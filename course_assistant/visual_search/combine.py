"""Combine retrieval candidates and rerank into final evidence (owner: Shrihari).

This is the handoff point to Preston and Brennen: everything the retrievers
(keyword, text embedding, visual) surface is merged by ``chunk_id`` into one
ranked list of ``EvidenceItem`` records.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from course_assistant.text_search.models import SearchResult
from course_assistant.visual_search.models import EvidenceItem, VisualHit
from course_assistant.visual_search.rerank import RerankItem, RerankerClient, RerankServiceError


@dataclass
class _Candidate:
    doc_id: str
    doc_name: str
    page_number: int
    text: str
    image_path: str
    source_format: str
    chunk_id: str = ""
    methods: set[str] = field(default_factory=set)
    method_scores: dict[str, float] = field(default_factory=dict)

    @property
    def best_score(self) -> float:
        """Highest score a single retrieval method assigned this candidate."""
        return max(self.method_scores.values()) if self.method_scores else 0.0

    def add(self, method: str, score: float) -> None:
        self.methods.add(method)
        self.method_scores[method] = max(
            self.method_scores.get(method, float("-inf")), score
        )

    def to_evidence(self, rerank_score: float) -> EvidenceItem:
        return EvidenceItem(
            chunk_id=self.chunk_id,
            doc_id=self.doc_id,
            doc_name=self.doc_name,
            page_number=self.page_number,
            text=self.text,
            image_path=self.image_path,
            source_format=self.source_format,
            rerank_score=rerank_score,
            retrieved_by=tuple(sorted(self.methods)),
        )


def combine_and_rerank(
    query: str,
    keyword_results: Sequence[SearchResult],
    text_embedding_results: Sequence[SearchResult],
    visual_results: Sequence[VisualHit],
    *,
    use_reranker: bool = True,
    reranker: RerankerClient | None = None,
) -> list[EvidenceItem]:
    """Merge all retrieval methods and return one ordered evidence list.

    Candidates from keyword and text embedding are deduplicated by ``chunk_id``
    (retaining the source fields and every method that found them). A visual hit
    attributes ``"visual"`` to any text candidate on the same ``doc_id`` /
    ``page_number``; a visual hit that matches no text candidate is kept as an
    image-only evidence item.

    When ``use_reranker`` is true and ``reranker`` is given, candidates are
    ordered by the class reranker's relevance score. Otherwise candidates fall
    back to a deterministic order: most retrieval methods first, then best
    per-method score (raw cross-method scores are otherwise not compared).
    A reranking outage degrades to that same fallback order.
    """

    candidates: dict[str, _Candidate] = {}
    for result in list(keyword_results) + list(text_embedding_results):
        candidate = candidates.setdefault(
            result.chunk_id,
            _Candidate(
                chunk_id=result.chunk_id,
                doc_id=result.doc_id,
                doc_name=result.doc_name,
                page_number=result.page_number,
                text=result.text,
                image_path=result.image_path,
                source_format=result.source_format,
            ),
        )
        candidate.add(result.search_method, result.score)

    # Deduplicate visual hits by page, keeping the strongest score.
    visual_by_page: dict[tuple[str, int], VisualHit] = {}
    for hit in visual_results:
        key = (hit.doc_id, hit.page_number)
        if key not in visual_by_page or hit.score > visual_by_page[key].score:
            visual_by_page[key] = hit

    image_only: dict[tuple[str, int], _Candidate] = {}
    for (doc_id, page_number), hit in visual_by_page.items():
        page_candidates = [
            c
            for c in candidates.values()
            if c.doc_id == doc_id and c.page_number == page_number
        ]
        if page_candidates:
            for candidate in page_candidates:
                candidate.add("visual", hit.score)
        else:
            image_only_candidate = image_only.setdefault(
                (doc_id, page_number),
                _Candidate(
                    doc_id=doc_id,
                    doc_name=hit.doc_name,
                    page_number=page_number,
                    text="",
                    image_path=hit.image_path,
                    source_format=hit.source_format,
                ),
            )
            image_only_candidate.add("visual", hit.score)

    final = list(candidates.values()) + list(image_only.values())
    if use_reranker and reranker is not None and final:
        try:
            return _reranked(query, reranker, final)
        except RerankServiceError:
            # Degrade gracefully to the deterministic order on an outage.
            pass
    return _fallback_ordered(final)


def _reranked(
    query: str,
    reranker: RerankerClient,
    candidates: Sequence[_Candidate],
) -> list[EvidenceItem]:
    items = [
        RerankItem(text=candidate.text, image_path=candidate.image_path or None)
        for candidate in candidates
    ]
    scores = reranker.rerank(query, items)
    score_by_index = {score.index: score.relevance_score for score in scores}
    ordered = sorted(
        range(len(candidates)),
        key=lambda index: -score_by_index.get(index, 0.0),
    )
    return [
        candidates[index].to_evidence(score_by_index[index]) for index in ordered
    ]


def _fallback_ordered(candidates: Sequence[_Candidate]) -> list[EvidenceItem]:
    ordered = sorted(
        candidates,
        key=lambda c: (
            -len(c.methods),
            -c.best_score,
            c.doc_id,
            c.page_number,
            c.chunk_id,
        ),
    )
    return [candidate.to_evidence(candidate.best_score) for candidate in ordered]
