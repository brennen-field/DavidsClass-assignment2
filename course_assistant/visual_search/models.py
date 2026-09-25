"""Data contracts owned by visual search and combine-and-rerank (owner: Shrihari)."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class EvidenceItem:
    """One final ranked piece of evidence handed to answers/quiz.

    Consumes the source-carrying fields of a text ``SearchResult`` (from Nick)
    or the visual index, and adds the final cross-method rerank score plus the
    retrieval methods that surfaced it. Handoff: Shrihari -> Preston, Brennen.

    ``retrieved_by`` lists every method that found this piece of evidence, e.g.
    ``("bm25", "text_embedding", "visual")``. ``text`` may be empty for
    image-only pages: the visual index owns retrieval for pages with no text,
    so downstream display should rely on ``image_path`` when ``text`` is empty.
    """

    chunk_id: str
    doc_id: str
    doc_name: str
    page_number: int
    text: str
    image_path: str
    source_format: str
    rerank_score: float
    retrieved_by: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class VisualPage:
    """One stored page record in the visual index (one per page/slide image)."""

    doc_id: str
    doc_name: str
    page_number: int
    image_path: str
    source_format: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class VisualHit:
    """One visual-search candidate: a page ranked by query-image similarity."""

    doc_id: str
    doc_name: str
    page_number: int
    image_path: str
    source_format: str
    score: float

    @classmethod
    def from_page(cls, page: VisualPage, *, score: float) -> "VisualHit":
        return cls(
            doc_id=page.doc_id,
            doc_name=page.doc_name,
            page_number=page.page_number,
            image_path=page.image_path,
            source_format=page.source_format,
            score=score,
        )

    def to_dict(self) -> dict:
        return asdict(self)
