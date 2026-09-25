import pytest

from course_assistant.text_search.models import SearchResult
from course_assistant.visual_search.combine import combine_and_rerank
from course_assistant.visual_search.models import VisualHit
from course_assistant.visual_search.rerank import RerankScore, RerankServiceError


def sr(chunk_id, method, score, doc="doc1", page=1, text="passage"):
    return SearchResult(
        chunk_id=chunk_id,
        doc_id=doc,
        doc_name=f"{doc}.pptx",
        page_number=page,
        text=text,
        image_path=f"/data/{doc}/page-{page:03d}.png",
        source_format="pptx",
        score=score,
        search_method=method,
    )


def vh(page, score, doc="doc1", image="chart"):
    return VisualHit(
        doc_id=doc,
        doc_name=f"{doc}.pptx",
        page_number=page,
        image_path=f"/data/{doc}/{image}.png",
        source_format="pptx",
        score=score,
    )


class FakeReranker:
    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def rerank(self, query, items):
        self.calls.append((query, items))
        return self.scores


def test_chunks_found_by_multiple_methods_dedupe_into_one_evidence_item():
    results = combine_and_rerank(
        "q",
        [sr("c1", "bm25", 0.9)],
        [sr("c1", "text_embedding", 0.8)],
        [],
        use_reranker=False,
    )
    assert len(results) == 1
    item = results[0]
    assert item.chunk_id == "c1"
    assert item.retrieved_by == ("bm25", "text_embedding")


def test_fallback_ranks_more_methods_first_even_with_lower_score():
    results = combine_and_rerank(
        "q",
        [sr("cA", "bm25", 0.1), sr("cSolo", "bm25", 0.9)],
        [sr("cA", "text_embedding", 0.1)],
        [],
        use_reranker=False,
    )
    # cA was found by two methods and must beat the higher-scoring one-method hit.
    assert [r.chunk_id for r in results] == ["cA", "cSolo"]


def test_fallback_breaks_ties_with_best_per_method_score():
    results = combine_and_rerank(
        "q",
        [sr("cA", "bm25", 0.2, doc="a"), sr("cB", "bm25", 0.9, doc="b")],
        [sr("cA", "text_embedding", 0.2), sr("cB", "text_embedding", 0.3)],
        [],
        use_reranker=False,
    )
    # Both found by two methods; cB has the higher best score -> first.
    assert [r.chunk_id for r in results] == ["cB", "cA"]


def test_visual_hit_attributes_visual_to_text_candidate_on_same_page():
    results = combine_and_rerank(
        "q",
        [sr("c1", "bm25", 0.5, page=2)],
        [],
        [vh(page=2, score=0.8)],
        use_reranker=False,
    )
    assert len(results) == 1
    assert results[0].retrieved_by == ("bm25", "visual")


def test_visual_hit_with_no_matching_page_becomes_image_only_evidence():
    results = combine_and_rerank(
        "q",
        [sr("c1", "bm25", 0.5, page=1)],
        [],
        [vh(page=7, score=0.9, image="meme")],
        use_reranker=False,
    )
    image_item = results[0]  # image-only, found by one method, page 7
    assert image_item.page_number == 7
    assert image_item.text == ""
    assert image_item.image_path.endswith("meme.png")
    assert image_item.retrieved_by == ("visual",)


def test_reranked_order_uses_reranker_scores():
    fake = FakeReranker(
        [RerankScore(index=1, relevance_score=0.9), RerankScore(index=0, relevance_score=0.2)]
    )
    results = combine_and_rerank(
        "q",
        [sr("cA", "bm25", 0.5), sr("cB", "text_embedding", 0.4)],
        [],
        [],
        use_reranker=True,
        reranker=fake,
    )
    assert [r.chunk_id for r in results] == ["cB", "cA"]
    assert results[0].rerank_score == pytest.approx(0.9)
    assert results[1].rerank_score == pytest.approx(0.2)
    # The query and a text-bearing RerankItem reached the fake reranker.
    assert fake.calls[0][0] == "q"
    assert fake.calls[0][1][0].text == "passage"


def test_reranker_outage_degrades_to_fallback_order():
    class FailingReranker(FakeReranker):
        def rerank(self, query, items):
            raise RerankServiceError("down")

    results = combine_and_rerank(
        "q",
        [sr("cA", "bm25", 0.1)],
        [sr("cA", "text_embedding", 0.1)],
        [],
        use_reranker=True,
        reranker=FailingReranker([]),
    )
    # Even though rerank failed, retrieval still works via the fallback order.
    assert [r.chunk_id for r in results] == ["cA"]
    assert results[0].retrieved_by == ("bm25", "text_embedding")


def test_no_reranker_given_falls_back_even_when_flag_is_on():
    results = combine_and_rerank(
        "q", [sr("c1", "bm25", 0.5)], [], [], use_reranker=True, reranker=None
    )
    assert [r.chunk_id for r in results] == ["c1"]


def test_empty_inputs_return_empty_list():
    assert combine_and_rerank("q", [], [], [], use_reranker=False) == []
