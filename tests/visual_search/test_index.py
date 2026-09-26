from dataclasses import dataclass
from pathlib import Path

import pytest

from course_assistant.text_search.models import PageLike
from course_assistant.visual_search.embedding import (
    VisualEmbeddingServiceError,
    VisualIndex,
)
from course_assistant.visual_search.persistent import ChromaVisualIndex


@dataclass(frozen=True)
class SamplePage:
    doc_id: str
    doc_name: str
    page_number: int
    text: str
    image_path: str
    source_format: str = "pptx"


def page(doc_id, page_number, image_name, text="", source_format="pptx") -> SamplePage:
    image_path = "" if image_name is None else f"/data/{doc_id}/{image_name}.png"
    return SamplePage(
        doc_id=doc_id,
        doc_name=f"{doc_id}.{source_format}",
        page_number=page_number,
        text=text,
        image_path=image_path,
        source_format=source_format,
    )


class FakeVisualEmbedder:
    """Encodes whether a path/query mentions 'chart' or 'diagram'."""

    def __init__(self):
        self.image_calls = []
        self.text_calls = []

    def _vector(self, text):
        text = text.lower()
        return (
            float("chart" in text),
            float("diagram" in text),
            0.1,
        )

    def embed_images(self, image_paths):
        self.image_calls.append(list(image_paths))
        return [self._vector(Path(p).stem) for p in image_paths]

    def embed_text(self, text):
        self.text_calls.append(text)
        return self._vector(text)


@pytest.fixture
def embedder():
    return FakeVisualEmbedder()


def chart_diagram_pages():
    return [
        page("decks", 1, "week1-chart", "Revenue chart"),
        page("decks", 2, "week1-diagram", "Architecture diagram"),
    ]


def test_visual_index_returns_the_relevant_page_image_first(embedder):
    index = VisualIndex(embedder)
    index.index_pages(chart_diagram_pages())

    results = index.search("chart on revenue", top_k=1)

    assert len(results) == 1
    hit = results[0]
    assert hit.doc_id == "decks"
    assert hit.page_number == 1
    assert hit.image_path.endswith("week1-chart.png")
    assert hit.source_format == "pptx"
    assert embedder.text_calls == ["chart on revenue"]


def test_visual_index_skips_pages_without_an_image(embedder):
    index = VisualIndex(embedder)
    index.index_pages([page("decks", 1, None), page("decks", 2, "chart")])

    assert [p.page_number for p in index.pages] == [2]


def test_visual_reindex_and_delete_are_complete_and_idempotent(embedder):
    index = VisualIndex(embedder)
    index.index_pages([page("decks", 1, "chart")])

    index.index_pages([page("decks", 2, "diagram")])
    assert len(index.pages) == 1
    assert index.pages[0].page_number == 2

    index.delete_document("decks")
    index.delete_document("decks")
    assert index.search("chart") == []


def test_failed_reindex_keeps_existing_document_available(embedder):
    class FailingFake(FakeVisualEmbedder):
        fail = False

        def embed_images(self, image_paths):
            if self.fail:
                raise VisualEmbeddingServiceError("service unavailable")
            return super().embed_images(image_paths)

    index = VisualIndex(FailingFake())
    index.index_pages([page("decks", 1, "chart")])

    index.embedder.fail = True
    with pytest.raises(VisualEmbeddingServiceError):
        index.index_pages([page("decks", 2, "diagram")])

    assert index.pages[0].page_number == 1


def test_reindex_document_to_no_images_clears_in_memory_entries(embedder):
    index = VisualIndex(embedder)
    index.index_pages([page("decks", 1, "chart")])
    assert len(index.pages) == 1

    # Re-index the same document with only an image-less page: the old visual
    # entries for the document must be removed, not left searchable.
    index.index_pages([page("decks", 2, None)])

    assert index.pages == ()
    assert index.search("chart") == []


def test_visual_index_requires_positive_top_k(embedder):
    index = VisualIndex(embedder)
    with pytest.raises(ValueError):
        index.search("anything", top_k=0)
    assert index.search("") == []


# --- Persistent (Chroma) index ---


def make_persistent_index(tmp_path, embedder):
    return ChromaVisualIndex(
        embedder,
        path=tmp_path / "visual_vectors",
        collection_name="test_page_images",
    )


def test_chroma_index_ranks_and_replaces(embedder, tmp_path):
    with make_persistent_index(tmp_path, embedder) as index:
        index.index_pages(chart_diagram_pages())
        hits = index.search("diagram", top_k=1)
        assert hits[0].page_number == 2

        index.index_pages([page("decks", 3, "chart")])
        assert [p.page_number for p in index.pages] == [3]


def test_chroma_index_persists_across_restarts(embedder, tmp_path):
    path = tmp_path / "visual_vectors"
    with ChromaVisualIndex(embedder, path=path, collection_name="test_page_images") as index:
        index.index_pages(chart_diagram_pages())

    # A fresh index on the same backing directory serves stored vectors without
    # needing anything re-embedded (embedder sees no new calls).
    embedder.image_calls.clear()
    embedder.text_calls.clear()
    with ChromaVisualIndex(embedder, path=path, collection_name="test_page_images") as reopened:
        hits = reopened.search("chart", top_k=5)

    # Persisted: chart page ranks above the diagram page, in rank order.
    assert [h.page_number for h in hits] == [1, 2]
    assert hits[0].image_path.endswith("week1-chart.png")


def test_chroma_index_delete_is_idempotent(embedder, tmp_path):
    index = make_persistent_index(tmp_path, embedder)
    index.index_pages(chart_diagram_pages())
    index.delete_document("decks")
    index.delete_document("decks")
    assert index.search("chart") == []
    index.close()


def test_chroma_reindex_to_no_images_clears_previous_entries(embedder, tmp_path):
    index = make_persistent_index(tmp_path, embedder)
    index.index_pages([page("decks", 1, "chart")])
    assert len(index.pages) == 1

    # Re-index the same document with only an image-less page: the previously
    # persisted visual entry must be removed, not left searchable.
    index.index_pages([page("decks", 2, None)])

    assert index.pages == ()
    assert index.search("chart") == []
    index.close()
