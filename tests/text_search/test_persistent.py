from dataclasses import dataclass

from course_assistant.text_search import ChromaSemanticIndex


@dataclass(frozen=True)
class SamplePage:
    doc_id: str
    doc_name: str
    page_number: int
    text: str
    image_path: str
    source_format: str = "pdf"


def page(doc_id: str, page_number: int, text: str) -> SamplePage:
    return SamplePage(
        doc_id=doc_id,
        doc_name=f"{doc_id}.pdf",
        page_number=page_number,
        text=text,
        image_path=f"/data/{doc_id}/page-{page_number:03d}.png",
    )


class MeaningEmbedder:
    def _vector(self, text):
        text = text.lower()
        return (
            float("grading" in text or "grade" in text),
            float("office" in text),
            0.1,
        )

    def embed_documents(self, texts):
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        return self._vector(text)


def test_chroma_index_survives_restart_and_preserves_sources(tmp_path):
    index = ChromaSemanticIndex(MeaningEmbedder(), path=tmp_path / "vectors")
    index.index_pages(
        [
            page("calendar", 1, "Office hours are Tuesday."),
            page("syllabus", 2, "The final project affects the course grade."),
        ]
    )
    index.close()

    reopened = ChromaSemanticIndex(MeaningEmbedder(), path=tmp_path / "vectors")
    result = reopened.search("How is grading determined?", top_k=1)[0]

    assert result.doc_id == "syllabus"
    assert result.doc_name == "syllabus.pdf"
    assert result.page_number == 2
    assert result.image_path.endswith("page-002.png")
    assert result.search_method == "text_embedding"
    reopened.close()


def test_chroma_reindex_replaces_stale_chunks_and_delete_is_idempotent(tmp_path):
    index = ChromaSemanticIndex(MeaningEmbedder(), path=tmp_path / "vectors")
    index.index_pages([page("syllabus", 1, "Old grading rule")])

    index.index_pages([page("syllabus", 3, "New grading rule")])

    assert len(index.chunks) == 1
    assert index.chunks[0].page_number == 3

    index.delete_document("syllabus")
    index.delete_document("syllabus")
    assert index.search("grading") == []
    index.close()
