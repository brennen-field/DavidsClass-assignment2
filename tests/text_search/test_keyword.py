from dataclasses import dataclass

import pytest

from course_assistant.text_search import KeywordIndex, chunk_pages, connect_document_store


@dataclass(frozen=True)
class SamplePage:
    doc_id: str
    doc_name: str
    page_number: int
    text: str
    image_path: str
    source_format: str = "pdf"


def page(doc_id: str, page_number: int, text: str, name: str = "Syllabus.pdf") -> SamplePage:
    return SamplePage(
        doc_id=doc_id,
        doc_name=name,
        page_number=page_number,
        text=text,
        image_path=f"/data/{doc_id}/page-{page_number:03d}.png",
    )


def test_chunk_pages_preserves_source_details_and_normalizes_whitespace():
    source = page("syllabus", 3, "Grading   policy\n\nAssignments are worth 40 percent.")

    chunks = chunk_pages([source], chunk_size=32, chunk_overlap=8)

    assert len(chunks) > 1
    assert all(chunk.doc_id == "syllabus" for chunk in chunks)
    assert all(chunk.doc_name == "Syllabus.pdf" for chunk in chunks)
    assert all(chunk.page_number == 3 for chunk in chunks)
    assert all(chunk.image_path.endswith("page-003.png") for chunk in chunks)
    assert all("\n" not in chunk.text and "  " not in chunk.text for chunk in chunks)
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)


def test_empty_page_has_no_text_chunks():
    assert chunk_pages([page("slides", 4, "  \n ")]) == []


def test_bm25_returns_the_relevant_sourced_passage_first():
    index = KeywordIndex(chunk_size=500, chunk_overlap=50)
    index.index_pages(
        [
            page("syllabus", 1, "Office hours are held on Tuesday afternoon."),
            page("syllabus", 2, "The grading policy assigns 40 percent to the final project."),
            page("week-1", 7, "Retrieval augmented generation combines search with an LLM.", "Week 1.pdf"),
        ]
    )

    results = index.search("What is the grading policy?", top_k=2)

    assert results[0].doc_id == "syllabus"
    assert results[0].page_number == 2
    assert "40 percent" in results[0].text
    assert results[0].search_method == "bm25"
    assert results[0].score > 0


def test_reindexing_a_document_replaces_its_old_chunks():
    index = KeywordIndex()
    index.index_pages([page("syllabus", 1, "Old attendance rule")])

    index.index_pages([page("syllabus", 1, "New participation policy")])

    assert index.search("attendance") == []
    assert index.search("participation")[0].text == "New participation policy"
    assert len(index.chunks) == 1


def test_delete_document_is_complete_and_idempotent():
    index = KeywordIndex()
    index.index_pages(
        [
            page("syllabus", 1, "Grading policy"),
            page("week-1", 1, "Generative AI", "Week 1.pdf"),
        ]
    )

    index.delete_document("syllabus")
    index.delete_document("syllabus")

    assert index.search("grading") == []
    assert index.search("generative")[0].doc_id == "week-1"


def test_store_adapter_registers_add_and_delete_hooks():
    class FakeStore:
        def on_document_added(self, listener):
            self.add_listener = listener

        def register_index_deleter(self, deleter):
            self.deleter = deleter

    store = FakeStore()
    index = KeywordIndex()
    connect_document_store(store, index)

    store.add_listener([page("syllabus", 1, "Late work policy")])
    assert index.search("late work")[0].doc_id == "syllabus"

    store.deleter("syllabus")
    assert index.search("late work") == []


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (10, -1), (10, 10)],
)
def test_chunk_configuration_is_validated(chunk_size, overlap):
    with pytest.raises(ValueError):
        chunk_pages([page("doc", 1, "text")], chunk_size=chunk_size, chunk_overlap=overlap)
