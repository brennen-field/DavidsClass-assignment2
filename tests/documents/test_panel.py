import gradio as gr

from course_assistant.documents.panel import add_files, build_panel, remove_document
from course_assistant.documents.store import DocumentStore
from tests.documents.test_store import make_pdf


def test_upload_then_remove_workflow(tmp_path):
    store = DocumentStore(tmp_path / "data")
    syllabus = make_pdf(tmp_path / "Syllabus.pdf", ["Grading policy"])
    week1 = make_pdf(tmp_path / "Week 1.pdf", ["Intro", "RAG"])
    notes = tmp_path / "notes.txt"
    notes.write_text("hello")

    status, rows, dropdown, cleared = add_files(store, [str(syllabus), str(week1), str(syllabus), str(notes)])

    assert "Syllabus.pdf: added" in status
    assert "Week 1.pdf: added" in status
    assert "already loaded" in status
    assert "unsupported" in status
    assert [r[0] for r in rows] == ["Syllabus.pdf", "Week 1.pdf"]
    assert cleared is None

    week1_id = next(d.doc_id for d in store.list_documents() if d.doc_name == "Week 1.pdf")
    status, rows, dropdown, preview = remove_document(store, week1_id)

    assert "removed" in status
    assert [r[0] for r in rows] == ["Syllabus.pdf"]
    assert dropdown["choices"] == [("Syllabus.pdf", store.list_documents()[0].doc_id)]
    assert preview == []


def test_add_with_no_files(tmp_path):
    status, *_ = add_files(DocumentStore(tmp_path / "data"), None)
    assert "Choose" in status


def test_remove_with_nothing_selected(tmp_path):
    status, *_ = remove_document(DocumentStore(tmp_path / "data"), None)
    assert "Select" in status


def test_panel_builds(tmp_path):
    with gr.Blocks():
        build_panel(DocumentStore(tmp_path / "data"))
