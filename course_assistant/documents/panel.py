"""Documents tab (owner: Chase): upload, list, preview, and remove course files."""

from pathlib import Path

import gradio as gr

from course_assistant.documents.store import SUPPORTED_FORMATS, DocumentStore, get_store

TABLE_HEADERS = ["Document", "Format", "Pages", "Added (UTC)", "Text from"]


def _table_rows(store: DocumentStore) -> list[list]:
    return [
        [d.doc_name, d.source_format.upper(), d.page_count, d.added_at[:16].replace("T", " "), d.text_source]
        for d in store.list_documents()
    ]


def _choices(store: DocumentStore) -> list[tuple[str, str]]:
    return [(d.doc_name, d.doc_id) for d in store.list_documents()]


def _preview(store: DocumentStore, doc_id: str | None) -> list[tuple[str, str]]:
    if not doc_id:
        return []
    doc = store.get_document(doc_id)
    label = "Slide" if doc and doc.source_format == "pptx" else "Page"
    return [(p.image_path, f"{label} {p.page_number}") for p in store.get_pages(doc_id)]


def add_files(store: DocumentStore, files: list[str] | None):
    """Add uploaded files and return (status, table, dropdown update, cleared upload)."""
    if not files:
        message = "Choose one or more files first."
    else:
        lines = []
        for file_path in files:
            result = store.add_file(Path(file_path), doc_name=Path(file_path).name)
            icon = {"added": "✅", "duplicate": "ℹ️", "error": "❌"}[result.status]
            lines.append(f"{icon} {result.message}")
            lines.extend(f"⚠️ {w}" for w in result.warnings)
        message = "\n\n".join(lines)
    return message, _table_rows(store), gr.update(choices=_choices(store)), None


def remove_document(store: DocumentStore, doc_id: str | None):
    """Remove the selected document and return (status, table, dropdown update, preview)."""
    if not doc_id:
        return "Select a document to remove.", _table_rows(store), gr.update(), []
    result = store.remove(doc_id)
    message = f"{'✅' if result.ok else '❌'} {result.message}"
    kept = None if result.ok else doc_id
    return (
        message,
        _table_rows(store),
        gr.update(choices=_choices(store), value=kept),
        _preview(store, kept),
    )


def build_panel(store: DocumentStore | None = None) -> None:
    """Add the document manager components. Called by app.py inside a gr.Tab."""
    store = store or get_store()
    formats = sorted(SUPPORTED_FORMATS)

    gr.Markdown(
        f"Add course files ({', '.join(f.lstrip('.').upper() for f in formats)}). "
        "PowerPoint and Word files are converted to PDF with LibreOffice. "
        "Uploading a file that's already loaded is skipped."
    )
    with gr.Row():
        uploads = gr.File(label="Upload files", file_count="multiple", file_types=formats, type="filepath")
    add_button = gr.Button("Add documents", variant="primary")
    status = gr.Markdown()

    table = gr.Dataframe(
        headers=TABLE_HEADERS,
        value=lambda: _table_rows(store),
        interactive=False,
        label="Loaded documents",
    )

    with gr.Row():
        selected = gr.Dropdown(
            label="Select a document",
            choices=_choices(store),
            value=None,
            scale=3,
        )
        remove_button = gr.Button("Remove document", variant="stop", scale=1)

    preview = gr.Gallery(label="Pages", columns=4, height="auto", object_fit="contain")

    add_button.click(
        lambda files: add_files(store, files),
        inputs=uploads,
        outputs=[status, table, selected, uploads],
        api_name="add_documents",
    )
    selected.change(lambda doc_id: _preview(store, doc_id), inputs=selected, outputs=preview, api_name="preview_document")
    remove_button.click(
        lambda doc_id: remove_document(store, doc_id),
        inputs=selected,
        outputs=[status, table, selected, preview],
        api_name="remove_document",
    )
