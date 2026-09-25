"""Documents tab (owner: Chase): upload, list, preview, and remove course files."""

import html
import random
import uuid
from pathlib import Path

import gradio as gr

from course_assistant.documents.store import SUPPORTED_FORMATS, AddResult, DocumentStore, get_store
from course_assistant.theme import CU_DARK_GRAY, CU_GOLD, CU_LIGHT_GRAY

TABLE_HEADERS = ["Document", "Format", "Pages", "Added (UTC)", "Text from"]
_CONFETTI_COLORS = [CU_GOLD, CU_GOLD, "#FFFFFF", CU_LIGHT_GRAY, CU_DARK_GRAY, "#F5EEDA"]


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


def stats_html(store: DocumentStore) -> str:
    docs = store.list_documents()
    if not docs:
        return (
            '<div class="buffs-empty">🦬 No documents yet. '
            "Drop in the syllabus or some slides above to get started!</div>"
        )
    decks = sum(d.source_format == "pptx" for d in docs)
    pages = sum(d.page_count for d in docs)
    tiles = [
        (len(docs), _plural(len(docs), "document") + " loaded"),
        (pages, "pages & slides searchable" if pages != 1 else "page searchable"),
        (decks, _plural(decks, "slide deck")),
    ]
    return '<div class="buffs-stats">' + "".join(
        f'<div class="buffs-stat"><div class="num">{n}</div><div class="label">{label}</div></div>'
        for n, label in tiles
    ) + "</div>"


def _plural(n: int, word: str) -> str:
    return word if n == 1 else word + "s"


def _message(kind: str, icon: str, text: str) -> str:
    return f'<div class="buffs-msg {kind}">{icon} {html.escape(text)}</div>'


def _celebration(added: list[AddResult]) -> str:
    pages = sum(r.document.page_count for r in added)
    what = f"{len(added)} {_plural(len(added), 'document')} · {pages} {_plural(pages, 'page')} ready"
    confetti = "".join(
        f'<span class="confetti" style="left:{random.randint(2, 98)}%;'
        f"background:{random.choice(_CONFETTI_COLORS)};"
        f'animation-delay:{random.uniform(0, 0.6):.2f}s;animation-duration:{random.uniform(1.4, 2.2):.2f}s"></span>'
        for _ in range(28)
    )
    # A fresh id makes Gradio replace the element, so the animation replays every time.
    return (
        f'<div class="buffs-celebrate" data-run="{uuid.uuid4().hex[:8]}" role="status">'
        f'{confetti}<div class="cheer">Sko Buffs!</div><div class="sub">{what}</div>'
        '<span class="dust" aria-hidden="true">💨</span><span class="stampede" aria-hidden="true">🦬</span>'
        "</div>"
    )


def add_files(store: DocumentStore, files: list[str] | None):
    """Add uploaded files. Returns (status, table, dropdown update, cleared upload, stats)."""
    if not files:
        status = _message("warning", "📂", "Choose one or more files first.")
        return status, _table_rows(store), gr.update(), None, stats_html(store)

    icons = {"added": "✅", "duplicate": "🔁", "error": "❌"}
    results = [store.add_file(Path(f), doc_name=Path(f).name) for f in files]
    lines = []
    for result in results:
        lines.append(_message(result.status, icons[result.status], result.message))
        lines.extend(_message("warning", "⚠️", w) for w in result.warnings)

    added = [r for r in results if r.status == "added"]
    status = (_celebration(added) if added else "") + "".join(lines)
    return status, _table_rows(store), gr.update(choices=_choices(store)), None, stats_html(store)


def remove_document(store: DocumentStore, doc_id: str | None):
    """Remove the selected document. Returns (status, table, dropdown update, preview, stats)."""
    if not doc_id:
        status = _message("warning", "👆", "Select a document to remove.")
        return status, _table_rows(store), gr.update(), [], stats_html(store)
    result = store.remove(doc_id)
    status = _message("added" if result.ok else "error", "🗑️" if result.ok else "❌", result.message)
    kept = None if result.ok else doc_id
    return (
        status,
        _table_rows(store),
        gr.update(choices=_choices(store), value=kept),
        _preview(store, kept),
        stats_html(store),
    )


def build_panel(store: DocumentStore | None = None) -> None:
    """Add the document manager components. Called by app.py inside a gr.Tab."""
    store = store or get_store()
    formats = sorted(SUPPORTED_FORMATS)

    gr.Markdown(
        "### Load your course materials\n"
        f"Accepts **{', '.join(f.lstrip('.').upper() for f in formats)}**. "
        "PowerPoint and Word files are converted automatically with LibreOffice, and each page or "
        "slide is saved as an image so you can search pictures, charts, and memes too. "
        "Uploading a file that's already loaded is skipped."
    )
    uploads = gr.File(label="Drop files here", file_count="multiple", file_types=formats, type="filepath")
    add_button = gr.Button("🦬 Add documents", variant="primary", size="lg")
    status = gr.HTML()

    stats = gr.HTML(value=lambda: stats_html(store))
    table = gr.Dataframe(
        headers=TABLE_HEADERS,
        value=lambda: _table_rows(store),
        interactive=False,
        label="Loaded documents",
    )

    with gr.Row(equal_height=True):
        selected = gr.Dropdown(label="Select a document to preview or remove", choices=_choices(store), value=None, scale=3)
        remove_button = gr.Button("🗑️ Remove document", variant="stop", scale=1)

    preview = gr.Gallery(label="Pages", columns=4, height="auto", object_fit="contain")

    add_button.click(
        lambda files: add_files(store, files),
        inputs=uploads,
        outputs=[status, table, selected, uploads, stats],
        api_name="add_documents",
    )
    selected.change(lambda doc_id: _preview(store, doc_id), inputs=selected, outputs=preview, api_name="preview_document")
    # Refresh the list on page load so a new browser tab sees documents added elsewhere.
    gr.on(triggers=None, fn=lambda: gr.update(choices=_choices(store)), outputs=selected, api_visibility="private")
    remove_button.click(
        lambda doc_id: remove_document(store, doc_id),
        inputs=selected,
        outputs=[status, table, selected, preview, stats],
        api_name="remove_document",
    )
