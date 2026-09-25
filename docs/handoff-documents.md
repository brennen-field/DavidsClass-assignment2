# Handoff: Documents → Text search & Visual search

Owner: Chase. Code: `course_assistant/documents/`.

The documents part turns uploaded files into **pages**. Nick (text search) and
Shrihari (visual search) index those pages. When a document is removed, the
documents part tells both indexes to delete it.

## What you receive: `Page`

```python
from course_assistant.documents.models import Page

Page(
    doc_id="3f9a1c0b7e2d4a61",           # stable ID, same file = same ID
    doc_name="Week 2 - Vibe Coding.pptx",  # original file name, for citations
    page_number=14,                        # 1-based page/slide number
    text="Vibe coding in prod ...",        # may be "" for image-only slides
    image_path="/abs/path/data/documents/3f9a1c0b7e2d4a61/pages/page-014.png",
    source_format="pptx",                  # "pdf" | "pptx" | "docx"
)
```

- One `Page` per PDF page or per slide. For PPTX, `page_number` is the slide number.
- `image_path` is an absolute path to a PNG (150 DPI). It stays valid until the document is removed.
- `doc_id` is the first 16 hex characters of the SHA-256 of the file bytes. Use it as the key
  (or metadata filter) for everything you store about a document.
- Use `page.to_dict()` if you need a plain dict (e.g. chromadb metadata).

## How you get pages

```python
from course_assistant.documents.store import get_store

store = get_store()

# Option 1: get notified whenever a document is added (recommended)
def index_document(pages: list[Page]) -> None:
    ...  # chunk / embed / add to your index

store.on_document_added(index_document)

# Option 2: pull pages yourself
for doc in store.list_documents():
    pages = store.get_pages(doc.doc_id)
```

A listener that raises an error doesn't undo the upload. The error is shown to the
user as a warning so they know the document may not be searchable.

## What you give back: a delete function

```python
def delete_document(doc_id: str) -> None:
    ...  # remove every entry for doc_id from your index

store.register_index_deleter(delete_document)
```

Requirements for your delete function:

- **Idempotent:** deleting a `doc_id` you don't have (or already deleted) must not raise.
- **Raise on real failure:** if the delete fails, raise. The document then stays listed
  so the user can retry, rather than disappearing from the list while still being
  searchable.

Removal order: every registered deleter runs first. Only if all of them succeed are the
document's files and catalog entry removed.

## Where to register

Register your listener and deleter once at startup. Brennen's `app.py` is the natural
place (or an `init()` in your package that `app.py` calls). Tests can build their own
store with `DocumentStore(tmp_path)`.
