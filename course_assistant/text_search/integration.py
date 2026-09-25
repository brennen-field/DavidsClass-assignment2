"""Connect text search to the document-store hooks defined by Chase."""

from typing import Protocol

from course_assistant.text_search.keyword import KeywordIndex


class DocumentStoreHooks(Protocol):
    def on_document_added(self, listener) -> None: ...

    def register_index_deleter(self, deleter) -> None: ...


def connect_document_store(store: DocumentStoreHooks, index: KeywordIndex) -> None:
    """Register keyword indexing and deletion exactly once on ``store``."""

    store.on_document_added(index.index_pages)
    store.register_index_deleter(index.delete_document)
