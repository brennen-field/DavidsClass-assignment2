"""Connect a text index to the document-store hooks defined by Chase."""

from typing import Protocol

class DocumentStoreHooks(Protocol):
    def on_document_added(self, listener) -> None: ...

    def register_index_deleter(self, deleter) -> None: ...


class TextIndexHooks(Protocol):
    def index_pages(self, pages): ...

    def delete_document(self, doc_id: str) -> None: ...


def connect_document_store(store: DocumentStoreHooks, index: TextIndexHooks) -> None:
    """Register one text index's add and delete callbacks on ``store``."""

    store.on_document_added(index.index_pages)
    store.register_index_deleter(index.delete_document)
