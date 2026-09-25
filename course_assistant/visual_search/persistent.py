"""Persistent Chroma storage for page-image embeddings (owner: Shrihari)."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from course_assistant.text_search.models import PageLike
from course_assistant.visual_search.embedding import (
    VisualEmbeddingClient,
    VisualEmbedder,
    VisualEmbeddingServiceError,
    _page_id,
    _validate_vector_dimensions,
    _visual_page,
)
from course_assistant.visual_search.models import VisualHit, VisualPage

DEFAULT_COLLECTION_NAME = "course_page_images"


class ChromaVisualIndex:
    """Persist page-image vectors locally while keeping their source metadata.

    Mirrors ``ChromaSemanticIndex`` (Nick) but stores one vector per page rather
    than per text chunk, so image-only slides persist across app restarts.
    """

    def __init__(
        self,
        embedder: VisualEmbedder,
        *,
        path: str | Path,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        import chromadb

        self.embedder = embedder
        self.path = Path(path)
        self._client = chromadb.PersistentClient(path=str(self.path))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @classmethod
    def from_environment(cls) -> "ChromaVisualIndex":
        """Use the configured class endpoint and local application data folder."""

        from course_assistant import config

        return cls(
            VisualEmbeddingClient.from_environment(),
            path=config.DATA_DIR / "visual_vectors",
        )

    def __enter__(self) -> "ChromaVisualIndex":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        """Release Chroma file handles, which is important on Windows."""

        self._client.close()

    @property
    def pages(self) -> tuple[VisualPage, ...]:
        stored = self._collection.get(include=["metadatas"])
        return tuple(
            _stored_page(page_id, metadata)
            for page_id, metadata in zip(
                stored["ids"],
                stored["metadatas"] or [],
                strict=True,
            )
        )

    def index_pages(self, pages: Iterable[PageLike]) -> list[VisualPage]:
        """Embed page images and persist them, replacing stale docs by ID."""

        indexed = [_visual_page(page) for page in pages if page.image_path]
        doc_ids = {page.doc_id for page in indexed}
        vectors = (
            self.embedder.embed_images([page.image_path for page in indexed])
            if indexed
            else []
        )
        if len(vectors) != len(indexed):
            raise VisualEmbeddingServiceError(
                "The visual-embedding service returned an unexpected vector count"
            )
        _validate_vector_dimensions(vectors)

        old_ids_by_doc = {
            doc_id: set(
                self._collection.get(where={"doc_id": doc_id}, include=[])["ids"]
            )
            for doc_id in doc_ids
        }
        new_ids_by_doc = {
            doc_id: {_page_id(page.doc_id, page.page_number) for page in indexed if page.doc_id == doc_id}
            for doc_id in doc_ids
        }

        # Upsert first so an indexing failure cannot erase the previous document.
        if indexed:
            self._collection.upsert(
                ids=[_page_id(page.doc_id, page.page_number) for page in indexed],
                embeddings=[list(vector) for vector in vectors],
                documents=[page.image_path for page in indexed],
                metadatas=[_page_metadata(page) for page in indexed],
            )

        for doc_id in doc_ids:
            stale_ids = sorted(old_ids_by_doc[doc_id] - new_ids_by_doc[doc_id])
            if stale_ids:
                self._collection.delete(ids=stale_ids)
        return indexed

    def delete_document(self, doc_id: str) -> None:
        """Delete one document's persisted page vectors; missing IDs are a no-op."""

        stored = self._collection.get(where={"doc_id": doc_id}, include=[])
        if stored["ids"]:
            self._collection.delete(ids=stored["ids"])

    def search(self, query: str, *, top_k: int = 5) -> list[VisualHit]:
        """Return persisted page images ordered by Chroma cosine similarity."""

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        query = query.strip()
        count = self._collection.count()
        if not query or count == 0:
            return []

        query_vector = self.embedder.embed_text(query)
        _validate_vector_dimensions([query_vector])
        stored = self._collection.query(
            query_embeddings=[list(query_vector)],
            n_results=min(top_k, count),
            include=["metadatas", "distances"],
        )
        ids = stored["ids"][0]
        metadatas = (stored["metadatas"] or [[]])[0]
        distances = (stored["distances"] or [[]])[0]
        return [
            VisualHit.from_page(_stored_page(page_id, metadata), score=1.0 - float(distance))
            for page_id, metadata, distance in zip(
                ids,
                metadatas,
                distances,
                strict=True,
            )
        ]


def _page_metadata(page: VisualPage) -> dict[str, str | int]:
    return {
        "doc_id": page.doc_id,
        "doc_name": page.doc_name,
        "page_number": page.page_number,
        "image_path": page.image_path,
        "source_format": page.source_format,
    }


def _stored_page(page_id: str, metadata: dict) -> VisualPage:
    return VisualPage(
        doc_id=str(metadata["doc_id"]),
        doc_name=str(metadata["doc_name"]),
        page_number=int(metadata["page_number"]),
        image_path=str(metadata["image_path"]),
        source_format=str(metadata["source_format"]),
    )
