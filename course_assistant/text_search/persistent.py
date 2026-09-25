"""Persistent Chroma storage for semantic text retrieval."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from course_assistant.text_search.chunking import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    chunk_pages,
)
from course_assistant.text_search.embedding import (
    Embedder,
    TextEmbeddingClient,
    _validate_vector_dimensions,
)
from course_assistant.text_search.models import PageLike, SearchResult, TextChunk


class ChromaSemanticIndex:
    """Persist chunk embeddings locally while retaining their source metadata."""

    def __init__(
        self,
        embedder: Embedder,
        *,
        path: str | Path,
        collection_name: str = "course_text_chunks",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be at least zero and smaller than chunk_size"
            )

        import chromadb

        self.embedder = embedder
        self.path = Path(path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._client = chromadb.PersistentClient(path=str(self.path))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @classmethod
    def from_environment(cls) -> "ChromaSemanticIndex":
        """Use the configured class endpoint and local application data folder."""

        from course_assistant import config

        return cls(
            TextEmbeddingClient.from_environment(),
            path=config.DATA_DIR / "text_vectors",
        )

    def __enter__(self) -> "ChromaSemanticIndex":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        """Release Chroma file handles, which is important on Windows."""

        self._client.close()

    @property
    def chunks(self) -> tuple[TextChunk, ...]:
        stored = self._collection.get(include=["documents", "metadatas"])
        return tuple(
            _stored_chunk(chunk_id, document, metadata)
            for chunk_id, document, metadata in zip(
                stored["ids"],
                stored["documents"] or [],
                stored["metadatas"] or [],
                strict=True,
            )
        )

    def index_pages(self, pages: Iterable[PageLike]) -> list[TextChunk]:
        """Embed pages and persist them, replacing stale chunks by document ID."""

        page_list = list(pages)
        doc_ids = {page.doc_id for page in page_list}
        chunks = chunk_pages(
            page_list,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        vectors = (
            self.embedder.embed_documents([chunk.text for chunk in chunks])
            if chunks
            else []
        )
        if len(vectors) != len(chunks):
            raise ValueError("The embedder returned an unexpected vector count")
        _validate_vector_dimensions(vectors)

        old_ids_by_doc = {
            doc_id: set(self._collection.get(where={"doc_id": doc_id}, include=[])["ids"])
            for doc_id in doc_ids
        }
        new_ids_by_doc = {
            doc_id: {chunk.chunk_id for chunk in chunks if chunk.doc_id == doc_id}
            for doc_id in doc_ids
        }

        # Upsert first so an indexing failure cannot erase the previous document.
        if chunks:
            self._collection.upsert(
                ids=[chunk.chunk_id for chunk in chunks],
                embeddings=[list(vector) for vector in vectors],
                documents=[chunk.text for chunk in chunks],
                metadatas=[_chunk_metadata(chunk) for chunk in chunks],
            )

        for doc_id in doc_ids:
            stale_ids = sorted(old_ids_by_doc[doc_id] - new_ids_by_doc[doc_id])
            if stale_ids:
                self._collection.delete(ids=stale_ids)
        return chunks

    def delete_document(self, doc_id: str) -> None:
        """Delete one document's persisted chunks; missing IDs are a no-op."""

        stored = self._collection.get(where={"doc_id": doc_id}, include=[])
        if stored["ids"]:
            self._collection.delete(ids=stored["ids"])

    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        """Return persisted chunks ordered by Chroma cosine similarity."""

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        query = query.strip()
        count = self._collection.count()
        if not query or count == 0:
            return []

        query_vector = self.embedder.embed_query(query)
        _validate_vector_dimensions([query_vector])
        stored = self._collection.query(
            query_embeddings=[list(query_vector)],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )
        ids = stored["ids"][0]
        documents = (stored["documents"] or [[]])[0]
        metadatas = (stored["metadatas"] or [[]])[0]
        distances = (stored["distances"] or [[]])[0]
        return [
            SearchResult.from_chunk(
                _stored_chunk(chunk_id, document, metadata),
                score=1.0 - float(distance),
                search_method="text_embedding",
            )
            for chunk_id, document, metadata, distance in zip(
                ids,
                documents,
                metadatas,
                distances,
                strict=True,
            )
        ]


def _chunk_metadata(chunk: TextChunk) -> dict[str, str | int]:
    return {
        "doc_id": chunk.doc_id,
        "doc_name": chunk.doc_name,
        "page_number": chunk.page_number,
        "chunk_index": chunk.chunk_index,
        "image_path": chunk.image_path,
        "source_format": chunk.source_format,
    }


def _stored_chunk(
    chunk_id: str,
    document: str,
    metadata: dict,
) -> TextChunk:
    return TextChunk(
        chunk_id=chunk_id,
        doc_id=str(metadata["doc_id"]),
        doc_name=str(metadata["doc_name"]),
        page_number=int(metadata["page_number"]),
        chunk_index=int(metadata["chunk_index"]),
        text=document,
        image_path=str(metadata["image_path"]),
        source_format=str(metadata["source_format"]),
    )
