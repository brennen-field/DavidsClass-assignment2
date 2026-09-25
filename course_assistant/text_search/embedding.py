"""Text-embedding client and in-memory semantic retrieval."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Protocol

import requests

from course_assistant.text_search.chunking import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    chunk_pages,
)
from course_assistant.text_search.models import PageLike, SearchResult, TextChunk

DEFAULT_TEXT_EMBED_MODEL = "nvidia/Nemotron-3-Embed-1B-BF16"
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_BATCH_SIZE = 32


class EmbeddingServiceError(RuntimeError):
    """A safe, credential-free error raised for embedding service failures."""


class _ResponseLike(Protocol):
    def raise_for_status(self) -> None: ...

    def json(self) -> object: ...


class _HttpClient(Protocol):
    def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, object],
        timeout: float,
    ) -> _ResponseLike: ...


class Embedder(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[tuple[float, ...]]: ...

    def embed_query(self, text: str) -> tuple[float, ...]: ...


class TextEmbeddingClient:
    """Call the class vLLM ``/v2/embed`` service without exposing its key."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str = DEFAULT_TEXT_EMBED_MODEL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        http_client: _HttpClient | None = None,
    ) -> None:
        if not base_url.strip():
            raise ValueError("The text-embedding URL is not configured")
        if not api_key.strip():
            raise ValueError("CLASS_API_KEY is not configured")
        if not model.strip():
            raise ValueError("The text-embedding model is not configured")
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        self.endpoint = _embedding_endpoint(base_url)
        self._api_key = api_key
        self.model = model
        self.timeout = timeout
        self.batch_size = batch_size
        self._http_client = http_client or requests

    @classmethod
    def from_environment(cls) -> "TextEmbeddingClient":
        """Build a client from values loaded by ``course_assistant.config``."""

        from course_assistant import config

        return cls(
            base_url=config.TEXT_EMBED_URL,
            api_key=config.CLASS_API_KEY,
            model=config.TEXT_EMBED_MODEL,
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(endpoint={self.endpoint!r}, "
            f"model={self.model!r}, api_key='[REDACTED]')"
        )

    def embed_documents(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        """Embed source passages using the model's document retrieval prompt."""

        vectors: list[tuple[float, ...]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(
                self._embed(
                    texts[start : start + self.batch_size],
                    input_type="document",
                )
            )
        return vectors

    def embed_query(self, text: str) -> tuple[float, ...]:
        """Embed one search query using the model's query retrieval prompt."""

        vectors = self._embed([text], input_type="query")
        return vectors[0]

    def _embed(
        self,
        texts: Sequence[str],
        *,
        input_type: str,
    ) -> list[tuple[float, ...]]:
        clean_texts = [text.strip() for text in texts]
        if not clean_texts or any(not text for text in clean_texts):
            raise ValueError("Embedding inputs must contain non-empty text")

        try:
            response = self._http_client.post(
                self.endpoint,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "input_type": input_type,
                    "texts": clean_texts,
                    "embedding_types": ["float"],
                    "truncate": "END",
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as error:
            raise EmbeddingServiceError(
                "The text-embedding service request failed"
            ) from error

        return _parse_embeddings(payload, expected_count=len(clean_texts))


class SemanticIndex:
    """An in-memory vector index that preserves each chunk's source details."""

    def __init__(
        self,
        embedder: Embedder,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be at least zero and smaller than chunk_size"
            )
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._chunks: dict[str, TextChunk] = {}
        self._vectors: dict[str, tuple[float, ...]] = {}

    @property
    def chunks(self) -> tuple[TextChunk, ...]:
        return tuple(self._chunks.values())

    def index_pages(self, pages: Iterable[PageLike]) -> list[TextChunk]:
        """Embed new chunks and atomically replace entries for their documents."""

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
            raise EmbeddingServiceError(
                "The text-embedding service returned an unexpected vector count"
            )
        _validate_vector_dimensions(vectors)

        # Do not discard an existing document until its replacement embeddings
        # have been generated and validated successfully.
        for doc_id in doc_ids:
            self.delete_document(doc_id)
        for chunk, vector in zip(chunks, vectors, strict=True):
            self._chunks[chunk.chunk_id] = chunk
            self._vectors[chunk.chunk_id] = vector
        return chunks

    def delete_document(self, doc_id: str) -> None:
        """Delete all chunks and vectors for a document; missing IDs are a no-op."""

        remove_ids = {
            chunk_id
            for chunk_id, chunk in self._chunks.items()
            if chunk.doc_id == doc_id
        }
        for chunk_id in remove_ids:
            self._chunks.pop(chunk_id, None)
            self._vectors.pop(chunk_id, None)

    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        """Return chunks ordered by cosine similarity to ``query``."""

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        query = query.strip()
        if not query or not self._chunks:
            return []

        query_vector = self.embedder.embed_query(query)
        _validate_vector_dimensions([query_vector, *self._vectors.values()])
        scored = [
            (_cosine_similarity(query_vector, self._vectors[chunk_id]), chunk)
            for chunk_id, chunk in self._chunks.items()
        ]
        scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
        return [
            SearchResult.from_chunk(
                chunk,
                score=score,
                search_method="text_embedding",
            )
            for score, chunk in scored[:top_k]
        ]


def _embedding_endpoint(base_url: str) -> str:
    normalized = base_url.strip().rstrip("/")
    if normalized.endswith("/v2/embed"):
        return normalized
    return f"{normalized}/v2/embed"


def _parse_embeddings(
    payload: object,
    *,
    expected_count: int,
) -> list[tuple[float, ...]]:
    try:
        raw_vectors = payload["embeddings"]["float"]  # type: ignore[index]
        vectors = [tuple(float(value) for value in vector) for vector in raw_vectors]
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise EmbeddingServiceError(
            "The text-embedding service returned an invalid response"
        ) from error

    if len(vectors) != expected_count:
        raise EmbeddingServiceError(
            "The text-embedding service returned an unexpected vector count"
        )
    _validate_vector_dimensions(vectors)
    return vectors


def _validate_vector_dimensions(vectors: Sequence[Sequence[float]]) -> None:
    if not vectors:
        return
    dimension = len(vectors[0])
    if dimension == 0 or any(len(vector) != dimension for vector in vectors):
        raise EmbeddingServiceError(
            "The text-embedding service returned inconsistent vector dimensions"
        )
    if any(not math.isfinite(value) for vector in vectors for value in vector):
        raise EmbeddingServiceError(
            "The text-embedding service returned a non-finite vector value"
        )


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (
        left_norm * right_norm
    )
