"""Visual-embedding client for page images (owner: Shrihari).

Mirrors the text-embedding client (credential-safe errors, batching,
``__repr__`` redaction) but targets the class *multimodal* embedding service at
``:9003``. The Qwen3-VL-Embedding model serves one ``input_type`` (``"default"``)
for both text and images, and embeds images as inline base64 data URLs.
"""

from __future__ import annotations

import base64
import hashlib
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Protocol

import requests

from course_assistant.text_search.models import PageLike
from course_assistant.visual_search.models import VisualHit, VisualPage

DEFAULT_VISUAL_EMBED_MODEL = "Qwen/Qwen3-VL-Embedding-2B"
DEFAULT_TIMEOUT_SECONDS = 120.0
# Images are token-heavy, so stay well under the model's 8192-token context.
DEFAULT_BATCH_SIZE = 4

_MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


class VisualEmbeddingServiceError(RuntimeError):
    """A safe, credential-free error raised for visual-embedding failures."""


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


class VisualEmbeddingClient:
    """Call the class vLLM ``/v2/embed`` service without exposing its key."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str = DEFAULT_VISUAL_EMBED_MODEL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        http_client: _HttpClient | None = None,
    ) -> None:
        if not base_url.strip():
            raise ValueError("The visual-embedding URL is not configured")
        if not api_key.strip():
            raise ValueError("CLASS_API_KEY is not configured")
        if not model.strip():
            raise ValueError("The visual-embedding model is not configured")
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
    def from_environment(cls) -> "VisualEmbeddingClient":
        """Build a client from values loaded by ``course_assistant.config``."""

        from course_assistant import config

        return cls(
            base_url=config.VISUAL_EMBED_URL,
            api_key=config.CLASS_API_KEY,
            model=config.VISUAL_EMBED_MODEL,
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(endpoint={self.endpoint!r}, "
            f"model={self.model!r}, api_key='[REDACTED]')"
        )

    def embed_images(self, image_paths: Sequence[str | Path]) -> list[tuple[float, ...]]:
        """Embed page images; one vector per image, preserving input order.

        Images are read from disk, base64-encoded, and sent as inline data URLs
        so the service never needs to reach the user's filesystem.
        """

        paths = [Path(path) for path in image_paths]
        if not paths:
            return []

        data_urls = [_data_url(path) for path in paths]
        vectors: list[tuple[float, ...]] = []
        for start in range(0, len(data_urls), self.batch_size):
            batch = data_urls[start : start + self.batch_size]
            vectors.extend(self._embed_images_batch(batch))
        return vectors

    def embed_text(self, text: str) -> tuple[float, ...]:
        """Embed one query into the same space as the page-image vectors."""

        text = text.strip()
        if not text:
            raise ValueError("Embedding inputs must contain non-empty text")
        inputs = [{"content": [{"type": "text", "text": text}]}]
        vectors = self._embed(inputs, expected_count=1)
        return vectors[0]

    def _embed_images_batch(self, data_urls: Sequence[str]) -> list[tuple[float, ...]]:
        inputs = [
            {
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    }
                ]
            }
            for data_url in data_urls
        ]
        return self._embed(inputs, expected_count=len(data_urls))

    def _embed(
        self,
        inputs: Sequence[dict[str, object]],
        *,
        expected_count: int,
    ) -> list[tuple[float, ...]]:
        try:
            response = self._http_client.post(
                self.endpoint,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "input_type": "default",
                    "inputs": list(inputs),
                    "embedding_types": ["float"],
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as error:
            raise VisualEmbeddingServiceError(
                "The visual-embedding service request failed"
            ) from error

        return _parse_embeddings(payload, expected_count=expected_count)


def _embedding_endpoint(base_url: str) -> str:
    normalized = base_url.strip().rstrip("/")
    if normalized.endswith("/v2/embed"):
        return normalized
    return f"{normalized}/v2/embed"


def _data_url(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"Image file not found: {path}")
    raw = path.read_bytes()
    if not raw:
        raise ValueError(f"Image file is empty: {path}")
    suffix = path.suffix.lower()
    mime = _MIME_BY_SUFFIX.get(suffix, "image/png")
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _parse_embeddings(
    payload: object,
    *,
    expected_count: int,
) -> list[tuple[float, ...]]:
    try:
        raw_vectors = payload["embeddings"]["float"]  # type: ignore[index]
        vectors = [tuple(float(value) for value in vector) for vector in raw_vectors]
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise VisualEmbeddingServiceError(
            "The visual-embedding service returned an invalid response"
        ) from error

    if len(vectors) != expected_count:
        raise VisualEmbeddingServiceError(
            "The visual-embedding service returned an unexpected vector count"
        )
    _validate_vector_dimensions(vectors)
    return vectors


def _validate_vector_dimensions(vectors: Sequence[Sequence[float]]) -> None:
    if not vectors:
        return
    dimension = len(vectors[0])
    if dimension == 0 or any(len(vector) != dimension for vector in vectors):
        raise VisualEmbeddingServiceError(
            "The visual-embedding service returned inconsistent vector dimensions"
        )
    if any(not math.isfinite(value) for vector in vectors for value in vector):
        raise VisualEmbeddingServiceError(
            "The visual-embedding service returned a non-finite vector value"
        )


class VisualEmbedder(Protocol):
    """The subset of the client used by visual indexes (supports fakes)."""

    def embed_images(self, image_paths: Sequence[str | Path]) -> list[tuple[float, ...]]: ...

    def embed_text(self, text: str) -> tuple[float, ...]: ...


class VisualIndex:
    """An in-memory index over page/slide image vectors.

    One vector per page (not per text chunk): the visual index owns retrieval
    for slides whose ``text`` is empty, and provides image-level evidence that
    the combine-and-rerank stage maps back onto text chunks by ``doc_id`` +
    ``page_number``.
    """

    def __init__(self, embedder: VisualEmbedder) -> None:
        self.embedder = embedder
        self._pages: dict[str, VisualPage] = {}
        self._vectors: dict[str, tuple[float, ...]] = {}

    @property
    def pages(self) -> tuple[VisualPage, ...]:
        return tuple(self._pages.values())

    def index_pages(self, pages: Iterable[PageLike]) -> list[VisualPage]:
        """Embed each page's image and atomically replace entries for its doc.

        Pages without an image are skipped, leaving the visual index to own
        only pages that actually have slide images.
        """

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

        # Do not discard an existing document until its replacement embeddings
        # have been generated and validated successfully.
        for doc_id in doc_ids:
            self.delete_document(doc_id)
        for page, vector in zip(indexed, vectors, strict=True):
            page_id = _page_id(page.doc_id, page.page_number)
            self._pages[page_id] = page
            self._vectors[page_id] = vector
        return indexed

    def delete_document(self, doc_id: str) -> None:
        """Delete every page vector for ``doc_id``; missing IDs are a no-op."""

        remove_ids = {
            page_id
            for page_id, page in self._pages.items()
            if page.doc_id == doc_id
        }
        for page_id in remove_ids:
            self._pages.pop(page_id, None)
            self._vectors.pop(page_id, None)

    def search(self, query: str, *, top_k: int = 5) -> list[VisualHit]:
        """Return page images ranked by cosine similarity to ``query``."""

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        query = query.strip()
        if not query or not self._pages:
            return []

        query_vector = self.embedder.embed_text(query)
        _validate_vector_dimensions([query_vector, *self._vectors.values()])
        scored = [
            (_cosine_similarity(query_vector, self._vectors[page_id]), page)
            for page_id, page in self._pages.items()
        ]
        scored.sort(key=lambda item: (-item[0], item[1].page_number))
        return [
            VisualHit.from_page(page, score=score)
            for score, page in scored[:top_k]
        ]


def _visual_page(page: PageLike) -> VisualPage:
    return VisualPage(
        doc_id=page.doc_id,
        doc_name=page.doc_name,
        page_number=page.page_number,
        image_path=page.image_path,
        source_format=page.source_format,
    )


def _page_id(doc_id: str, page_number: int) -> str:
    value = f"{doc_id}:{page_number}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()[:24]


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (
        left_norm * right_norm
    )
