"""Visual-embedding client for page images (owner: Shrihari).

Mirrors the text-embedding client (credential-safe errors, batching,
``__repr__`` redaction) but targets the class *multimodal* embedding service at
``:9003``. The Qwen3-VL-Embedding model serves one ``input_type`` (``"default"``)
for both text and images, and embeds images as inline base64 data URLs.
"""

from __future__ import annotations

import base64
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import requests

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
