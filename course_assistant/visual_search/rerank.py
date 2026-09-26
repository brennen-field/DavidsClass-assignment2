"""Multimodal reranker client for the class reranking service (owner: Shrihari)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import requests

from course_assistant.visual_search.embedding import _data_url

DEFAULT_RERANK_MODEL = "Qwen/Qwen3-VL-Reranker-2B"
DEFAULT_TIMEOUT_SECONDS = 120.0


class RerankServiceError(RuntimeError):
    """A safe, credential-free error raised for reranking service failures."""


@dataclass(frozen=True)
class RerankItem:
    """One candidate to score against the query: text, image, or both."""

    text: str = ""
    image_path: str | None = None


@dataclass(frozen=True)
class RerankScore:
    """One reranked result; ``index`` maps back onto the input candidate."""

    index: int
    relevance_score: float


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


class RerankerClient:
    """Score retrieval candidates via the class multimodal ``/v2/rerank`` service.

    Candidates may carry text, an image path, or both; images are sent as inline
    base64 data URLs. The caller keeps the returned ``RerankScore.index`` values
    aligned with the documents they passed in (order preserved).
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str = DEFAULT_RERANK_MODEL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        http_client: _HttpClient | None = None,
    ) -> None:
        if not base_url.strip():
            raise ValueError("The reranking URL is not configured")
        if not api_key.strip():
            raise ValueError("CLASS_API_KEY is not configured")
        if not model.strip():
            raise ValueError("The reranking model is not configured")
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")

        self.endpoint = _reranking_endpoint(base_url)
        self._api_key = api_key
        self.model = model
        self.timeout = timeout
        self._http_client = http_client or requests

    @classmethod
    def from_environment(cls) -> "RerankerClient":
        """Build a client from values loaded by ``course_assistant.config``."""

        from course_assistant import config

        return cls(
            base_url=config.RERANK_URL,
            api_key=config.CLASS_API_KEY,
            model=config.RERANK_MODEL,
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(endpoint={self.endpoint!r}, "
            f"model={self.model!r}, api_key='[REDACTED]')"
        )

    def rerank(
        self,
        query_text: str,
        documents: Sequence[RerankItem],
        *,
        top_n: int | None = None,
    ) -> list[RerankScore]:
        """Rerank ``documents`` against ``query_text``.

        Returns scores ordered by descending relevance (ties broken by input
        index), each carrying the original candidate position.
        """

        query_text = query_text.strip()
        if not query_text:
            raise ValueError("The rerank query must be non-empty")
        documents = list(documents)
        if not documents:
            return []

        if top_n is not None and top_n <= 0:
            raise ValueError("top_n must be greater than zero")

        total = len(documents)
        payload: dict[str, object] = {
            "model": self.model,
            "query": {"content": [{"type": "text", "text": query_text}]},
            "documents": [_document_payload(item) for item in documents],
            "return_documents": False,
        }
        if top_n is not None:
            payload["top_n"] = top_n

        try:
            response = self._http_client.post(
                self.endpoint,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
        except (requests.RequestException, ValueError) as error:
            raise RerankServiceError("The reranking service request failed") from error

        # With top_n the service returns only that many results (a subset).
        expected = min(top_n, total) if top_n is not None else total
        return _parse_scores(data, expected_count=expected, total_count=total)


def _reranking_endpoint(base_url: str) -> str:
    normalized = base_url.strip().rstrip("/")
    if normalized.endswith("/v2/rerank"):
        return normalized
    return f"{normalized}/v2/rerank"


def _document_payload(item: RerankItem) -> dict[str, object]:
    content: list[dict[str, object]] = []
    if item.text and item.text.strip():
        content.append({"type": "text", "text": item.text.strip()})
    if item.image_path:
        content.append(
            {"type": "image_url", "image_url": {"url": _data_url(Path(item.image_path))}}
        )
    if not content:
        raise ValueError("A rerank candidate must provide text and/or an image path")
    return {"content": content}


def _parse_scores(data: object, *, expected_count: int, total_count: int) -> list[RerankScore]:
    try:
        results = data["results"]  # type: ignore[index]
        raw = [
            (int(entry["index"]), float(entry["relevance_score"]))
            for entry in results
        ]
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise RerankServiceError("The reranking service returned an invalid response") from error

    indices = [index for index, _ in raw]
    if len(raw) != expected_count:
        raise RerankServiceError(
            "The reranking service returned an unexpected number of scores"
        )
    if len(set(indices)) != len(indices):
        raise RerankServiceError("The reranking service returned duplicate result indexes")
    if any(index < 0 or index >= total_count for index in indices):
        raise RerankServiceError("The reranking service returned an out-of-range result index")
    if any(not math.isfinite(score) for _, score in raw):
        raise RerankServiceError("The reranking service returned a non-finite score")

    raw.sort(key=lambda entry: (-entry[1], entry[0]))
    return [RerankScore(index=index, relevance_score=score) for index, score in raw]
