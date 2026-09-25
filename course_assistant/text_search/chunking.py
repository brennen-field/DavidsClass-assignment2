"""Split processed pages into stable, source-preserving text chunks."""

import hashlib
import re
from collections.abc import Iterable

from course_assistant.text_search.models import PageLike, TextChunk

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100


def chunk_pages(
    pages: Iterable[PageLike],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[TextChunk]:
    """Return chunks for ``pages`` while keeping citation metadata.

    Empty or whitespace-only pages have no text passage to index and are
    skipped. Their page images remain available to the visual-search index.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be at least zero and smaller than chunk_size")

    chunks: list[TextChunk] = []
    for page in pages:
        if page.page_number < 1:
            raise ValueError("page_number must be 1-based")
        for chunk_index, text in enumerate(
            _split_text(page.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        ):
            chunks.append(
                TextChunk(
                    chunk_id=_chunk_id(page.doc_id, page.page_number, chunk_index, text),
                    doc_id=page.doc_id,
                    doc_name=page.doc_name,
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                    text=text,
                    image_path=page.image_path,
                    source_format=page.source_format,
                )
            )
    return chunks


def _split_text(text: str, *, chunk_size: int, chunk_overlap: int) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        if end < len(normalized):
            word_boundary = normalized.rfind(" ", start + 1, end + 1)
            if word_boundary > start:
                end = word_boundary

        passage = normalized[start:end].strip()
        if passage:
            chunks.append(passage)
        if end >= len(normalized):
            break

        next_start = max(0, end - chunk_overlap)
        if next_start > 0:
            previous_boundary = normalized.rfind(" ", 0, next_start + 1)
            if previous_boundary >= start:
                next_start = previous_boundary + 1
        start = next_start if next_start > start else end

    return chunks


def _chunk_id(doc_id: str, page_number: int, chunk_index: int, text: str) -> str:
    value = f"{doc_id}:{page_number}:{chunk_index}:{text}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()[:24]
