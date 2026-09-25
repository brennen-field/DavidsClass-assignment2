"""Data contracts owned by text search."""

from dataclasses import asdict, dataclass
from typing import Protocol


class PageLike(Protocol):
    """The fields text search consumes from Chase's ``Page`` record."""

    doc_id: str
    doc_name: str
    page_number: int
    text: str
    image_path: str
    source_format: str


@dataclass(frozen=True)
class TextChunk:
    """A searchable passage that retains its original source details."""

    chunk_id: str
    doc_id: str
    doc_name: str
    page_number: int
    chunk_index: int
    text: str
    image_path: str
    source_format: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SearchResult:
    """One text-search candidate passed to combine-and-rerank."""

    chunk_id: str
    doc_id: str
    doc_name: str
    page_number: int
    text: str
    image_path: str
    source_format: str
    score: float
    search_method: str

    @classmethod
    def from_chunk(
        cls,
        chunk: TextChunk,
        *,
        score: float,
        search_method: str,
    ) -> "SearchResult":
        return cls(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            doc_name=chunk.doc_name,
            page_number=chunk.page_number,
            text=chunk.text,
            image_path=chunk.image_path,
            source_format=chunk.source_format,
            score=score,
            search_method=search_method,
        )

    def to_dict(self) -> dict:
        return asdict(self)
