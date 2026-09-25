"""Source-preserving BM25 and semantic text retrieval (owner: Nick)."""

from course_assistant.text_search.chunking import chunk_pages
from course_assistant.text_search.embedding import (
    EmbeddingServiceError,
    SemanticIndex,
    TextEmbeddingClient,
)
from course_assistant.text_search.integration import connect_document_store
from course_assistant.text_search.keyword import KeywordIndex
from course_assistant.text_search.models import SearchResult, TextChunk

__all__ = [
    "KeywordIndex",
    "EmbeddingServiceError",
    "SearchResult",
    "SemanticIndex",
    "TextChunk",
    "TextEmbeddingClient",
    "chunk_pages",
    "connect_document_store",
]
