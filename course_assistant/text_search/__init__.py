"""Text search (owner: Nick).

The first milestone provides source-preserving chunking and BM25 keyword
retrieval. Text-embedding retrieval will be added behind the same result
contract once the class service details are available.
"""

from course_assistant.text_search.chunking import chunk_pages
from course_assistant.text_search.integration import connect_document_store
from course_assistant.text_search.keyword import KeywordIndex
from course_assistant.text_search.models import SearchResult, TextChunk

__all__ = [
    "KeywordIndex",
    "SearchResult",
    "TextChunk",
    "chunk_pages",
    "connect_document_store",
]
