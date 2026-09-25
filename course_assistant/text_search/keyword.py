"""In-memory BM25 keyword retrieval for course-material text."""

import math
import re
from collections import Counter
from collections.abc import Iterable

from course_assistant.text_search.chunking import chunk_pages
from course_assistant.text_search.models import PageLike, SearchResult, TextChunk

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def tokenize(text: str) -> list[str]:
    """Tokenize text consistently for both indexing and querying."""

    return _TOKEN_PATTERN.findall(text.lower())


class KeywordIndex:
    """A small BM25 index whose entries can be replaced by ``doc_id``."""

    def __init__(
        self,
        *,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k1 = k1
        self.b = b
        self._chunks: dict[str, TextChunk] = {}

    @property
    def chunks(self) -> tuple[TextChunk, ...]:
        return tuple(self._chunks.values())

    def index_pages(self, pages: Iterable[PageLike]) -> list[TextChunk]:
        """Chunk and index pages, replacing existing entries for their documents."""

        page_list = list(pages)
        for doc_id in {page.doc_id for page in page_list}:
            self.delete_document(doc_id)

        chunks = chunk_pages(
            page_list,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        self._chunks.update({chunk.chunk_id: chunk for chunk in chunks})
        return chunks

    def delete_document(self, doc_id: str) -> None:
        """Delete every chunk for ``doc_id``; missing IDs are a successful no-op."""

        self._chunks = {
            chunk_id: chunk
            for chunk_id, chunk in self._chunks.items()
            if chunk.doc_id != doc_id
        }

    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        """Return the highest-scoring BM25 passages for ``query``."""

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        query_terms = tokenize(query)
        chunks = list(self._chunks.values())
        if not query_terms or not chunks:
            return []

        tokenized = [tokenize(chunk.text) for chunk in chunks]
        lengths = [len(tokens) for tokens in tokenized]
        average_length = sum(lengths) / len(lengths)
        document_frequency = Counter(
            term for tokens in tokenized for term in set(tokens)
        )

        scored: list[tuple[float, TextChunk]] = []
        for chunk, tokens, length in zip(chunks, tokenized, lengths, strict=True):
            frequencies = Counter(tokens)
            score = sum(
                self._term_score(
                    term,
                    frequency=frequencies[term],
                    document_frequency=document_frequency[term],
                    document_count=len(chunks),
                    document_length=length,
                    average_length=average_length,
                )
                for term in query_terms
            )
            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
        return [
            SearchResult.from_chunk(chunk, score=score, search_method="bm25")
            for score, chunk in scored[:top_k]
        ]

    def _term_score(
        self,
        term: str,
        *,
        frequency: int,
        document_frequency: int,
        document_count: int,
        document_length: int,
        average_length: float,
    ) -> float:
        if frequency == 0 or document_frequency == 0:
            return 0.0
        inverse_document_frequency = math.log(
            1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
        )
        length_ratio = document_length / average_length if average_length else 0.0
        denominator = frequency + self.k1 * (1 - self.b + self.b * length_ratio)
        return inverse_document_frequency * (frequency * (self.k1 + 1)) / denominator
