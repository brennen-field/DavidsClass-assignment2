"""Visual (page-image) retrieval and combine-and-rerank (owner: Shrihari).

Public surface for Preston and Brennen: ``combine_and_rerank(...)`` turns every
retriever's output into one ordered ``EvidenceItem`` list, and
``connect_document_store`` wires the visual indexes into Chase's document store.
"""

from course_assistant.text_search.integration import connect_document_store
from course_assistant.visual_search.combine import combine_and_rerank
from course_assistant.visual_search.embedding import (
    VisualEmbeddingClient,
    VisualEmbeddingServiceError,
    VisualIndex,
)
from course_assistant.visual_search.models import (
    EvidenceItem,
    VisualHit,
    VisualPage,
)
from course_assistant.visual_search.persistent import ChromaVisualIndex
from course_assistant.visual_search.rerank import (
    RerankItem,
    RerankScore,
    RerankServiceError,
    RerankerClient,
)

__all__ = [
    "ChromaVisualIndex",
    "EvidenceItem",
    "RerankItem",
    "RerankScore",
    "RerankServiceError",
    "RerankerClient",
    "VisualEmbeddingClient",
    "VisualEmbeddingServiceError",
    "VisualHit",
    "VisualIndex",
    "VisualPage",
    "combine_and_rerank",
    "connect_document_store",
]
