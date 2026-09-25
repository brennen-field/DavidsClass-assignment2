from dataclasses import dataclass
from pathlib import Path

from course_assistant import visual_search
from course_assistant.visual_search.embedding import VisualIndex

PUBLIC_EXPORTS = [
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


def test_public_api_is_exported():
    for name in PUBLIC_EXPORTS:
        assert hasattr(visual_search, name), f"missing export {name}"


@dataclass(frozen=True)
class PingPage:
    doc_id: str
    doc_name: str
    page_number: int
    text: str
    image_path: str
    source_format: str = "pptx"


class PingEmbedder:
    def _v(self, text):
        text = text.lower()
        return (float("chart" in text), 0.1)

    def embed_images(self, image_paths):
        return [self._v(Path(p).stem) for p in image_paths]

    def embed_text(self, text):
        return self._v(text)


class FakeStore:
    def __init__(self):
        self.add_listener = None
        self.deleter = None

    def on_document_added(self, listener):
        self.add_listener = listener

    def register_index_deleter(self, deleter):
        self.deleter = deleter


def test_connect_document_store_wires_the_visual_index():
    store = FakeStore()
    index = VisualIndex(PingEmbedder())
    visual_search.connect_document_store(store, index)

    store.add_listener(
        [PingPage("deck", "deck.pptx", 1, "", "/data/deck/chart.png")]
    )
    assert index.pages[0].page_number == 1

    store.deleter("deck")
    assert index.search("chart") == []
