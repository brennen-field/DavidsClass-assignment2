from dataclasses import dataclass

import pytest
import requests

from course_assistant.text_search import (
    EmbeddingServiceError,
    SemanticIndex,
    TextEmbeddingClient,
    connect_document_store,
)


@dataclass(frozen=True)
class SamplePage:
    doc_id: str
    doc_name: str
    page_number: int
    text: str
    image_path: str
    source_format: str = "pdf"


def page(doc_id: str, page_number: int, text: str) -> SamplePage:
    return SamplePage(
        doc_id=doc_id,
        doc_name=f"{doc_id}.pdf",
        page_number=page_number,
        text=text,
        image_path=f"/data/{doc_id}/page-{page_number:03d}.png",
    )


class FakeResponse:
    def __init__(self, payload=None, error=None):
        self.payload = payload
        self.error = error

    def raise_for_status(self):
        if self.error:
            raise self.error

    def json(self):
        return self.payload


class RecordingHttpClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


def test_client_sends_the_document_request_expected_by_the_class_service():
    http = RecordingHttpClient(
        FakeResponse({"embeddings": {"float": [[1.0, 0.0], [0.0, 1.0]]}})
    )
    client = TextEmbeddingClient(
        base_url="http://dobolyi.com:9002",
        api_key="unit-test-placeholder",
        http_client=http,
    )

    vectors = client.embed_documents(["First passage", "Second passage"])

    assert vectors == [(1.0, 0.0), (0.0, 1.0)]
    url, request = http.calls[0]
    assert url == "http://dobolyi.com:9002/v2/embed"
    assert request["headers"]["Authorization"] == "Bearer unit-test-placeholder"
    assert request["json"] == {
        "model": "nvidia/Nemotron-3-Embed-1B-BF16",
        "input_type": "document",
        "texts": ["First passage", "Second passage"],
        "embedding_types": ["float"],
        "truncate": "END",
    }


def test_client_uses_query_input_type_and_does_not_show_the_key_in_repr():
    http = RecordingHttpClient(FakeResponse({"embeddings": {"float": [[0.2, 0.8]]}}))
    client = TextEmbeddingClient(
        base_url="http://dobolyi.com:9002/v2/embed",
        api_key="unit-test-placeholder",
        http_client=http,
    )

    assert client.embed_query("grading policy") == (0.2, 0.8)
    assert http.calls[0][1]["json"]["input_type"] == "query"
    assert "unit-test-placeholder" not in repr(client)


def test_client_batches_large_document_collections():
    http = RecordingHttpClient(FakeResponse({"embeddings": {"float": [[1.0, 0.0]]}}))
    client = TextEmbeddingClient(
        base_url="http://dobolyi.com:9002",
        api_key="unit-test-placeholder",
        batch_size=1,
        http_client=http,
    )

    assert client.embed_documents(["one", "two"]) == [(1.0, 0.0), (1.0, 0.0)]
    assert [call[1]["json"]["texts"] for call in http.calls] == [["one"], ["two"]]


@pytest.mark.parametrize(
    "response",
    [
        FakeResponse(error=requests.ConnectionError("offline")),
        FakeResponse({"unexpected": []}),
        FakeResponse({"embeddings": {"float": [[1.0], [2.0]]}}),
    ],
)
def test_client_reports_safe_errors_for_unavailable_or_invalid_responses(response):
    client = TextEmbeddingClient(
        base_url="http://dobolyi.com:9002",
        api_key="unit-test-placeholder",
        http_client=RecordingHttpClient(response),
    )

    with pytest.raises(EmbeddingServiceError) as captured:
        client.embed_query("question")

    assert "unit-test-placeholder" not in str(captured.value)


class MeaningEmbedder:
    def __init__(self):
        self.document_calls = []
        self.query_calls = []

    def _vector(self, text):
        text = text.lower()
        return (
            float("grading" in text or "grade" in text),
            float("office" in text),
            0.1,
        )

    def embed_documents(self, texts):
        self.document_calls.append(list(texts))
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        self.query_calls.append(text)
        return self._vector(text)


def test_semantic_index_returns_the_relevant_sourced_passage_first():
    embedder = MeaningEmbedder()
    index = SemanticIndex(embedder, chunk_size=500, chunk_overlap=50)
    index.index_pages(
        [
            page("syllabus", 1, "Office hours are Tuesday afternoon."),
            page("syllabus", 2, "Final grades include the team project."),
        ]
    )

    results = index.search("How is grading determined?", top_k=1)

    assert results[0].doc_id == "syllabus"
    assert results[0].page_number == 2
    assert results[0].image_path.endswith("page-002.png")
    assert results[0].search_method == "text_embedding"
    assert embedder.query_calls == ["How is grading determined?"]


def test_semantic_reindex_and_delete_are_complete_and_idempotent():
    index = SemanticIndex(MeaningEmbedder())
    index.index_pages([page("syllabus", 1, "Old grading rule")])
    old_chunk_id = index.chunks[0].chunk_id

    index.index_pages([page("syllabus", 2, "New grading rule")])

    assert len(index.chunks) == 1
    assert index.chunks[0].chunk_id != old_chunk_id
    assert index.chunks[0].page_number == 2

    index.delete_document("syllabus")
    index.delete_document("syllabus")
    assert index.search("grading") == []


def test_failed_reindex_keeps_existing_document_available():
    class FailingEmbedder(MeaningEmbedder):
        fail = False

        def embed_documents(self, texts):
            if self.fail:
                raise EmbeddingServiceError("service unavailable")
            return super().embed_documents(texts)

    embedder = FailingEmbedder()
    index = SemanticIndex(embedder)
    index.index_pages([page("syllabus", 1, "Existing grading rule")])
    embedder.fail = True

    with pytest.raises(EmbeddingServiceError):
        index.index_pages([page("syllabus", 2, "Replacement grading rule")])

    assert index.chunks[0].page_number == 1


def test_store_adapter_supports_the_semantic_index():
    class FakeStore:
        def on_document_added(self, listener):
            self.add_listener = listener

        def register_index_deleter(self, deleter):
            self.deleter = deleter

    store = FakeStore()
    index = SemanticIndex(MeaningEmbedder())
    connect_document_store(store, index)

    store.add_listener([page("syllabus", 3, "Grading details")])
    assert index.search("grades")[0].page_number == 3

    store.deleter("syllabus")
    assert index.search("grades") == []
