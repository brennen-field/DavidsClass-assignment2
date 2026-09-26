import base64

import pytest
import requests

from course_assistant.visual_search.rerank import (
    RerankItem,
    RerankScore,
    RerankServiceError,
    RerankerClient,
)

_TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/"
    "qM+36gAAAABJRU5ErkJggg=="
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


def scores_response():
    return {
        "results": [
            {"index": 0, "relevance_score": 0.2},
            {"index": 1, "relevance_score": 0.9},
        ]
    }


def make_client(response):
    return RerankerClient(
        base_url="http://dobolyi.com:9004",
        api_key="unit-test-placeholder",
        http_client=RecordingHttpClient(response),
    )


def test_client_sends_the_rerank_request_expected_by_the_class_service(tmp_path):
    image = tmp_path / "a.png"
    image.write_bytes(_TINY_PNG)
    http = RecordingHttpClient(FakeResponse(scores_response()))
    client = RerankerClient(
        base_url="http://dobolyi.com:9004/v2/rerank",
        api_key="unit-test-placeholder",
        http_client=http,
    )

    client.rerank(
        "Which chart shows growth?",
        [RerankItem(text="Sales grew 20%", image_path=str(image)), RerankItem(text="Plain passage")],
    )

    url, request = http.calls[0]
    assert url == "http://dobolyi.com:9004/v2/rerank"
    assert request["headers"]["Authorization"] == "Bearer unit-test-placeholder"
    payload = request["json"]
    assert payload["model"] == "Qwen/Qwen3-VL-Reranker-2B"
    assert payload["query"] == {
        "content": [{"type": "text", "text": "Which chart shows growth?"}]
    }
    doc0, doc1 = payload["documents"]
    assert doc0["content"][0] == {"type": "text", "text": "Sales grew 20%"}
    assert doc0["content"][1]["image_url"]["url"].startswith(
        "data:image/png;base64," + base64.b64encode(_TINY_PNG).decode()
    )
    assert doc1["content"] == [{"type": "text", "text": "Plain passage"}]
    assert payload["return_documents"] is False


def test_client_returns_scores_sorted_descending_and_index_aligned():
    client = make_client(FakeResponse(scores_response()))

    scores = client.rerank("query", [RerankItem(text="a"), RerankItem(text="b")])

    assert scores == [
        RerankScore(index=1, relevance_score=0.9),
        RerankScore(index=0, relevance_score=0.2),
    ]


def test_client_sends_top_n_and_underscores_the_service():
    http = RecordingHttpClient(
        FakeResponse({"results": [{"index": 1, "relevance_score": 0.9}]})
    )
    client = RerankerClient(
        base_url="http://dobolyi.com:9004",
        api_key="unit-test-placeholder",
        http_client=http,
    )

    docs = [RerankItem(text="a"), RerankItem(text="b")]
    client.rerank("q", docs, top_n=1)

    assert http.calls[0][1]["json"]["top_n"] == 1


def test_empty_documents_returns_empty_list_without_calling():
    http = RecordingHttpClient(FakeResponse({}))
    client = make_client(http)
    assert client.rerank("q", []) == []
    assert http.calls == []


def test_client_rejects_empty_query():
    client = make_client(FakeResponse({}))
    with pytest.raises(ValueError, match="non-empty"):
        client.rerank("   ", [RerankItem(text="a")])
    with pytest.raises(ValueError, match="top_n"):
        client.rerank("q", [RerankItem(text="a")], top_n=0)


def test_candidate_must_have_text_or_image():
    client = make_client(FakeResponse({}))
    with pytest.raises(ValueError, match="text and/or an image"):
        client.rerank("q", [RerankItem()])


def test_client_does_not_expose_the_key_in_repr_or_error():
    http_error_client = make_client(
        FakeResponse(error=requests.ConnectionError("offline"))
    )
    assert "unit-test-placeholder" not in repr(http_error_client)
    with pytest.raises(RerankServiceError) as captured:
        http_error_client.rerank("q", [RerankItem(text="a")])
    assert "unit-test-placeholder" not in str(captured.value)


@pytest.mark.parametrize(
    "response",
    [
        FakeResponse(error=requests.ConnectionError("offline")),
        FakeResponse({"unexpected": []}),
        FakeResponse(
            {
                "results": [
                    {"index": 0, "relevance_score": 0.5},
                    {"index": 5, "relevance_score": 0.4},  # index out of range
                ]
            }
        ),
        FakeResponse({"results": [{"index": 0, "relevance_score": float("nan")}]}),
    ],
)
def test_client_reports_safe_errors_for_bad_responses(response):
    client = make_client(response)
    with pytest.raises(RerankServiceError) as captured:
        client.rerank("q", [RerankItem(text="a"), RerankItem(text="b")])
    assert "unit-test-placeholder" not in str(captured.value)


def test_client_rejects_missing_configuration():
    with pytest.raises(ValueError, match="URL is not configured"):
        RerankerClient(base_url="", api_key="k")
    with pytest.raises(ValueError, match="CLASS_API_KEY"):
        RerankerClient(base_url="http://dobolyi.com:9004", api_key="")


def test_top_n_accepts_a_valid_subset_of_results():
    # top_n=1 of 2 documents -> the service legitimately returns ONE result.
    http = RecordingHttpClient(FakeResponse({"results": [{"index": 1, "relevance_score": 0.9}]}))
    client = RerankerClient(
        base_url="http://dobolyi.com:9004",
        api_key="unit-test-placeholder",
        http_client=http,
    )

    scores = client.rerank("q", [RerankItem(text="a"), RerankItem(text="b")], top_n=1)

    assert scores == [RerankScore(index=1, relevance_score=0.9)]
    assert http.calls[0][1]["json"]["top_n"] == 1


def test_top_n_rejects_duplicate_indexes():
    client = make_client(
        FakeResponse(
            {
                "results": [
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 1, "relevance_score": 0.8},
                ]
            }
        )
    )
    with pytest.raises(RerankServiceError):
        client.rerank("q", [RerankItem(text="a"), RerankItem(text="b")])


def test_top_n_rejects_out_of_range_indexes():
    client = make_client(FakeResponse({"results": [{"index": 5, "relevance_score": 0.9}]}))
    with pytest.raises(RerankServiceError):
        client.rerank("q", [RerankItem(text="a"), RerankItem(text="b")], top_n=1)


def test_top_n_rejects_wrong_subset_size():
    client = make_client(FakeResponse({"results": []}))
    with pytest.raises(RerankServiceError):
        client.rerank("q", [RerankItem(text="a"), RerankItem(text="b")], top_n=1)
