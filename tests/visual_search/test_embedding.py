import base64

import pytest
import requests

from course_assistant.visual_search.embedding import (
    VisualEmbeddingClient,
    VisualEmbeddingServiceError,
)

# A tiny valid 1x1 PNG (red pixel).
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


def write_image(tmp_path, name="a.png", data=None):
    path = tmp_path / name
    path.write_bytes(_TINY_PNG if data is None else data)
    return path


def one_vector(value=1.0, dimension=2):
    return [float(value)] * dimension


def test_client_sends_the_image_request_expected_by_the_class_service(tmp_path):
    first = write_image(tmp_path, "a.png")
    second = write_image(tmp_path, "b.jpg")
    http = RecordingHttpClient(
        FakeResponse(
            {
                "embeddings": {
                    "float": [one_vector(), one_vector()],
                }
            }
        )
    )
    client = VisualEmbeddingClient(
        base_url="http://dobolyi.com:9003/v1",
        api_key="unit-test-placeholder",
        http_client=http,
    )

    vectors = client.embed_images([first, second])

    assert vectors == [tuple(one_vector()), tuple(one_vector())]
    url, request = http.calls[0]
    assert url == "http://dobolyi.com:9003/v1/v2/embed"
    assert request["headers"]["Authorization"] == "Bearer unit-test-placeholder"
    payload = request["json"]
    assert payload["model"] == "Qwen/Qwen3-VL-Embedding-2B"
    assert payload["input_type"] == "default"
    assert payload["embedding_types"] == ["float"]
    contents = [
        item["content"] for item in payload["inputs"]
    ]
    assert contents[0] == [
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64," + base64.b64encode(_TINY_PNG).decode()},
        }
    ]
    # The jpg file should be labelled with its correct mime type.
    assert contents[1][0]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_client_embeds_a_text_query_through_the_same_endpoint():
    http = RecordingHttpClient(FakeResponse({"embeddings": {"float": [[0.2, 0.8]]}}))
    client = VisualEmbeddingClient(
        base_url="http://dobolyi.com:9003",
        api_key="unit-test-placeholder",
        http_client=http,
    )

    vector = client.embed_text("What does the chart show?")

    assert vector == (0.2, 0.8)
    payload = http.calls[0][1]["json"]
    assert payload["inputs"] == [
        {"content": [{"type": "text", "text": "What does the chart show?"}]}
    ]


def test_client_batches_large_image_collections(tmp_path):
    paths = [write_image(tmp_path, f"{i}.png") for i in range(3)]
    http = RecordingHttpClient(
        FakeResponse({"embeddings": {"float": [one_vector()]}})
    )
    client = VisualEmbeddingClient(
        base_url="http://dobolyi.com:9003",
        api_key="unit-test-placeholder",
        batch_size=1,
        http_client=http,
    )

    client.embed_images(paths)

    assert len(http.calls) == 3
    batch_sizes = [len(call[1]["json"]["inputs"]) for call in http.calls]
    assert batch_sizes == [1, 1, 1]


def test_client_does_not_expose_the_key_in_repr_or_error(tmp_path):
    image = write_image(tmp_path)

    http_error_client = VisualEmbeddingClient(
        base_url="http://dobolyi.com:9003",
        api_key="unit-test-placeholder",
        http_client=RecordingHttpClient(
            FakeResponse(error=requests.ConnectionError("offline"))
        ),
    )
    assert "unit-test-placeholder" not in repr(http_error_client)
    with pytest.raises(VisualEmbeddingServiceError) as captured:
        http_error_client.embed_images([image])
    assert "unit-test-placeholder" not in str(captured.value)


@pytest.mark.parametrize(
    "response",
    [
        FakeResponse(error=requests.ConnectionError("offline")),
        FakeResponse({"unexpected": []}),
        FakeResponse({"embeddings": {"float": [[1.0], [2.0]]}}),
    ],
)
def test_client_reports_safe_errors_for_unavailable_or_invalid_responses(
    response, tmp_path
):
    image = write_image(tmp_path)
    client = VisualEmbeddingClient(
        base_url="http://dobolyi.com:9003",
        api_key="unit-test-placeholder",
        http_client=RecordingHttpClient(response),
    )

    with pytest.raises(VisualEmbeddingServiceError) as captured:
        client.embed_images([image])

    assert "unit-test-placeholder" not in str(captured.value)


def test_missing_or_empty_image_reports_a_clear_error(tmp_path):
    missing = tmp_path / "nope.png"
    client = VisualEmbeddingClient(
        base_url="http://dobolyi.com:9003",
        api_key="unit-test-placeholder",
        http_client=RecordingHttpClient(FakeResponse({"embeddings": {"float": []}})),
    )
    with pytest.raises(ValueError, match="Image file not found"):
        client.embed_images([missing])

    empty = write_image(tmp_path, "empty.png", data=b"")
    with pytest.raises(ValueError, match="empty"):
        client.embed_images([empty])


def test_client_rejects_missing_configuration():
    with pytest.raises(ValueError, match="URL is not configured"):
        VisualEmbeddingClient(base_url="", api_key="k")
    with pytest.raises(ValueError, match="CLASS_API_KEY"):
        VisualEmbeddingClient(base_url="http://dobolyi.com:9003", api_key="")
