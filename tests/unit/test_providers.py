import json

import httpx
import pytest

from app.config import Settings
from app.domain.models import MultimodalInput
from app.infrastructure.providers import (
    DashScopeEmbeddingProvider,
    DashScopeRerankerProvider,
    DashScopeVLMProvider,
    MockEmbeddingProvider,
    OpenAICompatibleVLMProvider,
    PaddleOCRProvider,
    VLLMEmbeddingProvider,
    VLLMRerankerProvider,
)


@pytest.mark.asyncio
async def test_mock_embedding_is_deterministic_and_semantic() -> None:
    provider = MockEmbeddingProvider(128)
    first = await provider.embed_query(MultimodalInput(text="多模态检索"))
    second = await provider.embed_query(MultimodalInput(text="多模态检索"))
    unrelated = await provider.embed_query(MultimodalInput(text="天气预报"))
    assert first == second
    assert sum(a * b for a, b in zip(first, second, strict=True)) > sum(
        a * b for a, b in zip(first, unrelated, strict=True)
    )


@pytest.mark.asyncio
async def test_paddleocr_provider_submits_polls_and_parses_jsonl() -> None:
    polls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal polls
        if request.method == "POST":
            assert request.headers["Authorization"] == "bearer token"
            return httpx.Response(200, json={"data": {"jobId": "job-1"}})
        if str(request.url) == "https://result.local/result.jsonl":
            line = {
                "result": {
                    "layoutParsingResults": [{"markdown": {"text": "识别结果"}}]
                }
            }
            return httpx.Response(200, text=json.dumps(line, ensure_ascii=False))
        polls += 1
        state = "running" if polls == 1 else "done"
        data = {"state": state}
        if state == "done":
            data["resultUrl"] = {"jsonUrl": "https://result.local/result.jsonl"}
        return httpx.Response(200, json={"data": data})

    settings = Settings(
        _env_file=None,
        ocr_provider="paddleocr",
        paddleocr_access_token="token",
        paddleocr_poll_interval_seconds=0,
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = PaddleOCRProvider(settings, client)
    result = await provider.recognize(b"image", language=["ch"])
    await client.aclose()
    assert result.normalized_text == "识别结果"
    assert result.request_id == "job-1"


@pytest.mark.asyncio
async def test_dashscope_vlm_parses_structured_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["model"] == "qwen3.7-flash"
        assert body["messages"][0]["content"][0]["type"] == "image_url"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"summary":"架构图","entities":["服务"],'
                                '"relations":[],"chart_type":"diagram",'
                                '"search_terms":["RAG"],"warnings":[]}'
                            )
                        }
                    }
                ]
            },
        )

    settings = Settings(
        _env_file=None,
        vlm_provider="dashscope",
        dashscope_api_key="key",
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = DashScopeVLMProvider(settings, client)
    result = await provider.describe_image(b"\x89PNG\r\n", caption=None, section_path=[])
    await client.aclose()
    assert result.summary == "架构图"
    assert result.chart_type == "diagram"


@pytest.mark.asyncio
async def test_openai_compatible_vlm_sends_vllm_request_and_parses_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://vlm.local/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer local-key"
        body = json.loads(request.content)
        assert body["model"] == "qwen-vl"
        assert body["max_tokens"] == 512
        assert "enable_thinking" not in body
        image_url = body["messages"][0]["content"][0]["image_url"]["url"]
        assert image_url.startswith("data:image/png;base64,")
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"summary":"人民建议征集表","entities":["居民"],'
                                '"relations":[],"chart_type":"other",'
                                '"search_terms":["人民建议"],"warnings":[]}'
                            )
                        }
                    }
                ]
            },
        )

    settings = Settings(
        _env_file=None,
        vlm_provider="vllm",
        vlm_base_url="http://vlm.local/v1/",
        vlm_api_key="local-key",
        vlm_model="qwen-vl",
        vlm_max_tokens=512,
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleVLMProvider(settings, client)
    result = await provider.describe_image(
        b"\x89PNG\r\n", caption="表单", section_path=["附件"]
    )
    await client.aclose()

    assert result.summary == "人民建议征集表"
    assert result.entities == ["居民"]


@pytest.mark.asyncio
async def test_openai_compatible_vlm_allows_vllm_without_api_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "Authorization" not in request.headers
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"summary":"图片"}'}}]},
        )

    settings = Settings(
        _env_file=None,
        vlm_provider="openai_compatible",
        vlm_api_key="",
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleVLMProvider(settings, client)
    result = await provider.describe_image(b"image", caption=None, section_path=[])
    await client.aclose()

    assert result.summary == "图片"


@pytest.mark.asyncio
async def test_vllm_embedding_uses_multimodal_chat_embeddings_api() -> None:
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://embedding.local/v1/embeddings"
        assert request.headers["Authorization"] == "Bearer embedding-key"
        body = json.loads(request.content)
        requests.append(body)
        assert body["continue_final_message"] is True
        assert body["add_special_tokens"] is True
        assert body["messages"][0][0]["role"] == "system"
        assert body["messages"][0][0]["content"][0]["text"] == "query instruction"
        image_url = body["messages"][1][1]["content"][0]["image_url"]["url"]
        assert image_url.startswith("data:image/png;base64,")
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.0, 2.0, 0.0]},
                    {"index": 0, "embedding": [3.0, 0.0, 0.0]},
                ]
            },
        )

    settings = Settings(
        _env_file=None,
        embedding_provider="vllm",
        embedding_dimension=3,
        embedding_vllm_base_url="http://embedding.local/v1/",
        embedding_vllm_api_key="embedding-key",
        embedding_vllm_model="embedding-model",
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = VLLMEmbeddingProvider(settings, client)
    result = await provider.embed_documents(
        [
            MultimodalInput(text="query", instruction="query instruction"),
            MultimodalInput(text="document", image=b"\x89PNG\r\n"),
        ]
    )
    await client.aclose()

    assert len(requests) == 1
    assert result == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]


@pytest.mark.asyncio
async def test_vllm_reranker_uses_multimodal_rerank_api() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://reranker.local/v1/rerank"
        assert "Authorization" not in request.headers
        body = json.loads(request.content)
        assert body["model"] == "reranker-model"
        assert body["query"] == "人民建议"
        assert body["documents"][0] == "无关内容"
        content = body["documents"][1]["content"]
        assert content[0] == {"type": "text", "text": "建议征集表"}
        assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
        assert body["top_n"] == 2
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 1, "relevance_score": 0.91},
                    {"index": 0, "relevance_score": 0.12},
                ]
            },
        )

    settings = Settings(
        _env_file=None,
        reranker_provider="vllm",
        reranker_vllm_base_url="http://reranker.local/v1/",
        reranker_vllm_api_key="",
        reranker_vllm_model="reranker-model",
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = VLLMRerankerProvider(settings, client)
    result = await provider.rerank(
        MultimodalInput(text="人民建议"),
        [
            MultimodalInput(text="无关内容"),
            MultimodalInput(text="建议征集表", image=b"\x89PNG\r\n"),
        ],
        top_n=2,
    )
    await client.aclose()

    assert [(item.index, item.score) for item in result] == [(1, 0.91), (0, 0.12)]


@pytest.mark.asyncio
async def test_dashscope_embedding_requests_one_fused_vector() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://dashscope.local/multimodal-embedding"
        assert request.headers["Authorization"] == "Bearer dashscope-key"
        body = json.loads(request.content)
        assert body["model"] == "qwen3-vl-embedding"
        assert body["input"]["contents"][0] == {
            "text": "document instruction\n图片说明"
        }
        image = body["input"]["contents"][1]["image"]
        assert image.startswith("data:image/png;base64,")
        assert body["parameters"] == {"enable_fusion": True, "dimension": 3}
        return httpx.Response(
            200,
            json={
                "output": {
                    "embeddings": [
                        {"index": 0, "embedding": [0.0, 4.0, 0.0], "type": "fused"}
                    ]
                }
            },
        )

    settings = Settings(
        _env_file=None,
        embedding_provider="dashscope",
        embedding_dimension=3,
        dashscope_api_key="dashscope-key",
        dashscope_embedding_url="https://dashscope.local/multimodal-embedding",
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = DashScopeEmbeddingProvider(settings, client)
    result = await provider.embed_query(
        MultimodalInput(
            text="图片说明",
            image=b"\x89PNG\r\n",
            instruction="document instruction",
        )
    )
    await client.aclose()

    assert result == [0.0, 1.0, 0.0]


@pytest.mark.asyncio
async def test_dashscope_reranker_uses_native_nested_protocol() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://dashscope.local/text-rerank"
        assert request.headers["Authorization"] == "Bearer dashscope-key"
        body = json.loads(request.content)
        assert body["model"] == "qwen3-vl-rerank"
        assert body["input"]["query"] == {"text": "人民建议"}
        assert body["input"]["documents"][0] == {"text": "无关内容"}
        image = body["input"]["documents"][1]["image"]
        assert image.startswith("data:image/png;base64,")
        assert body["parameters"]["return_documents"] is False
        assert body["parameters"]["top_n"] == 2
        assert body["parameters"]["instruct"].startswith("Retrieve text passages")
        return httpx.Response(
            200,
            json={
                "output": {
                    "results": [
                        {"index": 1, "relevance_score": 0.88},
                        {"index": 0, "relevance_score": 0.21},
                    ]
                }
            },
        )

    settings = Settings(
        _env_file=None,
        reranker_provider="dashscope",
        dashscope_api_key="dashscope-key",
        dashscope_reranker_url="https://dashscope.local/text-rerank",
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = DashScopeRerankerProvider(settings, client)
    result = await provider.rerank(
        MultimodalInput(text="人民建议"),
        [
            MultimodalInput(text="无关内容"),
            MultimodalInput(text="图片说明", image=b"\x89PNG\r\n"),
        ],
        top_n=2,
    )
    await client.aclose()

    assert [(item.index, item.score) for item in result] == [(1, 0.88), (0, 0.21)]
