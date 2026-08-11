import json

import httpx
import pytest

from app.config import Settings
from app.domain.models import MultimodalInput
from app.infrastructure.providers import (
    DashScopeVLMProvider,
    MockEmbeddingProvider,
    OpenAICompatibleVLMProvider,
    PaddleOCRProvider,
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
