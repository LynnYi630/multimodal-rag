import json

import httpx
import pytest

from app.config import Settings
from app.domain.models import MultimodalInput
from app.infrastructure.providers import (
    DashScopeVLMProvider,
    MockEmbeddingProvider,
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

