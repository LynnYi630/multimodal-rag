from __future__ import annotations

import asyncio
import base64
import hashlib
import importlib
import json
import re
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
from PIL import Image

from app.config import Settings
from app.domain.models import (
    ExternalProviderError,
    ImageDescription,
    MultimodalInput,
    OCRResult,
    RerankScore,
)
from app.domain.services import normalize


class DisabledOCRProvider:
    async def recognize(self, image: bytes, *, language: list[str]) -> OCRResult:
        return OCRResult(
            normalized_text="",
            provider="disabled",
            provider_version="disabled",
            request_id="",
            raw={},
        )


class PaddleOCRProvider:
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self.job_url = settings.paddleocr_job_url.rstrip("/")
        self.token = settings.paddleocr_access_token
        self.model = settings.paddleocr_model
        self.poll_interval = settings.paddleocr_poll_interval_seconds
        self.timeout = settings.paddleocr_timeout_seconds
        self.client = client or httpx.AsyncClient(timeout=httpx.Timeout(120, connect=15))
        self._owns_client = client is None

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    async def recognize(self, image: bytes, *, language: list[str]) -> OCRResult:
        headers = {"Authorization": f"bearer {self.token}"}
        optional_payload = {
            "useDocOrientationClassify": False,
            "useDocUnwarping": False,
            "useChartRecognition": False,
        }
        response = await self.client.post(
            self.job_url,
            headers=headers,
            data={
                "model": self.model,
                "optionalPayload": json.dumps(optional_payload),
            },
            files={"file": ("image.png", image, "image/png")},
        )
        self._raise_api_error(response, "submit OCR job")
        try:
            job_id = response.json()["data"]["jobId"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ExternalProviderError("PaddleOCR response does not contain data.jobId") from exc

        deadline = time.monotonic() + self.timeout
        job_data: dict[str, Any] = {}
        while time.monotonic() < deadline:
            result_response = await self.client.get(f"{self.job_url}/{job_id}", headers=headers)
            self._raise_api_error(result_response, "poll OCR job")
            job_data = result_response.json().get("data", {})
            state = job_data.get("state")
            if state == "done":
                break
            if state == "failed":
                raise ExternalProviderError(
                    f"PaddleOCR job failed: {job_data.get('errorMsg', 'unknown error')}"
                )
            if state not in {"pending", "running"}:
                raise ExternalProviderError(f"PaddleOCR returned unknown job state: {state!r}")
            await asyncio.sleep(self.poll_interval)
        else:
            raise ExternalProviderError(f"PaddleOCR job {job_id} timed out")

        result_url = (job_data.get("resultUrl") or {}).get("jsonUrl")
        if not result_url:
            raise ExternalProviderError("PaddleOCR completed without resultUrl.jsonUrl")
        jsonl_response = await self.client.get(result_url)
        self._raise_api_error(jsonl_response, "download OCR result")
        raw_pages: list[dict[str, Any]] = []
        texts: list[str] = []
        for raw_line in jsonl_response.text.splitlines():
            if not raw_line.strip():
                continue
            parsed = json.loads(raw_line)
            raw_pages.append(parsed)
            texts.extend(_extract_paddle_text(parsed))
        normalized = dict.fromkeys(text.strip() for text in texts if text.strip())
        return OCRResult(
            normalized_text="\n\n".join(normalized),
            provider="paddleocr_aistudio",
            provider_version=self.model,
            request_id=job_id,
            raw={"job": job_data, "pages": raw_pages},
        )

    @staticmethod
    def _raise_api_error(response: httpx.Response, action: str) -> None:
        if response.is_success:
            return
        body = response.text[:500]
        raise ExternalProviderError(
            f"failed to {action}: HTTP {response.status_code}: {body}"
        )


def _extract_paddle_text(payload: dict[str, Any]) -> list[str]:
    result = payload.get("result", payload)
    values: list[str] = []
    direct = result.get("markdownText") or result.get("markdown_text")
    if isinstance(direct, str):
        values.append(direct)
    layouts = result.get("layoutParsingResults") or result.get("layout_parsing_results") or []
    for layout in layouts:
        markdown = layout.get("markdown")
        if isinstance(markdown, dict) and isinstance(markdown.get("text"), str):
            values.append(markdown["text"])
        elif isinstance(markdown, str):
            values.append(markdown)
        pruned = layout.get("prunedResult")
        if isinstance(pruned, dict):
            for item in pruned.get("parsing_res_list", []):
                text = item.get("block_content")
                if isinstance(text, str):
                    values.append(text)
    return values


class DisabledVLMProvider:
    async def describe_image(
        self,
        image: bytes,
        *,
        caption: str | None,
        section_path: list[str],
    ) -> ImageDescription:
        return ImageDescription(summary="")


class _ChatCompletionsVLMProvider:
    provider_name = "OpenAI-compatible VLM"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float,
        max_tokens: int | None = None,
        extra_body: dict[str, Any] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.extra_body = extra_body or {}
        self.client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds, connect=15)
        )
        self._owns_client = client is None

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    async def describe_image(
        self,
        image: bytes,
        *,
        caption: str | None,
        section_path: list[str],
    ) -> ImageDescription:
        media_type = _detect_image_media_type(image)
        encoded = base64.b64encode(image).decode("ascii")
        prompt = (
            "分析这张文档图片。只输出合法 JSON，不要输出 Markdown 代码块。"
            '格式为 {"summary":"图片核心内容","entities":[],"relations":[],'
            '"chart_type":"flowchart|table|photo|diagram|other",'
            '"search_terms":[],"warnings":[]}。'
            f"\n章节：{' / '.join(section_path[:10]) or '未知'}"
            f"\n已有标题：{(caption or '')[:300]}"
        )
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{encoded}",
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "temperature": 0.1,
            **self.extra_body,
        }
        if self.max_tokens is not None:
            body["max_tokens"] = self.max_tokens
        response = await self.client.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=body,
        )
        if not response.is_success:
            raise ExternalProviderError(
                f"{self.provider_name} HTTP {response.status_code}: {response.text[:500]}"
            )
        try:
            content = response.json()["choices"][0]["message"]["content"]
            if isinstance(content, list):
                content = "".join(
                    item.get("text", "") for item in content if isinstance(item, dict)
                )
            parsed = _parse_json_object(content)
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ExternalProviderError(
                f"{self.provider_name} returned invalid JSON"
            ) from exc
        return ImageDescription(
            summary=str(parsed.get("summary", ""))[:800],
            entities=_string_list(parsed.get("entities")),
            relations=_string_list(parsed.get("relations")),
            chart_type=str(parsed.get("chart_type", "other")),
            search_terms=_string_list(parsed.get("search_terms")),
            warnings=_string_list(parsed.get("warnings")),
            raw=response.json(),
        )


class DashScopeVLMProvider(_ChatCompletionsVLMProvider):
    provider_name = "DashScope VLM"

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        super().__init__(
            base_url=settings.dashscope_base_url,
            api_key=settings.dashscope_api_key,
            model=settings.dashscope_vlm_model,
            timeout_seconds=settings.vlm_timeout_seconds,
            extra_body={"enable_thinking": False},
            client=client,
        )


class OpenAICompatibleVLMProvider(_ChatCompletionsVLMProvider):
    """VLM served by vLLM or another OpenAI-compatible Chat Completions API."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        super().__init__(
            base_url=settings.vlm_base_url,
            api_key=settings.vlm_api_key,
            model=settings.vlm_model,
            timeout_seconds=settings.vlm_timeout_seconds,
            max_tokens=settings.vlm_max_tokens,
            client=client,
        )


class MockEmbeddingProvider:
    """Deterministic, dependency-free embedding used for local development."""

    model_name = "mock-multimodal-embedding"
    revision = "hash-v1"

    def __init__(self, dimension: int = 2048) -> None:
        self.dimension = dimension

    async def embed_query(self, query: MultimodalInput) -> list[float]:
        return self._embed(query)

    async def embed_documents(
        self,
        documents: list[MultimodalInput],
    ) -> list[list[float]]:
        return [self._embed(document) for document in documents]

    def _embed(self, value: MultimodalInput) -> list[float]:
        vector = [0.0] * self.dimension
        text = " ".join(filter(None, [value.instruction, value.text]))
        for token in _semantic_tokens(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
            index = int.from_bytes(digest[:8], "big") % self.dimension
            sign = 1.0 if digest[8] & 1 else -1.0
            vector[index] += sign
        if value.image:
            digest = hashlib.sha256(value.image).digest()
            for offset in range(0, len(digest), 2):
                index = int.from_bytes(digest[offset : offset + 2], "big") % self.dimension
                vector[index] += 0.1
        return normalize(vector)


class MockRerankerProvider:
    model_name = "mock-multimodal-reranker"
    revision = "lexical-v1"

    async def rerank(
        self,
        query: MultimodalInput,
        candidates: list[MultimodalInput],
        *,
        top_n: int,
    ) -> list[RerankScore]:
        query_tokens = set(_semantic_tokens(query.text or ""))
        scores: list[RerankScore] = []
        for index, candidate in enumerate(candidates):
            candidate_tokens = set(_semantic_tokens(candidate.text or ""))
            union = query_tokens | candidate_tokens
            lexical = len(query_tokens & candidate_tokens) / len(union) if union else 0.0
            scores.append(RerankScore(index=index, score=lexical))
        return sorted(scores, key=lambda item: (-item.score, item.index))[:top_n]


class VLLMEmbeddingProvider:
    """Qwen3-VL embedding served by vLLM's multimodal Embeddings API."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self.model_name = settings.embedding_model
        self.revision = settings.embedding_revision
        self.dimension = settings.embedding_dimension
        self.base_url = settings.embedding_vllm_base_url.rstrip("/")
        self.api_key = settings.embedding_vllm_api_key
        self.served_model = settings.embedding_vllm_model
        self.batch_size = settings.embedding_vllm_batch_size
        self.client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(settings.embedding_vllm_timeout_seconds, connect=15)
        )
        self._owns_client = client is None

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    async def embed_query(self, query: MultimodalInput) -> list[float]:
        return (await self.embed_documents([query]))[0]

    async def embed_documents(
        self,
        documents: list[MultimodalInput],
    ) -> list[list[float]]:
        result: list[list[float]] = []
        for offset in range(0, len(documents), self.batch_size):
            batch = documents[offset : offset + self.batch_size]
            result.extend(await self._embed_batch(batch))
        return result

    async def _embed_batch(
        self,
        documents: list[MultimodalInput],
    ) -> list[list[float]]:
        response = await self.client.post(
            f"{self.base_url}/embeddings",
            headers=_bearer_headers(self.api_key),
            json={
                "model": self.served_model,
                "messages": [_vllm_embedding_messages(item) for item in documents],
                "encoding_format": "float",
                "continue_final_message": True,
                "add_special_tokens": True,
            },
        )
        if not response.is_success:
            raise ExternalProviderError(
                f"vLLM embedding HTTP {response.status_code}: {response.text[:500]}"
            )
        try:
            data = response.json()["data"]
            ordered = sorted(data, key=lambda item: int(item["index"]))
            if len(ordered) != len(documents):
                raise ValueError("embedding count mismatch")
            if [int(item["index"]) for item in ordered] != list(range(len(documents))):
                raise ValueError("embedding indexes are invalid")
            vectors = [list(map(float, item["embedding"])) for item in ordered]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ExternalProviderError("vLLM embedding returned invalid JSON") from exc
        if any(len(vector) != self.dimension for vector in vectors):
            raise ExternalProviderError(
                f"vLLM embedding output must be {self.dimension} dimensions"
            )
        return [normalize(vector) for vector in vectors]


class VLLMRerankerProvider:
    """Qwen3-VL reranker served by vLLM's Jina/Cohere-compatible API."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self.model_name = settings.reranker_model
        self.revision = settings.reranker_revision
        self.base_url = settings.reranker_vllm_base_url.rstrip("/")
        self.api_key = settings.reranker_vllm_api_key
        self.served_model = settings.reranker_vllm_model
        self.client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(settings.reranker_vllm_timeout_seconds, connect=15)
        )
        self._owns_client = client is None

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    async def rerank(
        self,
        query: MultimodalInput,
        candidates: list[MultimodalInput],
        *,
        top_n: int,
    ) -> list[RerankScore]:
        if not candidates or top_n <= 0:
            return []
        requested_top_n = min(top_n, len(candidates))
        response = await self.client.post(
            f"{self.base_url}/rerank",
            headers=_bearer_headers(self.api_key),
            json={
                "model": self.served_model,
                "query": _vllm_score_input(query),
                "documents": [_vllm_score_input(item) for item in candidates],
                "top_n": requested_top_n,
            },
        )
        if not response.is_success:
            raise ExternalProviderError(
                f"vLLM reranker HTTP {response.status_code}: {response.text[:500]}"
            )
        try:
            values = response.json()["results"]
            scores = [
                RerankScore(
                    index=int(item["index"]),
                    score=float(item["relevance_score"]),
                )
                for item in values
            ]
            if len(scores) != requested_top_n:
                raise ValueError("reranker result count mismatch")
            if len({item.index for item in scores}) != len(scores):
                raise ValueError("reranker returned duplicate indexes")
            if any(item.index < 0 or item.index >= len(candidates) for item in scores):
                raise ValueError("reranker index out of range")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ExternalProviderError("vLLM reranker returned invalid JSON") from exc
        return sorted(scores, key=lambda item: (-item.score, item.index))


class QwenLocalEmbeddingProvider:
    model_name: str
    revision: str

    def __init__(self, settings: Settings) -> None:
        self.model_name = settings.embedding_model
        self.revision = settings.embedding_revision
        self.dimension = settings.embedding_dimension
        self.module_name = settings.qwen_embedding_module
        self.class_name = settings.qwen_embedding_class
        self.model_path = settings.qwen_embedding_model_path
        self.repository_path = (
            settings.qwen_embedding_repository_path or settings.qwen_repository_path
        )
        self._model: Any = None
        self._lock = asyncio.Lock()

    async def embed_query(self, query: MultimodalInput) -> list[float]:
        return (await self.embed_documents([query]))[0]

    async def embed_documents(
        self,
        documents: list[MultimodalInput],
    ) -> list[list[float]]:
        async with self._lock:
            return await asyncio.to_thread(self._process, documents)

    def _process(self, documents: list[MultimodalInput]) -> list[list[float]]:
        model = self._load()
        inputs = [_qwen_input(item) for item in documents]
        output = model.process(inputs)
        values = output.detach().cpu().tolist() if hasattr(output, "detach") else output
        result = [list(map(float, row)) for row in values]
        if any(len(row) != self.dimension for row in result):
            raise RuntimeError(f"Qwen embedding output must be {self.dimension} dimensions")
        return result

    def _load(self) -> Any:
        if self._model is None:
            if self.repository_path:
                path = str(Path(self.repository_path).resolve())
                if path not in sys.path:
                    sys.path.insert(0, path)
            module = importlib.import_module(self.module_name)
            cls = getattr(module, self.class_name)
            self._model = cls(model_name_or_path=self.model_path)
        return self._model


class QwenLocalRerankerProvider:
    model_name: str
    revision: str

    def __init__(self, settings: Settings) -> None:
        self.model_name = settings.reranker_model
        self.revision = settings.reranker_revision
        self.module_name = settings.qwen_reranker_module
        self.class_name = settings.qwen_reranker_class
        self.model_path = settings.qwen_reranker_model_path
        self.repository_path = (
            settings.qwen_reranker_repository_path or settings.qwen_repository_path
        )
        self._model: Any = None
        self._lock = asyncio.Lock()

    async def rerank(
        self,
        query: MultimodalInput,
        candidates: list[MultimodalInput],
        *,
        top_n: int,
    ) -> list[RerankScore]:
        async with self._lock:
            values = await asyncio.to_thread(self._process, query, candidates)
        return sorted(
            [RerankScore(index=index, score=float(score)) for index, score in enumerate(values)],
            key=lambda item: (-item.score, item.index),
        )[:top_n]

    def _process(
        self,
        query: MultimodalInput,
        candidates: list[MultimodalInput],
    ) -> list[float]:
        model = self._load()
        output = model.process(
            {
                "instruction": (
                    "Retrieve text passages or document images relevant to the user's query."
                ),
                "query": _qwen_input(query, include_instruction=False),
                "documents": [
                    _qwen_input(candidate, include_instruction=False)
                    for candidate in candidates
                ],
            }
        )
        if hasattr(output, "detach"):
            output = output.detach().cpu().tolist()
        return [float(value) for value in output]

    def _load(self) -> Any:
        if self._model is None:
            if self.repository_path:
                path = str(Path(self.repository_path).resolve())
                if path not in sys.path:
                    sys.path.insert(0, path)
            module = importlib.import_module(self.module_name)
            cls = getattr(module, self.class_name)
            self._model = cls(model_name_or_path=self.model_path)
        return self._model


def _qwen_input(
    value: MultimodalInput,
    *,
    include_instruction: bool = True,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if value.text:
        result["text"] = value.text
    if value.image:
        image = Image.open(BytesIO(value.image))
        image.load()
        result["image"] = image.convert("RGB")
    if include_instruction and value.instruction:
        result["instruction"] = value.instruction
    return result


def _bearer_headers(api_key: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _vllm_embedding_messages(value: MultimodalInput) -> list[dict[str, Any]]:
    content = _vllm_content(value)
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": value.instruction or "Represent the user's input.",
                }
            ],
        },
        {"role": "user", "content": content},
        {"role": "assistant", "content": [{"type": "text", "text": ""}]},
    ]


def _vllm_score_input(value: MultimodalInput) -> str | dict[str, Any]:
    text = "\n".join(filter(None, [value.instruction, value.text]))
    if not value.image:
        return text
    content: list[dict[str, Any]] = []
    if text:
        content.append({"type": "text", "text": text})
    content.append(_vllm_image_part(value.image))
    return {"content": content}


def _vllm_content(value: MultimodalInput) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    if value.image:
        content.append(_vllm_image_part(value.image))
    content.append({"type": "text", "text": value.text or ""})
    return content


def _vllm_image_part(image: bytes) -> dict[str, Any]:
    media_type = _detect_image_media_type(image)
    encoded = base64.b64encode(image).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{media_type};base64,{encoded}"},
    }


def _semantic_tokens(text: str) -> list[str]:
    lowered = text.lower()
    words = re.findall(r"[a-z0-9_]+", lowered)
    chinese_runs = re.findall(r"[\u3400-\u9fff]+", lowered)
    chinese = []
    for run in chinese_runs:
        chinese.extend(run)
        chinese.extend(run[index : index + 2] for index in range(max(0, len(run) - 1)))
    return words + chinese


def _parse_json_object(value: str) -> dict[str, Any]:
    stripped = value.strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
    stripped = re.sub(r"\s*```$", "", stripped)
    start, end = stripped.find("{"), stripped.rfind("}")
    if start < 0 or end < start:
        raise json.JSONDecodeError("missing JSON object", stripped, 0)
    return json.loads(stripped[start : end + 1])


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None]


def _detect_image_media_type(image: bytes) -> str:
    if image.startswith(b"\x89PNG"):
        return "image/png"
    if image.startswith(b"RIFF") and image[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"
