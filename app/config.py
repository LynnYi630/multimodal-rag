from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_env: str = "development"
    app_name: str = "multimodal-rag"
    log_level: str = "INFO"
    api_prefix: str = "/v1"

    database_url: str = "sqlite+aiosqlite:///./data/rag.db"
    postgres_database_url: str = ""
    postgres_password: str = "123456"
    auto_create_schema: bool = True

    redis_url: str = "redis://:123456@localhost:6379/0"
    redis_password: str = "123456"
    celery_task_always_eager: bool = True

    vector_provider: Literal["memory", "qdrant"] = "memory"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "knowledge_nodes"

    storage_provider: Literal["filesystem", "minio"] = "filesystem"
    local_storage_root: Path = Path("./data/objects")
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = ""
    minio_secret_key: str = ""
    minio_secure: bool = False
    minio_bucket_originals: str = "rag-originals"
    minio_bucket_assets: str = "rag-assets"
    minio_bucket_derived: str = "rag-derived"

    parser_provider: Literal["docling", "plain_text"] = "docling"
    docling_artifacts_path: Path | None = None
    docling_ocr_engine: Literal["rapidocr", "auto"] = "rapidocr"
    docling_rapidocr_backend: Literal[
        "onnxruntime", "openvino", "paddle", "torch"
    ] = "torch"
    docling_ocr_languages: str = "chinese"
    mineru_enabled: bool = False
    mineru_base_url: str = ""

    embedding_provider: Literal["mock", "qwen_local", "vllm", "dashscope"] = "mock"
    embedding_model: str = "Qwen/Qwen3-VL-Embedding-2B"
    embedding_revision: str = "locked-revision"
    embedding_dimension: int = 2048
    embedding_vllm_base_url: str = "http://127.0.0.1:8200/v1"
    embedding_vllm_api_key: str = ""
    embedding_vllm_model: str = "qwen3-vl-embedding-2b"
    embedding_vllm_batch_size: int = 8
    embedding_vllm_timeout_seconds: float = 120
    dashscope_embedding_url: str = (
        "https://dashscope.aliyuncs.com/api/v1/services/embeddings/"
        "multimodal-embedding/multimodal-embedding"
    )
    dashscope_embedding_model: str = "qwen3-vl-embedding"
    dashscope_embedding_concurrency: int = 4
    dashscope_embedding_timeout_seconds: float = 120
    qwen_embedding_module: str = "src.models.qwen3_vl_embedding"
    qwen_embedding_class: str = "Qwen3VLEmbedder"
    qwen_embedding_model_path: str = "./models/Qwen3-VL-Embedding-2B"
    qwen_embedding_repository_path: str = ""

    reranker_provider: Literal["mock", "qwen_local", "vllm", "dashscope"] = "mock"
    reranker_model: str = "Qwen/Qwen3-VL-Reranker-2B"
    reranker_revision: str = "locked-revision"
    reranker_vllm_base_url: str = "http://127.0.0.1:8300/v1"
    reranker_vllm_api_key: str = ""
    reranker_vllm_model: str = "qwen3-vl-reranker-2b"
    reranker_vllm_timeout_seconds: float = 120
    dashscope_reranker_url: str = (
        "https://dashscope.aliyuncs.com/api/v1/services/rerank/"
        "text-rerank/text-rerank"
    )
    dashscope_reranker_model: str = "qwen3-vl-rerank"
    dashscope_reranker_timeout_seconds: float = 120
    dashscope_reranker_instruction: str = (
        "Retrieve text passages or document images relevant to the user's query."
    )
    qwen_reranker_module: str = "src.models.qwen3_vl_reranker"
    qwen_reranker_class: str = "Qwen3VLReranker"
    qwen_reranker_model_path: str = "./models/Qwen3-VL-Reranker-2B"
    qwen_reranker_repository_path: str = ""

    ocr_provider: Literal["disabled", "paddleocr"] = "disabled"
    paddleocr_job_url: str = "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"
    paddleocr_access_token: str = ""
    paddleocr_model: str = "PaddleOCR-VL-1.6"
    paddleocr_poll_interval_seconds: float = 5
    paddleocr_timeout_seconds: float = 600

    vlm_provider: Literal[
        "disabled", "dashscope", "vllm", "openai_compatible"
    ] = "disabled"
    dashscope_api_key: str = ""
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    dashscope_vlm_model: str = "qwen3.7-flash"
    vlm_base_url: str = "http://127.0.0.1:8100/v1"
    vlm_api_key: str = ""
    vlm_model: str = "qwen3-vl-8b-instruct-fp8"
    vlm_max_tokens: int = 800
    vlm_timeout_seconds: float = 120

    max_upload_bytes: int = 100 * 1024 * 1024
    max_results_per_document: int = 5
    rrf_k: int = 60
    rerank_fallback: bool = True

    default_tenant_id: str = Field(
        default="00000000-0000-0000-0000-000000000001",
        exclude=True,
    )
    default_user_id: str = Field(
        default="00000000-0000-0000-0000-000000000001",
        exclude=True,
    )

    @model_validator(mode="after")
    def validate_external_credentials(self) -> Settings:
        if not any(
            language.strip() for language in self.docling_ocr_languages.split(",")
        ):
            raise ValueError("DOCLING_OCR_LANGUAGES must contain at least one language")
        if self.ocr_provider == "paddleocr" and not self.paddleocr_access_token:
            raise ValueError("PADDLEOCR_ACCESS_TOKEN is required when OCR_PROVIDER=paddleocr")
        dashscope_enabled = "dashscope" in {
            self.vlm_provider,
            self.embedding_provider,
            self.reranker_provider,
        }
        if dashscope_enabled and not self.dashscope_api_key:
            raise ValueError("DASHSCOPE_API_KEY is required for DashScope providers")
        if self.vlm_provider in {"vllm", "openai_compatible"}:
            if not self.vlm_base_url.strip():
                raise ValueError(
                    "VLM_BASE_URL is required when VLM_PROVIDER=vllm/openai_compatible"
                )
            if not self.vlm_model.strip():
                raise ValueError(
                    "VLM_MODEL is required when VLM_PROVIDER=vllm/openai_compatible"
                )
            if self.vlm_max_tokens <= 0:
                raise ValueError("VLM_MAX_TOKENS must be greater than zero")
        if self.embedding_provider == "vllm":
            if not self.embedding_vllm_base_url.strip():
                raise ValueError(
                    "EMBEDDING_VLLM_BASE_URL is required when EMBEDDING_PROVIDER=vllm"
                )
            if not self.embedding_vllm_model.strip():
                raise ValueError(
                    "EMBEDDING_VLLM_MODEL is required when EMBEDDING_PROVIDER=vllm"
                )
            if self.embedding_vllm_batch_size <= 0:
                raise ValueError("EMBEDDING_VLLM_BATCH_SIZE must be greater than zero")
        if self.embedding_provider == "dashscope":
            if not self.dashscope_embedding_url.strip():
                raise ValueError(
                    "DASHSCOPE_EMBEDDING_URL is required when EMBEDDING_PROVIDER=dashscope"
                )
            if not self.dashscope_embedding_model.strip():
                raise ValueError(
                    "DASHSCOPE_EMBEDDING_MODEL is required when "
                    "EMBEDDING_PROVIDER=dashscope"
                )
            if self.dashscope_embedding_concurrency <= 0:
                raise ValueError("DASHSCOPE_EMBEDDING_CONCURRENCY must be greater than zero")
        if self.reranker_provider == "vllm":
            if not self.reranker_vllm_base_url.strip():
                raise ValueError(
                    "RERANKER_VLLM_BASE_URL is required when RERANKER_PROVIDER=vllm"
                )
            if not self.reranker_vllm_model.strip():
                raise ValueError(
                    "RERANKER_VLLM_MODEL is required when RERANKER_PROVIDER=vllm"
                )
        if self.reranker_provider == "dashscope":
            if not self.dashscope_reranker_url.strip():
                raise ValueError(
                    "DASHSCOPE_RERANKER_URL is required when RERANKER_PROVIDER=dashscope"
                )
            if not self.dashscope_reranker_model.strip():
                raise ValueError(
                    "DASHSCOPE_RERANKER_MODEL is required when "
                    "RERANKER_PROVIDER=dashscope"
                )
        if self.storage_provider == "minio" and (
            not self.minio_access_key or not self.minio_secret_key
        ):
            raise ValueError("MINIO_ACCESS_KEY and MINIO_SECRET_KEY are required")
        if not self.celery_task_always_eager and self.vector_provider == "memory":
            raise ValueError(
                "VECTOR_PROVIDER=qdrant is required when Celery runs out of process"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
