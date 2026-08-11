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
    mineru_enabled: bool = False
    mineru_base_url: str = ""

    embedding_provider: Literal["mock", "qwen_local"] = "mock"
    embedding_model: str = "Qwen/Qwen3-VL-Embedding-2B"
    embedding_revision: str = "locked-revision"
    embedding_dimension: int = 2048
    qwen_embedding_module: str = "src.models.qwen3_vl_embedding"
    qwen_embedding_class: str = "Qwen3VLEmbedder"
    qwen_embedding_model_path: str = "./models/Qwen3-VL-Embedding-2B"
    qwen_embedding_repository_path: str = ""
    qwen_repository_path: str = ""

    reranker_provider: Literal["mock", "qwen_local"] = "mock"
    reranker_model: str = "Qwen/Qwen3-VL-Reranker-2B"
    reranker_revision: str = "locked-revision"
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
        if self.ocr_provider == "paddleocr" and not self.paddleocr_access_token:
            raise ValueError("PADDLEOCR_ACCESS_TOKEN is required when OCR_PROVIDER=paddleocr")
        if self.vlm_provider == "dashscope" and not self.dashscope_api_key:
            raise ValueError("DASHSCOPE_API_KEY is required when VLM_PROVIDER=dashscope")
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
