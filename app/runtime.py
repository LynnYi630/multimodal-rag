from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.application.services import (
    AssetService,
    DocumentService,
    IngestionService,
    SearchService,
)
from app.config import Settings
from app.infrastructure.database import Database
from app.infrastructure.parsers import DoclingAdapter, PlainTextParser
from app.infrastructure.providers import (
    DashScopeEmbeddingProvider,
    DashScopeRerankerProvider,
    DashScopeVLMProvider,
    DisabledOCRProvider,
    DisabledVLMProvider,
    MockEmbeddingProvider,
    MockRerankerProvider,
    OpenAICompatibleVLMProvider,
    PaddleOCRProvider,
    QwenLocalEmbeddingProvider,
    QwenLocalRerankerProvider,
    VLLMEmbeddingProvider,
    VLLMRerankerProvider,
)
from app.infrastructure.storage import FileSystemStorage, MinioStorage
from app.infrastructure.vector import InMemoryVectorRepository, QdrantVectorRepository


@dataclass(slots=True)
class Container:
    settings: Settings
    database: Database
    storage: Any
    vector: Any
    parser: Any
    ocr: Any
    vlm: Any
    embedding: Any
    reranker: Any
    documents: DocumentService
    ingestion: IngestionService
    search: SearchService
    assets: AssetService

    async def initialize(self) -> None:
        await self.database.initialize(self.settings.auto_create_schema)
        await self.storage.initialize()
        await self.vector.initialize()

    async def close(self) -> None:
        for provider in (
            self.ocr,
            self.vlm,
            self.embedding,
            self.reranker,
            self.vector,
        ):
            close = getattr(provider, "close", None)
            if close is not None:
                result = close()
                if result is not None:
                    await result
        await self.database.close()


def build_container(settings: Settings) -> Container:
    database = Database(settings)
    storage = (
        FileSystemStorage(settings.local_storage_root)
        if settings.storage_provider == "filesystem"
        else MinioStorage(settings)
    )
    vector = (
        InMemoryVectorRepository(settings.embedding_dimension)
        if settings.vector_provider == "memory"
        else QdrantVectorRepository(settings)
    )
    parser = (
        PlainTextParser()
        if settings.parser_provider == "plain_text"
        else DoclingAdapter(
            settings.docling_artifacts_path,
            ocr_engine=settings.docling_ocr_engine,
            rapidocr_backend=settings.docling_rapidocr_backend,
            ocr_languages=[
                language.strip()
                for language in settings.docling_ocr_languages.split(",")
                if language.strip()
            ],
        )
    )
    ocr = (
        PaddleOCRProvider(settings)
        if settings.ocr_provider == "paddleocr"
        else DisabledOCRProvider()
    )
    if settings.vlm_provider == "dashscope":
        vlm = DashScopeVLMProvider(settings)
    elif settings.vlm_provider in {"vllm", "openai_compatible"}:
        vlm = OpenAICompatibleVLMProvider(settings)
    else:
        vlm = DisabledVLMProvider()
    if settings.embedding_provider == "qwen_local":
        embedding = QwenLocalEmbeddingProvider(settings)
    elif settings.embedding_provider == "vllm":
        embedding = VLLMEmbeddingProvider(settings)
    elif settings.embedding_provider == "dashscope":
        embedding = DashScopeEmbeddingProvider(settings)
    else:
        embedding = MockEmbeddingProvider(settings.embedding_dimension)
    if settings.reranker_provider == "qwen_local":
        reranker = QwenLocalRerankerProvider(settings)
    elif settings.reranker_provider == "vllm":
        reranker = VLLMRerankerProvider(settings)
    elif settings.reranker_provider == "dashscope":
        reranker = DashScopeRerankerProvider(settings)
    else:
        reranker = MockRerankerProvider()
    documents = DocumentService(
        settings=settings,
        database=database,
        storage=storage,
        parser=parser,
        embedding=embedding,
        vector=vector,
    )
    ingestion = IngestionService(
        settings=settings,
        database=database,
        storage=storage,
        vector=vector,
        parser=parser,
        ocr=ocr,
        vlm=vlm,
        embedding=embedding,
    )
    search = SearchService(
        settings=settings,
        database=database,
        storage=storage,
        vector=vector,
        embedding=embedding,
        reranker=reranker,
    )
    assets = AssetService(database, storage)
    return Container(
        settings=settings,
        database=database,
        storage=storage,
        vector=vector,
        parser=parser,
        ocr=ocr,
        vlm=vlm,
        embedding=embedding,
        reranker=reranker,
        documents=documents,
        ingestion=ingestion,
        search=search,
        assets=assets,
    )
