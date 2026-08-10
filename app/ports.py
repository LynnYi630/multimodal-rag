from __future__ import annotations

from collections.abc import Sequence
from typing import BinaryIO, Protocol

from app.domain.models import (
    ImageDescription,
    MultimodalInput,
    ObjectStream,
    OCRResult,
    RerankScore,
    StoredObject,
    UnifiedDocument,
    UnifiedNode,
    VectorHit,
)


class DocumentParser(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    def supports(self, media_type: str, filename: str) -> bool: ...

    def parse(
        self,
        file_obj: BinaryIO,
        *,
        filename: str,
        document_id: str,
        version_id: str,
    ) -> UnifiedDocument: ...


class OCRProvider(Protocol):
    async def recognize(self, image: bytes, *, language: list[str]) -> OCRResult: ...


class VLMProvider(Protocol):
    async def describe_image(
        self,
        image: bytes,
        *,
        caption: str | None,
        section_path: list[str],
    ) -> ImageDescription: ...


class EmbeddingProvider(Protocol):
    model_name: str
    revision: str
    dimension: int

    async def embed_query(self, query: MultimodalInput) -> list[float]: ...

    async def embed_documents(
        self,
        documents: list[MultimodalInput],
    ) -> list[list[float]]: ...


class RerankerProvider(Protocol):
    model_name: str
    revision: str

    async def rerank(
        self,
        query: MultimodalInput,
        candidates: list[MultimodalInput],
        *,
        top_n: int,
    ) -> list[RerankScore]: ...


class VectorRepository(Protocol):
    async def initialize(self) -> None: ...

    async def upsert(
        self,
        nodes: Sequence[UnifiedNode],
        vectors: Sequence[Sequence[float]],
        *,
        acl_scope_ids: Sequence[str],
        status: str,
    ) -> None: ...

    async def search(
        self,
        vector: Sequence[float],
        *,
        tenant_id: str,
        node_type: str,
        acl_scope_ids: Sequence[str],
        document_ids: Sequence[str] | None,
        limit: int,
    ) -> list[VectorHit]: ...

    async def set_version_status(self, version_id: str, status: str) -> None: ...

    async def delete_document(self, document_id: str) -> None: ...


class AssetStorage(Protocol):
    async def initialize(self) -> None: ...

    async def put(
        self,
        bucket: str,
        key: str,
        content: bytes,
        media_type: str,
    ) -> StoredObject: ...

    async def get(self, bucket: str, key: str) -> ObjectStream: ...

    async def read(self, bucket: str, key: str) -> bytes: ...

    async def delete_prefix(self, bucket: str, prefix: str) -> None: ...
