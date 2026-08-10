from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class NodeType(StrEnum):
    TEXT = "text"
    IMAGE = "image"


class ImageKind(StrEnum):
    FIGURE = "figure"
    PAGE_RENDER = "page_render"


class VersionStatus(StrEnum):
    UPLOADED = "uploaded"
    QUEUED = "queued"
    PARSING = "parsing"
    BASE_INDEXING = "base_indexing"
    SEARCH_READY = "search_ready"
    ENRICHING = "enriching"
    ENRICHED_READY = "enriched_ready"
    FAILED = "failed"
    SUPERSEDED = "superseded"
    DELETING = "deleting"
    DELETED = "deleted"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(slots=True)
class UnifiedNode:
    node_id: str
    tenant_id: str
    document_id: str
    version_id: str
    node_type: NodeType
    page_no: int | None
    section_path: list[str]
    ordinal: int
    text: str | None = None
    asset_id: str | None = None
    image_kind: ImageKind | None = None
    caption: str | None = None
    ocr_text: str | None = None
    description: str | None = None
    related_text: list[str] = field(default_factory=list)
    bbox: list[float] | None = None
    parent_node_id: str | None = None
    previous_node_id: str | None = None
    next_node_id: str | None = None
    content_hash: str = ""
    parser_name: str = ""
    parser_version: str = ""
    embedding_model: str = ""
    embedding_dimension: int = 2048
    enrichment_status: str = "pending"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ParsedBlock:
    text: str
    page_no: int | None
    section_path: list[str]
    ordinal: int
    bbox: list[float] | None = None
    kind: str = "paragraph"


@dataclass(slots=True)
class ParsedImage:
    content: bytes
    media_type: str
    page_no: int | None
    ordinal: int
    kind: ImageKind = ImageKind.FIGURE
    caption: str | None = None
    bbox: list[float] | None = None
    section_path: list[str] = field(default_factory=list)


@dataclass(slots=True)
class UnifiedDocument:
    blocks: list[ParsedBlock]
    images: list[ParsedImage]
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MultimodalInput:
    text: str | None = None
    image: bytes | None = None
    instruction: str | None = None


@dataclass(slots=True)
class OCRResult:
    normalized_text: str
    provider: str
    provider_version: str
    request_id: str
    raw: dict[str, Any]
    confidence_summary: float | None = None


@dataclass(slots=True)
class ImageDescription:
    summary: str
    entities: list[str] = field(default_factory=list)
    relations: list[str] = field(default_factory=list)
    chart_type: str = "other"
    search_terms: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class VectorHit:
    node_id: str
    score: float
    payload: dict[str, Any]


@dataclass(slots=True)
class RerankScore:
    index: int
    score: float


@dataclass(slots=True)
class AuthContext:
    tenant_id: str
    user_id: str
    roles: list[str] = field(default_factory=list)
    groups: list[str] = field(default_factory=list)

    @property
    def scope_ids(self) -> list[str]:
        return [
            f"user:{self.user_id}",
            *(f"role:{role}" for role in self.roles),
            *(f"group:{group}" for group in self.groups),
        ]


@dataclass(slots=True)
class StoredObject:
    bucket: str
    key: str
    etag: str
    media_type: str
    size: int
    version_id: str | None = None


@dataclass(slots=True)
class ObjectStream:
    chunks: Any
    etag: str
    media_type: str
    size: int


@dataclass(slots=True)
class JobSnapshot:
    id: str
    status: JobStatus
    current_step: str
    completed: int
    total: int
    errors: list[dict[str, Any]]
    created_at: datetime
    updated_at: datetime


class DomainError(Exception):
    """Base error translated by the API layer."""


class NotFoundError(DomainError):
    pass


class PermissionDeniedError(DomainError):
    pass


class InvalidDocumentError(DomainError):
    pass


class ProviderUnavailableError(DomainError):
    pass


class ParserUnavailableError(ProviderUnavailableError):
    pass


class ExternalProviderError(DomainError):
    pass


class ConsistencyError(DomainError):
    pass

