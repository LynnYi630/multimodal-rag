from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    document_id: str
    version_id: str
    job_id: str
    status: str
    existing: bool = False


class VersionInfo(BaseModel):
    version_id: str
    version_no: int
    status: str
    file_hash: str
    parser_name: str
    parser_version: str
    embedding_model: str
    created_at: datetime


class DocumentInfo(BaseModel):
    document_id: str
    name: str
    source_type: str
    status: str
    active_version_id: str | None
    created_at: datetime


class JobProgress(BaseModel):
    completed: int
    total: int


class JobResponse(BaseModel):
    job_id: str
    status: str
    current_step: str
    progress: JobProgress
    errors: list[dict]
    created_at: datetime
    updated_at: datetime


class SearchFilters(BaseModel):
    document_ids: list[str] | None = None
    version_policy: Literal["active_only"] = "active_only"


class SearchInclude(BaseModel):
    snippets: bool = True
    image_metadata: bool = True


class SearchRequest(BaseModel):
    query: Annotated[str, Field(min_length=1, max_length=4000)]
    top_k: Annotated[int, Field(ge=1, le=50)] = 10
    text_candidate_k: Annotated[int, Field(ge=0, le=100)] = 30
    image_candidate_k: Annotated[int, Field(ge=0, le=100)] = 20
    filters: SearchFilters = Field(default_factory=SearchFilters)
    include: SearchInclude = Field(default_factory=SearchInclude)


class DocumentRef(BaseModel):
    document_id: str
    version_id: str
    name: str


class Location(BaseModel):
    page_no: int | None
    section_path: list[str]


class TextResult(BaseModel):
    content: str
    highlight: str | None = None


class ImageResult(BaseModel):
    asset_id: str
    uri: str
    kind: str
    caption: str | None
    description: str | None


class SearchResult(BaseModel):
    result_id: str
    type: Literal["text", "image"]
    score: float
    rrf_score: float
    document: DocumentRef
    location: Location
    text: TextResult | None = None
    image: ImageResult | None = None


class Timing(BaseModel):
    query_embedding: int
    retrieval: int
    candidate_loading: int
    rerank: int
    total: int


class SearchResponse(BaseModel):
    query: str
    ranking_mode: str
    results: list[SearchResult]
    timing_ms: Timing
    trace_id: str


class DeleteResponse(BaseModel):
    document_id: str
    status: str

