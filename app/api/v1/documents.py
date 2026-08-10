from __future__ import annotations

from typing import Annotated

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    Query,
    Request,
    UploadFile,
    status,
)

from app.api.deps import AuthDependency, ContainerDependency
from app.application.schemas import DeleteResponse, DocumentInfo, UploadResponse, VersionInfo
from app.domain.models import InvalidDocumentError
from app.workers.tasks import enqueue_ingestion

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=UploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: Annotated[UploadFile, File()],
    auth: AuthDependency,
    container: ContainerDependency,
    name: Annotated[str | None, Form()] = None,
    parser: Annotated[str, Form()] = "docling",
    acl_subjects: Annotated[list[str] | None, Form()] = None,
) -> UploadResponse:
    if parser != container.parser.name:
        raise InvalidDocumentError(f"configured parser is {container.parser.name!r}")
    content = await _read_upload(file, container.settings.max_upload_bytes)
    result = await container.documents.upload_new(
        filename=file.filename or "document",
        content=content,
        media_type=file.content_type,
        name=name,
        auth=auth,
        acl_scopes=acl_subjects or [],
    )
    if not result.existing:
        enqueue_ingestion(container, result.version_id, background_tasks)
    return result


@router.post(
    "/{document_id}/versions",
    response_model=UploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_version(
    document_id: str,
    background_tasks: BackgroundTasks,
    file: Annotated[UploadFile, File()],
    auth: AuthDependency,
    container: ContainerDependency,
    force: Annotated[bool, Query()] = False,
) -> UploadResponse:
    content = await _read_upload(file, container.settings.max_upload_bytes)
    result = await container.documents.upload_version(
        document_id=document_id,
        filename=file.filename or "document",
        content=content,
        media_type=file.content_type,
        auth=auth,
        force=force,
    )
    if not result.existing or force:
        enqueue_ingestion(container, result.version_id, background_tasks)
    return result


@router.get("/{document_id}", response_model=DocumentInfo)
async def get_document(
    document_id: str,
    auth: AuthDependency,
    container: ContainerDependency,
) -> DocumentInfo:
    return await container.documents.get_document(document_id, auth)


@router.get("/{document_id}/versions", response_model=list[VersionInfo])
async def list_versions(
    document_id: str,
    auth: AuthDependency,
    container: ContainerDependency,
) -> list[VersionInfo]:
    return await container.documents.list_versions(document_id, auth)


@router.delete(
    "/{document_id}",
    response_model=DeleteResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def delete_document(
    document_id: str,
    auth: AuthDependency,
    container: ContainerDependency,
) -> DeleteResponse:
    return await container.documents.delete(document_id, auth)


async def _read_upload(file: UploadFile, max_bytes: int) -> bytes:
    chunks = []
    size = 0
    while chunk := await file.read(1024 * 1024):
        size += len(chunk)
        if size > max_bytes:
            raise InvalidDocumentError("file exceeds MAX_UPLOAD_BYTES")
        chunks.append(chunk)
    return b"".join(chunks)
