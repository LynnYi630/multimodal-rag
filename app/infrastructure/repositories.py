from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import and_, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models import (
    AuthContext,
    ImageKind,
    JobSnapshot,
    JobStatus,
    NodeType,
    NotFoundError,
    UnifiedNode,
    VersionStatus,
)
from app.infrastructure.database import (
    AssetORM,
    DocumentACLORM,
    DocumentORM,
    DocumentVersionORM,
    IngestionJobORM,
    NodeORM,
    OutboxEventORM,
)


class Repository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_document(
        self,
        *,
        document: DocumentORM,
        version: DocumentVersionORM,
        job: IngestionJobORM,
        acl_scopes: Sequence[str],
    ) -> None:
        self.session.add_all([document, version, job])
        for scope in set(acl_scopes):
            subject_type, subject_id = scope.split(":", 1)
            self.session.add(
                DocumentACLORM(
                    document_id=document.id,
                    subject_type=subject_type,
                    subject_id=subject_id,
                    permission="read",
                )
            )
        self.session.add(
            DocumentACLORM(
                document_id=document.id,
                subject_type="user",
                subject_id=document.created_by,
                permission="admin",
            )
        )
        await self.session.commit()

    async def add_version(
        self,
        version: DocumentVersionORM,
        job: IngestionJobORM,
    ) -> None:
        self.session.add_all([version, job])
        await self.session.commit()

    async def get_document(self, document_id: str) -> DocumentORM:
        document = await self.session.get(DocumentORM, document_id)
        if document is None or document.deleted_at is not None:
            raise NotFoundError("document not found")
        return document

    async def get_version(self, version_id: str) -> DocumentVersionORM:
        version = await self.session.get(DocumentVersionORM, version_id)
        if version is None:
            raise NotFoundError("document version not found")
        return version

    async def get_version_by_content(
        self,
        document_id: str,
        file_hash: str,
        parser_version: str,
        embedding_model: str,
    ) -> DocumentVersionORM | None:
        query = select(DocumentVersionORM).where(
            DocumentVersionORM.document_id == document_id,
            DocumentVersionORM.file_hash == file_hash,
            DocumentVersionORM.parser_version == parser_version,
            DocumentVersionORM.embedding_model == embedding_model,
        )
        return (await self.session.execute(query)).scalar_one_or_none()

    async def next_version_no(self, document_id: str) -> int:
        query = select(func.max(DocumentVersionORM.version_no)).where(
            DocumentVersionORM.document_id == document_id
        )
        value = (await self.session.execute(query)).scalar_one()
        return (value or 0) + 1

    async def list_versions(self, document_id: str) -> list[DocumentVersionORM]:
        query = (
            select(DocumentVersionORM)
            .where(DocumentVersionORM.document_id == document_id)
            .order_by(DocumentVersionORM.version_no.desc())
        )
        return list((await self.session.execute(query)).scalars())

    async def get_job(self, job_id: str) -> JobSnapshot:
        job = await self.session.get(IngestionJobORM, job_id)
        if job is None:
            raise NotFoundError("job not found")
        return JobSnapshot(
            id=job.id,
            status=JobStatus(job.status),
            current_step=job.current_step,
            completed=job.completed,
            total=job.total,
            errors=job.errors,
            created_at=job.created_at,
            updated_at=job.updated_at,
        )

    async def get_job_for_version(self, version_id: str) -> IngestionJobORM:
        query = (
            select(IngestionJobORM)
            .where(IngestionJobORM.version_id == version_id)
            .order_by(IngestionJobORM.created_at.desc())
        )
        job = (await self.session.execute(query)).scalars().first()
        if job is None:
            raise NotFoundError("job not found")
        return job

    async def update_job(
        self,
        job_id: str,
        *,
        status: str | None = None,
        current_step: str | None = None,
        completed: int | None = None,
        total: int | None = None,
        error: dict[str, Any] | None = None,
        clear_errors: bool = False,
    ) -> None:
        job = await self.session.get(IngestionJobORM, job_id)
        if job is None:
            raise NotFoundError("job not found")
        if status is not None:
            job.status = status
        if current_step is not None:
            job.current_step = current_step
        if completed is not None:
            job.completed = completed
        if total is not None:
            job.total = total
        if clear_errors:
            job.errors = []
        if error is not None:
            job.errors = [*job.errors, error]
        job.updated_at = datetime.now(UTC)
        await self.session.commit()

    async def set_version_status(self, version_id: str, status: str) -> None:
        version = await self.get_version(version_id)
        version.status = status
        if status in {VersionStatus.SEARCH_READY, VersionStatus.ENRICHED_READY}:
            version.search_ready_at = datetime.now(UTC)
        if status == VersionStatus.ENRICHED_READY:
            version.enriched_ready_at = datetime.now(UTC)
        await self.session.commit()

    async def replace_nodes(
        self,
        version_id: str,
        nodes: Sequence[UnifiedNode],
        assets: Sequence[AssetORM],
    ) -> None:
        existing_nodes = list(
            (
                await self.session.execute(
                    select(NodeORM).where(NodeORM.version_id == version_id)
                )
            ).scalars()
        )
        for node in existing_nodes:
            await self.session.delete(node)
        existing_assets = list(
            (
                await self.session.execute(
                    select(AssetORM).where(AssetORM.version_id == version_id)
                )
            ).scalars()
        )
        for asset in existing_assets:
            await self.session.delete(asset)
        await self.session.flush()
        self.session.add_all([node_to_orm(node) for node in nodes])
        self.session.add_all(list(assets))
        await self.session.commit()

    async def load_nodes(self, node_ids: Sequence[str]) -> list[NodeORM]:
        if not node_ids:
            return []
        query = select(NodeORM).where(NodeORM.id.in_(list(node_ids)))
        found = {node.id: node for node in (await self.session.execute(query)).scalars()}
        return [found[node_id] for node_id in node_ids if node_id in found]

    async def get_asset(self, asset_id: str) -> AssetORM:
        asset = await self.session.get(AssetORM, asset_id)
        if asset is None:
            raise NotFoundError("asset not found")
        return asset

    async def document_names(self, document_ids: Sequence[str]) -> dict[str, str]:
        if not document_ids:
            return {}
        query = select(DocumentORM.id, DocumentORM.name).where(
            DocumentORM.id.in_(list(document_ids))
        )
        return dict((await self.session.execute(query)).all())

    async def acl_scope_ids(self, document_id: str) -> list[str]:
        query = select(
            DocumentACLORM.subject_type,
            DocumentACLORM.subject_id,
        ).where(
            DocumentACLORM.document_id == document_id,
            DocumentACLORM.permission.in_(["read", "write", "admin"]),
        )
        return [f"{kind}:{subject_id}" for kind, subject_id in (await self.session.execute(query))]

    async def can_read(self, document_id: str, auth: AuthContext) -> bool:
        return await self.has_permission(
            document_id,
            auth,
            permissions=["read", "write", "admin"],
        )

    async def can_write(self, document_id: str, auth: AuthContext) -> bool:
        return await self.has_permission(
            document_id,
            auth,
            permissions=["write", "admin"],
        )

    async def can_admin(self, document_id: str, auth: AuthContext) -> bool:
        return await self.has_permission(document_id, auth, permissions=["admin"])

    async def has_permission(
        self,
        document_id: str,
        auth: AuthContext,
        *,
        permissions: Sequence[str],
    ) -> bool:
        conditions = []
        for scope in auth.scope_ids:
            subject_type, subject_id = scope.split(":", 1)
            conditions.append(
                and_(
                    DocumentACLORM.subject_type == subject_type,
                    DocumentACLORM.subject_id == subject_id,
                )
            )
        if not conditions:
            return False
        query = select(DocumentACLORM.document_id).where(
            DocumentACLORM.document_id == document_id,
            DocumentACLORM.permission.in_(list(permissions)),
            or_(*conditions),
        )
        return (await self.session.execute(query)).first() is not None

    async def activate_version(self, document_id: str, version_id: str) -> str | None:
        document = await self.get_document(document_id)
        previous = document.active_version_id
        document.active_version_id = version_id
        document.status = "active"
        await self.session.execute(
            update(DocumentVersionORM)
            .where(
                DocumentVersionORM.document_id == document_id,
                DocumentVersionORM.id != version_id,
                DocumentVersionORM.status.in_(
                    [
                        VersionStatus.SEARCH_READY,
                        VersionStatus.ENRICHED_READY,
                    ]
                ),
            )
            .values(status=VersionStatus.SUPERSEDED)
        )
        version = await self.get_version(version_id)
        version.status = VersionStatus.ENRICHED_READY
        version.search_ready_at = datetime.now(UTC)
        version.enriched_ready_at = datetime.now(UTC)
        self.session.add(
            OutboxEventORM(
                id=str(uuid.uuid4()),
                event_type="ACTIVATE_VERSION",
                aggregate_id=version_id,
                payload={
                    "document_id": document_id,
                    "version_id": version_id,
                    "previous_version_id": previous,
                },
                processed=False,
            )
        )
        await self.session.commit()
        return previous

    async def mark_document_deleting(self, document_id: str) -> DocumentORM:
        document = await self.get_document(document_id)
        document.status = "deleting"
        await self.session.execute(
            update(DocumentVersionORM)
            .where(DocumentVersionORM.document_id == document_id)
            .values(status=VersionStatus.DELETING)
        )
        await self.session.commit()
        return document

    async def mark_document_deleted(self, document_id: str) -> None:
        document = await self.session.get(DocumentORM, document_id)
        if document is None:
            return
        document.status = "deleted"
        document.deleted_at = datetime.now(UTC)
        await self.session.execute(
            update(DocumentVersionORM)
            .where(DocumentVersionORM.document_id == document_id)
            .values(status=VersionStatus.DELETED)
        )
        await self.session.commit()


def node_to_orm(node: UnifiedNode) -> NodeORM:
    return NodeORM(
        id=node.node_id,
        tenant_id=node.tenant_id,
        document_id=node.document_id,
        version_id=node.version_id,
        node_type=node.node_type.value,
        image_kind=node.image_kind.value if node.image_kind else None,
        page_no=node.page_no,
        ordinal=node.ordinal,
        section_path=node.section_path,
        text=node.text,
        asset_id=node.asset_id,
        caption=node.caption,
        ocr_text=node.ocr_text,
        description=node.description,
        related_text=node.related_text,
        bbox=node.bbox,
        content_hash=node.content_hash,
        enrichment_status=node.enrichment_status,
        qdrant_point_id=node.node_id,
        node_metadata=node.metadata,
    )


def orm_to_node(node: NodeORM, *, embedding_model: str = "") -> UnifiedNode:
    return UnifiedNode(
        node_id=node.id,
        tenant_id=node.tenant_id,
        document_id=node.document_id,
        version_id=node.version_id,
        node_type=NodeType(node.node_type),
        page_no=node.page_no,
        section_path=node.section_path,
        ordinal=node.ordinal,
        text=node.text,
        asset_id=node.asset_id,
        image_kind=ImageKind(node.image_kind) if node.image_kind else None,
        caption=node.caption,
        ocr_text=node.ocr_text,
        description=node.description,
        related_text=node.related_text,
        bbox=node.bbox,
        content_hash=node.content_hash,
        embedding_model=embedding_model,
        enrichment_status=node.enrichment_status,
        metadata=node.node_metadata,
    )
