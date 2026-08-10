from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections import Counter
from collections.abc import Sequence
from io import BytesIO
from typing import Any

from PIL import Image
from sqlalchemy.exc import IntegrityError

from app.application.schemas import (
    DeleteResponse,
    DocumentInfo,
    DocumentRef,
    ImageResult,
    JobProgress,
    JobResponse,
    Location,
    SearchRequest,
    SearchResponse,
    SearchResult,
    TextResult,
    Timing,
    UploadResponse,
    VersionInfo,
)
from app.config import Settings
from app.domain.models import (
    AuthContext,
    ImageKind,
    InvalidDocumentError,
    JobStatus,
    MultimodalInput,
    NodeType,
    NotFoundError,
    PermissionDeniedError,
    ProviderUnavailableError,
    UnifiedNode,
    VersionStatus,
)
from app.domain.services import (
    StructureAwareChunker,
    deterministic_asset_id,
    deterministic_node_id,
    deterministic_version_id,
    reciprocal_rank_fusion,
    safe_filename,
    sha256_bytes,
)
from app.infrastructure.database import (
    AssetORM,
    Database,
    DocumentORM,
    DocumentVersionORM,
    IngestionJobORM,
)
from app.infrastructure.repositories import Repository
from app.ports import (
    AssetStorage,
    DocumentParser,
    EmbeddingProvider,
    OCRProvider,
    RerankerProvider,
    VectorRepository,
    VLMProvider,
)

logger = logging.getLogger(__name__)

QUERY_INSTRUCTION = "Retrieve text passages or document images relevant to the user's query."
TEXT_DOCUMENT_INSTRUCTION = "Represent this document passage for retrieval."
IMAGE_DOCUMENT_INSTRUCTION = "Represent this document image and its context for retrieval."

SUPPORTED_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".txt": "text/plain",
    ".md": "text/markdown",
}


class DocumentService:
    def __init__(
        self,
        *,
        settings: Settings,
        database: Database,
        storage: AssetStorage,
        parser: DocumentParser,
        embedding: EmbeddingProvider,
        vector: VectorRepository,
    ) -> None:
        self.settings = settings
        self.database = database
        self.storage = storage
        self.parser = parser
        self.embedding = embedding
        self.vector = vector

    async def upload_new(
        self,
        *,
        filename: str,
        content: bytes,
        media_type: str | None,
        name: str | None,
        auth: AuthContext,
        acl_scopes: Sequence[str],
    ) -> UploadResponse:
        checked_type = self._validate(filename, content, media_type)
        document_id = str(uuid.uuid4())
        return await self._create_version(
            document_id=document_id,
            filename=filename,
            content=content,
            media_type=checked_type,
            name=name or filename,
            auth=auth,
            acl_scopes=[*acl_scopes, f"user:{auth.user_id}"],
            new_document=True,
        )

    async def upload_version(
        self,
        *,
        document_id: str,
        filename: str,
        content: bytes,
        media_type: str | None,
        auth: AuthContext,
        force: bool,
    ) -> UploadResponse:
        checked_type = self._validate(filename, content, media_type)
        async with self.database.session() as session:
            repo = Repository(session)
            document = await repo.get_document(document_id)
            if document.tenant_id != auth.tenant_id or not await repo.can_write(
                document_id, auth
            ):
                raise PermissionDeniedError("document not found")
        return await self._create_version(
            document_id=document_id,
            filename=filename,
            content=content,
            media_type=checked_type,
            name=document.name,
            auth=auth,
            acl_scopes=[],
            new_document=False,
            force=force,
        )

    async def _create_version(
        self,
        *,
        document_id: str,
        filename: str,
        content: bytes,
        media_type: str,
        name: str,
        auth: AuthContext,
        acl_scopes: Sequence[str],
        new_document: bool,
        force: bool = False,
    ) -> UploadResponse:
        file_hash = sha256_bytes(content)
        version_id = deterministic_version_id(
            document_id,
            file_hash,
            self.parser.version,
            self.settings.embedding_model,
        )
        async with self.database.session() as session:
            repo = Repository(session)
            if not new_document:
                existing = await repo.get_version_by_content(
                    document_id,
                    file_hash,
                    self.parser.version,
                    self.settings.embedding_model,
                )
                if existing and not force:
                    job = await repo.get_job_for_version(existing.id)
                    return UploadResponse(
                        document_id=document_id,
                        version_id=existing.id,
                        job_id=job.id,
                        status=job.status,
                        existing=True,
                    )
                if existing and force:
                    job = await repo.get_job_for_version(existing.id)
                    await repo.set_version_status(existing.id, VersionStatus.QUEUED)
                    await repo.update_job(
                        job.id,
                        status=JobStatus.QUEUED,
                        current_step="queued",
                        completed=0,
                        total=5,
                    )
                    return UploadResponse(
                        document_id=document_id,
                        version_id=existing.id,
                        job_id=job.id,
                        status=JobStatus.QUEUED,
                        existing=False,
                    )
                version_no = await repo.next_version_no(document_id)
            else:
                version_no = 1

            key = (
                f"{auth.tenant_id}/{document_id}/{version_id}/source/"
                f"{safe_filename(filename)}"
            )
            stored = await self.storage.put(
                self.settings.minio_bucket_originals,
                key,
                content,
                media_type,
            )
            job_id = str(uuid.uuid4())
            version = DocumentVersionORM(
                id=version_id,
                document_id=document_id,
                version_no=version_no,
                file_hash=file_hash,
                source_object_key=stored.key,
                source_object_etag=stored.etag,
                source_media_type=media_type,
                source_filename=filename,
                parser_name=self.parser.name,
                parser_version=self.parser.version,
                embedding_model=self.settings.embedding_model,
                embedding_revision=self.settings.embedding_revision,
                embedding_dimension=self.settings.embedding_dimension,
                status=VersionStatus.QUEUED,
            )
            job = IngestionJobORM(
                id=job_id,
                tenant_id=auth.tenant_id,
                document_id=document_id,
                version_id=version_id,
                status=JobStatus.QUEUED,
                current_step="queued",
                completed=0,
                total=5,
                errors=[],
            )
            try:
                if new_document:
                    document = DocumentORM(
                        id=document_id,
                        tenant_id=auth.tenant_id,
                        name=name,
                        source_type=media_type,
                        status="processing",
                        created_by=auth.user_id,
                    )
                    await repo.create_document(
                        document=document,
                        version=version,
                        job=job,
                        acl_scopes=acl_scopes,
                    )
                else:
                    await repo.add_version(version, job)
            except IntegrityError:
                await session.rollback()
                existing = await repo.get_version_by_content(
                    document_id,
                    file_hash,
                    self.parser.version,
                    self.settings.embedding_model,
                )
                if existing is None:
                    if new_document:
                        await self._cleanup_failed_source(key)
                    raise
                existing_job = await repo.get_job_for_version(existing.id)
                return UploadResponse(
                    document_id=document_id,
                    version_id=existing.id,
                    job_id=existing_job.id,
                    status=existing_job.status,
                    existing=True,
                )
            except Exception:
                await session.rollback()
                if new_document:
                    await self._cleanup_failed_source(key)
                raise
        return UploadResponse(
            document_id=document_id,
            version_id=version_id,
            job_id=job_id,
            status=JobStatus.QUEUED,
        )

    async def _cleanup_failed_source(self, key: str) -> None:
        try:
            await self.storage.delete_prefix(self.settings.minio_bucket_originals, key)
        except Exception:
            logger.warning("failed to clean up source object after database error", exc_info=True)

    async def get_document(self, document_id: str, auth: AuthContext) -> DocumentInfo:
        async with self.database.session() as session:
            repo = Repository(session)
            document = await repo.get_document(document_id)
            if document.tenant_id != auth.tenant_id or not await repo.can_admin(
                document_id, auth
            ):
                raise PermissionDeniedError("document not found")
            return DocumentInfo(
                document_id=document.id,
                name=document.name,
                source_type=document.source_type,
                status=document.status,
                active_version_id=document.active_version_id,
                created_at=document.created_at,
            )

    async def list_versions(self, document_id: str, auth: AuthContext) -> list[VersionInfo]:
        await self.get_document(document_id, auth)
        async with self.database.session() as session:
            versions = await Repository(session).list_versions(document_id)
            return [
                VersionInfo(
                    version_id=item.id,
                    version_no=item.version_no,
                    status=item.status,
                    file_hash=item.file_hash,
                    parser_name=item.parser_name,
                    parser_version=item.parser_version,
                    embedding_model=item.embedding_model,
                    created_at=item.created_at,
                )
                for item in versions
            ]

    async def get_job(self, job_id: str, auth: AuthContext) -> JobResponse:
        async with self.database.session() as session:
            repo = Repository(session)
            snapshot = await repo.get_job(job_id)
            job = await session.get(IngestionJobORM, job_id)
            if job is None or job.tenant_id != auth.tenant_id:
                raise PermissionDeniedError("job not found")
            if not await repo.can_read(job.document_id, auth):
                raise PermissionDeniedError("job not found")
            return JobResponse(
                job_id=snapshot.id,
                status=snapshot.status,
                current_step=snapshot.current_step,
                progress=JobProgress(
                    completed=snapshot.completed,
                    total=snapshot.total,
                ),
                errors=snapshot.errors,
                created_at=snapshot.created_at,
                updated_at=snapshot.updated_at,
            )

    async def delete(self, document_id: str, auth: AuthContext) -> DeleteResponse:
        async with self.database.session() as session:
            repo = Repository(session)
            document = await repo.get_document(document_id)
            if document.tenant_id != auth.tenant_id or not await repo.can_read(document_id, auth):
                raise PermissionDeniedError("document not found")
            await repo.mark_document_deleting(document_id)
        await self.vector.delete_document(document_id)
        prefix = f"{auth.tenant_id}/{document_id}/"
        for bucket in (
            self.settings.minio_bucket_originals,
            self.settings.minio_bucket_assets,
            self.settings.minio_bucket_derived,
        ):
            await self.storage.delete_prefix(bucket, prefix)
        async with self.database.session() as session:
            await Repository(session).mark_document_deleted(document_id)
        return DeleteResponse(document_id=document_id, status="deleted")

    def _validate(self, filename: str, content: bytes, media_type: str | None) -> str:
        suffix = "." + filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
        expected = SUPPORTED_TYPES.get(suffix)
        if expected is None or not self.parser.supports(media_type or expected, filename):
            raise InvalidDocumentError("supported formats: PDF, DOCX, PPTX")
        if not content:
            raise InvalidDocumentError("file is empty")
        if len(content) > self.settings.max_upload_bytes:
            raise InvalidDocumentError("file exceeds MAX_UPLOAD_BYTES")
        return expected


class IngestionService:
    def __init__(
        self,
        *,
        settings: Settings,
        database: Database,
        storage: AssetStorage,
        vector: VectorRepository,
        parser: DocumentParser,
        ocr: OCRProvider,
        vlm: VLMProvider,
        embedding: EmbeddingProvider,
    ) -> None:
        self.settings = settings
        self.database = database
        self.storage = storage
        self.vector = vector
        self.parser = parser
        self.ocr = ocr
        self.vlm = vlm
        self.embedding = embedding
        self.chunker = StructureAwareChunker()

    async def run(self, version_id: str) -> None:
        async with self.database.session() as session:
            repo = Repository(session)
            version = await repo.get_version(version_id)
            job = await repo.get_job_for_version(version_id)
            document = await repo.get_document(version.document_id)
            if job.status == JobStatus.SUCCEEDED:
                return
            await repo.update_job(
                job.id,
                status=JobStatus.RUNNING,
                current_step="parsing",
                completed=0,
                total=5,
                clear_errors=True,
            )
            await repo.set_version_status(version_id, VersionStatus.PARSING)
            tenant_id = document.tenant_id
            document_id = document.id
        try:
            source = await self.storage.read(
                self.settings.minio_bucket_originals,
                version.source_object_key,
            )
            parsed = await asyncio.to_thread(
                self.parser.parse,
                BytesIO(source),
                filename=version.source_filename,
                document_id=document_id,
                version_id=version_id,
            )
            await self._progress(job.id, "persisting", 1)
            nodes, assets = await self._build_nodes(
                tenant_id=tenant_id,
                document_id=document_id,
                version_id=version_id,
                parsed=parsed,
            )
            await self.storage.put(
                self.settings.minio_bucket_derived,
                f"{tenant_id}/{document_id}/{version_id}/parser/docling_document.json",
                json.dumps(parsed.raw, ensure_ascii=False, default=str).encode("utf-8"),
                "application/json",
            )
            async with self.database.session() as session:
                repo = Repository(session)
                await repo.replace_nodes(version_id, nodes, assets)
                acl_scopes = await repo.acl_scope_ids(document_id)
                await repo.set_version_status(version_id, VersionStatus.BASE_INDEXING)
            await self._progress(job.id, "embedding", 3)
            inputs = await self._embedding_inputs(nodes, assets)
            vectors = await self.embedding.embed_documents(inputs)
            await self.vector.upsert(
                nodes,
                vectors,
                acl_scope_ids=acl_scopes,
                status="building",
            )
            await self._progress(job.id, "activating", 4)
            async with self.database.session() as session:
                previous = await Repository(session).activate_version(document_id, version_id)
            await self.vector.set_version_status(version_id, "active")
            if previous:
                await self.vector.set_version_status(previous, "superseded")
            await self._progress(job.id, "complete", 5, status=JobStatus.SUCCEEDED)
        except Exception as exc:
            logger.exception("ingestion failed", extra={"version_id": version_id})
            async with self.database.session() as session:
                repo = Repository(session)
                await repo.set_version_status(version_id, VersionStatus.FAILED)
                await repo.update_job(
                    job.id,
                    status=JobStatus.FAILED,
                    current_step="failed",
                    error={
                        "type": type(exc).__name__,
                        "message": str(exc)[:1000],
                    },
                )

    async def _build_nodes(
        self,
        *,
        tenant_id: str,
        document_id: str,
        version_id: str,
        parsed: Any,
    ) -> tuple[list[UnifiedNode], list[AssetORM]]:
        nodes: list[UnifiedNode] = []
        assets: list[AssetORM] = []
        chunks = self.chunker.chunk(parsed.blocks)
        for chunk in chunks:
            content_hash = sha256_bytes(chunk.text.encode("utf-8"))
            node_id = deterministic_node_id(
                version_id,
                NodeType.TEXT,
                chunk.page_no,
                chunk.ordinal,
                content_hash,
            )
            nodes.append(
                UnifiedNode(
                    node_id=node_id,
                    tenant_id=tenant_id,
                    document_id=document_id,
                    version_id=version_id,
                    node_type=NodeType.TEXT,
                    page_no=chunk.page_no,
                    section_path=chunk.section_path[:10],
                    ordinal=chunk.ordinal,
                    text=chunk.text,
                    bbox=chunk.bbox,
                    content_hash=content_hash,
                    parser_name=self.parser.name,
                    parser_version=self.parser.version,
                    embedding_model=self.settings.embedding_model,
                    embedding_dimension=self.settings.embedding_dimension,
                    enrichment_status="complete",
                )
            )
        text_count = len(nodes)
        for image_index, image in enumerate(parsed.images):
            content_hash = sha256_bytes(image.content)
            ordinal = text_count + image_index
            node_id = deterministic_node_id(
                version_id,
                NodeType.IMAGE,
                image.page_no,
                ordinal,
                content_hash,
            )
            extension = "png" if image.media_type == "image/png" else "jpg"
            folder = "figures" if image.kind == ImageKind.FIGURE else "pages"
            filename = (
                f"{node_id}.{extension}"
                if image.kind == ImageKind.FIGURE
                else f"page_{(image.page_no or 0):06d}.{extension}"
            )
            key = f"{tenant_id}/{document_id}/{version_id}/{folder}/{filename}"
            stored = await self.storage.put(
                self.settings.minio_bucket_assets,
                key,
                image.content,
                image.media_type,
            )
            asset_id = deterministic_asset_id(version_id, key, content_hash)
            related = _related_text(image.page_no, image.section_path, chunks)
            ocr_text, description, enrichment = await self._enrich_image(
                image.content,
                image.caption,
                image.section_path,
                tenant_id=tenant_id,
                document_id=document_id,
                version_id=version_id,
                asset_id=asset_id,
            )
            width, height = _image_dimensions(image.content)
            nodes.append(
                UnifiedNode(
                    node_id=node_id,
                    tenant_id=tenant_id,
                    document_id=document_id,
                    version_id=version_id,
                    node_type=NodeType.IMAGE,
                    page_no=image.page_no,
                    section_path=image.section_path[:10],
                    ordinal=ordinal,
                    asset_id=asset_id,
                    image_kind=image.kind,
                    caption=(image.caption or "")[:300] or None,
                    ocr_text=ocr_text[:1000] or None,
                    description=description[:800] or None,
                    related_text=related,
                    bbox=image.bbox,
                    content_hash=content_hash,
                    parser_name=self.parser.name,
                    parser_version=self.parser.version,
                    embedding_model=self.settings.embedding_model,
                    embedding_dimension=self.settings.embedding_dimension,
                    enrichment_status=enrichment,
                    metadata={"media_type": image.media_type},
                )
            )
            assets.append(
                AssetORM(
                    id=asset_id,
                    tenant_id=tenant_id,
                    document_id=document_id,
                    version_id=version_id,
                    node_id=node_id,
                    bucket_name=stored.bucket,
                    object_key=stored.key,
                    object_version_id=stored.version_id,
                    object_etag=stored.etag,
                    media_type=image.media_type,
                    byte_size=stored.size,
                    width=width,
                    height=height,
                    sha256=content_hash,
                )
            )
        for index, node in enumerate(nodes):
            node.previous_node_id = nodes[index - 1].node_id if index else None
            node.next_node_id = nodes[index + 1].node_id if index + 1 < len(nodes) else None
        return nodes, assets

    async def _enrich_image(
        self,
        content: bytes,
        caption: str | None,
        section_path: list[str],
        *,
        tenant_id: str,
        document_id: str,
        version_id: str,
        asset_id: str,
    ) -> tuple[str, str, str]:
        ocr_text = ""
        description = ""
        status = "complete"
        try:
            ocr = await self.ocr.recognize(content, language=["ch", "en"])
            ocr_text = ocr.normalized_text
            if ocr.raw:
                await self.storage.put(
                    self.settings.minio_bucket_derived,
                    f"{tenant_id}/{document_id}/{version_id}/ocr/{asset_id}.json",
                    json.dumps(ocr.raw, ensure_ascii=False).encode("utf-8"),
                    "application/json",
                )
        except Exception:
            logger.warning("image OCR failed", exc_info=True, extra={"asset_id": asset_id})
            status = "partial"
        try:
            described = await self.vlm.describe_image(
                content,
                caption=caption,
                section_path=section_path,
            )
            description = described.summary
            if described.raw:
                await self.storage.put(
                    self.settings.minio_bucket_derived,
                    f"{tenant_id}/{document_id}/{version_id}/vlm/{asset_id}.json",
                    json.dumps(described.raw, ensure_ascii=False).encode("utf-8"),
                    "application/json",
                )
        except Exception:
            logger.warning("image VLM failed", exc_info=True, extra={"asset_id": asset_id})
            status = "partial"
        return ocr_text, description, status

    async def _embedding_inputs(
        self,
        nodes: Sequence[UnifiedNode],
        assets: Sequence[AssetORM],
    ) -> list[MultimodalInput]:
        asset_by_id = {asset.id: asset for asset in assets}
        result = []
        for node in nodes:
            section = " / ".join(node.section_path)
            if node.node_type == NodeType.TEXT:
                result.append(
                    MultimodalInput(
                        text=f"章节：{section}\n正文：{node.text or ''}",
                        instruction=TEXT_DOCUMENT_INSTRUCTION,
                    )
                )
                continue
            asset = asset_by_id[node.asset_id or ""]
            image = await self.storage.read(asset.bucket_name, asset.object_key)
            related = "\n".join(node.related_text)[:3200]
            result.append(
                MultimodalInput(
                    text=(
                        f"章节：{section}\n标题：{node.caption or ''}\n"
                        f"OCR：{node.ocr_text or ''}\n图片描述：{node.description or ''}\n"
                        f"相关正文：{related}"
                    ),
                    image=image,
                    instruction=IMAGE_DOCUMENT_INSTRUCTION,
                )
            )
        return result

    async def _progress(
        self,
        job_id: str,
        step: str,
        completed: int,
        *,
        status: str | None = None,
    ) -> None:
        async with self.database.session() as session:
            await Repository(session).update_job(
                job_id,
                status=status,
                current_step=step,
                completed=completed,
                total=5,
            )


class SearchService:
    def __init__(
        self,
        *,
        settings: Settings,
        database: Database,
        storage: AssetStorage,
        vector: VectorRepository,
        embedding: EmbeddingProvider,
        reranker: RerankerProvider,
    ) -> None:
        self.settings = settings
        self.database = database
        self.storage = storage
        self.vector = vector
        self.embedding = embedding
        self.reranker = reranker

    async def search(
        self,
        request: SearchRequest,
        auth: AuthContext,
        *,
        trace_id: str,
    ) -> SearchResponse:
        started = time.perf_counter()
        phase = started
        query_vector = await self.embedding.embed_query(
            MultimodalInput(text=request.query, instruction=QUERY_INSTRUCTION)
        )
        embedding_ms, phase = _elapsed_ms(phase)
        async def retrieve(node_type: NodeType, limit: int) -> list[Any]:
            if limit == 0:
                return []
            return await self.vector.search(
                query_vector,
                tenant_id=auth.tenant_id,
                node_type=node_type,
                acl_scope_ids=auth.scope_ids,
                document_ids=request.filters.document_ids,
                limit=limit,
            )

        text_hits, image_hits = await asyncio.gather(
            retrieve(NodeType.TEXT, request.text_candidate_k),
            retrieve(NodeType.IMAGE, request.image_candidate_k),
        )
        retrieval_ms, phase = _elapsed_ms(phase)
        fused = reciprocal_rank_fusion(
            [
                [hit.node_id for hit in text_hits],
                [hit.node_id for hit in image_hits],
            ],
            k=self.settings.rrf_k,
            limit=min(50, request.text_candidate_k + request.image_candidate_k),
        )
        rrf_by_id = dict(fused)
        node_ids = [node_id for node_id, _ in fused]
        async with self.database.session() as session:
            repo = Repository(session)
            nodes = await repo.load_nodes(node_ids)
            names = await repo.document_names(list({node.document_id for node in nodes}))
            asset_records = {
                node.asset_id: await repo.get_asset(node.asset_id)
                for node in nodes
                if node.asset_id
            }
        loading_ms, phase = _elapsed_ms(phase)
        inputs: list[MultimodalInput] = []
        for node in nodes:
            if node.node_type == NodeType.TEXT:
                inputs.append(
                    MultimodalInput(
                        text=f"章节：{' / '.join(node.section_path)}\n正文：{node.text or ''}"
                    )
                )
            else:
                asset = asset_records[node.asset_id]
                image = await self.storage.read(asset.bucket_name, asset.object_key)
                inputs.append(
                    MultimodalInput(
                        text=(
                            f"标题：{node.caption or ''}\nOCR：{node.ocr_text or ''}\n"
                            f"图片描述：{node.description or ''}"
                        ),
                        image=image,
                    )
                )
        ranking_mode = "qwen3_vl_reranker"
        try:
            scores = await self.reranker.rerank(
                MultimodalInput(text=request.query),
                inputs,
                top_n=len(inputs),
            )
        except Exception as exc:
            if not self.settings.rerank_fallback:
                raise ProviderUnavailableError("reranker unavailable") from exc
            logger.warning("reranker unavailable, using RRF", exc_info=True)
            ranking_mode = "rrf_fallback"
            scores = []
        rerank_ms, _ = _elapsed_ms(phase)
        if scores:
            ranked = [(nodes[item.index], item.score) for item in scores]
            if self.settings.reranker_provider == "mock":
                ranking_mode = "mock_reranker"
        else:
            node_by_id = {node.id: node for node in nodes}
            ranked = [
                (node_by_id[node_id], score)
                for node_id, score in fused
                if node_id in node_by_id
            ]
        result_items = self._deduplicate(
            ranked,
            rrf_by_id,
            names,
            request,
        )
        total_ms = int((time.perf_counter() - started) * 1000)
        return SearchResponse(
            query=request.query,
            ranking_mode=ranking_mode,
            results=result_items[: request.top_k],
            timing_ms=Timing(
                query_embedding=embedding_ms,
                retrieval=retrieval_ms,
                candidate_loading=loading_ms,
                rerank=rerank_ms,
                total=total_ms,
            ),
            trace_id=trace_id,
        )

    def _deduplicate(
        self,
        ranked: Sequence[tuple[Any, float]],
        rrf_by_id: dict[str, float],
        names: dict[str, str],
        request: SearchRequest,
    ) -> list[SearchResult]:
        seen: set[str] = set()
        document_counts: Counter[str] = Counter()
        figure_pages = {
            (node.document_id, node.page_no)
            for node, _ in ranked
            if node.image_kind == ImageKind.FIGURE
        }
        results = []
        for node, score in ranked:
            if node.id in seen:
                continue
            if document_counts[node.document_id] >= self.settings.max_results_per_document:
                continue
            if (
                node.image_kind == ImageKind.PAGE_RENDER
                and (node.document_id, node.page_no) in figure_pages
            ):
                continue
            seen.add(node.id)
            document_counts[node.document_id] += 1
            result = SearchResult(
                result_id=node.id,
                type=node.node_type,
                score=float(score),
                rrf_score=rrf_by_id[node.id],
                document=DocumentRef(
                    document_id=node.document_id,
                    version_id=node.version_id,
                    name=names.get(node.document_id, ""),
                ),
                location=Location(
                    page_no=node.page_no,
                    section_path=node.section_path,
                ),
            )
            if node.node_type == NodeType.TEXT:
                result.text = TextResult(
                    content=node.text or "",
                    highlight=_highlight(node.text or "", request.query)
                    if request.include.snippets
                    else None,
                )
            elif request.include.image_metadata:
                result.image = ImageResult(
                    asset_id=node.asset_id,
                    uri=f"/v1/assets/{node.asset_id}",
                    kind=node.image_kind or ImageKind.FIGURE,
                    caption=node.caption,
                    description=node.description,
                )
            results.append(result)
        return results


class AssetService:
    def __init__(self, database: Database, storage: AssetStorage) -> None:
        self.database = database
        self.storage = storage

    async def get(self, asset_id: str, auth: AuthContext) -> Any:
        async with self.database.session() as session:
            repo = Repository(session)
            asset = await repo.get_asset(asset_id)
            try:
                document = await repo.get_document(asset.document_id)
            except NotFoundError as exc:
                raise PermissionDeniedError("asset not found") from exc
            if (
                document.tenant_id != auth.tenant_id
                or asset.tenant_id != auth.tenant_id
                or not await repo.can_read(asset.document_id, auth)
            ):
                raise PermissionDeniedError("asset not found")
        return await self.storage.get(asset.bucket_name, asset.object_key)


def _related_text(
    page_no: int | None,
    section_path: Sequence[str],
    chunks: Sequence[Any],
) -> list[str]:
    ranked = []
    for chunk in chunks:
        score = 0.0
        if chunk.section_path == list(section_path):
            score += 0.5
        if page_no is not None and chunk.page_no is not None:
            distance = abs(page_no - chunk.page_no)
            if distance <= 1:
                score += 0.35 - 0.05 * distance
        if score > 0:
            ranked.append((score, chunk.ordinal, chunk.text))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    result: list[str] = []
    total = 0
    for _, _, text in ranked[:4]:
        remaining = 3200 - total
        if remaining <= 0:
            break
        result.append(text[:remaining])
        total += len(result[-1])
    return result


def _image_dimensions(content: bytes) -> tuple[int | None, int | None]:
    try:
        with Image.open(BytesIO(content)) as image:
            return image.width, image.height
    except Exception:
        return None, None


def _highlight(text: str, query: str) -> str | None:
    terms = [term for term in query.replace("，", " ").split() if len(term) >= 2]
    lowered = text.lower()
    positions = [lowered.find(term.lower()) for term in terms]
    positions = [position for position in positions if position >= 0]
    if not positions:
        return text[:240] if text else None
    start = max(0, min(positions) - 80)
    end = min(len(text), start + 240)
    return ("…" if start else "") + text[start:end] + ("…" if end < len(text) else "")


def _elapsed_ms(started: float) -> tuple[int, float]:
    now = time.perf_counter()
    return int((now - started) * 1000), now
