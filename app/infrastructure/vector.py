from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient, models

from app.config import Settings
from app.domain.models import UnifiedNode, VectorHit


class QdrantVectorRepository:
    def __init__(self, settings: Settings) -> None:
        self.client = AsyncQdrantClient(url=settings.qdrant_url)
        self.collection = settings.qdrant_collection
        self.dimension = settings.embedding_dimension
        self.model_name = settings.embedding_model
        self.model_revision = settings.embedding_revision

    async def initialize(self) -> None:
        collections = await self.client.get_collections()
        names = {item.name for item in collections.collections}
        if self.collection not in names:
            await self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=self.dimension,
                    distance=models.Distance.COSINE,
                ),
                on_disk_payload=True,
            )
        for field in [
            "tenant_id",
            "node_type",
            "document_id",
            "version_id",
            "status",
            "acl_scope_ids",
            "image_kind",
        ]:
            try:
                await self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            except Exception as exc:
                if "already exists" not in str(exc).lower():
                    raise

    async def upsert(
        self,
        nodes: Sequence[UnifiedNode],
        vectors: Sequence[Sequence[float]],
        *,
        acl_scope_ids: Sequence[str],
        status: str,
    ) -> None:
        if len(nodes) != len(vectors):
            raise ValueError("node/vector count mismatch")
        points = []
        for node, vector in zip(nodes, vectors, strict=True):
            if len(vector) != self.dimension:
                raise ValueError(f"expected vector dimension {self.dimension}")
            points.append(
                models.PointStruct(
                    id=node.node_id,
                    vector=list(vector),
                    payload={
                        "tenant_id": node.tenant_id,
                        "document_id": node.document_id,
                        "version_id": node.version_id,
                        "node_id": node.node_id,
                        "node_type": node.node_type.value,
                        "image_kind": node.image_kind.value if node.image_kind else None,
                        "page_no": node.page_no,
                        "section_path": node.section_path,
                        "ordinal": node.ordinal,
                        "asset_id": node.asset_id,
                        "caption": node.caption,
                        "status": status,
                        "acl_scope_ids": list(acl_scope_ids),
                        "embedding_model": self.model_name,
                        "embedding_revision": self.model_revision,
                        "embedding_dimension": self.dimension,
                        "content_hash": node.content_hash,
                        "enrichment_status": node.enrichment_status,
                    },
                )
            )
        for start in range(0, len(points), 128):
            await self.client.upsert(
                collection_name=self.collection,
                points=points[start : start + 128],
                wait=True,
            )

    async def search(
        self,
        vector: Sequence[float],
        *,
        tenant_id: str,
        node_type: str,
        acl_scope_ids: Sequence[str],
        document_ids: Sequence[str] | None,
        limit: int,
    ) -> list[VectorHit]:
        must = [
            models.FieldCondition(
                key="tenant_id",
                match=models.MatchValue(value=tenant_id),
            ),
            models.FieldCondition(
                key="status",
                match=models.MatchValue(value="active"),
            ),
            models.FieldCondition(
                key="node_type",
                match=models.MatchValue(value=node_type),
            ),
            models.FieldCondition(
                key="acl_scope_ids",
                match=models.MatchAny(any=list(acl_scope_ids)),
            ),
        ]
        if document_ids:
            must.append(
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchAny(any=list(document_ids)),
                )
            )
        response = await self.client.query_points(
            collection_name=self.collection,
            query=list(vector),
            query_filter=models.Filter(must=must),
            limit=limit,
            with_payload=True,
        )
        return [
            VectorHit(
                node_id=str(point.id),
                score=float(point.score),
                payload=dict(point.payload or {}),
            )
            for point in response.points
        ]

    async def set_version_status(self, version_id: str, status: str) -> None:
        await self.client.set_payload(
            collection_name=self.collection,
            payload={"status": status},
            points=models.Filter(
                must=[
                    models.FieldCondition(
                        key="version_id",
                        match=models.MatchValue(value=version_id),
                    )
                ]
            ),
            wait=True,
        )

    async def delete_document(self, document_id: str) -> None:
        await self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
            wait=True,
        )


@dataclass(slots=True)
class _MemoryPoint:
    vector: list[float]
    payload: dict


class InMemoryVectorRepository:
    """Process-local vector index for development and tests."""

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.points: dict[str, _MemoryPoint] = {}
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        return None

    async def upsert(
        self,
        nodes: Sequence[UnifiedNode],
        vectors: Sequence[Sequence[float]],
        *,
        acl_scope_ids: Sequence[str],
        status: str,
    ) -> None:
        if len(nodes) != len(vectors):
            raise ValueError("node/vector count mismatch")
        async with self._lock:
            for node, vector in zip(nodes, vectors, strict=True):
                if len(vector) != self.dimension:
                    raise ValueError(f"expected vector dimension {self.dimension}")
                self.points[node.node_id] = _MemoryPoint(
                    vector=list(vector),
                    payload={
                        "tenant_id": node.tenant_id,
                        "document_id": node.document_id,
                        "version_id": node.version_id,
                        "node_id": node.node_id,
                        "node_type": node.node_type.value,
                        "image_kind": node.image_kind.value if node.image_kind else None,
                        "asset_id": node.asset_id,
                        "status": status,
                        "acl_scope_ids": list(acl_scope_ids),
                    },
                )

    async def search(
        self,
        vector: Sequence[float],
        *,
        tenant_id: str,
        node_type: str,
        acl_scope_ids: Sequence[str],
        document_ids: Sequence[str] | None,
        limit: int,
    ) -> list[VectorHit]:
        scopes = set(acl_scope_ids)
        documents = set(document_ids or [])
        hits = []
        for node_id, point in self.points.items():
            payload = point.payload
            if (
                payload["tenant_id"] != tenant_id
                or payload["node_type"] != node_type
                or payload["status"] != "active"
                or not scopes.intersection(payload["acl_scope_ids"])
                or (documents and payload["document_id"] not in documents)
            ):
                continue
            hits.append(
                VectorHit(
                    node_id=node_id,
                    score=_cosine(vector, point.vector),
                    payload=dict(payload),
                )
            )
        return sorted(hits, key=lambda hit: (-hit.score, hit.node_id))[:limit]

    async def set_version_status(self, version_id: str, status: str) -> None:
        async with self._lock:
            for point in self.points.values():
                if point.payload["version_id"] == version_id:
                    point.payload["status"] = status

    async def delete_document(self, document_id: str) -> None:
        async with self._lock:
            for node_id in [
                node_id
                for node_id, point in self.points.items()
                if point.payload["document_id"] == document_id
            ]:
                del self.points[node_id]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)
