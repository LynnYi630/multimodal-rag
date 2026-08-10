from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncIterator
from io import BytesIO
from pathlib import Path

from minio import Minio

from app.config import Settings
from app.domain.models import ObjectStream, StoredObject


def _safe_path(root: Path, bucket: str, key: str) -> Path:
    target = (root / bucket / key).resolve()
    bucket_root = (root / bucket).resolve()
    if target != bucket_root and bucket_root not in target.parents:
        raise ValueError("invalid object key")
    return target


class FileSystemStorage:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    async def initialize(self) -> None:
        await asyncio.to_thread(self.root.mkdir, parents=True, exist_ok=True)

    async def put(
        self,
        bucket: str,
        key: str,
        content: bytes,
        media_type: str,
    ) -> StoredObject:
        path = _safe_path(self.root, bucket, key)
        await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(path.write_bytes, content)
        etag = hashlib.md5(content, usedforsecurity=False).hexdigest()  # noqa: S324
        return StoredObject(
            bucket=bucket,
            key=key,
            etag=etag,
            media_type=media_type,
            size=len(content),
        )

    async def get(self, bucket: str, key: str) -> ObjectStream:
        path = _safe_path(self.root, bucket, key)
        if not await asyncio.to_thread(path.is_file):
            raise FileNotFoundError(key)
        size = path.stat().st_size
        etag = await asyncio.to_thread(_file_md5, path)

        async def chunks() -> AsyncIterator[bytes]:
            file_obj = await asyncio.to_thread(path.open, "rb")
            try:
                while chunk := await asyncio.to_thread(file_obj.read, 64 * 1024):
                    yield chunk
            finally:
                await asyncio.to_thread(file_obj.close)

        return ObjectStream(
            chunks=chunks(),
            etag=etag,
            media_type=_guess_media_type(path),
            size=size,
        )

    async def read(self, bucket: str, key: str) -> bytes:
        return await asyncio.to_thread(_safe_path(self.root, bucket, key).read_bytes)

    async def delete_prefix(self, bucket: str, prefix: str) -> None:
        base = _safe_path(self.root, bucket, prefix)
        if not await asyncio.to_thread(base.exists):
            return
        files = await asyncio.to_thread(
            lambda: [path for path in base.rglob("*") if path.is_file()]
        )
        for path in files:
            await asyncio.to_thread(path.unlink)
        directories = await asyncio.to_thread(
            lambda: sorted(
                [path for path in base.rglob("*") if path.is_dir()],
                key=lambda path: len(path.parts),
                reverse=True,
            )
        )
        for directory in directories:
            await asyncio.to_thread(directory.rmdir)
        if base.is_dir():
            await asyncio.to_thread(base.rmdir)


class MinioStorage:
    def __init__(self, settings: Settings) -> None:
        self.client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )
        self.buckets = [
            settings.minio_bucket_originals,
            settings.minio_bucket_assets,
            settings.minio_bucket_derived,
        ]

    async def initialize(self) -> None:
        for bucket in self.buckets:
            exists = await asyncio.to_thread(self.client.bucket_exists, bucket)
            if not exists:
                await asyncio.to_thread(self.client.make_bucket, bucket)

    async def put(
        self,
        bucket: str,
        key: str,
        content: bytes,
        media_type: str,
    ) -> StoredObject:
        result = await asyncio.to_thread(
            self.client.put_object,
            bucket,
            key,
            BytesIO(content),
            len(content),
            content_type=media_type,
        )
        return StoredObject(
            bucket=bucket,
            key=key,
            etag=result.etag,
            media_type=media_type,
            size=len(content),
            version_id=result.version_id,
        )

    async def get(self, bucket: str, key: str) -> ObjectStream:
        stat = await asyncio.to_thread(self.client.stat_object, bucket, key)
        response = await asyncio.to_thread(self.client.get_object, bucket, key)

        async def chunks() -> AsyncIterator[bytes]:
            try:
                while chunk := await asyncio.to_thread(response.read, 64 * 1024):
                    yield chunk
            finally:
                response.close()
                response.release_conn()

        return ObjectStream(
            chunks=chunks(),
            etag=stat.etag,
            media_type=stat.content_type or "application/octet-stream",
            size=stat.size,
        )

    async def read(self, bucket: str, key: str) -> bytes:
        response = await asyncio.to_thread(self.client.get_object, bucket, key)
        try:
            return await asyncio.to_thread(response.read)
        finally:
            response.close()
            response.release_conn()

    async def delete_prefix(self, bucket: str, prefix: str) -> None:
        objects = await asyncio.to_thread(
            lambda: list(self.client.list_objects(bucket, prefix=prefix, recursive=True))
        )
        for item in objects:
            await asyncio.to_thread(self.client.remove_object, bucket, item.object_name)


def _file_md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)  # noqa: S324
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _guess_media_type(path: Path) -> str:
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".json": "application/json",
        ".jsonl": "application/x-ndjson",
    }.get(path.suffix.lower(), "application/octet-stream")
