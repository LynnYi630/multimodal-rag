from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.domain.models import ImageKind, ParsedBlock, ParsedImage, UnifiedDocument
from app.main import create_app


def test_upload_ingest_search_and_rbac(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        database_url=f"sqlite+aiosqlite:///{(tmp_path / 'rag.db').as_posix()}",
        local_storage_root=tmp_path / "objects",
        parser_provider="plain_text",
        vector_provider="memory",
        storage_provider="filesystem",
        embedding_provider="mock",
        reranker_provider="mock",
        celery_task_always_eager=True,
        auto_create_schema=True,
    )
    app = create_app(settings)
    owner_headers = {
        "X-Tenant-ID": "tenant-a",
        "X-User-ID": "owner",
    }
    with TestClient(app) as client:
        upload = client.post(
            "/v1/documents",
            headers=owner_headers,
            files={
                "file": (
                    "sample.txt",
                    "多模态检索系统使用统一向量空间。".encode(),
                    "text/plain",
                )
            },
            data={"parser": "plain_text"},
        )
        assert upload.status_code == 202, upload.text
        identifiers = upload.json()

        job = client.get(f"/v1/jobs/{identifiers['job_id']}", headers=owner_headers)
        assert job.status_code == 200
        assert job.json()["status"] == "succeeded"

        search = client.post(
            "/v1/search",
            headers=owner_headers,
            json={"query": "统一向量空间", "top_k": 5},
        )
        assert search.status_code == 200, search.text
        assert search.json()["results"][0]["type"] == "text"
        assert search.json()["ranking_mode"] == "mock_reranker"

        forced = client.post(
            f"/v1/documents/{identifiers['document_id']}/versions?force=true",
            headers=owner_headers,
            files={
                "file": (
                    "sample.txt",
                    "多模态检索系统使用统一向量空间。".encode(),
                    "text/plain",
                )
            },
        )
        assert forced.status_code == 202
        assert forced.json()["version_id"] == identifiers["version_id"]
        versions = client.get(
            f"/v1/documents/{identifiers['document_id']}/versions",
            headers=owner_headers,
        )
        assert len(versions.json()) == 1

        outsider = client.post(
            "/v1/search",
            headers={"X-Tenant-ID": "tenant-a", "X-User-ID": "outsider"},
            json={"query": "统一向量空间"},
        )
        assert outsider.status_code == 200
        assert outsider.json()["results"] == []

        deleted = client.delete(
            f"/v1/documents/{identifiers['document_id']}",
            headers=owner_headers,
        )
        assert deleted.status_code == 202
        after_delete = client.post(
            "/v1/search",
            headers=owner_headers,
            json={"query": "统一向量空间"},
        )
        assert after_delete.json()["results"] == []


def test_image_result_asset_etag_and_deleted_asset_are_protected(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        database_url=f"sqlite+aiosqlite:///{(tmp_path / 'images.db').as_posix()}",
        local_storage_root=tmp_path / "objects",
        parser_provider="plain_text",
        vector_provider="memory",
        storage_provider="filesystem",
        celery_task_always_eager=True,
    )
    app = create_app(settings)
    headers = {"X-Tenant-ID": "tenant-a", "X-User-ID": "owner"}
    with TestClient(app) as client:
        parser = _ImageFixtureParser()
        container = app.state.container
        container.parser = parser
        container.documents.parser = parser
        container.ingestion.parser = parser

        upload = client.post(
            "/v1/documents",
            headers=headers,
            files={"file": ("fixture.pdf", b"fixture", "application/pdf")},
            data={"parser": "fixture"},
        )
        assert upload.status_code == 202, upload.text
        document_id = upload.json()["document_id"]

        search = client.post(
            "/v1/search",
            headers=headers,
            json={
                "query": "多模态架构图",
                "top_k": 5,
                "text_candidate_k": 0,
                "image_candidate_k": 20,
            },
        )
        assert search.status_code == 200, search.text
        assert all(item["type"] == "image" for item in search.json()["results"])
        image_result = next(
            item for item in search.json()["results"] if item["type"] == "image"
        )
        uri = image_result["image"]["uri"]
        asset = client.get(uri, headers=headers)
        assert asset.status_code == 200
        assert asset.headers["content-type"] == "image/png"
        etag = asset.headers["etag"]
        assert client.get(
            uri,
            headers={**headers, "If-None-Match": etag},
        ).status_code == 304
        assert client.get(
            uri,
            headers={"X-Tenant-ID": "tenant-a", "X-User-ID": "outsider"},
        ).status_code == 404

        client.delete(f"/v1/documents/{document_id}", headers=headers)
        assert client.get(uri, headers=headers).status_code == 404


class _ImageFixtureParser:
    name = "fixture"
    version = "1"

    def supports(self, media_type: str, filename: str) -> bool:
        return True

    def parse(
        self,
        file_obj,
        *,
        filename: str,
        document_id: str,
        version_id: str,
    ) -> UnifiedDocument:
        output = BytesIO()
        Image.new("RGB", (8, 8), "white").save(output, format="PNG")
        return UnifiedDocument(
            blocks=[
                ParsedBlock(
                    text="这里是多模态架构的技术说明。",
                    page_no=1,
                    section_path=["系统架构"],
                    ordinal=0,
                )
            ],
            images=[
                ParsedImage(
                    content=output.getvalue(),
                    media_type="image/png",
                    page_no=1,
                    ordinal=0,
                    kind=ImageKind.FIGURE,
                    caption="多模态架构图",
                    section_path=["系统架构"],
                )
            ],
        )
