from __future__ import annotations

from fastapi import APIRouter, Header, Response, status
from fastapi.responses import StreamingResponse

from app.api.deps import AuthDependency, ContainerDependency

router = APIRouter(prefix="/assets", tags=["assets"])


@router.get("/{asset_id}")
async def get_asset(
    asset_id: str,
    auth: AuthDependency,
    container: ContainerDependency,
    if_none_match: str | None = Header(default=None),
) -> Response:
    item = await container.assets.get(asset_id, auth)
    etag = f'"{item.etag}"'
    headers = {
        "ETag": etag,
        "Cache-Control": "private, max-age=3600",
        "Content-Length": str(item.size),
        "X-Content-Type-Options": "nosniff",
    }
    if if_none_match in {etag, item.etag}:
        return Response(status_code=status.HTTP_304_NOT_MODIFIED, headers=headers)
    return StreamingResponse(
        item.chunks,
        media_type=item.media_type,
        headers=headers,
    )
