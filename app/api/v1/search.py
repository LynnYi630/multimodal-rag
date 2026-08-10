from __future__ import annotations

from fastapi import APIRouter, Request

from app.api.deps import AuthDependency, ContainerDependency
from app.application.schemas import SearchRequest, SearchResponse

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
async def search(
    payload: SearchRequest,
    request: Request,
    auth: AuthDependency,
    container: ContainerDependency,
) -> SearchResponse:
    return await container.search.search(
        payload,
        auth,
        trace_id=request.state.trace_id,
    )
