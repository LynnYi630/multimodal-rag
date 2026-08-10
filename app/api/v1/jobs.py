from __future__ import annotations

from fastapi import APIRouter

from app.api.deps import AuthDependency, ContainerDependency
from app.application.schemas import JobResponse

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: str,
    auth: AuthDependency,
    container: ContainerDependency,
) -> JobResponse:
    return await container.documents.get_job(job_id, auth)
