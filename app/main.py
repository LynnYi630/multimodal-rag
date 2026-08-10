from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.v1 import assets, documents, jobs, search
from app.config import Settings, get_settings
from app.domain.models import (
    DomainError,
    InvalidDocumentError,
    NotFoundError,
    PermissionDeniedError,
    ProviderUnavailableError,
)
from app.runtime import build_container


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or get_settings()
    logging.basicConfig(
        level=getattr(logging, resolved.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        container = build_container(resolved)
        await container.initialize()
        app.state.container = container
        try:
            yield
        finally:
            await container.close()

    app = FastAPI(
        title="Multimodal RAG Retrieval Service",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def trace_middleware(request: Request, call_next):
        trace_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.trace_id = trace_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = trace_id
        return response

    @app.exception_handler(DomainError)
    async def domain_error_handler(request: Request, exc: DomainError) -> JSONResponse:
        status_code = 400
        if isinstance(exc, (NotFoundError, PermissionDeniedError)):
            status_code = 404
        elif isinstance(exc, ProviderUnavailableError):
            status_code = 503
        elif isinstance(exc, InvalidDocumentError):
            status_code = 422
        return JSONResponse(
            status_code=status_code,
            content={
                "detail": str(exc),
                "trace_id": getattr(request.state, "trace_id", ""),
            },
        )

    @app.get("/health/live", tags=["health"])
    async def live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready", tags=["health"])
    async def ready(request: Request) -> dict[str, str]:
        container = request.app.state.container
        return {
            "status": "ready",
            "embedding": container.embedding.model_name,
            "reranker": container.reranker.model_name,
            "vector": container.settings.vector_provider,
            "storage": container.settings.storage_provider,
        }

    prefix = resolved.api_prefix
    app.include_router(documents.router, prefix=prefix)
    app.include_router(jobs.router, prefix=prefix)
    app.include_router(search.router, prefix=prefix)
    app.include_router(assets.router, prefix=prefix)
    return app


app = create_app()
