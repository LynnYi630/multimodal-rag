from __future__ import annotations

import asyncio

from fastapi import BackgroundTasks

from app.config import get_settings
from app.runtime import Container, build_container
from app.workers.celery_app import celery_app


def enqueue_ingestion(
    container: Container,
    version_id: str,
    background_tasks: BackgroundTasks,
) -> None:
    if container.settings.celery_task_always_eager:
        background_tasks.add_task(container.ingestion.run, version_id)
    else:
        ingest_document.delay(version_id)


@celery_app.task(
    bind=True,
    autoretry_for=(ConnectionError, TimeoutError),
    retry_backoff=True,
    retry_jitter=True,
    max_retries=5,
    name="rag.ingest_document",
)
def ingest_document(self, version_id: str) -> None:
    asyncio.run(_run_ingestion(version_id))


async def _run_ingestion(version_id: str) -> None:
    container = build_container(get_settings())
    await container.initialize()
    try:
        await container.ingestion.run(version_id)
    finally:
        await container.close()
