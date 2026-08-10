from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Header, Request

from app.config import Settings
from app.domain.models import AuthContext
from app.runtime import Container


def get_container(request: Request) -> Container:
    return request.app.state.container


def get_auth_context(
    request: Request,
    x_tenant_id: Annotated[str | None, Header()] = None,
    x_user_id: Annotated[str | None, Header()] = None,
    x_roles: Annotated[str | None, Header()] = None,
    x_groups: Annotated[str | None, Header()] = None,
) -> AuthContext:
    settings: Settings = request.app.state.container.settings
    return AuthContext(
        tenant_id=x_tenant_id or settings.default_tenant_id,
        user_id=x_user_id or settings.default_user_id,
        roles=_csv(x_roles),
        groups=_csv(x_groups),
    )


def _csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


AuthDependency = Annotated[AuthContext, Depends(get_auth_context)]
ContainerDependency = Annotated[Container, Depends(get_container)]
