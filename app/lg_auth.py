"""
LangGraph SDK Auth for local dev and production.

This module exports `auth`, an instance of `langgraph_sdk.Auth`, as required by
langgraph-api's custom auth loader (expects an Auth instance, not a Starlette backend).

Behavior
- Accepts Authorization: Bearer <token> OR the access cookie (ACCESS_COOKIE_NAME, default nx_access).
- Emits a simple identity string; scopes can be extended later if needed.
"""
from __future__ import annotations

import os
from fastapi import HTTPException
from starlette.authentication import AuthenticationError
from starlette.requests import Request
from langgraph_sdk import Auth

from app.auth import verify_jwt


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _allow_anon_identity() -> bool:
    if _is_truthy(os.getenv("LANGGRAPH_ALLOW_ANON")):
        return True
    variant = (os.getenv("LANGSMITH_LANGGRAPH_API_VARIANT", "") or "").strip().lower()
    return variant == "local_dev"


auth = Auth()


@auth.authenticate
async def authenticate(request: Request, authorization: str | None = None):
    cookie_name = os.getenv("ACCESS_COOKIE_NAME", "nx_access")
    # Prefer cookie-based token for LangGraph streams so cookie rotation keeps sessions alive
    token = request.cookies.get(cookie_name)
    if not token and authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    allow_anon = _allow_anon_identity()
    if not token:
        if allow_anon:
            return "anon"
        raise AuthenticationError("Missing credentials")
    try:
        claims = verify_jwt(token)
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        raise AuthenticationError(detail)
    tenant_id = claims.get("tenant_id")
    if tenant_id:
        request.state.tenant_id = tenant_id
        request.state.roles = claims.get("roles", [])
        return f"tenant:{tenant_id}"
    if allow_anon:
        return "anon"
    raise AuthenticationError("Missing tenant_id claim")
