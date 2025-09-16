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
from starlette.authentication import AuthenticationError
from starlette.requests import Request
from langgraph_sdk import Auth


auth = Auth()


@auth.authenticate
async def authenticate(request: Request, authorization: str | None = None):
    cookie_name = os.getenv("ACCESS_COOKIE_NAME", "nx_access")
    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    if not token:
        token = request.cookies.get(cookie_name)
    if not token:
        # Allow anonymous identity in local dev when enabled or when the runtime variant is local_dev
        allow = (os.getenv("LANGGRAPH_ALLOW_ANON", "false") or "false").strip().lower() in ("1", "true", "yes", "on")
        variant = (os.getenv("LANGSMITH_LANGGRAPH_API_VARIANT", "") or "").strip().lower()
        if allow or variant == "local_dev":
            return "anon"
        raise AuthenticationError("Missing credentials")
    # Minimal identity; attach tenant hint when provided
    tid = request.headers.get("x-tenant-id")
    user_id = f"tenant:{tid}" if tid else "user"
    # You can also return (permissions, user) where permissions is a list[str]
    return user_id
