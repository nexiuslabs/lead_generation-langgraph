from __future__ import annotations

from fastapi import APIRouter, Request, Response
import os
import httpx
import logging

router = APIRouter(prefix="/graph", tags=["graph-proxy"])
logger = logging.getLogger("graph_proxy")


def _target_base() -> str | None:
    # Remote LangGraph server base URL, e.g. https://graph.example.com
    url = (os.getenv("LANGGRAPH_REMOTE_URL") or "").strip()
    return url or None


def _auth_headers() -> dict[str, str]:
    # Inject server-held credentials so the UI can remain cookie-only.
    # Supports LangGraph Studio style X-Api-Key and/or Authorization.
    h: dict[str, str] = {}
    api_key = (os.getenv("LANGSMITH_API_KEY") or "").strip()
    if api_key:
        h["X-Api-Key"] = api_key
    bearer = (os.getenv("LANGGRAPH_BEARER_TOKEN") or "").strip()
    if bearer:
        h["Authorization"] = f"Bearer {bearer}"
    return h


async def _forward(request: Request, method: str, path: str) -> Response:
    base = _target_base()
    if not base:
        return Response(status_code=501, content=b"Graph proxy not configured")
    # Build target URL preserving path and query string
    target = base.rstrip("/") + "/" + path.lstrip("/")
    # Compose headers: server auth + selected pass-throughs
    headers: dict[str, str] = {}
    headers.update(_auth_headers())
    client_headers = request.headers
    for name in ("content-type", "x-tenant-id", "cookie"):
        v = client_headers.get(name)
        if v:
            headers[name] = v
    # Send request
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        body = await request.body() if method.upper() not in ("GET", "HEAD") else None
        # Inject tenant context into run-start payloads so the graph can always resolve tenant_id
        try:
            if body and headers.get("content-type", "").lower().startswith("application/json"):
                # Target path shapes:
                #   /threads/{id}/runs
                #   /threads/{id}/runs/stream
                # We inject { context: { tenant_id: <X-Tenant-ID> } } using the incoming header if present.
                if "/threads/" in path and "/runs" in path:
                    raw = body.decode("utf-8")
                    payload = {}
                    try:
                        import json as _json
                        payload = _json.loads(raw or "{}")
                    except Exception:
                        payload = {}
                    tenant_hdr = request.headers.get("x-tenant-id") or request.headers.get("X-Tenant-ID")
                    if tenant_hdr:
                        ctx = payload.get("context") or {}
                        if not isinstance(ctx, dict):
                            ctx = {}
                        # Only set if not already provided by the client
                        ctx.setdefault("tenant_id", tenant_hdr)
                        payload["context"] = ctx
                        body = _json.dumps(payload).encode("utf-8")
                        headers["content-type"] = "application/json"
        except Exception:
            # Best-effort only; never block the proxy
            pass
        tenant_header = headers.get("x-tenant-id") or headers.get("X-Tenant-ID") or request.headers.get("x-tenant-id")
        logger.info("graph_proxy forward method=%s path=%s tenant=%s", method.upper(), path, tenant_header)
        resp = await client.request(method=method.upper(), url=target, headers=headers, content=body)
        return Response(content=resp.content, status_code=resp.status_code, headers=resp.headers)


@router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
async def proxy_all(path: str, request: Request):
    return await _forward(request, request.method, path)
