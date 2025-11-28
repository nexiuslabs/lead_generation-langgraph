from __future__ import annotations

from fastapi import APIRouter, Request, Response
from fastapi.responses import StreamingResponse
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
    for name in (
        "content-type",
        "x-tenant-id",
        "cookie",
        "authorization",
        "accept",
    ):
        v = client_headers.get(name)
        if v:
            headers[name] = v
    # Ensure Accept header is present for stream endpoints
    if "/runs/stream" in path:
        acc = headers.get("accept")
        if not acc or "text/event-stream" not in acc:
            headers["accept"] = "text/event-stream"
    # Send request
    timeout_s = float(os.getenv("GRAPH_PROXY_TIMEOUT_SECONDS", "0")) or 600.0
    http_timeout = httpx.Timeout(timeout_s, read=timeout_s, write=timeout_s, connect=30.0)
    async with httpx.AsyncClient(timeout=http_timeout, follow_redirects=True) as client:
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
        logger.info(
            "graph_proxy forward method=%s path=%s tenant=%s",
            method.upper(),
            path,
            tenant_header,
        )
        req = client.build_request(
            method=method.upper(),
            url=target,
            headers=headers,
            content=body,
        )
        resp = await client.send(req, stream=True)
        if "/runs/stream" in path:
            logger.info(
                "graph_proxy upstream response path=%s status=%s content-type=%s",
                path,
                resp.status_code,
                resp.headers.get("content-type"),
            )

        hop_headers = {
            "connection",
            "keep-alive",
            "proxy-authenticate",
            "proxy-authorization",
            "te",
            "trailers",
            "transfer-encoding",
            "upgrade",
        }

        def _headers(drop_content_length: bool) -> dict[str, str]:
            excluded = set(hop_headers)
            if drop_content_length:
                excluded.add("content-length")
            return {
                key: value
                for key, value in resp.headers.items()
                if key.lower() not in excluded
            }

        content_type = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
        is_event_stream = content_type == "text/event-stream"

        if is_event_stream:
            async def resp_iterator():
                try:
                    async for chunk in resp.aiter_raw():
                        yield chunk
                except httpx.ReadError as exc:
                    logger.warning(
                        "graph_proxy stream closed early path=%s tenant=%s error=%s",
                        path,
                        tenant_header,
                        exc,
                    )
                finally:
                    await resp.aclose()

            return StreamingResponse(
                resp_iterator(),
                status_code=resp.status_code,
                headers=_headers(drop_content_length=True),
                media_type=resp.headers.get("content-type"),
            )

        content = await resp.aread()
        await resp.aclose()
        return Response(
            content=content,
            status_code=resp.status_code,
            headers=_headers(drop_content_length=False),
        )


@router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
async def proxy_all(path: str, request: Request):
    return await _forward(request, request.method, path)
