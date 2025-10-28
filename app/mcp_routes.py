from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request

from app.auth import require_optional_identity

router = APIRouter(prefix="/mcp", tags=["mcp-bridge"])

log = logging.getLogger("app.mcp")
if not log.handlers:
    h = logging.StreamHandler()
    log.addHandler(h)
log.setLevel(logging.INFO)


def _arg(payload: Dict[str, Any] | None, key: str, default: Any = None) -> Any:
    if not payload:
        return default
    if key in payload:
        return payload.get(key, default)
    # Also support {"arguments": {...}} wrapper
    args = payload.get("arguments") if isinstance(payload.get("arguments"), dict) else None  # type: ignore[assignment]
    if isinstance(args, dict) and key in args:
        return args.get(key, default)
    return default


@router.get("/servers")
async def list_servers(_: dict = Depends(require_optional_identity)) -> Dict[str, Any]:
    return {"servers": ["jina"]}


@router.get("/servers/{server}/tools")
async def list_tools(server: str, _: dict = Depends(require_optional_identity)) -> Dict[str, Any]:
    if not server:
        raise HTTPException(status_code=400, detail="missing server name")
    # Minimal tool listing for discovery/debugging
    return {
        "server": server,
        "tools": [
            "read_url",
            "search_web",
            "parallel_search_web",
        ],
    }


@router.post("/servers/{server}/tools/{tool}/invoke")
async def invoke_tool(
    request: Request,
    server: str,
    tool: str,
    payload: Dict[str, Any] | None = None,
    _: dict = Depends(require_optional_identity),
) -> Any:
    # Accept both direct args and {"arguments": {...}} shape
    tool = (tool or "").strip()
    server = (server or "").strip()
    if not server or not tool:
        raise HTTPException(status_code=400, detail="missing server/tool")

    # Light logging without leaking secrets
    try:
        auth_present = bool(request.headers.get("Authorization"))
        log.info("[mcp] invoke server=%s tool=%s auth=%s", server, tool, "yes" if auth_present else "no")
    except Exception:
        pass

    # Route to minimal supported tools; use direct remote MCP client to avoid recursion
    try:
        from src.services import mcp_remote
        if tool == "read_url":
            url = str(_arg(payload, "url", "") or "")
            timeout = float(_arg(payload, "timeout", 8.0) or 8.0)
            txt: Optional[str] = mcp_remote.read_url(url, timeout=timeout)
            return {"content": (txt or "")}
        if tool == "search_web":
            query = str(_arg(payload, "query", "") or "")
            country = _arg(payload, "country", None)
            # Bridge may send limit; remote uses per tool default; keep a cap
            limit = int(_arg(payload, "limit", 20) or 20)
            urls: List[str] = mcp_remote.search_web(query, country=country, max_results=limit)
            return {"results": urls[:limit]}
        if tool == "parallel_search_web":
            queries = _arg(payload, "queries", []) or []
            per_query = int(_arg(payload, "per_query", 10) or 10)
            if not isinstance(queries, list):
                queries = []
            mapping = mcp_remote.parallel_search_web(queries, per_query=per_query)
            # mapping already returns {query: [urls]}
            return mapping
        # Unknown tool
        return {"error": f"unsupported tool: {tool}"}
    except Exception as e:
        log.info("[mcp] invoke failed tool=%s err=%s", tool, e)
        # Return empty shape with 200 so bridge can fall back without hard fail
        if tool == "read_url":
            return {"content": ""}
        if tool == "search_web":
            return {"results": []}
        if tool == "parallel_search_web":
            return {"results": {}}
        return {"error": "invoke_error"}

