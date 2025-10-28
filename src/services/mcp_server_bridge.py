from __future__ import annotations

import json
import logging
import time
from typing import Optional, List, Dict, Any

import requests

from src.settings import (
    ENABLE_SERVER_MCP_BRIDGE,
    LGS_BASE_URL,
    MCP_SERVER_NAME,
    MCP_BRIDGE_INVOKE_URL,
    MCP_BRIDGE_HEADERS_JSON,
    MCP_API_KEY,
    MCP_BRIDGE_FORCE_AUTH,
    MCP_BRIDGE_COOL_OFF_S,
    MCP_BRIDGE_CONNECT_TIMEOUT_S,
    MCP_BRIDGE_READ_TIMEOUT_S,
)

log = logging.getLogger("mcp_server_bridge")

_BRIDGE_DOWN_UNTIL: float = 0.0  # cooldown when bridge repeatedly times out
_BRIDGE_COOL_OFF_S: float = MCP_BRIDGE_COOL_OFF_S


def _headers() -> Dict[str, str]:
    base = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    try:
        if MCP_BRIDGE_HEADERS_JSON:
            extra = json.loads(MCP_BRIDGE_HEADERS_JSON)
            if isinstance(extra, dict):
                for k, v in extra.items():
                    base[str(k)] = str(v)
    except Exception:
        pass
    # Add Authorization header only if explicitly configured. In local_dev, attaching
    # a non-JWT Authorization can cause the LangGraph auth middleware to return 403.
    if MCP_BRIDGE_FORCE_AUTH:
        try:
            auth_present = any(k.lower() == "authorization" for k in base.keys())
            if not auth_present and MCP_API_KEY:
                base["Authorization"] = f"Bearer {MCP_API_KEY}"
        except Exception:
            pass
    return base


def _invoke(tool: str, args: Dict[str, Any], timeout: float = 8.0) -> Optional[Dict[str, Any] | List[Any]]:
    global _BRIDGE_DOWN_UNTIL
    if not ENABLE_SERVER_MCP_BRIDGE:
        return None
    # If the bridge is in a cool-off, skip quickly to allow fallback paths
    now = time.time()
    if _BRIDGE_DOWN_UNTIL and now < _BRIDGE_DOWN_UNTIL:
        try:
            remain = int(max(0, _BRIDGE_DOWN_UNTIL - now))
            log.info("[mcp-bridge] skipped (cooldown active %ss) tool=%s", remain, tool)
        except Exception:
            pass
        return None
    # Prefer explicit template if provided
    url = MCP_BRIDGE_INVOKE_URL.strip()
    if url:
        url = url.replace("{server}", MCP_SERVER_NAME).replace("{tool}", tool)
    else:
        # Best-guess default shape used by recent server builds
        url = f"{LGS_BASE_URL.rstrip('/')}/mcp/servers/{MCP_SERVER_NAME}/tools/{tool}/invoke"
    payload = {"arguments": args}
    try:
        # Light header visibility without secrets
        hdrs = _headers()
        auth_present = any(k.lower() == "authorization" for k in hdrs.keys())
        log.info(
            "[mcp-bridge] invoke start tool=%s url=%s timeout=%ss auth=%s",
            tool,
            url,
            timeout,
            "yes" if auth_present else "no",
        )
        # Use separate connect/read timeouts to avoid hanging when server is wrong/absent
        connect_to = MCP_BRIDGE_CONNECT_TIMEOUT_S or min(3.05, max(1.0, timeout / 4.0))
        read_to = max(timeout, MCP_BRIDGE_READ_TIMEOUT_S or timeout)
        r = requests.post(
            url,
            headers=hdrs,
            json=payload,
            timeout=(connect_to, read_to),
        )
        if r.status_code >= 400:
            reason = getattr(r, "reason", "") or ""
            body_snip = (r.text or "")[:200]
            log.info(
                "[mcp-bridge] invoke fail tool=%s status=%s reason=%s body~%r",
                tool,
                r.status_code,
                reason,
                body_snip,
            )
            return None
        try:
            out = r.json()
            try:
                size_hint = 0
                if isinstance(out, dict):
                    size_hint = len(out.keys())
                elif isinstance(out, list):
                    size_hint = len(out)
                log.info("[mcp-bridge] invoke ok tool=%s status=%s size=%s", tool, r.status_code, size_hint)
            except Exception:
                pass
            return out
        except Exception:
            log.info("[mcp-bridge] invoke parse-failed tool=%s", tool)
            return None
    except Exception as e:
        # Mark bridge down briefly so subsequent calls fall back without waiting
        try:
            _BRIDGE_DOWN_UNTIL = time.time() + _BRIDGE_COOL_OFF_S
            log.info(
                "[mcp-bridge] invoke error tool=%s err=%s (cooldown %ss)",
                tool,
                e,
                int(_BRIDGE_COOL_OFF_S),
            )
        except Exception:
            log.info("[mcp-bridge] invoke error tool=%s err=%s", tool, e)
        return None


def read_url(url: str, *, timeout: float = 12.0) -> Optional[str]:
    res = _invoke("read_url", {"url": url, "timeout": timeout}, timeout=timeout)
    if isinstance(res, dict):
        txt = (res.get("content") or "").strip()
        if txt:
            log.info("[mcp-bridge] read_url ok bytes=%s", len(txt))
            return txt
        log.info("[mcp-bridge] read_url empty")
        return None
    return None


def search_web(query: str, *, country: Optional[str] = None, max_results: int = 20) -> List[str]:
    res = _invoke("search_web", {"query": query, "limit": max_results, "country": country or None}, timeout=8.0)
    out: List[str] = []
    if isinstance(res, list):
        for it in res:
            if isinstance(it, str):
                out.append(it)
            elif isinstance(it, dict) and it.get("url"):
                out.append(str(it["url"]))
        out = out[:max_results]
        log.info("[mcp-bridge] search_web ok count=%s", len(out))
        return out
    if isinstance(res, dict):
        items = res.get("results")
        if isinstance(items, list):
            for it in items:
                if isinstance(it, str):
                    out.append(it)
                elif isinstance(it, dict) and it.get("url"):
                    out.append(str(it["url"]))
    out = out[:max_results]
    if out:
        log.info("[mcp-bridge] search_web ok count=%s", len(out))
    else:
        log.info("[mcp-bridge] search_web empty")
    return out


def parallel_search_web(queries: List[str], *, per_query: int = 10) -> Dict[str, List[str]]:
    res = _invoke("parallel_search_web", {"queries": queries, "per_query": per_query}, timeout=12.0)
    out: Dict[str, List[str]] = {}
    if isinstance(res, dict):
        for k, v in res.items():
            if isinstance(v, list):
                out[str(k)] = [str(x.get("url") if isinstance(x, dict) else x) for x in v]
    log.info("[mcp-bridge] parallel_search_web ok queries=%s non_empty=%s", len(queries or []), sum(1 for v in out.values() if v))
    return out
