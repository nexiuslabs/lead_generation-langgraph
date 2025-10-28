from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

import requests

from src.settings import MCP_REMOTE_URL, MCP_API_KEY, MCP_TIMEOUT_S

log = logging.getLogger("mcp_remote")


def _headers() -> Dict[str, str]:
    h = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if MCP_API_KEY:
        h["Authorization"] = f"Bearer {MCP_API_KEY}"
    return h


def _rpc(method: str, params: Dict[str, Any], *, timeout: float) -> Optional[Dict[str, Any]]:
    payload = {"jsonrpc": "2.0", "id": "1", "method": method, "params": params}
    try:
        r = requests.post(MCP_REMOTE_URL, headers=_headers(), json=payload, timeout=timeout)
        if r.status_code >= 400:
            log.info("[mcp-remote] http status=%s", r.status_code)
            return None
        return r.json()
    except Exception as e:
        log.info("[mcp-remote] rpc error method=%s err=%s", method, e)
        return None


def _extract_text_content(result: Dict[str, Any]) -> str:
    try:
        content = (result or {}).get("result", {}).get("content")
        if isinstance(content, list):
            texts: List[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
                    texts.append(str(item["text"]))
            return "\n".join(texts)
        if isinstance(content, str):
            return content
    except Exception:
        pass
    return ""


def read_url(url: str, *, timeout: float = MCP_TIMEOUT_S) -> Optional[str]:
    res = _rpc("tools/call", {"name": "read_url", "arguments": {"url": url, "timeout": timeout}}, timeout=timeout)
    txt = _extract_text_content(res or {})
    if txt:
        log.info("[mcp-remote] read_url ok bytes=%s", len(txt))
        return txt
    log.info("[mcp-remote] read_url empty")
    return None


_URL_RE = re.compile(r"https?://\S+", re.I)


def search_web(query: str, *, country: Optional[str] = None, max_results: int = 20, timeout: float = MCP_TIMEOUT_S) -> List[str]:
    args: Dict[str, Any] = {"query": query, "limit": max_results}
    if country:
        args["country"] = country
    res = _rpc("tools/call", {"name": "search_web", "arguments": args}, timeout=timeout)
    # Try to parse content items for URLs; fallback to regex
    out: List[str] = []
    try:
        content = (res or {}).get("result", {}).get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
                    out.extend(_URL_RE.findall(str(item["text"]))[:max_results])
    except Exception:
        pass
    return out[:max_results]


def parallel_search_web(queries: List[str], *, per_query: int = 10, timeout: float = MCP_TIMEOUT_S) -> Dict[str, List[str]]:
    args: Dict[str, Any] = {"queries": queries, "per_query": per_query}
    res = _rpc("tools/call", {"name": "parallel_search_web", "arguments": args}, timeout=timeout)
    out: Dict[str, List[str]] = {}
    try:
        content = (res or {}).get("result", {}).get("content")
        if isinstance(content, list):
            # Treat each text block as k: v list pair lines
            texts = [str(it.get("text")) for it in content if isinstance(it, dict) and it.get("type") == "text" and it.get("text")]
            joined = "\n".join(texts)
            # Very light heuristic; users rely mainly on read_url path
            for q in queries:
                urls = _URL_RE.findall(joined)
                if urls:
                    out[str(q)] = urls[:per_query]
    except Exception:
        pass
    return out

