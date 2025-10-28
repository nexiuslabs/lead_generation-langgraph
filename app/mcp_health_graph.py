from __future__ import annotations

from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END


class HealthState(TypedDict, total=False):
    url: str
    ok: bool
    disabled: bool
    error: Optional[str]


def _check_node(state: HealthState) -> HealthState:
    try:
        from src.settings import ENABLE_MCP_READER
        if not ENABLE_MCP_READER:
            return {"ok": False, "disabled": True, "url": state.get("url") or "https://example.com"}
        from src.services.mcp_reader import read_url as _mcp_read
        url = state.get("url") or "https://example.com"
        txt = _mcp_read(url, timeout=1.0)
        if txt:
            return {"ok": True, "disabled": False, "url": url}
        return {"ok": False, "disabled": False, "url": url, "error": "empty"}
    except Exception as e:
        return {"ok": False, "disabled": False, "url": state.get("url") or "https://example.com", "error": type(e).__name__}


def make_graph():
    g = StateGraph(HealthState)
    g.add_node("check", _check_node)
    g.set_entry_point("check")
    g.add_edge("check", END)
    return g.compile()

