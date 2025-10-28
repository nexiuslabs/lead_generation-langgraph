from __future__ import annotations

import logging
import time
import random
from typing import Optional, List, Dict, Any

from src.obs import bump_vendor, log_event, get_run_context
from src.settings import (
    ENABLE_MCP_READER,
    ENABLE_SERVER_MCP_BRIDGE,
    MCP_ENDPOINT,
    MCP_API_KEY,
    MCP_TIMEOUT_S,
    MCP_MAX_PARALLEL,
    CB_ERROR_THRESHOLD,
    CB_COOL_OFF_S,
)
from src.retry import BackoffPolicy, CircuitBreaker, RetryableError


class NoSessionError(RetryableError):
    pass

log = logging.getLogger("mcp_reader")

_SESSION = None  # lazily initialized MCP client/session


def _ensure_session():
    global _SESSION
    if _SESSION is not None:
        return _SESSION
    try:
        # Prefer official client if installed; import is optional
        from jina_mcp import Client  # type: ignore
        try:
            redacted_key = "yes" if (MCP_API_KEY or "").strip() else "no"
            log.info(
                "[mcp] connecting base_url=%s timeout=%ss api_key_present=%s",
                MCP_ENDPOINT,
                MCP_TIMEOUT_S,
                redacted_key,
            )
        except Exception:
            pass
        _SESSION = Client(base_url=MCP_ENDPOINT, api_key=MCP_API_KEY, timeout=MCP_TIMEOUT_S)
        try:
            log.info("[mcp] connected ok base_url=%s", MCP_ENDPOINT)
        except Exception:
            pass
        return _SESSION
    except Exception as e:  # pragma: no cover
        log.warning("[mcp] jina_mcp client not installed: %s", e)
        # Fallback: if server bridge is enabled, synthesize a session backed by it
        if ENABLE_SERVER_MCP_BRIDGE:
            try:
                from src.services import mcp_server_bridge as bridge

                class _BridgeTools:
                    def invoke(self, tool: str, payload: dict):
                        if tool == "read_url":
                            return {"content": bridge.read_url(payload.get("url", ""), timeout=float(payload.get("timeout", MCP_TIMEOUT_S))) or ""}
                        if tool == "search_web":
                            return {
                                "results": bridge.search_web(
                                    payload.get("query", ""),
                                    country=payload.get("country"),
                                    max_results=int(payload.get("limit", 20) or 20),
                                )
                            }
                        if tool == "parallel_search_web":
                            return bridge.parallel_search_web(
                                payload.get("queries") or [], per_query=int(payload.get("per_query", 10) or 10)
                            )
                        raise NoSessionError(f"unknown-tool:{tool}")

                class _BridgeSession:
                    def __init__(self):
                        self.tools = _BridgeTools()

                _SESSION = _BridgeSession()
                log.info("[mcp] using server bridge as client fallback")
                return _SESSION
            except Exception:
                pass
        _SESSION = None
        return None


def _telemetry(tool: str, *, ok: bool, start: float, extra: Dict[str, Any] | None = None) -> None:
    dur_ms = int((time.perf_counter() - start) * 1000)
    # Optional Prometheus metrics
    try:  # pragma: no cover
        from prometheus_client import Counter, Histogram  # type: ignore

        global _MCP_REQS, _MCP_DUR
        try:
            _MCP_REQS
        except NameError:
            _MCP_REQS = Counter("mcp_requests_total", "MCP tool requests", ["tool", "status"])  # type: ignore
        try:
            _MCP_DUR
        except NameError:
            _MCP_DUR = Histogram("mcp_request_duration_seconds", "MCP tool request duration (seconds)", ["tool"])  # type: ignore
        _MCP_REQS.labels(tool, ("ok" if ok else "error")).inc()  # type: ignore
        _MCP_DUR.labels(tool).observe(max(0.0, dur_ms / 1000.0))  # type: ignore
    except Exception:
        pass

    run_id, tenant_id = get_run_context()
    if run_id and tenant_id:
        try:
            log_event(
                run_id,
                tenant_id,
                stage="mcp",
                event=tool,
                status=("ok" if ok else "error"),
                duration_ms=dur_ms,
                extra=extra,
            )
            bump_vendor(run_id, tenant_id, vendor="mcp", calls=1, errors=(0 if ok else 1))
        except Exception:
            pass


_BREAKERS = {
    "read_url": CircuitBreaker(CB_ERROR_THRESHOLD, CB_COOL_OFF_S),
    "search_web": CircuitBreaker(CB_ERROR_THRESHOLD, CB_COOL_OFF_S),
    "parallel_search_web": CircuitBreaker(CB_ERROR_THRESHOLD, CB_COOL_OFF_S),
}


def _with_retry_sync(fn, policy: BackoffPolicy) -> Any:  # type: ignore[name-defined]
    attempt = 0
    while attempt < policy.max_attempts:
        try:
            return fn()
        except Exception as e:
            # Do not retry when the MCP session is not available
            if isinstance(e, NoSessionError):
                raise e
            attempt += 1
            if attempt >= policy.max_attempts:
                raise e
            delay = min(policy.max_delay_ms, policy.base_delay_ms * (2 ** (attempt - 1)))
            jitter = 0.8 + 0.4 * random.random()
            time.sleep((delay * jitter) / 1000.0)


def _invoke_with_resilience(tool: str, payload: dict) -> dict | list | None:
    br = _BREAKERS.get(tool)
    run_id, tenant_id = get_run_context()
    tenant = int(tenant_id or 0)
    vendor = f"mcp:{tool}"
    if br and not br.allow(tenant, vendor):
        raise RetryableError(f"breaker-open:{tool}")

    def _call():
        sess = _ensure_session()
        if sess is None:
            raise NoSessionError("no-session")
        try:
            log.info("[mcp] invoke start tool=%s", tool)
            out = sess.tools.invoke(tool, payload)
            try:
                # keep details light to avoid sensitive data leakage
                if isinstance(out, dict):
                    size_hint = sum(len(str(v)[:50]) for v in out.values())
                elif isinstance(out, list):
                    size_hint = len(out)
                else:
                    size_hint = 0
                log.info("[mcp] invoke ok tool=%s size=%s", tool, size_hint)
            except Exception:
                pass
            return out
        except Exception as e:
            # reset session to force reconnect on next attempt
            try:
                global _SESSION
                _SESSION = None
            except Exception:
                pass
            try:
                log.info("[mcp] invoke failed tool=%s err=%s", tool, e)
            except Exception:
                pass
            raise

    res = _with_retry_sync(_call, BackoffPolicy(max_attempts=3, base_delay_ms=250, max_delay_ms=1500))
    if br:
        try:
            br.on_success(tenant, vendor)
        except Exception:
            pass
    return res


def read_url(url: str, timeout: Optional[float] = None) -> Optional[str]:
    if not ENABLE_MCP_READER:
        return None
    t0 = time.perf_counter()
    try:
        res = _invoke_with_resilience("read_url", {"url": url, "timeout": timeout or MCP_TIMEOUT_S})
        text = (res.get("content") or "").strip() if isinstance(res, dict) else ""
        _telemetry("read_url", ok=True, start=t0, extra={"bytes": len(text)})
        if text:
            try:
                log.info("[mcp] read_url ok bytes=%s", len(text))
            except Exception:
                pass
            return text
        try:
            log.info("[mcp] read_url empty")
        except Exception:
            pass
        # Try direct remote MCP JSON-RPC as last resort (bypass bridge cooldown)
        try:
            from src.services import mcp_remote as _remote
            rtxt = _remote.read_url(url, timeout=timeout or MCP_TIMEOUT_S)
            if rtxt:
                return rtxt
        except Exception:
            pass
        return None
    except Exception as e:
        br = _BREAKERS.get("read_url")
        if br:
            try:
                if not isinstance(e, NoSessionError):
                    run_id, tenant_id = get_run_context()
                    br.on_error(int(tenant_id or 0), "mcp:read_url")
            except Exception:
                pass
        log.info("[mcp] read_url error: %s", e)
        _telemetry("read_url", ok=False, start=t0, extra={"error": type(e).__name__})
        return None


def search_web(query: str, *, country: Optional[str] = None, max_results: int = 20) -> List[str]:
    if not ENABLE_MCP_READER:
        return []
    t0 = time.perf_counter()
    try:
        payload = {"query": query, "limit": max_results, "country": country}
        res = _invoke_with_resilience("search_web", payload)
        items = res.get("results") if isinstance(res, dict) else res
        out: List[str] = []
        for it in items or []:
            if isinstance(it, str):
                out.append(it)
            elif isinstance(it, dict) and it.get("url"):
                out.append(str(it["url"]))
        _telemetry("search_web", ok=True, start=t0, extra={"count": len(out)})
        out = out[:max_results]
        try:
            if out:
                log.info("[mcp] search_web ok count=%s", len(out))
            else:
                log.info("[mcp] search_web empty")
        except Exception:
            pass
        if not out:
            try:
                from src.services import mcp_remote as _remote
                out = _remote.search_web(query, country=country, max_results=max_results)
            except Exception:
                pass
        return out
    except Exception as e:
        br = _BREAKERS.get("search_web")
        if br:
            try:
                if not isinstance(e, NoSessionError):
                    run_id, tenant_id = get_run_context()
                    br.on_error(int(tenant_id or 0), "mcp:search_web")
            except Exception:
                pass
        log.info("[mcp] search_web error: %s", e)
        _telemetry("search_web", ok=False, start=t0, extra={"error": type(e).__name__})
        return []


def parallel_search_web(queries: List[str], *, per_query: int = 10) -> Dict[str, List[str]]:
    if not ENABLE_MCP_READER:
        return {}
    t0 = time.perf_counter()
    try:
        payload = {"queries": (queries or [])[:MCP_MAX_PARALLEL], "per_query": per_query}
        res = _invoke_with_resilience("parallel_search_web", payload)
        out: Dict[str, List[str]] = {}
        if isinstance(res, dict):
            for k, v in res.items():
                if isinstance(v, list):
                    out[str(k)] = [str(x.get("url") if isinstance(x, dict) else x) for x in v]
        _telemetry("parallel_search_web", ok=True, start=t0, extra={"queries": len(out)})
        try:
            non_empty = sum(1 for v in out.values() if v)
            log.info(
                "[mcp] parallel_search_web ok queries=%s non_empty=%s",
                len(queries or []),
                non_empty,
            )
        except Exception:
            pass
        if not out:
            try:
                from src.services import mcp_remote as _remote
                out = _remote.parallel_search_web(queries or [], per_query=per_query)
            except Exception:
                pass
        return out
    except Exception as e:
        br = _BREAKERS.get("parallel_search_web")
        if br:
            try:
                if not isinstance(e, NoSessionError):
                    run_id, tenant_id = get_run_context()
                    br.on_error(int(tenant_id or 0), "mcp:parallel_search_web")
            except Exception:
                pass
        log.info("[mcp] parallel_search_web error: %s", e)
        _telemetry("parallel_search_web", ok=False, start=t0, extra={"error": type(e).__name__})
        return {}
