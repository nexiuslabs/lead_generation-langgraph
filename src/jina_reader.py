import logging
import time
import random
from typing import Optional
from urllib.parse import urlparse

import requests

log = logging.getLogger("jina_reader")
if not log.handlers:
    h = logging.StreamHandler()
    log.addHandler(h)
log.setLevel(logging.INFO)


_FAIL_CACHE: dict[str, float] = {}

def _maybe_use_mcp() -> bool:
    try:
        from src import settings  # type: ignore
        return bool(getattr(settings, "ENABLE_MCP_READER", False))
    except Exception:
        return False

def _mcp_dual_read_pct() -> int:
    try:
        from src import settings  # type: ignore
        v = int(getattr(settings, "MCP_DUAL_READ_PCT", 0) or 0)
        return max(0, min(100, v))
    except Exception:
        return 0

def _reader_url_for(raw_url: str) -> str:
    url = (raw_url or "").strip()
    if not url:
        return "https://r.jina.ai/http://"
    parsed = urlparse(url if url.startswith("http") else ("https://" + url))
    scheme = parsed.scheme.lower() if parsed.scheme else "https"
    # r.jina mirrors the original scheme inside the path: /http://... or /https://...
    inner = f"{scheme}://{parsed.netloc}{parsed.path or ''}"
    if parsed.query:
        inner += f"?{parsed.query}"
    return f"https://r.jina.ai/{inner}"


def clean_jina_text(text: str) -> str:
    try:
        raw = text or ""
        # Drop common metadata headers from r.jina snapshots
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        filtered = [
            ln
            for ln in lines
            if not ln.lower().startswith((
                "title:",
                "url source:",
                "published time:",
                "markdown content:",
                "warning:",
            ))
        ]
        clean = " ".join(filtered) if filtered else " ".join(raw.split())
        return clean
    except Exception:
        return text or ""


def _host_of(u: str) -> str:
    try:
        p = urlparse(u if u.startswith("http") else ("https://" + u))
        return (p.netloc or "").lower()
    except Exception:
        return ""


def _read_url_http(url: str, timeout: float = 12.0) -> Optional[str]:
    try:
        # Skip known-bad hosts for a cooling period to avoid tight retry loops
        ttl = 60.0 * 60.0 * 6  # 6 hours default
        try:
            import os
            ttl = float(os.getenv("JINA_FAIL_TTL_S", str(ttl)))
        except Exception:
            pass
        host = _host_of(url)
        now = time.time()
        if host and host in _FAIL_CACHE and (now - _FAIL_CACHE[host]) < ttl:
            return None

        # Build conservative variants to try (scheme flip + www)
        variants: list[str] = []
        base = url if url.startswith("http") else ("https://" + url)
        try:
            p = urlparse(base)
            schemes = [p.scheme.lower()] if p.scheme else ["https"]
            if "https" in schemes:
                schemes.append("http")
            hosts = [p.netloc]
            if p.netloc and not p.netloc.lower().startswith("www."):
                hosts.append("www." + p.netloc)
            for sch in schemes:
                for h in hosts:
                    composed = f"{sch}://{h}{p.path or ''}"
                    if p.query:
                        composed += f"?{p.query}"
                    if composed not in variants:
                        variants.append(composed)
        except Exception:
            variants = [base]

        for v in variants:
            reader = _reader_url_for(v)
            log.info("[jina] GET %s", reader)
            r = requests.get(reader, timeout=timeout)
            if r.status_code >= 400:
                log.info("[jina] status=%s for %s", r.status_code, v)
                continue
            txt = (r.text or "")[:10000]
            return clean_jina_text(txt)
        # All variants failed; mark host as bad for a while
        if host:
            _FAIL_CACHE[host] = now
        return None
    except Exception as e:
        log.info("[jina] fetch failed url=%s err=%s", url, e)
        h = _host_of(url)
        if h:
            _FAIL_CACHE[h] = time.time()
        return None


def read_url(url: str, timeout: float = 12.0) -> Optional[str]:
    """Feature-flagged wrapper that prefers MCP read_url when enabled.

    - If MCP is enabled and dual-read sampling triggers, run HTTP (return it) and MCP (log parity) sequentially.
    - If MCP is enabled (no sampling), attempt MCP; on any error, fall back to HTTP.
    - If MCP is disabled, use HTTP reader.
    """
    # If MCP disabled, use legacy HTTP path
    if not _maybe_use_mcp():
        return _read_url_http(url, timeout=timeout)

    # Dual-read sampling: always return HTTP to avoid user-visible changes during validation
    pct = _mcp_dual_read_pct()
    do_dual = pct > 0 and (random.random() * 100.0) < float(pct)
    if do_dual:
        http_txt = _read_url_http(url, timeout=timeout)
        # Fire MCP best-effort (non-blocking for errors); log parity if available
        try:
            t0 = time.perf_counter()
            try:
                from src.services import mcp_reader as _mcp  # type: ignore
                mcp_txt = _mcp.read_url(url, timeout_s=timeout)
            except Exception as e:
                mcp_txt = None
                _log_mcp_dual_event(url, t0, error=type(e).__name__)
            else:
                _log_mcp_dual_event(url, t0, http_txt=http_txt, mcp_txt=mcp_txt)
        except Exception:
            pass
        return http_txt

    # MCP-only path with fallback
    try:
        t0 = time.perf_counter()
        from src.services import mcp_reader as _mcp  # type: ignore
        log.info("[mcp] enabled; using MCP reader for %s", url)
        txt = _mcp.read_url(url, timeout_s=timeout)
        return clean_jina_text(txt or "") if txt else None
    except Exception as e:
        # Fall back to HTTP and mark host as temporarily failed for MCP
        try:
            log.info("[mcp] fallback to HTTP for %s due to: %s", url, (str(e) or type(e).__name__))
        except Exception:
            pass
        return _read_url_http(url, timeout=timeout)


def _log_mcp_dual_event(url: str, t0: float, *, http_txt: Optional[str] = None, mcp_txt: Optional[str] = None, error: Optional[str] = None) -> None:
    try:
        from src import obs  # type: ignore
        run_id, tenant = obs.get_run_context()
    except Exception:
        run_id, tenant = None, None
    try:
        dur = int((time.perf_counter() - t0) * 1000)
    except Exception:
        dur = None  # type: ignore
    extra = {
        "url": url,
        "http_len": len(http_txt) if isinstance(http_txt, str) else None,
        "mcp_len": len(mcp_txt) if isinstance(mcp_txt, str) else None,
        "error": error,
    }
    try:
        if run_id is not None and tenant is not None:
            obs.log_event(
                run_id,
                tenant,
                stage="mcp_dual_read",
                event="finish" if not error else "error",
                status="ok" if not error else "error",
                duration_ms=dur,
                extra=extra,
            )
    except Exception:
        pass
