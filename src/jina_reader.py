import logging
import time
from typing import Optional
from urllib.parse import urlparse

import requests

log = logging.getLogger("jina_reader")
if not log.handlers:
    h = logging.StreamHandler()
    log.addHandler(h)
log.setLevel(logging.INFO)


_FAIL_CACHE: dict[str, float] = {}

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


def read_url(url: str, timeout: float = 12.0) -> Optional[str]:
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
