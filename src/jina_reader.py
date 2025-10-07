import logging
from typing import Optional
from urllib.parse import urlparse

import requests

log = logging.getLogger("jina_reader")
if not log.handlers:
    h = logging.StreamHandler()
    log.addHandler(h)
log.setLevel(logging.INFO)


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


def read_url(url: str, timeout: float = 12.0) -> Optional[str]:
    try:
        reader = _reader_url_for(url)
        log.info("[jina] GET %s", reader)
        r = requests.get(reader, timeout=timeout)
        if r.status_code >= 400:
            log.info("[jina] status=%s for %s", r.status_code, url)
            return None
        txt = (r.text or "")[:10000]
        return clean_jina_text(txt)
    except Exception as e:
        log.info("[jina] fetch failed url=%s err=%s", url, e)
        return None

