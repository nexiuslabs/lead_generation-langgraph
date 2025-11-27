import os
import re
import time
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import requests

from src.settings import (
    JINA_DEEP_RESEARCH_TIMEOUT_S,
)
from src.obs import bump_vendor as _obs_bump, log_event as _obs_log

log = logging.getLogger("jina_deep_research")


def _auth_header() -> Dict[str, str]:
    key = os.getenv("JINA_API_KEY") or os.getenv("JINA_DEEPSEARCH_API_KEY") or ""
    hdrs = {"Content-Type": "application/json"}
    if key:
        hdrs["Authorization"] = f"Bearer {key}"
    return hdrs


def _norm_host(u: str) -> Optional[str]:
    try:
        s = (u or "").strip()
        if not s:
            return None
        s = s.replace("http://", "").replace("https://", "")
        for sep in ("/", "?", "#"):
            if sep in s:
                s = s.split(sep, 1)[0]
        if s.startswith("www."):
            s = s[4:]
        if "." not in s:
            return None
        return s.lower()
    except Exception:
        return None


def _extract_urls_from_text(text: str) -> List[str]:
    try:
        urls = re.findall(r"https?://[^\s)]+", text or "")
        return urls
    except Exception:
        return []


def deep_research_query(seed: str, icp_context: Dict[str, Any], *, timeout_s: float | None = None) -> Dict[str, Any]:
    """Run Jina Deep Research to discover candidate domains for a seed + ICP context.

    Reference implementation aligns with docs/Jina_deepresearch.md sample.
    Returns pack: {domains: [...], snippets_by_domain: {...}, fast_facts: {...}, source: 'jina_deep_research'}
    """
    endpoint = "https://deepsearch.jina.ai/v1/chat/completions"
    hdrs = _auth_header()
    if not hdrs.get("Authorization"):
        log.info("JINA_API_KEY missing; deep_research_query skipping")
        return {"domains": [], "snippets_by_domain": {}, "fast_facts": {}, "source": "jina_deep_research"}
    # Build a concise user prompt with seed + ICP context
    industries = ", ".join([str(x) for x in (icp_context.get("industries") or [])][:5])
    buyers = ", ".join([str(x) for x in (icp_context.get("buyer_titles") or [])][:5])
    geo = ", ".join([str(x) for x in (icp_context.get("geo") or [])][:3])
    prompt = (
        f"Find up to 50 company websites similar to '{seed}'.\n"
        f"Industries: {industries or 'any'}. Buyer titles: {buyers or 'any'}. Geo: {geo or 'any'}.\n"
        f"Return diverse, commercial domains (no portals/aggregators)."
    )
    # Prefer non-stream for simpler parsing; respect sample fields
    payload: Dict[str, Any] = {
        "model": "jina-deepsearch-v1",
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "reasoning_effort": "low",
        "max_returned_urls": "50",
        "bad_hostnames": ["*.gov.sg", "*.gov", "*.xyz"],
    }
    t0 = time.perf_counter()
    try:
        r = requests.post(
            endpoint,
            headers=hdrs,
            data=json.dumps(payload),
            timeout=float(timeout_s or JINA_DEEP_RESEARCH_TIMEOUT_S),
        )
        r.raise_for_status()
        data = r.json()
        # Try to harvest domains from visitedURLs/readURLs first
        urls = []
        for key in ("visitedURLs", "readURLs"):
            try:
                arr = data.get(key) or []
                if isinstance(arr, list):
                    urls.extend([str(x) for x in arr if isinstance(x, str)])
            except Exception:
                pass
        # Fallback to choices[].message.content when present
        try:
            content = None
            ch = (data.get("choices") or [{}])[0]
            msg = ch.get("message") or {}
            content = msg.get("content")
            if isinstance(content, str):
                urls.extend(_extract_urls_from_text(content))
        except Exception:
            pass
        # Normalize to apex domains and dedupe
        hosts: List[str] = []
        seen = set()
        for u in urls:
            h = _norm_host(u)
            if not h or h in seen:
                continue
            seen.add(h)
            hosts.append(h)
        _obs_bump(run_id=0, tenant_id=0, vendor="jina_deep_research", calls=1)  # run/tenant resolved by caller typically
        return {
            "domains": hosts[:50],
            "snippets_by_domain": {},
            "fast_facts": {},
            "source": "jina_deep_research",
        }
    except Exception as exc:
        log.warning("deep_research_query failed: %s", exc)
        _obs_bump(run_id=0, tenant_id=0, vendor="jina_deep_research", calls=1, errors=1)
        return {"domains": [], "snippets_by_domain": {}, "fast_facts": {}, "source": "jina_deep_research"}


def deep_research_for_domain(domain: str, *, timeout_s: float | None = None) -> Dict[str, Any]:
    """Obtain a domain summary via Jina Deep Research (HTTP protocol).

    Returns pack: {domain, summary, pages, source}
    """
    endpoint = "https://deepsearch.jina.ai/v1/chat/completions"
    hdrs = _auth_header()
    if not hdrs.get("Authorization"):
        return {"domain": domain, "summary": "", "pages": [], "source": "jina_deep_research"}
    prompt = f"Summarize the main offering and ICP signals for https://{domain}. Return concise text."
    payload: Dict[str, Any] = {
        "model": "jina-deepsearch-v1",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "reasoning_effort": "low",
        "max_returned_urls": "10",
        "bad_hostnames": ["*.gov.sg", "*.gov", "*.xyz"],
    }
    try:
        r = requests.post(
            endpoint,
            headers=hdrs,
            data=json.dumps(payload),
            timeout=float(timeout_s or JINA_DEEP_RESEARCH_TIMEOUT_S),
        )
        r.raise_for_status()
        data = r.json()
        content = None
        try:
            ch = (data.get("choices") or [{}])[0]
            msg = ch.get("message") or {}
            content = msg.get("content")
        except Exception:
            content = None
        pages = []
        for key in ("visitedURLs", "readURLs"):
            try:
                arr = data.get(key) or []
                if isinstance(arr, list):
                    for u in arr[:5]:
                        pages.append({"url": u, "summary": None})
            except Exception:
                pass
        _obs_bump(run_id=0, tenant_id=0, vendor="jina_deep_research", calls=1)
        return {"domain": domain, "summary": (content or "")[:4000], "pages": pages, "source": "jina_deep_research"}
    except Exception:
        _obs_bump(run_id=0, tenant_id=0, vendor="jina_deep_research", calls=1, errors=1)
        return {"domain": domain, "summary": "", "pages": [], "source": "jina_deep_research"}


__all__ = [
    "deep_research_query",
    "deep_research_for_domain",
]

