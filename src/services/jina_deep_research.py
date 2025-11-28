import os
import re
import time
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import requests

from src.settings import (
    JINA_DEEP_RESEARCH_TIMEOUT_S,
    JINA_DEEP_RESEARCH_API_URL,
    JINA_DEEP_RESEARCH_MODEL,
    JINA_DEEP_RESEARCH_REASONING_EFFORT,
    JINA_DEEP_RESEARCH_BAD_HOSTNAMES,
    JINA_DEEP_RESEARCH_DISCOVERY_MAX_URLS,
    JINA_DEEP_RESEARCH_SUMMARY_MAX_URLS,
)
from src.obs import bump_vendor as _obs_bump, log_event as _obs_log

log = logging.getLogger("jina_deep_research")
if log.level == logging.NOTSET:
    log.setLevel(logging.INFO)


def _mask_key(key: str) -> str:
    try:
        if not key:
            return "missing"
        parts = key.strip().split("-")
        if len(parts) <= 3:
            return key[-6:]
        tail = "-".join(parts[-3:])
        return tail
    except Exception:
        return "invalid"


def _auth_header() -> Dict[str, str]:
    key = os.getenv("JINA_API_KEY") or ""
    hdrs = {"Content-Type": "application/json"}
    if key:
        hdrs["Authorization"] = f"Bearer {key}"
    try:
        masked = _mask_key(key)
        log.warning("[jina_dr] auth_header using key_tail=%s", masked)
    except Exception:
        pass
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


def _icp_summary_sentence(industries: str, geo: str) -> str:
    try:
        if industries and geo:
            return f"{industries} companies operating in {geo}"
        if industries:
            return f"{industries} companies"
        if geo:
            return f"companies operating in {geo}"
    except Exception:
        pass
    return "relevant B2B companies"


def deep_research_query(seed: str, icp_context: Dict[str, Any], *, timeout_s: float | None = None) -> Dict[str, Any]:
    """Run Jina Deep Research to discover candidate domains for a seed + ICP context.

    Reference implementation aligns with docs/Jina_deepresearch.md sample.
    Returns pack: {domains: [...], snippets_by_domain: {...}, fast_facts: {...}, source: 'jina_deep_research'}
    """
    endpoint = JINA_DEEP_RESEARCH_API_URL
    hdrs = _auth_header()
    if not hdrs.get("Authorization"):
        log.info("JINA_API_KEY missing; deep_research_query skipping")
        return {"domains": [], "snippets_by_domain": {}, "fast_facts": {}, "source": "jina_deep_research"}
    # Build a concise user prompt with seed + ICP context
    industries = ", ".join([str(x) for x in (icp_context.get("industries") or [])][:5])
    buyers = ", ".join([str(x) for x in (icp_context.get("buyer_titles") or [])][:5])
    geo = ", ".join([str(x) for x in (icp_context.get("geo") or [])][:3])
    icp_sentence = _icp_summary_sentence(industries, geo)
    prompt = (
        f"Find up to {JINA_DEEP_RESEARCH_DISCOVERY_MAX_URLS} company domains matching this ICP: {icp_sentence}.\n"
        f"Seed: '{seed}'.\n"
        "Your PRIMARY objective is to identify and return company names and their corresponding official domain URLs."
        " This is the ONLY acceptable output type. If the Seed does not directly match companies within the specified ICPs and regions,"
        " ignore the seed and instead identify up to the requested number of diverse, commercial companies that do match the ICPs and regions."
        " Under no circumstances should the absence of seed-related results lead to a numerical or summary output.\n"
        "Your response MUST be a valid JSON array of objects. Each object MUST contain two keys: 'company_name' (string) and 'domain_url' (string)."
        " The 'domain_url' MUST be the official website of the company. Ensure there are no portals or aggregators."
        " DO NOT provide any numerical answers, statistics, summaries, or other data aggregation."
        " DO NOT include any text or commentary outside of the JSON array."
        " Example: [{\"company_name\": \"Company A\", \"domain_url\": \"companya.com\"}]."
    )
    # Prefer non-stream for simpler parsing; respect sample fields
    payload: Dict[str, Any] = {
        "model": JINA_DEEP_RESEARCH_MODEL,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "reasoning_effort": JINA_DEEP_RESEARCH_REASONING_EFFORT,
        "max_returned_urls": str(JINA_DEEP_RESEARCH_DISCOVERY_MAX_URLS),
        "bad_hostnames": list(JINA_DEEP_RESEARCH_BAD_HOSTNAMES or []),
    }
    try:
        log.warning(
            "[jina_dr] discovery_call seed=%s industries=%s buyers=%s geo=%s",
            seed,
            industries,
            buyers,
            geo,
        )
        log.warning("[jina_dr] discovery_prompt=<<%s>>", prompt)
    except Exception:
        pass
    t0 = time.perf_counter()
    try:
        r = requests.post(
            endpoint,
            headers=hdrs,
            data=json.dumps(payload),
            timeout=float(timeout_s or JINA_DEEP_RESEARCH_TIMEOUT_S),
        )
        log.warning("[jina_dr] response_status=%s", r.status_code)
        r.raise_for_status()
        data = r.json()
        try:
            import json as _json

            preview = data.get("visitedURLs") or data.get("readURLs") or []
            log.warning("[jina_dr] response_preview=%s", preview[:5])
            log.warning(
                "[jina_dr] response_body=%s",
                (_json.dumps(data)[:2000] if isinstance(data, dict) else str(data)[:2000]),
            )
        except Exception:
            pass
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
    endpoint = JINA_DEEP_RESEARCH_API_URL
    hdrs = _auth_header()
    if not hdrs.get("Authorization"):
        return {"domain": domain, "summary": "", "pages": [], "source": "jina_deep_research"}
    prompt = f"Summarize the main offering and ICP signals for https://{domain}. Return concise text."
    payload: Dict[str, Any] = {
        "model": JINA_DEEP_RESEARCH_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "reasoning_effort": JINA_DEEP_RESEARCH_REASONING_EFFORT,
        "max_returned_urls": str(JINA_DEEP_RESEARCH_SUMMARY_MAX_URLS),
        "bad_hostnames": list(JINA_DEEP_RESEARCH_BAD_HOSTNAMES or []),
    }
    try:
        log.warning("[jina_dr] domain_call domain=%s", domain)
    except Exception:
        pass
    try:
        r = requests.post(
            endpoint,
            headers=hdrs,
            data=json.dumps(payload),
            timeout=float(timeout_s or JINA_DEEP_RESEARCH_TIMEOUT_S),
        )
        log.warning("[jina_dr] domain_response_status=%s", r.status_code)
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
