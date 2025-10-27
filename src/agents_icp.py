"""
LangChain/LangGraph ICP Agents

Implements LLM-based agents described in featurePRD19.md §6 using LangChain and LangGraph.
These agents operate on a simple dict state and can be composed into a StateGraph,
or called from existing flows (pre_sdr_graph) as helpers.
"""
from __future__ import annotations

import asyncio
import re
from typing import Any, Dict, List, Optional
import json
import html as html_lib
from bs4 import BeautifulSoup

from langchain_openai import ChatOpenAI
from src.settings import ENABLE_DDG_DISCOVERY, DDG_TIMEOUT_S, DDG_KL, DDG_MAX_CALLS
try:
    from src.settings import ICP_SG_PROFILES  # type: ignore
except Exception:  # pragma: no cover
    ICP_SG_PROFILES = False  # type: ignore
try:
    from src.settings import STRICT_INDUSTRY_QUERY_ONLY  # type: ignore
except Exception:  # pragma: no cover
    STRICT_INDUSTRY_QUERY_ONLY = True  # type: ignore
try:
    from src.settings import ENABLE_STRICT_DOMAIN_HYGIENE, DISCOVERY_ALLOW_PORTALS  # type: ignore
except Exception:  # pragma: no cover
    ENABLE_STRICT_DOMAIN_HYGIENE = True  # type: ignore
    DISCOVERY_ALLOW_PORTALS = False  # type: ignore
import logging
import requests
import os
import time
from urllib.parse import quote, urlparse, urljoin, parse_qs, unquote
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
# Strict DuckDuckGo-only mode: do not aggregate from other engines
# Ensure trailing slash so requests builds '/html/?q=...'
DDG_HTML_ENDPOINT = "https://html.duckduckgo.com/html/"
DDG_HTML_GET = "https://duckduckgo.com/html/"
DDG_LITE_GET = "https://lite.duckduckgo.com/lite/"
DDG_STD_LITE = "https://duckduckgo.com/lite/"

# Optional: ddgs library fallback (if available) — kept but no longer primary path
try:  # pragma: no cover
    from ddgs import DDGS  # type: ignore
except Exception:  # pragma: no cover
    DDGS = None  # type: ignore

# Enforce a minimum timeout for DDG endpoints (connect, read)
try:
    _DDG_TIMEOUT_MIN = 8.0
    TIMEOUT_S = float(DDG_TIMEOUT_S)
    if TIMEOUT_S < _DDG_TIMEOUT_MIN:
        TIMEOUT_S = _DDG_TIMEOUT_MIN
except Exception:
    TIMEOUT_S = 8.0

# Optional seed hints passed in by the caller (confirm flow) to improve queries
SEED_HINTS: List[str] = []

def set_seed_hints(hints: List[str] | None):
    global SEED_HINTS
    try:
        SEED_HINTS = [h.strip().lower() for h in (hints or []) if isinstance(h, str) and h.strip()][:10]
    except Exception:
        SEED_HINTS = []

from src.icp_pipeline import collect_evidence_for_domain
from src.jina_reader import read_url as jina_read
try:
    from src.config_profiles import (
        load_profiles as _load_sg_cfg,
        is_singapore_page as _is_sg_page,
        is_valid_fqdn as _is_valid_fqdn,
        is_denied_host as _is_denied_host,
        score_profile as _score_profile,
        bucket as _bucket,
    )
except Exception:  # pragma: no cover
    _load_sg_cfg = None  # type: ignore
    _is_sg_page = None  # type: ignore
    _is_valid_fqdn = None  # type: ignore
    _is_denied_host = None  # type: ignore


# Core state keys (subset)
# tenant_id, seeds[], icp_profile, discovery_candidates[], research_artifacts[], evidence[], scores[], queue[], errors[]

def _is_probable_domain(host: str) -> bool:
    try:
        h = (host or "").strip().lower()
        if not h:
            return False
        # Must contain at least one dot and valid labels
        import re as _re
        if not _re.match(r"^[a-z0-9-]+(\.[a-z0-9-]+)+$", h):
            return False
        # Block common file extensions posing as TLDs
        tld = h.rsplit(".", 1)[-1]
        bad_tlds = {
            "webp", "jpg", "jpeg", "png", "gif", "svg", "bmp", "ico",
            "pdf", "txt", "json", "xml", "csv", "zip", "gz", "tar",
            "mp4", "mp3", "mov", "avi", "css", "js", "html", "htm",
            # add text/doctype/file-like pseudo-TLDs often seen in scraped text
            "dtd",
        }
        if tld in bad_tlds:
            return False
        # Label length checks
        if not (2 <= len(tld) <= 24):
            return False
        # Exclude common shorteners/infrastructure
        if h in {"wa.me", "bit.ly", "t.co", "goo.gl", "tinyurl.com"}:
            return False
        return True
    except Exception:
        return False


def _uniq(seq: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for s in seq:
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _normalize_host(h: str) -> str:
    try:
        h = (h or "").strip().lower()
        if not h:
            return ""
        # strip protocol and path
        if h.startswith("http://"):
            h = h[7:]
        elif h.startswith("https://"):
            h = h[8:]
        for sep in ["/", "?", "#"]:
            if sep in h:
                h = h.split(sep, 1)[0]
        if h.startswith("www."):
            h = h[4:]
        return h
    except Exception:
        return (h or "").lower()


_MULTI_TLDS = {
    "co.uk", "org.uk", "ac.uk", "gov.uk",
    "com.sg", "net.sg", "org.sg", "edu.sg",
    "com.au", "net.au", "org.au",
    "co.jp", "ne.jp", "or.jp",
    "co.in", "net.in", "org.in",
}


def _apex_domain(h: str) -> str:
    try:
        host = _normalize_host(h)
        if not host:
            return ""
        parts = host.split(".")
        if len(parts) <= 2:
            return host
        last2 = ".".join(parts[-2:])
        last3 = ".".join(parts[-3:])
        if last2 in _MULTI_TLDS and len(parts) >= 3:
            return ".".join(parts[-3:])
        if last3 in _MULTI_TLDS and len(parts) >= 4:
            return ".".join(parts[-4:])
        return last2
    except Exception:
        return _normalize_host(h)


def _clean_possible_percent_encoded(s: str) -> str:
    """One-pass cleanup for percent-encoded and stray '2f' artifacts.

    - Unquote once
    - If URL-shaped, take netloc
    - Strip leading 'www.' and repeated '2f' prefixes
    - Lowercase and trim
    """
    try:
        raw = (s or "").strip()
        if not raw:
            return ""
        try:
            from urllib.parse import unquote as _unq, urlparse as _up
            u = _unq(raw)
            # If still contains a schema, extract host
            if "://" in u:
                host = (_up(u).netloc or u)
            else:
                host = u
        except Exception:
            host = raw
        host = host.strip().lower()
        # Remove protocol if any leaked through
        if host.startswith("http://"):
            host = host[7:]
        elif host.startswith("https://"):
            host = host[8:]
        # Trim path/query/fragment indicators
        for sep in ["/", "?", "#"]:
            if sep in host:
                host = host.split(sep, 1)[0]
        if host.startswith("www."):
            host = host[4:]
        # Strip leading repeated '2f' artifacts (from %2F)
        while host.startswith("2f"):
            host = host[2:]
        return host
    except Exception:
        return (s or "").strip().lower()


def _ddg_search_domains(query: str, max_results: int = 25, country: str | None = None, lang: str | None = None) -> List[str]:
    """Perform domain discovery by fetching DuckDuckGo HTML directly and parsing anchors.

    - Paginates up to 10 pages using the `s` offset, stops once `max_results` reached.
    - Parses DDG redirect anchors (`/l/?uddg=...`) or direct external hrefs.
    - Enforces `site:` filter (e.g., site:.sg) when present in the query.
    - Falls back to r.jina DDG snapshot + regex extraction only if all direct endpoints fail.
    """
    if not ENABLE_DDG_DISCOVERY:
        return []
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": (lang or "en-US,en;q=0.9"),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Referer": "https://duckduckgo.com/",
        "Cache-Control": "no-cache",
    }
    # Region hint to improve results (e.g., 'sg-en' when .sg seeds present)
    kl = None
    try:
        if country:
            country = country.lower()
            if country in {"sg", "singapore"}:
                kl = "sg-en"
            elif country in {"us", "usa", "united states"}:
                kl = "us-en"
            elif country in {"uk", "gb", "united kingdom"}:
                kl = "uk-en"
    except Exception:
        kl = None
    def _extract_domains_from_html(raw_html: str) -> List[str]:
        text = raw_html or ""
        text = html_lib.unescape(text)
        found: List[str] = []
        try:
            # Parse only anchor tags for performance
            soup = BeautifulSoup(text, "html.parser")
            for a in soup.find_all("a"):
                href = (a.get("href") or "").strip()
                if not href:
                    continue
                try:
                    # Resolve relative to DDG
                    href_abs = urljoin("https://duckduckgo.com", href)
                    u = urlparse(href_abs)
                    host = (u.netloc or "").lower()
                    if not host:
                        continue
                    # Handle DDG redirect pattern
                    if (host.endswith("duckduckgo.com") or host.endswith("r.duckduckgo.com")) and u.path.startswith("/l/"):
                        q = parse_qs(u.query)
                        target = q.get("uddg", [None])[0]
                        if target:
                            target_url = unquote(str(target))
                            host = (urlparse(target_url).netloc or "").lower()
                            if not host:
                                continue
                            found.append(host)
                            continue
                    # External direct link
                    found.append(host)
                except Exception:
                    continue
        except Exception:
            # Regex fallback
            for m in re.findall(r"/l/\?[^\s\"']*uddg=([^&\"']+)", text):
                try:
                    target_url = unquote(m)
                    host = (urlparse(target_url).netloc or "").lower()
                    if host:
                        found.append(host)
                except Exception:
                    continue
            for href in re.findall(r'href=[\"\']([^\"\']+)[\"\']', text):
                try:
                    href_abs = urljoin("https://duckduckgo.com", href.strip())
                    u = urlparse(href_abs)
                    host = (u.netloc or "").lower()
                    if not host:
                        continue
                    found.append(host)
                except Exception:
                    continue
        # Normalize and filter noisy/search/CDN/wiki hosts
        out: List[str] = []
        for h in _uniq(found):
            if any(x in h for x in (
                "duckduckgo.", "google.", "bing.", "brave.", "yahoo.", "yandex.", "mojeek.",
                "cloudflare.", "wikipedia.", "wikimedia.", "github.", "stackexchange.",
            )):
                continue
            if _is_probable_domain(h):
                out.append(h)
            if len(out) >= max_results:
                break
        # If no domains extracted via hrefs/redirects, fall back to text-based domain scanning
        if not out:
            try:
                text_domains = _extract_domains_from_text(text)
                # Preserve order and cap
                tmp: List[str] = []
                for d in text_domains:
                    if any(x in d for x in (
                        "duckduckgo.", "cloudflare.", "wikipedia.", "wikimedia.",
                    )):
                        continue
                    if _is_probable_domain(d) and d not in tmp:
                        tmp.append(d)
                    if len(tmp) >= max_results:
                        break
                if tmp:
                    return tmp
            except Exception:
                pass
        return out

    # Use a requests session with retries disabled to avoid noisy repeated attempts
    session = requests.Session()
    try:
        from requests.adapters import HTTPAdapter
        adapter = HTTPAdapter(max_retries=0)
        for h in ("https://", "http://"):
            session.mount(h, adapter)
    except Exception:
        pass

    def _get(url: str, params: dict | None = None, data: dict | None = None, method: str = "GET"):
        # Retry on transient connection/read issues with small backoff
        try:
            tries = max(1, int(os.getenv("DDG_REQUEST_RETRIES", "3") or 3))
        except Exception:
            tries = 3
        last_err = None
        for attempt in range(1, tries + 1):
            try:
                if method == "POST":
                    return session.post(url, data=(data or {}), headers=headers, timeout=(TIMEOUT_S, TIMEOUT_S))
                return session.get(url, params=(params or {}), headers=headers, timeout=(TIMEOUT_S, TIMEOUT_S))
            except Exception as e:
                last_err = e
                if attempt < tries:
                    try:
                        time.sleep(0.25 * attempt)
                    except Exception:
                        pass
                    continue
                raise last_err

    # Helper: fetch DDG HTML snapshot via r.jina.ai (fallback only)
    def _ddg_snapshot_via_jina(q: str, s_offset: int = 0) -> Optional[str]:
        try:
            kl_q = (DDG_KL or kl)
            # Primary: html.duckduckgo.com via Jina proxy
            url1 = f"https://r.jina.ai/https://html.duckduckgo.com/html/?q={quote(q)}" + (f"&kl={quote(kl_q)}" if kl_q else "") + (f"&s={s_offset}" if s_offset else "")
            r = session.get(url1, timeout=(TIMEOUT_S, TIMEOUT_S))
            if r.status_code < 400 and (r.text or "").strip():
                log.info("[ddg] r.jina snapshot via html.duckduckgo.com ok for query=%s", q)
                return r.text
            # Fallback: duckduckgo.com/html via Jina proxy
            url2 = f"https://r.jina.ai/https://duckduckgo.com/html/?q={quote(q)}" + (f"&kl={quote(kl_q)}" if kl_q else "") + (f"&s={s_offset}" if s_offset else "")
            r = session.get(url2, timeout=(TIMEOUT_S, TIMEOUT_S))
            if r.status_code < 400 and (r.text or "").strip():
                log.info("[ddg] r.jina snapshot via duckduckgo.com/html ok for query=%s", q)
                return r.text
            # Last: duckduckgo.com/lite via Jina proxy
            url3 = f"https://r.jina.ai/https://duckduckgo.com/lite/?q={quote(q)}" + (f"&kl={quote(kl_q)}" if kl_q else "") + (f"&s={s_offset}" if s_offset else "")
            r = session.get(url3, timeout=(TIMEOUT_S, TIMEOUT_S))
            if r.status_code < 400 and (r.text or "").strip():
                log.info("[ddg] r.jina snapshot via duckduckgo.com/lite ok for query=%s", q)
                return r.text
        except Exception as e:
            log.info("[ddg] r.jina snapshot fail: %s", e)
        return None

    # Site filter handling: if query contains `site:<token>`, enforce it
    def _site_filter(host: str) -> bool:
        try:
            m = re.search(r"\bsite:([^\s]+)", query)
            if not m:
                return True
            token = m.group(1).strip().lower()
            h = (host or "").lower()
            if not token:
                return True
            # site:.sg → require TLD endswith .sg
            if token.startswith('.'):
                return h.endswith(token)
            # site:example.com → require endswith example.com
            return h.endswith(token)
        except Exception:
            return True

    # 1) Fetch up to 10 pages directly from DDG endpoints; parse anchors
    collected: List[str] = []
    MAX_PAGES = 10
    PAGE_SIZE_GUESS = 30
    params_base = {"q": query}
    if (DDG_KL or kl):
        params_base["kl"] = (DDG_KL or kl)
    endpoints = [
        DDG_HTML_GET,       # duckduckgo.com/html/
        DDG_STD_LITE,       # duckduckgo.com/lite/
        DDG_HTML_ENDPOINT,  # html.duckduckgo.com/html/
        DDG_LITE_GET,       # lite.duckduckgo.com/lite/
    ]
    for page in range(MAX_PAGES):
        if len(collected) >= max_results:
            break
        s_off = page * PAGE_SIZE_GUESS
        page_ok = False
        found_any_for_page = False
        for ep in endpoints:
            try:
                params = dict(params_base)
                if s_off:
                    params["s"] = str(s_off)
                r = _get(ep, params=params)
                r.raise_for_status()
                page_hosts = _extract_domains_from_html(r.text)
                # Enforce site: filter and apex normalization
                page_domains: List[str] = []
                for h in page_hosts:
                    d = _apex_domain(_clean_possible_percent_encoded(h))
                    if d and _is_probable_domain(d) and _site_filter(d):
                        page_domains.append(d)
                # Log and merge
                try:
                    log.info(
                        "[ddg] page %d domains: %s",
                        page + 1,
                        ", ".join([f"https://{d}" for d in page_domains[:min(25, len(page_domains))]]),
                    )
                except Exception:
                    pass
                if page_domains:
                    found_any_for_page = True
                    for d in page_domains:
                        if d not in collected:
                            collected.append(d)
                        if len(collected) >= max_results:
                            break
                    # Move to next page only after extracting some domains
                    break
            except Exception as e:
                # Try next endpoint for this page
                continue
        if not found_any_for_page:
            # As a last resort, try r.jina snapshot for this page and parse via regex/text
            snapshot = _ddg_snapshot_via_jina(query, s_offset=s_off)
            if snapshot:
                snap_hosts = _extract_domains_from_html(snapshot)
                page_domains: List[str] = []
                for h in snap_hosts:
                    d = _apex_domain(_clean_possible_percent_encoded(h))
                    if d and _is_probable_domain(d) and _site_filter(d):
                        page_domains.append(d)
                try:
                    log.info(
                        "[ddg] page %d domains: %s",
                        page + 1,
                        ", ".join([f"https://{d}" for d in page_domains[:min(25, len(page_domains))]]),
                    )
                except Exception:
                    pass
                for d in page_domains:
                    if d not in collected:
                        collected.append(d)
                    if len(collected) >= max_results:
                        break
            else:
                # If first page fails entirely, return what we have (likely empty)
                if page == 0:
                    return [d for d in _uniq(collected) if _is_probable_domain(str(d))][:max_results]
                break
    return [d for d in _uniq(collected) if _is_probable_domain(str(d))][:max_results]


log = logging.getLogger("agents.icp")
if not log.handlers:
    h = logging.StreamHandler()
    fmt = logging.Formatter("[%(levelname)s] %(asctime)s %(name)s :: %(message)s", "%H:%M:%S")
    h.setFormatter(fmt)
    log.addHandler(h)
log.setLevel(logging.INFO)


class MicroICP(BaseModel):
    industries: Optional[List[str]] = None
    integrations: Optional[List[str]] = None
    buyer_titles: Optional[List[str]] = None
    size_bands: Optional[List[str]] = None
    triggers: Optional[List[str]] = None


def _icp_completeness(profile: Dict[str, Any]) -> int:
    """Rough completeness count across the 5 key fields."""
    keys = ["industries", "integrations", "buyer_titles", "size_bands", "triggers"]
    score = 0
    for k in keys:
        v = profile.get(k) or []
        if isinstance(v, list) and any(isinstance(x, str) and x.strip() for x in v):
            score += 1
    return score


def ensure_icp_enriched_with_jina(state: Dict[str, Any]) -> Dict[str, Any]:
    """If ICP profile is sparse, enrich it using DDG + r.jina snippets via LLM extraction.

    - Uses existing discovery planner to obtain domains and r.jina snippets (no direct HTTP crawl).
    - Runs structured LLM extraction (MicroICP) on concatenated snippets to fill the 5 key fields.
    - Merges new values into state['icp_profile'] with de-duplication.
    """
    try:
        icp = dict(state.get("icp_profile") or {})
        if _icp_completeness(icp) >= 3:
            return state
        # Ensure we have r.jina snippets from discovery
        if not state.get("jina_snippets"):
            try:
                _tmp = discovery_planner({"icp_profile": icp})
                # Carry over snippets and candidates
                if isinstance(_tmp.get("jina_snippets"), dict):
                    state["jina_snippets"] = _tmp["jina_snippets"]
                if isinstance(_tmp.get("discovery_candidates"), list):
                    state["discovery_candidates"] = _tmp["discovery_candidates"]
            except Exception as e:
                log.info("[enrich] discovery planner failed: %s", e)
        snips: Dict[str, str] = state.get("jina_snippets") or {}
        if not snips:
            return state
        # Build a concise but content-rich evidence text
        parts = []
        for i, (dom, txt) in enumerate(list(snips.items())[:10], 1):
            try:
                t = " ".join((txt or "").split())[:1200]
                parts.append(f"Domain: {dom}\n{t}")
            except Exception:
                continue
        evidence = "\n\n".join(parts)
        if not evidence.strip():
            return state
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        structured = llm.with_structured_output(MicroICP)
        msgs = ChatPromptTemplate.from_messages([
            ("system", "Extract micro-ICP lists from web page snippets. Return arrays for industries, integrations, buyer_titles, size_bands, triggers. Keep items concise but meaningful; dedupe and lowercase."),
            ("human", evidence),
        ])
        try:
            out = structured.invoke(msgs)
            data = (
                out.model_dump() if hasattr(out, "model_dump") else (
                    out.dict() if hasattr(out, "dict") else (out if isinstance(out, dict) else {})
                )
            )
            for k in ("industries", "integrations", "buyer_titles", "size_bands", "triggers"):
                vals = [
                    (v or "").strip().lower()
                    for v in (data.get(k) or [])
                    if isinstance(v, str) and v.strip()
                ]
                if vals:
                    icp[k] = _uniq((icp.get(k) or []) + vals)
            state["icp_profile"] = icp
            log.info("[enrich] ICP enriched via r.jina snippets; completeness=%d", _icp_completeness(icp))
        except Exception as e:
            log.info("[enrich] LLM extraction failed: %s", e)
    except Exception as e:
        log.info("[enrich] failed: %s", e)
    return state


def _fallback_home_snippet(domain: str) -> str:
    """Fetch a lightweight homepage title/description when Jina is unavailable.

    Returns a short string combining <title> and meta description if found.
    """
    try:
        host = _normalize_host(domain)
        if not host:
            return ""
        url = f"https://{host}"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        r = requests.get(url, headers=headers, timeout=6)
        html = r.text or ""
        if not html:
            return host
        # Extract <title>
        m_title = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.I|re.S)
        title = (m_title.group(1).strip() if m_title else "")
        # Extract meta description
        m_desc = re.search(r"<meta[^>]+name=\"description\"[^>]+content=\"([^\"]+)\"", html, flags=re.I)
        desc = (m_desc.group(1).strip() if m_desc else "")
        text = " - ".join([t for t in [title, desc] if t]) or host
        # normalize whitespace and cap length
        text = " ".join(text.split())
        return text[:400]
    except Exception:
        return (domain or "").strip()


def icp_synthesizer(state: Dict[str, Any]) -> Dict[str, Any]:
    """LLM agent that synthesizes a micro‑ICP from seed snippets and prior ICP.

    Inputs:
      state['seeds'] = [{'url':..., 'snippet':...}, ...] (optional)
      state['icp_profile'] (optional)
    Outputs:
      state['icp_profile'] = { 'industries': [...], 'integrations': [...], 'buyer_titles': [...], 'size_bands': [...], 'triggers': [...] }
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    seeds_text = "\n\n".join([f"URL: {s.get('url','')}\n{(s.get('snippet') or '')[:1000]}" for s in (state.get("seeds") or [])])
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You extract micro-ICP from short evidence. Return JSON with keys: industries[], integrations[], buyer_titles[], size_bands[], triggers[]. Keep values short, lowercase, and deduplicated."),
        ("human", "Existing ICP (optional): {prior}\n\nSeed evidence:\n{seeds}")
    ])
    try:
        structured = llm.with_structured_output(MicroICP)
        msgs = prompt.format_messages(prior=str(state.get("icp_profile") or {}), seeds=seeds_text)
        log.info("[synth] running icp_synthesizer; seeds=%d", len(state.get("seeds") or []))
        out = structured.invoke(msgs)
        data = (
            out.model_dump() if hasattr(out, "model_dump") else (
                out.dict() if hasattr(out, "dict") else (out if isinstance(out, dict) else {})
            )
        )
        icp = dict(state.get("icp_profile") or {})
        for k in ("industries", "integrations", "buyer_titles", "size_bands", "triggers"):
            vals = [v.strip().lower() for v in (data.get(k) or []) if isinstance(v, str) and v.strip()]
            if vals:
                icp[k] = _uniq(vals)
        state["icp_profile"] = icp
    except Exception as e:
        # Non-fatal; keep prior profile
        log.info("[synth] failed: %s", e)
    return state


def discovery_planner(state: Dict[str, Any]) -> Dict[str, Any]:
    """LLM + tool: generate queries from micro‑ICP and collect candidate domains.

    Inputs: state['icp_profile']
    Outputs: state['discovery_candidates'] = ['domain1.com', ...]
    """
    icp = state.get("icp_profile") or {}
    inds = ", ".join(icp.get("industries") or [])
    sig_list = (icp.get("integrations") or []) + (icp.get("triggers") or [])
    sigs = ", ".join(sig_list)
    titles = ", ".join(icp.get("buyer_titles") or [])
    # Derive a coarse country hint for DDG (e.g., 'sg') from seed domains
    country_hint: Optional[str] = None
    try:
        seeds = list(SEED_HINTS)
        if any(str(s).endswith('.sg') for s in seeds):
            country_hint = 'sg'
    except Exception:
        country_hint = None
    # If SG profiles feature is enabled, force SG region bias
    try:
        if ICP_SG_PROFILES:
            country_hint = 'sg'
    except Exception:
        pass

    # Compose one concise DDG query from ICP + website via LLM (fallback to heuristic)
    def _llm_compose_ddg_query(website_text: Optional[str]) -> Optional[str]:
        try:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            inds = ", ".join([s for s in (icp.get("industries") or []) if isinstance(s, str) and s.strip()])
            site = "site:.sg" if (country_hint == 'sg') else ""
            sys = (
                "Compose exactly ONE DuckDuckGo query to find companies strictly within the TARGET INDUSTRY, derived from the customer's WEBSITE TEXT and the industries list. "
                "Strict rules: use only industry words found in the website text or the given industries list; do not invent competitors or seeds; avoid buyer titles and triggers. "
                "Keep it concise (<= 12 words). Include the site filter verbatim if provided. "
                "If both website text and industries are empty, output exactly: b2b distributors (plus the site filter if provided). "
                "Output JUST the query."
            )
            human = (f"website_text: {(website_text or '')[:800]}\n"
                     f"industries: {inds}\n"
                     f"site_filter: {site}")
            from langchain_core.messages import SystemMessage, HumanMessage
            messages = [SystemMessage(content=sys), HumanMessage(content=human)]
            out = llm.invoke(messages)
            q = (getattr(out, "content", None) or "").strip().replace("\n", " ")
            # Guard rails: ensure site filter is present if required
            if country_hint == 'sg' and "site:.sg" not in q:
                q = (q + " site:.sg").strip()
            # Additional guard rails: reject generic/exampley outputs that are not grounded in industries
            try:
                raw_inds = [s for s in (icp.get("industries") or []) if isinstance(s, str)]
                tokens: list[str] = []
                for s in raw_inds:
                    for t in re.split(r"[^a-zA-Z&]+", s.lower()):
                        t = t.strip(" &").strip()
                        if len(t) >= 3:
                            tokens.append(t)
                tokset = set(tokens)
                qlow = q.lower()
                bad_examples = {
                    "food & beverage distributors",
                    "food and beverage distributors",
                    "consumer goods wholesale",
                    "logistics distributors",
                }
                # If industries given but no token overlap OR matches a known bad example, drop to heuristic
                if tokset and (not any(tok in qlow for tok in tokset) or qlow in bad_examples):
                    return None
            except Exception:
                pass
            return q if len(q) >= 4 else None
        except Exception as e:
            log.info("[plan] llm-query fail: %s", e)
            return None

    # Derive website text when available to ground the LLM query
    website_hint = None
    for k in ("website_domain", "website", "website_url", "home", "url", "site"):
        try:
            v = (icp.get(k) if isinstance(icp, dict) else None) or (state.get(k) if isinstance(state, dict) else None)
            if isinstance(v, str) and v.strip():
                website_hint = v.strip()
                break
        except Exception:
            continue
    website_text = None
    if website_hint:
        try:
            u = website_hint if website_hint.startswith("http") else ("https://" + website_hint)
            website_text = jina_read(u, timeout=6) or _fallback_home_snippet(u)
        except Exception:
            website_text = None
    # Prefer LLM-composed, single strict query; fallback to heuristic join
    query = None
    try:
        q_llm = _llm_compose_ddg_query(website_text)
        if q_llm and len(q_llm) >= 4:
            query = q_llm
            try:
                log.info("[planner] llm=1 website_used=%s", bool(website_text))
            except Exception:
                pass
    except Exception:
        query = None
    if not query:
        terms: List[str] = []
        for v in (icp.get("industries") or []):
            if isinstance(v, str) and v.strip():
                terms.append(v.strip())
        inline_site = "site:.sg" if (country_hint == 'sg') else ""
        base_query = (
            (", ".join(_uniq(terms)) + (" " + inline_site if inline_site else "")).strip()
            if terms else ""
        )
        if base_query:
            query = base_query
        else:
            fallback = ", ".join([s for s in [inds, titles, sigs] if s]).strip(", ")
            if country_hint == 'sg':
                query = (f"{fallback} site:.sg" if fallback else "b2b distributors site:.sg").strip()
            else:
                query = fallback or "b2b distributors"

    # Single-query discovery: paginate up to 8 pages, stop at 50
    log.info("[plan] ddg-only query: %s", query)
    domains: List[str] = []
    ddg_count = 0
    try:
        for dom in _ddg_search_domains(query, max_results=50, country=country_hint):
            domains.append(dom)
            ddg_count += 1
    except Exception as e:
        log.info("[plan] ddg fail: %s", e)
    try:
        log.info("[planner] tools.ddg=%d", ddg_count)
    except Exception:
        pass
    uniq = [d for d in _uniq(domains) if _is_probable_domain(str(d))]
    # Apply deny/host hygiene using profiles config (always on; not gated by ICP_SG_PROFILES)
    if uniq:
        cfg = (_load_sg_cfg() if _load_sg_cfg else {})
        _f: list[str] = []
        counts = {"kept": 0, "DOMAIN_HYGIENE": 0, "DENY_HOST": 0, "DENY_PATH": 0}
        for d in uniq:
            try:
                dn = str(d).strip().lower()
                if ENABLE_STRICT_DOMAIN_HYGIENE and _is_valid_fqdn and not _is_valid_fqdn(dn):
                    try:
                        log.info("[plan] drop %s reason=DOMAIN_HYGIENE", dn)
                    except Exception:
                        pass
                    counts["DOMAIN_HYGIENE"] += 1
                    continue
                if (not DISCOVERY_ALLOW_PORTALS) and _is_denied_host and _is_denied_host(dn, cfg):
                    try:
                        log.info("[plan] drop %s reason=DENY_HOST", dn)
                    except Exception:
                        pass
                    counts["DENY_HOST"] += 1
                    continue
                _f.append(dn)
                counts["kept"] += 1
            except Exception:
                continue
        uniq = _f
        try:
            log.info(
                "[plan] host_filter kept=%s drop_hygiene=%s drop_deny=%s drop_path=%s",
                counts.get("kept"), counts.get("DOMAIN_HYGIENE"), counts.get("DENY_HOST"), counts.get("DENY_PATH")
            )
        except Exception:
            pass
    # If nothing found, retry once with a safe default query focused on SG
    if not uniq and (country_hint == 'sg'):
        try:
            _q2 = "b2b distributors site:.sg"
            log.info("[plan] ddg retry with default query: %s", _q2)
            more2: List[str] = []
            for dom in _ddg_search_domains(_q2, max_results=50, country=country_hint):
                more2.append(dom)
            muniq = [d for d in _uniq(more2) if _is_probable_domain(str(d))]
            if muniq:
                domains.extend(muniq)
                uniq = [d for d in _uniq(domains) if _is_probable_domain(str(d))]
        except Exception as _e2:
            log.info("[plan] ddg default retry failed: %s", _e2)
    # If SG-biased discovery is active, also include a relaxed pass without site:.sg to collect non-SG
    try:
        if country_hint == 'sg':
            q_relaxed2 = re.sub(r"\bsite:[^\s]+", "", query).strip()
            if q_relaxed2 and q_relaxed2 != query:
                more2: List[str] = []
                for dom in _ddg_search_domains(q_relaxed2, max_results=50, country=country_hint):
                    more2.append(dom)
                if more2:
                    merged = [d for d in _uniq(uniq + more2) if _is_probable_domain(str(d))]
                    # Prioritize .sg domains first, then the rest
                    sg = [d for d in merged if str(d).endswith('.sg')]
                    non = [d for d in merged if not str(d).endswith('.sg')]
                    uniq = sg + non
    except Exception as _e_relax:
        pass

    # Snapshot full set prior to seed exclusion for logging/audit
    uniq_all: List[str] = list(uniq)
    # Do not alter the original query; skip relaxed re-query unless explicitly enabled
    try:
        import os as _os
        _relax = (_os.getenv("ENABLE_DDG_RELAX") or "").strip().lower() in {"1", "true", "yes"}
    except Exception:
        _relax = False
    if _relax:
        try:
            if len(uniq) < 10:
                q_relaxed = re.sub(r"\bsite:[^\s]+", "", query).strip()
                if q_relaxed and q_relaxed != query:
                    more: List[str] = []
                    try:
                        for dom in _ddg_search_domains(q_relaxed, max_results=50, country=country_hint):
                            more.append(dom)
                    except Exception:
                        more = []
                    if more:
                        uniq = [d for d in _uniq(uniq + more) if _is_probable_domain(str(d))]
                        try:
                            log.info("[plan] ddg relaxed domains added=%d", len(more))
                        except Exception:
                            pass
        except Exception:
            pass
    # Exclude seed domains (by apex) from discovery set to avoid reprocessing submitted customers
    try:
        seed_apex = {_apex_domain(s) for s in (SEED_HINTS or [])}
        uniq = [d for d in uniq if _apex_domain(d) not in seed_apex]
    except Exception:
        pass
    log.info("[plan] ddg domains found=%d (uniq=%d)", len(domains), len(uniq))
    # Emit full discovery list (including any seeds) for observability
    try:
        log.info("[plan] discovery all urls=%s", ", ".join([f"https://{d}" for d in uniq_all]))
    except Exception:
        pass
    # Optional Jina Reader fetch for quick homepage snippets of first few domains
    # Industry-based filter: keep only candidates whose snippets mention industry terms
    def _industry_terms(profile: Dict[str, Any]) -> list[str]:
        try:
            raw = [s for s in (profile.get("industries") or []) if isinstance(s, str)]
            toks: list[str] = []
            for s in raw:
                for t in re.split(r"[^a-zA-Z&]+", s.lower()):
                    t = t.strip(" &").strip()
                    if len(t) >= 3:
                        toks.append(t)
            return sorted(set(toks))
        except Exception:
            return []
    ind_toks = _industry_terms(icp)
    jina_snips: Dict[str, str] = {}
    # Limit homepage snapshot checks to a small set to reduce latency
    for d in uniq[:5]:
        try:
            url = f"https://{d}"
            reader = f"https://r.jina.ai/http://{d}"
            log.info("[jina] GET %s", reader)
            # Shorter timeout to avoid long stalls during planning
            r = requests.get(reader, timeout=6)
            txt = (r.text or "")[:8000]
            # Clean noisy prefixes often present in r.jina output
            lines = [ln.strip() for ln in (txt or "").splitlines() if ln.strip()]
            filtered = [
                ln for ln in lines
                if not re.match(r"^(Title:|URL Source:|Published Time:|Markdown Content:|Warning:)", ln, flags=re.I)
            ]
            clean = " ".join(filtered) if filtered else " ".join((txt or "").split())
            snip = clean[:400]
            low = clean.lower()
            if ind_toks:
                if not any(tok in low for tok in ind_toks):
                    # Skip off-industry candidates early
                    continue
            jina_snips[d] = snip
            log.info("[jina] ok len=%d domain=%s", len(txt), d)
        except Exception as e:
            log.info("[jina] fail domain=%s err=%s", d, e)
            continue
    # Final list of discovery candidates (cap at 50)
    state["discovery_candidates"] = uniq[:50]
    try:
        urls = [f"https://{d}" for d in state["discovery_candidates"]]
        log.info("[plan] discovery candidates total=%d", len(urls))
        # Emit full list (up to 50) as URLs for auditability
        log.info("[plan] discovery candidates urls=%s", ", ".join(urls))
    except Exception:
        pass
    if jina_snips:
        state["jina_snippets"] = jina_snips
    return state


def compliance_guard(state: Dict[str, Any]) -> Dict[str, Any]:
    """Compliance guard to prune discovery candidates prior to enrichment.

    - Enforces FQDN hygiene and deny host/suffix.
    - Applies SG markers gating when ICP_SG_PROFILES is enabled.
    - Emits drop counters and replaces state['discovery_candidates'] with filtered list.
    """
    try:
        cand = [str(d).strip().lower() for d in (state.get("discovery_candidates") or []) if isinstance(d, str) and d.strip()]
        if not cand:
            return state
        cfg = (_load_sg_cfg() if _load_sg_cfg else {})
        # Prepare optional SG marker check inputs
        jina_snips: Dict[str, str] = state.get("jina_snippets") or {}
        kept: list[str] = []
        counts = {"DOMAIN_HYGIENE": 0, "DENY_HOST": 0, "NO_SG_MARKERS": 0, "kept": 0}
        # Exclude seed apexes if provided via hints
        try:
            seed_apex = {_apex_domain(s) for s in (SEED_HINTS or [])}
        except Exception:
            seed_apex = set()
        ambiguous: list[str] = []
        for d in cand:
            try:
                dn = _apex_domain(_clean_possible_percent_encoded(d))
                if ENABLE_STRICT_DOMAIN_HYGIENE and _is_valid_fqdn and not _is_valid_fqdn(dn):
                    counts["DOMAIN_HYGIENE"] += 1
                    continue
                if _apex_domain(dn) in seed_apex:
                    counts.setdefault("SEED_EXCLUDED", 0)
                    counts["SEED_EXCLUDED"] += 1
                    continue
                if (not DISCOVERY_ALLOW_PORTALS) and _is_denied_host and _is_denied_host(dn, cfg):
                    counts["DENY_HOST"] += 1
                    continue
                # SG markers gating when enabled
                if ICP_SG_PROFILES:
                    if (not dn.endswith('.sg')):
                        snip = jina_snips.get(dn) or ""
                        try:
                            if not (_is_sg_page and _is_sg_page(snip or dn, cfg)):
                                counts["NO_SG_MARKERS"] += 1
                                ambiguous.append(dn)
                                continue
                        except Exception:
                            counts["NO_SG_MARKERS"] += 1
                            ambiguous.append(dn)
                            continue
                kept.append(dn)
                counts["kept"] += 1
            except Exception:
                counts["DOMAIN_HYGIENE"] += 1
                continue
        # LLM tie-breaker for up to 3 ambiguous non-.sg candidates lacking SG markers
        llm_tiebreak = 0
        try:
            if ambiguous:
                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
                for dn in ambiguous[:3]:
                    snip = jina_snips.get(dn) or ""
                    if not snip:
                        try:
                            snip = jina_read(f"https://{dn}", timeout=6) or ""
                        except Exception:
                            snip = ""
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", "Decide if this company is based in or primarily targeting Singapore. Reply 'keep' or 'drop' and a short reason."),
                        ("human", f"Domain: {dn}\nSnippet: {(snip or '')[:800]}"),
                    ])
                    out = llm.invoke(prompt)
                    txt = (getattr(out, "content", None) or "").lower()
                    if "keep" in txt and dn not in kept:
                        kept.append(dn)
                        counts["kept"] += 1
                        llm_tiebreak += 1
        except Exception:
            pass
        kept = _uniq(kept)
        state["discovery_candidates"] = kept
        state["guard_kept"] = counts.get("kept")
        state["guard_drops"] = {k: v for k, v in counts.items() if k != "kept"}
        try:
            log.info(
                "[guard] kept=%s drop_hygiene=%s drop_deny=%s drop_sg=%s guard.llm_tiebreak=%s",
                counts.get("kept"), counts.get("DOMAIN_HYGIENE"), counts.get("DENY_HOST"), counts.get("NO_SG_MARKERS"), llm_tiebreak
            )
        except Exception:
            pass
        return state
    except Exception:
        return state


async def mini_crawl_worker(state: Dict[str, Any]) -> Dict[str, Any]:
    """Jina-based: read homepage snapshots per candidate domain and attach text evidence.

    Inputs: state['tenant_id'], state['discovery_candidates']
    Outputs: state['evidence'] = [{domain, summary}, ...] where summary is plain text
    """
    out: List[Dict[str, Any]] = []
    cand = (state.get("discovery_candidates") or [])
    # Reduce page reads and shorten timeouts to keep confirm fast
    log.info("[mini] jina-read start count=%d", len(cand[:5]))
    for dom in (state.get("discovery_candidates") or [])[:5]:
        url = f"https://{dom}"
        try:
            txt = jina_read(url, timeout=6)
            if not txt:
                log.info("[mini] jina empty domain=%s", dom)
                continue
            out.append({"domain": dom, "summary": txt[:4000]})
        except Exception as e:
            log.info("[mini] jina fail domain=%s err=%s", dom, e)
            continue
    log.info("[mini] jina-read done ok=%d", len(out))
    state["evidence"] = out
    return state


def evidence_extractor(state: Dict[str, Any]) -> Dict[str, Any]:
    """LLM extraction: convert raw crawl summaries into normalized icp_evidence-like records.

    Inputs: state['evidence']
    Outputs: state['evidence'] augmented with 'signals' and normalized fields.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    class EvidenceOut(BaseModel):
        evidence_types: Optional[int] = None
        integrations: Optional[List[str]] = None
        buyer_titles: Optional[List[str]] = None
        hiring_open_roles: Optional[int] = None
        has_pricing: Optional[bool] = None
        has_case_studies: Optional[bool] = None

    structured = llm.with_structured_output(EvidenceOut)
    items = []
    for ev in (state.get("evidence") or []):
        try:
            txt = str(ev.get("summary") or "")
            out = structured.invoke([("system", "Extract ICP evidence fields."), ("human", txt)])
            data = (
                out.model_dump() if hasattr(out, "model_dump") else (
                    out.dict() if hasattr(out, "dict") else (out if isinstance(out, dict) else {})
                )
            )
            ev2 = dict(ev)
            for k in ("evidence_types", "integrations", "buyer_titles", "hiring_open_roles", "has_pricing", "has_case_studies"):
                if k in data and data[k] is not None:
                    ev2[k] = data[k]
            items.append(ev2)
        except Exception:
            items.append(ev)
    state["evidence"] = items
    return state


def scoring_and_gating(state: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic scoring stub. In production, use lead_scoring.py and DB features.

    Inputs: state['evidence']
    Outputs: state['scores'] with A/B/C buckets and reason stubs.
    """
    scores: List[Dict[str, Any]] = []
    for ev in (state.get("evidence") or []):
        score = 0
        # Emphasize integrations/titles/pricing per PRD19 tuning
        ev_types = int(ev.get("evidence_types") or 0)
        ints = ev.get("integrations") or []
        titles = ev.get("buyer_titles") or []
        roles = int(ev.get("hiring_open_roles") or 0)
        has_pricing = bool(ev.get("has_pricing"))
        has_cs = bool(ev.get("has_case_studies"))

        score += 15 if ev_types >= 3 else 0
        score += 35 if ints else 0
        score += 20 if titles else 0
        score += 5 if roles >= 3 else 0
        score += 15 if has_pricing else 0
        score += 10 if has_cs else 0

        bucket = "A" if score >= 70 else ("B" if score >= 50 else "C")
        reason = {
            "integrations": ints[:5],
            "titles": titles[:5],
            "pricing": has_pricing,
            "case_studies": has_cs,
        }
        scores.append({"domain": ev.get("domain"), "score": score, "bucket": bucket, "reason": reason})
    # Log simple analysis summary
    try:
        tops = sorted(scores, key=lambda x: int(x.get("score") or 0), reverse=True)[:3]
        log.info(
            "[analyze] scored=%d top=%s",
            len(scores),
            ", ".join([f"{t.get('domain')}:{t.get('score')}" for t in tops]),
        )
    except Exception:
        pass
    state["scores"] = scores
    return state


def plan_top10_with_reasons(icp_profile: Dict[str, Any], tenant_id: int | None = None) -> List[Dict[str, Any]]:
    """Convenience helper: discovery → mini-crawl → extract → score.

    Returns up to 50 candidates with fields: domain, score, bucket, why, snippet.
    The UI presents Top‑10; the remainder (up to 40) is queued for background enrich.
    """
    try:
        # 1) plan
        s = {"icp_profile": dict(icp_profile or {})}
        s = discovery_planner(s)
        cand: List[str] = [d for d in (s.get("discovery_candidates") or []) if _is_probable_domain(str(d))]
        # Exclude seed domains from discovery candidates (by apex)
        try:
            seed_apex = {_apex_domain(h) for h in (SEED_HINTS or [])}
        except Exception:
            seed_apex = set()
        cand = [d for d in cand if _apex_domain(str(d)) not in seed_apex]
        jina_snips: Dict[str, str] = s.get("jina_snippets") or {}
        if not cand:
            return []
        # 2) Jina reader snapshot (head only to keep latency low)
        ev_list: List[Dict[str, Any]] = []
        # Industry filter helpers
        def _ind_terms(icp_prof: Dict[str, Any]) -> list[str]:
            try:
                raw = [s for s in (icp_prof.get("industries") or []) if isinstance(s, str)]
                toks: list[str] = []
                for s in raw:
                    for t in re.split(r"[^a-zA-Z&]+", s.lower()):
                        t = t.strip(" &").strip()
                        if len(t) >= 3:
                            toks.append(t)
                return sorted(set(toks))
            except Exception:
                return []
        ind_toks = _ind_terms(icp_profile)
        # Analyze a smaller head to reduce latency while keeping quality reasonable
        # Expand head to improve recall; we will still cap UI later
        HEAD = 12
        # SG config (optional)
        cfg = (_load_sg_cfg() if _load_sg_cfg else {})
        for d in cand[:HEAD]:
            url = f"https://{d}"
            try:
                t = int(tenant_id or 0)
            except Exception:
                t = 0
            try:
                log.info("[mini] jina-read domain=%s", d)
                summ = jina_read(url, timeout=6)
                if not summ:
                    # Fallback: fetch homepage title/description when Jina is rate limited
                    try:
                        fb_txt = _fallback_home_snippet(d)
                    except Exception:
                        fb_txt = ""
                    if not fb_txt:
                        # Still record a minimal placeholder to keep candidate in flow
                        fb_txt = d
                    # Relax gating on fallback snippets: do not drop due to missing industry tokens
                    # SG gating: accept .sg; for non-.sg require SG markers in fallback text when feature is on
                    allow = True
                    if ICP_SG_PROFILES:
                        try:
                            allow = (str(d).lower().endswith('.sg')) or (
                                _is_sg_page and _is_sg_page(str(fb_txt), cfg)
                            )
                        except Exception:
                            allow = str(d).lower().endswith('.sg')
                    if allow:
                        ev_list.append({"domain": d, "summary": str(fb_txt)[:4000]})
                    else:
                        try:
                            log.info("[mini] drop %s reason=NO_SG_MARKERS_FALLBACK", d)
                        except Exception:
                            pass
                else:
                    low = str(summ).lower()
                    # Industry gating
                    if ind_toks and not any(tok in low for tok in ind_toks):
                        # Skip off-industry entries
                        try:
                            log.info("[mini] drop %s reason=INDUSTRY_GATING", d)
                        except Exception:
                            pass
                        continue
                    # SG gating when feature is enabled: accept .sg or require SG markers in text
                    if ICP_SG_PROFILES:
                        try:
                            if (not str(d).lower().endswith('.sg')) and not (_is_sg_page and _is_sg_page(str(summ), cfg)):
                                # Skip non-SG pages with no SG markers
                                try:
                                    log.info("[mini] drop %s reason=NO_SG_MARKERS", d)
                                except Exception:
                                    pass
                                continue
                        except Exception:
                            if not str(d).lower().endswith('.sg'):
                                try:
                                    log.info("[mini] drop %s reason=NO_SG_MARKERS", d)
                                except Exception:
                                    pass
                                continue
                    ev_list.append({"domain": d, "summary": str(summ)[:4000]})
            except Exception as e:
                log.info("[mini] jina fail domain=%s err=%s", d, e)
                continue
        if not ev_list:
            # Final guard: build minimal evidence from homepage titles to avoid empty list
            for d in cand[:HEAD]:
                try:
                    fb = _fallback_home_snippet(d)
                except Exception:
                    fb = d
                if fb:
                    ev_list.append({"domain": d, "summary": str(fb)[:4000]})
        # 3) extract (micro-ICP signals) then 4) score using profile-weighted helper
        st2 = {"evidence": ev_list}
        st2 = evidence_extractor(st2)
        cfg = (_load_sg_cfg() if _load_sg_cfg else {})
        profile_name = None
        try:
            p = (icp_profile or {}).get("lead_profile")
            if isinstance(p, str) and p.strip():
                profile_name = p.strip()
            else:
                profile_name = (cfg.get("profile") or "sg_employer_buyers")
        except Exception:
            profile_name = (cfg.get("profile") or "sg_employer_buyers")
        # Build items by rescoring each evidence summary
        top: List[Dict[str, Any]] = []
        for ev in (st2.get("evidence") or []):
            try:
                dom = ev.get("domain")
                summ = ev.get("summary") or ""
                sc, why_bits, br = _score_profile(str(summ), str(dom), profile_name, cfg)
                # Prefer Jina short snippet if available
                snip_raw = jina_snips.get(dom) if isinstance(jina_snips, dict) else None
                snip = None
                try:
                    snip = (" ".join((snip_raw or "").split()))[:180]
                except Exception:
                    snip = None
                top.append({
                    "domain": dom,
                    "score": int(sc),
                    "bucket": _bucket(sc),
                    "why": "; ".join(why_bits) if why_bits else "signal match",
                    "snippet": snip,
                    "breakdown": br,
                    "lead_profile": profile_name,
                })
            except Exception:
                continue
        # Sort by score desc and cap
        top = sorted([it for it in top if it.get("domain")], key=lambda x: int(x.get("score") or 0), reverse=True)
        # If fewer than desired, backfill using additional candidates (heuristic scoring) and seeds/legacy fallbacks
        DESIRED = 50
        if len(top) < DESIRED:
            # 4a) Heuristic backfill from remaining discovery candidates
            seen = {str((it.get("domain") or "").strip().lower()) for it in top}
            extra: List[Dict[str, Any]] = []
            # Cap additional checks to avoid long serial fetches
            for d in cand[HEAD:HEAD+20]:
                try:
                    if not d:
                        continue
                    dn = str(d).strip().lower()
                    if dn in seen or (not _is_probable_domain(dn)):
                        continue
                    if _apex_domain(dn) in seed_apex:
                        continue
                    url = f"https://{d}"
                    body = jina_read(url, timeout=5) or ""
                    from_fallback = False
                    if not body:
                        # Use relaxed fallback when Jina times out or is rate limited
                        try:
                            clean = _fallback_home_snippet(d) or d
                        except Exception:
                            clean = d
                        from_fallback = True
                    else:
                        clean = " ".join((body or "").split())
                    low = str(clean or "").lower()
                    # Industry gating
                    def _ind_terms2(icp_prof: Dict[str, Any]) -> list[str]:
                        try:
                            raw = [s for s in (icp_prof.get("industries") or []) if isinstance(s, str)]
                            toks: list[str] = []
                            for s in raw:
                                for t in re.split(r"[^a-zA-Z&]+", s.lower()):
                                    t = t.strip(" &").strip()
                                    if len(t) >= 3:
                                        toks.append(t)
                            return sorted(set(toks))
                        except Exception:
                            return []
                    ind_toks2 = _ind_terms2(icp_profile)
                    # Relax gating when we are using fallback snippets
                    if not from_fallback and ind_toks2 and not any(tok in low for tok in ind_toks2):
                        continue
                    score = 0
                    why_bits: List[str] = []
                    if "integrat" in low:
                        score += 35
                        why_bits.append("integrations signals")
                    if "pricing" in low or "plans" in low:
                        score += 15
                        why_bits.append("pricing page found")
                    if "case stud" in low or "customers" in low:
                        score += 10
                        why_bits.append("case studies")
                    if "careers" in low or "jobs" in low or "hiring" in low:
                        score += 5
                        why_bits.append("hiring")
                    snip = clean[:180]
                    bucket = "A" if score >= 70 else ("B" if score >= 50 else "C")
                    extra.append({
                        "domain": d,
                        "score": score,
                        "bucket": bucket,
                        "why": "; ".join(why_bits) if why_bits else "signal match",
                        "snippet": snip,
                    })
                    seen.add(dn)
                    if len(top) + len(extra) >= DESIRED:
                        # stop once we have enough
                        break
                except Exception:
                    continue
            if extra:
                extra = sorted(extra, key=lambda x: int(x.get("score") or 0), reverse=True)
                top = (top + extra)
            # 4b) Disable any additional DDG-based fallbacks to enforce single-query discovery.
            #     We intentionally do NOT run seed-competitor queries or legacy heuristic queries
            #     that would trigger extra DDG calls. Only the first r.jina+DDG query is used.
        # Final fill: ensure all discovered candidates are collected (with minimal placeholders)
        if len(top) < len(cand):
            present = {str((it.get("domain") or "").strip().lower()) for it in top}
            fillers: List[Dict[str, Any]] = []
            for d in cand:
                dn = str(d or "").strip().lower()
                if not dn or dn in present:
                    continue
                snip = None
                try:
                    snip = (_fallback_home_snippet(d) or "").strip()[:180]
                except Exception:
                    snip = None
                fillers.append({
                    "domain": d,
                    "score": 0,
                    "bucket": "C",
                    "why": "signal match",
                    "snippet": snip,
                })
                if len(top) + len(fillers) >= DESIRED:
                    break
            if fillers:
                top.extend(fillers)
        # Return at most 50 for persistence (UI will only show Top‑10)
        total_ret = min(DESIRED, len(top))
        top = top[:total_ret]
        try:
            # Keep original Top‑10 log for backwards compatibility
            log.info("[confirm] agent top10 count=%d", min(10, len(top)))
            # New: log total planned count
            log.info("[confirm] agent planned total=%d", len(top))
        except Exception:
            pass
        # Display message after analysis for backend logs
        log.info("ICP Profile")
        return top
    except Exception as e:
        log.info("[top10] failed: %s", e)
        return []


def plan_top10_with_reasons_fallback(icp_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Fallback Top‑10 planner that does not require LLMs.

    - Builds simple queries from industries/titles/triggers.
    - Uses strict DuckDuckGo HTML search for domains.
    - Fetches homepage text via r.jina.ai for snippets.
    - Scores heuristically based on keyword signals.
    """
    try:
        icp = dict(icp_profile or {})
        inds = [s.strip() for s in (icp.get("industries") or []) if isinstance(s, str) and s.strip()]
        # Strict industry-only DDG query composition
        titles: List[str] = []
        sigs: List[str] = []
        queries: List[str] = []
        if inds:
            queries.append(f"B2B companies in {', '.join(inds[:2])}")
        if not queries:
            # minimal generic industry-only fallbacks
            queries = ["B2B distributors", "B2B companies"]

        # Discover domains strictly via DDG
        domains: List[str] = []
        for q in queries[:3]:
            for dom in _ddg_search_domains(q, max_results=25):
                domains.append(dom)
        uniq = [d for d in _uniq(domains) if _is_probable_domain(str(d))][:20]
        if not uniq:
            return []

        # Fetch snippets via Jina Reader and score heuristically
        out: List[Dict[str, Any]] = []
        for d in uniq[:10]:
            try:
                reader = f"https://r.jina.ai/http://{d}"
                r = requests.get(reader, timeout=10)
                txt = (r.text or "")[:8000]
                from src.jina_reader import clean_jina_text as _clean
                clean = _clean(txt)
                snip = clean[:180]
                # Heuristic scoring
                low = clean.lower()
                score = 0
                why_bits: List[str] = []
                if "integrat" in low:
                    score += 35
                    why_bits.append("integrations signals")
                if "pricing" in low or "plans" in low:
                    score += 15
                    why_bits.append("pricing page found")
                if "case stud" in low or "customers" in low:
                    score += 10
                    why_bits.append("case studies")
                if "careers" in low or "jobs" in low or "hiring" in low:
                    score += 5
                    why_bits.append("hiring")
                bucket = "A" if score >= 70 else ("B" if score >= 50 else "C")
                out.append({
                    "domain": d,
                    "score": score,
                    "bucket": bucket,
                    "why": "; ".join(why_bits) if why_bits else "signal match",
                    "snippet": snip,
                })
            except Exception:
                continue
        out = sorted(out, key=lambda x: int(x.get("score") or 0), reverse=True)[:10]
        return out
    except Exception as e:
        log.info("[top10-fallback] failed: %s", e)
        return []


def _extract_domains_from_text(text: str) -> List[str]:
    try:
        pats = re.findall(r"\b((?:[a-z0-9-]+\.)+[a-z]{2,})(?:/|\b)", text, flags=re.I)
        outs: List[str] = []
        for h in _uniq([p.lower() for p in pats]):
            if any(x in h for x in (
                "duckduckgo.", "google.", "bing.", "brave.", "yahoo.", "yandex.",
                "cloudflare.", "wikipedia.", "wikimedia.", "github.", "stackexchange.",
                "facebook.", "linkedin.", "twitter.", "x.com", "instagram.", "youtube.",
            )):
                continue
            if _is_probable_domain(h):
                outs.append(h)
        return [d for d in outs if _is_probable_domain(d)]
    except Exception:
        return []


def fallback_top10_from_seeds(seed_domains: List[str], icp_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Seeds fallback without crawling seed sites.

    Strategy:
    - For each seed, derive 1–2 compact DDG queries (e.g., "<seed_label> competitor", "b2b distributor similar to <seed_label>")
    - Aggregate DDG domains, exclude seeds, validate, dedupe
    - Fetch r.jina snippets for candidates; score heuristically using ICP cues
    - Return Top‑10 [{domain, score, bucket, why, snippet}]
    """
    try:
        seeds = [d.strip().lower() for d in (seed_domains or []) if isinstance(d, str) and d.strip()]
        if not seeds:
            return []
        seed_set = set(seeds)
        # Create a small query set using seed labels (no seed crawling)
        qset: List[str] = []
        for d in seeds[:5]:
            # Derive brand label from apex domain (avoid 'www')
            try:
                apex = _apex_domain(d)
                parts = [p for p in apex.split('.') if p]
                label = parts[0] if parts else (d.split('.')[0])
                if label == 'www' and len(parts) >= 2:
                    label = parts[0]
            except Exception:
                label = d.split('.')[0]
            # lightweight, concise variants
            if label and label != 'www':
                qset.append(f"{label} competitors")
                qset.append(f"similar to {label} distributor")
        # Region hint: prefer .sg if any seed endswith .sg
        use_sg = any(s.endswith(".sg") for s in seeds)
        if use_sg:
            qset = [q + " site:.sg" for q in qset]
        # Run DDG for each query (cap)
        cand: List[str] = []
        for q in qset[:6]:
            for dom in _ddg_search_domains(q, max_results=25):
                cand.append(dom)
        uniq = [h for h in _uniq(cand) if _is_probable_domain(h) and h not in seed_set][:60]
        if not uniq:
            return []
        inds = ", ".join([s for s in (icp_profile.get("industries") or []) if isinstance(s, str)])
        out: List[Dict[str, Any]] = []
        # Industry tokens for gating
        def _ind_terms3(icp_prof: Dict[str, Any]) -> list[str]:
            try:
                raw = [s for s in (icp_prof.get("industries") or []) if isinstance(s, str)]
                toks: list[str] = []
                for s in raw:
                    for t in re.split(r"[^a-zA-Z&]+", s.lower()):
                        t = t.strip(" &").strip()
                        if len(t) >= 3:
                            toks.append(t)
                return sorted(set(toks))
            except Exception:
                return []
        ind_toks3 = _ind_terms3(icp_profile)
        for d in uniq[:30]:
            try:
                # Prefer Jina but fall back to direct homepage title/description on 429/network errors
                try:
                    reader = f"https://r.jina.ai/http://{d}"
                    r = requests.get(reader, timeout=10)
                    raw = (r.text or "")[:8000]
                    from src.jina_reader import clean_jina_text as _clean
                    clean = _clean(raw)
                except Exception:
                    clean = _fallback_home_snippet(d) or d
                snip = (clean or "")[:180]
                low = (clean or "").lower()
                if ind_toks3 and not any(tok in low for tok in ind_toks3):
                    continue
                score = 0
                why_bits: List[str] = []
                if "integrat" in low:
                    score += 35
                    why_bits.append("integrations signals")
                if "pricing" in low or "plans" in low:
                    score += 15
                    why_bits.append("pricing page found")
                if "case stud" in low or "customers" in low:
                    score += 10
                    why_bits.append("case studies")
                if "careers" in low or "jobs" in low or "hiring" in low:
                    score += 5
                    why_bits.append("hiring")
                if any(w in low for w in ["food service", "foodservice", "distribution", "distributor"]):
                    score += 10
                if inds and any(w.strip().lower() in low for w in inds.split(",")):
                    score += 10
                bucket = "A" if score >= 70 else ("B" if score >= 50 else "C")
                out.append({
                    "domain": d,
                    "score": score,
                    "bucket": bucket,
                    "why": "; ".join(why_bits) if why_bits else "signal match",
                    "snippet": snip,
                })
            except Exception:
                continue
        out = sorted(out, key=lambda x: int(x.get("score") or 0), reverse=True)[:10]
        return out
    except Exception as e:
        log.info("[top10-seeds-fallback] failed: %s", e)
        return []


def fallback_top10_via_seed_outlinks(seed_domains: List[str], icp_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Fallback that avoids DDG entirely by mining outbound domains from seed homepages via r.jina.ai.

    - Fetch each seed's homepage snapshot (r.jina.ai)
    - Extract domain mentions from the text
    - Filter out social/CDN/seeds; prefer same-region TLDs (e.g., .sg) when present
    - Fetch a short snippet for each candidate and score heuristically using ICP cues
    - Return Top‑10 [{domain, score, bucket, why, snippet}]
    """
    try:
        seeds = [d.strip().lower() for d in (seed_domains or []) if isinstance(d, str) and d.strip()]
        if not seeds:
            return []
        seed_apex = {_apex_domain(s) for s in seeds}
        prefer_sg = any(s.endswith(".sg") for s in seeds)
        # Collect outlink domains from seed pages
        cand: List[str] = []
        for s in seeds[:6]:
            try:
                reader = f"https://r.jina.ai/http://{_normalize_host(s)}"
                log.info("[jina] GET %s", reader)
                r = requests.get(reader, timeout=10)
                raw = (r.text or "")[:10000]
                from src.jina_reader import clean_jina_text as _clean
                clean = _clean(raw)
                for h in _extract_domains_from_text(clean):
                    cand.append(h)
            except Exception as e:
                log.info("[jina] seed read fail %s err=%s", s, e)
                continue
        uniq: List[str] = []
        seen = set()
        for h in cand:
            try:
                if not _is_probable_domain(h):
                    continue
                apex = _apex_domain(h)
                if apex in seed_apex:
                    continue
                if apex not in seen:
                    seen.add(apex)
                    uniq.append(apex)
            except Exception:
                continue
        # Prefer same-region TLDs first
        if prefer_sg:
            sg = [d for d in uniq if d.endswith(".sg")]
            non = [d for d in uniq if not d.endswith(".sg")]
            uniq = sg + non
        uniq = uniq[:50]
        if not uniq:
            return []
        # Score candidates using lightweight homepage snippets
        def _ind_terms4(icp_prof: Dict[str, Any]) -> list[str]:
            try:
                raw = [s for s in (icp_prof.get("industries") or []) if isinstance(s, str)]
                toks: list[str] = []
                for s in raw:
                    for t in re.split(r"[^a-zA-Z&]+", s.lower()):
                        t = t.strip(" &").strip()
                        if len(t) >= 3:
                            toks.append(t)
                return sorted(set(toks))
            except Exception:
                return []
        ind_toks = _ind_terms4(icp_profile)
        out: List[Dict[str, Any]] = []
        for d in uniq[:20]:
            try:
                try:
                    reader = f"https://r.jina.ai/http://{d}"
                    r = requests.get(reader, timeout=10)
                    from src.jina_reader import clean_jina_text as _clean
                    clean = _clean((r.text or "")[:8000])
                except Exception:
                    clean = _fallback_home_snippet(d) or d
                low = (clean or "").lower()
                if ind_toks and not any(tok in low for tok in ind_toks):
                    continue
                score = 0
                why_bits: List[str] = []
                if "integrat" in low:
                    score += 35
                    why_bits.append("integrations signals")
                if "pricing" in low or "plans" in low:
                    score += 15
                    why_bits.append("pricing page found")
                if "case stud" in low or "customers" in low:
                    score += 10
                    why_bits.append("case studies")
                if "careers" in low or "jobs" in low or "hiring" in low:
                    score += 5
                    why_bits.append("hiring")
                snip = (clean or "")[:180]
                bucket = "A" if score >= 70 else ("B" if score >= 50 else "C")
                out.append({
                    "domain": d,
                    "score": score,
                    "bucket": bucket,
                    "why": "; ".join(why_bits) if why_bits else "signal match",
                    "snippet": snip,
                })
            except Exception:
                continue
        out = sorted(out, key=lambda x: int(x.get("score") or 0), reverse=True)[:10]
        return out
    except Exception as e:
        log.info("[top10-seed-outlinks] failed: %s", e)
        return []

def build_icp_agents_graph():
    """Build a small LangGraph with the core nodes described in PRD19 §6.

    Note: This graph operates on an in‑memory state and does not write to the DB.
    Integrate with persistence by calling existing modules (e.g., lead_scoring, icp_pipeline) where appropriate.
    """
    from langgraph.graph import StateGraph

    def _ensure_async(fn):
        if asyncio.iscoroutinefunction(fn):
            return fn
        async def aw(state):
            return fn(state)
        return aw

    g = StateGraph(dict)
    g.add_node("synthesize_icp", _ensure_async(icp_synthesizer))
    g.add_node("plan_discovery", _ensure_async(discovery_planner))
    g.add_node("mini_crawl", mini_crawl_worker)
    g.add_node("extract_evidence", _ensure_async(evidence_extractor))
    g.add_node("score_gate", _ensure_async(scoring_and_gating))

    g.set_entry_point("synthesize_icp")
    g.add_edge("synthesize_icp", "plan_discovery")
    g.add_edge("plan_discovery", "mini_crawl")
    g.add_edge("mini_crawl", "extract_evidence")
    g.add_edge("extract_evidence", "score_gate")
    return g.compile()
