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
    ENABLE_JINA_DEEP_RESEARCH_HEURISTICS,
    JINA_DEEP_RESEARCH_HEURISTIC_CHECK_MAX,
    JINA_DEEP_RESEARCH_HEURISTIC_TIMEOUT_S,
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


def _extract_domains_from_markdown_table(text: str) -> List[str]:
    """Extract domain candidates from a Markdown table with a 'domain_url' column.

    Looks for a header row containing 'domain_url' and parses the rows beneath,
    returning the raw cell values from that column (e.g., 'example.com' or 'foo.sg').
    """
    out: List[str] = []
    if not isinstance(text, str) or not text:
        return out
    lines = [ln.strip() for ln in text.splitlines()]
    # Find header row
    header_idx = -1
    for i, ln in enumerate(lines):
        if "|" in ln and "domain_url" in ln.lower():
            header_idx = i
            break
    if header_idx < 0:
        return out
    headers = [h.strip().lower() for h in lines[header_idx].strip("|").split("|")]
    try:
        col_idx = headers.index("domain_url")
    except ValueError:
        return out
    # Rows typically start after a separator row; skip next line if it's a --- table rule
    row_start = header_idx + 1
    if row_start < len(lines) and set(lines[row_start].replace("|", "").replace("-", "").strip()) == set():
        row_start += 1
    # Parse subsequent rows until a blank or non-table line
    for j in range(row_start, len(lines)):
        ln = lines[j]
        if "|" not in ln:
            # End of table
            break
        cols = [c.strip().strip("`") for c in ln.strip("|").split("|")]
        if not cols or col_idx >= len(cols):
            continue
        cell = cols[col_idx]
        if cell:
            out.append(cell)
    return out


def _extract_bare_domains(text: str) -> List[str]:
    """Extract bare domain-like tokens (without scheme) from text.

    Matches tokens like 'example.com', 'foo.co.sg', ignoring trailing punctuation.
    """
    out: List[str] = []
    if not isinstance(text, str) or not text:
        return out
    # Simple domain pattern; exclude extremely long TLD parts
    for m in re.finditer(r"\b([a-z0-9][-a-z0-9]*\.)+[a-z]{2,}\b", text, re.IGNORECASE):
        token = m.group(0).strip().strip(".,;:()[]{}\"'`")
        if token and len(token) <= 255:
            out.append(token)
    return out


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
    # Prefer non-stream for simpler parsing; add explicit system guard for JSON-only output
    payload: Dict[str, Any] = {
        "model": JINA_DEEP_RESEARCH_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Return ONLY a JSON array of objects with keys 'company_name' (string) and 'domain_url' (string). "
                    "The 'domain_url' MUST be an official corporate website. Exclude aggregators/directories (LinkedIn, ZoomInfo, Crunchbase). "
                    "No prose or commentary outside of the JSON array."
                ),
            },
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
            log.warning("[jina_dr] response_preview=%s", preview)
            # Log entire response body (no truncation) as requested
            log.warning(
                "[jina_dr] response_body=%s",
                (_json.dumps(data) if isinstance(data, dict) else str(data)),
            )
        except Exception:
            pass
        # Candidate URLs come ONLY from assistant content (JSON or regex/table fallback).
        # We intentionally exclude visitedURLs/readURLs to avoid aggregator/reference links.
        urls: List[str] = []
        # Parse assistant content: prefer strict JSON (even when wrapped in ```json fences);
        # fallback to regex URL extraction only when JSON parse fails.
        parsed_content_urls: List[str] = []
        try:
            content = None
            ch = (data.get("choices") or [{}])[0]
            msg = ch.get("message") or {}
            content = msg.get("content")
            if isinstance(content, str):
                try:
                    text = content.strip()
                    # Attempt to extract JSON array inside fenced block first
                    m = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", text, re.IGNORECASE)
                    json_text = m.group(1) if m else None
                    if not json_text:
                        # Try direct parse; if fails, try to locate first plausible array substring
                        try:
                            parsed = json.loads(text)
                            if isinstance(parsed, list):
                                json_text = text
                        except Exception:
                            m2 = re.search(r"\[[\s\S]*\]", text)
                            json_text = m2.group(0) if m2 else None
                    if json_text:
                        parsed = json.loads(json_text)
                        if isinstance(parsed, list):
                            for item in parsed:
                                if isinstance(item, dict):
                                    du = item.get("domain_url") or item.get("domain") or item.get("url")
                                    if isinstance(du, str) and du.strip():
                                        parsed_content_urls.append(du.strip())
                    else:
                        # No JSON array found — try Markdown table and bare domains, then regex URLs
                        parsed_content_urls.extend(_extract_domains_from_markdown_table(text))
                        parsed_content_urls.extend(_extract_bare_domains(text))
                        parsed_content_urls.extend(_extract_urls_from_text(text))
                except Exception:
                    # Non-JSON content — try Markdown table and bare domains, then regex URLs
                    parsed_content_urls.extend(_extract_domains_from_markdown_table(content))
                    parsed_content_urls.extend(_extract_bare_domains(content))
                    parsed_content_urls.extend(_extract_urls_from_text(content))
        except Exception:
            pass
        # Use only assistant content for discovery
        if parsed_content_urls:
            urls = list(parsed_content_urls)
        # Normalize → filter bad hostnames → dedupe → sort by rough "corporate-likeness"
        def _is_bad(host: str) -> bool:
            try:
                host_l = (host or "").lower()
                for pat in (JINA_DEEP_RESEARCH_BAD_HOSTNAMES or []):
                    p = (pat or "").lower()
                    if not p:
                        continue
                    if p.startswith("*."):
                        suf = p[1:]
                        if host_l.endswith(suf):
                            return True
                    elif host_l == p or host_l.endswith("." + p):
                        return True
                return False
            except Exception:
                return False

        hosts: List[str] = []
        seen = set()
        for u in urls:
            # Accept either full URLs (http...) or bare domains
            h = _norm_host(u)
            if not h:
                # _norm_host rejects bare domains without scheme; treat u as domain candidate
                try:
                    # Add https:// scheme to normalize
                    h = _norm_host("https://" + str(u))
                except Exception:
                    h = None
            if not h or h in seen:
                continue
            if _is_bad(h):
                continue
            seen.add(h)
            hosts.append(h)

        def _score(h: str) -> int:
            try:
                parts = h.split(".")
                tld = parts[-1] if len(parts) >= 2 else ""
                score = 0
                if len(parts) == 2:  # brand.tld
                    score += 2
                if tld in {"com", "co", "net", "sg", "io", "ai"}:
                    score += 1
                return score
            except Exception:
                return 0
        hosts.sort(key=_score, reverse=True)

        # Optional lightweight homepage heuristic check using Jina MCP
        if ENABLE_JINA_DEEP_RESEARCH_HEURISTICS and hosts:
            try:
                from src.jina_reader import read_url as _jina_read  # lazy import to avoid circular deps
                checked = []
                limit = max(1, int(JINA_DEEP_RESEARCH_HEURISTIC_CHECK_MAX))
                for h in hosts[:limit]:
                    try:
                        url = f"https://{h}"
                        text = _jina_read(url, timeout=float(JINA_DEEP_RESEARCH_HEURISTIC_TIMEOUT_S)) or ""
                        text_l = text.lower()
                        brand = h.split(".")[0].lower()
                        ok_brand = (brand in text_l)
                        ok_about = ("/about" in text_l) or (" about " in text_l)
                        ok_contact = ("/contact" in text_l) or (" contact " in text_l)
                        score_bump = 0
                        if ok_brand:
                            score_bump += 2
                        if ok_about:
                            score_bump += 1
                        if ok_contact:
                            score_bump += 1
                        checked.append((h, score_bump))
                    except Exception:
                        checked.append((h, 0))
                # Reorder top candidates by heuristic bump
                score_map = {h: bump for (h, bump) in checked}
                hosts.sort(key=lambda hh: (_score(hh), score_map.get(hh, 0)), reverse=True)
            except Exception:
                pass
        _obs_bump(run_id=0, tenant_id=0, vendor="jina_deep_research", calls=1)  # run/tenant resolved by caller typically
        try:
            # Log discovered domains explicitly (as URLs for readability)
            url_list = [f"https://{h}" for h in hosts[:50]]
            log.warning("[jina_dr] parsed_json_items=%s", len(parsed_content_urls) if parsed_content_urls else 0)
            log.warning("[jina_dr] discovered_domains=%s", url_list)
        except Exception:
            pass
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


__all__ = [
    "deep_research_query",
    "deep_research_contacts",
]


def deep_research_contacts(company_name: str, domain: str, *, timeout_s: float | None = None) -> Dict[str, Any]:
    """Query Jina Deep Research to find a decision-maker contact for IT adoption.

    Uses the exact user-provided educational prompt, substituting company and domain.
    Returns a best-effort structured dict with keys:
      - content: raw assistant content (string)
      - visitedURLs/readURLs: arrays from the API when present
      - source: 'jina_deep_research'

    Parsing of contacts is handled by the caller to keep this helper focused on transport.
    """
    endpoint = JINA_DEEP_RESEARCH_API_URL
    hdrs = _auth_header()
    if not hdrs.get("Authorization"):
        return {"content": "", "visitedURLs": [], "readURLs": [], "source": "jina_deep_research"}
    prompt = (
        "For eductional purpose, I want to teach a class about business development."
        f"Help me find the contact person for {company_name} and {domain}. "
        "who is the decision maker for adopting IT solutions"
    )
    payload: Dict[str, Any] = {
        "model": JINA_DEEP_RESEARCH_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "reasoning_effort": JINA_DEEP_RESEARCH_REASONING_EFFORT,
        "max_returned_urls": str(JINA_DEEP_RESEARCH_SUMMARY_MAX_URLS),
        "bad_hostnames": list(JINA_DEEP_RESEARCH_BAD_HOSTNAMES or []),
    }
    try:
        log.warning("[jina_dr] contacts_call name=%s domain=%s", company_name, domain)
        log.warning("[jina_dr] contacts_prompt=<<%s>>", prompt)
    except Exception:
        pass
    try:
        r = requests.post(
            endpoint,
            headers=hdrs,
            data=json.dumps(payload),
            timeout=float(timeout_s or JINA_DEEP_RESEARCH_TIMEOUT_S),
        )
        log.warning("[jina_dr] contacts_response_status=%s", r.status_code)
        r.raise_for_status()
        data = r.json()
        try:
            import json as _json
            # Log full contacts response body (no truncation)
            log.warning(
                "[jina_dr] contacts_response_body=%s",
                (_json.dumps(data) if isinstance(data, dict) else str(data)),
            )
        except Exception:
            pass
        content = None
        try:
            ch = (data.get("choices") or [{}])[0]
            msg = ch.get("message") or {}
            content = msg.get("content")
        except Exception:
            content = None
        visited = []
        read = []
        for key in ("visitedURLs", "readURLs"):
            try:
                arr = data.get(key) or []
                if isinstance(arr, list):
                    if key == "visitedURLs":
                        visited = [str(x) for x in arr if isinstance(x, str)]
                    else:
                        read = [str(x) for x in arr if isinstance(x, str)]
            except Exception:
                pass
        _obs_bump(run_id=0, tenant_id=0, vendor="jina_deep_research", calls=1)
        try:
            clen = len(content or "")
            log.warning(
                "[jina_dr] contacts_parsed content_len=%s visited=%s read=%s",
                clen,
                len(visited),
                len(read),
            )
        except Exception:
            pass
        return {
            "content": (content or ""),
            "visitedURLs": visited,
            "readURLs": read,
            "source": "jina_deep_research",
        }
    except Exception as exc:
        _obs_bump(run_id=0, tenant_id=0, vendor="jina_deep_research", calls=1, errors=1)
        log.warning("deep_research_contacts failed: %s", exc)
        return {"content": "", "visitedURLs": [], "readURLs": [], "source": "jina_deep_research"}
