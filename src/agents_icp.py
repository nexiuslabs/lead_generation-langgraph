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
import html as html_lib

from langchain_openai import ChatOpenAI
import logging
import requests
from urllib.parse import quote, urlparse, urljoin, parse_qs, unquote
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
# Strict DuckDuckGo-only mode: do not aggregate from other engines
DDG_HTML_ENDPOINT = "https://html.duckduckgo.com/html"
DDG_HTML_GET = "https://duckduckgo.com/html/"
DDG_LITE_GET = "https://lite.duckduckgo.com/lite/"

# Optional seed hints passed in by the caller (confirm flow) to improve queries
SEED_HINTS: List[str] = []

def set_seed_hints(hints: List[str] | None):
    global SEED_HINTS
    try:
        SEED_HINTS = [h.strip().lower() for h in (hints or []) if isinstance(h, str) and h.strip()][:10]
    except Exception:
        SEED_HINTS = []

from src.icp_pipeline import collect_evidence_for_domain
from src.crawler import crawl_site


# Core state keys (subset)
# tenant_id, seeds[], icp_profile, discovery_candidates[], research_artifacts[], evidence[], scores[], queue[], errors[]


def _uniq(seq: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for s in seq:
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _ddg_search_domains(query: str, max_results: int = 25, country: str | None = None, lang: str | None = None) -> List[str]:
    """Perform a DuckDuckGo HTML search and extract external result domains.

    - Uses only DuckDuckGo (`html.duckduckgo.com/html`).
    - Parses redirect links (`/l/?uddg=...`) and extracts the target domain.
    - Filters obvious search/CDN/wiki hosts.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116 Safari/537.36",
        "Accept-Language": (lang or "en-US,en;q=0.9"),
    }
    def _extract_domains_from_html(raw_html: str) -> List[str]:
        text = raw_html or ""
        text = html_lib.unescape(text)
        found: List[str] = []
        # 1) Extract via explicit DDG redirect pattern `/l/?uddg=` (covers html + standard pages)
        for m in re.findall(r"/l/\?[^\s\"']*uddg=([^&\"']+)", text):
            try:
                target_url = unquote(m)
                host = (urlparse(target_url).netloc or "").lower()
                if host:
                    found.append(host)
            except Exception:
                continue
        # 2) Fallback: collect anchor hrefs and take external hosts directly
        for href in re.findall(r'href=[\"\']([^\"\']+)[\"\']', text):
            try:
                href = href.strip()
                if href.startswith("/"):
                    href_abs = urljoin("https://duckduckgo.com", href)
                else:
                    href_abs = href
                u = urlparse(href_abs)
                host = (u.netloc or "").lower()
                if not host:
                    continue
                if (host.endswith("duckduckgo.com") or host.endswith("r.duckduckgo.com")) and u.path.startswith("/l/"):
                    q = parse_qs(u.query)
                    target = q.get("uddg", [None])[0]
                    if target:
                        target_url = unquote(str(target))
                        host = (urlparse(target_url).netloc or "").lower()
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
            out.append(h)
            if len(out) >= max_results:
                break
        return out

    domains: List[str] = []
    # Attempt 1: html.duckduckgo.com via POST (classic HTML endpoint)
    try:
        r = requests.post(DDG_HTML_ENDPOINT, data={"q": query}, headers=headers, timeout=15)
        r.raise_for_status()
        domains = _extract_domains_from_html(r.text)
        if domains:
            log.info("[ddg] POST parsed domains=%d for query=%s", len(domains), query)
            return domains
    except Exception as e:
        log.info("[ddg] POST search failed: %s", e)

    # Attempt 2: duckduckgo.com/html via GET
    try:
        url = DDG_HTML_GET + "?q=" + quote(query)
        r2 = requests.get(url, headers=headers, timeout=15)
        r2.raise_for_status()
        domains = _extract_domains_from_html(r2.text)
        if domains:
            log.info("[ddg] GET parsed domains=%d for query=%s", len(domains), query)
            return domains
    except Exception as e2:
        log.info("[ddg] GET fallback failed: %s", e2)

    # Attempt 3: lite.duckduckgo.com page which often bypasses heavy markup
    try:
        url = DDG_LITE_GET + "?q=" + quote(query)
        r3 = requests.get(url, headers=headers, timeout=15)
        r3.raise_for_status()
        domains = _extract_domains_from_html(r3.text)
        if domains:
            log.info("[ddg] LITE parsed domains=%d for query=%s", len(domains), query)
            return domains
    except Exception as e3:
        log.info("[ddg] LITE fallback failed: %s", e3)

    return _uniq(domains)


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
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You write 3 web search queries to find B2B companies matching a micro‑ICP. Keep concise. Cite which ICP fields influenced each query. Return as lines: query — because ..."),
        ("human", "Industries: {inds}\nSignals: {sigs}\nTitles: {titles}")
    ])
    msgs = prompt.format_messages(inds=inds, sigs=sigs, titles=titles)
    log.info("[plan] building queries from ICP; inds=%s sigs=%s titles=%s", inds, sigs, titles)
    text = llm.invoke(msgs).content or ""
    # Extract queries (strip trailing explanations) and sanitize numbering like '1. '
    raw_queries = [ln.split(" — ", 1)[0] for ln in text.splitlines() if ln.strip()]
    queries: List[str] = []
    for q in raw_queries:
        q = re.sub(r"^\s*\d+\.\s*", "", q or "").strip()
        # Strip surrounding quotes that tend to reduce recall on DDG
        if (q.startswith('"') and q.endswith('"')) or (q.startswith("'") and q.endswith("'")):
            q = q[1:-1].strip()
        if q:
            queries.append(q)
    domains: List[str] = []
    for q in queries[:6]:
        try:
            log.info("[plan] ddg-only query: %s", q)
            for dom in _ddg_search_domains(q, max_results=25):
                domains.append(dom)
        except Exception as e:
            log.info("[plan] ddg fail: %s", e)
            continue
    uniq = _uniq(domains)
    if not uniq:
        # Deterministic fallback query pack to improve recall while staying DDG-only
        fb: List[str] = [
            "foodservice distributors",
            "broadline foodservice distributors",
            "restaurant supply distributors",
            "food & beverage wholesale distributors",
        ]
        # Region hint: if any seed ends with .sg, prefer .sg-specific queries
        try:
            seeds = list(SEED_HINTS)
        except Exception:
            seeds = []
        if any(str(s).endswith('.sg') for s in seeds):
            fb = [
                "site:.sg foodservice distributor",
                "site:.sg food & beverage distributor",
                "site:.sg fmcg distributor",
            ] + fb
        for q in fb:
            try:
                log.info("[plan] ddg-only fallback query: %s", q)
                for dom in _ddg_search_domains(q, max_results=25):
                    domains.append(dom)
            except Exception as e:
                log.info("[plan] ddg fail: %s", e)
                continue
        uniq = _uniq(domains)
    log.info("[plan] ddg domains found=%d (uniq=%d)", len(domains), len(uniq))
    # Optional Jina Reader fetch for quick homepage snippets of first few domains
    jina_snips: Dict[str, str] = {}
    for d in uniq[:10]:
        try:
            url = f"https://{d}"
            reader = f"https://r.jina.ai/http://{d}"
            log.info("[jina] GET %s", reader)
            r = requests.get(reader, timeout=10)
            txt = (r.text or "")[:8000]
            # Clean noisy prefixes often present in r.jina output
            lines = [ln.strip() for ln in (txt or "").splitlines() if ln.strip()]
            filtered = [
                ln for ln in lines
                if not re.match(r"^(Title:|URL Source:|Published Time:|Markdown Content:|Warning:)", ln, flags=re.I)
            ]
            clean = " ".join(filtered) if filtered else " ".join((txt or "").split())
            snip = clean[:400]
            jina_snips[d] = snip
            log.info("[jina] ok len=%d domain=%s", len(txt), d)
        except Exception as e:
            log.info("[jina] fail domain=%s err=%s", d, e)
            continue
    state["discovery_candidates"] = uniq[:50]
    if jina_snips:
        state["jina_snippets"] = jina_snips
    return state


async def mini_crawl_worker(state: Dict[str, Any]) -> Dict[str, Any]:
    """Tool-based: crawl a small bundle per candidate domain and attach evidence payloads.

    Inputs: state['tenant_id'], state['discovery_candidates']
    Outputs: state['evidence'] = [{domain, summary, signals?}, ...]
    """
    tenant_id = int(state.get("tenant_id") or 0)
    out: List[Dict[str, Any]] = []
    cand = (state.get("discovery_candidates") or [])
    log.info("[mini] crawl start count=%d", len(cand[:10]))
    for dom in (state.get("discovery_candidates") or [])[:10]:
        url = f"https://{dom}"
        try:
            summary = await crawl_site(url, max_pages=4)
            out.append({"domain": dom, "summary": summary})
        except Exception as e:
            log.info("[mini] crawl fail domain=%s err=%s", dom, e)
            continue
    log.info("[mini] crawl done ok=%d", len(out))
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
    """Convenience helper: discovery → mini-crawl → extract → score; returns Top‑10 with 'why'.

    Returns: [{domain, score, bucket, why}...]
    """
    try:
        # 1) plan
        s = {"icp_profile": dict(icp_profile or {})}
        s = discovery_planner(s)
        cand: List[str] = s.get("discovery_candidates") or []
        jina_snips: Dict[str, str] = s.get("jina_snippets") or {}
        if not cand:
            return []
        # 2) mini crawl (top 10)
        ev_list: List[Dict[str, Any]] = []
        for d in cand[:10]:
            url = f"https://{d}"
            try:
                t = int(tenant_id or 0)
            except Exception:
                t = 0
            try:
                log.info("[mini] crawl domain=%s", d)
                summ = asyncio.run(crawl_site(url, max_pages=4))
                ev_list.append({"domain": d, "summary": summ})
            except Exception as e:
                log.info("[mini] crawl fail domain=%s err=%s", d, e)
                continue
        if not ev_list:
            return []
        # 3) extract
        st2 = {"evidence": ev_list}
        st2 = evidence_extractor(st2)
        # 4) score
        st3 = scoring_and_gating(st2)
        scores = st3.get("scores") or []
        # Build reason lines
        top = []
        for srow in scores:
            why_bits = []
            r = srow.get("reason") or {}
            ints = r.get("integrations") or []
            if ints:
                why_bits.append(f"integrations: {', '.join(ints[:3])}")
            titles = r.get("titles") or []
            if titles:
                why_bits.append(f"titles: {', '.join(titles[:3])}")
            if r.get("pricing"):
                why_bits.append("pricing page found")
            if r.get("case_studies"):
                why_bits.append("case studies")
            dom = srow.get("domain")
            # Surface Jina snippet (short)
            snip_raw = jina_snips.get(dom) if isinstance(jina_snips, dict) else None
            try:
                snip = (" ".join((snip_raw or "").split()))[:180]
            except Exception:
                snip = None
            top.append({
                "domain": srow.get("domain"),
                "score": srow.get("score"),
                "bucket": srow.get("bucket"),
                "why": "; ".join(why_bits) if why_bits else "signal match",
                "snippet": snip,
            })
        # Sort by score desc and return top 10
        top = sorted(top, key=lambda x: int(x.get("score") or 0), reverse=True)[:10]
        log.info("[confirm] agent top10 count=%d", len(top))
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
        titles = [s.strip() for s in (icp.get("buyer_titles") or []) if isinstance(s, str) and s.strip()]
        sigs = [s.strip() for s in ((icp.get("integrations") or []) + (icp.get("triggers") or [])) if isinstance(s, str) and s.strip()]
        queries: List[str] = []
        if inds:
            queries.append(f"B2B companies in {', '.join(inds[:2])}")
        if sigs:
            queries.append(f"B2B solutions {', '.join(sigs[:2])}")
        if titles:
            queries.append(f"companies hiring {', '.join(titles[:2])}")
        if not queries:
            queries = ["B2B SaaS companies", "B2B platforms", "enterprise software vendors"]

        # Discover domains strictly via DDG
        domains: List[str] = []
        for q in queries[:3]:
            for dom in _ddg_search_domains(q, max_results=25):
                domains.append(dom)
        uniq = _uniq(domains)[:20]
        if not uniq:
            return []

        # Fetch snippets via Jina Reader and score heuristically
        out: List[Dict[str, Any]] = []
        for d in uniq[:10]:
            try:
                reader = f"https://r.jina.ai/http://{d}"
                r = requests.get(reader, timeout=10)
                txt = (r.text or "")[:8000]
                clean = " ".join((txt or "").split())
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
