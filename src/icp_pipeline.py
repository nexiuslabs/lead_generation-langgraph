import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from psycopg2.extras import Json

from src.jina_reader import read_url as jina_read
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from src.database import get_conn
from src.icp_intake import fuzzy_map_seed_to_acra
from src.database import get_conn
from src.settings import TAVILY_API_KEY

try:
    from tavily import TavilyClient  # type: ignore
except Exception:  # pragma: no cover
    TavilyClient = None  # type: ignore

log = logging.getLogger("icp_pipeline")


# -----------------------------
# Small helpers
# -----------------------------

def _norm_domain(domain: Optional[str]) -> Optional[str]:
    if not domain:
        return None
    d = (domain or "").strip().lower()
    d = d.replace("http://", "").replace("https://", "")
    if d.startswith("www."):
        d = d[4:]
    for sep in ["/", "?", "#"]:
        if sep in d:
            d = d.split(sep, 1)[0]
    return d or None


def _dedupe_seeds(seeds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for s in seeds or []:
        name = (s.get("seed_name") or "").strip()
        dom = _norm_domain(s.get("domain"))
        if not name:
            continue
        key = (name.lower(), dom or "")
        if key in seen:
            continue
        seen.add(key)
        out.append({"seed_name": name[:120], "domain": dom})
    return out


def _is_vendor_or_directory(url: str) -> bool:
    u = (url or "").lower()
    bad = [
        "crunchbase.com",
        "linkedin.com/company",
        "facebook.com",
        "twitter.com",
        "x.com",
        "instagram.com",
        "g2.com",
        "capterra.com",
        "builtwith.com",
        "stackshare.io",
        "glassdoor",
    ]
    return any(b in u for b in bad)


# -----------------------------
# Step 2: Seed Normalization
# -----------------------------

def find_seed_domains_via_tavily(company: str, geo_hint: Optional[str] = None) -> List[Dict[str, Any]]:
    """Use Tavily to propose candidate domains for a seed company with evidence and confidence.

    Returns a list of {domain, url, confidence, why} sorted by confidence.
    """
    results: List[Dict[str, Any]] = []
    if not TAVILY_API_KEY or TavilyClient is None or not company:
        return results
    try:
        client = TavilyClient(TAVILY_API_KEY)
        queries = [f"{company} official site"]
        if geo_hint:
            queries.append(f"{company} official site {geo_hint}")
            queries.append(f"{company} {geo_hint} careers press")
        else:
            queries.append(f"{company} careers press integrations")
        seen = set()
        for q in queries:
            resp = client.search(q)
            for item in (resp.get("results") or []):
                url = (item.get("url") or "").strip()
                if not url or _is_vendor_or_directory(url):
                    continue
                dom = _norm_domain(url)
                if not dom:
                    continue
                if dom in seen:
                    continue
                seen.add(dom)
                title = (item.get("title") or "").strip()
                snippet = (item.get("content") or "").strip()
                why = f"{title[:80]} – {snippet[:140]}" if (title or snippet) else "from search"
                results.append({"domain": dom, "url": url, "confidence": "low", "why": why})
    except Exception as e:
        log.info("tavily search failed: %s", e)
        return []

    return results


# -----------------------------
# Step 3–4: Resolver + Evidence
# -----------------------------

@dataclass
class ResolverCard:
    seed_name: str
    domain: str
    confidence: str
    fast_facts: Dict[str, Any]
    why: str


async def build_resolver_cards(seeds: List[Dict[str, Any]]) -> List[ResolverCard]:
    """For each seed, pick top domain (via Tavily if needed), read via r.jina, extract fast facts via LLM.

    - No deterministic crawler; uses Jina Reader homepage text
    - Extract fast facts: industry guess, size band guess, geo hints, buyer titles, integrations mentions
    - Confidence derived from text richness combined with search confidence
    """
    cards: List[ResolverCard] = []
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract quick company fast-facts from homepage text. Return JSON with keys: industry_guess, size_band_guess, geo_guess, buyer_titles (array), integrations_mentions (array). Keep values concise."),
        ("human", "Seed: {name}\nDomain: {domain}\n\n{body}"),
    ])

    class FastFacts(BaseModel):
        industry_guess: Optional[str] = None
        size_band_guess: Optional[str] = None
        geo_guess: Optional[str] = None
        buyer_titles: Optional[List[str]] = None
        integrations_mentions: Optional[List[str]] = None

    structured = llm.with_structured_output(FastFacts)
    for s in _dedupe_seeds(seeds):
        name = (s.get("seed_name") or "").strip()
        dom = _norm_domain(s.get("domain"))
        cand = dom
        why = "provided"
        conf = "medium"
        if not cand:
            cands = find_seed_domains_via_tavily(name)
            if cands:
                top = cands[0]
                cand = top.get("domain")
                conf = top.get("confidence") or "low"
                why = top.get("why") or "search"
        if not cand:
            continue
        url = f"https://{cand}"
        body = jina_read(url, timeout=12) or ""
        if not body:
            log.info("jina empty for %s", url)
            continue
        # LLM fast-facts from homepage text
        try:
            msgs = prompt.format_messages(name=name, domain=cand, body=body[:4000])
            out = structured.invoke(msgs)
            data = out.model_dump() if hasattr(out, "model_dump") else (
                out.dict() if hasattr(out, "dict") else {}
            )
            industry_guess = (data.get("industry_guess") or None)
            size_guess = (data.get("size_band_guess") or None)
            geo_guess = (data.get("geo_guess") or None)
            buyer_titles = (data.get("buyer_titles") or []) or []
            integrations = (data.get("integrations_mentions") or []) or []
        except Exception as e:
            log.info("fast-facts extraction failed for %s: %s", url, e)
            industry_guess = None
            size_guess = None
            geo_guess = None
            buyer_titles = []
            integrations = []

        # Confidence: richer body + higher search confidence
        card_conf = conf or "low"
        if len(body) > 1200 and (industry_guess or buyer_titles or integrations):
            if card_conf == "low":
                card_conf = "medium"
        cards.append(
            ResolverCard(
                seed_name=name,
                domain=cand,
                confidence=card_conf,
                fast_facts={
                    "industry_guess": industry_guess,
                    "size_band_guess": size_guess,
                    "geo_guess": geo_guess,
                    "buyer_titles": buyer_titles,
                    "integrations_mentions": integrations,
                },
                why=why,
            )
        )
    return cards


def persist_evidence_records(
    tenant_id: int, company_id: Optional[int], records: List[Dict[str, Any]], source: str
) -> int:
    """Insert evidence rows. Each record: {signal_key, value, confidence, why}.
    Returns count written.
    """
    if not records:
        return 0
    written = 0
    with get_conn() as conn, conn.cursor() as cur:
        for r in records:
            key = (r.get("signal_key") or "").strip()
            val = r.get("value")
            if not key:
                continue
            try:
                cur.execute(
                    """
                    INSERT INTO icp_evidence(tenant_id, company_id, signal_key, value, source)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (tenant_id, company_id, key, Json({"data": val, "confidence": r.get("confidence"), "why": r.get("why")}), source),
                )
                written += 1
            except Exception as e:
                log.info("evidence insert failed: %s", e)
    return written


async def collect_evidence_for_domain(
    tenant_id: int, company_id: Optional[int], domain: str
) -> int:
    """Read homepage via r.jina and extract ICP evidence using LLM."""
    url = f"https://{_norm_domain(domain)}"
    body = jina_read(url, timeout=12) or ""
    if not body:
        return 0
    # Resolve a company_id by website_domain if not provided to satisfy NOT NULL constraints
    try:
        if company_id is None:
            apex = _norm_domain(domain)
            if apex:
                with get_conn() as conn, conn.cursor() as cur:
                    cur.execute("SELECT company_id FROM companies WHERE website_domain=%s LIMIT 1", (apex,))
                    row = cur.fetchone()
                    company_id = int(row[0]) if row and row[0] is not None else None
    except Exception:
        company_id = None
    recs: List[Dict[str, Any]] = []
    # LLM normalization of evidence (PRD19 §6 Evidence Extractor) from Jina body
    try:
        from src.agents_icp import evidence_extractor as _agent_evidence_extractor  # type: ignore
        st = {"evidence": [{"summary": body[:4000]}]}
        out = _agent_evidence_extractor(st)
        ev = (out.get("evidence") or [{}])[0]
        # Map normalized fields into recs where helpful
        if isinstance(ev.get("integrations"), list) and ev.get("integrations"):
            recs.append({
                "signal_key": "integrations",
                "value": ev.get("integrations"),
                "confidence": 0.6,
                "why": "LLM extraction (jina)",
            })
        if isinstance(ev.get("buyer_titles"), list) and ev.get("buyer_titles"):
            recs.append({
                "signal_key": "buyer_titles",
                "value": ev.get("buyer_titles"),
                "confidence": 0.6,
                "why": "LLM extraction (jina)",
            })
        if isinstance(ev.get("hiring_open_roles"), int) and ev.get("hiring_open_roles"):
            recs.append({
                "signal_key": "hiring_open_roles",
                "value": {"count": ev.get("hiring_open_roles")},
                "confidence": 0.5,
                "why": "LLM extraction (jina)",
            })
        if isinstance(ev.get("has_case_studies"), bool):
            recs.append({
                "signal_key": "has_case_studies",
                "value": ev.get("has_case_studies"),
                "confidence": 0.5,
                "why": "LLM extraction (jina)",
            })
        if isinstance(ev.get("has_pricing"), bool):
            recs.append({
                "signal_key": "has_pricing",
                "value": ev.get("has_pricing"),
                "confidence": 0.5,
                "why": "LLM extraction (jina)",
            })
    except Exception:
        pass
    # Only persist when we have a concrete company_id
    if company_id is None:
        return 0
    return persist_evidence_records(tenant_id, company_id, recs, source="jina_reader")


# -----------------------------
# Step 5: ACRA/SSIC anchoring
# -----------------------------

def acra_anchor_seed(
    tenant_id: int, seed_name: str, company_id: Optional[int]
) -> Optional[Dict[str, Any]]:
    with get_conn() as conn, conn.cursor() as cur:
        acra = fuzzy_map_seed_to_acra(cur, seed_name)
        if not acra:
            return None
        # Ensure we have a concrete company_id: prefer provided, else by UEN, else create minimal row
        cid = company_id
        try:
            if cid is None:
                uen = (acra.get("uen") or "").strip()
                nm = (acra.get("entity_name") or seed_name or "").strip()
                code = acra.get("primary_ssic_code")
                ind_code = str(code) if code is not None else None
                if uen:
                    # Upsert by UEN
                    cur.execute(
                        """
                        INSERT INTO companies(uen, name, industry_code, last_seen)
                        VALUES (%s, %s, %s, NOW())
                        ON CONFLICT (uen) DO UPDATE SET name=EXCLUDED.name, industry_code=COALESCE(EXCLUDED.industry_code, companies.industry_code), last_seen=NOW()
                        RETURNING company_id
                        """,
                        (uen, nm, ind_code),
                    )
                    row = cur.fetchone()
                    if row and row[0] is not None:
                        cid = int(row[0])
                else:
                    # Try exact name match first
                    cur.execute("SELECT company_id FROM companies WHERE LOWER(name)=LOWER(%s) LIMIT 1", (nm,))
                    r = cur.fetchone()
                    if r and r[0] is not None:
                        cid = int(r[0])
                    else:
                        cur.execute(
                            """
                            INSERT INTO companies(name, industry_code, last_seen)
                            VALUES (%s, %s, NOW())
                            RETURNING company_id
                            """,
                            (nm, ind_code),
                        )
                        rr = cur.fetchone()
                        if rr and rr[0] is not None:
                            cid = int(rr[0])
        except Exception as e:
            log.info("acra ensure company failed: %s", e)
        # Insert SSIC evidence only when we have a valid company_id to satisfy NOT NULL schema
        if cid is not None:
            try:
                cur.execute(
                    """
                    INSERT INTO icp_evidence(tenant_id, company_id, signal_key, value, source)
                    VALUES (%s,%s,'ssic',%s,'acra')
                    """,
                    (tenant_id, cid, Json({
                        "ssic": acra.get("primary_ssic_code"),
                        "uen": acra.get("uen"),
                        "matched_name": acra.get("entity_name"),
                    })),
                )
            except Exception as e:
                log.info("acra evidence insert failed: %s", e)
        return acra


# -----------------------------
# Step 6–8: Patterns → Candidates → Micro‑ICPs
# -----------------------------

def winner_profile(tenant_id: int) -> Dict[str, Any]:
    """Compute simple frequency stats from icp_evidence for seeds.
    Returns dict with top ssic, common stacks, median size guess etc.
    """
    out: Dict[str, Any] = {"ssic": [], "stacks": [], "signals": []}
    with get_conn() as conn, conn.cursor() as cur:
        try:
            cur.execute(
                """
                SELECT value->>'ssic', COUNT(*)
                FROM icp_evidence
                WHERE tenant_id=%s AND signal_key='ssic' AND value ? 'ssic'
                GROUP BY 1 ORDER BY 2 DESC LIMIT 5
                """,
                (tenant_id,),
            )
            out["ssic"] = [(r[0], int(r[1])) for r in cur.fetchall() or []]
        except Exception:
            pass
        try:
            cur.execute(
                """
                SELECT (value->'data'->'analytics') IS NOT NULL,
                       (value->'data'->'crm') IS NOT NULL,
                       COUNT(*)
                FROM icp_evidence
                WHERE tenant_id=%s AND signal_key='tech_stack'
                GROUP BY 1,2 ORDER BY 3 DESC
                """,
                (tenant_id,),
            )
            rows = cur.fetchall() or []
            if rows:
                out["stacks"] = [
                    {
                        "analytics": bool(r[0]),
                        "crm": bool(r[1]),
                        "count": int(r[2]),
                    }
                    for r in rows
                ]
        except Exception:
            pass
    return out


def candidate_lookalikes_from_patterns(tenant_id: int, profile: Dict[str, Any], limit: int = 25) -> List[Dict[str, Any]]:
    """Best-effort lookalikes using company table by SSIC or tech evidence density."""
    # Use icp_patterns MV if present; else fall back to companies table
    out: List[Dict[str, Any]] = []
    codes = [c for (c, _cnt) in (profile.get("ssic") or [])]
    with get_conn() as conn, conn.cursor() as cur:
        try:
            if codes:
                cur.execute(
                    """
                    SELECT company_id, name, uen, website_domain, industry_code
                    FROM companies
                    WHERE regexp_replace(industry_code::text, '\\D', '', 'g') = ANY(%s)
                    ORDER BY employees_est DESC NULLS LAST, name ASC
                    LIMIT %s
                    """,
                    (codes, int(limit)),
                )
                for r in cur.fetchall() or []:
                    out.append({
                        "id": int(r[0]),
                        "name": r[1],
                        "uen": r[2],
                        "domain": r[3],
                        "industry_code": r[4],
                    })
        except Exception:
            pass
    return out


def micro_icp_suggestions_from_profile(profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for code, cnt in (profile.get("ssic") or [])[:5]:
        items.append({
            "id": f"ssic:{code}",
            "title": f"SSIC {code}",
            "rationale": f"Shared by {cnt} winners",
            "evidence_count": int(cnt),
        })
    return items
