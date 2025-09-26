import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from psycopg2.extras import Json

from src.crawler import crawl_site
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
    """For each seed, pick top domain (via Tavily if needed), crawl key sections, extract fast facts.

    - Robots.txt aware deterministic crawl with a small page budget
    - Extract fast facts: industry guess, size band guess, geo hints, buyer titles (from value props), integrations mentions
    - Confidence requires ≥2 agreeing sources (about + careers etc.) to be considered strong
    """
    cards: List[ResolverCard] = []
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
        try:
            summary = await crawl_site(url, max_pages=4)
        except Exception as e:
            log.info("crawl failed for %s: %s", url, e)
            continue
        sig = summary.get("signals", {})
        # Heuristic extraction for fast facts; optionally refined later by LLM
        industry_guess = "b2b" if "solutions" in (sig.get("title") or "").lower() else None
        size_guess = summary.get("company_size_guess") or None
        geo_guess = None
        if (summary.get("title") or "").lower().find("singapore") >= 0:
            geo_guess = "Singapore"
        buyer_titles = []
        for vp in (sig.get("value_props") or [])[:8]:
            if re.search(r"(ops|revenue|marketing|sales|growth|hr|talent|data)", vp, re.I):
                buyer_titles.append(vp)
        buyer_titles = buyer_titles[:5]
        integrations = []
        for ps in (sig.get("products_services") or [])[:20]:
            if re.search(r"integrations?", ps, re.I):
                integrations.append(ps)

        # Confidence threshold: require at least two agreeing sources for a strong card
        sources_ok = int(bool(sig.get("has_careers_page"))) + int(bool(sig.get("has_case_studies")))
        card_conf = conf
        if sources_ok >= 2 and conf == "high":
            card_conf = "high"
        elif sources_ok >= 1 and conf in ("high", "medium"):
            card_conf = "medium"
        else:
            card_conf = "low"
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
    """Deterministic crawl + light extraction → evidence rows.
    Quality gates: robots compliance, page caps, triangulation.
    """
    url = f"https://{_norm_domain(domain)}"
    try:
        summary = await crawl_site(url, max_pages=6)
    except Exception as e:
        log.info("crawl_site failed: %s", e)
        return 0
    sig = summary.get("signals", {})
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
    # Firmographics
    if sig.get("hiring", {}).get("open_roles"):
        recs.append({
            "signal_key": "hiring_open_roles",
            "value": sig.get("hiring"),
            "confidence": 0.6,
            "why": "Detected careers/hiring cues",
        })
    if sig.get("tech"):
        recs.append({
            "signal_key": "tech_stack",
            "value": sig.get("tech"),
            "confidence": 0.5,
            "why": "Found vendor script tags",
        })
    if sig.get("pricing"):
        recs.append({
            "signal_key": "pricing_page",
            "value": sig.get("pricing")[:5],
            "confidence": 0.5,
            "why": "Pricing list items found",
        })
    # Triangulation bump
    appears = int(bool(sig.get("has_careers_page"))) + int(bool(sig.get("has_case_studies"))) + int(bool(sig.get("has_testimonials")))
    for r in recs:
        if appears >= 2:
            r["confidence"] = min(0.95, float(r.get("confidence") or 0.5) + 0.25)
    # Only persist when we have a concrete company_id
    if company_id is None:
        return 0
    return persist_evidence_records(tenant_id, company_id, recs, source="crawler")


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
        try:
            cur.execute(
                """
                INSERT INTO icp_evidence(tenant_id, company_id, signal_key, value, source)
                VALUES (%s,%s,'ssic',%s,'acra')
                """,
                (tenant_id, company_id, Json({
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
