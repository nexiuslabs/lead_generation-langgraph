from __future__ import annotations

import asyncio
import json
import inspect
import logging
import os
import re
from typing import Any, Dict, List, Optional, TypedDict
try:  # Python 3.9/3.10 fallback
    from typing import Annotated  # type: ignore
except Exception:  # pragma: no cover
    from typing_extensions import Annotated  # type: ignore

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from uuid import uuid4
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from app.odoo_store import OdooStore
from psycopg2.extras import Json
from src.database import get_pg_pool, get_conn
from src.icp import _find_ssic_codes_by_terms, _select_acra_by_ssic_codes
try:
    # New helper for accurate ACRA totals
    from src.icp import _count_acra_by_ssic_codes  # type: ignore
except Exception:  # pragma: no cover
    _count_acra_by_ssic_codes = None  # type: ignore
from src.enrichment import enrich_company_with_tavily
# ICP Finder helpers (Feature 17)
try:
    from src.icp_intake import (
        save_icp_intake as _icp_save_intake,
        map_seeds_to_evidence as _icp_map_seeds,
        refresh_icp_patterns as _icp_refresh_patterns,
        generate_suggestions as _icp_generate_suggestions,
        store_intake_evidence as _icp_store_intake_evidence,
    )
    from src.icp_pipeline import (
        build_resolver_cards as _icp_build_resolver_cards,
        collect_evidence_for_domain as _icp_collect_evidence_for_domain,
        acra_anchor_seed as _icp_acra_anchor_seed,
        winner_profile as _icp_winner_profile,
        micro_icp_suggestions_from_profile as _icp_micro_suggestions,
    )
except Exception:  # pragma: no cover
    _icp_save_intake = None  # type: ignore
    _icp_map_seeds = None  # type: ignore
    _icp_refresh_patterns = None  # type: ignore
    _icp_generate_suggestions = None  # type: ignore
    _icp_build_resolver_cards = None  # type: ignore
    _icp_collect_evidence_for_domain = None  # type: ignore
    _icp_acra_anchor_seed = None  # type: ignore
    _icp_winner_profile = None  # type: ignore
    _icp_micro_suggestions = None  # type: ignore
from src.lead_scoring import lead_scoring_agent
from src.settings import ODOO_POSTGRES_DSN
try:
    from src.settings import ENABLE_ICP_INTAKE  # type: ignore
except Exception:  # pragma: no cover
    ENABLE_ICP_INTAKE = False  # type: ignore
try:
    from src.settings import ICP_WIZARD_FAST_START_ONLY  # type: ignore
except Exception:  # pragma: no cover
    ICP_WIZARD_FAST_START_ONLY = True  # type: ignore

# ---------- logging ----------
logger = logging.getLogger("presdr")
_level = os.getenv("LOG_LEVEL", "INFO").upper()
if not logger.handlers:
    h = logging.StreamHandler()
    fmt = logging.Formatter(
        "[%(levelname)s] %(asctime)s %(name)s :: %(message)s", "%H:%M:%S"
    )
    h.setFormatter(fmt)
    logger.addHandler(h)
logger.setLevel(_level)

# Session-boot token: used to gate explicit actions after server restarts
BOOT_TOKEN = os.getenv("LG_SERVER_BOOT_TOKEN") or str(uuid4())

# --- Helper: ensure required Fast-Start fields are present (asked and captured) ---
def _icp_required_fields_done(icp: dict, ask_counts: dict | None = None) -> bool:
    """Return True when all required Fast-Start fields have been captured.

    Required keys (no industry):
      - website_url
      - seeds_list (>=5)
      - lost_churned (present; can be empty if user skipped)
      - employees_min or employees_max
      - geos (list)
      - integrations_required (present; can be empty)
      - acv_usd (present; can be None)
      - cycle_weeks_min or cycle_weeks_max (present; can be None)
      - price_floor_usd (present; can be None)
      - champion_titles (present; can be empty)
      - triggers (present; can be empty)
    """
    icp = icp or {}
    def present(k: str) -> bool:
        return k in icp
    if not icp.get("website_url"):
        return False
    seeds = icp.get("seeds_list") or []
    if not isinstance(seeds, list) or len(seeds) < 5:
        return False
    if not present("lost_churned"):
        return False
    if not (present("employees_min") or present("employees_max")):
        return False
    if not (isinstance(icp.get("geos"), list)):
        return False
    for k in ("integrations_required", "acv_usd", "price_floor_usd"):
        if not present(k):
            return False
    if not (present("cycle_weeks_min") or present("cycle_weeks_max")):
        return False
    if not present("champion_titles"):
        return False
    if not present("triggers"):
        return False
    return True

# ---------- DB table names (env-overridable) ----------
COMPANY_TABLE = os.getenv("COMPANY_TABLE", "companies")
LEAD_SCORES_TABLE = os.getenv("LEAD_SCORES_TABLE", "lead_scores")


class PreSDRState(TypedDict, total=False):
    # Use LangGraph message reducer so message updates append instead of replace
    messages: Annotated[List[BaseMessage], add_messages]
    icp: Dict[str, Any]
    candidates: List[Dict[str, Any]]
    results: List[Dict[str, Any]]


# ---------- ICP persistence helpers ----------
async def _resolve_tenant_id_for_write(state: dict) -> Optional[int]:
    # Prefer explicit tenant in state
    try:
        v = state.get("tenant_id") if isinstance(state, dict) else None
        if v is not None:
            return int(v)
    except Exception:
        pass
    # Default tenant for server-side jobs (env)
    try:
        v = os.getenv("DEFAULT_TENANT_ID")
        if v and v.isdigit():
            return int(v)
    except Exception:
        pass
    # Infer from ODOO_POSTGRES_DSN via odoo_connections
    try:
        inferred_db = None
        if ODOO_POSTGRES_DSN:
            from urllib.parse import urlparse
            u = urlparse(ODOO_POSTGRES_DSN)
            inferred_db = (u.path or "/").lstrip("/") or None
        if inferred_db:
            with get_conn() as _c, _c.cursor() as _cur:
                _cur.execute(
                    "SELECT tenant_id FROM odoo_connections WHERE (db_name=%s OR db_name=%s) AND active=TRUE LIMIT 1",
                    (inferred_db, ODOO_POSTGRES_DSN),
                )
                _row = _cur.fetchone()
                if _row:
                    return int(_row[0])
    except Exception:
        pass
    # Last resort: first active
    try:
        with get_conn() as _c, _c.cursor() as _cur:
            _cur.execute("SELECT tenant_id FROM odoo_connections WHERE active=TRUE LIMIT 1")
            _row = _cur.fetchone()
            if _row:
                return int(_row[0])
    except Exception:
        pass
    return None


def _icp_payload_from_state_icp(icp: dict) -> dict:
    # Normalize chat state into orchestrator-friendly payload
    inds = []
    if isinstance(icp.get("industries"), list):
        inds = [str(s).strip() for s in icp.get("industries") if isinstance(s, str) and s.strip()]
    emp_min = icp.get("employees_min")
    emp_max = icp.get("employees_max")
    y_min = icp.get("year_min")
    y_max = icp.get("year_max")
    payload: dict[str, Any] = {}
    if inds:
        payload["industries"] = inds
    if isinstance(emp_min, int) or isinstance(emp_max, int):
        payload["employee_range"] = {"min": emp_min if isinstance(emp_min, int) else None, "max": emp_max if isinstance(emp_max, int) else None}
    if isinstance(y_min, int) or isinstance(y_max, int):
        payload["incorporation_year"] = {"min": y_min if isinstance(y_min, int) else None, "max": y_max if isinstance(y_max, int) else None}
    if isinstance(icp.get("geos"), list):
        geos = [str(s).strip() for s in icp.get("geos") if isinstance(s, str) and s.strip()]
        if geos:
            payload["geos"] = geos
    if isinstance(icp.get("signals"), list):
        sigs = [str(s).strip() for s in icp.get("signals") if isinstance(s, str) and s.strip()]
        if sigs:
            payload["signals"] = sigs
    return payload


def _save_icp_rule_sync(tid: int, payload: dict, name: str = "Default ICP") -> None:
    # Insert a new ICP rule row for this tenant; rely on RLS via GUC
    with get_conn() as conn, conn.cursor() as cur:
        try:
            cur.execute("SELECT set_config('request.tenant_id', %s, true)", (str(tid),))
        except Exception:
            pass
        cur.execute(
            """
            INSERT INTO icp_rules(tenant_id, name, payload)
            VALUES (%s, %s, %s)
            """,
            (tid, name, Json(payload)),
        )


def _resolve_tenant_id_for_write_sync(state: dict) -> Optional[int]:
    try:
        v = state.get("tenant_id") if isinstance(state, dict) else None
        if v is not None:
            return int(v)
    except Exception:
        pass
    try:
        v = os.getenv("DEFAULT_TENANT_ID")
        if v and v.isdigit():
            return int(v)
    except Exception:
        pass
    try:
        inferred_db = None
        if ODOO_POSTGRES_DSN:
            from urllib.parse import urlparse
            u = urlparse(ODOO_POSTGRES_DSN)
            inferred_db = (u.path or "/") .lstrip("/") or None
        if inferred_db:
            with get_conn() as _c, _c.cursor() as _cur:
                _cur.execute(
                    "SELECT tenant_id FROM odoo_connections WHERE (db_name=%s OR db_name=%s) AND active=TRUE LIMIT 1",
                    (inferred_db, ODOO_POSTGRES_DSN),
                )
                _row = _cur.fetchone()
                if _row:
                    return int(_row[0])
    except Exception:
        pass
    try:
        with get_conn() as _c, _c.cursor() as _cur:
            _cur.execute("SELECT tenant_id FROM odoo_connections WHERE active=TRUE LIMIT 1")
            _row = _cur.fetchone()
            if _row:
                return int(_row[0])
    except Exception:
        pass
    return None


def _last_text(msgs) -> str:
    if not msgs:
        return ""
    m = msgs[-1]
    if isinstance(m, BaseMessage):
        return m.content or ""
    if isinstance(m, dict):
        return m.get("content") or ""
    return str(m)


def _parse_website(text: str) -> Optional[str]:
    try:
        s = (text or "").strip()
        if not s:
            return None
        m = re.search(r"https?://[^\s]+", s, re.IGNORECASE)
        if m:
            return m.group(0)
        # fallback: bare domain
        m = re.search(r"\b([a-z0-9-]+\.)+[a-z]{2,}\b", s, re.IGNORECASE)
        return m.group(0) if m else None
    except Exception:
        return None


def _parse_seeds(text: str) -> list[dict]:
    """Parse seeds in forms like 'Company — domain' separated by newlines/semicolons/commas."""
    out: list[dict] = []
    try:
        s = (text or "").strip()
        if not s:
            return out
        # split into items generously
        parts = re.split(r"[\n;]+|\s{2,}", s)
        for p in parts:
            pp = p.strip().strip(",")
            if not pp or len(pp) < 3:
                continue
            # prefer em dash / hyphen separator
            if "—" in pp:
                name, dom = [x.strip() for x in pp.split("—", 1)]
            elif " - " in pp:
                name, dom = [x.strip() for x in pp.split(" - ", 1)]
            else:
                # fallback: take first token as name and look for domain inside
                m = re.search(r"\b([a-z0-9-]+\.)+[a-z]{2,}\b", pp, re.IGNORECASE)
                dom = m.group(0) if m else None
                name = pp if dom is None else pp.replace(dom, "").strip(" -—,|:")
            if name and dom:
                # Filter out protocol placeholders and too-short names (avoid parsing bare URLs as seeds)
                low = name.strip().lower()
                if low in {"http", "https", "www"}:
                    continue
                if not re.search(r"[a-z]", low):
                    continue
                if len(low) < 3:
                    continue
                out.append({"seed_name": name[:120], "domain": dom.lower()})
        # dedupe by domain/name
        seen = set()
        dedup: list[dict] = []
        for it in out:
            key = (it.get("seed_name"), it.get("domain"))
            if key in seen:
                continue
            seen.add(key)
            dedup.append(it)
        return dedup[:20]
    except Exception:
        return out


def _parse_lost_churned(text: str) -> list[dict]:
    """Parse entries like 'Company — domain — reason' per line."""
    out: list[dict] = []
    try:
        s = (text or "").strip()
        if not s:
            return out
        lines = re.split(r"[\n;]+", s)
        for ln in lines:
            ln = ln.strip().strip(",")
            if not ln:
                continue
            parts = [p.strip() for p in re.split(r"\s+—\s+|\s+-\s+|\s+--\s+", ln)]
            # Expect 2 or 3 parts: name, domain[, reason]
            if len(parts) >= 2:
                name, dom = parts[0], parts[1]
                reason = parts[2] if len(parts) >= 3 else None
                # basic filters
                if not re.search(r"[a-z]", name.lower()):
                    continue
                m = re.search(r"\b([a-z0-9-]+\.)+[a-z]{2,}\b", dom, re.IGNORECASE)
                domain = m.group(0).lower() if m else None
                if name and domain:
                    rec = {"seed_name": name[:120], "domain": domain}
                    if reason and isinstance(reason, str) and reason.strip():
                        rec["reason"] = reason.strip()[:240]
                    out.append(rec)
        # dedupe by (name, domain)
        seen = set()
        dedup = []
        for it in out:
            key = (it.get("seed_name"), it.get("domain"))
            if key in seen:
                continue
            seen.add(key)
            dedup.append(it)
        return dedup[:20]
    except Exception:
        return out


def _parse_titles_list(text: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"[,/\n;]+", text or "") if p and p.strip()]
    out: list[str] = []
    for p in parts:
        if len(p) < 2:
            continue
        if p.lower() in {"skip", "none", "any"}:
            continue
        out.append(p[:60])
    # dedupe preserving order
    seen = set()
    ded = []
    for t in out:
        if t.lower() in seen:
            continue
        seen.add(t.lower())
        ded.append(t)
    return ded[:10]


def _parse_currency_usd(text: str) -> float | None:
    try:
        s = (text or "").lower()
        # extract like $18k, 18k, $18000, 18,000
        m = re.search(r"\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)(\s*[km])?", s)
        if not m:
            return None
        num = float(m.group(1).replace(",", ""))
        suffix = (m.group(2) or "").strip().lower()
        if suffix == "k":
            num *= 1_000
        elif suffix == "m":
            num *= 1_000_000
        return float(num)
    except Exception:
        return None


def _parse_weeks_range(text: str) -> tuple[float | None, float | None]:
    try:
        s = (text or "").lower()
        # strip non-digit separators, read patterns like 4–8 weeks, 4-8w, 6 weeks
        m = re.search(r"(\d{1,3})(?:\s*(?:[-–]|to)\s*(\d{1,3}))?\s*(w|wk|wks|week|weeks)?", s)
        if not m:
            return (None, None)
        lo = float(m.group(1))
        hi = float(m.group(2)) if m.group(2) else None
        return (lo, hi)
    except Exception:
        return (None, None)
    except Exception:
        return out


def _log_state(prefix: str, state: Dict[str, Any]):
    prev = _last_text(state.get("messages"))
    logger.info("%s last='%s' keys=%s", prefix, prev[:120], list(state.keys()))


def log_node(name: str):
    def deco(fn):
        if inspect.iscoroutinefunction(fn):

            async def aw(state, *a, **kw):
                _log_state(f"▶ {name}", state)
                out = await fn(state, *a, **kw)
                logger.info("✔ %s → keys=%s", name, list(out.keys()))
                return out

            return aw
        else:

            def sw(state, *a, **kw):
                _log_state(f"▶ {name}", state)
                out = fn(state, *a, **kw)
                logger.info("✔ %s → keys=%s", name, list(out.keys()))
                return out

            return sw

    return deco


def _last_is_ai(messages) -> bool:
    if not messages:
        return False
    m = messages[-1]
    if isinstance(m, BaseMessage):
        return isinstance(m, AIMessage)
    if isinstance(m, dict):
        role = (m.get("type") or m.get("role") or "").lower()
        return role in ("ai", "assistant")
    return False


@log_node("icp")
def icp_discovery(state: PreSDRState) -> PreSDRState:
    # If the user already confirmed, don't re-ask; let routing advance.
    if _user_just_confirmed(state):
        state["icp_confirmed"] = True
        return state
    icp = state.get("icp") or {}
    state["icp"] = icp
    text = _last_text(state.get("messages")).lower()

    # Feature 17: ICP Finder (agent-led, minimal inputs). When enabled, prefer
    # asking for website + seed customers instead of industry/employees.
    try:
        if ENABLE_ICP_INTAKE:
            last = _last_text(state.get("messages"))
            # Ask website first
            if not icp.get("website_url"):
                url = _parse_website(last)
                if url:
                    icp["website_url"] = url
                else:
                    state["messages"].append(
                        AIMessage("Let's infer your ICP from evidence. What's your website URL?")
                    )
                    return state
            # Ask for seeds next (quality gate: require ≥5 best seeds)
            if not icp.get("seeds_list"):
                seeds = _parse_seeds(last)
                if seeds:
                    if len(seeds) >= 5:
                        icp["seeds_list"] = seeds
                    else:
                        state["messages"].append(
                            AIMessage(
                                f"I need at least 5 best customers. You shared {len(seeds)}; please add a few more (Company — website)."
                            )
                        )
                        return state
                else:
                    state["messages"].append(
                        AIMessage(
                            "Share 5–15 best customers (Company — website). Optionally 2–3 lost/churned with a short reason."
                        )
                    )
                    return state
            # Ready to confirm
            state["messages"].append(
                AIMessage(
                    "Thanks! I’ll crawl your site + seed sites, map to ACRA/SSIC, and propose micro‑ICPs with evidence. Reply confirm to proceed, or add more seeds."
                )
            )
            return state
    except Exception:
        pass

    # Legacy prompts only when Finder is disabled
    if not ENABLE_ICP_INTAKE:
        if "industry" not in icp:
            state["messages"].append(
                AIMessage("Which industries or problem spaces? (e.g., SaaS, Pro Services)")
            )
            icp["industry"] = True
            return state
        if "employees" not in icp:
            state["messages"].append(
                AIMessage("Typical company size? (e.g., 10–200 employees)")
            )
            icp["employees"] = True
            return state
        if "geo" not in icp:
            state["messages"].append(AIMessage("Primary geographies? (SG, SEA, global)"))
            icp["geo"] = True
            return state
        if "signals" not in icp:
            state["messages"].append(
                AIMessage("Buying signals? (hiring, stack, certifications)")
            )
            icp["signals"] = True
            return state

    state["messages"].append(
        AIMessage("Great. Reply **confirm** to save, or tell me what to change.")
    )
    return state


@log_node("confirm")
def icp_confirm(state: PreSDRState) -> PreSDRState:
    # Persist ICP from the basic flow when user confirms
    try:
        # Feature 17: ICP Finder — save intake (website + seeds) and run mapping/suggestions
        if ENABLE_ICP_INTAKE:
            icp = dict(state.get("icp") or {})
            url = icp.get("website_url")
            seeds = icp.get("seeds_list") or []
            if url and isinstance(seeds, list) and seeds:
                tid = _resolve_tenant_id_for_write_sync(state)
                if isinstance(tid, int) and _icp_save_intake is not None:
                    answers = {"website": url}
                    payload = {"answers": answers, "seeds": seeds}
                    try:
                        _icp_save_intake(tid, "chat", payload)
                        # Synchronously map and refresh patterns for fast suggestions
                        if _icp_map_seeds and _icp_refresh_patterns and _icp_generate_suggestions:
                            _icp_map_seeds(tid)
                            _icp_refresh_patterns()
                            items = _icp_generate_suggestions(tid)
                            if items:
                                lines = ["Here are draft micro‑ICPs:"]
                                for i, it in enumerate(items, 1):
                                    title = it.get("title") or it.get("id")
                                    ev = it.get("evidence_count") or 0
                                    lines.append(f"{i}) {title} (evidence: {ev})")
                                state["messages"].append(AIMessage("\n".join(lines)))
                                state["micro_icp_suggestions"] = items
                                state["finder_suggestions_done"] = True
                    except Exception:
                        pass
        # Legacy flow persistence
        icp = dict(state.get("icp") or {})
        payload = _icp_payload_from_state_icp(icp)
        if payload:
            tid = _resolve_tenant_id_for_write_sync(state)
            if isinstance(tid, int):
                _save_icp_rule_sync(tid, payload, name="Default ICP")
    except Exception:
        pass
    # Compose preview message with accurate counts and plan for enrichment
    icp = state.get("icp") or {}
    terms = []
    try:
        # Support either single 'industry' or list 'industries'
        if isinstance(icp.get("industry"), str) and icp.get("industry").strip():
            terms.append(icp.get("industry").strip().lower())
        inds = icp.get("industries") or []
        if isinstance(inds, list):
            terms.extend([s.strip().lower() for s in inds if isinstance(s, str) and s.strip()])
        terms = sorted(set(terms))
    except Exception:
        terms = []

    # Seed candidates from sync_head_company_ids when none provided
    if not state.get("candidates"):
        try:
            ids = state.get("sync_head_company_ids") or []
            if isinstance(ids, list) and ids:
                with get_conn() as _c, _c.cursor() as _cur:
                    _cur.execute(
                        "SELECT company_id, name, uen FROM companies WHERE company_id = ANY(%s)",
                        (ids,),
                    )
                    rows = _cur.fetchall()
                    cand = []
                    for r in rows:
                        cid = int(r[0]) if r and r[0] is not None else None
                        nm = (r[1] or "").strip() if len(r) > 1 else ""
                        uen = (r[2] or "").strip() if len(r) > 2 else None
                        if cid and nm:
                            cand.append({"id": cid, "name": nm, "uen": uen})
                    if cand:
                        state["candidates"] = cand
        except Exception:
            pass
    # Current candidate count (preview list)
    n = len(state.get("candidates") or [])

    # Count ICP-matched total in companies
    icp_total = 0
    try:
        clauses: list[str] = []
        params: list = []
        if terms:
            clauses.append("LOWER(industry_norm) = ANY(%s)")
            params.append(terms)
        sql = "SELECT COUNT(*) FROM companies " + ("WHERE " + " AND ".join(clauses) if clauses else "")
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
            icp_total = int(row[0]) if row and row[0] is not None else 0
    except Exception:
        icp_total = n

    msg_lines: list[str] = []
    msg_lines.append(f"Got {n} companies.")

    # SSIC resolution and ACRA totals
    ssic_matches = []
    try:
        if terms:
            ssic_matches = _find_ssic_codes_by_terms(terms)
            if ssic_matches:
                top_code, top_title, _ = ssic_matches[0]
                msg_lines.append(f"Matched {len(ssic_matches)} SSIC codes (top: {top_code} {top_title} …)")
            else:
                msg_lines.append("Matched 0 SSIC codes")
            # ACRA total and sample
            codes = {c for (c, _t, _s) in ssic_matches}
            total_acra = 0
            rows = []
            try:
                if _count_acra_by_ssic_codes:
                    total_acra = _count_acra_by_ssic_codes(codes)  # type: ignore[arg-type]
                rows = _select_acra_by_ssic_codes(codes, 10)
            except Exception:
                total_acra = 0
                rows = []
            if total_acra:
                msg_lines.append(f"Found {total_acra} ACRA candidates. Sample:")
                for r in rows[:2]:
                    uen = (r.get("uen") or "").strip()
                    nm = (r.get("entity_name") or "").strip()
                    code = (r.get("primary_ssic_code") or "").strip()
                    status = (r.get("entity_status_description") or "").strip()
                    msg_lines.append(f"UEN: {uen} – {nm} – SSIC {code} – status: {status}")
            else:
                msg_lines.append("Found 0 ACRA candidates.")
    except Exception:
        # Non-blocking
        pass

    # Planned enrichment counts
    try:
        enrich_now_limit = int(os.getenv("CHAT_ENRICH_LIMIT", os.getenv("RUN_NOW_LIMIT", "10") or 10))
    except Exception:
        enrich_now_limit = 10
    do_now = min(n, enrich_now_limit) if n else 0
    if n > 0:
        # Prefer ACRA total by suggested SSICs, else by matched terms, else company total
        nightly = 0
        try:
            # From micro‑ICP suggestions if present
            sugg = state.get("micro_icp_suggestions") or []
            codes_from_suggestions: list[str] = []
            for it in sugg:
                sid = (it.get("id") or "") if isinstance(it, dict) else ""
                if isinstance(sid, str) and sid.lower().startswith("ssic:"):
                    code = sid.split(":", 1)[1]
                    if code and code.strip():
                        codes_from_suggestions.append(code.strip())
            if codes_from_suggestions and _count_acra_by_ssic_codes:
                total = _count_acra_by_ssic_codes(set(codes_from_suggestions))
                nightly = max(int(total) - do_now, 0)
            elif 'total_acra' in locals():
                nightly = max(int(total_acra) - do_now, 0)
            else:
                nightly = max(int(icp_total) - do_now, 0)
        except Exception:
            nightly = max(int(icp_total) - do_now, 0)
        msg_lines.append(
            f"Ready to enrich {do_now} now; {nightly} scheduled for nightly. Type 'run enrichment' after accepting a micro‑ICP."
        )
    else:
        msg_lines.append("No candidates yet. I’ll keep collecting ICP details.")

    text = "\n\n".join([ln for ln in msg_lines if ln])
    state["icp_match_total"] = icp_total
    state["enrich_now_planned"] = do_now
    state["messages"].append(AIMessage(text))
    return state


@log_node("candidates")
def parse_candidates(state: PreSDRState) -> PreSDRState:
    last = _last_text(state.get("messages"))
    names = [n.strip() for n in last.split(",") if 1 < len(n.strip()) < 120]
    if names:
        state["candidates"] = [{"name": n} for n in names]
        state["messages"].append(
            AIMessage(f"Got {len(names)} companies. Running Enrichment...")
        )
    else:
        state["messages"].append(
            AIMessage("Please paste a few company names (comma-separated).")
        )
    return state


@log_node("enrich")
async def run_enrichment(state: PreSDRState) -> PreSDRState:
    # Persist ICP for the basic flow as soon as enrichment starts
    try:
        icp_cur = dict(state.get("icp") or {})
        payload = _icp_payload_from_state_icp(icp_cur)
        if payload:
            tid = _resolve_tenant_id_for_write_sync(state)  # basic flow uses sync helper
            if isinstance(tid, int):
                _save_icp_rule_sync(tid, payload, name="Default ICP")
    except Exception:
        pass

    candidates = state.get("candidates") or []
    # Prefer sync_head_company_ids captured during chat normalize (10 upserts)
    if not candidates:
        try:
            ids = state.get("sync_head_company_ids") or []
            if isinstance(ids, list) and ids:
                async with (await get_pg_pool()).acquire() as conn:
                    rows = await conn.fetch(
                        "SELECT company_id AS id, name, uen FROM companies WHERE company_id = ANY($1::int[])",
                        [int(i) for i in ids if isinstance(i, int) or (isinstance(i, str) and str(i).isdigit())],
                    )
                cand = []
                for r in rows:
                    nm = r.get("name") or ""
                    if nm:
                        cand.append({"id": int(r.get("id")), "name": nm, "uen": r.get("uen")})
                if cand:
                    candidates = cand
                    state["candidates"] = cand
        except Exception:
            pass
    if not candidates:
        return state

    pool = await get_pg_pool()
    # Resolve tenant for Odoo with robust fallbacks (env → DSN→mapping → first active)
    _tid = None
    try:
        _tid_env = os.getenv("DEFAULT_TENANT_ID")
        _tid = int(_tid_env) if _tid_env and _tid_env.isdigit() else None
    except Exception:
        _tid = None
    if _tid is None:
        try:
            from src.settings import ODOO_POSTGRES_DSN
            inferred_db = None
            if ODOO_POSTGRES_DSN:
                from urllib.parse import urlparse
                u = urlparse(ODOO_POSTGRES_DSN)
                inferred_db = (u.path or "/").lstrip("/") or None
            if inferred_db:
                with get_conn() as _c, _c.cursor() as _cur:
                    _cur.execute(
                        "SELECT tenant_id FROM odoo_connections WHERE (db_name=%s OR db_name=%s) AND active=TRUE LIMIT 1",
                        (inferred_db, ODOO_POSTGRES_DSN),
                    )
                    _row = _cur.fetchone()
                    if _row:
                        _tid = int(_row[0])
        except Exception:
            pass
    if _tid is None:
        try:
            with get_conn() as _c, _c.cursor() as _cur:
                _cur.execute("SELECT tenant_id FROM odoo_connections WHERE active=TRUE LIMIT 1")
                _row = _cur.fetchone()
                if _row:
                    _tid = int(_row[0])
        except Exception:
            pass
    store = None
    try:
        logger.info("odoo resolve: tenant_id=%s (env DEFAULT_TENANT_ID=%s)", _tid, os.getenv("DEFAULT_TENANT_ID"))
    except Exception:
        pass
    try:
        store = OdooStore(tenant_id=_tid)
    except Exception as _init_exc:
        # Fallback: derive DSN for current tenant first; only use first active when tenant is unknown
        try:
            db_name = None
            with get_conn() as _c, _c.cursor() as _cur:
                if _tid is not None:
                    _cur.execute(
                        "SELECT db_name FROM odoo_connections WHERE tenant_id=%s AND active=TRUE LIMIT 1",
                        (_tid,),
                    )
                    _row = _cur.fetchone()
                    db_name = _row[0] if _row and _row[0] else None
                if db_name is None and _tid is None:
                    _cur.execute("SELECT db_name FROM odoo_connections WHERE active=TRUE LIMIT 1")
                    _row = _cur.fetchone()
                    db_name = _row[0] if _row and _row[0] else None
            if db_name:
                tpl = (os.getenv("ODOO_BASE_DSN_TEMPLATE", "") or "").strip()
                if tpl:
                    dsn = tpl.format(db_name=db_name)
                else:
                    dsn = db_name if str(db_name).startswith("postgresql://") else None
                if dsn:
                    logger.info(
                        "odoo init: fallback DSN via mapping db=%s%s",
                        db_name,
                        f" (tenant_id={_tid})" if _tid is not None else "",
                    )
                    store = OdooStore(dsn=dsn)
        except Exception as _fb_exc:
            logger.warning("odoo init fallback error: %s", _fb_exc)
        if store is None:
            logger.warning("odoo init skipped: %s", _init_exc)

    async def _enrich_one(c: Dict[str, Any]) -> Dict[str, Any]:
        name = c["name"]
        cid = c.get("id") or await _ensure_company_row(pool, name)
        uen = c.get("uen")
        await enrich_company_with_tavily(cid, name, uen)
        return {"company_id": cid, "name": name, "uen": uen}

    results = await asyncio.gather(*[_enrich_one(c) for c in candidates])
    state["results"] = results

    ids = [r["company_id"] for r in results if r.get("company_id") is not None]
    if not ids:
        return state

    icp = state.get("icp") or {}
    scoring_initial_state = {
        "candidate_ids": ids,
        "lead_features": [],
        "lead_scores": [],
        "icp_payload": {
            "employee_range": {
                "min": icp.get("employees_min"),
                "max": icp.get("employees_max"),
            },
            "revenue_bucket": icp.get("revenue_bucket"),
            "incorporation_year": {
                "min": icp.get("year_min"),
                "max": icp.get("year_max"),
            },
        },
    }
    scoring_state = await lead_scoring_agent.ainvoke(scoring_initial_state)
    scores = {s["company_id"]: s for s in scoring_state.get("lead_scores", [])}
    features = {f["company_id"]: f for f in scoring_state.get("lead_features", [])}

    async with pool.acquire() as conn:
        comp_rows = await conn.fetch(
            """
            SELECT company_id, name, uen, industry_norm, employees_est,
                   revenue_bucket, incorporation_year, website_domain
            FROM companies WHERE company_id = ANY($1::int[])
            """,
            ids,
        )
        comps = {r["company_id"]: dict(r) for r in comp_rows}
        email_rows = await conn.fetch(
            "SELECT company_id, email FROM lead_emails WHERE company_id = ANY($1::int[])",
            ids,
        )
        emails: Dict[int, str] = {}
        for row in email_rows:
            cid = row["company_id"]
            emails.setdefault(cid, row["email"])

    for cid in ids:
        comp = comps.get(cid, {})
        if not comp:
            continue
        score = scores.get(cid)
        email = emails.get(cid)
        try:
            odoo_id = await store.upsert_company(
                comp.get("name"),
                comp.get("uen"),
                industry_norm=comp.get("industry_norm"),
                employees_est=comp.get("employees_est"),
                revenue_bucket=comp.get("revenue_bucket"),
                incorporation_year=comp.get("incorporation_year"),
                website_domain=comp.get("website_domain"),
            )
            if email:
                try:
                    await store.add_contact(odoo_id, email)
                    logger.info("odoo export: contact added email=%s for partner_id=%s", email, odoo_id)
                except Exception as _contact_exc:
                    logger.warning("odoo export: add_contact failed email=%s err=%s", email, _contact_exc)
            try:
                logger.info("odoo export: upsert company partner_id=%s name=%s", odoo_id, comp.get("name"))
            except Exception:
                pass
            try:
                await store.merge_company_enrichment(odoo_id, {})
            except Exception:
                pass
            if score:
                try:
                    await store.create_lead_if_high(
                        odoo_id,
                        comp.get("name"),
                        score.get("score"),
                        features.get(cid, {}),
                        score.get("rationale", ""),
                        email,
                    )
                except Exception as _lead_exc:
                    logger.warning("odoo export: create_lead failed partner_id=%s err=%s", odoo_id, _lead_exc)
        except Exception as exc:
            logger.exception("odoo sync failed for company_id=%s", cid)

    state["messages"].append(
        AIMessage(f"Enrichment complete for {len(results)} companies.")
    )
    return state


def route(state: PreSDRState) -> str:
    text = _last_text(state.get("messages")).lower()
    if "confirm" in text:
        dest = "confirm"
    elif "run enrichment" in text:
        dest = "enrich"
    elif "," in text or "auto" in text:
        dest = "candidates"
    else:
        dest = "icp"
    logger.info("↪ router -> %s", dest)
    return dest


def build_presdr_graph():
    g = StateGraph(PreSDRState)
    g.add_node("icp", icp_discovery)
    g.add_node("confirm", icp_confirm)
    g.add_node("candidates", parse_candidates)
    g.add_node("enrich", run_enrichment)

    g.set_entry_point("icp")

    # IMPORTANT: these keys must match what route() returns
    g.add_conditional_edges(
        "icp",
        route,
        {
            "confirm": "confirm",
            "enrich": "enrich",
            "candidates": "candidates",
            "icp": "icp",
        },
    )
    g.add_conditional_edges(
        "confirm",
        route,
        {
            "enrich": "enrich",
            "candidates": "candidates",
            "icp": "icp",
        },
    )
    g.add_conditional_edges(
        "candidates",
        route,
        {
            "enrich": "enrich",
            "icp": "icp",
        },
    )
    g.add_edge("enrich", END)
    return g.compile()


# ------------------------------
# New LLM-driven Pre-SDR graph (dynamic Q&A, structured extraction)
# ------------------------------


class GraphState(TypedDict):
    # Ensure appends across runs/nodes
    messages: Annotated[List[BaseMessage], add_messages]
    icp: Dict[str, Any]
    candidates: List[Dict[str, Any]]
    results: List[Dict[str, Any]]
    confirmed: bool
    icp_confirmed: bool
    ask_counts: Dict[str, int]  # how many times we asked each slot
    scored: List[Dict[str, Any]]


# ------------------------------
# LLMs
# ------------------------------

QUESTION_LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
EXTRACT_LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ------------------------------
# Helpers
# ------------------------------


def _to_text(content: Any) -> str:
    """Coerce Chat UI content (string OR list of blocks) into a plain string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if "text" in block and isinstance(block["text"], str):
                    parts.append(block["text"])
                elif "content" in block and isinstance(block.get("content"), str):
                    parts.append(block["content"])
                else:
                    parts.append(str(block))
            else:
                parts.append(str(block))
        return "\n".join(p.strip() for p in parts if p)
    return str(content)


def _last_user_text(state: GraphState) -> str:
    for msg in reversed(state.get("messages") or []):
        if isinstance(msg, HumanMessage):
            return _to_text(msg.content).strip()
    return _to_text((state.get("messages") or [AIMessage("")])[-1].content).strip()


# None/skip/any detector for buying signals
NEG_NONE = {
    "none",
    "no",
    "n/a",
    "na",
    "skip",
    "any",
    "nope",
    "not important",
    "no preference",
    "doesn't matter",
    "dont care",
    "don't care",
    "anything",
    "no specific",
    "no specific signals",
    "no signal",
    "no signals",
}


def _says_none(text: str) -> bool:
    t = text.strip().lower()
    return any(p in t for p in NEG_NONE)


def _user_just_confirmed(state: dict) -> bool:
    msgs = state.get("messages") or []
    for m in reversed(msgs):
        if isinstance(m, HumanMessage):
            txt = (getattr(m, "content", "") or "").strip().lower()
            return txt in {"confirm", "yes", "y", "ok", "okay", "looks good", "lgtm"}
    return False


def _icp_complete(icp: Dict[str, Any]) -> bool:
    has_industries = bool(icp.get("industries"))
    has_employees = bool(icp.get("employees_min") or icp.get("employees_max"))
    has_geos = bool(icp.get("geos"))
    signals_done = bool(icp.get("signals")) or bool(icp.get("signals_done"))
    # Require industries + employees + geos, and either explicit signals or explicit skip (signals_done)
    return has_industries and has_employees and has_geos and signals_done


def _is_company_like(token: str) -> bool:
    """Heuristic to distinguish company names/domains from industries/geos.

    Rules (conservative):
    - Domains (contain a dot) => True
    - Contains company suffix (inc, ltd, corp, llc, pte, plc, gmbh) => True
    - If multi-word: reject if composed only of geo/common words (e.g., "SG and SEA").
      Otherwise require at least one capitalized word (proper noun) to reduce false positives.
    - Single all-lowercase words (e.g., "saas", "fintech") => False
    - Very short tokens (<= 2) => False
    """
    t = (token or "").strip()
    if not t:
        return False
    tl = t.lower()
    if "." in t:
        return True
    # company suffixes
    suffixes = [
        " inc",
        " inc.",
        " ltd",
        " corp",
        " co",
        " llc",
        " pte",
        " plc",
        " gmbh",
        " limited",
        " company",
    ]
    if any(s in tl for s in suffixes):
        return True
    if len(t) <= 2:
        return False
    # Reject single all-lowercase words
    if t.isalpha() and t == t.lower():
        return False
    # Multi-word handling
    if " " in t:
        words = [w for w in re.split(r"\s+", tl) if w]
        geo_words = {
            "sg",
            "singapore",
            "sea",
            "apac",
            "emea",
            "global",
            "us",
            "usa",
            "europe",
            "uk",
            "india",
            "na",
            "latam",
            "southeast",
            "asia",
            "north",
            "south",
            "america",
        }
        connectors = {"and", "&", "/", "-", "or", "the", "of"}
        if all((w in geo_words) or (w in connectors) for w in words):
            return False
        # Require at least one capitalized word (proper noun) to count as company-like
        caps = any(part and part[0].isupper() for part in t.split())
        return caps
    # Mixed-case single word, likely a proper noun (company)
    return any(ch.isupper() for ch in t)


def _parse_company_list(text: str) -> List[str]:
    raw = re.split(r"[,|\n]+", text or "")
    names = [n.strip() for n in raw if n and n.strip()]
    names = [
        n for n in names if n.lower() not in {"start", "confirm", "run enrichment"}
    ]
    # Keep only tokens that look like companies/domains
    names = [n for n in names if _is_company_like(n)]
    # Guard: when ICP Finder is enabled, avoid treating a single URL/domain as an
    # explicit company list (the user likely pasted their website URL).
    try:
        if ENABLE_ICP_INTAKE and len(names) == 1:
            t = names[0]
            if re.search(r"^https?://", t, flags=re.IGNORECASE) or re.search(
                r"\b([a-z0-9-]+\.)+[a-z]{2,}\b", t, flags=re.IGNORECASE
            ):
                return []
    except Exception:
        pass
    return names


# ------------------------------
# Structured extraction
# ------------------------------


class ICPUpdate(BaseModel):
    industries: List[str] = Field(default_factory=list)
    employees_min: Optional[int] = Field(default=None)
    employees_max: Optional[int] = Field(default=None)
    # New: revenue bucket and incorporation year range
    revenue_bucket: Optional[str] = Field(
        default=None, description="small|medium|large"
    )
    year_min: Optional[int] = Field(default=None)
    year_max: Optional[int] = Field(default=None)
    geos: List[str] = Field(default_factory=list)
    signals: List[str] = Field(default_factory=list)
    confirm: bool = Field(default=False)
    pasted_companies: List[str] = Field(default_factory=list)
    signals_done: bool = Field(
        default=False,
        description="True if user said skip/none/any for buying signals",
    )


EXTRACT_SYS = SystemMessage(
    content=(
        "You extract ICP details from user messages.\n"
        "Return JSON ONLY with industries (list[str]), employees_min/max (ints if present), "
        "revenue_bucket (one of 'small','medium','large' if present), year_min/year_max (ints for incorporation year range if present), "
        "geos (list[str]), signals (list[str]), confirm (bool), pasted_companies (list[str]), and signals_done (bool).\n"
        "If the user indicates no preference for buying signals (e.g., 'none', 'any', 'skip'), "
        "set signals_done=true and signals=[]. If the user pasted company names (comma or newline separated), "
        "put them into pasted_companies."
    )
)


async def extract_update_from_text(text: str) -> ICPUpdate:
    structured = EXTRACT_LLM.with_structured_output(ICPUpdate)
    return await structured.ainvoke([EXTRACT_SYS, HumanMessage(text)])


# ------------------------------
# Dynamic question generation
# ------------------------------

QUESTION_SYS = SystemMessage(
    content=(
        "You are an expert SDR assistant. Ask exactly ONE short question at a time to help define an Ideal Customer Profile (ICP). "
        "Keep it brief, concrete, and practical. If ICP looks complete, ask the user to confirm or adjust."
    )
)


def _fmt_icp(icp: Dict[str, Any]) -> str:
    inds = ", ".join(icp.get("industries") or []) or "Any"
    emp_min = icp.get("employees_min")
    emp_max = icp.get("employees_max")
    if emp_min and emp_max:
        emp = f"{emp_min}–{emp_max}"
    elif emp_min:
        emp = f"{emp_min}+"
    elif emp_max:
        emp = f"up to {emp_max}"
    else:
        emp = "Any"
    geos = ", ".join(icp.get("geos") or []) or "Any"
    rev = icp.get("revenue_bucket") or "Any"
    y_min = icp.get("year_min")
    y_max = icp.get("year_max")
    if y_min and y_max:
        years = f"{y_min}–{y_max}"
    elif y_min:
        years = f"{y_min}+"
    elif y_max:
        years = f"up to {y_max}"
    else:
        years = "Any"
    sigs_list = icp.get("signals") or []
    if not sigs_list and icp.get("signals_done"):
        sigs = "None specified"
    else:
        sigs = ", ".join(sigs_list) or "None specified"
    return "\n".join(
        [
            f"- Industries: {inds}",
            f"- Employees: {emp}",
            f"- Revenue: {rev}",
            f"- Inc. Years: {years}",
            f"- Geos: {geos}",
            f"- Signals: {sigs}",
        ]
    )


def next_icp_question(icp: Dict[str, Any]) -> tuple[str, str]:
    order: List[str] = []
    if not icp.get("industries"):
        order.append("industries")
    if not (icp.get("employees_min") or icp.get("employees_max")):
        order.append("employees")
    if not icp.get("revenue_bucket"):
        order.append("revenue")
    if not (icp.get("year_min") or icp.get("year_max")):
        order.append("inc_year")
    if not icp.get("geos"):
        order.append("geos")
    if not icp.get("signals") and not icp.get("signals_done", False):
        order.append("signals")

    if not order:
        summary = _fmt_icp(icp)
        return (
            f"Does ICPs look right? Type **confirm** to enrichment.\n\n{summary}",
            "confirm",
        )

    focus = order[0]
    prompts = {
        "industries": "Which industries or problem spaces should we target? (e.g., SaaS, logistics, fintech)",
        "employees": "What's the typical employee range? (e.g., 10–200)",
        "revenue": "Preferred revenue bucket? (small / medium / large)",
        "inc_year": "Incorporation year range? (e.g., 2015–2024)",
        "geos": "Which geographies or markets? (e.g., SG, SEA, global)",
        "signals": "What specific buying signals are you looking for (e.g., hiring for data roles, ISO 27001, AWS partner)?",
    }
    return (prompts[focus], focus)


# ------------------------------
# Persistence helpers
# ------------------------------


async def _ensure_company_row(pool, name: str) -> int:
    """
    Find an existing company row by name and return its primary key (company_id).
    Only uses the canonical column `company_id` as defined in the schema.
    As a last resort, inserts a minimal row and returns the new company_id.
    """
    async with pool.acquire() as conn:
        # 1) Try company_id first (most common in this repo)
        row = await conn.fetchrow(
            "SELECT company_id FROM companies WHERE name = $1",
            name,
        )
        if row and "company_id" in row:
            return int(row["company_id"])  # type: ignore[index]

        # No legacy `id` support — schema defines only `company_id`.

        # 3) Insert minimal row; prefer returning company_id if present
        # Try RETURNING company_id
        try:
            row = await conn.fetchrow(
                "INSERT INTO companies(name) VALUES ($1) RETURNING company_id",
                name,
            )
            if row and "company_id" in row:
                return int(row["company_id"])  # type: ignore[index]
        except Exception:
            pass
        # Do not attempt RETURNING id; only company_id exists

        # 4) As a final fallback (schemas without defaults), synthesize a new company_id
        #    WARNING: This is best-effort and not concurrency-safe, but unblocks local flows.
        try:
            # Determine next id value from max(company_id)
            row = await conn.fetchrow(
                "SELECT COALESCE(MAX(company_id), 0) + 1 AS nid FROM companies"
            )
            nid = int(row["nid"]) if row and "nid" in row else None  # type: ignore[index]
            if nid is not None:
                await conn.execute(
                    "INSERT INTO companies(company_id, name) VALUES ($1, $2)",
                    nid,
                    name,
                )
                return nid
        except Exception:
            pass

        raise RuntimeError("Could not create or locate a company row for enrichment")


async def _default_candidates(
    pool, icp: Dict[str, Any], limit: int = 20
) -> List[Dict[str, Any]]:
    """
    Pull candidates from companies using basic ICP filters:
    - industry (industry_norm ILIKE)
    - employees_min/max (employees_est range)
    - geos (hq_country/hq_city ILIKE any)
    Falls back gracefully if filters are missing.
    """
    icp = icp or {}
    # Normalize industries; accept multiple. Use exact match on industry_norm (case-insensitive).
    industries_param: List[str] = []
    # Back-compat: allow single 'industry' or list 'industries'
    industry_single = icp.get("industry")
    if isinstance(industry_single, str) and industry_single.strip():
        industries_param.append(industry_single.strip().lower())
    inds = icp.get("industries") or []
    if isinstance(inds, list):
        industries_param.extend(
            [s.strip().lower() for s in inds if isinstance(s, str) and s.strip()]
        )
    # Dedupe
    industries_param = sorted(set(industries_param))
    emp_min = icp.get("employees_min")
    emp_max = icp.get("employees_max")
    rev_bucket = (
        (icp.get("revenue_bucket") or "").strip().lower()
        if isinstance(icp.get("revenue_bucket"), str)
        else None
    )
    y_min = icp.get("year_min")
    y_max = icp.get("year_max")
    geos = icp.get("geos") or []

    base_select = f"""
        SELECT
            c.company_id AS id,
            c.name,
            c.website_domain AS domain,
            c.industry_norm AS industry,
            c.employees_est AS employee_count,
            c.company_size,
            c.hq_city,
            c.hq_country,
            c.linkedin_url
        FROM public.{COMPANY_TABLE} c
    """

    clauses: List[str] = []
    params: List[Any] = []

    if industries_param:
        # Exact equality against normalized industry names
        clauses.append(f"LOWER(c.industry_norm) = ANY(${len(params)+1})")
        params.append(industries_param)
    if isinstance(emp_min, int):
        clauses.append(f"c.employees_est >= ${len(params)+1}")
        params.append(emp_min)
    if isinstance(emp_max, int):
        clauses.append(f"c.employees_est <= ${len(params)+1}")
        params.append(emp_max)
    if rev_bucket in ("small", "medium", "large"):
        clauses.append(f"LOWER(c.revenue_bucket) = ${len(params)+1}")
        params.append(rev_bucket)
    if isinstance(y_min, int):
        clauses.append(f"c.incorporation_year >= ${len(params)+1}")
        params.append(y_min)
    if isinstance(y_max, int):
        clauses.append(f"c.incorporation_year <= ${len(params)+1}")
        params.append(y_max)
    if isinstance(geos, list) and geos:
        # Build an OR group for geos across hq_country/hq_city
        geo_like_params = []
        geo_subclauses = []
        for g in geos:
            if not isinstance(g, str) or not g.strip():
                continue
            like_val = f"%{g.strip()}%"
            # country match
            geo_subclauses.append(
                f"c.hq_country ILIKE ${len(params)+len(geo_like_params)+1}"
            )
            geo_like_params.append(like_val)
            # city match
            geo_subclauses.append(
                f"c.hq_city ILIKE ${len(params)+len(geo_like_params)+1}"
            )
            geo_like_params.append(like_val)
        if geo_subclauses:
            clauses.append("(" + " OR ".join(geo_subclauses) + ")")
            params.extend(geo_like_params)

    where_clause = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    order_by = "ORDER BY c.employees_est DESC NULLS LAST, c.name ASC"

    sql = f"""
        {base_select}
        {where_clause}
        {order_by}
        LIMIT ${len(params)+1}
    """
    params.append(limit)

    async def _run_query(p, q):
        async with pool.acquire() as _conn:
            return await _conn.fetch(q, *p)

    # Pass 1: strict (all available filters)
    rows = await _run_query(params, sql)
    if not rows:
        # Pass 2: relax employees + geo filters, but NEVER drop industry if provided
        r_clauses: List[str] = []
        r_params: List[Any] = []
        if industries_param:
            r_clauses.append(f"LOWER(c.industry_norm) = ANY(${len(r_params)+1})")
            r_params.append(industries_param)
        r_where = ("WHERE " + " AND ".join(r_clauses)) if r_clauses else ""
        r_sql = f"{base_select} {r_where} {order_by} LIMIT ${len(r_params)+1}"
        r_params.append(limit)
        rows = await _run_query(r_params, r_sql)
        if not rows:
            # Pass 3: only if no industry given, show something to unblock the user
            if not industries_param:
                any_sql = f"{base_select} {order_by} LIMIT $1"
                rows = await _run_query([limit], any_sql)
            else:
                # Pass 3b: map industries -> SSIC codes via ssic_ref, then fetch by industry_code
                try:
                    codes = [c for (c, _t, _s) in _find_ssic_codes_by_terms(industries_param)]
                    if codes:
                        code_sql = f"""
                            {base_select}
                            WHERE regexp_replace(c.industry_code::text, '\\D', '', 'g') = ANY($1::text[])
                            {order_by}
                            LIMIT $2
                        """
                        rows = await _run_query([codes, limit], code_sql)
                except Exception:
                    # Do not block on fallback errors
                    pass

    out: List[Dict[str, Any]] = []
    for r in rows:
        d = dict(r)
        d["name"] = d.get("name") or (d.get("domain") or "Unknown")
        out.append(d)
    return out


# ------------------------------
# ICP industry helpers
# ------------------------------

def _extract_icp_industries_from_text(text: str) -> list[str]:
    """Extract user-intended industry phrases from free text (last human message).

    - Splits on common separators/connectors
    - Removes conversational fillers and bullets
    - Prefers multi-word phrases and drops contained single-word generics
    """
    if not text:
        return []
    # Strip URLs to avoid domain tokens becoming "industries"
    try:
        text = re.sub(r"https?://\S+", " ", text)
    except Exception:
        pass
    parts = re.split(r"[,\n;]+|\band\b|\bor\b|/|\\\\|\|", text, flags=re.IGNORECASE)
    stop = {
        "sg",
        "singapore",
        "global",
        "worldwide",
        "sea",
        "apac",
        "emea",
        "us",
        "usa",
        "uk",
        "eu",
        "startup",
        "startups",
        "smb",
        "sme",
        "enterprise",
        "b2b",
        "b2c",
        "small",
        "medium",
        "large",
        "which",
        "which geographies",
        "which industries",
        "problem spaces",
        "should we target",
        "e.g.",
        "eg",
        # URL/common web tokens
        "http",
        "https",
        "www",
    }
    raw: list[str] = []
    for p in parts:
        s = (p or "").strip()
        if not s or len(s) < 2:
            continue
        if s.strip().startswith("-"):
            continue
        if any(ch in s for ch in ("?", "(", ")", ":")):
            continue
        if not re.search(r"[A-Za-z]", s):
            continue
        # Skip bare domains (example.com)
        try:
            if re.search(r"\b([a-z0-9-]+\.)+[a-z]{2,}\b", s, flags=re.IGNORECASE):
                continue
        except Exception:
            pass
        sl = s.lower()
        if sl in stop:
            continue
        sl = re.sub(r"\s+", " ", sl)
        raw.append(sl)
    # Dedupe preserve order
    seen = set()
    out: list[str] = []
    for t in raw:
        if t not in seen:
            seen.add(t)
            out.append(t)
    # Prefer multiword phrases; drop single-word tokens contained in multiwords
    multi = [t for t in out if " " in t]
    if multi:
        singles = {t for t in out if " " not in t}
        singles = {s for s in singles if any(s in m.split() for m in multi)}
        out = [t for t in out if not (" " not in t and t in singles)]
    return out[:10]


# ------------------------------
# LangGraph nodes
# ------------------------------


async def icp_node(state: GraphState) -> GraphState:
    # If the user already confirmed but ICP is incomplete, proactively ask the next question
    # to avoid router loops where the last message remains the user's "confirm".
    if _user_just_confirmed(state):
        state["icp_confirmed"] = True
        try:
            if ENABLE_ICP_INTAKE:
                icp_f = dict(state.get("icp") or {})
                if not icp_f.get("website_url"):
                    state["messages"] = add_messages(
                        state.get("messages") or [],
                        [AIMessage("Before we proceed, what’s your website URL?")],
                    )
                    state["icp"] = icp_f
                    return state
                if not icp_f.get("seeds_list"):
                    asks = dict(state.get("ask_counts") or {})
                    if asks.get("seeds", 0) == 0:
                        asks["seeds"] = 1
                        state["ask_counts"] = asks
                    state["messages"] = add_messages(
                        state.get("messages") or [],
                        [AIMessage("Share 5–15 best customers (Company — website). You can type 'skip' for optional fields later.")],
                    )
                    state["icp"] = icp_f
                    return state
                # If website + seeds exist but core ICP not complete, ask the next missing item
                if not _icp_complete(icp_f):
                    asks = dict(state.get("ask_counts") or {})
                    # Industries → Employees → Geos → Signals
                    if not icp_f.get("industries"):
                        asks["industries"] = asks.get("industries", 0) + 1
                        state["ask_counts"] = asks
                        state["messages"] = add_messages(
                            state.get("messages") or [],
                            [AIMessage("Which industries or problem spaces? (e.g., SaaS, professional services, logistics)")],
                        )
                        state["icp"] = icp_f
                        return state
                    if not (icp_f.get("employees_min") or icp_f.get("employees_max")):
                        asks["employees"] = asks.get("employees", 0) + 1
                        state["ask_counts"] = asks
                        state["messages"] = add_messages(
                            state.get("messages") or [],
                            [AIMessage("Typical company size? (e.g., 10–200 employees)")],
                        )
                        state["icp"] = icp_f
                        return state
                    if not icp_f.get("geos"):
                        asks["geos"] = asks.get("geos", 0) + 1
                        state["ask_counts"] = asks
                        state["messages"] = add_messages(
                            state.get("messages") or [],
                            [AIMessage("Primary geographies? (e.g., SG, SEA, global)")],
                        )
                        state["icp"] = icp_f
                        return state
                    if not icp_f.get("signals") and not icp_f.get("signals_done"):
                        asks["signals"] = asks.get("signals", 0) + 1
                        state["ask_counts"] = asks
                        state["messages"] = add_messages(
                            state.get("messages") or [],
                            [AIMessage("Buying signals? (e.g., hiring, specific tech stack, certifications). Reply 'skip' to continue.")],
                        )
                        state["icp"] = icp_f
                        return state
        except Exception:
            pass
        # If Finder is off or core ICP is already complete, allow router to branch.
        return state

    text = _last_user_text(state)

    # Reset flow when user types 'start' explicitly
    if re.search(r"\bstart\b", text.strip(), flags=re.IGNORECASE):
        state["icp"] = {}
        # Clear transient outputs so we ask fresh ICP questions
        for k in ("candidates", "results", "scored", "ask_counts", "enrichment_completed"):
            try:
                if k in state:
                    del state[k]  # type: ignore[index]
            except Exception:
                pass
        # Proceed to question generation with an empty ICP
        text = ""  # ensure extractor doesn't pollute with previous context

    # Feature 17: ICP Finder intake (website + seeds + core key points) when enabled
    try:
        if ENABLE_ICP_INTAKE:
            icp_f = dict(state.get("icp") or {})
            # Ask for website URL first
            if not icp_f.get("website_url"):
                url = _parse_website(text)
                if url:
                    icp_f["website_url"] = url
                else:
                    state["messages"] = add_messages(
                        state.get("messages") or [],
                        [AIMessage("Let's infer your ICP from evidence. What's your website URL?")],
                    )
                    state["icp"] = icp_f
                    return state
            # Ask for seed customers next (ask-then-parse pattern to avoid parsing the website URL as a seed)
            if not icp_f.get("seeds_list"):
                asks = dict(state.get("ask_counts") or {})
                if asks.get("seeds", 0) == 0:
                    asks["seeds"] = 1
                    state["ask_counts"] = asks
                    state["messages"] = add_messages(
                        state.get("messages") or [],
                        [
                            AIMessage(
                                "Share 5–15 best customers (Company — website). Optionally 2–3 lost/churned with a short reason."
                            )
                        ],
                    )
                    state["icp"] = icp_f
                    return state
                else:
                    seeds = _parse_seeds(text)
                    if seeds:
                        if len(seeds) >= 5:
                            icp_f["seeds_list"] = seeds
                        else:
                            state["messages"] = add_messages(
                                state.get("messages") or [],
                                [AIMessage(f"I need at least 5 best customers. You shared {len(seeds)}; please add a few more (Company — website).")],
                            )
                            state["icp"] = icp_f
                            return state
                    else:
                        state["messages"] = add_messages(
                            state.get("messages") or [],
                            [AIMessage("Got it. Please list seeds as 'Company — domain' per line.")],
                        )
                        state["icp"] = icp_f
                        return state
            # If Fast-Start is enabled, skip detailed prompts and proceed to confirmation
            if ICP_WIZARD_FAST_START_ONLY and not icp_f.get("fast_start_explained"):
                expl = [
                    "I will infer industries from evidence instead of asking.",
                    "What I will crawl:",
                    "- Your site: Industries served, Customers/Case Studies, Integrations, Pricing (ACV hints), Careers (buyer/team clues), Partners, blog topics.",
                    "- Seed and anti-customer sites: industry labels, product lines, About text, Careers (roles/scale), Integrations pages, locations.",
                    "I'll map seeds → SSIC codes via ssic_ref → ACRA (primary_ssic_code) to learn which SSICs and bands dominate winners.",
                ]
                state["messages"] = add_messages(state.get("messages") or [], [AIMessage("\n".join(expl))])
                icp_f["fast_start_explained"] = True
                state["icp"] = icp_f

            # Collect core ICP points before confirmation: industries -> employees -> geos -> (optional) signals
            # 1) Industries (skip when fast-start is enabled; we infer from evidence)
            if not ICP_WIZARD_FAST_START_ONLY and not icp_f.get("industries"):
                asks = dict(state.get("ask_counts") or {})
                if asks.get("industries", 0) == 0:
                    asks["industries"] = 1
                    state["ask_counts"] = asks
                    state["messages"] = add_messages(
                        state.get("messages") or [],
                        [AIMessage("Which industries or problem spaces? (e.g., SaaS, professional services, logistics)")],
                    )
                    state["icp"] = icp_f
                    return state
                else:
                    try:
                        from typing import List as _List
                        from_lang = _extract_icp_industries_from_text(text)
                        inds: _List[str] = [s.strip() for s in from_lang if s and s.strip()]
                    except Exception:
                        inds = []  # type: ignore
                    if inds:
                        icp_f["industries"] = inds
                    else:
                        state["messages"] = add_messages(
                            state.get("messages") or [],
                            [AIMessage("A few examples help (e.g., SaaS, logistics). You can also type 'skip' if not sure.")],
                        )
                        state["icp"] = icp_f
                        return state
            # 2) Employees
            if not (icp_f.get("employees_min") or icp_f.get("employees_max")):
                asks = dict(state.get("ask_counts") or {})
                if asks.get("employees", 0) == 0:
                    asks["employees"] = 1
                    state["ask_counts"] = asks
                    state["messages"] = add_messages(
                        state.get("messages") or [],
                        [AIMessage("Typical company size? (e.g., 10–200 employees)")],
                    )
                    state["icp"] = icp_f
                    return state
                else:
                    m = re.search(r"(\d{1,6})\s*(?:[-–]|to)\s*(\d{1,6})", text)
                    if m:
                        try:
                            lo = int(m.group(1))
                            hi = int(m.group(2))
                            if lo > 0 and hi >= lo:
                                icp_f["employees_min"], icp_f["employees_max"] = lo, hi
                        except Exception:
                            pass
                    if not (icp_f.get("employees_min") or icp_f.get("employees_max")):
                        state["messages"] = add_messages(
                            state.get("messages") or [],
                            [AIMessage("Please provide a range like 10–200, 50-500, or 10 to 200.")],
                        )
                        state["icp"] = icp_f
                        return state
            # 3) Geos
            if not icp_f.get("geos"):
                asks = dict(state.get("ask_counts") or {})
                if asks.get("geos", 0) == 0:
                    asks["geos"] = 1
                    state["ask_counts"] = asks
                    state["messages"] = add_messages(
                        state.get("messages") or [],
                        [AIMessage("Primary geographies? (e.g., SG, SEA, global)")],
                    )
                    state["icp"] = icp_f
                    return state
                else:
                    parts = [p.strip() for p in re.split(r"[,\n]+", text) if p and p.strip()]
                    geos = [p for p in parts if 1 <= len(p) <= 40 and p.lower() not in {"confirm", "start"}]
                    geo_words = {"sg", "singapore", "sea", "apac", "global", "worldwide", "us", "usa", "uk", "eu", "emea", "asia"}
                    geos = [g for g in geos if g.lower() in geo_words or len(g) <= 20]
                    if geos:
                        icp_f["geos"] = sorted(set(geos))
                    else:
                        state["messages"] = add_messages(
                            state.get("messages") or [],
                            [AIMessage("Examples: SG, SEA, APAC, Global. You can list multiple separated by commas.")],
                        )
                        state["icp"] = icp_f
                        return state
            # Optional: signals (skip if none)
            if not icp_f.get("signals") and not icp_f.get("signals_done"):
                if re.search(r"\b(none|skip|any)\b", text.strip(), flags=re.IGNORECASE):
                    icp_f["signals_done"] = True
                else:
                    # Ask once; subsequent turn can proceed with confirm if skipped
                    asks = dict(state.get("ask_counts") or {})
                    if asks.get("signals", 0) < 1:
                        asks["signals"] = 1
                        state["ask_counts"] = asks
                        state["messages"] = add_messages(
                            state.get("messages") or [],
                            [AIMessage("Buying signals? (e.g., hiring, specific tech stack, certifications). Reply 'skip' to continue.")],
                        )
                        state["icp"] = icp_f
                        return state
            # --- Fast‑Start additions ---
            # 4) Lost/Churned (name — website — 1-line reason)
            if not icp_f.get("lost_churned") and icp_f.get("seeds_list"):
                asks = dict(state.get("ask_counts") or {})
                if asks.get("lost_churned", 0) == 0:
                    asks["lost_churned"] = 1
                    state["ask_counts"] = asks
                    state["messages"] = add_messages(
                        state.get("messages") or [],
                        [AIMessage("List 3 lost/churned (Company — website — 1‑line reason). Type 'skip' to continue.")],
                    )
                    state["icp"] = icp_f
                    return state
                else:
                    if re.search(r"\b(skip|none|n/a)\b", text.strip(), flags=re.IGNORECASE):
                        icp_f["lost_churned"] = []
                    else:
                        lc = _parse_lost_churned(text)
                        if lc:
                            icp_f["lost_churned"] = lc
                        else:
                            state["messages"] = add_messages(
                                state.get("messages") or [],
                                [AIMessage("Please format as 'Company — domain — reason'. Or type 'skip'.")],
                            )
                            state["icp"] = icp_f
                            return state
            # 5) Must‑have integrations
            if not icp_f.get("integrations_required"):
                asks = dict(state.get("ask_counts") or {})
                if asks.get("integrations", 0) == 0:
                    asks["integrations"] = 1
                    state["ask_counts"] = asks
                    state["messages"] = add_messages(
                        state.get("messages") or [],
                        [AIMessage("Must‑have integrations? (e.g., HubSpot, Salesforce, Shopify). Comma‑separated; 'skip' to continue.")],
                    )
                    state["icp"] = icp_f
                    return state
                else:
                    if re.search(r"\b(skip|none|n/a)\b", text.strip(), flags=re.IGNORECASE):
                        icp_f["integrations_required"] = []
                    else:
                        icp_f["integrations_required"] = _parse_titles_list(text)
                        if not icp_f["integrations_required"]:
                            state["messages"] = add_messages(
                                state.get("messages") or [],
                                [AIMessage("List integrations separated by commas (e.g., HubSpot, Salesforce). Or type 'skip'.")],
                            )
                            state["icp"] = icp_f
                            return state
            # 6) Average deal size (ACV)
            if not icp_f.get("acv_usd"):
                asks = dict(state.get("ask_counts") or {})
                if asks.get("acv", 0) == 0:
                    asks["acv"] = 1
                    state["ask_counts"] = asks
                    state["messages"] = add_messages(
                        state.get("messages") or [],
                        [AIMessage("Average deal size (ACV)? e.g., $18k. Type 'skip' to continue.")],
                    )
                    state["icp"] = icp_f
                    return state
                else:
                    if re.search(r"\b(skip|none|n/a)\b", text.strip(), flags=re.IGNORECASE):
                        icp_f["acv_usd"] = None
                    else:
                        v = _parse_currency_usd(text)
                        if v and v > 0:
                            icp_f["acv_usd"] = v
                        else:
                            state["messages"] = add_messages(
                                state.get("messages") or [],
                                [AIMessage("Please provide a number like $18k, 18,000, or type 'skip'.")],
                            )
                            state["icp"] = icp_f
                            return state
            # 7) Deal cycle length
            if not (icp_f.get("cycle_weeks_min") or icp_f.get("cycle_weeks_max")):
                asks = dict(state.get("ask_counts") or {})
                if asks.get("cycle", 0) == 0:
                    asks["cycle"] = 1
                    state["ask_counts"] = asks
                    state["messages"] = add_messages(
                        state.get("messages") or [],
                        [AIMessage("Typical deal cycle length? e.g., 4–8 weeks. Type 'skip' to continue.")],
                    )
                    state["icp"] = icp_f
                    return state
                else:
                    if re.search(r"\b(skip|none|n/a)\b", text.strip(), flags=re.IGNORECASE):
                        icp_f["cycle_weeks_min"] = None
                        icp_f["cycle_weeks_max"] = None
                    else:
                        lo, hi = _parse_weeks_range(text)
                        if lo:
                            icp_f["cycle_weeks_min"] = lo
                        if hi:
                            icp_f["cycle_weeks_max"] = hi
                        if not (lo or hi):
                            state["messages"] = add_messages(
                                state.get("messages") or [],
                                [AIMessage("Please provide a value like 4–8 weeks or 6 weeks, or 'skip'.")],
                            )
                            state["icp"] = icp_f
                            return state
            # 8) Price floor
            if not icp_f.get("price_floor_usd"):
                asks = dict(state.get("ask_counts") or {})
                if asks.get("price_floor", 0) == 0:
                    asks["price_floor"] = 1
                    state["ask_counts"] = asks
                    state["messages"] = add_messages(
                        state.get("messages") or [],
                        [AIMessage("Price floor? Deals usually fail when budget < … e.g., <$8k ACV. Type 'skip' to continue.")],
                    )
                    state["icp"] = icp_f
                    return state
                else:
                    if re.search(r"\b(skip|none|n/a)\b", text.strip(), flags=re.IGNORECASE):
                        icp_f["price_floor_usd"] = None
                    else:
                        v = _parse_currency_usd(text)
                        if v and v > 0:
                            icp_f["price_floor_usd"] = v
                        else:
                            state["messages"] = add_messages(
                                state.get("messages") or [],
                                [AIMessage("Provide a number like $8k, 8000, or type 'skip'.")],
                            )
                            state["icp"] = icp_f
                            return state
            # 9) Champion titles
            if not icp_f.get("champion_titles"):
                asks = dict(state.get("ask_counts") or {})
                if asks.get("champions", 0) == 0:
                    asks["champions"] = 1
                    state["ask_counts"] = asks
                    state["messages"] = add_messages(
                        state.get("messages") or [],
                        [AIMessage("Champion title(s)? e.g., RevOps Lead, Head of Sales. Comma‑separated; 'skip' to continue.")],
                    )
                    state["icp"] = icp_f
                    return state
                else:
                    if re.search(r"\b(skip|none|n/a)\b", text.strip(), flags=re.IGNORECASE):
                        icp_f["champion_titles"] = []
                    else:
                        t = _parse_titles_list(text)
                        if t:
                            icp_f["champion_titles"] = t
                        else:
                            state["messages"] = add_messages(
                                state.get("messages") or [],
                                [AIMessage("List titles separated by commas (e.g., RevOps Lead, Head of Sales), or 'skip'.")],
                            )
                            state["icp"] = icp_f
                            return state
            # 10) Predictive events
            if not icp_f.get("triggers"):
                asks = dict(state.get("ask_counts") or {})
                if asks.get("triggers", 0) == 0:
                    asks["triggers"] = 1
                    state["ask_counts"] = asks
                    state["messages"] = add_messages(
                        state.get("messages") or [],
                        [AIMessage("3 events that predict a good fit? e.g., Hiring RevOps, migrating to HubSpot. Comma‑separated; 'skip' to continue.")],
                    )
                    state["icp"] = icp_f
                    return state
                else:
                    if re.search(r"\b(skip|none|n/a)\b", text.strip(), flags=re.IGNORECASE):
                        icp_f["triggers"] = []
                    else:
                        icp_f["triggers"] = _parse_titles_list(text)
                        if not icp_f["triggers"]:
                            state["messages"] = add_messages(
                                state.get("messages") or [],
                                [AIMessage("List events separated by commas (e.g., Hiring RevOps, HubSpot migration), or 'skip'.")],
                            )
                            state["icp"] = icp_f
                            return state
            # Ready to confirm
            state["messages"] = add_messages(
                state.get("messages") or [],
                [
                    AIMessage(
                        "Thanks! I’ll crawl your site + seed sites, map to ACRA/SSIC, enrich evidence, mine patterns, and propose micro‑ICPs. Reply confirm to proceed, or adjust any detail."
                    )
                ],
            )
            state["icp"] = icp_f
            return state
    except Exception:
        # Non-blocking: fall through to legacy questions if anything goes wrong
        pass

    # Legacy Q&A path (industries, size, geos, signals)
    # 1) Extract structured update
    update = await extract_update_from_text(text)

    icp = dict(state.get("icp") or {})

    # 2) Merge extractor output into ICP (prefer precise phrases from user text)
    if True:
        human_terms = _extract_icp_industries_from_text(text)
        llm_terms = [s.strip() for s in (update.industries or []) if s and s.strip()]
        merged: list[str] = []
        # human phrases first, preserve order
        for t in human_terms:
            if t not in merged:
                merged.append(t)
        # then LLM terms if new (case-insensitive dedupe)
        low = {t.lower(): t for t in merged}
        for t in llm_terms:
            tl = t.lower()
            if tl not in low:
                merged.append(t)
                low[tl] = t
        # Prefer multiword phrases: drop single-word generics contained in multiwords
        multi = [t for t in merged if " " in t]
        if multi:
            singles = {t for t in merged if " " not in t}
            singles = {s for s in singles if any(s.lower() in m.split() for m in multi)}
            merged = [t for t in merged if not (" " not in t and t in singles)]
        if merged:
            icp["industries"] = merged
    if update.employees_min is not None:
        icp["employees_min"] = update.employees_min
    if update.employees_max is not None:
        icp["employees_max"] = update.employees_max
    # New: revenue_bucket and incorporation year
    if getattr(update, "revenue_bucket", None):
        # normalize to lowercase canonical values if possible
        rb = (update.revenue_bucket or "").strip().lower()
        if rb in ("small", "medium", "large"):
            icp["revenue_bucket"] = rb
    if getattr(update, "year_min", None) is not None:
        icp["year_min"] = update.year_min
    if getattr(update, "year_max", None) is not None:
        icp["year_max"] = update.year_max
    if update.geos:
        icp["geos"] = sorted(set([s.strip() for s in update.geos if s.strip()]))
    if update.signals:
        icp["signals"] = sorted(set([s.strip() for s in update.signals if s.strip()]))

    # 3) Treat explicit “none/skip/any” as signals_done
    if _says_none(text) or getattr(update, "signals_done", False):
        icp["signals"] = []
        icp["signals_done"] = True

    new_msgs: List[BaseMessage] = []

    # If user pasted companies, preserve previous behavior
    if update.pasted_companies:
        state["candidates"] = [{"name": n} for n in update.pasted_companies]
        new_msgs.append(
            AIMessage(
                content=f"Got {len(update.pasted_companies)} companies. Type **run enrichment** to start."
            )
        )

    # 4) Back-off: if we already asked about 'signals' once and still don't have them, stop asking
    ask_counts = dict(state.get("ask_counts") or {})
    q, focus = next_icp_question(icp)
    if (
        focus == "signals"
        and ask_counts.get("signals", 0) >= 1
        and not icp.get("signals")
    ):
        icp["signals_done"] = True
        q, focus = next_icp_question(icp)

    ask_counts[focus] = ask_counts.get(focus, 0) + 1
    state["ask_counts"] = ask_counts

    new_msgs.append(AIMessage(content=q))

    state["icp"] = icp
    state["messages"] = add_messages(state.get("messages") or [], new_msgs)
    return state


async def candidates_node(state: GraphState) -> GraphState:
    # Prefer 10 upserted IDs captured during chat normalize if present
    if not state.get("candidates"):
        try:
            ids = state.get("sync_head_company_ids") or []
            if isinstance(ids, list) and ids:
                async with (await get_pg_pool()).acquire() as conn:
                    rows = await conn.fetch(
                        "SELECT company_id AS id, name, uen FROM companies WHERE company_id = ANY($1::int[])",
                        [int(i) for i in ids if isinstance(i, int) or (isinstance(i, str) and str(i).isdigit())],
                    )
                cand = []
                for r in rows:
                    nm = r.get("name") or ""
                    if nm:
                        cand.append({"id": int(r.get("id")), "name": nm, "uen": r.get("uen")})
                if cand:
                    state["candidates"] = cand
        except Exception:
            pass
    if not state.get("candidates"):
        pool = await get_pg_pool()
        cand = await _default_candidates(pool, state.get("icp") or {}, limit=20)
        state["candidates"] = cand

    n = len(state["candidates"]) if state.get("candidates") else 0

    # Consolidated message: Got N companies + SSIC + ACRA sample + start note
    icp = state.get("icp") or {}
    terms = [
        s.strip().lower()
        for s in (icp.get("industries") or [])
        if isinstance(s, str) and s.strip()
    ]
    # New: derive SSIC codes from Finder suggestions if present
    sugg = state.get("micro_icp_suggestions") or []
    codes_from_suggestions: list[str] = []
    try:
        if isinstance(sugg, list) and sugg:
            for it in sugg:
                sid = (it.get("id") or "") if isinstance(it, dict) else ""
                if isinstance(sid, str) and sid.lower().startswith("ssic:"):
                    code = sid.split(":", 1)[1]
                    if code and code.strip():
                        codes_from_suggestions.append(code.strip())
    except Exception:
        codes_from_suggestions = []
    lines: list[str] = []
    # Compute ICP total in companies (not just the preview list)
    icp_total = 0
    try:
        clauses: list[str] = []
        params: list = []
        if terms:
            clauses.append("LOWER(industry_norm) = ANY($1)")
            params.append(terms)
        # If no free-text terms but we have SSIC codes from suggestions, count by industry_code
        if not terms and codes_from_suggestions:
            clauses.append("regexp_replace(industry_code::text, '\\D', '', 'g') = ANY($" + str(len(params)+1) + "::text[])")
            params.append(codes_from_suggestions)
        sql = "SELECT COUNT(*) FROM companies " + ("WHERE " + " AND ".join(clauses) if clauses else "")
        async with (await get_pg_pool()).acquire() as conn:
            row = await conn.fetchrow(sql, *params)
            icp_total = int(row[0]) if row and row[0] is not None else 0
    except Exception:
        icp_total = n
    try:
        # Prefer free-text terms flow when provided; otherwise leverage SSIC codes from suggestions
        if terms or codes_from_suggestions:
            ssic_matches = _find_ssic_codes_by_terms(terms) if terms else [(c, "", 1.0) for c in codes_from_suggestions]
            if ssic_matches:
                top_code, top_title, _ = ssic_matches[0]
                lines.append(
                    f"Matched {len(ssic_matches)} SSIC codes (top: {top_code} {top_title} …)"
                )
            else:
                lines.append("Matched 0 SSIC codes")
            # ACRA count + sample
            try:
                codes = {c for (c, _t, _s) in ssic_matches}
                total_acra = 0
                try:
                    if _count_acra_by_ssic_codes:
                        total_acra = await asyncio.to_thread(_count_acra_by_ssic_codes, codes)
                except Exception:
                    total_acra = 0
                rows = await asyncio.to_thread(_select_acra_by_ssic_codes, codes, 10)
            except Exception:
                rows = []
                total_acra = 0
            if total_acra:
                lines.append(f"- Found {total_acra} ACRA candidates. Sample:")
                for r in rows[:2]:
                    uen = (r.get("uen") or "").strip()
                    nm = (r.get("entity_name") or "").strip()
                    code = (r.get("primary_ssic_code") or "").strip()
                    status = (r.get("entity_status_description") or "").strip()
                    lines.append(
                        f"UEN: {uen} – {nm} – SSIC {code} – status: {status}"
                    )
                # If we have fewer candidates than the immediate enrichment cap, top up from ACRA by ensuring company rows
                try:
                    import os as _os
                    enrich_now_limit = int(_os.getenv("CHAT_ENRICH_LIMIT", _os.getenv("RUN_NOW_LIMIT", "10") or 10))
                except Exception:
                    enrich_now_limit = 10
                if (state.get("candidates") or []) and len(state.get("candidates") or []) < enrich_now_limit:
                    try:
                        pool = await get_pg_pool()
                        needed = enrich_now_limit - len(state.get("candidates") or [])
                        added: list[dict] = []
                        for r in rows:
                            if needed <= 0:
                                break
                            nm = (r.get("entity_name") or "").strip()
                            if not nm:
                                continue
                            try:
                                cid = await _ensure_company_row(pool, nm)
                            except Exception:
                                continue
                            added.append({"id": cid, "name": nm, "uen": (r.get("uen") or "").strip() or None})
                            needed -= 1
                        if added:
                            state["candidates"] = (state.get("candidates") or []) + added
                            n = len(state["candidates"])
                    except Exception:
                        pass
            else:
                lines.append("- Found 0 ACRA candidates.")

        # If still no candidates and we have SSIC codes from suggestions, try pulling directly by code from companies
        if not (state.get("candidates") or []) and codes_from_suggestions:
            try:
                async with (await get_pg_pool()).acquire() as conn:
                    rows = await conn.fetch(
                        """
                        SELECT company_id AS id, name, uen, website_domain AS domain
                        FROM companies
                        WHERE regexp_replace(industry_code::text, '\\D', '', 'g') = ANY($1::text[])
                        ORDER BY employees_est DESC NULLS LAST, name ASC
                        LIMIT 20
                        """,
                        codes_from_suggestions,
                    )
                cand = [{"id": int(r["id"]), "name": r["name"], "uen": r["uen"], "domain": r["domain"]} for r in rows]
                if cand:
                    state["candidates"] = cand
                    n = len(cand)
            except Exception:
                pass
            # If still empty, attempt ACRA→companies ensure by codes
            if not (state.get("candidates") or []):
                try:
                    rows = await asyncio.to_thread(_select_acra_by_ssic_codes, set(codes_from_suggestions), 20)
                except Exception:
                    rows = []
                if rows:
                    try:
                        pool = await get_pg_pool()
                        added: list[dict] = []
                        for r in rows:
                            nm = (r.get("entity_name") or "").strip()
                            if not nm:
                                continue
                            try:
                                cid = await _ensure_company_row(pool, nm)
                                added.append({"id": cid, "name": nm, "uen": (r.get("uen") or "").strip() or None})
                            except Exception:
                                continue
                        if added:
                            state["candidates"] = added
                            n = len(added)
                    except Exception:
                        pass
    except Exception:
        pass
    # After potential top-up, refresh n and state, then report count
    n = len(state.get("candidates") or [])
    lines.insert(0, f"Got {n} companies. ")
    # Plan and communicate enrichment counts
    try:
        import os as _os
        enrich_now_limit = int(_os.getenv("CHAT_ENRICH_LIMIT", _os.getenv("RUN_NOW_LIMIT", "10") or 10))
    except Exception:
        enrich_now_limit = 10
    do_now = min(n, enrich_now_limit) if n else 0
    if n > 0:
        # If user already confirmed (router sent us here due to no candidates yet), reflect that enrichment will start now without extra prompt
        try:
            just_confirmed = _user_just_confirmed(state)
        except Exception:
            just_confirmed = False
        # Prefer ACRA total for nightly planning using suggested SSIC codes when available
        scheduled = 0
        try:
            sugg = state.get("micro_icp_suggestions") or []
            codes_from_suggestions: list[str] = []
            for it in sugg:
                sid = (it.get("id") or "") if isinstance(it, dict) else ""
                if isinstance(sid, str) and sid.lower().startswith("ssic:"):
                    code = sid.split(":", 1)[1]
                    if code and code.strip():
                        codes_from_suggestions.append(code.strip())
            if codes_from_suggestions and _count_acra_by_ssic_codes:
                acra_total = await asyncio.to_thread(_count_acra_by_ssic_codes, set(codes_from_suggestions))
                scheduled = max(int(acra_total) - do_now, 0)
            elif 'total_acra' in locals():
                scheduled = max(int(total_acra) - do_now, 0)
            else:
                scheduled = max(int(icp_total) - do_now, 0)
        except Exception:
            scheduled = max(int(icp_total) - do_now, 0)
        lines.append(
            f"Ready to enrich {do_now} now; {scheduled} scheduled for nightly. Accept a micro‑ICP, then type 'run enrichment' to proceed."
        )
    else:
        lines.append("No candidates yet. I’ll keep collecting ICP details.")
    msg = "\n".join([ln for ln in lines if ln])

    state["messages"] = add_messages(state.get("messages") or [], [AIMessage(content=msg)])
    return state


async def confirm_node(state: GraphState) -> GraphState:
    state["confirmed"] = True
    logger.info("[confirm] Entered confirm_node")
    # Persist ICP captured in the dynamic graph flow
    try:
        icp = dict(state.get("icp") or {})
        try:
            logger.info("[confirm] ICP keys present: %s", sorted(list(icp.keys())))
        except Exception:
            pass
        # Feature 17: if Finder inputs present, save intake and generate suggestions immediately
        try:
            if ENABLE_ICP_INTAKE and icp.get("website_url") and icp.get("seeds_list"):
                tid = await _resolve_tenant_id_for_write(state)
                if isinstance(tid, int):
                    chips: list[str] = []
                    # Collect Fast‑Start answers for persistence
                    answers = {
                        "website": icp.get("website_url"),
                        "industries": icp.get("industries"),
                        "employees_min": icp.get("employees_min"),
                        "employees_max": icp.get("employees_max"),
                        "geos": icp.get("geos"),
                        "signals": icp.get("signals") if icp.get("signals") else [],
                        "lost_churned": icp.get("lost_churned") if icp.get("lost_churned") else [],
                        "integrations_required": icp.get("integrations_required") if icp.get("integrations_required") else [],
                        "acv_usd": icp.get("acv_usd"),
                        "cycle_weeks_min": icp.get("cycle_weeks_min"),
                        "cycle_weeks_max": icp.get("cycle_weeks_max"),
                        "price_floor_usd": icp.get("price_floor_usd"),
                        "champion_titles": icp.get("champion_titles") if icp.get("champion_titles") else [],
                        "triggers": icp.get("triggers") if icp.get("triggers") else [],
                    }
                    try:
                        logger.info("[confirm] Intake Answers JSON: %s", json.dumps(answers, ensure_ascii=False))
                        logger.info("[confirm] Seeds JSON: %s", json.dumps(icp.get("seeds_list") or [], ensure_ascii=False))
                    except Exception:
                        pass
                    payload_in = {"answers": answers, "seeds": icp.get("seeds_list")}
                    # Persist intake if storage function is available; otherwise continue Finder pipeline
                    if _icp_save_intake:
                        logger.info("[confirm] Saving intake (tenant_id=%s)", tid)
                        await asyncio.to_thread(_icp_save_intake, tid, "chat", payload_in)
                        chips.append("Intake saved ✓")
                    else:
                        logger.info("[confirm] _icp_save_intake unavailable; skipping DB persist but continuing")
                    # Store intake-derived evidence
                    try:
                        if _icp_store_intake_evidence:
                            cnt = await asyncio.to_thread(_icp_store_intake_evidence, tid, answers)
                            logger.info("[confirm] Intake evidence rows inserted: %s", cnt)
                            chips.append("Evidence (from answers) ✓")
                    except Exception:
                        pass
                    # Seed Normalization + Resolver (Step 2–3): propose domains and show fast facts
                    try:
                        seeds_in = list(icp.get("seeds_list") or [])
                        if _icp_build_resolver_cards and isinstance(seeds_in, list) and seeds_in:
                            logger.info("[confirm] Building resolver cards for %d seeds", len(seeds_in))
                            cards = await _icp_build_resolver_cards(seeds_in)
                            if cards:
                                logger.info("[confirm] Resolver cards built: %d", len(cards))
                                lines = ["Domain resolver preview:"]
                                low_conf = 0
                                for i, c in enumerate(cards, 1):
                                    facts = []
                                    ff = c.fast_facts or {}
                                    if ff.get("industry_guess"):
                                        facts.append(f"industry={ff.get('industry_guess')}")
                                    if ff.get("size_band_guess"):
                                        facts.append(f"size={ff.get('size_band_guess')}")
                                    if ff.get("geo_guess"):
                                        facts.append(f"geo={ff.get('geo_guess')}")
                                    buyers = ", ".join(ff.get("buyer_titles") or [])
                                    if buyers:
                                        facts.append(f"buyers={buyers}")
                                    integ = ", ".join(ff.get("integrations_mentions") or [])
                                    if integ:
                                        facts.append(f"integrations={integ}")
                                    fact_str = f" ({'; '.join(facts)})" if facts else ""
                                    lines.append(f"{i}) {c.seed_name} → {c.domain} [{c.confidence}] — {c.why}{fact_str}")
                                    if (c.confidence or "").lower() == "low":
                                        low_conf += 1
                                if low_conf:
                                    lines.append(f"{low_conf} low‑confidence matches. Reply with edits if any domain looks off.")
                        state["messages"] = add_messages(state.get("messages") or [], [AIMessage("\n".join(lines))])
                        chips.append("Domain resolve ✓")
                    except Exception:
                        pass

                    # Evidence collection (Step 4) and ACRA anchoring (Step 5) — best‑effort batch
                    try:
                        ev_count = 0
                        acra_count = 0
                        # Crawl user's own website first for industries/customers/integrations/pricing/careers/partners/blog
                        try:
                            site_url = (icp.get("website_url") or "").strip()
                            if site_url and _icp_collect_evidence_for_domain:
                                # Normalize to apex domain
                                from urllib.parse import urlparse
                                _apex = (urlparse(site_url).netloc or site_url).lower()
                                if _apex:
                                    n0 = await _icp_collect_evidence_for_domain(tid, None, _apex)
                                    logger.info("[confirm] Evidence rows for tenant website=%s: %s", _apex, n0)
                                    ev_count += int(n0 or 0)
                        except Exception:
                            pass
                        async def _collect_for_seed(s: dict):
                            """Collect crawl evidence and ACRA SSIC anchoring for each seed.
                            Do BOTH when possible so we still get SSIC evidence even if crawl yields 0.
                            Returns tuple: (evidence_rows_added, acra_rows_added)
                            """
                            name = (s.get("seed_name") or "").strip()
                            dom = (s.get("domain") or "").strip()
                            ev_added = 0
                            acra_added = 0
                            # Crawl evidence (best effort)
                            if _icp_collect_evidence_for_domain and dom:
                                try:
                                    n = await _icp_collect_evidence_for_domain(tid, None, dom)
                                    logger.info("[confirm] Evidence rows for domain=%s: %s", dom, n)
                                    ev_added += int(n or 0)
                                except Exception:
                                    pass
                            # Always attempt ACRA anchoring from seed name as well
                            if _icp_acra_anchor_seed and name:
                                try:
                                    acra = await asyncio.to_thread(_icp_acra_anchor_seed, tid, name, None)
                                    logger.info(
                                        "[confirm] ACRA anchored name=%s uen=%s ssic=%s",
                                        name,
                                        (acra or {}).get("uen"),
                                        (acra or {}).get("primary_ssic_code"),
                                    )
                                    if acra:
                                        acra_added += 1
                                except Exception:
                                    pass
                            return (ev_added, acra_added)
                        seeds_list = list(icp.get("seeds_list") or [])
                        for n, a in await asyncio.gather(*[_collect_for_seed(s) for s in seeds_list]):
                            ev_count += int(n)
                            acra_count += int(a)
                        logger.info("[confirm] Evidence total=%s ACRA total=%s", ev_count, acra_count)
                        if seeds_list:
                            chips.append("Evidence ✓")
                            chips.append("ACRA ✓")
                    except Exception:
                        pass

                    # Pattern mining + Micro‑ICPs (Step 6–8)
                    try:
                        if _icp_winner_profile and _icp_micro_suggestions:
                            logger.info("[confirm] Building winners profile")
                            prof = await asyncio.to_thread(_icp_winner_profile, tid)
                            items2 = await asyncio.to_thread(_icp_micro_suggestions, prof)
                            chips.append("Patterns ✓")
                            if items2:
                                logger.info("[confirm] Micro‑ICP suggestions=%d", len(items2))
                                lines = ["Early micro‑ICP suggestions:"]
                                for i, it in enumerate(items2, 1):
                                    title = it.get("title") or it.get("id")
                                    ev = it.get("evidence_count") or 0
                                    lines.append(f"{i}) {title} (evidence: {ev})")
                                # Also show industry titles for SSIC-based suggestions
                                try:
                                    codes = []
                                    for it in items2:
                                        sid = (it.get("id") or "") if isinstance(it, dict) else ""
                                        if isinstance(sid, str) and sid.lower().startswith("ssic:"):
                                            codes.append(sid.split(":", 1)[1])
                                    titles: list[str] = []
                                    if codes:
                                        # Lookup titles from ssic_ref
                                        from src.database import get_conn as _get_conn
                                        with _get_conn() as _c, _c.cursor() as _cur:
                                            _cur.execute(
                                                "SELECT code, title FROM ssic_ref WHERE regexp_replace(code::text,'\\D','','g') = ANY(%s::text[])",
                                                (codes,),
                                            )
                                            for r in _cur.fetchall() or []:
                                                titles.append(f"{r[1]} (SSIC {r[0]})")
                                    if titles:
                                        lines.append("")
                                        lines.append("Targeted industry preview:")
                                        for t in titles[:3]:
                                            lines.append(f"- {t}")
                                    # ACRA total based on suggested SSIC codes
                                    if codes:
                                        try:
                                            from src.icp import _count_acra_by_ssic_codes as _cnt_acra, _select_acra_by_ssic_codes as _sel_acra
                                            total = await asyncio.to_thread(_cnt_acra, set(codes)) if _cnt_acra else 0
                                            lines.append(f"Found ~{total} ACRA candidates matching suggested SSICs.")
                                            if total:
                                                rows = await asyncio.to_thread(_sel_acra, set(codes), 2)
                                                for r in rows[:2]:
                                                    uen = (r.get("uen") or "").strip()
                                                    nm = (r.get("entity_name") or "").strip()
                                                    code = (r.get("primary_ssic_code") or "").strip()
                                                    status = (r.get("entity_status_description") or "").strip()
                                                    lines.append(f"UEN: {uen} – {nm} – SSIC {code} – status: {status}")
                                        except Exception:
                                            pass
                                except Exception:
                                    pass
                                state["messages"] = add_messages(state.get("messages") or [], [AIMessage("\n".join(lines))])
                                chips.append("Suggestions ✓")
                                state["micro_icp_suggestions"] = items2
                            state["finder_suggestions_done"] = True
                    except Exception:
                        pass

                    # Synchronously map seeds and refresh patterns for fast suggestions
                    if _icp_map_seeds and _icp_refresh_patterns and _icp_generate_suggestions:
                        logger.info("[confirm] Map seeds to evidence + refresh patterns + generate suggestions")
                        await asyncio.to_thread(_icp_map_seeds, tid)
                        await asyncio.to_thread(_icp_refresh_patterns)
                        items = await asyncio.to_thread(_icp_generate_suggestions, tid)
                        if items:
                            lines = ["Here are draft micro‑ICPs:"]
                            for i, it in enumerate(items, 1):
                                title = it.get("title") or it.get("id")
                                ev = it.get("evidence_count") or 0
                                lines.append(f"{i}) {title} (evidence: {ev})")
                            # Titles for SSIC codes
                            try:
                                codes = []
                                for it in items:
                                    sid = (it.get("id") or "") if isinstance(it, dict) else ""
                                    if isinstance(sid, str) and sid.lower().startswith("ssic:"):
                                        codes.append(sid.split(":", 1)[1])
                                titles: list[str] = []
                                if codes:
                                    from src.database import get_conn as _get_conn
                                    with _get_conn() as _c, _c.cursor() as _cur:
                                        _cur.execute(
                                            "SELECT code, title FROM ssic_ref WHERE regexp_replace(code::text,'\\D','','g') = ANY(%s::text[])",
                                            (codes,),
                                        )
                                        for r in _cur.fetchall() or []:
                                            titles.append(f"{r[1]} (SSIC {r[0]})")
                                if titles:
                                    lines.append("")
                                    lines.append("Targeted industry preview:")
                                    for t in titles[:3]:
                                        lines.append(f"- {t}")
                                # ACRA total based on suggested SSIC codes
                                if codes:
                                    try:
                                        from src.icp import _count_acra_by_ssic_codes as _cnt_acra, _select_acra_by_ssic_codes as _sel_acra
                                        total = await asyncio.to_thread(_cnt_acra, set(codes)) if _cnt_acra else 0
                                        lines.append(f"Found ~{total} ACRA candidates matching suggested SSICs.")
                                        if total:
                                            rows = await asyncio.to_thread(_sel_acra, set(codes), 2)
                                            for r in rows[:2]:
                                                uen = (r.get("uen") or "").strip()
                                                nm = (r.get("entity_name") or "").strip()
                                                code = (r.get("primary_ssic_code") or "").strip()
                                                status = (r.get("entity_status_description") or "").strip()
                                                lines.append(f"UEN: {uen} – {nm} – SSIC {code} – status: {status}")
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                            state["messages"] = add_messages(state.get("messages") or [], [AIMessage("\n".join(lines))])
                            if "Suggestions ✓" not in chips:
                                chips.append("Suggestions ✓")

                    # Emit progress chips as a single chat message
                    if chips:
                        progress = "Progress: " + " → ".join(chips)
                        state["messages"] = add_messages(state.get("messages") or [], [AIMessage(progress)])
        except Exception as _e:
            logger.warning("[confirm] Finder confirm pipeline failed: %s", _e)
            # Non-blocking; fall back to legacy persistence
            pass

        # Legacy persistence for industry-based ICP
        payload = _icp_payload_from_state_icp(icp)
        if payload:
            tid = await _resolve_tenant_id_for_write(state)
            if isinstance(tid, int):
                _save_icp_rule_sync(tid, payload, name="Default ICP")
    except Exception:
        pass

    # Ensure we have candidates to work with post-confirm
    if not state.get("candidates"):
        try:
            pool = await get_pg_pool()
            cand = await _default_candidates(pool, state.get("icp") or {}, limit=20)
            state["candidates"] = cand
        except Exception:
            state["candidates"] = []

    # Ensure we have up to 10 candidates ready (prefer sync head IDs; then top up from ACRA)
    try:
        enrich_now_limit = int(os.getenv("CHAT_ENRICH_LIMIT", os.getenv("RUN_NOW_LIMIT", "10") or 10))
    except Exception:
        enrich_now_limit = 10
    if not state.get("candidates"):
        try:
            ids = state.get("sync_head_company_ids") or []
            if isinstance(ids, list) and ids:
                async with (await get_pg_pool()).acquire() as conn:
                    rows = await conn.fetch(
                        "SELECT company_id AS id, name, uen FROM companies WHERE company_id = ANY($1::int[])",
                        [int(i) for i in ids if isinstance(i, int) or (isinstance(i, str) and str(i).isdigit())],
                    )
                cand = []
                for r in rows:
                    nm = r.get("name") or ""
                    if nm:
                        cand.append({"id": int(r.get("id")), "name": nm, "uen": r.get("uen")})
                if cand:
                    state["candidates"] = cand[:enrich_now_limit]
        except Exception:
            pass
    # If still fewer than limit and we have SSIC codes, top up from ACRA by ensuring rows
    n = len(state.get("candidates") or [])
    if n < enrich_now_limit:
        try:
            icp = state.get("icp") or {}
            terms = [
                s.strip().lower() for s in (icp.get("industries") or []) if isinstance(s, str) and s.strip()
            ]
            if terms:
                ssic_matches = _find_ssic_codes_by_terms(terms)
                codes = {c for (c, _t, _s) in ssic_matches}
                if codes:
                    rows = await asyncio.to_thread(_select_acra_by_ssic_codes, codes, enrich_now_limit)
                    pool = await get_pg_pool()
                    added: list[dict] = []
                    needed = enrich_now_limit - n
                    for r in rows:
                        if needed <= 0:
                            break
                        nm = (r.get("entity_name") or "").strip()
                        if not nm:
                            continue
                        try:
                            cid = await _ensure_company_row(pool, nm)
                        except Exception:
                            continue
                        added.append({"id": cid, "name": nm, "uen": (r.get("uen") or "").strip() or None})
                        needed -= 1
                    if added:
                        state["candidates"] = (state.get("candidates") or []) + added
                        n = len(state["candidates"])  # refresh
        except Exception:
            pass
    # Cap to limit
    if n > enrich_now_limit:
        state["candidates"] = (state.get("candidates") or [])[:enrich_now_limit]
        n = enrich_now_limit
    # Compute total ICP-matched companies for transparency (not just the preview list)
    async def _count_companies_by_icp(icp: Dict[str, Any]) -> int:
        icp = icp or {}
        industries_param: List[str] = []
        # Back-compat: allow single 'industry' or list 'industries'
        if isinstance(icp.get("industry"), str) and icp.get("industry").strip():
            industries_param.append(icp.get("industry").strip().lower())
        inds = icp.get("industries") or []
        if isinstance(inds, list):
            industries_param.extend([s.strip().lower() for s in inds if isinstance(s, str) and s.strip()])
        industries_param = sorted(set(industries_param))
        emp_min = icp.get("employees_min")
        emp_max = icp.get("employees_max")
        rev_bucket = (
            (icp.get("revenue_bucket") or "").strip().lower()
            if isinstance(icp.get("revenue_bucket"), str)
            else None
        )
        y_min = icp.get("year_min")
        y_max = icp.get("year_max")
        geos = icp.get("geos") or []

        clauses: List[str] = []
        params: List[Any] = []
        if industries_param:
            clauses.append(f"LOWER(industry_norm) = ANY(${len(params)+1})")
            params.append(industries_param)
        if isinstance(emp_min, int):
            clauses.append(f"employees_est >= ${len(params)+1}")
            params.append(emp_min)
        if isinstance(emp_max, int):
            clauses.append(f"employees_est <= ${len(params)+1}")
            params.append(emp_max)
        if rev_bucket in ("small", "medium", "large"):
            clauses.append(f"LOWER(revenue_bucket) = ${len(params)+1}")
            params.append(rev_bucket)
        if isinstance(y_min, int):
            clauses.append(f"incorporation_year >= ${len(params)+1}")
            params.append(y_min)
        if isinstance(y_max, int):
            clauses.append(f"incorporation_year <= ${len(params)+1}")
            params.append(y_max)
        if isinstance(geos, list) and geos:
            geo_like_params: List[str] = []
            geo_subclauses: List[str] = []
            for g in geos:
                if not isinstance(g, str) or not g.strip():
                    continue
                like_val = f"%{g.strip()}%"
                geo_subclauses.append(f"hq_country ILIKE ${len(params)+len(geo_like_params)+1}")
                geo_like_params.append(like_val)
                geo_subclauses.append(f"hq_city ILIKE ${len(params)+len(geo_like_params)+1}")
                geo_like_params.append(like_val)
            if geo_subclauses:
                clauses.append("(" + " OR ".join(geo_subclauses) + ")")
                params.extend(geo_like_params)

        where_clause = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT COUNT(*) FROM companies {where_clause}"
        async with (await get_pg_pool()).acquire() as conn:
            row = await conn.fetchrow(sql, *params)
            try:
                return int(row[0]) if row else 0
            except Exception:
                return 0

    icp_total = 0
    try:
        icp_total = await _count_companies_by_icp(icp)
    except Exception:
        icp_total = n

    # Resolve SSIC by industries (if provided)
    icp = state.get("icp") or {}
    terms = [
        s.strip().lower()
        for s in (icp.get("industries") or [])
        if isinstance(s, str) and s.strip()
    ]
    ssic_matches = []
    msg_lines: list[str] = []

    # Start with candidate count
    msg_lines.append(f"Got {n} companies. ")

    try:
        if terms:
            ssic_matches = _find_ssic_codes_by_terms(terms)
            if ssic_matches:
                top_code, top_title, _ = ssic_matches[0]
                msg_lines.append(
                    f"Matched {len(ssic_matches)} SSIC codes (top: {top_code} {top_title} …)"
                )
            else:
                msg_lines.append("Matched 0 SSIC codes")

            # Fetch ACRA sample
            try:
                codes = {c for (c, _t, _s) in ssic_matches}
                # Count all candidates and fetch a small sample for display
                from src.icp import _count_acra_by_ssic_codes
                total_acra = await asyncio.to_thread(_count_acra_by_ssic_codes, codes)
                rows = await asyncio.to_thread(_select_acra_by_ssic_codes, codes, 10)
            except Exception:
                rows = []
                total_acra = 0
            if total_acra:
                msg_lines.append(f"- Found {total_acra} ACRA candidates. Sample:")
                for r in rows[:2]:
                    uen = (r.get("uen") or "").strip()
                    nm = (r.get("entity_name") or "").strip()
                    code = (r.get("primary_ssic_code") or "").strip()
                    status = (r.get("entity_status_description") or "").strip()
                    msg_lines.append(
                        f"UEN: {uen} – {nm} – SSIC {code} – status: {status}"
                    )
                # Fallback: seed candidates directly from ACRA when none exist
                if n == 0 and not state.get("candidates"):
                    try:
                        derived = []
                        for r in rows[:20]:
                            nm = (r.get("entity_name") or "").strip()
                            if not nm:
                                continue
                            derived.append({"name": nm, "uen": (r.get("uen") or "").strip() or None})
                        if derived:
                            state["candidates"] = derived
                            n = len(derived)
                    except Exception:
                        pass
            else:
                msg_lines.append("- Found 0 ACRA candidates.")
    except Exception:
        # Don’t block on SSIC/ACRA preview errors
        pass

    # Plan enrichment counts: how many now vs later
    do_now = min(n, enrich_now_limit) if n else 0
    if n > 0:
        # Compute nightly remainder from ACRA total when available
        scheduled = max((total_acra if 'total_acra' in locals() else icp_total) - do_now, 0)
        msg_lines.append(
            f"Ready to enrich {do_now} now; {scheduled} scheduled for nightly. Accept a micro‑ICP, then type 'run enrichment' to proceed."
        )
    else:
        msg_lines.append("No candidates yet. I’ll keep collecting ICP details.")
    text = "\n".join([ln for ln in msg_lines if ln])

    # Signal that we've shown the SSIC/ACRA preview
    state["ssic_probe_done"] = True

    # Keep totals in state for later nodes
    state["icp_match_total"] = icp_total
    state["enrich_now_planned"] = do_now
    state["messages"] = add_messages(state.get("messages") or [], [AIMessage(content=text)])
    return state


async def enrich_node(state: GraphState) -> GraphState:
    # Persist current ICP immediately when enrichment is requested, even if user skipped explicit confirm
    try:
        icp_cur = dict(state.get("icp") or {})
        payload = _icp_payload_from_state_icp(icp_cur)
        if payload:
            tid = await _resolve_tenant_id_for_write(state)
            if isinstance(tid, int):
                _save_icp_rule_sync(tid, payload, name="Default ICP")
    except Exception:
        pass

    text = _last_user_text(state)
    if not state.get("candidates"):
        # Prefer sync_head_company_ids captured during chat normalize (10 upserts)
        try:
            ids = state.get("sync_head_company_ids") or []
            if isinstance(ids, list) and ids:
                pool = await get_pg_pool()
                async with pool.acquire() as conn:
                    rows = await conn.fetch(
                        "SELECT company_id AS id, name, uen FROM companies WHERE company_id = ANY($1::int[])",
                        [int(i) for i in ids if isinstance(i, int) or (isinstance(i, str) and str(i).isdigit())],
                    )
                cand = []
                for r in rows:
                    nm = r.get("name") or ""
                    if nm:
                        cand.append({"id": int(r.get("id")), "name": nm, "uen": r.get("uen")})
                if cand:
                    state["candidates"] = cand
        except Exception:
            # Best-effort: fall back to pasted/default candidates below
            pass
    if not state.get("candidates"):
        pasted = _parse_company_list(text)
        if pasted:
            state["candidates"] = [{"name": n} for n in pasted]
        else:
            # If user requested enrichment without pasting names, use ICP-derived suggestions
            try:
                pool = await get_pg_pool()
                cand = await _default_candidates(pool, state.get("icp") or {}, limit=20)
                state["candidates"] = cand
            except Exception as _e:
                # Fall-through to user prompt below
                pass

    candidates = state.get("candidates") or []
    # Cap immediate enrichment to avoid heavy runs in chat context
    try:
        enrich_now_limit = int(os.getenv("CHAT_ENRICH_LIMIT", os.getenv("RUN_NOW_LIMIT", "10") or 10))
    except Exception:
        enrich_now_limit = 10
    if len(candidates) > enrich_now_limit:
        candidates = candidates[:enrich_now_limit]
        state["candidates"] = candidates
    if not candidates:
        # Offer clear next steps when no candidates could be found
        state["messages"] = add_messages(
            state.get("messages") or [],
            [
                AIMessage(
                    content=(
                        "I couldn't find any companies for this ICP. "
                        "Try relaxing employee/geography filters, or paste a few company names (comma-separated)."
                    )
                )
            ],
        )
        return state

    pool = await get_pg_pool()

    # Limit immediate enrichment to a small batch (default 10) and defer the rest to nightly
    try:
        import os
        enrich_now_limit = int(os.getenv("CHAT_ENRICH_LIMIT", os.getenv("RUN_NOW_LIMIT", "10") or 10))
    except Exception:
        enrich_now_limit = 10

    total_candidates = len(candidates)
    if total_candidates > enrich_now_limit:
        # Ensure company rows exist for all candidates so nightly can pick them up later
        ensured_ids: list[int] = []
        for c in candidates:
            try:
                nm = c.get("name") if isinstance(c, dict) else None
                if not nm:
                    continue
                cid = c.get("id") or await _ensure_company_row(pool, nm)
                ensured_ids.append(int(cid))
            except Exception:
                # Best-effort; if ensure fails for some, still proceed with available ones
                pass

        # Choose the first N candidates to process now (preserve current order)
        selected_ids = set(ensured_ids[:enrich_now_limit]) if ensured_ids else set()
        if selected_ids:
            candidates = [c for c in candidates if (c.get("id") in selected_ids) or (not c.get("id") and False)] or candidates[:enrich_now_limit]
        else:
            candidates = candidates[:enrich_now_limit]

    async def _enrich_one(c: Dict[str, Any]) -> Dict[str, Any]:
        name = c["name"]
        cid = c.get("id") or await _ensure_company_row(pool, name)
        uen = c.get("uen")
        final_state = await enrich_company_with_tavily(cid, name, uen)
        completed = (
            bool(final_state.get("completed"))
            if isinstance(final_state, dict)
            else False
        )
        err = None
        try:
            err = final_state.get("error") if isinstance(final_state, dict) else None
        except Exception:
            err = None
        return {"company_id": cid, "name": name, "uen": uen, "completed": completed, "error": err}

    results = await asyncio.gather(*[_enrich_one(c) for c in candidates])
    all_done = all(bool(r.get("completed")) for r in results) if results else False
    state["results"] = results
    state["enrichment_completed"] = all_done

    if all_done:
        # Compose completion message; include ICP total and remainder planned for nightly if available
        icp_total = 0
        try:
            icp_total = int(state.get("icp_match_total") or 0)
        except Exception:
            icp_total = 0
        remaining = max(icp_total - len(results), 0) if icp_total else None
        suffix = (
            f" ICP-matched total: {icp_total}. Remaining scheduled nightly: {remaining}."
            if icp_total
            else " The enrichment pipeline will continue by nightly runner."
        )
        done_msg = f"Enrichment complete for {len(results)} companies." + suffix
        state["messages"] = add_messages(
            state.get("messages") or [],
            [AIMessage(content=done_msg)],
        )
        # Trigger lead scoring pipeline and persist scores for UI consumption
        try:
            # Include all results for scoring, but export to Odoo only for non-skipped rows
            ids = [r.get("company_id") for r in results if r.get("company_id") is not None]
            ids_export = [
                r.get("company_id")
                for r in results
                if r.get("company_id") is not None
                and bool(r.get("completed"))
                and (r.get("error") or "") != "skip_prior_enrichment"
            ]
            if ids:
                scoring_initial_state = {
                    "candidate_ids": ids,
                    "lead_features": [],
                    "lead_scores": [],
                    "icp_payload": {
                        "employee_range": {
                            "min": (state.get("icp") or {}).get("employees_min"),
                            "max": (state.get("icp") or {}).get("employees_max"),
                        },
                        # New: pass-through revenue_bucket and incorporation_year
                        "revenue_bucket": (state.get("icp") or {}).get(
                            "revenue_bucket"
                        ),
                        "incorporation_year": {
                            "min": (state.get("icp") or {}).get("year_min"),
                            "max": (state.get("icp") or {}).get("year_max"),
                        },
                    },
                }
                await lead_scoring_agent.ainvoke(scoring_initial_state)
                # Immediately render scores into chat for better UX
                state = await score_node(state)

                # Best-effort Odoo sync for completed companies (skip ones we skipped enriching)
                try:
                    pool = await get_pg_pool()
                    async with pool.acquire() as conn:
                        comp_rows = await conn.fetch(
                            """
                            SELECT company_id, name, uen, industry_norm, employees_est,
                                   revenue_bucket, incorporation_year, website_domain
                            FROM companies WHERE company_id = ANY($1::int[])
                            """,
                            ids_export,
                        )
                        comps = {r["company_id"]: dict(r) for r in comp_rows}

                        email_rows = await conn.fetch(
                            "SELECT company_id, email FROM lead_emails WHERE company_id = ANY($1::int[])",
                            ids_export,
                        )
                        emails: Dict[int, str] = {}
                        for row in email_rows:
                            cid = row["company_id"]
                            emails.setdefault(cid, row["email"])

                        score_rows = await conn.fetch(
                            "SELECT company_id, score, rationale FROM lead_scores WHERE company_id = ANY($1::int[])",
                            ids_export,
                        )
                        scores = {r["company_id"]: dict(r) for r in score_rows}

                    from app.odoo_store import OdooStore

                    try:
                        try:
                            logger.info("odoo resolve: tenant_id=%s (env DEFAULT_TENANT_ID=%s)", _tid, os.getenv("DEFAULT_TENANT_ID"))
                        except Exception:
                            pass
                        # Resolve tenant for OdooStore in this order:
                        # 1) DEFAULT_TENANT_ID (for non-HTTP runs)
                        # 2) Map DSN path -> odoo_connections.tenant_id
                        # 3) First active mapping in odoo_connections
                        # Prefer tenant_id from state (multi-user safe)
                        _tid_val = state.get("tenant_id") if isinstance(state, dict) else None
                        try:
                            _tid = int(_tid_val) if _tid_val is not None else None
                        except Exception:
                            _tid = None
                        # Env fallback for dev/single-user
                        try:
                            if _tid is None:
                                _tid_env = os.getenv("DEFAULT_TENANT_ID")
                                _tid = int(_tid_env) if _tid_env and _tid_env.isdigit() else None
                        except Exception:
                            _tid = None

                        if _tid is None:
                            try:
                                from src.settings import ODOO_POSTGRES_DSN
                                inferred_db = None
                                if ODOO_POSTGRES_DSN:
                                    from urllib.parse import urlparse
                                    u = urlparse(ODOO_POSTGRES_DSN)
                                    inferred_db = (u.path or "/").lstrip("/") or None
                                if inferred_db:
                                    with get_conn() as _c, _c.cursor() as _cur:
                                        _cur.execute(
                                            "SELECT tenant_id FROM odoo_connections WHERE (db_name=%s OR db_name=%s) AND active=TRUE LIMIT 1",
                                            (inferred_db, ODOO_POSTGRES_DSN),
                                        )
                                        _row = _cur.fetchone()
                                        if _row:
                                            _tid = int(_row[0])
                            except Exception:
                                pass

                        if _tid is None:
                            try:
                                with get_conn() as _c, _c.cursor() as _cur:
                                    _cur.execute("SELECT tenant_id FROM odoo_connections WHERE active=TRUE LIMIT 1")
                                    _row = _cur.fetchone()
                                    if _row:
                                        _tid = int(_row[0])
                            except Exception:
                                pass

                        store = None
                        try:
                            store = OdooStore(tenant_id=_tid)
                        except Exception as _odoo_init_exc:
                            # Fallback DSN from mapping + template
                            try:
                                db_name = None
                                with get_conn() as _c, _c.cursor() as _cur:
                                    if _tid is not None:
                                        _cur.execute(
                                            "SELECT db_name FROM odoo_connections WHERE tenant_id=%s AND active=TRUE LIMIT 1",
                                            (_tid,),
                                        )
                                        _row = _cur.fetchone()
                                        db_name = _row[0] if _row and _row[0] else None
                                    if db_name is None and _tid is None:
                                        _cur.execute(
                                            "SELECT db_name FROM odoo_connections WHERE active=TRUE LIMIT 1"
                                        )
                                        _row = _cur.fetchone()
                                        db_name = _row[0] if _row and _row[0] else None
                                if db_name:
                                    tpl = (os.getenv("ODOO_BASE_DSN_TEMPLATE", "") or "").strip()
                                    if tpl:
                                        dsn = tpl.format(db_name=db_name)
                                    else:
                                        dsn = db_name if str(db_name).startswith("postgresql://") else None
                                    if dsn:
                                        logger.info(
                                            "odoo init: fallback DSN via mapping db=%s (tenant_id=%s)",
                                            db_name,
                                            _tid,
                                        )
                                        store = OdooStore(dsn=dsn)
                            except Exception as _fb_exc:
                                logger.warning("odoo init fallback error: %s", _fb_exc)
                            if store is None:
                                logger.warning("odoo init skipped: %s", _odoo_init_exc)

                    except Exception as _tid_init_block_exc:
                        # Catch-all for any unexpected errors during tenant resolution
                        # and initial OdooStore creation so the outer block can continue.
                        logger.warning("odoo init block error: %s", _tid_init_block_exc)
                        store = None

                    if store:
                        for cid in ids_export:
                            comp = comps.get(cid, {})
                            if not comp:
                                continue
                            score = scores.get(cid) or {}
                            email = emails.get(cid)
                            try:
                                odoo_id = await store.upsert_company(
                                    comp.get("name"),
                                    comp.get("uen"),
                                    industry_norm=comp.get("industry_norm"),
                                    employees_est=comp.get("employees_est"),
                                    revenue_bucket=comp.get("revenue_bucket"),
                                    incorporation_year=comp.get("incorporation_year"),
                                    website_domain=comp.get("website_domain"),
                                )
                                try:
                                    logger.info(
                                        "odoo export: upsert company partner_id=%s name=%s",
                                        odoo_id,
                                        comp.get("name"),
                                    )
                                except Exception:
                                    pass
                                if email:
                                    try:
                                        await store.add_contact(odoo_id, email)
                                        logger.info(
                                            "odoo export: contact added email=%s partner_id=%s",
                                            email,
                                            odoo_id,
                                        )
                                    except Exception as _c_exc:
                                        logger.warning(
                                            "odoo export: add_contact failed email=%s err=%s",
                                            email,
                                            _c_exc,
                                        )
                                try:
                                    await store.merge_company_enrichment(odoo_id, {})
                                except Exception:
                                    pass
                                if "score" in score:
                                    try:
                                        await store.create_lead_if_high(
                                            odoo_id,
                                            comp.get("name"),
                                            float(score.get("score") or 0.0),
                                            {},
                                            str(score.get("rationale") or ""),
                                            email,
                                        )
                                    except Exception as _lead_exc:
                                        logger.warning(
                                            "odoo export: create_lead failed partner_id=%s err=%s",
                                            odoo_id,
                                            _lead_exc,
                                        )
                            except Exception as exc:
                                logger.exception(
                                    "odoo sync failed for company_id=%s", cid
                                )
                except Exception as _odoo_exc:
                    logger.exception("odoo sync block failed")
        except Exception as _score_exc:
            logger.exception("lead scoring failed")
    else:
        done = sum(1 for r in results if r.get("completed"))
        total = len(results)
        state["messages"] = add_messages(
            state.get("messages") or [],
            [
                AIMessage(
                    content=f"Enrichment finished with issues ({done}/{total} completed). I’ll wait to score until all complete."
                )
            ],
        )
    return state


def _fmt_table(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "No candidates found."
    headers = ["Name", "Domain", "Industry", "Employees", "Score", "Bucket", "Rationale", "Contact"]
    md = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for r in rows:
        rationale = str(r.get("lead_rationale", ""))
        md.append(
            "| "
            + " | ".join([
                str(r.get("name", "")),
                str(r.get("domain", "")),
                str(r.get("industry", "")),
                str(r.get("employee_count", "")),
                str(r.get("lead_score", "")),
                str(r.get("lead_bucket", "")),
                rationale,
                str(r.get("contact_email", "")),
            ])
            + " |"
        )
    return "\n".join(md)


async def score_node(state: GraphState) -> GraphState:
    pool = await get_pg_pool()
    cands = state.get("candidates") or []
    ids = [c.get("id") for c in cands if c.get("id") is not None]
    # Fallback: derive ids from enrichment results if candidates lack ids
    if not ids:
        results = state.get("results") or []
        ids = [r.get("company_id") for r in results if r.get("company_id") is not None]

    if not ids:
        table = _fmt_table([])
        state["messages"] = add_messages(
            state.get("messages") or [],
            [AIMessage(content=f"Here are your leads:\n\n{table}")],
        )
        return state

    async with pool.acquire() as conn:
        # 1) Fetch latest scores for the candidate IDs
        score_rows = await conn.fetch(
            f"""
            SELECT company_id, score, bucket, rationale
            FROM public.{LEAD_SCORES_TABLE}
            WHERE company_id = ANY($1::int[])
            """,
            ids,
        )
        by_score = {r["company_id"]: dict(r) for r in score_rows}

        # 2) Fetch fresh company fields to display up-to-date values
        comp_rows = await conn.fetch(
            """
            SELECT company_id, name, website_domain, industry_norm, employees_est
            FROM public.companies
            WHERE company_id = ANY($1::int[])
            """,
            ids,
        )
        by_comp = {r["company_id"]: dict(r) for r in comp_rows}

        # 3) Fetch a contact email when available
        email_rows = await conn.fetch(
            "SELECT company_id, email FROM public.lead_emails WHERE company_id = ANY($1::int[])",
            ids,
        )
        by_email = {}
        for _er in email_rows:
            _cid = _er["company_id"]
            if _cid not in by_email:
                by_email[_cid] = _er.get("email")

    # 3) Merge fresh company data with scores, preserving candidate order
    scored: List[Dict[str, Any]] = []
    for c in cands:
        cid = c.get("id")
        comp = by_comp.get(cid, {})
        sc = by_score.get(cid)
        # Build row with refreshed fields; fallback to existing candidate values if missing
        row: Dict[str, Any] = {
            "id": cid,
            "name": comp.get("name") or c.get("name"),
            "domain": comp.get("website_domain") or c.get("domain"),
            "industry": comp.get("industry_norm") or c.get("industry"),
            "employee_count": (
                comp.get("employees_est")
                if comp.get("employees_est") is not None
                else c.get("employee_count")
            ),
            "contact_email": (by_email.get(cid) if "by_email" in locals() else None) or c.get("email") or "",
        }
        if sc:
            row["lead_score"] = sc.get("score")
            row["lead_bucket"] = sc.get("bucket")
            row["lead_rationale"] = sc.get("rationale")
        scored.append(row)

    state["scored"] = scored
    table = _fmt_table(scored)
    state["messages"] = add_messages(
        state.get("messages") or [],
        [AIMessage(content=f"Here are your leads:\n\n{table}")],
    )
    return state


# ------------------------------
# Router
# ------------------------------


def router(state: GraphState) -> str:
    msgs = state.get("messages") or []
    icp = state.get("icp") or {}

    text = _last_user_text(state).lower()

    # Boot-session initialization: record current message count so we only honor
    # commands after a NEW human message arrives post-boot.
    try:
        if state.get("boot_init_token") != BOOT_TOKEN:
            state["boot_init_token"] = BOOT_TOKEN
            state["boot_seen_messages_len"] = len(msgs)
        else:
            # On subsequent router cycles, if new messages appended and last is Human → mark as fresh user action
            if len(msgs) > int(state.get("boot_seen_messages_len") or 0):
                last = msgs[-1] if msgs else None
                def _is_human(m) -> bool:
                    try:
                        if isinstance(m, HumanMessage):
                            return True
                        if isinstance(m, dict):
                            role = (m.get("type") or m.get("role") or "").lower()
                            return role in {"human", "user"}
                    except Exception:
                        return False
                    return False
                if last and _is_human(last):
                    state["last_user_boot_token"] = BOOT_TOKEN
                state["boot_seen_messages_len"] = len(msgs)
    except Exception:
        pass

    # Do not auto-run anything if assistant spoke last, unless user explicitly typed a new command.
    # This prevents enrichment from resuming on server restart. Nightly jobs should invoke nodes directly.
    if _last_is_ai(msgs) and "run enrichment" not in text and not re.search(r"\baccept\s+micro[- ]icp\b", text):
        logger.info("router -> end (assistant last; no explicit user action)")
        return "end"

    # Enrichment gating: require required Fast-Start fields; otherwise route back to ICP Q&A
    if (
        "run enrichment" in text
        and state.get("candidates")
        and not state.get("results")
    ):
        icp_f = dict(state.get("icp") or {})
        asks = dict(state.get("ask_counts") or {})
        if ENABLE_ICP_INTAKE and not _icp_required_fields_done(icp_f, asks):
            logger.info("router -> icp (ask remaining required Fast-Start fields before enrichment)")
            return "icp"
        logger.info("router -> enrich (explicit user override)")
        return "enrich"

    # Accept micro‑ICP selection
    if re.search(r"\baccept\s+micro[- ]icp\b", text):
        logger.info("router -> accept (user accepted micro‑ICP)")
        return "accept"

    # 1) Pipeline progression
    # Finder gating: do NOT auto-advance to enrichment until micro-ICP suggestions are done
    if ENABLE_ICP_INTAKE and not state.get("finder_suggestions_done"):
        # If we already have candidates but haven't finished Finder suggestions, hold and wait for user
        if state.get("candidates") and not state.get("results"):
            logger.info("router -> end (Finder: hold before enrichment until suggestions)")
            return "end"
    # Additionally require explicit selection (or user override) before enrichment
    if ENABLE_ICP_INTAKE and not state.get("micro_icp_selected") and "run enrichment" not in text:
        if state.get("candidates") and not state.get("results"):
            logger.info("router -> end (Finder: wait for micro‑ICP acceptance)")
            return "end"
    has_candidates = bool(state.get("candidates"))
    has_results = bool(state.get("results"))
    has_scored = bool(state.get("scored"))
    enrichment_completed = bool(state.get("enrichment_completed"))

    # Only progress when a human spoke last. This stops auto-run after server restarts.
    if has_candidates and not has_results and not _last_is_ai(msgs):
        logger.info("router -> enrich (have candidates, no enrichment)")
        return "enrich"
    if has_results and enrichment_completed and not has_scored:
        logger.info("router -> score (have enrichment, no scores, all completed)")
        return "score"
    if has_results and not enrichment_completed and not has_scored:
        logger.info("router -> end (enrichment not fully completed)")
        return "end"

    # 2) If assistant spoke last and no pending work, wait for user input (covered by early guard)

    # 3) Fast-path: user requested enrichment
    if "run enrichment" in text:
        # Finder context: if suggestions not finished yet, hold until they are done
        if ENABLE_ICP_INTAKE and not state.get("finder_suggestions_done"):
            logger.info("router -> end (Finder: hold until suggestions are done before run enrichment)")
            return "end"
        # Proceed with enrichment when candidates exist, regardless of ICP completeness
        if state.get("candidates"):
            logger.info("router -> enrich (user requested enrichment)")
            return "enrich"
        logger.info("router -> candidates (prepare candidates before enrichment)")
        return "candidates"

    # 4) If user pasted an explicit company list, jump to candidates
    # Avoid misclassifying comma-separated industry/geo lists as companies.
    pasted = _parse_company_list(text)
    # Only jump early if at least one looks like a domain or multi-word name
    if pasted and any(("." in n) or (" " in n) for n in pasted):
        if ENABLE_ICP_INTAKE and not _icp_complete(icp):
            logger.info("router -> icp (Finder gating: ignore explicit list until ICP set)")
            return "icp"
        logger.info("router -> candidates (explicit company list)")
        return "candidates"

    # 5) User said confirm: proceed forward once (avoid loops)
    if _user_just_confirmed(state):
        # Finder: if minimal intake present (website + seeds), allow confirm pipeline even if core ICP incomplete
        if ENABLE_ICP_INTAKE:
            has_minimal = bool(icp.get("website_url")) and bool(icp.get("seeds_list"))
            if has_minimal:
                logger.info("router -> confirm (Finder minimal intake present; proceeding to confirm pipeline)")
                return "confirm"
            # Otherwise continue asking for missing pieces
            if not _icp_complete(icp):
                logger.info("router -> icp (Finder gating: need website + seeds before confirm)")
                return "icp"
        # If we already derived candidates, move ahead to enrichment; else collect candidates first.
        if state.get("candidates"):
            logger.info("router -> enrich (user confirmed ICP; have candidates)")
            return "enrich"
        logger.info("router -> candidates (user confirmed ICP)")
        return "candidates"

    # 6) If ICP is not complete yet, continue ICP Q&A
    if not _icp_complete(icp):
        logger.info("router -> icp (need more ICP)")
        return "icp"

    # 7) Default
    logger.info("router -> icp (default)")
    return "icp"


def router_entry(state: GraphState) -> GraphState:
    """No-op node so we can attach conditional edges to a central router hub."""
    return state


# ------------------------------
# Graph builder
# ------------------------------


@log_node("accept")
def accept_micro_icp(state: GraphState) -> GraphState:
    """Persist selected micro‑ICP (basic SSIC-based payload) and unlock enrichment."""
    text = _last_user_text(state)
    # Parse selection number
    m = re.search(r"accept\s+micro[- ]icp\s*(\d+)", text, flags=re.IGNORECASE)
    idx = int(m.group(1)) if m else None
    suggestions = state.get("micro_icp_suggestions") or []
    if not suggestions or not isinstance(suggestions, list):
        state["messages"] = add_messages(state.get("messages") or [], [AIMessage("No micro‑ICP suggestions available yet. Type 'show micro-icp'.")])
        return state
    if not idx or idx < 1 or idx > len(suggestions):
        lines = ["Please specify which micro‑ICP to accept (e.g., 'accept micro-icp 1').", "Options:"]
        for i, it in enumerate(suggestions, 1):
            title = it.get("title") or it.get("id")
            ev = it.get("evidence_count") or 0
            lines.append(f"{i}) {title} (evidence: {ev})")
        state["messages"] = add_messages(state.get("messages") or [], [AIMessage("\n".join(lines))])
        return state
    choice = suggestions[idx - 1]
    title = choice.get("title") or choice.get("id") or f"Micro-ICP {idx}"
    # Derive a simple payload: support SSIC code suggestions id='ssic:CODE'
    payload: Dict[str, Any] = {}
    cid = (choice.get("id") or "").strip().lower()
    if cid.startswith("ssic:"):
        payload["ssic_codes"] = [cid.split(":", 1)[1]]
    # Persist rule best-effort
    try:
        tid = _resolve_tenant_id_for_write_sync(state)
        if isinstance(tid, int):
            _save_icp_rule_sync(tid, payload or {"note": title}, name=f"{title}")
    except Exception:
        pass
    # Unlock enrichment
    state["micro_icp_selected"] = True
    state["messages"] = add_messages(state.get("messages") or [], [AIMessage(f"Accepted: {title}. You can now type 'run enrichment' to proceed.")])
    return state

def build_graph():
    g = StateGraph(GraphState)
    # Central router node (no-op) to hub all control flow
    g.add_node("router", router_entry)
    g.add_node("icp", icp_node)
    g.add_node("candidates", candidates_node)
    g.add_node("confirm", confirm_node)
    g.add_node("enrich", enrich_node)
    g.add_node("score", score_node)
    # Accept micro‑ICP selection node
    g.add_node("accept", accept_micro_icp)
    # Central router: every node returns here so we can advance the workflow
    mapping = {
        "icp": "icp",
        "candidates": "candidates",
        "confirm": "confirm",
        "accept": "accept",
        "enrich": "enrich",
        "score": "score",
        "end": END,
    }
    # Start in the router so we always decide the right first step
    g.set_entry_point("router")
    g.add_conditional_edges("router", router, mapping)
    # Every worker node loops back to the router
    for node in ("icp", "candidates", "confirm", "accept", "enrich", "score"):
        g.add_edge(node, "router")
    return g.compile()


GRAPH = build_graph()
