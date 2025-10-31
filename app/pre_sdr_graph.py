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
    from src.settings import ENABLE_AGENT_DISCOVERY  # type: ignore
except Exception:  # pragma: no cover
    ENABLE_AGENT_DISCOVERY = False  # type: ignore
try:
    # Optional LLM agents (PRD19 §6)
    from src.agents_icp import icp_synthesizer as _agent_icp_synth  # type: ignore
    from src.agents_icp import discovery_planner as _agent_plan_discovery  # type: ignore
except Exception:  # pragma: no cover
    _agent_icp_synth = None  # type: ignore
    _agent_plan_discovery = None  # type: ignore
try:
    from src.settings import ENABLE_ICP_INTAKE  # type: ignore
except Exception:  # pragma: no cover
    ENABLE_ICP_INTAKE = False  # type: ignore
try:
    from src.settings import ICP_WIZARD_FAST_START_ONLY  # type: ignore
except Exception:  # pragma: no cover
    ICP_WIZARD_FAST_START_ONLY = True  # type: ignore
try:
    from src.settings import ENABLE_ACRA_IN_CHAT  # type: ignore
except Exception:  # pragma: no cover
    ENABLE_ACRA_IN_CHAT = False  # type: ignore
try:
    from src.settings import STRICT_DDG_ONLY  # type: ignore
except Exception:  # pragma: no cover
    STRICT_DDG_ONLY = True  # type: ignore
try:
    # Lead‑gen conversational Q&A helpers
    from src.conversation_agent import answer_leadgen_question as _qa_answer  # type: ignore
except Exception:  # pragma: no cover
    _qa_answer = None  # type: ignore

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


# --- Helper: announce completion for queued background next-40 jobs ---
def _announce_completed_bg_jobs(state) -> None:
    try:
        pend = list(state.get("pending_bg_jobs") or [])
        if not pend:
            return
        done_ids: list[int] = []
        msgs: list[str] = []
        with get_conn() as conn, conn.cursor() as cur:
            for jid in pend:
                try:
                    cur.execute(
                        "SELECT status, processed, total, error, params FROM background_jobs WHERE job_id=%s",
                        (int(jid),),
                    )
                    r = cur.fetchone()
                    if not r:
                        continue
                    status, processed, total, error, params = r
                    if str(status or "").lower() != "done":
                        continue
                    # Build A/B/C bucket summary from lead_scores for the job's company_ids
                    ids = []
                    try:
                        ids = [int(i) for i in ((params or {}).get("company_ids") or []) if str(i).strip()]
                    except Exception:
                        ids = []
                    counts: dict[str, int] = {"A": 0, "B": 0, "C": 0}
                    if ids:
                        cur.execute(
                            "SELECT bucket, COUNT(*) FROM lead_scores WHERE company_id = ANY(%s) GROUP BY bucket",
                            (ids,),
                        )
                        for br in cur.fetchall() or []:
                            b, c = (br[0] or "C"), int(br[1] or 0)
                            counts[str(b)] = c
                    err_txt = f" error={error}" if error else ""
                    msgs.append(
                        f"Background enrichment finished (job {jid}). Processed {int(processed or 0)}/{int(total or 0)}. "
                        f"Buckets: A={counts.get('A',0)}, B={counts.get('B',0)}, C={counts.get('C',0)}.{err_txt}"
                    )
                    done_ids.append(int(jid))
                except Exception:
                    continue
        if msgs:
            for m in msgs:
                state["messages"] = add_messages(state.get("messages") or [], [AIMessage(content=m)])
        if done_ids:
            try:
                state["pending_bg_jobs"] = [j for j in pend if int(j) not in set(done_ids)]
            except Exception:
                state["pending_bg_jobs"] = [j for j in pend if j not in done_ids]
    except Exception:
        return


# --- Helper: enqueue Non‑SG next‑40 after Top‑10 enrichment completes ---
def _enqueue_next40_if_applicable(state) -> None:
    try:
        # Avoid duplicate enqueue within the same conversation thread
        if bool(state.get("next40_enqueued")):
            return
        # Always attempt to enqueue the next 40 for background enrichment
        # (no longer gated by Non‑SG only)
        from src.database import get_conn as __get_conn
        from src.jobs import enqueue_web_discovery_bg_enrich as __enqueue_bg
        # Resolve tenant id from state/env/mapping
        try:
            tid = _resolve_tenant_id_for_write_sync(state)
        except Exception:
            tid = None
        _tidv = int(tid) if isinstance(tid, int) else None
        if _tidv is None:
            try:
                with __get_conn() as _c, _c.cursor() as _cur:
                    _cur.execute("SELECT tenant_id FROM odoo_connections WHERE active=TRUE LIMIT 1")
                    _r = _cur.fetchone()
                    _tidv = int(_r[0]) if _r and _r[0] is not None else None
            except Exception:
                _tidv = None
        if _tidv is None:
            return
        bg_limit = 40
        try:
            bg_limit = int((os.getenv("BG_NEXT_COUNT") or "40")) or 40
        except Exception:
            bg_limit = 40
        doms: list[str] = []
        preview_total = 0
        preview_doms: list[str] = []
        with __get_conn() as __c2, __c2.cursor() as __cur2:
            try:
                # Fetch latest preview batch_id for this tenant (most recent preview row)
                __cur2.execute(
                    """
                    SELECT ai_metadata->>'batch_id'
                      FROM staging_global_companies
                     WHERE tenant_id=%s AND COALESCE((ai_metadata->>'preview')::boolean,false)=true
                     ORDER BY created_at DESC
                     LIMIT 1
                    """,
                    (_tidv,),
                )
                row_bid = __cur2.fetchone()
                batch_id = (row_bid and row_bid[0]) or None
                preview_doms = []
                if batch_id:
                    __cur2.execute(
                        """
                        SELECT domain
                          FROM staging_global_companies
                         WHERE tenant_id=%s AND COALESCE((ai_metadata->>'preview')::boolean,false)=true
                           AND ai_metadata->>'batch_id'=%s
                         ORDER BY created_at DESC
                         LIMIT 10
                        """,
                        (_tidv, batch_id),
                    )
                    _p = __cur2.fetchall() or []
                    preview_doms = [str(r[0]) for r in _p if r and r[0]]
                    __cur2.execute(
                        """
                        SELECT domain
                          FROM staging_global_companies
                         WHERE tenant_id=%s
                           AND source = 'web_discovery'
                           AND COALESCE((ai_metadata->>'preview')::boolean,false)=false
                           AND ai_metadata->>'batch_id'=%s
                           AND NOT LOWER(domain) = ANY(%s)
                         ORDER BY created_at DESC
                         LIMIT %s
                        """,
                        (_tidv, batch_id, [d.lower() for d in preview_doms] or [''], bg_limit),
                    )
                else:
                    # Fallback for older runs without batch_id (use recent preview as exclusion)
                    __cur2.execute(
                        """
                        SELECT domain
                          FROM staging_global_companies
                         WHERE tenant_id=%s AND COALESCE((ai_metadata->>'preview')::boolean,false)=true
                         ORDER BY created_at DESC
                         LIMIT 10
                        """,
                        (_tidv,),
                    )
                    _p = __cur2.fetchall() or []
                    preview_doms = [str(r[0]) for r in _p if r and r[0]]
                    __cur2.execute(
                        """
                        SELECT domain
                          FROM staging_global_companies
                         WHERE tenant_id=%s
                           AND source = 'web_discovery'
                           AND COALESCE((ai_metadata->>'preview')::boolean,false)=false
                           AND NOT LOWER(domain) = ANY(%s)
                         ORDER BY created_at DESC
                         LIMIT %s
                        """,
                        (_tidv, [d.lower() for d in preview_doms] or [''], bg_limit),
                    )
                rows = __cur2.fetchall() or []
                doms = [str(r[0]) for r in rows if r and r[0]]
                # Compute preview_total within this batch when available
                if batch_id:
                    __cur2.execute(
                        "SELECT COUNT(*) FROM staging_global_companies WHERE tenant_id=%s AND COALESCE((ai_metadata->>'preview')::boolean,false)=true AND ai_metadata->>'batch_id'=%s",
                        (_tidv, batch_id),
                    )
                else:
                    __cur2.execute(
                        "SELECT COUNT(*) FROM staging_global_companies WHERE tenant_id=%s AND COALESCE((ai_metadata->>'preview')::boolean,false)=true",
                        (_tidv,),
                    )
                cr = __cur2.fetchone()
                preview_total = int(cr[0] or 0) if cr else 0
            except Exception:
                doms = []
        if not doms:
            return
        # Map to company_ids
        bg_ids: list[int] = []
        with __get_conn() as __c3, __c3.cursor() as __cur3:
            try:
                __cur3.execute(
                    "SELECT company_id, website_domain FROM companies WHERE LOWER(website_domain) = ANY(%s)",
                    ([d.lower() for d in doms],),
                )
                _rows = __cur3.fetchall() or []
                _map = {str((r[1] or "").lower()): int(r[0]) for r in _rows if r and r[0] is not None}
                for d in doms:
                    key = str(d.lower())
                    _cid = _map.get(key)
                    if _cid:
                        bg_ids.append(_cid)
                    else:
                        # Ensure a companies row exists for this domain, then include it
                        try:
                            ensured = _ensure_company_by_domain(d)
                            if ensured:
                                bg_ids.append(int(ensured))
                        except Exception:
                            continue
            except Exception:
                bg_ids = []
        if not bg_ids:
            return
        job = __enqueue_bg(int(_tidv), bg_ids)
        jid = (job or {}).get("job_id")
        # Verify that the background job row exists for additional certainty
        try:
            with __get_conn() as _vc, _vc.cursor() as _vcur:
                _vcur.execute(
                    "SELECT job_type, status FROM background_jobs WHERE job_id=%s",
                    (int(jid) if jid else -1,),
                )
                _v = _vcur.fetchone()
                verified = bool(_v)
        except Exception:
            verified = False
        try:
            logger.info(
                "[enrich] enqueue next40 count=%s job_id=%s (preview_total=%s tenant_id=%s verified=%s) ids_head=%s",
                str(len(bg_ids)),
                str(jid),
                str(preview_total),
                str(_tidv),
                str(verified),
                ",".join([str(i) for i in (bg_ids[:5] if isinstance(bg_ids, list) else [])]),
            )
        except Exception:
            pass
        # Track and inform user in chat
        pend = list(state.get("pending_bg_jobs") or [])
        if jid:
            pend.append(int(jid))
        state["pending_bg_jobs"] = pend
        state["next40_enqueued"] = True
        state["messages"] = add_messages(
            state.get("messages") or [],
            [AIMessage(content=f"I’m enriching the next {min(len(bg_ids), bg_limit)} in the background (job {jid}). I’ll reply here when it’s done. You can also check /jobs/{jid}.")],
        )
    except Exception:
        # best-effort; do not block
        return


def _is_non_sg_active_profile(state: Dict[str, Any]) -> bool:
    """Heuristic to decide Non‑SG based on icp_profile geos/hq_country.

    Returns True if we cannot positively detect Singapore as an active geo.
    """
    try:
        prof = dict(state.get("icp_profile") or {})
        geos = prof.get("geos") or []
        if isinstance(geos, list):
            for g in geos:
                if isinstance(g, str) and g.strip().lower() == "singapore":
                    return False
        hq = (prof.get("hq_country") or "").strip().lower()
        if hq == "singapore":
            return False
    except Exception:
        pass
    return True

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
STAGING_GLOBAL_TABLE = os.getenv("STAGING_GLOBAL_TABLE", "staging_global_companies")


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
    # Next: explicit DEFAULT_TENANT_ID from current session env (set by /session/odoo_info)
    try:
        _tid_env = os.getenv("DEFAULT_TENANT_ID")
        if _tid_env and _tid_env.isdigit():
            tid_env = int(_tid_env)
            # Validate it's an active odoo mapping
            with get_conn() as _c, _c.cursor() as _cur:
                _cur.execute("SELECT 1 FROM odoo_connections WHERE tenant_id=%s AND active=TRUE", (tid_env,))
                if _cur.fetchone():
                    return tid_env
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


def _merge_icp_profile_into_payload(payload: dict, icp_profile: dict) -> dict:
    """Merge LLM-derived icp_profile keys into icp_rules payload structure.

    Adds: industries, integrations, buyer_titles, triggers, size_bands (if present).
    """
    out = dict(payload or {})
    prof = dict(icp_profile or {})
    def _arr(key: str) -> list[str]:
        vals = prof.get(key) or []
        if not isinstance(vals, list):
            return []
        return [str(v).strip() for v in vals if isinstance(v, str) and v.strip()]
    for k in ("industries", "integrations", "buyer_titles", "triggers", "size_bands"):
        arr = _arr(k)
        if arr:
            out[k] = sorted(set(arr))
    return out


def _ensure_company_by_domain(domain: str) -> int | None:
    """Ensure a companies row exists for an apex domain; return company_id or None."""
    try:
        dom = (domain or "").strip().lower()
        if not dom:
            return None
        # strip protocol and path
        import re as _re
        dom = dom.replace("http://", "").replace("https://", "")
        if dom.startswith("www."):
            dom = dom[4:]
        for sep in ["/", "?", "#"]:
            if sep in dom:
                dom = dom.split(sep, 1)[0]
        with get_conn() as _c, _c.cursor() as _cur:
            _cur.execute("SELECT company_id FROM companies WHERE website_domain=%s", (dom,))
            r = _cur.fetchone()
            if r and r[0] is not None:
                return int(r[0])
            _cur.execute(
                "INSERT INTO companies(name, website_domain, last_seen) VALUES (%s,%s,NOW()) RETURNING company_id",
                (dom, dom),
            )
            rr = _cur.fetchone()
            return int(rr[0]) if rr and rr[0] is not None else None
    except Exception:
        return None


def _clean_snippet(s: str) -> str:
    try:
        t = (s or "").strip()
        if not t:
            return ""
        import re as _re
        # Drop common r.jina metadata and JSON-y fragments
        t = _re.sub(r"\b(Title:|URL Source:|Published Time:|Markdown Content:|Warning:)\b.*?\s+", "", t, flags=_re.I)
        t = _re.sub(r"\{[^}]{0,200}\}", " ", t)  # small JSON blobs
        t = " ".join(t.split())
        return t[:180]
    except Exception:
        return (s or "")[:180]


def _fmt_top10_md(rows: list[dict]) -> str:
    if not rows:
        return "No lookalikes found."
    hdr = ["#", "Domain", "Score", "Why", "Snippet"]
    out = [
        "| " + " | ".join(hdr) + " |",
        "|" + "|".join(["---"] * len(hdr)) + "|",
    ]
    for idx, r in enumerate(rows, 1):
        dom = str(r.get("domain") or "")
        score = str(int(r.get("score") or 0))
        why = str(r.get("why") or "")
        snip = _clean_snippet(str(r.get("snippet") or ""))
        out.append(f"| {idx} | {dom} | {score} | {why} | {snip} |")
    return "\n".join(out)


def _stash_top10_in_thread_memory(state: dict, top: list[dict]) -> None:
    """Store Top-10 domains in thread memory (message additional_kwargs) for reuse.

    We avoid relying on env DEFAULT_TENANT_ID by keeping shortlist in-thread.
    """
    try:
        # Keep only first 10; store minimal fields to reduce payload size
        rows = []
        for it in (top or [])[:10]:
            if not isinstance(it, dict):
                continue
            dom = (it.get("domain") or "").strip().lower()
            if not dom:
                continue
            rows.append({
                "domain": dom,
                "score": it.get("score"),
                "why": it.get("why"),
                "snippet": it.get("snippet"),
            })
        if not rows:
            return
        mem_msg = AIMessage(content="", additional_kwargs={"top10_memory": rows})
        state["messages"] = add_messages(state.get("messages") or [], [mem_msg])
    except Exception:
        # Never block routing on memory write
        pass


def _load_top10_from_thread_memory(state: dict) -> list[dict]:
    """Read Top-10 shortlist from thread memory messages, newest-first."""
    try:
        for msg in reversed(state.get("messages") or []):
            if not isinstance(msg, AIMessage):
                continue
            try:
                extra = getattr(msg, "additional_kwargs", {}) or {}
            except Exception:
                extra = {}
            mem = extra.get("top10_memory")
            if isinstance(mem, list) and mem:
                out = []
                for it in mem:
                    if isinstance(it, dict) and (it.get("domain")):
                        out.append(it)
                    elif isinstance(it, str):
                        out.append({"domain": it})
                if out:
                    return out
    except Exception:
        return []
    return []


def _save_icp_rule_sync(tid: int, payload: dict, name: str = "Default ICP") -> None:
    # Upsert an ICP rule row for this tenant; rely on RLS via GUC
    with get_conn() as conn, conn.cursor() as cur:
        try:
            cur.execute("SELECT set_config('request.tenant_id', %s, true)", (str(tid),))
        except Exception:
            pass
        try:
            keys = list((payload or {}).keys())
            logger.info("[icp_rules] upsert → tenant_id=%s name=%s keys=%s", tid, name, keys[:8])
        except Exception:
            pass
        try:
            cur.execute(
                """
                INSERT INTO icp_rules(tenant_id, name, payload)
                VALUES (%s, %s, %s)
                ON CONFLICT (tenant_id, name) DO UPDATE SET
                  payload = EXCLUDED.payload,
                  created_at = NOW()
                """,
                (tid, name, Json(payload)),
            )
        except Exception as _up_e:
            try:
                logger.warning("[icp_rules] upsert failed tenant_id=%s name=%s err=%s", tid, name, _up_e)
            except Exception:
                pass
            raise
        try:
            logger.info("[icp_rules] upsert OK → tenant_id=%s name=%s", tid, name)
        except Exception:
            pass


def _upsert_icp_profile_sync(tid: int, icp_profile: dict, name: str = "Default ICP") -> None:
    """Merge/update icp_profile fields into current icp_rules payload for tenant.

    Keys merged: industries, integrations, buyer_titles, size_bands, triggers.
    """
    try:
        try:
            logger.info(
                "[icp_rules] merge profile → tenant_id=%s name=%s icp_profile_keys=%s",
                tid,
                name,
                list((icp_profile or {}).keys())[:8],
            )
        except Exception:
            pass
        base: dict = {}
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT payload FROM icp_rules WHERE tenant_id=%s AND name=%s ORDER BY created_at DESC LIMIT 1",
                (tid, name),
            )
            row = cur.fetchone()
            if row and row[0]:
                if isinstance(row[0], dict):
                    base = dict(row[0])
        merged = _merge_icp_profile_into_payload(base, dict(icp_profile or {}))
        try:
            logger.info(
                "[icp_rules] merged payload keys → tenant_id=%s name=%s keys=%s",
                tid,
                name,
                list(merged.keys())[:8],
            )
        except Exception:
            pass
        _save_icp_rule_sync(tid, merged, name=name)
    except Exception:
        # Non-fatal; ignore if table missing or RLS blocks
        pass


def _resolve_tenant_id_for_write_sync(state: dict) -> Optional[int]:
    try:
        v = state.get("tenant_id") if isinstance(state, dict) else None
        if v is not None:
            return int(v)
    except Exception:
        pass
    # Next: explicit DEFAULT_TENANT_ID from current session env (set by /session/odoo_info)
    try:
        _tid_env = os.getenv("DEFAULT_TENANT_ID")
        if _tid_env and _tid_env.isdigit():
            tid_env = int(_tid_env)
            with get_conn() as _c, _c.cursor() as _cur:
                _cur.execute("SELECT 1 FROM odoo_connections WHERE tenant_id=%s AND active=TRUE", (tid_env,))
                if _cur.fetchone():
                    return tid_env
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


def _persist_web_candidates_to_staging(
    domains: list[str],
    tenant_id: Optional[int] = None,
    ai_metadata: Optional[dict] = None,
    per_domain_meta: Optional[dict[str, dict]] = None,
) -> int:
    """Persist discovered global (non‑SG) candidate domains into a lightweight staging table.

    - Table: staging_global_companies(id PK, tenant_id, domain, source, created_at, ai_metadata JSONB)
    - Idempotent on (tenant_id, domain, source) via ON CONFLICT DO NOTHING.
    - Accepts a single `ai_metadata` blob (applied to all) and/or a per-domain metadata map.
    """
    if not domains:
        return 0
    ds = [str(d).strip().lower() for d in domains if isinstance(d, str) and d.strip()]
    ds = sorted(set(ds))
    if not ds:
        return 0
    try:
        with get_conn() as conn, conn.cursor() as cur:
            # Set tenant GUC for RLS consistency (even if staging has no RLS, keep uniform)
            try:
                if isinstance(tenant_id, int):
                    cur.execute("SELECT set_config('request.tenant_id', %s, true)", (str(tenant_id),))
            except Exception:
                pass
            # Ensure table exists (dev-safe) including ai_metadata column
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {STAGING_GLOBAL_TABLE} (
                  id BIGSERIAL PRIMARY KEY,
                  tenant_id BIGINT NULL,
                  domain TEXT NOT NULL,
                  source TEXT NOT NULL DEFAULT 'web_discovery',
                  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                  ai_metadata JSONB NOT NULL DEFAULT '{{}}',
                  UNIQUE (tenant_id, domain, source)
                )
                """
            )
            rows = 0
            for d in ds:
                try:
                    meta = (per_domain_meta or {}).get(d) if isinstance(per_domain_meta, dict) else None
                    if not isinstance(meta, dict):
                        meta = dict(ai_metadata or {}) if isinstance(ai_metadata, dict) else {}
                    # Ensure a small provenance trail at minimum
                    if "provenance" not in meta:
                        meta["provenance"] = {"agent": "agents_icp.discovery", "stage": "staging"}
                    cur.execute(
                        f"""
                        INSERT INTO {STAGING_GLOBAL_TABLE}(tenant_id, domain, source, ai_metadata)
                        VALUES (%s,%s,'web_discovery',%s)
                        ON CONFLICT DO NOTHING
                        """,
                        (tenant_id, d, Json(meta)),
                    )
                    rows += cur.rowcount if isinstance(cur.rowcount, int) else 0
                except Exception:
                    # continue best-effort
                    pass
            return rows
    except Exception:
        return 0


def _persist_top10_preview(tid: Optional[int], top: list[dict]) -> None:
    """Persist Top‑10 preview into staging, companies, icp_evidence, and lead_scores."""
    try:
        # Staging with per-domain ai_metadata (score/why/snippet provenance)
        per_meta: dict[str, dict] = {}
        # Detect a lead_profile from any item (agents may attach it), else omit
        _lead_profile = None
        for it in (top or [])[:10]:
            try:
                lp = it.get("lead_profile") if isinstance(it, dict) else None
                if isinstance(lp, str) and lp.strip():
                    _lead_profile = lp.strip()
                    break
            except Exception:
                continue
        for it in (top or [])[:10]:
            if not isinstance(it, dict):
                continue
            dom = (it.get("domain") or "").strip().lower()
            if not dom:
                continue
            per_meta[dom] = {
                "preview": True,
                "score": it.get("score"),
                "bucket": it.get("bucket"),
                "why": it.get("why"),
                "snippet": (it.get("snippet") or "")[:200],
                "provenance": {"agent": "agents_icp.plan_top10", "stage": "preview"},
                **({"lead_profile": _lead_profile} if _lead_profile else {}),
            }
        # Persist Top‑10 preview rows
        _ = _persist_web_candidates_to_staging(
            [str(it.get("domain")) for it in (top or [])[:10] if isinstance(it, dict) and it.get("domain")],
            int(tid) if isinstance(tid, int) else None,
            ai_metadata={"provenance": {"agent": "agents_icp.plan_top10"}, **({"lead_profile": _lead_profile} if _lead_profile else {})},
            per_domain_meta=per_meta,
        )
        # Persist remainder (beyond Top‑10) as non‑preview rows so Next‑40 can be enqueued reliably
        rest = [
            str(it.get("domain")).strip().lower()
            for it in (top[10:] if isinstance(top, list) and len(top) > 10 else [])
            if isinstance(it, dict) and it.get("domain")
        ]
        if rest:
            _persist_web_candidates_to_staging(
                rest,
                int(tid) if isinstance(tid, int) else None,
                ai_metadata={"provenance": {"agent": "agents_icp.plan_top10", "stage": "staging"}, **({"lead_profile": _lead_profile} if _lead_profile else {})},
            )
    except Exception:
        pass
    try:
        with get_conn() as _c, _c.cursor() as _cur:
            # Ensure tenant GUC is set for RLS writes
            try:
                if isinstance(tid, int):
                    _cur.execute("SELECT set_config('request.tenant_id', %s, true)", (str(tid),))
            except Exception:
                pass
            for it in (top or [])[:10]:
                dom = (it.get("domain") or "").strip().lower() if isinstance(it, dict) else ""
                if not dom:
                    continue
                _cur.execute("SELECT company_id, name FROM companies WHERE website_domain=%s", (dom,))
                r = _cur.fetchone()
                if r and r[0] is not None:
                    cid = int(r[0])
                else:
                    _cur.execute(
                        "INSERT INTO companies(name, website_domain, last_seen) VALUES (%s,%s,NOW()) RETURNING company_id",
                        (dom, dom),
                    )
                    cid = int((_cur.fetchone() or [None])[0])
                why = it.get("why") or ""
                snip = it.get("snippet") or ""
                try:
                    _cur.execute(
                        "INSERT INTO icp_evidence(tenant_id, company_id, signal_key, value, source) VALUES (%s,%s,%s,%s,'web_preview')",
                        (int(tid) if isinstance(tid, int) else None, cid, "top10_preview", Json({"why": why, "snippet": snip})),
                    )
                except Exception:
                    pass
                try:
                    score = float(it.get("score") or 0)
                except Exception:
                    score = 0.0
                bucket = it.get("bucket") or ("A" if score >= 70 else ("B" if score >= 50 else "C"))
                rationale = why or snip
                try:
                    _cur.execute(
                        """
                        INSERT INTO lead_scores(company_id, score, bucket, rationale, cache_key)
                        VALUES (%s,%s,%s,%s,NULL)
                        ON CONFLICT (company_id) DO UPDATE SET
                          score = EXCLUDED.score,
                          bucket = EXCLUDED.bucket,
                          rationale = EXCLUDED.rationale
                        """,
                        (cid, score, bucket, rationale),
                    )
                except Exception:
                    pass
    except Exception:
        pass


def _load_persisted_top10(tid: Optional[int]) -> list[dict]:
    """Load recently persisted Top‑10 preview.

    Preference:
    1) If tenant_id is known, read from `icp_evidence` + `lead_scores` for that tenant (RLS requires tenant).
    2) If nothing found (or tenant unknown), read from `staging_global_companies` where ai_metadata.preview=true,
       filtered by tenant when available; otherwise read the latest preview rows (dev-safe fallback).
    """
    try:
        with get_conn() as _c, _c.cursor() as _cur:
            # 1) Tenant-scoped evidence table (requires tenant for RLS)
            if isinstance(tid, int):
                try:
                    _cur.execute("SELECT set_config('request.tenant_id', %s, true)", (str(tid),))
                except Exception:
                    pass
                try:
                    _cur.execute(
                        """
                        SELECT c.website_domain AS domain,
                               COALESCE(ls.score,0) AS score,
                               COALESCE(ls.bucket,'C') AS bucket,
                               (ev.value->>'why') AS why,
                               (ev.value->>'snippet') AS snippet
                          FROM icp_evidence ev
                          JOIN companies c ON c.company_id = ev.company_id
                          LEFT JOIN lead_scores ls ON ls.company_id = ev.company_id
                         WHERE ev.tenant_id = %s
                           AND ev.signal_key = 'top10_preview'
                         ORDER BY ev.created_at DESC
                         LIMIT 10
                        """,
                        (tid,),
                    )
                    rows = _cur.fetchall() or []
                    out: list[dict] = []
                    for r in rows:
                        d = {
                            "domain": (r[0] or "").strip().lower(),
                            "score": float(r[1] or 0),
                            "bucket": r[2] or "C",
                            "why": r[3] or "",
                            "snippet": r[4] or "",
                        }
                        if d["domain"]:
                            out.append(d)
                    if out:
                        return out
                except Exception:
                    # fall through to staging fallback
                    pass

            # 2) Fallback: staging preview rows (dev-safe; may be global if tenant unknown)
            try:
                _cur.execute(
                    f"""
                    SELECT domain,
                           COALESCE((ai_metadata->>'score')::float, 0) AS score,
                           COALESCE((ai_metadata->>'bucket'), 'C') AS bucket,
                           (ai_metadata->>'why') AS why,
                           (ai_metadata->>'snippet') AS snippet
                      FROM {STAGING_GLOBAL_TABLE}
                     WHERE (tenant_id = %s OR %s IS NULL)
                       AND COALESCE((ai_metadata->>'preview')::boolean, false) = true
                     ORDER BY created_at DESC
                     LIMIT 10
                    """,
                    (tid if isinstance(tid, int) else None, None if tid is None else tid),
                )
                rows2 = _cur.fetchall() or []
                out2: list[dict] = []
                for r in rows2:
                    d = {
                        "domain": (r[0] or "").strip().lower(),
                        "score": float(r[1] or 0),
                        "bucket": r[2] or "C",
                        "why": r[3] or "",
                        "snippet": r[4] or "",
                    }
                    if d["domain"]:
                        out2.append(d)
                return out2
            except Exception:
                return []
    except Exception:
        return []


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
                                "List 5–15 best customers (Company — website). Optionally 2–3 lost/churned with a short reason."
                            )
                    )
                    return state
            # Optionally run LLM ICP synthesizer to augment inferred profile
            try:
                if ENABLE_AGENT_DISCOVERY and _agent_icp_synth is not None:
                    s = {
                        "icp_profile": dict(state.get("icp") or {}),
                        "seeds": [{"url": u, "snippet": ""} for u in (icp.get("seeds_list") or [])],
                    }
                    icp_prof = _agent_icp_synth(s).get("icp_profile")
                    if icp_prof:
                        state["icp_profile"] = icp_prof
                        # Inform the user in chat once the AI agents have synthesized the ICP
                        try:
                            ind = ", ".join((icp_prof.get("industries") or [])[:3]) or "n/a"
                            titles = ", ".join((icp_prof.get("buyer_titles") or [])[:3]) or "n/a"
                            sizes = ", ".join((icp_prof.get("size_bands") or [])[:3]) or "n/a"
                            sigs = ", ".join((icp_prof.get("integrations") or [])[:3]) or "n/a"
                            trig = ", ".join((icp_prof.get("triggers") or [])[:3]) or "n/a"
                            lines = [
                                "ICP profile ready.",
                                f"- Industries: {ind}",
                                f"- Buyer titles: {titles}",
                                f"- Size bands: {sizes}",
                                f"- Integrations/signals: {sigs}",
                                f"- Triggers: {trig}",
                                "Reply confirm to proceed or edit any field.",
                            ]
                            state["messages"].append(AIMessage("\n".join(lines)))
                        except Exception:
                            pass
            except Exception:
                pass
            # Ready to confirm
            state["messages"].append(
                AIMessage(
                    "Thanks! I’ll crawl your site + seed sites, plan web discovery, and propose a Top‑10 with evidence. ACRA is only used later in the nightly SG pass. Reply confirm to proceed, or add more seeds."
                )
            )
            return state
    except Exception:
        pass

    state["messages"].append(
        AIMessage("Great. Reply **confirm** to save, or tell me what to change.")
    )
    return state


@log_node("confirm")
def icp_confirm(state: PreSDRState) -> PreSDRState:
    # Announce any completed background jobs (next-40) from prior turns
    try:
        _announce_completed_bg_jobs(state)
    except Exception:
        pass
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
        # Merge any synthesized icp_profile fields into payload before persist
        try:
            prof = dict(state.get("icp_profile") or {})
            if prof:
                payload = _merge_icp_profile_into_payload(payload, prof)
        except Exception:
            pass
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

    # Optionally plan agent-driven discovery candidates for preview (non-blocking)
    try:
        if ENABLE_AGENT_DISCOVERY and _agent_plan_discovery is not None:
            # Plan discovery and then apply compliance guard before persisting
            s = {"icp_profile": state.get("icp_profile") or {}}
            planned = _agent_plan_discovery(s)
            acand = planned.get("discovery_candidates") or []
            # Try compliance guard if available
            try:
                from src.agents_icp import compliance_guard as _agent_compliance_guard  # type: ignore
            except Exception:
                _agent_compliance_guard = None  # type: ignore
            if _agent_compliance_guard is not None and acand:
                try:
                    guard_in = dict(planned)
                    guarded = _agent_compliance_guard(guard_in) or {}
                    gcand = guarded.get("discovery_candidates") or acand
                    acand = [str(d).strip().lower() for d in gcand if isinstance(d, str) and d.strip()]
                except Exception:
                    pass
            # Bounded one-shot re-plan when guard prunes all and relaxation is allowed
            try:
                if not acand and (os.getenv("ENABLE_DDG_RELAX", "").lower() in {"1","true","yes"}):
                    logger.info("[guard] replan requested: relaxing site filter once")
                    # Re-run planner once with same icp_profile
                    planned2 = _agent_plan_discovery({"icp_profile": state.get("icp_profile") or {}})
                    guarded2 = _agent_compliance_guard(planned2) if _agent_compliance_guard else planned2
                    ac2 = (guarded2 or {}).get("discovery_candidates") or []
                    acand = [str(d).strip().lower() for d in ac2 if isinstance(d, str) and d.strip()]
            except Exception:
                pass
            if acand:
                state["agent_candidates"] = acand
                # Persist all discovered (guarded) domains into staging for audit/queueing
                try:
                    tid = _resolve_tenant_id_for_write_sync(state)
                    icp_prof = dict(state.get("icp_profile") or {})
                    icp_keys = sorted([k for k, v in icp_prof.items() if v])
                    meta = {"provenance": {"agent": "agents_icp.discovery_planner", "stage": "plan"}, "icp_profile_keys": icp_keys[:8]}
                    added = _persist_web_candidates_to_staging([str(d) for d in acand if isinstance(d, str)], tid, ai_metadata=meta)
                    state["web_discovery_total"] = len(acand)
                    logger.info(
                        "[discovery] found_total=%d persisted=%d table=%s",
                        len(acand),
                        int(added or 0),
                        STAGING_GLOBAL_TABLE,
                    )
                except Exception as _pe:
                    try:
                        logger.info("[candidates] staging persist skipped: %s", _pe)
                    except Exception:
                        pass
    except Exception:
        pass
    # Build a concise status message with correct totals
    msg_lines: list[str] = []
    # Compute discovered candidates (domain URLs) count for display
    # Prefer in-memory discovery results strictly for accuracy per run
    discovered_total = 0
    try:
        ac = state.get("agent_candidates") or []
        if isinstance(ac, list) and ac:
            discovered_total = len(ac)
    except Exception:
        discovered_total = 0
    if discovered_total <= 0:
        try:
            at = state.get("agent_top10") or []
            if isinstance(at, list) and at:
                discovered_total = len(at)
        except Exception:
            discovered_total = 0
    if discovered_total <= 0:
        # Last resort: use a cached count if present; avoid DB/staging totals to prevent mismatch
        try:
            discovered_total = int(state.get("web_discovery_total") or 0)
        except Exception:
            discovered_total = 0
    display_total = discovered_total
    if display_total > 0:
        msg_lines.append(f"Found {display_total} ICP candidates. Showing Top‑10 preview below.")
    else:
        msg_lines.append("Collecting ICP candidates…")

    # Web discovery Top‑10 (agent-driven) — present to the user; move ACRA to nightly
    try:
        web_cand = state.get("agent_candidates") or []
        if isinstance(web_cand, list) and web_cand:
            msg_lines.append(f"Planned web discovery: found {len(web_cand)} candidate domains.")
            show = web_cand[:10]
            for i, dom in enumerate(show, 1):
                try:
                    d = dom if isinstance(dom, str) else str(dom)
                    msg_lines.append(f"{i}) {d}")
                except Exception:
                    continue
            msg_lines.append("I’ll use these to extract evidence and score fit.")
    except Exception:
        pass

    # Compute and display Top‑10 with "why" lines
    try:
        if ENABLE_AGENT_DISCOVERY:
            try:
                from src.agents_icp import plan_top10_with_reasons as _agent_top10  # type: ignore
            except Exception:
                _agent_top10 = None  # type: ignore
            if _agent_top10 is not None:
                top = _agent_top10(state.get("icp_profile") or {}, state.get("tenant_id"))
                if top:
                    # Store full planned set in memory; UI will still show Top‑10
                    state["agent_top10"] = top
                    # Persist preview so later runs reuse the same Top‑10
                    try:
                        tid = _resolve_tenant_id_for_write_sync(state)
                    except Exception:
                        tid = None
                    _persist_top10_preview(int(tid) if isinstance(tid, int) else None, top)
                    # Also persist the icp_profile we just used/derived so it evolves over time
                    try:
                        if isinstance(tid, int):
                            _upsert_icp_profile_sync(tid, state.get("icp_profile") or {}, name="Default ICP")
                    except Exception:
                        pass
                    # Pretty Top‑10 table
                    try:
                        table = _fmt_top10_md(top)
                        # Include total so users see "10 of N"
                        _tot = display_total if isinstance(display_total, int) and display_total > 0 else len(top)
                        msg_lines.append(f"Top-listed lookalikes (with why) — showing 10 of {_tot}:\n\n" + table)
                    except Exception:
                        # Fallback to simple lines
                        _tot = display_total if isinstance(display_total, int) and display_total > 0 else 10
                        msg_lines.append(f"Top‑listed lookalikes (with why) — showing 10 of {_tot}:")
                        for i, row in enumerate(top, 1):
                            dom = row.get("domain")
                            why = row.get("why") or "signal match"
                            score = int(row.get("score") or 0)
                            snip = _clean_snippet(row.get("snippet") or "")
                            if snip:
                                msg_lines.append(f"{i}) {dom} — {why} (score {score}) — {snip}")
                            else:
                                msg_lines.append(f"{i}) {dom} — {why} (score {score})")
                    # Avoid a second discovery pass: skip ensure_icp_enriched_with_jina here.
                    # Discovery just ran to produce Top‑10; calling enrichment would re-trigger DDG.
                    try:
                        logger.info("[confirm] Skipping ICP enrichment-from-discovery to prevent duplicate DDG runs")
                    except Exception:
                        pass
                    # After showing results, show a detailed ICP Profile summary for the user
                    try:
                        icp_prof = state.get("icp_profile") or {}
                        inds = (icp_prof.get("industries") or [])
                        titles_l = (icp_prof.get("buyer_titles") or [])
                        sizes_l = (icp_prof.get("size_bands") or [])
                        ints = (icp_prof.get("integrations") or [])
                        trigs = (icp_prof.get("triggers") or [])
                        msg_lines.append("ICP Profile")
                        msg_lines.append(f"- Industries: {', '.join(inds[:6]) if inds else 'n/a'}")
                        msg_lines.append(f"- Buyer titles: {', '.join(titles_l[:6]) if titles_l else 'n/a'}")
                        msg_lines.append(f"- Company sizes: {', '.join(sizes_l[:6]) if sizes_l else 'n/a'}")
                        sigs_arr = (ints or []) + (trigs or [])
                        msg_lines.append(f"- Signals: {', '.join(sigs_arr[:8]) if sigs_arr else 'n/a'}")
                        # Persist ICP profile to icp_rules for reuse across threads/API
                        try:
                            if isinstance(tid, int):
                                base = _icp_payload_from_state_icp(state.get("icp") or {})
                                merged = _merge_icp_profile_into_payload(base, icp_prof)
                                if merged:
                                    _save_icp_rule_sync(int(tid), merged, name="Default ICP")
                                    logger.info("[confirm] Persisted ICP profile (icp_rules) with keys=%s", list(merged.keys()))
                        except Exception:
                            pass
                    except Exception:
                        msg_lines.append("ICP Profile")
    except Exception:
        pass

    # Planned enrichment counts
    try:
        enrich_now_limit = int(os.getenv("CHAT_ENRICH_LIMIT", os.getenv("RUN_NOW_LIMIT", "10") or 10))
    except Exception:
        enrich_now_limit = 10
    # If we have an agent Top‑10, prefer that count to set expectations accurately
    try:
        top_for_count = state.get("agent_top10") or []
        if isinstance(top_for_count, list) and top_for_count:
            do_now = min(len(top_for_count), enrich_now_limit)
        else:
            do_now = min(n, enrich_now_limit) if n else 0
    except Exception:
        do_now = min(n, enrich_now_limit) if n else 0
        if n > 0:
            # Prefer ACRA total by suggested SSICs, else by matched terms, else company total
            nightly = 0
            try:
                # Prefer stored ACRA total from earlier SSIC probe
                try:
                    acra_total_state = int(state.get("acra_total_suggested") or 0)
                except Exception:
                    acra_total_state = 0
                if acra_total_state:
                    nightly = max(int(acra_total_state) - do_now, 0)
                else:
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
            f"We can enrich {do_now} companies now. The nightly runner will process the remaining ICP companies. Type 'run enrichment' after accepting a micro‑ICP."
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
                # Merge any previously learned icp_profile fields as well
                prof = dict(state.get("icp_profile") or {})
                if prof:
                    try:
                        _upsert_icp_profile_sync(tid, prof, name="Default ICP")
                    except Exception:
                        pass
                _save_icp_rule_sync(tid, payload, name="Default ICP")
    except Exception:
        pass

    candidates = state.get("candidates") or []
    # Force strict Top‑10 when a persisted Top‑10 preview exists for this tenant
    try:
        tid_for_read = _resolve_tenant_id_for_write_sync(state)
    except Exception:
        tid_for_read = None
    try:
        if not state.get("strict_top10"):
            persisted_top = _load_persisted_top10(int(tid_for_read) if isinstance(tid_for_read, int) else None)
            if isinstance(persisted_top, list) and persisted_top:
                # Map domains to company rows (create if missing)
                from src.database import get_conn as __get_conn
                mapped: list[dict] = []
                with __get_conn() as _c2, _c2.cursor() as _cur2:
                    for it in persisted_top[:10]:
                        try:
                            dom = (it.get("domain") or "").strip().lower()
                            if not dom:
                                continue
                            _cur2.execute("SELECT company_id, name FROM companies WHERE website_domain=%s", (dom,))
                            r = _cur2.fetchone()
                            if r and r[0] is not None:
                                cid = int(r[0]); name = (r[1] or dom)
                            else:
                                _cur2.execute(
                                    "INSERT INTO companies(name, website_domain, last_seen) VALUES (%s,%s,NOW()) RETURNING company_id",
                                    (dom, dom),
                                )
                                cid = int((_cur2.fetchone() or [None])[0]); name = dom
                            mapped.append({"id": cid, "name": name})
                        except Exception:
                            continue
                if mapped:
                    candidates = mapped
                    state["candidates"] = mapped
                    state["strict_top10"] = True
    except Exception:
        pass
    # Ensure: if a Top‑10 list from web discovery exists, enrich exactly those first.
    if isinstance(state.get("agent_top10"), list) and state.get("agent_top10"):
        try:
            from src.database import get_conn as _get_conn
            with _get_conn() as _c, _c.cursor() as _cur:
                cand: list[dict] = []
                for it in (state.get("agent_top10") or [])[:10]:
                    dom = (it.get("domain") or "").strip().lower() if isinstance(it, dict) else ""
                    if not dom:
                        continue
                    _cur.execute("SELECT company_id, name FROM companies WHERE website_domain=%s", (dom,))
                    row = _cur.fetchone()
                    if row and row[0] is not None:
                        cid = int(row[0])
                        name = (row[1] or dom)
                    else:
                        _cur.execute(
                            "INSERT INTO companies(name, website_domain, last_seen) VALUES (%s,%s,NOW()) RETURNING company_id",
                            (dom, dom),
                        )
                        cid = int((_cur.fetchone() or [None])[0])
                        name = dom
                    cand.append({"id": cid, "name": name})
                if cand:
                    candidates = cand
                    state["candidates"] = cand
                    state["strict_top10"] = True
                    # Non‑SG: enqueue background next‑40 enrichment and inform user (best effort)
                    try:
                        if not _is_non_sg_active_profile(state):
                            raise RuntimeError("active profile is SG; skip next-40 background")
                        from src.database import get_conn as __get_conn
                        from src.jobs import enqueue_web_discovery_bg_enrich as __enqueue_bg
                        bg_limit = 40
                        try:
                            import os as __os
                            bg_limit = int(__os.getenv("BG_NEXT_COUNT", "40") or 40)
                        except Exception:
                            bg_limit = 40
                        # Load next‑40 preview domains from staging (ordered by preview score)
                        doms: list[str] = []
                        with __get_conn() as __c2, __c2.cursor() as __cur2:
                            try:
                                # Resolve tenant id for scoping (reuse DEFAULT_TENANT_ID or first active mapping)
                                _tid_env = os.getenv("DEFAULT_TENANT_ID")
                                _tidv = int(_tid_env) if _tid_env and _tid_env.isdigit() else None
                            except Exception:
                                _tidv = None
                            if _tidv is None:
                                try:
                                    __cur2.execute("SELECT tenant_id FROM odoo_connections WHERE active=TRUE LIMIT 1")
                                    _r = __cur2.fetchone()
                                    _tidv = int(_r[0]) if _r and _r[0] is not None else None
                                except Exception:
                                    _tidv = None
                            if _tidv is not None:
                                __cur2.execute(
                                    """
                                    SELECT domain
                                    FROM staging_global_companies
                                    WHERE tenant_id=%s AND COALESCE((ai_metadata->>'preview')::boolean,false)=true
                                    ORDER BY COALESCE((ai_metadata->>'score')::float,0) DESC
                                    OFFSET 10 LIMIT %s
                                    """,
                                    (_tidv, bg_limit),
                                )
                                _r2 = __cur2.fetchall() or []
                                doms = [str(r[0]) for r in _r2 if r and r[0]]
                                try:
                                    __cur2.execute(
                                        """
                                        SELECT COUNT(*) FROM staging_global_companies
                                        WHERE tenant_id=%s AND COALESCE((ai_metadata->>'preview')::boolean,false)=true
                                        """,
                                        (_tidv,),
                                    )
                                    _cnt_row = __cur2.fetchone()
                                    _preview_total = int(_cnt_row[0] or 0) if _cnt_row else 0
                                except Exception:
                                    _preview_total = 0
                        # Map to company_ids
                        bg_ids: list[int] = []
                        if doms:
                            with __get_conn() as __c3, __c3.cursor() as __cur3:
                                __cur3.execute(
                                    "SELECT company_id, website_domain FROM companies WHERE LOWER(website_domain) = ANY(%s)",
                                    ([d.lower() for d in doms],),
                                )
                                _rows = __cur3.fetchall() or []
                                _map = {str((r[1] or "").lower()): int(r[0]) for r in _rows if r and r[0] is not None}
                                for d in doms:
                                    _cid = _map.get(str(d.lower()))
                                    if _cid:
                                        bg_ids.append(_cid)
                        if bg_ids and _tidv is not None:
                            _job = __enqueue_bg(int(_tidv), bg_ids)
                            _jid = (_job or {}).get("job_id")
                            try:
                                logger.info(
                                    "[next40] preview_total=%s enrich_now=%s queued_next=%s job_id=%s tenant_id=%s",
                                    str(_preview_total if '_preview_total' in locals() else '?'),
                                    "10",
                                    str(len(bg_ids)),
                                    str(_jid),
                                    str(_tidv),
                                )
                            except Exception:
                                pass
                            # Track pending background job in state for later completion announcement
                            try:
                                pend = list(state.get("pending_bg_jobs") or [])
                                if _jid:
                                    pend.append(int(_jid))
                                state["pending_bg_jobs"] = pend
                            except Exception:
                                pass
                            state["messages"] = add_messages(
                                state.get("messages") or [],
                                [AIMessage(content=f"Enriching the next {min(len(bg_ids), bg_limit)} in the background (job {_jid}). I will reply here when it finishes. You can also check status via /jobs/{_jid}.")],
                            )
                    except Exception:
                        pass
        except Exception:
            pass
    # If Top‑10 was shown earlier but agent_top10 is missing in this state (new cycle/thread),
    # load the last persisted Top‑10 preview and use it strictly for enrichment.
    if not candidates:
        try:
            tid = _resolve_tenant_id_for_write_sync(state)
        except Exception:
            tid = None
        tid_int = int(tid) if isinstance(tid, int) else None
        try:
            persisted_top = _load_persisted_top10(tid_int)
        except Exception:
            persisted_top = []
        if not persisted_top:
            regenerated_top = await _regenerate_top10_if_missing(state, tid_int)
            if regenerated_top:
                persisted_top = regenerated_top
        if persisted_top:
            try:
                from src.database import get_conn as _get_conn
                with _get_conn() as _c, _c.cursor() as _cur:
                    cand: list[dict] = []
                    for it in persisted_top[:10]:
                        dom = (it.get("domain") or "").strip().lower() if isinstance(it, dict) else ""
                        if not dom:
                            continue
                        _cur.execute("SELECT company_id, name FROM companies WHERE website_domain=%s", (dom,))
                        row = _cur.fetchone()
                        if row and row[0] is not None:
                            cid = int(row[0])
                            name = (row[1] or dom)
                        else:
                            _cur.execute(
                                "INSERT INTO companies(name, website_domain, last_seen) VALUES (%s,%s,NOW()) RETURNING company_id",
                                (dom, dom),
                            )
                            cid = int((_cur.fetchone() or [None])[0])
                            name = dom
                        cand.append({"id": cid, "name": name})
                    if cand:
                        candidates = cand
                        state["candidates"] = cand
                        state["strict_top10"] = True
                        _persist_top10_preview(tid_int, persisted_top)
            except Exception:
                pass
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
        # Strict policy: do not enrich arbitrary companies when Top‑10 is missing
        try:
            # Helpful diagnostics for operators (logs only)
            try:
                _tid_dbg = _resolve_tenant_id_for_write_sync(state)
                _dbg_top = _load_persisted_top10(int(_tid_dbg) if isinstance(_tid_dbg, int) else None)
                logger.info("[enrich] strict Top-10 missing; tenant_id=%s persisted_top=%d", _tid_dbg, len(_dbg_top or []))
            except Exception:
                pass
            state["messages"] = add_messages(
                state.get("messages") or [],
                [AIMessage(content="I couldn’t find the last Top‑10 shortlist to enrich. Please type ‘confirm’ to regenerate and lock a Top‑10, then try 'run enrichment' again.")],
            )
        except Exception:
            pass
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

    # Best-effort: enqueue remainder for nightly based on accepted micro‑ICP SSIC titles (only once)
    try:
        if not state.get("nightly_enqueued"):
            sugg = state.get("micro_icp_suggestions") or []
            codes: list[str] = []
            for it in (sugg if isinstance(sugg, list) else []):
                sid = (it.get("id") or "") if isinstance(it, dict) else ""
                if isinstance(sid, str) and sid.lower().startswith("ssic:"):
                    code = sid.split(":", 1)[1]
                    if code and code.strip():
                        codes.append(code.strip())
            titles: list[str] = []
            if codes:
                # Lookup human-readable titles for SSIC codes
                from src.database import get_conn as _get_conn
                with _get_conn() as _c, _c.cursor() as _cur:
                    _cur.execute(
                        "SELECT title FROM ssic_ref WHERE regexp_replace(code::text,'\\D','','g') = ANY(%s::text[])",
                        (codes,),
                    )
                    titles = [r[0] for r in (_cur.fetchall() or []) if r and r[0]]
            if titles:
                try:
                    from src.jobs import enqueue_staging_upsert as _enqueue
                    _enqueue(None, titles)
                    state["nightly_enqueued"] = True
                except Exception:
                    pass
    except Exception:
        pass

    async def _enrich_one(c: Dict[str, Any]) -> Dict[str, Any]:
        name = c["name"]
        cid = c.get("id") or await _ensure_company_row(pool, name)
        uen = c.get("uen")
        await enrich_company_with_tavily(cid, name, uen, search_policy="require_existing")
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

    odoo_upserts = 0
    odoo_contacts = 0
    odoo_leads = 0
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
            try:
                odoo_upserts += 1 if isinstance(odoo_id, int) and odoo_id > 0 else 0
            except Exception:
                pass
            if email:
                try:
                    await store.add_contact(odoo_id, email)
                    logger.info("odoo export: contact added email=%s for partner_id=%s", email, odoo_id)
                    odoo_contacts += 1
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
                    odoo_leads += 1
                except Exception as _lead_exc:
                    logger.warning("odoo export: create_lead failed partner_id=%s err=%s", odoo_id, _lead_exc)
        except Exception as exc:
            logger.exception("odoo sync failed for company_id=%s", cid)

    # Present a scored shortlist table in chat (top by score)
    try:
        logger.info("[odoo] export summary upserts=%s contacts=%s leads=%s", odoo_upserts, odoo_contacts, odoo_leads)
    except Exception:
        pass
    try:
        scored_rows = [scores.get(cid) for cid in ids if scores.get(cid)]
        # Fallback: if in-memory scores are empty, try DB lead_scores
        if not scored_rows:
            try:
                async with pool.acquire() as conn:
                    rows = await conn.fetch(
                        "SELECT company_id, score::float, bucket FROM lead_scores WHERE company_id = ANY($1::int[])",
                        ids,
                    )
                for r in rows:
                    scored_rows.append({
                        "company_id": int(r["company_id"]),
                        "score": float(r["score"] or 0.0),
                        "bucket": (r["bucket"] or "").strip() or "-",
                    })
            except Exception:
                scored_rows = []
        scored_rows.sort(key=lambda r: float(r.get("score") or 0.0), reverse=True)
        head = scored_rows[: min(10, len(scored_rows))]
        if head:
            try:
                logger.info("[shortlist] rendering scored table rows=%s", len(head))
            except Exception:
                pass
            lines = ["Lead Shortlist (top scored):", "", "| Company | Score | Bucket |", "|---|---:|:---:|"]
            for r in head:
                cid = int(r.get("company_id")) if r.get("company_id") is not None else None
                comp = comps.get(cid, {}) if cid is not None else {}
                nm = comp.get("name") or f"Company {cid}"
                sc = f"{float(r.get('score') or 0):.1f}"
                bk = (r.get("bucket") or "").strip() or "-"
                lines.append(f"| {nm} | {sc} | {bk} |")
            state["messages"].append(AIMessage("\n".join(lines)))
        else:
            # Last fallback: present a simple table without scores
            lines = ["Lead Shortlist:", "", "| Company |", "|---|"]
            for cid in ids[: min(10, len(ids))]:
                nm = (comps.get(cid, {}) or {}).get("name") or f"Company {cid}"
                lines.append(f"| {nm} |")
            state["messages"].append(AIMessage("\n".join(lines)))
    except Exception:
        state["messages"].append(
            AIMessage(f"Enrichment complete for {len(results)} companies.")
        )
    return state


def _boot_guard_route(state: PreSDRState) -> bool:
    """Return True if we should halt routing until a fresh human message after boot.

    Mirrors the boot-resume guard used in the LLM router, so both simple and
    LLM-driven graphs honor the same behavior across server restarts.
    """
    try:
        msgs = state.get("messages") or []
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
        if state.get("boot_init_token") != BOOT_TOKEN:
            state["boot_init_token"] = BOOT_TOKEN
            state["boot_seen_messages_len"] = len(msgs)
            last = msgs[-1] if msgs else None
            # Halt only if there isn't a fresh human message
            if not (last and _is_human(last)):
                logger.info("router -> end (boot resume guard: waiting for new user input)")
                return True
            # Fresh human: clear duplicate guard to allow routing
            state["last_user_boot_token"] = BOOT_TOKEN
            try:
                if "last_routed_text" in state:
                    del state["last_routed_text"]
            except Exception:
                pass
        else:
            if len(msgs) > int(state.get("boot_seen_messages_len") or 0):
                last = msgs[-1] if msgs else None
                if last and _is_human(last):
                    state["last_user_boot_token"] = BOOT_TOKEN
                    try:
                        if "last_routed_text" in state:
                            del state["last_routed_text"]
                    except Exception:
                        pass
                state["boot_seen_messages_len"] = len(msgs)
    except Exception:
        return False
    return False


def route(state: PreSDRState) -> str:
    # Enforce boot-resume guard for the simple router as well
    if _boot_guard_route(state):
        return "end"
    text = _last_text(state.get("messages")).lower()
    # Map common start intents to ICP intake
    if re.search(r"\b(start\s+lead\s*gen|start\s+leadgen|start\s+lead\s+generation|find\s+leads)\b", text):
        logger.info("router -> icp (explicit start intent)")
        try:
            state["last_routed_text"] = text
            state["icp_in_progress"] = True
        except Exception:
            pass
        return "icp"
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

from src.settings import LANGCHAIN_MODEL
QUESTION_LLM = ChatOpenAI(model=LANGCHAIN_MODEL, temperature=0.2)
EXTRACT_LLM = ChatOpenAI(model=LANGCHAIN_MODEL, temperature=0)

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


def _top10_preview_was_sent(state: GraphState) -> bool:
    """Return True if an AI message already presented a Top-10 lookalike table."""
    try:
        for msg in reversed(state.get("messages") or []):
            if not isinstance(msg, AIMessage):
                continue
            text = _to_text(getattr(msg, "content", "") or "").strip().lower()
            if not text:
                continue
            if "top-10 lookalikes" in text or "top‑10 lookalikes" in text:
                return True
    except Exception:
        pass
    return False


async def _regenerate_top10_if_missing(
    state: GraphState, tenant_id: Optional[int]
) -> List[Dict[str, Any]]:
    """Rebuild the Top-10 list when the preview was shown but persistence missed it."""
    if not _top10_preview_was_sent(state):
        return []
    try:
        from src.agents_icp import plan_top10_with_reasons as _agent_top10  # type: ignore
    except Exception as _imp_err:  # pragma: no cover - import edge case
        logger.info("[top10] regeneration skipped: %s", _imp_err)
        return []
    icp_prof = dict(state.get("icp_profile") or {})
    try:
        regenerated = await asyncio.to_thread(_agent_top10, icp_prof, tenant_id)
    except Exception as _regen_exc:
        logger.info("[top10] regeneration failed: %s", _regen_exc)
        return []
    if isinstance(regenerated, list) and regenerated:
        state["agent_top10"] = regenerated
        try:
            _persist_top10_preview(tenant_id, regenerated)
        except Exception:
            pass
        return regenerated
    return []


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
            raw = (getattr(m, "content", "") or "").strip().lower()
            # normalize punctuation
            import re as _re
            txt = _re.sub(r"[\s.!?]+$", "", raw)
            # accept common confirmations
            confirmed_set = {
                "confirm",
                "confirmed",
                "yes",
                "y",
                "ok",
                "okay",
                "looks good",
                "lgtm",
                "go ahead",
                "proceed",
                "continue",
                "sounds good",
                "done",
            }
            return txt in confirmed_set
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
    # Mark that we are in the ICP intake flow so router can keep sending user replies here
    try:
        state["icp_in_progress"] = True
    except Exception:
        pass
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
                        [AIMessage("List 5–15 best customers (Company — website). You can type 'skip' for optional fields later.")],
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
                    state["icp_last_focus"] = "seeds"
                    state["messages"] = add_messages(
                        state.get("messages") or [],
                        [
                            AIMessage(
                                "List 5–15 best customers (Company — website). Optionally 2–3 lost/churned with a short reason."
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
                            state["icp_last_focus"] = "seeds"
                            state["messages"] = add_messages(
                                state.get("messages") or [],
                                [AIMessage(f"That doesn’t quite answer my question yet — I need at least 5 best customers. You shared {len(seeds)}; please add a few more (format: Company — website).")],
                            )
                            state["icp"] = icp_f
                            return state
                    else:
                        state["icp_last_focus"] = "seeds"
                        state["messages"] = add_messages(
                            state.get("messages") or [],
                            [AIMessage("That doesn’t seem to answer my question about best customers. Please list seeds as 'Company — domain' per line, one per line. Thanks!")],
                        )
                        state["icp"] = icp_f
                        return state
            # If Fast-Start is enabled, skip detailed prompts and proceed to confirmation immediately
            if ICP_WIZARD_FAST_START_ONLY:
                if not icp_f.get("fast_start_explained"):
                    expl = [
                        "I will infer industries from evidence instead of asking.",
                        "What I will crawl:",
                        "- Your site: Industries served, Customers/Case Studies, Integrations, Pricing (ACV hints), Careers (buyer/team clues), Partners, blog topics.",
                        "- Seed and anti-customer sites: industry labels, product lines, About text, Careers (roles/scale), Integrations pages, locations.",
                        "Then I’ll run web discovery to propose a Top‑10 lookalikes list with evidence. ACRA is only used later in the nightly SG pass.",
                    ]
                    state["messages"] = add_messages(state.get("messages") or [], [AIMessage("\n".join(expl))])
                    icp_f["fast_start_explained"] = True
                # Go straight to confirmation prompt
                state["messages"] = add_messages(
                    state.get("messages") or [],
                    [
                        AIMessage(
                            "Thanks! I’ll crawl your site + seed sites, run web discovery, extract evidence, and propose a Top‑10 with why‑us fit. ACRA is used later during the SG nightly pass. Reply confirm to proceed, or adjust any detail."
                        )
                    ],
                )
                state["icp"] = icp_f
                return state

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
                        "Thanks! I’ll crawl your site + seed sites, run web discovery, extract evidence, and propose a Top‑10 with why‑us fit. ACRA is used later during the SG nightly pass. Reply confirm to proceed, or adjust any detail."
                    )
                ],
            )
            state["icp"] = icp_f
            return state
    except Exception:
        # Non-blocking: simplify to confirmation to keep PRD19 minimal flow
        state["messages"] = add_messages(
            state.get("messages") or [],
            [AIMessage("Great. Reply **confirm** to save, or tell me what to change.")],
        )
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
                try:
                    state["acra_total_suggested"] = int(total_acra)
                except Exception:
                    pass
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
            # Prefer ACRA total stored from earlier probe/suggestions
            try:
                acra_total_state = int(state.get("acra_total_suggested") or 0)
            except Exception:
                acra_total_state = 0
            if acra_total_state:
                scheduled = max(int(acra_total_state) - do_now, 0)
            else:
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
            f"We can enrich {do_now} companies now. The nightly runner will process the remaining ICP companies. Accept a micro‑ICP, then type 'run enrichment' to proceed."
        )
    else:
        lines.append("No candidates yet. I’ll keep collecting ICP details.")
    msg = "\n".join([ln for ln in lines if ln])

    state["messages"] = add_messages(state.get("messages") or [], [AIMessage(content=msg)])
    return state


async def confirm_node(state: GraphState) -> GraphState:
    state["confirmed"] = True
    logger.info("[confirm] Entered confirm_node")
    # Emit a quick progress note so the user sees immediate feedback
    try:
        state["messages"] = add_messages(
            state.get("messages") or [],
            [AIMessage(content="Confirm received. Gathering evidence and planning Top‑10…")],
        )
    except Exception:
        pass
    try:
        from src.settings import ENABLE_AGENT_DISCOVERY as _ead  # type: ignore
        logger.info("[confirm] ENABLE_AGENT_DISCOVERY=%s", _ead)
    except Exception:
        logger.info("[confirm] ENABLE_AGENT_DISCOVERY unavailable (default False)")
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
                                # Merge resolver fast-facts into icp_profile to avoid 'n/a' later
                                try:
                                    prof = dict(state.get("icp_profile") or {})
                                    inds: list[str] = []
                                    titles: list[str] = []
                                    sizes: list[str] = []
                                    integrs: list[str] = []
                                    for c in cards:
                                        ff = c.fast_facts or {}
                                        if ff.get("industry_guess"):
                                            inds.append(str(ff.get("industry_guess")).strip().lower())
                                        if ff.get("size_band_guess"):
                                            sizes.append(str(ff.get("size_band_guess")).strip().lower())
                                        for t in (ff.get("buyer_titles") or []) or []:
                                            if isinstance(t, str) and t.strip():
                                                titles.append(t.strip().lower())
                                        for it in (ff.get("integrations_mentions") or []) or []:
                                            if isinstance(it, str) and it.strip():
                                                integrs.append(it.strip().lower())
                                    # de-dupe & assign only when present
                                    if inds:
                                        prof.setdefault("industries", [])
                                        prof["industries"] = sorted(set((prof.get("industries") or []) + inds))
                                    if titles:
                                        prof.setdefault("buyer_titles", [])
                                        prof["buyer_titles"] = sorted(set((prof.get("buyer_titles") or []) + titles))
                                    if sizes:
                                        prof.setdefault("size_bands", [])
                                        prof["size_bands"] = sorted(set((prof.get("size_bands") or []) + sizes))
                                    if integrs:
                                        prof.setdefault("integrations", [])
                                        prof["integrations"] = sorted(set((prof.get("integrations") or []) + integrs))
                                    state["icp_profile"] = prof
                                except Exception:
                                    pass
                                lines = ["Domain resolver preview:"]
                                low_conf = 0
                                for i, c in enumerate(cards, 1):
                                    facts = []
                                    ff = c.fast_facts or {}
                                    if ff.get("industry_guess"):
                                        facts.append(f"Industry: {ff.get('industry_guess')}")
                                    if ff.get("size_band_guess"):
                                        facts.append(f"Size: {ff.get('size_band_guess')}")
                                    if ff.get("geo_guess"):
                                        facts.append(f"Geo: {ff.get('geo_guess')}")
                                    buyers = ", ".join(ff.get("buyer_titles") or [])
                                    if buyers:
                                        facts.append(f"Buyers: {buyers}")
                                    integ = ", ".join(ff.get("integrations_mentions") or [])
                                    if integ:
                                        facts.append(f"Integrations: {integ}")
                                    fact_str = f" — {'; '.join(facts)}" if facts else ""
                                    lines.append(
                                        f"{i}) Seed: {c.seed_name} — Domain: {c.domain} — Confidence: {c.confidence} — Reason: {c.why}{fact_str}"
                                    )
                                    if (c.confidence or "").lower() == "low":
                                        low_conf += 1
                                if low_conf:
                                    lines.append(f"{low_conf} low‑confidence matches. Reply with edits if any domain looks off.")
                        # Do not show resolver results in chat; log instead
                        try:
                            logger.info("[confirm] Resolver preview (suppressed in chat):\n%s", "\n".join(lines))
                        except Exception:
                            pass
                        chips.append("Domain resolve ✓")
                    except Exception:
                        pass

                    # Evidence collection (Step 4) — best‑effort batch (ACRA anchoring disabled by default per PRD19)
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
                                    # Ensure companies row exists so evidence persists
                                    try:
                                        _cid = await asyncio.to_thread(_ensure_company_by_domain, _apex)
                                    except Exception:
                                        _cid = None
                                    n0 = await _icp_collect_evidence_for_domain(tid, _cid, _apex)
                                    logger.info("[confirm] Evidence rows for tenant website=%s: %s", _apex, n0)
                                    ev_count += int(n0 or 0)
                        except Exception:
                            pass
                        async def _collect_for_seed(s: dict):
                            """Collect crawl evidence for each seed; ACRA anchoring only when enabled.
                            Returns tuple: (evidence_rows_added, acra_rows_added)
                            """
                            name = (s.get("seed_name") or "").strip()
                            dom = (s.get("domain") or "").strip()
                            ev_added = 0
                            acra_added = 0
                            # Ensure company row for seed domain so evidence can persist
                            cid = None
                            try:
                                if dom:
                                    cid = await asyncio.to_thread(_ensure_company_by_domain, dom)
                            except Exception:
                                cid = None
                            # Crawl evidence (best effort)
                            if _icp_collect_evidence_for_domain and dom:
                                try:
                                    n = await _icp_collect_evidence_for_domain(tid, cid, dom)
                                    logger.info("[confirm] Evidence rows for domain=%s: %s", dom, n)
                                    ev_added += int(n or 0)
                                except Exception:
                                    pass
                            # Optional: ACRA anchoring from seed name
                            if ENABLE_ACRA_IN_CHAT and _icp_acra_anchor_seed and name:
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
                            if ENABLE_ACRA_IN_CHAT and acra_count:
                                chips.append("ACRA ✓")
                    except Exception:
                        pass

                    # Agent-driven discovery Top-10 with "why" + Jina snippets (PRD19)
                    try:
                        if ENABLE_AGENT_DISCOVERY:
                            logger.info("[confirm] Agent discovery enabled — importing agents_icp")
                            try:
                                from src.agents_icp import plan_top10_with_reasons as _agent_top10  # type: ignore
                                from src.agents_icp import icp_synthesizer as _agent_synth  # type: ignore
                                logger.info("[confirm] agents_icp import ok")
                            except Exception as e:
                                logger.warning("[confirm] agents_icp import failed: %s", e)
                                _agent_top10 = None  # type: ignore
                                _agent_synth = None  # type: ignore
                        if _agent_top10 is not None:
                                icp_prof = state.get("icp_profile") or {}
                                # If no profile yet, derive minimal profile from current ICP answers
                                if not icp_prof:
                                    try:
                                        _icp = dict(state.get("icp") or {})
                                        prof0: dict[str, list[str]] = {}
                                        inds = []
                                        if isinstance(_icp.get("industries"), list):
                                            inds = [str(s).strip() for s in _icp.get("industries") if isinstance(s, str) and s.strip()]
                                        elif isinstance(_icp.get("industry"), str) and _icp.get("industry").strip():
                                            inds = [str(_icp.get("industry")).strip()]
                                        if inds:
                                            prof0["industries"] = inds
                                        titles = []
                                        if isinstance(_icp.get("champion_titles"), list):
                                            titles = [str(s).strip() for s in _icp.get("champion_titles") if isinstance(s, str) and s.strip()]
                                        if titles:
                                            prof0["buyer_titles"] = titles
                                        sizes = []
                                        if isinstance(_icp.get("size_bands"), list):
                                            sizes = [str(s).strip() for s in _icp.get("size_bands") if isinstance(s, str) and s.strip()]
                                        if sizes:
                                            prof0["size_bands"] = sizes
                                        signals = []
                                        if isinstance(_icp.get("signals"), list):
                                            signals.extend([str(s).strip() for s in _icp.get("signals") if isinstance(s, str) and s.strip()])
                                        if isinstance(_icp.get("integrations_required"), list):
                                            signals.extend([str(s).strip() for s in _icp.get("integrations_required") if isinstance(s, str) and s.strip()])
                                        if signals:
                                            prof0["integrations"] = list({s.lower(): s for s in signals}.values())
                                        trigs = []
                                        if isinstance(_icp.get("triggers"), list):
                                            trigs = [str(s).strip() for s in _icp.get("triggers") if isinstance(s, str) and s.strip()]
                                        if trigs:
                                            prof0["triggers"] = trigs
                                        if prof0:
                                            icp_prof = prof0
                                            state["icp_profile"] = icp_prof
                                            logger.info("[confirm] Derived icp_profile from ICP answers keys=%s", list(prof0.keys()))
                                    except Exception:
                                        pass
                                # If empty, synthesize from seeds quickly
                                if not icp_prof and _agent_synth is not None:
                                    try:
                                        seeds_ev = [{"url": s.get("domain"), "snippet": s.get("seed_name")} for s in (icp.get("seeds_list") or [])]
                                        sstate = {"icp_profile": {}, "seeds": seeds_ev}
                                        logger.info("[confirm] invoking icp_synthesizer on %d seeds", len(seeds_ev))
                                        out = await asyncio.to_thread(_agent_synth, sstate)
                                        icp_prof = out.get("icp_profile") or {}
                                        state["icp_profile"] = icp_prof
                                    except Exception as e:
                                        logger.warning("[confirm] icp_synthesizer failed: %s", e)
                                        icp_prof = {}
                                # Best-effort tenant id
                                tnet = None
                                try:
                                    tnet = int(tid) if tid is not None else None
                                except Exception:
                                    tnet = None
                                # Pass seed domains as discovery hints for DDG
                                try:
                                    seed_domains = []
                                    for s in (icp.get("seeds_list") or []):
                                        dom = (s.get("domain") or "").strip().lower() if isinstance(s, dict) else ""
                                        if dom:
                                            seed_domains.append(dom)
                                    if seed_domains:
                                        from src.agents_icp import set_seed_hints as _set_seed_hints  # type: ignore
                                        try:
                                            _set_seed_hints(seed_domains)
                                            logger.info("[confirm] set %d seed hints for discovery", len(seed_domains))
                                        except Exception:
                                            pass
                                except Exception:
                                    pass
                                logger.info("[confirm] invoking plan_top10_with_reasons; have_icp=%s", bool(icp_prof))
                                top = await asyncio.to_thread(_agent_top10, icp_prof, tnet)
                                # Log Top‑10 (cap to 10) for consistency with UI
                                logger.info("[confirm] agent top10 count=%d", min(10, len(top or [])))
                                if not top:
                                    # Seeds-based competitor query fallback is disabled when STRICT_INDUSTRY_QUERY_ONLY
                                    try:
                                        from src.settings import STRICT_INDUSTRY_QUERY_ONLY as _strict  # type: ignore
                                    except Exception:
                                        _strict = True
                                    if not _strict:
                                        try:
                                            from src.agents_icp import fallback_top10_from_seeds as _fb_seeds  # type: ignore
                                            seed_domains = []
                                            for s in (icp.get("seeds_list") or []):
                                                dom = (s.get("domain") or "").strip().lower()
                                                if dom:
                                                    seed_domains.append(dom)
                                            if seed_domains:
                                                top = await asyncio.to_thread(_fb_seeds, seed_domains, icp_prof)
                                                logger.info("[confirm] seeds-fallback top10 count=%d", len(top or []))
                                        except Exception as _fb1_e:
                                            logger.warning("[confirm] seeds-fallback failed: %s", _fb1_e)
                                if not top and not STRICT_DDG_ONLY:
                                    # Fallback 2: Jina outlinks from seeds (no DDG) — disabled when STRICT_DDG_ONLY
                                    try:
                                        from src.agents_icp import fallback_top10_via_seed_outlinks as _fb_outlinks  # type: ignore
                                        seed_domains = []
                                        for s in (icp.get("seeds_list") or []):
                                            dom = (s.get("domain") or "").strip().lower()
                                            if dom:
                                                seed_domains.append(dom)
                                        if seed_domains:
                                            top = await asyncio.to_thread(_fb_outlinks, seed_domains, icp_prof)
                                            logger.info("[confirm] seeds-outlinks fallback top10 count=%d", len(top or []))
                                    except Exception as _fb2_e:
                                        logger.warning("[confirm] seeds-outlinks fallback failed: %s", _fb2_e)
                                if not top:
                                    # Fallback 3: legacy DDG+Jina heuristic (may be disabled by env)
                                    try:
                                        from src.agents_icp import plan_top10_with_reasons_fallback as _top10_fb  # type: ignore
                                        top = await asyncio.to_thread(_top10_fb, icp_prof)
                                        logger.info("[confirm] legacy fallback top10 count=%d", len(top or []))
                                    except Exception as _fb_e:
                                        logger.warning("[confirm] top10 legacy fallback failed: %s", _fb_e)
                                if top:
                                    # Persist and stash for reuse during 'run enrichment'
                                    try:
                                        tid2 = await _resolve_tenant_id_for_write(state)
                                    except Exception:
                                        tid2 = None
                                    try:
                                        _persist_top10_preview(int(tid2) if isinstance(tid2, int) else None, top)
                                    except Exception:
                                        pass
                                    try:
                                        state["agent_top10"] = top
                                    except Exception:
                                        pass
                                    # Also stash into thread memory so next turn can reuse without DB
                                    try:
                                        _stash_top10_in_thread_memory(state, top)
                                    except Exception:
                                        pass
                                    # Pretty Top‑10 table (separate message from ICP Profile)
                                    top_lines: list[str]
                                    try:
                                        table = _fmt_top10_md(top)
                                        top_lines = ["Top‑listed lookalikes (with why):\n\n" + table]
                                    except Exception:
                                        top_lines = ["Top‑listed lookalikes (with why):"]
                                        for i, row in enumerate(top, 1):
                                            dom = row.get("domain")
                                            why = row.get("why") or "signal match"
                                            score = int(row.get("score") or 0)
                                            snip = _clean_snippet(row.get("snippet") or "")
                                            if snip:
                                                top_lines.append(f"{i}) {dom} — {why} (score {score}) — {snip}")
                                            else:
                                                top_lines.append(f"{i}) {dom} — {why} (score {score})")
                                    # Enrich ICP from r.jina+ddg if sparse, then show a detailed profile summary
                                    try:
                                        from src.agents_icp import ensure_icp_enriched_with_jina as _icp_enrich  # type: ignore
                                        try:
                                            out = _icp_enrich({"icp_profile": state.get("icp_profile") or {}})
                                            if isinstance(out.get("icp_profile"), dict):
                                                state["icp_profile"] = out.get("icp_profile")
                                        except Exception:
                                            pass
                                    except Exception:
                                        pass
                                    # Build ICP Profile as a separate message block
                                    profile_lines: list[str] = []
                                    try:
                                        icp_prof = state.get("icp_profile") or {}
                                        inds = (icp_prof.get("industries") or [])
                                        titles_l = (icp_prof.get("buyer_titles") or [])
                                        sizes_l = (icp_prof.get("size_bands") or [])
                                        ints = (icp_prof.get("integrations") or [])
                                        trigs = (icp_prof.get("triggers") or [])
                                        profile_lines.append("ICP Profile")
                                        profile_lines.append(f"- Industries: {', '.join(inds[:6]) if inds else 'n/a'}")
                                        profile_lines.append(f"- Buyer titles: {', '.join(titles_l[:6]) if titles_l else 'n/a'}")
                                        profile_lines.append(f"- Company sizes: {', '.join(sizes_l[:6]) if sizes_l else 'n/a'}")
                                        sigs_arr = (ints or []) + (trigs or [])
                                        profile_lines.append(f"- Signals: {', '.join(sigs_arr[:8]) if sigs_arr else 'n/a'}")
                                    except Exception:
                                        profile_lines.append("ICP Profile")
                                    # Emit Top‑10 table and ICP Profile as two separate messages
                                    state["messages"] = add_messages(
                                        state.get("messages") or [],
                                        [AIMessage("\n".join(top_lines)), AIMessage("\n".join(profile_lines))],
                                    )
                                    chips.append("Top‑10 ✓")
                                else:
                                    # No Top‑10 from web discovery; advise user to adjust ICP rather than echo seeds
                                    try:
                                        state["messages"] = add_messages(
                                            state.get("messages") or [],
                                            [AIMessage("Web discovery returned no lookalikes. Try broadening industries or adding signals (integrations, buyer titles).")],
                                        )
                                    except Exception:
                                        pass
                    except Exception as e:
                        logger.warning("[confirm] agent discovery block failed: %s", e)

                    # Ensure ICP profile is persisted even if Top‑10 is empty
                    try:
                        icp_prof2 = dict(state.get("icp_profile") or {})
                        if icp_prof2:
                            try:
                                tid3 = await _resolve_tenant_id_for_write(state)
                            except Exception:
                                tid3 = None
                            if isinstance(tid3, int):
                                try:
                                    base_payload: dict = {}
                                except Exception:
                                    base_payload = {}
                                try:
                                    merged_payload = _merge_icp_profile_into_payload(base_payload, icp_prof2)
                                except Exception:
                                    merged_payload = icp_prof2
                                try:
                                    _save_icp_rule_sync(int(tid3), merged_payload, name="Default ICP")
                                    logger.info("[confirm] Persisted ICP profile fallback (icp_rules) keys=%s", list((merged_payload or {}).keys())[:8])
                                except Exception as _pf_e:
                                    logger.warning("[confirm] ICP profile persist fallback failed: %s", _pf_e)
                    except Exception:
                        pass

                    # Pattern mining + Micro‑ICPs (Step 6–8) — disabled in chat when ACRA is off (nightly only)
                    try:
                        if ENABLE_ACRA_IN_CHAT and _icp_winner_profile and _icp_micro_suggestions:
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
                                            try:
                                                state["acra_total_suggested"] = int(total)
                                            except Exception:
                                                pass
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

                    # Synchronously map seeds and refresh patterns — disabled in chat by default (nightly only)
                    if ENABLE_ACRA_IN_CHAT and _icp_map_seeds and _icp_refresh_patterns and _icp_generate_suggestions:
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
                                        try:
                                            state["acra_total_suggested"] = int(total)
                                        except Exception:
                                            pass
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
    # If still fewer than limit, do not top up from ACRA in chat (PRD19); leave to nightly
    n = len(state.get("candidates") or [])
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

    # Start with discovered candidates count (domain URLs)
    # Prefer in-memory discovery results strictly for accuracy per run
    discovered_total = 0
    try:
        ac = state.get("agent_candidates") or []
        if isinstance(ac, list) and ac:
            discovered_total = len(ac)
    except Exception:
        discovered_total = 0
    if discovered_total <= 0:
        try:
            at = state.get("agent_top10") or []
            if isinstance(at, list) and at:
                discovered_total = len(at)
        except Exception:
            discovered_total = 0
    if discovered_total <= 0:
        # Last resort: use cached state count if present; avoid DB/staging totals
        try:
            discovered_total = int(state.get("web_discovery_total") or 0)
        except Exception:
            discovered_total = 0
    display_total = discovered_total
    if display_total > 0:
        msg_lines.append(f"Found {display_total} ICP candidates.")
    else:
        msg_lines.append("Collecting ICP candidates…")

    # PRD19: hide SSIC/ACRA preview in chat; ACRA is for nightly SG pass only

    # Plan enrichment counts: how many now vs later
    do_now = min(n, enrich_now_limit) if n else 0
    if n > 0:
        # Compute nightly remainder from ACRA total when available; prefer stored value
        try:
            acra_total_state = int(state.get("acra_total_suggested") or 0)
        except Exception:
            acra_total_state = 0
        if acra_total_state:
            scheduled = max(int(acra_total_state) - do_now, 0)
        else:
            scheduled = max((total_acra if 'total_acra' in locals() else icp_total) - do_now, 0)
        msg_lines.append(
            f"We can enrich {do_now} companies now. The nightly runner will process the remaining ICP companies. Accept a micro‑ICP, then type 'run enrichment' to proceed."
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
    # Announce any completed background jobs (next-40) from prior turns
    try:
        _announce_completed_bg_jobs(state)
    except Exception:
        pass
    # Mark the last routed human command so the router's duplicate-guard
    # can halt re-entry loops (e.g., repeated "run enrichment").
    try:
        state["last_routed_text"] = (_last_user_text(state) or "").strip().lower()
    except Exception:
        pass
    # Defer detailed begin log until after we resolve Top‑10 vs fallback candidates
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

    # Strict Top‑10 enrichment: if a Top‑10 list from web discovery exists,
    # build candidates from those domains and do not mix with ACRA/top-ups.
    # Guard: do NOT override existing candidate list with a too-short Top‑10.
    try:
        top10 = state.get("agent_top10") or []
    except Exception:
        top10 = []
    # Try loading Top‑10 from thread memory first (avoids re-discovery on 'run enrichment')
    if not (isinstance(top10, list) and top10):
        mem_top = _load_top10_from_thread_memory(state)
        if isinstance(mem_top, list) and mem_top:
            top10 = mem_top

    # Try loading last persisted Top‑10 preview for this tenant if still not present
    if not (isinstance(top10, list) and top10):
        try:
            tid = await _resolve_tenant_id_for_write(state)
        except Exception:
            tid = None
        persisted_top = _load_persisted_top10(int(tid) if isinstance(tid, int) else None)
        if persisted_top:
            top10 = persisted_top
        else:
            # Strict mode: do not regenerate Top‑10 during enrichment; require user to confirm again
            if bool(state.get("strict_top10")):
                try:
                    logger.info("[enrich] strict mode: no persisted Top‑10; aborting (no discovery)")
                except Exception:
                    pass
            # Last resort (non-strict only): if the chat previously showed Top‑10, attempt regeneration once
            elif _top10_preview_was_sent(state):
                try:
                    regen = await _regenerate_top10_if_missing(state, int(tid) if isinstance(tid, int) else None)
                    if regen:
                        top10 = regen
                except Exception:
                    pass
    # If Top‑10 is still missing, regenerate a fresh Top‑10 now (fallback), persist, then proceed
    if not (isinstance(top10, list) and top10):
        try:
            from src.agents_icp import plan_top10_with_reasons as _agent_top10  # type: ignore
        except Exception:
            _agent_top10 = None  # type: ignore
        if _agent_top10 is not None:
            try:
                icp_prof = dict(state.get("icp_profile") or {})
            except Exception:
                icp_prof = {}
            try:
                t_id = await _resolve_tenant_id_for_write(state)
            except Exception:
                t_id = None
            try:
                regenerated = await asyncio.to_thread(_agent_top10, icp_prof, (int(t_id) if isinstance(t_id, int) else None))
            except Exception:
                regenerated = []
            if regenerated:
                # Persist and stash full planned set; enrichment still caps run-now
                try:
                    state["agent_top10"] = regenerated
                    _persist_top10_preview((int(t_id) if isinstance(t_id, int) else None), regenerated)
                except Exception:
                    pass
                top10 = regenerated
        # If still missing after regeneration, ask user to confirm again
        if not (isinstance(top10, list) and top10):
            try:
                state["messages"] = add_messages(
                    state.get("messages") or [],
                    [AIMessage(content="I can’t find the Top‑10 shortlist. I tried to regenerate it but got no results. Please type ‘confirm’ to rebuild it, then use 'run enrichment'.")],
                )
            except Exception:
                pass
            return state
    if isinstance(top10, list) and top10:
        try:
            from src.database import get_conn as _get_conn
            with _get_conn() as _c, _c.cursor() as _cur:
                cand: list[dict] = []
                for it in top10[:10]:
                    dom = (it.get("domain") or "").strip().lower() if isinstance(it, dict) else ""
                    if not dom:
                        continue
                    _cur.execute("SELECT company_id, name FROM companies WHERE website_domain=%s", (dom,))
                    row = _cur.fetchone()
                    if row and row[0] is not None:
                        cid = int(row[0])
                        name = (row[1] or dom)
                    else:
                        _cur.execute(
                            "INSERT INTO companies(name, website_domain, last_seen) VALUES (%s,%s,NOW()) RETURNING company_id",
                            (dom, dom),
                        )
                        cid = int((_cur.fetchone() or [None])[0])
                        name = dom
                    cand.append({"id": cid, "name": name})
            if cand:
                # Always use the discovered Top‑10 exclusively, even if fewer than limit
                try:
                    enrich_now_limit = int(os.getenv("CHAT_ENRICH_LIMIT", os.getenv("RUN_NOW_LIMIT", "10") or 10))
                except Exception:
                    enrich_now_limit = 10
                sel = cand[:enrich_now_limit] if len(cand) > enrich_now_limit else cand
                state["candidates"] = sel
                state["strict_top10"] = True
                try:
                    logger.info("[enrich] using agent/persisted top10 count=%d", len(sel))
                except Exception:
                    pass
        except Exception:
            # Non-fatal; fall back to existing selection logic
            pass
        # Persist preview if tenant known
        try:
            tid = await _resolve_tenant_id_for_write(state)
        except Exception:
            tid = None
        _persist_top10_preview(int(tid) if isinstance(tid, int) else None, top10 if isinstance(top10, list) else [])
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
    # If we have fewer than the desired head count, try to top up from ACRA by SSIC codes
    # Skip top-ups when strict Top‑10 is active
    if len(candidates) < enrich_now_limit and not state.get("strict_top10"):
        try:
            # Derive SSIC codes from selected suggestions
            sugg = state.get("micro_icp_suggestions") or []
            codes: list[str] = []
            for it in (sugg if isinstance(sugg, list) else []):
                sid = (it.get("id") or "") if isinstance(it, dict) else ""
                if isinstance(sid, str) and sid.lower().startswith("ssic:"):
                    code = sid.split(":", 1)[1]
                    if code and code.strip():
                        codes.append(code.strip())
            # Fallback: infer SSIC from ICP industries
            if not codes and (state.get("icp") or {}).get("industries"):
                try:
                    terms = [str(t).strip().lower() for t in (state.get("icp") or {}).get("industries") or [] if str(t).strip()]
                    matches = _find_ssic_codes_by_terms(terms)
                    codes = [c for (c, _t, _s) in matches]
                except Exception:
                    codes = []
            if codes:
                # Fetch ACRA rows and ensure company rows to top up count
                try:
                    rows = await asyncio.to_thread(_select_acra_by_ssic_codes, set(codes), enrich_now_limit * 3)
                except Exception:
                    rows = []
                if rows:
                    pool = await get_pg_pool()
                    needed = enrich_now_limit - len(candidates)
                    added: list[dict] = []
                    have_ids = {c.get("id") for c in candidates if isinstance(c, dict)}
                    for r in rows:
                        if needed <= 0:
                            break
                        nm = (r.get("entity_name") or "").strip()
                        if not nm:
                            continue
                        try:
                            cid = await _ensure_company_row(pool, nm)
                            if cid in have_ids:
                                continue
                            added.append({"id": cid, "name": nm, "uen": (r.get("uen") or "").strip() or None})
                            have_ids.add(cid)
                            needed -= 1
                        except Exception:
                            continue
                    if added:
                        candidates = (state.get("candidates") or []) + added
                        state["candidates"] = candidates
        except Exception:
            pass
    if len(candidates) > enrich_now_limit:
        candidates = candidates[:enrich_now_limit]
        state["candidates"] = candidates
    # Finalize start-of-enrichment log with strict flag after Top‑10 detection/regen and candidate selection
    try:
        logger.info(
            "[enrich] begin strict_top10=%s candidates=%d",
            str(bool(state.get("strict_top10"))),
            int(len(state.get("candidates") or [])),
        )
    except Exception:
        pass
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
        final_state = await enrich_company_with_tavily(cid, name, uen, search_policy="require_existing")
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

    # Always enqueue background for the next 40 preview candidates (once per thread),
    # even if some of the head-of-line enrichments failed or were skipped.
    try:
        _enqueue_next40_if_applicable(state)
    except Exception as _enq_exc:
        try:
            logger.warning("[next40] enqueue failed: %s", _enq_exc)
        except Exception:
            pass

    if all_done:
        # Compose completion message; include ACRA/ICP totals and nightly remainder
        icp_total = 0
        acra_total_state = 0
        try:
            icp_total = int(state.get("icp_match_total") or 0)
        except Exception:
            icp_total = 0
        try:
            acra_total_state = int(state.get("acra_total_suggested") or 0)
        except Exception:
            acra_total_state = 0
        # Do not surface counts; keep a simple nightly note
        done_msg = (
            f"Enrichment complete for {len(results)} companies. The nightly runner will process the remaining ICP companies."
        )
        state["messages"] = add_messages(
            state.get("messages") or [],
            [AIMessage(content=done_msg)],
        )
        # (Already enqueued next‑40 above if applicable)
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

                # Nightly remainder scheduling: enqueue staging_upsert for the accepted micro‑ICP (or derived SSIC)
                try:
                    # Compute remainder based on ACRA/ICP totals vs run-now count
                    run_now_count = len(results)
                    # Prefer previously measured ACRA total
                    total_candidates = int(state.get("acra_total_suggested") or state.get("icp_match_total") or 0)
                    remainder = max(total_candidates - run_now_count, 0) if total_candidates else 0
                except Exception:
                    remainder = 0

                # Enqueue when there is any clear targeting (titles/codes), even if remainder computed as 0
                # because ACRA_total might be unknown at runtime. Guard duplicate enqueues with a state flag.
                if not state.get("nightly_enqueued"):
                    # Derive SSIC titles from selected micro‑ICP suggestion if present; else from dominant evidence
                    titles: list[str] = []
                    codes: list[str] = []
                    try:
                        sugg = state.get("micro_icp_suggestions") or []
                        for it in (sugg if isinstance(sugg, list) else []):
                            sid = (it.get("id") or "") if isinstance(it, dict) else ""
                            if isinstance(sid, str) and sid.lower().startswith("ssic:"):
                                code = sid.split(":", 1)[1]
                                if code and code.strip():
                                    codes.append(code.strip())
                        if codes:
                            with get_conn() as _c:
                                cur = _c.cursor()
                                cur.execute(
                                    "SELECT title FROM ssic_ref WHERE regexp_replace(code::text,'\\D','','g') = ANY(%s::text[])",
                                    (codes,),
                                )
                                titles = [r[0] for r in (cur.fetchall() or []) if r and r[0]]
                        # If we still have no titles, fall back to top SSIC from evidence
                        if not titles:
                            with get_conn() as _c:
                                cur = _c.cursor()
                                cur.execute(
                                    """
                                    SELECT s.title
                                    FROM icp_evidence e
                                    JOIN ssic_ref s
                                      ON regexp_replace(s.code::text,'\\D','','g') = regexp_replace((e.value->>'ssic')::text,'\\D','','g')
                                    WHERE e.signal_key='ssic'
                                    GROUP BY s.title
                                    ORDER BY COUNT(*) DESC
                                    LIMIT 3
                                    """
                                )
                                titles = [r[0] for r in (cur.fetchall() or []) if r and r[0]]
                        # If still empty, attempt to resolve SSIC by ICP industries terms
                        if not titles and (state.get("icp") or {}).get("industries"):
                            try:
                                terms = [str(t).strip().lower() for t in (state.get("icp") or {}).get("industries") or [] if str(t).strip()]
                                matches = _find_ssic_codes_by_terms(terms)
                                codes = [c for (c, _t, _s) in matches]
                                if codes:
                                    with get_conn() as _c:
                                        cur = _c.cursor()
                                        cur.execute(
                                            "SELECT title FROM ssic_ref WHERE regexp_replace(code::text,'\\D','','g') = ANY(%s::text[])",
                                            (codes,),
                                        )
                                        titles = [r[0] for r in (cur.fetchall() or []) if r and r[0]]
                            except Exception:
                                pass
                    except Exception:
                        titles = []

                    # Enqueue when we have any titles/codes or a positive remainder
                    if titles or codes or remainder > 0:
                        try:
                            # Resolve tenant and enqueue a single staging_upsert job for the SSIC titles
                            tid = await _resolve_tenant_id_for_write(state)
                            from src.jobs import enqueue_staging_upsert
                            upsert_terms = titles if titles else [f"SSIC {c}" for c in codes]
                            enqueue_staging_upsert(int(tid) if isinstance(tid, int) else None, upsert_terms)
                            state["nightly_enqueued"] = True
                            logger.info("nightly remainder queued: titles/codes=%s remainder=%s", upsert_terms, remainder)
                        except Exception as _qexc:
                            logger.warning("nightly remainder enqueue failed: %s", _qexc)

                # Best-effort Odoo sync for completed companies (skip ones we skipped enriching)
                # Guarded by env flag and readiness; skip in local dev unless explicitly enabled.
                try:
                    def _odoo_export_enabled() -> bool:
                        # Policy: if ODOO_EXPORT_ENABLED is explicitly set, honor it.
                        # Otherwise, auto-enable when a mapping/DSN is present.
                        try:
                            raw = os.getenv("ODOO_EXPORT_ENABLED")
                            if raw is not None:
                                v = raw.strip().lower()
                                return v in ("1", "true", "yes", "on")
                        except Exception:
                            pass
                        # Auto-enable if we can resolve any active mapping or DSN
                        try:
                            from src.settings import ODOO_POSTGRES_DSN as _ODSN
                            if _ODSN:
                                return True
                        except Exception:
                            pass
                        try:
                            with get_conn() as _c, _c.cursor() as _cur:
                                _cur.execute("SELECT 1 FROM odoo_connections WHERE active=TRUE LIMIT 1")
                                return bool(_cur.fetchone())
                        except Exception:
                            return False

                    if not _odoo_export_enabled():
                        logger.info("odoo export skipped: disabled by policy (no mapping/flag)")
                        # Soft-skip export without raising; proceed with rest of flow
                        raise StopIteration
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
                except StopIteration:
                    pass
                except Exception as _odoo_exc:
                    logger.exception("odoo sync block failed")
        except Exception as _score_exc:
            logger.exception("lead scoring failed")
    else:
        # Partial completion: score and export the completed subset now; enqueue the remainder for background.
        done = [r for r in results if r.get("completed")]
        pending = [r for r in results if not r.get("completed")]
        done_ids = [int(r.get("company_id")) for r in done if r.get("company_id") is not None]
        pending_ids = [int(r.get("company_id")) for r in pending if r.get("company_id") is not None]

        # Score the completed subset and render the table immediately
        try:
            if done_ids:
                scoring_initial_state = {
                    "candidate_ids": done_ids,
                    "lead_features": [],
                    "lead_scores": [],
                    "icp_payload": {
                        "employee_range": {
                            "min": (state.get("icp") or {}).get("employees_min"),
                            "max": (state.get("icp") or {}).get("employees_max"),
                        },
                        "revenue_bucket": (state.get("icp") or {}).get("revenue_bucket"),
                        "incorporation_year": {
                            "min": (state.get("icp") or {}).get("year_min"),
                            "max": (state.get("icp") or {}).get("year_max"),
                        },
                    },
                }
                await lead_scoring_agent.ainvoke(scoring_initial_state)
                # Filter candidates to the completed subset for display
                try:
                    cands = state.get("candidates") or []
                    state["candidates"] = [c for c in cands if int(c.get("id") or 0) in set(done_ids)]
                except Exception:
                    pass
                state = await score_node(state)
        except Exception:
            logger.exception("partial scoring failed")

        # Attempt Odoo export for completed subset (best-effort)
        try:
            if done_ids:
                from app.odoo_store import OdooStore
                # Resolve tenant id, mirroring the all_done path
                _tid_val = state.get("tenant_id") if isinstance(state, dict) else None
                try:
                    _tid = int(_tid_val) if _tid_val is not None else None
                except Exception:
                    _tid = None
                if _tid is None:
                    try:
                        _tid_env = os.getenv("DEFAULT_TENANT_ID")
                        _tid = int(_tid_env) if _tid_env and _tid_env.isdigit() else None
                    except Exception:
                        _tid = None
                store = None
                try:
                    store = OdooStore(tenant_id=_tid)
                except Exception:
                    store = None
                if store is not None:
                    async with pool.acquire() as conn:
                        comp_rows = await conn.fetch(
                            """
                            SELECT company_id, name, uen, industry_norm, employees_est,
                                   revenue_bucket, incorporation_year, website_domain
                            FROM companies WHERE company_id = ANY($1::int[])
                            """,
                            done_ids,
                        )
                        comps = {r["company_id"]: dict(r) for r in comp_rows}
                        email_rows = await conn.fetch(
                            "SELECT company_id, email FROM lead_emails WHERE company_id = ANY($1::int[])",
                            done_ids,
                        )
                        emails: Dict[int, str] = {}
                        for row in email_rows:
                            emails.setdefault(row["company_id"], row["email"])
                        score_rows = await conn.fetch(
                            "SELECT company_id, score, rationale FROM lead_scores WHERE company_id = ANY($1::int[])",
                            done_ids,
                        )
                        scores = {r["company_id"]: dict(r) for r in score_rows}
                    for cid in done_ids:
                        comp = comps.get(cid) or {}
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
                                except Exception:
                                    pass
                            sc = scores.get(cid)
                            if sc:
                                try:
                                    await store.create_lead_if_high(
                                        odoo_id,
                                        comp.get("name"),
                                        float(sc.get("score") or 0.0),
                                        {},
                                        str(sc.get("rationale") or ""),
                                        email,
                                    )
                                except Exception:
                                    pass
                        except Exception:
                            logger.exception("odoo sync failed for company_id=%s", cid)
        except Exception:
            logger.exception("partial odoo export failed")

        # Enqueue pending items for background enrichment if any
        try:
            if pending_ids:
                tid = await _resolve_tenant_id_for_write(state)
                if tid is not None:
                    from src.jobs import enqueue_web_discovery_bg_enrich as __enqueue_bg
                    __enqueue_bg(int(tid), pending_ids)
        except Exception as _bg_exc:
            try:
                logger.warning("[enqueue-pending] failed: %s", _bg_exc)
            except Exception:
                pass

        # Notify user and continue
        total = len(results)
        msg = f"Enrichment finished with issues ({len(done_ids)}/{total} completed). I scored and listed completed ones here; the rest are queued for background."
        state["messages"] = add_messages(
            state.get("messages") or [],
            [AIMessage(content=msg)],
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
        # Apply RLS tenant context if known so we can read tenant-scoped lead_scores
        try:
            tid = await _resolve_tenant_id_for_write(state)
            if tid is not None:
                await conn.execute("SELECT set_config('request.tenant_id', $1, true)", str(tid))
        except Exception:
            pass
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

    # Boot-session initialization: record current message count and STOP
    # any resumed run immediately after server restart. We only honor commands
    # after a NEW human message arrives post-boot.
    try:
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

        if state.get("boot_init_token") != BOOT_TOKEN:
            state["boot_init_token"] = BOOT_TOKEN
            state["boot_seen_messages_len"] = len(msgs)
            last = msgs[-1] if msgs else None
            # Halt only if there isn't a fresh human message
            if not (last and _is_human(last)):
                logger.info("router -> end (boot resume guard: waiting for new user input)")
                return "end"
            # Fresh human: clear duplicate guard to allow routing
            state["last_user_boot_token"] = BOOT_TOKEN
            try:
                if "last_routed_text" in state:
                    del state["last_routed_text"]
            except Exception:
                pass
        else:
            # On subsequent router cycles, if new messages appended and last is Human → mark as fresh user action
            if len(msgs) > int(state.get("boot_seen_messages_len") or 0):
                last = msgs[-1] if msgs else None
                if last and _is_human(last):
                    state["last_user_boot_token"] = BOOT_TOKEN
                    # New human input: allow routing again by clearing last_routed_text
                    try:
                        if "last_routed_text" in state:
                            del state["last_routed_text"]
                    except Exception:
                        pass
                state["boot_seen_messages_len"] = len(msgs)
    except Exception:
        pass

    # If we've already routed for this exact human text, wait for new input to avoid loops
    try:
        if (state.get("last_routed_text") or "") == text and text.strip():
            logger.info("router -> end (duplicate command; waiting for new user input)")
            return "end"
    except Exception:
        pass

    # Note: Do not route to enrichment here. We intentionally handle
    # explicit 'run enrichment' after the assistant-last guard below to
    # avoid loops when the assistant just spoke.

    # If the assistant spoke last, do not route again until a NEW human message arrives.
    # This prevents loops where the router keeps re-reading the same last human text.
    if _last_is_ai(msgs):
        logger.info("router -> end (assistant last)")
        return "end"

    # Intent detection: only begin ICP when the user explicitly asks for lead generation
    # Note: do NOT treat 'run enrichment' as generic lead-intent here; it has an
    # explicit branch later that routes to enrichment. Including it here would
    # misroute to the ICP node and block enrichment.
    lead_intent = False
    try:
        # phrases like: start lead gen, start lead generation, start discovery, start prospecting
        if re.search(r"\bstart\s+(lead(\s*gen|\s*generation)?|prospect(ing)?|discovery|enrich(ment)?)\b", text):
            lead_intent = True
        # commands like: find/generate/prospect/discover leads/companies
        if re.search(r"\b(find|generate|prospect|discover)\s+(leads?|companies)\b", text):
            lead_intent = True
    except Exception:
        lead_intent = False

    if lead_intent:
        logger.info("router -> icp (explicit lead-gen intent)")
        try:
            state["last_routed_text"] = text
            state["icp_in_progress"] = True
        except Exception:
            pass
        return "icp"

    # Greetings: acknowledge and explain how to start lead-gen, without auto-starting
    if re.search(r"\b(hello|hi|hey|howdy)\b", text) and not bool(state.get("welcomed")):
        logger.info("router -> welcome (greeting)")
        try:
            state["last_routed_text"] = text
        except Exception:
            pass
        return "welcome"

    # If user pasted a website/domain and Finder is enabled, treat it as starting/continuing ICP
    try:
        url = _parse_website(text)
    except Exception:
        url = None
    if url:
        logger.info("router -> icp (website/domain provided)")
        try:
            state["last_routed_text"] = text
            state["icp_in_progress"] = True
            # Initialize icp dict if missing so icp node can store website
            if not state.get("icp"):
                state["icp"] = {}
        except Exception:
            pass
        return "icp"

    # (Handled above) Explicit enrichment command routes immediately

    # Accept micro‑ICP selection
    if re.search(r"\baccept\s+micro[- ]icp\b", text):
        logger.info("router -> accept (user accepted micro‑ICP)")
        try:
            state["last_routed_text"] = text
        except Exception:
            pass
        return "accept"

    # 1) Pipeline progression (explicit only)
    # Finder gating: do NOT auto-advance to enrichment until micro-ICP suggestions are done
    # Finder gating: allow explicit 'run enrichment' to override the hold
    if ENABLE_ICP_INTAKE and not state.get("finder_suggestions_done") and ("run enrichment" not in text):
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

    # Do not auto-progress; enrichment only runs on explicit command handled below.
    if has_results and enrichment_completed and not has_scored:
        logger.info("router -> score (have enrichment, no scores, all completed)")
        return "score"
    if has_results and not enrichment_completed and not has_scored:
        logger.info("router -> end (enrichment not fully completed)")
        return "end"

    # 2) If assistant spoke last and no pending work, wait for user input (covered by early guard)

    # 3) Fast-path: user requested enrichment (only when last turn was human)
    if "run enrichment" in text:
        logger.info("router -> enrich (user requested enrichment)")
        try:
            state["last_routed_text"] = text
            # Enrichment should strictly use the previously persisted Top‑10;
            # never re-run discovery during this command.
            state["strict_top10"] = True
        except Exception:
            pass
        return "enrich"

    # 4) If user pasted an explicit company list, jump to candidates
    # Avoid misclassifying comma-separated industry/geo lists as companies.
    pasted = _parse_company_list(text)
    # Only jump early if at least one looks like a domain or multi-word name
    if pasted and any(("." in n) or (" " in n) for n in pasted):
        if ENABLE_ICP_INTAKE and not _icp_complete(icp):
            logger.info("router -> icp (Finder gating: ignore explicit list until ICP set)")
            try:
                state["last_routed_text"] = text
            except Exception:
                pass
            return "icp"
        logger.info("router -> candidates (explicit company list)")
        try:
            state["last_routed_text"] = text
        except Exception:
            pass
        return "candidates"

    # 5) Free‑form lead‑gen Q&A: if the user asked a general question, answer it directly.
    # Avoid hijacking explicit commands.
    try:
        if (
            _qa_answer is not None
            and not any(k in text for k in ("run enrichment", "accept micro", "accept micro-icp"))
            # Detect question intent: question mark OR question verbs anywhere in text
            and ("?" in text or re.search(r"\b(how|what|why|when|where|which|tips|best|increase|improve|explain|define|definition|meaning)\b", text, flags=re.I))
            and (state.get("last_answered_text") != text)
        ):
            logger.info("router -> leadgen_qa (free‑form question detected)")
            try:
                state["last_routed_text"] = text
            except Exception:
                pass
            return "leadgen_qa"
    except Exception:
        pass

    # 6) User said confirm: proceed forward once (avoid loops)
    if _user_just_confirmed(state):
        # Finder: if minimal intake present (website + seeds), allow confirm pipeline even if core ICP incomplete
        if ENABLE_ICP_INTAKE:
            has_minimal = bool(icp.get("website_url")) and bool(icp.get("seeds_list"))
            if has_minimal:
                logger.info("router -> confirm (Finder minimal intake present; proceeding to confirm pipeline)")
                try:
                    state["last_routed_text"] = text
                except Exception:
                    pass
                return "confirm"
            # Otherwise continue asking for missing pieces
            if not _icp_complete(icp):
                logger.info("router -> icp (Finder gating: need website + seeds before confirm)")
                return "icp"
        # After confirm, do not auto-run enrichment. Hold for explicit 'run enrichment'.
        if not state.get("candidates"):
            logger.info("router -> candidates (user confirmed ICP; prepare candidates)")
            try:
                state["last_routed_text"] = text
            except Exception:
                pass
            return "candidates"
        logger.info("router -> end (confirmed; waiting for 'run enrichment')")
        return "end"

    # 6) ICP intake progression: if ICP is in progress and incomplete, route to ICP to parse the user's reply
    try:
        asks = dict(state.get("ask_counts") or {})
    except Exception:
        asks = {}
    try:
        if ENABLE_ICP_INTAKE:
            if bool(state.get("icp_in_progress")) and not _icp_required_fields_done(icp or {}, asks):
                logger.info("router -> icp (continue ICP intake)")
                try:
                    state["last_routed_text"] = text
                except Exception:
                    pass
                return "icp"
        else:
            # Non-Finder mode: continue asking until basic ICP is complete
            if (icp and not _icp_complete(icp)):
                logger.info("router -> icp (continue ICP basics)")
                try:
                    state["last_routed_text"] = text
                except Exception:
                    pass
                return "icp"
    except Exception:
        pass

    # 7) Default: do nothing until explicit intent
    logger.info("router -> end (no explicit lead‑gen intent)")
    return "end"


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
    # Welcome node for greetings without kicking off lead-gen automatically
    def welcome(state: GraphState) -> GraphState:
        try:
            msg = (
                "Hi! I can answer questions about lead generation. "
                "When you’re ready to begin, say ‘start lead gen’, ‘find leads’, or ‘run enrichment’."
            )
            state["messages"] = add_messages(state.get("messages") or [], [AIMessage(msg)])
            state["welcomed"] = True
        except Exception:
            pass
        return state
    g.add_node("welcome", welcome)
    # Lead‑gen Q&A node for free‑form questions
    def leadgen_qa(state: GraphState) -> GraphState:
        try:
            q = _last_user_text(state)
            # Build a richer runtime context for smart, system-only answers
            try:
                run_now_limit = int(os.getenv("CHAT_ENRICH_LIMIT", os.getenv("RUN_NOW_LIMIT", "10") or 10))
            except Exception:
                run_now_limit = 10
            ctx = {
                "icp": state.get("icp") or {},
                "candidates_count": (len(state.get("candidates") or []) if isinstance(state.get("candidates"), list) else 0),
                "results_count": (len(state.get("results") or []) if isinstance(state.get("results"), list) else 0),
                "scored_count": (len(state.get("scored") or []) if isinstance(state.get("scored"), list) else 0),
                "ENABLE_ICP_INTAKE": bool(ENABLE_ICP_INTAKE),
                "ENABLE_AGENT_DISCOVERY": bool(ENABLE_AGENT_DISCOVERY),
                "ENABLE_ACRA_IN_CHAT": bool(ENABLE_ACRA_IN_CHAT),
                "finder_suggestions_done": bool(state.get("finder_suggestions_done")),
                "micro_icp_selected": bool(state.get("micro_icp_selected")),
                "icp_match_total": state.get("icp_match_total"),
                "acra_total_suggested": state.get("acra_total_suggested"),
                "enrich_now_planned": state.get("enrich_now_planned"),
                "RUN_NOW_LIMIT": run_now_limit,
            }
            if _qa_answer is not None:
                ans = _qa_answer(q, context=ctx)
            else:
                ans = "I can answer lead‑gen questions once the LLM is configured."
            state["messages"] = add_messages(state.get("messages") or [], [AIMessage(ans)])
            state["last_answered_text"] = (q or "").strip().lower()
        except Exception as e:
            try:
                state["messages"] = add_messages(state.get("messages") or [], [AIMessage(f"I hit an error answering that: {e}")])
            except Exception:
                pass
        return state
    g.add_node("leadgen_qa", leadgen_qa)
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
        "leadgen_qa": "leadgen_qa",
        "welcome": "welcome",
        "end": END,
    }
    # Start in the router so we always decide the right first step
    g.set_entry_point("router")
    g.add_conditional_edges("router", router, mapping)
    # Every worker node loops back to the router
    # Route all worker nodes back to router EXCEPT welcome, which should end the run
    for node in ("icp", "candidates", "confirm", "accept", "enrich", "score"):
        g.add_edge(node, "router")
    return g.compile()


GRAPH = build_graph()
