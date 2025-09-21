from __future__ import annotations

import asyncio
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
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from app.odoo_store import OdooStore
from psycopg2.extras import Json
from src.database import get_pg_pool, get_conn
from src.icp import _find_ssic_codes_by_terms, _select_acra_by_ssic_codes
from src.enrichment import enrich_company_with_tavily
from src.lead_scoring import lead_scoring_agent
from src.settings import ODOO_POSTGRES_DSN

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
        icp = dict(state.get("icp") or {})
        payload = _icp_payload_from_state_icp(icp)
        if payload:
            tid = _resolve_tenant_id_for_write_sync(state)
            if isinstance(tid, int):
                _save_icp_rule_sync(tid, payload, name="Default ICP")
    except Exception:
        pass
    state["messages"].append(
        AIMessage(
            "✅ ICP saved. Paste companies (comma-separated), or type **run enrichment**."
        )
    )
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
# LangGraph nodes
# ------------------------------


async def icp_node(state: GraphState) -> GraphState:
    # If the user already confirmed, don't speak again; allow router to branch to confirm.
    if _user_just_confirmed(state):
        state["icp_confirmed"] = True
        return state

    text = _last_user_text(state)

    # 1) Extract structured update
    update = await extract_update_from_text(text)

    icp = dict(state.get("icp") or {})

    # 2) Merge extractor output into ICP
    if update.industries:
        icp["industries"] = sorted(
            set([s.strip() for s in update.industries if s.strip()])
        )
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
    lines: list[str] = []
    lines.append(f"Got {n} companies. ")
    try:
        if terms:
            ssic_matches = _find_ssic_codes_by_terms(terms)
            if ssic_matches:
                top_code, top_title, _ = ssic_matches[0]
                lines.append(
                    f"Matched {len(ssic_matches)} SSIC codes (top: {top_code} {top_title} …)"
                )
            else:
                lines.append("Matched 0 SSIC codes")
            # ACRA sample
            try:
                codes = {c for (c, _t, _s) in ssic_matches}
                rows = await asyncio.to_thread(_select_acra_by_ssic_codes, codes, 50)
            except Exception:
                rows = []
            m = len(rows)
            if m:
                lines.append(f"- Found {m} ACRA candidates. Sample:")
                for r in rows[:2]:
                    uen = (r.get("uen") or "").strip()
                    nm = (r.get("entity_name") or "").strip()
                    code = (r.get("primary_ssic_code") or "").strip()
                    status = (r.get("entity_status_description") or "").strip()
                    lines.append(
                        f"UEN: {uen} – {nm} – SSIC {code} – status: {status}"
                    )
                # Fallback: if we still have no candidates selected from companies,
                # derive initial candidates directly from ACRA sample (name + UEN).
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
                lines.append("- Found 0 ACRA candidates.")
    except Exception:
        pass
    if n > 0:
        lines.append("Started Enrichment. Please wait...")
    else:
        lines.append("No candidates yet. I’ll keep collecting ICP details.")
    msg = "\n".join([ln for ln in lines if ln])

    state["messages"] = add_messages(state.get("messages") or [], [AIMessage(content=msg)])
    return state


async def confirm_node(state: GraphState) -> GraphState:
    state["confirmed"] = True
    # Persist ICP captured in the dynamic graph flow
    try:
        icp = dict(state.get("icp") or {})
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

    n = len(state.get("candidates") or [])

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
                rows = await asyncio.to_thread(_select_acra_by_ssic_codes, codes, 50)
            except Exception:
                rows = []
            m = len(rows)
            if m:
                msg_lines.append(f"- Found {m} ACRA candidates. Sample:")
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

    if n > 0:
        msg_lines.append("Started Enrichment. Please wait...")
    else:
        msg_lines.append("No candidates yet. I’ll keep collecting ICP details.")
    text = "\n".join([ln for ln in msg_lines if ln])

    # Signal that we've shown the SSIC/ACRA preview
    state["ssic_probe_done"] = True

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
        return {"company_id": cid, "name": name, "uen": uen, "completed": completed}

    results = await asyncio.gather(*[_enrich_one(c) for c in candidates])
    all_done = all(bool(r.get("completed")) for r in results) if results else False
    state["results"] = results
    state["enrichment_completed"] = all_done

    if all_done:
        # Compose completion message; mention nightly continuation generically
        done_msg = f"Enrichment complete for {len(results)} companies. The enrichment pipeline will continue by nightly runner."
        state["messages"] = add_messages(
            state.get("messages") or [],
            [AIMessage(content=done_msg)],
        )
        # Trigger lead scoring pipeline and persist scores for UI consumption
        try:
            ids = [
                r.get("company_id") for r in results if r.get("company_id") is not None
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

                # Best-effort Odoo sync for completed companies
                try:
                    pool = await get_pg_pool()
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

                        score_rows = await conn.fetch(
                            "SELECT company_id, score, rationale FROM lead_scores WHERE company_id = ANY($1::int[])",
                            ids,
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
                        for cid in ids:
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

    # 1) Pipeline progression (allow auto-scoring even if assistant spoke last)
    has_candidates = bool(state.get("candidates"))
    has_results = bool(state.get("results"))
    has_scored = bool(state.get("scored"))
    enrichment_completed = bool(state.get("enrichment_completed"))

    if has_candidates and not has_results:
        logger.info("router -> enrich (have candidates, no enrichment)")
        return "enrich"
    if has_results and enrichment_completed and not has_scored:
        logger.info("router -> score (have enrichment, no scores, all completed)")
        return "score"
    if has_results and not enrichment_completed and not has_scored:
        logger.info("router -> end (enrichment not fully completed)")
        return "end"

    # 2) If assistant spoke last and no pending work, wait for user input
    if _last_is_ai(msgs):
        logger.info("router -> end (assistant last, waiting on user)")
        return "end"

    # 3) Fast-path: user requested enrichment
    if "run enrichment" in text:
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
        logger.info("router -> candidates (explicit company list)")
        return "candidates"

    # 5) User said confirm: proceed forward once (avoid loops)
    if _user_just_confirmed(state):
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


def build_graph():
    g = StateGraph(GraphState)
    # Central router node (no-op) to hub all control flow
    g.add_node("router", router_entry)
    g.add_node("icp", icp_node)
    g.add_node("candidates", candidates_node)
    g.add_node("confirm", confirm_node)
    g.add_node("enrich", enrich_node)
    g.add_node("score", score_node)
    # Central router: every node returns here so we can advance the workflow
    mapping = {
        "icp": "icp",
        "candidates": "candidates",
        "confirm": "confirm",
        "enrich": "enrich",
        "score": "score",
        "end": END,
    }
    # Start in the router so we always decide the right first step
    g.set_entry_point("router")
    g.add_conditional_edges("router", router, mapping)
    # Every worker node loops back to the router
    for node in ("icp", "candidates", "confirm", "enrich", "score"):
        g.add_edge(node, "router")
    return g.compile()


GRAPH = build_graph()
