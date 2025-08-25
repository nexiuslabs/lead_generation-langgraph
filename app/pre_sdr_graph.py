from __future__ import annotations
from typing import TypedDict, List, Dict, Any
import os, asyncio, logging, inspect
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from src.database import get_pg_pool
from src.enrichment import enrich_company_with_tavily
from src.lead_scoring import lead_scoring_agent
from typing import Optional
import re

# ---------- logging ----------
logger = logging.getLogger("presdr")
_level = os.getenv("LOG_LEVEL","INFO").upper()
if not logger.handlers:
    h = logging.StreamHandler()
    fmt = logging.Formatter("[%(levelname)s] %(asctime)s %(name)s :: %(message)s", "%H:%M:%S")
    h.setFormatter(fmt)
    logger.addHandler(h)
logger.setLevel(_level)

# ---------- DB table names (env-overridable) ----------
COMPANY_TABLE = os.getenv("COMPANY_TABLE", "companies")
LEAD_SCORES_TABLE = os.getenv("LEAD_SCORES_TABLE", "lead_scores")

class PreSDRState(TypedDict, total=False):
    messages: List[BaseMessage]
    icp: Dict[str, Any]
    candidates: List[Dict[str, Any]]
    results: List[Dict[str, Any]]

def _last_text(msgs) -> str:
    if not msgs:
        return ""
    m = msgs[-1]
    if isinstance(m, BaseMessage):
        return (m.content or "")
    if isinstance(m, dict):
        return (m.get("content") or "")
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
        state["messages"].append(AIMessage("Which industries or problem spaces? (e.g., SaaS, Pro Services)"))
        icp["industry"] = True
        return state
    if "employees" not in icp:
        state["messages"].append(AIMessage("Typical company size? (e.g., 10–200 employees)"))
        icp["employees"] = True
        return state
    if "geo" not in icp:
        state["messages"].append(AIMessage("Primary geographies? (SG, SEA, global)"))
        icp["geo"] = True
        return state
    if "signals" not in icp:
        state["messages"].append(AIMessage("Buying signals? (hiring, stack, certifications)"))
        icp["signals"] = True
        return state

    state["messages"].append(AIMessage("Great. Reply **confirm** to save, or tell me what to change."))
    return state

@log_node("confirm")
def icp_confirm(state: PreSDRState) -> PreSDRState:
    state["messages"].append(AIMessage("✅ ICP saved. Paste companies (comma-separated), or type **run enrichment**."))
    return state

@log_node("candidates")
def parse_candidates(state: PreSDRState) -> PreSDRState:
    last = _last_text(state.get("messages"))
    names = [n.strip() for n in last.split(",") if 1 < len(n.strip()) < 120]
    if names:
        state["candidates"] = [{"name": n} for n in names]
        state["messages"].append(AIMessage(f"Got {len(names)} companies. Type **run enrichment** to start."))
    else:
        state["messages"].append(AIMessage("Please paste a few company names (comma-separated)."))
    return state

@log_node("enrich")
async def run_enrichment(state: PreSDRState) -> PreSDRState:
    # PLACEHOLDER: you still call your enrichment + Odoo writes here.
    await asyncio.sleep(0.01)
    cands = state.get("candidates") or []
    state["results"] = [{"name": c["name"], "score": 0.7} for c in cands]
    state["messages"].append(AIMessage(f"Enrichment complete for {len(cands)} companies."))
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
    g.add_conditional_edges("icp", route, {
        "confirm": "confirm",
        "enrich": "enrich",
        "candidates": "candidates",
        "icp": "icp",
    })
    g.add_conditional_edges("confirm", route, {
        "enrich": "enrich",
        "candidates": "candidates",
        "icp": "icp",
    })
    g.add_conditional_edges("candidates", route, {
        "enrich": "enrich",
        "icp": "icp",
    })
    g.add_edge("enrich", END)
    return g.compile()

# ------------------------------
# New LLM-driven Pre-SDR graph (dynamic Q&A, structured extraction)
# ------------------------------

class GraphState(TypedDict):
    messages: List[BaseMessage]
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
    "none","no","n/a","na","skip","any","nope","not important",
    "no preference","doesn't matter","dont care","don't care",
    "anything","no specific","no specific signals","no signal","no signals"
}
def _says_none(text: str) -> bool:
    t = text.strip().lower()
    return any(p in t for p in NEG_NONE)

def _user_just_confirmed(state: dict) -> bool:
    msgs = state.get("messages") or []
    for m in reversed(msgs):
        if isinstance(m, HumanMessage):
            txt = (getattr(m, "content", "") or "").strip().lower()
            return txt in {"confirm","yes","y","ok","okay","looks good","lgtm"}
    return False

def _icp_complete(icp: Dict[str, Any]) -> bool:
    has_industries = bool(icp.get("industries"))
    has_employees = bool(icp.get("employees_min") or icp.get("employees_max"))
    has_geos = bool(icp.get("geos"))
    signals_done = bool(icp.get("signals")) or bool(icp.get("signals_done"))
    # Require industries + employees + geos, and either explicit signals or explicit skip (signals_done)
    return has_industries and has_employees and has_geos and signals_done

def _parse_company_list(text: str) -> List[str]:
    raw = re.split(r"[,|\n]+", text)
    names = [n.strip() for n in raw if n.strip()]
    return [n for n in names if n.lower() not in {"start", "confirm", "run enrichment"}]

# ------------------------------
# Structured extraction
# ------------------------------

class ICPUpdate(BaseModel):
    industries: List[str] = Field(default_factory=list)
    employees_min: Optional[int] = Field(default=None)
    employees_max: Optional[int] = Field(default=None)
    geos: List[str] = Field(default_factory=list)
    signals: List[str] = Field(default_factory=list)
    confirm: bool = Field(default=False)
    pasted_companies: List[str] = Field(default_factory=list)
    signals_done: bool = Field(
        default=False,
        description="True if user said skip/none/any for buying signals",
    )

EXTRACT_SYS = SystemMessage(content=(
    "You extract ICP details from user messages.\n"
    "Return JSON ONLY with industries (list[str]), employees_min/max (ints if present), "
    "geos (list[str]), signals (list[str]), confirm (bool), pasted_companies (list[str]), "
    "and signals_done (bool).\n"
    "If the user indicates no preference for buying signals (e.g., 'none', 'any', 'skip'), "
    "set signals_done=true and signals=[]. If the user pasted company names (comma or newline separated), "
    "put them into pasted_companies."
))

async def extract_update_from_text(text: str) -> ICPUpdate:
    structured = EXTRACT_LLM.with_structured_output(ICPUpdate)
    return await structured.ainvoke([EXTRACT_SYS, HumanMessage(text)])

# ------------------------------
# Dynamic question generation
# ------------------------------

QUESTION_SYS = SystemMessage(content=(
    "You are an expert SDR assistant. Ask exactly ONE short question at a time to help define an Ideal Customer Profile (ICP). "
    "Keep it brief, concrete, and practical. If ICP looks complete, ask the user to confirm or adjust."
))

def next_icp_question(icp: Dict[str, Any]) -> tuple[str, str]:
    order: List[str] = []
    if not icp.get("industries"):
        order.append("industries")
    if not (icp.get("employees_min") or icp.get("employees_max")):
        order.append("employees")
    if not icp.get("geos"):
        order.append("geos")
    if not icp.get("signals") and not icp.get("signals_done", False):
        order.append("signals")

    if not order:
        return ("Does this ICP look right? Reply **confirm** to save, or tell me what to change.", "confirm")

    focus = order[0]
    prompts = {
        "industries": "Which industries or problem spaces should we target? (e.g., SaaS, logistics, fintech)",
        "employees": "What's the typical employee range? (e.g., 10–200)",
        "geos": "Which geographies or markets? (e.g., SG, SEA, global)",
        "signals": "What specific buying signals are you looking for (e.g., hiring for data roles, ISO 27001, AWS partner)?",
    }
    return (prompts[focus], focus)

# ------------------------------
# Persistence helpers
# ------------------------------

async def _ensure_company_row(pool, name: str) -> int:
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT id FROM companies WHERE name = $1", name)
        if row:
            return row["id"]
        row = await conn.fetchrow("INSERT INTO companies(name) VALUES ($1) RETURNING id", name)
        return row["id"]

async def _default_candidates(pool, icp: Dict[str, Any], limit: int = 20) -> List[Dict[str, Any]]:
    """
    Pull candidates from companies using basic ICP filters:
    - industry (industry_norm ILIKE)
    - employees_min/max (employees_est range)
    - geos (hq_country/hq_city ILIKE any)
    Falls back gracefully if filters are missing.
    """
    icp = icp or {}
    industry = icp.get("industry")
    if not industry:
        inds = icp.get("industries") or []
        if isinstance(inds, list) and inds:
            industry = inds[0]
    emp_min = icp.get("employees_min")
    emp_max = icp.get("employees_max")
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

    if industry:
        clauses.append(f"c.industry_norm ILIKE ${len(params)+1}")
        params.append(f"%{industry}%")
    if isinstance(emp_min, int):
        clauses.append(f"c.employees_est >= ${len(params)+1}")
        params.append(emp_min)
    if isinstance(emp_max, int):
        clauses.append(f"c.employees_est <= ${len(params)+1}")
        params.append(emp_max)
    if isinstance(geos, list) and geos:
        # Build an OR group for geos across hq_country/hq_city
        geo_like_params = []
        geo_subclauses = []
        for g in geos:
            if not isinstance(g, str) or not g.strip():
                continue
            like_val = f"%{g.strip()}%"
            # country match
            geo_subclauses.append(f"c.hq_country ILIKE ${len(params)+len(geo_like_params)+1}")
            geo_like_params.append(like_val)
            # city match
            geo_subclauses.append(f"c.hq_city ILIKE ${len(params)+len(geo_like_params)+1}")
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

    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, *params)

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
        icp["industries"] = sorted(set([s.strip() for s in update.industries if s.strip()]))
    if update.employees_min is not None:
        icp["employees_min"] = update.employees_min
    if update.employees_max is not None:
        icp["employees_max"] = update.employees_max
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
        new_msgs.append(AIMessage(content=f"Got {len(update.pasted_companies)} companies. Type **run enrichment** to start."))

    # 4) Back-off: if we already asked about 'signals' once and still don't have them, stop asking
    ask_counts = dict(state.get("ask_counts") or {})
    q, focus = next_icp_question(icp)
    if focus == "signals" and ask_counts.get("signals", 0) >= 1 and not icp.get("signals"):
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
    state["messages"] = add_messages(
        state.get("messages") or [],
        [AIMessage(content=f"Got {n} companies. Type **run enrichment** to start.")],
    )
    return state

async def confirm_node(state: GraphState) -> GraphState:
    state["confirmed"] = True
    state["messages"] = add_messages(
        state.get("messages") or [],
        [AIMessage(content="✅ ICP saved. Paste companies (comma-separated), or type **run enrichment**.")],
    )
    return state

async def enrich_node(state: GraphState) -> GraphState:
    text = _last_user_text(state)
    if not state.get("candidates"):
        pasted = _parse_company_list(text)
        if pasted:
            state["candidates"] = [{"name": n} for n in pasted]

    candidates = state.get("candidates") or []
    if not candidates:
        state["messages"] = add_messages(
            state.get("messages") or [],
            [AIMessage(content="I need some companies. Paste a list (comma-separated) or say **run enrichment** to use suggestions.")],
        )
        return state

    pool = await get_pg_pool()

    async def _enrich_one(c: Dict[str, Any]) -> Dict[str, Any]:
        name = c["name"]
        cid = c.get("id") or await _ensure_company_row(pool, name)
        uen = c.get("uen")
        # Your pipeline is async in this codebase; run concurrently via gather
        await enrich_company_with_tavily(cid, name, uen)
        return {"company_id": cid, "name": name, "uen": uen}

    results = await asyncio.gather(*[_enrich_one(c) for c in candidates])
    state["results"] = results
    state["messages"] = add_messages(
        state.get("messages") or [],
        [AIMessage(content=f"Enrichment complete for {len(results)} companies.")],
    )
    # Trigger lead scoring pipeline and persist scores for UI consumption
    try:
        ids = [r.get("company_id") for r in results if r.get("company_id") is not None]
        if ids:
            scoring_initial_state = {
                "candidate_ids": ids,
                "lead_features": [],
                "lead_scores": [],
                "icp_payload": {
                    "employee_range": {
                        "min": (state.get("icp") or {}).get("employees_min"),
                        "max": (state.get("icp") or {}).get("employees_max"),
                    }
                },
            }
            await lead_scoring_agent.ainvoke(scoring_initial_state)
    except Exception as _score_exc:
        logger.warning("lead scoring failed: %s", _score_exc)
    return state

def _fmt_table(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "No candidates found."
    headers = ["Name", "Domain", "Industry", "Employees", "Score", "Bucket"]
    md = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"]*len(headers)) + "|"]
    for r in rows:
        md.append("| " + " | ".join([
            str(r.get("name", "")),
            str(r.get("domain", "")),
            str(r.get("industry", "")),
            str(r.get("employee_count", "")),
            str(r.get("lead_score", "")),
            str(r.get("lead_bucket", "")),
        ]) + " |")
    return "\n".join(md)

async def score_node(state: GraphState) -> GraphState:
    pool = await get_pg_pool()
    cands = state.get("candidates") or []
    ids = [c.get("id") for c in cands if c.get("id") is not None]

    if not ids:
        table = _fmt_table([])
        state["messages"] = add_messages(
            state.get("messages") or [],
            [AIMessage(content=f"Here are your leads:\n\n{table}")],
        )
        return state

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT company_id, score, bucket, rationale
            FROM public.{LEAD_SCORES_TABLE}
            WHERE company_id = ANY($1::int[])
            """,
            ids,
        )
    by_id = {r["company_id"]: dict(r) for r in rows}

    scored: List[Dict[str, Any]] = []
    for c in cands:
        sc = by_id.get(c.get("id"))
        c_out = {**c}
        if sc:
            c_out["lead_score"] = sc.get("score")
            c_out["lead_bucket"] = sc.get("bucket")
            c_out["lead_rationale"] = sc.get("rationale")
        scored.append(c_out)

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

    # 2) Fast-path: user said confirm -> proceed to candidates (allow light ICP)
    if _user_just_confirmed(state):
        logger.info("router -> candidates (user confirmed ICP)")
        return "candidates"

    # 3) Fast-path: user requested enrichment
    if "run enrichment" in text:
        if state.get("candidates"):
            logger.info("router -> enrich (user requested enrichment)")
            return "enrich"
        else:
            logger.info("router -> candidates (prepare candidates before enrichment)")
            return "candidates"

    # 4) If user pasted an explicit company list, jump to candidates
    if _parse_company_list(text):
        logger.info("router -> candidates (explicit company list)")
        return "candidates"

    # 5) If ICP is not complete yet, continue ICP Q&A
    if not _icp_complete(icp):
        logger.info("router -> icp (need more ICP)")
        return "icp"

    # 6) Pipeline progression (allow auto-scoring even if assistant spoke last)
    has_candidates = bool(state.get("candidates"))
    has_results = bool(state.get("results"))
    has_scored = bool(state.get("scored"))

    if has_candidates and not has_results:
        logger.info("router -> enrich (have candidates, no enrichment)")
        return "enrich"
    if has_results and not has_scored:
        logger.info("router -> score (have enrichment, no scores)")
        return "score"

    # 6b) If no pending work, pause when assistant was last speaker
    if _last_is_ai(msgs):
        logger.info("router -> end (no pending work; assistant last)")
        return "end"

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
