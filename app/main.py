# app/main.py
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from app.pre_sdr_graph import build_graph
from src.database import get_pg_pool
import csv
from io import StringIO

app = FastAPI(title="Pre-SDR LangGraph Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

graph = build_graph()

def _role_to_type(role: str) -> str:
    r = (role or "").lower()
    if r in ("user", "human"): return "human"
    if r in ("assistant", "ai"): return "ai"
    if r == "system": return "system"
    return "human"

def _to_message(msg: dict) -> BaseMessage:
    # Accepts {"role":"human","content":"..."} or {"type":"ai","content":"..."}
    mtype = msg.get("type") or _role_to_type(msg.get("role", "human"))
    content = msg.get("content", "")
    if mtype == "human":
        return HumanMessage(content=content)
    if mtype == "system":
        return SystemMessage(content=content)
    return AIMessage(content=content)  # default to AI

def normalize_input(payload: dict) -> dict:
    """
    Accept a variety of UI payloads and emit the graph state:
      {"messages": [BaseMessage, ...], "candidates": [...]}
    """
    data = payload.get("input", payload) or {}
    msgs = data.get("messages") or []
    if isinstance(msgs, dict):  # sometimes a single message object is sent
        msgs = [msgs]
    norm_msgs = [_to_message(m) if not isinstance(m, BaseMessage) else m for m in msgs]

    # Ensure we always have at least one message
    if not norm_msgs:
        norm_msgs = [HumanMessage(content="")]

    state = {"messages": norm_msgs}

    # pass-through optional fields you use (companies/candidates)
    if "candidates" in data:
        state["candidates"] = data["candidates"]
    elif "companies" in data:
        state["candidates"] = data["companies"]

    return state

ui_adapter = RunnableLambda(normalize_input) | graph

# expose adapted runnable so /agent accepts role-based payloads
add_routes(app, ui_adapter, path="/agent")

@app.get("/health")
def health():
    return {"ok": True}


# --- Lightweight tenant middleware (optional header-based) ---
@app.middleware("http")
async def tenant_middleware(request: Request, call_next):
    # Extract tenant_id from header (e.g., set by SSO/edge); no validation here
    tenant_id = request.headers.get("X-Tenant-ID") or None
    request.state.tenant_id = tenant_id
    return await call_next(request)


# --- Export endpoints (JSON/CSV) ---
@app.get("/export/latest_scores.json")
async def export_latest_scores_json(limit: int = 200):
    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT c.company_id, c.name, c.website_domain, c.industry_norm, c.employees_est,
                   s.score, s.bucket, s.rationale
            FROM companies c
            JOIN lead_scores s ON s.company_id = c.company_id
            ORDER BY s.score DESC NULLS LAST
            LIMIT $1
            """,
            limit,
        )
    return [dict(r) for r in rows]


@app.get("/export/latest_scores.csv")
async def export_latest_scores_csv(limit: int = 200):
    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT c.company_id, c.name, c.website_domain, c.industry_norm, c.employees_est,
                   s.score, s.bucket, s.rationale
            FROM companies c
            JOIN lead_scores s ON s.company_id = c.company_id
            ORDER BY s.score DESC NULLS LAST
            LIMIT $1
            """,
            limit,
        )
    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()) if rows else [
        "company_id","name","website_domain","industry_norm","employees_est","score","bucket","rationale"
    ])
    writer.writeheader()
    for r in rows:
        writer.writerow(dict(r))
    return Response(content=buf.getvalue(), media_type="text/csv")
