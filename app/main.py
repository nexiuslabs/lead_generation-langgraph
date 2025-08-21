# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from app.pre_sdr_graph import build_graph

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
