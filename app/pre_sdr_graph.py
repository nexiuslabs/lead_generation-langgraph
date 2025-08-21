from __future__ import annotations
from typing import TypedDict, List, Dict, Any, Optional
import os, asyncio
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from src.enrichment import enrich_company_with_tavily
from app.odoo_store import OdooStore

store = OdooStore()

class PreSDRState(TypedDict, total=False):
    messages: List[BaseMessage]          # REQUIRED for Chat UI
    icp: Dict[str, Any]
    candidates: List[Dict[str, Any]]     # [{name, uen?}]
    progress: Dict[str, Any]
    results: List[Dict[str, Any]]

# ---- ICP discovery (simple, safe defaults for non-experts) ----
def icp_discovery(state: PreSDRState) -> PreSDRState:
    # Ask stepwise until we have enough to proceed
    text = (state["messages"][-1].content or "").lower()
    if not state.get("icp"):
        state["icp"] = {}
    icp = state["icp"]
    if "industry" not in icp:
        state["messages"].append(AIMessage("Let’s narrow your ICP. What industries or problem spaces do you target? (e.g., SaaS, Professional Services)"))
        icp["industry"] = True; return state
    if "employees" not in icp:
        state["messages"].append(AIMessage("Typical company size? (e.g., 10–200 employees)"))
        icp["employees"] = True; return state
    if "geo" not in icp:
        state["messages"].append(AIMessage("Primary geographies? (SG only, SEA, or global)"))
        icp["geo"] = True; return state
    if "signals" not in icp:
        state["messages"].append(AIMessage("Buying signals you care about? (e.g., hiring, analytics tags, case studies)"))
        icp["signals"] = True; return state
    state["messages"].append(AIMessage("Great. Reply **confirm** to save or tell me what to change."))
    return state

def icp_confirm(state: PreSDRState) -> PreSDRState:
    state["messages"].append(AIMessage("✅ ICP saved. Send company names (or type **auto** to pull from Odoo) and then type **run enrichment**."))
    return state

# ---- Candidate collection ----
async def candidates_from_odoo(state: PreSDRState) -> PreSDRState:
    # In MVP: parse any company names the user typed, else leave empty
    human = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    names = [n.strip() for n in human.split(",") if 1 < len(n.strip()) < 80]
    state["candidates"] = [{"name": n} for n in names] if names else []
    if not state["candidates"]:
        state["messages"].append(AIMessage("Please paste a small list of company names, or type `auto` (I’ll fetch from your nightly ACRA/ICP shortlist)."))
    return state

# ---- Enrichment runner (calls your existing LangGraph) ----
async def run_enrichment(state: PreSDRState) -> PreSDRState:
    out=[]
    for c in state.get("candidates", []):
        # upsert minimal company shell in Odoo so we have an id
        company_id = await store.upsert_company(name=c["name"], uen=c.get("uen"))
        final = await enrich_company_with_tavily(company_id, c["name"], uen=c.get("uen"))  # <- your graph
        data = final.get("data") or {}
        # persist extra signals (emails->contacts, score->lead)
        primary_email = (data.get("email") or [None])[0]
        await store.merge_company_enrichment(company_id, {
            "jobs_count": data.get("jobs_count"),
            "tech_stack": data.get("tech_stack") or [],
            "about_text": data.get("about_text"),
            "website_domain": data.get("website_domain"),
        })
        if primary_email:
            await store.add_contact(company_id, primary_email)
        # naive score from signals; you can replace with your lead_scoring_agent later
        score = min(0.2 + 0.02*len(data.get("tech_stack") or []) + 0.1*(1 if primary_email else 0), 0.95)
        await store.create_lead_if_high(
            company_id, f"[Pre-SDR] {c['name']}", score,
            {"tech_hits":len(data.get("tech_stack") or []),"has_email":bool(primary_email)},
            f"Tech stack {len(data.get('tech_stack') or [])}; email={'yes' if primary_email else 'no'}.",
            primary_email, float(os.getenv("LEAD_THRESHOLD","0.66"))
        )
        out.append({"company_id": company_id, "name": c["name"], "score": score})
    state["results"] = out
    state["messages"].append(AIMessage(f"Enrichment completed for {len(out)} companies."))
    return state

# ---- Router ----
async def route(state: PreSDRState) -> str:
    last_human = next((m.content.lower() for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    if "confirm" in last_human: return "confirm"
    if "run enrichment" in last_human: return "enrich"
    if ("auto" in last_human) or ("," in last_human): return "candidates"
    return "icp"

def build_presdr_graph():
    g = StateGraph(PreSDRState)
    g.add_node("icp", icp_discovery)
    g.add_node("confirm", icp_confirm)
    g.add_node("candidates", candidates_from_odoo)
    g.add_node("enrich", run_enrichment)
    g.set_entry_point("icp")
    g.add_conditional_edges("icp", route, {"confirm":"confirm","enrich":"enrich","candidates":"candidates","icp":"icp"})
    g.add_conditional_edges("confirm", route, {"icp":"icp","enrich":"enrich","candidates":"candidates"})
    g.add_conditional_edges("candidates", route, {"enrich":"enrich","icp":"icp"})
    g.add_edge("enrich", END)
    return g.compile()
