"""
Node implementations for the unified orchestrator.

These implementations leverage LangChain/OpenAI when available and gracefully
degrade to rule-based heuristics when credentials are missing. Downstream
LangGraph agents (normalization, ICP refresh, etc.) are reused wherever
possible so behavior matches the legacy stack.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List

from src.icp import icp_by_ssic_agent, icp_refresh_agent, normalize_agent
from src.lead_scoring import lead_scoring_agent
from src.jobs import enqueue_web_discovery_bg_enrich

from .llm import call_llm_json
from .state import OrchestrationState, ProfileState

try:  # optional dependency (network / vendor credentials)
    from src.enrichment import enrich_company_with_tavily
except Exception:  # pragma: no cover
    enrich_company_with_tavily = None

try:
    from src.agents_icp import plan_top10_with_reasons
except Exception:  # pragma: no cover
    plan_top10_with_reasons = None

logger = logging.getLogger(__name__)
OFFLINE = (os.getenv("ORCHESTRATOR_OFFLINE") or "").lower() in {"1", "true", "yes"}

# ---------------------------------------------------------------------------
# Conversation helpers
# ---------------------------------------------------------------------------


def _ensure_messages(state: OrchestrationState) -> None:
    state.setdefault("messages", [])


def _append_message(state: OrchestrationState, role: str, content: str) -> None:
    if not content.strip():
        return
    _ensure_messages(state)
    state["messages"].append({"role": role, "content": content})


def _set_status(state: OrchestrationState, phase: str, message: str) -> None:
    state["status"] = {"phase": phase, "message": message}
    history = state.setdefault("status_history", [])
    history.append(
        {
            "phase": phase,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )


def _ensure_profile_state(state: OrchestrationState) -> ProfileState:
    profile = state.get("profile_state") or {}
    profile.setdefault("outstanding_prompts", [])
    state["profile_state"] = profile
    return profile


def _simple_intent(text: str) -> str:
    text = (text or "").lower()
    if not text:
        return "idle"
    if "run enrichment" in text or "finish pipeline" in text:
        return "run_enrichment"
    if "confirm icp" in text:
        return "confirm_icp"
    if "confirm company" in text:
        return "confirm_company"
    if "accept micro-icp" in text:
        return "accept_micro_icp"
    return "chat"


async def ingest_message(state: OrchestrationState) -> OrchestrationState:
    """Normalize incoming payloads using an LLM (fallback to heuristics)."""
    incoming = str(state.get("input") or "").strip()
    role = state.get("input_role", "user")
    if incoming:
        _append_message(state, role, incoming)
        prompt = f"""
You normalize chat inputs for a lead-generation orchestrator.
Respond with JSON {{"normalized_text": "...", "intent": "...", "tags": [...]}}.
Input: {incoming!r}
Intent options: run_enrichment, confirm_company, confirm_icp, accept_micro_icp, chat, idle.
"""
        fallback = {
            "normalized_text": incoming,
            "intent": _simple_intent(incoming),
            "tags": [],
        }
        analysis = call_llm_json(prompt, fallback)
        ctx = state.get("entry_context") or {}
        ctx.update(
            {
                "last_user_command": analysis.get("normalized_text", incoming),
                "intent": analysis.get("intent", fallback["intent"]),
                "tags": analysis.get("tags", []),
                "last_received_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        state["entry_context"] = ctx
    msg = "Processed user input" if incoming else "Waiting for user input"
    _set_status(state, "ingest", msg)
    return state


async def profile_builder(state: OrchestrationState) -> OrchestrationState:
    """Update profile flags using the latest conversation snippet."""
    profile = _ensure_profile_state(state)
    history = state.get("messages", [])[-5:]
    prompt = f"""
You maintain company + ICP profile confirmations based on conversation snippets.
Return JSON with keys company_profile_confirmed, icp_profile_confirmed, micro_icp_selected (bools).
History: {history!r}
Current flags: {profile!r}
"""
    fallback = {
        "company_profile_confirmed": bool(profile.get("company_profile_confirmed")),
        "icp_profile_confirmed": bool(profile.get("icp_profile_confirmed")),
        "micro_icp_selected": bool(profile.get("micro_icp_selected")),
    }
    result = call_llm_json(prompt, fallback)
    for key in fallback:
        profile[key] = bool(result.get(key, fallback[key]))
    profile["last_updated_at"] = datetime.now(timezone.utc).isoformat()
    _set_status(state, "profile_builder", "Profile snapshot updated")
    return state


async def journey_guard(state: OrchestrationState) -> OrchestrationState:
    """Ensure prerequisites exist before backend work."""
    profile = _ensure_profile_state(state)
    company_ready = bool(profile.get("company_profile_confirmed"))
    icp_ready = bool(profile.get("icp_profile_confirmed"))
    if company_ready and icp_ready:
        profile["outstanding_prompts"] = []
        msg = "Prerequisites satisfied. Proceeding to normalization."
    else:
        missing = []
        if not company_ready:
            missing.append("company profile confirmation")
        if not icp_ready:
            missing.append("ICP confirmation")
        prompt = f"Please provide {', '.join(missing)}."
        profile["outstanding_prompts"] = [prompt]
        msg = prompt
    _set_status(state, "journey_guard", msg)
    return state


# ---------------------------------------------------------------------------
# Normalization / discovery
# ---------------------------------------------------------------------------


async def normalize(state: OrchestrationState) -> OrchestrationState:
    result = {"processed_rows": 0, "errors": [], "last_run_at": datetime.now(timezone.utc).isoformat()}
    if OFFLINE:
        result["processed_rows"] = 5
    else:
        try:
            norm_state = await normalize_agent.ainvoke({"raw_records": [], "normalized_records": []})
            result["processed_rows"] = len(norm_state.get("normalized_records") or [])
        except Exception as exc:  # pragma: no cover - external dependencies
            logger.warning("normalize_agent failed: %s", exc)
            result["errors"].append(str(exc))
    state["normalize"] = result
    _set_status(state, "normalize", f"Normalized {result['processed_rows']} rows")
    return state


async def refresh_icp(state: OrchestrationState) -> OrchestrationState:
    payload = state.get("icp_payload") or {}
    discovery = state.get("discovery") or {}
    if OFFLINE:
        discovery["candidate_ids"] = [101, 102, 103]
        discovery["diagnostics"] = {"payload": payload, "offline": True}
        msg = f"[offline] Found {len(discovery['candidate_ids'])} candidates"
    else:
        try:
            icp_state = await icp_refresh_agent.ainvoke({"payload": payload, "candidate_ids": []})
            discovery["candidate_ids"] = icp_state.get("candidate_ids") or []
            discovery["diagnostics"] = {"payload": payload}
            msg = f"Found {len(discovery['candidate_ids'])} candidates"
        except Exception as exc:  # pragma: no cover
            logger.warning("icp_refresh_agent failed: %s", exc)
            discovery["candidate_ids"] = []
            discovery["diagnostics"] = {"error": str(exc)}
            msg = "ICP refresh failed"
    state["discovery"] = discovery
    _set_status(state, "refresh_icp", msg)
    return state


async def decide_strategy(state: OrchestrationState) -> OrchestrationState:
    discovery = state.get("discovery") or {}
    candidate_count = len(discovery.get("candidate_ids") or [])
    last_ssic = discovery.get("last_ssic_attempt")
    prompt = f"""
Choose discovery action (use_cached, regenerate, ssic_fallback).
Return JSON {{"action": "...", "reason": "..."}}.
Candidate count: {candidate_count}
Last SSIC attempt: {last_ssic}
"""
    fallback = {"action": "use_cached" if candidate_count else "ssic_fallback", "reason": "heuristic fallback"}
    result = call_llm_json(prompt, fallback)
    discovery["strategy"] = result.get("action", fallback["action"])
    state["discovery"] = discovery
    _set_status(state, "decide_strategy", result.get("reason", fallback["reason"]))
    return state


async def ssic_fallback(state: OrchestrationState) -> OrchestrationState:
    discovery = state.get("discovery") or {}
    profile = state.get("profile_state") or {}
    industries = profile.get("icp_profile", {}).get("industries") or []
    terms = [str(i).lower() for i in industries if isinstance(i, str)]
    try:
        ssic_state = await icp_by_ssic_agent.ainvoke({"terms": terms})
        rows = ssic_state.get("acra_candidates") or []
        candidate_ids = [int(r.get("company_id")) for r in rows if r.get("company_id") is not None]
        discovery["candidate_ids"] = candidate_ids
        discovery["last_ssic_attempt"] = datetime.now(timezone.utc).isoformat()
        msg = f"SSIC fallback produced {len(candidate_ids)} IDs"
    except Exception as exc:  # pragma: no cover
        logger.warning("SSIC fallback failed: %s", exc)
        msg = "SSIC fallback failed"
    state["discovery"] = discovery
    _set_status(state, "ssic_fallback", msg)
    return state


async def plan_top10(state: OrchestrationState) -> OrchestrationState:
    icp_profile = state.get("profile_state", {}).get("icp_profile") or {}
    items: List[Dict[str, Any]] = []
    if OFFLINE:
        items = [
            {"company_id": cid, "name": f"Offline Co {cid}", "reason": "stubbed"}
            for cid in state.get("discovery", {}).get("candidate_ids", [])[:3]
        ]
        msg = f"[offline] Planned {len(items)} candidates"
    else:
        try:
            if plan_top10_with_reasons is None:
                raise RuntimeError("plan_top10_with_reasons unavailable")
            items = plan_top10_with_reasons(icp_profile, tenant_id=None) or []
            msg = f"Planned {len(items)} candidates"
        except Exception as exc:  # pragma: no cover
            logger.warning("plan_top10 failed: %s", exc)
            msg = "Top-10 planning failed"
    state["top10"] = {"items": items, "generated_at": datetime.now(timezone.utc).isoformat()}
    _set_status(state, "plan_top10", msg)
    return state


# ---------------------------------------------------------------------------
# Enrichment / scoring / export
# ---------------------------------------------------------------------------


async def enrich_batch(state: OrchestrationState) -> OrchestrationState:
    candidate_ids = state.get("discovery", {}).get("candidate_ids") or []
    results: List[Dict[str, Any]] = []
    if not candidate_ids:
        msg = "No candidates to enrich"
    elif OFFLINE:
        for cid in candidate_ids:
            results.append({"company_id": int(cid), "completed": True, "error": None, "offline": True})
        msg = f"[offline] Enrichment simulated for {len(results)} companies"
    elif enrich_company_with_tavily is None:  # pragma: no cover
        msg = "Enrichment helper unavailable; skipping"
    else:
        for cid in candidate_ids:  # pragma: no cover - network I/O
            try:
                asyncio.run(enrich_company_with_tavily(int(cid), search_policy="require_existing"))
                results.append({"company_id": int(cid), "completed": True, "error": None})
            except Exception as exc:
                logger.warning("enrich_company_with_tavily failed for %s: %s", cid, exc)
                results.append({"company_id": int(cid), "completed": False, "error": str(exc)})
        msg = f"Enrichment attempted for {len(results)} companies"
    state["enrichment_results"] = results
    _set_status(state, "enrich_batch", msg)
    return state


async def score_leads(state: OrchestrationState) -> OrchestrationState:
    candidate_ids = state.get("discovery", {}).get("candidate_ids") or []
    if OFFLINE:
        scores = [{"company_id": cid, "score": 0.8, "reason": "offline stub"} for cid in candidate_ids]
        msg = f"[offline] Scored {len(scores)} companies"
    else:
        try:
            scoring_state = lead_scoring_agent.invoke(
                {"candidate_ids": candidate_ids, "lead_features": [], "lead_scores": [], "icp_payload": state.get("icp_payload", {})}
            )
            scores = scoring_state.get("lead_scores") or []
            msg = f"Scored {len(scores)} companies"
        except Exception as exc:  # pragma: no cover
            logger.warning("lead_scoring_agent failed: %s", exc)
            scores = []
            msg = "Lead scoring failed"
    state["scoring"] = {"scores": scores, "last_run_at": datetime.now(timezone.utc).isoformat()}
    _set_status(state, "score_leads", msg)
    return state


async def export_results(state: OrchestrationState) -> OrchestrationState:
    exports = state.get("exports") or {
        "next40_enqueued": False,
        "odoo_exported": False,
        "job_ids": [],
    }
    exports["last_run_at"] = datetime.now(timezone.utc).isoformat()
    # Background Next-40 enqueue
    tenant_id = (state.get("entry_context") or {}).get("tenant_id")
    ids = [r.get("company_id") for r in state.get("enrichment_results") or [] if r.get("completed")]
    if OFFLINE:
        msg_extra = "; offline mode: export skipped"
    elif tenant_id and ids:
        try:
            job = enqueue_web_discovery_bg_enrich(int(tenant_id), [int(i) for i in ids if i])
            exports["next40_enqueued"] = bool(job.get("job_id"))
            exports.setdefault("job_ids", []).append(job.get("job_id"))
            msg_extra = f"; Next-40 job {job.get('job_id')} queued"
        except Exception as exc:  # pragma: no cover
            logger.warning("enqueue_web_discovery_bg_enrich failed: %s", exc)
            msg_extra = "; Next-40 enqueue failed"
    else:
        msg_extra = "; no Next-40 enqueue"
    # TODO: wire Odoo export when available
    state["exports"] = exports
    _set_status(state, "export", f"Export stage complete{msg_extra}")
    return state


async def progress_report(state: OrchestrationState) -> OrchestrationState:
    summary = {
        "phase": state.get("status", {}).get("phase"),
        "candidates": len(state.get("discovery", {}).get("candidate_ids") or []),
        "confirmed_company": bool(state.get("profile_state", {}).get("company_profile_confirmed")),
        "confirmed_icp": bool(state.get("profile_state", {}).get("icp_profile_confirmed")),
    }
    prompt = f"""
Summarize the orchestration status for the user in ≤3 sentences as JSON {{"message": "..."}}
State: {summary!r}
"""
    fallback = {"message": f"Stage: {summary['phase']} | Candidates: {summary['candidates']}"}
    result = call_llm_json(prompt, fallback)
    message = result.get("message", fallback["message"])
    _set_status(state, "progress_report", message)
    _append_message(state, "assistant", message)
    return state


async def finalize(state: OrchestrationState) -> OrchestrationState:
    msg = state.get("status", {}).get("message") or "Run complete"
    _append_message(state, "assistant", msg)
    _set_status(state, "summary", msg)
    return state
