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
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List
from urllib.parse import urlparse

from src.icp import icp_by_ssic_agent, icp_refresh_agent, normalize_agent
from src.lead_scoring import lead_scoring_agent
from src.jobs import enqueue_web_discovery_bg_enrich
from src.jina_reader import read_url as jina_read

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


def _log_step(step: str, **info: Any) -> None:
    """Helper to emit structured info logs for LangGraph tracing."""
    if not logger.isEnabledFor(logging.INFO):
        return
    payload = " ".join(f"{k}={info[k]!r}" for k in sorted(info))
    logger.info("%s %s", step, payload)

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
    profile.setdefault("company_profile", {})
    profile.setdefault("icp_profile", {})
    state["profile_state"] = profile
    return profile


def _needs_company_website(profile: ProfileState) -> bool:
    company = profile.get("company_profile") or {}
    return not bool((company.get("website") or "").strip())


def _name_from_domain(url: str) -> str:
    try:
        parsed = urlparse(url)
        host = parsed.netloc or parsed.path
    except Exception:  # pragma: no cover
        host = url
    host = host.lower().lstrip("www.")
    base = host.split("/")[0]
    base = base.split(":")[0]
    base = base.split(".")[0]
    value = base.replace("-", " ").title()
    return value or "Your company"


def _fetch_site_excerpt(url: str, max_chars: int = 1500) -> str | None:
    if not url:
        return None
    try:
        raw = jina_read(url, timeout=8) or ""
    except Exception as exc:  # pragma: no cover - network variability
        logger.warning("jina_read failed for %s: %s", url, exc)
        raw = ""
    cleaned = re.sub(r"\s+", " ", raw).strip()
    if not cleaned:
        return None
    return cleaned[:max_chars]


def _structure_company_profile(website: str, snippet: str, recent: List[str]) -> Dict[str, Any]:
    prompt = f"""
You extract structured company profile data for a lead-generation workflow.
Website: {website!r}
Website snippet: {snippet!r}
Recent user snippets: {recent!r}
Return JSON with keys:
  summary (<=2 sentences string)
  industries (list of up to 3 industry labels)
  offerings (list of up to 4 products/services)
  ideal_customers (list of buyer/company descriptors)
  proof_points (list of metrics, awards, logos, or quotes)
If a field is unknown, return an empty string or empty list.
"""
    fallback = {
        "summary": (snippet or "").strip()[:240] or f"I'm reviewing {website}. Please share a short description.",
        "industries": [],
        "offerings": [],
        "ideal_customers": [],
        "proof_points": [],
    }
    result = call_llm_json(prompt, fallback)
    structured = {
        "summary": (result.get("summary") or fallback["summary"]).strip(),
        "industries": [s.strip() for s in (result.get("industries") or []) if str(s).strip()],
        "offerings": [s.strip() for s in (result.get("offerings") or []) if str(s).strip()],
        "ideal_customers": [s.strip() for s in (result.get("ideal_customers") or []) if str(s).strip()],
        "proof_points": [s.strip() for s in (result.get("proof_points") or []) if str(s).strip()],
    }
    return structured


def _summarize_company_site(website: str, history: Iterable[Any]) -> Dict[str, Any]:
    recent = [_message_text(m) for m in history if _message_role(m) == "user"][-3:]
    excerpt = _fetch_site_excerpt(website)
    snippet = excerpt or ""
    prompt = f"""
You draft concise company profiles for a lead-generation assistant.
Website: {website!r}
Recent user snippets: {recent!r}
Website content snippet: {snippet!r}
Return JSON {{"summary": "<=2 sentence description", "name": "Short name if mentioned"}}.
If you lack info, infer based on the domain (e.g., Nexius Labs) and invite the user to refine.
"""
    fallback_summary = (
        (snippet[:240].rsplit(" ", 1)[0] + "…")
        if snippet and len(snippet) > 240
        else (snippet or f"I'm reviewing {website}. Let me know what your company does so I can capture it accurately.")
    )
    fallback = {
        "summary": fallback_summary,
        "name": _name_from_domain(website),
    }
    result = call_llm_json(prompt, fallback)
    summary = (result.get("summary") or fallback_summary).strip()
    name = (result.get("name") or fallback["name"]).strip()
    structured = _structure_company_profile(website, snippet or summary, recent)
    source = "snippet" if snippet else "fallback"
    _log_step(
        "company_profile_summary",
        website=website,
        source=source,
        summary_preview=structured.get("summary", summary)[:120],
    )
    structured["name"] = name
    structured["website"] = website
    structured["summary_source"] = source
    return structured


def _maybe_capture_company_website(profile: ProfileState, history: Iterable[Any]) -> bool:
    company = profile.get("company_profile") or {}
    if company.get("website"):
        return False
    history_list = list(history)
    updated = False
    for message in reversed(history_list):
        if _message_role(message) != "user":
            continue
        text = _message_text(message)
        match = URL_RE.search(text)
        if match:
            website = match.group(1).rstrip(".,)")
            company["website"] = website
            summary_pack = _summarize_company_site(website, history_list)
            company.update(summary_pack)
            updated = True
            break
    profile["company_profile"] = company
    return updated


def _ensure_company_summary(profile: ProfileState, history: Iterable[Any]) -> bool:
    company = profile.get("company_profile") or {}
    if not company.get("website"):
        return False
    summary = (company.get("summary") or "").strip()
    if summary and company.get("summary_source") not in {"", "fallback"}:
        return False
    history_list = list(history)
    summary_pack = _summarize_company_site(company["website"], history_list)
    company.update(summary_pack)
    profile["company_profile"] = company
    return True


def _format_company_profile(company: Dict[str, Any]) -> str:
    summary = (company.get("summary") or "").strip()
    industries = company.get("industries") or []
    offerings = company.get("offerings") or []
    ideal_customers = company.get("ideal_customers") or []
    proof_points = company.get("proof_points") or []
    parts: List[str] = []
    if summary:
        parts.append(f"Summary: {summary}")
    if industries:
        parts.append("Industries: " + ", ".join(industries[:3]))
    if offerings:
        parts.append("Offerings: " + ", ".join(offerings[:4]))
    if ideal_customers:
        parts.append("Ideal customers: " + ", ".join(ideal_customers[:3]))
    if proof_points:
        parts.append("Proof points: " + ", ".join(proof_points[:3]))
    return " | ".join(parts)


def _heuristic_intent(text: str) -> str:
    lowered = text.lower()
    if any(keyword in lowered for keyword in ("http://", "https://")):
        return "confirm_company"
    if "confirm" in lowered and "company" in lowered:
        return "confirm_company"
    if "confirm" in lowered and "icp" in lowered:
        return "confirm_icp"
    if "micro icp" in lowered or lowered.strip().endswith("?"):
        return "question"
    if "run enrichment" in lowered or "start enrichment" in lowered:
        return "run_enrichment"
    return "chat"


def _simple_intent(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return "idle"
    heuristic = _heuristic_intent(raw)
    prompt = f"""
Classify the user's intention for the lead-generation orchestrator.
Return JSON {{"intent": one of [run_enrichment, confirm_icp, confirm_company, accept_micro_icp, question, chat]}}.
Text: {raw!r}
"""
    fallback = {"intent": heuristic}
    result = call_llm_json(prompt, fallback)
    return (result or fallback)["intent"]


def _maybe_explain_question(question: str, intent: str | None = None) -> str | None:
    clean = (question or "").strip()
    if not clean:
        return None
    normalized_intent = (intent or "").strip().lower()
    forced_question = normalized_intent == "question"
    prompt = f"""
You are a lead-generation assistant. Determine if the user is asking a question and, if so, answer it in ≤3 sentences.
Return JSON {{"is_question": true/false, "answer": "<friendly explanation or empty string>"}}.
User text: {clean!r}
"""
    fallback = {"is_question": forced_question or bool("?" in clean), "answer": ""}
    result = call_llm_json(prompt, fallback)
    answer_data = result or fallback
    is_question = bool(answer_data.get("is_question")) or forced_question
    if not is_question:
        return None
    answer = answer_data.get("answer") or ""
    if not answer:
        text = clean.lower()
        if "micro icp" in text:
            answer = "A micro ICP is the most focused slice of your ideal customer profile—usually a specific industry + size + trigger that converts best."
        elif "company profile" in text:
            answer = "A company profile is the quick snapshot of your business—website, what you do, customers you serve, and any proof points—so I can ground the rest of the orchestration."
        else:
            answer = "I'm here to help with ICP questions. Once you're ready, please share your company profile."
    _log_step("question_explanation", intent=normalized_intent or None, answer_preview=answer[:80])
    return answer


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
Intent options: run_enrichment, confirm_company, confirm_icp, accept_micro_icp, question, chat, idle.
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
        _log_step(
            "ingest",
            role=role,
            intent=ctx["intent"],
            tags=ctx.get("tags", []),
            normalized_text=ctx["last_user_command"],
        )
    msg = "Processed user input" if incoming else "Waiting for user input"
    _set_status(state, "ingest", msg)
    return state


async def profile_builder(state: OrchestrationState) -> OrchestrationState:
    """Update profile flags using the latest conversation snippet."""
    profile = _ensure_profile_state(state)
    history = state.get("messages", [])[-5:]
    prompt = f"""
You maintain company + ICP profile confirmations and draft summaries from chat snippets.
Return JSON with keys:
  - company_profile_confirmed (bool)
  - icp_profile_confirmed (bool)
  - micro_icp_selected (bool)
  - company_profile (object with website/name/summary fields when mentioned)
  - icp_profile (object summarizing the ICP when provided)
History: {history!r}
Current profile: {profile!r}
"""
    fallback = {
        "company_profile_confirmed": bool(profile.get("company_profile_confirmed")),
        "icp_profile_confirmed": bool(profile.get("icp_profile_confirmed")),
        "micro_icp_selected": bool(profile.get("micro_icp_selected")),
        "company_profile": profile.get("company_profile") or {},
        "icp_profile": profile.get("icp_profile") or {},
    }
    result = call_llm_json(prompt, fallback)
    for key in ("company_profile_confirmed", "icp_profile_confirmed", "micro_icp_selected"):
        profile[key] = bool(result.get(key, fallback[key]))
    profile["company_profile"] = result.get("company_profile") or fallback["company_profile"]
    profile["icp_profile"] = result.get("icp_profile") or fallback["icp_profile"]
    captured_new_site = _maybe_capture_company_website(profile, history)
    if captured_new_site:
        profile["company_profile_confirmed"] = False
    elif _ensure_company_summary(profile, history):
        profile["company_profile_confirmed"] = False
    profile["last_updated_at"] = datetime.now(timezone.utc).isoformat()
    _log_step(
        "profile_builder",
        company=profile.get("company_profile_confirmed"),
        icp=profile.get("icp_profile_confirmed"),
        micro=profile.get("micro_icp_selected"),
    )
    _set_status(state, "profile_builder", "Profile snapshot updated")
    return state


async def journey_guard(state: OrchestrationState) -> OrchestrationState:
    """Ensure prerequisites exist before backend work."""
    profile = _ensure_profile_state(state)
    company_ready = bool(profile.get("company_profile_confirmed"))
    icp_ready = bool(profile.get("icp_profile_confirmed"))
    journey_ready = bool(company_ready and icp_ready)
    ctx = state.get("entry_context") or {}
    question = ctx.get("last_user_command") or _latest_user_text(state)
    last_intent = ctx.get("intent") or (_heuristic_intent(question) if question else None)
    state["journey_ready"] = journey_ready
    company_profile = profile.get("company_profile") or {}
    needs_website = (not company_ready) and _needs_company_website(profile)
    if not needs_website:
        if _ensure_company_summary(profile, state.get("messages", [])[-8:]):
            company_profile = profile.get("company_profile") or {}
    if company_ready and icp_ready:
        profile["outstanding_prompts"] = []
        msg = "Prerequisites satisfied. Proceeding to normalization."
    else:
        explanation = _maybe_explain_question(question, last_intent)
        prompt_parts: List[str] = []
        if explanation:
            prompt_parts.append(explanation)

        if needs_website:
            prompt_parts.append("To get started, please share your business website so I can draft your company profile.")
        elif not company_ready:
            summary = (company_profile.get("summary") or "").strip()
            company_name = company_profile.get("name") or "your company"
            website = company_profile.get("website")
            summary_source = company_profile.get("summary_source")
            if summary:
                prefix = "Here's what I gathered" if summary_source != "fallback" else "I pulled an initial overview"
                formatted = _format_company_profile(company_profile)
                prompt_parts.append(f"{prefix} for {company_name}{' (' + website + ')' if website else ''}: {formatted}")
            elif website:
                prompt_parts.append(f"I've saved {website} as your website. I'll summarize it once you confirm.")
            prompt_parts.append("Does that look right? Please confirm so we can continue to your ICP.")
        elif not icp_ready:
            prompt_parts.append("Great! Next, could you describe your ideal customer profile (industries, sizes, triggers) so I can line up candidates?")
        else:
            prompt_parts.append("Almost ready—just a quick confirmation on your inputs and I'll proceed.")

        prompt = " ".join(p for p in prompt_parts if p)
        profile["outstanding_prompts"] = [prompt]
        msg = prompt
        _append_message(state, "assistant", prompt)
    _log_step(
        "journey_guard",
        company_ready=company_ready,
        icp_ready=icp_ready,
        intent=last_intent,
        explanation_provided=bool(explanation),
        needs_website=needs_website,
    )
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
    _log_step("normalize", processed=result["processed_rows"], errors=result.get("errors"))
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
    _log_step(
        "refresh_icp",
        candidates=len(discovery.get("candidate_ids") or []),
        diagnostics=discovery.get("diagnostics"),
    )
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
    action = result.get("action", fallback["action"])
    discovery["strategy"] = action
    state["discovery"] = discovery
    _log_step("decide_strategy", action=action, reason=result.get("reason", fallback["reason"]))
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
    _log_step("ssic_fallback", produced=len(discovery.get("candidate_ids") or []), last_attempt=discovery.get("last_ssic_attempt"))
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
    _log_step("plan_top10", planned=len(items))
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
    _log_step("enrich_batch", attempts=len(results), offline=OFFLINE, had_candidates=bool(candidate_ids))
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
    _log_step("score_leads", scored=len(scores))
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
    _log_step(
        "export",
        next40=exports.get("next40_enqueued"),
        jobs=exports.get("job_ids"),
        msg_extra=msg_extra,
    )
    _set_status(state, "export", f"Export stage complete{msg_extra}")
    return state


async def progress_report(state: OrchestrationState) -> OrchestrationState:
    summary = {
        "phase": state.get("status", {}).get("phase"),
        "candidates": len(state.get("discovery", {}).get("candidate_ids") or []),
        "confirmed_company": bool(state.get("profile_state", {}).get("company_profile_confirmed")),
        "confirmed_icp": bool(state.get("profile_state", {}).get("icp_profile_confirmed")),
    }
    outstanding = (state.get("profile_state") or {}).get("outstanding_prompts") or []
    if outstanding:
        message = outstanding[0]
    else:
        prompt = f"""
Summarize the orchestration status for the user in ≤2 sentences using plain language.
Focus on next steps instead of internal phase names.
State: {summary!r}
Return JSON {{"message": "..."}}
"""
        fallback_message = "I'm reviewing your inputs. Please confirm your company profile and ideal customer profile so I can continue."
        result = call_llm_json(prompt, {"message": fallback_message})
        message = result.get("message", fallback_message)
    _log_step("progress_report", outstanding=bool(outstanding), message=message)
    _set_status(state, "progress_report", message)
    _append_message(state, "assistant", message)
    return state


async def finalize(state: OrchestrationState) -> OrchestrationState:
    msg = state.get("status", {}).get("message") or "Run complete"
    _append_message(state, "assistant", msg)
    _set_status(state, "summary", msg)
    _log_step("summary", message=msg)
    return state
URL_RE = re.compile(r"(https?://[^\s]+)", re.IGNORECASE)


def _message_role(message: Any) -> str | None:
    if isinstance(message, dict):
        role = message.get("role")
    else:
        role = getattr(message, "role", None) or getattr(message, "type", None)
    if role in {"human", "user"}:
        return "user"
    if role in {"ai", "assistant"}:
        return "assistant"
    return role


def _message_text(message: Any) -> str:
    if isinstance(message, dict):
        content = message.get("content", "")
    else:
        content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for chunk in content:
            if isinstance(chunk, dict):
                parts.append(str(chunk.get("text") or chunk.get("content") or ""))
            else:
                parts.append(str(chunk))
        return " ".join(p for p in parts if p)
    return str(content)


def _latest_user_text(state: OrchestrationState) -> str:
    for message in reversed(state.get("messages") or []):
        if _message_role(message) == "user":
            return _message_text(message)
    return ""


def _message_role(message: Any) -> str | None:
    if isinstance(message, dict):
        role = message.get("role")
    else:
        role = getattr(message, "role", None) or getattr(message, "type", None)
    if role in {"human", "user"}:
        return "user"
    if role in {"ai", "assistant"}:
        return "assistant"
    return role


def _message_text(message: Any) -> str:
    if isinstance(message, dict):
        content = message.get("content", "")
    else:
        content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for chunk in content:
            if isinstance(chunk, dict):
                parts.append(str(chunk.get("text") or chunk.get("content") or ""))
            else:
                parts.append(str(chunk))
        return " ".join(p for p in parts if p)
    return str(content)
