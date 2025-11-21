"""
Node implementations for the unified orchestrator.

These implementations leverage LangChain/OpenAI when available and gracefully
degrade to rule-based heuristics when credentials are missing. Downstream
LangGraph agents (normalization, ICP refresh, etc.) are reused wherever
possible so behavior matches the legacy stack.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Sequence
from urllib.parse import urlparse
import requests

from src.icp import icp_by_ssic_agent, icp_refresh_agent, normalize_agent
from src.lead_scoring import lead_scoring_agent
from src.jobs import enqueue_web_discovery_bg_enrich
from src.jina_reader import read_url as jina_read
from app.pre_sdr_graph import _persist_company_profile_sync, _persist_icp_profile_sync  # type: ignore

try:
    from src.settings import ENABLE_AGENT_DISCOVERY as _SETTINGS_AGENT_DISCOVERY  # type: ignore
except Exception:  # pragma: no cover - optional settings module
    _SETTINGS_AGENT_DISCOVERY = False  # type: ignore

try:
    from src.agents_icp import compliance_guard as _agent_compliance_guard  # type: ignore
    from src.agents_icp import discovery_planner as _agent_discovery_planner  # type: ignore
except Exception:  # pragma: no cover - agent bundle optional
    _agent_discovery_planner = None  # type: ignore
    _agent_compliance_guard = None  # type: ignore


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

RUN_ENRICH_PHRASES = {
    "enrich",
    "start enrichment",
    "run enrichment",
    "enrich 10",
    "top 10",
    "enrich ten",
    "please enrich",
}
DEFAULT_COMPANY_PROFILE = {
    "name": "Nexius Labs",
    "website": "https://nexiuslabs.com",
    "summary": (
        "NEXIUS Labs provides AI-powered automation to help lean teams win more customers "
        "while reducing operational overhead, centered on the Nexius Agent AI business partner "
        "plus courses and pre-built workflows for sales, invoicing, and ops."
    ),
    "industries": ["AI & Automation", "SaaS", "Business Operations"],
    "offerings": [
        "Nexius Agent (AI business partner for automating key operations)",
        "NEXIUS Academy (AI courses and resources)",
        "Pre-built automation workflows (lead engine, invoicing, ops dashboard)",
        "Open-source AI stacks and integrations",
    ],
    "ideal_customers": [
        "Founders and solo founders",
        "Lean teams and small businesses",
        "SMBs scaling without expensive software",
    ],
    "proof_points": [
        "Average 45% efficiency boost (claimed)",
        "Trusted by founders and lean teams",
        "Starter workflows live in days",
    ],
    "summary_source": "seeded",
}
_EVENT_TOKEN_RE = re.compile(r"^[a-z0-9_]+:[a-z0-9_]+$", re.IGNORECASE)


def _ensure_mcp_reader_enabled() -> None:
    try:
        from src import settings as _app_settings  # type: ignore
    except Exception:  # pragma: no cover
        return
    try:
        if not getattr(_app_settings, "ENABLE_MCP_READER", False):
            setattr(_app_settings, "ENABLE_MCP_READER", True)
            logger.info("ENABLE_MCP_READER not set; defaulting to True for orchestrator runs.")
    except Exception:
        pass


_ensure_mcp_reader_enabled()
OFFLINE = (os.getenv("ORCHESTRATOR_OFFLINE") or "").lower() in {"1", "true", "yes"}
AGENT_DISCOVERY_ENABLED = bool(_SETTINGS_AGENT_DISCOVERY)


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
    if content is None:
        return
    text = str(content)
    if _should_suppress_message(role, text):
        return
    _ensure_messages(state)
    state["messages"].append({"role": role, "content": text})


def _looks_like_json_blob(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped[0] not in "{[":
        return False
    if stripped[-1] not in "]}":
        return False
    try:
        payload = json.loads(stripped)
    except Exception:
        return False
    return isinstance(payload, (dict, list))


def _should_suppress_message(role: str, content: str) -> bool:
    stripped = content.strip()
    if not stripped:
        return True
    if role != "assistant":
        return False
    if _EVENT_TOKEN_RE.match(stripped):
        logger.debug("suppressing telemetry event: %s", stripped)
        return True
    if _looks_like_json_blob(stripped):
        logger.debug("suppressing JSON payload: %s", stripped[:120])
        return True
    return False


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
    profile.setdefault("icp_profile_generated", bool(profile.get("icp_profile")))
    profile.setdefault("icp_discovery_confirmed", False)
    profile.setdefault("awaiting_discovery_confirmation", False)
    profile.setdefault("awaiting_enrichment_confirmation", False)
    profile.setdefault("enrichment_confirmed", False)
    profile.setdefault("discovery_retry_requested", False)
    profile.setdefault("customer_websites", [])
    seeded = False
    if not profile["company_profile"]:
        profile["company_profile"] = copy.deepcopy(DEFAULT_COMPANY_PROFILE)
        profile["company_profile_confirmed"] = False
        seeded = True
    if "seeded_company_profile" not in profile:
        profile["seeded_company_profile"] = seeded
    elif seeded:
        profile["seeded_company_profile"] = True
    state["profile_state"] = profile
    return profile


def _needs_company_website(profile: ProfileState) -> bool:
    company = profile.get("company_profile") or {}
    seeded = bool(profile.get("seeded_company_profile"))
    has_website = bool((company.get("website") or "").strip())
    return seeded or not has_website


def _domain_from_value(url: str) -> str:
    try:
        parsed = urlparse(url)
        host = (parsed.netloc or parsed.path or url or "").strip().lower()
    except Exception:
        host = (url or "").strip().lower()
    if not host:
        return ""
    if host.startswith("http://"):
        host = host[7:]
    elif host.startswith("https://"):
        host = host[8:]
    for sep in ("/", "?", "#"):
        if sep in host:
            host = host.split(sep, 1)[0]
    if host.startswith("www."):
        host = host[4:]
    return host


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
    if not raw:
        raw = _http_fetch_plain(url)
    cleaned = re.sub(r"\s+", " ", raw or "").strip()
    if not cleaned:
        return None
    return cleaned[:max_chars]


def _http_fetch_plain(url: str, timeout: int = 8) -> str:
    try:
        headers = {"User-Agent": "LeadGen-Orchestrator/0.1"}
        resp = requests.get(url, timeout=timeout, headers=headers)
        if resp.status_code >= 400:
            alt = url.replace("https://", "http://") if url.startswith("https://") else url.replace("http://", "https://", 1)
            if alt != url:
                resp = requests.get(alt, timeout=timeout, headers=headers)
        return (resp.text or "")[:8000]
    except Exception as exc:
        logger.warning("direct fetch failed for %s: %s", url, exc)
        return ""


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


def _maybe_capture_company_website(state: OrchestrationState, profile: ProfileState, history: Iterable[Any]) -> bool:
    company = profile.get("company_profile") or {}
    if company.get("website") and not profile.get("seeded_company_profile"):
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
            _persist_company_profile_state(state, company, confirmed=False)
            updated = True
            break
    profile["company_profile"] = company
    if updated:
        profile["seeded_company_profile"] = False
    return updated


def _ensure_company_summary(state: OrchestrationState, profile: ProfileState, history: Iterable[Any]) -> bool:
    company = profile.get("company_profile") or {}
    if not company.get("website"):
        return False
    summary = (company.get("summary") or "").strip()
    if summary and company.get("summary_source") not in {"", "fallback"}:
        return False
    history_list = list(history)
    summary_pack = _summarize_company_site(company["website"], history_list)
    company.update(summary_pack)
    _persist_company_profile_state(state, company, confirmed=False)
    profile["company_profile"] = company
    profile["seeded_company_profile"] = False
    return True


def _format_company_profile(company: Dict[str, Any]) -> str:
    summary = (company.get("summary") or "").strip()
    industries = company.get("industries") or []
    offerings = company.get("offerings") or []
    ideal_customers = company.get("ideal_customers") or []
    proof_points = company.get("proof_points") or []
    blocks: List[str] = []
    if summary:
        blocks.append(f"**Summary**\n{summary}")

    def _format_section(label: str, values: Sequence[str], limit: int) -> None:
        cleaned = [str(v).strip() for v in values if str(v).strip()]
        if not cleaned:
            return
        snippet_lines = "\n".join(f"- {item}" for item in cleaned[:limit])
        blocks.append(f"**{label}**\n{snippet_lines}")

    _format_section("Industries", industries, 3)
    _format_section("Offerings", offerings, 4)
    _format_section("Ideal Customers", ideal_customers, 3)
    _format_section("Proof Points", proof_points, 3)
    return "\n\n".join(blocks)


def _format_icp_profile(icp: Dict[str, Any]) -> str:
    summary = (icp.get("summary") or "").strip()
    sections: List[str] = []
    if summary:
        sections.append(f"**Summary**\n{summary}")

    def _append(label: str, values: List[str], limit: int = 4) -> None:
        cleaned = [str(v).strip() for v in (values or []) if str(v).strip()]
        if not cleaned:
            return
        bullets = "\n".join(f"- {item}" for item in cleaned[:limit])
        sections.append(f"**{label}**\n{bullets}")

    _append("Industries", icp.get("industries") or [], limit=3)
    _append("Company Sizes", icp.get("company_sizes") or [], limit=3)
    _append("Regions", icp.get("regions") or [], limit=3)
    _append("Key Pains", icp.get("pains") or [], limit=3)
    _append("Buying Triggers", icp.get("buying_triggers") or [], limit=3)
    _append("Personas", icp.get("persona_titles") or [], limit=3)
    _append("Proof Points", icp.get("proof_points") or [], limit=3)
    return "\n\n".join(sections)


def _persist_company_profile_state(state: OrchestrationState, profile: Dict[str, Any], confirmed: bool = False) -> None:
    ctx = state.get("entry_context") or {}
    tenant_id = ctx.get("tenant_id") or state.get("tenant_id")
    if not tenant_id:
        return
    state["tenant_id"] = tenant_id
    persist_state = {
        "tenant_id": tenant_id,
        "company_profile_confirmed": confirmed,
        "site_profile_bootstrap_url": profile.get("website"),
    }
    try:
        _persist_company_profile_sync(persist_state, dict(profile or {}), confirmed=confirmed, source_url=profile.get("website"))
    except Exception:
        logger.warning("Failed to persist company profile", exc_info=True)


def _persist_icp_profile_state(state: OrchestrationState, icp_profile: Dict[str, Any], seed_urls: List[str]) -> None:
    ctx = state.get("entry_context") or {}
    tenant_id = ctx.get("tenant_id") or state.get("tenant_id")
    if not tenant_id:
        return
    persist_state = {"tenant_id": tenant_id}
    try:
        _persist_icp_profile_sync(persist_state, dict(icp_profile or {}), confirmed=True, seed_urls=seed_urls, user_confirmed=False)
    except Exception:
        logger.warning("Failed to persist ICP profile", exc_info=True)


def _format_candidate_table(records: Sequence[Dict[str, Any]], limit: int = 50) -> str:
    if not records:
        return ""
    header = ["#", "Name", "Domain", "Bucket", "Score", "ICP Match"]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for idx, item in enumerate(records[:limit], start=1):
        domain = _domain_from_value(str(item.get("domain") or ""))
        display_name = item.get("name") or _name_from_domain(domain)
        bucket = str(item.get("bucket") or "").upper() or "-"
        score = item.get("score")
        try:
            score_txt = f"{int(score)}"
        except Exception:
            score_txt = "-" if score in (None, "", []) else str(score)
        reason = (
            item.get("why")
            or item.get("reason")
            or item.get("match_reason")
            or item.get("snippet")
            or "Matches your ICP focus."
        )
        reason_clean = " ".join(str(reason or "").split())
        if len(reason_clean) > 120:
            reason_clean = reason_clean[:117].rstrip() + "..."
        lines.append(f"| {idx} | {display_name} | {domain or '-'} | {bucket} | {score_txt} | {reason_clean or '-'} |")
    return "\n".join(lines)


def _plan_web_candidates(icp_profile: Dict[str, Any]) -> Dict[str, Any]:
    if not AGENT_DISCOVERY_ENABLED:
        return {}
    details: List[Dict[str, Any]] = []
    snippets: Dict[str, str] = {}
    domains: List[str] = []
    if plan_top10_with_reasons is not None:
        try:
            planned_items = plan_top10_with_reasons(icp_profile, tenant_id=None) or []
            for item in planned_items:
                domain = _domain_from_value(str(item.get("domain") or ""))
                if not domain:
                    continue
                details.append(
                    {
                        "domain": domain,
                        "name": _name_from_domain(domain),
                        "bucket": (item.get("bucket") or "").upper() or None,
                        "score": item.get("score"),
                        "why": item.get("why") or item.get("reason"),
                        "snippet": item.get("snippet"),
                        "lead_profile": item.get("lead_profile"),
                    }
                )
                snippet_val = str(item.get("snippet") or "").strip()
                if snippet_val:
                    snippets[domain] = snippet_val
            if details:
                domains = [d["domain"] for d in details]
        except Exception:  # pragma: no cover - optional agent
            details = []
            snippets = {}
            domains = []
    if not domains and not (_agent_discovery_planner and AGENT_DISCOVERY_ENABLED):
        return {}
    try:
        if not domains and _agent_discovery_planner is not None:
            planned = _agent_discovery_planner({"icp_profile": icp_profile or {}}) or {}
            guarded = planned
            if _agent_compliance_guard is not None:
                try:
                    guarded = _agent_compliance_guard(dict(planned)) or guarded
                except Exception:
                    pass
            raw_domains = [
                _domain_from_value(str(d))
                for d in (guarded.get("discovery_candidates") or planned.get("discovery_candidates") or [])
                if isinstance(d, str) and d.strip()
            ]
            domains = [d for d in raw_domains if d][:50]
            guard_snips = guarded.get("jina_snippets") or planned.get("jina_snippets") or {}
            if guard_snips:
                snippets.update({d: guard_snips.get(d) for d in domains if guard_snips.get(d)})
    except Exception as exc:  # pragma: no cover - agent optional
        logger.warning("web discovery planner failed: %s", exc)
        return {}
    result: Dict[str, Any] = {"domains": domains[:50]}
    if snippets:
        result["snippets"] = {d: snippets.get(d) for d in domains if snippets.get(d)}
    if details:
        domain_rank = {d: idx for idx, d in enumerate(domains)}
        ordered = sorted(details, key=lambda item: domain_rank.get(item.get("domain") or "", 0))
        result["details"] = ordered[:50]
    return result


def _generate_icp_from_customers(state: OrchestrationState, profile: ProfileState) -> bool:
    sites = profile.get("customer_websites") or []
    if len(sites) < 5:
        return False
    snippets = []
    for url in sites[:5]:
        snippet = _fetch_site_excerpt(url)
        if snippet:
            snippets.append({"url": url, "snippet": snippet})
    if not snippets:
        return False
    prompt = f"""
You are an ICP strategist. Analyze these existing customer websites and summarize the ideal customer profile for a lead-generation campaign.
Customers:
{snippets!r}
Return JSON with keys: summary, industries, company_sizes, regions, pains, buying_triggers, persona_titles, proof_points.
Each list should contain up to 4 concise items.
"""
    fallback = {
        "summary": "Ideal customers resemble the provided reference customers—please refine if needed.",
        "industries": [],
        "company_sizes": [],
        "regions": [],
        "pains": [],
        "buying_triggers": [],
        "persona_titles": [],
        "proof_points": [],
    }
    result = call_llm_json(prompt, fallback)
    icp = {
        "summary": (result.get("summary") or fallback["summary"]).strip(),
        "industries": [s.strip() for s in (result.get("industries") or []) if str(s).strip()],
        "company_sizes": [s.strip() for s in (result.get("company_sizes") or []) if str(s).strip()],
        "regions": [s.strip() for s in (result.get("regions") or []) if str(s).strip()],
        "pains": [s.strip() for s in (result.get("pains") or []) if str(s).strip()],
        "buying_triggers": [s.strip() for s in (result.get("buying_triggers") or []) if str(s).strip()],
        "persona_titles": [s.strip() for s in (result.get("persona_titles") or []) if str(s).strip()],
        "proof_points": [s.strip() for s in (result.get("proof_points") or []) if str(s).strip()],
        "sources": sites[:5],
    }
    profile["icp_profile"] = icp
    profile["icp_profile_generated"] = True
    profile["icp_profile_confirmed"] = False
    _persist_icp_profile_state(state, icp, sites[:5])
    _log_step("icp_profile_generated", sites=sites[:5], summary_preview=icp["summary"][:120])
    return True


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
    if "run enrichment" in lowered or "start enrichment" in lowered or "enrich" in lowered:
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


def _wants_enrichment(intent: str | None, normalized_command: str) -> bool:
    normalized_intent = (intent or "").strip().lower()
    normalized_text = (normalized_command or "").strip().lower()
    return normalized_intent == "run_enrichment" or any(phrase in normalized_text for phrase in RUN_ENRICH_PHRASES)


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
        if ctx.get("tenant_id"):
            state["tenant_id"] = ctx["tenant_id"]
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
  - icp_discovery_confirmed (bool)
  - company_profile (object with website/name/summary fields when mentioned)
  - icp_profile (object summarizing the ICP when provided)
History: {history!r}
Current profile: {profile!r}
"""
    fallback = {
        "company_profile_confirmed": bool(profile.get("company_profile_confirmed")),
        "icp_profile_confirmed": bool(profile.get("icp_profile_confirmed")),
        "icp_discovery_confirmed": bool(profile.get("icp_discovery_confirmed")),
        "micro_icp_selected": bool(profile.get("micro_icp_selected")),
        "company_profile": profile.get("company_profile") or {},
        "icp_profile": profile.get("icp_profile") or {},
    }
    result = call_llm_json(prompt, fallback)
    for key in ("company_profile_confirmed", "icp_profile_confirmed", "micro_icp_selected", "icp_discovery_confirmed"):
        profile[key] = bool(result.get(key, fallback[key]))
    profile["company_profile"] = result.get("company_profile") or fallback["company_profile"]
    profile["icp_profile"] = result.get("icp_profile") or fallback["icp_profile"]
    captured_new_site = _maybe_capture_company_website(state, profile, history)
    if captured_new_site:
        profile["company_profile_confirmed"] = False
    elif _ensure_company_summary(state, profile, history):
        profile["company_profile_confirmed"] = False
    messages = state.get("messages", [])
    company_url = (profile.get("company_profile") or {}).get("website")
    profile["customer_websites"] = _collect_customer_sites(messages, company_url)
    customer_sites = profile.get("customer_websites") or []
    if len(customer_sites) >= 5 and not profile.get("icp_profile_generated"):
        _generate_icp_from_customers(state, profile)

    ctx = state.get("entry_context") or {}
    last_intent = (ctx.get("intent") or "").strip().lower()
    last_command = (ctx.get("last_user_command") or "").strip().lower()
    normalized_command = last_command.translate(str.maketrans("", "", ".!?")).strip()

    def _includes_phrase(text: str, phrases: Iterable[str]) -> bool:
        if not text:
            return False
        return any(phrase in text for phrase in phrases)

    if not profile.get("company_profile_confirmed"):
        company_confirm_phrases = {
            "confirm company",
            "company looks good",
            "company ok",
            "company okay",
            "that's correct",
            "that's right",
            "looks good",
            "sounds good",
            "ok confirmed",
            "okay confirmed",
            "confirm",
        }
        if last_intent == "confirm_company" or _includes_phrase(normalized_command, company_confirm_phrases):
            profile["company_profile_confirmed"] = True

    if profile.get("company_profile_confirmed") and not profile.get("icp_profile_confirmed"):
        icp_confirm_phrases = {
            "confirm icp",
            "icp looks good",
            "icp ok",
            "icp okay",
            "looks great",
            "looks good",
            "sounds good",
            "ready for discovery",
            "start discovery",
            "begin discovery",
            "proceed",
            "go ahead",
            "do it",
        }
        if last_intent in {"confirm_icp", "run_enrichment"} or _includes_phrase(normalized_command, icp_confirm_phrases):
            profile["icp_profile_confirmed"] = True

    enrichment_requested = _wants_enrichment(last_intent, normalized_command)
    if profile.get("icp_discovery_confirmed") and enrichment_requested:
        profile["enrichment_confirmed"] = True
        profile["awaiting_enrichment_confirmation"] = False

    awaiting_discovery = bool(profile.get("awaiting_discovery_confirmation"))
    if awaiting_discovery and not profile.get("icp_discovery_confirmed"):
        positive_intents = {"run_enrichment", "accept_micro_icp", "confirm_icp"}
        positive_phrases = {
            "yes",
            "yep",
            "yeah",
            "yup",
            "sure",
            "ok",
            "okay",
            "k",
            "kk",
            "alright",
            "do it",
            "go ahead",
            "go for it",
            "looks good",
            "start discovery",
            "start icp",
            "run discovery",
            "run icp",
            "begin discovery",
            "begin icp",
            "proceed",
        }
        normalized = normalized_command
        confirm_discovery = False
        if last_intent in positive_intents or any(phrase in normalized for phrase in positive_phrases):
            confirm_discovery = True
        elif "discovery" in normalized and any(
            trigger in normalized for trigger in ("start", "run", "begin", "kick off", "ready", "lets", "let's", "proceed")
        ):
            confirm_discovery = True
        elif "icp" in normalized and any(
            trigger in normalized for trigger in ("start", "run", "begin", "ready", "proceed", "go ahead")
        ):
                confirm_discovery = True
        if confirm_discovery:
            profile["icp_discovery_confirmed"] = True
            profile["awaiting_discovery_confirmation"] = False

    if profile.get("icp_discovery_confirmed"):
        profile["awaiting_discovery_confirmation"] = False

    retry_triggers = ("retry discovery", "retry icp", "search again")
    if any(trigger in last_command for trigger in retry_triggers):
        profile["discovery_retry_requested"] = True
        profile["awaiting_enrichment_confirmation"] = False
        profile["enrichment_confirmed"] = False
    if profile.get("awaiting_enrichment_confirmation") and not profile.get("enrichment_confirmed"):
        confirm_phrases = (
            "enrich",
            "start enrichment",
            "proceed",
            "go ahead",
            "looks good",
            "confirm",
            "yes",
            "top 10",
            "ten",
        )
        normalized = normalized_command
        if enrichment_requested or any(phrase in normalized for phrase in confirm_phrases):
            profile["enrichment_confirmed"] = True
            profile["awaiting_enrichment_confirmation"] = False

    if profile.get("seeded_company_profile"):
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
    discovery_ready = bool(profile.get("icp_discovery_confirmed"))
    enrichment_ready = bool(profile.get("enrichment_confirmed"))
    ctx = state.get("entry_context") or {}
    question = ctx.get("last_user_command") or _latest_user_text(state)
    last_intent = (ctx.get("intent") or (_heuristic_intent(question) if question else "")).strip().lower()
    last_command = (ctx.get("last_user_command") or "").strip().lower()
    normalized_command = last_command.translate(str.maketrans("", "", ".!?")).strip()
    if discovery_ready and not enrichment_ready and _wants_enrichment(last_intent, normalized_command):
        profile["enrichment_confirmed"] = True
        profile["awaiting_enrichment_confirmation"] = False
        enrichment_ready = True
    journey_ready = bool(company_ready and icp_ready and discovery_ready and enrichment_ready)
    state["journey_ready"] = journey_ready
    company_profile = profile.get("company_profile") or {}
    icp_profile = profile.get("icp_profile") or {}
    needs_website = (not company_ready) and _needs_company_website(profile)
    customer_sites = profile.get("customer_websites") or []
    discovery_state = state.get("discovery") or {}
    if company_ready and company_profile:
        _persist_company_profile_state(state, company_profile, confirmed=True)
    if not needs_website:
        if _ensure_company_summary(state, profile, state.get("messages", [])[-8:]):
            company_profile = profile.get("company_profile") or {}
    explanation = None
    if journey_ready:
        profile["outstanding_prompts"] = []
        profile["awaiting_discovery_confirmation"] = False
        msg = "Prerequisites satisfied. Proceeding to normalization."
    else:
        explanation = _maybe_explain_question(question, last_intent)
        prompt_parts: List[str] = []
        if explanation:
            prompt_parts.append(explanation)

        if needs_website:
            profile["awaiting_discovery_confirmation"] = False
            prompt_parts.append("To get started, please share your business website so I can draft your company profile.")
        elif company_ready and len(customer_sites) < 5:
            profile["awaiting_discovery_confirmation"] = False
            remaining = 5 - len(customer_sites)
            prompt_parts.append(
                f"Great! Now share {remaining} more of your best customer website URLs (total of 5) so I can learn from them."
            )
        elif not company_ready:
            profile["awaiting_discovery_confirmation"] = False
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
            profile["awaiting_discovery_confirmation"] = False
            icp_overview = _format_icp_profile(icp_profile)
            if icp_overview:
                prompt_parts.append(f"Here's the ICP I drafted from the customer sites you shared: {icp_overview}")
                prompt_parts.append("Does that look right? Confirm or suggest edits before I start discovery.")
            else:
                prompt_parts.append("I'm synthesizing your ICP from the customer sites you shared—one moment while I finish up.")
        elif not discovery_ready:
            profile["awaiting_discovery_confirmation"] = True
            prompt_parts.append("You're all set on profiles. Ready for me to start ICP discovery and pull candidate lists?")
            prompt_parts.append("Reply with 'start discovery' (or similar) when you're ready, or share tweaks if you'd like adjustments first.")
        elif discovery_ready and not enrichment_ready:
            profile["awaiting_discovery_confirmation"] = False
            if profile.get("discovery_retry_requested"):
                discovery_state.pop("web_candidates", None)
                discovery_state.pop("web_candidate_details", None)
                discovery_state.pop("web_candidate_table", None)
            candidates = discovery_state.get("web_candidates") or []
            candidate_details = discovery_state.get("web_candidate_details") or []
            if not candidates:
                plan = _plan_web_candidates(icp_profile) or {}
                candidates = plan.get("domains") or []
                if candidates:
                    discovery_state["web_candidates"] = candidates
                snippets = plan.get("snippets") or {}
                if snippets:
                    discovery_state["web_snippets"] = snippets
                details = plan.get("details") or []
                if details:
                    discovery_state["web_candidate_details"] = details
                    candidate_details = details
                profile["discovery_retry_requested"] = False
            if candidates:
                if not candidate_details:
                    candidate_details = [
                        {
                            "domain": d,
                            "name": _name_from_domain(d),
                            "bucket": None,
                            "score": None,
                            "why": "Matches your ICP inputs.",
                        }
                        for d in candidates
                    ]
                    discovery_state["web_candidate_details"] = candidate_details
                table = _format_candidate_table(candidate_details, limit=50)
                discovery_state["web_candidate_table"] = table
                found_count = min(50, len(candidate_details))
                prompt_parts.append(
                    f"I found {len(candidates)} ICP candidate websites. Here's the latest {found_count} with match details:\n{table}"
                )
                prompt_parts.append(
                    "\n\nReady for me to enrich the best 10 now? Reply 'enrich 10' (or similar) to proceed, "
                    "or say 'retry discovery' and I'll search again."
                )
                profile["awaiting_enrichment_confirmation"] = True
            else:
                prompt_parts.append("I couldn't find solid ICP candidates. Say 'retry discovery' to search again or adjust your ICP details.")
                profile["awaiting_enrichment_confirmation"] = False
        else:
            prompt_parts.append("Almost ready—just a quick confirmation on your inputs and I'll proceed.")

        prompt = " ".join(p for p in prompt_parts if p)
        profile["outstanding_prompts"] = [prompt]
        msg = prompt
        _append_message(state, "assistant", prompt)
    state["discovery"] = discovery_state
    _log_step(
        "journey_guard",
        company_ready=company_ready,
        icp_ready=icp_ready,
        discovery_ready=discovery_ready,
        enrichment_ready=enrichment_ready,
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
    profile = _ensure_profile_state(state)
    if not profile.get("enrichment_confirmed"):
        _log_step("refresh_icp", skipped=True, reason="awaiting_enrichment_confirmation")
        _set_status(state, "refresh_icp", "Waiting for enrichment confirmation")
        return state
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
        web_candidates=len(discovery.get("web_candidates") or []),
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
        "web_candidates": len(state.get("discovery", {}).get("web_candidates") or []),
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
        all_web_candidates = state.get("discovery", {}).get("web_candidates") or []
        if all_web_candidates:
            preview = ", ".join(all_web_candidates[:5])
            fallback_message = (
                f"Pulled {len(all_web_candidates)} fresh ICP candidates via live web search. "
                f"Sample: {preview}. Moving on to enrichment."
            )
        else:
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
def _normalize_url(url: str) -> str:
    try:
        parsed = urlparse(url if url.startswith("http") else ("https://" + url))
        scheme = parsed.scheme or "https"
        netloc = (parsed.netloc or parsed.path).lower()
        path = parsed.path if parsed.netloc else ""
        base = f"{scheme}://{netloc}{path}"
        return base.rstrip("/")
    except Exception:
        return url.rstrip("/")


def _extract_urls(text: str) -> List[str]:
    if not text:
        return []
    return [match.rstrip(".,)") for match in URL_RE.findall(text)]


def _collect_customer_sites(messages: List[Dict[str, Any]], company_url: Optional[str]) -> List[str]:
    normalized_company = _normalize_url(company_url) if company_url else None
    seen = set()
    urls: List[str] = []
    for message in messages:
        if _message_role(message) != "user":
            continue
        for raw in _extract_urls(_message_text(message)):
            norm = _normalize_url(raw)
            if normalized_company and norm == normalized_company:
                continue
            if norm in seen:
                continue
            seen.add(norm)
            urls.append(norm)
            if len(urls) >= 5:
                return urls
    return urls
