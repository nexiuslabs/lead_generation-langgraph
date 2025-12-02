"""
Node implementations for the unified orchestrator.

These implementations leverage LangChain/OpenAI when available and gracefully
degrade to rule-based heuristics when credentials are missing. Downstream
LangGraph agents (normalization, ICP refresh, etc.) are reused wherever
possible so behavior matches the legacy stack.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse
import requests

from src.icp import icp_by_ssic_agent, icp_refresh_agent, normalize_agent
from src.lead_scoring import lead_scoring_agent
from src.jobs import _odoo_export_for_ids

try:
    from src.jobs import enqueue_web_discovery_bg_enrich as _enqueue_next40  # legacy stub
except Exception:  # pragma: no cover
    _enqueue_next40 = None  # type: ignore

enqueue_web_discovery_bg_enrich = _enqueue_next40
from src.jina_reader import read_url as jina_read
from src.database import get_conn
from app.pre_sdr_graph import (
    _persist_company_profile_sync,
    _persist_icp_profile_sync,
    _persist_web_candidates_to_staging,
    _fetch_company_profile_record,
    _fetch_icp_profile_record,
    _load_persisted_top10,
)  # type: ignore

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
from langchain_core.runnables.config import var_child_runnable_config  # to read LangGraph run config
try:
    # Available when running under LangGraph Server; provides request/headers/thread metadata
    from langgraph_api.metadata import get_current_metadata as _lg_get_current_metadata  # type: ignore
except Exception:  # pragma: no cover
    _lg_get_current_metadata = None  # type: ignore
from .state import OrchestrationState, ProfileState

# Optional: Command return for custom UI events when running under LangGraph server.
try:
    # Newer LangGraph versions expose Command here
    from langgraph.graph import Command as _LGCommand  # type: ignore
except Exception:  # pragma: no cover
    try:
        # Fallback import path if API differs
        from langgraph_api.types import Command as _LGCommand  # type: ignore
    except Exception:  # pragma: no cover
        _LGCommand = None  # type: ignore

try:  # optional dependency (network / vendor credentials)
    from src.enrichment import enrich_company_with_tavily
except Exception:  # pragma: no cover
    enrich_company_with_tavily = None

try:
    from src.agents_icp import plan_top10_with_reasons
except Exception:  # pragma: no cover
    plan_top10_with_reasons = None

try:
    from src.icp_pipeline import collect_evidence_for_domain
except Exception:  # pragma: no cover
    collect_evidence_for_domain = None  # type: ignore

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
DISCOVERY_CONFIRM_PHRASES = {
    "start discovery",
    "begin discovery",
    "run discovery",
    "start icp",
    "run icp",
    "begin icp",
    "start icp discovery",
    "kick off discovery",
    "kickoff discovery",
    "ready for discovery",
    "ready to start discovery",
    "proceed with discovery",
    "proceed to discovery",
    "go ahead with discovery",
}
DISCOVERY_CONFIRM_TRIGGERS = {
    "start",
    "run",
    "begin",
    "kick off",
    "kickoff",
    "ready",
    "lets",
    "let's",
    "proceed",
    "go ahead",
    "do it",
    "beginning",
    "launch",
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
    if role == "assistant":
        if _EVENT_TOKEN_RE.match(stripped):
            logger.debug("suppressing telemetry event: %s", stripped)
            return True
        if _looks_like_json_blob(stripped):
            logger.debug("suppressing JSON payload: %s", stripped[:120])
            return True
        return False
    if role and role != "user" and _looks_like_json_blob(stripped):
        logger.debug("suppressing %s payload: %s", role, stripped[:120])
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


def _get_run_mode(state: OrchestrationState) -> str:
    ctx = state.get("entry_context") or {}
    raw = str(ctx.get("run_mode") or "").strip().lower()
    if raw in {"chat_top10", "nightly_acra", "acra_direct"}:
        mode = raw
    else:
        mode = "chat_top10"
    if ctx.get("run_mode") != mode:
        ctx["run_mode"] = mode
        state["entry_context"] = ctx
    return mode


def _get_tenant_id(state: OrchestrationState) -> Optional[int]:
    ctx = state.get("entry_context") or {}
    tenant_id = ctx.get("tenant_id") or state.get("tenant_id")
    if isinstance(tenant_id, int):
        return tenant_id
    if isinstance(tenant_id, str):
        raw = tenant_id.strip()
        if raw.isdigit():
            try:
                return int(raw)
            except Exception:
                return None
    return None


def _get_config_int(name: str, default: int) -> int:
    try:
        v = int(os.getenv(name, str(default)) or default)
        return v
    except Exception:
        return default


def _get_config_bool(name: str, default: bool) -> bool:
    try:
        raw = (os.getenv(name) or "").strip().lower()
        if not raw:
            return default
        return raw in {"1", "true", "yes", "on"}
    except Exception:
        return default


def _ensure_profile_state(state: OrchestrationState) -> ProfileState:
    profile = state.get("profile_state") or {}
    profile.setdefault("outstanding_prompts", [])
    profile.setdefault("company_profile", {})
    profile.setdefault("icp_profile", {})
    profile.setdefault("icp_profile_generated", bool(profile.get("icp_profile")))
    profile.setdefault("icp_discovery_confirmed", False)
    profile.setdefault("awaiting_discovery_confirmation", False)
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


def _looks_like_discovery_confirmation(text: str) -> bool:
    normalized = (text or "").strip().lower()
    if not normalized:
        return False
    normalized = normalized.translate(str.maketrans("", "", ".!?"))
    if any(phrase in normalized for phrase in DISCOVERY_CONFIRM_PHRASES):
        return True
    if ("discovery" in normalized or "icp" in normalized) and any(
        trigger in normalized for trigger in DISCOVERY_CONFIRM_TRIGGERS
    ):
        return True
    return False


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


def _parse_structured_company_profile(text: str) -> Dict[str, Any]:
    """Parse a plain-text snippet into a structured company profile.

    Supports simple section headers like:
      Summary\n...
      Industries\nitem\nitem
      Offerings\nitem\n...
      Ideal Customers\n...
      Proof Points\n...

    Returns an empty dict when nothing is confidently extracted.
    """
    if not isinstance(text, str) or not text.strip():
        return {}
    lines = [ln.strip() for ln in text.splitlines()]
    # Normalize: collapse multiple blank lines, keep order
    norm: list[str] = []
    for ln in lines:
        if not ln and (not norm or not norm[-1]):
            continue
        norm.append(ln)
    # Section names we recognize (case-insensitive)
    sections = {
        "summary": {"summary"},
        "industries": {"industry", "industries"},
        "offerings": {"offering", "offerings", "products", "services"},
        "ideal_customers": {"ideal customers", "ideal customer", "customers", "buyers", "personas"},
        "proof_points": {"proof points", "proofs", "evidence", "social proof", "results"},
    }
    # Build reverse lookup mapping of header label -> canonical key
    lookup: dict[str, str] = {}
    for key, labels in sections.items():
        for lab in labels:
            lookup[lab] = key
    # Scan for header indices
    header_idx: list[tuple[int, str]] = []
    for i, ln in enumerate(norm):
        hdr = ln.strip().lower().rstrip(":")
        if hdr in lookup:
            header_idx.append((i, lookup[hdr]))
    if not header_idx:
        return {}
    header_idx.sort(key=lambda x: x[0])
    # Add sentinel end
    header_idx.append((len(norm), "__end"))
    result: Dict[str, Any] = {}
    for (start, key), (end, _next) in zip(header_idx, header_idx[1:]):
        if key == "__end":
            continue
        # Capture the block following the header until the next header or blank break
        block = [ln for ln in norm[start + 1 : end] if ln]
        if not block:
            continue
        if key == "summary":
            summary = " ".join(block).strip()
            # Trim extremely long text to keep UI concise
            result["summary"] = summary[:600]
        else:
            # Split list-like lines into items; also split on bullets or commas or slashes
            items: list[str] = []
            for ln in block:
                raw = ln.lstrip("-•\u2022 ")  # remove common bullet prefixes
                parts = [p.strip() for p in re.split(r"[,/]|\s{2,}", raw) if p.strip()]
                # When the line is meant as a single item (e.g., a short phrase), avoid over-splitting
                if len(parts) == 1:
                    items.append(parts[0])
                else:
                    items.extend(parts)
            # Deduplicate while preserving order
            seen: set[str] = set()
            cleaned: list[str] = []
            for it in items:
                v = it.strip()
                if not v:
                    continue
                low = v.lower()
                if low in seen:
                    continue
                seen.add(low)
                cleaned.append(v)
            result[key] = cleaned
    return result


def _apply_user_company_profile_text(state: OrchestrationState, profile: ProfileState, last_user_text: Optional[str]) -> bool:
    """Best-effort parse of the latest user message into company_profile fields.

    When a user provides a structured snippet with headers like "Summary", "Industries",
    etc., extract fields and persist to `tenant_company_profiles` immediately so the
    orchestrator can continue without requiring LLM parsing.
    """
    if not last_user_text:
        return False
    extracted = _parse_structured_company_profile(last_user_text)
    if not extracted:
        return False
    company = dict(profile.get("company_profile") or {})
    company.update(extracted)
    # Prefer any existing website/name already known; do not overwrite here
    if company.get("summary"):
        company["summary_source"] = company.get("summary_source") or "user"
    profile["company_profile"] = company
    # Draft persist (not confirmed yet); confirmation handled downstream by phrase detection
    try:
        _persist_company_profile_state(state, company, confirmed=False)
    except Exception:
        logger.debug("rule-based company profile persist failed", exc_info=True)
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


def _normalized_icp_for_persistence(icp_profile: Dict[str, Any]) -> Dict[str, Any]:
    profile = dict(icp_profile or {})

    def _copy_list(src: str, dest: str) -> None:
        if profile.get(dest):
            return
        raw = profile.get(src)
        if isinstance(raw, list):
            cleaned = [str(v).strip() for v in raw if isinstance(v, str) and v.strip()]
        elif isinstance(raw, str) and raw.strip():
            cleaned = [raw.strip()]
        else:
            cleaned = []
        if cleaned:
            profile[dest] = cleaned

    _copy_list("company_sizes", "size_bands")
    _copy_list("persona_titles", "buyer_titles")
    _copy_list("buying_triggers", "triggers")
    _copy_list("regions", "geos")
    return profile


def _persist_icp_profile_state(state: OrchestrationState, icp_profile: Dict[str, Any], seed_urls: List[str]) -> None:
    ctx = state.get("entry_context") or {}
    tenant_id = ctx.get("tenant_id") or state.get("tenant_id")
    if not tenant_id:
        return
    persist_state = {"tenant_id": tenant_id}
    try:
        normalized_profile = _normalized_icp_for_persistence(icp_profile)
        _persist_icp_profile_sync(
            persist_state,
            normalized_profile,
            confirmed=True,
            seed_urls=seed_urls,
            user_confirmed=False,
        )
    except Exception:
        logger.warning("Failed to persist ICP profile", exc_info=True)


def _persist_discovery_candidates(state: OrchestrationState, details: Sequence[Dict[str, Any]]) -> None:
    ctx = state.get("entry_context") or {}
    tenant_id = ctx.get("tenant_id") or state.get("tenant_id")
    if not tenant_id:
        return
    domains: List[str] = []
    per_meta: Dict[str, Dict[str, Any]] = {}
    for idx, item in enumerate(details or []):
        if not isinstance(item, dict):
            continue
        dom = (item.get("domain") or "").strip().lower()
        if not dom:
            continue
        domains.append(dom)
        meta = {
            "preview": idx < 10,
            "score": item.get("score"),
            "bucket": item.get("bucket"),
            "why": item.get("why"),
            "table_row": item,
            "provenance": {
                "agent": "orchestrator.discovery",
                "stage": "preview" if idx < 10 else "staging",
            },
        }
        per_meta[dom] = meta
    if not domains:
        return
    try:
        _persist_web_candidates_to_staging(
            domains,
            int(tenant_id) if isinstance(tenant_id, int) or str(tenant_id).isdigit() else None,
            per_domain_meta=per_meta,
            ai_metadata={"provenance": {"agent": "orchestrator.discovery"}},
        )
    except Exception:
        logger.debug("staging persistence skipped", exc_info=True)


def _cache_discovery_details(
    state: OrchestrationState, discovery_state: Dict[str, Any], details: Sequence[Dict[str, Any]]
) -> None:
    if not details:
        return
    detail_list = [item for item in details if isinstance(item, dict)]
    if not detail_list:
        return
    deduped: List[Dict[str, Any]] = []
    domains: List[str] = []
    seen: set[str] = set()
    for item in detail_list:
        dom = _domain_from_value(str(item.get("domain") or ""))
        if not dom or dom in seen:
            continue
        seen.add(dom)
        domains.append(dom)
        deduped.append(dict(item))
    if not deduped:
        return
    discovery_state["planned_candidates"] = deduped
    discovery_state["web_candidate_details"] = deduped
    if domains:
        discovery_state["web_candidates"] = domains
    top_slice = deduped[:10]
    next_slice = deduped[10:50]
    discovery_state["top10_details"] = top_slice
    discovery_state["next40_details"] = next_slice
    discovery_state["top10_domains"] = domains[:10]
    discovery_state["next40_domains"] = domains[10:50]
    tenant_ctx = _get_tenant_id(state)
    id_map = _ensure_company_ids_for_domains(domains, tenant_ctx)
    top_ids = [id_map.get(dom) for dom in discovery_state["top10_domains"] if id_map.get(dom)]
    next_ids = [id_map.get(dom) for dom in discovery_state["next40_domains"] if id_map.get(dom)]
    if top_ids:
        discovery_state["top10_ids"] = top_ids
        discovery_state["candidate_ids"] = top_ids
    if next_ids:
        discovery_state["next40_ids"] = next_ids


def _candidate_domain_lookup(discovery_state: Dict[str, Any]) -> Dict[int, str]:
    lookup: Dict[int, str] = {}
    primary_ids = discovery_state.get("top10_ids") or []
    primary_domains = discovery_state.get("top10_domains") or []
    for raw_id, raw_dom in zip(primary_ids, primary_domains):
        try:
            cid = int(raw_id)
        except Exception:
            continue
        dom = _domain_from_value(str(raw_dom or ""))
        if dom:
            lookup[cid] = dom
    if lookup:
        return lookup
    fallback_ids = discovery_state.get("candidate_ids") or []
    fallback_domains = discovery_state.get("web_candidates") or []
    for raw_id, raw_dom in zip(fallback_ids, fallback_domains):
        try:
            cid = int(raw_id)
        except Exception:
            continue
        dom = _domain_from_value(str(raw_dom or ""))
        if dom:
            lookup[cid] = dom
    return lookup


async def _attempt_jina_enrichment(
    tenant_id: Optional[int], company_id: int, domain: str
) -> Tuple[bool, Optional[str]]:
    cleaned = _domain_from_value(domain)
    if not cleaned:
        return False, "missing_domain"
    if collect_evidence_for_domain is not None and tenant_id:
        try:
            inserted = await collect_evidence_for_domain(int(tenant_id), int(company_id), cleaned)
            return (inserted > 0, None if inserted else "no_evidence")
        except Exception as exc:  # pragma: no cover - network/DB variability
            return False, str(exc)
    try:
        snapshot = jina_read(f"https://{cleaned}", timeout=8)
        if snapshot:
            return True, None
        return False, "empty_snapshot"
    except Exception as exc:  # pragma: no cover
        return False, str(exc)


async def _export_top10_to_odoo(tenant_id: Optional[int], company_ids: Sequence[int]) -> bool:
    if tenant_id is None:
        return False
    normalized: List[int] = []
    seen: set[int] = set()
    for raw in company_ids or []:
        try:
            cid = int(raw)
        except Exception:
            continue
        if cid <= 0 or cid in seen:
            continue
        seen.add(cid)
        normalized.append(cid)
    if not normalized:
        return False
    rows: List[Tuple[int, str, Optional[str]]] = []
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT company_id, name, uen FROM companies WHERE company_id = ANY(%s)",
                (normalized,),
            )
            rows = cur.fetchall() or []
    except Exception as exc:
        logger.warning("fetching companies for Odoo export failed: %s", exc)
        return False
    if not rows:
        return False
    try:
        await _odoo_export_for_ids(int(tenant_id), rows)
        return True
    except Exception as exc:  # pragma: no cover - network variability
        logger.warning("odoo export failed: %s", exc)
        return False


def _ensure_company_ids_for_domains(domains: Sequence[str], tenant_id: Optional[int]) -> Dict[str, int]:
    """Ensure each domain has a row in companies and return {domain: company_id}."""
    cleaned: List[str] = []
    seen = set()
    for raw in domains or []:
        dom = _domain_from_value(str(raw or ""))
        if dom and dom not in seen:
            seen.add(dom)
            cleaned.append(dom)
    if not cleaned:
        return {}
    mapping: Dict[str, int] = {}
    try:
        with get_conn() as conn, conn.cursor() as cur:
            if isinstance(tenant_id, int):
                try:
                    cur.execute("SELECT set_config('request.tenant_id', %s, true)", (str(tenant_id),))
                except Exception:
                    pass
            for dom in cleaned:
                cid: Optional[int] = None
                try:
                    cur.execute("SELECT company_id FROM companies WHERE website_domain=%s LIMIT 1", (dom,))
                    row = cur.fetchone()
                    if row and row[0] is not None:
                        cid = int(row[0])
                except Exception:
                    cid = None
                if cid is None:
                    try:
                        cur.execute(
                            "INSERT INTO companies(name, website_domain, last_seen) VALUES (%s,%s,NOW()) RETURNING company_id",
                            (dom, dom),
                        )
                        row = cur.fetchone()
                        if row and row[0] is not None:
                            cid = int(row[0])
                    except Exception:
                        cid = None
                if cid is not None:
                    mapping[dom] = cid
            conn.commit()
    except Exception:
        logger.debug("ensure_company_ids failed", exc_info=True)
    return mapping


def _lookup_primary_emails(company_ids: Sequence[int]) -> Dict[int, str]:
    if not company_ids:
        return {}
    normalized_ids = sorted({int(cid) for cid in company_ids if isinstance(cid, int)})
    if not normalized_ids:
        return {}
    emails: Dict[int, str] = {}
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT company_id, email FROM lead_emails WHERE company_id = ANY(%s)",
                (normalized_ids,),
            )
            for row in cur.fetchall() or []:
                try:
                    cid = int(row[0])
                except Exception:
                    continue
                email = (row[1] or "").strip()
                if email:
                    emails[cid] = email
    except Exception:
        logger.debug("email lookup failed", exc_info=True)
    return emails


def _format_lead_score_table(scores: Sequence[Dict[str, Any]], discovery: Dict[str, Any], email_map: Dict[int, str]) -> str:
    if not scores:
        return ""
    detail_lookup: Dict[int, Dict[str, Any]] = {}
    details = discovery.get("top10_details") or []
    ids = discovery.get("top10_ids") or []
    for raw_id, detail in zip(ids, details):
        if not isinstance(detail, dict):
            continue
        try:
            cid = int(raw_id)
        except Exception:
            continue
        detail_lookup[cid] = detail
    has_email = any(
        email_map.get(int(row.get("company_id") or 0))
        for row in scores
        if row.get("company_id") is not None
    )
    header = ["#", "Company", "Domain", "Score", "Bucket"]
    if has_email:
        header.append("Email")
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for idx, score_row in enumerate(scores, start=1):
        try:
            cid = int(score_row.get("company_id"))
        except Exception:
            cid = None
        detail = detail_lookup.get(cid or -1, {}) if cid is not None else {}
        name = detail.get("name") or _name_from_domain(detail.get("domain") or "")
        domain = _domain_from_value(detail.get("domain") or "")
        score_val = score_row.get("score")
        try:
            shown_score = f"{int(round(float(score_val)))}"
        except Exception:
            shown_score = str(score_val or "-")
        bucket = str(score_row.get("bucket") or detail.get("bucket") or "-").upper()
        row = [str(idx), name or "-", domain or "-", shown_score, bucket or "-"]
        if has_email:
            email = email_map.get(cid or -1, "") if cid is not None else ""
            row.append(email or "-")
        lines.append("| " + " | ".join(row) + " |")
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
    # Treat punctuation-agnostic direct questions as questions (even without '?')
    _q_norm = lowered.strip().replace("’", "'")
    if any(_q_norm.startswith(p) for p in (
        "what is my ", "what's my ", "whats my ", "what is our ", "what's our ", "whats our ",
        "what is the ", "what's the ", "whats the ",
    )):
        return "question"
    if any(_q_norm.startswith(p) for p in (
        "show me ", "show ", "give me ", "tell me ", "where is ", "list ",
    )):
        return "question"
    # Update/edit intents
    if any(_q_norm.startswith(p) for p in (
        "update ", "change ", "set ", "remove ", "add ", "edit ", "delete ",
    )):
        return "update"
    if any(keyword in lowered for keyword in ("status", "progress", "job status", "queue", "queued", "running", "done")):
        return "status"
    if any(keyword in lowered for keyword in ("http://", "https://")):
        return "confirm_company"
    if "confirm" in lowered and "company" in lowered:
        return "confirm_company"
    if "confirm" in lowered and "icp" in lowered:
        return "confirm_icp"
    if _looks_like_discovery_confirmation(text):
        return "confirm_discovery"
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
Return JSON {{"intent": one of [status, run_enrichment, confirm_icp, confirm_company, confirm_discovery, accept_micro_icp, question, chat]}}.
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
    # Best-effort: hydrate tenant context from LangGraph run config / SDK context / thread metadata
    try:
        # 0) Read tenant from 'context' on state (LangGraph SDK clients often attach it there)
        try:
            ctx_in = state.get("context") or {}
            if isinstance(ctx_in, dict):
                tid0 = ctx_in.get("tenant_id") or ctx_in.get("tenantId")
                if tid0 is not None:
                    tid_int = int(str(tid0).strip())
                    ctx0 = state.get("entry_context") or {}
                    if ctx0.get("tenant_id") != tid_int:
                        ctx0["tenant_id"] = tid_int
                        state["entry_context"] = ctx0
                    state["tenant_id"] = tid_int
                    _log_step("tenant_context", source="state.context", tenant_id=tid_int)
                em0 = ctx_in.get("notify_email") or ctx_in.get("user_email")
                if isinstance(em0, str) and "@" in em0:
                    ctx0 = state.get("entry_context") or {}
                    if not ctx0.get("notify_email"):
                        ctx0["notify_email"] = em0.strip()
                        state["entry_context"] = ctx0
        except Exception:
            pass
        cfg = var_child_runnable_config.get() or {}
        conf = cfg.get("configurable", {}) if isinstance(cfg, dict) else {}
        # Common places the SDK/server pass tenant metadata
        meta = {}
        try:
            # sometimes nested under configurable.metadata
            m = conf.get("metadata") or {}
            if isinstance(m, dict):
                meta.update(m)
        except Exception:
            pass
        try:
            # updates may carry context from the client submit
            c = conf.get("context") or {}
            if isinstance(c, dict):
                meta.update(c)
        except Exception:
            pass
        # Some deployments may put tenant directly at top-level configurable
        for key in ("tenant_id", "tenantId"):
            if key in conf and conf.get(key) is not None:
                meta.setdefault("tenant_id", conf.get(key))
        tid = meta.get("tenant_id")
        if tid is not None:
            try:
                tid_int = int(str(tid).strip())
                ctx0 = state.get("entry_context") or {}
                if ctx0.get("tenant_id") != tid_int:
                    ctx0["tenant_id"] = tid_int
                    state["entry_context"] = ctx0
                state["tenant_id"] = tid_int
                _log_step("tenant_context", source="configurable/context/metadata", tenant_id=tid_int)
            except Exception:
                pass
        # Optional: pick up notify_email if the client provided it in context
        em = meta.get("notify_email") or meta.get("user_email")
        if isinstance(em, str) and "@" in em:
            ctx0 = state.get("entry_context") or {}
            if not ctx0.get("notify_email"):
                ctx0["notify_email"] = em.strip()
                state["entry_context"] = ctx0
        # Finally: check LangGraph Server request metadata for X-Tenant-ID header or thread metadata
        if _lg_get_current_metadata is not None and (state.get("tenant_id") is None):
            try:
                md = _lg_get_current_metadata() or {}
                headers = {}
                # Typical shapes: { headers: {...} } or { request: { headers: {...} } }
                if isinstance(md.get("headers"), dict):
                    headers = md.get("headers")  # type: ignore[assignment]
                elif isinstance(md.get("request"), dict) and isinstance(md["request"].get("headers"), dict):  # type: ignore[index]
                    headers = md["request"]["headers"]  # type: ignore[index,assignment]
                tid_hdr = None
                if headers:
                    # Lower/upper variants depending on server
                    tid_hdr = headers.get("x-tenant-id") or headers.get("X-Tenant-ID")
                # Thread metadata (if present)
                if tid_hdr is None:
                    tmeta = None
                    if isinstance(md.get("thread"), dict):
                        t = md.get("thread")  # type: ignore[assignment]
                        tmeta = t.get("metadata") if isinstance(t.get("metadata"), dict) else None  # type: ignore[index]
                    elif isinstance(md.get("metadata"), dict):
                        tmeta = md.get("metadata")  # type: ignore[assignment]
                    if tmeta and isinstance(tmeta.get("tenant_id"), (str, int)):
                        tid_hdr = tmeta.get("tenant_id")
                if tid_hdr is not None:
                    tid_int = int(str(tid_hdr).strip())
                    ctx0 = state.get("entry_context") or {}
                    if ctx0.get("tenant_id") != tid_int:
                        ctx0["tenant_id"] = tid_int
                        state["entry_context"] = ctx0
                    state["tenant_id"] = tid_int
                    _log_step("tenant_context", source="lg_metadata(headers/thread)", tenant_id=tid_int)
            except Exception:
                pass
    except Exception:
        # Never fail ingestion due to config inspection
        pass

    incoming = str(state.get("input") or "").strip()
    role = state.get("input_role", "user")
    # When no new user input, mark suppress_output to avoid repeating assistant messages on SDK re-connects
    state["suppress_output"] = False if incoming else True
    if incoming:
        _append_message(state, role, incoming)
        _log_step("user_message", role=role, text=incoming[:500])
        prompt = f"""
You normalize chat inputs for a lead-generation orchestrator.
Respond with JSON {{"normalized_text": "...", "intent": "...", "tags": [...]}}.
Input: {incoming!r}
Intent options: run_enrichment, confirm_company, confirm_icp, confirm_discovery, accept_micro_icp, question, chat, idle.
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
        # If the latest input is a question/update, clear/downgrade outstanding prompts to avoid overshadowing the action
        try:
            _last = ctx.get("last_user_command") or ""
            _int = ctx.get("intent") or _heuristic_intent(_last)
            if _int in {"question", "update"}:
                profile = _ensure_profile_state(state)
                profile["outstanding_prompts"] = []
                state["profile_state"] = profile
        except Exception:
            pass
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


async def return_user_probe(state: OrchestrationState) -> OrchestrationState:
    """Probe for prior tenant context and cached discovery to minimize re-asks.

    Loads persisted company/ICP profiles and Top-10 preview (if any) and
    computes a minimal decision: reuse cached vs request re-run. Designed to be
    idempotent and fast; falls back to heuristics when DB helpers are missing.
    """
    profile = _ensure_profile_state(state)
    # Resolve tenant context early so persisted snapshots can be loaded
    tenant_id = _get_tenant_id(state)
    if tenant_id is None:
        try:
            # Attempt to read from state.context / configurable metadata
            ctx_in = state.get("context") or {}
            if isinstance(ctx_in, dict):
                tid0 = ctx_in.get("tenant_id") or ctx_in.get("tenantId")
                if tid0 is not None:
                    tid_int = int(str(tid0).strip())
                    ctx0 = state.get("entry_context") or {}
                    ctx0["tenant_id"] = tid_int
                    state["entry_context"] = ctx0
                    state["tenant_id"] = tid_int
                    tenant_id = tid_int
                    _log_step("tenant_context", source="probe/state.context", tenant_id=tid_int)
        except Exception:
            pass
        try:
            cfg = var_child_runnable_config.get() or {}
            conf = cfg.get("configurable", {}) if isinstance(cfg, dict) else {}
            meta = {}
            if isinstance(conf.get("metadata"), dict):
                meta.update(conf.get("metadata"))
            if isinstance(conf.get("context"), dict):
                meta.update(conf.get("context"))
            for key in ("tenant_id", "tenantId"):
                if key in conf and conf.get(key) is not None:
                    meta.setdefault("tenant_id", conf.get(key))
            tid = meta.get("tenant_id")
            if tid is not None:
                tid_int = int(str(tid).strip())
                ctx0 = state.get("entry_context") or {}
                ctx0["tenant_id"] = tid_int
                state["entry_context"] = ctx0
                state["tenant_id"] = tid_int
                tenant_id = tid_int
                _log_step("tenant_context", source="probe/configurable/context/metadata", tenant_id=tid_int)
        except Exception:
            pass
        if tenant_id is None and _lg_get_current_metadata is not None:
            try:
                md = _lg_get_current_metadata() or {}
                headers = {}
                if isinstance(md.get("headers"), dict):
                    headers = md.get("headers")
                elif isinstance(md.get("request"), dict) and isinstance(md["request"].get("headers"), dict):
                    headers = md["request"]["headers"]  # type: ignore[index]
                tid_hdr = headers.get("x-tenant-id") or headers.get("X-Tenant-ID") if headers else None
                if tid_hdr is None:
                    tmeta = None
                    if isinstance(md.get("thread"), dict):
                        tmeta = md.get("thread").get("metadata") if isinstance(md.get("thread").get("metadata"), dict) else None  # type: ignore[index]
                    elif isinstance(md.get("metadata"), dict):
                        tmeta = md.get("metadata")
                    if tmeta and isinstance(tmeta.get("tenant_id"), (str, int)):
                        tid_hdr = tmeta.get("tenant_id")
                if tid_hdr is not None:
                    tid_int = int(str(tid_hdr).strip())
                    ctx0 = state.get("entry_context") or {}
                    ctx0["tenant_id"] = tid_int
                    state["entry_context"] = ctx0
                    state["tenant_id"] = tid_int
                    tenant_id = tid_int
                    _log_step("tenant_context", source="probe/lg_metadata", tenant_id=tid_int)
            except Exception:
                pass
        # Final DB fallback in dev: pick first active Odoo tenant
        if tenant_id is None:
            try:
                with get_conn() as conn, conn.cursor() as cur:
                    from src.settings import ALLOW_DB_TENANT_FALLBACK as _ALLOW
                    if _ALLOW:
                        cur.execute("SELECT tenant_id FROM odoo_connections WHERE active=TRUE LIMIT 1")
                        row = cur.fetchone()
                        if row and row[0] is not None:
                            tid_int = int(row[0])
                            ctx0 = state.get("entry_context") or {}
                            ctx0["tenant_id"] = tid_int
                            state["entry_context"] = ctx0
                            state["tenant_id"] = tid_int
                            tenant_id = tid_int
                            _log_step("tenant_context", source="probe/db_fallback", tenant_id=tid_int)
            except Exception:
                pass
    decisions: Dict[str, Any] = {
        "is_return_user": False,
        "use_cached": False,
        "rerun_icp": False,
        "reason": "no_prior_context",
        "stale_signals": {},
        "diffs": {},
    }
    candidate_details: List[Dict[str, Any]] = []
    last_preview_ts: Optional[str] = None
    loaded_any = False

    # Attempt to load persisted profiles for the tenant
    if isinstance(tenant_id, int) and tenant_id > 0:
        try:
            comp = _fetch_company_profile_record(int(tenant_id))
        except Exception:
            comp = None
        try:
            icp = _fetch_icp_profile_record(int(tenant_id))
        except Exception:
            icp = None
        if comp and isinstance(comp.get("profile"), dict):
            company = dict(comp.get("profile") or {})
            # Ensure website is set from source_url if missing
            src_url = comp.get("source_url")
            if (not company.get("website")) and isinstance(src_url, str) and src_url.strip():
                company["website"] = src_url.strip()
            if company:
                profile["company_profile"] = company
                profile["company_profile_confirmed"] = bool(comp.get("confirmed")) or bool(company.get("website"))
                # Hydration implies we seeded from persistence; prevent re-ask wording
                profile["seeded_company_profile"] = False
                loaded_any = True
        if icp and isinstance(icp.get("profile"), dict):
            profile["icp_profile"] = dict(icp.get("profile") or {})
            profile["icp_profile_confirmed"] = bool(icp.get("confirmed")) or bool(profile.get("icp_profile"))
            loaded_any = True
        # Load any recently persisted Top‑10 preview for reuse
        try:
            top = _load_persisted_top10(int(tenant_id)) or []
        except Exception:
            top = []
        if top:
            candidate_details = top[:10]
            # Read latest preview timestamp for staleness evaluation
            try:
                with get_conn() as conn, conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT created_at
                          FROM staging_global_companies
                         WHERE (tenant_id = %s OR %s IS NULL)
                           AND COALESCE((ai_metadata->>'preview')::boolean,false) = true
                         ORDER BY created_at DESC
                         LIMIT 1
                        """,
                        (int(tenant_id), int(tenant_id)),
                    )
                    r = cur.fetchone()
                    if r and r[0] is not None:
                        last_preview_ts = str(r[0])
            except Exception:
                last_preview_ts = None

    # Decide reuse vs rerun (heuristic + optional LLM policy)
    candidate_count = len(candidate_details)
    stale_days = None
    is_stale = False
    try:
        from src.settings import DISCOVERY_STALENESS_DAYS as _STALE
    except Exception:
        _STALE = 14
    if candidate_count:
        if last_preview_ts:
            try:
                t = datetime.fromisoformat(last_preview_ts.replace("Z", "+00:00"))
                age = (datetime.now(timezone.utc) - t).days
                stale_days = age
                is_stale = age > int(_STALE)
            except Exception:
                is_stale = False
        if not is_stale:
            decisions.update({
                "is_return_user": True,
                "use_cached": True,
                "rerun_icp": False,
                "reason": "cached_top10_preview",
            })
        else:
            decisions.update({
                "is_return_user": True,
                "use_cached": False,
                "rerun_icp": True,
                "reason": f"stale_preview_over_{_STALE}d",
                "stale_signals": {"last_preview_at": last_preview_ts, "window_days": _STALE},
            })
    elif loaded_any:
        decisions.update({
            "is_return_user": True,
            "use_cached": False,
            "rerun_icp": False,
            "reason": "profiles_loaded_no_candidates",
        })

    # Optional LLM policy to refine the decision (fallback to heuristics)
    try:
        company = (profile.get("company_profile") or {})
        icp_prof = (profile.get("icp_profile") or {})
        intent = (state.get("entry_context") or {}).get("intent")
        text = (state.get("entry_context") or {}).get("last_user_command")
        prompt = f"""
You are a cautious orchestrator policy assistant. Decide whether to reuse cached ICP discovery results or re-run discovery.
Return ONLY JSON with keys: is_return_user, use_cached, rerun_icp, reason, stale_signals, diffs.

company_profile: {company!r}
icp_profile: {icp_prof!r}
cached_discovery: {{"count": {candidate_count}, "top10_age_days": {stale_days}, "last_run_at": {last_preview_ts!r}}}
rules: {{"icp_rule_name": "default", "profile_staleness_days": 14, "discovery_staleness_days": {_STALE}}}
incoming_intent: {intent!r}
incoming_text: {text!r}

Heuristics:
- use_cached when: candidates exist, fresh (<= discovery_staleness_days), and no breaking diffs.
- rerun_icp when: website/industries changed, rule drift, or stale (> discovery_staleness_days).
Respond with JSON only.
"""
        fallback = {
            "is_return_user": bool(decisions.get("is_return_user")),
            "use_cached": bool(decisions.get("use_cached")),
            "rerun_icp": bool(decisions.get("rerun_icp")),
            "reason": str(decisions.get("reason") or "heuristic"),
            "stale_signals": decisions.get("stale_signals") or {},
            "diffs": decisions.get("diffs") or {},
        }
        llm_dec = call_llm_json(prompt, fallback) or fallback
        # Normalize
        decisions["is_return_user"] = bool(llm_dec.get("is_return_user", fallback["is_return_user"]))
        decisions["use_cached"] = bool(llm_dec.get("use_cached", fallback["use_cached"]))
        decisions["rerun_icp"] = bool(llm_dec.get("rerun_icp", fallback["rerun_icp"]))
        if decisions["use_cached"] and decisions["rerun_icp"]:
            # prefer rerun if conflict
            decisions["use_cached"] = False
        reason = llm_dec.get("reason") or decisions.get("reason") or "policy"
        decisions["reason"] = str(reason)
        if isinstance(llm_dec.get("stale_signals"), dict):
            decisions["stale_signals"] = llm_dec.get("stale_signals")
        if isinstance(llm_dec.get("diffs"), dict):
            decisions["diffs"] = llm_dec.get("diffs")
    except Exception:
        pass

    # Guard: On first-run with no cached candidates, do NOT suggest rerun.
    # This prevents the confusing "Re-run discovery" prompt when there is nothing to rerun.
    if not candidate_details:
        decisions["rerun_icp"] = False

    # Write discovery cache for downstream nodes to optionally reuse
    if candidate_details:
        disc = state.get("discovery") or {}
        disc["web_candidate_details"] = candidate_details
        disc["top10_details"] = candidate_details[:10]
        # Derive domains + ids if possible (ids may be absent in preview; leave empty)
        doms = [_domain_from_value(d.get("domain") or "") for d in candidate_details if isinstance(d, dict)]
        disc["top10_domains"] = [d for d in doms if d]
        state["discovery"] = disc

    # Persist updated profile_state back
    state["profile_state"] = profile
    state["decisions"] = decisions
    state["is_return_user"] = bool(decisions.get("is_return_user"))

    # Status/logs
    if decisions.get("is_return_user"):
        msg = "Found prior context; will reuse cached candidates" if decisions.get("use_cached") else "Found prior context; no cached candidates to reuse"
    else:
        msg = "No prior context found"
    _log_step(
        "return_user_probe",
        tenant_id=_get_tenant_id(state),
        is_return=bool(decisions.get("is_return_user")),
        use_cached=bool(decisions.get("use_cached")),
        reason=decisions.get("reason"),
        cached=len(candidate_details),
    )
    _set_status(state, "return_user_probe", msg)
    return state


async def profile_builder(state: OrchestrationState) -> OrchestrationState:
    """Update profile flags using the latest conversation snippet."""
    profile = _ensure_profile_state(state)
    history = state.get("messages", [])[-5:]
    # First try a deterministic parse of a user‑provided structured snippet
    try:
        last_user = next((m for m in reversed(history) if _message_role(m) == "user"), None)
        last_user_text = _message_text(last_user) if last_user else None
        if _apply_user_company_profile_text(state, profile, last_user_text):
            # If we captured new fields from user text, treat as not yet confirmed
            profile["company_profile_confirmed"] = False
    except Exception:
        pass
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
        if (
            last_intent in {"confirm_icp", "run_enrichment"}
            or _includes_phrase(normalized_command, icp_confirm_phrases)
            or _looks_like_discovery_confirmation(normalized_command)
        ):
            # Only confirm ICP if it was generated from customer sites and we have at least 5
            sites = profile.get("customer_websites") or []
            have5 = isinstance(sites, list) and len([u for u in sites if isinstance(u, str) and u.strip()]) >= 5
            generated = bool(profile.get("icp_profile_generated")) and bool(profile.get("icp_profile"))
            if generated and have5:
                profile["icp_profile_confirmed"] = True

    enrichment_requested = _wants_enrichment(last_intent, normalized_command)
    if profile.get("icp_discovery_confirmed") and enrichment_requested:
        profile["enrichment_confirmed"] = True
        profile["awaiting_enrichment_confirmation"] = False

    awaiting_discovery = bool(profile.get("awaiting_discovery_confirmation"))
    if awaiting_discovery and not profile.get("icp_discovery_confirmed"):
        positive_intents = {"run_enrichment", "accept_micro_icp", "confirm_icp", "confirm_discovery"}
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
        elif _looks_like_discovery_confirmation(normalized):
            confirm_discovery = True
        if confirm_discovery:
            profile["icp_discovery_confirmed"] = True
            profile["awaiting_discovery_confirmation"] = False

    if profile.get("icp_discovery_confirmed"):
        profile["awaiting_discovery_confirmation"] = False

    retry_triggers = ("retry discovery", "retry icp", "search again")
    if any(trigger in last_command for trigger in retry_triggers):
        profile["discovery_retry_requested"] = True
        profile["enrichment_confirmed"] = False

    if profile.get("seeded_company_profile"):
        profile["company_profile_confirmed"] = False

    # Final guardrail: never allow ICP to be marked confirmed unless we have a generated ICP
    # AND at least 5 customer websites. This overrides any premature LLM booleans.
    sites = profile.get("customer_websites") or []
    have5 = isinstance(sites, list) and len([u for u in sites if isinstance(u, str) and u.strip()]) >= 5
    generated_icp = bool(profile.get("icp_profile_generated")) and bool(profile.get("icp_profile"))
    if profile.get("icp_profile_confirmed") and not (generated_icp and have5):
        profile["icp_profile_confirmed"] = False

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
    enrichment_ready = True if BG_DISCOVERY_AND_ENRICH else bool(profile.get("enrichment_confirmed"))
    ctx = state.get("entry_context") or {}
    question = ctx.get("last_user_command") or _latest_user_text(state)
    last_intent = (ctx.get("intent") or (_heuristic_intent(question) if question else "")).strip().lower()
    last_command = (ctx.get("last_user_command") or "").strip().lower()
    normalized_command = last_command.translate(str.maketrans("", "", ".!?")).strip()

    # Honor return-user decisions: reuse cached vs ask to re-run
    try:
        decisions = state.get("decisions") or {}
        if isinstance(decisions, dict) and decisions:
            if decisions.get("use_cached") and (company_ready and icp_ready):
                profile["icp_discovery_confirmed"] = True
                profile["awaiting_discovery_confirmation"] = False
                discovery_ready = True
            elif decisions.get("rerun_icp") and (company_ready and icp_ready):
                # Require a positive confirmation before proceeding
                ask = f"Re-run discovery now due to {decisions.get('reason') or 'policy'}? (yes/no)"
                # Look for a yes-like response in the last turn
                positive_intents = {"confirm_discovery", "confirm", "run_enrichment"}
                positive_phrases = {"yes", "yep", "ok", "okay", "start discovery", "run discovery", "begin discovery", "proceed"}
                normalized = normalized_command
                wants = False
                if last_intent in positive_intents or any(p in normalized for p in positive_phrases) or _looks_like_discovery_confirmation(normalized):
                    wants = True
                if wants:
                    profile["icp_discovery_confirmed"] = True
                    profile["awaiting_discovery_confirmation"] = False
                    discovery_ready = True
                else:
                    profile["awaiting_discovery_confirmation"] = True
                    # Make this the primary prompt so it isn't overshadowed by older asks
                    profile["outstanding_prompts"] = [ask]
                    _append_message(state, "assistant", ask)
    except Exception:
        pass
    if discovery_ready and not enrichment_ready and _wants_enrichment(last_intent, normalized_command):
        profile["enrichment_confirmed"] = True
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
    # Persist ICP profile as confirmed when user has confirmed it in chat
    if icp_ready and icp_profile:
        try:
            normalized = _normalized_icp_for_persistence(icp_profile)
            _persist_icp_profile_sync(
                {"tenant_id": _get_tenant_id(state)},
                normalized,
                confirmed=True,
                seed_urls=(customer_sites or [])[:5],
                user_confirmed=True,
            )
        except Exception:
            logger.debug("icp confirmed persist failed", exc_info=True)
    if not needs_website:
        if _ensure_company_summary(state, profile, state.get("messages", [])[-8:]):
            company_profile = profile.get("company_profile") or {}
    explanation = None
    # Background job creation should happen only after both profiles are confirmed
    # and (optionally) after collecting a minimum number of customer websites.
    if BG_DISCOVERY_AND_ENRICH and company_ready and icp_ready:
        # Require customer websites before enqueueing (configurable, default 5)
        min_customers = _get_config_int("CUSTOMER_SITES_REQUIRED", 5)
        have_customers = isinstance(customer_sites, list) and len([u for u in customer_sites if str(u).strip()]) >= max(0, min_customers)
        if not have_customers and min_customers > 0:
            ask = (
                f"Great — company and ICP are confirmed. Please share {min_customers} customer website URLs "
                f"(e.g., your best-fit customers) so I can tune discovery before I queue the background job."
            )
            # Always surface the latest gating prompt as primary
            profile["outstanding_prompts"] = [ask]
            _append_message(state, "assistant", ask)
            _set_status(state, "journey_guard", "Awaiting customer websites")
            _log_step("journey_guard", company_ready=company_ready, icp_ready=icp_ready, need_customer_urls=True, required=min_customers)
            state["profile_state"] = profile
            return state

        # Ensure tenant_id is set; if missing, attempt to read from client-provided context and LG server metadata
        tenant_id = _get_tenant_id(state) or 0
        if not tenant_id:
            try:
                ctx_in = state.get("context") or {}
                if isinstance(ctx_in, dict):
                    tid0 = ctx_in.get("tenant_id") or ctx_in.get("tenantId")
                    if tid0 is not None:
                        tid_int = int(str(tid0).strip())
                        ctx2 = state.get("entry_context") or {}
                        ctx2["tenant_id"] = tid_int
                        state["entry_context"] = ctx2
                        state["tenant_id"] = tid_int
                        tenant_id = tid_int
                        _log_step("tenant_context", source="journey_guard/state.context", tenant_id=tid_int)
            except Exception:
                pass
        if not tenant_id and _lg_get_current_metadata is not None:
            try:
                md = _lg_get_current_metadata() or {}
                headers = {}
                if isinstance(md.get("headers"), dict):
                    headers = md.get("headers")  # type: ignore[assignment]
                elif isinstance(md.get("request"), dict) and isinstance(md["request"].get("headers"), dict):  # type: ignore[index]
                    headers = md["request"]["headers"]  # type: ignore[index,assignment]
                tid = headers.get("x-tenant-id") or headers.get("X-Tenant-ID")
                if tid is None:
                    tmeta = None
                    if isinstance(md.get("thread"), dict):
                        t = md.get("thread")  # type: ignore[assignment]
                        tmeta = t.get("metadata") if isinstance(t.get("metadata"), dict) else None  # type: ignore[index]
                    elif isinstance(md.get("metadata"), dict):
                        tmeta = md.get("metadata")  # type: ignore[assignment]
                    if tmeta and isinstance(tmeta.get("tenant_id"), (str, int)):
                        tid = tmeta.get("tenant_id")
                if tid is not None:
                    try:
                        tid_int = int(str(tid).strip())
                        ctx2 = state.get("entry_context") or {}
                        ctx2["tenant_id"] = tid_int
                        state["entry_context"] = ctx2
                        state["tenant_id"] = tid_int
                        tenant_id = tid_int
                        _log_step("tenant_context", source="journey_guard/lg_metadata", tenant_id=tid_int)
                    except Exception:
                        pass
            except Exception:
                pass
        # Final fallback (dev-friendly): if still missing, try first active Odoo connection
        if not tenant_id:
            try:
                with get_conn() as conn, conn.cursor() as cur:
                    from src.settings import ALLOW_DB_TENANT_FALLBACK as _ALLOW
                    if _ALLOW:
                        cur.execute("SELECT tenant_id FROM odoo_connections WHERE active=TRUE LIMIT 1")
                        row = cur.fetchone()
                        if row and row[0] is not None:
                            tid_int = int(row[0])
                            ctx2 = state.get("entry_context") or {}
                            ctx2["tenant_id"] = tid_int
                            state["entry_context"] = ctx2
                            state["tenant_id"] = tid_int
                            tenant_id = tid_int
                            _log_step("tenant_context", source="journey_guard/db_fallback", tenant_id=tid_int)
            except Exception:
                pass
        # Short-term UI auto-enqueue path: emit custom event for the UI to call FastAPI with user cookies
        ui_enqueue = str(os.getenv("UI_ENQUEUE_JOBS", "")).strip().lower() in {"1", "true", "yes", "on"}
        if ui_enqueue and _LGCommand is not None and tenant_id:
            evt = {
                "type": "queue_job",
                "tenant_id": int(tenant_id),
                "payload": {"kind": "icp_discovery_enrich"},
            }
            msg = "Queuing background discovery and enrichment for your ICP…"
            profile["outstanding_prompts"] = [msg]
            _append_message(state, "assistant", msg)
            _set_status(state, "journey_guard", msg)
            _log_step("enqueue_ui", tenant_id=int(tenant_id), event_type="queue_job")
            return _LGCommand(update={}, custom=evt)  # type: ignore

        # Resolve notify email using policy similar to API endpoint
        def _resolve_email() -> Optional[str]:
            try:
                ctx2 = state.get("entry_context") or {}
                em = (ctx2.get("notify_email") or ctx2.get("user_email"))
                if em and isinstance(em, str) and "@" in em:
                    return em.strip()
            except Exception:
                pass
            try:
                if EMAIL_DEV_ACCEPT_TENANT_USER_ID_AS_EMAIL and tenant_id:
                    with get_conn() as conn, conn.cursor() as cur:
                        cur.execute("SELECT user_id FROM tenant_users WHERE tenant_id=%s LIMIT 5", (int(tenant_id),))
                        for r in cur.fetchall() or []:
                            u = r[0] if r and r[0] is not None else None
                            if u and "@" in str(u):
                                return str(u).strip()
            except Exception:
                pass
            try:
                if DEFAULT_NOTIFY_EMAIL and "@" in str(DEFAULT_NOTIFY_EMAIL):
                    return str(DEFAULT_NOTIFY_EMAIL).strip()
            except Exception:
                pass
            return None

        notify_email = _resolve_email()
        job_id = None
        try:
            if _enqueue_unified is not None and tenant_id:
                res = _enqueue_unified(int(tenant_id), notify_email=notify_email)
                job_id = (res or {}).get("job_id")
                if not job_id:
                    _log_step("enqueue_bg", ok=False, reason="no_job_id", tenant_id=tenant_id, response=res)
        except Exception:
            job_id = None
            _log_step("enqueue_bg", ok=False, reason="exception", tenant_id=tenant_id)
        if tenant_id and job_id:
            msg = (
                f"Thanks — I’ve queued background discovery and enrichment for your ICP. "
                f"I’ll email the results to you at {notify_email or 'your address on file'} when it finishes. "
                f"Job ID: {job_id}."
            )
            profile["enrichment_confirmed"] = True
        elif not tenant_id:
            msg = (
                "I couldn’t resolve your tenant session, so I can’t queue the background job yet. "
                "Please sign in via the app and try again."
            )
        else:
            msg = (
                f"I attempted to queue the background discovery and enrichment job but couldn’t confirm the job id. "
                f"Please try again or contact support."
            )
        profile["outstanding_prompts"] = [msg]
        _append_message(state, "assistant", msg)
        _set_status(state, "journey_guard", msg)
        return state

    # If the last input is a question, defer to progress_report Q&A and do not append prompts here.
    is_question_turn = (last_intent == "question") or (_heuristic_intent(question or "") == "question")
    is_update_turn = (last_intent == "update") or (_heuristic_intent(question or "") == "update")

    if is_question_turn or is_update_turn:
        # Do not mutate outstanding_prompts here; progress_report will answer.
        state["journey_ready"] = False
        _set_status(state, "journey_guard", "question_or_update_turn")
        _log_step("journey_guard", company_ready=company_ready, icp_ready=icp_ready, discovery_ready=discovery_ready, enrichment_ready=enrichment_ready, intent=last_intent, explanation_provided=False, needs_website=needs_website, question_turn=is_question_turn, update_turn=is_update_turn)
        return state

    if journey_ready:
        profile["outstanding_prompts"] = []
        profile["awaiting_discovery_confirmation"] = False
        msg = "Prerequisites satisfied. Proceeding to normalization."
    else:
        explanation = None  # generic explanations disabled; use unified Q&A in progress_report
        prompt_parts: List[str] = []
        # never inject generic explanations here; prompts only for gating needs

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
                _cache_discovery_details(state, discovery_state, candidate_details)
                if candidate_details and not discovery_state.get("staging_persisted"):
                    _persist_discovery_candidates(state, candidate_details)
                    discovery_state["staging_persisted"] = True
                prompt_parts.append(
                    f"I found {len(candidates)} ICP candidate websites and queued them for background enrichment."
                )
            else:
                prompt_parts.append("I couldn't find solid ICP candidates. Say 'retry discovery' to search again or adjust your ICP details.")
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
        explanation_provided=False,
        needs_website=needs_website,
    )
    _set_status(state, "journey_guard", msg)
    return state


# ---------------------------------------------------------------------------
# Normalization / discovery
# ---------------------------------------------------------------------------


async def normalize(state: OrchestrationState) -> OrchestrationState:
    result = {"processed_rows": 0, "errors": [], "last_run_at": datetime.now(timezone.utc).isoformat()}
    if _get_run_mode(state) == "chat_top10":
        _log_step("normalize", skipped=True, reason="chat_top10")
        _set_status(state, "normalize", "Skipped normalization for chat flow")
        state["normalize"] = result
        return state
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
    run_mode = _get_run_mode(state)
    if run_mode == "chat_top10" and (discovery.get("top10_ids") or discovery.get("top10_details")):
        _log_step("refresh_icp", skipped=True, reason="chat_top10_cached")
        _set_status(state, "refresh_icp", "Using confirmed Top-10 candidates")
        state["discovery"] = discovery
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
    profile = state.get("profile_state") or {}
    run_mode = _get_run_mode(state)
    has_cached = bool(discovery.get("top10_details") or discovery.get("web_candidate_details"))
    wants_retry = bool(profile.get("discovery_retry_requested"))
    if has_cached and not wants_retry:
        reason = "chat_top10_cached" if run_mode == "chat_top10" else "cached_discovery"
        _log_step("decide_strategy", action="use_cached", reason=reason)
        discovery["strategy"] = "use_cached"
        state["discovery"] = discovery
        message = "Reusing confirmed Top-10 candidates" if run_mode == "chat_top10" else "Reusing cached discovery candidates"
        _set_status(state, "decide_strategy", message)
        return state
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
    strategy = discovery.get("strategy")
    if strategy == "use_cached" and (discovery.get("top10_details") or discovery.get("web_candidate_details")):
        _log_step("ssic_fallback", skipped=True, reason="cached_discovery")
        _set_status(state, "ssic_fallback", "SSIC fallback skipped (cached discovery)")
        state["discovery"] = discovery
        return state
    if _get_run_mode(state) == "chat_top10" and (discovery.get("top10_ids") or discovery.get("top10_details")):
        _log_step("ssic_fallback", skipped=True, reason="chat_top10_cached")
        _set_status(state, "ssic_fallback", "SSIC fallback skipped for confirmed Top-10 run")
        state["discovery"] = discovery
        return state
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
    profile_state = state.get("profile_state") or {}
    icp_profile = profile_state.get("icp_profile") or {}
    items: List[Dict[str, Any]] = []
    run_mode = _get_run_mode(state)
    discovery = state.get("discovery") or {}
    cached_details = discovery.get("web_candidate_details") or discovery.get("top10_details") or []
    strategy = discovery.get("strategy")
    wants_retry = bool(profile_state.get("discovery_retry_requested"))
    if strategy == "use_cached" and cached_details and not wants_retry:
        _cache_discovery_details(state, discovery, cached_details)
        state["top10"] = {"items": cached_details, "generated_at": datetime.now(timezone.utc).isoformat()}
        state["discovery"] = discovery
        profile_state["discovery_retry_requested"] = False
        state["profile_state"] = profile_state
        reuse_reason = "chat_top10_cached" if run_mode == "chat_top10" else "cached_discovery"
        _log_step("plan_top10", planned=len(cached_details), reused=True, reason=reuse_reason)
        _set_status(state, "plan_top10", "Reusing cached discovery candidates")
        return state
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
    if items:
        _cache_discovery_details(state, discovery, items)
    state["discovery"] = discovery
    _log_step("plan_top10", planned=len(items))
    _set_status(state, "plan_top10", msg)
    return state


# ---------------------------------------------------------------------------
# Enrichment / scoring / export
# ---------------------------------------------------------------------------


async def enrich_batch(state: OrchestrationState) -> OrchestrationState:
    discovery = state.get("discovery") or {}
    tenant_id = _get_tenant_id(state)
    candidate_ids = discovery.get("top10_ids") or discovery.get("candidate_ids") or []
    unique_ids: List[int] = []
    seen_ids: set[int] = set()
    for raw in candidate_ids:
        if not isinstance(raw, (int, str)) or not str(raw).strip():
            continue
        try:
            cid = int(raw)
        except Exception:
            continue
        if cid in seen_ids:
            continue
        seen_ids.add(cid)
        unique_ids.append(cid)
    domain_lookup = _candidate_domain_lookup(discovery)
    results: List[Dict[str, Any]] = []
    if not unique_ids:
        msg = "No candidates to enrich"
        state["enrichment_results"] = results
        _log_step("enrich_batch", attempts=0, offline=OFFLINE, had_candidates=False)
        _set_status(state, "enrich_batch", msg)
        return state
    if OFFLINE:
        for cid in candidate_ids:
            results.append({"company_id": int(cid), "completed": True, "error": None, "source": "offline"})
        msg = f"[offline] Enrichment simulated for {len(results)} companies"
        state["enrichment_results"] = results
        _log_step("enrich_batch", attempts=len(results), offline=True, had_candidates=True)
        _set_status(state, "enrich_batch", msg)
        return state

    for cid in unique_ids:
        domain = domain_lookup.get(int(cid))
        success = False
        error: Optional[str] = None
        source = "mcp"
        if domain:
            success, error = await _attempt_jina_enrichment(tenant_id, int(cid), domain)
        else:
            error = "missing_domain"
        if not success and enrich_company_with_tavily is not None:
            try:
                await enrich_company_with_tavily(int(cid), search_policy="require_existing")
                success = True
                error = None
                source = "tavily"
            except Exception as exc:  # pragma: no cover - network variability
                logger.warning("enrich_company_with_tavily failed for %s: %s", cid, exc)
                error = str(exc)
        results.append({"company_id": int(cid), "completed": success, "error": error, "source": source})

    state["enrichment_results"] = results
    completed = len([r for r in results if r.get("completed")])
    msg = (
        f"Enrichment attempted for {len(results)} companies; {completed} succeeded."
        if results
        else "No candidates to enrich"
    )
    _log_step(
        "enrich_batch",
        attempts=len(results),
        succeeded=completed,
        used_fallback=any(r.get("source") == "tavily" for r in results),
        offline=OFFLINE,
        had_candidates=True,
    )
    _set_status(state, "enrich_batch", msg)
    return state


async def score_leads(state: OrchestrationState) -> OrchestrationState:
    discovery = state.get("discovery") or {}
    candidate_ids = discovery.get("top10_ids") or discovery.get("candidate_ids") or []
    candidate_ids = [int(cid) for cid in candidate_ids if isinstance(cid, (int, str)) and str(cid).strip()]
    if OFFLINE:
        scores = [{"company_id": cid, "score": 0.8, "reason": "offline stub"} for cid in candidate_ids]
        msg = f"[offline] Scored {len(scores)} companies"
    else:
        try:
            scoring_state = await lead_scoring_agent.ainvoke(
                {"candidate_ids": candidate_ids, "lead_features": [], "lead_scores": [], "icp_payload": state.get("icp_payload", {})}
            )
            scores = scoring_state.get("lead_scores") or []
            msg = f"Scored {len(scores)} companies"
        except Exception as exc:  # pragma: no cover
            logger.warning("lead_scoring_agent failed: %s", exc)
            scores = []
            msg = "Lead scoring failed"
    state["scoring"] = {"scores": scores, "last_run_at": datetime.now(timezone.utc).isoformat()}
    if scores:
        email_map = _lookup_primary_emails([int(s.get("company_id")) for s in scores if s.get("company_id") is not None])
        table = _format_lead_score_table(scores, discovery, email_map)
        if table:
            _append_message(state, "assistant", "Lead Score summary:\n" + table)
    _log_step("score_leads", scored=len(scores))
    _set_status(state, "score_leads", msg)
    return state


async def export_results(state: OrchestrationState) -> OrchestrationState:
    exports = state.get("exports") or {
        "odoo_exported": False,
        "job_ids": [],
    }
    exports["last_run_at"] = datetime.now(timezone.utc).isoformat()
    tenant_id = (state.get("entry_context") or {}).get("tenant_id")
    msg_extra = "; background discovery/enrichment already queued"

    # Trigger Odoo export for completed Top-10 enrichment when tenant context present
    enriched_ids = [
        int(r.get("company_id"))
        for r in state.get("enrichment_results") or []
        if r.get("completed") and isinstance(r.get("company_id"), (int, str)) and str(r.get("company_id")).strip()
    ]
    odoo_msg = "; Odoo export skipped"
    if tenant_id and enriched_ids:
        exported = await _export_top10_to_odoo(int(tenant_id), enriched_ids)
        exports["odoo_exported"] = exported
        odoo_msg = "; Odoo export triggered" if exported else "; Odoo export failed"
    elif tenant_id:
        exports["odoo_exported"] = False
    else:
        odoo_msg = "; Odoo tenant missing"

    state["exports"] = exports
    _log_step(
        "export",
        jobs=exports.get("job_ids"),
        msg_extra=f"{msg_extra}{odoo_msg}",
    )
    _set_status(state, "export", f"Export stage complete{msg_extra}{odoo_msg}")
    return state


async def progress_report(state: OrchestrationState) -> OrchestrationState:
    # Update/Q&A precedence: if the last message is an update or looks like a direct question (even without '?'),
    # attempt to perform the update or answer before replaying outstanding prompts.
    last_text = _latest_user_text(state)
    ctx = state.get("entry_context") or {}
    intent_norm = (ctx.get("intent") or _heuristic_intent(last_text)).strip().lower()
    is_question = intent_norm == "question"
    is_update = intent_norm == "update"
    if is_update:
        updated, message = _try_apply_update_command(state, last_text)
        if updated:
            _log_step("icp_update", message=message[:140])
            _set_status(state, "progress_report", message)
            # Clear prompts and acknowledge update
            try:
                profile = _ensure_profile_state(state)
                profile["outstanding_prompts"] = []
                state["profile_state"] = profile
            except Exception:
                pass
            if not state.get("suppress_output"):
                _append_message(state, "assistant", message)
            return state
    if is_question:
        # Answer from thread/persistence; fall back to LLM using a compact context snapshot
        answer = _smart_answer(state, last_text)
        if answer:
            _log_step("qa_answer", message=answer[:140])
            _set_status(state, "progress_report", answer)
            # Clear outstanding prompts since user's question took precedence
            try:
                profile = _ensure_profile_state(state)
                profile["outstanding_prompts"] = []
                state["profile_state"] = profile
            except Exception:
                pass
            if not state.get("suppress_output"):
                _append_message(state, "assistant", answer)
            return state

    summary = {
        "phase": state.get("status", {}).get("phase"),
        "candidates": len(state.get("discovery", {}).get("candidate_ids") or []),
        "confirmed_company": bool(state.get("profile_state", {}).get("company_profile_confirmed")),
        "confirmed_icp": bool(state.get("profile_state", {}).get("icp_profile_confirmed")),
        "web_candidates": len(state.get("discovery", {}).get("web_candidates") or []),
    }
    # Optional: if user asked for status, include latest background job info
    try:
        ctx = state.get("entry_context") or {}
        if (ctx.get("intent") or "").strip().lower() == "status":
            tid = _get_tenant_id(state)
            if tid:
                try:
                    with get_conn() as conn, conn.cursor() as cur:
                        cur.execute(
                            """
                            SELECT job_id, status, processed, total, created_at, started_at, ended_at
                              FROM background_jobs
                             WHERE tenant_id=%s AND job_type='icp_discovery_enrich'
                             ORDER BY created_at DESC
                             LIMIT 1
                            """,
                            (int(tid),),
                        )
                        row = cur.fetchone()
                    if row:
                        summary["latest_job"] = {
                            "job_id": row[0],
                            "status": row[1],
                            "processed": row[2],
                            "total": row[3],
                            "created_at": str(row[4]) if row[4] else None,
                            "started_at": str(row[5]) if row[5] else None,
                            "ended_at": str(row[6]) if row[6] else None,
                        }
                except Exception:
                    pass
    except Exception:
        pass
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
    # Avoid duplicating outputs when this run has no new user input
    if not state.get("suppress_output"):
        _append_message(state, "assistant", message)
    return state


def _answer_direct_question(state: OrchestrationState, text: str | None) -> str | None:
    """Answer simple, direct facts from thread or persistence.

    Currently supports company website/url queries even without a trailing question mark.
    """
    if not text:
        return None
    msg = text.strip().lower().replace("’", "'")
    # Normalize stray trailing punctuation/symbols
    while msg and msg[-1] in ">:;,. ":
        msg = msg[:-1]
    # Detect company url/website questions
    url_tokens = {"url", "website", "site", "domain"}
    trig = (msg.startswith("what is my ") or msg.startswith("what's my ") or msg.startswith("whats my ") or
            msg.startswith("what is our ") or msg.startswith("what's our ") or msg.startswith("whats our ") or
            "what is the" in msg or "what's the" in msg or "whats the" in msg)
    mentions_url = any(tok in msg for tok in url_tokens) and ("company" in msg or True)
    if trig and mentions_url:
        # Try thread state first
        profile = _ensure_profile_state(state)
        company = profile.get("company_profile") or {}
        website = (company.get("website") or "").strip()
        if website:
            return f"Your company website is {website}."
        # Fallback: read from persisted tenant_company_profiles (source_url)
        tid = _get_tenant_id(state)
        if tid is not None:
            try:
                rec = _fetch_company_profile_record(int(tid))
            except Exception:
                rec = None
            if rec:
                src = (rec.get("source_url") or "").strip() if isinstance(rec.get("source_url"), str) else ""
                if src:
                    return f"Your company website is {src}."
        return "I don’t have your website on file yet. Please share it (e.g., https://example.com)."
    return None


def _qa_context_snapshot(state: OrchestrationState) -> Dict[str, Any]:
    """Collect a compact snapshot for Q&A answers (thread + persistence + latest job)."""
    snap: Dict[str, Any] = {}
    profile = _ensure_profile_state(state)
    snap["company_profile"] = dict(profile.get("company_profile") or {})
    snap["icp_profile"] = dict(profile.get("icp_profile") or {})
    snap["customer_websites"] = list(profile.get("customer_websites") or [])
    snap["flags"] = {
        "company_confirmed": bool(profile.get("company_profile_confirmed")),
        "icp_confirmed": bool(profile.get("icp_profile_confirmed")),
        "icp_generated": bool(profile.get("icp_profile_generated")),
    }
    # Discovery snapshot (counts + preview)
    disc = state.get("discovery") or {}
    try:
        web_cands = list(disc.get("web_candidates") or [])
    except Exception:
        web_cands = []
    try:
        top10 = list(disc.get("top10_domains") or [])
    except Exception:
        top10 = []
    try:
        next40 = list(disc.get("next40_domains") or [])
    except Exception:
        next40 = []
    try:
        cand_ids = list(disc.get("candidate_ids") or [])
    except Exception:
        cand_ids = []
    snap["discovery"] = {
        "web_candidate_count": len(web_cands),
        "top10_count": len(top10),
        "next40_count": len(next40),
        "candidate_ids_count": len(cand_ids),
        "web_candidates_preview": web_cands[:10],
        "top10_preview": top10[:10],
        "next40_preview": next40[:10],
    }

    # Scoring snapshot (bucket counts + preview)
    scoring = state.get("scoring") or {}
    scores = scoring.get("scores") or []
    buckets = {"A": 0, "B": 0, "C": 0}
    try:
        for row in scores:
            b = str(row.get("bucket") or "C").upper()
            if b in buckets:
                buckets[b] += 1
    except Exception:
        pass
    try:
        preview_scores = [
            {"company_id": r.get("company_id"), "score": r.get("score"), "bucket": r.get("bucket")}
            for r in (scores or [])[:5]
            if isinstance(r, dict)
        ]
    except Exception:
        preview_scores = []
    snap["scoring"] = {
        "total": len(scores or []),
        "buckets": buckets,
        "preview": preview_scores,
    }

    # Include latest background job (if any)
    tid = _get_tenant_id(state)
    if tid is not None:
        try:
            with get_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT job_id, status, processed, total
                      FROM background_jobs
                     WHERE tenant_id=%s AND job_type='icp_discovery_enrich'
                     ORDER BY created_at DESC
                     LIMIT 1
                    """,
                    (int(tid),),
                )
                row = cur.fetchone()
                if row:
                    snap["latest_job"] = {
                        "job_id": row[0],
                        "status": row[1],
                        "processed": row[2],
                        "total": row[3],
                    }
        except Exception:
            pass
        # Also include persisted website as a fallback
        try:
            rec = _fetch_company_profile_record(int(tid))
        except Exception:
            rec = None
        if rec and isinstance(rec.get("source_url"), str) and rec.get("source_url").strip():
            snap.setdefault("company_profile", {}).setdefault("website", rec.get("source_url").strip())
    return snap


def _smart_answer(state: OrchestrationState, text: str | None) -> str | None:
    """Answer general user questions using thread + persisted context (and LLM fallback).

    - Tries direct fact lookups for known fields (site, name, summaries, lists).
    - Falls back to a concise LLM answer grounded in a compact context snapshot.
    """
    if not text:
        return None
    raw = text.strip()
    msg = raw.lower().replace("’", "'")

    # 1) Deterministic facts
    prof = _ensure_profile_state(state)
    company = dict(prof.get("company_profile") or {})
    icp = dict(prof.get("icp_profile") or {})
    customers = list(prof.get("customer_websites") or [])

    def has_any(words: Iterable[str]) -> bool:
        return any(w in msg for w in words)

    # Company website / URL / domain
    if has_any([" company url", " company website", " company site", " company domain"]) or (
        msg.startswith("what is my ") and has_any(["url", "website", "site", "domain"])):
        ans = _answer_direct_question(state, raw)
        if ans:
            return ans

    # Company name
    if has_any(["company name", "what is my company name", "what's my company name"]):
        name = (company.get("name") or "").strip()
        if name:
            return f"Your company name is {name}."

    # Company summary / what we do
    if has_any(["company summary", "what do we do", "what does my company do", "what's our summary"]):
        s = (company.get("summary") or "").strip()
        if s:
            return s

    # Company fields
    def _fmt_list(key: str, label: str, limit: int = 6) -> Optional[str]:
        vals = company.get(key) if isinstance(company, dict) else None
        if isinstance(vals, list) and vals:
            items = ", ".join([str(v) for v in vals[:limit]])
            return f"{label}: {items}."
        return None

    if has_any(["industries"]):
        t = _fmt_list("industries", "Industries")
        if t:
            return t
    if has_any(["offerings", "products", "services"]):
        t = _fmt_list("offerings", "Offerings")
        if t:
            return t
    if has_any(["ideal customers", "ideal buyer", "buyers"]):
        t = _fmt_list("ideal_customers", "Ideal customers")
        if t:
            return t
    if has_any(["proof points", "proof", "evidence"]):
        t = _fmt_list("proof_points", "Proof points")
        if t:
            return t

    # ICP questions
    def _icp_list(key: str, label: str, limit: int = 6) -> Optional[str]:
        vals = icp.get(key) if isinstance(icp, dict) else None
        if isinstance(vals, list) and vals:
            items = ", ".join([str(v) for v in vals[:limit]])
            return f"{label}: {items}."
        return None

    if has_any(["icp summary", "what is my icp", "what's my icp", "ideal customer profile"]):
        s = (icp.get("summary") or "").strip()
        if s:
            return s
    if has_any(["icp industries", "target industries"]):
        t = _icp_list("industries", "ICP industries")
        if t:
            return t
    if has_any(["company sizes", "size bands", "target sizes"]):
        t = _icp_list("company_sizes", "Company sizes") or _icp_list("size_bands", "Company sizes")
        if t:
            return t
    if has_any(["regions", "geos", "target regions", "target geos"]):
        t = _icp_list("regions", "Regions") or _icp_list("geos", "Regions")
        if t:
            return t
    if has_any(["pains", "pain points", "key pains"]):
        t = _icp_list("pains", "Key pains")
        if t:
            return t
    if has_any(["triggers", "buying triggers"]):
        t = _icp_list("buying_triggers", "Buying triggers") or _icp_list("triggers", "Buying triggers")
        if t:
            return t
    if has_any(["personas", "titles", "roles"]):
        t = _icp_list("persona_titles", "Personas") or _icp_list("buyer_titles", "Personas")
        if t:
            return t

    # Customer sites
    if has_any(["customer websites", "customer urls", "seed urls", "seeds"]):
        if customers:
            preview = ", ".join(customers[:5])
            return f"You’ve shared {len(customers)} customer websites. Example: {preview}."

    # Candidates / Top‑10 / Next‑40
    if has_any(["candidates", "domains", "top 10", "top-10", "next 40", "next-40", "web candidates", "lookalikes"]):
        snap = _qa_context_snapshot(state)
        disc = snap.get("discovery") or {}
        top10 = disc.get("top10_preview") or []
        web = disc.get("web_candidates_preview") or []
        next40 = disc.get("next40_preview") or []
        total = int(disc.get("web_candidate_count") or 0) or int(disc.get("candidate_ids_count") or 0)
        if top10:
            sample = ", ".join(top10[:5])
            return f"Top‑10 planned; sample: {sample}. Total planned candidates: {total or len(top10)}."
        if web:
            sample = ", ".join(web[:5])
            return f"Planned candidates: {len(web)} (preview). Sample: {sample}."
        if next40:
            sample = ", ".join(next40[:5])
            return f"Next‑40 queued; sample: {sample}."
        return "I don’t have candidate domains yet. Once discovery runs, I’ll share a preview."

    # Lead scores / buckets
    if has_any(["scores", "lead scores", "buckets", "grade", "how many a", "how many b", "how many c"]):
        snap = _qa_context_snapshot(state)
        sc = snap.get("scoring") or {}
        total = int(sc.get("total") or 0)
        buckets = sc.get("buckets") or {}
        if total:
            a = buckets.get("A", 0)
            b = buckets.get("B", 0)
            c = buckets.get("C", 0)
            return f"Scored {total} companies. Buckets: A={a}, B={b}, C={c}."
        return "I don’t have scores yet — they’ll appear after enrichment runs."

    # Job status / what did you queue
    if has_any(["job", "status", "queue", "queued", "running", "progress", "what did you queue", "what did you enqueue", "what was queued"]):
        snap = _qa_context_snapshot(state)
        job = snap.get("latest_job") or {}
        if job:
            jid = job.get("job_id")
            st = job.get("status")
            p = job.get("processed")
            t = job.get("total")
            return f"Latest job {jid} is {st}. Progress: {p}/{t}."

    # General lead‑generation guidance / how-to use the system
    if has_any(["lead generation", "generate leads", "prospecting", "mql", "sql", "pipeline", "enrichment", "discovery", "icp best", "how to find leads"]):
        return (
            "Lead generation basics: 1) Define ICP (industries, size, regions, pains, triggers), "
            "2) Build lookalike list (discovery) from ICP, 3) Enrich companies (web + signals) and score (A/B/C), "
            "4) Prioritize outreach with clear offers and fast feedback loops. In this system: share your website, "
            "confirm the profile, add 5 customer sites → I synthesize ICP → I plan candidates (Top‑10/Next‑40) → "
            "enrichment + scoring runs in the background → export via /export/latest_scores.csv."
        )

    if has_any(["help", "how to use", "how this works", "guide", "instructions", "what can you do", "what does this do"]):
        return (
            "How to use: 1) Share your business website; confirm your company profile. "
            "2) Share 5 customer websites; I’ll generate your ICP. 3) Confirm ICP; I queue discovery + enrichment in the background. "
            "4) Check progress by asking ‘job status’, and download scores at /export/latest_scores.csv. "
            "Tip: You can ask me anything (e.g., ‘what’s our ICP industries?’ or ‘show Top‑10 candidates’)."
        )

    # 2) LLM fallback using compact context snapshot
    snapshot = _qa_context_snapshot(state)
    prompt = f"""
You are a precise assistant. Answer the user's question using ONLY this context.
If unknown, say you don't know and ask for the missing info.

Question: {raw!r}
Context: {snapshot!r}
Return JSON {{"answer": "<one or two sentences>"}}.
"""
    fallback = {"answer": "I don't have enough info to answer that yet."}
    result = call_llm_json(prompt, fallback)
    ans = (result or fallback).get("answer") or fallback["answer"]
    return ans.strip() if isinstance(ans, str) else None


# -----------------------------
# Update command parser & applier
# -----------------------------

def _canonical_icp_key(section: str) -> Optional[str]:
    key = (section or "").strip().lower()
    mapping = {
        "industries": "industries",
        "industry": "industries",
        "regions": "regions",
        "geos": "regions",
        "region": "regions",
        "company sizes": "company_sizes",
        "company size": "company_sizes",
        "sizes": "company_sizes",
        "size bands": "company_sizes",
        "pains": "pains",
        "pain points": "pains",
        "triggers": "buying_triggers",
        "buying triggers": "buying_triggers",
        "personas": "persona_titles",
        "titles": "persona_titles",
        "buyer titles": "persona_titles",
        "proof points": "proof_points",
        "proofs": "proof_points",
        "seed urls": "seed_urls",
        "seeds": "seed_urls",
    }
    return mapping.get(key)


def _parse_update_command(text: str | None) -> Optional[Dict[str, Any]]:
    if not isinstance(text, str) or not text.strip():
        return None
    s = text.strip()
    low = s.lower()
    # Patterns: remove X, Y in <section>; remove X from <section>
    import re
    m = re.match(r"^(remove|add|set|change|edit|delete)\s+(.*)$", low)
    if not m:
        return None
    op = m.group(1)
    rest = s[len(op):].strip()
    # Try 'in <section>' or 'from <section>' or 'to <section>' or 'set <section> to ...'
    # Normalize separators
    candidates: list[str] = []
    section = None
    if op == "set" or low.startswith("set "):
        # set <section> to ITEMS
        sm = re.match(r"^\s*([a-zA-Z /&_-]+?)\s+to\s+(.+)$", rest, re.IGNORECASE)
        if sm:
            section_raw = sm.group(1)
            # Decide target by section name and presence of 'company'
            prefer_company = ("company" in low) or ("company profile" in low)
            c_key = _canonical_company_key(section_raw)
            i_key = _canonical_icp_key(section_raw)
            if prefer_company and c_key:
                section = c_key
                target = "company"
            elif i_key and not c_key:
                section = i_key
                target = "icp"
            elif c_key:
                section = c_key
                target = "company"
            else:
                section = i_key
                target = "icp"
            raw_items = sm.group(2).strip()
            candidates = [i.strip() for i in re.split(r",|/|;|\n", raw_items) if i.strip()]
            return {"op": "set", "section": section, "items": candidates, "target": target}
    # remove/add/edit/delete/change patterns with 'in/from/to'
    pm = re.match(r"^(.+?)\s+(in|from|to)\s+([a-zA-Z /&_-]+)$", rest, re.IGNORECASE)
    if pm:
        item_part = pm.group(1)
        preposition = pm.group(2).lower()
        section_raw = pm.group(3)
        prefer_company = ("company" in low) or ("company profile" in low)
        c_key = _canonical_company_key(section_raw)
        i_key = _canonical_icp_key(section_raw)
        if prefer_company and c_key:
            section = c_key
            target = "company"
        elif i_key and not c_key:
            section = i_key
            target = "icp"
        elif c_key:
            section = c_key
            target = "company"
        else:
            section = i_key
            target = "icp"
        raw_items = item_part
        # Strip parentheses and extra spaces
        raw_items = re.sub(r"[()]+", " ", raw_items)
        candidates = [i.strip() for i in re.split(r",|/|;|\n", raw_items) if i.strip()]
        normalized_op = "remove" if op in {"remove", "delete"} else ("add" if op == "add" or preposition == "to" else "change")
        return {"op": normalized_op, "section": section, "items": candidates, "target": target}
    # Fallback: single-section direct (e.g., remove Malaysia in Regions without clear tokens)
    # Not matched → ignore
    return None


def _apply_icp_update(state: OrchestrationState, spec: Dict[str, Any]) -> Tuple[bool, str]:
    section = spec.get("section")
    op = spec.get("op")
    items = spec.get("items") or []
    if not section or not isinstance(items, list):
        return False, "I couldn’t parse which section to update."
    profile = _ensure_profile_state(state)
    icp = dict(profile.get("icp_profile") or {})
    # Map canonical section to both icp keys we keep
    def _get_lists_for_section(sec: str) -> list[Tuple[dict, str]]:
        # Return (dict, key) pairs to update in place
        if sec == "company_sizes":
            return [(icp, "company_sizes"), (icp, "size_bands")]
        if sec == "regions":
            return [(icp, "regions"), (icp, "geos")]
        if sec == "buying_triggers":
            return [(icp, "buying_triggers"), (icp, "triggers")]
        if sec == "persona_titles":
            return [(icp, "persona_titles"), (icp, "buyer_titles")]
        return [(icp, sec)]

    targets = _get_lists_for_section(section)
    # Normalize helpers
    def _norm_list(v):
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        return []
    def _ci_match(a: str, b: str) -> bool:
        return a.strip().lower() == b.strip().lower()
    def _ci_contains(longer: str, shorter: str) -> bool:
        return shorter.strip().lower() in longer.strip().lower()

    summary_changes: list[str] = []
    for obj, key in targets:
        cur = _norm_list(obj.get(key) or [])
        if op == "set":
            obj[key] = items
            summary_changes.append(f"set {key} to {', '.join(items) if items else 'empty'}")
        elif op == "add":
            for it in items:
                if not any(_ci_match(x, it) for x in cur):
                    cur.append(it)
            obj[key] = cur
            summary_changes.append(f"added {', '.join(items)} to {key}")
        elif op in {"remove", "delete"}:
            newlist = []
            removed_local: list[str] = []
            for x in cur:
                if any(_ci_match(x, it) or _ci_contains(x, it) for it in items):
                    removed_local.append(x)
                else:
                    newlist.append(x)
            obj[key] = newlist
            if removed_local:
                summary_changes.append(f"removed {', '.join(removed_local)} from {key}")
        else:
            # change/edit → treat like set
            obj[key] = items
            summary_changes.append(f"set {key} to {', '.join(items)}")

    profile["icp_profile"] = icp
    # Keep ICP confirmed and persist
    profile["icp_profile_confirmed"] = True
    state["profile_state"] = profile
    try:
        seeds = list(profile.get("customer_websites") or [])
        _persist_icp_profile_state(state, icp, seeds[:5])
    except Exception:
        pass
    msg = "; ".join(summary_changes) if summary_changes else f"updated {section}"
    return True, f"Updated ICP: {msg}."


def _try_apply_update_command(state: OrchestrationState, text: str | None) -> Tuple[bool, str]:
    spec = _parse_update_command(text)
    if not spec:
        return False, ""
    if spec.get("target") == "company" or not spec.get("target") and spec.get("section") in {"industries", "offerings", "ideal_customers", "proof_points", "summary", "name", "website"}:
        ok, msg = _apply_company_update(state, spec)
        return ok, msg
    ok, msg = _apply_icp_update(state, spec)
    return ok, msg


def _canonical_company_key(section: str) -> Optional[str]:
    key = (section or "").strip().lower()
    mapping = {
        "industries": "industries",
        "industry": "industries",
        "offerings": "offerings",
        "products": "offerings",
        "services": "offerings",
        "ideal customers": "ideal_customers",
        "ideal customer": "ideal_customers",
        "customers": "ideal_customers",
        "proof points": "proof_points",
        "proof": "proof_points",
        "summary": "summary",
        "name": "name",
        "website": "website",
        "url": "website",
        "site": "website",
        "domain": "website",
    }
    return mapping.get(key)


def _apply_company_update(state: OrchestrationState, spec: Dict[str, Any]) -> Tuple[bool, str]:
    section = spec.get("section")
    op = spec.get("op")
    items = spec.get("items") or []
    if not section:
        return False, "I couldn’t parse which company section to update."
    profile = _ensure_profile_state(state)
    company = dict(profile.get("company_profile") or {})

    list_fields = {"industries", "offerings", "ideal_customers", "proof_points"}
    string_fields = {"summary", "name", "website"}

    def _norm_list(v):
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        return []

    changes: list[str] = []
    if section in list_fields:
        cur = _norm_list(company.get(section) or [])
        if op == "set" or op == "change":
            company[section] = items
            changes.append(f"set {section} to {', '.join(items) if items else 'empty'}")
        elif op == "add":
            for it in items:
                if it and it.lower() not in [x.lower() for x in cur]:
                    cur.append(it)
            company[section] = cur
            changes.append(f"added {', '.join(items)} to {section}")
        elif op in {"remove", "delete"}:
            lowered = [x.lower() for x in items]
            newlist = [x for x in cur if all(li not in x.lower() for li in lowered)]
            removed = [x for x in cur if x not in newlist]
            company[section] = newlist
            if removed:
                changes.append(f"removed {', '.join(removed)} from {section}")
        else:
            company[section] = items
            changes.append(f"set {section} to {', '.join(items)}")
    elif section in string_fields:
        # For string fields, use first item or join items
        value = ", ".join(items) if items else ""
        company[section] = value
        if section == "website" and value:
            state["site_profile_bootstrap_url"] = value
        changes.append(f"set {section} to {value if value else 'empty'}")
    else:
        return False, "I couldn’t map that company section."

    # Persist and confirm
    profile["company_profile"] = company
    profile["company_profile_confirmed"] = True
    state["profile_state"] = profile
    try:
        _persist_company_profile_sync(state, company, confirmed=True)
    except Exception:
        pass
    msg = "; ".join(changes) if changes else f"updated {section}"
    return True, f"Updated company profile: {msg}."


async def finalize(state: OrchestrationState) -> OrchestrationState:
    msg = state.get("status", {}).get("message") or "Run complete"
    if not state.get("suppress_output"):
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
def _coerce_url_str(value: Any) -> str:
    try:
        if value is None:
            return ""
        if isinstance(value, list):
            for v in value:
                if isinstance(v, str) and v.strip():
                    return v.strip()
            return ""
        if isinstance(value, dict):
            for key in ("url", "website", "href"):
                v = value.get(key)
                if isinstance(v, str) and v.strip():
                    return v.strip()
            return ""
        if isinstance(value, str):
            return value.strip()
        s = str(value)
        return s.strip()
    except Exception:
        return ""


def _normalize_url(url: Any) -> str:
    url = _coerce_url_str(url)
    try:
        if not url:
            return ""
        parsed = urlparse(url if url.startswith("http") else ("https://" + url))
        scheme = parsed.scheme or "https"
        netloc = (parsed.netloc or parsed.path).lower()
        path = parsed.path if parsed.netloc else ""
        base = f"{scheme}://{netloc}{path}"
        return base.rstrip("/")
    except Exception:
        try:
            return url.rstrip("/") if isinstance(url, str) else ""
        except Exception:
            return ""

try:
    from src.settings import (
        BG_DISCOVERY_AND_ENRICH,
        CHAT_DISCOVERY_ENABLED,
        DEFAULT_NOTIFY_EMAIL,
        EMAIL_DEV_ACCEPT_TENANT_USER_ID_AS_EMAIL,
    )
except Exception:  # pragma: no cover
    BG_DISCOVERY_AND_ENRICH = False  # type: ignore
    CHAT_DISCOVERY_ENABLED = False  # type: ignore
    DEFAULT_NOTIFY_EMAIL = None  # type: ignore
    EMAIL_DEV_ACCEPT_TENANT_USER_ID_AS_EMAIL = True  # type: ignore

try:
    from src.jobs import enqueue_icp_discovery_enrich as _enqueue_unified
except Exception:  # pragma: no cover
    _enqueue_unified = None  # type: ignore


def _extract_urls(text: str) -> List[str]:
    if not text:
        return []
    return [match.rstrip(".,)") for match in URL_RE.findall(text)]


def _collect_customer_sites(messages: List[Dict[str, Any]], company_url: Optional[str]) -> List[str]:
    normalized_company = _normalize_url(company_url)
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
