"""
Conversational agent utilities for lead generation Q&A and relevance checking.

Provides two primary helpers:
- answer_leadgen_question(question, context?): LLM-backed expert reply.
- check_answer_relevance(question, user_answer): Classifies if the answer is on-topic.

Also exposes a simple handle_user_turn() helper to orchestrate a turn-by-turn
interaction where the agent can ask a question, validate the user's response,
and politely ask to try again when the response is not relevant.

All functions degrade gracefully when LLM is not configured.
"""
from __future__ import annotations

from typing import Any, Dict, Optional
import json
import logging

from langchain_openai import ChatOpenAI  # type: ignore
from src.settings import LANGCHAIN_MODEL, TEMPERATURE
from langchain_core.prompts import ChatPromptTemplate  # type: ignore
from pydantic import BaseModel

log = logging.getLogger("agents.conversation")
if not log.handlers:
    h = logging.StreamHandler()
    fmt = logging.Formatter("[%(levelname)s] %(asctime)s %(name)s :: %(message)s", "%H:%M:%S")
    h.setFormatter(fmt)
    log.addHandler(h)
log.setLevel(logging.INFO)


class RelevanceJudgment(BaseModel):
    is_relevant: bool
    reason: str
    missing_elements: Optional[list[str]] = None


# A concise, embedded reference of what this agent can do and how it works.
# Q&A answers MUST rely only on this reference and any provided runtime context.
SYSTEM_REFERENCE = (
    "Agent: Pre‑SDR lead‑gen assistant under LangGraph Server. Router-controlled chat graph.\n"
    "Entry: app/lg_entry.py:make_graph → app/pre_sdr_graph.build_graph.\n"
    "Safety: No auto‑start or auto‑enrichment; explicit user commands required. Greetings and Q&A are terminal.\n"
    "Tone: Explain jargon inline (e.g., 'ICP (ideal customer profile)').\n\n"
    "User Commands:\n"
    "- start lead gen | find leads | discover leads | prospect leads: begin ICP flow.\n"
    "- confirm profile: acknowledge the captured company profile before progressing.\n"
    "- confirm: persist current ICP/intake and advance to candidates.\n"
    "- confirm micro-icp: approve the suggested micro‑ICP (focused ICP) list after providing anti-ICP notes.\n"
    "- accept micro-icp N: select a suggested micro‑ICP (often SSIC‑based) to unlock enrichment.\n"
    "- run enrichment: enrich current candidates now (small batch), schedule remainder nightly.\n\n"
    "ICP Intake (Finder on):\n"
    "1) Ask website URL.\n"
    "2) Ask 5–15 best customers as 'Company — website' (seeds).\n"
    "3) Ask industries, employee range, geographies, buying signals (skip allowed).\n"
    "4) Optionals: ACV, cycle length, price floor, champion titles, triggers.\n"
    "5) After synthesizing the company profile, show the summary back and wait for **confirm profile** or corrections.\n"
    "After confirm, system proposes micro‑ICPs with evidence, captures anti‑ICP (companies to avoid), "
    "and requires **confirm micro-icp** before selection.\n\n"
    "Candidates: From (a) recent upserts (sync head), (b) DB by ICP filters, and may preview counts.\n"
    "Micro‑ICPs: Suggested segments with evidence; user accepts with 'accept micro-icp N'.\n\n"
    "Enrichment (run enrichment):\n"
    "- Immediate batch capped by CHAT_ENRICH_LIMIT/RUN_NOW_LIMIT (~10).\n"
    "- Ensures company rows, enriches with Tavily, stores contact emails when found.\n"
    "- Tops up candidates using SSIC→ACRA where appropriate (e.g., Singapore dataset) or defaults to ICP-derived DB lookups.\n"
    "- Remainder queued via staging_upsert for nightly processing.\n"
    "- After enrichment, run lead scoring and render a table with Name, Domain, Industry, Employees, Score, Bucket, Rationale, Contact.\n"
    "- Best-effort Odoo export for enriched companies and high-scoring leads if tenant/DSN configured.\n\n"
    "Notes:\n"
    "- ACRA/SSIC used primarily for Singapore (nightly scale); global discovery uses web enrichment + DB filters.\n"
    "- 'start' resets transient state (candidates/results) and restarts ICP Q&A.\n"
    "- Router gates enrichment until micro‑ICP is accepted (when Finder is on) or user explicitly overrides with 'run enrichment'.\n"
)


def _llm_safely(model: str | None = None, temperature: float | None = None):
    try:
        _model = model or LANGCHAIN_MODEL
        # If a specific temperature is not provided, default to configured TEMPERATURE
        _temp = TEMPERATURE if temperature is None else temperature
        return ChatOpenAI(model=_model, temperature=_temp)
    except Exception as e:  # no API key or network
        log.info("LLM unavailable: %s", e)
        return None


def _is_in_scope(question: str) -> bool:
    """Liberal scope: treat most lead‑gen questions as in-scope, but allow graceful refusal.

    We only refuse clearly unrelated topics; the system prompt already constrains answers.
    """
    q = (question or "").strip().lower()
    if not q:
        return False
    # Obvious off-topic buckets (examples; not exhaustive)
    off = ["weather", "sports", "stock price", "recipe", "movie", "song", "travel visa"]
    return not any(t in q for t in off)


def answer_leadgen_question(question: str, context: Optional[Dict[str, Any]] = None) -> str:
    """Answer strictly from this system’s setup and lead‑gen workflow.

    Rules:
    - Closed-book: Use only the embedded SYSTEM_REFERENCE and provided context.
    - No external facts, browsing, or speculative advice.
    - If out-of-scope, respond that you can only discuss this agent and its workflow.
    - If LLM is unavailable, return a concise, static response built from SYSTEM_REFERENCE.
    """
    question = (question or "").strip()
    if not question:
        return "I can answer questions about this agent’s lead‑gen setup and workflow."

    # Guard: obvious out-of-scope → brief refusal
    if not _is_in_scope(question):
        return "I can only discuss this agent and its lead‑gen workflow."

    llm = _llm_safely()
    if not llm:
        # Static, closed-book fallback using the embedded reference
        return (
            "System-only mode: This agent runs under LangGraph Server and supports: "
            "start lead gen → ICP intake → confirm → candidates → accept micro‑ICP → run enrichment → scoring. "
            "It never auto‑starts or auto‑enriches. Ask about these steps or commands."
        )

    sys = (
        "You answer ONLY from the provided Reference and Context. Do not use external/web knowledge.\n"
        "If asked beyond what the Reference/Context covers, say what’s available in this system and what isn’t.\n"
        "Be specific, concise, and recommend the next command the user can run (e.g., 'start lead gen', 'confirm profile', "
        "'confirm micro-icp', 'accept micro-icp N', 'run enrichment').\n\n"
        f"Reference:\n{SYSTEM_REFERENCE}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", sys),
        (
            "human",
            "Question: {q}\nContext (state): {ctx}\n\n"
            "Answer strictly from the Reference and Context."
        ),
    ])
    # Keep context compact; include only high-signal fields
    # Pass through a richer, but compact, runtime context
    ctx = {
        "icp": (context or {}).get("icp") if isinstance((context or {}).get("icp"), dict) else {},
        "counts": {
            "candidates": (context or {}).get("candidates_count"),
            "results": (context or {}).get("results_count"),
            "scored": (context or {}).get("scored_count"),
        },
        "flags": {
            "ENABLE_ICP_INTAKE": (context or {}).get("ENABLE_ICP_INTAKE"),
            "ENABLE_AGENT_DISCOVERY": (context or {}).get("ENABLE_AGENT_DISCOVERY"),
            "ENABLE_ACRA_IN_CHAT": (context or {}).get("ENABLE_ACRA_IN_CHAT"),
        },
        "gating": {
            "finder_suggestions_done": (context or {}).get("finder_suggestions_done"),
            "micro_icp_selected": (context or {}).get("micro_icp_selected"),
        },
        "metrics": {
            "icp_match_total": (context or {}).get("icp_match_total"),
            "acra_total_suggested": (context or {}).get("acra_total_suggested"),
            "enrich_now_planned": (context or {}).get("enrich_now_planned"),
            "RUN_NOW_LIMIT": (context or {}).get("RUN_NOW_LIMIT"),
        },
        "commands": [
            "start lead gen",
            "confirm profile",
            "confirm",
            "confirm micro-icp",
            "accept micro-icp N",
            "run enrichment",
        ],
    }
    msgs = prompt.format_messages(q=question, ctx=json.dumps(ctx, ensure_ascii=False))
    try:
        return (llm.invoke(msgs).content or "").strip()
    except Exception as e:
        log.info("answer_llm_fail: %s", e)
        return (
            "I can only discuss this agent’s setup and workflow. Try asking about: start/confirm, ICP intake, candidates, micro‑ICPs, or enrichment."
        )


def check_answer_relevance(question: str, user_answer: str) -> RelevanceJudgment:
    """Judge whether a user's answer is relevant to the agent's question.

    Returns a RelevanceJudgment with a boolean and brief reason. If not relevant,
    includes missing_elements summarizing what should be addressed.
    """
    q = (question or "").strip()
    a = (user_answer or "").strip()
    if not q:
        return RelevanceJudgment(is_relevant=True, reason="No specific question provided.")
    if not a:
        return RelevanceJudgment(is_relevant=False, reason="Empty response.", missing_elements=["answer the question directly"])

    llm = _llm_safely(temperature=0.0)
    if not llm:
        # Heuristic fallback: simple keyword overlap
        q_terms = {t.lower() for t in q.replace("?", " ").split() if len(t) > 3}
        a_terms = {t.lower() for t in a.replace("?", " ").split() if len(t) > 3}
        overlap = q_terms.intersection(a_terms)
        ok = len(overlap) >= max(1, len(q_terms) // 6)
        reason = "keyword overlap" if ok else "low keyword overlap"
        return RelevanceJudgment(is_relevant=ok, reason=reason)

    judge_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a strict grader. Determine if the user's answer directly addresses the agent's question. "
            "Only consider topical relevance, not writing style. Be fair but firm."
        ),
        (
            "human",
            "Question: {q}\nAnswer: {a}\n\n"
            "Return a strict JSON object with keys: is_relevant (boolean), reason (short), missing_elements (string list)."
        ),
    ])
    msgs = judge_prompt.format_messages(q=q, a=a)
    try:
        structured = llm.with_structured_output(RelevanceJudgment)
        result = structured.invoke(msgs)
        # Ensure pydantic instance is returned
        if isinstance(result, RelevanceJudgment):
            return result
        if hasattr(result, "model_dump"):
            data = result.model_dump()
        elif hasattr(result, "dict"):
            data = result.dict()
        else:
            data = result if isinstance(result, dict) else {}
        return RelevanceJudgment(**data)
    except Exception as e:
        log.info("relevance_llm_fail: %s", e)
        # Conservative default: require another try
        return RelevanceJudgment(is_relevant=False, reason="judgment error", missing_elements=["address the question directly"])


def handle_user_turn(last_agent_question: Optional[str], user_message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Orchestrate a conversational turn.

    - If we have a pending agent question, verify the user's message is relevant.
      If not, return a gentle re-ask with brief guidance.
    - Else, treat the user message as a question to the lead-gen expert and answer it.

    Returns a dict: { "reply": str, "relevant": bool, "missing": list[str] }
    """
    context = context or {}
    pending = (last_agent_question or "").strip()
    msg = (user_message or "").strip()
    if pending:
        judge = check_answer_relevance(pending, msg)
        if not judge.is_relevant:
            hint = " ".join((judge.reason or "").split())
            missing = ", ".join(judge.missing_elements or [])
            ask = f"I may have missed it — could you answer the question: ‘{pending}’?"
            if missing:
                ask += f" A quick note: consider covering {missing}."
            return {"reply": ask, "relevant": False, "missing": judge.missing_elements or []}
        # Acknowledge and optionally follow-up with a brief next question if needed
        return {"reply": "Thanks — got it.", "relevant": True, "missing": []}

    # No pending question: interpret user's message as a question and answer
    answer = answer_leadgen_question(msg, context=context)
    return {"reply": answer, "relevant": True, "missing": []}


def phrase_system_update(
    heading: Optional[str] = None,
    body: Optional[str] = None,
    block: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> str:
    """Return a concise, LLM‑phrased user message for system updates.

    - Produces a short 1–2 sentence summary using the provided heading/body/context.
    - If `block` is provided (e.g., a table), it is appended verbatim after the sentence.
    - Falls back to the original text when the LLM is unavailable.
    """
    context = context or {}
    llm = _llm_safely(temperature=0.2)
    preface: str
    if llm:
        try:
            sys = (
                "You rephrase internal system updates into a concise, user‑facing sentence (<= 2 sentences). "
                "Do not invent facts. You may mention counts or next actions from the provided Context. "
                "Do not include code fences. The UI will append any provided block verbatim."
            )
            prompt = ChatPromptTemplate.from_messages([
                ("system", sys),
                (
                    "human",
                    "Heading: {h}\nBody: {b}\nContext: {c}\n\nRewrite as a concise user update.",
                ),
            ])
            msgs = prompt.format_messages(
                h=(heading or ""), b=(body or ""), c=json.dumps(context, ensure_ascii=False)
            )
            preface = (llm.invoke(msgs).content or "").strip()
        except Exception as e:
            log.info("phrase_system_update_llm_fail: %s", e)
            preface = (heading or (body.splitlines()[0] if body else "Update")).strip()
    else:
        preface = (heading or (body.splitlines()[0] if body else "Update")).strip()

    if block:
        return f"{preface}\n\n{block}"
    if body:
        return f"{preface}\n\n{body}"
    return preface


__all__ = [
    "answer_leadgen_question",
    "phrase_system_update",
    "check_answer_relevance",
    "handle_user_turn",
    "RelevanceJudgment",
]
