from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from src.notifications.render import render_summary_html, build_csv_bytes
from src.notifications.sendgrid import send_leads_email
from src.settings import LANGCHAIN_MODEL, TEMPERATURE, EMAIL_ENABLED

log = logging.getLogger("notifications.agent")


@tool
async def send_email_tool(to: str, subject: str, intro_html: str, tenant_id: int, limit: int = 200) -> Dict[str, Any]:
    """Send the shortlist email. Provide:
    - to: recipient email address
    - subject: concise subject line (<=80 chars)
    - intro_html: a brief 1–3 sentence HTML intro for the email body
    - tenant_id: numeric tenant id of the shortlist to include
    - limit: max rows to include from the shortlist (default 200)
    Returns delivery status dict.
    """
    subj_auto, table_html, csv_link = render_summary_html(int(tenant_id), int(limit))
    subject_final = subject or subj_auto
    html = f"{intro_html}{table_html}"
    # Build CSV attachment
    try:
        csv_bytes, csv_name = build_csv_bytes(int(tenant_id), 500)
    except Exception:
        csv_bytes, csv_name = None, None  # type: ignore[assignment]
    res = await send_leads_email(
        to,
        subject_final,
        html,
        attachment_bytes=csv_bytes,
        attachment_filename=csv_name,
        attachment_content_type="text/csv",
    )
    # attach csv_link hint for caller telemetry
    try:
        res["csv_link"] = csv_link
    except Exception:
        pass
    return res


def _make_llm() -> ChatOpenAI:
    # Reuse the same model config as enrichment
    kwargs: dict = {"model": LANGCHAIN_MODEL}
    if TEMPERATURE is not None and not str(LANGCHAIN_MODEL).lower().startswith("gpt-5"):
        kwargs["temperature"] = TEMPERATURE
    return ChatOpenAI(**kwargs)


async def agentic_send_results(to: Optional[str], tenant_id: Optional[int], *, limit: int = 200) -> Dict[str, Any]:
    """Compose and send shortlist email via an LLM + tool-calling agent.

    Returns dict: { status, to?, email_status?, request_id?, error?, csv_link? }
    """
    if not to or not tenant_id:
        return {"status": "skipped", "error": "missing to/tenant_id"}
    if not EMAIL_ENABLED:
        try:
            log.info("email skipped: EMAIL_ENABLED is false; to=%s tenant_id=%s", to, tenant_id)
        except Exception:
            pass
        return {"status": "skipped_no_config"}

    # Prepare context summary for the model (counts + a few top names)
    try:
        log.info("agentic_email:start to=%s tenant_id=%s limit=%s", to, tenant_id, limit)
    except Exception:
        pass
    subj_auto, table_html, csv_link = render_summary_html(int(tenant_id), int(limit))
    # Derive a tiny, safe preview list (no PII) to guide tone
    preview_names: list[str] = []
    try:
        import re
        # naive parse of first 5 names from the table_html
        rows = re.findall(r"<tr>.*?</tr>", table_html, flags=re.S)
        for row in rows[1:6]:  # skip header
            m = re.search(r"<td[^>]*>(.*?)</td>", row)
            if m:
                name = re.sub(r"<[^>]+>", "", m.group(1)).strip()
                if name:
                    preview_names.append(name)
    except Exception:
        preview_names = []

    sys = SystemMessage(content=(
        "You are an SDR assistant. Write a concise, neutral email intro summarizing a lead shortlist. "
        "Keep it 1–3 short sentences. Do not invent data; avoid personal details."
    ))
    human = HumanMessage(content=(
        "Task: Compose a subject line and a brief HTML intro for the shortlist email.\n"
        "Audience: internal pre-SDR user.\n"
        f"Guidance: Mention the total leads and that a CSV is attached via link.\n"
        f"Sample top names: {', '.join(preview_names) if preview_names else 'n/a'}.\n"
        f"Auto-subject (fallback): {subj_auto}.\n"
        "Constraints: Subject <= 80 chars. Intro only; the table and CSV link are appended by the system."
    ))

    llm = _make_llm().bind_tools([send_email_tool])
    ai = await llm.ainvoke([sys, human])

    # Execute any tool calls (expecting one call to send_email_tool)
    calls = getattr(ai, "tool_calls", None) or []
    sent = None
    for call in calls:
        try:
            if call["name"] == "send_email_tool":
                args = dict(call.get("args") or {})
                # Force authoritative args (LLM must not override recipient or tenant)
                if args.get("to") and str(args.get("to")) != str(to):
                    try:
                        log.info("agent attempted to override recipient; ignoring")
                    except Exception:
                        pass
                args["to"] = to
                args["tenant_id"] = int(tenant_id)
                args["limit"] = int(limit)
                # If the model didn't provide intro/subject, use safe defaults
                args.setdefault("intro_html", "<p>Here is your latest shortlisted leads summary. You can download the full CSV from the link below.</p>")
                args.setdefault("subject", subj_auto)
                sent = await send_email_tool.ainvoke(args)
                break
        except Exception as e:
            log.exception("agent tool execution failed: %s", e)
            sent = {"status": "failed", "error": str(e)}
            break
    if not sent:
        # Fallback: send with auto subject and generic intro
        try:
            generic_intro = "<p>Here is your latest shortlisted leads summary. You can download the full CSV from the link below.</p>"
            sent = await send_email_tool.ainvoke({
                "to": to,
                "tenant_id": int(tenant_id),
                "limit": int(limit),
                "subject": subj_auto,
                "intro_html": generic_intro,
            })
        except Exception as e:
            sent = {"status": "failed", "error": str(e)}

    out = {"status": sent.get("status"), "to": to}
    for k in ("request_id", "http_status", "error", "csv_link"):
        if sent.get(k) is not None:
            out[k] = sent.get(k)
    try:
        log.info(
            "agentic_email:done to=%s tenant_id=%s status=%s http_status=%s request_id=%s",
            to,
            tenant_id,
            out.get("status"),
            out.get("http_status"),
            out.get("request_id"),
        )
    except Exception:
        pass
    return out
