# Simplified entry wrapper that routes every turn to the orchestrator graph.

from __future__ import annotations

from typing import Any, Dict, List, Union

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from my_agent.agent import build_orchestrator_graph
from my_agent.utils.state import OrchestrationState

Content = Union[str, List[dict], dict, None]

ORCHESTRATOR = build_orchestrator_graph()


def _flatten_content(content: Content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict) and isinstance(content.get("text"), str):
        return content["text"]
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                texts.append(item["text"])
        return "\n".join(texts)
    return str(content)


async def handle_turn(payload: Dict[str, Any], config: RunnableConfig | None = None) -> Dict[str, Any]:
    """Invoke the orchestrator graph with the raw SDK payload."""
    text = _flatten_content(payload.get("input"))
    role = payload.get("role", "user")
    thread_id = (config or {}).get("configurable", {}).get("thread_id") if config else None
    run_mode = str(payload.get("run_mode") or "chat_top10")
    tenant_id = payload.get("tenant_id")
    user_id = payload.get("user_id")
    entry_context = {"thread_id": thread_id, "run_mode": run_mode}
    if tenant_id is not None:
        entry_context["tenant_id"] = tenant_id
    if user_id is not None:
        entry_context["user_id"] = user_id
    state: OrchestrationState = {
        "messages": [],
        "input": text,
        "input_role": role,
        "entry_context": entry_context,
        "icp_payload": payload.get("icp_payload") or {},
    }
    if tenant_id is not None:
        state["tenant_id"] = tenant_id
    # Copy prior messages if provided
    prior_messages = payload.get("messages") or []
    for msg in prior_messages:
        if isinstance(msg, AIMessage):
            nodes._append_message(state, "assistant", msg.content)
        elif isinstance(msg, HumanMessage):
            nodes._append_message(state, "user", msg.content)

    result = await ORCHESTRATOR.ainvoke(state, config=config)
    status = (result or {}).get("status") or {}
    message = status.get("message", "Processing complete.")
    status_history = result.get("status_history") or []
    filtered_messages = []
    for msg in result.get("messages") or []:
        role = msg.get("role") if isinstance(msg, dict) else None
        content = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(role, str) and isinstance(content, str):
            if nodes._should_suppress_message(role, content):
                continue
        filtered_messages.append(msg)
    # Return minimal structure for the chat UI
    return {
        "messages": filtered_messages,
        "status": status,
        "status_history": status_history,
        "output": message,
    }
