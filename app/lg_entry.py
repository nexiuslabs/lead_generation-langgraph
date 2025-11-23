# Simplified entry wrapper that routes every turn to the orchestrator graph.

from __future__ import annotations

import os
from typing import Any, Dict, List, Union

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import var_child_runnable_config
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import CONFIG_KEY_CHECKPOINTER

from my_agent.agent import build_orchestrator_graph
from my_agent.utils.state import OrchestrationState

Content = Union[str, List[dict], dict, None]

CHECKPOINTER = MemorySaver()
ORCHESTRATOR = build_orchestrator_graph()


def normalize_payload(payload: dict) -> dict:
    """Convert SDK payloads into LangChain message objects while preserving state."""
    if not isinstance(payload, dict):
        return {}
    messages = payload.get("messages")
    if isinstance(messages, list) and all(isinstance(m, (HumanMessage, AIMessage)) for m in messages):
        return payload
    state = dict(payload.get("state") or {})
    normalized = []
    for item in payload.get("input", {}).get("messages", []):
        if not isinstance(item, dict):
            continue
        role = (item.get("role") or item.get("type") or "").lower()
        content = item.get("content")
        if not isinstance(content, str):
            continue
        if role in {"human", "user"}:
            normalized.append(HumanMessage(content=content))
        elif role in {"assistant", "ai"}:
            normalized.append(AIMessage(content=content))
    if normalized:
        state["messages"] = normalized
    return state


def _load_checkpoint_state(config: RunnableConfig | None) -> dict | None:
    cfg = config or var_child_runnable_config.get()
    if not cfg:
        return None
    configurable = cfg.get("configurable") or {}
    thread_id = configurable.get("thread_id")
    saver = configurable.get(CONFIG_KEY_CHECKPOINTER)
    if not thread_id or saver is None:
        return None
    request = {"configurable": {"thread_id": thread_id}}
    try:
        checkpoint = saver.get_tuple(request)
    except Exception:
        return None
    if not checkpoint:
        return None
    return (checkpoint.checkpoint or {}).get("channel_values")


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
    state: OrchestrationState = {
        "messages": [],
        "input": text,
        "input_role": role,
        "entry_context": {"thread_id": thread_id, "run_mode": run_mode},
        "icp_payload": payload.get("icp_payload") or {},
    }
    # Copy prior messages if provided
    prior_messages = payload.get("messages") or []
    for msg in prior_messages:
        if isinstance(msg, AIMessage):
            state["messages"].append({"role": "assistant", "content": msg.content})
        elif isinstance(msg, HumanMessage):
            state["messages"].append({"role": "user", "content": msg.content})

    result = await ORCHESTRATOR.ainvoke(state, config=config)
    status = (result or {}).get("status") or {}
    message = status.get("message", "Processing complete.")
    status_history = result.get("status_history") or []
    # Return minimal structure for the chat UI
    return {
        "messages": result.get("messages") or [],
        "status": status,
        "status_history": status_history,
        "output": message,
    }
