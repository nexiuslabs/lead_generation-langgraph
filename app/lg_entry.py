# app/lg_entry.py
from typing import Dict, Any, List, Union
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END
from app.pre_sdr_graph import build_graph, GraphState  # new dynamic builder

Content = Union[str, List[dict], dict, None]


def _role_to_type(role: str) -> str:
    r = (role or "").lower()
    if r in ("user", "human"):
        return "human"
    if r in ("assistant", "ai"):
        return "ai"
    if r == "system":
        return "system"
    return "human"


def _flatten_content(content: Content) -> str:
    """
    Accepts UI message content in various shapes and returns a plain string.
    Examples:
      - "hello"
      - [{"type":"input_text","text":"hello"}, {"type":"image_url",...}]
      - {"text": "..."}
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        # Common shape from SDKs
        if "text" in content and isinstance(content["text"], str):
            return content["text"]
        return str(content)
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                if "text" in item and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif item.get("type") in ("input_text", "text") and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif "image_url" in item:
                    parts.append("[image]")
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return "\n".join([p for p in parts if p])
    # Fallback stringify
    return str(content)


def _to_message(msg: dict | BaseMessage) -> BaseMessage:
    if isinstance(msg, BaseMessage):
        # Ensure content is a string
        if not isinstance(msg.content, str):
            # Best-effort conversion
            text = _flatten_content(msg.content)  # type: ignore[arg-type]
            # Recreate message with string content to avoid mutating internals
            if isinstance(msg, HumanMessage):
                return HumanMessage(content=text)
            if isinstance(msg, SystemMessage):
                return SystemMessage(content=text)
            return AIMessage(content=text)
        return msg
    mtype = msg.get("type") or _role_to_type(msg.get("role", "human"))
    content = _flatten_content(msg.get("content"))
    if mtype == "human":
        return HumanMessage(content=content)
    if mtype == "system":
        return SystemMessage(content=content)
    return AIMessage(content=content)


def _normalize(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Agent Chat UI will call /threads/.../runs with a body like:
      {"assistant_id":"agent","input":{"messages":[{"role":"human","content":"start"}]}}
    We map it to the graph state: {"messages": [BaseMessage,...], "candidates": ...}
    """
    data = payload.get("input", payload) or {}
    msgs = data.get("messages") or []
    if isinstance(msgs, dict):  # sometimes a single message object is sent
        msgs = [msgs]

    norm_msgs = [_to_message(m) for m in msgs] or [HumanMessage(content="")]
    state: Dict[str, Any] = {"messages": norm_msgs}

    # optional “companies”/“candidates” passthrough for your graph
    if "candidates" in data:
        state["candidates"] = data["candidates"]
    elif "companies" in data:
        state["candidates"] = data["companies"]

    return state


def make_graph(config: Dict[str, Any] | None = None):
    """Called by `langgraph dev` to get a valid compiled Graph.

    We wrap the existing compiled pre-SDR graph with a tiny outer graph that
    normalizes Chat UI payloads into the expected PreSDRState. Returning a
    compiled StateGraph ensures the dev server's graph validation passes.
    """
    inner = build_graph()  # compiled inner graph (dynamic Pre-SDR pipeline)

    def normalize_node(payload: Dict[str, Any]) -> GraphState:
        # Accept raw UI payload and coerce into graph state
        state = _normalize(payload)
        # type: ignore[return-value] — runtime shape matches PreSDRState
        return state  # type: ignore

    outer = StateGraph(GraphState)
    outer.add_node("normalize", normalize_node)
    outer.add_node("presdr", inner)
    outer.set_entry_point("normalize")
    outer.add_edge("normalize", "presdr")
    outer.add_edge("presdr", END)
    return outer.compile()
