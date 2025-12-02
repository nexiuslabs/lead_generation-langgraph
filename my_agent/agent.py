"""
Graph construction entry point for the unified orchestrator.

This currently builds a linear placeholder pipeline that wires together the
node functions declared in `my_agent.utils.nodes`. As implementation
progresses, replace the TODO areas with the real business logic.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
import os
from langgraph.graph import StateGraph

from .utils import nodes
from .utils.state import OrchestrationState


def build_orchestrator_graph():
    graph = StateGraph(OrchestrationState)

    graph.add_node("ingest", nodes.ingest_message)
    # New: probe returning-user context before building profiles
    graph.add_node("return_user_probe", nodes.return_user_probe)
    graph.add_node("profile_builder", nodes.profile_builder)
    graph.add_node("journey_guard", nodes.journey_guard)
    graph.add_node("normalize", nodes.normalize)
    graph.add_node("refresh_icp", nodes.refresh_icp)
    graph.add_node("decide_strategy", nodes.decide_strategy)
    graph.add_node("ssic_fallback", nodes.ssic_fallback)
    graph.add_node("progress_report", nodes.progress_report)
    graph.add_node("summary", nodes.finalize)

    graph.set_entry_point("ingest")
    graph.add_edge("ingest", "return_user_probe")
    graph.add_edge("return_user_probe", "profile_builder")
    graph.add_edge("profile_builder", "journey_guard")
    graph.add_conditional_edges(
        "journey_guard",
        lambda state: "ready" if state.get("journey_ready") else "pending",
        {
            "ready": "normalize",
            "pending": "progress_report",
        },
    )
    graph.add_edge("normalize", "refresh_icp")
    graph.add_edge("refresh_icp", "decide_strategy")
    graph.add_edge("decide_strategy", "ssic_fallback")
    graph.add_edge("ssic_fallback", "progress_report")
    graph.add_edge("progress_report", "summary")

    # Prefer durable checkpoint when available, else fall back to memory.
    checkpointer = None
    try:
        # Use sqlite-backed saver when library is available
        from langgraph.checkpoint.sqlite import SqliteSaver  # type: ignore

        dir_path = os.getenv("LANGGRAPH_CHECKPOINT_DIR", ".langgraph_api").rstrip("/")
        os.makedirs(dir_path, exist_ok=True)
        db_path = os.path.join(dir_path, "orchestrator.sqlite")
        # Some versions expose a convenient constructor:
        try:
            checkpointer = SqliteSaver.from_conn_string(f"sqlite:///{db_path}")  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - different API surface
            # Fallback: attempt simple path-based init
            checkpointer = SqliteSaver(db_path)  # type: ignore[call-arg]
    except Exception:
        checkpointer = MemorySaver()

    return graph.compile(checkpointer=checkpointer)
