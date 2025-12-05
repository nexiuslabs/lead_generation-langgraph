"""
Graph construction entry point for the unified orchestrator.

This currently builds a linear placeholder pipeline that wires together the
node functions declared in `my_agent.utils.nodes`. As implementation
progresses, replace the TODO areas with the real business logic.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
import contextlib
import atexit
import os
import logging
from langgraph.graph import StateGraph

from .utils import nodes
from .utils.state import OrchestrationState


def build_orchestrator_graph():
    graph = StateGraph(OrchestrationState)

    graph.add_node("ingest", nodes.ingest_message)
    # New: probe returning-user context before building profiles
    graph.add_node("return_user_probe", nodes.return_user_probe)
    graph.add_node("profile_builder", nodes.profile_builder)
    # New: mid-run update/cancel gating
    graph.add_node("run_guard", nodes.run_guard)
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
    # Interpose run_guard between profile_builder and journey_guard
    graph.add_edge("profile_builder", "run_guard")
    graph.add_edge("run_guard", "journey_guard")
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
    dir_path = os.getenv("LANGGRAPH_CHECKPOINT_DIR", ".langgraph_api").rstrip("/")
    os.makedirs(dir_path, exist_ok=True)
    db_path = os.path.join(dir_path, "orchestrator.sqlite")
    try:
        # Use async sqlite saver when available (required for ainvoke)
        try:
            import importlib.util as _ilu
            if _ilu.find_spec("aiosqlite") is None:
                raise RuntimeError("aiosqlite not installed")
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # type: ignore

            # Construct from connection string with explicit aiosqlite driver
            conn_str = f"sqlite+aiosqlite:///{db_path}"
            cp = AsyncSqliteSaver.from_conn_string(conn_str)  # type: ignore[attr-defined]
            # Defensive: if this version returns an async context manager, fall back to memory
            if not hasattr(cp, "get_next_version"):
                logging.getLogger(__name__).warning(
                    "AsyncSqliteSaver.from_conn_string returned context manager; falling back to MemorySaver"
                )
                checkpointer = MemorySaver()
            else:
                checkpointer = cp
        except Exception as _e:
            # Fallback to memory; sync SqliteSaver is not compatible with async ainvoke
            logging.getLogger(__name__).info(
                "AsyncSqliteSaver unavailable (%s); using in-memory checkpointing", _e
            )
            checkpointer = MemorySaver()
    except Exception as _e:
        logging.getLogger(__name__).warning("Checkpoint init failed; using memory: %s", _e)
        checkpointer = MemorySaver()

    return graph.compile(checkpointer=checkpointer)
