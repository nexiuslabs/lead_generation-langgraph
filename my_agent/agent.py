"""
Graph construction entry point for the unified orchestrator.

This currently builds a linear placeholder pipeline that wires together the
node functions declared in `my_agent.utils.nodes`. As implementation
progresses, replace the TODO areas with the real business logic.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph

from .utils import nodes
from .utils.state import OrchestrationState


def build_orchestrator_graph():
    graph = StateGraph(OrchestrationState)

    graph.add_node("ingest", nodes.ingest_message)
    graph.add_node("profile_builder", nodes.profile_builder)
    graph.add_node("journey_guard", nodes.journey_guard)
    graph.add_node("normalize", nodes.normalize)
    graph.add_node("refresh_icp", nodes.refresh_icp)
    graph.add_node("decide_strategy", nodes.decide_strategy)
    graph.add_node("ssic_fallback", nodes.ssic_fallback)
    graph.add_node("plan_top10", nodes.plan_top10)
    graph.add_node("enrich_batch", nodes.enrich_batch)
    graph.add_node("score_leads", nodes.score_leads)
    graph.add_node("export", nodes.export_results)
    graph.add_node("progress_report", nodes.progress_report)
    graph.add_node("summary", nodes.finalize)

    graph.set_entry_point("ingest")
    graph.add_edge("ingest", "profile_builder")
    graph.add_edge("profile_builder", "journey_guard")
    graph.add_edge("journey_guard", "normalize")
    graph.add_edge("normalize", "refresh_icp")
    graph.add_edge("refresh_icp", "decide_strategy")
    graph.add_edge("decide_strategy", "ssic_fallback")
    graph.add_edge("ssic_fallback", "plan_top10")
    graph.add_edge("plan_top10", "enrich_batch")
    graph.add_edge("enrich_batch", "score_leads")
    graph.add_edge("score_leads", "export")
    graph.add_edge("export", "progress_report")
    graph.add_edge("progress_report", "summary")

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)
