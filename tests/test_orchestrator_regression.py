import uuid

import pytest

from my_agent.agent import build_orchestrator_graph
from my_agent.utils import nodes


class _AsyncAgent:
    def __init__(self, result):
        self._result = result

    async def ainvoke(self, *_args, **_kwargs):  # pragma: no cover - simple stub
        return self._result


@pytest.fixture(autouse=True)
def stub_agents(monkeypatch):
    monkeypatch.setattr(nodes, "call_llm_json", lambda prompt, fallback: fallback)
    monkeypatch.setattr(nodes, "normalize_agent", _AsyncAgent({"normalized_records": [{}]}))
    monkeypatch.setattr(nodes, "icp_refresh_agent", _AsyncAgent({"candidate_ids": [1, 2]}))
    monkeypatch.setattr(nodes, "icp_by_ssic_agent", _AsyncAgent({"acra_candidates": []}))
    monkeypatch.setattr(nodes, "plan_top10_with_reasons", lambda *_args, **_kwargs: [{"company_id": 1}])
    monkeypatch.setattr(nodes, "lead_scoring_agent", _AsyncAgent({"lead_scores": [{"company_id": 1, "score": 80}], "lead_features": []}))
    monkeypatch.setattr(nodes, "enrich_company_with_tavily", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(nodes, "enqueue_web_discovery_bg_enrich", lambda *_args, **_kwargs: {"job_id": 42})
    yield


def _base_state():
    return {
        "messages": [],
        "input": "run enrichment",
        "input_role": "user",
        "entry_context": {"thread_id": str(uuid.uuid4())},
        "icp_payload": {"industries": ["software"]},
    }


@pytest.mark.asyncio
async def test_orchestrator_reaches_summary():
    graph = build_orchestrator_graph()
    state = _base_state()
    result = await graph.ainvoke(state, config={"configurable": {"thread_id": state["entry_context"]["thread_id"]}})
    assert result["status"]["phase"] == "summary"


@pytest.mark.asyncio
async def test_orchestrator_multiple_runs_consistent():
    graph = build_orchestrator_graph()
    outputs = []
    for _ in range(5):
        state = _base_state()
        res = await graph.ainvoke(state, config={"configurable": {"thread_id": state["entry_context"]["thread_id"]}})
        outputs.append(res["status"]["phase"])
    assert all(phase == "summary" for phase in outputs)
