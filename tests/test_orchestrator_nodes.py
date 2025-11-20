import uuid

import pytest

from my_agent.utils import nodes
from my_agent.utils.state import OrchestrationState


@pytest.fixture(autouse=True)
def _mock_llm(monkeypatch):
    monkeypatch.setattr(nodes, "call_llm_json", lambda prompt, fallback: fallback)


@pytest.mark.asyncio
async def test_ingest_message_sets_intent():
    state: OrchestrationState = {
        "input": "Please confirm company",
        "input_role": "user",
        "messages": [],
        "entry_context": {},
        "icp_payload": {},
    }
    out = await nodes.ingest_message(state)
    assert out["status"]["phase"] == "ingest"
    assert out["entry_context"]["intent"] == "confirm_company"
    assert out["status_history"][-1]["phase"] == "ingest"


@pytest.mark.asyncio
async def test_journey_guard_prompts_when_missing_confirmation():
    state: OrchestrationState = {
        "messages": [],
        "entry_context": {},
        "icp_payload": {},
        "profile_state": {"company_profile_confirmed": False, "icp_profile_confirmed": True, "outstanding_prompts": []},
    }
    out = await nodes.journey_guard(state)
    assert out["status"]["phase"] == "journey_guard"
    assert "company profile" in out["status"]["message"]
    assert out["profile_state"]["outstanding_prompts"]
    assert out["status_history"][-1]["phase"] == "journey_guard"


@pytest.mark.asyncio
async def test_progress_report_appends_assistant_message():
    state: OrchestrationState = {
        "messages": [{"role": "user", "content": "hi"}],
        "status": {"phase": "score_leads", "message": "Scored"},
        "discovery": {"candidate_ids": [1, 2]},
        "profile_state": {"company_profile_confirmed": True, "icp_profile_confirmed": True},
    }
    out = await nodes.progress_report(state)
    assert out["status"]["phase"] == "progress_report"
    assert out["messages"][-1]["role"] == "assistant"
    assert "Stage" in out["messages"][-1]["content"]
    assert out["status_history"][-1]["phase"] == "progress_report"
