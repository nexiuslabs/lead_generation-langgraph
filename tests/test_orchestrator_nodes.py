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
async def test_journey_guard_surfaces_icp_summary_for_confirmation():
    state: OrchestrationState = {
        "messages": [],
        "entry_context": {},
        "icp_payload": {},
        "profile_state": {
            "company_profile_confirmed": True,
            "icp_profile_confirmed": False,
            "icp_profile_generated": True,
            "customer_websites": [
                "https://a.com",
                "https://b.com",
                "https://c.com",
                "https://d.com",
                "https://e.com",
            ],
            "icp_profile": {
                "summary": "Ideal customers are mid-market SaaS revenue teams.",
                "industries": ["SaaS"],
                "company_sizes": ["50-250 employees"],
                "regions": ["North America"],
                "pains": ["Manual lead routing"],
                "buying_triggers": ["Hiring sales ops"],
                "persona_titles": ["RevOps Lead"],
                "proof_points": ["Boosts SDR throughput"],
            },
            "outstanding_prompts": [],
        },
    }
    out = await nodes.journey_guard(state)
    assert "ICP I drafted" in out["status"]["message"]
    assert "Ideal customers" in out["status"]["message"]
    assert out["messages"][-1]["role"] == "assistant"
    assert out["profile_state"]["outstanding_prompts"]


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
    assert out["messages"][-1]["content"]
    assert out["status_history"][-1]["phase"] == "progress_report"
