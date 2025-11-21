import uuid

import pytest

from my_agent.utils import nodes
from my_agent.utils.state import OrchestrationState


@pytest.fixture(autouse=True)
def _mock_llm(monkeypatch):
    monkeypatch.setattr(nodes, "call_llm_json", lambda prompt, fallback: fallback)


def test_mcp_reader_auto_enabled(monkeypatch):
    try:
        from src import settings as _settings  # type: ignore
    except Exception:  # pragma: no cover
        pytest.skip("settings module unavailable")
    monkeypatch.setattr(_settings, "ENABLE_MCP_READER", False, raising=False)
    nodes._ensure_mcp_reader_enabled()
    assert getattr(_settings, "ENABLE_MCP_READER") is True


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
async def test_journey_guard_requests_discovery_confirmation():
    state: OrchestrationState = {
        "messages": [],
        "entry_context": {},
        "icp_payload": {},
        "profile_state": {
            "company_profile_confirmed": True,
            "icp_profile_confirmed": True,
            "icp_discovery_confirmed": False,
            "customer_websites": [
                "https://a.com",
                "https://b.com",
                "https://c.com",
                "https://d.com",
                "https://e.com",
            ],
            "icp_profile": {
                "summary": "Ideal customers are lean SaaS teams.",
                "industries": ["SaaS"],
            },
            "outstanding_prompts": [],
        },
    }
    out = await nodes.journey_guard(state)
    assert "start ICP discovery" in out["status"]["message"]
    assert out["profile_state"]["awaiting_discovery_confirmation"] is True
    assert out["status"]["phase"] == "journey_guard"


@pytest.mark.asyncio
async def test_journey_guard_prompts_for_enrichment_confirmation(monkeypatch):
    monkeypatch.setattr(nodes, "_plan_web_candidates", lambda icp: {"domains": ["alpha.com", "beta.com"], "snippets": {}})
    state: OrchestrationState = {
        "messages": [],
        "entry_context": {},
        "icp_payload": {},
        "profile_state": {
            "company_profile_confirmed": True,
            "icp_profile_confirmed": True,
            "icp_discovery_confirmed": True,
            "customer_websites": [
                "https://a.com",
                "https://b.com",
                "https://c.com",
                "https://d.com",
                "https://e.com",
            ],
            "icp_profile": {"summary": "test", "industries": ["F&B"]},
            "outstanding_prompts": [],
        },
    }
    out = await nodes.journey_guard(state)
    assert "enrich the best 10" in out["status"]["message"]
    assert out["profile_state"]["awaiting_enrichment_confirmation"] is True
    assert out["discovery"]["web_candidates"] == ["alpha.com", "beta.com"]


@pytest.mark.asyncio
async def test_journey_guard_ready_when_discovery_confirmed():
    state: OrchestrationState = {
        "messages": [],
        "entry_context": {},
        "icp_payload": {},
        "profile_state": {
            "company_profile_confirmed": True,
            "icp_profile_confirmed": True,
            "icp_discovery_confirmed": True,
            "enrichment_confirmed": True,
            "outstanding_prompts": ["pending"],
        },
    }
    out = await nodes.journey_guard(state)
    assert out["journey_ready"] is True
    assert not out["profile_state"]["outstanding_prompts"]
    assert out["status"]["phase"] == "journey_guard"


@pytest.mark.asyncio
async def test_profile_builder_marks_discovery_confirmed_on_positive_intent():
    state: OrchestrationState = {
        "messages": [],
        "entry_context": {"intent": "run_enrichment", "last_user_command": "start discovery"},
        "profile_state": {
            "company_profile_confirmed": True,
            "icp_profile_confirmed": True,
            "icp_profile": {"summary": "test"},
            "awaiting_discovery_confirmation": True,
            "icp_discovery_confirmed": False,
        },
    }
    out = await nodes.profile_builder(state)
    assert out["profile_state"]["icp_discovery_confirmed"] is True
    assert out["profile_state"]["awaiting_discovery_confirmation"] is False


@pytest.mark.asyncio
async def test_profile_builder_marks_enrichment_confirmed_on_command():
    state: OrchestrationState = {
        "messages": [],
        "entry_context": {"intent": "run_enrichment", "last_user_command": "enrich 10 now"},
        "profile_state": {
            "company_profile_confirmed": True,
            "icp_profile_confirmed": True,
            "icp_discovery_confirmed": True,
            "awaiting_enrichment_confirmation": True,
            "enrichment_confirmed": False,
            "icp_profile": {"summary": "test"},
        },
    }
    out = await nodes.profile_builder(state)
    assert out["profile_state"]["enrichment_confirmed"] is True
    assert out["profile_state"]["awaiting_enrichment_confirmation"] is False


@pytest.mark.asyncio
async def test_profile_builder_sets_retry_flag():
    state: OrchestrationState = {
        "messages": [],
        "entry_context": {"intent": "chat", "last_user_command": "please retry discovery"},
        "profile_state": {
            "company_profile_confirmed": True,
            "icp_profile_confirmed": True,
            "icp_discovery_confirmed": True,
            "awaiting_enrichment_confirmation": True,
            "enrichment_confirmed": False,
            "icp_profile": {"summary": "test"},
        },
    }
    out = await nodes.profile_builder(state)
    assert out["profile_state"]["discovery_retry_requested"] is True
    assert out["profile_state"]["awaiting_enrichment_confirmation"] is False


@pytest.mark.asyncio
async def test_refresh_icp_waits_for_enrichment_confirmation():
    state: OrchestrationState = {
        "profile_state": {"enrichment_confirmed": False},
        "discovery": {},
        "icp_payload": {},
    }
    out = await nodes.refresh_icp(state)
    assert out["status"]["message"] == "Waiting for enrichment confirmation"
    assert out.get("discovery", {}).get("candidate_ids") is None


@pytest.mark.asyncio
async def test_refresh_icp_runs_after_enrichment_confirmation(monkeypatch):
    class DummyRefresh:
        async def ainvoke(self, payload):
            return {"candidate_ids": [7, 8]}

    monkeypatch.setattr(nodes, "icp_refresh_agent", DummyRefresh())
    state: OrchestrationState = {
        "profile_state": {"enrichment_confirmed": True},
        "discovery": {},
        "icp_payload": {},
    }
    out = await nodes.refresh_icp(state)
    assert out["discovery"]["candidate_ids"] == [7, 8]


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
