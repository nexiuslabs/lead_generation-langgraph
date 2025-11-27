import uuid

import pytest
from typing import Dict, List

from my_agent.utils import nodes
from my_agent.utils.state import OrchestrationState


@pytest.fixture(autouse=True)
def _mock_llm(monkeypatch):
    monkeypatch.setattr(nodes, "call_llm_json", lambda prompt, fallback: fallback)


REAL_COMPANY = {
    "name": "Real Co",
    "website": "https://realco.com",
    "summary": "We do real things.",
    "summary_source": "user",
    "industries": ["SaaS"],
}


def test_simple_intent_detects_discovery():
    assert nodes._simple_intent("Start discovery now please!") == "confirm_discovery"


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
    assert "share your business website" in out["status"]["message"]
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
            "company_profile": dict(REAL_COMPANY),
            "seeded_company_profile": False,
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
            "company_profile": dict(REAL_COMPANY),
            "seeded_company_profile": False,
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
async def test_profile_builder_confirms_discovery_from_start_command():
    state: OrchestrationState = {
        "messages": [
            {
                "role": "user",
                "content": "https://a.com https://b.com https://c.com https://d.com https://e.com",
            }
        ],
        "entry_context": {"intent": "chat", "last_user_command": "start discovery"},
        "icp_payload": {},
        "profile_state": {
            "company_profile_confirmed": True,
            "icp_profile_confirmed": True,
            "icp_discovery_confirmed": False,
            "awaiting_discovery_confirmation": True,
            "company_profile": dict(REAL_COMPANY),
            "icp_profile": {"summary": "Lean food distributors.", "industries": ["F&B"]},
            "seeded_company_profile": False,
            "outstanding_prompts": [],
        },
    }
    out = await nodes.profile_builder(state)
    profile = out["profile_state"]
    assert profile["icp_discovery_confirmed"] is True
    assert profile["awaiting_discovery_confirmation"] is False


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
            "company_profile": dict(REAL_COMPANY),
            "seeded_company_profile": False,
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
    assert "enrich 10" in out["status"]["message"]
    assert "| # |" in out["status"]["message"]
    assert out["profile_state"]["awaiting_enrichment_confirmation"] is True
    assert out["discovery"]["web_candidates"] == ["alpha.com", "beta.com"]
    assert len(out["discovery"].get("web_candidate_details") or []) == 2
    assert [row.get("domain") for row in (out["discovery"].get("top10_details") or [])] == ["alpha.com", "beta.com"]
    assert [row.get("domain") for row in (out["discovery"].get("top10_details") or [])] == ["alpha.com", "beta.com"]


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
            "company_profile": dict(REAL_COMPANY),
            "seeded_company_profile": False,
            "outstanding_prompts": ["pending"],
        },
    }
    out = await nodes.journey_guard(state)
    assert out["journey_ready"] is True
    assert not out["profile_state"]["outstanding_prompts"]
    assert out["status"]["phase"] == "journey_guard"


@pytest.mark.asyncio
async def test_journey_guard_accepts_enrichment_command():
    state: OrchestrationState = {
        "messages": [],
        "entry_context": {"intent": "run_enrichment", "last_user_command": "enrich 10"},
        "icp_payload": {},
        "profile_state": {
            "company_profile_confirmed": True,
            "icp_profile_confirmed": True,
            "icp_discovery_confirmed": True,
            "awaiting_enrichment_confirmation": True,
            "enrichment_confirmed": False,
            "company_profile": dict(REAL_COMPANY),
            "seeded_company_profile": False,
        },
    }
    out = await nodes.journey_guard(state)
    assert out["journey_ready"] is True
    assert out["profile_state"]["enrichment_confirmed"] is True
    assert out["status"]["message"].startswith("Prerequisites satisfied")


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
            "company_profile": dict(REAL_COMPANY),
            "seeded_company_profile": False,
        },
    }
    out = await nodes.profile_builder(state)
    assert out["profile_state"]["icp_discovery_confirmed"] is True
    assert out["profile_state"]["awaiting_discovery_confirmation"] is False


@pytest.mark.asyncio
async def test_profile_builder_confirms_company_from_affirmative_text():
    state: OrchestrationState = {
        "messages": [],
        "entry_context": {"intent": "confirm_company", "last_user_command": "company looks good"},
        "profile_state": {
            "company_profile_confirmed": False,
            "icp_profile_confirmed": False,
            "icp_profile": {"summary": "test"},
            "company_profile": dict(REAL_COMPANY),
            "seeded_company_profile": False,
        },
    }
    out = await nodes.profile_builder(state)
    assert out["profile_state"]["company_profile_confirmed"] is True


@pytest.mark.asyncio
async def test_profile_builder_captures_user_website_when_seeded():
    state: OrchestrationState = {
        "messages": [
            {"role": "assistant", "content": "please share your site"},
            {"role": "user", "content": "https://example.com is my site"},
        ],
        "entry_context": {"intent": "chat", "last_user_command": "https://example.com is my site"},
        "profile_state": {},
    }
    out = await nodes.profile_builder(state)
    company = out["profile_state"]["company_profile"]
    assert company["website"] == "https://example.com"
    assert out["profile_state"]["seeded_company_profile"] is False
    assert out["profile_state"]["company_profile_confirmed"] is False


@pytest.mark.asyncio
async def test_profile_builder_confirms_icp_from_affirmative_text():
    state: OrchestrationState = {
        "messages": [],
        "entry_context": {"intent": "chat", "last_user_command": "looks great and start discovery"},
        "profile_state": {
            "company_profile_confirmed": True,
            "icp_profile_confirmed": False,
            "icp_profile": {"summary": "test"},
            "company_profile": dict(REAL_COMPANY),
            "seeded_company_profile": False,
        },
    }
    out = await nodes.profile_builder(state)
    assert out["profile_state"]["icp_profile_confirmed"] is True


@pytest.mark.asyncio
async def test_profile_builder_enrichment_confirmed_without_explicit_intent():
    state: OrchestrationState = {
        "messages": [],
        "entry_context": {"intent": "chat", "last_user_command": "please enrich 10 now"},
        "profile_state": {
            "company_profile_confirmed": True,
            "icp_profile_confirmed": True,
            "icp_discovery_confirmed": True,
            "awaiting_enrichment_confirmation": True,
            "enrichment_confirmed": False,
            "icp_profile": {"summary": "test"},
            "company_profile": dict(REAL_COMPANY),
            "seeded_company_profile": False,
        },
    }
    out = await nodes.profile_builder(state)
    assert out["profile_state"]["enrichment_confirmed"] is True
    assert out["profile_state"]["awaiting_enrichment_confirmation"] is False


@pytest.mark.asyncio
async def test_profile_builder_handles_informal_discovery_confirmation():
    state: OrchestrationState = {
        "messages": [],
        "entry_context": {"intent": "chat", "last_user_command": "Ok let's start discovery now!"},
        "profile_state": {
            "company_profile_confirmed": True,
            "icp_profile_confirmed": True,
            "icp_profile": {"summary": "test"},
            "awaiting_discovery_confirmation": True,
            "icp_discovery_confirmed": False,
            "company_profile": dict(REAL_COMPANY),
            "seeded_company_profile": False,
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
            "company_profile": dict(REAL_COMPANY),
            "seeded_company_profile": False,
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
            "company_profile": dict(REAL_COMPANY),
            "seeded_company_profile": False,
        },
    }
    out = await nodes.profile_builder(state)
    assert out["profile_state"]["discovery_retry_requested"] is True
    assert out["profile_state"]["awaiting_enrichment_confirmation"] is False


@pytest.mark.asyncio
async def test_decide_strategy_reuses_cached_candidates():
    state: OrchestrationState = {
        "discovery": {
            "top10_details": [{"domain": "alpha.com"}],
            "candidate_ids": [1, 2, 3],
        },
        "profile_state": {"discovery_retry_requested": False},
        "entry_context": {},
    }
    out = await nodes.decide_strategy(state)
    assert out["discovery"]["strategy"] == "use_cached"
    assert "Reusing" in out["status"]["message"]


@pytest.mark.asyncio
async def test_plan_top10_skips_when_cached(monkeypatch):
    def _should_not_run(*_args, **_kwargs):
        raise AssertionError("plan_top10_with_reasons should be skipped")

    monkeypatch.setattr(nodes, "plan_top10_with_reasons", _should_not_run, raising=False)
    cached = [
        {"domain": "alpha.com", "name": "Alpha"},
        {"domain": "beta.com", "name": "Beta"},
    ]
    state: OrchestrationState = {
        "discovery": {"strategy": "use_cached", "web_candidate_details": cached},
        "profile_state": {"discovery_retry_requested": False},
        "entry_context": {},
    }
    out = await nodes.plan_top10(state)
    assert out["status"]["message"] == "Reusing cached discovery candidates"
    assert [row.get("domain") for row in out["discovery"].get("top10_details") or []][:2] == ["alpha.com", "beta.com"]


@pytest.mark.asyncio
async def test_enrich_batch_prefers_jina(monkeypatch):
    called: List[tuple] = []

    async def fake_collect(tenant_id, company_id, domain):
        called.append((tenant_id, company_id, domain))
        return 1

    monkeypatch.setattr(nodes, "collect_evidence_for_domain", fake_collect, raising=False)
    monkeypatch.setattr(nodes, "enrich_company_with_tavily", None, raising=False)
    state: OrchestrationState = {
        "entry_context": {"tenant_id": 7},
        "discovery": {"top10_ids": [123], "top10_domains": ["alpha.com"]},
    }
    out = await nodes.enrich_batch(state)
    assert called
    assert out["enrichment_results"][0]["completed"] is True
    assert out["enrichment_results"][0]["source"] == "mcp"


@pytest.mark.asyncio
async def test_enrich_batch_falls_back_to_tavily(monkeypatch):
    async def fake_collect(tenant_id, company_id, domain):
        return 0

    fallback_called: List[int] = []

    async def fake_tavily(company_id, **kwargs):
        fallback_called.append(int(company_id))

    monkeypatch.setattr(nodes, "collect_evidence_for_domain", fake_collect, raising=False)
    monkeypatch.setattr(nodes, "enrich_company_with_tavily", fake_tavily, raising=False)
    state: OrchestrationState = {
        "entry_context": {"tenant_id": 7},
        "discovery": {"top10_ids": [456], "top10_domains": ["beta.com"]},
    }
    out = await nodes.enrich_batch(state)
    assert fallback_called == [456]
    assert out["enrichment_results"][0]["completed"] is True
    assert out["enrichment_results"][0]["source"] == "tavily"


@pytest.mark.asyncio
async def test_enrich_batch_dedupes_candidate_ids(monkeypatch):
    called: List[int] = []

    async def fake_collect(tenant_id, company_id, domain):
        called.append(company_id)
        return 1

    monkeypatch.setattr(nodes, "collect_evidence_for_domain", fake_collect, raising=False)
    monkeypatch.setattr(nodes, "enrich_company_with_tavily", None, raising=False)
    state: OrchestrationState = {
        "entry_context": {"tenant_id": 9},
        "discovery": {
            "candidate_ids": [111, "111", 222],
            "top10_ids": [111, 222],
            "top10_domains": ["alpha.com", "beta.com"],
        },
    }
    out = await nodes.enrich_batch(state)
    assert called == [111, 222]
    assert len(out["enrichment_results"]) == 2


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
async def test_score_leads_uses_async_agent(monkeypatch):
    called: Dict[str, Any] = {}

    class FakeAgent:
        async def ainvoke(self, payload):
            called["payload"] = payload
            return {"lead_scores": [{"company_id": 99, "score": 42, "bucket": "B"}]}

    monkeypatch.setattr(nodes, "lead_scoring_agent", FakeAgent())
    monkeypatch.setattr(nodes, "_lookup_primary_emails", lambda ids: {99: "test@example.com"})
    state: OrchestrationState = {
        "discovery": {"candidate_ids": ["99"], "top10_ids": [99], "top10_details": [{"name": "Alpha", "domain": "alpha.com"}]},
        "icp_payload": {"foo": "bar"},
        "profile_state": {},
        "messages": [],
    }
    out = await nodes.score_leads(state)
    assert called["payload"]["candidate_ids"] == [99]
    assert out["scoring"]["scores"] == [{"company_id": 99, "score": 42, "bucket": "B"}]
    assert "Lead Score summary" in out["messages"][-1]["content"]
    assert "test@example.com" in out["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_progress_report_appends_assistant_message():
    state: OrchestrationState = {
        "messages": [{"role": "user", "content": "hi"}],
        "status": {"phase": "score_leads", "message": "Scored"},
        "discovery": {"candidate_ids": [1, 2]},
        "profile_state": {
            "company_profile_confirmed": True,
            "icp_profile_confirmed": True,
            "company_profile": dict(REAL_COMPANY),
            "seeded_company_profile": False,
        },
    }
    out = await nodes.progress_report(state)
    assert out["status"]["phase"] == "progress_report"
    assert out["messages"][-1]["role"] == "assistant"
    assert out["messages"][-1]["content"]
    assert out["status_history"][-1]["phase"] == "progress_report"


def test_append_message_filters_json_payloads():
    state: OrchestrationState = {}
    nodes._append_message(state, "assistant", '{"is_question": false, "answer": ""}')
    assert not state.get("messages")
    nodes._append_message(state, "assistant", "icp:toplikes_ready")
    assert not state.get("messages")
    nodes._append_message(state, "assistant", "Here is a summary.")
    assert state["messages"][-1]["content"] == "Here is a summary."
    nodes._append_message(state, "user", '{"is_question": false, "answer": ""}')
    assert state["messages"][-1]["content"] == '{"is_question": false, "answer": ""}'


def test_append_message_suppresses_system_json():
    state: OrchestrationState = {}
    nodes._append_message(state, "system", '{"foo": "bar"}')
    assert not state.get("messages")
    nodes._append_message(state, "user", '{"foo": "bar"}')
    assert state["messages"][-1]["content"].startswith('{"foo"')


def test_profile_state_seeded_with_default_company():
    state: OrchestrationState = {}
    profile = nodes._ensure_profile_state(state)
    company = profile.get("company_profile") or {}
    assert company.get("name") == "Nexius Labs"
    assert company.get("summary")
    assert profile.get("company_profile_confirmed") is False
    assert profile.get("seeded_company_profile") is True


def test_format_company_profile_returns_markdown():
    company = {
        "summary": "Test summary.",
        "industries": ["One", "Two"],
        "offerings": ["A", "B", "C"],
        "ideal_customers": ["Alpha"],
        "proof_points": ["Proof"],
    }
    rendered = nodes._format_company_profile(company)
    assert "**Summary**" in rendered
    assert "- One" in rendered
    assert "**Offerings**" in rendered


@pytest.mark.asyncio
async def test_profile_builder_does_not_confirm_seeded_profile():
    state: OrchestrationState = {
        "messages": [],
        "entry_context": {"intent": "confirm_company", "last_user_command": "confirm"},
        "profile_state": {},
    }
    out = await nodes.profile_builder(state)
    assert out["profile_state"]["company_profile_confirmed"] is False
    assert out["profile_state"]["seeded_company_profile"] is True


def test_format_icp_profile_returns_markdown():
    icp = {
        "summary": "ICP summary.",
        "industries": ["F&B"],
        "company_sizes": ["10-50"],
        "regions": ["SG"],
        "pains": ["Pain"],
        "buying_triggers": ["Trigger"],
        "persona_titles": ["Buyer"],
        "proof_points": ["Proof"],
    }
    rendered = nodes._format_icp_profile(icp)
    assert "**Summary**" in rendered
    assert "- F&B" in rendered
    assert "**Company Sizes**" in rendered


def test_normalized_icp_profile_maps_synonyms():
    profile = {
        "company_sizes": ["50-250"],
        "persona_titles": ["Ops Lead"],
        "buying_triggers": ["Opening outlets"],
        "regions": ["Singapore"],
    }
    normalized = nodes._normalized_icp_for_persistence(profile)
    assert normalized["size_bands"] == ["50-250"]
    assert normalized["buyer_titles"] == ["Ops Lead"]
    assert normalized["triggers"] == ["Opening outlets"]
    assert normalized["geos"] == ["Singapore"]
