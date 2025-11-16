import asyncio
import os
import sys
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.messages import HumanMessage, AIMessage


@pytest.mark.asyncio
async def test_icp_update_without_keyword_routes_to_icp(monkeypatch):
    import app.pre_sdr_graph as presdr

    async def fake_extract_icp(text: str):
        return presdr.ICPProfileUpdate(industries=["food distribution"])

    monkeypatch.setattr(presdr, "extract_icp_profile_update", fake_extract_icp)
    monkeypatch.setattr(presdr, "_persist_icp_profile_sync", lambda *args, **kwargs: None)

    state = {
        "messages": [HumanMessage(content="remove hospitality in industry")],
        "icp": {"seeds_list": [{}] * 5},
        "icp_profile": {"industries": ["food service", "hospitality"]},
        "icp_profile_summary_sent": True,
        "awaiting_icp_profile_confirmation": True,
        "last_profile_prompt_type": "icp",
    }

    out = await presdr.icp_node(state)  # type: ignore[arg-type]

    assert out.get("awaiting_icp_profile_confirmation") is True
    msgs = out.get("messages") or []
    assert isinstance(msgs[-1], AIMessage)
    assert "icp profile" in (msgs[-1].content or "").lower()


@pytest.mark.asyncio
async def test_icp_update_honors_explicit_icp_even_when_company_pending(monkeypatch):
    import app.pre_sdr_graph as presdr

    async def fake_extract_icp(text: str):
        return presdr.ICPProfileUpdate(industries=["wholesale"])

    monkeypatch.setattr(presdr, "extract_icp_profile_update", fake_extract_icp)
    monkeypatch.setattr(presdr, "_persist_icp_profile_sync", lambda *args, **kwargs: None)

    state = {
        "messages": [HumanMessage(content="add wholesale to industry of icp profile")],
        "icp": {"seeds_list": [{}] * 5},
        "icp_profile": {"industries": ["food service"]},
        "icp_profile_summary_sent": True,
        "awaiting_icp_profile_confirmation": True,
        "last_profile_prompt_type": "icp",
        "company_profile_pending": True,
    }

    out = await presdr.icp_node(state)  # type: ignore[arg-type]

    assert out.get("awaiting_icp_profile_confirmation") is True
    assert not out.get("company_profile_pending")


def test_company_pending_clears_after_confirmation(monkeypatch):
    import app.pre_sdr_graph as presdr

    monkeypatch.setattr(presdr, "_persist_company_profile_sync", lambda *args, **kwargs: None)

    state = {
        "messages": [],
        "company_profile": {"industries": ["logistics"]},
        "icp": {},
    }

    presdr._prompt_profile_confirmation(state)
    assert state.get("company_profile_pending") is True

    state["messages"] = presdr.add_messages(
        state.get("messages") or [], [HumanMessage(content="confirm profile")]
    )

    presdr._handle_profile_confirmation_gate(state, "confirm profile")

    assert state.get("company_profile_pending") is False
    assert state.get("awaiting_profile_confirmation") is False


@pytest.mark.asyncio
async def test_company_pending_reopens_on_company_update_request(monkeypatch):
    import app.pre_sdr_graph as presdr

    async def fake_extract_company(text: str):
        return presdr.CompanyProfileUpdate(industries=["automation"])

    monkeypatch.setattr(presdr, "extract_company_profile_update", fake_extract_company)

    state = {
        "messages": [HumanMessage(content="add automation to company profile industries")],
        "icp": {"seeds_list": [{}] * 5},
        "company_profile": {"industries": ["logistics"]},
        "company_profile_confirmed": True,
        "company_profile_pending": False,
        "icp_profile": {"industries": ["logistics"]},
        "icp_profile_summary_sent": True,
    }

    out = await presdr.icp_node(state)  # type: ignore[arg-type]

    assert out.get("company_profile_pending") is True
    assert out.get("awaiting_profile_confirmation") is True


@pytest.mark.asyncio
async def test_icp_prompt_sent_when_seed_domains_missing(monkeypatch):
    import app.pre_sdr_graph as presdr

    monkeypatch.setattr(presdr, "ENABLE_AGENT_DISCOVERY", True, raising=False)
    monkeypatch.setattr(presdr, "_agent_icp_synth", lambda payload: {"icp_profile": {}}, raising=False)

    state = {
        "messages": [],
        "icp": {
            "website_url": "https://example.com",
            "seeds_list": [{"seed_name": f"Seed {i}", "domain": ""} for i in range(5)],
        },
    }

    await presdr._maybe_bootstrap_icp_profile_from_seeds(state)

    assert state.get("awaiting_icp_profile_confirmation") is True
    assert state.get("icp_profile_pending") is True
    summary = ((state.get("icp_profile") or {}).get("summary") or "").lower()
    assert "seed" in summary


@pytest.mark.asyncio
async def test_confirm_profile_routes_to_icp_when_icp_pending(monkeypatch):
    import app.pre_sdr_graph as presdr

    monkeypatch.setattr(presdr, "_persist_icp_profile_sync", lambda *args, **kwargs: None)
    monkeypatch.setattr(presdr, "ENABLE_AGENT_DISCOVERY", False, raising=False)

    customer_lines = "\n".join(
        [
            "Lim Siang Huat — https://limsianghuat.com",
            "FoodXervices Inc — https://www.foodxervices.com",
            "Bidfood Singapore — https://bidfood.com.sg",
            "Makoto-Ya — https://makoto-ya.sg",
            "GJH Global — https://www.gjhglobal.com",
        ]
    )

    state = {
        "messages": [HumanMessage(content=customer_lines)],
        "icp": {},
        "company_profile": {"industries": ["logistics"]},
        "company_profile_confirmed": True,
        "icp_profile_confirmed": True,
        "icp_profile_summary_sent": True,
        "awaiting_icp_profile_confirmation": False,
        "icp_profile_pending": False,
    }

    out = await presdr.icp_node(state)  # type: ignore[arg-type]

    assert out.get("awaiting_icp_profile_confirmation") is True
    assert out.get("icp_profile_pending") is True
    assert not out.get("icp_profile_confirmed")
    last_ai = next((msg for msg in reversed(out.get("messages") or []) if isinstance(msg, AIMessage)), None)
    assert last_ai is not None
    assert "icp profile" in (last_ai.content or "").lower()


@pytest.mark.asyncio
async def test_icp_update_honors_explicit_icp_even_when_company_pending(monkeypatch):
    import app.pre_sdr_graph as presdr

    async def fake_extract_icp(text: str):
        return presdr.ICPProfileUpdate(industries=["wholesale"])

    monkeypatch.setattr(presdr, "extract_icp_profile_update", fake_extract_icp)
    monkeypatch.setattr(presdr, "_persist_icp_profile_sync", lambda *args, **kwargs: None)

    state = {
        "messages": [HumanMessage(content="add wholesale to industry of icp profile")],
        "icp": {"seeds_list": [{}] * 5},
        "icp_profile": {"industries": ["food service"]},
        "icp_profile_summary_sent": True,
        "awaiting_icp_profile_confirmation": True,
        "last_profile_prompt_type": "icp",
        "company_profile_pending": True,
    }

    out = await presdr.icp_node(state)  # type: ignore[arg-type]

    assert out.get("awaiting_icp_profile_confirmation") is True
    assert not out.get("company_profile_pending")


def test_company_pending_clears_after_confirmation(monkeypatch):
    import app.pre_sdr_graph as presdr

    monkeypatch.setattr(presdr, "_persist_company_profile_sync", lambda *args, **kwargs: None)

    state = {
        "messages": [],
        "company_profile": {"industries": ["logistics"]},
        "icp": {},
    }

    presdr._prompt_profile_confirmation(state)
    assert state.get("company_profile_pending") is True

    state["messages"] = presdr.add_messages(
        state.get("messages") or [], [HumanMessage(content="confirm profile")]
    )

    presdr._handle_profile_confirmation_gate(state, "confirm profile")

    assert state.get("company_profile_pending") is False
    assert state.get("awaiting_profile_confirmation") is False


@pytest.mark.asyncio
async def test_company_pending_reopens_on_company_update_request(monkeypatch):
    import app.pre_sdr_graph as presdr

    async def fake_extract_company(text: str):
        return presdr.CompanyProfileUpdate(industries=["automation"])

    monkeypatch.setattr(presdr, "extract_company_profile_update", fake_extract_company)

    state = {
        "messages": [HumanMessage(content="add automation to company profile industries")],
        "icp": {"seeds_list": [{}] * 5},
        "company_profile": {"industries": ["logistics"]},
        "company_profile_confirmed": True,
        "company_profile_pending": False,
        "icp_profile": {"industries": ["logistics"]},
        "icp_profile_summary_sent": True,
    }

    out = await presdr.icp_node(state)  # type: ignore[arg-type]

    assert out.get("company_profile_pending") is True
    assert out.get("awaiting_profile_confirmation") is True


@pytest.mark.asyncio
async def test_icp_prompt_sent_when_seed_domains_missing(monkeypatch):
    import app.pre_sdr_graph as presdr

    monkeypatch.setattr(presdr, "ENABLE_AGENT_DISCOVERY", True, raising=False)
    monkeypatch.setattr(presdr, "_agent_icp_synth", lambda payload: {"icp_profile": {}}, raising=False)

    state = {
        "messages": [],
        "icp": {
            "website_url": "https://example.com",
            "seeds_list": [{"seed_name": f"Seed {i}", "domain": ""} for i in range(5)],
        },
    }

    await presdr._maybe_bootstrap_icp_profile_from_seeds(state)

    assert state.get("awaiting_icp_profile_confirmation") is True
    assert state.get("icp_profile_pending") is True
    summary = ((state.get("icp_profile") or {}).get("summary") or "").lower()
    assert "seed" in summary


@pytest.mark.asyncio
async def test_confirm_profile_requires_icp_keyword_when_pending(monkeypatch):
    import app.pre_sdr_graph as presdr

    monkeypatch.setattr(presdr, "_persist_icp_profile_sync", lambda *args, **kwargs: None)

    seeds = [{"seed_name": f"Seed {i}", "domain": f"seed{i}.com"} for i in range(5)]
    state = {
        "messages": [HumanMessage(content="confirm profile")],
        "icp": {"seeds_list": seeds},
        "icp_profile": {"industries": ["food service"]},
        "icp_profile_summary_sent": True,
        "awaiting_icp_profile_confirmation": True,
        "icp_profile_pending": True,
        "company_profile": {"industries": ["logistics"]},
        "company_profile_confirmed": True,
        "company_profile_pending": False,
    }

    out = await presdr.icp_node(state)  # type: ignore[arg-type]

    assert not out.get("icp_profile_confirmed")
    assert out.get("awaiting_icp_profile_confirmation") is True
    last_ai = next((msg for msg in reversed(out.get("messages") or []) if isinstance(msg, AIMessage)), None)
    assert last_ai is not None
    assert "confirm icp profile" in (last_ai.content or "").lower()


@pytest.mark.asyncio
async def test_confirm_icp_profile_keyword_unlocks_discovery(monkeypatch):
    import app.pre_sdr_graph as presdr

    monkeypatch.setattr(presdr, "_persist_icp_profile_sync", lambda *args, **kwargs: None)

    seeds = [{"seed_name": f"Seed {i}", "domain": f"seed{i}.com"} for i in range(5)]
    state = {
        "messages": [HumanMessage(content="confirm icp profile")],
        "icp": {"seeds_list": seeds},
        "icp_profile": {"industries": ["food service"]},
        "icp_profile_summary_sent": True,
        "awaiting_icp_profile_confirmation": True,
        "icp_profile_pending": True,
        "company_profile": {"industries": ["logistics"]},
        "company_profile_confirmed": True,
        "company_profile_pending": False,
    }

    out = await presdr.icp_node(state)  # type: ignore[arg-type]

    assert out.get("icp_profile_confirmed") is True
    assert not out.get("awaiting_icp_profile_confirmation")
    assert not out.get("icp_profile_pending")


@pytest.mark.asyncio
async def test_confirm_profile_skips_duplicate_website_prompt(monkeypatch):
    import app.pre_sdr_graph as presdr

    monkeypatch.setattr(presdr, "_persist_company_profile_sync", lambda *args, **kwargs: None)
    monkeypatch.setattr(presdr, "ENABLE_ICP_INTAKE", True, raising=False)

    state = {
        "messages": [
            AIMessage(content="Here’s the latest company profile snapshot."),
            HumanMessage(content="confirm profile"),
        ],
        "company_profile": {
            "industries": ["logistics"],
            "website_url": "https://example.com",
        },
        "icp": {},
        "site_profile_summary_sent": True,
        "awaiting_profile_confirmation": True,
    }

    out = await presdr.icp_node(state)  # type: ignore[arg-type]

    msgs = out.get("messages") or []
    last_ai = next((msg for msg in reversed(msgs) if isinstance(msg, AIMessage)), None)
    assert last_ai is not None
    content = (last_ai.content or "").lower()
    assert "list 5" in content
    assert "website url" not in content


def test_icp_manual_edit_updates_memory_snapshot(monkeypatch):
    import app.pre_sdr_graph as presdr

    def fake_extract_icp(text: str):
        return presdr.ICPProfileUpdate(industries=["automation"])

    monkeypatch.setattr(presdr, "extract_icp_profile_update_sync", fake_extract_icp)

    state = {
        "messages": [HumanMessage(content="add automation to the icp industries")],
        "icp": {"seeds_list": [{}] * 5},
        "icp_profile": {"industries": ["logistics"]},
        "icp_profile_summary_sent": True,
        "awaiting_icp_profile_confirmation": True,
        "short_term_memory": {"semantic": {}},
    }

    result = presdr._handle_icp_profile_confirmation_gate(state, "add automation to the icp industries")

    assert result == "prompted"
    semantic = (state.get("short_term_memory") or {}).get("semantic") or {}
    snapshot = semantic.get("icp_profile") or {}
    assert "automation" in snapshot.get("industries", [])


@pytest.mark.asyncio
async def test_icp_recap_progress_lists_completed_steps():
    import app.pre_sdr_graph as presdr

    seeds = [{"seed_name": f"Seed {i}", "domain": f"seed{i}.com"} for i in range(5)]
    state = {
        "messages": [HumanMessage(content="recap")],
        "icp": {
            "website_url": "https://example.com",
            "seeds_list": seeds,
            "lost_churned": [],
            "industries": ["SaaS"],
            "employees_min": 10,
            "employees_max": 200,
            "geos": ["SG", "SEA"],
            "integrations_required": ["HubSpot"],
            "acv_usd": 18000,
            "cycle_weeks_min": 4,
            "cycle_weeks_max": 8,
            "price_floor_usd": 8000,
            "champion_titles": ["RevOps Lead"],
            "triggers": ["Hiring RevOps"],
        },
        "company_profile_confirmed": True,
    }

    out = await presdr.icp_node(state)  # type: ignore[arg-type]

    last_ai = next((msg for msg in reversed(out.get("messages") or []) if isinstance(msg, AIMessage)), None)
    assert last_ai is not None
    content = (last_ai.content or "").lower()
    assert "captured so far" in content
    assert "company profile" in content
    assert "best customers" in content


@pytest.mark.asyncio
async def test_icp_recap_handles_missing_inputs():
    import app.pre_sdr_graph as presdr

    state = {
        "messages": [HumanMessage(content="recap please")],
        "icp": {},
    }

    out = await presdr.icp_node(state)  # type: ignore[arg-type]

    last_ai = next((msg for msg in reversed(out.get("messages") or []) if isinstance(msg, AIMessage)), None)
    assert last_ai is not None
    content = (last_ai.content or "").lower()
    assert "website" in content
    assert "best customers" in content


@pytest.mark.asyncio
async def test_progress_recap_emitted_after_website_capture(monkeypatch):
    import app.pre_sdr_graph as presdr

    async def fake_bootstrap(state, url, *, force_refresh=False):
        return None

    monkeypatch.setattr(presdr, "_maybe_bootstrap_profile_from_site", fake_bootstrap)

    state = {
        "messages": [HumanMessage(content="https://example.com")],
        "icp": {},
    }

    out = await presdr.icp_node(state)  # type: ignore[arg-type]

    recap_msg = next(
        (msg for msg in reversed(out.get("messages") or []) if isinstance(msg, AIMessage) and "progress recap" in (msg.content or "").lower()),
        None,
    )
    assert recap_msg is not None
    assert "company profile" in (recap_msg.content or "").lower()


@pytest.mark.asyncio
async def test_progress_recap_emitted_after_seed_collection(monkeypatch):
    import app.pre_sdr_graph as presdr

    async def fake_bootstrap_icp(state):
        return None

    monkeypatch.setattr(presdr, "_maybe_bootstrap_icp_profile_from_seeds", fake_bootstrap_icp)

    seeds_text = "\n".join([f"Seed {i} — https://seed{i}.com" for i in range(5)])
    state = {
        "messages": [HumanMessage(content=seeds_text)],
        "icp": {"website_url": "https://example.com"},
        "company_profile_confirmed": True,
    }

    out = await presdr.icp_node(state)  # type: ignore[arg-type]

    recap_msg = next(
        (msg for msg in reversed(out.get("messages") or []) if isinstance(msg, AIMessage) and "progress recap" in (msg.content or "").lower()),
        None,
    )
    assert recap_msg is not None
    content = (recap_msg.content or "").lower()
    assert "best customers" in content
    assert "company profile" in content


@pytest.mark.asyncio
async def test_progress_recap_sent_on_initial_seed_prompt():
    import app.pre_sdr_graph as presdr

    state = {
        "messages": [HumanMessage(content="start lead gen")],
        "icp": {},
    }

    out = await presdr.icp_node(state)  # type: ignore[arg-type]

    recap_msg = next(
        (msg for msg in reversed(out.get("messages") or []) if isinstance(msg, AIMessage) and "progress recap" in (msg.content or "").lower()),
        None,
    )
    assert recap_msg is not None
    content = (recap_msg.content or "").lower()
    assert "progress recap" in content
    assert "company profile" in content or "haven’t captured enough" in content or "haven't captured enough" in content


def test_seed_submission_persists_intake_once(monkeypatch):
    import app.pre_sdr_graph as presdr

    monkeypatch.setattr(presdr, "ENABLE_ICP_INTAKE", True, raising=False)

    saved: list[dict] = []

    def fake_save(tid: int, submitted_by: str, payload: dict):
        saved.append({"tid": tid, "payload": payload})

    monkeypatch.setattr(presdr, "_icp_save_intake", fake_save)

    state = {"tenant_id": 77, "icp": {"website_url": "https://example.com"}}
    seeds = [
        {"seed_name": "Acme", "domain": "acme.com"},
        {"seed_name": "Beta", "domain": "beta.com"},
    ]

    presdr._persist_customer_seeds(state, seeds)
    presdr._persist_customer_seeds(state, seeds)

    assert len(saved) == 1
    assert saved[0]["tid"] == 77
    stored = saved[0]["payload"].get("seeds") or []
    assert any(s.get("seed_name") == "Acme" for s in stored)


@pytest.mark.asyncio
async def test_icp_bootstrap_persists_unconfirmed_profile(monkeypatch):
    import app.pre_sdr_graph as presdr

    monkeypatch.setattr(presdr, "ENABLE_AGENT_DISCOVERY", True, raising=False)

    def fake_agent(payload):
        return {"icp_profile": {"industries": ["SaaS"]}}

    monkeypatch.setattr(presdr, "_agent_icp_synth", fake_agent)

    captured: dict = {}

    def fake_persist(state, profile, confirmed=None, seed_urls=None):
        captured["profile"] = dict(profile or {})
        captured["confirmed"] = confirmed
        captured["seed_urls"] = seed_urls

    monkeypatch.setattr(presdr, "_persist_icp_profile_sync", fake_persist)

    state = {
        "tenant_id": 55,
        "icp": {
            "seeds_list": [
                {"seed_name": "Acme", "domain": "acme.com"},
                {"seed_name": "Beta", "domain": "beta.com"},
                {"seed_name": "Gamma", "domain": "gamma.com"},
                {"seed_name": "Delta", "domain": "delta.com"},
                {"seed_name": "Epsilon", "domain": "epsilon.com"},
            ]
        },
    }

    await presdr._maybe_bootstrap_icp_profile_from_seeds(state)

    assert captured.get("profile", {}).get("industries") == ["SaaS"]
    assert captured.get("confirmed") is False
    assert len(captured.get("seed_urls") or []) >= 5
