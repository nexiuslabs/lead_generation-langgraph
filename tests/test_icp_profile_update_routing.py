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
async def test_confirm_profile_prefers_icp_when_icp_pending(monkeypatch):
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

    assert out.get("icp_profile_confirmed") is True
    assert not out.get("awaiting_icp_profile_confirmation")
    assert not out.get("company_profile_pending")
    assert not out.get("awaiting_profile_confirmation")
    last_ai = next((msg for msg in reversed(out.get("messages") or []) if isinstance(msg, AIMessage)), None)
    assert last_ai is not None
    assert "company profile" not in (last_ai.content or "").lower()
