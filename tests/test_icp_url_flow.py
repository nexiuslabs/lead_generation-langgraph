import asyncio
import os
import sys
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.messages import HumanMessage, AIMessage


@pytest.mark.asyncio
async def test_url_only_input_prompts_seeds_and_no_industry(monkeypatch):
    import app.main as main_app
    import app.pre_sdr_graph as presdr

    # 1) URL/domain input should not produce industry terms
    terms = main_app._collect_industry_terms([HumanMessage(content="https://nexiuslabs.com")])
    assert terms == []

    # 2) Stub out LLM extraction to avoid network calls
    async def fake_extract_update(text: str):
        return presdr.ICPUpdate()  # defaults: no fields extracted

    monkeypatch.setattr(presdr, "extract_update_from_text", fake_extract_update)
    monkeypatch.setattr(presdr, "ENABLE_AGENT_DISCOVERY", False, raising=False)
    monkeypatch.setattr(
        presdr,
        "jina_read",
        lambda url, timeout=8.0: "url: https://nexiuslabs.com\ncontent: NexiusLabs builds lead generation copilots.",
    )

    # 3) Invoke icp_node with only a website message
    state = {"messages": [HumanMessage(content="https://nexiuslabs.com")]}  # minimal GraphState
    out = await presdr.icp_node(state)  # type: ignore[arg-type]

    # 4) Should pause on profile confirmation so the user can edit before seeds
    assert out.get("icp_last_focus") == "profile"
    assert out.get("awaiting_profile_confirmation") is True
    msgs = out.get("messages") or []
    assert isinstance(msgs[-1], AIMessage)
    prompt_text = (msgs[-1].content or "").lower()
    assert "confirm profile" in prompt_text


@pytest.mark.asyncio
async def test_website_input_honored_even_when_seeds_prompted(monkeypatch):
    import app.pre_sdr_graph as presdr

    async def fake_extract_update(text: str):
        return presdr.ICPUpdate()

    monkeypatch.setattr(presdr, "extract_update_from_text", fake_extract_update)
    monkeypatch.setattr(
        presdr,
        "jina_read",
        lambda url, timeout=8.0: "url: https://nexiuslabs.com\ncontent: NexiusLabs builds lead generation copilots.",
    )

    state = {
        "messages": [HumanMessage(content="https://nexiuslabs.com")],
        "icp": {},
        "icp_last_focus": "seeds",
        "ask_counts": {"seeds": 1},
    }
    out = await presdr.icp_node(state)  # type: ignore[arg-type]

    assert out.get("icp_last_focus") == "profile"
    assert out.get("awaiting_profile_confirmation") is True
