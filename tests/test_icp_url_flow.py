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

    # 3) Invoke icp_node with only a website message
    state = {"messages": [HumanMessage(content="https://nexiuslabs.com")]}  # minimal GraphState
    out = await presdr.icp_node(state)  # type: ignore[arg-type]

    # 4) Should set focus to 'seeds' and ask for seed customers next
    assert out.get("icp_last_focus") == "seeds"
    msgs = out.get("messages") or []
    assert isinstance(msgs[-1], AIMessage)
    assert "List 5–15" in (msgs[-1].content or "")

