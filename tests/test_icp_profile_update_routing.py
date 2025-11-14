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
