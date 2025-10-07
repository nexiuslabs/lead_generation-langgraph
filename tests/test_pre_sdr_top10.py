import asyncio
import sys
import types

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app import pre_sdr_graph


@pytest.mark.parametrize(
    "messages, expected",
    [
        ([AIMessage(content="Top-10 lookalikes (with why):")], True),
        ([AIMessage(content="Top‑10 lookalikes — sample")], True),
        ([AIMessage(content="No shortlist yet."), HumanMessage(content="ok")], False),
        ([], False),
    ],
)
def test_top10_preview_was_sent(messages, expected):
    state = {"messages": messages}
    assert pre_sdr_graph._top10_preview_was_sent(state) is expected


def test_regenerate_top10_if_missing_requires_preview(monkeypatch):
    state = {"messages": [AIMessage(content="No shortlist yet.")], "icp_profile": {}}

    called = False

    def fake_persist(tid, top):
        nonlocal called
        called = True

    async def fake_to_thread(func, *args, **kwargs):  # pragma: no cover - should not run
        raise AssertionError("plan_top10_with_reasons should not run")

    monkeypatch.setattr(pre_sdr_graph, "_persist_top10_preview", fake_persist)
    monkeypatch.setattr(pre_sdr_graph.asyncio, "to_thread", fake_to_thread)

    result = asyncio.run(pre_sdr_graph._regenerate_top10_if_missing(state, None))

    assert result == []
    assert state.get("agent_top10") is None
    assert called is False


def test_regenerate_top10_if_missing_rebuilds_shortlist(monkeypatch):
    ai_message = AIMessage(content="Top‑10 lookalikes (with why):")
    icp_profile = {"industries": ["fmcg"]}
    state = {"messages": [ai_message], "icp_profile": icp_profile}

    captured_persist = {}

    def fake_persist(tid, top):
        captured_persist["tid"] = tid
        captured_persist["top"] = top

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    def fake_plan(icp_prof, tenant_id):
        return [
            {"domain": "example.com", "score": 42, "why": "match", "snippet": "sample"}
        ]

    monkeypatch.setattr(pre_sdr_graph, "_persist_top10_preview", fake_persist)
    monkeypatch.setattr(pre_sdr_graph.asyncio, "to_thread", fake_to_thread)
    stub = types.SimpleNamespace(plan_top10_with_reasons=fake_plan)
    monkeypatch.setitem(sys.modules, "src.agents_icp", stub)

    result = asyncio.run(pre_sdr_graph._regenerate_top10_if_missing(state, 7))

    assert result == [
        {"domain": "example.com", "score": 42, "why": "match", "snippet": "sample"}
    ]
    assert state["agent_top10"] == result
    assert captured_persist == {
        "tid": 7,
        "top": result,
    }

