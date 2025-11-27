import pytest

from app import lg_entry


class _DummyGraph:
    def __init__(self):
        self.last_state = None

    async def ainvoke(self, state, config=None):
        self.last_state = state
        return {"status": {"message": "ok"}, "status_history": [], "messages": []}


@pytest.mark.asyncio
async def test_handle_turn_propagates_tenant_context(monkeypatch):
    dummy = _DummyGraph()
    monkeypatch.setattr(lg_entry, "ORCHESTRATOR", dummy)
    payload = {
        "input": "hello there",
        "role": "user",
        "tenant_id": 987,
        "user_id": "tester",
        "run_mode": "chat_top10",
    }
    config = {"configurable": {"thread_id": "thread-123"}}

    result = await lg_entry.handle_turn(payload, config=config)

    assert dummy.last_state["entry_context"]["tenant_id"] == 987
    assert dummy.last_state["tenant_id"] == 987
    assert result["status"]["message"] == "ok"
