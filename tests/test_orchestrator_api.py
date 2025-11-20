import uuid

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from app.main import app
from app.auth import require_auth
from my_agent.utils import nodes


class _StubAgent:
    def __init__(self, result):
        self._result = result

    def invoke(self, state):
        return self._result


@pytest.fixture(autouse=True)
def auth_override():
    async def _fake_auth(request: Request):
        request.state.tenant_id = 1
        return {"sub": "test-user", "tenant_id": 1}

    app.dependency_overrides[require_auth] = _fake_auth
    yield
    app.dependency_overrides.pop(require_auth, None)


@pytest.fixture(autouse=True)
def stub_agents(monkeypatch):
    monkeypatch.setattr(nodes, "call_llm_json", lambda prompt, fallback: fallback)
    monkeypatch.setattr(nodes, "normalize_agent", _StubAgent({"normalized_records": [{}]}))
    monkeypatch.setattr(nodes, "icp_refresh_agent", _StubAgent({"candidate_ids": [1, 2]}))
    monkeypatch.setattr(nodes, "plan_top10_with_reasons", lambda icp_profile, tenant_id=None: [{"company_id": 1}])
    monkeypatch.setattr(nodes, "lead_scoring_agent", _StubAgent({"lead_scores": [{"company_id": 1, "score": 75}], "lead_features": []}))
    monkeypatch.setattr(nodes, "enrich_company_with_tavily", lambda cid, search_policy="require_existing": None)
    monkeypatch.setattr(nodes, "enqueue_web_discovery_bg_enrich", lambda tenant_id, company_ids: {"job_id": 123})
    yield


def test_orchestrator_api_basic():
    client = TestClient(app)
    client.headers.update({"x-tenant-id": "1"})
    payload = {
        "input": "run enrichment",
        "thread_id": str(uuid.uuid4()),
        "role": "user",
        "messages": [],
        "icp_payload": {"industries": ["software"]},
    }
    resp = client.post("/api/orchestrations", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["thread_id"]
    assert "status" in data
    assert isinstance(data.get("status_history"), list)
    assert isinstance(data["output"], str)

    status_resp = client.get(f"/api/orchestrations/{data['thread_id']}")
    assert status_resp.status_code == 200
    snapshot = status_resp.json()
    assert snapshot["thread_id"] == data["thread_id"]
    assert "status" in snapshot
    assert isinstance(snapshot.get("status_history"), list)
