import os
import sys
from types import SimpleNamespace

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test")
os.environ.setdefault("ENABLE_ICP_INTAKE", "true")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient


def test_icp_suggestions_payload_includes_negative_and_targeting(monkeypatch):
    # Import after setting env so /icp router mounts
    from app.main import app
    import app.icp_endpoints as ep

    # Bypass auth and force a tenant in header
    async def fake_auth(request):
        return {"user_id": "u", "roles": ["ops"]}

    monkeypatch.setattr(ep, "require_auth", fake_auth)

    # Provide answers for negative ICPderivation
    class _Cur:
        def execute(self, sql, args=None):
            self._row = ({"lost_or_churned": [{"reason": "budget too low"}]},)

        def fetchone(self):
            return self._row

    class _Conn:
        def cursor(self):
            return _Cur()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_conn():
        return _Conn()

    monkeypatch.setattr(ep, "get_conn", fake_conn)

    # Stub suggestions and helpers
    def fake_suggestions(tid, limit=5):
        return [{"id": "ssic:46900", "title": "SSIC 46900", "evidence_count": 2}]

    def fake_pack(card):
        return {"ssic_filters": ["46900"], "technographic_filters": ["has_crm"], "pitch": "Fast ROI"}

    def fake_neg(answers):
        return [{"theme": "budget_too_low", "count": 1, "reason": "derived"}]

    monkeypatch.setattr(ep, "generate_suggestions", fake_suggestions)
    monkeypatch.setattr(ep, "build_targeting_pack", fake_pack)
    monkeypatch.setattr(ep, "_derive_negative_icp_flags", fake_neg)

    client = TestClient(app)
    r = client.get("/icp/suggestions", headers={"X-Tenant-ID": "1", "Authorization": "Bearer test"})
    assert r.status_code == 200
    js = r.json()
    assert isinstance(js, list) and js
    one = js[0]
    assert one["targeting_pack"]["ssic_filters"] == ["46900"]
    assert one["negative_icp"][0]["theme"] == "budget_too_low"


def test_icp_intake_returns_job_id(monkeypatch):
    from app.main import app
    import app.icp_endpoints as ep

    async def fake_auth(request):
        return {"user_id": "u", "roles": ["ops"]}

    monkeypatch.setattr(ep, "require_auth", fake_auth)

    # Enqueue and run job stubs
    def fake_enqueue(tid):
        return {"job_id": 123}

    async def fake_run(job_id: int):
        return None

    monkeypatch.setattr(ep, "enqueue_icp_intake_process", fake_enqueue)
    monkeypatch.setattr(ep, "run_icp_intake_process", fake_run)

    client = TestClient(app)
    payload = {"answers": {"website": "https://acme.com"}, "seeds": [{"seed_name": "Acme", "domain": "acme.com"}]}
    r = client.post("/icp/intake", json=payload, headers={"X-Tenant-ID": "1", "Authorization": "Bearer test"})
    assert r.status_code == 200
    js = r.json()
    assert js.get("status") == "queued"
    assert js.get("job_id") == 123


def test_icp_accept_persists_rule(monkeypatch):
    from app.main import app
    import app.icp_endpoints as ep

    async def fake_auth(request):
        return {"user_id": "u", "roles": ["ops"]}

    monkeypatch.setattr(ep, "require_auth", fake_auth)

    # Capture saved payload
    saved = {}

    def fake_save(tid, payload, name="Default"):
        saved["tid"] = tid
        saved["payload"] = payload
        saved["name"] = name

    monkeypatch.setattr(ep, "_save_icp_rule", fake_save)

    from fastapi.testclient import TestClient
    client = TestClient(app)
    r = client.post("/icp/accept", json={"suggestion_id": "ssic:46900"}, headers={"X-Tenant-ID": "7", "Authorization": "Bearer test"})
    assert r.status_code == 200
    assert saved["tid"] == 7
    assert saved["payload"]["ssic_codes"] == ["46900"]


def test_icp_suggestions_enforces_auth(monkeypatch):
    from app.main import app
    import app.icp_endpoints as ep

    async def fake_auth_fail(request):
        from fastapi import HTTPException
        raise HTTPException(status_code=401, detail="no auth")

    monkeypatch.setattr(ep, "require_auth", fake_auth_fail)

    client = TestClient(app)
    r = client.get("/icp/suggestions")
    assert r.status_code == 401

