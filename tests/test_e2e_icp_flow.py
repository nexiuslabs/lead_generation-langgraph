import os
import sys
import pytest

os.environ.setdefault("OPENAI_API_KEY", "test")
os.environ.setdefault("ENABLE_ICP_INTAKE", "true")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient


@pytest.mark.anyio
async def test_e2e_intake_suggest_accept(monkeypatch):
    from app.main import app
    import app.icp_endpoints as ep

    # Auth stub
    async def ok_auth(request):
        return {"user_id": "qa", "roles": ["ops"]}

    monkeypatch.setattr(ep, "require_auth", ok_auth)

    # Intake enqueue
    def fake_enqueue(tid):
        return {"job_id": 456}

    async def fake_run(job_id: int):
        # no-op run
        return None

    monkeypatch.setattr(ep, "enqueue_icp_intake_process", fake_enqueue)
    monkeypatch.setattr(ep, "run_icp_intake_process", fake_run)

    # Suggestions + helpers
    def fake_suggestions(tid, limit=5):
        return [{"id": "ssic:62010", "title": "SSIC 62010", "evidence_count": 3}]

    def fake_pack(card):
        return {"ssic_filters": ["62010"], "technographic_filters": ["has_analytics"], "pitch": "Quick start"}

    def fake_neg(answers):
        return [{"theme": "budget_too_low", "count": 1, "reason": "derived"}]

    monkeypatch.setattr(ep, "generate_suggestions", fake_suggestions)
    monkeypatch.setattr(ep, "build_targeting_pack", fake_pack)
    monkeypatch.setattr(ep, "_derive_negative_icp_flags", fake_neg)

    # Accept capture
    saved = {}

    def fake_save(tid, payload, name="Default"):
        saved["tid"] = tid
        saved["payload"] = payload
        saved["name"] = name

    monkeypatch.setattr(ep, "_save_icp_rule", fake_save)

    client = TestClient(app)

    # Intake
    payload = {"answers": {"website": "https://acme.io"}, "seeds": [{"seed_name": "Acme", "domain": "acme.io"}]}
    r1 = client.post("/icp/intake", json=payload, headers={"Authorization": "Bearer test", "X-Tenant-ID": "9"})
    assert r1.status_code == 200
    assert r1.json().get("job_id") == 456

    # Suggestions
    r2 = client.get("/icp/suggestions", headers={"Authorization": "Bearer test", "X-Tenant-ID": "9"})
    assert r2.status_code == 200
    js = r2.json()
    assert js and js[0]["targeting_pack"]["ssic_filters"] == ["62010"]
    assert js[0]["negative_icp"][0]["theme"] == "budget_too_low"

    # Accept
    r3 = client.post("/icp/accept", json={"suggestion_id": "ssic:62010"}, headers={"Authorization": "Bearer test", "X-Tenant-ID": "9"})
    assert r3.status_code == 200
    assert saved["tid"] == 9
    assert saved["payload"]["ssic_codes"] == ["62010"]

