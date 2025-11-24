import pytest
from typing import List

from my_agent.utils import nodes
from my_agent.utils.state import OrchestrationState


def _base_state() -> OrchestrationState:
    return {
        "messages": [],
        "entry_context": {"thread_id": "test-thread", "run_mode": "chat_top10", "tenant_id": 77},
        "icp_payload": {"industries": ["F&B"]},
        "profile_state": {"icp_profile": {"industries": ["F&B"]}},
        "discovery": {},
    }


@pytest.mark.asyncio
async def test_plan_top10_records_top_and_next(monkeypatch):
    raw_domains = [
        "site1.com",
        "site2.com",
        "site2.com",
        "site3.com",
        "site4.com",
        "site3.com",
        "site5.com",
        "site6.com",
        "site7.com",
        "site8.com",
        "site9.com",
        "site10.com",
        "site11.com",
        "site12.com",
    ]
    domains: List[str] = []  # type: ignore[assignment]
    for dom in raw_domains:
        if dom not in domains:
            domains.append(dom)

    def fake_plan(_icp, tenant_id=None):
        return [{"domain": d, "score": idx} for idx, d in enumerate(raw_domains, start=1)]

    def fake_ensure(ids, tenant_id):
        return {dom: idx for idx, dom in enumerate(ids, start=100)}

    monkeypatch.setattr(nodes, "plan_top10_with_reasons", fake_plan)
    monkeypatch.setattr(nodes, "_ensure_company_ids_for_domains", fake_ensure)

    state = _base_state()
    out = await nodes.plan_top10(state)
    discovery = out["discovery"]
    assert discovery["top10_domains"] == domains[:10]
    assert discovery["next40_domains"] == domains[10:50]
    assert discovery["top10_ids"] == list(range(100, 110))
    assert discovery["next40_ids"] == list(range(110, 112))


@pytest.mark.asyncio
async def test_export_enqueues_next40_only(monkeypatch):
    captured = {}

    def fake_enqueue(tenant_id, ids, notify_email=None):
        captured["tenant"] = tenant_id
        captured["ids"] = ids
        return {"job_id": 555}

    async def fake_odoo_export(tenant_id, company_ids):
        captured["odoo_tenant"] = tenant_id
        captured["odoo_ids"] = company_ids
        return True

    monkeypatch.setattr(nodes, "enqueue_web_discovery_bg_enrich", fake_enqueue)
    monkeypatch.setattr(nodes, "_export_top10_to_odoo", fake_odoo_export)
    state: OrchestrationState = _base_state()
    state["discovery"] = {"next40_ids": [201, 202, 203]}
    state["enrichment_results"] = [{"company_id": 1, "completed": True}]
    out = await nodes.export_results(state)
    assert out["exports"]["next40_enqueued"] is True
    assert captured["tenant"] == 77
    assert captured["ids"] == [201, 202, 203]
    assert captured["odoo_tenant"] == 77
    assert captured["odoo_ids"] == [1]


@pytest.mark.asyncio
async def test_plan_top10_reuses_cached_list(monkeypatch):
    called = {"ran": False}

    def fail_plan(*_a, **_k):
        called["ran"] = True
        raise AssertionError("should not call planner")

    monkeypatch.setattr(nodes, "plan_top10_with_reasons", fail_plan)
    state = _base_state()
    state["discovery"] = {
        "strategy": "use_cached",
        "top10_details": [{"domain": "alpha.com", "score": 50}],
        "top10_ids": [101],
    }
    out = await nodes.plan_top10(state)
    assert out["status"]["message"] == "Reusing cached discovery candidates"
    assert called["ran"] is False


@pytest.mark.asyncio
async def test_refresh_icp_skips_when_top10_cached():
    state = _base_state()
    state["discovery"] = {"top10_ids": [1, 2, 3]}
    out = await nodes.refresh_icp(state)
    assert out["status"]["message"] == "Using confirmed Top-10 candidates"
