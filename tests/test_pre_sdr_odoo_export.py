from typing import Any, Dict, Optional

import pytest

from app import pre_sdr_graph


class _DummyOdooStore:
    instances: list["_DummyOdooStore"] = []

    def __init__(self, tenant_id: Optional[int] = None, dsn: Optional[str] = None):
        self.tenant_id = tenant_id
        self.dsn = dsn
        self.upserts: list[Dict[str, Any]] = []
        self.contacts: list[tuple[int, str]] = []
        self.leads: list[tuple[int, float, Optional[str]]] = []
        self.enrichment_merges: list[tuple[int, Dict[str, Any]]] = []
        _DummyOdooStore.instances.append(self)

    async def upsert_company(self, name: str, uen: Optional[str], **fields: Any) -> int:
        self.upserts.append({"name": name, "uen": uen, **fields})
        return 9001

    async def add_contact(self, partner_id: int, email: str) -> None:
        self.contacts.append((partner_id, email))

    async def merge_company_enrichment(self, partner_id: int, data: Dict[str, Any]) -> None:
        self.enrichment_merges.append((partner_id, data))

    async def create_lead_if_high(
        self,
        company_id: int,
        title: str,
        score: float,
        features: Dict[str, Any],
        rationale: str,
        primary_email: Optional[str],
        threshold: float = 0,
    ) -> Optional[int]:
        self.leads.append((company_id, score, primary_email))
        return 111


class _DummyLeadScoringAgent:
    async def ainvoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "lead_scores": [
                {"company_id": 101, "score": 0.88, "rationale": "Great fit"}
            ],
            "lead_features": [],
        }


class _FakeSyncConn:
    def __enter__(self) -> "_FakeSyncConn":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def cursor(self) -> "_FakeSyncConn":
        return self

    def execute(self, *args, **kwargs) -> None:
        return None

    def fetchone(self):
        return None

    def fetchall(self):
        return []


class _FakeAsyncConn:
    async def fetch(self, query: str, ids):
        if "FROM companies" in query:
            cid = ids[0] if isinstance(ids, (list, tuple)) else ids
            return [
                {
                    "company_id": cid,
                    "name": "Example Co",
                    "uen": "UEN123",
                    "industry_norm": "Manufacturing",
                    "employees_est": 120,
                    "revenue_bucket": "5-10M",
                    "incorporation_year": 2010,
                    "website_domain": "example.com",
                }
            ]
        if "FROM lead_emails" in query:
            cid = ids[0] if isinstance(ids, (list, tuple)) else ids
            return [{"company_id": cid, "email": "contact@example.com"}]
        if "FROM lead_scores" in query:
            cid = ids[0] if isinstance(ids, (list, tuple)) else ids
            return [{"company_id": cid, "score": 0.88, "rationale": "Great fit"}]
        return []


class _FakeAcquire:
    def __init__(self, conn: _FakeAsyncConn):
        self._conn = conn

    async def __aenter__(self) -> _FakeAsyncConn:
        return self._conn

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class _FakePool:
    def __init__(self, conn: _FakeAsyncConn):
        self._conn = conn

    def acquire(self) -> _FakeAcquire:
        return _FakeAcquire(self._conn)


@pytest.mark.asyncio
async def test_enrich_node_attempts_odoo_export(monkeypatch):
    _DummyOdooStore.instances.clear()

    monkeypatch.setattr(pre_sdr_graph, "OdooStore", _DummyOdooStore)

    monkeypatch.setattr(pre_sdr_graph, "lead_scoring_agent", _DummyLeadScoringAgent())

    async def _fake_enrich_company_with_tavily(cid, name, uen, search_policy="require_existing"):
        return {"completed": True}

    monkeypatch.setattr(pre_sdr_graph, "enrich_company_with_tavily", _fake_enrich_company_with_tavily)

    async def _fake_score_node(state):
        state["score_node_called"] = True
        return state

    monkeypatch.setattr(pre_sdr_graph, "score_node", _fake_score_node)

    monkeypatch.setattr(pre_sdr_graph, "_enqueue_next40_if_applicable", lambda _state: None)
    monkeypatch.setattr(pre_sdr_graph, "_save_icp_rule_sync", lambda *a, **k: None)
    monkeypatch.setattr(pre_sdr_graph, "_announce_completed_bg_jobs", lambda *a, **k: None)

    async def _fake_resolve_tenant(state):
        return state.get("tenant_id")

    monkeypatch.setattr(pre_sdr_graph, "_resolve_tenant_id_for_write", _fake_resolve_tenant)
    monkeypatch.setattr(pre_sdr_graph, "get_conn", lambda: _FakeSyncConn())

    async def _fake_get_pg_pool():
        return _FakePool(_FakeAsyncConn())

    monkeypatch.setattr(pre_sdr_graph, "get_pg_pool", _fake_get_pg_pool)

    monkeypatch.setenv("ODOO_EXPORT_ENABLED", "1")

    state = {
        "messages": [],
        "agent_top10": [{"domain": "example.com"}],
        "candidates": [
            {"id": 101, "name": "Example Co", "uen": "UEN123", "domain": "example.com"}
        ],
        "icp": {
            "employees_min": 10,
            "employees_max": 500,
            "year_min": 2000,
            "year_max": 2024,
            "revenue_bucket": "5-10M",
        },
        "tenant_id": 42,
        "micro_icp_suggestions": [],
    }

    result = await pre_sdr_graph.enrich_node(state)

    assert result["enrichment_completed"] is True
    assert _DummyOdooStore.instances, "OdooStore should be instantiated"
    store = _DummyOdooStore.instances[0]
    assert store.tenant_id == 42
    assert store.upserts, "Company upsert should be attempted"
    assert store.contacts == [(9001, "contact@example.com")]
    assert store.leads == [(9001, 0.88, "contact@example.com")]
    assert store.enrichment_merges == [(9001, {})]
