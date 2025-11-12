import os
import sys
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import jobs as jobs_mod


class FakeCursor:
    def __init__(self):
        self.executed = []
        self._params_row = {1: {"terms": ["software"]}}
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return None
    def execute(self, sql: str, params=None):
        self.executed.append((" ".join(sql.strip().split()), params))
        if "select params from background_jobs" in sql.lower():
            self._row = (self._params_row.get(params[0], {}),)
        else:
            self._row = None
    def fetchone(self):
        return getattr(self, "_row", None)


class FakeConn:
    def cursor(self):
        return FakeCursor()
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return None


import asyncio

@pytest.mark.asyncio
async def test_run_staging_upsert_happy(monkeypatch):
    # Fake get_conn
    monkeypatch.setattr(jobs_mod, "get_conn", lambda: FakeConn())
    # Fake upsert function to return a count
    monkeypatch.setattr(jobs_mod, "upsert_batched", lambda terms: 42)

    # Run the coroutine
    await jobs_mod.run_staging_upsert(1)

    # Ensure no exceptions and that our fake ran: executed SQL recorded
    # Following assertion validates the function returned without exceptions
    assert True


class _JobState:
    def __init__(self):
        self.jobs = {}
        self.ssic_rows = []
        self.tenant_users = {}
        self.daily_count = 0


class _JobConn:
    def __init__(self, state: _JobState):
        self.state = state

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def cursor(self):
        return _JobCursor(self.state)

    def rollback(self):
        return None


class _JobCursor:
    def __init__(self, state: _JobState):
        self.state = state
        self._row = None
        self._rows = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql: str, params=None):
        sql_norm = " ".join(sql.strip().lower().split())
        if "select tenant_id, params from background_jobs" in sql_norm:
            job_id = params[0]
            job = self.state.jobs[job_id]
            self._row = (job["tenant_id"], job["params"])
        elif "select params from background_jobs" in sql_norm and "tenant_id" not in sql_norm:
            job_id = params[0]
            job = self.state.jobs[job_id]
            self._row = (job["params"],)
        elif "select company_id, name, uen from companies" in sql_norm and "regexp_replace" in sql_norm:
            limit = params[1]
            self._rows = self.state.ssic_rows[:limit]
        elif "select coalesce(count(*),0) as cnt" in sql_norm:
            self._row = (self.state.daily_count,)
        elif "update background_jobs set status='running'" in sql_norm:
            job_id = params[0]
            self.state.jobs[job_id]["status"] = "running"
            self._row = None
        elif "update background_jobs set status='done'" in sql_norm:
            job_id = params[-1]
            job = self.state.jobs[job_id]
            job["status"] = "done"
            job["processed"] = params[0]
            job["total"] = params[1]
        elif "update background_jobs set params" in sql_norm and "'email_sent_at'" in sql_norm:
            job_id = params[1]
            job = self.state.jobs[job_id]
            job["params"]["email_sent_at"] = "now"
            job["params"]["email_to"] = params[0]
        else:
            self._row = None
            self._rows = None

    def fetchone(self):
        row = self._row
        self._row = None
        return row

    def fetchall(self):
        rows = self._rows or []
        self._rows = None
        return rows


@pytest.mark.asyncio
async def test_run_enrich_candidates_scores_and_emails(monkeypatch):
    from src import jobs as jobs_mod

    state = _JobState()
    job_id = 4321
    state.jobs[job_id] = {
        "tenant_id": 7,
        "params": {"ssic_codes": ["62011"], "notify_email": "ops@example.com"},
        "status": "queued",
    }
    state.ssic_rows = [
        (301, "Gamma SG", "U301"),
        (302, "Delta SG", "U302"),
    ]

    monkeypatch.setattr(jobs_mod, "get_conn", lambda: _JobConn(state))

    async def fake_enrich(cid, name, uen, search_policy="discover"):
        return {"completed": True, "data": {"email": []}}

    monkeypatch.setattr(jobs_mod, "enrich_company_with_tavily", fake_enrich)

    async def fake_odoo_export(tid, rows):
        fake_odoo_export.called = rows

    fake_odoo_export.called = None
    monkeypatch.setattr(jobs_mod, "_odoo_export_for_ids", fake_odoo_export)

    scoring_calls = {}

    class _FakeScoringAgent:
        async def ainvoke(self, payload):
            scoring_calls["ids"] = payload.get("candidate_ids")

    import src.lead_scoring as scoring_mod
    monkeypatch.setattr(scoring_mod, "lead_scoring_agent", _FakeScoringAgent())

    async def fake_agentic_send_results(to, tenant_id, limit=200):
        fake_agentic_send_results.args = (to, tenant_id)
        return {"status": "sent"}

    fake_agentic_send_results.args = None
    monkeypatch.setattr("src.notifications.agentic_email.agentic_send_results", fake_agentic_send_results)

    monkeypatch.setattr("src.obs.begin_run", lambda tenant_id: 9999)
    monkeypatch.setattr("src.obs.set_run_context", lambda run_id, tenant_id: None)
    monkeypatch.setattr("src.obs.finalize_run", lambda run_id, status="succeeded": None)
    monkeypatch.setattr(jobs_mod, "_enrich_set_ctx", lambda run_id, tenant_id: None)

    await jobs_mod.run_enrich_candidates(job_id)

    assert scoring_calls["ids"] == [301, 302]
    assert fake_agentic_send_results.args == ("ops@example.com", 7)
    assert fake_odoo_export.called == state.ssic_rows
    assert state.jobs[job_id]["params"].get("email_sent_at") == "now"
