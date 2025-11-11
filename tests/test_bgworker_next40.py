import asyncio
import os

import pytest


@pytest.mark.asyncio
async def test_bgworker_runs_next40(monkeypatch):
    # Import module under test
    import scripts.run_bg_worker as bgmod

    # Capture calls to the job runner
    called = {"job_id": None}

    async def fake_run_web_discovery_bg_enrich(job_id: int) -> None:
        called["job_id"] = int(job_id)
        # Simulate scoring + email completes by touching globals
        fake_run_web_discovery_bg_enrich.invoked = True
    fake_run_web_discovery_bg_enrich.invoked = False

    # Patch the symbol used inside BGWorker._run_job
    monkeypatch.setattr(bgmod, "run_web_discovery_bg_enrich", fake_run_web_discovery_bg_enrich)

    # Provide a fake asyncpg pool/connection used by _spawn_until_full
    class _FakeConn:
        async def execute(self, *args, **kwargs):
            # Accept maintenance SQL and SELECTs without side effects
            return None

        async def fetchval(self, *args, **kwargs):
            return 0

    class _Acquire:
        def __init__(self, conn):
            self._conn = conn

        async def __aenter__(self):
            return self._conn

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class _FakePool:
        def __init__(self, conn):
            self._conn = conn

        def acquire(self):
            return _Acquire(self._conn)

    fake_pool = _FakePool(_FakeConn())

    # Instantiate the worker with small concurrency and interval
    worker = bgmod.BGWorker(dsn="postgresql://test", max_concurrency=1, sweep_interval=0.1)

    # Make the claim call return a single job id, then stop
    claimed = {"done": False}

    async def _fake_claim_one(conn, job_type):
        if not claimed["done"] and job_type == "web_discovery_bg_enrich":
            claimed["done"] = True
            return 789
        return None

    # Patch the instance method on this worker
    monkeypatch.setattr(
        worker,
        "_claim_one",
        lambda conn, job_type: asyncio.get_running_loop().create_task(_fake_claim_one(conn, job_type)),
    )

    # Run one spawn cycle and wait any spawned tasks to complete
    await worker._spawn_until_full(fake_pool)
    if worker._tasks:
        await asyncio.gather(*list(worker._tasks))

    # Assert the Next‑40 job runner was invoked with our job id
    assert called["job_id"] == 789


class _JobState:
    def __init__(self):
        self.jobs = {}
        self.companies = {}


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
        elif "select company_id, name, uen from companies where company_id = any" in sql_norm:
            ids = list(params[0])
            rows = []
            for cid in ids:
                comp = self.state.companies.get(cid, {})
                rows.append((cid, comp.get("name"), comp.get("uen")))
            self._rows = rows
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
            self._row = None
        elif "update background_jobs set params" in sql_norm and "'email_sent_at'" in sql_norm:
            job_id = params[1]
            job = self.state.jobs[job_id]
            job["params"]["email_sent_at"] = "now"
            job["params"]["email_to"] = params[0]
            self._row = None
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
async def test_run_web_discovery_bg_enrich_scores_and_emails(monkeypatch):
    from src import jobs as jobs_mod

    state = _JobState()
    job_id = 900
    state.jobs[job_id] = {
        "tenant_id": 55,
        "params": {"company_ids": [101, 102], "notify_email": "ops@example.com"},
        "status": "queued",
    }
    state.companies = {
        101: {"name": "Alpha Pte Ltd", "uen": "U101"},
        102: {"name": "Beta Pte Ltd", "uen": "U102"},
    }

    monkeypatch.setattr(jobs_mod, "get_conn", lambda: _JobConn(state))

    enrich_calls = []

    async def fake_enrich(cid, name, uen, search_policy="require_existing"):
        enrich_calls.append(cid)
        return {"completed": True, "data": {"email": []}}

    monkeypatch.setattr(jobs_mod, "enrich_company_with_tavily", fake_enrich)

    scoring_calls = {}

    class _FakeScoringAgent:
        async def ainvoke(self, payload):
            scoring_calls["ids"] = payload.get("candidate_ids")

    import src.lead_scoring as scoring_mod
    monkeypatch.setattr(scoring_mod, "lead_scoring_agent", _FakeScoringAgent())

    email_calls = {}

    async def fake_agentic_send_results(to, tenant_id, limit=200):
        email_calls["args"] = (to, tenant_id)
        return {"status": "sent"}

    monkeypatch.setattr("src.notifications.agentic_email.agentic_send_results", fake_agentic_send_results)

    await jobs_mod.run_web_discovery_bg_enrich(job_id)

    assert scoring_calls["ids"] == [101, 102]
    assert email_calls["args"] == ("ops@example.com", 55)
    assert state.jobs[job_id]["params"].get("email_sent_at") == "now"
