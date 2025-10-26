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
    monkeypatch.setattr(worker, "_claim_one", lambda conn, job_type: asyncio.get_running_loop().create_task(_fake_claim_one(conn, job_type)))

    # Run one spawn cycle and wait any spawned tasks to complete
    await worker._spawn_until_full(fake_pool)
    if worker._tasks:
        await asyncio.gather(*list(worker._tasks))

    # Assert the Next‑40 job runner was invoked with our job id
    assert called["job_id"] == 789

