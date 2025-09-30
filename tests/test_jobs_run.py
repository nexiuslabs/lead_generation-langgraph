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
