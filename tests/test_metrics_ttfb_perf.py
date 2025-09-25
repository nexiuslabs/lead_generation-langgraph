import os
import sys
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app.main as main_app


class FakeCursor:
    def __init__(self, ttfb_values_ms):
        self._rows = []
        self._one = None
        self._ttfb = ttfb_values_ms

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def execute(self, sql: str, params=None):
        s = " ".join(sql.strip().split()).lower()
        if "from background_jobs where status='queued'" in s:
            self._one = (0,)
        elif "sum(processed)" in s and "from background_jobs" in s:
            self._one = (0,)
        elif "from lead_scores" in s:
            self._one = (0,)
        elif "select processed, extract(epoch" in s:
            # No background jobs in this perf test
            self._rows = []
        elif "from run_event_logs" in s and "ttfb" in s:
            # Simulate high load: 1000 rows with 950 fast (100ms) and 50 slow (400ms)
            self._rows = [(v,) for v in self._ttfb]
        else:
            self._one = None
            self._rows = []

    def fetchone(self):
        return self._one

    def fetchall(self):
        return self._rows


class FakeConn:
    def __init__(self, cur: FakeCursor):
        self._cur = cur

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def cursor(self):
        return self._cur


@pytest.mark.asyncio
async def test_ttfb_p95_under_load_is_below_threshold(monkeypatch):
    # Prepare distribution: 950 values at 100ms, 50 values at 400ms
    ttfb = [100] * 950 + [400] * 50
    cur = FakeCursor(ttfb)

    monkeypatch.setattr(main_app, "get_conn", lambda: FakeConn(cur))

    out = await main_app.metrics({})
    assert out["chat_ttfb_p95_ms"] is not None
    # For n=1000, k=int(0.95*(n-1)) -> 949 -> value 100
    assert int(out["chat_ttfb_p95_ms"]) == 100
    # Acceptance threshold
    assert out["chat_ttfb_p95_ms"] <= 300

