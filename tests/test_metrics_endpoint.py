import os
import sys
import types
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app.main as main_app


class FakeCursor:
    def __init__(self):
        self._rows = []
        self._one = None
        self.description = []
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return None
    def execute(self, sql: str, params=None):
        s = " ".join(sql.strip().split()).lower()
        if "from background_jobs where status='queued'" in s:
            self._one = (3,)
        elif "sum(processed)" in s and "from background_jobs" in s:
            self._one = (1200,)
        elif "from lead_scores" in s:
            self._one = (456,)
        elif "select processed, extract(epoch" in s:
            # two jobs: (600 rows, 60s), (300 rows, 30s)
            self._rows = [(600, 60.0), (300, 30.0)]
        elif "from run_event_logs" in s and "ttfb" in s:
            self._rows = [(100,), (250,), (500,), (50,), (300,)]
        else:
            self._one = None
            self._rows = []
    def fetchone(self):
        return self._one
    def fetchall(self):
        return self._rows


class FakeConn:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return None
    def cursor(self):
        return FakeCursor()


@pytest.mark.asyncio
async def test_metrics_rich(monkeypatch):
    monkeypatch.setattr(main_app, "get_conn", lambda: FakeConn())
    out = await main_app.metrics({})
    assert out["job_queue_depth"] == 3
    assert out["jobs_processed_total"] == 1200
    assert out["lead_scores_total"] == 456
    # rows/min from jobs (600/60*60=600, 300/30*60=600) avg=600
    assert int(out["rows_per_min"]) == 600
    # p95_job_ms from durations [60000,30000] -> pick last -> 60000
    assert int(out["p95_job_ms"]) == 60000
    # ttfb p95 from sorted [50,100,250,300,500] -> index 0.95*4=3 -> 4th value 300
    assert int(out["chat_ttfb_p95_ms"]) == 300

