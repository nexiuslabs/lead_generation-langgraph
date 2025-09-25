import os
import sys
from datetime import datetime, timedelta
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app.main as main_app


class FakeCursor:
    def __init__(self, rows_page1, rows_page2):
        self._rows_page1 = rows_page1
        self._rows_page2 = rows_page2
        self._rows = []
        self._one = None
        self.executed_sql = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def execute(self, sql: str, params=None):
        s = " ".join(sql.strip().split())
        self.executed_sql.append(s)
        # Decide which page to return based on presence of afterUpdatedAt/afterId in params
        # First call: params ends with (limit,)
        # Second call: params ends with (..., afterUpdatedAt, afterId, limit)
        if params and len(params) >= 3:
            # treat as page 2
            self._rows = self._rows_page2
        else:
            self._rows = self._rows_page1

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
async def test_candidates_pagination_next_cursor_and_no_offset(monkeypatch):
    now = datetime.utcnow()
    # Rows are ordered by last_seen DESC, company_id DESC (already sorted)
    rows_page1 = [
        (105, "C", "software", "c.com", (now - timedelta(minutes=0))),
        (104, "B", "software", "b.com", (now - timedelta(minutes=1))),
    ]
    rows_page2 = [
        (103, "A", "software", "a.com", (now - timedelta(minutes=2))),
    ]
    cur = FakeCursor(rows_page1, rows_page2)

    def _fake_get_conn():
        return FakeConn(cur)

    monkeypatch.setattr(main_app, "get_conn", _fake_get_conn)

    # Page 1
    res1 = await main_app.candidates_latest(limit=2, afterUpdatedAt=None, afterId=None, industry=None, _={})
    assert len(res1["items"]) == 2
    assert res1["nextCursor"] is not None
    assert res1["nextCursor"]["afterId"] == 104
    assert res1["nextCursor"]["afterUpdatedAt"] == rows_page1[-1][4].isoformat()

    # Page 2 using nextCursor
    nc = res1["nextCursor"]
    res2 = await main_app.candidates_latest(limit=2, afterUpdatedAt=datetime.fromisoformat(nc["afterUpdatedAt"]), afterId=nc["afterId"], industry=None, _={})
    assert len(res2["items"]) == 1
    assert res2["nextCursor"] is None  # last page shorter than limit

    # Ensure no SQL used OFFSET (keyset pagination)
    assert all("OFFSET" not in s.upper() for s in cur.executed_sql)

