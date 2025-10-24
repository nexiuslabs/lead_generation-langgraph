import os
import types

os.environ.setdefault("OPENAI_API_KEY", "test")


def test_sg_registered_requires_confidence_threshold():
    from src import icp as icp_mod

    # Case 1: High confidence (matching/very similar names) + live status → sg_registered True
    row_high = {
        "entity_name": "Alpha Beta Pte Ltd",
        "name": "Alpha Beta Pte Ltd",
        "uen": "201234567A",
        "entity_status_description": "Live Company",
        "primary_ssic_description": "Software development",
        "primary_ssic_code": "62010",
    }
    norm_high = icp_mod._normalize_row(row_high)
    assert norm_high.get("uen_confidence") is not None
    assert float(norm_high.get("uen_confidence") or 0.0) >= float(icp_mod.UEN_CONFIDENCE_MIN)
    assert norm_high.get("sg_registered") is True

    # Case 2: Low confidence (dissimilar names) even with live status → sg_registered not set
    row_low = {
        "entity_name": "Gamma Solutions Pte Ltd",
        "name": "Zeta Holdings LLC",
        "uen": "201234567B",
        "entity_status_description": "LIVE",
        "primary_ssic_description": "Consulting",
        "primary_ssic_code": "70209",
    }
    norm_low = icp_mod._normalize_row(row_low)
    assert norm_low.get("uen_confidence") is not None
    assert float(norm_low.get("uen_confidence") or 0.0) < float(icp_mod.UEN_CONFIDENCE_MIN)
    # Expect gate to refuse setting sg_registered
    assert norm_low.get("sg_registered") in (None, False)


def test_upsert_includes_uen_confidence(monkeypatch):
    from src import icp as icp_mod

    # Fake DB connection and cursor to capture SQL
    class FakeCursor:
        def __init__(self):
            self.sql = None
            self.params = None
            self.rowcount = 1

        def execute(self, sql, params=None):
            self.sql = sql
            self.params = params

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeConn:
        def __init__(self):
            self.cur = FakeCursor()
            self.committed = False

        def cursor(self):
            return self.cur

        def commit(self):
            self.committed = True

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    fake_conn = FakeConn()

    # Monkeypatch get_conn and _table_columns so icp uses our fake conn and sees expected columns
    monkeypatch.setattr(icp_mod, "get_conn", lambda: fake_conn)
    monkeypatch.setattr(
        icp_mod,
        "_table_columns",
        lambda _conn, _t: {
            "company_id",
            "uen",
            "uen_confidence",
            "name",
            "industry_norm",
            "industry_code",
            "website_domain",
            "incorporation_year",
            "founded_year",
            "ownership_type",
            "sg_registered",
            "last_seen",
        },
    )

    # Build a normalized row with uen_confidence present
    row = {
        "entity_name": "Alpha Beta Pte Ltd",
        "name": "Alpha Beta Pte Ltd",
        "uen": "201234567A",
        "entity_status_description": "Live Company",
        "primary_ssic_description": "Software development",
        "primary_ssic_code": "62010",
    }
    norm = icp_mod._normalize_row(row)
    assert norm.get("uen_confidence") is not None

    # Run upsert (will hit our fake cursor)
    affected = icp_mod._upsert_companies_batch([norm])
    assert affected == 1

    # Validate SQL contains uen_confidence and params include the value
    assert isinstance(fake_conn.cur.sql, str) and "INSERT INTO companies" in fake_conn.cur.sql
    assert "uen_confidence" in fake_conn.cur.sql
    assert norm.get("uen_confidence") in (fake_conn.cur.params or [])

