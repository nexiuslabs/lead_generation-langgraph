import os
import sys

os.environ.setdefault("OPENAI_API_KEY", "test")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Any, List, Optional

import app.main as main_app


class FakeDesc:
    def __init__(self, name: str):
        self.name = name


class FakeCursor:
    def __init__(self) -> None:
        self.description: List[FakeDesc] = []
        self.rowcount = 0
        self._pending_fetchall: Optional[List[tuple]] = None
        self._pending_fetchone: Optional[tuple] = None
        self.insert_calls: List[tuple[str, Optional[List[Any]]]] = []
        self.update_calls: List[tuple[str, Optional[List[Any]]]] = []

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401 - part of context manager protocol
        return None

    def execute(self, sql: str, params: Any = None) -> None:
        normalized = " ".join(sql.strip().split())
        self.rowcount = 0
        if "information_schema.columns" in normalized:
            self._pending_fetchall = [
                ("uen",),
                ("entity_name",),
                ("primary_ssic_description",),
                ("primary_ssic_code",),
            ]
            self.description = []
        elif "FROM staging_acra_companies" in normalized and normalized.startswith("SELECT"):
            self._pending_fetchall = [
                ("201234567A", "Acme Tech", "Software Development Services", "62010"),
            ]
            self.description = [
                FakeDesc("uen"),
                FakeDesc("entity_name"),
                FakeDesc("primary_ssic_description"),
                FakeDesc("primary_ssic_code"),
            ]
        elif "SELECT company_id FROM companies WHERE uen" in normalized:
            self._pending_fetchone = None
        elif "SELECT company_id FROM companies WHERE LOWER(name)" in normalized:
            self._pending_fetchone = None
        elif "SELECT company_id FROM companies WHERE website_domain" in normalized:
            self._pending_fetchone = None
        elif normalized.startswith("INSERT INTO companies"):
            self.insert_calls.append((sql, params))
            self.rowcount = 1
            self._pending_fetchone = (123,)
        elif normalized.startswith("UPDATE companies SET last_seen"):
            self.update_calls.append((sql, params))
            self.rowcount = 1
            self._pending_fetchone = None
        elif normalized.startswith("UPDATE companies SET"):
            self.update_calls.append((sql, params))
            self.rowcount = 1
            self._pending_fetchone = None
        else:
            raise AssertionError(f"Unexpected SQL: {sql}")

    def fetchall(self) -> List[tuple]:
        rows = self._pending_fetchall or []
        self._pending_fetchall = None
        return rows

    def fetchone(self) -> Optional[tuple]:
        row = self._pending_fetchone
        self._pending_fetchone = None
        return row


class FakeConnection:
    def __init__(self) -> None:
        self.cursor_obj = FakeCursor()

    def __enter__(self) -> "FakeConnection":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401 - context manager protocol
        return None

    def cursor(self, name: str | None = None) -> FakeCursor:
        if name:
            raise AssertionError(f"Unexpected named cursor request: {name}")
        return self.cursor_obj


def test_upsert_handles_missing_optional_staging_columns(monkeypatch):
    fake_conn = FakeConnection()

    monkeypatch.setattr(main_app, "get_conn", lambda: fake_conn)
    monkeypatch.setattr(
        main_app,
        "_find_ssic_codes_by_terms",
        lambda terms: [("62010", "Software development", 0.9)],
        raising=False,
    )

    affected = main_app._upsert_companies_from_staging_by_industries(["Software Development"])

    assert affected == 1
    assert fake_conn.cursor_obj.insert_calls, "Expected an INSERT into companies"
    insert_sql, insert_params = fake_conn.cursor_obj.insert_calls[0]
    assert "website_domain" not in insert_sql
    assert insert_params[:4] == [
        "201234567A",
        "Acme Tech",
        "software development services",
        "62010",
    ]
    assert insert_params[4:] in ([], [False])
