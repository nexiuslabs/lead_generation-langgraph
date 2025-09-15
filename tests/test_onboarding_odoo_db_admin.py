import os
import sys
import pytest

# ensure path for import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app.onboarding as ob


def test_ensure_odoo_db_and_admin_propagates_module_error(monkeypatch):
    monkeypatch.setattr(ob, "_odoo_db_list", lambda server: [])
    monkeypatch.setattr(ob, "_odoo_db_create", lambda *a, **kw: None)
    monkeypatch.setattr(ob, "_odoo_admin_user_create", lambda *a, **kw: None)
    monkeypatch.setattr(ob, "_odoo_run_migration", lambda db: None)

    def fail_install(*a, **kw):
        raise RuntimeError("module fail")

    monkeypatch.setattr(ob, "_odoo_install_modules", fail_install)

    with pytest.raises(RuntimeError, match="module fail"):
        ob._ensure_odoo_db_and_admin("srv", "master", "db", "user@example.com")


def test_ensure_odoo_db_and_admin_propagates_migration_error(monkeypatch):
    monkeypatch.setattr(ob, "_odoo_db_list", lambda server: [])
    monkeypatch.setattr(ob, "_odoo_db_create", lambda *a, **kw: None)
    monkeypatch.setattr(ob, "_odoo_admin_user_create", lambda *a, **kw: None)
    monkeypatch.setattr(ob, "_odoo_install_modules", lambda *a, **kw: None)

    def fail_migration(db):
        raise RuntimeError("migration fail")

    monkeypatch.setattr(ob, "_odoo_run_migration", fail_migration)

    with pytest.raises(RuntimeError, match="migration fail"):
        ob._ensure_odoo_db_and_admin("srv", "master", "db", "user@example.com")


@pytest.mark.anyio("asyncio")
async def test_handle_first_login_surfaces_odoo_errors(monkeypatch):
    monkeypatch.setenv("ODOO_ENABLE_HTTP_PROVISION", "true")
    monkeypatch.setenv("ODOO_SERVER_URL", "http://fake")
    monkeypatch.setenv("ODOO_MASTER_PASSWORD", "master")

    monkeypatch.setattr(ob, "_ensure_tables", lambda: None)
    monkeypatch.setattr(ob, "_ensure_tenant_and_user", lambda email, tenant_id: 1)
    monkeypatch.setattr(ob, "_has_active_mapping", lambda tid: False)
    async def fake_mapping(tid, email, admin_password_override=None):
        return None

    monkeypatch.setattr(ob, "_ensure_odoo_mapping", fake_mapping)

    class DummyCursor:
        def execute(self, q, p):
            pass

        def fetchone(self):
            return ("db1",)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

    class DummyConn:
        def cursor(self):
            return DummyCursor()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

    monkeypatch.setattr(ob, "get_conn", lambda: DummyConn())

    recorded = []

    def fake_status(tid, status, err=None):
        recorded.append((status, err))

    monkeypatch.setattr(ob, "_insert_or_update_status", fake_status)

    def fail_ensure(*a, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(ob, "_ensure_odoo_db_and_admin", fail_ensure)

    result = await ob.handle_first_login("user@example.com", None)
    assert result["status"] == ob.ONBOARDING_ERROR
    assert "boom" in result["error"]
    assert (ob.ONBOARDING_ERROR, "odoo db auto-create failed: boom") in recorded
