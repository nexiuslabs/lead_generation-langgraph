import os
from typing import Optional, Tuple
import asyncpg  # noqa: F401  # reserved for potential future async DB ops
from src.database import get_conn
from app.odoo_store import OdooStore

ONBOARDING_READY = "ready"
ONBOARDING_PROVISIONING = "provisioning"
ONBOARDING_SYNCING = "syncing"
ONBOARDING_ERROR = "error"


def _ensure_tables():
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
      CREATE TABLE IF NOT EXISTS onboarding_status (
        tenant_id INT PRIMARY KEY REFERENCES tenants(tenant_id) ON DELETE CASCADE,
        status TEXT NOT NULL,
        error TEXT,
        updated_at TIMESTAMPTZ DEFAULT now()
      );
      """
        )


def _insert_or_update_status(tenant_id: int, status: str, error: Optional[str] = None):
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
      INSERT INTO onboarding_status(tenant_id, status, error, updated_at)
      VALUES (%s, %s, %s, now())
      ON CONFLICT (tenant_id) DO UPDATE SET status=EXCLUDED.status, error=EXCLUDED.error, updated_at=now();
      """,
            (tenant_id, status, error),
        )


def _get_status(tenant_id: int) -> Tuple[str, Optional[str]]:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT status, error FROM onboarding_status WHERE tenant_id=%s", (tenant_id,)
        )
        row = cur.fetchone()
        if not row:
            return (ONBOARDING_PROVISIONING, None)
        return (row[0], row[1])


def _ensure_tenant_and_user(email: str, tenant_id_claim: Optional[int]) -> int:
    with get_conn() as conn, conn.cursor() as cur:
        # If claim exists and tenant row exists, reuse it; else create a new tenant.
        tid: Optional[int] = None
        if tenant_id_claim is not None:
            cur.execute(
                "SELECT tenant_id FROM tenants WHERE tenant_id=%s", (tenant_id_claim,)
            )
            r = cur.fetchone()
            if r:
                tid = r[0]
        if tid is None:
            cur.execute(
                "INSERT INTO tenants(name, status) VALUES(%s,'active') RETURNING tenant_id",
                (email.split("@")[0],),
            )
            tid = cur.fetchone()[0]
        # Link user to tenant with default role viewer
        cur.execute(
            """
      INSERT INTO tenant_users(tenant_id, user_id, roles)
      VALUES (%s, %s, %s)
      ON CONFLICT (tenant_id, user_id) DO UPDATE SET roles=EXCLUDED.roles
      """,
            (tid, email, ["viewer"]),
        )
        # Seed ICP template if none exists
        cur.execute("SELECT 1 FROM icp_rules WHERE tenant_id=%s LIMIT 1", (tid,))
        if not cur.fetchone():
            cur.execute(
                "INSERT INTO icp_rules(tenant_id, name, payload) VALUES (%s, %s, %s)",
                (tid, "Default ICP", {"industries": ["software"], "employee_range": {"min": 10, "max": 200}}),
            )
        return tid


async def _ensure_odoo_mapping(tenant_id: int):
    base_tpl = os.getenv("ODOO_BASE_DSN_TEMPLATE")
    default_db = os.getenv("ODOO_DEFAULT_DB_NAME", "odoo")
    db_name = (base_tpl and f"odoo_tenant_{tenant_id}") or default_db
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
      INSERT INTO odoo_connections(tenant_id, db_name, auth_type, secret, active)
      VALUES (%s, %s, %s, %s, TRUE)
      ON CONFLICT (tenant_id) DO UPDATE SET db_name=EXCLUDED.db_name, active=TRUE
      """,
            (tenant_id, db_name, "service_account", None),
        )
    # Connectivity smoke
    store = OdooStore(tenant_id=tenant_id)
    await store.connectivity_smoke_test()


async def handle_first_login(email: str, tenant_id_claim: Optional[int]) -> dict:
    _ensure_tables()
    tid = _ensure_tenant_and_user(email, tenant_id_claim)
    _insert_or_update_status(tid, ONBOARDING_PROVISIONING)
    try:
        # Ensure mapping and smoke test
        await _ensure_odoo_mapping(tid)
        # Move to syncing while we seed minimal baseline
        _insert_or_update_status(tid, ONBOARDING_SYNCING)
        try:
            store = OdooStore(tenant_id=tid)
            await store.seed_baseline_entities(tenant_id=tid, email=email)
        except Exception as _seed_err:
            # Non-fatal; continue to ready but include error detail in status
            _insert_or_update_status(tid, ONBOARDING_SYNCING, str(_seed_err))
        _insert_or_update_status(tid, ONBOARDING_READY)
    except Exception as e:
        _insert_or_update_status(tid, ONBOARDING_ERROR, str(e))
        return {"tenant_id": tid, "status": ONBOARDING_ERROR, "error": str(e)}
    return {"tenant_id": tid, "status": ONBOARDING_READY}


def get_onboarding_status(tenant_id: int) -> dict:
    _ensure_tables()
    status, error = _get_status(tenant_id)
    return {"tenant_id": tenant_id, "status": status, "error": error}
