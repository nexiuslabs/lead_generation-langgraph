import os
import logging
from typing import Optional, Tuple
import asyncpg  # noqa: F401  # reserved for potential future async DB ops
from src.database import get_conn
from app.odoo_store import OdooStore

ONBOARDING_READY = "ready"
ONBOARDING_PROVISIONING = "provisioning"
ONBOARDING_SYNCING = "syncing"
ONBOARDING_ERROR = "error"

logger = logging.getLogger("onboarding")


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
        # Always honor the tenant_id claim when provided
        if tenant_id_claim is not None:
            # Ensure tenants row exists with this exact id
            cur.execute(
                "SELECT tenant_id FROM tenants WHERE tenant_id=%s",
                (tenant_id_claim,),
            )
            r = cur.fetchone()
            if not r:
                cur.execute(
                    "INSERT INTO tenants(tenant_id, name, status) VALUES (%s, %s, 'active')",
                    (tenant_id_claim, email.split("@")[0]),
                )
            # Link user to claimed tenant (viewer by default)
            cur.execute(
                """
      INSERT INTO tenant_users(tenant_id, user_id, roles)
      VALUES (%s, %s, %s)
      ON CONFLICT (tenant_id, user_id) DO UPDATE SET roles=EXCLUDED.roles
      """,
                (tenant_id_claim, email, ["viewer"]),
            )
            # Seed ICP template if none exists for this tenant
            cur.execute(
                "SELECT 1 FROM icp_rules WHERE tenant_id=%s LIMIT 1",
                (tenant_id_claim,),
            )
            if not cur.fetchone():
                cur.execute(
                    "INSERT INTO icp_rules(tenant_id, name, payload) VALUES (%s, %s, %s)",
                    (tenant_id_claim, "Default ICP", {"industries": ["software"], "employee_range": {"min": 10, "max": 200}}),
                )
            return tenant_id_claim

        # No claim: if user already linked to a tenant, reuse it
        cur.execute(
            "SELECT tenant_id FROM tenant_users WHERE user_id=%s LIMIT 1",
            (email,),
        )
        r = cur.fetchone()
        if r:
            tid = r[0]
        else:
            cur.execute(
                "INSERT INTO tenants(name, status) VALUES(%s,'active') RETURNING tenant_id",
                (email.split("@")[0],),
            )
            tid = cur.fetchone()[0]
        # Link user and seed defaults
        cur.execute(
            """
      INSERT INTO tenant_users(tenant_id, user_id, roles)
      VALUES (%s, %s, %s)
      ON CONFLICT (tenant_id, user_id) DO UPDATE SET roles=EXCLUDED.roles
      """,
            (tid, email, ["viewer"]),
        )
        cur.execute("SELECT 1 FROM icp_rules WHERE tenant_id=%s LIMIT 1", (tid,))
        if not cur.fetchone():
            cur.execute(
                "INSERT INTO icp_rules(tenant_id, name, payload) VALUES (%s, %s, %s)",
                (tid, "Default ICP", {"industries": ["software"], "employee_range": {"min": 10, "max": 200}}),
            )
        return tid


async def _ensure_odoo_mapping(tenant_id: int):
    """Ensure there is an active odoo_connections row for this tenant.

    Behavior:
    - If a mapping exists, try to use it. If connectivity fails and a default DB can be inferred, fallback.
    - If missing, insert mapping inferred from ODOO_POSTGRES_DSN (db name from DSN path).
    - If neither exists, raise with guidance.
    """
    from src.settings import ODOO_POSTGRES_DSN
    inferred_db: Optional[str] = None
    if ODOO_POSTGRES_DSN:
        try:
            from urllib.parse import urlparse
            u = urlparse(ODOO_POSTGRES_DSN)
            path = (u.path or "/").lstrip("/")
            inferred_db = path or None
        except Exception:
            inferred_db = None

    def upsert_mapping(db_name: str):
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT db_name FROM odoo_connections WHERE tenant_id=%s",
                (tenant_id,),
            )
            row = cur.fetchone()
            if row and row[0] == db_name:
                cur.execute(
                    "UPDATE odoo_connections SET active=TRUE WHERE tenant_id=%s",
                    (tenant_id,),
                )
            else:
                cur.execute(
                    """
          INSERT INTO odoo_connections(tenant_id, db_name, auth_type, secret, active)
          VALUES (%s, %s, %s, %s, TRUE)
          ON CONFLICT (tenant_id) DO UPDATE SET db_name=EXCLUDED.db_name, active=TRUE
          """,
                    (tenant_id, db_name, "service_account", None),
                )

    # Prefer existing mapping if any
    existing_db: Optional[str] = None
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT db_name FROM odoo_connections WHERE tenant_id=%s",
            (tenant_id,),
        )
        row = cur.fetchone()
        if row and row[0]:
            existing_db = row[0]

    if existing_db:
        logger.info(
            "onboarding:odoo_mapping use_existing tenant_id=%s db_name=%s",
            tenant_id,
            existing_db,
        )
        upsert_mapping(existing_db)
        store = OdooStore(tenant_id=tenant_id)
        try:
            await store.connectivity_smoke_test()
            return
        except Exception as e:
            logger.warning(
                "onboarding:odoo_connectivity_failed tenant_id=%s db_name=%s error=%s",
                tenant_id,
                existing_db,
                e,
            )
            # Fallback to inferred DB if available
            if inferred_db and inferred_db != existing_db:
                logger.info(
                    "onboarding:odoo_mapping fallback_to_inferred tenant_id=%s db_name=%s",
                    tenant_id,
                    inferred_db,
                )
                upsert_mapping(inferred_db)
                store = OdooStore(tenant_id=tenant_id)
                await store.connectivity_smoke_test()
                return
            raise

    # No mapping yet; create from inferred db
    if inferred_db:
        logger.info("onboarding:odoo_mapping create_inferred tenant_id=%s db_name=%s", tenant_id, inferred_db)
        upsert_mapping(inferred_db)
        store = OdooStore(tenant_id=tenant_id)
        await store.connectivity_smoke_test()
        return

    # No way to infer mapping
    raise RuntimeError(
        f"No odoo_connections mapping for tenant_id={tenant_id} and unable to infer db_name. "
        f"Insert into odoo_connections(tenant_id, db_name, active) or set ODOO_POSTGRES_DSN."
    )


async def handle_first_login(email: str, tenant_id_claim: Optional[int]) -> dict:
    _ensure_tables()
    logger.info("onboarding:first_login start email=%s tenant_id_claim=%s", email, tenant_id_claim)
    tid = _ensure_tenant_and_user(email, tenant_id_claim)
    logger.info("onboarding:first_login resolved_tenant_id=%s email=%s", tid, email)
    _insert_or_update_status(tid, ONBOARDING_PROVISIONING)
    try:
        # Ensure mapping and smoke test
        await _ensure_odoo_mapping(tid)
        # Log current odoo_connections mapping for visibility
        try:
            with get_conn() as conn, conn.cursor() as cur:
                cur.execute("SELECT db_name FROM odoo_connections WHERE tenant_id=%s", (tid,))
                row = cur.fetchone()
                logger.info("onboarding:odoo_mapping tenant_id=%s db_name=%s", tid, row[0] if row else None)
        except Exception:
            pass
        # Move to syncing while we seed minimal baseline
        _insert_or_update_status(tid, ONBOARDING_SYNCING)
        try:
            store = OdooStore(tenant_id=tid)
            await store.seed_baseline_entities(tenant_id=tid, email=email)
        except Exception as _seed_err:
            # Non-fatal; continue to ready but include error detail in status
            logger.exception("Onboarding baseline seed failed tenant_id=%s", tid)
            _insert_or_update_status(tid, ONBOARDING_SYNCING, str(_seed_err))
        _insert_or_update_status(tid, ONBOARDING_READY)
    except Exception as e:
        logger.exception("Onboarding failed tenant_id=%s", tid)
        _insert_or_update_status(tid, ONBOARDING_ERROR, str(e))
        return {"tenant_id": tid, "status": ONBOARDING_ERROR, "error": str(e)}
    return {"tenant_id": tid, "status": ONBOARDING_READY}


def get_onboarding_status(tenant_id: int) -> dict:
    _ensure_tables()
    status, error = _get_status(tenant_id)
    return {"tenant_id": tenant_id, "status": status, "error": error}
