import os
import json
import logging
from typing import Optional, Tuple
import asyncpg  # noqa: F401  # reserved for potential future async DB ops
import requests
from src.database import get_conn
from app.odoo_store import OdooStore

ONBOARDING_READY = "ready"
ONBOARDING_PROVISIONING = "provisioning"
ONBOARDING_SYNCING = "syncing"
ONBOARDING_ERROR = "error"

# PRD-detailed phases
ONBOARDING_STARTING = "starting"
ONBOARDING_CREATING_ODOO = "creating_odoo"
ONBOARDING_CONFIGURING_OIDC = "configuring_oidc"
ONBOARDING_SEEDING = "seeding"

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
                try:
                    from psycopg2.extras import Json  # type: ignore
                    payload = Json({"industries": ["software"], "employee_range": {"min": 10, "max": 200}})
                except Exception:
                    payload = json.dumps({"industries": ["software"], "employee_range": {"min": 10, "max": 200}})
                cur.execute(
                    "INSERT INTO icp_rules(tenant_id, name, payload) VALUES (%s, %s, %s)",
                    (tenant_id_claim, "Default ICP", payload),
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
            # If tenant row is missing for this tid (orphaned mapping), create it idempotently
            cur.execute("SELECT 1 FROM tenants WHERE tenant_id=%s", (tid,))
            if not cur.fetchone():
                cur.execute(
                    "INSERT INTO tenants(tenant_id, name, status) VALUES (%s, %s, 'active') ON CONFLICT (tenant_id) DO NOTHING",
                    (tid, email.split("@")[0]),
                )
        else:
            cur.execute(
                "INSERT INTO tenants(name, status) VALUES(%s,'active') RETURNING tenant_id",
                (email.split("@")[0],),
            )
            tid = cur.fetchone()[0]
        # Ensure tenants row exists for tid (idempotent upsert) in case of prior inconsistent data
        try:
            cur.execute(
                "INSERT INTO tenants(tenant_id, name, status) VALUES (%s, %s, 'active') ON CONFLICT (tenant_id) DO NOTHING",
                (tid, email.split("@")[0]),
            )
        except Exception:
            pass
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
            try:
                from psycopg2.extras import Json  # type: ignore
                payload = Json({"industries": ["software"], "employee_range": {"min": 10, "max": 200}})
            except Exception:
                payload = json.dumps({"industries": ["software"], "employee_range": {"min": 10, "max": 200}})
            cur.execute(
                "INSERT INTO icp_rules(tenant_id, name, payload) VALUES (%s, %s, %s)",
                (tid, "Default ICP", payload),
            )
        return tid


def _derive_db_name_from_email(email: str) -> str:
    try:
        local, domain = (email or "").split("@", 1)
        # Strip TLD and non-alnum, keep first label
        parts = [p for p in domain.split('.') if p]
        base = parts[0] if parts else (domain or "tenant")
        import re
        base = re.sub(r"[^a-zA-Z0-9_]", "", base.lower())
        if not base:
            base = "tenant"
        return f"odoo_{base}"
    except Exception:
        return "odoo_tenant"


async def _ensure_odoo_mapping(tenant_id: int, email: Optional[str] = None):
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

    def upsert_mapping(db_name: str, base_url: Optional[str] = None):
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT db_name FROM odoo_connections WHERE tenant_id=%s",
                (tenant_id,),
            )
            row = cur.fetchone()
            if row and row[0] == db_name:
                cur.execute(
                    "UPDATE odoo_connections SET active=TRUE, base_url=COALESCE(%s, base_url) WHERE tenant_id=%s",
                    (base_url, tenant_id),
                )
            else:
                cur.execute(
                    """
          INSERT INTO odoo_connections(tenant_id, base_url, db_name, auth_type, secret, active)
          VALUES (%s, %s, %s, %s, %s, TRUE)
          ON CONFLICT (tenant_id) DO UPDATE SET db_name=EXCLUDED.db_name, base_url=COALESCE(EXCLUDED.base_url, odoo_connections.base_url), active=TRUE
          """,
                    (tenant_id, base_url, db_name, "service_account", None),
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

    # No mapping yet; try server-level automation using suggested db_name from email
    server = os.getenv('ODOO_SERVER_URL')
    master = os.getenv('ODOO_MASTER_PASSWORD')
    if server and master and email:
        try:
            desired = _derive_db_name_from_email(email)
            # If DB exists on server, just upsert mapping; else create and map
            if desired in _odoo_db_list(server):
                upsert_mapping(desired, base_url=server)
            else:
                # Create DB and basic admin user using email as login
                import secrets
                admin_pass = secrets.token_urlsafe(24)
                _odoo_db_create(server, master, desired, email, admin_pass)
                _odoo_admin_user_create(server, desired, email, admin_pass, email)
                upsert_mapping(desired, base_url=server)
            store = OdooStore(tenant_id=tenant_id)
            await store.connectivity_smoke_test()
            return
        except Exception as e:
            logger.warning("odoo mapping via server failed tenant_id=%s email=%s error=%s", tenant_id, email, e)

    # Fallback: create mapping from inferred DSN path
    if inferred_db:
        logger.info("onboarding:odoo_mapping create_inferred tenant_id=%s db_name=%s", tenant_id, inferred_db)
        upsert_mapping(inferred_db, base_url=None)
        store = OdooStore(tenant_id=tenant_id)
        await store.connectivity_smoke_test()
        return

    # No way to infer mapping
    raise RuntimeError(
        f"No odoo_connections mapping for tenant_id={tenant_id} and unable to infer db_name. "
        f"Insert into odoo_connections(tenant_id, db_name, active) or set ODOO_POSTGRES_DSN."
    )


async def handle_first_login(email: str, tenant_id_claim: Optional[int]) -> dict:
    try:
        _ensure_tables()
        logger.info("onboarding:first_login start email=%s tenant_id_claim=%s", email, tenant_id_claim)
        # Odoo-first: ensure DB exists before creating tenant rows
        _ensure_odoo_db_first_by_email(email)
        # Create/find tenant rows in app DB
        tid = _ensure_tenant_and_user(email, tenant_id_claim)
        logger.info("onboarding:first_login resolved_tenant_id=%s email=%s", tid, email)
        _insert_or_update_status(tid, ONBOARDING_STARTING)
    except Exception as e:
        logger.exception("onboarding:first_login early failure email=%s", email)
        return {"tenant_id": None, "status": ONBOARDING_ERROR, "error": str(e)}
    try:
        # Ensure mapping and smoke test / infer db
        _insert_or_update_status(tid, ONBOARDING_CREATING_ODOO)
        await _ensure_odoo_mapping(tid, email=email)

        # Optional: automatically create Odoo DB (HTTP DB manager) and bootstrap admin
        try:
            from urllib.parse import urlparse
            server = os.getenv('ODOO_SERVER_URL')
            master = os.getenv('ODOO_MASTER_PASSWORD')
            if server and master:
                # derive db_name
                db_name = None
                with get_conn() as conn, conn.cursor() as cur:
                    cur.execute("SELECT db_name FROM odoo_connections WHERE tenant_id=%s", (tid,))
                    row = cur.fetchone()
                    db_name = (row[0] if row else None)
                if db_name:
                    # Use first-login email as admin login
                    _ensure_odoo_db_and_admin(server, master, db_name, email)
        except Exception as e:
            logger.warning("odoo db auto-create skipped: %s", e)

        # Configure OIDC provider in Odoo (optional)
        _insert_or_update_status(tid, ONBOARDING_CONFIGURING_OIDC)
        try:
            server = os.getenv('ODOO_SERVER_URL')
            issuer = os.getenv('NEXIUS_ISSUER')
            cid = os.getenv('NEXIUS_CLIENT_ID')
            secret = os.getenv('NEXIUS_CLIENT_SECRET')
            if server and issuer and cid and secret:
                with get_conn() as conn, conn.cursor() as cur:
                    cur.execute("SELECT db_name FROM odoo_connections WHERE tenant_id=%s", (tid,))
                    row = cur.fetchone()
                    db_name = (row[0] if row else None)
                if db_name:
                    # Log in with tenant admin (same email) and configure provider
                    admin_login = email
                    admin_password = os.getenv('ODOO_TENANT_ADMIN_PASSWORD_FALLBACK', '')
                    if admin_password:
                        _odoo_configure_oidc(server, db_name, admin_login, admin_password, issuer, cid, secret)
        except Exception as e:
            logger.warning("odoo oidc auto-config skipped: %s", e)

        # Move to seeding while we seed minimal baseline
        _insert_or_update_status(tid, ONBOARDING_SEEDING)
        try:
            store = OdooStore(tenant_id=tid)
            await store.seed_baseline_entities(tenant_id=tid, email=email)
        except Exception as _seed_err:
            # Non-fatal; continue to ready but include error detail in status
            logger.exception("Onboarding baseline seed failed tenant_id=%s", tid)
            _insert_or_update_status(tid, ONBOARDING_SEEDING, str(_seed_err))
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


# --- Odoo DB + OIDC helpers (best effort; optional based on env) ---
def _requests_verify() -> bool | str:
    v = (os.getenv('ODOO_TLS_VERIFY', 'true') or 'true').strip().lower()
    enabled = v in ('1', 'true', 'yes', 'on')
    ca = os.getenv('ODOO_CA_BUNDLE', '').strip()
    if enabled and ca:
        return ca
    return enabled


def _odoo_db_list(server: str) -> list:
    payload = {"jsonrpc": "2.0", "method": "call", "params": {"service": "db", "method": "list", "args": []}}
    r = requests.post(server.rstrip('/') + "/jsonrpc", json=payload, timeout=20, verify=_requests_verify())
    r.raise_for_status()
    return (r.json() or {}).get("result", [])


def _odoo_db_create(server: str, master_pwd: str, db_name: str, admin_login: str, admin_password: str):
    import requests
    args = [master_pwd, db_name, False, os.getenv("ODOO_LANG", "en_US"), admin_password, admin_login, os.getenv("ODOO_COUNTRY", "SG"), "", admin_login]
    payload = {"jsonrpc": "2.0", "method": "call", "params": {"service": "db", "method": "create_database", "args": args}}
    r = requests.post(server.rstrip('/') + "/jsonrpc", json=payload, timeout=180, verify=_requests_verify())
    r.raise_for_status()


def _odoo_admin_user_create(server: str, db_name: str, admin_login: str, admin_password: str, tenant_admin_email: str):
    import xmlrpc.client
    common = xmlrpc.client.ServerProxy(f"{server.rstrip('/')}/xmlrpc/2/common")
    uid = common.authenticate(db_name, admin_login, admin_password, {})
    models = xmlrpc.client.ServerProxy(f"{server.rstrip('/')}/xmlrpc/2/object")
    group_system_id = models.execute_kw(db_name, uid, admin_password, 'ir.model.data', 'xmlid_to_res_id', ['base.group_system'])
    # Ensure a separate admin user if needed
    users = models.execute_kw(db_name, uid, admin_password, 'res.users', 'search', [[['login', '=', tenant_admin_email]]])
    if not users:
        models.execute_kw(db_name, uid, admin_password, 'res.users', 'create', [{
            'name': 'Tenant Admin', 'login': tenant_admin_email, 'email': tenant_admin_email, 'groups_id': [(6, 0, [group_system_id])]
        }])


def _odoo_configure_oidc(server: str, db_name: str, admin_login: str, admin_password: str, issuer: str, cid: str, secret: str):
    import xmlrpc.client
    common = xmlrpc.client.ServerProxy(f"{server.rstrip('/')}/xmlrpc/2/common")
    uid = common.authenticate(db_name, admin_login, admin_password, {})
    models = xmlrpc.client.ServerProxy(f"{server.rstrip('/')}/xmlrpc/2/object")
    # Install auth_oauth
    ids = models.execute_kw(db_name, uid, admin_password, 'ir.module.module', 'search', [[['name', '=', 'auth_oauth']]])
    if ids:
        try:
            models.execute_kw(db_name, uid, admin_password, 'ir.module.module', 'button_immediate_install', [ids])
        except Exception:
            pass
    auth_ep = issuer.rstrip('/') + '/protocol/openid-connect/auth'
    token_ep = issuer.rstrip('/') + '/protocol/openid-connect/token'
    userinfo_ep = issuer.rstrip('/') + '/protocol/openid-connect/userinfo'
    vals = { 'name': 'Nexius', 'client_id': cid, 'client_secret': secret, 'auth_endpoint': auth_ep, 'scope': 'openid email profile', 'validation_endpoint': userinfo_ep }
    prov = models.execute_kw(db_name, uid, admin_password, 'auth.oauth.provider', 'search', [[['name', '=', 'Nexius']]])
    if prov:
        models.execute_kw(db_name, uid, admin_password, 'auth.oauth.provider', 'write', [prov, vals])
    else:
        models.execute_kw(db_name, uid, admin_password, 'auth.oauth.provider', 'create', [vals])


def _ensure_odoo_db_and_admin(server: str, master_pwd: str, db_name: str, email: str):
    try:
        if db_name not in _odoo_db_list(server):
            admin_login = email
            import secrets
            admin_password = secrets.token_urlsafe(int(os.getenv('ODOO_TENANT_ADMIN_PASSWORD_LENGTH', '24')))
            _odoo_db_create(server, master_pwd, db_name, admin_login, admin_password)
            _odoo_admin_user_create(server, db_name, admin_login, admin_password, email)
    except Exception as e:
        logger.warning("ensure_odoo_db_and_admin failed: %s", e)


def _ensure_odoo_db_first_by_email(email: str) -> Optional[str]:
    """Ensure an Odoo database exists for the user's email domain BEFORE tenant creation.

    - Derives db_name as odoo_<first label of domain>.
    - If ODOO_SERVER_URL/ODOO_MASTER_PASSWORD not set, returns None (skip).
    - If DB exists: return db_name.
    - Else create DB + admin user (login=email) and return db_name.
    """
    server = os.getenv('ODOO_SERVER_URL')
    master = os.getenv('ODOO_MASTER_PASSWORD')
    if not server or not master:
        return None
    try:
        desired = _derive_db_name_from_email(email)
        dbs = _odoo_db_list(server)
        if desired in dbs:
            logger.info("odoo:first ensure exists db=%s email=%s", desired, email)
            return desired
        # Create DB and bootstrap admin user
        import secrets
        admin_pass = secrets.token_urlsafe(int(os.getenv('ODOO_TENANT_ADMIN_PASSWORD_LENGTH', '24')))
        _odoo_db_create(server, master, desired, email, admin_pass)
        _odoo_admin_user_create(server, desired, email, admin_pass, email)
        logger.info("odoo:first created db=%s email=%s", desired, email)
        return desired
    except Exception as e:
        logger.warning("odoo:first ensure failed email=%s error=%s", email, e)
        return None
