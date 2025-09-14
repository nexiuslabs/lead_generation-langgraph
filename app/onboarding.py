import os
import json
import logging
import time
from contextlib import closing
from typing import Optional, Tuple
import asyncpg  # noqa: F401  # reserved for potential future async DB ops
import requests
from passlib.hash import pbkdf2_sha512
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
        tenant_label = _derive_tenant_label(email)
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
                    (tenant_id_claim, tenant_label),
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
                    (tid, tenant_label),
                )
        else:
            cur.execute(
                "INSERT INTO tenants(name, status) VALUES(%s,'active') RETURNING tenant_id",
                (tenant_label,),
            )
            tid = cur.fetchone()[0]
        # Ensure tenants row exists for tid (idempotent upsert) in case of prior inconsistent data
        try:
            cur.execute(
                "INSERT INTO tenants(tenant_id, name, status) VALUES (%s, %s, 'active') ON CONFLICT (tenant_id) DO NOTHING",
                (tid, tenant_label),
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


def _derive_tenant_label(email: str) -> str:
    try:
        _local, domain = (email or "").split("@", 1)
        parts = [p for p in (domain or "").split(".") if p]
        base = parts[0] if parts else (domain or "tenant")
        import re
        base = base.replace('-', '_').lower()
        base = re.sub(r"[^a-z0-9_]", "", base)
        return base or "tenant"
    except Exception:
        return "tenant"


def _derive_db_name_from_email(email: str) -> str:
    # Odoo DB equals the tenant label
    return _derive_tenant_label(email)


def _truthy(name: str, default: str = "false") -> bool:
    val = (os.getenv(name, default) or default).strip().lower()
    return val in {"1", "true", "yes", "on"}


def _has_active_mapping(tenant_id: int) -> bool:
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM odoo_connections WHERE tenant_id=%s AND active", (tenant_id,)
            )
            return bool(cur.fetchone())
    except Exception:
        return False


async def _ensure_odoo_mapping(tenant_id: int, email: Optional[str] = None, admin_password_override: Optional[str] = None):
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

    # Compute preferred tenant base_url like https://{tenant}.agent.nexiusagent.com
    tenant_label = _derive_tenant_label(email or "") if email else None
    base_template = (os.getenv('ODOO_BASE_URL_TEMPLATE') or '').strip()
    base_domain = (os.getenv('ODOO_BASE_DOMAIN') or '').strip()
    scheme = (os.getenv('ODOO_BASE_SCHEME') or 'https').strip()
    preferred_base_url: Optional[str] = None
    try:
        if tenant_label:
            if base_template:
                preferred_base_url = base_template.format(tenant=tenant_label)
            elif base_domain:
                preferred_base_url = f"{scheme}://{tenant_label}.{base_domain}"
            else:
                # Fallback: derive base domain from ODOO_SERVER_URL
                srv = (os.getenv('ODOO_SERVER_URL') or '').strip()
                if srv:
                    from urllib.parse import urlparse
                    u = urlparse(srv)
                    host = (u.netloc or u.path or '').strip().lstrip('/')
                    if host:
                        preferred_base_url = f"{scheme}://{tenant_label}.{host}"
    except Exception:
        preferred_base_url = None

    if existing_db:
        logger.info(
            "onboarding:odoo_mapping use_existing tenant_id=%s db_name=%s",
            tenant_id,
            existing_db,
        )
        upsert_mapping(existing_db, base_url=preferred_base_url)
        store = OdooStore(tenant_id=tenant_id)
        try:
            await store.connectivity_smoke_test()
            # If signup password was provided, align Odoo admin credentials now (existing DB case)
            if admin_password_override and email:
                try:
                    server = (os.getenv('ODOO_SERVER_URL') or '').strip()
                    if server:
                        _odoo_set_admin_credentials(server, existing_db, email, admin_password_override)
                        logger.info("onboarding:odoo_admin aligned for existing db tenant_id=%s", tenant_id)
                except Exception as _align_exc:
                    logger.warning("onboarding:odoo_admin align failed tenant_id=%s db=%s error=%s", tenant_id, existing_db, _align_exc)
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
    # Attempt HTTP provisioning whenever server+master are configured (do not gate on the flag in dev)
    if server and master and email:
        try:
            desired = _derive_db_name_from_email(email)
            # If DB exists on server, just upsert mapping; else create and map
            if desired in _odoo_db_list(server):
                upsert_mapping(desired, base_url=preferred_base_url or server)
            else:
                # Create DB and basic admin user using email as login
                import secrets
                admin_pass = admin_password_override or secrets.token_urlsafe(24)
                logger.info("odoo:create start db=%s email=%s server=%s", desired, email, server)
                _odoo_db_create(server, master, desired, email, admin_pass)
                # Wait for the new DB to appear and be ready for XML-RPC auth
                ready = _odoo_wait_db(server, desired, timeout_s=90, interval_s=2.0)
                if not ready:
                    raise RuntimeError(f"odoo db '{desired}' did not appear within timeout after create_database")
                # Set admin credentials explicitly to the signup email/password when available
                try:
                    if admin_password_override:
                        # Try authenticating with the credentials used by create_database first
                        ok = _odoo_set_admin_credentials(
                            server,
                            desired,
                            email,
                            admin_password_override,
                            auth_login=email,
                            auth_password=admin_pass,
                        )
                        if not ok:
                            # Some Odoo versions ignore the admin_login param and set 'admin'
                            _odoo_set_admin_credentials(
                                server,
                                desired,
                                email,
                                admin_password_override,
                                auth_login='admin',
                                auth_password=admin_pass,
                            )
                    else:
                        # Retry admin user bootstrap a few times (registry may still be warming up)
                        last_exc: Exception | None = None
                        for _ in range(5):
                            try:
                                _odoo_admin_user_create(server, desired, email, admin_pass, email)
                                last_exc = None
                                break
                            except Exception as e:
                                last_exc = e
                                time.sleep(2.0)
                        if last_exc is not None:
                            logger.warning("odoo:create admin user failed for db=%s email=%s error=%s", desired, email, last_exc)
                except Exception as _cred_exc:
                    logger.warning("odoo:create admin credential set failed db=%s email=%s error=%s", desired, email, _cred_exc)
                # Upsert mapping regardless so platform can connect via Postgres DSN
                upsert_mapping(desired, base_url=preferred_base_url or server)
            # Best-effort connectivity check; do not fail provisioning if DSN is unreachable (e.g., tunnel closed)
            try:
                store = OdooStore(tenant_id=tenant_id)
                await store.connectivity_smoke_test()
            except Exception as e:
                logger.warning("onboarding:odoo_connectivity_post_create_failed tenant_id=%s db=%s error=%s", tenant_id, desired if email else "?", e)
            return
        except Exception as e:
            logger.warning("odoo mapping via server failed tenant_id=%s email=%s error=%s", tenant_id, email, e)
            # Fallback: If web DB-manager is disabled (list_db=false) or RPC signature differs, try template-based provisioning
            try:
                desired = _derive_db_name_from_email(email)
                admin_pass = admin_password_override or _random_password()
                if _odoo_provision_from_template(desired, email, admin_pass):
                    upsert_mapping(desired, base_url=preferred_base_url or server)
                    try:
                        store = OdooStore(tenant_id=tenant_id)
                        await store.connectivity_smoke_test()
                    except Exception as e2:
                        logger.warning("onboarding:odoo_connectivity_post_template_failed tenant_id=%s db=%s error=%s", tenant_id, desired, e2)
                    return
            except Exception as e2:
                logger.warning("odoo template provision failed tenant_id=%s email=%s error=%s", tenant_id, email, e2)

    # Fallback: create mapping from inferred DSN path
    if inferred_db:
        logger.info("onboarding:odoo_mapping create_inferred tenant_id=%s db_name=%s", tenant_id, inferred_db)
        upsert_mapping(inferred_db, base_url=None)
        try:
            store = OdooStore(tenant_id=tenant_id)
            await store.connectivity_smoke_test()
            # Align admin credentials if we have a signup password and server
            if admin_password_override and email:
                try:
                    server = (os.getenv('ODOO_SERVER_URL') or '').strip()
                    if server:
                        _odoo_set_admin_credentials(server, inferred_db, email, admin_password_override)
                        logger.info("onboarding:odoo_admin aligned for inferred db tenant_id=%s", tenant_id)
                except Exception as _align_exc:
                    logger.warning("onboarding:odoo_admin align failed tenant_id=%s db=%s error=%s", tenant_id, inferred_db, _align_exc)
        except Exception as e:
            logger.warning("onboarding:odoo_connectivity_inferred_failed tenant_id=%s db=%s error=%s", tenant_id, inferred_db, e)
        return

    # No way to infer mapping
    raise RuntimeError(
        f"No odoo_connections mapping for tenant_id={tenant_id} and unable to infer db_name. "
        f"Set ODOO_POSTGRES_DSN or ODOO_BASE_DSN_TEMPLATE for Postgres connectivity, "
        f"or configure ODOO_SERVER_URL and ODOO_MASTER_PASSWORD so auto-provision can run."
    )


async def handle_first_login(email: str, tenant_id_claim: Optional[int], user_password: Optional[str] = None) -> dict:
    try:
        _ensure_tables()
        logger.info("onboarding:first_login start email=%s tenant_id_claim=%s", email, tenant_id_claim)
        # Optional: Odoo-first HTTP ensure (disabled by default)
        if _truthy('ODOO_ENABLE_HTTP_PROVISION', 'false'):
            _ensure_odoo_db_first_by_email(email)
        # Create/find tenant rows in app DB
        tid = _ensure_tenant_and_user(email, tenant_id_claim)
        logger.info("onboarding:first_login resolved_tenant_id=%s email=%s", tid, email)
        _insert_or_update_status(tid, ONBOARDING_STARTING)
    except Exception as e:
        logger.exception("onboarding:first_login early failure email=%s", email)
        return {"tenant_id": None, "status": ONBOARDING_ERROR, "error": str(e)}
    try:
        # Determine if mapping already existed before this run
        mapping_exists_before = _has_active_mapping(tid)
        # Ensure mapping and smoke test / infer db
        _insert_or_update_status(tid, ONBOARDING_CREATING_ODOO)
        await _ensure_odoo_mapping(tid, email=email, admin_password_override=user_password)

        # Optional: automatically create Odoo DB (HTTP DB manager) and bootstrap admin
        if not mapping_exists_before and _truthy('ODOO_ENABLE_HTTP_PROVISION', 'false'):
            try:
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
        if not mapping_exists_before and _truthy('ODOO_ENABLE_OIDC_AUTOCONFIG', 'false'):
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

        # Move to seeding only if this was a newly provisioned mapping
        if not mapping_exists_before:
            _insert_or_update_status(tid, ONBOARDING_SEEDING)
            try:
                store = OdooStore(tenant_id=tid)
                await store.seed_baseline_entities(tenant_id=tid, email=email)
            except Exception as _seed_err:
                # Non-fatal; continue to ready but include error detail in status
                logger.exception("Onboarding baseline seed failed tenant_id=%s", tid)
                _insert_or_update_status(tid, ONBOARDING_SEEDING, str(_seed_err))
        _insert_or_update_status(tid, ONBOARDING_READY)
        logger.info("onboarding:first_login status=ready tenant_id=%s email=%s", tid, email)
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
    data = r.json() or {}
    if "error" in data:
        raise RuntimeError(f"odoo db.list failed: {data['error']}")
    return data.get("result", [])


def _odoo_db_create(server: str, master_pwd: str, db_name: str, admin_login: str, admin_password: str):
    import requests
    lang = os.getenv("ODOO_LANG", "en_US")
    country = os.getenv("ODOO_COUNTRY", "SG")
    # Try shortest/newer signatures first (common in recent Odoo), then legacy 9-arg
    attempts = [
        # 6-arg: master, db, demo, lang, admin_password, admin_login
        [master_pwd, db_name, False, lang, admin_password, admin_login],
        # 7-arg: + country_code (some versions accept country as positional)
        [master_pwd, db_name, False, lang, admin_password, admin_login, country],
        # 9-arg legacy: + country_code, phone, admin_login (again)
        [master_pwd, db_name, False, lang, admin_password, admin_login, country, "", admin_login],
    ]
    last_err = None
    for args in attempts:
        payload = {
            "jsonrpc": "2.0",
            "method": "call",
            "params": {"service": "db", "method": "create_database", "args": args},
        }
        try:
            r = requests.post(
                server.rstrip("/") + "/jsonrpc",
                json=payload,
                timeout=180,
                verify=_requests_verify(),
            )
            r.raise_for_status()
            data = r.json() or {}
            if "error" in data:
                # On signature mismatch, try next variant; otherwise fail
                err_msg = (
                    (data.get("error", {}).get("data", {}) or {}).get("message")
                    or data.get("error", {}).get("message")
                    or str(data.get("error"))
                )
                last_err = err_msg
                low = (err_msg or "").lower()
                if (
                    "too many positional arguments" in low
                    or ("missing" in low and "argument" in low)
                ):
                    continue
                raise RuntimeError(f"odoo create_database failed: {data['error']}")
            # Success
            return
        except Exception as e:
            last_err = str(e)
            continue
    # If all attempts failed, raise the last error
    raise RuntimeError(f"odoo create_database failed: {last_err}")


def _odoo_wait_db(server: str, db_name: str, timeout_s: int = 60, interval_s: float = 1.5) -> bool:
    """Wait until the Odoo DB shows up in the db manager list.

    Returns True when the DB appears within timeout; False otherwise.
    """
    deadline = time.time() + max(1, timeout_s)
    while time.time() < deadline:
        try:
            if db_name in _odoo_db_list(server):
                return True
        except Exception:
            pass
        time.sleep(max(0.5, interval_s))
    return False


def _odoo_admin_user_create(server: str, db_name: str, admin_login: str, admin_password: str, tenant_admin_email: str):
    import xmlrpc.client
    common = xmlrpc.client.ServerProxy(f"{server.rstrip('/')}/xmlrpc/2/common")
    uid = common.authenticate(db_name, admin_login, admin_password, {})
    if not uid:
        raise RuntimeError("odoo xmlrpc authentication failed for admin user")
    models = xmlrpc.client.ServerProxy(f"{server.rstrip('/')}/xmlrpc/2/object")

    def _resolve_group_system_id() -> int | None:
        # Try several methods to resolve 'base.group_system' across Odoo versions
        try:
            gid = models.execute_kw(
                db_name, uid, admin_password,
                'ir.model.data', 'xmlid_to_res_model_res_id', ['base.group_system']
            )
            # Some versions return (model, id)
            if isinstance(gid, (list, tuple)) and len(gid) == 2:
                return int(gid[1])
            if isinstance(gid, int):
                return int(gid)
        except Exception:
            pass
        try:
            pair = models.execute_kw(
                db_name, uid, admin_password,
                'ir.model.data', 'get_object_reference', ['base', 'group_system']
            )
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                return int(pair[1])
        except Exception:
            pass
        try:
            # Fallback search by XML-ID stored in ir_model_data
            recs = models.execute_kw(
                db_name, uid, admin_password,
                'ir.model.data', 'search_read', [[['module', '=', 'base'], ['name', '=', 'group_system']]],
                {'fields': ['res_id'], 'limit': 1}
            )
            if recs and recs[0].get('res_id'):
                return int(recs[0]['res_id'])
        except Exception:
            pass
        # Last resort: search group by implied system name
        try:
            gids = models.execute_kw(
                db_name, uid, admin_password,
                'res.groups', 'search', [[['technical_name', '=', 'group_system']]],
                {'limit': 1}
            )
            if gids:
                return int(gids[0])
        except Exception:
            pass
        return None

    group_system_id = _resolve_group_system_id()

    # Ensure a separate admin user if needed
    users = models.execute_kw(
        db_name, uid, admin_password,
        'res.users', 'search', [[['login', '=', tenant_admin_email]]]
    )
    if not users:
        vals = {
            'name': 'Tenant Admin',
            'login': tenant_admin_email,
            'email': tenant_admin_email,
        }
        if group_system_id:
            vals['groups_id'] = [(6, 0, [group_system_id])]
        models.execute_kw(db_name, uid, admin_password, 'res.users', 'create', [vals])


def _odoo_set_admin_credentials(
    server: str,
    db_name: str,
    admin_email: str,
    new_password: str,
    *,
    auth_login: Optional[str] = None,
    auth_password: Optional[str] = None,
) -> bool:
    """Set the tenant admin login/password via XML-RPC using template admin credentials.

    This ensures the Odoo admin login matches the signup email/password, even when the
    database already existed or the create_database signature didn’t set the login.
    Returns True on success, False if preconditions are missing.
    """
    import xmlrpc.client
    template_login = os.getenv('ODOO_TEMPLATE_ADMIN_LOGIN', 'admin')
    template_pw = os.getenv('ODOO_TEMPLATE_ADMIN_PASSWORD')
    # Prefer explicit auth creds; fallback to template admin env
    auth_login = auth_login or template_login
    auth_password = auth_password or template_pw
    if not auth_login or not auth_password:
        # No usable credentials to perform XML-RPC write
        return False

    def _verify_login(email: str, pwd: str) -> bool:
        try:
            c = xmlrpc.client.ServerProxy(f"{server.rstrip('/')}/xmlrpc/2/common")
            return bool(c.authenticate(db_name, email, pwd, {}))
        except Exception:
            return False

    common = xmlrpc.client.ServerProxy(f"{server.rstrip('/')}/xmlrpc/2/common")
    uid = common.authenticate(db_name, auth_login, auth_password, {})
    if not uid:
        raise RuntimeError("odoo xmlrpc authentication failed for credential used to set admin password (check dbfilter/password)")
    models = xmlrpc.client.ServerProxy(f"{server.rstrip('/')}/xmlrpc/2/object")
    # Identify admin user record (by template login or id=2 fallback)
    ids = models.execute_kw(
        db_name, uid, auth_password,
        'res.users', 'search', [[['login', '=', template_login]]], {'limit': 1}
    )
    if not ids:
        ids = [2]
    # Apply login + password (Odoo hashes the password)
    models.execute_kw(
        db_name, uid, auth_password, 'res.users', 'write', [ids, {'login': admin_email, 'active': True, 'password': new_password}]
    )
    try:
        recs = models.execute_kw(db_name, uid, auth_password, 'res.users', 'read', [ids, ['partner_id']])
        if recs and recs[0].get('partner_id'):
            pid = recs[0]['partner_id'][0]
            try:
                models.execute_kw(db_name, uid, auth_password, 'res.partner', 'write', [[pid], {'email': admin_email}])
            except Exception:
                pass
    except Exception:
        pass
    # Verify login works
    if _verify_login(admin_email, new_password):
        return True

    # Fallback: DB-level update with correct hash so Odoo can authenticate
    try:
        target_dsn = _odoo_admin_dsn_for(db_name)
        if not target_dsn:
            return False
        import psycopg2
        with closing(psycopg2.connect(target_dsn)) as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                # Check if password_crypt exists (Odoo 16+)
                cur.execute(
                    """
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name='res_users' AND column_name='password_crypt' LIMIT 1
                    """
                )
                has_crypt = bool(cur.fetchone())
                pwd_hash = _odoo_admin_hash(new_password)
                # Resolve admin user id (prefer id from xmlrpc search, fallback to 2)
                admin_id = ids[0] if ids else 2
                # Ensure login and password stored in the correct column
                if has_crypt:
                    cur.execute(
                        "UPDATE res_users SET login=%s, password_crypt=%s, active=TRUE WHERE id=%s",
                        (admin_email, pwd_hash, admin_id),
                    )
                else:
                    cur.execute(
                        "UPDATE res_users SET login=%s, password=%s, active=TRUE WHERE id=%s",
                        (admin_email, pwd_hash, admin_id),
                    )
                # Best-effort partner email sync
                try:
                    cur.execute("SELECT partner_id FROM res_users WHERE id=%s", (admin_id,))
                    r = cur.fetchone()
                    if r and r[0]:
                        cur.execute("UPDATE res_partner SET email=%s WHERE id=%s", (admin_email, int(r[0])))
                except Exception:
                    pass
        # Verify again via XML-RPC
        return _verify_login(admin_email, new_password)
    except Exception:
        return False


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

    - Derives db_name as <first label of domain>.
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
# --- Template-based provisioning fallback (works when list_db=false) ---

def _random_password(length: int = 24) -> str:
    import secrets
    return secrets.token_urlsafe(length)


def _odoo_admin_hash(password: str, rounds: int = 29000) -> str:
    """Generate a pbkdf2_sha512 hash using passlib."""
    return pbkdf2_sha512.hash(password, rounds=rounds)


def _odoo_admin_dsn_for(db_name: str) -> Optional[str]:
    tpl = (os.getenv('ODOO_BASE_DSN_TEMPLATE') or '').strip()
    if tpl:
        return tpl.format(db_name=db_name)
    dsn = (os.getenv('ODOO_POSTGRES_DSN') or '').strip()
    if not dsn:
        return None
    # Replace path with db_name
    try:
        from urllib.parse import urlparse, urlunparse
        u = urlparse(dsn)
        u = u._replace(path='/' + db_name)
        return urlunparse(u)
    except Exception:
        return None


def _pg_exec(dsn: str, sql: str, params: tuple | None = None) -> None:
    import psycopg2
    with closing(psycopg2.connect(dsn)) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(sql, params or ())


def _wait_pg_connect(dsn: str, timeout_s: int = 60) -> bool:
    import psycopg2
    deadline = time.time() + max(1, timeout_s)
    while time.time() < deadline:
        try:
            with closing(psycopg2.connect(dsn)) as conn:
                return True
        except Exception:
            time.sleep(1.0)
    return False


def _sanitize_db_name(name: str) -> str:
    import re
    return re.sub(r"[^a-z0-9_\-]", "", name.lower())


def _odoo_provision_from_template(db_name: str, admin_email: str, admin_password: str) -> bool:
    """Provision an Odoo DB by copying a template DB and setting admin credentials.

    Requires env ODOO_TEMPLATE_DB and DSN template/DSN to connect to Postgres.
    """
    template = (os.getenv('ODOO_TEMPLATE_DB') or '').strip()
    if not template:
        raise RuntimeError('Missing ODOO_TEMPLATE_DB for template-based provisioning')

    db_name = _sanitize_db_name(db_name)
    # Ensure SSH tunnel to Postgres is open (best-effort)
    try:
        admin_dsn_postgres = _odoo_admin_dsn_for('postgres')
        if not admin_dsn_postgres:
            raise RuntimeError('Cannot build Odoo admin DSN for postgres DB')
        # Open tunnel
        _ = OdooStore(dsn=admin_dsn_postgres)
    except Exception:
        pass

    admin_dsn_postgres = _odoo_admin_dsn_for('postgres')
    if not admin_dsn_postgres:
        raise RuntimeError('Cannot build Odoo admin DSN for postgres DB')

    # Create DB from template
    logger.info('odoo:template create db=%s from=%s', db_name, template)
    try:
        _pg_exec(admin_dsn_postgres, f'CREATE DATABASE "{db_name}" TEMPLATE "{template}"')
    except Exception as e:
        msg = str(e)
        if 'already exists' in msg.lower():
            logger.info('odoo:template db already exists, continuing db=%s', db_name)
        else:
            raise

    # Wait until new DB accepts connections
    target_dsn = _odoo_admin_dsn_for(db_name)
    if not target_dsn:
        raise RuntimeError('Cannot build DSN for new Odoo DB')
    if not _wait_pg_connect(target_dsn, timeout_s=60):
        raise RuntimeError(f'New DB {db_name} did not accept connections in time')

    # Prefer setting credentials via XML-RPC so Odoo applies the right hash
    try:
        server = os.getenv('ODOO_SERVER_URL')
        template_admin_login = os.getenv('ODOO_TEMPLATE_ADMIN_LOGIN', 'admin')
        template_admin_password = os.getenv('ODOO_TEMPLATE_ADMIN_PASSWORD')
        if server and template_admin_password:
            import xmlrpc.client
            common = xmlrpc.client.ServerProxy(f"{server.rstrip('/')}/xmlrpc/2/common")
            uid = common.authenticate(db_name, template_admin_login, template_admin_password, {})
            if uid:
                models = xmlrpc.client.ServerProxy(f"{server.rstrip('/')}/xmlrpc/2/object")
                # Find admin user (login='admin' or id=2)
                ids = models.execute_kw(db_name, uid, template_admin_password, 'res.users', 'search', [[['login', '=', template_admin_login]]], {'limit': 1})
                if not ids:
                    ids = [2]
                # Set login + password (Odoo will hash it)
                models.execute_kw(db_name, uid, template_admin_password, 'res.users', 'write', [ids, {'login': admin_email, 'password': admin_password}])
                # Update partner email as well
                recs = models.execute_kw(db_name, uid, template_admin_password, 'res.users', 'read', [ids, ['partner_id']])
                if recs and recs[0].get('partner_id'):
                    pid = recs[0]['partner_id'][0]
                    try:
                        models.execute_kw(db_name, uid, template_admin_password, 'res.partner', 'write', [[pid], {'email': admin_email}])
                    except Exception:
                        pass
                logger.info('odoo:template set admin via xmlrpc db=%s user=%s', db_name, admin_email)
                return True
    except Exception as e:
        logger.warning('odoo:template xmlrpc admin set failed db=%s error=%s', db_name, e)

    # Fallback: Update admin user credentials directly in DB (hash locally)
    logger.info('odoo:template set admin credentials (DB) db=%s email=%s', db_name, admin_email)
    pwd_hash = _odoo_admin_hash(admin_password)
    import psycopg2
    with closing(psycopg2.connect(target_dsn)) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            # Find admin user id
            admin_id = 2
            partner_id = None
            try:
                cur.execute("SELECT id, partner_id FROM res_users WHERE login='admin' ORDER BY id ASC LIMIT 1")
                r = cur.fetchone()
                if r:
                    admin_id = int(r[0]) if r[0] else admin_id
                    partner_id = int(r[1]) if r[1] else None
                else:
                    cur.execute("SELECT partner_id FROM res_users WHERE id=2")
                    r2 = cur.fetchone()
                    if r2 and r2[0]:
                        partner_id = int(r2[0])
            except Exception:
                pass
            # Determine which password column exists
            has_crypt = False
            try:
                cur.execute(
                    """
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name='res_users' AND column_name='password_crypt' LIMIT 1
                    """
                )
                has_crypt = bool(cur.fetchone())
            except Exception:
                has_crypt = False
            # Update login + password
            if has_crypt:
                cur.execute(
                    "UPDATE res_users SET login=%s, password_crypt=%s, active=TRUE WHERE id=%s",
                    (admin_email, pwd_hash, admin_id),
                )
            else:
                cur.execute(
                    "UPDATE res_users SET login=%s, password=%s, active=TRUE WHERE id=%s",
                    (admin_email, pwd_hash, admin_id),
                )
            try:
                if partner_id:
                    cur.execute(
                        "UPDATE res_partner SET email=%s WHERE id=%s",
                        (admin_email, partner_id),
                    )
            except Exception:
                pass
    return True
