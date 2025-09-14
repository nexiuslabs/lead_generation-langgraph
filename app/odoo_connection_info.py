from __future__ import annotations

import logging
from typing import Optional, Dict, Any

from src.database import get_conn
from app.odoo_store import OdooStore
import os

logger = logging.getLogger("onboarding")


def _infer_db_from_dsn() -> Optional[str]:
    try:
        from src.settings import ODOO_POSTGRES_DSN
        if not ODOO_POSTGRES_DSN:
            return None
        from urllib.parse import urlparse
        u = urlparse(ODOO_POSTGRES_DSN)
        path = (u.path or "/").lstrip("/")
        return path or None
    except Exception:
        return None


def _resolve_tenant_id(email: str, claim_tid: Optional[int]) -> Optional[int]:
    # Priority 1: DSN → active odoo_connections mapping
    inferred_db = _infer_db_from_dsn()
    if inferred_db:
        try:
            with get_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT tenant_id FROM odoo_connections WHERE db_name=%s AND active=TRUE LIMIT 1",
                    (inferred_db,),
                )
                row = cur.fetchone()
                if row:
                    return int(row[0])
        except Exception:
            pass

    # Priority 2: tenant_id from claim (if present)
    if claim_tid is not None:
        return int(claim_tid)

    # Priority 3: Existing user mapping by email
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT tenant_id FROM tenant_users WHERE user_id=%s LIMIT 1", (email,))
            row = cur.fetchone()
            if row:
                return int(row[0])
    except Exception:
        pass
    return None


def _current_odoo_db_name(tenant_id: int) -> Optional[str]:
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT db_name FROM odoo_connections WHERE tenant_id=%s", (tenant_id,))
            row = cur.fetchone()
            if row and row[0]:
                return str(row[0])
    except Exception:
        pass
    return None


async def get_odoo_connection_info(email: str, claim_tid: Optional[int]) -> Dict[str, Any]:
    tid = _resolve_tenant_id(email, claim_tid)
    db_name = None
    base_url = None
    if tid is not None:
        try:
            with get_conn() as conn, conn.cursor() as cur:
                cur.execute("SELECT db_name, base_url FROM odoo_connections WHERE tenant_id=%s", (tid,))
                row = cur.fetchone()
                if row:
                    db_name = row[0]
                    base_url = row[1]
        except Exception:
            db_name = _current_odoo_db_name(tid)

    ready = False
    error = None
    login_url = None
    if tid is not None:
        try:
            store = OdooStore(tenant_id=int(tid))
            await store.connectivity_smoke_test()
            ready = True
        except Exception as e:
            error = str(e)
    # Build a login URL that works with subdomain dbfilter or db param fallback
    try:
        if base_url:
            # base_url is expected to be tenant-specific, e.g. https://<db>.example.com
            login_url = f"{base_url.rstrip('/')}/web/login"
        elif db_name:
            server = (os.getenv("ODOO_SERVER_URL") or "").rstrip("/")
            if server:
                login_url = f"{server}/web/login?db={db_name}"
    except Exception:
        login_url = None

    logger.info(
        "session:odoo_info email=%s tenant_id=%s db_name=%s ready=%s",
        email,
        tid,
        db_name,
        ready,
    )

    return {
        "email": email,
        "tenant_id": tid,
        "odoo": {
            "db_name": db_name,
            "base_url": base_url,
            "ready": ready,
            "error": error,
            "login_url": login_url,
        },
    }
