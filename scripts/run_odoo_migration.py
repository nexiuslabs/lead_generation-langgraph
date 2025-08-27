import os
import subprocess
import sys
import time
from urllib.parse import urlparse, urlunparse

import psycopg2

# Ensure project root is on sys.path so we can import src.settings
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from src.settings import ODOO_POSTGRES_DSN  # type: ignore
except Exception as e:
    print("ERROR: Could not import ODOO_POSTGRES_DSN from src/settings.py:", e)
    sys.exit(1)

# Optional SSH tunnel configuration
SSH_HOST = os.getenv("SSH_HOST")
SSH_PORT = int(os.getenv("SSH_PORT", "22"))
SSH_USER = os.getenv("SSH_USER")
SSH_PASSWORD = os.getenv("SSH_PASSWORD")
DB_HOST_IN_DROPLET = os.getenv("DB_HOST_IN_DROPLET")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "demo")
DB_USER = os.getenv("DB_USER", "odoo")
DB_PASSWORD = os.getenv("DB_PASSWORD", "odoo")
LOCAL_PORT = int(os.getenv("LOCAL_PORT", "25060"))

MIGRATION_FILE = os.path.join(ROOT, "app", "migrations", "001_presdr_odoo.sql")


def _mask_dsn(dsn: str) -> str:
    """Return DSN with password stripped for display."""
    try:
        parts = urlparse(dsn)
        if "@" in parts.netloc and ":" in parts.netloc.split("@")[0]:
            user, _ = parts.netloc.split("@")[0].split(":", 1)
            host = parts.netloc.split("@", 1)[1]
            netloc = f"{user}:***@{host}"
        else:
            netloc = parts.netloc
        return urlunparse(
            (
                parts.scheme,
                netloc,
                parts.path,
                parts.params,
                parts.query,
                parts.fragment,
            )
        )
    except Exception:
        return "<hidden>"


def _open_ssh_tunnel():
    """Open an SSH tunnel if credentials are provided.

    Returns the subprocess handle or None if no tunnel was started.
    """
    if not (SSH_HOST and SSH_USER and DB_HOST_IN_DROPLET):
        return None
    cmd = [
        "ssh",
        "-N",
        "-L",
        f"{LOCAL_PORT}:{DB_HOST_IN_DROPLET}:{DB_PORT}",
        f"{SSH_USER}@{SSH_HOST}",
        "-p",
        str(SSH_PORT),
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "StrictHostKeyChecking=no",
    ]
    if SSH_PASSWORD:
        cmd = ["sshpass", "-p", SSH_PASSWORD, *cmd]
    proc = subprocess.Popen(cmd)
    time.sleep(1)  # give tunnel a moment
    return proc


def main():
    dsn = ODOO_POSTGRES_DSN
    if not dsn:
        dsn = f"postgresql://{DB_USER}:{DB_PASSWORD}@localhost:{LOCAL_PORT}/{DB_NAME}"

    tunnel = _open_ssh_tunnel()

    if not dsn:
        print("ERROR: Odoo DSN not configured")
        sys.exit(1)
    if not os.path.exists(MIGRATION_FILE):
        print(f"ERROR: Migration file not found: {MIGRATION_FILE}")
        sys.exit(1)

    print("Using Odoo Postgres DSN:", _mask_dsn(dsn))
    conn = psycopg2.connect(dsn=dsn)
    try:
        with conn:
            with conn.cursor() as cur:
                # Check for required Odoo tables
                def has_table(name: str) -> bool:
                    cur.execute(
                        """
                        SELECT EXISTS (
                          SELECT 1
                          FROM pg_class c
                          JOIN pg_namespace n ON n.oid = c.relnamespace
                          WHERE c.relkind = 'r' AND c.relname = %s
                        );
                        """,
                        (name,),
                    )
                    return bool(cur.fetchone()[0])

                has_res_partner = has_table("res_partner")
                has_crm_lead = has_table("crm_lead")

                if not (has_res_partner and has_crm_lead):
                    print("\n❌ Odoo core tables not found in the target database.")
                    print("   Expected tables: res_partner, crm_lead")
                    print("   Current DSN:", _mask_dsn(dsn))
                    print("\nAction needed:")
                    print(
                        " - Point ODOO_POSTGRES_DSN or DB_* env vars to the actual Odoo Postgres database."
                    )
                    print(
                        " - Ensure the Odoo server has initialized its schema (start Odoo once).\n"
                    )
                    sys.exit(2)

                print("✅ Odoo core tables verified")
                print(f"Applying Odoo migration: {MIGRATION_FILE}")
                with open(MIGRATION_FILE, "r", encoding="utf-8") as f:
                    sql = f.read()
                cur.execute(sql)
        print("✅ Migration applied (safe/idempotent)")
    finally:
        conn.close()
        if tunnel:
            tunnel.terminate()


if __name__ == "__main__":
    main()
