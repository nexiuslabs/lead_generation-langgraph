import os
import sys
import psycopg2

# Ensure project root is on sys.path so we can import src.settings
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from src.settings import POSTGRES_DSN  # type: ignore
except Exception as e:
    print("ERROR: Could not import POSTGRES_DSN from src/settings.py:", e)
    sys.exit(1)

MIGRATION_FILE = os.path.join(ROOT, "app", "migrations", "001_presdr_odoo.sql")


def main():
    if not POSTGRES_DSN:
        print("ERROR: POSTGRES_DSN not set in environment/.env")
        sys.exit(1)
    if not os.path.exists(MIGRATION_FILE):
        print(f"ERROR: Migration file not found: {MIGRATION_FILE}")
        sys.exit(1)

    print(f"Applying Odoo migration: {MIGRATION_FILE}")
    with open(MIGRATION_FILE, "r", encoding="utf-8") as f:
        sql = f.read()

    conn = psycopg2.connect(dsn=POSTGRES_DSN)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql)
        print("✅ Migration applied (safe/idempotent)")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
