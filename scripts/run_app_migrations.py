import os
import sys
import glob
import psycopg2

# Ensure project root import path
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from src.settings import POSTGRES_DSN  # type: ignore
except Exception as e:
    print("ERROR: Could not import POSTGRES_DSN from src/settings.py:", e)
    sys.exit(1)

MIGRATIONS_DIR = os.path.join(ROOT, "app", "migrations")


def main():
    if not POSTGRES_DSN:
        print("ERROR: POSTGRES_DSN not set in environment/.env")
        sys.exit(1)
    files = sorted(glob.glob(os.path.join(MIGRATIONS_DIR, "[0-9][0-9][0-9]_*.sql")))
    # Skip Odoo-specific 001 migration; apply generic ones (002+)
    files = [f for f in files if not os.path.basename(f).startswith("001_")]
    if not files:
        print("No app migrations to apply.")
        return
    conn = psycopg2.connect(dsn=POSTGRES_DSN)
    try:
        with conn:
            with conn.cursor() as cur:
                for path in files:
                    print(f"Applying migration: {os.path.basename(path)}")
                    with open(path, "r", encoding="utf-8") as f:
                        sql = f.read()
                    cur.execute(sql)
        print("✅ App migrations applied.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()

