import os
import sys
import glob
import psycopg2
from psycopg2.extras import DictCursor

from dotenv import load_dotenv


def get_dsn() -> str:
    # Load .env so local runs work without exporting
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
    dsn = os.getenv("POSTGRES_DSN") or os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("POSTGRES_DSN or DATABASE_URL must be set in env/.env")
    return dsn


def ensure_schema_table(cur):
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            filename TEXT PRIMARY KEY,
            applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """
    )


def already_applied(cur, filename: str) -> bool:
    cur.execute("SELECT 1 FROM schema_migrations WHERE filename=%s", (filename,))
    return cur.fetchone() is not None


def mark_applied(cur, filename: str):
    cur.execute(
        "INSERT INTO schema_migrations(filename) VALUES (%s) ON CONFLICT DO NOTHING",
        (filename,),
    )


def run():
    dsn = get_dsn()
    migrations_dir = os.path.join(os.path.dirname(__file__), "..", "migrations")
    files = sorted(glob.glob(os.path.join(migrations_dir, "*.sql")))
    if not files:
        print("No migrations found.")
        return 0
    with psycopg2.connect(dsn) as conn:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            ensure_schema_table(cur)
            ran = 0
            for path in files:
                fname = os.path.basename(path)
                if already_applied(cur, fname):
                    continue
                with open(path, "r", encoding="utf-8") as fh:
                    sql = fh.read()
                print(f"Applying migration: {fname}")
                cur.execute(sql)
                mark_applied(cur, fname)
                ran += 1
            print(f"Migrations applied: {ran}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(run())
    except Exception as e:
        print(f"Migration failed: {e}")
        sys.exit(1)

