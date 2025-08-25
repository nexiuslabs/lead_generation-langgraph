import os
import sys
import psycopg2

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from src.settings import POSTGRES_DSN  # type: ignore
except Exception as e:
    print("ERROR: Could not import POSTGRES_DSN from src/settings.py:", e)
    sys.exit(1)


def main():
    if not POSTGRES_DSN:
        print("ERROR: POSTGRES_DSN not set")
        sys.exit(1)
    conn = psycopg2.connect(dsn=POSTGRES_DSN)
    try:
        with conn:
            with conn.cursor() as cur:
                print("Refreshing materialized view icp_candidate_companies…")
                cur.execute("REFRESH MATERIALIZED VIEW icp_candidate_companies;")
        print("✅ Refreshed icp_candidate_companies")
    finally:
        conn.close()


if __name__ == "__main__":
    main()

