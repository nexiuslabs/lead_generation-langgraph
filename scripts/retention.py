from src.database import get_conn


def run_retention() -> None:
    with get_conn() as conn, conn.cursor() as cur:
        # Keep detailed events for 30 days
        try:
            cur.execute("DELETE FROM run_event_logs WHERE ts < NOW() - INTERVAL '30 days'")
        except Exception:
            pass
        # Keep QA samples for 90 days
        try:
            cur.execute("DELETE FROM qa_samples WHERE created_at < NOW() - INTERVAL '90 days'")
        except Exception:
            pass


if __name__ == "__main__":
    run_retention()

