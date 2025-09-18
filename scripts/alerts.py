import os
import httpx
from src.database import get_conn
import os

SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK_URL")


def _post(msg: str) -> None:
    if not SLACK_WEBHOOK:
        return
    try:
        httpx.post(SLACK_WEBHOOK, json={"text": msg}, timeout=5.0)
    except Exception:
        pass


def check_last_run_alerts() -> None:
    """Post a simple alert to Slack when the last run per tenant did not succeed."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
          WITH last_runs AS (
            SELECT tenant_id, MAX(run_id) AS run_id
            FROM enrichment_runs
            GROUP BY tenant_id
          )
          SELECT l.tenant_id, r.run_id, r.status
          FROM last_runs l
          JOIN enrichment_runs r USING(run_id)
            """
        )
        last = cur.fetchall()
        for tid, run_id, status in last:
            if str(status).lower() != "succeeded":
                _post(f"⚠️ Nightly run {run_id} for tenant {tid} status={status}")

    # Additional alerts per PRD
    crawl_err_max = float(os.getenv("ALERT_TAVILY_ERROR_RATE_MAX", "0.30") or 0.30)
    candidate_floor = int(os.getenv("ALERT_CANDIDATE_FLOOR", "20") or 20)
    qa_min_pass = float(os.getenv("ALERT_QA_PASS_RATE_MIN", "0.80") or 0.80)

    with get_conn() as conn, conn.cursor() as cur:
        # Crawl/Tavily vendor error rate on last run per tenant
        cur.execute(
            """
            WITH last_runs AS (
              SELECT tenant_id, MAX(run_id) AS run_id FROM enrichment_runs GROUP BY tenant_id
            )
            SELECT v.tenant_id, v.run_id,
                   COALESCE(NULLIF(v.calls,0),0) AS calls,
                   v.errors
            FROM last_runs l
            JOIN run_vendor_usage v ON v.run_id=l.run_id AND v.tenant_id=l.tenant_id AND v.vendor='tavily'
            """
        )
        for tid, run_id, calls, errors in cur.fetchall():
            try:
                rate = (errors / calls) if calls else 0.0
            except Exception:
                rate = 0.0
            if rate > crawl_err_max:
                _post(f"⚠️ High Tavily error rate {rate:.0%} for tenant {tid} run {run_id} (>{crawl_err_max:.0%})")

        # Vendor rate limits / quota
        cur.execute(
            """
            WITH last_runs AS (
              SELECT tenant_id, MAX(run_id) AS run_id FROM enrichment_runs GROUP BY tenant_id
            )
            SELECT v.tenant_id, v.run_id, v.vendor, v.rate_limit_hits, v.quota_exhausted
            FROM last_runs l
            JOIN run_vendor_usage v ON v.run_id=l.run_id AND v.tenant_id=l.tenant_id
            WHERE v.rate_limit_hits > 0 OR v.quota_exhausted
            """
        )
        for tid, run_id, vendor, rl, quota in cur.fetchall():
            if rl and rl > 0:
                _post(f"⚠️ Vendor {vendor} rate limits for tenant {tid} run {run_id}: hits={rl}")
            if quota:
                _post(f"⚠️ Vendor {vendor} quota exhausted for tenant {tid} run {run_id}")

        # Low candidate count (post-filter)
        cur.execute(
            """
            WITH last_runs AS (
              SELECT tenant_id, MAX(run_id) AS run_id FROM enrichment_runs GROUP BY tenant_id
            )
            SELECT s.tenant_id, s.run_id, s.candidates, s.processed
            FROM last_runs l
            JOIN run_summaries s USING(run_id, tenant_id)
            """
        )
        for tid, run_id, candidates, processed in cur.fetchall():
            if int(candidates or 0) < candidate_floor:
                _post(f"⚠️ Low candidates {candidates} (<{candidate_floor}) for tenant {tid} run {run_id}")

        # QA pass rate (only count reviewed rows)
        cur.execute(
            """
            WITH last_runs AS (
              SELECT tenant_id, MAX(run_id) AS run_id FROM enrichment_runs GROUP BY tenant_id
            ), qa AS (
              SELECT q.tenant_id, q.run_id,
                     SUM(CASE WHEN q.result='pass' THEN 1 ELSE 0 END) AS pass_cnt,
                     SUM(CASE WHEN q.result='fail' THEN 1 ELSE 0 END) AS fail_cnt
              FROM qa_samples q JOIN last_runs l USING(run_id, tenant_id)
              GROUP BY q.tenant_id, q.run_id
            )
            SELECT tenant_id, run_id, pass_cnt, fail_cnt FROM qa
            """
        )
        for tid, run_id, pass_cnt, fail_cnt in cur.fetchall():
            total = int(pass_cnt or 0) + int(fail_cnt or 0)
            if total > 0:
                rate = (pass_cnt / total) if total else 1.0
                if rate < qa_min_pass:
                    _post(f"⚠️ QA pass rate {rate:.0%} (<{qa_min_pass:.0%}) for tenant {tid} run {run_id}")


if __name__ == "__main__":
    check_last_run_alerts()
