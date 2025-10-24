import os
import sys
import json
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


def main() -> None:
    if not POSTGRES_DSN:
        print("ERROR: POSTGRES_DSN not set in environment/.env")
        sys.exit(1)
    conn = psycopg2.connect(dsn=POSTGRES_DSN)
    try:
        with conn:
            with conn.cursor() as cur:
                # Find duplicate groups for Next-40 (queued/running) where batch_id present
                cur.execute(
                    """
                    SELECT tenant_id,
                           params->>'batch_id' AS batch_id,
                           json_agg(json_build_object('job_id', job_id, 'status', status) ORDER BY job_id DESC) AS jobs
                      FROM background_jobs
                     WHERE job_type='web_discovery_bg_enrich'
                       AND status IN ('queued','running')
                       AND params ? 'batch_id'
                  GROUP BY tenant_id, params->>'batch_id'
                    HAVING COUNT(*) > 1
                    """
                )
                groups = cur.fetchall() or []
                total_pruned = 0
                for tenant_id, batch_id, jobs_json in groups:
                    jobs = jobs_json or []
                    # Determine the job to keep:
                    # - If any running: keep the latest running (highest job_id)
                    # - Else: keep the latest queued (highest job_id)
                    keep_id = None
                    running_ids = [int(j["job_id"]) for j in jobs if (j.get("status") == "running")]
                    if running_ids:
                        keep_id = max(running_ids)
                    else:
                        keep_id = int(jobs[0]["job_id"])  # jobs ordered DESC by job_id
                    prune_ids = [int(j["job_id"]) for j in jobs if int(j["job_id"]) != keep_id]
                    if not prune_ids:
                        continue
                    # Mark duplicates as done with an explanatory note to avoid retries
                    cur.execute(
                        """
                        UPDATE background_jobs
                           SET status='done',
                               error=COALESCE(error,'') || CASE WHEN COALESCE(error,'') = '' THEN '' ELSE '; ' END || 'duplicate pruned by cleanup',
                               ended_at=now()
                         WHERE job_type='web_discovery_bg_enrich'
                           AND job_id = ANY(%s)
                        """,
                        (prune_ids,),
                    )
                    total_pruned += cur.rowcount or 0
                    print(f"Pruned {len(prune_ids)} duplicates for tenant={tenant_id} batch_id={batch_id}; kept job_id={keep_id}")
        print(f"✅ Cleanup complete. Total duplicates pruned: {total_pruned}")
    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()

