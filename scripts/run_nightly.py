import asyncio
from typing import Optional, List

from src.database import get_conn
from src.jobs import run_staging_upsert, run_enrich_candidates


async def run_queued_jobs(limit: Optional[int] = None) -> int:
    """Pick up queued background_jobs of type 'staging_upsert' and process them sequentially.

    Returns the number of jobs processed.
    """
    jobs: list[tuple[int, str]] = []
    with get_conn() as conn, conn.cursor() as cur:
        sql = "SELECT job_id, job_type FROM background_jobs WHERE status='queued' AND job_type IN ('staging_upsert','enrich_candidates','icp_intake_process') ORDER BY job_id ASC"
        if isinstance(limit, int) and limit > 0:
            sql += f" LIMIT {int(limit)}"
        cur.execute(sql)
        rows = cur.fetchall() or []
        jobs = [(int(r[0]), str(r[1])) for r in rows if r and r[0] is not None]
    for jid, jtype in jobs:
        if jtype == 'staging_upsert':
            await run_staging_upsert(int(jid))
        elif jtype == 'enrich_candidates':
            await run_enrich_candidates(int(jid))
        elif jtype == 'icp_intake_process':
            from src.jobs import run_icp_intake_process
            await run_icp_intake_process(int(jid))
    return len(jobs)


async def run_tenant_partial(tenant_id: int, max_now: int = 10) -> None:
    """Compatibility function referenced by /scheduler/run_now.

    In this codebase, queued jobs are tenant-agnostic. This function simply
    processes up to `max_now` queued jobs.
    """
    await run_queued_jobs(limit=max_now)


async def run_all(limit: Optional[int] = None) -> int:
    """Process all queued staging_upsert jobs.

    This function is awaited by the scheduler. It simply dispatches to
    run_queued_jobs with an optional limit.
    """
    return await run_queued_jobs(limit=limit)


def list_active_tenants() -> List[int]:
    """Return a list of active tenant IDs for post-run acceptance checks.

    Falls back to an empty list if the mapping table is missing.
    """
    tenants: List[int] = []
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT tenant_id FROM odoo_connections WHERE active=TRUE ORDER BY tenant_id"
            )
            rows = cur.fetchall() or []
            tenants = [int(r[0]) for r in rows if r and r[0] is not None]
    except Exception:
        tenants = []
    return tenants


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Process queued staging_upsert jobs")
    ap.add_argument("--limit", type=int, default=None, help="Max number of jobs to process")
    args = ap.parse_args()
    asyncio.run(run_queued_jobs(limit=args.limit))
