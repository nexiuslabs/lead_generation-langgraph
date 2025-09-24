import asyncio
from typing import Optional

from src.database import get_conn
from src.jobs import run_staging_upsert


async def run_queued_jobs(limit: Optional[int] = None) -> int:
    """Pick up queued background_jobs of type 'staging_upsert' and process them sequentially.

    Returns the number of jobs processed.
    """
    jobs: list[int] = []
    with get_conn() as conn, conn.cursor() as cur:
        sql = "SELECT job_id FROM background_jobs WHERE job_type='staging_upsert' AND status='queued' ORDER BY job_id ASC"
        if isinstance(limit, int) and limit > 0:
            sql += f" LIMIT {int(limit)}"
        cur.execute(sql)
        rows = cur.fetchall() or []
        jobs = [int(r[0]) for r in rows if r and r[0] is not None]
    for jid in jobs:
        await run_staging_upsert(int(jid))
    return len(jobs)


async def run_tenant_partial(tenant_id: int, max_now: int = 10) -> None:
    """Compatibility function referenced by /scheduler/run_now.

    In this codebase, queued jobs are tenant-agnostic. This function simply
    processes up to `max_now` queued jobs.
    """
    await run_queued_jobs(limit=max_now)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Process queued staging_upsert jobs")
    ap.add_argument("--limit", type=int, default=None, help="Max number of jobs to process")
    args = ap.parse_args()
    asyncio.run(run_queued_jobs(limit=args.limit))

