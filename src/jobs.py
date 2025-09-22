import asyncio
import logging
import os
from typing import List, Optional

from src.database import get_conn
from psycopg2.extras import Json
import threading

# Reuse the batched/streaming implementation from lg_entry
try:
    from app.lg_entry import _upsert_companies_from_staging_by_industries as upsert_batched
except Exception:  # pragma: no cover
    upsert_batched = None  # type: ignore

UPSERT_MAX_PER_JOB = int(os.getenv("UPSERT_MAX_PER_JOB", "2000") or 2000)
STAGING_BATCH_SIZE = int(os.getenv("STAGING_BATCH_SIZE", "500") or 500)

log = logging.getLogger("jobs")


def _insert_job(tenant_id: Optional[int], terms: List[str]) -> int:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO background_jobs(tenant_id, job_type, status, params) VALUES (%s,'staging_upsert','queued', %s) RETURNING job_id",
            (tenant_id, Json({"terms": terms})),
        )
        row = cur.fetchone()
        return int(row[0]) if row else 0


def enqueue_staging_upsert(tenant_id: Optional[int], terms: List[str]) -> dict:
    """Queue a staging_upsert job for the nightly runner without executing now.

    Nightly execution can call run_staging_upsert(job_id) or a dispatcher that
    picks up queued jobs. This function only enqueues and returns the job id.
    """
    job_id = _insert_job(tenant_id, terms)
    return {"job_id": job_id}


async def run_staging_upsert(job_id: int) -> None:
    # Mark running
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE background_jobs SET status='running', started_at=now() WHERE job_id=%s",
            (job_id,),
        )
    processed = 0
    total = 0
    # Load params
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT params FROM background_jobs WHERE job_id=%s", (job_id,))
        r = cur.fetchone()
        params = (r and r[0]) or {}
        terms = [((t or '').strip().lower()) for t in (params.get('terms') or []) if (t or '').strip()]
    try:
        if not upsert_batched:
            raise RuntimeError("upsert function unavailable")
        # Call batched upsert once (internally streams and batches)
        processed = upsert_batched(terms)
        total = processed
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE background_jobs SET status='done', processed=%s, total=%s, ended_at=now() WHERE job_id=%s",
                (processed, total, job_id),
            )
    except Exception as e:  # pragma: no cover
        log.exception("staging_upsert job failed: %s", e)
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE background_jobs SET status='error', error=%s, processed=%s, total=%s, ended_at=now() WHERE job_id=%s",
                (str(e), processed, total, job_id),
            )
