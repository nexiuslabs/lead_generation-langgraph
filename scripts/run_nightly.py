import asyncio
import logging
import os
from typing import Optional, List
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
import re

from src.database import get_conn
from src.jobs import run_staging_upsert, run_enrich_candidates, enqueue_staging_upsert
from datetime import timedelta, datetime


def _configure_logging() -> None:
    env = (os.getenv("ENVIRONMENT") or os.getenv("PY_ENV") or os.getenv("NODE_ENV") or "dev").strip().lower()
    log_dir = os.getenv("TROUBLESHOOT_API_LOG_DIR")
    if not log_dir and env in {"dev", "development", "local", "localhost"}:
        log_dir = ".log_api"
    if not log_dir:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        return
    try:
        path = Path(log_dir).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        file_path = path / "nightly.log"
        root = logging.getLogger()
        if not any(isinstance(h, TimedRotatingFileHandler) and getattr(h, "baseFilename", None) == str(file_path) for h in root.handlers):
            fh = TimedRotatingFileHandler(
                file_path,
                when="midnight",
                interval=1,
                backupCount=14,
                encoding="utf-8",
                utc=True,
            )
            fh.suffix = "%Y-%m-%d"
            fh.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}$")  # type: ignore[attr-defined]
            fh.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S"))
            root.addHandler(fh)
        if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, TimedRotatingFileHandler) for h in root.handlers):
            sh = logging.StreamHandler()
            sh.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S"))
            root.addHandler(sh)
        if root.level == logging.NOTSET:
            root.setLevel(logging.INFO)
    except Exception:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


async def run_queued_jobs(limit: Optional[int] = None) -> int:
    """Pick up queued background_jobs of type 'staging_upsert' and process them sequentially.

    Returns the number of jobs processed.
    """
    jobs: list[tuple[int, str]] = []
    with get_conn() as conn, conn.cursor() as cur:
        # Maintenance: requeue stale 'running' jobs and retry eligible 'error' jobs
        try:
            # Staleness threshold for 'running' jobs (minutes)
            try:
                stale_minutes = int(os.getenv("JOB_STALE_MINUTES", "30") or 30)
            except Exception:
                stale_minutes = 30
            # Max automatic retries for 'error' jobs
            try:
                max_retries = int(os.getenv("JOB_MAX_RETRIES", "2") or 2)
            except Exception:
                max_retries = 2
            # 1) Requeue stale running ACRA jobs
            cur.execute(
                """
                UPDATE background_jobs b
                SET status='queued', started_at=NULL, ended_at=NULL
                WHERE b.status='running'
                  AND b.job_type IN ('staging_upsert','enrich_candidates')
                  AND b.started_at IS NOT NULL
                  AND b.started_at < now() - make_interval(mins => %s)
                  AND (b.ended_at IS NULL OR b.ended_at < b.started_at)
                RETURNING b.job_id
                """,
                (stale_minutes,),
            )
            _ = cur.fetchall()  # consume
            # 2) Requeue eligible error ACRA jobs with bounded retries (stored in params.retries)
            # Ensure params is jsonb; increment retries atomically
            cur.execute(
                """
                UPDATE background_jobs b
                SET status='queued', started_at=NULL, ended_at=NULL, error=NULL,
                    params = jsonb_set(COALESCE(b.params, '{}'::jsonb), '{retries}',
                                       to_jsonb(COALESCE((b.params->>'retries')::int, 0) + 1), true)
                WHERE b.status='error'
                  AND b.job_type IN ('staging_upsert','enrich_candidates')
                  AND COALESCE((b.params->>'retries')::int, 0) < %s
                RETURNING b.job_id
                """,
                (max_retries,),
            )
            _ = cur.fetchall()
        except Exception:
            # Maintenance is best-effort; continue to job selection
            pass
        # Nightly runner: only process ACRA pipeline jobs (staging_upsert → enrich_candidates).
        # Web discovery next‑40 runs immediately on enqueue and is excluded from nightly.
        sql = (
            "SELECT job_id, job_type FROM background_jobs "
            "WHERE status='queued' AND job_type IN ('staging_upsert','enrich_candidates') "
            "ORDER BY CASE job_type "
            "WHEN 'staging_upsert' THEN 0 "
            "WHEN 'enrich_candidates' THEN 1 "
            "ELSE 2 END, job_id ASC"
        )
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
        # Other job types are intentionally skipped by the nightly runner
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
    # 0) Bootstrap: for first-time tenants, auto-enqueue an ACRA staging_upsert
    #    based on their accepted ICP (preferred titles / SSIC codes) so the
    #    nightly run can source candidates from staging_acra_companies.
    try:
        _bootstrap_first_time_acra_jobs()
    except Exception:
        # Best-effort; continue with queued jobs even if bootstrap fails
        pass
    # 1) Process queued jobs (staging_upsert -> enrich_candidates)
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


def _tenant_has_acra_jobs(tenant_id: int) -> bool:
    """Return True if any ACRA-related background jobs exist for tenant.

    We treat job types 'staging_upsert' and 'enrich_candidates' as ACRA jobs
    for the nightly runner. Any status qualifies to avoid duplicate bootstraps.
    """
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT 1
                FROM background_jobs
                WHERE tenant_id=%s AND job_type IN ('staging_upsert','enrich_candidates')
                LIMIT 1
                """,
                (tenant_id,),
            )
            return cur.fetchone() is not None
    except Exception:
        return True  # be conservative; assume present to avoid repeated enqueue


def _resolve_acra_terms_for_tenant(tenant_id: int) -> list[str]:
    """Resolve industry/SSIC titles for a tenant from icp_rules.

    Priority:
      1) payload.preferred_titles (list of strings)
      2) payload.industries (list of strings)
      3) payload.ssic_codes -> map to ssic_ref.title
    Returns a deduped list of non-empty strings.
    """
    titles: list[str] = []
    payload: dict | None = None
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT payload
                FROM icp_rules
                WHERE tenant_id=%s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (tenant_id,),
            )
            row = cur.fetchone()
            payload = row[0] if row and row[0] else None
    except Exception:
        payload = None
    if isinstance(payload, dict):
        # preferred_titles or industries straight from payload
        for key in ("preferred_titles", "industries"):
            val = payload.get(key)
            if isinstance(val, list):
                titles.extend([str(x).strip() for x in val if (str(x) or "").strip()])
        # Map ssic_codes to titles if no titles found
        if not titles:
            codes = payload.get("ssic_codes")
            if isinstance(codes, list) and codes:
                try:
                    with get_conn() as conn, conn.cursor() as cur:
                        cur.execute(
                            "SELECT DISTINCT title FROM ssic_ref WHERE regexp_replace(code::text,'\\D','','g') = ANY(%s::text[])",
                            ([str(c) for c in codes if str(c).strip()],),
                        )
                        rows = cur.fetchall() or []
                        titles.extend([str(r[0]).strip() for r in rows if r and r[0]])
                except Exception:
                    pass
    # Dedupe and normalize
    seen = set()
    out: list[str] = []
    for t in titles:
        tt = (t or "").strip()
        if tt and tt.lower() not in seen:
            seen.add(tt.lower())
            out.append(tt)
    return out


def _bootstrap_first_time_acra_jobs() -> None:
    """Auto-enqueue a staging_upsert job per tenant on first nightly run.

    Criteria for "first-time": no existing background_jobs of types
    ('staging_upsert','enrich_candidates') for the tenant. When true, resolve
    terms from icp_rules and enqueue one 'staging_upsert' job for the nightly.
    """
    tenants = list_active_tenants()
    for tid in tenants:
        try:
            if _tenant_has_acra_jobs(tid):
                continue  # already initialized or running
            terms = _resolve_acra_terms_for_tenant(tid)
            if not terms:
                # No ICP-derived terms; skip bootstrap for this tenant
                continue
            enqueue_staging_upsert(tid, terms)
        except Exception:
            # Skip tenant on error; continue others
            continue


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Process queued staging_upsert jobs")
    ap.add_argument("--limit", type=int, default=None, help="Max number of jobs to process")
    args = ap.parse_args()
    _configure_logging()
    asyncio.run(run_queued_jobs(limit=args.limit))
