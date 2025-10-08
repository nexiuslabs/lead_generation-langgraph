import asyncio
import logging
import os
from typing import List, Optional, Dict, Any

from src.database import get_conn
from psycopg2.extras import Json
import threading

# Reuse the batched/streaming implementation from lg_entry
try:
    from app.lg_entry import _upsert_companies_from_staging_by_industries as upsert_batched
except Exception:  # pragma: no cover
    upsert_batched = None  # type: ignore

try:
    from src.enrichment import enrich_company_with_tavily  # async
except Exception:  # pragma: no cover
    enrich_company_with_tavily = None  # type: ignore

UPSERT_MAX_PER_JOB = int(os.getenv("UPSERT_MAX_PER_JOB", "2000") or 2000)
STAGING_BATCH_SIZE = int(os.getenv("STAGING_BATCH_SIZE", "500") or 500)
CRAWL_CONCURRENCY = int(os.getenv("CRAWL_CONCURRENCY", "4") or 4)

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


def enqueue_icp_intake_process(tenant_id: int) -> dict:
    """Queue a full intake pipeline run for a tenant (map→crawl→patterns)."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO background_jobs(tenant_id, job_type, status, params) VALUES (%s,'icp_intake_process','queued', '{}'::jsonb) RETURNING job_id",
            (tenant_id,),
        )
        row = cur.fetchone()
        return {"job_id": int(row[0]) if row and row[0] is not None else 0}


async def run_staging_upsert(job_id: int) -> None:
    import time
    t0 = time.perf_counter()
    log.info("{\"job\":\"staging_upsert\",\"phase\":\"start\",\"job_id\":%s}", job_id)
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
        # Best-effort: if terms look like industry/SSIC titles, enqueue enrichment job for the same selection
        try:
            codes: List[str] = []
            with get_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT regexp_replace(code::text,'\\D','','g') FROM ssic_ref WHERE LOWER(title) = ANY(%s::text[])",
                    ([t.strip().lower() for t in terms if (t or '').strip()],),
                )
                rows = cur.fetchall() or []
                codes = [str(r[0]) for r in rows if r and r[0] is not None]
            if codes:
                enqueue_enrich_candidates(None, codes)
        except Exception:
            pass
        dur_ms = int((time.perf_counter() - t0) * 1000)
        log.info("{\"job\":\"staging_upsert\",\"phase\":\"finish\",\"job_id\":%s,\"processed\":%s,\"duration_ms\":%s}", job_id, processed, dur_ms)
    except Exception as e:  # pragma: no cover
        log.exception("staging_upsert job failed: %s", e)
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE background_jobs SET status='error', error=%s, processed=%s, total=%s, ended_at=now() WHERE job_id=%s",
                (str(e), processed, total, job_id),
            )
        dur_ms = int((time.perf_counter() - t0) * 1000)
        log.info("{\"job\":\"staging_upsert\",\"phase\":\"error\",\"job_id\":%s,\"processed\":%s,\"duration_ms\":%s,\"error\":%s}", job_id, processed, dur_ms, str(e))


async def run_icp_intake_process(job_id: int) -> None:
    """Map seeds → crawl evidence (user + seeds) → refresh patterns → (optional) generate suggestions."""
    import time
    t0 = time.perf_counter()
    log.info("{\"job\":\"icp_intake_process\",\"phase\":\"start\",\"job_id\":%s}", job_id)
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("UPDATE background_jobs SET status='running', started_at=now() WHERE job_id=%s", (job_id,))
        cur.execute("SELECT tenant_id FROM background_jobs WHERE job_id=%s", (job_id,))
        row = cur.fetchone()
        tenant_id = int(row[0]) if row and row[0] is not None else None
    if not tenant_id:
        return
    # 1) Map seeds to ACRA evidence
    try:
        from src.icp_intake import map_seeds_to_evidence
        await asyncio.to_thread(map_seeds_to_evidence, tenant_id)
    except Exception:
        pass
    # 2) Crawl evidence for user site + seeds
    try:
        from src.icp_pipeline import collect_evidence_for_domain
        # Fetch last website apex
        website = None
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT answers_jsonb FROM icp_intake_responses WHERE tenant_id=%s ORDER BY submitted_at DESC LIMIT 1",
                (tenant_id,),
            )
            r = cur.fetchone()
            ans = (r and r[0]) or {}
            website = (ans.get("website") or ans.get("answers", {}).get("website") if isinstance(ans, dict) else None) or None
        tasks = []
        if website:
            try:
                from urllib.parse import urlparse
                apex = (urlparse(website).netloc or website)
            except Exception:
                apex = website
            tasks.append(collect_evidence_for_domain(tenant_id, None, apex))
        # Crawl seeds with domains
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT domain FROM customer_seeds WHERE tenant_id=%s AND NULLIF(TRIM(domain),'') IS NOT NULL", (tenant_id,))
            domains = [d[0] for d in (cur.fetchall() or []) if d and d[0]]
        if domains:
            for d in domains[:100]:  # soft cap
                tasks.append(collect_evidence_for_domain(tenant_id, None, d))
        if tasks:
            # run with limited concurrency
            sem = asyncio.Semaphore(CRAWL_CONCURRENCY)

            async def _guarded(coro):
                async with sem:
                    try:
                        await coro
                    except Exception:
                        pass

            await asyncio.gather(*[_guarded(t) for t in tasks])
    except Exception:
        pass
    # 3) Refresh patterns
    try:
        from src.icp_intake import refresh_icp_patterns
        await asyncio.to_thread(refresh_icp_patterns)
    except Exception:
        pass
    # 4) Done
    dur_ms = int((time.perf_counter() - t0) * 1000)
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE background_jobs SET status='done', processed=%s, total=%s, ended_at=now() WHERE job_id=%s",
            (0, 0, job_id),
        )
    log.info("{\"job\":\"icp_intake_process\",\"phase\":\"finish\",\"job_id\":%s,\"duration_ms\":%s}", job_id, dur_ms)


def enqueue_manual_research_enrich(tenant_id: Optional[int], company_ids: List[int]) -> dict:
    """Queue enrichment for a set of company IDs sourced from ResearchOps import.

    This is a lightweight per-company enrichment runner used to prioritize
    manual_research candidates per DevPlan19.
    """
    ids = [int(i) for i in (company_ids or []) if str(i).strip()]
    if not ids:
        return {"job_id": 0}
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO background_jobs(tenant_id, job_type, status, params) VALUES (%s,'manual_research_enrich','queued', %s) RETURNING job_id",
            (tenant_id, Json({"company_ids": ids})),
        )
        row = cur.fetchone()
        return {"job_id": int(row[0]) if row and row[0] is not None else 0}


async def run_manual_research_enrich(job_id: int) -> None:
    """Process a list of company_ids and run enrichment for each.

    Uses the same enrich function as other flows (Tavily/Apify pipeline),
    but scoped to the provided IDs.
    """
    import time
    t0 = time.perf_counter()
    log.info("{\"job\":\"manual_research_enrich\",\"phase\":\"start\",\"job_id\":%s}", job_id)
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("UPDATE background_jobs SET status='running', started_at=now() WHERE job_id=%s", (job_id,))
        cur.execute("SELECT params FROM background_jobs WHERE job_id=%s", (job_id,))
        row = cur.fetchone()
        params = (row and row[0]) or {}
        ids = [int(i) for i in (params.get('company_ids') or []) if str(i).strip()]
    processed = 0
    try:
        if not ids:
            raise RuntimeError("company_ids required")
        if enrich_company_with_tavily is None:
            raise RuntimeError("enrich unavailable")
        # Fetch names/uen for display and vendor input
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT company_id, name, uen FROM companies WHERE company_id = ANY(%s)",
                (ids,),
            )
            rows = cur.fetchall() or []
        for cid, name, uen in rows:
            try:
                await enrich_company_with_tavily(int(cid), name, uen)
                processed += 1
            except Exception:
                continue
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE background_jobs SET status='done', processed=%s, total=%s, ended_at=now() WHERE job_id=%s",
                (processed, len(ids), job_id),
            )
        dur_ms = int((time.perf_counter() - t0) * 1000)
        log.info("{\"job\":\"manual_research_enrich\",\"phase\":\"finish\",\"job_id\":%s,\"processed\":%s,\"duration_ms\":%s}", job_id, processed, dur_ms)
    except Exception as e:  # pragma: no cover
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE background_jobs SET status='error', error=%s, processed=%s, total=%s, ended_at=now() WHERE job_id=%s",
                (str(e), processed, len(ids) if 'ids' in locals() else 0, job_id),
            )
        dur_ms = int((time.perf_counter() - t0) * 1000)
        log.info("{\"job\":\"manual_research_enrich\",\"phase\":\"error\",\"job_id\":%s,\"processed\":%s,\"duration_ms\":%s,\"error\":%s}", job_id, processed, dur_ms, str(e))


def enqueue_web_discovery_bg_enrich(tenant_id: int, company_ids: list[int]) -> dict:
    """Queue background enrichment for next‑40 web_discovery preview companies.

    Stores the company_ids list in background_jobs.params for processing by the nightly dispatcher.
    """
    ids = [int(i) for i in (company_ids or []) if str(i).strip()]
    if not ids:
        return {"job_id": 0}
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO background_jobs(tenant_id, job_type, status, params) VALUES (%s,'web_discovery_bg_enrich','queued', %s) RETURNING job_id",
            (tenant_id, Json({"company_ids": ids})),
        )
        row = cur.fetchone()
        return {"job_id": int(row[0]) if row and row[0] is not None else 0}


async def run_web_discovery_bg_enrich(job_id: int) -> None:
    """Process a list of company_ids for Non‑SG next‑40 background enrichment.

    Uses the same enrich function as other flows (Tavily/Apify pipeline), scoped to provided IDs.
    """
    import time
    t0 = time.perf_counter()
    log.info("{\"job\":\"web_discovery_bg_enrich\",\"phase\":\"start\",\"job_id\":%s}", job_id)
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("UPDATE background_jobs SET status='running', started_at=now() WHERE job_id=%s", (job_id,))
        cur.execute("SELECT tenant_id, params FROM background_jobs WHERE job_id=%s", (job_id,))
        row = cur.fetchone()
        tenant_id = int(row[0]) if row and row[0] is not None else None
        params = (row and row[1]) or {}
        ids = [int(i) for i in (params.get('company_ids') or []) if str(i).strip()]
    processed = 0
    try:
        if not ids:
            raise RuntimeError("company_ids required")
        if enrich_company_with_tavily is None:
            raise RuntimeError("enrich unavailable")
        for cid in ids:
            try:
                await enrich_company_with_tavily(int(cid))
                processed += 1
            except Exception:
                # continue best-effort
                pass
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE background_jobs SET status='done', processed=%s, total=%s, ended_at=now() WHERE job_id=%s",
                (processed, len(ids), job_id),
            )
        dur_ms = int((time.perf_counter() - t0) * 1000)
        log.info("{\"job\":\"web_discovery_bg_enrich\",\"phase\":\"finish\",\"job_id\":%s,\"processed\":%s,\"duration_ms\":%s}", job_id, processed, dur_ms)
    except Exception as e:  # pragma: no cover
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE background_jobs SET status='error', error=%s, processed=%s, ended_at=now() WHERE job_id=%s",
                (str(e), processed, job_id),
            )
        dur_ms = int((time.perf_counter() - t0) * 1000)
        log.info("{\"job\":\"web_discovery_bg_enrich\",\"phase\":\"error\",\"job_id\":%s,\"processed\":%s,\"duration_ms\":%s,\"error\":%s}", job_id, processed, dur_ms, str(e))


def enqueue_enrich_candidates(tenant_id: Optional[int], ssic_codes: List[str]) -> dict:
    """Queue an enrich_candidates job keyed by SSIC codes (normalized numeric strings)."""
    params: Dict[str, Any] = {"ssic_codes": [str(c) for c in (ssic_codes or []) if str(c).strip()]}
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO background_jobs(tenant_id, job_type, status, params) VALUES (%s,'enrich_candidates','queued', %s) RETURNING job_id",
            (tenant_id, Json(params)),
        )
        row = cur.fetchone()
        return {"job_id": int(row[0]) if row and row[0] is not None else 0}


async def run_enrich_candidates(job_id: int) -> None:
    """Run enrichment for companies matching given SSIC codes in params.ssic_codes.

    Processes in chunks with a soft cap per run.
    """
    import time
    t0 = time.perf_counter()
    log.info("{\"job\":\"enrich_candidates\",\"phase\":\"start\",\"job_id\":%s}", job_id)
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("UPDATE background_jobs SET status='running', started_at=now() WHERE job_id=%s", (job_id,))
    try:
        # Load params
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT tenant_id, params FROM background_jobs WHERE job_id=%s", (job_id,))
            row = cur.fetchone() or [None, {}]
            tenant_id = row[0]
            params = row[1] or {}
        codes = [str(c).strip() for c in (params.get('ssic_codes') or []) if str(c).strip()]
        if not codes:
            raise RuntimeError("ssic_codes missing for enrich_candidates")
        # Select candidate companies by SSIC
        batch_cap = int(os.getenv("ENRICH_BATCH_SIZE", "200") or 200)
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT company_id, name, uen
                FROM companies
                WHERE regexp_replace(coalesce(industry_code::text,''), '\\D', '', 'g') = ANY(%s::text[])
                ORDER BY company_id
                LIMIT %s
                """,
                (codes, batch_cap),
            )
            rows = cur.fetchall() or []
        processed = 0
        if enrich_company_with_tavily and rows:
            import asyncio as _asyncio

            async def _run():
                async def _one(cid: int, name: str, uen: Optional[str]):
                    try:
                        await enrich_company_with_tavily(cid, name, uen)
                    except Exception:
                        pass
                await _asyncio.gather(*[_one(int(r[0]), str(r[1]), (r[2] or None)) for r in rows])

            await _run()
            processed = len(rows)
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE background_jobs SET status='done', processed=%s, total=%s, ended_at=now() WHERE job_id=%s",
                (processed, processed, job_id),
            )
        dur_ms = int((time.perf_counter() - t0) * 1000)
        log.info("{\"job\":\"enrich_candidates\",\"phase\":\"finish\",\"job_id\":%s,\"processed\":%s,\"duration_ms\":%s}", job_id, processed, dur_ms)
    except Exception as e:  # pragma: no cover
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE background_jobs SET status='error', error=%s, ended_at=now() WHERE job_id=%s",
                (str(e), job_id),
            )
        dur_ms = int((time.perf_counter() - t0) * 1000)
        log.exception("enrich_candidates job failed: %s", e)
        log.info("{\"job\":\"enrich_candidates\",\"phase\":\"error\",\"job_id\":%s,\"duration_ms\":%s,\"error\":%s}", job_id, dur_ms, str(e))
