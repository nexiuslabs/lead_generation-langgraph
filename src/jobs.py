import asyncio
import logging
import os
from typing import List, Optional, Dict, Any

from src.database import get_conn
import psycopg2
import asyncio as _asyncio
from psycopg2.extras import Json
import json
import threading

# Reuse the batched/streaming implementation from lg_entry
try:
    from app.lg_entry import _upsert_companies_from_staging_by_industries as upsert_batched
except Exception:  # pragma: no cover
    upsert_batched = None  # type: ignore

try:
    from src.enrichment import enrich_company_with_tavily, set_run_context as _enrich_set_ctx  # async
except Exception:  # pragma: no cover
    enrich_company_with_tavily = None  # type: ignore
    _enrich_set_ctx = None  # type: ignore

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
    # Load params + tenant for downstream enqueue
    tenant_id: int | None = None
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT tenant_id, params FROM background_jobs WHERE job_id=%s", (job_id,))
        r = cur.fetchone()
        if r:
            try:
                tenant_id = int(r[0]) if r[0] is not None else None
            except Exception:
                tenant_id = None
            params = r[1] or {}
        else:
            params = {}
        terms = [((t or '').strip().lower()) for t in (params.get('terms') or []) if (t or '').strip()]
    try:
        if not upsert_batched:
            raise RuntimeError("upsert function unavailable")
        # Call batched upsert once (internally streams and batches)
        processed = upsert_batched(terms)
        total = processed
        # Robust status update with retry — avoid flipping a successful job to error due
        # to a transient DB write failure at the end of processing.
        for attempt in range(3):
            try:
                with get_conn() as conn, conn.cursor() as cur:
                    cur.execute(
                        "UPDATE background_jobs SET status='done', processed=%s, total=%s, ended_at=now() WHERE job_id=%s",
                        (processed, total, job_id),
                    )
                break
            except psycopg2.Error:
                if attempt == 2:
                    # Give up on marking done; keep going without raising to avoid mislabeling the job as error
                    pass
                else:
                    await _asyncio.sleep(0.5 * (attempt + 1))
        # Best-effort: resolve SSIC codes from the same free-text terms and enqueue enrichment
        try:
            from src.icp import _find_ssic_codes_by_terms as _resolve_ssic
            resolved = _resolve_ssic(terms) if terms else []
            codes = [str(c).strip() for (c, _title, _score) in (resolved or []) if str(c).strip()]
            if codes:
                # Pass through the same tenant to scope observability and caps
                _res = enqueue_enrich_candidates(tenant_id, codes)
                try:
                    preview = ", ".join(codes[:10]) + (f", ... (+{len(codes)-10} more)" if len(codes) > 10 else "")
                    log.info(
                        "{\"job\":\"staging_upsert\",\"phase\":\"post\",\"job_id\":%s,\"tenant_id\":%s,\"enrich_job_id\":%s,\"codes_count\":%s,\"codes_preview\":\"%s\"}",
                        job_id,
                        tenant_id,
                        (_res.get("job_id") if isinstance(_res, dict) else None),
                        len(codes),
                        preview,
                    )
                except Exception:
                    pass
        except Exception:
            # Non-fatal; enrichment will be skipped if codes can't be resolved
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
                await enrich_company_with_tavily(int(cid), name, uen, search_policy="require_existing")
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


def enqueue_icp_discovery_enrich(tenant_id: int, notify_email: Optional[str] = None) -> dict:
    """Queue a unified background job that performs ICP discovery (50) and enrichment end-to-end.

    Params are stored in background_jobs.params with optional notify_email.
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO background_jobs(tenant_id, job_type, status, params) VALUES (%s,'icp_discovery_enrich','queued', %s) RETURNING job_id",
            (tenant_id, Json({**({"notify_email": notify_email} if notify_email else {})})),
        )
        row = cur.fetchone()
        jid = int(row[0]) if row and row[0] is not None else 0
        try:
            if jid:
                payload = json.dumps({"job_id": jid, "type": "icp_discovery_enrich"})
                cur.execute("NOTIFY bg_jobs, %s", (payload,))
        except Exception:
            pass
        return {"job_id": jid}


async def run_icp_discovery_enrich(job_id: int) -> None:
    """Run discovery (50 candidates) then enrichment for each; export + email on completion.

    Skeleton implementation wiring existing pipelines; discovery uses Deep Research client when configured.
    """
    import time
    from src.obs import (
        begin_run as _begin_run,
        finalize_run as _finalize_run,
        stage_timer as _stage_timer,
        log_event as _log_event,
        write_summary as _write_summary,
        persist_manifest as _persist_manifest,
    )
    from src.services.jina_deep_research import deep_research_query
    from src.lead_scoring import lead_scoring_agent
    from src.notifications.agentic_email import agentic_send_results
    from src.settings import ENABLE_JINA_DEEP_RESEARCH_DISCOVERY
    t0 = time.perf_counter()
    log.info("{\"job\":\"icp_discovery_enrich\",\"phase\":\"start\",\"job_id\":%s}", job_id)
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("UPDATE background_jobs SET status='running', started_at=now() WHERE job_id=%s", (job_id,))
        cur.execute("SELECT tenant_id, params FROM background_jobs WHERE job_id=%s", (job_id,))
        row = cur.fetchone()
        tenant_id = int(row[0]) if row and row[0] is not None else None
        params = (row and row[1]) or {}
    if not tenant_id:
        return
    notify_email = None
    try:
        notify_email = params.get("notify_email") if isinstance(params, dict) else None
    except Exception:
        notify_email = None

    run_id = _begin_run(tenant_id)
    try:
        # 1) Discovery via Deep Research (simple seed = tenant name or fallback)
        # Load a seed/company name from tenant profile or companies table as a placeholder
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT name FROM tenants WHERE tenant_id=%s", (tenant_id,))
            row = cur.fetchone()
            seed = (row and row[0]) or f"tenant_{tenant_id}"
        icp_context = {"industries": [], "buyer_titles": [], "geo": []}
        domains: List[str] = []
        with _stage_timer(run_id, tenant_id, "bg_discovery", total_inc=1):
            if ENABLE_JINA_DEEP_RESEARCH_DISCOVERY:
                tdr = time.perf_counter()
                pack = deep_research_query(str(seed), icp_context)
                domains = list(pack.get("domains") or [])
                try:
                    _log_event(
                        run_id,
                        tenant_id,
                        "bg_discovery",
                        event="dr_query",
                        status="ok",
                        duration_ms=int((time.perf_counter() - tdr) * 1000),
                        extra={"domains": len(domains)},
                    )
                except Exception:
                    pass
            # persist into staging_global_companies
            if domains:
                with get_conn() as conn, conn.cursor() as cur:
                    for d in domains[:50]:
                        try:
                            cur.execute(
                                "INSERT INTO staging_global_companies(tenant_id, domain, ai_metadata) VALUES (%s,%s,%s)",
                                (tenant_id, d, Json({"provenance": {"source": "jina_deep_research"}})),
                            )
                        except Exception:
                            continue
        # 2) Enrichment per company_id resolved by domain (best-effort)
        processed = 0
        # Create minimal company rows if not present
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT domain FROM staging_global_companies WHERE tenant_id=%s ORDER BY id ASC LIMIT 50",
                (tenant_id,),
            )
            stage_domains = [r[0] for r in (cur.fetchall() or []) if r and r[0]]
        company_ids: List[int] = []
        with get_conn() as conn, conn.cursor() as cur:
            for d in stage_domains:
                try:
                    cur.execute("SELECT company_id FROM companies WHERE website_domain=%s LIMIT 1", (d,))
                    r = cur.fetchone()
                    if r and r[0]:
                        company_ids.append(int(r[0]))
                    else:
                        cur.execute(
                            "INSERT INTO companies(name, website_domain, last_seen) VALUES (%s,%s,now()) RETURNING company_id",
                            (d.split(".")[0].title(), d),
                        )
                        rr = cur.fetchone()
                        if rr and rr[0]:
                            company_ids.append(int(rr[0]))
                except Exception:
                    continue
        # Persist manifest early for traceability
        try:
            _persist_manifest(run_id, tenant_id, company_ids)
        except Exception:
            pass
        # Run enrichment for each
        if enrich_company_with_tavily is None:
            raise RuntimeError("enrich unavailable")
        with _stage_timer(run_id, tenant_id, "bg_enrich_run", total_inc=len(company_ids)):
            for cid in company_ids:
                t1 = time.perf_counter()
                try:
                    await enrich_company_with_tavily(cid, search_policy="require_existing")
                    processed += 1
                    _log_event(run_id, tenant_id, "bg_enrich_run", event="company", status="ok", company_id=cid, duration_ms=int((time.perf_counter()-t1)*1000))
                except Exception as e:
                    _log_event(run_id, tenant_id, "bg_enrich_run", event="company", status="error", company_id=cid, error_code=type(e).__name__)
                    continue
        # Scoring for all enriched companies (best-effort)
        try:
            if company_ids:
                tsc = time.perf_counter()
                scoring_state = {
                    "candidate_ids": company_ids,
                    "lead_features": [],
                    "lead_scores": [],
                    "icp_payload": {},
                }
                await lead_scoring_agent.ainvoke(scoring_state)
                try:
                    _log_event(
                        run_id,
                        tenant_id,
                        "scoring",
                        event="score_batch",
                        status="ok",
                        duration_ms=int((time.perf_counter() - tsc) * 1000),
                        extra={"companies": len(company_ids)},
                    )
                except Exception:
                    pass
        except Exception:
            pass
        # Export to Odoo (best-effort)
        try:
            rows: list[tuple] = []
            with get_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT company_id, name, uen FROM companies WHERE company_id = ANY(%s)",
                    (company_ids,),
                )
                rows = cur.fetchall() or []
            if rows:
                texp = time.perf_counter()
                await _odoo_export_for_ids(tenant_id, rows)
                try:
                    _log_event(
                        run_id,
                        tenant_id,
                        "odoo_export",
                        event="export_batch",
                        status="ok",
                        duration_ms=int((time.perf_counter() - texp) * 1000),
                        extra={"companies": len(rows)},
                    )
                except Exception:
                    pass
        except Exception:
            pass
        # Email notification (best-effort)
        try:
            if notify_email:
                t2 = time.perf_counter()
                res = await agentic_send_results(notify_email, tenant_id, limit=500)
                _log_event(run_id, tenant_id, "email_notify", event="send", status=str(res.get("status") or "ok"), duration_ms=int((time.perf_counter()-t2)*1000), extra={"to": notify_email})
        except Exception:
            pass
        # Run summary
        try:
            _write_summary(run_id, tenant_id, candidates=len(company_ids), processed=processed, batches=1)
        except Exception:
            pass
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE background_jobs SET status='done', processed=%s, total=%s, ended_at=now() WHERE job_id=%s",
                (processed, len(company_ids), job_id),
            )
        dur_ms = int((time.perf_counter() - t0) * 1000)
        log.info("{\"job\":\"icp_discovery_enrich\",\"phase\":\"finish\",\"job_id\":%s,\"processed\":%s,\"duration_ms\":%s}", job_id, processed, dur_ms)
    except Exception as e:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE background_jobs SET status='error', error=%s, ended_at=now() WHERE job_id=%s",
                (str(e), job_id),
            )
        log.exception("icp_discovery_enrich failed: %s", e)
    finally:
        _finalize_run(run_id, status="succeeded")


def enqueue_web_discovery_bg_enrich(tenant_id: int, company_ids: list[int], notify_email: Optional[str] = None) -> dict:
    """Queue background enrichment for next‑40 web_discovery preview companies.

    Stores the company_ids list in background_jobs.params for processing by the nightly dispatcher.
    """
    ids = [int(i) for i in (company_ids or []) if str(i).strip()]
    if not ids:
        return {"job_id": 0}
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO background_jobs(tenant_id, job_type, status, params) VALUES (%s,'web_discovery_bg_enrich','queued', %s) RETURNING job_id",
            (tenant_id, Json({"company_ids": ids, **({"notify_email": notify_email} if notify_email else {})})),
        )
        row = cur.fetchone()
        jid = int(row[0]) if row and row[0] is not None else 0
        # Notify listeners (bg worker) so they can pick it up immediately
        try:
            if jid:
                payload = json.dumps({"job_id": jid, "type": "web_discovery_bg_enrich"})
                cur.execute("NOTIFY bg_jobs, %s", (payload,))
        except Exception:
            # Best-effort: notification is optional
            pass
        return {"job_id": jid}


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
        # Resolve company names/uen for enrichment call signature
        comp_map: dict[int, tuple[str, str | None]] = {}
        try:
            with get_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT company_id, name, uen FROM companies WHERE company_id = ANY(%s)",
                    (ids,),
                )
                rows = cur.fetchall() or []
                for r in rows:
                    try:
                        cid = int(r[0]) if r and r[0] is not None else None
                        nm = (r[1] or "").strip() if len(r) > 1 else ""
                        uen = (r[2] or None) if len(r) > 2 else None
                        if cid and nm:
                            comp_map[cid] = (nm, uen)
                    except Exception:
                        continue
        except Exception:
            comp_map = {}
        scored_ids: list[int] = []
        for cid in ids:
            try:
                name, uen = comp_map.get(int(cid), (str(cid), None))
                # Pre-step log per company
                try:
                    log.info(
                        "{\"job\":\"web_discovery_bg_enrich\",\"phase\":\"company\",\"job_id\":%s,\"tenant_id\":%s,\"company_id\":%s,\"name\":\"%s\"}",
                        job_id,
                        tenant_id,
                        int(cid),
                        (name or "").replace("\"", "'")[:200],
                    )
                except Exception:
                    pass
                final_state = await enrich_company_with_tavily(
                    int(cid), name, uen, search_policy="require_existing"
                )
                # Post-step detailed summary per company
                try:
                    completed = bool(final_state.get("completed")) if isinstance(final_state, dict) else None
                    error = final_state.get("error") if isinstance(final_state, dict) else None
                    domains = final_state.get("domains") if isinstance(final_state, dict) else None
                    pages = final_state.get("extracted_pages") if isinstance(final_state, dict) else None
                    chunks = final_state.get("chunks") if isinstance(final_state, dict) else None
                    data = final_state.get("data") if isinstance(final_state, dict) else None
                    degraded = final_state.get("degraded_reasons") if isinstance(final_state, dict) else None
                    log.info(
                        "{\"job\":\"web_discovery_bg_enrich\",\"phase\":\"result\",\"job_id\":%s,\"tenant_id\":%s,\"company_id\":%s,\"completed\":%s,\"error\":%s,\"domains\":%s,\"pages\":%s,\"chunks\":%s,\"emails\":%s,\"degraded\":%s}",
                        job_id,
                        tenant_id,
                        int(cid),
                        completed,
                        (json.dumps(error) if error is not None else "null"),
                        (len(domains) if isinstance(domains, list) else 0),
                        (len(pages) if isinstance(pages, list) else 0),
                        (len(chunks) if isinstance(chunks, list) else 0),
                        (len((data or {}).get("email", [])) if isinstance(data, dict) else 0),
                        (json.dumps(degraded) if degraded is not None else "null"),
                    )
                except Exception:
                    pass
                processed += 1
                scored_ids.append(int(cid))
            except Exception:
                # continue best-effort
                pass
        # Robust status update with retry — avoid failing the job after work completes due to transient DB errors
        for attempt in range(3):
            try:
                with get_conn() as conn, conn.cursor() as cur:
                    cur.execute(
                        "UPDATE background_jobs SET status='done', processed=%s, total=%s, ended_at=now() WHERE job_id=%s",
                        (processed, len(ids), job_id),
                    )
                break
            except psycopg2.Error:
                if attempt == 2:
                    raise
                await _asyncio.sleep(0.5 * (attempt + 1))
        # Run lead scoring for enriched companies best-effort
        try:
            if scored_ids:
                from src.lead_scoring import lead_scoring_agent
                scoring_state = {
                    "candidate_ids": scored_ids,
                    "lead_features": [],
                    "lead_scores": [],
                    "icp_payload": {},
                }
                await lead_scoring_agent.ainvoke(scoring_state)
        except Exception as e:
            try:
                log.warning("{\"job\":\"web_discovery_bg_enrich\",\"phase\":\"scoring_error\",\"job_id\":%s,\"error\":\"%s\"}", job_id, str(e)[:200])
            except Exception:
                pass
        dur_ms = int((time.perf_counter() - t0) * 1000)
        log.info("{\"job\":\"web_discovery_bg_enrich\",\"phase\":\"finish\",\"job_id\":%s,\"processed\":%s,\"duration_ms\":%s}", job_id, processed, dur_ms)
    except Exception as e:  # pragma: no cover
        # Best-effort error write with retry as well
        for attempt in range(3):
            try:
                with get_conn() as conn, conn.cursor() as cur:
                    cur.execute(
                        "UPDATE background_jobs SET status='error', error=%s, processed=%s, ended_at=now() WHERE job_id=%s",
                        (str(e), processed, job_id),
                    )
                break
            except psycopg2.Error:
                if attempt == 2:
                    break
                await _asyncio.sleep(0.5 * (attempt + 1))
        dur_ms = int((time.perf_counter() - t0) * 1000)
        log.info("{\"job\":\"web_discovery_bg_enrich\",\"phase\":\"error\",\"job_id\":%s,\"processed\":%s,\"duration_ms\":%s,\"error\":%s}", job_id, processed, dur_ms, str(e))
    # After completion (success or failure on some items), send email once if requested and not already sent
    try:
        to_email = None
        sent_at = None
        try:
            to_email = (params or {}).get("notify_email")
            sent_at = (params or {}).get("email_sent_at")
        except Exception:
            to_email = None
        # Resolve from tenant_users when missing
        if tenant_id and not to_email:
            try:
                with get_conn() as conn, conn.cursor() as cur:
                    cur.execute(
                        "SELECT user_id FROM tenant_users WHERE tenant_id=%s ORDER BY user_id LIMIT 1",
                        (int(tenant_id),),
                    )
                    row = cur.fetchone()
                    if row and row[0]:
                        candidate = str(row[0])
                        # Only accept tenant_users.user_id as email when it contains '@' (dev-only guard)
                        try:
                            from src.settings import EMAIL_DEV_ACCEPT_TENANT_USER_ID_AS_EMAIL as _ACCEPT_TU_EMAIL
                        except Exception:
                            _ACCEPT_TU_EMAIL = True
                        if _ACCEPT_TU_EMAIL and ("@" in candidate):
                            to_email = candidate
                            # persist source
                            cur.execute(
                                "UPDATE background_jobs SET params = COALESCE(params,'{}'::jsonb) || jsonb_build_object('notify_email', %s, 'notify_email_source', 'tenant_users') WHERE job_id=%s",
                                (to_email, job_id),
                            )
            except Exception:
                pass
        # Fallback to DEFAULT_NOTIFY_EMAIL when still missing (useful in dev-bypass)
        if tenant_id and not to_email:
            try:
                from src.settings import DEFAULT_NOTIFY_EMAIL as _DEF_TO
                if _DEF_TO and ("@" in str(_DEF_TO)):
                    to_email = str(_DEF_TO)
                    # persist fallback into params to avoid repeated lookups and document provenance
                    with get_conn() as conn, conn.cursor() as cur:
                        cur.execute(
                            "UPDATE background_jobs SET params = COALESCE(params,'{}'::jsonb) || jsonb_build_object('notify_email', %s, 'notify_email_source', 'default_env') WHERE job_id=%s",
                            (to_email, job_id),
                        )
            except Exception:
                pass
        if tenant_id and to_email and not sent_at:
            from src.notifications.agentic_email import agentic_send_results
            res = await agentic_send_results(str(to_email), int(tenant_id))
            # Log outcome for observability
            try:
                log.info(
                    "{\"job\":\"web_discovery_bg_enrich\",\"phase\":\"email\",\"job_id\":%s,\"tenant_id\":%s,\"to\":\"%s\",\"status\":%s}",
                    job_id,
                    tenant_id,
                    (to_email or "").replace("\"", "'")[:200],
                    (res or {}).get("status"),
                )
            except Exception:
                pass
            # Mark as sent if success to prevent duplicates
            if (res or {}).get("status") == "sent":
                with get_conn() as conn, conn.cursor() as cur:
                    cur.execute(
                        "UPDATE background_jobs SET params = COALESCE(params,'{}'::jsonb) || jsonb_build_object('email_sent_at', now(), 'email_to', %s) WHERE job_id=%s",
                        (to_email, job_id),
                    )
        else:
            try:
                reason = "already_sent" if sent_at else "missing_to"
                log.info(
                    "{\"job\":\"web_discovery_bg_enrich\",\"phase\":\"email_skip\",\"job_id\":%s,\"tenant_id\":%s,\"reason\":\"%s\"}",
                    job_id,
                    tenant_id,
                    reason,
                )
            except Exception:
                pass
    except Exception:
        # Non-fatal
        pass


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
        try:
            log.info(
                "{\"job\":\"enrich_candidates\",\"phase\":\"tenant\",\"job_id\":%s,\"tenant_id\":%s,\"codes_count\":%s,\"codes_preview\":\"%s\"}",
                job_id,
                tenant_id,
                len(codes),
                ", ".join(codes[:10]) + (f", ... (+{len(codes)-10} more)" if len(codes) > 10 else ""),
            )
        except Exception:
            pass
        # Begin an observability run and set enrichment context for proper tenant scoping
        run_id = None
        try:
            from src import obs as _obs
            if tenant_id is not None:
                run_id = _obs.begin_run(int(tenant_id))
                try:
                    _obs.set_run_context(int(run_id), int(tenant_id))
                except Exception:
                    pass
                # Propagate context to enrichment module so company_enrichment_runs rows link to this run_id
                try:
                    if _enrich_set_ctx and run_id is not None:
                        _enrich_set_ctx(int(run_id), int(tenant_id))
                except Exception:
                    pass
        except Exception:
            run_id = None

        # Enforce per-tenant daily cap for ACRA nightly enrichment
        try:
            DAILY_CAP = int(os.getenv("ACRA_DAILY_ENRICH_LIMIT", "20") or 20)
        except Exception:
            DAILY_CAP = 20
        remaining_today = DAILY_CAP
        try:
            if tenant_id is not None:
                with get_conn() as conn, conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT COALESCE(COUNT(*),0) AS cnt
                        FROM company_enrichment_runs cer
                        JOIN enrichment_runs er ON er.run_id = cer.run_id
                        WHERE er.tenant_id = %s
                          AND er.started_at >= date_trunc('day', now())
                        """,
                        (int(tenant_id),),
                    )
                    r = cur.fetchone()
                    already = int(r[0] or 0) if r else 0
                    remaining_today = max(0, DAILY_CAP - already)
        except Exception:
            # If we cannot compute, fall back to the cap to avoid over-processing
            remaining_today = DAILY_CAP

        # Log capacity snapshot per tenant
        try:
            log.info(
                "{\"job\":\"enrich_candidates\",\"phase\":\"capacity\",\"job_id\":%s,\"tenant_id\":%s,\"cap\":%s,\"remaining_today\":%s}",
                job_id, tenant_id, DAILY_CAP, remaining_today,
            )
        except Exception:
            pass

        if remaining_today <= 0:
            # Defer this job to the next nightly window
            with get_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    "UPDATE background_jobs SET status='queued', error=%s WHERE job_id=%s",
                    ("deferred: daily cap reached", job_id),
                )
            try:
                log.info(
                    "{\"job\":\"enrich_candidates\",\"phase\":\"deferred\",\"job_id\":%s,\"tenant_id\":%s,\"reason\":\"daily_cap_reached\",\"cap\":%s}",
                    job_id,
                    tenant_id,
                    DAILY_CAP,
                )
            except Exception:
                pass
            # Finalize run header if we started one
            try:
                if run_id is not None:
                    from src import obs as _obs
                    _obs.finalize_run(int(run_id), status="succeeded")
            except Exception:
                pass
            return

        # Select candidate companies by SSIC, limited by daily remaining and batch cap
        batch_cap = int(os.getenv("ENRICH_BATCH_SIZE", "200") or 200)
        effective_limit = min(batch_cap, max(0, int(remaining_today)))
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT company_id, name, uen
                FROM companies
                WHERE regexp_replace(coalesce(industry_code::text,''), '\\D', '', 'g') = ANY(%s::text[])
                ORDER BY company_id
                LIMIT %s
                """,
                (codes, effective_limit),
            )
            rows = cur.fetchall() or []
        try:
            log.info(
                "{\"job\":\"enrich_candidates\",\"phase\":\"select\",\"job_id\":%s,\"tenant_id\":%s,\"selected\":%s,\"effective_limit\":%s}",
                job_id,
                tenant_id,
                len(rows),
                effective_limit,
            )
        except Exception:
            pass
        processed = 0
        scored_ids: list[int] = []
        if enrich_company_with_tavily and rows:
            import asyncio as _asyncio

            async def _run():
                async def _one(cid: int, name: str, uen: Optional[str]):
                    try:
                        await enrich_company_with_tavily(cid, name, uen, search_policy="discover")
                    except Exception:
                        pass
                await _asyncio.gather(*[_one(int(r[0]), str(r[1]), (r[2] or None)) for r in rows])

            await _run()
            processed = len(rows)
            scored_ids = [int(r[0]) for r in rows if r and r[0] is not None]
            # Best-effort Odoo export for enriched companies
            try:
                await _odoo_export_for_ids(tenant_id, rows)
            except Exception:
                pass
            # Run lead scoring on the enriched batch so nightly runs update lead_scores immediately
            try:
                if scored_ids:
                    from src.lead_scoring import lead_scoring_agent
                    scoring_state = {
                        "candidate_ids": scored_ids,
                        "lead_features": [],
                        "lead_scores": [],
                        "icp_payload": {},
                    }
                    await lead_scoring_agent.ainvoke(scoring_state)
            except Exception as e:
                try:
                    log.warning("{\"job\":\"enrich_candidates\",\"phase\":\"scoring_error\",\"job_id\":%s,\"error\":\"%s\"}", job_id, str(e)[:200])
                except Exception:
                    pass
        # If we fully utilized today's quota, keep the job queued for the next night; else mark done
        job_completed = True
        if processed >= effective_limit and effective_limit > 0:
            # Re-queue for next window
            with get_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    "UPDATE background_jobs SET status='queued', processed=COALESCE(processed,0)+%s, total=COALESCE(total,0)+%s, ended_at=now() WHERE job_id=%s",
                    (processed, processed, job_id),
                )
            try:
                log.info(
                    "{\"job\":\"enrich_candidates\",\"phase\":\"requeue\",\"job_id\":%s,\"tenant_id\":%s,\"processed\":%s,\"effective_limit\":%s}",
                    job_id,
                    tenant_id,
                    processed,
                    effective_limit,
                )
            except Exception:
                pass
            job_completed = False
        else:
            with get_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    "UPDATE background_jobs SET status='done', processed=%s, total=%s, ended_at=now() WHERE job_id=%s",
                    (processed, processed, job_id),
                )
        # Finalize the run header
        try:
            if run_id is not None:
                from src import obs as _obs
                _obs.finalize_run(int(run_id), status="succeeded")
        except Exception:
            pass
        dur_ms = int((time.perf_counter() - t0) * 1000)
        log.info(
            "{\"job\":\"enrich_candidates\",\"phase\":\"finish\",\"job_id\":%s,\"tenant_id\":%s,\"processed\":%s,\"duration_ms\":%s}",
            job_id,
            tenant_id,
            processed,
            dur_ms,
        )
        # If the nightly batch fully completed, send the shortlist email once scoring is done
        if job_completed and tenant_id is not None and processed > 0:
            try:
                current_params = params
                try:
                    with get_conn() as conn, conn.cursor() as cur:
                        cur.execute("SELECT params FROM background_jobs WHERE job_id=%s", (job_id,))
                        row = cur.fetchone()
                        if row and row[0]:
                            current_params = row[0] or {}
                except Exception:
                    pass
                to_email = (current_params or {}).get("notify_email") if isinstance(current_params, dict) else None
                sent_at = (current_params or {}).get("email_sent_at") if isinstance(current_params, dict) else None
                # Resolve tenant_users fallback
                if not to_email:
                    try:
                        with get_conn() as conn, conn.cursor() as cur:
                            cur.execute(
                                "SELECT user_id FROM tenant_users WHERE tenant_id=%s ORDER BY user_id LIMIT 1",
                                (int(tenant_id),),
                            )
                            row = cur.fetchone()
                            if row and row[0]:
                                candidate = str(row[0])
                                try:
                                    from src.settings import EMAIL_DEV_ACCEPT_TENANT_USER_ID_AS_EMAIL as _ACCEPT_TU_EMAIL
                                except Exception:
                                    _ACCEPT_TU_EMAIL = True
                                if _ACCEPT_TU_EMAIL and ("@" in candidate):
                                    to_email = candidate
                                    with get_conn() as conn, conn.cursor() as cur2:
                                        cur2.execute(
                                            "UPDATE background_jobs SET params = COALESCE(params,'{}'::jsonb) || jsonb_build_object('notify_email', %s, 'notify_email_source', 'tenant_users') WHERE job_id=%s",
                                            (to_email, job_id),
                                        )
                    except Exception:
                        pass
                if not to_email:
                    try:
                        from src.settings import DEFAULT_NOTIFY_EMAIL as _DEF_TO
                        if _DEF_TO and ("@" in str(_DEF_TO)):
                            to_email = str(_DEF_TO)
                            with get_conn() as conn, conn.cursor() as cur:
                                cur.execute(
                                    "UPDATE background_jobs SET params = COALESCE(params,'{}'::jsonb) || jsonb_build_object('notify_email', %s, 'notify_email_source', 'default_env') WHERE job_id=%s",
                                    (to_email, job_id),
                                )
                    except Exception:
                        pass
                if to_email and not sent_at:
                    from src.notifications.agentic_email import agentic_send_results
                    res = await agentic_send_results(str(to_email), int(tenant_id))
                    try:
                        log.info(
                            "{\"job\":\"enrich_candidates\",\"phase\":\"email\",\"job_id\":%s,\"tenant_id\":%s,\"to\":\"%s\",\"status\":%s}",
                            job_id,
                            tenant_id,
                            (to_email or "").replace("\"", "'")[:200],
                            (res or {}).get("status"),
                        )
                    except Exception:
                        pass
                    if (res or {}).get("status") == "sent":
                        with get_conn() as conn, conn.cursor() as cur:
                            cur.execute(
                                "UPDATE background_jobs SET params = COALESCE(params,'{}'::jsonb) || jsonb_build_object('email_sent_at', now(), 'email_to', %s) WHERE job_id=%s",
                                (to_email, job_id),
                            )
                else:
                    try:
                        reason = "already_sent" if sent_at else "missing_to"
                        log.info(
                            "{\"job\":\"enrich_candidates\",\"phase\":\"email_skip\",\"job_id\":%s,\"tenant_id\":%s,\"reason\":\"%s\"}",
                            job_id,
                            tenant_id,
                            reason,
                        )
                    except Exception:
                        pass
            except Exception:
                pass
    except Exception as e:  # pragma: no cover
        try:
            with get_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    "UPDATE background_jobs SET status='error', error=%s, ended_at=now() WHERE job_id=%s",
                    (str(e), job_id),
                )
        except Exception:
            pass
        dur_ms = int((time.perf_counter() - t0) * 1000)
        log.exception("enrich_candidates job failed: %s", e)
        log.info("{\"job\":\"enrich_candidates\",\"phase\":\"error\",\"job_id\":%s,\"duration_ms\":%s,\"error\":%s}", job_id, dur_ms, str(e))


async def _odoo_export_for_ids(tenant_id: Optional[int], company_rows: list[tuple]) -> None:
    """Best-effort Odoo sync for a batch of companies after nightly enrichment.

    For each (company_id, name, uen) in company_rows, upsert to Odoo partner,
    add primary contact if present, and create a lead with the latest score.
    Non-fatal: all exceptions are swallowed per item; logs in OdooStore cover details.
    """
    if not company_rows:
        return
    try:
        from app.odoo_store import OdooStore  # async methods
    except Exception:
        return
    try:
        store = OdooStore(tenant_id=int(tenant_id) if tenant_id is not None else None)
    except Exception:
        return
    # Fetch company core fields, scores and a primary email
    ids = [int(r[0]) for r in company_rows if r and r[0] is not None]
    if not ids:
        return
    comps: dict[int, dict] = {}
    emails: dict[int, str | None] = {}
    scores: dict[int, float] = {}
    rationales: dict[int, str] = {}
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT company_id, name, uen, industry_norm, employees_est, revenue_bucket,
                       incorporation_year, website_domain
                  FROM companies
                 WHERE company_id = ANY(%s)
                """,
                (ids,),
            )
            for row in cur.fetchall() or []:
                try:
                    comps[int(row[0])] = {
                        "name": row[1],
                        "uen": row[2],
                        "industry_norm": row[3],
                        "employees_est": row[4],
                        "revenue_bucket": row[5],
                        "incorporation_year": row[6],
                        "website_domain": row[7],
                    }
                except Exception:
                    continue
            cur.execute(
                "SELECT company_id, email FROM lead_emails WHERE company_id = ANY(%s)",
                (ids,),
            )
            for r in cur.fetchall() or []:
                try:
                    emails[int(r[0])] = r[1]
                except Exception:
                    continue
            cur.execute(
                "SELECT company_id, score, rationale FROM lead_scores WHERE company_id = ANY(%s)",
                (ids,),
            )
            for r in cur.fetchall() or []:
                try:
                    scores[int(r[0])] = float(r[1] or 0.0)
                    rationales[int(r[0])] = r[2] or ""
                except Exception:
                    continue
    except Exception:
        # proceed with what we have
        pass
    # Export sequentially to avoid hammering Odoo
    for cid in ids:
        comp = comps.get(cid) or {}
        try:
            partner_id = await store.upsert_company(
                comp.get("name"),
                comp.get("uen"),
                industry_norm=comp.get("industry_norm"),
                employees_est=comp.get("employees_est"),
                revenue_bucket=comp.get("revenue_bucket"),
                incorporation_year=comp.get("incorporation_year"),
                website_domain=comp.get("website_domain"),
            )
            email = emails.get(cid)
            if email:
                try:
                    await store.add_contact(partner_id, email)
                except Exception:
                    pass
            score = scores.get(cid, 0.0)
            rationale = rationales.get(cid, "")
            try:
                await store.create_lead_if_high(partner_id, comp.get("name"), score, {}, rationale, email)
            except Exception:
                pass
        except Exception:
            # continue best-effort for the rest
            continue
