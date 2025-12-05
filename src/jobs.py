import asyncio
import logging
import os
from typing import List, Optional, Dict, Any, Tuple
from urllib.parse import urlparse
import hashlib
import zlib

from src.database import get_conn
import psycopg2
import asyncio as _asyncio
from psycopg2.extras import Json
import json
import threading
from src.settings import ICP_RULE_NAME
from src.troubleshoot_log import log_json
from urllib.parse import urlparse as _urlparse

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
_DEFAULT_DISCOVERY_COUNTRY = (os.getenv("MCP_SEARCH_COUNTRY") or os.getenv("DEFAULT_DISCOVERY_COUNTRY") or "").strip() or None


def request_cancel(job_id: int) -> dict:
    """Mark a queued/running job as cancel_requested.

    Uses both a dedicated boolean column (when available via migrations) and params JSONB for compatibility.
    """
    with get_conn() as conn, conn.cursor() as cur:
        # Write params flag unconditionally; set cancel_requested column when it exists
        try:
            cur.execute(
                """
                UPDATE background_jobs
                   SET params = COALESCE(params,'{}'::jsonb) || jsonb_build_object('cancel_requested', true),
                       cancel_requested = TRUE
                 WHERE job_id=%s AND status IN ('queued','running')
             RETURNING job_id, status
                """,
                (int(job_id),),
            )
        except Exception:
            cur.execute(
                """
                UPDATE background_jobs
                   SET params = COALESCE(params,'{}'::jsonb) || jsonb_build_object('cancel_requested', true)
                 WHERE job_id=%s AND status IN ('queued','running')
             RETURNING job_id, status
                """,
                (int(job_id),),
            )
        row = cur.fetchone()
        if not row:
            return {"ok": False, "error": "not_cancellable"}
        try:
            log.info('{"job":"cancel_request","job_id":%s,"status":"%s"}', int(row[0]), str(row[1]))
        except Exception:
            pass
        return {"ok": True, "job_id": int(row[0]), "status": str(row[1])}


def request_cancel_current(tenant_id: int) -> dict:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT job_id
              FROM background_jobs
             WHERE tenant_id=%s AND job_type='icp_discovery_enrich' AND status IN ('queued','running')
             ORDER BY job_id DESC
             LIMIT 1
            """,
            (int(tenant_id),),
        )
        row = cur.fetchone()
        if not row or row[0] is None:
            return {"ok": False, "error": "no_active_job"}
        jid = int(row[0])
    return request_cancel(jid)


def _should_cancel(job_id: int) -> bool:
    """Return True when cancel has been requested for the job."""
    try:
        with get_conn() as conn, conn.cursor() as cur:
            try:
                cur.execute(
                    "SELECT cancel_requested, params FROM background_jobs WHERE job_id=%s",
                    (int(job_id),),
                )
            except Exception:
                cur.execute(
                    "SELECT NULL::boolean AS cancel_requested, params FROM background_jobs WHERE job_id=%s",
                    (int(job_id),),
                )
            row = cur.fetchone()
            if not row:
                return False
            col = bool(row[0]) if row[0] is not None else False
            params = row[1] or {}
            flag = False
            try:
                flag = bool(params.get("cancel_requested")) if isinstance(params, dict) else False
            except Exception:
                flag = False
            return bool(col or flag)
    except Exception:
        return False


def _dsn_hint() -> dict:
    """Return a sanitized DSN hint for logs (host + db name only)."""
    try:
        from src.settings import POSTGRES_DSN as _DSN  # type: ignore
    except Exception:
        _DSN = os.getenv("POSTGRES_DSN", "")
    if not _DSN:
        return {"host": None, "database": None}
    try:
        pr = _urlparse(_DSN)
        host = pr.hostname
        db = (pr.path or "/").lstrip("/")
        return {"host": host, "database": db}
    except Exception:
        return {"host": None, "database": None}


def _normalize_domain(value: str) -> Optional[str]:
    try:
        s = (value or "").strip()
        if not s:
            return None
        if not s.startswith("http://") and not s.startswith("https://"):
            s = "https://" + s
        parsed = urlparse(s)
        host = (parsed.netloc or parsed.path or "").strip().lower()
        if not host:
            return None
        if host.startswith("www."):
            host = host[4:]
        if ":" in host:
            host = host.split(":", 1)[0]
        if not host or "." not in host:
            return None
        return host
    except Exception:
        return None


def _build_discovery_queries(seed: str, country_hint: Optional[str]) -> List[str]:
    site_filter = ""
    try:
        if country_hint and country_hint.lower() in {"sg", "singapore"}:
            site_filter = "site:.sg"
        elif country_hint and country_hint.strip():
            site_filter = f"site:.{country_hint.strip().lower()}"
    except Exception:
        site_filter = ""

    base = (seed or "").strip()
    queries: List[str] = []

    def _push(q: str) -> None:
        qq = q.strip()
        if not qq:
            return
        if site_filter and "site:" not in qq:
            qq = f"{qq} {site_filter}".strip()
        if qq not in queries:
            queries.append(qq)

    if base:
        _push(f"{base} competitors")
        _push(f"{base} alternatives")
        _push(f"companies like {base}")
    _push("b2b distributors")
    _push("enterprise software vendors")
    return queries


def _fallback_discovery_domains(
    seed: str,
    *,
    limit: int,
    country_hint: Optional[str] = None,
) -> Tuple[List[str], str]:
    queries = _build_discovery_queries(seed, country_hint)
    if not queries:
        queries = ["b2b distributors"]
    seen: set[str] = set()
    domains: List[str] = []
    # 1) Try MCP search tools first
    try:
        from src.services import mcp_reader as _mcp  # type: ignore

        for q in queries:
            try:
                urls = _mcp.search_web(q, country=country_hint, max_results=limit) or []
            except Exception as exc:  # pragma: no cover
                try:
                    log.info("{\"stage\":\"bg_discovery\",\"event\":\"mcp_fallback_error\",\"query\":\"%s\",\"error\":\"%s\"}", q, str(exc)[:180])
                except Exception:
                    pass
                continue
            for u in urls:
                host = _normalize_domain(u)
                if not host or host in seen:
                    continue
                seen.add(host)
                domains.append(host)
                if len(domains) >= limit:
                    return (domains[:limit], "jina_mcp_search")
        if domains:
            return (domains[:limit], "jina_mcp_search")
    except Exception as exc:  # pragma: no cover
        try:
            log.info("{\"stage\":\"bg_discovery\",\"event\":\"mcp_import_error\",\"error\":\"%s\"}", str(exc)[:180])
        except Exception:
            pass

    # 2) Fallback to DDG HTML via r.jina helper
    source = "ddg_search"
    try:
        from src.ddg_simple import search_domains as _ddg_search

        for q in queries:
            try:
                hosts = _ddg_search(q, max_results=limit, country=country_hint)
            except Exception as exc:  # pragma: no cover
                try:
                    log.info("{\"stage\":\"bg_discovery\",\"event\":\"ddg_error\",\"query\":\"%s\",\"error\":\"%s\"}", q, str(exc)[:180])
                except Exception:
                    pass
                continue
            for h in hosts:
                host = _normalize_domain(h)
                if not host or host in seen:
                    continue
                seen.add(host)
                domains.append(host)
                if len(domains) >= limit:
                    return (domains[:limit], source)
    except Exception as exc:  # pragma: no cover
        try:
            log.warning("{\"stage\":\"bg_discovery\",\"event\":\"fallback_failed\",\"error\":\"%s\"}", str(exc)[:180])
        except Exception:
            pass
    return (domains[:limit], source)


def _canonical_icp_profile(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Project ICP payload to a stable subset for hashing.

    We keep only fields that influence discovery/targeting and sort list values implicitly via JSON dumps.
    """
    keys = (
        "industries",
        "buyer_titles",
        "size_bands",
        "company_sizes",
        "geos",
        "signals",
        "integrations",
        "triggers",
        "summary",
        "seed_urls",
    )
    out: Dict[str, Any] = {}
    try:
        for k in keys:
            v = payload.get(k)
            if v is None:
                continue
            out[k] = v
    except Exception:
        return {}
    return out


def _icp_fingerprint(payload: Dict[str, Any], rule_name: Optional[str]) -> str:
    try:
        subset = _canonical_icp_profile(payload or {})
        blob = json.dumps(subset, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        rn = (rule_name or payload.get("rule_name") or "Default ICP").strip()
        text = f"{blob}|{rn}"
        return hashlib.sha1(text.encode("utf-8")).hexdigest()
    except Exception:
        return ""


def _fp_to_key(fp: str) -> int:
    """Map a hex fingerprint to a 32-bit int for advisory locks."""
    if not fp:
        return 0
    try:
        # Fast 32bit key
        return zlib.crc32(fp.encode("utf-8")) & 0xFFFFFFFF
    except Exception:
        return 0


def _load_icp_profile(tenant_id: int) -> Dict[str, Any]:
    try:
        with get_conn() as conn, conn.cursor() as cur:
            for query, params in (
                (
                    """
                    SELECT payload
                    FROM icp_rules
                    WHERE tenant_id=%s AND name=%s
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (int(tenant_id), ICP_RULE_NAME),
                ),
                (
                    """
                    SELECT payload
                    FROM icp_rules
                    WHERE tenant_id=%s
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (int(tenant_id),),
                ),
            ):
                cur.execute(query, params)
                row = cur.fetchone()
                if not row:
                    continue
                val = row[0]
                if isinstance(val, dict):
                    return val
                if isinstance(val, str):
                    try:
                        import json as _json

                        data = _json.loads(val)
                        if isinstance(data, dict):
                            return data
                    except Exception:
                        continue
    except Exception:
        return {}
    return {}


def _clean_str_list(value: Any, limit: int = 5) -> List[str]:
    items: List[str] = []
    try:
        if isinstance(value, list):
            for v in value:
                if isinstance(v, str) and v.strip():
                    items.append(v.strip())
        elif isinstance(value, str) and value.strip():
            items.append(value.strip())
    except Exception:
        return []
    if limit:
        return items[:limit]
    return items


def _icp_context_from_profile(profile: Dict[str, Any]) -> Dict[str, List[str]]:
    context = {
        "industries": [],
        "buyer_titles": [],
        "geo": [],
    }
    if not profile:
        return context
    context["industries"] = _clean_str_list(
        profile.get("industries")
        or profile.get("industry")
        or profile.get("icp_industries")
        or [],
    )
    context["buyer_titles"] = _clean_str_list(
        profile.get("buyer_titles")
        or profile.get("titles")
        or profile.get("preferred_titles")
        or [],
    )
    context["geo"] = _clean_str_list(
        profile.get("geo")
        or profile.get("geos")
        or profile.get("regions")
        or [],
    )
    return context


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

    Idempotency: compute an ICP fingerprint from the latest `icp_rules` payload combined with rule name.
    If an identical fingerprint is already queued/running for the same tenant, return the existing job_id.
    """
    icp_payload = _load_icp_profile(int(tenant_id)) or {}
    fp = _icp_fingerprint(icp_payload, ICP_RULE_NAME)
    fp_key = _fp_to_key(fp)
    params = {"notify_email": notify_email} if notify_email else {}
    if fp:
        params["fp"] = fp
    # Emit a structured "try" log to aid diagnosis
    try:
        log_json(
            "jobs",
            "info",
            "enqueue_try",
            {
                "tenant_id": int(tenant_id),
                "job_type": "icp_discovery_enrich",
                "fp": fp or None,
                "dsn_hint": _dsn_hint(),
            },
        )
    except Exception:
        pass
    try:
        with get_conn() as conn, conn.cursor() as cur:
            # Acquire a short-lived advisory lock on (tenant_id, fp_key) to prevent races
            # If this fails on the managed DB, log + rollback and continue without the lock.
            try:
                if fp_key:
                    cur.execute("SELECT pg_try_advisory_xact_lock(%s, %s)", (int(tenant_id), int(fp_key)))
            except Exception as aex:
                try:
                    conn.rollback()
                except Exception:
                    pass
                try:
                    log_json(
                        "jobs",
                        "warning",
                        "advisory_lock_failed",
                        {
                            "tenant_id": int(tenant_id),
                            "job_type": "icp_discovery_enrich",
                            "dsn_hint": _dsn_hint(),
                            "error": str(aex),
                        },
                    )
                except Exception:
                    pass
                # Best-effort: proceed without the lock; dedup SELECT guards duplicates
            # Check for an existing queued/running job with the same fingerprint
            if fp:
                try:
                    cur.execute(
                        """
                        SELECT job_id
                          FROM background_jobs
                         WHERE tenant_id = %s
                           AND job_type = 'icp_discovery_enrich'
                           AND status IN ('queued','running')
                           AND COALESCE(params->>'fp','') = %s
                         ORDER BY job_id DESC
                         LIMIT 1
                        """,
                        (int(tenant_id), fp),
                    )
                    r = cur.fetchone()
                    if r and r[0] is not None:
                        jid = int(r[0])
                        try:
                            log.info(
                                '{"job":"enqueue","job_type":"icp_discovery_enrich","tenant_id":%s,"job_id":%s,"dedup":"existing","fp":"%s"}',
                                int(tenant_id),
                                jid,
                                fp,
                            )
                        except Exception:
                            pass
                        return {"job_id": jid, "dedup": True}
                except Exception as dex:
                    # If the dedup SELECT fails (e.g., table missing/perms), rollback and raise a clear error
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                    raise RuntimeError(f"dedup_select_failed: {dex}")
            # Insert new job
            cur.execute(
                "INSERT INTO background_jobs(tenant_id, job_type, status, params) VALUES (%s,'icp_discovery_enrich','queued', %s) RETURNING job_id",
                (int(tenant_id), Json(params)),
            )
            row = cur.fetchone()
            jid = int(row[0]) if row and row[0] is not None else 0
            try:
                log.info(
                    '{"job":"enqueue","job_type":"icp_discovery_enrich","tenant_id":%s,"job_id":%s,"notify_email":%s,"fp":"%s"}',
                    int(tenant_id),
                    int(jid),
                    json.dumps(notify_email) if notify_email else 'null',
                    fp or ''
                )
            except Exception:
                pass
            try:
                if jid:
                    payload = json.dumps({"job_id": jid, "type": "icp_discovery_enrich"})
                    cur.execute("NOTIFY bg_jobs, %s", (payload,))
            except Exception:
                pass
            try:
                log_json(
                    "jobs",
                    "info",
                    "enqueue_ok",
                    {"tenant_id": int(tenant_id), "job_id": int(jid), "job_type": "icp_discovery_enrich"},
                )
            except Exception:
                pass
            return {"job_id": jid}
    except Exception as e:
        # Surface detailed diagnostics while preserving exception behavior for callers
        try:
            log_json(
                "jobs",
                "error",
                "enqueue_exception",
                {
                    "tenant_id": int(tenant_id),
                    "job_type": "icp_discovery_enrich",
                    "fp": fp or None,
                    "dsn_hint": _dsn_hint(),
                    "error": str(e),
                },
            )
        except Exception:
            pass
        raise


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
        try:
            log.info("[db] UPDATE background_jobs status=running job_id=%s", job_id)
        except Exception:
            pass
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
        # Early cancel check right after start
        if _should_cancel(job_id):
            with get_conn() as conn, conn.cursor() as cur:
                try:
                    cur.execute("UPDATE background_jobs SET status='cancelled', canceled_at=now(), ended_at=now() WHERE job_id=%s", (job_id,))
                except Exception:
                    cur.execute("UPDATE background_jobs SET status='cancelled', ended_at=now() WHERE job_id=%s", (job_id,))
            try:
                log.info('{"job":"icp_discovery_enrich","phase":"cancelled","job_id":%s,"at":"start"}', job_id)
            except Exception:
                pass
            try:
                log_json("background_worker", "info", "cancelled", {"job_id": int(job_id), "tenant_id": int(tenant_id) if tenant_id is not None else None, "phase": "start"})
            except Exception:
                pass
            return
        # 1) Discovery via Deep Research (simple seed = tenant name or fallback)
        # Load a seed/company name from tenant profile or companies table as a placeholder
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT name FROM tenants WHERE tenant_id=%s", (tenant_id,))
            row = cur.fetchone()
            seed = (row and row[0]) or f"tenant_{tenant_id}"
        icp_profile = _load_icp_profile(tenant_id)
        icp_context = _icp_context_from_profile(icp_profile)
        try:
            log.info(
                "{\"job\":\"icp_discovery_enrich\",\"phase\":\"icp_context\",\"job_id\":%s,\"industries\":%s,\"buyer_titles\":%s,\"geo\":%s}",
                job_id,
                len(icp_context.get("industries") or []),
                len(icp_context.get("buyer_titles") or []),
                len(icp_context.get("geo") or []),
            )
        except Exception:
            pass
        domains: List[str] = []
        discovery_source = "jina_deep_research"
        names_by_domain: dict[str, str] = {}
        with _stage_timer(run_id, tenant_id, "bg_discovery", total_inc=1):
            if ENABLE_JINA_DEEP_RESEARCH_DISCOVERY:
                tdr = time.perf_counter()
                try:
                    log.info(
                        "{\"job\":\"icp_discovery_enrich\",\"phase\":\"deep_research_call\",\"job_id\":%s,\"seed\":\"%s\",\"industries\":%s,\"buyer_titles\":%s,\"geo\":%s}",
                        job_id,
                        str(seed).replace("\"", "'")[:120],
                        ", ".join(icp_context.get("industries") or [])[:200],
                        ", ".join(icp_context.get("buyer_titles") or [])[:200],
                        ", ".join(icp_context.get("geo") or [])[:200],
                    )
                except Exception:
                    pass
                pack = deep_research_query(str(seed), icp_context)
                domains = list(pack.get("domains") or [])
                # Optional: names mapped by discovered domain (when DR returned JSON with company_name)
                try:
                    cand = pack.get("company_names_by_domain") or {}
                    if isinstance(cand, dict):
                        names_by_domain = {str(k): str(v) for k, v in cand.items() if k and v}
                except Exception:
                    names_by_domain = {}
                try:
                    _log_event(
                        run_id,
                        tenant_id,
                        "bg_discovery",
                        event="dr_query",
                        status="ok",
                        duration_ms=int((time.perf_counter() - tdr) * 1000),
                        extra={"domains": len(domains), "source": "jina_deep_research"},
                    )
                    log.info(
                        "{\"job\":\"icp_discovery_enrich\",\"phase\":\"discovery\",\"job_id\":%s,\"source\":\"jina_deep_research\",\"domains\":%s}",
                        job_id,
                        len(domains),
                    )
                    # Also log the exact domain list discovered (up to 50)
                    import json as _json
                    log.info(
                        "{\"job\":\"icp_discovery_enrich\",\"phase\":\"discovery_domains\",\"job_id\":%s,\"list\":%s}",
                        job_id,
                        _json.dumps([f"https://{d}" if not d.startswith("http") else d for d in domains[:50]]),
                    )
                except Exception:
                    pass
            # Cooperate: cancel after discovery
            if _should_cancel(job_id):
                with get_conn() as conn, conn.cursor() as cur:
                    try:
                        cur.execute("UPDATE background_jobs SET status='cancelled', canceled_at=now(), ended_at=now() WHERE job_id=%s", (job_id,))
                    except Exception:
                        cur.execute("UPDATE background_jobs SET status='cancelled', ended_at=now() WHERE job_id=%s", (job_id,))
                try:
                    log.info('{"job":"icp_discovery_enrich","phase":"cancelled","job_id":%s,"at":"post_discovery"}', job_id)
                except Exception:
                    pass
                try:
                    log_json("background_worker", "info", "cancelled", {"job_id": int(job_id), "tenant_id": int(tenant_id) if tenant_id is not None else None, "phase": "post_discovery"})
                except Exception:
                    pass
                return
            if not domains:
                from src.settings import JINA_DEEP_RESEARCH_DISCOVERY_MAX_URLS

                fallback_domains, fallback_source = _fallback_discovery_domains(
                    str(seed),
                    limit=JINA_DEEP_RESEARCH_DISCOVERY_MAX_URLS,
                    country_hint=_DEFAULT_DISCOVERY_COUNTRY,
                )
                discovery_source = fallback_source or discovery_source
                domains = list(fallback_domains or [])
                try:
                    _log_event(
                        run_id,
                        tenant_id,
                        "bg_discovery",
                        event="fallback_discovery",
                        status="ok" if domains else "empty",
                        extra={
                            "source": discovery_source,
                            "domains": len(domains),
                            "fallback_only": not ENABLE_JINA_DEEP_RESEARCH_DISCOVERY,
                        },
                    )
                    log.info(
                        "{\"job\":\"icp_discovery_enrich\",\"phase\":\"discovery_fallback\",\"job_id\":%s,\"source\":\"%s\",\"domains\":%s}",
                        job_id,
                        discovery_source,
                        len(domains),
                    )
                    # Log fallback domain list as well
                    import json as _json
                    log.info(
                        "{\"job\":\"icp_discovery_enrich\",\"phase\":\"discovery_domains_fallback\",\"job_id\":%s,\"source\":\"%s\",\"list\":%s}",
                        job_id,
                        discovery_source,
                        _json.dumps([f"https://{d}" if not d.startswith("http") else d for d in (domains[:50] or [])]),
                    )
                except Exception:
                    pass
            # persist into staging_global_companies (ensure company_name column exists)
            if domains:
                with get_conn() as conn, conn.cursor() as cur:
                    try:
                        cur.execute("ALTER TABLE staging_global_companies ADD COLUMN IF NOT EXISTS company_name TEXT")
                    except Exception:
                        pass
                    for d in domains[:50]:
                        cname = None
                        try:
                            # domains are typically apex hosts without scheme for DR results
                            key = d
                            if isinstance(key, str) and key.startswith("http"):
                                from urllib.parse import urlparse as _urlparse
                                key = _urlparse(key).netloc
                            cname = names_by_domain.get(key)
                        except Exception:
                            cname = None
                        # Insert with conflict guard; log insert or reuse
                        cur.execute(
                            """
                            INSERT INTO staging_global_companies(tenant_id, domain, company_name, ai_metadata)
                            VALUES (%s,%s,%s,%s)
                            ON CONFLICT (tenant_id, domain, source) DO NOTHING
                            RETURNING id
                            """,
                            (tenant_id, d, cname, Json({"provenance": {"source": discovery_source}})),
                        )
                        sid_row = cur.fetchone()
                        if sid_row and sid_row[0] is not None:
                            try:
                                log.info(
                                    "{\"job\":\"icp_discovery_enrich\",\"phase\":\"staging_insert\",\"job_id\":%s,\"tenant_id\":%s,\"id\":%s,\"domain\":\"%s\",\"company_name\":%s}",
                                    job_id,
                                    tenant_id,
                                    int(sid_row[0]),
                                    d,
                                    json.dumps(cname) if cname else 'null',
                                )
                            except Exception:
                                pass
                        else:
                            # Reuse existing row id for logging clarity
                            try:
                                cur.execute(
                                    "SELECT id FROM staging_global_companies WHERE tenant_id=%s AND domain=%s AND source='web_discovery' LIMIT 1",
                                    (tenant_id, d),
                                )
                                rrow = cur.fetchone()
                                rid = int(rrow[0]) if rrow and rrow[0] is not None else None
                            except Exception:
                                rid = None
                            try:
                                log.info(
                                    "{\"job\":\"icp_discovery_enrich\",\"phase\":\"staging_reuse\",\"job_id\":%s,\"tenant_id\":%s,\"id\":%s,\"domain\":\"%s\"}",
                                    job_id,
                                    tenant_id,
                                    rid,
                                    d,
                                )
                            except Exception:
                                pass
                try:
                    log.info(
                        "{\"job\":\"icp_discovery_enrich\",\"phase\":\"discovery_store\",\"job_id\":%s,\"domains\":%s,\"source\":\"%s\"}",
                        job_id,
                        min(len(domains), 50),
                        discovery_source,
                    )
                except Exception:
                    pass
            else:
                log.info(
                    "{\"job\":\"icp_discovery_enrich\",\"phase\":\"discovery\",\"job_id\":%s,\"event\":\"no_domains\"}",
                    job_id,
                )
        # 2) Sequential enrichment per discovered domain (preserve order)
        processed = 0
        stage_domains = list(domains or [])
        total_seq = len(stage_domains)
        try:
            log.info(
                "{\"job\":\"icp_discovery_enrich\",\"phase\":\"discovery_stage\",\"job_id\":%s,\"domains\":%s}",
                job_id,
                total_seq,
            )
        except Exception:
            pass
        company_ids: List[int] = []
        if enrich_company_with_tavily is None:
            raise RuntimeError("enrich unavailable")
        with _stage_timer(run_id, tenant_id, "bg_enrich_run", total_inc=total_seq):
            try:
                log.warning(
                    "{\"job\":\"icp_discovery_enrich\",\"phase\":\"enrich_start\",\"job_id\":%s,\"companies\":%s}",
                    job_id,
                    total_seq,
                )
            except Exception:
                pass
            for idx, d in enumerate(stage_domains, start=1):
                if _should_cancel(job_id):
                    with get_conn() as conn, conn.cursor() as cur:
                        try:
                            cur.execute(
                                "UPDATE background_jobs SET status='cancelled', canceled_at=now(), ended_at=now(), processed=%s, total=%s WHERE job_id=%s",
                                (processed, total_seq, job_id),
                            )
                        except Exception:
                            cur.execute(
                                "UPDATE background_jobs SET status='cancelled', ended_at=now(), processed=%s, total=%s WHERE job_id=%s",
                                (processed, total_seq, job_id),
                            )
                    try:
                        log.info('{"job":"icp_discovery_enrich","phase":"cancelled","job_id":%s,"at":"enrich_loop","processed":%s}', job_id, processed)
                    except Exception:
                        pass
                    try:
                        log_json("background_worker", "info", "cancelled", {"job_id": int(job_id), "tenant_id": int(tenant_id) if tenant_id is not None else None, "phase": "enrich_loop", "processed": int(processed), "total": int(total_seq)})
                    except Exception:
                        pass
                    return
                # Prepare or reuse company row for this domain
                cid = None
                cname_hint = None
                try:
                    key = d
                    if isinstance(key, str) and key.startswith("http"):
                        from urllib.parse import urlparse as _urlparse
                        key = _urlparse(key).netloc
                    cname_hint = names_by_domain.get(key) if 'names_by_domain' in locals() else None
                except Exception:
                    cname_hint = None
                try:
                    with get_conn() as conn, conn.cursor() as cur:
                        cur.execute("SELECT company_id FROM companies WHERE website_domain=%s LIMIT 1", (d,))
                        r = cur.fetchone()
                        if r and r[0]:
                            cid = int(r[0])
                            try:
                                log.info(
                                    "{\"job\":\"icp_discovery_enrich\",\"phase\":\"companies_reuse\",\"job_id\":%s,\"seq\":%s,\"company_id\":%s,\"domain\":\"%s\"}",
                                    job_id,
                                    idx,
                                    cid,
                                    d,
                                )
                            except Exception:
                                pass
                        else:
                            # Fallback: derive a simple title-cased name from apex host
                            try:
                                host = d
                                if isinstance(host, str) and host.startswith("http"):
                                    from urllib.parse import urlparse as _urlparse
                                    host = _urlparse(host).netloc
                                core = (host.split(".")[0] or "").replace("-", " ")
                                fallback_name = core.title() if core else None
                            except Exception:
                                fallback_name = None
                            cname = cname_hint or fallback_name or None
                            cur.execute(
                                "INSERT INTO companies(name, website_domain, last_seen) VALUES (%s,%s,now()) RETURNING company_id",
                                (cname, d),
                            )
                            rr = cur.fetchone()
                            if rr and rr[0]:
                                cid = int(rr[0])
                                try:
                                    log.info(
                                        "{\"job\":\"icp_discovery_enrich\",\"phase\":\"companies_insert\",\"job_id\":%s,\"seq\":%s,\"company_id\":%s,\"name\":%s,\"domain\":\"%s\"}",
                                        job_id,
                                        idx,
                                        cid,
                                        json.dumps(cname) if cname else 'null',
                                        d,
                                    )
                                except Exception:
                                    pass
                except Exception as prep_exc:
                    _log_event(run_id, tenant_id, "bg_enrich_run", event="company", status="error", error_code=type(prep_exc).__name__, extra={"step":"prepare","domain":d})
                    continue

                if not cid:
                    continue
                company_ids.append(cid)
                # Enrich sequentially for this company
                t1 = time.perf_counter()
                try:
                    try:
                        log.info(
                            "{\"job\":\"icp_discovery_enrich\",\"phase\":\"enrich_company_start\",\"job_id\":%s,\"seq\":%s,\"total\":%s,\"company_id\":%s,\"domain\":\"%s\"}",
                            job_id, idx, total_seq, cid, d,
                        )
                    except Exception:
                        pass
                    await enrich_company_with_tavily(cid, company_name=cname_hint, search_policy="require_existing")
                    processed += 1
                    _log_event(run_id, tenant_id, "bg_enrich_run", event="company", status="ok", company_id=cid, duration_ms=int((time.perf_counter()-t1)*1000))
                except Exception as e:
                    _log_event(run_id, tenant_id, "bg_enrich_run", event="company", status="error", company_id=cid, error_code=type(e).__name__)
                    continue
        try:
            log.info(
                "{\"job\":\"icp_discovery_enrich\",\"phase\":\"enrich\",\"job_id\":%s,\"processed\":%s,\"total\":%s}",
                job_id,
                processed,
                total_seq,
            )
        except Exception:
            pass
        # Scoring for all enriched companies (best-effort)
        try:
            if company_ids:
                tsc = time.perf_counter()
                scoring_state = {
                    "candidate_ids": company_ids,
                    "lead_features": [],
                    "lead_scores": [],
                    "icp_payload": {},
                    "tenant_id": int(tenant_id) if tenant_id is not None else None,
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
                try:
                    log.info(
                        "{\"job\":\"icp_discovery_enrich\",\"phase\":\"scoring\",\"job_id\":%s,\"companies\":%s}",
                        job_id,
                        len(company_ids),
                    )
                except Exception:
                    pass
        except Exception:
            pass
        
        # Email notification (best-effort)
        try:
            to_email = notify_email
            # If not provided on enqueue, try to resolve from job params, tenant_users, or DEFAULT_NOTIFY_EMAIL
            if not to_email:
                try:
                    with get_conn() as conn, conn.cursor() as cur:
                        cur.execute("SELECT params FROM background_jobs WHERE job_id=%s", (job_id,))
                        row = cur.fetchone()
                        current_params = row[0] if row and row[0] else {}
                        if isinstance(current_params, dict):
                            to_email = current_params.get("notify_email")
                except Exception:
                    to_email = None
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
                            from src.settings import EMAIL_DEV_ACCEPT_TENANT_USER_ID_AS_EMAIL as _ACCEPT_TU_EMAIL  # type: ignore
                            if _ACCEPT_TU_EMAIL and ("@" in candidate):
                                to_email = candidate
                                cur.execute(
                                    "UPDATE background_jobs SET params=COALESCE(params,'{}'::jsonb) || jsonb_build_object('notify_email', %s, 'notify_email_source','tenant_users') WHERE job_id=%s",
                                    (to_email, job_id),
                                )
                except Exception:
                    pass
            if not to_email:
                try:
                    from src.settings import DEFAULT_NOTIFY_EMAIL as _DEF_TO  # type: ignore
                    if _DEF_TO and ("@" in str(_DEF_TO)):
                        to_email = str(_DEF_TO)
                        with get_conn() as conn, conn.cursor() as cur:
                            cur.execute(
                                "UPDATE background_jobs SET params=COALESCE(params,'{}'::jsonb) || jsonb_build_object('notify_email', %s, 'notify_email_source','default_env') WHERE job_id=%s",
                                (to_email, job_id),
                            )
                except Exception:
                    pass
            if to_email and processed > 0:
                t2 = time.perf_counter()
                try:
                    log.info(
                        "{\"job\":\"icp_discovery_enrich\",\"phase\":\"email_start\",\"job_id\":%s,\"to\":\"%s\"}",
                        job_id,
                        (to_email or "").replace("\"", "'")[:200],
                    )
                except Exception:
                    pass
                res = await agentic_send_results(to_email, tenant_id, limit=500)
                elapsed_ms = int((time.perf_counter() - t2) * 1000)
                _log_event(
                    run_id,
                    tenant_id,
                    "email_notify",
                    event="send",
                    status=str(res.get("status") or "ok"),
                    duration_ms=elapsed_ms,
                    extra={"to": to_email, "http_status": res.get("http_status"), "request_id": res.get("request_id")},
                )
                try:
                    log_json(
                        "background_worker",
                        "info",
                        "email",
                        {
                            "job_id": int(job_id),
                            "tenant_id": int(tenant_id) if tenant_id is not None else None,
                            "to": to_email,
                            "status": res.get("status"),
                            "http_status": res.get("http_status"),
                            "request_id": res.get("request_id"),
                            "duration_ms": elapsed_ms,
                        },
                    )
                except Exception:
                    pass
                try:
                    log.info(
                        "{\"job\":\"icp_discovery_enrich\",\"phase\":\"email\",\"job_id\":%s,\"to\":\"%s\",\"status\":\"%s\",\"http_status\":%s,\"request_id\":%s,\"duration_ms\":%s}",
                        job_id,
                        (to_email or "").replace("\"","'")[:200],
                        str(res.get("status") or "ok"),
                        json.dumps(res.get("http_status")) if isinstance(res.get("http_status"), int) else json.dumps(res.get("http_status")),
                        json.dumps(res.get("request_id")),
                        elapsed_ms,
                    )
                except Exception:
                    pass
                if (res or {}).get("status") == "sent":
                    with get_conn() as conn, conn.cursor() as cur:
                        cur.execute(
                            "UPDATE background_jobs SET params=COALESCE(params,'{}'::jsonb) || jsonb_build_object('email_sent_at', now(), 'email_to', %s) WHERE job_id=%s",
                            (to_email, job_id),
                        )
        except Exception:
            pass
        # Export to Odoo (best-effort) — after email
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
                odoo_counts = await _odoo_export_for_ids(tenant_id, rows)
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
                try:
                    log_json(
                        "background_worker",
                        "info",
                        "odoo_export",
                        {
                            "job_id": int(job_id),
                            "tenant_id": int(tenant_id) if tenant_id is not None else None,
                            "companies": len(rows),
                            "upserts": int(odoo_counts.get("upserts", 0)),
                            "contacts": int(odoo_counts.get("contacts", 0)),
                            "leads": int(odoo_counts.get("leads", 0)),
                        },
                    )
                except Exception:
                    pass
        except Exception:
            pass
        # Job end summary (scoring/email/Odoo export)
        try:
            summary_email_status = None
            summary_email_http = None
            try:
                # best effort: check last email event emitted above
                # we can't read it back; instead, reuse local variable if present
                summary_email_status = locals().get("res", {}).get("status")  # type: ignore[arg-type]
                summary_email_http = locals().get("res", {}).get("http_status")  # type: ignore[arg-type]
            except Exception:
                pass
            log_json(
                "background_worker",
                "info",
                "job_end",
                {
                    "job_id": int(job_id),
                    "tenant_id": int(tenant_id) if tenant_id is not None else None,
                    "companies": len(company_ids or []),
                    "processed": int(processed),
                    "scored": len(company_ids or []),
                    "email_status": summary_email_status,
                    "email_http_status": summary_email_http,
                },
            )
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
            try:
                log.info(
                    "[db] UPDATE background_jobs status=done job_id=%s processed=%s total=%s",
                    job_id, processed, len(company_ids)
                )
            except Exception:
                pass
        dur_ms = int((time.perf_counter() - t0) * 1000)
        log.info("{\"job\":\"icp_discovery_enrich\",\"phase\":\"finish\",\"job_id\":%s,\"processed\":%s,\"duration_ms\":%s}", job_id, processed, dur_ms)
    except Exception as e:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE background_jobs SET status='error', error=%s, ended_at=now() WHERE job_id=%s",
                (str(e), job_id),
            )
            try:
                log.info("[db] UPDATE background_jobs status=error job_id=%s", job_id)
            except Exception:
                pass
        log.exception("icp_discovery_enrich failed: %s", e)
    finally:
        _finalize_run(run_id, status="succeeded")


def enqueue_web_discovery_bg_enrich(tenant_id: int, company_ids: list[int], notify_email: Optional[str] = None) -> dict:
    """Deprecated alias delegating to enqueue_icp_discovery_enrich."""
    log.warning("enqueue_web_discovery_bg_enrich is deprecated; delegating to enqueue_icp_discovery_enrich")
    return enqueue_icp_discovery_enrich(tenant_id, notify_email=notify_email)

async def run_web_discovery_bg_enrich(job_id: int) -> None:
    """Deprecated runner delegating to run_icp_discovery_enrich."""
    log.warning("run_web_discovery_bg_enrich is deprecated; delegating job_id=%s to run_icp_discovery_enrich", job_id)
    await run_icp_discovery_enrich(job_id)

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
            # Run lead scoring first so email can reflect latest shortlist
            try:
                if scored_ids:
                    from src.lead_scoring import lead_scoring_agent
                    scoring_state = {
                        "candidate_ids": scored_ids,
                        "lead_features": [],
                        "lead_scores": [],
                        "icp_payload": {},
                        "tenant_id": int(tenant_id) if tenant_id is not None else None,
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
        # If the nightly batch fully completed, send the shortlist email once scoring is done,
        # then perform Odoo export (scoring -> email -> Odoo export).
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
                # After email attempt, perform Odoo export for the selected rows
                try:
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


async def _odoo_export_for_ids(tenant_id: Optional[int], company_rows: list[tuple]) -> dict:
    """Best-effort Odoo sync for a batch of companies after nightly enrichment.

    For each (company_id, name, uen) in company_rows, upsert to Odoo partner,
    add primary contact if present, and create a lead with the latest score.
    Non-fatal: all exceptions are swallowed per item; logs in OdooStore cover details.
    """
    if not company_rows:
        return {"total": 0, "upserts": 0, "contacts": 0, "leads": 0}
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
        return {"total": 0, "upserts": 0, "contacts": 0, "leads": 0}
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
    upserts = 0
    contacts_added = 0
    leads_created = 0
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
            try:
                upserts += 1
                log_json(
                    "odoo_export",
                    "info",
                    "upsert_company",
                    {
                        "tenant_id": int(tenant_id) if tenant_id is not None else None,
                        "company_id": int(cid),
                        "partner_id": int(partner_id) if isinstance(partner_id, int) else partner_id,
                        "name": comp.get("name"),
                        "domain": comp.get("website_domain"),
                    },
                )
            except Exception:
                pass
            email = emails.get(cid)
            if email:
                try:
                    await store.add_contact(partner_id, email)
                    contacts_added += 1
                    try:
                        log_json(
                            "odoo_export",
                            "info",
                            "add_contact",
                            {
                                "tenant_id": int(tenant_id) if tenant_id is not None else None,
                                "company_id": int(cid),
                                "partner_id": int(partner_id) if isinstance(partner_id, int) else partner_id,
                                "email": email,
                            },
                        )
                    except Exception:
                        pass
                except Exception:
                    pass
            score = scores.get(cid, 0.0)
            rationale = rationales.get(cid, "")
            try:
                await store.create_lead_if_high(partner_id, comp.get("name"), score, {}, rationale, email)
                leads_created += 1
                try:
                    log_json(
                        "odoo_export",
                        "info",
                        "create_lead",
                        {
                            "tenant_id": int(tenant_id) if tenant_id is not None else None,
                            "company_id": int(cid),
                            "partner_id": int(partner_id) if isinstance(partner_id, int) else partner_id,
                            "score": float(score),
                            "email": email,
                        },
                    )
                except Exception:
                    pass
            except Exception:
                pass
        except Exception:
            # continue best-effort for the rest
            continue
    try:
        log_json(
            "odoo_export",
            "info",
            "batch_summary",
            {
                "tenant_id": int(tenant_id) if tenant_id is not None else None,
                "total": len(ids),
                "upserts": upserts,
                "contacts": contacts_added,
                "leads": leads_created,
            },
        )
    except Exception:
        pass
    return {"total": len(ids), "upserts": upserts, "contacts": contacts_added, "leads": leads_created}
