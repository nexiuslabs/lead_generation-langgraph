from __future__ import annotations
import time
from contextlib import contextmanager
from typing import Optional, Any, Dict
from src.database import get_conn
import logging as _logging
import json as _json

def begin_run(tenant_id: int, trace_url: Optional[str] = None) -> int:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO enrichment_runs(tenant_id, started_at, status, langsmith_trace_url)
            VALUES (%s, NOW(), %s, %s)
            RETURNING run_id
            """,
            (tenant_id, 'running', trace_url),
        )
        rid = int(cur.fetchone()[0])
        try:
            _logging.getLogger("obs.db").info(
                "[db] INSERT enrichment_runs tenant_id=%s status=%s run_id=%s",
                tenant_id, 'running', rid,
            )
        except Exception:
            pass
        return rid

# Simple global run context for vendor usage in modules that don't receive run_id/tenant_id explicitly
_CTX: Dict[str, Optional[int]] = {"run_id": None, "tenant_id": None}

def set_run_context(run_id: int, tenant_id: int) -> None:
    _CTX["run_id"] = int(run_id)
    _CTX["tenant_id"] = int(tenant_id)

def get_run_context() -> tuple[Optional[int], Optional[int]]:
    return _CTX.get("run_id"), _CTX.get("tenant_id")

def finalize_run(run_id: int, status: str = "succeeded") -> None:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE enrichment_runs SET ended_at = NOW(), status = %s WHERE run_id = %s",
            (status, run_id),
        )
        try:
            _logging.getLogger("obs.db").info("[db] UPDATE enrichment_runs run_id=%s status=%s", run_id, status)
        except Exception:
            pass

def mark_run_degraded(run_id: int) -> None:
    """Best-effort update to flag a run as degraded when any fallback/degradation occurred.

    No-ops silently if the column doesn't exist.
    """
    try:
        with get_conn() as conn, conn.cursor() as cur:
            try:
                cur.execute("ALTER TABLE enrichment_runs ADD COLUMN IF NOT EXISTS degraded BOOL DEFAULT FALSE")
            except Exception:
                # Ignore if no permission or already present
                pass
            try:
                cur.execute("UPDATE enrichment_runs SET degraded=TRUE WHERE run_id=%s", (run_id,))
            except Exception:
                pass
    except Exception:
        pass

def persist_manifest(run_id: int, tenant_id: int, selected_ids: list[int]) -> None:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO run_manifests(run_id, tenant_id, selected_ids)
            VALUES (%s,%s,%s)
            ON CONFLICT (run_id) DO UPDATE SET selected_ids = EXCLUDED.selected_ids
            """,
            (run_id, tenant_id, selected_ids),
        )
        try:
            _logging.getLogger("obs.db").info(
                "[db] UPSERT run_manifests run_id=%s tenant_id=%s count=%s",
                run_id, tenant_id, len(selected_ids or []),
            )
        except Exception:
            pass

def write_summary(run_id: int, tenant_id: int, *, candidates: int, processed: int, batches: int) -> None:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO run_summaries(run_id, tenant_id, candidates, processed, batches)
            VALUES (%s,%s,%s,%s,%s)
            ON CONFLICT (run_id) DO UPDATE SET candidates=EXCLUDED.candidates, processed=EXCLUDED.processed, batches=EXCLUDED.batches
            """,
            (run_id, tenant_id, candidates, processed, batches),
        )
        try:
            _logging.getLogger("obs.db").info(
                "[db] UPSERT run_summaries run_id=%s tenant_id=%s candidates=%s processed=%s batches=%s",
                run_id, tenant_id, candidates, processed, batches,
            )
        except Exception:
            pass

def log_event(run_id: int, tenant_id: int, stage: str, event: str, status: str,
              *, company_id: Optional[int] = None, error_code: Optional[str] = None,
              duration_ms: Optional[int] = None, trace_id: Optional[str] = None,
              extra: Optional[Dict[str, Any]] = None) -> None:
    # Structured JSON log
    try:
        _logging.getLogger("obs.event").info(_json.dumps({
            "run_id": run_id,
            "tenant_id": tenant_id,
            "stage": stage,
            "company_id": company_id,
            "event": event,
            "status": status,
            "error_code": error_code,
            "duration_ms": duration_ms,
            "trace_id": trace_id,
            "extra": extra,
        }))
    except Exception:
        pass
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO run_event_logs(run_id, tenant_id, stage, company_id, event, status, error_code, duration_ms, trace_id, extra)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (run_id, tenant_id, stage, company_id, event, status, error_code, duration_ms, trace_id, extra),
        )
        try:
            _logging.getLogger("obs.db").info(
                "[db] INSERT run_event_logs run_id=%s tenant_id=%s stage=%s event=%s status=%s company_id=%s",
                run_id, tenant_id, stage, event, status, company_id,
            )
        except Exception:
            pass

def bump_vendor(run_id: int, tenant_id: int, vendor: str, *, calls: int = 0, errors: int = 0,
                tokens_in: int = 0, tokens_out: int = 0, cost_usd: float = 0.0,
                rate_limit_hits: int = 0, quota_exhausted: bool = False) -> None:
    # Structured JSON log
    try:
        _logging.getLogger("obs.vendor").info(_json.dumps({
            "run_id": run_id,
            "tenant_id": tenant_id,
            "vendor": vendor,
            "calls": calls,
            "errors": errors,
            "tokens_input": tokens_in,
            "tokens_output": tokens_out,
            "cost_usd": round(cost_usd, 6),
            "rate_limit_hits": rate_limit_hits,
            "quota_exhausted": quota_exhausted,
        }))
    except Exception:
        pass
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO run_vendor_usage(run_id, tenant_id, vendor, calls, errors, tokens_input, tokens_output, cost_usd, rate_limit_hits, quota_exhausted)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (run_id, tenant_id, vendor) DO UPDATE SET
              calls = run_vendor_usage.calls + EXCLUDED.calls,
              errors = run_vendor_usage.errors + EXCLUDED.errors,
              tokens_input = run_vendor_usage.tokens_input + EXCLUDED.tokens_input,
              tokens_output = run_vendor_usage.tokens_output + EXCLUDED.tokens_output,
              cost_usd = run_vendor_usage.cost_usd + EXCLUDED.cost_usd,
              rate_limit_hits = run_vendor_usage.rate_limit_hits + EXCLUDED.rate_limit_hits,
              quota_exhausted = run_vendor_usage.quota_exhausted OR EXCLUDED.quota_exhausted
            """,
            (run_id, tenant_id, vendor, calls, errors, tokens_in, tokens_out, cost_usd, rate_limit_hits, quota_exhausted),
        )
        try:
            _logging.getLogger("obs.db").info(
                "[db] UPSERT run_vendor_usage run_id=%s tenant_id=%s vendor=%s calls=%s errors=%s cost=%s",
                run_id, tenant_id, vendor, calls, errors, round(cost_usd, 6),
            )
        except Exception:
            pass

@contextmanager
def stage_timer(run_id: int, tenant_id: int, stage: str, *, total_inc: int = 0):
    t0 = time.perf_counter()
    log_event(run_id, tenant_id, stage, event="start", status="ok")
    try:
        yield
        dur = int((time.perf_counter() - t0) * 1000)
        log_event(run_id, tenant_id, stage, event="finish", status="ok", duration_ms=dur)
        _inc_stage(run_id, tenant_id, stage, total=total_inc, ok=total_inc)
    except Exception as e:
        dur = int((time.perf_counter() - t0) * 1000)
        log_event(run_id, tenant_id, stage, event="error", status="error", duration_ms=dur, error_code=type(e).__name__)
        _inc_stage(run_id, tenant_id, stage, total=total_inc, err=total_inc)
        raise

def _inc_stage(run_id: int, tenant_id: int, stage: str, *, total: int = 0, ok: int = 0, err: int = 0) -> None:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO run_stage_stats(run_id, tenant_id, stage, count_total, count_success, count_error)
            VALUES (%s,%s,%s,%s,%s,%s)
            ON CONFLICT (run_id, tenant_id, stage) DO UPDATE SET
              count_total = run_stage_stats.count_total + EXCLUDED.count_total,
              count_success = run_stage_stats.count_success + EXCLUDED.count_success,
              count_error = run_stage_stats.count_error + EXCLUDED.count_error
            """,
            (run_id, tenant_id, stage, total, ok, err),
        )
        try:
            _logging.getLogger("obs.db").info(
                "[db] UPSERT run_stage_stats run_id=%s tenant_id=%s stage=%s total=%s ok=%s err=%s",
                run_id, tenant_id, stage, total, ok, err,
            )
        except Exception:
            pass


def aggregate_percentiles(run_id: int, tenant_id: int) -> None:
    """Compute p50/p95/p99 per stage from run_event_logs and update run_stage_stats."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            WITH stats AS (
              SELECT stage,
                     percentile_cont(0.5) WITHIN GROUP (ORDER BY duration_ms) AS p50,
                     percentile_cont(0.95) WITHIN GROUP (ORDER BY duration_ms) AS p95,
                     percentile_cont(0.99) WITHIN GROUP (ORDER BY duration_ms) AS p99
              FROM run_event_logs
              WHERE run_id = %s AND tenant_id = %s AND duration_ms IS NOT NULL
              GROUP BY stage
            )
            UPDATE run_stage_stats rss
            SET p50_ms = COALESCE(s.p50, 0)::INT,
                p95_ms = COALESCE(s.p95, 0)::INT,
                p99_ms = COALESCE(s.p99, 0)::INT
            FROM stats s
            WHERE rss.run_id = %s AND rss.tenant_id = %s AND rss.stage = s.stage
            """,
            (run_id, tenant_id, run_id, tenant_id),
        )

