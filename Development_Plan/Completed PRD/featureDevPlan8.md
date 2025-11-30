# Dev Plan — Feature 8: Observability, QA & Runbooks

Source: featurePRD8.md (Observability, QA & Runbooks)

Objective: Implement per-tenant/per-run metrics, structured logs, QA sampling, light dashboards, alerts, and runbooks with minimal moving parts and clear DB-backed aggregation. Keep secrets off the client, scope everything to `tenant_id`, and ensure retention hygiene.

---

## Architecture Overview

- DB-first observability: capture raw events in `run_event_logs`, aggregate into `run_stage_stats` and `run_vendor_usage`, and sample QA into `qa_samples`.
- Lightweight helpers in `src/obs.py` to record runs, stages, vendor usage, and events from runners and enrichment code.
- Optional Prometheus later — start with SQL/Metabase dashboards and FastAPI endpoints for JSON/CSV.

---

## Migrations (SQL)

File suggestion: `lead_generation-main/app/migrations/008_observability.sql`

```sql
-- Enrichment runs: add a trace URL for LangSmith correlation
ALTER TABLE IF EXISTS enrichment_runs
  ADD COLUMN IF NOT EXISTS langsmith_trace_url TEXT;

-- Stage-level stats per run
CREATE TABLE IF NOT EXISTS run_stage_stats (
  run_id           BIGINT      NOT NULL,
  tenant_id        INT         NOT NULL,
  stage            VARCHAR(64) NOT NULL,
  count_total      INT         DEFAULT 0,
  count_success    INT         DEFAULT 0,
  count_error      INT         DEFAULT 0,
  p50_ms           INT         DEFAULT 0,
  p95_ms           INT         DEFAULT 0,
  p99_ms           INT         DEFAULT 0,
  cost_usd         NUMERIC(12,4) DEFAULT 0,
  tokens_input     INT         DEFAULT 0,
  tokens_output    INT         DEFAULT 0,
  PRIMARY KEY (run_id, tenant_id, stage)
);
CREATE INDEX IF NOT EXISTS idx_rss_tenant_run_stage ON run_stage_stats(tenant_id, run_id, stage);

-- Vendor usage per run (OpenAI, Tavily, ZeroBounce, apify_linkedin, etc.)
CREATE TABLE IF NOT EXISTS run_vendor_usage (
  run_id           BIGINT      NOT NULL,
  tenant_id        INT         NOT NULL,
  vendor           VARCHAR(64) NOT NULL,
  calls            INT         DEFAULT 0,
  errors           INT         DEFAULT 0,
  cost_usd         NUMERIC(12,4) DEFAULT 0,
  rate_limit_hits  INT         DEFAULT 0,
  quota_exhausted  BOOL        DEFAULT FALSE,
  tokens_input     INT         DEFAULT 0,
  tokens_output    INT         DEFAULT 0,
  PRIMARY KEY (run_id, tenant_id, vendor)
);
CREATE INDEX IF NOT EXISTS idx_rvu_tenant_run_vendor ON run_vendor_usage(tenant_id, run_id, vendor);

-- Fine-grain event log for debugging (short retention)
CREATE TABLE IF NOT EXISTS run_event_logs (
  run_id       BIGINT      NOT NULL,
  tenant_id    INT         NOT NULL,
  stage        VARCHAR(64) NOT NULL,
  company_id   INT         NULL,
  event        VARCHAR(64) NOT NULL,  -- start|finish|retry|vendor_call|error|qa_sampled|...
  status       VARCHAR(32) NOT NULL,  -- ok|error|skip|retry
  error_code   VARCHAR(64) NULL,
  duration_ms  INT         NULL,
  trace_id     TEXT        NULL,
  extra        JSONB       NULL,
  ts           TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_rel_run_stage_ts ON run_event_logs(run_id, stage, ts DESC);
CREATE INDEX IF NOT EXISTS idx_rel_tenant_run ON run_event_logs(tenant_id, run_id);

-- QA samples for manual checks per run
CREATE TABLE IF NOT EXISTS qa_samples (
  run_id     BIGINT      NOT NULL,
  tenant_id  INT         NOT NULL,
  company_id INT         NOT NULL,
  bucket     VARCHAR(16) NOT NULL,   -- High|Medium|Low
  checks     JSONB       NOT NULL,   -- skeleton checklist
  result     VARCHAR(16) NOT NULL DEFAULT 'needs_review', -- pass|fail|needs_review
  notes      TEXT        NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (run_id, tenant_id, company_id)
);
CREATE INDEX IF NOT EXISTS idx_qa_tenant_run ON qa_samples(tenant_id, run_id);

-- (Optional) RLS enablement follows the project pattern; apply only if RLS is enabled globally
-- ALTER TABLE run_stage_stats ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE run_vendor_usage ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE run_event_logs ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE qa_samples ENABLE ROW LEVEL SECURITY;
```

Retention job (SQL):

```sql
-- keep detailed events for 30 days
DELETE FROM run_event_logs WHERE ts < NOW() - INTERVAL '30 days';
-- keep QA samples for 90 days
DELETE FROM qa_samples WHERE created_at < NOW() - INTERVAL '90 days';
```

---

## Library: `src/obs.py`

Purpose: Small, dependency-light helpers for recording runs, stage timers, vendor usage, events, and aggregations.

```python
# lead_generation-main/src/obs.py
from __future__ import annotations
import time
from contextlib import contextmanager
from typing import Optional, Any, Dict
from src.database import get_conn

def begin_run(tenant_id: int, trace_url: Optional[str] = None) -> int:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO enrichment_runs(tenant_id, started_at, langsmith_trace_url) VALUES (%s, NOW(), %s) RETURNING run_id",
            (tenant_id, trace_url),
        )
        return int(cur.fetchone()[0])

def finalize_run(run_id: int, status: str = "succeeded") -> None:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE enrichment_runs SET ended_at = NOW(), status = %s WHERE run_id = %s",
            (status, run_id),
        )

def log_event(run_id: int, tenant_id: int, stage: str, event: str, status: str,
              *, company_id: Optional[int] = None, error_code: Optional[str] = None,
              duration_ms: Optional[int] = None, trace_id: Optional[str] = None,
              extra: Optional[Dict[str, Any]] = None) -> None:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO run_event_logs(run_id, tenant_id, stage, company_id, event, status, error_code, duration_ms, trace_id, extra)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (run_id, tenant_id, stage, company_id, event, status, error_code, duration_ms, trace_id, extra),
        )

def bump_vendor(run_id: int, tenant_id: int, vendor: str, *, calls: int = 0, errors: int = 0,
                cost_usd: float = 0.0, rate_limit_hits: int = 0,
                tokens_in: int = 0, tokens_out: int = 0) -> None:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO run_vendor_usage(run_id, tenant_id, vendor, calls, errors, cost_usd, rate_limit_hits, tokens_input, tokens_output)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (run_id, tenant_id, vendor) DO UPDATE SET
              calls = run_vendor_usage.calls + EXCLUDED.calls,
              errors = run_vendor_usage.errors + EXCLUDED.errors,
              cost_usd = run_vendor_usage.cost_usd + EXCLUDED.cost_usd,
              rate_limit_hits = run_vendor_usage.rate_limit_hits + EXCLUDED.rate_limit_hits,
              tokens_input = run_vendor_usage.tokens_input + EXCLUDED.tokens_input,
              tokens_output = run_vendor_usage.tokens_output + EXCLUDED.tokens_output
            """,
            (run_id, tenant_id, vendor, calls, errors, cost_usd, rate_limit_hits, tokens_in, tokens_out),
        )

@contextmanager
def stage_timer(run_id: int, tenant_id: int, stage: str, *, total_inc: int = 0):
    t0 = time.time()
    log_event(run_id, tenant_id, stage, event="start", status="ok")
    try:
        yield
        dur = int((time.time() - t0) * 1000)
        log_event(run_id, tenant_id, stage, event="finish", status="ok", duration_ms=dur)
        _inc_stage(run_id, tenant_id, stage, total=total_inc, ok=total_inc)
    except Exception as e:
        dur = int((time.time() - t0) * 1000)
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

def aggregate_percentiles(run_id: int, tenant_id: int) -> None:
    """Compute p50/p95/p99 per stage from event durations and update run_stage_stats."""
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
```

---

## Runner Integration (`scripts/run_nightly.py`)

Key changes (pseudocode):

```python
from src import obs

async def run_tenant(tenant_id: int):
    os.environ["DEFAULT_TENANT_ID"] = str(tenant_id)
    run_id = obs.begin_run(tenant_id)
    try:
        with obs.stage_timer(run_id, tenant_id, "mv_refresh"):
            refresh_icp_candidates()

        with obs.stage_timer(run_id, tenant_id, "select_targets"):
            targets = select_target_set(...)

        with obs.stage_timer(run_id, tenant_id, "crawl", total_inc=len(targets)):
            await enrich_many(targets)  # inside, call obs.log_event per company and vendor bumps

        with obs.stage_timer(run_id, tenant_id, "score", total_inc=len(targets)):
            scoring_state = await lead_scoring_agent.ainvoke(...)

        with obs.stage_timer(run_id, tenant_id, "export_odoo"):
            export_to_odoo(scoring_state)

        obs.aggregate_percentiles(run_id, tenant_id)
        obs.finalize_run(run_id, status="succeeded")
    except Exception:
        obs.finalize_run(run_id, status="failed")
        raise
```

Vendor usage examples inside enrichment and LLM code paths:

```python
# OpenAI
obs.bump_vendor(run_id, tenant_id, "openai", calls=1, tokens_in=tok_in, tokens_out=tok_out, cost_usd=estimate_cost_usd)

# Tavily, ZeroBounce, Apify LinkedIn
obs.bump_vendor(run_id, tenant_id, "tavily", calls=1)
obs.bump_vendor(run_id, tenant_id, "zerobounce", calls=len(batch), errors=err_cnt)
obs.bump_vendor(run_id, tenant_id, "apify_linkedin", calls=1, errors=int(failed))
```

Per-company events in enrichment:

```python
obs.log_event(run_id, tenant_id, stage="crawl", event="vendor_call", status="ok", company_id=cid, duration_ms=dur_ms)
obs.log_event(run_id, tenant_id, stage="verify", event="vendor_call", status="error", company_id=cid, error_code="HTTP_429")
```

---

## FastAPI Endpoints

Add to `app/main.py` (pattern mirrors `/shortlist/status`):

```python
@app.get("/runs/{run_id}/stats")
async def get_run_stats(run_id: int, _: dict = Depends(require_auth)):
    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        rows1 = await conn.fetch("SELECT * FROM run_stage_stats WHERE run_id=$1 ORDER BY stage", run_id)
        rows2 = await conn.fetch("SELECT * FROM run_vendor_usage WHERE run_id=$1 ORDER BY vendor", run_id)
    return {"run_id": run_id, "stage_stats": [dict(r) for r in rows1], "vendor_usage": [dict(r) for r in rows2]}

@app.get("/export/run_events.csv")
async def export_run_events(run_id: int, _: dict = Depends(require_auth)):
    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT run_id, tenant_id, stage, company_id, event, status, error_code, duration_ms, trace_id, extra, ts FROM run_event_logs WHERE run_id=$1 ORDER BY ts",
            run_id,
        )
    import csv, io
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["run_id","tenant_id","stage","company_id","event","status","error_code","duration_ms","trace_id","extra","ts"])
    for r in rows:
        w.writerow([r[k] for k in ("run_id","tenant_id","stage","company_id","event","status","error_code","duration_ms","trace_id","extra","ts")])
    from fastapi.responses import Response
    return Response(content=buf.getvalue(), media_type="text/csv")
```

---

## QA Sampling

Helper to create QA samples after scoring (legacy helper now removed from the codebase; retained here for archival context):

```python
def create_qa_samples(run_id: int, tenant_id: int, high_bucket_company_ids: list[int], limit: int = 10):
    import random
    picks = random.sample(high_bucket_company_ids, min(limit, len(high_bucket_company_ids)))
    checklist = {"domain": False, "about_text": False, "contacts": False, "rationale": False}
    with get_conn() as conn, conn.cursor() as cur:
        for cid in picks:
            cur.execute(
                """
                INSERT INTO qa_samples(run_id, tenant_id, company_id, bucket, checks, result)
                VALUES (%s,%s,%s,%s,%s::jsonb,'needs_review')
                ON CONFLICT DO NOTHING
                """,
                (run_id, tenant_id, cid, "High", json.dumps(checklist)),
            )
```

Export QA to CSV (reuse `/export/run_events.csv` pattern or create `/export/qa.csv?run_id=`).

---

## Alerts (Simple Worker)

Script `scripts/alerts.py` checks last run per tenant and posts to Slack on conditions (MV failure, high error rate, quotas):

```python
import os, asyncio, httpx
from src.database import get_conn

SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK_URL")

def _post(msg: str):
    if not SLACK_WEBHOOK: return
    try:
        httpx.post(SLACK_WEBHOOK, json={"text": msg}, timeout=5.0)
    except Exception: pass

def check_last_run_alerts():
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""
          WITH last_runs AS (
            SELECT tenant_id, MAX(run_id) AS run_id FROM enrichment_runs GROUP BY tenant_id
          )
          SELECT l.tenant_id, r.run_id, r.status
          FROM last_runs l JOIN enrichment_runs r USING(run_id)
        """)
        for tid, run_id, status in cur.fetchall():
            if status != 'succeeded':
                _post(f"Run {run_id} for tenant {tid} failed: status={status}")
```

Cron: run every 5 minutes or after nightly completes.

---

## Dashboards (SQL starters)

- Tenant overview:

```sql
SELECT rss.tenant_id, rss.run_id,
       SUM(rss.count_total) AS total_items,
       SUM(rss.count_error) AS total_errors,
       MAX(rss.p95_ms) AS max_p95_ms,
       MAX(rss.p99_ms) AS max_p99_ms
FROM run_stage_stats rss
GROUP BY 1,2
ORDER BY 2 DESC;
```

- Vendor usage:

```sql
SELECT tenant_id, run_id, vendor, calls, errors, cost_usd, rate_limit_hits
FROM run_vendor_usage
ORDER BY run_id DESC, vendor;
```

---

## Testing & Verification

- Local dry run: insert fake `run_event_logs` rows and run `aggregate_percentiles` to verify p50/p95/p99.
- E2E: run nightly on a staging tenant; confirm rows appear in all four tables and `/runs/{run_id}/stats` returns sane values.
- QA: ensure `qa_samples` has 10 High entries; manually fill `result` and export.

---

## Rollout Steps

1) Apply migration 008 and deploy `src/obs.py`.
2) Integrate helpers into `scripts/run_nightly.py` and critical vendor call sites.
3) Add `/runs/{run_id}/stats` and `/export/run_events.csv` endpoints.
4) Set up Metabase charts; wire Slack webhook in staging; simulate alerts.
5) Add retention job (cron/worker) for event logs and QA samples.
6) Document runbooks MV-01, CR-02, VQ-03, OA-04, CC-05, OE-06 in ops wiki.
