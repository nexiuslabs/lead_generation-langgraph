**Feature Dev Plan 18 — Responsive ICP Industry Search & Background Upsert**

Source: Development/featurePRD18.md

Goal: Remove heavy staging→companies upserts from the chat request path, batch and cap ingestion, add the right DB indexes, and improve UX responsiveness with background jobs, pagination, and guardrails.

—

Implementation Overview
- Flags: introduce runtime flags to safely switch behavior on/off and tune limits.
- DB: add hot-path indexes; optional simple jobs table for status (or reuse run_* tables).
- Backend: upsert + enrich up to 10 records synchronously for immediate UX; enqueue remaining ICP‑matched records for the nightly runner; provide job status endpoints; adopt the batched/streaming upsert implementation; add keyset pagination utilities.
- Frontend: debounce + cancel in-flight; show preview; poll job status; virtualize long lists.
- Observability: structured logs + counters around batching and job lifecycle.

—

Environment Flags
- STAGING_UPSERT_MODE: `background` | `off` | `sync_preview` (default: `background`).
- UPSERT_SYNC_LIMIT: max rows allowed synchronously in request path (default: 10; 0 disables).
- UPSERT_MAX_PER_JOB: cap per background job (default: 2000).
- STAGING_BATCH_SIZE: server-side cursor batch size (default: 500).
- CHAT_DEBOUNCE_MS: recommended 300–500ms on the client.

—

Database Changes
1) Indexes (run CONCURRENTLY in production)
```sql
-- 010_icp_perf.sql
CREATE INDEX IF NOT EXISTS idx_companies_industry_norm_lower
  ON companies (LOWER(industry_norm));

CREATE INDEX IF NOT EXISTS idx_companies_name_lower
  ON companies (LOWER(name));

CREATE INDEX IF NOT EXISTS idx_companies_website_domain
  ON companies (website_domain);

-- Optional: if sorting by score frequently
CREATE INDEX IF NOT EXISTS idx_lead_scores_score_id
  ON lead_scores (score DESC, company_id DESC);
```

2) Optional lightweight jobs table (if not reusing run tables)
```sql
-- 011_background_jobs.sql
CREATE TABLE IF NOT EXISTS background_jobs (
  job_id      bigserial PRIMARY KEY,
  tenant_id   int,
  job_type    text NOT NULL,          -- e.g., 'staging_upsert'
  status      text NOT NULL,          -- queued | running | done | error
  created_at  timestamptz NOT NULL DEFAULT now(),
  started_at  timestamptz,
  ended_at    timestamptz,
  params      jsonb,
  processed   int DEFAULT 0,
  total       int DEFAULT 0,
  error       text
);
```

We can alternatively record per-batch stats in `run_stage_stats` with `stage='staging_upsert'` and keep a thin `background_jobs` row for lifecycle.

—

Backend Changes (FastAPI)
1) Normalize Input: upsert+enrich 10 synchronously; enqueue remainder for nightly *(legacy path retired; snippet kept for archival context.)*
```py
# app/main.py (excerpt)
STAGING_UPSERT_MODE = os.getenv("STAGING_UPSERT_MODE", "background").lower()
UPSERT_SYNC_LIMIT = int(os.getenv("UPSERT_SYNC_LIMIT", "10") or 10)

def _enqueue_staging_upsert(tenant_id: int | None, terms: list[str]) -> dict:
    from src.jobs import enqueue_staging_upsert
    return enqueue_staging_upsert(tenant_id, terms)

def normalize_input(payload: dict) -> dict:
    data = payload.get("input", payload) or {}
    msgs = data.get("messages") or []
    # ... existing message normalization ...
    state = {"messages": norm_msgs}

    try:
        inds = _collect_industry_terms(state.get("messages"))
        if inds:
            if STAGING_UPSERT_MODE == "off":
                pass
            elif STAGING_UPSERT_MODE == "sync_preview":
                # Resolve SSIC codes + fetch a tiny sample (<=20) for UX feedback only
                from src.icp import _find_ssic_codes_by_terms, _select_acra_by_ssic_codes
                codes = [c for c, _t, _s in _find_ssic_codes_by_terms([i.lower() for i in inds])]
                sample = []
                if codes:
                    import asyncio
                    sample = asyncio.run(_select_acra_by_ssic_codes(set(codes), 20))  # best-effort
                # attach sample suggestion (names/uen) to state for UI rendering
                state["candidates"] = [{"name": (r.get("entity_name") or "").strip(), "uen": r.get("uen")} for r in (sample or []) if r]
            else:  # background schedule with small synchronous head (10)
                # First, upsert + enrich up to UPSERT_SYNC_LIMIT records synchronously for immediate usability
                try:
                    from app.lg_entry import upsert_and_enrich_by_industries_head
                    upsert_and_enrich_by_industries_head(inds, limit=UPSERT_SYNC_LIMIT)
                except Exception:
                    logger.info("sync head upsert+enrich skipped/failed; continuing")
                # Resolve tenant and enqueue remainder for nightly runner (queued only)
                tid = None
                try:
                    from app.odoo_connection_info import get_odoo_connection_info
                    info = asyncio.run(get_odoo_connection_info(email=None, claim_tid=None))  # may return None
                    tid = info.get("tenant_id") if isinstance(info, dict) else None
                except Exception:
                    tid = None
                _enqueue_staging_upsert(tid, inds)  # queued; picked up by nightly worker
    except Exception as _e:
        logger.warning("input-normalization staging enqueue failed: %s", _e)
    return state
```

2) Job Enqueue + Nightly Worker
```py
# src/jobs.py (new helper module)
import os, asyncio, logging
from datetime import datetime
from typing import List, Optional
from src.database import get_conn
from app.lg_entry import _upsert_companies_from_staging_by_industries as upsert_batched

UPSERT_MAX_PER_JOB = int(os.getenv("UPSERT_MAX_PER_JOB", "2000") or 2000)
STAGING_BATCH_SIZE = int(os.getenv("STAGING_BATCH_SIZE", "500") or 500)
log = logging.getLogger("jobs")

def _insert_job(tenant_id: Optional[int], terms: List[str]) -> int:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO background_jobs(tenant_id, job_type, status, params) VALUES (%s,'staging_upsert','queued', %s) RETURNING job_id",
            (tenant_id, {"terms": terms}),
        )
        return int(cur.fetchone()[0])

def enqueue_staging_upsert(tenant_id: Optional[int], terms: List[str]) -> dict:
    job_id = _insert_job(tenant_id, terms)
    # Do not execute immediately; the nightly runner will pick this up.
    return {"job_id": job_id}

async def run_staging_upsert(job_id: int) -> None:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("UPDATE background_jobs SET status='running', started_at=now() WHERE job_id=%s", (job_id,))
    processed = 0
    total = 0
    terms: List[str] = []
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT params FROM background_jobs WHERE job_id=%s", (job_id,))
        row = cur.fetchone(); params = (row and row[0]) or {}
        terms = [((t or '').strip().lower()) for t in (params.get('terms') or []) if (t or '').strip()]
    try:
        # upsert_batched already streams and batches internally; we loop with a hard cap
        # (For simplicity, invoke once; inside it will process all matches up to its LIMIT/itersize.)
        processed = upsert_batched(terms)
        total = processed
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE background_jobs SET status='done', processed=%s, total=%s, ended_at=now() WHERE job_id=%s",
                (processed, total, job_id),
            )
    except Exception as e:
        log.exception("staging_upsert job failed: %s", e)
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE background_jobs SET status='error', error=%s, processed=%s, total=%s, ended_at=now() WHERE job_id=%s",
                (str(e), processed, total, job_id),
            )
```

3) Job Endpoints
```py
# app/main.py (add routes)
@app.post("/jobs/staging_upsert")
async def jobs_staging_upsert(body: dict, claims: dict = Depends(require_optional_identity)):
    terms = (body or {}).get("terms") or []
    if not isinstance(terms, list) or not terms:
        raise HTTPException(status_code=400, detail="terms[] required")
    # resolve tenant best-effort
    email = claims.get("email") or claims.get("preferred_username") or claims.get("sub")
    claim_tid = claims.get("tenant_id")
    from app.odoo_connection_info import get_odoo_connection_info
    info = await get_odoo_connection_info(email=email, claim_tid=claim_tid)
    from src.jobs import enqueue_staging_upsert
    res = enqueue_staging_upsert(info.get("tenant_id"), terms)
    return res

@app.get("/jobs/{job_id}")
async def jobs_status(job_id: int, _: dict = Depends(require_optional_identity)):
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT job_id, job_type, status, processed, total, error, created_at, started_at, ended_at FROM background_jobs WHERE job_id=%s", (job_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="job not found")
        cols = [d[0] for d in cur.description]
        return dict(zip(cols, row))
```

4) Keyset Pagination utilities (SQL examples)
```sql
-- latest scores (keyset)
-- Inputs: :limit, :after_score, :after_id (nullable)
SELECT c.company_id, c.name, c.industry_norm, c.employees_est,
       s.score, s.bucket, s.rationale
FROM companies c
JOIN lead_scores s ON s.company_id = c.company_id
WHERE (:after_score IS NULL OR (s.score, c.company_id) < (:after_score, :after_id))
ORDER BY s.score DESC, c.company_id DESC
LIMIT :limit;
```

—

Frontend Changes (agent-chat-ui)
1) Debounce + cancel on industry input
```ts
// hooks/useDebouncedFetch.ts
export function useDebouncedFetch(delay = 400) {
  const ctrlRef = useRef<AbortController | null>(null)
  const timeoutRef = useRef<any>()
  const run = (fn: () => Promise<any>) => {
    if (timeoutRef.current) clearTimeout(timeoutRef.current)
    if (ctrlRef.current) ctrlRef.current.abort()
    const ctrl = new AbortController()
    ctrlRef.current = ctrl
    return new Promise((resolve) => {
      timeoutRef.current = setTimeout(async () => {
        const res = await fn()
        resolve(res)
      }, delay)
    })
  }
  return { run }
}
```

2) Enqueue background upsert and poll job
```ts
// On submit with terms ['manufacturing']
const { job_id } = await authFetch(`${apiBase}/jobs/staging_upsert`, { method: 'POST', body: JSON.stringify({ terms }) })
let status
do {
  await new Promise(r => setTimeout(r, 1500))
  status = await authFetch(`${apiBase}/jobs/${job_id}`)
  setProgress({ processed: status.processed, total: status.total })
} while (status.status === 'queued' || status.status === 'running')
```

3) Virtualized table (react-window) for long lists
```tsx
<FixedSizeList height={600} itemCount={items.length} itemSize={48}>
  {Row}
</FixedSizeList>
```

—

User Journey & Program Flow
- User types “Manufacturing” in the chat.
- Frontend debounces and sends the message once stable; server:
  - Parses terms, resolves SSIC codes for preview (optional), upserts + enriches up to 10 records synchronously, enqueues the remainder for the nightly runner, and responds immediately.
- UI shows “Upserted and enriched 10 now; scheduled the rest for nightly processing…” with a small sample (≤ 20) if available.
- UI polls `/jobs/{id}` for the scheduled work; users can continue the conversation; no blocking.
- Once nightly upsert progresses, candidate selection and enrichment run as usual; shortlists/loaders are paginated and virtualized.

—

Observability
- Logs: `stage=staging_upsert` events for batch_start/batch_end/error with counts and `duration_ms`.
- Metrics (emit to logs first): processed/min, p95 batch time, job success/error counts, chat TTFB.

—

Testing & Acceptance
- Unit: terms parsing; job enqueue; job status DAO.
- Integration: sync head upsert+enrich (10) + enqueue remainder (seeded staging rows) → nightly job status updates.
- Performance: P95 TTFB under 300 ms for “Manufacturing” message with ≥ 1000 potential rows.
- Pagination: verify nextCursor correctness and no OFFSET usage on large sets.

—

Rollout Plan
- Phase 1: Ship indexes + flags; switch request path to `sync_preview` for safety; add `/jobs/*` endpoints; behind feature flag for selected tenants.
- Phase 2: Enable `background` mode; add UI polling + virtualization; tune caps.
- Phase 3: Adopt keyset pagination across list endpoints; finalize dashboards and alerts; remove legacy synchronous upsert path.
