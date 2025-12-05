# Development Plan — Bugfix: Return‑User Threads (DB‑Backed)

Aligns with: `Development_Plan/bugsfix_returnUserThread.md`, `Development_Plan/devplanPRD_returnUserThread.md`, and the embedded LangGraph orchestrator.

## Goals
- Replace in‑memory `_THREADS` with a tenant‑scoped Postgres `threads` table and API.
- Bind all graph runs to durable checkpoints via `configurable.thread_id` (existing `SqliteSaver`).
- Enforce single‑active thread per `(tenant_id,user_id,agent,context_key)` with locking/resume and auto‑archive.
- Keep return‑user node semantics unchanged; fix lifecycle via DB‑backed threads.

## Non‑Goals
- Moving business data (company/ICP snapshots, candidates, scores) out of Postgres.
- Changing LangGraph node order/logic beyond thread lifecycle.
- Building a separate threads microservice (server remains embedded in FastAPI).

## Architecture & Data Flow
- Threads metadata and lifecycle live in Postgres.
- Run checkpoints live in LangGraph’s durable checkpointer (SQLite at `.langgraph_api/orchestrator.sqlite`).
- API mediates thread creation/resume and invokes the graph with `{"configurable":{"thread_id": <uuid>}}`.
- Business writes (discovery candidates, scoring, exports) continue in Postgres tables as today.

## Database Migration
Create `app/migrations/00X_threads.sql`:

```sql
CREATE TABLE IF NOT EXISTS threads (
  id UUID PRIMARY KEY,
  tenant_id INT NULL,
  user_id TEXT NULL,
  agent TEXT NOT NULL DEFAULT 'icp_finder',
  context_key TEXT NOT NULL,
  label TEXT NULL,
  status TEXT NOT NULL DEFAULT 'open', -- 'open'|'locked'|'archived'
  locked_at TIMESTAMPTZ NULL,
  archived_at TIMESTAMPTZ NULL,
  reason TEXT NULL,
  last_updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_threads_tenant ON threads(tenant_id);
CREATE INDEX IF NOT EXISTS idx_threads_lookup ON threads(tenant_id, user_id, agent, context_key, status);
CREATE INDEX IF NOT EXISTS idx_threads_updated ON threads(last_updated_at DESC);

-- One open thread per context (strong safety)
CREATE UNIQUE INDEX IF NOT EXISTS uq_open_thread_per_context
  ON threads(tenant_id, user_id, agent, context_key)
  WHERE status = 'open';
```

Optional RLS (recommended; relies on GUC `request.tenant_id`):

```sql
ALTER TABLE threads ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_threads_read ON threads
  USING (tenant_id IS NULL OR tenant_id = current_setting('request.tenant_id', true)::int);
CREATE POLICY tenant_threads_write ON threads
  FOR UPDATE TO PUBLIC
  USING (tenant_id IS NULL OR tenant_id = current_setting('request.tenant_id', true)::int)
  WITH CHECK (tenant_id IS NULL OR tenant_id = current_setting('request.tenant_id', true)::int);
```

## Context Key (single source of truth)
Reuse `app/main.py::_context_key(payload, tenant_id)`: prefer `domain:<apex>` else `icp:<rule_name>#<payload_hash>`.

## API Design (FastAPI)
Add `app/threads_routes.py` and mount in `app/main.py`.

- POST `/threads` — create open thread (transactional)
  - Body: `{ input?: string, icp_payload?: object, label?: string }`
  - Steps:
    1) Compute `context_key` via `_context_key`.
    2) INSERT new row `{status:'open'}`; set `tenant_id`, `user_id`, `agent`.
    3) Lock prior open rows for the same `(tenant_id,user_id,agent,context_key)` → `{status:'locked', locked_at: now(), reason: 'new_thread_created'}`.
    4) Optionally auto‑archive stale locked rows.
    5) Return `{ id, status, context_key, label }`.

- POST `/threads/{id}/resume` — resume a specific open thread
  - 409 if `status!='open'` → `{ error: 'thread_locked' }`.
  - Optional `input` to immediately invoke orchestrator run.

- POST `/threads/resume-eligible` — resolve auto‑resume
  - Body: `{ input?: string, icp_payload?: object }` (used to compute `context_key`).
  - Query open threads for `(tenant_id,user_id,agent,context_key)` within `THREAD_RESUME_WINDOW_DAYS`.
  - Exactly one → return `{ auto_resumed: true, thread }`.
  - Multiple → return top 2–3 candidates with labels for UI disambiguation.
  - None → create new via POST `/threads` and return it.

- GET `/threads` — list threads (open by default; `show_archived` toggle).
- GET `/threads/{id}` — fetch thread metadata.

Auth: require JWT; middleware sets `request.state.tenant_id` and DB GUC `request.tenant_id`. RLS enforces isolation.

## Orchestrator Integration
- Graph built via `build_orchestrator_graph()` (already uses `SqliteSaver`).
- Always invoke with `{"configurable":{"thread_id": <uuid>}}`.
- `/api/orchestrations` updates:
  - Resolve or create a thread (using the DB endpoints/path) before invoking the graph.
  - Reject runs to non‑`open` threads with 409.
  - Update `threads.last_updated_at = now()` when runs complete/stream ends.

## Thread Policy & Logic
- Single‑active‑per‑context: thread creation locks prior open threads for the same `(tenant_id,user_id,agent,context_key)` atomically.
- Auto‑resume: if one eligible open thread within `THREAD_RESUME_WINDOW_DAYS` → resume; multiple → ask user to choose; none → create.
- Read‑only: resuming or writing to non‑`open` threads returns 409 (`thread_locked`).
- Auto‑archive: when `AUTO_ARCHIVE_STALE_LOCKED=true`, move locked threads older than `THREAD_STALE_DAYS` to `archived`.

## Feature Flags & Settings
Add to `src/settings.py`:
- `USE_DB_THREADS=true`
- `THREAD_RESUME_WINDOW_DAYS=7`
- `THREAD_STALE_DAYS=30`
- `AUTO_ARCHIVE_STALE_LOCKED=true`
- `MAX_OPEN_THREADS_PER_AGENT=1`
- Existing: `ENABLE_EMBEDDED_LG_SERVER`, `LANGGRAPH_CHECKPOINT_DIR`

## Implementation Tasks (by file)
- DB
  - Add migration `app/migrations/00X_threads.sql` (table, indexes, RLS, unique index).
- API/Server
  - Add `app/threads_routes.py` with the endpoints above.
  - Wire router in `app/main.py`; ensure tenant GUC is set each request.
  - Update `/api/orchestrations` to resolve/create thread in DB and pass `configurable.thread_id`.
  - Deprecate `_THREADS` and `.langgraph_api/threads.json`; keep dev fallback behind `USE_DB_THREADS=false`.
- Settings
  - Add flags and defaults in `src/settings.py`.
- Optional Dev Backfill
  - On dev boot, import legacy `.langgraph_api/threads.json` into DB then remove the file.

## Pseudocode (critical paths)

Create thread (transaction):
```python
with db.tx() as cur:
  context_key = _context_key(payload, tenant_id)
  tid = uuid4()
  cur.execute("""
    INSERT INTO threads(id, tenant_id, user_id, agent, context_key, label, status)
    VALUES ($1,$2,$3,'icp_finder',$4,$5,'open')
  """, (tid, tenant_id, user_id, context_key, label))
  cur.execute("""
    UPDATE threads
       SET status='locked', locked_at=now(), reason='new_thread_created'
     WHERE tenant_id=$1 AND user_id=$2 AND agent='icp_finder'
       AND context_key=$3 AND status='open' AND id<>$4
  """, (tenant_id, user_id, context_key, tid))
  if settings.AUTO_ARCHIVE_STALE_LOCKED:
    cur.execute("""
      UPDATE threads
         SET status='archived', archived_at=now()
       WHERE tenant_id=$1 AND status='locked'
         AND last_updated_at < now() - make_interval(days => $2)
    """, (tenant_id, settings.THREAD_STALE_DAYS))
```

Resume‑eligible:
```python
rows = cur.fetch("""
  SELECT id, label, last_updated_at
    FROM threads
   WHERE tenant_id=$1 AND user_id=$2 AND agent='icp_finder'
     AND context_key=$3 AND status='open'
     AND last_updated_at >= now() - make_interval(days => $4)
   ORDER BY last_updated_at DESC
   LIMIT 3
""", (tenant_id, user_id, context_key, settings.THREAD_RESUME_WINDOW_DAYS))
```

## Testing Plan
- Unit
  - `_context_key` (domain vs ICP hash cases).
  - SQL helpers for create/lock/archive transactions.
- API
  - POST `/threads` creates open thread; locks prior; returns metadata.
  - POST `/threads/{id}/resume` → 409 when status != open.
  - POST `/threads/resume-eligible` → none/one/multiple cases.
  - GET `/threads` respects tenant and `show_archived`.
- RLS/Security
  - Enforce tenant isolation via `request.tenant_id`.
  - Cross‑tenant access denied; admin override (if applicable) works.
- Concurrency
  - Parallel creates for same context: unique index prevents duplicates; one succeeds, others fail or see locked prior.
- Orchestrator integration
  - Run with `thread_id` persists checkpoints; second run resumes.
  - Non‑open thread run attempt returns 409 from wrapper.
- Fallback
  - With `USE_DB_THREADS=false`, legacy in‑memory path works in dev.

## Observability
- Structured logs for lifecycle: create → lock(priors) → resume → archive → run start/complete.
- Metrics: count open/locked/archived; auto‑resume hit rate; 409 occurrences.
- Correlate API and graph with a shared trace/request id.

## Risks & Mitigations
- Concurrency races on create/lock → single transaction + partial unique index on open threads.
- Staleness thresholds vs UX → env‑driven; log decisions for tuning.
- Dual backends confusion → `USE_DB_THREADS` flag; clear startup log of active backend.

## Rollout Plan
1) Add migration; apply in staging.
2) Implement routes; enable `USE_DB_THREADS=true` in staging; verify E2E.
3) Switch `/api/orchestrations` to DB threads; monitor logs/metrics.
4) Enable in production; keep legacy fallback disabled by default; remove later.

## Acceptance Criteria
- Creating a thread locks prior open threads for same context in a single transaction.
- Resuming a non‑open thread returns 409; open threads resume with checkpoint continuity.
- Auto‑resume: exactly one eligible → resume; multiple → list options; none → create.
- RLS prevents cross‑tenant access; admin behavior per policy.
- Observability shows correct lifecycle; no duplicate open threads per context.

## Example Requests (manual QA)
- Create
  ```bash
  curl -s -X POST http://localhost:2024/threads \
    -H 'Authorization: Bearer dev' -H 'X-Tenant-ID: 1' \
    -H 'Content-Type: application/json' \
    -d '{"input":"our site https://acme.ai"}'
  ```
- Auto‑resume
  ```bash
  curl -s -X POST http://localhost:2024/threads/resume-eligible \
    -H 'Authorization: Bearer dev' -H 'X-Tenant-ID: 1' \
    -H 'Content-Type: application/json' \
    -d '{"input":"our site https://acme.ai"}'
  ```
- Locked resume attempt
  ```bash
  curl -s -X POST http://localhost:2024/threads/<old_id>/resume \
    -H 'Authorization: Bearer dev' -H 'X-Tenant-ID: 1'
  ```
- Orchestrator run
  ```bash
  curl -s -X POST http://localhost:2024/api/orchestrations \
    -H 'Authorization: Bearer dev' -H 'X-Tenant-ID: 1' \
    -H 'Content-Type: application/json' \
    -d '{"thread_id":"<uuid>","input":"start discovery"}'
  ```

