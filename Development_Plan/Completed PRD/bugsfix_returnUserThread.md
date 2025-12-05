# Bugfix/Implementation Plan — Tenant‑Scoped Threads for Return‑User Flow

Aligns with: Development_Plan/featurePRD_returnUserThread.md and current embedded LangGraph orchestrator.

## Summary
- Replace the in‑memory thread registry used by `/api/orchestrations` with a tenant‑scoped `threads` table and FastAPI endpoints.
- Bind every graph run to a durable checkpoint via `configurable.thread_id` (SqliteSaver already wired in `my_agent/agent.py`).
- Enforce single‑active‑thread per (tenant_id, user_id, agent, context_key) at create/resume time; auto‑lock prior open threads; optionally auto‑archive stale locked.

Rationale: Threads become the durable anchor for checkpointed graph state (via LangGraph checkpointer). The server controls the business policy (context_key, tenant filtering, locking) atomically in SQL.

## Current State (Codebase)
- `app/main.py` exposes `/api/orchestrations` which currently uses an in‑memory map `_THREADS` persisted to `.langgraph_api/threads.json` to approximate threads creation/locking.
- Orchestrator (`my_agent/agent.py`) compiles the LangGraph with a durable checkpointer (`SqliteSaver`) bound to `.langgraph_api/orchestrator.sqlite` and already passes `configurable.thread_id`.
- Return‑user gating and graph order are implemented: `ingest → return_user_probe → profile_builder → journey_guard → ...`.

Gap: `_THREADS` needs to be replaced with a DB‑backed `threads` table and endpoints so policies survive process restarts and support multi‑instance deployments.

## DB Schema (Postgres)
Create an additive migration `app/migrations/00X_threads.sql`:

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

-- Helpful indexes
CREATE INDEX IF NOT EXISTS idx_threads_tenant ON threads(tenant_id);
CREATE INDEX IF NOT EXISTS idx_threads_lookup ON threads(tenant_id, user_id, agent, context_key, status);
CREATE INDEX IF NOT EXISTS idx_threads_updated ON threads(last_updated_at DESC);
```

Optional RLS (recommended; mirrors existing `request.tenant_id` usage):

```sql
ALTER TABLE threads ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_threads_read ON threads
  USING (tenant_id IS NULL OR tenant_id = current_setting('request.tenant_id', true)::int);
CREATE POLICY tenant_threads_write ON threads
  FOR UPDATE TO PUBLIC
  USING (tenant_id IS NULL OR tenant_id = current_setting('request.tenant_id', true)::int)
  WITH CHECK (tenant_id IS NULL OR tenant_id = current_setting('request.tenant_id', true)::int);
```

Note: Set `request.tenant_id` per request as we already do in multiple DB helpers.

## Context Key (reuse existing logic)
Use `app/main.py::_context_key(payload, tenant_id)` which prefers `domain:<apex>` else `icp:<rule_name>#<payload_hash>`. Keep that single source of truth to prevent drift.

## FastAPI Endpoints (new)
Add a small router (e.g., `app/threads_routes.py`) with tenant‑scoped endpoints that wrap the embedded graph server calls:

- POST `/threads` → create a new open thread
  - Body: `{ input?: string, icp_payload?: object, label?: string }`
  - Steps (transaction):
    1) Compute `context_key` via `_context_key`.
    2) INSERT new row with `status='open'`, label (e.g., website/rule), user_id from JWT, tenant_id from request.
    3) UPDATE prior open rows for same `(tenant_id,user_id,agent,context_key)` → `status='locked', locked_at=now(), reason='new_thread_created'`.
    4) Return `{ id, status, context_key }`.

- POST `/threads/{id}/resume` → resume a specific open thread
  - 409 if `status!='open'` with `{ error:'thread_locked' }`.
  - Optionally allow an `input` payload to directly invoke the orchestrator.

- POST `/threads/resume-eligible` → server‑side resolver for auto‑resume
  - Input: `{ input?: string, icp_payload?: object }` (used to compute `context_key`).
  - Query candidates for same `(tenant_id,user_id,agent,context_key)` with `status='open'` and `last_updated_at >= now() - 'THREAD_RESUME_WINDOW_DAYS'`.
  - If exactly one candidate → return it and a hint `{ auto_resumed: true }`.
  - If multiple → return `2–3` labeled candidates for disambiguation.
  - If none → create a new thread (calls POST `/threads`).

- GET `/threads` → list threads for current tenant/user (open by default; `show_archived` toggle).
- GET `/threads/{id}` → fetch thread metadata.

Authorization: `require_auth` (existing) — admin or same‑tenant users; rely on RLS for isolation.

## Orchestrator Invocation (unchanged mechanics)
- Build graph via `build_orchestrator_graph()` (already uses `SqliteSaver`).
- When invoking, always pass config: `{"configurable": {"thread_id": <uuid>}}` to bind checkpoints to the thread.
- Use the existing `/api/orchestrations` handler as a thin wrapper that, under the hood, first resolves/creates a thread id using the new endpoints/DB, then routes to the graph.

Adjustment: Deprecate `_THREADS` + `.langgraph_api/threads.json`. Keep as dev fallback if DB is unreachable (best‑effort).

## Server Flow Details

Create (atomic):
1) Compute `context_key`.
2) INSERT row (`status=open`).
3) Lock prior open rows for same key.
4) Kick an initial orchestrator run with `configurable.thread_id` (optional) to seed `return_user_probe`.

Resume logic:
- With `thread_id`: resume or 409 if not open.
- Without `thread_id`:
  - Query eligible open candidates by `(tenant_id,user_id,agent,context_key)` and `last_updated_at >= now() - THREAD_RESUME_WINDOW_DAYS`.
  - Exactly one → auto‑resume.
  - Multiple → return top 2–3 by `last_updated_at` with labels for UI disambiguation.
  - None → create.

Auto‑archive:
- On create, after locking prior open rows, also move stale locked rows to `archived` when `AUTO_ARCHIVE_STALE_LOCKED=true` and `last_updated_at < now() - THREAD_STALE_DAYS`.

## Return‑User Nodes
- `return_user_probe(state)` (already present): 
  - Loads persisted company/ICP snapshots and computes decisions `{ use_cached, rerun_icp, reason }`.
  - Compatible with thread checkpointing; idempotent.
- `journey_guard` honors `decisions`: 
  - `use_cached` → set discovery ready; skip refresh.
  - `rerun_icp` → ask “Re‑run discovery due to X?”; proceed on explicit confirmation.

No changes to node semantics are required; only the thread lifecycle and resume now use DB.

## UI/UX Alignment
- When API returns `status='locked'` for a thread, UI shows Read‑only badge and disables composer; POST to runs returns 409.
- “Start new ICP session” calls POST `/threads` (auto‑locks prior open threads for the same context), then routes to the new thread.
- If `/threads/resume-eligible` returns multiple, UI shows a compact chooser (2–3 labeled candidates).

## Enrichment & Idempotency
- Continue to use `icp_hash + rule_name` as the idempotency fingerprint for background jobs (already implemented in `src/jobs.enqueue_icp_discovery_enrich`).
- Top‑N inline vs enqueue remaining: unchanged. Threads do not duplicate jobs because `journey_guard` gates enqueue and the job layer dedupes by fingerprint.

## Example SQL — resume candidate query
```sql
SELECT id, label, last_updated_at
FROM threads
WHERE tenant_id = $1
  AND user_id = $2
  AND agent = 'icp_finder'
  AND context_key = $3
  AND status = 'open'
  AND last_updated_at >= now() - make_interval(days => $4)
ORDER BY last_updated_at DESC
LIMIT 3;
```

## Operational Settings
- `THREAD_RESUME_WINDOW_DAYS` (default 7)
- `THREAD_STALE_DAYS` (default 30)
- `AUTO_ARCHIVE_STALE_LOCKED` (default true)
- `MAX_OPEN_THREADS_PER_AGENT=1` (enforced per context_key)

## Security & RLS
- Require JWT; set `request.tenant_id` GUC per request for DB session.
- Enable RLS on `threads`; add read/update policies keyed by `request.tenant_id`.
- Admin override (admin roles) may list cross‑tenant with explicit intent (optional admin endpoints).

## API Changes in Codebase
- Add `app/threads_routes.py` and mount in `app/main.py`.
- Update `/api/orchestrations` to consult/create `threads` in DB instead of `_THREADS`.
- Keep `/api/orchestrations/{thread_id}` unchanged (reads checkpoints via LangGraph state snapshot).

## Migration & Backfill
- Run migration `00X_threads.sql`.
- Optional: on boot in dev, import legacy `.langgraph_api/threads.json` into `threads` (best‑effort) and then delete the file.

## Testing (manual quickstart)
- Create new thread (no thread_id):
  - `curl -s -X POST /threads -H 'Authorization: Bearer dev' -H 'X-Tenant-ID: 1' -H 'Content-Type: application/json' -d '{"input":"our site https://acme.ai"}'`
- Auto‑resume (same context, within window):
  - `curl -s -X POST /threads/resume-eligible -H 'Authorization: Bearer dev' -H 'X-Tenant-ID: 1' -H 'Content-Type: application/json' -d '{"input":"our site https://acme.ai"}'`
- Locked thread attempt:
  - After creating a new thread for same context, POST `/threads/{old}/resume` should 409 with `{error:'thread_locked'}`.
- Orchestrator run:
  - POST `/api/orchestrations` with `{thread_id: <id>, input:"start discovery"}` should reuse the thread and the checkpoint.

## Risks
- Partial adoption where `_THREADS` and DB both exist: mitigate by feature‑flagging the DB route and deprecating `_THREADS` after verifying DB health.
- Concurrency: ensure create + prior‑lock run inside a single transaction; use row‑level UPDATE target filters.
- Staleness thresholds not aligned with UX: expose day values via env and log decisions for tuning.

## Rollout
1) Ship DB migration.
2) Implement endpoints + switch `/api/orchestrations` to DB under a feature flag; verify in staging.
3) Enable in production; leave legacy fallback disabled by default.

