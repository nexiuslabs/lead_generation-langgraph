# Feature PRD — Mid‑Run ICP Update and Cancellation

## Purpose
- Objective: Improve UX when a user changes the ICP while a discovery/enrichment run is active. The agent should detect the intent, ask whether to cancel the current run and apply the change now, or keep running and apply later.
- Goals:
  - Provide clear, deterministic conversational gating for mid‑run ICP updates.
  - Add cooperative cancellation to background jobs (soft cancel) with proper observability.
  - Maintain single‑active‑thread per context and avoid duplicate/overlapping runs.

## Scope
- In‑scope:
  - Orchestrator node `run_guard` (after `profile_builder`, before `journey_guard`).
  - Intent refinement for `update_icp` and cancel/keep confirmations.
  - DB support for cooperative cancellation (`background_jobs.cancel_requested`, `canceled_at`).
  - Worker support for cancel checks inside `run_icp_discovery_enrich`.
  - Minimal API endpoints to request cancellation.
  - Status surfacing in `progress_report`/`summary` for cancel states.
  - Adopted choices: soft cancel policy; cancel scope limited to `icp_discovery_enrich`; public cancel API only via `POST /jobs/{job_id}/cancel`.
- Out‑of‑scope:
  - Hard kill of external vendors (cancels are cooperative, not force kill of network calls).
  - Broad cancel across all job types (restricted to `icp_discovery_enrich` initially).
  - Multi‑agent meltdown recovery beyond this flow.

## Architecture & Integration
- Orchestrator (LangGraph):
  - Graph order: `ingest` → `return_user_probe` → `profile_builder` → `run_guard` → `journey_guard` → `normalize` → `refresh_icp` → `decide_strategy` → `ssic_fallback` → `progress_report` → `finalize`.
  - New node `run_guard(state)`:
    - Reads tenant context and checks `background_jobs` for active `icp_discovery_enrich` (queued|running).
    - If last intent is `update_icp` and an active job exists:
      - Set `state.run = { active_job_id, status: 'running', awaiting_cancel_confirmation: true }`.
      - Emit prompt asking to cancel and update now vs keep running.
      - Branch to `progress_report` (pending).
    - If waiting for confirmation and the user replies:
      - “cancel and update” → request cancel via API/DB, set `status='pending_cancel'`, clear discovery cache, set `icp_profile_confirmed=false`, branch back to `profile_builder`.
      - “keep running” → clear `awaiting_cancel_confirmation`, optionally stash ICP draft, continue to `journey_guard`.
- State additions (`my_agent/utils/state.py`):
  - `run: { active_job_id?: int, status?: 'idle'|'running'|'pending_cancel'|'cancelled', awaiting_cancel_confirmation?: bool }`.
- Intent refinement (`ingest_message`):
  - Map update phrases mentioning ICP/target/persona into `update_icp`.
  - Binary confirmation for cancel vs keep: 
    - Cancel: {"cancel and update", "cancel", "stop run", "abort"}.
    - Keep: {"keep running", "continue", "don’t cancel"}.
- Gating (`journey_guard`):
  - If `run.status in {'pending_cancel','running'}` and `awaiting_cancel_confirmation`, avoid enqueueing; surface pending prompt instead.
- Progress reporting (`progress_report`/`finalize`):
  - If `pending_cancel`, poll job status; announce `cancel_request` received and later `cancelled`.
- DB migration:
  - Add columns to `background_jobs`: `cancel_requested boolean default false`, `canceled_at timestamptz NULL`.
- Worker (`scripts/run_bg_worker.py` + `src/jobs.py`):
  - In `run_icp_discovery_enrich(job_id)`, check `cancel_requested` at stage boundaries and per-company loop.
  - On cancel: mark `status='cancelled'`, set `canceled_at=now()`, `ended_at=now()`, log and exit.
- API routes (`app/main.py`):
  - POST `/jobs/{job_id}/cancel` → sets `cancel_requested=true` (admin/owner only).
  - POST `/jobs/cancel_current?tenant_id=…` → resolves latest active `icp_discovery_enrich` and sets the flag.
- Interop with return‑user logic:
  - `return_user_probe` remains unchanged. `run_guard` executes after profile hydration.

## User Stories
- As a user, while a run is in progress, I can change ICP and be asked whether to cancel the current run to apply the change immediately.
- As a user, if I keep the current run, my new ICP is preserved as a draft and suggested later.
- As an admin, I can cancel the currently running discovery/enrichment job for a tenant.

## User Flows
1) No Active Job (Simple Update)
- User: “Change ICP to HR tech in APAC.”
- `run_guard`: no active job → pass through; `profile_builder` applies/update ICP; `journey_guard` proceeds normally.

2) Active Job Detected → Ask to Cancel or Keep
- User: “Change ICP to HR tech in APAC.”
- `run_guard`: finds active job; sets `awaiting_cancel_confirmation=true`; assistant asks using templated copy (see below): “A job (ID 123) is running. Cancel and update now, or keep it running?”

3) Cancel and Update
- User: “Cancel and update.”
- API: `/jobs/{id}/cancel` → `cancel_requested=true`.
- Orchestrator: `run.status='pending_cancel'`; clears discovery cache, sets `icp_profile_confirmed=false`, prompts for ICP confirm; returns to `profile_builder`.
- Worker: within ≤1 iteration, marks `status='cancelled'` and exits.
- Assistant: announces “Cancelled job 123. Updated ICP ready — continue to discovery?”

4) Keep Running
- User: “Keep running.”
- Orchestrator: clear `awaiting_cancel_confirmation`; optionally stash ICP draft; continue with `journey_guard`.
- Assistant: “Okay, I’ll keep this run going. I’ll apply your ICP change after it finishes.”

5) Race: Job Finishes Before Decision
- If job finishes during prompt: assistant updates “The job just finished. Apply your new ICP and start a new run?”

6) Retry/Resume
- If a cancel is requested but job already finished, assistant reflects final status and proceeds to apply the ICP update without cancellation.

## Decisions & Policy
- Signals:
  - Active job exists: `background_jobs where job_type='icp_discovery_enrich' and status in ('queued','running')`.
  - Update intent: “update icp”, “change target”, “revise ideal customer”, etc.
- Outcomes:
  - `cancel_and_update`: request cancel, reset ICP confirmation, clear discovery cache.
  - `keep_running`: continue current run; stash ICP draft.
- Cancellation policy: soft cancel only (finish current item, then stop). `CANCEL_POLICY=soft` is enforced for this feature.
- Thread policy: single‑active‑thread per context unchanged; `run_guard` complements `featurePRD_returnUserThread` by gating mid‑run ICP edits.

### Prompt templates (assistant copy)
- Cancel/keep prompt: "A discovery/enrichment job (ID {{job_id}}) is currently {{status}}. Do you want to cancel it and apply your ICP changes now, or keep it running?"
- Cancel acknowledged: "Cancel request sent for job {{job_id}}. I’ll stop the run safely and apply your ICP update next."
- Cancel complete: "Job {{job_id}} was cancelled. Your updated ICP is ready — shall I proceed with discovery?"
- Keep running: "Okay — I’ll keep job {{job_id}} running. I’ve saved your ICP changes and will apply them after this run finishes."

## APIs & Server Hooks
- Public:
  - POST `/jobs/{job_id}/cancel` → body `{ reason?: string }`
    - Auth: admin or tenant owner.
    - Response `{ ok: true, job_id, status: 'pending_cancel' }`.
  - GET `/jobs/{job_id}` → include `cancel_requested`, `canceled_at` in response (additive fields).
- Internal (server‑only helper):
  - `request_cancel_current(tenant_id: int)` resolves the latest active `icp_discovery_enrich` and sets `cancel_requested=true`.

## Configuration
- `CANCEL_POLICY=soft` (enforced for this feature; hard cancel not exposed).
- `CANCEL_POLL_INTERVAL_MS=500` (runner loop check cadence; soft bound, actual checks at stage boundaries).
- Existing: `SINGLE_THREAD_PER_CONTEXT`, `THREAD_RESUME_WINDOW_DAYS`, `BG_JOB_STALE_MINUTES`, etc.

## Acceptance Criteria
- When user proposes an ICP change and an active job exists, assistant asks to cancel or keep.
- “Cancel and update” sets `cancel_requested=true`, the job stops within an iteration, and status reads `cancelled`.
- After cancellation, ICP confirmation resets, discovery cache clears, and the assistant prompts to confirm the new ICP.
- “Keep running” preserves the current run and stashes ICP updates for later without enqueueing a conflicting job.
- `progress_report` reflects pending cancel and final cancelled states using the templates above.
- No duplicate jobs are enqueued while a job is `queued|running|pending_cancel`.
- Only the public endpoint `POST /jobs/{job_id}/cancel` is exposed; no `cancel_current` public route.

## Non‑Functional Requirements
- Reliability: cancel does not corrupt partial results; soft cancel only at safe boundaries.
- Observability: structured logs for `cancel_request`, `cancel_ack`, `cancelled`, and user decisions.
- Security: endpoints require auth; enforce tenant isolation.
- Performance: additional cancel checks do not materially slow the enrichment loop.

## Testing Strategy
- Unit:
  - Intent classification for `update_icp` and yes/no decisions.
  - State transitions for `run` sub‑state (awaiting → pending_cancel → cancelled).
- Integration:
  - `run_guard` behavior with and without active jobs.
  - API cancel request flips runner within ≤1 loop.
  - `progress_report` messages across pending and final states.
- E2E:
  - Mid‑run ICP update → cancel and update → new ICP confirmed → discovery and enrichment resume.
  - Mid‑run ICP update → keep running → ICP draft applied post‑run.

## Risks & Mitigations
- Cancel latency (long step): keep soft cancel and communicate “cancel pending” status in chat.
- Race conditions: handle finished job during prompt; re‑query job status before acting.
- Partial state writes: only cancel between safe boundaries; mark job `cancelled` to distinguish from `error`.

## Rollout Plan
- Phase 1: implement `run_guard`, intent tweaks, API cancel endpoints, runner cancel checks; add DB columns; ship behind feature flag if needed.
- Phase 2: FE affordances (cancel button, status badges), richer metrics.
- Phase 3: expand to other job types if required.

## Migrations
- SQL (additive):
  - `ALTER TABLE background_jobs ADD COLUMN IF NOT EXISTS cancel_requested boolean DEFAULT false;`
  - `ALTER TABLE background_jobs ADD COLUMN IF NOT EXISTS canceled_at timestamptz NULL;`

## Tenancy & Auth
- Validate Nexius SSO JWT.
- Authorization: platform admin or tenant owner/admin roles (e.g., `admin`, `tenant_admin`).
- Apply RLS and per‑tenant filters on `background_jobs` reads.

## Observability
- Emit `run_event_logs` entries for cancel lifecycle and user decisions.
- Include cancel state in `/progress_report` messages and `GET /jobs/{id}`.

## Dependencies
- None new; extends `my_agent` orchestrator, `app/main.py`, `src/jobs.py`, `scripts/run_bg_worker.py`.

## Backward Compatibility
- Additive DB columns and endpoints are backward compatible; previous flows unaffected when no active job or no update intent.

## Open Questions — Resolved
- Cancel scope: Limit to `icp_discovery_enrich` only for this feature.  
  - Rationale: Tenant‑wide nightly jobs (`staging_upsert`/`enrich_candidates`) are scheduled and multi‑tenant in nature; cancelling them from chat risks surprise cross‑team effects. We’ll evaluate broader cancel in a later iteration.
- Cancel mode: Soft cancel only (finish current item, then stop).  
  - Rationale: Ensures DB consistency and avoids half‑written items during enrichment. Hard kill may leave inconsistent vendor state and partial records; we can revisit once we have idempotent checkpoints per step.

## TODOs — Adopted Plan (decisions + justification)
- PM (due T+2d): Implement templated, contextual assistant copy for prompts (see templates above).  
  - Justification: Clear guidance, consistent tone, and easy localization/testing.
- Backend (due T+3d): Expose only `POST /jobs/{job_id}/cancel` as a public endpoint; add internal helper `request_cancel_current(tenant_id)` for server flows.  
  - Justification: Minimal public surface and least privilege; avoids misuse and race conditions.
- Security (due T+4d): Enforce authorization for cancel endpoint: platform `admin` or tenant‑scoped owner/admin (e.g., `tenant_admin`).  
  - Justification: Cancelling impacts shared tenant outcomes; restrict to responsible roles.
- QA (due T+6d): Add API tests and orchestrator integration tests; include one targeted E2E (worker + DB) for cooperative cancel.  
  - Justification: High value coverage with manageable CI complexity; validates end‑to‑end behavior.
