# TODO — Mid‑Run ICP Update and Cancellation

Source: `lead_generation-main/Development_Plan/devplan_midRunIcpUpdate_and_Cancel.md`

## Orchestrator (LangGraph)
- [x] Add node `run_guard` in `my_agent/agent.py` after `profile_builder` and before `journey_guard`.
- [x] Wire edges: `ingest` → `return_user_probe` → `profile_builder` → `run_guard` → `journey_guard`.
- [x] Extend state typing in `my_agent/utils/state.py` with `run: TypedDict(total=False)` keys:
  - `active_job_id: int | None`
  - `status: Literal['idle','running','pending_cancel','cancelled']`
  - `awaiting_cancel_confirmation: bool`

## Intent & Prompts
- [x] Extend `ingest_message` in `my_agent/utils/nodes.py` to detect:
  - Update phrases → `update_icp`.
  - Confirmations when awaiting cancel: yes → `cancel_and_update`, no → `keep_running`.
- [x] Add prompt templates for `run_guard`/`progress_report`:
  - Cancel/keep prompt, cancel acknowledged, cancel complete, keep running.

## New Node: `run_guard`
- [x] Implement `run_guard(state)` in `my_agent/utils/nodes.py`:
  - [x] Resolve `tenant_id` from state/context.
  - [x] Lookup active job:
        `SELECT job_id, status FROM background_jobs WHERE tenant_id=%s AND job_type='icp_discovery_enrich' AND status IN ('queued','running') ORDER BY job_id DESC LIMIT 1`.
  - [x] If no active job → pass through to `journey_guard`.
  - [x] If active job and last intent `update_icp`:
        set `state.run={active_job_id, status:'running', awaiting_cancel_confirmation:true}`;
        append cancel/keep prompt; branch to `progress_report`.
  - [x] If `awaiting_cancel_confirmation`:
        - [x] On `cancel_and_update`: call `request_cancel(job_id)`; set `status='pending_cancel'`;
              reset ICP readiness (`profile_state.icp_profile_confirmed=false`) and clear discovery cache;
              append “cancel acknowledged”; return pending for `progress_report`.
        - [x] On `keep_running`: clear `awaiting_cancel_confirmation`; preserve any ICP draft; proceed to `journey_guard`.

## Gating & Status Surfaces
- [x] Update `journey_guard` to gate when `awaiting_cancel_confirmation` or `status='pending_cancel'`; branch to `progress_report`.
- [x] Update `progress_report` to poll `background_jobs` and render cancel states:
  - [x] If `status='cancelled'`: append “cancel complete” and proceed to profile update path.
  - [x] Else: surface “cancel acknowledged” + job status.

## API & Server Helpers
- [x] In `app/main.py`, add `POST /jobs/{job_id}/cancel` (admin or tenant owner):
      sets `cancel_requested=true`; returns `{ ok, job_id, status: 'pending_cancel' }`.
- [x] Ensure `GET /jobs/{job_id}` includes `cancel_requested`, `canceled_at` (additive change).
- [x] In `src/jobs.py`, add helpers:
  - [x] `request_cancel(job_id: int) -> dict` — set `cancel_requested=true` if status in ('queued','running').
  - [x] `request_cancel_current(tenant_id: int) -> dict` — best‑effort resolve latest active `icp_discovery_enrich` for tenant and set flag.

## Worker & Runner (Cooperative Cancel)
- [x] In `src/jobs.py` (`run_icp_discovery_enrich`):
  - [x] Implement `_should_cancel(job_id)` reading `background_jobs.cancel_requested`.
  - [x] Insert checks:
        after marking `running`, after discovery (pre‑enrichment), and each company iteration.
  - [x] On cancel: set `status='cancelled'`, `canceled_at=now()`, `ended_at=now()`; log `cancelled`; exit.

## Data & Migrations
- [x] Create `app/migrations/00X_add_cancel_flags_to_background_jobs.sql`:
  - [x] `ALTER TABLE background_jobs ADD COLUMN IF NOT EXISTS cancel_requested boolean DEFAULT false;`
  - [x] `ALTER TABLE background_jobs ADD COLUMN IF NOT EXISTS canceled_at timestamptz NULL;`

## Configuration
- [ ] Add `CANCEL_POLL_INTERVAL_MS=500` (poll cadence for `progress_report`).
- [x] Keep cancel policy as soft cancel; do not expose env switch.

## Observability
- [x] Emit structured logs: `cancel_request` (API), `cancel_ack` (orchestrator), `cancelled` (runner), including `tenant_id`, `job_id`.
- [x] Include cancel state in `progress_report` status and `GET /jobs/{job_id}` response.

## Documentation
- [x] Update README/ICPs docs to mention cancel capability and flow.

## Testing
- [ ] Acceptance (conversational):
  - [ ] Mid‑run ICP change triggers cancel/keep prompt.
  - [ ] Choosing cancel → `pending_cancel`, clears discovery cache, resets ICP confirmation.
  - [ ] Choosing keep → proceed without enqueueing; draft ICP preserved and applied post‑run.
- [ ] Unit:
  - [ ] Intent parsing maps phrases to `update_icp`, `cancel_and_update`, `keep_running`.
  - [ ] `run_guard` state transitions: `awaiting_cancel_confirmation` → `pending_cancel` → `cancelled`.
  - [x] Helpers: `request_cancel(job_id)`, `_should_cancel(job_id)` behavior.
- [ ] API:
  - [ ] `POST /jobs/{job_id}/cancel`: happy path, role gating, invalid job/status.
  - [ ] `GET /jobs/{job_id}` includes `cancel_requested`, `canceled_at`.
- [ ] Worker:
  - [ ] Simulate loop with injected `_should_cancel(job_id)`; verify cooperative stop sets `cancelled`, `ended_at`, `canceled_at`.
  - [ ] Boundary checks before next iteration.
- [ ] Migration:
  - [ ] Columns exist; defaults set; existing rows unaffected.
- [ ] Prompt snapshots:
  - [ ] Templates render with `job_id`/`status` and remain stable.

## Rollout
- [ ] Dev: implement behind standard auth; ship migration; run unit/integration tests.
- [ ] Staging: manual sanity (cancel mid‑run), verify logs and status messages, ensure no duplicate jobs.
- [ ] Production: deploy with migration; monitor `background_jobs` status distribution and `cancelled` logs; prepare quick rollback by disabling the endpoint if needed.

## Risks & Mitigations Checklist
- [x] Mitigate cancel latency during long vendor calls with safe‑boundary checks and clear messaging.
- [ ] Handle race where job finishes before user decides: re‑query job state; branch to “job finished” copy; proceed with ICP update without cancel.
- [x] Restrict cancel endpoint to admin/tenant‑admin roles and validate tenant ownership.
- [x] Use narrow queries (single job row by PK) for polling.

## Definition of Done
- [ ] All tests green; coverage stable or increased.
- [ ] No flaky sleeps; DB calls mocked in unit/integration; E2E behind CI flag.
- [ ] Docs updated; observability logs present; rollout plan executed.
