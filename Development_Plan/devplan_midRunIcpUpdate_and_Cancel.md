# Dev Plan — Mid‑Run ICP Update and Cancellation

Source PRD: Development_Plan/featurePRD_midRunIcpUpdate_and_Cancel.md

## Objectives
- Detect user intent to change ICP while a run is active, ask to cancel or keep, and handle both paths cleanly.
- Add cooperative, observable cancellation for `icp_discovery_enrich` jobs.
- Preserve single‑active‑thread per context and avoid duplicate jobs.

## Deliverables
- Orchestrator changes (`my_agent`) with a new `run_guard` node and intent refinements.
- API endpoint to request job cancellation by `job_id` with role checks.
- Worker and job runner updates for cooperative cancel.
- DB migration adding cancel flags to `background_jobs`.
- Tests (API + orchestrator integration + one E2E) and concise docs.

## Design

### 1) Orchestrator wiring (LangGraph)
- Update `my_agent/agent.py`:
  - Add node: `run_guard` (after `profile_builder`, before `journey_guard`).
  - Edges: `ingest` → `return_user_probe` → `profile_builder` → `run_guard` → `journey_guard`.
- Update state typing (`my_agent/utils/state.py`):
  - Add `run: TypedDict(total=False)` with keys:
    - `active_job_id: int | None`
    - `status: Literal['idle','running','pending_cancel','cancelled']`
    - `awaiting_cancel_confirmation: bool`

### 2) Intent refinement and prompts
- `my_agent/utils/nodes.py`:
  - Extend `ingest_message` intent classification:
    - Map update phrases mentioning ICP/target/persona ("update icp", "change target", "revise ideal customer") to intent `update_icp`.
    - Map cancel confirmations: yes → `cancel_and_update`, no → `keep_running` when `awaiting_cancel_confirmation=true`.
  - Add prompt templates (as constants) used by `run_guard` and `progress_report`:
    - Cancel/keep prompt, cancel acknowledged, cancel complete, keep running.

### 3) New node: `run_guard`
- Responsibilities:
  - Resolve `tenant_id` from state/context as in existing nodes.
  - Lookup active job: `SELECT job_id, status FROM background_jobs WHERE tenant_id=%s AND job_type='icp_discovery_enrich' AND status IN ('queued','running') ORDER BY job_id DESC LIMIT 1`.
  - If no active job → pass through unchanged to `journey_guard`.
  - If active job exists and last intent is `update_icp`:
    - Set `state.run = { active_job_id: jid, status: 'running', awaiting_cancel_confirmation: true }`.
    - Append assistant message using the templated cancel/keep prompt; branch to `progress_report` (via the existing pending edge from `journey_guard`).
  - If `awaiting_cancel_confirmation`:
    - If intent `cancel_and_update`:
      - Call internal helper `request_cancel(job_id)` (see §5) to set `cancel_requested=true`.
      - Set `state.run.status='pending_cancel'` and keep `active_job_id`.
      - Reset ICP readiness: `profile_state.icp_profile_confirmed=false` and clear discovery cache (`state.discovery` top10/next40/candidate_ids/web_candidates).
      - Append “cancel acknowledged” message; return (pending) so `progress_report` can poll and inform.
    - If intent `keep_running`:
      - Clear `awaiting_cancel_confirmation`; stash any ICP draft (leave `profile_state.icp_profile_confirmed=false` if changed); proceed to `journey_guard`.

### 4) Gating and status surfaces
- `journey_guard`:
  - If `state.run.awaiting_cancel_confirmation` or `state.run.status='pending_cancel'`, avoid enqueueing new work; surface the prompt/status as the outstanding message and branch to `progress_report`.
- `progress_report`:
  - If `pending_cancel` and `active_job_id` exists, poll: `SELECT status, canceled_at FROM background_jobs WHERE job_id=%s`.
  - If `status='cancelled'`: append “cancel complete” message; proceed to profile update path.
  - Else: keep “cancel acknowledged” + job status.

### 5) API and server helpers
- `app/main.py`:
  - Add `POST /jobs/{job_id}/cancel` (admin or tenant owner): sets `cancel_requested=true` and returns `{ ok, job_id, status: 'pending_cancel' }`.
  - Keep existing `GET /jobs/{job_id}`; include `cancel_requested`, `canceled_at` in response (additive).
- `src/jobs.py` (internal helper):
  - `def request_cancel(job_id: int) -> dict:`
    - Update `background_jobs` row: `cancel_requested=true` if status in ('queued','running').
  - `def request_cancel_current(tenant_id: int) -> dict:` (server‑side only): best‑effort resolve latest active `icp_discovery_enrich` for the tenant and set flag. Used by orchestrator when only tenant context is available.

### 6) Worker and runner: cooperative cancel
- `scripts/run_bg_worker.py`: no structural change; runner remains the same.
- `src/jobs.py` (`run_icp_discovery_enrich`):
  - Add a small utility `_should_cancel(job_id)` that returns `background_jobs.cancel_requested`.
  - Insert checks at safe boundaries:
    - After setting `status='running'`.
    - After discovery (before enrichment loop).
    - Inside the enrichment loop (each company) before starting the next iteration.
  - On cancel request:
    - Mark `status='cancelled', canceled_at=now(), ended_at=now()`; log `cancelled` event; exit.

### 7) Data & migrations
- New SQL migration (additive): `app/migrations/00X_add_cancel_flags_to_background_jobs.sql`
  - `ALTER TABLE background_jobs ADD COLUMN IF NOT EXISTS cancel_requested boolean DEFAULT false;`
  - `ALTER TABLE background_jobs ADD COLUMN IF NOT EXISTS canceled_at timestamptz NULL;`
- Backfill not required; existing rows default to `cancel_requested=false`.

### 8) Configuration
- `CANCEL_POLL_INTERVAL_MS=500`: `progress_report` poll cadence (soft bound; actual polls happen on user turns or periodic status queries).
- Cancel policy fixed to soft cancel for this feature; no env switch is exposed.

### 9) Observability
- Emit structured logs via existing troubleshoot/metrics channels:
  - `cancel_request` (API), `cancel_ack` (orchestrator), `cancelled` (runner), with `tenant_id`, `job_id`.
- Include cancel state in `progress_report` status and in `/jobs/{job_id}`.

## Implementation Steps
1) State typing: add `run` to `OrchestrationState` in `my_agent/utils/state.py`.
2) Intent: extend `ingest_message` for `update_icp`, `cancel_and_update`, `keep_running` mappings.
3) Node: implement `run_guard(state)` in `my_agent/utils/nodes.py` with logic above.
4) Wiring: add node and edge in `my_agent/agent.py`.
5) Gating: harden `journey_guard` to respect `awaiting_cancel_confirmation`/`pending_cancel`.
6) Status: update `progress_report` to poll and render cancel states with templates.
7) Migration: add SQL file under `app/migrations/` for cancel flags.
8) API: implement `POST /jobs/{job_id}/cancel` in `app/main.py` with role checks; include flags in `GET /jobs/{job_id}`.
9) Jobs: add `request_cancel(job_id)` and `request_cancel_current(tenant_id)` helpers in `src/jobs.py`.
10) Runner: add `_should_cancel(job_id)` checks in `run_icp_discovery_enrich`.
11) Docs: update Readme/ICPs docs minimally to mention cancel capability.

## Test-Driven Development (TDD)
- Process: write failing tests → implement minimal code → refactor → repeat.
- Order of work:
  1) Acceptance tests first (high-level) for cancel/keep conversational flows and assistant messages.
  2) Unit tests for parser, state transitions, and job helpers.
  3) Integration tests for orchestrator + API; then one targeted E2E.
- Acceptance tests (failing first):
  - Mid-run ICP change triggers cancel/keep prompt; choosing cancel transitions to `pending_cancel`, clears discovery cache, and resets ICP confirmation.
  - Choosing keep proceeds without enqueueing; draft ICP preserved and applied post-run.
- Unit tests:
  - Intent parsing: phrases map to `update_icp`, `cancel_and_update`, `keep_running`.
  - `run_guard` state machine: `awaiting_cancel_confirmation` → `pending_cancel` → `cancelled`.
  - Helpers: `request_cancel(job_id)`, `_should_cancel(job_id)` behavior.
- API tests:
  - `POST /jobs/{job_id}/cancel`: happy path, role gating, invalid job/status.
  - `GET /jobs/{job_id}` includes `cancel_requested`, `canceled_at` fields.
- Worker tests:
  - Simulate loop with injected `_should_cancel(job_id)` toggles; verify cooperative stop marks `cancelled` and sets `ended_at`/`canceled_at`.
  - Boundary checks before next iteration.
- Migration tests:
  - New columns exist; defaults set; existing rows unaffected.
- Prompt snapshot tests:
  - Templates render with `job_id`/`status` and remain stable (copy assertions).
- Definition of Done:
  - All tests green; coverage stable or increased; no flaky sleeps; DB calls mocked in unit/integration; E2E isolated behind CI flag.

## Testing Strategy
- Unit
  - Intent parsing: `update_icp`, `cancel_and_update`, `keep_running` mapping from sample phrases.
  - State machines: transitions for `run.awaiting_cancel_confirmation` → `pending_cancel` → `cancelled`.
- Integration
  - Orchestrator: simulate active job row; verify `run_guard` prompt and gating (no enqueue); confirm cancel path resets discovery and ICP flags.
  - API: cancel endpoint sets `cancel_requested`; `GET /jobs/{id}` reflects flags.
- E2E (conditional on DB availability in CI)
  - Start worker in test mode; enqueue `icp_discovery_enrich`; issue cancel; assert job finishes with `status='cancelled'` and orchestrator surfaces the messages.

## Risks & Mitigations
- Cancel latency during long vendor calls: mitigate with soft cancel at iteration boundaries and clear messaging (“stopping safely”).
- Race where job finishes before user decides: re‑query job state; branch to “job finished” copy and proceed with ICP update without cancel.
- Over‑broad permissions: restrict cancel endpoint to admin/tenant‑admin roles and validate tenant ownership.
- Query overhead: use narrow queries (single job row by PK) for polling.

## Rollout Plan
- Dev: implement behind standard auth; ship migration; run unit/integration tests.
- Staging: manual sanity (cancel mid‑run), verify logs and status messages, ensure no duplicate jobs.
- Production: deploy with migration; monitor `background_jobs` status distribution and logs for `cancelled` events; add quick rollback by disabling the endpoint if needed.

## Retired Requirements
- None. This feature augments the existing flow; no prior requirements are removed.
