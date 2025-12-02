# Feature PRD — Return User Logic and Thread Management

## Purpose
- Objective: Improve ICP Finder UX for returning users by reusing prior company/ICP data, minimizing re-asks, and deciding when to use cached candidates vs re-run discovery. Add thread management that enforces a single active thread per ICP context while keeping history accessible (read-only).
- Chosen approach: Embedded LangGraph server (local) with a durable checkpointer and thread metadata. No external LangGraph dependency.

## Scope
- In-scope:
  - `return_user_probe` node to load company/ICP snapshots and compute decisions.
  - Thread policy: single active thread per context; auto-lock prior threads; read-only history.
  - Thread resolver: auto-resume single eligible open thread; otherwise prompt user to choose.
  - Gating updates in `journey_guard` to honor `use_cached` vs `rerun_icp`.
  - Minimal UI affordances to show locked status and disable composer.
- Out-of-scope:
  - A bespoke threads database (we rely on the embedded server + checkpointer).
  - Merging threads; admin merges may come later.
  - Non-ICP agents (policy is generalizable; initial rollout is ICP Finder).

## Architecture & Integration
- Orchestrator (LangGraph):
  - Graph order: `ingest` → `return_user_probe` → `profile_builder` → `journey_guard` → `normalize` → `refresh_icp` → `decide_strategy` → `ssic_fallback` → `progress_report` → `finalize`.
  - State additions (TypedDict):
    - `tenant_id: int | None`, `thread_id: str | None`
    - `is_return_user: bool`
    - `decisions: { use_cached: bool, rerun_icp: bool, reason: string }`
- Thread model (embedded server):
  - Use LangGraph server endpoints embedded in FastAPI; durable checkpointer (SQLite/Postgres).
  - Thread metadata: `tenant_id`, `user_id`, `agent='icp_finder'`, `context_key`, `status: 'open'|'locked'|'archived'`, `locked_at`, `archived_at`, `reason`.
  - Context key: `domain:<company_website_domain>` when present; else `icp:<rule_name>#<payload_hash>`.
- Persistence (profiles):
  - Reuse existing tables: `tenant_company_profiles` (company snapshot + confirmed) and `icp_rules` (name='Default ICP') for ICP payload + confirmed.
  - Optional: add `state_hash` columns if we need deterministic re-run triggers beyond diffs.

## User Stories
- As a returning user, I want the agent to remember my company and ICP so I don’t re-enter details and can continue where I left off.
- As a user, when I start a new session for the same ICP context, older sessions should become read-only but remain visible in history.
- As a user, if I have multiple active sessions, the agent should help me choose the right one to continue.
- As a user, if my website/ICP changed, the agent should guide me to re-run discovery with a short justification.

## User Flow
### Flow 1 — New User
- UI: Create new thread (no `thread_id`).
- Server: Creates `open` thread, computes `context_key`.
- Graph:
  - `return_user_probe`: no prior profiles → none loaded.
  - `profile_builder`: asks for website and 5 customer sites.
  - `journey_guard`: prompts step-by-step until prerequisites met.

### Flow 2 — Returning User (Single Eligible → Auto-resume)
- UI: Start without `thread_id`.
- Server: Find one eligible open thread for `(tenant_id,user_id,agent,context_key)` within `THREAD_RESUME_WINDOW_DAYS` → reuse.
- Graph:
  - `return_user_probe`: load persisted company/ICP; set confirmations; compute `decisions`.
  - `journey_guard`: skip redundant prompts;
    - If `use_cached`: set discovery ready and proceed.
    - If `rerun_icp`: ask “Re-run discovery due to X?” and wait for confirmation.

### Flow 3 — Returning User (Multiple Eligible → Disambiguate)
- Server: Multiple candidates; return assistant prompt listing ~2–3 threads by label (website or rule).
- User: Chooses; server resumes chosen thread; continue Flow 2.

### Flow 4 — New Thread For Same Context (Auto-lock older)
- UI: “New ICP session” or explicit “start fresh”.
- Server:
  - Create new `open` thread with same `context_key`.
  - Lock prior open threads with same `(tenant_id,user_id,agent,context_key)` → `status='locked'`, `locked_at=now()`.
  - If `AUTO_ARCHIVE_STALE_LOCKED=true` and prior thread is stale (`last_updated_at>THREAD_STALE_DAYS`), set `status='archived'`.
- Graph: seeds from persisted snapshots and proceeds like Flow 2.

### Flow 5 — Context Change Mid-Thread
- `return_user_probe`: detect website/ICP drift via diffs/hashes.
- Assistant: “Website/ICP changed — continue here and re-run discovery, or start a new session?”
- User chooses:
  - Continue: same thread, `decisions.rerun_icp=true`.
  - New: lock current thread, open new `open` thread with new `context_key`.

### Flow 6 — Rerun Decision (use_cached vs rerun_icp)
- `return_user_probe` sets:
  - `use_cached` when recent candidates exist and no breaking diffs/staleness/rule drift.
  - `rerun_icp` when website/industry changed, rule drift, or stale by policy.
- `journey_guard`:
  - If `use_cached`: marks discovery ready; skips refresh.
  - If `rerun_icp`: asks “Re-run discovery due to X?”; proceeds on confirmation.

### Flow 7 — Enrichment + Background Jobs
- On “run enrichment” or post‑discovery confirmation:
  - Top‑10 runs inline (cap by `RUN_NOW_LIMIT`); remainder is enqueued (Next‑40) if enabled.
  - Idempotency uses `icp_hash + rule_name`; duplicate runs are skipped or merged.

## Decisions & Policy
- Signals:
  - Profile existence (company website present, ICP confirmed), staleness (`PROFILE_STALENESS_DAYS`, `DISCOVERY_STALENESS_DAYS`), rule drift (`ICP_RULE_NAME`), structural diffs.
- Outcomes:
  - `use_cached`: recent candidates + no breaking diffs → skip refresh; still allow `ssic_fallback`.
  - `rerun_icp`: website/industry changed, rule drift, or stale → ask to re-run; proceed on confirmation.
- Thread resolve (server):
  - With `thread_id`: always use it.
  - Without `thread_id`: exactly one eligible → reuse; multiple → prompt user; none → create new.

## Thread Management
- Status semantics:
  - `open`: accepts new runs/messages.
  - `locked`: read-only; history visible; server rejects new runs with 409 `{error:'thread_locked'}`.
  - `archived`: hidden by default (UI “Show archived” reveals).
- Auto-lock rule:
  - On creating a new `open` thread for the same `context_key`, lock prior open threads with same `(tenant_id,user_id,agent)`.
- Limits:
  - `MAX_OPEN_THREADS_PER_AGENT=1` for `icp_finder` (per context). Other agents unaffected.
- UI behavior:
  - Locked threads show a “Read-only” badge; composer disabled; show “Resume in new thread” action.

## APIs & Server Hooks
- Embedded LangGraph server routes (mounted in FastAPI): `/threads`, `/threads/{id}`, `/threads/{id}/runs`, `/threads/search`, `/runs/stream`.
- Thread create hook:
  - Compute `context_key`; set metadata; set `status='open'`; lock prior open threads for same `(tenant_id,user_id,agent,context_key)`.
- Run start gate:
  - If `status!='open'` → 409 `{error:'thread_locked', hint:'create_new'}`.
  - Soft guard: graph reads thread metadata and replies with a read-only message if reached.
- Orchestrator nodes:
  - `return_user_probe(state)` loads snapshots via legacy helpers and computes `decisions`.
  - `journey_guard` honors `decisions`; prompts on `rerun_icp`; marks discovery ready on `use_cached`.

## Configuration
- `SINGLE_THREAD_PER_CONTEXT=true` (enable auto-locking).
- `THREAD_RESUME_WINDOW_DAYS=7` (resume eligibility window).
- `THREAD_STALE_DAYS=30` (auto-archive threshold).
- `AUTO_ARCHIVE_STALE_LOCKED=true` (promote stale locked → archived).
- `RETURN_USER_STRICT=true` (require user choice when multiple candidates).
- `PROFILE_STALENESS_DAYS=14`, `DISCOVERY_STALENESS_DAYS=14`.
- `MAX_OPEN_THREADS_PER_AGENT=1` (ICP Finder default).
- Existing env remain (e.g., `ICP_RULE_NAME`, `RUN_NOW_LIMIT`, etc.).

## Acceptance Criteria
- Starting without `thread_id`:
  - If exactly one eligible thread → auto-resume (thread_id returned).
  - If multiple → assistant prompt lists top candidates; user choice resumes selected thread.
  - If none → new thread created; prior open threads with same context are locked.
- Opening a new thread for same context locks prior open threads and marks them read-only.
- Locked threads: returning runs/messages are rejected with 409; UI disables composer and shows a badge.
- Return-user decisions: `use_cached` sets discovery ready without running refresh; `rerun_icp` prompts and re-runs on confirmation.
- No duplicate background jobs when identical `icp_hash + rule_name` exists (idempotency).
- Tenant isolation enforced for all thread operations and profile loads.

## Non-Functional Requirements
- Performance: thread search < 50 ms typical; probe + gating overhead < 150 ms.
- Reliability: no duplicate enrichment jobs; advisory locks around enqueue operations.
- Security: honor RLS per-tenant; never resume across tenants; require JWT except local_dev.
- Observability: structured logs for thread decisions, probe outcomes, gates, and auto-locks.

## Testing Strategy
- Unit: hash/diff functions; decision matrix; `context_key` computation; thread resolver policy.
- Integration: auto-resume (single candidate), disambiguation (multiple), new-thread auto-lock; `use_cached` vs `rerun_icp` paths; locked thread run returns 409.
- E2E: new user → confirm website/ICP → candidates → enrichment (Top‑10) → Next‑40 enqueue; returning user → auto-resume; change website → prompt for re-run; choose new → lock/continue.
- Regression: `/export/latest_scores.csv` unaffected; nightly scheduler and ACRA ingests unaffected.

## Risks & Mitigations
- Ambiguous context keys → user confusion: always prompt when >1 candidate; show clear labels (website, rule).
- Mid-run locking: defer lock until run completes or mark pending-lock.
- Long-term thread growth: auto-archive stale locked threads; UI hides archived by default.
- DB coupling: profiles read via existing helpers; optional `state_hash` migration only if needed.

## Rollout Plan
- Phase 1: implement `return_user_probe` + `journey_guard` decisions; add thread resolver logic; lock prior open threads on create; UI read-only state.
- Phase 2: enable `AUTO_ARCHIVE_STALE_LOCKED`; add optional `state_hash` fields if policy needs deterministic triggers; expand to other agents with per-agent caps.
- Ops: start in dev with `RETURN_USER_STRICT=true`; adjust resume window after feedback.

## Completion Checklist
- Code:
  - `my_agent/agent.py`: insert `return_user_probe`.
  - `my_agent/utils/nodes.py`: add probe and decisions; gate in `journey_guard`.
  - `my_agent/utils/state.py`: add fields for `tenant_id`, `thread_id`, `is_return_user`, `decisions`.
  - Embedded server mounted with durable checkpointer; thread create/run hooks enforce policy.
- Docs:
  - Update `AGENTS.md` (Supervisor Prompt: return-user policy).
  - Add this PRD to repo; add a short README in `Development_Plan` referencing it.
- Tests:
  - Unit/integration/E2E added and passing.
- Ops:
  - Env flags set; metrics/logs confirmed in dev.
- UI:
  - Composer disabled for locked threads; “Resume in new thread” action shown.

