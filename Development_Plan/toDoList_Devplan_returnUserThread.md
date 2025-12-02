# To‑Do List — Return User Thread (Tracking)

Purpose: Track implementation of the updated chat journey in `Development_Plan/devplanPRD_returnUserThread.md` where the chat orchestrator no longer runs ICP discovery/enrichment inline and instead queues background work.

Status legend: [ ] pending · [~] in progress · [x] completed

## 0) Scope & Outcomes
- [x] Chat collects confirmations (company + ICP), then queues a background job.
- [x] Discovery/enrichment/scoring/email/Odoo run in background workers only.
- [x] Doc updated to reflect simplified chat graph + runner ordering.

## 1) Orchestrator Graph Wiring (my_agent)
- [ ] Add `return_user_probe` node; wire: `ingest -> return_user_probe -> profile_builder`.
  - File(s): `my_agent/agent.py`, `my_agent/utils/nodes.py`
  - Accept: node exists; runs before `profile_builder`; sets `state.is_return_user` and seeds profiles when available.
- [x] Keep simplified leg after `ssic_fallback` → `progress_report -> summary` (no inline Top‑10/enrich/score/export in chat graph).
  - File(s): `my_agent/agent.py`
  - Accept: Graph has no `plan_top10/enrich_batch/score_leads/export_results` nodes.

## 2) Return‑User Hydration & Decisions
- [ ] Implement `return_user_probe` to load persisted snapshots (company + ICP) from Postgres by tenant.
  - Use helpers: `_fetch_company_profile_record`, `_fetch_icp_profile_record` (from `app/pre_sdr_graph.py`).
  - Set: `state.profile_state.company_profile`, `state.profile_state.icp_profile`, and confirmation flags.
  - Set: `state.is_return_user = True` when snapshots exist; compute `state.decisions` (use_cached vs rerun_icp) by diffs/staleness.
  - Accept: On an existing tenant with snapshots, chat skips redundant asks.

## 3) journey_guard — Queue Background Job Only
- [~] Always queue unified background job when `company_profile_confirmed` and `icp_profile_confirmed` are true; no inline discovery/enrich.
  - File(s): `my_agent/utils/nodes.py` (`journey_guard` already contains enqueue logic; verify defaults and remove inline variants).
  - Accept: Chat replies with queued message + job id; no inline enrich path executed.

## 4) Thread Policy & Context
- [ ] Single active thread per context; compute and store `context_key`; auto‑lock prior open threads.
  - File(s): `app/main.py` (embedded server bootstrap/policies) or helper.
  - Accept: Creating a new thread with same context locks previous; reads resume window.
- [ ] Durable checkpointer for threads/runs in embedded server.
  - Replace `MemorySaver` when server enabled (use filesystem checkpointer under `LANGGRAPH_CHECKPOINT_DIR`).
  - Accept: Threads persist in `.langgraph_api/` or configured dir; survive restarts.

## 5) Configuration Defaults (src/settings.py)
- [ ] Enforce background behavior:
  - `BG_DISCOVERY_AND_ENRICH=true`
  - `CHAT_DISCOVERY_ENABLED=false`
  - `UI_ENQUEUE_JOBS=true` (when UI should handle queue events)
  - Accept: With defaults, chat queues job; does not perform inline discovery/enrich.

## 6) Background Runners — Ordering & Behavior
- [x] Unified background worker (icp_discovery_enrich): enrich → lead scoring → notify email → Odoo export.
- [x] Nightly ACRA (enrich_candidates): enrich → lead scoring → notify email → Odoo export.
- [x] ACRA Direct: enrichment only (no post‑enrichment).
  - File(s): `src/jobs.py`, `src/acra_direct.py`
  - Accept: Logs and sequence confirm the order per runner.

## 7) Observability
- [x] Lead scoring logs (counts/timings) and orchestrator export path logs.
- [x] Background job logs: scoring/email/Odoo export with durations and IDs.
- [ ] Chat “status” intent (optional): summarize background job progress on demand.
  - File(s): `my_agent/utils/nodes.py` (ingest intent + handler) or API endpoint reuse.

## 8) Tests
- [ ] Remove/skip inline Top‑10/enrich/score chat tests; add tests for enqueue + status messages.
  - File(s): `tests/` (new or adapt existing).
  - Accept: CI covers enqueue path and runner ordering; no inline enrich in chat tests.

## 9) Docs & UX
- [x] Update `devplanPRD_returnUserThread.md` (chat defers heavy work; simplified graph).
- [ ] Add short section in README/UI copy: “Results will be emailed and synced to Odoo when the background job completes; ask for status anytime.”

## 10) Cleanup
- [ ] Remove unused imports in chat nodes (e.g., `enrich_company_with_tavily`, `plan_top10_with_reasons`) if not referenced post‑change.
- [ ] Mark inline Top‑10/enrich/score/export helpers as deprecated or move to legacy module if still needed for tests.

---

### Cross‑linking
- PRD: `Development_Plan/devplanPRD_returnUserThread.md`
- Orchestrator: `my_agent/agent.py`, `my_agent/utils/nodes.py`
- Legacy helpers: `app/pre_sdr_graph.py`
- Background jobs: `src/jobs.py`, `scripts/run_nightly.py`, `src/acra_direct.py`
- Server/bootstrap: `app/main.py`

### Recent Work (Checked)
- [x] Background job ordering unified (scoring → email → Odoo export) where required.
- [x] Structured logs added for scoring/email/Odoo export.
- [x] PRD updated to reflect background‑only discovery/enrichment.

### Open Risks / Notes
- Thread auto‑resume depends on embedded server policies + durable checkpointing; confirm in non‑dev environments.
- Return‑user probe must hydrate from Postgres (company/ICP) to minimize re‑asks; not yet implemented.
- Ensure env defaults are applied in deployments (BG only) to prevent accidental inline execution.

