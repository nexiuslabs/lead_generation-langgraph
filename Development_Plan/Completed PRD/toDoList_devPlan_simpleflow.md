# toDoList_devPlan_simpleflow.md

Tracking checklist for implementing `Development_Plan/devPlan_simplyflow.md`.

- [x] Inventory existing profile/ICP state flags (`GraphState`, `awaiting_*`, `ask_counts` usage).
- [x] Update `ProfileWorkflowStatus` and `_profile_workflow_status` to only expose `company_ready`, `icp_ready`, `discovery_ready`.
- [x] Remove obsolete fields/flags from `GraphState` (awaiting/pending/profile prompt counters, seed wait markers).
- [x] Simplify `_prompt_profile_confirmation` to snapshot-only behavior (no gating side effects).
- [x] Remove `_handle_profile_confirmation_gate` and migrate any needed logic.
- [x] Harden website ingestion so seed lists don’t reset the company profile inadvertently.
- [x] Simplify `_prompt_icp_profile_confirmation` to snapshot-only behavior.
- [x] Remove `_handle_icp_profile_confirmation_gate` and integrate show/update parsing into `icp_node`.
- [x] Ensure `_maybe_bootstrap_icp_profile_from_seeds` runs automatically when ≥5 seeds exist; drop seed wait helpers/telemetry.
- [x] Delete lingering pre-discovery ICP logic (ask_count loops, `seed_wait` logs, retry timers).
- [x] Refactor `router_entry` to the new straight-through gating (cpgen → icpgen → discovery prompt).
- [x] Add guard so “run discovery/enrichment” is blocked with a prompt until `discovery_ready=True`, while other user turns are processed normally.
- [x] Rework `icp_discovery_prompt_node` to handle approvals without `awaiting_*` flags and set `icp_discovery_confirmed`.
- [x] Verify show/update commands still work mid-flow (company and ICP snapshots).
- [x] Adjust persistence helpers (`_persist_company_profile_sync`, `_auto_confirm_*`) to ignore removed flags.
- [x] Update checkpoint restore logic in `app/lg_entry.py` to avoid referencing deleted state fields.
- [x] Rewrite/expand tests (router gating, profile status, ICP updates) for the new flow.
- [x] Update docs/telemetry references to the old gates; document the simplified flow.
- [x] Manual QA run-through of the end-to-end flow (website → seeds → ICP → discovery approval → enrichment).
