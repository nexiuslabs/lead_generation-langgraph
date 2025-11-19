# PR 04 – Final Sweep & QA

**Goal:** tidy remaining references, update docs, and validate the full simplified flow now that both gates are gone.

## Scope

1. Remove any dangling state fields from `GraphState` (e.g., `company_profile_confirmed` if no longer needed, or confine its usage to persistence).
2. Update checkpoint serialization (`app/lg_entry.py`, `tests/test_lg_entry.py`) to drop deleted fields.
3. Refresh documentation:
   - `docs/icp_pipeline_playbook.md`
   - `Development_Plan/simply_flow.md`
   - Any onboarding notes referencing “confirm profile” loops.
4. Add/adjust progress-recorder tests to ensure router honors the new three-gate flow.
5. Verify telemetry/log strings (remove `icp_flow: seed_wait`, etc.).

## Files

- `app/pre_sdr_graph.py`
- `app/lg_entry.py`
- `tests/test_lg_entry.py`
- Docs under `docs/` and `Development_Plan/`

## Verification

- Full pytest run.
- Manual QA using the conversation checklist (website → seeds → show/update → discovery confirm).
