# PR 01 – Snapshot Helpers Cleanup

**Goal:** make `_prompt_profile_confirmation` and `_prompt_icp_profile_confirmation` purely responsible for rendering messages, so later PRs can delete the gating flags without changing formatting logic.

## Scope

1. Update both helpers to:
   - Default `require_confirmation=False`.
   - Skip all state mutations unrelated to message formatting (`awaiting_*`, `*_pending`, `*_prompt_message_count`).
   - Emit a single telemetry event `*_snapshot_shared`.
2. Touch every direct caller to remove now-unused keyword arguments.
3. Keep existing gates intact; they will still set their flags, but they will no longer rely on helper side effects.

## Files

- `app/pre_sdr_graph.py`: helper definitions + call sites.
- `tests/test_icp_profile_update_routing.py`, `tests/test_profile_workflow_status.py`, `tests/test_icp_url_flow.py`: update expectations that previously counted confirmation prompts or state changes caused by the helpers.

## Verification

- `pytest tests/test_profile_workflow_status.py tests/test_icp_profile_update_routing.py tests/test_icp_url_flow.py`
- Manual sanity check: run the graph locally and confirm that “show profile” messages still appear.
