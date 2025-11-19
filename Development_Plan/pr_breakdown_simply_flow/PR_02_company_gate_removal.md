# PR 02 – Remove Company-Profile Gate

**Goal:** eliminate `_handle_profile_confirmation_gate` and all company-confirmation state, now that snapshots are passive.

## Scope

1. Delete:
   - `_handle_profile_confirmation_gate`
   - `awaiting_profile_confirmation`, `company_profile_pending`, `profile_prompt_message_count`, `profile_feedback_*`
2. Replace gate invocations (router, ICP node, Finder intake) with direct checks on `_has_company_profile_context`.
3. Update persistence helpers (`_persist_company_profile_sync`, `_auto_confirm_company_profile`) to ignore the removed flags.
4. Adjust tests:
   - `tests/test_icp_profile_update_routing.py`
   - `tests/test_profile_workflow_status.py`
   - `tests/test_icp_url_flow.py`

## Files

- `app/pre_sdr_graph.py`
- `tests/test_icp_profile_update_routing.py`
- `tests/test_profile_workflow_status.py`
- `tests/test_icp_url_flow.py`
- `app/lg_entry.py` and `tests/test_lg_entry.py` (checkpoint data).

## Verification

- `pytest tests/test_profile_workflow_status.py tests/test_icp_profile_update_routing.py tests/test_icp_url_flow.py tests/test_lg_entry.py`
- Manual: start lead-gen session, ensure “show company profile” works and no repeated confirmation loops occur.
