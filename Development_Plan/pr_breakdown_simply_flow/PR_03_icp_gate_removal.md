# PR 03 – Remove ICP-Profile Gate

**Goal:** mirror PR 02 for the ICP confirmation flow so the agent no longer tracks `awaiting_icp_profile_confirmation` or related flags.

## Scope

1. Delete `_handle_icp_profile_confirmation_gate` and fields:
   - `awaiting_icp_profile_confirmation`
   - `icp_profile_pending`
   - `icp_profile_prompt_message_count`
   - `icp_profile_feedback_handled_len`
   - `icp_profile_feedback_last_text`
2. Simplify ICP intake (`icp_node`, Finder) to rely on `_has_icp_profile_context`.
3. Remove seed-wait integrations tied to the old gate (any “prompt best customers” branches that reopened the gate).
4. Update tests mirroring PR 02.

## Files

- `app/pre_sdr_graph.py`
- `tests/test_icp_profile_update_routing.py`
- `tests/test_icp_url_flow.py`
- `tests/test_router_profile_gating.py`

## Verification

- `pytest tests/test_icp_profile_update_routing.py tests/test_icp_url_flow.py tests/test_router_profile_gating.py`
- Manual: feed seeds → ensure ICP snapshot appears once, edits work without confirmation loops.
