# devPlan_simplyflow.md

Detailed implementation plan for the simplified intake flow described in `Development_Plan/simply_flow.md`.

---

## 0. Prep & Tracing

1. **Inventory state fields**
   - Inspect `GraphState` in `app/pre_sdr_graph.py` (`ProfileWorkflowState` type hints around line ~2950) and note all `company_profile_*`, `awaiting_*`, and seed-gate flags.
   - Run `rg "awaiting_profile_confirmation"` and `rg "company_profile_pending"` to ensure we know every read/write before deleting them.
2. **Snapshot router behavior**
   - Document the current transitions emanating from `router_entry` (lines ~8700-8920) so we can methodically replace the gating branches.
   - Capture existing unit tests that rely on the old flags: `tests/test_profile_workflow_status.py`, `tests/test_icp_profile_update_routing.py`, `tests/test_router_profile_gating.py`, `tests/test_lg_entry.py`.

---

## 1. Data Model & State Cleanup

1. **Shrink `ProfileWorkflowStatus`**
   - Update the dataclass to hold only `company_ready`, `icp_ready`, `discovery_ready`.
   - Rewrite `_profile_workflow_status` so `company_ready=bool(state.get("company_profile"))`, `icp_ready=bool(state.get("icp_profile"))`, `discovery_ready=company_ready and icp_ready and bool(state.get("icp_discovery_confirmed"))`.
2. **Prune obsolete flags**
   - Remove fields from the `TypedDict` (`awaiting_profile_confirmation`, `profile_prompt_message_count`, `awaiting_best_customers_reply`, etc.) when they only served the confirmation loops.
   - Delete helpers that only maintained those flags (`_mark_waiting_for_seed_customers`, `_clear_waiting_for_seed_customers`, `_seed_prompt_has_new_human_input`, `_awaiting_best_customers_without_new_input`, `_nudge_waiting_for_best_customers`), or refactor them if portions remain useful.
3. **Persist helpers**
   - Ensure `_persist_company_profile_sync` and `_persist_icp_profile_sync` no longer reference the removed flags (e.g., `company_profile_pending`).

---

## 2. Company Profile Flow

1. **Simplify `_prompt_profile_confirmation`**
   - Rename or internally gate so `require_confirmation` defaults to `False`. The helper just sends the snapshot; it should not touch `awaiting_profile_confirmation` anymore.
   - Drop `profile_prompt_message_count`, `company_profile_pending`, and `profile_feedback` trackers unless still referenced elsewhere.
2. **Remove `_handle_profile_confirmation_gate`**
   - Inline any remaining useful logic (e.g., “if user shares a new site, reset profile”) into a new lightweight helper invoked from `_maybe_bootstrap_profile_from_site` or `icp_node`.
   - Delete invocations in `icp_node` and `router_entry`; they will rely strictly on presence/absence of profile payloads.
3. **Website ingestion**
   - Ensure `_parse_website` usage doesn’t unintentionally reset the profile when parsing seed lists. Consider moving the reset logic into `_maybe_bootstrap_profile_from_site` so only explicit “new site” commands trigger it.

---

## 3. ICP Profile Flow

1. **Simplify `_prompt_icp_profile_confirmation`**
   - Mirror the company snapshot behavior: drop `awaiting_icp_profile_confirmation`, `icp_profile_pending`, etc.
   - Keep ability to show snapshots on demand and after edits.
2. **Remove `_handle_icp_profile_confirmation_gate`**
   - Merge any “show / update” parsing into `icp_node` so user commands still work without gating.
3. **Seed handling**
   - Ensure `_maybe_bootstrap_icp_profile_from_seeds` is called whenever `len(seeds_list) >= 5` irrespective of previous gates.
    - Drop `_mark_waiting_for_seed_customers` and associated state; the ICP node can still re-ask politely if seeds < 5, but it should not block the router or set “awaiting” flags.
4. **Cull pre-discovery cruft**
   - Remove or refactor any helper that only existed to service the old gating path (e.g., `seed_wait` telemetry, “prompt_best_customers” message counters, retry timers).
   - After seeds are captured and the ICP profile is built, ensure no leftover logic continues to run (e.g., `ask_counts["seeds"]` loops, `icp_last_focus="seeds"` checks) before discovery; all state should reflect the new streamlined path.

---

## 4. Router Refactor

1. **Straight-through gating**
   - Replace the block from `if just_confirmed` down to the existing gating logic (~8724-8870) with:
     ```python
     status = _profile_workflow_status(state)
     if not status.company_ready:
         return _route_to_cpgen(state, text)
     if not status.icp_ready:
         return _route_to_icpgen(state, text)
     if not status.discovery_ready:
         return "icp_discovery_prompt"
     ```
   - Remove `just_confirmed`, `recent_company_confirm`, `awaiting_best_customers_reply`, and the associated `ask_counts` branches.
2. **Discovery command guard**
   - When `discovery_ready=False`, intercept “run discovery”/“run enrichment”/“accept micro-icp” commands and respond with the approval prompt rather than routing to candidates/enrich nodes.
   - Once `icp_discovery_confirmed=True`, allow these commands to flow as before.
3. **Lead-gen Q&A / other nodes**
   - Verify the simplified gating still allows Q&A, candidate drill-down, etc., since they now only depend on `company_ready`/`icp_ready`.

---

## 5. Discovery Approval UX

1. **`icp_discovery_prompt_node`**
   - Keep the prompt text but ensure it doesn’t assume an “awaiting” flag; instead, log the last prompt message ID to avoid spamming.
   - Add parsing for user confirmations (e.g., “yes, run discovery”) that sets `icp_discovery_confirmed=True`.
2. **Router integration**
   - When `discovery_ready=False`, any explicit confirm should flip the flag and immediately re-route the user’s turn to the requested action.
   - If the user responds with unrelated text, answer normally but keep `icp_discovery_confirmed=False` until they explicitly approve.

---

## 6. Interactive Controls

1. **Show commands**
   - `_is_profile_show_request` logic stays; ensure it no longer depends on `awaiting_*`.
   - Both `_prompt_*` helpers should be callable at any time without toggling workflow flags.
2. **Update commands**
   - `_is_profile_update_request` should continue to run LLM extraction and apply deltas to the respective profile, then immediately call the snapshot helper to show changes.
   - No gating logic should re-open or hold the router after an update; the router simply sees that profiles remain present and moves forward.

---

## 7. Persistence & Bootstrapping

1. **`_auto_confirm_company_profile` / `_auto_confirm_icp_profile`**
   - Rename or repurpose to simply “persist snapshot”. Remove any references to `*_pending`.
2. **`lg_entry.py` checkpoint merge**
   - Ensure the restored state no longer sets the removed flags so legacy checkpoints do not keep dangling state.

---

## 8. Tests & Tooling

1. **Rewrite gating tests**
   - `tests/test_profile_workflow_status.py`: new expectations that gating depends solely on profile presence + discovery confirmation.
   - `tests/test_router_profile_gating.py` and `tests/test_icp_profile_update_routing.py`: remove assertions around `awaiting_*` and add new cases verifying that:
     - router loops through cpgen → icpgen → discovery prompt in order,
     - show/update commands work mid-flow,
     - discovery commands are blocked until `icp_discovery_confirmed=True`.
2. **Update golden transcripts (if any)**
   - Search for fixtures referencing “confirm profile” loops and adjust them to match the new UX where the assistant shares snapshots without asking for confirmation.
3. **Lint / type checks**
   - Run `make lint` or the project’s standard `pytest` commands to ensure no stale references remain.

---

## 9. Documentation & Telemetry

1. **Docs**
   - Update `docs/icp_pipeline_playbook.md` and any onboarding docs that describe the old gates.
   - Link `Development_Plan/simply_flow.md` and this implementation plan from the main README/PRD if helpful.
2. **Telemetry / logging**
   - Remove log lines referencing the old gate names (`icp_flow: seed_wait`, `icp_flow: prompt_company_profile_confirmation require=True`, etc.) or rewrite them to describe the new flow (“sharing company snapshot”, “discovery approval requested”).
   - Ensure analytics dashboards that previously looked at `company_profile_verified` events still receive meaningful events (e.g., emit them when we share the snapshot or when the user approves discovery).

---

## 10. Rollout Checklist

1. **Backward compatibility**
   - Confirm that existing checkpoints without the new flags still hydrate because `GraphState` defaults to False/None.
2. **Dry-run**
   - Run through the lead-gen UI manually: website → seeds → ask questions → approve discovery to ensure no branch crashes.
3. **Feature toggle (optional)**
   - If needed, guard the new behavior behind an env flag (e.g., `SIMPLIFIED_PROFILE_FLOW=1`) for staged rollout.
4. **QA sign-off**
   - Document before/after behavior for the team, highlighting removal of repeated confirmation prompts and the new discovery approval gate.

---

Following these steps will replace the brittle gate maze with the streamlined three-stage guardrail described in `simply_flow.md`, while preserving the user’s ability to inspect or edit the company and ICP profiles at any point in the conversation.

---

## Example Conversation Checklist

Validate the implementation end-to-end by replaying these chat snippets (mirrors the appendix in `simply_flow.md`):

1. **Website ingestion & snapshot** – prompt for site, share company profile immediately.
2. **Show/update company profile** – user requests snapshot, then adds a note; assistant updates and replays.
3. **Seed capture reminder** – user sends fewer than five seeds, assistant politely re-asks until ≥5 gathered.
4. **ICP synthesis** – assistant shares the ICP snapshot; user can “show ICP profile” or edit buyer titles inline.
5. **Discovery approval gate** – assistant asks for confirmation; “run discovery” replies get blocked until “confirm discovery.”
6. **Post-approval actions** – “run discovery” succeeds once approved; user can still “show company profile.”
7. **Editing after approval** – user tweaks ICP industries and receives a refreshed snapshot without reopening gates.
8. **Website change resets approval** – user supplies a new site; company + ICP rebuild, discovery approval resets automatically.
9. **Partial seed drop reminder** – assistant asks for the remaining seeds without entering an awaiting state.

Each scenario should be covered in automated tests or QA notes to ensure no regression in the streamlined flow.
