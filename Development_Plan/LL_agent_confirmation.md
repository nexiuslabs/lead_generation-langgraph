  # LLM Agent Confirmation Flow (Snapshot Sharing + Discovery Approval)

  ## 1. Overview
  The legacy “company confirmation gate” and “ICP confirmation gate” have been removed. The assistant now follows a simple
  three-step progression:

  1. Share the **company profile snapshot** as soon as the website crawl finishes.
  2. Share the **ICP profile snapshot** once ≥5 normalized seeds exist (either from the user or Finder).
  3. Ask for a single **discovery approval** before running heavy actions (discovery / enrichment).

  Snapshots never block routing—they are informational. Users can interrupt at any point with “show company profile,” “show ICP profile,”
  or natural-language edits and the agent must respond immediately.

  ## 2. Confirm Commands
  - “Confirm profile” now only matters for historical telemetry; it no longer re-opens a gate.
  - “Confirm icp profile” is treated as an explicit opt-in to run discovery. If both profiles exist, the router immediately routes to the discovery approval node (`icp_discovery_prompt`) which flips `icp_discovery_confirmed=True`.
  - If the user asks to “run discovery” before approval, the assistant simply replays the approval prompt; all other commands are honored.

  ## 3. Functional Requirements

  ### 3.1 Intent Detection
  - `_matches_profile_confirm_command`, `_is_profile_show_request`, `_is_profile_update_request`, and `classify_profile_intent` continue to detect user intent so the right snapshot helper is called.
  - These helpers should *never* mutate workflow flags—only render the requested snapshot or apply deltas.

  ### 3.2 Snapshot Behavior
  - `_prompt_profile_confirmation` / `_prompt_icp_profile_confirmation` always share the latest snapshot, even mid-flow, and do not toggle any `awaiting_*` state.
  - `_maybe_bootstrap_profile_from_site` and `_maybe_bootstrap_icp_profile_from_seeds` call `_auto_confirm_*` immediately after storing the snapshot so `company_profile_confirmed` / `icp_profile_confirmed` simply mirror persistence status.

  ### 3.3 Discovery Guardrail
  - `icp_discovery_prompt_node` is the only guard: it logs the last approval message, waits for an affirmative reply (“confirm discovery”, “looks good”, “run discovery”), then sets `icp_discovery_confirmed=True`.
  - Router logic:
    - No company profile → route to ICP node to collect the website.
    - Company profile but no ICP profile → stay in ICP node collecting seeds.
    - Both profiles exist but discovery not yet approved → route to `icp_discovery_prompt`.
    - Once approved, “run discovery/enrichment” flows as normal.

  ### 3.4 Memory Persistence
  - Every company or ICP edit still calls `_remember_*` so semantic memory and downstream persistence stay in sync.

  ## 4. Testing
  - Ensure “show” and “update” commands respond immediately, even after discovery approval.
  - Verify “run discovery” replies with the approval prompt until `icp_discovery_confirmed=True`.
  - Validate that providing a new website resets company + ICP snapshots and reverts `icp_discovery_confirmed` to `False`.

  `PYTHONDONTWRITEBYTECODE=1 pytest tests/test_icp_profile_update_routing.py tests/test_router_profile_gating.py`

  ---

  **Implementation Checklist**
  - [ ] Snapshot helpers never set `awaiting_*` flags; they purely render content.
  - [ ] `_auto_confirm_*` persists every snapshot immediately.
  - [ ] Router evaluates `company_ready`, `icp_ready`, `discovery_ready` in order before picking the next node.
  - [ ] Discovery prompt accepts “confirm discovery”/“run discovery” as approval, otherwise stays active without blocking other conversation turns.
