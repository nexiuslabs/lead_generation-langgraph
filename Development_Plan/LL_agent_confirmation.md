  # LLM Agent Confirmation Flow (Company → ICP → Discovery)

  ## 1. Overview
  We enforce a strict progression:

  1. **Company Profile Confirmation**
  2. **ICP Profile Confirmation**
  3. **ICP Discovery + Enrichment**

  The LLM-driven agent must understand which profile the user is referencing, capture every edit in memory, and only allow discovery
  once the ICP profile is explicitly confirmed.

  ## 2. Key Decision: Generic “confirm profile”
  - **Policy**: Do *not* auto-resolve generic “confirm profile” to the ICP gate.
  - **Rationale**: “Profile” historically refers to the company/site profile. Auto-promoting would risk confirming an ICP users haven’t
  reviewed.
  - **Behavior**: If the company gate is already closed and the user says “confirm profile,” respond with a clarification such as:

    > “The company profile is already locked. If you’re happy with the ICP profile shown above, reply **confirm icp profile** (or tell
  me what to change).”

  This keeps telemetry clean and ensures the user intentionally confirms the ICP snapshot.

  ## 3. Functional Requirements

  ### 3.1 Intent Detection
  - `_matches_profile_confirm_command(text, icp=True/False)` uses keyword detectors (`ICP_KEYWORD_RE`, `COMPANY_PROFILE_KEYWORD_RE`).
  - `_is_profile_show_request` / `_is_profile_update_request` accept an `icp` flag to keep “show/update” routing scoped.
  - `classify_profile_intent` (LLM) backs up regex rules.

  ### 3.2 Company Profile Gate
  - `awaiting_profile_confirmation` controls prompts.
  - “Confirm icp…” while this gate is open ⇒ send clarification and stay in the gate.
  - On true confirmation:
    - `company_profile_confirmed=True`
    - `company_profile_pending=False`
    - `_remember_company_profile` + `_persist_company_profile_sync`
    - Ask for seeds if `<5`.

  ### 3.3 ICP Profile Gate
  - Requires ≥5 seeds, ICP snapshot, and `awaiting_icp_profile_confirmation`.
  - Confirmation text **must include `icp` or “ideal customer.”**
  - Generic “confirm profile” when company gate already closed ⇒ send clarification (above) and keep waiting.
  - On confirmation:
    - `icp_profile_confirmed=True`
    - `icp_profile_pending=False`
    - `_remember_icp_profile` + `_persist_icp_profile_sync`
    - Trigger discovery planner/micro-ICPs.

  ### 3.4 Memory Persistence
  - Every edit (manual note, LLM merge) calls `_remember_company_profile` or `_remember_icp_profile`.

  ### 3.5 Discovery Guardrail
  - Router checks `icp_profile_confirmed` before invoking planner, Top‑10, enrichment.
  - If user tries to “run discovery” early, reply with: “Please confirm your ICP profile first so I can use it for discovery.”

  - Include tenant_id, confirmation text, profile keys.

  ## 4. Testing
  Add/extend tests ensuring:
  1. Generic “confirm profile” during ICP gate only clarifies (no state change).
  2. Clarification not sent when `confirm icp profile` is used.
  3. Memory snapshots updated when user edits without confirming.

  `PYTHONDONTWRITEBYTECODE=1 pytest tests/test_icp_profile_update_routing.py`

  ---

  **Implementation Checklist**
  - [ ] Update `_matches_profile_confirm_command`, `_is_profile_show_request`, `_is_profile_update_request`.
  - [ ] Add clarifying message branch in ICP gate for generic confirmations.
  - [ ] Ensure `_remember_*` calls exist for every edit path.
  - [ ] Guard discovery nodes with `icp_profile_confirmed`.
  - [ ] Extend telemetry + tests.