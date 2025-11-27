
  # Bugsfix Onboarding Notes

  ## Current Setup Recap

  The LangGraph pipeline defined in `app/pre_sdr_graph.py` centers on the `router` node. Every worker node (`icp`, `candidates`,
  `confirm`, `enrich`, etc.) completes its work, writes updates into the shared `state`, and then yields back to `router` so the next
  hop can be decided in a single LangGraph run.

  - **start lead gen** triggers inside `icp_node` (≈5690). It renders `_build_icp_progress_recap(...)` immediately and, when
  `icp_profile_confirmed` is already true, calls `_prompt_icp_profile_confirmation(..., require_confirmation=False)` so the user sees
  the saved snapshot without needing to re-confirm.
  - After re-showing the snapshot, the command blanks out the processed text and keeps all confirmation flags, so `router` can route
  directly into ICP Finder intake again. The 2025‑11‑16 logs confirm the router flowing `router -> icp -> router -> icp` with seeds re-
  crawled and the ICP profile re-synthesized automatically after the recap.
  - `_collect_icp_progress_steps` (≈520) intentionally omits inferred fields (industries, size bands, geos, integrations) from “Still
  collecting” so the recap only shows inputs that require explicit user answers. Remaining ⏳ entries (average deal size, deal cycle,
  price floor, champion titles, predictive events) correspond to `icp` fields we cannot infer from crawls.
  - Router gating (≈8466‑8614) enforces the lead-gen lifecycle: it blocks enrichment until ICP + micro‑ICP are confirmed, pushes
  partially-complete intakes back through the `icp` node, and moves to `candidates`/`confirm` when the Finder milestones are satisfied.

  ## Cleanup & Tight ICP Flow

  Target flow: **Company Profile Generation → ICP Profile Generation → ICP Discovery**. Routing should be deterministic at each stage so
  the assistant does not stall after `start lead gen`.

  ### 1. Company Profile Generation

  1. Ask for the company website URL.
  2. Synthesize/summarize the site, persist the profile (`tenant_company_profiles`), and show the summary.
  3. Enter the “confirm company profile” loop until `company_profile_confirmed` is true.
  4. Once confirmed, prompt for customer seeds (or accept them inline if they appeared during the profile loop).

  ### 2. ICP Profile Generation

  1. Collect best customer websites (minimum 5 seeds) plus optional lost/churned URLs.
  2. Run the ICP synthesizer to populate industries, buyer titles, size bands, integrations, and triggers, storing the snapshot in
  `icp_rules` with both `icp_profile_confirmed` and `icp_profile_user_confirmed` flags.
  3. Show the synthesized ICP profile and keep asking for confirmation/edits via `_prompt_icp_profile_confirmation` until the user locks
  ### 3. ICP Discovery & Beyond

  1. When both company and ICP profiles are confirmed (or retrieved from storage for a returning tenant), emit a progress recap
  summarizing what is already captured.
  2. Immediately route toward ICP Discovery: generate micro‑ICP suggestions (`candidates` node), collect explicit confirmations
  (`confirm` node), then unlock enrichment once a micro‑ICP segment is accepted.
  3. If a tenant already has stored profiles, skip the confirm loops, show the recap + snapshot, and ask whether to proceed directly to
  micro‑ICP discovery (`start lead gen` should always re-enter the router with `icp_last_focus` set to `icp_profile`).

  By codifying this linear flow we keep the intake loops tight, avoid repeated confirmation prompts for saved tenants, and guarantee
  that `start lead gen` always leads into ICP discovery once the known prerequisites are satisfied.

  ## ICP Intake Simplification (Two Questions Only)

  We need to remove every remaining manual question so the intake only asks for:

  1. **Company website URL** – synthesize, summarize, persist, and confirm the company profile.
  2. **Best customer websites (5–15 seeds)** – synthesize, summarize, persist, and confirm the ICP profile.

  Action items:

  - Delete the prompts for industries, employee range, geographies, buying signals, ACV/price floor, deal cycle, champion titles,
    predictive triggers, and lost/churned references. Those attributes must now be inferred automatically (from crawled evidence or
    Finder outputs) or omitted entirely.
  - Update `_collect_icp_progress_steps` and any recap text so only the two captured items appear in “Still collecting”.
  - Ensure `_icp_complete` / `_icp_required_fields_done` only gate on the website + seeds so the router never loops waiting for fields
    we no longer ask about.
  - After both confirmations (or when loading persisted profiles), show the recap and prompt the user to proceed directly into ICP
    discovery (micro‑ICP suggestions). `start lead gen` must always re-enter the router with `icp_last_focus="icp_profile"` so we go
    straight into discovery for returning tenants.
