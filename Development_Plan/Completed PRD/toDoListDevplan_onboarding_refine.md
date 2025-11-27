# DevPlan 007 — Pre-SDR Onboarding Refinements To-Do List (Status)

Scope: Greeting injection, company profile verification, micro-ICP confirmation, anti-ICP capture, telemetry, and supporting schema updates.
This file tracks implementation status against `Development_Plan/devplan_007_onboarding_refinements.md`.

## Status Snapshot
- [x] Greeting node + state flags gate conversation entry (`app/pre_sdr_graph.py`, `app/lg_entry.py`).
- [x] Company profile verification loop with immediate acknowledgement + telemetry logging (`app/pre_sdr_graph.py`).
- [x] Micro-ICP confirmation and anti-ICP capture gating enrichment and storing structured notes (`app/pre_sdr_graph.py`).
- [x] Prompt scaffolding + copy governance updated for tone/jargon guidance (`src/conversation_agent.py`, `app/pre_sdr_graph.py`).
- [ ] State persistence, API serialization, and schema/migration updates for new fields (GraphState + `migrations/032_company_profile_storage.sql` landed, but no API surface exposes the new flags yet).
- [ ] Telemetry events, docs, and regression tests aligned with the new flow (events exist, but docs/runbooks/tests still missing the new checkpoints).

## LangGraph Workflow & Flow Control
- [x] Insert greeting node triggered on session bootstrap when no prior turns exist; branch into verification stages (`app/pre_sdr_graph.py:7705-7775`, `app/lg_entry.py:690-723`).
- [x] Extend `pre_sdr_flow` edges so conversation cannot bypass greeting/company/micro-ICP checkpoints (`app/pre_sdr_graph.py:7740-7925`).
- [x] Ensure greeting stored as audit-friendly conversation entry and `state.greeting_sent` flips immediately to prevent duplicates (`app/lg_entry.py:690-723`).

## Company Profile Verification Loop
- [x] After company URL ingestion + synthesis, send acknowledgement message explaining profile drafting (`app/pre_sdr_graph.py:5218-5256`).
- [x] Present synthesized profile for confirmation; capture corrections and loop until approval (`app/pre_sdr_graph.py:3083-3750`).
- [ ] Store sanitized delta for analytics + future reminders (current loop appends raw `manual_notes`; no sanitized delta or reminder hooks yet).
- [ ] Rate-limit clarification loops (≤3 attempts) and escalate via fallback copy afterward (no prompt counters or escalation copy implemented).

## Micro-ICP Confirmation & Anti-ICP Capture
- [x] Present plain-language micro-ICP summary and wait for explicit confirmation before enabling enrichment commands (`app/pre_sdr_graph.py:3946-4055`, router gating at `app/pre_sdr_graph.py:7858-7898`).
- [ ] Handle adjustment feedback by updating segments and re-confirming (feedback today only triggers generic acknowledgements, no regeneration of suggestions).
- [ ] Collect anti-ICP examples/traits, normalize into structured dict, and persist to state (`anti_icp_notes` currently stores raw strings only; no structured normalization or downstream usage beyond the list).
- [ ] On returning sessions with stored anti-ICP data, prompt user to confirm/amend exclusions (no reminder logic detected).

## Prompt & Copy Governance
- [x] Update system prompt to enforce friendly tone, define jargon inline, and highlight confirmation commands (`src/conversation_agent.py:32-78`, `app/pre_sdr_graph.py:2941-2960`).
- [x] Add reusable snippets for verification and anti-ICP capture (`app/pre_sdr_graph.py:_prompt_profile_confirmation`, `_prompt_icp_profile_confirmation`, `_prompt_micro_icp_confirmation`).
- [ ] Provide copy variants for acknowledgement, confirmation, and fallback loops; route through marketing review (copy variants exist inline but no tracked approvals or alternates documented).

## State, API & Persistence
- [x] Extend conversation state with `greeting_sent`, `company_profile_confirmed`, `micro_icp_confirmed`, `anti_icp_notes` (GraphState + short-term memory in `app/pre_sdr_graph.py:2688-2875`).
- [ ] Update API serialization payload to expose new flags so FE can render checkpoints (no conversation routes expose these booleans yet).
- [ ] Confirm persistence layer accepts new JSON fields (referenced `app/models/conversation_state.py` file does not exist; no verification in repo).
- [ ] Add DB migration (Supabase/Postgres) or schema patch to store anti-ICP data without null issues (only company profile table added in `app/migrations/032_company_profile_storage.sql`).

## Telemetry & Analytics
- [x] Emit `company_profile_verified`, `company_profile_adjusted`, `micro_icp_confirmed`, `anti_icp_logged` events (`app/pre_sdr_graph.py:3235-4050`).
- [ ] Decide + document telemetry payload content (sanitized summary vs raw); align with data governance (no doc or sanitization helpers for these events).
- [ ] Ensure analytics dashboards or exports ingest the new events (no ingestion wiring or dashboards referenced).

## Tooling, Docs & Integration Updates
- [ ] Refresh mock transcripts/docs to show greeting + verification checkpoints (docs such as `docs/icp_to_enrichment_flow.md` still omit greeting + anti-ICP reminders).
- [ ] Coordinate marketing copy sign-off; log blockers if pending beyond Day 4 milestone (no evidence of sign-off tracking).
- [ ] Document new flow + flags in README/runbooks, including quick-confirm paths and anti-ICP reminders (README/docs have not been updated with these instructions).

## Testing & QA
- [x] Unit tests for graph transitions enforcing greeting/verification gates (`tests/test_router_profile_gating.py`, `tests/test_router_enrich_override.py`, `tests/test_icp_url_flow.py`).
- [ ] Integration tests simulating end-to-end onboarding transcript with correction loops (no automated chat transcripts covering corrections/anti-ICP capture).
- [ ] Regression tests covering legacy flows where URL or micro-ICP data is missing to ensure guardrails still allow progress (only targeted router tests exist; legacy flow coverage missing).
- [ ] Manual QA script for staging run: single greeting, tone compliance, telemetry event emissions (not documented).

## Risks & Open Questions
- [ ] Validate mitigation for advanced-user friction (quick confirm instructions; highlight `confirm` command) — not revisited.
- [ ] Resolve anti-ICP data sync scope (internal-only vs CRM export) before build freeze — still open.
- [ ] Finalize telemetry payload sanitization rules to avoid privacy issues — unsolved.
