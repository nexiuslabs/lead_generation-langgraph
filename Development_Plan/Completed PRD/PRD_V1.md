# Lead Generation LangGraph — PRD v1

## Context
The chat assistant successfully produces a Top‑10 lookalike table during confirmation, yet enrichment sometimes fails because the shortlist is missing from state despite being shown to the user.

## Desired Outcome
Restore a reliable path from confirmation to enrichment so that a displayed Top‑10 shortlist is always available when the user types `run enrichment`.

## Clarifications
- TODO: Confirm whether rehydrating the shortlist via a fresh `plan_top10_with_reasons` call is acceptable when the persisted preview cannot be loaded.
- TODO: Validate that persisting the regenerated shortlist should also update the staging tables for downstream jobs.

