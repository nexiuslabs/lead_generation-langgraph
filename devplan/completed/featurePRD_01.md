# Feature PRD 01 — Restore Top‑10 availability for enrichment

## Story
As a sales operator, when I confirm my ICP and receive a Top‑10 lookalike table, I expect the agent to reuse that shortlist when I type `run enrichment`, without asking me to confirm again.

## Acceptance Criteria
- When a Top‑10 table has been shown in chat, enrichment must succeed without additional confirmation steps.
- If the shortlist is missing from memory, the system must reconstruct it best-effort and persist it so downstream enrichment operates on the same 10 domains.
- The fallback must not trigger when no Top‑10 preview was ever produced (e.g., user skipped confirmation).

## Dependencies
- Access to `src.agents_icp.plan_top10_with_reasons` for regeneration.
- Database connection for `_persist_top10_preview`.

## Open Questions
1. TODO: Should the regeneration path respect micro-ICP gating before running enrichment again?
2. TODO: Are there rate limits on repeated `plan_top10_with_reasons` calls we need to guard against?

