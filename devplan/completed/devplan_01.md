# Dev Plan 01 — Restore Top‑10 availability for enrichment

## Design
- Detect whether the Top‑10 preview was emitted by scanning AI messages for the `Top‑10 lookalikes` heading.
- During enrichment, if no in-memory or persisted shortlist exists but the preview was shown, regenerate the shortlist with `plan_top10_with_reasons` and attach it to state.
- Persist the regenerated shortlist through `_persist_top10_preview` so subsequent stages use the same data.

## Data Model / Persistence
- Reuse existing `_persist_top10_preview` writes; no schema changes required.

## Migrations
- None.

## Testing Strategy
- Add targeted unit coverage for the helper that detects whether a preview was shown.
- Exercise the enrichment fallback via a focused async test that simulates missing state yet records a previous preview message.

## Risks
- Regeneration may take a few seconds; ensure we only trigger it when the preview was actually shown.
- Re-running `plan_top10_with_reasons` might return a slightly different ordering; mitigate by persisting immediately so subsequent runs stay stable.

