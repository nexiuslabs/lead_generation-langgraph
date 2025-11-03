---
owner: Codex Agent – Frontend Generator
status: draft
last_reviewed: 2025-03-24
---

# Enhancement PRD — Lead Discovery & Scoring Data Hygiene

## Story
Sales operations relies on the automated ICP discovery pipeline to surface actionable leads with reliable firmographics. Today, malformed domains, irrelevant directory listings, and empty enrichment fields force manual triage before outreach lists are usable. As a growth analyst preparing campaigns, I need the discovery-to-scoring pipeline to deliver clean, contextually relevant leads with populated industries and headcount so I can launch sequences without scrubbing every batch.

## Background & Problem
Recent pipeline runs show three recurring defects:
1. **Malformed discovery domains** such as `2ffinestservices.com.sg` slip through normalization, causing domain-resolution failures and wasting enrichment slots.
2. **Fallback Top-10 results include irrelevant portals** like `gov.sg` or `w3.org`, diluting the ranked candidate list with non-prospects.
3. **Firmographic enrichment is blank** (Industry, Employees remain `None`), yet the scorer still grants 100-point "high" buckets based solely on research-event counts.

These issues erode trust in the pipeline, slow down SDR workflows, and generate misleading performance metrics.

## Goals
- Ensure discovery outputs only persist resolvable, normalized domains before enrichment.
- Tighten Top-10 fallback logic so only ICP-relevant companies reach the candidate table.
- Restore firmographic capture and align scoring weights so leads without core data no longer reach the highest priority bucket.

## Non-Goals
- Overhauling the entire crawler or adding new data vendors.
- Changing the nightly job cadence or batching configuration.
- Redesigning the UI that renders lead tables.

## Acceptance Criteria
- Discovery planner unit tests prove that URL-encoded domain variants (e.g., `2fexample.com.sg`) are deduped down to a single valid domain in the persisted candidate list.
- Top-10 planner integration test shows irrelevant directories/government portals are excluded, while legitimate matches remain.
- Enrichment pipeline populates industry and employee estimates for ≥90% of leads that previously returned `None`, verified via regression dataset.
- Lead scoring logic demotes leads lacking firmographics or ICP-aligned industries out of the "high" bucket even when research-event evidence exists.
- Updated documentation captures the new normalization, filtering, and scoring rules for future maintainers.

## Dependencies
- `src/agents_icp.py` domain extraction and Top-10 planner logic.
- `src/enrichment.py` company persistence and vendor response parsing.
- `src/lead_scoring.py` scoring heuristics and bucket thresholds.
- Existing unit/integration test harnesses under `tests/` for planner, enrichment, and scoring modules.

## Success Metrics
- ≥95% of enriched leads in staging runs include non-empty industry and employee fields.
- False-positive rate (non-ICP sites reaching "high" bucket) drops below 5% during QA validation.
- Enrichment failure rate attributable to malformed domains decreases by at least 80% compared to the previous sprint baseline.

## Risks & Mitigations
- **Over-filtering legitimate prospects.** Tightening filters may exclude edge-case ICP matches. *Mitigation:* instrument debug logging and run QA on historical leads before rollout.
- **Vendor data gaps.** External enrichers might still omit firmographics. *Mitigation:* add fallback parsing from existing snippets and flag residual blanks for manual review.
- **Scoring regression.** Adjusted heuristics could lower scores for true positives. *Mitigation:* A/B compare against archived batches and tune thresholds iteratively.

## Open Questions
- Should we maintain a configurable denylist for known directories/government domains, or rely solely on heuristic filters?
- What is the acceptable latency impact if additional enrichment passes are required to backfill missing firmographics?
