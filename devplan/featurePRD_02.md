---
owner: Codex Agent – Frontend Generator
status: draft
last_reviewed: 2025-03-20
---

# Feature PRD 02 — High-throughput ACRA Enrichment Agent

## Story
Operations needs a dedicated high-frequency enrichment path so the ACRA corporate registry stays fresh for outbound prospecting. As a data ops specialist managing the ACRA dataset, I want an express worker that bypasses the nightly queue caps, so I can continuously refresh high-priority company records without waiting for the standard nightly dispatcher to finish.

## Acceptance Criteria
- A new dispatcher process (e.g., `scripts/run_acra_express.py`) continuously polls only the ACRA express job types and can be scheduled independently from the nightly run.
- Express staging jobs enqueue matching express enrichment jobs and respect per-job throttle overrides stored in `background_jobs.params`.
- Operators can configure express throughput (batch size, daily caps, cron frequency) via environment variables without code changes.
- The express agent publishes run metrics (processed companies, successes/failures, queue depth) so ops can validate the higher throughput.
- Nightly dispatcher behavior and SLAs remain unaffected when the express agent is active.

## Dependencies
- Existing job queue infrastructure: `background_jobs` table, `enqueue_staging_upsert`, `enqueue_enrich_candidates`, and related runner scripts.
- Scheduler framework (`scripts/run_scheduler.py`) or infrastructure to host a dedicated long-running worker.
- Environment/configuration management for new variables controlling express throughput and metrics publishing.
- Observability tooling (logs, dashboards, alerts) to monitor express agent performance and failures.

## Success Metrics
- ACRA express agent processes at least 3× the daily throughput of the standard nightly pipeline for ACRA-targeted jobs.
- Express runs complete within 95th percentile latency of <1 hour from staging enqueue to enrichment completion.
- No increase in failed or throttled jobs in the nightly queue after enabling the express agent.

## Risks & Mitigations
- **Resource contention:** Express agent could starve nightly jobs if it consumes shared worker capacity. *Mitigation:* isolate job types and optionally run on separate worker pool or enforce queue prioritization.
- **API rate limits:** Higher frequency could hit vendor caps. *Mitigation:* allow per-job throttle overrides and add monitoring to alert when limits approach.
- **Operational complexity:** Multiple dispatchers may confuse operators. *Mitigation:* document startup/shutdown procedures in the runbook and add health indicators.

## Decisions
1. **Express cadence and concurrency.** Trigger the express dispatcher every 15 minutes with a worker concurrency of 2. This cadence sustains sub-hour freshness while respecting DuckDuckGo’s 200 requests/hour envelope and r.jina.ai’s 120 calls/hour budget. If load testing surfaces excess headroom, we can increase concurrency to 3 without breaching provider guidance.
2. **Default throttle overrides.** Apply a default express batch size of 5 companies per staging job and a soft daily cap of 300 companies. These defaults deliver roughly 3× nightly throughput yet remain below ZeroBounce’s 500 verifications/day allocation and r.jina.ai’s 5k tokens/minute allowance. Individual jobs may still override the values when explicitly authorized.
3. **Observability hooks.** Emit Prometheus counters (`acra_express_companies_processed_total`, `acra_express_failures_total`) and histograms (`acra_express_latency_seconds`), and send Slack alerts to `#data-ops-alerts` when failures exceed 5% over any 30-minute window so operators see express health alongside nightly dashboards.
4. **Credential and tenant isolation.** Reuse the nightly worker service account while pinning the express dispatcher to the `acra_express` tenant role via environment variable. This keeps credential rotation manageable and preserves RLS scoping; we can provision dedicated credentials later if tenant isolation requirements expand.

## Open Questions
- None. Follow-up exports remain manual at launch, and stakeholders confirmed no launch backfill or additional compliance review is required.
