# Runbooks — Feature 8: Observability, QA & Alerts

This runbook compiles the procedures outlined in Feature PRD 8 and DevPlan 8.

## MV-01: MV Refresh Failure
- Symptom: `mv_refresh` stage errors or no candidate IDs.
- Checks: DB connection, locks on MV, recent DDL.
- Mitigation:
  - `REFRESH MATERIALIZED VIEW CONCURRENTLY icp_candidate_companies;`
  - Re-apply migrations if schema drift; rerun nightly.

## CR-02: Crawl Error Rate High
- Symptom: High Tavily error rate in alerts or vendor usage.
- Checks: robots blocks, DNS/TLS errors, `CRAWLER_DOMAIN_MIN_INTERVAL_S` settings.
- Mitigation: Increase delay, reduce concurrency, ensure Tavily key; rerun batch.

## VQ-03: Vendor Quota/Rate Limit
- Symptom: Alerts on `rate_limit_hits > 0` or `quota_exhausted = TRUE` (e.g., OpenAI or ZeroBounce).
- Checks: key validity, quotas, usage spikes.
- Mitigation: Retry later, reduce batch size, top-up credits.

## OA-04: OpenAI Rate Limit
- Symptom: 429s observed; latency spikes.
- Checks: model usage, parallelism.
- Mitigation: Backoff with jitter; reduce concurrency; switch to a cheaper/faster model temporarily.

## CC-05: Candidate Count Low
- Symptom: Candidates below floor in `run_summaries`.
- Checks: `icp_rules` validity; MV contents; industry fallback executed.
- Mitigation: Widen filters, refresh staging data, confirm tenant-provided ICP.

## OE-06: Odoo Export Errors
- Symptom: Export step errors or DB constraint failures (e.g., `autopost_bills` not null).
- Checks: Odoo mapping (`odoo_connections`), user/role, schema constraints.
- Mitigation: Set defaults for new columns, skip non-essential fields; rerun export only.

# Observability Quickstart
- Latest run stats: `GET /runs/{run_id}/stats`
- Event CSV: `GET /export/run_events.csv?run_id=...`
- QA CSV: `GET /export/qa.csv?run_id=...`
- Status for tenant (includes last run fields): `GET /shortlist/status`

# Alert Configuration (env)
- `SLACK_WEBHOOK_URL`: Slack incoming webhook URL.
- `ALERTS_CRON`: cron for alerts schedule (default: `*/5 * * * *`).
- `ALERT_TAVILY_ERROR_RATE_MAX`: max allowed error rate for Tavily (default 0.30).
- `ALERT_CANDIDATE_FLOOR`: minimum acceptable candidates per run (default 20).
- `ALERT_QA_PASS_RATE_MIN`: minimum QA pass rate on reviewed items (default 0.80).

# Cost Config (env)
- `OPENAI_PRICE_INPUT_PER_1K`, `OPENAI_PRICE_OUTPUT_PER_1K` (USD per 1K tokens).
- `ZEROBOUNCE_COST_PER_CHECK` (USD per check).

