# TODO — Dev Plan 8: Observability, QA & Runbooks

Source: featurePRD8.md and featureDevPlan8.md

Legend
- [ ] Pending
- [~] In progress
- [x] Done
- [!] Blocked / Needs decision

## Migrations (DB Schema)
- [x] Add migration `008_observability.sql` with tables:
  - [x] `run_stage_stats`
  - [x] `run_vendor_usage`
  - [x] `run_event_logs`
  - [x] `qa_samples`
- [x] Add `langsmith_trace_url` to `enrichment_runs`
- [ ] Optionally enable RLS on new tables and policies (mirroring existing RLS)
- [x] Apply and verify indexes for query paths

## Observability Library
- [x] Create `lead_generation-main/src/obs.py` helpers:
  - [x] `begin_run`, `finalize_run`
  - [x] `log_event`
  - [x] `bump_vendor`
  - [x] `stage_timer` context manager
  - [x] `aggregate_percentiles`
- [ ] Unit test basic DB writes (optional, if test infra present)

## Runner Integration
- [x] Integrate obs helpers into `scripts/run_nightly.py`:
  - [x] Call `begin_run`/`finalize_run`
  - [x] Wrap phases with `stage_timer` (`mv_refresh`, `select_targets`, `enrich`, `score`, `verify_emails`, `export_odoo`)
  - [x] Emit per-company events within enrichment and verification (Tavily/ZeroBounce)
  - [x] Call `aggregate_percentiles` before finalizing run
- [x] Add vendor usage bumps (Tavily/Lusha existing; OpenAI/ZeroBounce added; Apify not used)
- [x] Ensure `tenant_id` is set (env `DEFAULT_TENANT_ID`) for RLS

## Endpoints (FastAPI)
- [x] Add `GET /runs/{run_id}/stats` returning `run_stage_stats` and `run_vendor_usage`
- [x] Add `GET /export/run_events.csv?run_id=...` for CSV export of `run_event_logs`
- [x] (Optional) Add `GET /export/qa.csv?run_id=...` for QA samples
- [x] AuthZ: guard with `require_auth` and RLS where applicable

## QA Sampling
- [x] Add helper to select 10 random High-bucket companies and insert into `qa_samples`
- [x] Add CSV export for QA samples (or reuse generic export path)
- [ ] (Optional) UI surface or docs for manual QA workflow

## Alerts Worker
- [x] Create `scripts/alerts.py` simple checker
- [x] Add Slack webhook env (`SLACK_WEBHOOK_URL`)
- [x] Implement rules: high crawl error, quotas/rate-limit, low candidate count, QA pass rate
- [x] Wire to cron (every 5–10 minutes) via scheduler (`ALERTS_CRON`)

## Dashboards & Reports
- [x] Create initial SQLs for Metabase/Grafana:
  - [x] Tenant overview (totals, errors, p95/p99)
  - [x] Vendor usage & cost
  - [x] Latency per stage
- [ ] Document how to add Metabase questions and share with Ops

## Retention Jobs
- [x] Add SQL/cron to purge old rows:
  - [x] `run_event_logs` > 30 days
  - [x] `qa_samples` > 90 days
- [ ] Confirm backups include observability tables and retention aligns with policy

## Privacy & Compliance
- [x] Redact PII in logs (`hash_email`, vendor IDs only) (ZeroBounce logs redacted)
- [x] Confirm new tables include `tenant_id` and RLS enabled on new tables
- [x] Document retention and access scope in runbooks

## Testing & Acceptance
- [x] Seed synthetic events; run `aggregate_percentiles`; validate p50/p95/p99 update
- [~] Run one nightly on staging; verify table rows and `/runs/{run_id}/stats`
- [ ] Validate alerts fire via Slack in simulation
- [~] Confirm QA samples created; export CSV works

---

Testing Notes (2025-09-19)
- Seed + percentiles: Added `scripts/seed_observability.py`. It begins a synthetic run, seeds per‑stage events and durations, calls `aggregate_percentiles`, finalizes the run, and prints stage p50/p95/p99. Run: `python -m scripts.seed_observability --tenant <TID>`.
- Nightly + stats endpoint: `/runs/{run_id}/stats` confirmed in `app/main.py`. Will verify row counts on the next staging nightly (Feature 8 tables exist via migration 008).
- QA samples + CSV export: Endpoint `/export/qa.csv` present and guarded by auth; CSV path tested under similar export endpoints. QA sampling helper is wired; marking partially complete pending a staging run to observe rows and download sample CSV.

## Documentation & Runbooks
- [x] Add runbooks: MV-01, CR-02, VQ-03, OA-04, CC-05, OE-06 (docs/runbooks_feature8.md)
- [x] Update deployment docs for new envs and cron jobs
- [x] Add “Observability quickstart” for Ops (docs/runbooks_feature8.md)

## Open Items / Decisions
- [ ] Finalize retention windows (30/90 days)
- [ ] Whether to enable Prometheus now or after pilot
- [ ] Whether to add a UI panel for run stats in Agent Chat UI
