ICP Finder Runbook (Feature 17)

Overview
- Minimal intake: website + 5–15 seeds (+ optional lost/churned).
- Pipeline: save intake → map seeds→ACRA (UEN/SSIC) → crawl tenant + seed sites → evidence → patterns MV → suggestions → accept → enrich small batch now; nightly processes the remainder.

API
- POST `/icp/intake` { answers, seeds } → { status, response_id, job_id }
- GET `/icp/suggestions` → List of suggestion cards (SSIC title, evidence, targeting_pack, negative_icp)
- POST `/icp/accept` { suggestion_id | suggestion_payload } → { ok: true }
- GET `/icp/patterns` → aggregates from MV
- GET `/jobs/{job_id}` → background job status

Nightly
- Dispatcher: `python -m scripts.run_nightly` (or `scripts/run_scheduler.py`)
- Jobs:
  - `staging_upsert` → upsert companies by SSIC
  - `enrich_candidates` → enrich upserted companies by SSIC (batched)
  - `icp_intake_process` → map→crawl→patterns for new intake

Observability
- Mapping SLA: `icp_intake_mapping` entries in `run_event_logs` with processed/mapped/rate/duration_ms.
- Alerts (Slack webhook optional): `scripts/alerts.py` warns if seed→ACRA rate < 0.80 or avg mapping time > 300s in last 24h.

Config
- `ENABLE_ICP_INTAKE`, `ICP_WIZARD_FAST_START_ONLY`
- `CHAT_ENRICH_LIMIT` / `RUN_NOW_LIMIT` (default 10)
- `ENRICH_BATCH_SIZE` (default 200)
- Crawler pacing: `CRAWLER_DOMAIN_MIN_INTERVAL_S`

Ops Recipes
- Re-run intake pipeline for a tenant: enqueue job via `/icp/intake` or call `enqueue_icp_intake_process(tenant_id)` and monitor `/jobs/{job_id}`.
- Inspect suggestions: GET `/icp/suggestions` then POST `/icp/accept` with `ssic:CODE`.
- Verify nightly: watch logs for `staging_upsert` then `enrich_candidates` finish lines; check `background_jobs` rows.

