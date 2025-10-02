ICP Finder Runbook — Replaced by PRD19

Overview
- The legacy ICP Finder (Feature 17) has been fully replaced by the agent‑driven ICP Finder described in `Development_Plan/Development/featureDevPlan19.md`.
- New flow: two‑message chat (user site, then customer sites), agent ICP synthesis, discovery planning, mini‑crawl + evidence extraction, scoring/gating, and immediate head enrichment + nightly processing.

API
- POST `/icp/intake` { answers, seeds } → persists ICP intake and triggers mapping/crawl.
- GET `/icp/suggestions` → returns micro‑ICP suggestions (now enhanced by agents when enabled).
- POST `/icp/accept` { suggestion_id | suggestion_payload } → persists active ICP; enriches small head and schedules remainder.
- GET `/icp/patterns` → aggregates from MV
- POST `/icp/research/import` → import ResearchOps Markdown under `docs/` into `icp_research_artifacts` and `icp_evidence`.

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
- `ENABLE_ICP_INTAKE` (default true), `ICP_WIZARD_FAST_START_ONLY`
- `ENABLE_AGENT_DISCOVERY` (default true), `AGENT_MODEL_DISCOVERY`
- `CHAT_ENRICH_LIMIT` / `RUN_NOW_LIMIT` (default 10)
- `ENRICH_BATCH_SIZE` (default 200)
- Crawler pacing: `CRAWLER_DOMAIN_MIN_INTERVAL_S`

Ops Recipes
- Re-run intake pipeline for a tenant: enqueue job via `/icp/intake` or call `enqueue_icp_intake_process(tenant_id)` and monitor `/jobs/{job_id}`.
- Inspect suggestions: GET `/icp/suggestions` then POST `/icp/accept` with `ssic:CODE`.
- Verify nightly: watch logs for `staging_upsert` then `enrich_candidates` finish lines; check `background_jobs` rows.
