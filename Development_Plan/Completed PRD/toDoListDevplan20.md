---
owner: Codex Agent – Backend/Jobs
status: in-progress
last_reviewed: 2025-10-15
---

# To‑Do List — Feature 20: Direct ACRA Enrichment

Legend: [ ] pending, [~] in progress, [x] done

## Planning & Design
- [x] PRD finalized without Odoo sync — file: Development_Plan/Development/featurePRD20.md
- [x] Dev plan drafted with steps & architecture — file: Development_Plan/Development/featureDevPlan20.md

## Scaffolding
- [ ] Add script entry `scripts/run_acra_direct.py` (argparse/env, main loop)
- [ ] Add helper module `src/acra_direct.py` with function stubs
- [ ] Create optional table `service_progress(key text primary key, last_id bigint)`

## Selection Layer (no ICP)
- [ ] Implement streaming selection from `staging_acra_companies` (ordered, filters)
- [ ] Implement resume via `ACRA_DIRECT_START_AFTER_ID` / `ACRA_DIRECT_START_FROM_UEN`
- [ ] Implement dedupe with anti‑join to recent `company_enrichment_runs`

## Upsert Layer
- [ ] Normalize SSIC code (digits)
- [ ] Resolve industry via `ssic_ref` → set `companies.industry_norm` (and `companies.industry` if present)
- [ ] Upsert into `companies` by `uen` (fallback by name/year)
- [ ] Return `company_id`

## Enrichment & Persistence
- [ ] Wire `enrich_company_with_tavily(..., search_policy='discover')`
- [ ] Persist updates to `companies` (about_text, website_domain, employees_est, revenue_bucket, linkedin_url, hq_city, hq_country, last_seen)
- [ ] Insert `company_enrichment_runs` rows linked to run_id
- [ ] Optional: write `lead_scores`

## Progress, Metrics, Alerts
- [ ] Write `service_progress` marker after each processed row
- [ ] Expose Prometheus metrics (counters + latency histogram) — optional
- [ ] Add Slack alert on failure ratio spike — optional

## Logging
- [ ] Structured logs per company: select → upsert → discover → extract → persist → summary
- [ ] Run summary: processed, skipped, failed, duration

## Testing
- [ ] Unit tests: SSIC mapping (staging → ssic_ref → industry)
- [ ] Unit tests: UPSERT by UEN and fallback path
- [ ] Integration test: small batch run asserts companies + history writes

## Ops & Rollout
- [ ] Add run instructions to README/devplan
- [ ] Dry‑run 50 rows in staging env, confirm metrics and logs
- [ ] Green‑light for wider backfill with caps

## Nice‑to‑Haves (Post‑MVP)
- [ ] Concurrency >1 with backpressure
- [ ] Configurable exclusion lists (e.g., dissolved, too new)
- [ ] Automatic pause on high error rate

