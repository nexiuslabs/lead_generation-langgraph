---
owner: Codex Agent – Backend/Jobs
status: draft
last_reviewed: 2025-10-15
---

# Feature Dev Plan 20 — Direct ACRA Enrichment (No ICP)

## Objective
Implement a manual, one‑by‑one background enrichment service operating directly on ACRA staging data without relying on ICP/industry filters. It upserts into `companies`, enriches via the existing pipeline (`search_policy='discover'`) that uses DuckDuckGo (DDG) for domain discovery and r.jina + HTTP for page content (no Tavily), writes history into `company_enrichment_runs`, and optionally scores leads. No scheduler; runs on demand.

## Architecture & Components
- Script: `scripts/run_acra_direct.py` (new)
  - CLI/env‑driven runtime (batch/concurrency/anchors/caps)
  - Orchestrates selection → upsert → enrichment → persistence → progress → metrics
- Helper module: `src/acra_direct.py` (new)
  - `stream_staging_candidates(...)` — yields staging rows using server‑side cursor
  - `upsert_company_from_staging(row) -> int` — maps staging→companies with SSIC mapping via `ssic_ref`
  - `should_skip(company_id) -> bool` — respects `ENRICH_SKIP_IF_ANY_HISTORY`/`ENRICH_RECHECK_DAYS`
  - `persist_enrichment(company_id, state, run_id)` — projects updates + history row
  - `mark_progress(last_id)` — writes `service_progress` marker (optional)
- Existing: `src/enrichment.py` — call the enrichment pipeline via `enrich_company_with_tavily(..., search_policy='discover')`. Implementation relies on DDG for domain discovery and r.jina + HTTP for content; Tavily is not used.
- Tables: `staging_acra_companies`, `companies`, `ssic_ref`, `enrichment_runs`, `company_enrichment_runs`, `lead_scores` (optional), `service_progress` (optional)

## Data Mapping
- Normalize primary SSIC code to digits; set `companies.industry_code`.
- Resolve canonical industry via `ssic_ref` match on normalized code; set `companies.industry_norm` and (if present) `companies.industry`.
- Fallback to `primary_ssic_description` only if `ssic_ref` misses.
- Extract incorporation year from date; carry status; set website_domain when hinted in staging.

## Detailed Implementation Steps
1. Scaffolding
  - Add `scripts/run_acra_direct.py` entrypoint with argparse/env parsing
  - Add `src/acra_direct.py` with function stubs and minimal DSN plumbing
2. Selection (no ICP)
  - Implement `stream_staging_candidates(limit, start_after_id, start_from_uen, recheck_days)`
    - SQL: ordered by `id ASC`, optional status filters
    - Anti‑join to `company_enrichment_runs` via `companies(uen)` when `recheck_days > 0`
    - Supports LIMIT for batch testing
3. Upsert helper
  - `upsert_company_from_staging(row)`
    - Normalize SSIC code to digits
    - Resolve `industry_title` via `ssic_ref`; update `industry_norm` (and `industry` if exists)
    - UPSERT by `uen`; fallback by `(LOWER(name), incorporation_year)` if `uen` is null
    - RETURNING `company_id`
4. Skip logic
  - `should_skip(company_id)` uses `ENRICH_SKIP_IF_ANY_HISTORY` else `ENRICH_RECHECK_DAYS`
5. Enrichment call
  - Invoke the enrichment pipeline (`enrich_company_with_tavily(company_id, name, uen, search_policy='discover')`) which performs DDG domain discovery and r.jina/HTTP page extraction (no Tavily calls)
  - Add per‑company structured logs (selection, upsert fields, discovery/pages/chunks/emails, degraded)
6. Persistence
  - Update `companies` with enriched fields (best‑effort update list per PRD)
  - Insert `company_enrichment_runs` row; rely on helpers to only include existing columns
7. Progress & resume
  - Implement optional `service_progress(key='acra_direct', last_id)` table and writes
  - On startup, read marker and/or prefer dedupe via enrichment recency
8. Metrics & alerts (optional, gated by env)
  - Expose Prometheus counters/histograms in a lightweight HTTP server
  - Slack alert when failure ratio over last 30 min > threshold
9. Testing
  - Unit: SSIC mapping function against sample codes/titles
  - Unit: UPSERT with/without UEN; ensures `industry_norm` updated
  - Integration: small batch run in test DB; assert companies updated and history row inserted
10. Runbook & docs
  - Update README/devplan with run instructions and failure recovery

## Pseudocode (script)
```
def main():
  cfg = load_config_from_env()
  run_id = create_run_header(tenant_id=cfg.tenant_id)
  processed = fails = 0
  for row in stream_staging_candidates(cfg):
    try:
      cid = upsert_company_from_staging(row)
      if should_skip(cid):
        log_skip(cid)
        continue
      state = enrich_company_with_tavily(cid, row.entity_name, row.uen, search_policy='discover')
      persist_enrichment(cid, state, run_id)
      mark_progress(row.id)
      processed += 1
    except Exception as e:
      fails += 1
      log_error(row.id, e)
      continue
  maybe_emit_metrics(processed, fails)
```

## Config & Flags (env)
- POSTGRES_DSN (required)
- ACRA_DIRECT_BATCH_LIMIT, ACRA_DIRECT_CONCURRENCY (default 1)
- ACRA_DIRECT_START_AFTER_ID, ACRA_DIRECT_START_FROM_UEN
- ENRICH_SKIP_IF_ANY_HISTORY, ENRICH_RECHECK_DAYS
- ENABLE_DDG_DISCOVERY, STRICT_DDG_ONLY; CRAWLER_TIMEOUT_S, CRAWL_MAX_PAGES, LLM_MAX_CHUNKS; ENABLE_APIFY_LINKEDIN
- ACRA_DIRECT_DAILY_CAP, ACRA_DIRECT_ERROR_ALERT_THRESHOLD
- METRICS_PROMETHEUS_ENABLE, METRICS_PORT, SLACK_WEBHOOK_URL (optional)

## Acceptance Validation
- Dry run 50 ACRA rows; verify:
  - companies rows upserted with industry_code + industry_norm (via ssic_ref)
  - company_enrichment_runs rows linked to a run_id
  - logs show selection→upsert→discover→extract→persist
  - throughput/latency within expected bounds under caps

## Timeline (suggested)
- Day 1: scaffolding, selection, SSIC mapping UPSERT
- Day 2: enrichment wiring, persistence, progress marker
- Day 3: metrics/alerts (optional), tests, dry run

## Risks & Mitigations
- High volume → vendor caps: enforce caps and timeouts; monitor metrics
- Schema drift (industry vs industry_norm): detect columns via information_schema and write the supported one(s)
- Duplicate enrichments: rely on skip/recheck logic and UEN UPSERT identity
