---
owner: Codex Agent – Backend/Jobs
status: draft
last_reviewed: 2025-10-15
---

# Feature PRD 20 — Direct ACRA Background Enrichment Service (No ICP)

## Story
As a data operations engineer, I want a separate background enrichment service that works directly on ACRA data without requiring an ICP profile or industry filters, so that we can steadily backfill and refresh the corporate registry end‑to‑end, independent of nightly ICP‑driven runs.

## Summary
Clone the nightly ACRA enrichment flow but remove two behaviors:
- No automatic scheduling (it is started manually or by ops tooling).
- No ICP/industry filtering during upsert. Instead, iterate ACRA records directly.

This service enriches companies one‑by‑one from ACRA staging, inserting/updating the `companies` table and associated history as it goes.

## Acceptance Criteria
- A new long‑running script (e.g., `scripts/run_acra_direct.py`) that:
  - Connects to Postgres and iterates ACRA staging rows without using ICP filters.
  - For each row, upserts a single company into `companies` and runs enrichment end‑to‑end.
  - Writes enrichment results into `company_enrichment_runs` and updates core columns in `companies`.
  - Skips companies with recent enrichment unless override flags are set.
  - Emits detailed logs per company: selection → upsert → domain discovery → crawl/extract → contacts → persist.
- Start/stop is manual (no cron). The process resumes from last progress markers when restarted.
- Does not modify nightly ACRA or Top‑10/Next‑40 behaviors.

## Non‑Goals
- No ICP profile creation or usage.
- No changes to the nightly scheduler, queue priorities, or chat flows.

## Dependencies
- Database tables: `staging_acra_companies` (or equivalent), `companies`, `company_enrichment_runs`, `enrichment_runs`, `lead_scores` (optional), reference tables (`ssic_ref`).
- Existing enrichment pipeline in `src/enrichment.py` (DDG domain discovery, deterministic crawl, LLM extraction, optional contacts via Apify).
- Settings in `src/settings.py` (DSN, timeouts, page caps). No Tavily key required (DDG + HTTP fallback used).

## Detailed Process Flow
1) Initialization
- Load env and connect to Postgres using `POSTGRES_DSN`.
- Resolve run parameters:
  - `ACRA_DIRECT_BATCH_LIMIT` (max rows to attempt this run; default unlimited).
  - `ACRA_DIRECT_CONCURRENCY` (in‑flight enrichments; default 1 for strict one‑by‑one).
  - `ACRA_DIRECT_START_AFTER_ID` or `ACRA_DIRECT_START_FROM_UEN` (optional cursor/anchor).
  - `ENRICH_SKIP_IF_ANY_HISTORY`, `ENRICH_RECHECK_DAYS` (reuse global skip policy).
- Create a new `enrichment_runs` header row for metrics attribution.

2) Selection (no ICP filter)
- Use a streaming cursor over `staging_acra_companies` with optional WHERE clauses:
  - Exclude obviously invalid or dissolved entities if available (e.g., `entity_status_description`).
  - Support resume: start after last processed `staging_id` or match `uen` anchor.
  - Avoid duplicates by joining with `companies` on `uen` and skipping only when recent enrichment exists and skip policy is active.
  - Order by stable key (e.g., `uen` ascending, then `staging_id`).

3) Per‑company upsert
- Map staging fields to core `companies` columns (best‑effort with existing helpers):
  - name, `uen`, `primary_ssic_code`, `primary_ssic_description`, incorporation year/date, status, and any available contact domain hints.
- Insert or update the `companies` row (by `uen` when present, else by `(name, incorporation_year)` heuristic as a fallback).
- Capture `company_id` for enrichment.

4) Enrichment (individual)
- Invoke `enrich_company_with_tavily(company_id, name, uen, search_policy='discover')` so domain discovery is allowed for ACRA rows without known websites.
- Stages executed by the graph:
  - `find_domain` (DB → DDG fallback → heuristic probe).
  - `deterministic_crawl` (homepage + about/contact/careers seeded pages; r.jina + HTTP fallback).
  - `discover_urls` + `expand_crawl` (site navigation based, bounded by caps).
  - `extract_pages` (TavilyExtract if available, else HTTP fallback), `build_chunks`, `llm_extract`.
  - `apify_contacts` (if enabled) and `persist_core`/`persist_legacy`.
- Skip rules apply via `ENRICH_SKIP_IF_ANY_HISTORY` and `ENRICH_RECHECK_DAYS`.

5) Persistence
- Update `companies` with enriched fields (about text, website domain, employees_est, revenue_bucket, location, linkedin url, etc.).
- Insert a row in `company_enrichment_runs` linked to the `enrichment_runs` header; persist merged corpus when configured.
- Optionally generate/update `lead_scores` for UI filtering and QA dashboards.

6) Progress tracking & resume
- Maintain a lightweight progress marker:
  - Option A: store last processed `staging_acra_companies.id` in a key‑value table (e.g., `service_progress(key, last_id)` with key `acra_direct`).
  - Option B: query next candidate via anti‑join on `company_enrichment_runs` by `uen` or `company_id` with recency window.
- On restart, resume from the last cursor or deduplicate via skip rules.

7) Observability & logging
- Per‑company structured logs: selection, upsert, domain discovery result, pages extracted, chunks built, emails found, degraded fallbacks.
- Run summaries: processed, skipped, failed, duration (attach `run_id`).
- Error handling: transient failures retried with backoff; hard failures logged and the run continues.

## Interfaces & Scripts
- New script: `scripts/run_acra_direct.py`
  - Flags/env (examples):
    - `ACRA_DIRECT_BATCH_LIMIT=1000`
    - `ACRA_DIRECT_CONCURRENCY=1`
    - `ACRA_DIRECT_START_AFTER_ID=0`
    - `ACRA_DIRECT_START_FROM_UEN=SOMEUEN123`
    - Reuse global enrichment envs (HTTP timeouts, caps, skip rules).
  - Behavior: starts immediately and runs until batch limit reached or source exhausted. Exits with nonzero code on fatal connectivity errors, otherwise 0.
- No scheduler entry (no cron). Ops run it on demand or via their own orchestrator.

## Configuration
- Required
  - `POSTGRES_DSN`: PostgreSQL DSN (e.g., `postgresql://user:pass@host:5432/db`).
- Core runtime
  - `ACRA_DIRECT_BATCH_LIMIT`: maximum ACRA rows to attempt in a single run (default: unlimited).
  - `ACRA_DIRECT_CONCURRENCY`: number of in-flight enrichments (default: 1 for strict one-by-one).
  - `ACRA_DIRECT_START_AFTER_ID`: numeric cursor to start after a given staging row id.
  - `ACRA_DIRECT_START_FROM_UEN`: string cursor to start from a given UEN (exclusive).
- Enrichment behavior (reused)
  - `ENRICH_SKIP_IF_ANY_HISTORY` (bool): skip when any prior enrichment exists.
  - `ENRICH_RECHECK_DAYS` (int): re-enrich only if last update older than N days.
  - `CRAWLER_TIMEOUT_S`, `CRAWL_MAX_PAGES`, `LLM_MAX_CHUNKS`: performance/cost guards.
  - `ENABLE_APIFY_LINKEDIN` (bool): enable contact discovery via Apify where configured.

## Selection SQL Examples
- Baseline (stream all active staging rows):
```
SELECT id, uen, entity_name, primary_ssic_code, primary_ssic_description,
       registration_incorporation_date, entity_status_description
FROM staging_acra_companies
WHERE COALESCE(NULLIF(TRIM(entity_status_description), ''), 'Active') NOT ILIKE 'dissolved%'
  AND id > $START_AFTER
ORDER BY id ASC
LIMIT $BATCH;
```

- Deduplicate by recent enrichment (skip if enriched within N days):
```
SELECT s.*
FROM staging_acra_companies s
LEFT JOIN companies c ON c.uen = s.uen
LEFT JOIN company_enrichment_runs r ON r.company_id = c.company_id
  AND r.updated_at >= now() - ($RECHECK_DAYS || ' days')::interval
WHERE s.id > $START_AFTER
  AND r.company_id IS NULL
ORDER BY s.id ASC
LIMIT $BATCH;
```

## Upsert Mapping (staging → companies)
- `entity_name` → `companies.name`
- `uen` → `companies.uen`
- `primary_ssic_code` → `companies.industry_code` (normalized digits)
- SSIC → industry mapping: join `ssic_ref` by normalized `primary_ssic_code` to obtain canonical title; write to:
  - `companies.industry_norm` (preferred column in this codebase), and
  - `companies.industry` if that column exists in your schema (back‑compat UI support).
- `primary_ssic_description` → fallback for `industry_norm` only if `ssic_ref` lookup is missing.
- `registration_incorporation_date` → `companies.incorporation_year` (year extract)
- `entity_status_description` → `companies.status`
- Optional: `website`‑like hints if present in staging → `companies.website_domain`

Example upsert (pseudo‑SQL):
```
INSERT INTO companies(uen, name, industry_code, industry_norm, incorporation_year, status, website_domain)
VALUES ($uen, $name, $code, $title, $year, $status, $domain)
ON CONFLICT (uen) DO UPDATE SET
  name = EXCLUDED.name,
  industry_code = EXCLUDED.industry_code,
  industry_norm = EXCLUDED.industry_norm,
  incorporation_year = EXCLUDED.incorporation_year,
  status = EXCLUDED.status,
  website_domain = COALESCE(companies.website_domain, EXCLUDED.website_domain);
```

Industry resolution via `ssic_ref` (pseudo‑SQL):
```
WITH src AS (
  SELECT $primary_ssic_code::text AS code_digits
), norm AS (
  SELECT regexp_replace(code_digits, '\\D', '', 'g') AS code_norm FROM src
), ref AS (
  SELECT r.title
  FROM ssic_ref r
  JOIN norm n ON regexp_replace(r.code::text, '\\D', '', 'g') = n.code_norm
  LIMIT 1
)
SELECT COALESCE((SELECT title FROM ref), $primary_ssic_description) AS industry_title;
```

## Control Flow (pseudo‑code)
```
run_id = insert into enrichment_runs returning id
for each staging_row in stream_selection():
  # 1) upsert → get company_id
  company_id = upsert_company(staging_row)
  # 2) skip logic
  if should_skip(company_id):
    log(company_id, "skip_prior_enrichment")
    continue
  # 3) enrich (search_policy='discover')
  state = enrich_company_with_tavily(company_id, staging_row.entity_name, staging_row.uen, search_policy='discover')
  # 4) persist core + history
  persist_company_updates(company_id, state)
  insert_company_enrichment_run(run_id, company_id, state)
  # 5) optional scoring
  maybe_update_lead_scores(company_id, state)
```

## Step‑by‑Step Process (with DB tables)

This section expands the flow into concrete steps, showing exactly which tables are read/written at each point. It assumes a single‑process, one‑by‑one run (`ACRA_DIRECT_CONCURRENCY=1`).

1) Open Run Context
- Tables: `enrichment_runs`
- Actions:
  - Begin process; set tenant GUC if RLS is enabled:
    - `SELECT set_config('request.tenant_id', $TENANT_ID::text, true);`
  - Create a run header:
    - `INSERT INTO enrichment_runs(tenant_id) VALUES ($TENANT_ID) RETURNING run_id;`

2) Select Next ACRA Candidate (no ICP)
- Tables: `staging_acra_companies`, `companies`, `company_enrichment_runs`
- Actions:
  - Stream rows from `staging_acra_companies` using a server‑side cursor, ordered by `id ASC`.
  - Optional filters: exclude dissolved statuses.
  - Dedupe/recency guard (anti‑join or window): left join through `companies(uen)` and skip when a recent row exists in `company_enrichment_runs` for that `company_id` within `ENRICH_RECHECK_DAYS`.
  - Resume using `ACRA_DIRECT_START_AFTER_ID` or `ACRA_DIRECT_START_FROM_UEN`.

3) Upsert Company Core
- Tables: `companies`
- Actions:
  - Prefer identity by `uen`:
    - If exists: `UPDATE companies SET ... WHERE uen = $uen`.
    - Else: `INSERT ... ON CONFLICT (uen) DO UPDATE ...`.
  - Fallback identity (when `uen` is null): heuristic match on `(LOWER(name), incorporation_year)` if needed (best‑effort).
  - Fields mapped from staging:
    - name, uen, industry_code (digits of primary_ssic_code), industry_norm (primary_ssic_description), incorporation_year (extract from date), status, website_domain (if provided).
  - Retrieve `company_id` (from RETURNING or SELECT after UPDATE).

4) Apply Skip Policy
- Tables: `company_enrichment_runs`
- Actions:
  - If `ENRICH_SKIP_IF_ANY_HISTORY` is true, `SELECT 1 FROM company_enrichment_runs WHERE company_id=$id LIMIT 1`; skip if found.
  - Else if `ENRICH_RECHECK_DAYS>0`, skip when `updated_at >= now() - interval '$DAYS days'` for this `company_id`.

5) Enrichment Invocation
- Tables: read `companies` (for existing website_domain); write happens later in persistence.
- Actions:
  - Call `enrich_company_with_tavily(company_id, name, uen, search_policy='discover')`.
  - Internals (high‑level):
    - `find_domain`: DB website_domain → DDG fallback → HTTP probe.
    - `deterministic_crawl`: homepage + common subpages; r.jina with HTTP fallback.
    - `discover_urls`/`expand_crawl`: gathers additional site URLs (bounded by caps).
    - `extract_pages` → `build_chunks` → `llm_extract`.
    - `apify_contacts` (if enabled).

6) Persist Enrichment Core
- Tables: `companies`, `company_enrichment_runs`
- Actions:
  - Update `companies` with enriched fields (best‑effort): `website_domain`, `about_text`, `employees_est`, `revenue_bucket`, `linkedin_url`, `hq_city`, `hq_country`, `last_seen`.
  - Insert a history row:
    - `INSERT INTO company_enrichment_runs(run_id, tenant_id, company_id, about_text, public_emails, tech_stack, jobs_count, linkedin_url, updated_at)`
    - Only include columns that exist (guard by information_schema, as in existing helpers) and rely on DB defaults.

7) Optional: Lead Scoring
- Tables: `lead_scores`
- Actions:
  - Compute basic features and write/update `lead_scores` for the `company_id` (optional, for UI filtering/QA).

8) Progress Marker
- Tables: `service_progress` (new, simple key/value), or reuse selection logic.
- Actions:
  - Option A: `INSERT INTO service_progress(key, last_id) VALUES ('acra_direct', $staging_id) ON CONFLICT (key) DO UPDATE SET last_id=EXCLUDED.last_id;`
  - Option B: no explicit marker; rely on `ENRICH_RECHECK_DAYS` anti‑join to naturally skip recent companies.

9) Error Handling & Retry
- Tables: none (logging only).
- Actions:
  - Catch transient exceptions and retry with backoff inside enrichment steps.
  - On hard failure, log the error and continue to the next company.

10) Run Summary & Exit
- Tables: `enrichment_runs` (optional update)
- Actions:
  - Optionally update `enrichment_runs` end timestamp/counters (processed, failures).
  - Exit 0 on success; non‑zero on fatal DB connection errors.

### Table Reference (quick)
- `staging_acra_companies` — source dataset (ACRA). Key fields: `id`, `uen`, `entity_name`, `primary_ssic_code`, `primary_ssic_description`, `registration_incorporation_date`, `entity_status_description`.
- `companies` — target core table (upsert). Key fields: `company_id` (PK), `uen` (unique), `name`, `industry_code`, `industry_norm`, `incorporation_year`, `status`, `website_domain`, `about_text`, `employees_est`, `revenue_bucket`, `linkedin_url`, `last_seen`.
- `enrichment_runs` — run headers for grouping and metrics: `run_id`, `tenant_id`, `started_at`, `ended_at`.
- `company_enrichment_runs` — per‑company enrichment history linked to `run_id`.
- `lead_scores` (optional) — per‑company score snapshots for UI/QA.
- `service_progress` (optional) — KV marker for resume: `key`, `last_id`.

## Run Instructions
- Set DSN in your shell or `.env`:
```
export POSTGRES_DSN=postgresql://user:pass@localhost:5432/app
```
- Dry run 50 rows starting after a known staging id:
```
ACRA_DIRECT_BATCH_LIMIT=50 ACRA_DIRECT_START_AFTER_ID=100000 \
python -m scripts.run_acra_direct
```
- Resume from a UEN anchor with strict one‑by‑one processing:
```
ACRA_DIRECT_CONCURRENCY=1 ACRA_DIRECT_START_FROM_UEN=201912345N \
python -m scripts.run_acra_direct
```

Expected logs per company:
- select → upsert → find_domain → crawl/extract → chunks → llm_extract → contacts → persist → summary
Each step logs counts (domains/pages/chunks/emails) and degraded fallbacks when used.

## Safety & Idempotency
- Skips repeat enrichment by default using `ENRICH_SKIP_IF_ANY_HISTORY` or `ENRICH_RECHECK_DAYS`.
- Upsert by UEN ensures stable identity; name/year fallback only when UEN is missing.
- Conservative HTTP timeouts and page caps avoid excessive vendor usage.
- All operations are best‑effort; errors on one company do not halt the run.

## Data Handling
- Source: `staging_acra_companies` (names, UEN, SSIC codes/descriptions, incorporation dates, statuses).
- Target: `companies` upserted by UEN (preferred) or name/year fallback; enrichment augments core fields.
- History: `company_enrichment_runs` linked to `enrichment_runs`.
- Idempotency: enforced by skip rules and UEN‑based upserts.
- Privacy/PII: email extraction and contact data is limited to public sources; contact vendors gated by env flags.

## Success Metrics
- Sustained throughput of ≥ N companies/hour (configurable), independent of nightly flows.
- 95th percentile latency from selection → persistence < 1 hour under default caps.
- ≥ 90% success rate on domain discovery for active ACRA companies.
- Error rate < 5% per run, with no impact on nightly job latency.

## Risks & Mitigations
- Volume surge could increase external calls (DDG/Jina/Apify). Mitigate with runtime caps, daily caps, and concurrency controls.
- Low‑quality or dissolved entities may waste cycles. Mitigate with optional staging filters (status/date) and quicker timeouts.
- Duplicate enrichment of the same UEN. Mitigate with skip rules and upsert‑by‑UEN.
 - Resource contention with nightly/other services. Mitigate by running on a separate worker/host and enforcing vendor caps.

## Decisions
1. **One‑by‑one processing:** Default `ACRA_DIRECT_CONCURRENCY=1` to mirror the requirement precisely; allow increasing to 2 for ops if needed.
2. **Search policy:** Use `search_policy='discover'` to maximize success without requiring pre‑existing domains.
3. **Progress model:** Prefer Option B (deduplicate on recent enrichment history) to simplify state management; retain Option A as a fallback.
4. **No queue coupling:** This service does not consume or produce `background_jobs`; it reads from staging and writes results immediately.
5. **Observability hooks:** Expose Prometheus counters/histograms and optional Slack alerts for failure spikes to align with ops expectations.
6. **Credential/tenant isolation:** Reuse the standard service account and set `request.tenant_id` per run if RLS is active; revisit dedicated creds if isolation needs grow.

## Configuration
- Required
  - `POSTGRES_DSN`: PostgreSQL DSN (e.g., `postgresql://user:pass@host:5432/db`).
- Core runtime
  - `ACRA_DIRECT_BATCH_LIMIT`: maximum ACRA rows to attempt in a single run (default: unlimited).
  - `ACRA_DIRECT_CONCURRENCY`: number of in‑flight enrichments (default: 1 for strict one‑by‑one).
  - `ACRA_DIRECT_START_AFTER_ID`: numeric cursor to start after a given staging row id.
  - `ACRA_DIRECT_START_FROM_UEN`: string cursor to start from a given UEN (exclusive).
- Enrichment behavior (reused)
  - `ENRICH_SKIP_IF_ANY_HISTORY` (bool): skip when any prior enrichment exists.
  - `ENRICH_RECHECK_DAYS` (int): re‑enrich only if last update older than N days.
  - `CRAWLER_TIMEOUT_S`, `CRAWL_MAX_PAGES`, `LLM_MAX_CHUNKS`: performance/cost guards.
  - `ENABLE_APIFY_LINKEDIN` (bool): enable contact discovery via Apify where configured.
- Throughput & caps
  - `ACRA_DIRECT_DAILY_CAP`: soft daily cap on processed companies (default: 0 = unlimited).
  - `ACRA_DIRECT_ERROR_ALERT_THRESHOLD`: percent failures over a sliding window for alerts (default: 5).
- Metrics & alerts (optional)
  - `METRICS_PROMETHEUS_ENABLE` (bool) and `METRICS_PORT` (int) to expose `/metrics` with:
    - `acra_direct_companies_processed_total`
    - `acra_direct_failures_total`
    - `acra_direct_latency_seconds` (histogram)
  - `SLACK_WEBHOOK_URL`: send alert when failure ratio exceeds threshold.

## Rollout & Ops
- Phase 1: Dry run on a small batch with verbose logs; validate persistence and skip behavior.
- Phase 2: Increase batch limit and run time window under ops supervision.
- Runbook updates: how to start/stop, resume after failure, and interpret logs/metrics.

## Testing Strategy
- Unit test selection logic with staging mocks (filters, resume from UEN/ID).
- Integration test a small batch against a test DB: assert companies upserted, enrichment runs present, and logs emitted.
- Non‑prod smoke run before enabling on real ACRA dataset.
