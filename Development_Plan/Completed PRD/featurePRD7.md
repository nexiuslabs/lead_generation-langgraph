# Feature PRD 7 — Scheduling & Throughput

Purpose: Define the nightly scheduling, throughput targets, and pipeline orchestration to produce a daily shortlist per tenant, with robust fallbacks and vendor rate-limit awareness. This elaborates PRD.txt Section 7.

## Scope
- Nightly end-to-end batch for all tenants from data refresh to Odoo sync.
- Deterministic crawl → Tavily fallback → AI extraction → Lusha fallback → ZeroBounce verification → Scoring → Odoo sync.
- Concurrency, rate-limits, and SLAs to achieve < 90 minutes for 1k shortlisted companies.

## Out of Scope
- Full observability dashboards and alerts (covered in PRD 8).
- PDPA and compliance details (covered in PRD 9).
- Cost guardrails specifics (covered in PRD 10).

## Goals
- Produce a fresh, capped shortlist per tenant nightly, with up-to-date features, scores, and contacts.
- Default daily cap per tenant is 20 (overridable per env/tenant policy).
- Keep runtime scalable (historical target: < 90 minutes for 1k) while respecting vendor quotas.
- Ensure idempotence across steps so re-runs and partial failures do not duplicate records.

## Assumptions
- ACRA ingestion into `staging_acra_companies` exists and runs independently (`acra_webhook`).
- App DB and Odoo connections are provisioned (Section 6 onboarding). RLS is enabled per tenant.
- External vendor keys may be absent; fallbacks allow graceful degradation.

## Personas
- Ops Engineer: sets schedules, monitors runs, configures quotas.
- SDR/Analyst: consumes shortlist and rationale; expects daily updates.

## Architecture Overview
- Scheduler triggers a per-tenant nightly DAG:
  1) Refresh companies from ACRA staging (if new data since last run)
  2) Refresh ICP candidates (per tenant)
  3) Select daily target set (cap) and orchestrate enrichment
  4) Deterministic crawl (per-domain rate limit)
  5) Tavily crawl fallback → merge raw → AI extraction
  6) Lusha fallback (contacts)
  7) ZeroBounce verification (batched)
  8) Scoring + rationale + persist
  9) Odoo pre-create/update companies & contacts (idempotent)

## Detailed Workflow

1) Refresh companies from ACRA staging
- Detect new or updated `staging_acra_companies` rows since last nightly run.
- Upsert into `companies` using existing normalization/upsert logic.
  - Implementation anchor: `src/icp.py` functions `_fetch_staging_rows`, `_normalize_row`, `_upsert_companies_batch` and best-effort staging→companies sync in `app/main.py` input normalization. Nightly run should execute a dedicated upsert over a broader slice (not just chat-triggered).

2) Refresh ICP candidates per tenant
- Build or refresh a materialized view or cached set (`icp_candidate_companies`) per tenant based on the tenant’s ICP payload.
- Implementation anchor: `src/icp.py` (`icp_refresh_agent`, `_select_icp_candidates`).

3) Select daily target set (cap)
- Compute a per-tenant daily cap (default 20). Prioritize candidates with stale/missing features or contacts.
- Sort priority: previously unseen domains > stale companies > recently updated.
- Persist a run manifest (tenant_id, run_id, selected company_ids, timestamp, caps and vendor budget snapshot).

4) Deterministic crawl (primary)
- For each company, derive a candidate URL (from `website_domain`); crawl homepage + up to N key pages using per-domain rate limit.
- Implementation anchor: `src/crawler.py` (`crawl_site`), which respects `robots.txt`, skips media, and collects signals (emails, tech hints, pricing, cases, careers).
- Config: `CRAWL_MAX_PAGES`, `CRAWLER_TIMEOUT_S`, `CRAWLER_USER_AGENT` (in `src/settings.py`).

5) Tavily fallback → merge raw → AI extraction
- When deterministic crawl is blocked/thin, use Tavily crawl/extract to build a merged corpus; then run AI extraction against the corpus with token guardrails.
- Implementation anchor: `src/enrichment.py` (Tavily client, `PERSIST_CRAWL_CORPUS`, `EXTRACT_CORPUS_CHAR_LIMIT`, `LANGCHAIN_MODEL`).
- Cost guardrails: Respect PRD 10 budgets; cap pages and corpus length.

6) Lusha fallback (contacts)
- When no contacts found from site/corpus, fetch contacts from Lusha by domain.
- Implementation anchor: `src/enrichment.py` (`AsyncLushaClient`, `upsert_contacts_from_lusha`, `ENABLE_LUSHA_FALLBACK`, `LUSHA_PREFERRED_TITLES`).
- Cap lookups per run; persist vendor IDs for reuse.

7) ZeroBounce verification (batched)
- Verify newly discovered emails in batches; cache results to avoid repeat charges.
- Implementation anchor: `src/enrichment.py` (email verification cache helpers), with `ZEROBOUNCE_API_KEY`.
- Batch size and concurrency configurable; respect vendor throttling.

8) Scoring + rationale + persist
- Compute feature vectors and scores; bucket into High/Medium/Low; generate concise rationale; persist `lead_features` and `lead_scores` (RLS-aware, tenant_id when present).
- Implementation anchor: `src/lead_scoring.py` (`lead_scoring_agent`).
- Ensure per-tenant GUC is set for RLS in non-HTTP runs (uses `DEFAULT_TENANT_ID`).

9) Odoo pre-create/update (idempotent)
- For High/Medium (threshold configurable), upsert companies and primary contacts into Odoo; optionally create leads when score ≥ threshold.
- Implementation anchor: `app/main.py` (`/export/odoo/sync`) and `app/odoo_store.py`.
- Must be idempotent; avoid duplicate partners/contacts; log partner_id and any merge actions.

## Scheduling Model

Time window and cron
- Default nightly window: 01:00 SGT start. Single run per tenant per day.
- Global cron can stagger tenants to smooth vendor quotas.

Tenant iteration and concurrency
- Discover active tenants from `tenants` joined to `odoo_connections`.
- Process tenants with a fixed concurrency (e.g., 2–3 tenants in parallel), configurable.

Per-tenant concurrency
- Within a tenant run, process companies concurrently (e.g., 8 workers) with a per-domain rate limit (e.g., ≤ 1 req/0.5s per domain) to be polite and reduce blocks.
- Deterministic crawl is sequential per domain; vendor fallbacks are bounded by per-tenant caps.

Retries and backoff
- Transient HTTP/vendor errors: retry with exponential backoff (jitter); bound retries (e.g., ≤ 3 attempts).
- Persist partial progress; resume from manifest on rerun.

Idempotence and dedup
- All upserts are idempotent (companies, lead_features, lead_scores, contacts, Odoo). Use conflict targets and key-based merging.

Pause/skip controls
- Allow excluding tenants (maintenance), or skipping sub-steps via config flags.

## Configuration (env)
- Core
  - `SCHED_ENABLED=true|false` (default true)
  - `SCHED_START_CRON=0 1 * * *` (01:00 SGT)
  - `SCHED_TENANT_CONCURRENCY=3`
  - `SCHED_COMPANY_CONCURRENCY=8`
  - `SCHED_DAILY_CAP_PER_TENANT=20`
  - `SCHED_TENANT_INCLUDE` / `SCHED_TENANT_EXCLUDE` (CSV of tenant_ids)

- Crawl and extraction (already present where applicable)
  - `CRAWL_MAX_PAGES`, `CRAWLER_TIMEOUT_S`, `CRAWLER_USER_AGENT`, `EXTRACT_CORPUS_CHAR_LIMIT`, `PERSIST_CRAWL_CORPUS`

- Vendor budgets (per tenant, per run unless noted)
  - `OPENAI_DAILY_TOKENS_CAP` (e.g., 150k)
  - `TAVILY_MAX_QUERIES` (e.g., 800)
  - `LUSHA_MAX_CONTACT_LOOKUPS` (e.g., 500)
  - `ZEROBOUNCE_MAX_VERIFICATIONS` (e.g., 1000)

- Thresholds
  - `SCORE_MIN_EXPORT=0.66` (create lead in Odoo when ≥)
  - `SHORTLIST_DAILY_CAP=20` (alias of SCHED_DAILY_CAP_PER_TENANT)

## Data Model and Persistence
- `enrichment_runs` (already scaffolded): record run_id, tenant_id, started_at/finished_at, step counters, error summary.
- `company_enrichment_runs`: per-company enrichment output snapshot with optional raw corpus reference (guarded by `PERSIST_CRAWL_CORPUS`).
- `lead_features`, `lead_scores`: already used by scoring; ensure `tenant_id` columns with RLS when migrations are applied.
- Optional MV `icp_candidate_companies` per tenant (or equivalent cache) refreshed nightly.

## APIs and Tooling
- CLI script `scripts/run_nightly.py` (or `scripts/run_scheduler.py`) to:
  - Enumerate tenants → build manifests → execute pipeline with concurrency controls.
  - Provide `--tenant`, `--limit`, `--dry-run`, and `--from-manifest` flags.
- Admin HTTP kickoff endpoint (optional, ops-only): `POST /admin/runs/nightly?tenant_id=...` with auth and role guard.

## Throughput Targets and Budgeting
- SLA: With the default cap of 20 per tenant per night, end-to-end time should be just a few minutes. Historical scalability target remains < 90 minutes for 1k shortlisted candidates per tenant (p90), assuming vendor availability and typical site responsiveness.
- Budget breakdown example (ballpark):
  - Crawl (deterministic): ~2–4s per company avg with 8-way concurrency → ~4–8 minutes/1k when all fast; long-tail mitigated by skip thresholds.
  - Tavily fallback + extract: bounded to top-N pages and char limit; cap total Tavily calls per run.
  - Lusha fallback: 200–500 contacts per run; parallelism ≤ 5.
  - ZeroBounce: batch in 100s; parallelism ≤ 5; cache hits reduce calls.
  - Rationale generation: short prompts; consider caching via feature hash; cap tokens/day per tenant.

- ## Acceptance Criteria
- Functional
  - Produces a per-tenant shortlist capped at `SHORTLIST_DAILY_CAP` (default 20) with features, scores, buckets, rationale.
  - Odoo sync upserts companies and primary contacts idempotently; leads created when score ≥ `SCORE_MIN_EXPORT`.
  - RLS isolation respected for all writes and reads; `request.tenant_id` GUC is set during non-HTTP runs via `DEFAULT_TENANT_ID` or equivalent.
- Performance
  - For a tenant with 1k shortlisted candidates, end-to-end time ≤ 90 minutes (p90) on reference hardware and network.
  - Vendor quotas not exceeded; runs gracefully degrade (fewer contacts or delayed verification) rather than fail.
- Resilience
  - Partial failures do not corrupt data; reruns pick up from manifest and remain idempotent.
  - Retries/backoff applied to transient errors; final status recorded per tenant.

## Risks and Mitigations
- Vendor rate limits or outages → enforce per-run caps; retry/backoff; cache results; degrade gracefully.
- Slow or blocked sites → limit per-domain pages and timeouts; use Tavily fallback; skip low-signal cases.
- DB contention → batch writes and use upserts; avoid long transactions.
- Token/cost overruns → daily caps per tenant; rationale caching by feature hash.

## Open Questions
- Exact per-tenant daily cap defaults and overrides.
- Ordering heuristics for target set (e.g., favor stale High/Mid vs. new candidates).
- Admin endpoint vs. CLI-only kickoff.
- Whether to persist full merged corpora by default in prod (privacy/cost tradeoff).

## Implementation Notes (tie-in to current codebase)
- Use `src/icp.py` agents for normalization and ICP candidate refresh to seed nightly runs.
- Use `src/enrichment.py` for crawl/Tavily/extraction, contacts, and email verification; ensure new caps are read from env.
- Use `src/lead_scoring.py` for scoring and persistence; set `DEFAULT_TENANT_ID` per tenant in worker before DB calls to set GUC.
- Use `app/odoo_store.py` and `app/main.py:/export/odoo/sync` semantics for Odoo upsert and lead creation thresholds.
