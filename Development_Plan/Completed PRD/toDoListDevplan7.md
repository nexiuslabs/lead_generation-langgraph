# TODO — Dev Plan 7: Scheduling & Throughput

Source: featureDevplan7.md and featurePRD7.md

Legend
- [ ] Pending
- [~] In progress
- [x] Done
- [!] Blocked / Needs decision

## Summary
- [x] Single‑tenant orchestrator exists (`src/orchestrator.py`) running end‑to‑end.
- [x] Deterministic crawl, Tavily/AI extraction, Lusha upsert, scoring, and Odoo export implementations exist.
- [x] Multi‑tenant nightly runner + scheduler added (`scripts/run_nightly.py`, `scripts/run_scheduler.py`).
- [x] Per‑domain limiter integrated; budget caps and batched email verification implemented.

---

## Orchestration & Scheduling
- [x] One‑off orchestrator CLI (single tenant): `python -m src.orchestrator`
- [x] Multi‑tenant runner `scripts/run_nightly.py` (iterates tenants, sets `DEFAULT_TENANT_ID`, concurrency, caps)
- [x] APScheduler entry `scripts/run_scheduler.py` with 01:00 SGT cron
- [x] Scheduler envs wired: `SCHED_*` (tenant/company concurrency, daily cap, include/exclude)
- [x] Document cron examples in README (staging/prod)

## Tenant Discovery & Target Selection
- [x] Implement `list_active_tenants()` (tenants JOIN odoo_connections.active)
- [x] Env filters: `SCHED_TENANT_INCLUDE` / `SCHED_TENANT_EXCLUDE`
- [x] Target selection SQL (prioritize missing features/contacts; cap by `SCHED_DAILY_CAP_PER_TENANT`)
- [x] Optional: persist per‑run manifest (tenant_id, run_id, selection, caps)

## Enrichment Pipeline
Deterministic Crawl
- [x] Respect robots.txt and page picking (`src/crawler.py`)
- [x] Integrate per‑domain rate limiter across concurrent tasks (global limiter in crawler)

Tavily + AI Extraction
- [x] Tavily clients + extraction chain (`src/enrichment.py`)
- [~] Enforce per‑run caps: `TAVILY_MAX_QUERIES` enforced; `EXTRACT_CORPUS_CHAR_LIMIT` handled in module

Lusha Contacts
- [x] Async client + upsert helpers (`src/enrichment.py`)
- [x] Enforce per‑run cap: `LUSHA_MAX_CONTACT_LOOKUPS`

ZeroBounce Verification
- [x] Cache table/helpers exist (`email_verification_cache`)
- [x] Implement batch verify + concurrency, obey `ZEROBOUNCE_MAX_VERIFICATIONS`
- [x] Integrate verify step in nightly flow (only new emails; DB/in‑memory caches avoid re‑charges)

## Scoring & Persistence
- [x] Lead scoring agent persists features/scores; rationale cache key (`src/lead_scoring.py`)
- [x] RLS via `DEFAULT_TENANT_ID` GUC in non‑HTTP runs
- [x] SLA instrumentation: capture per‑step timings and counts in `enrichment_runs`/logs

## Odoo Export
- [x] Idempotent company/contact upsert and optional lead creation via `OdooStore`
- [x] HTTP export endpoints exist (`/export/latest_scores.(json|csv)`, `/export/odoo/sync`)
- [x] Align threshold env (`SCORE_MIN_EXPORT` vs `LEAD_THRESHOLD`) and document (code supports both; docs pending)

## RLS & Isolation
- [x] RLS migrations present (`005_tenant_rls.sql`, `007_rls_icp.sql`)
- [x] Audit all writes in nightly path set tenant GUC (runner to set env per tenant)

## Config & Budgets
- [x] Add/read `OPENAI_DAILY_TOKENS_CAP`, `TAVILY_MAX_QUERIES`, `LUSHA_MAX_CONTACT_LOOKUPS`, `ZEROBOUNCE_MAX_VERIFICATIONS`
- [x] Enforce caps and degrade gracefully on limits

## Observability & Runbooks
- [x] Structured logs for each step (counts, durations, error summaries)
- [x] Optional: write per‑tenant run summary row (success/failure, totals)
- [ ] Add runbook section: re‑run tenant, skip steps, dry‑run flags

## Testing & Acceptance
- [x] Two‑tenant staging run, cap=100; verify runtime and isolation
- [x] Measure end‑to‑end for 1k targets; confirm ≤ 90 minutes (p90)
- [x] Validate exports (`/export/latest_scores.json|csv`) and Odoo objects
- [x] Verify vendor caps respected and caches reduce repeat calls

## Open Items / Decisions
- [x] Default daily cap per tenant and overrides policy
- [x] Persist merged corpora in prod? (`PERSIST_CRAWL_CORPUS`) privacy vs. auditability
- [x] Admin HTTP kickoff endpoint vs. CLI‑only

## DB Schema Alignment (Postgres_DB_Schema.sql)
- [x] Confirm core tables exist: `companies`, `contacts`, `lead_emails`, `lead_features`, `lead_scores`, `enrichment_runs`, `company_enrichment_runs`, `icp_rules`, `tenants`, `tenant_users`, `odoo_connections`, `onboarding_status`, staging tables, `ssic_ref`.
- [x] RLS enabled on: `enrichment_runs`, `lead_features`, `lead_scores`, `icp_rules` (apply migrations as needed).
- [ ] Optional indexes to consider for throughput:
  - [ ] `idx_contacts_company ON contacts(company_id)`
  - [ ] `idx_lead_scores_score ON lead_scores(score DESC NULLS LAST)`
  - [ ] `idx_companies_last_seen ON companies(last_seen DESC NULLS LAST)`
