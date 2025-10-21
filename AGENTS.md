Agents Guide — lead_generation-main

Purpose
- Pre-SDR pipeline: normalize → ICP candidates → deterministic crawl + Tavily/Apify → ZeroBounce verify → scoring + rationale → export → optional Odoo sync.

Run API
- python -m venv .venv && source .venv/bin/activate
- pip install -r requirements.txt
- uvicorn app.main:app --host 0.0.0.0 --port 2024
- Endpoints: /export/latest_scores.csv, /docs

Run Orchestrator (one-off)
- source .venv/bin/activate
- python -m src.orchestrator

Key Env Vars (src/settings.py)
- POSTGRES_DSN (required)
- OPENAI_API_KEY, LANGCHAIN_MODEL=gpt-4o-mini, TEMPERATURE=0.3
- TAVILY_API_KEY?, ZEROBOUNCE_API_KEY?
- APIFY_TOKEN (required for Apify), ENABLE_APIFY_LINKEDIN=true
- Optional: APIFY_INPUT_JSON to pass a custom actor input JSON template.
  - Use placeholders: %%QUERY%% for a single query string, %%QUERIES%% for an array of queries.
- Optional: APIFY_SEARCH_ACTOR_ID for resolving LinkedIn profile URLs when the main actor requires `profileUrls`.
- Optional: APIFY_COMPANY_FINDER_BY_DOMAIN_ACTOR_ID for resolving a LinkedIn company URL from a website domain (Top‑10/Next‑40). Default: `s-r~free-linkedin-company-finder---linkedin-address-from-any-site`.
- Optional: APIFY_DEBUG_LOG_ITEMS=true to log a small sample of Apify dataset items and normalized contacts. Control size via APIFY_LOG_SAMPLE_SIZE (default 3).
- Optional: APIFY_USE_COMPANY_EMPLOYEE_CHAIN=true to use the chain company-by-name → employees → profile-details.
- APIFY_COMPANY_ACTOR_ID=harvestapi~linkedin-company, APIFY_EMPLOYEES_ACTOR_ID=harvestapi~linkedin-company-employees, APIFY_LINKEDIN_ACTOR_ID=dev_fusion~linkedin-profile-scraper
- ENABLE_LUSHA_FALLBACK=false (optional; set true only if you want Lusha as a fallback), LUSHA_API_KEY? (optional)
- ICP_RULE_NAME=default, CRAWL_MAX_PAGES=6, EXTRACT_CORPUS_CHAR_LIMIT=35000
- ODOO_POSTGRES_DSN (or resolve per-tenant via odoo_connections)

Migrations
- Apply multi-tenant + MV: app/migrations/004_multi_tenant_icp.sql
- Odoo columns: app/migrations/001_presdr_odoo.sql

Tenancy & Auth (Section 6)
- Production: Validate Nexius SSO JWT, set request.state.tenant_id and roles.
- Enforce RLS/filters on tenant-owned tables; set GUC request.tenant_id per request.
- Dev: X-Tenant-ID header may be accepted for local testing only.

Common Ops
- Refresh MV: REFRESH MATERIALIZED VIEW CONCURRENTLY icp_candidate_companies;
- Export shortlist: curl "http://localhost:2024/export/latest_scores.csv?limit=200" -o shortlist.csv
- Logs: tail -f .logs/*.log

Troubleshooting
- Postgres connect errors: verify POSTGRES_DSN and DB reachable.
- Tavily/Apify/ZeroBounce: missing keys → fallbacks/pathways skip gracefully; check settings flags.

Apify Usage
- Nightly ACRA/ACRA Direct: use existing Apify name→company→employees→profiles chain as before.
- Top‑10/Next‑40: domain is known from ICP; resolve LinkedIn company URL via `APIFY_COMPANY_FINDER_BY_DOMAIN_ACTOR_ID`, then employees→profiles with title filtering.

Scheduler & Cron
- Use the async scheduler entry: `python lead_generation-main/scripts/run_scheduler.py`
- Configure start time via `SCHED_START_CRON` (default `0 1 * * *` for 01:00 SGT)
- Limits & caps:
  - `SCHED_DAILY_CAP_PER_TENANT` (default 20 in .env)
  - `SCHED_COMPANY_BATCH_SIZE` (per-batch company count; default 1)
  - Vendor caps (coarse): `TAVILY_MAX_QUERIES`, `APIFY_DAILY_CAP`, `LUSHA_MAX_CONTACT_LOOKUPS` (only if Lusha fallback enabled)
  - ZeroBounce: `ZEROBOUNCE_MAX_VERIFICATIONS`, `ZEROBOUNCE_BATCH_SIZE`

Admin Kickoff Endpoint
- POST `/admin/runs/nightly` (requires admin role)
  - Run all tenants: `curl -X POST http://localhost:2024/admin/runs/nightly -b 'nx_access=...'`
  - Run a single tenant: `curl -X POST 'http://localhost:2024/admin/runs/nightly?tenant_id=123' -b 'nx_access=...'`
