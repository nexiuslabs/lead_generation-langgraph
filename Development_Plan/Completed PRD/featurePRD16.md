**Feature PRD 16 — Replace Lusha Fallback with Apify LinkedIn Actor**

- **Objective:** Replace Lusha-based contact discovery with an Apify Actor that scrapes public LinkedIn profiles to retrieve role-qualified contacts for shortlisted companies, while adhering to compliance and cost guardrails.
- **Primary Users:** Ops, SDRs, Enrichment pipeline operators.
- **Non-Goals:** Circumventing website or platform Terms; personal email discovery; bulk non-consensual harvesting.

**Vendor & API**
- **Platform:** Apify
- **Auth:** `APIFY_TOKEN` (server-side secret; never exposed to the browser)
- **Actors & Endpoints:**
  - Company (by name): `harvestapi~linkedin-company`
    - `POST /v2/acts/harvestapi~linkedin-company/run-sync-get-dataset-items?token=TOKEN&format=json`
    - Body: `{ "companies": ["<Company Name>"] }`
  - Employees (by company URL): `harvestapi~linkedin-company-employees`
    - `POST /v2/acts/harvestapi~linkedin-company-employees/run-sync-get-dataset-items?token=TOKEN&format=json`
    - Body: `{ "companies": ["https://www.linkedin.com/company/<slug>/"], "maxItems": 50, "profileScraperMode": "Full ($8 per 1k)" }`
  - Profile details (by profile URLs): `dev_fusion~linkedin-profile-scraper`
    - `POST /v2/acts/dev_fusion~linkedin-profile-scraper/run-sync-get-dataset-items?token=TOKEN`

**Input Strategy**
- Default: chain company-by-name → employees → profile details. Filter contacts by preferred titles after profile scrape.
- Fallback: when chain disabled or insufficient data, build queries from company name + preferred titles and call the profile actor directly.
- Example query: `"{company_name}" AND (Founder OR CEO OR "Head of Growth")`
- Batch and cap per tenant to respect budgets.

**Output Mapping**
- Parse dataset items to extract fields:
  - `full_name`, `headline/title`, `company_current`, `profile_url`, `location`, `about/snippet`.
  - Email is typically not present on public profiles; do not infer or guess addresses. If email is provided by the Actor, mark as `source=apify_linkedin` and verify via ZeroBounce before use.
- Persist into:
  - `contacts` (new or existing table): `contact_id, tenant_id, company_id, full_name, title, linkedin_url, location, source, created_at`.
  - `lead_emails` (optional): only when email present and verified; store `status` from ZeroBounce.
  - `contact_discovery_events` (new table): `run_id, tenant_id, company_id, contact_id, vendor='apify_linkedin', status, error_code, duration_ms, raw_ref (dataset item key/id)`.

**Pipeline Placement**
- New fallback in Error Handling chain (PRD 11):
  - Crawl → Tavily fallback → (if contacts still missing) Apify LinkedIn Actor → proceed to scoring with contact features available.
- Disable Lusha code paths behind feature flag: `ENABLE_LUSHA_FALLBACK=false` (default). New flag: `ENABLE_APIFY_LINKEDIN=true`.

- **Configuration**
  - Flags: `ENABLE_APIFY_LINKEDIN=true`, `APIFY_USE_COMPANY_EMPLOYEE_CHAIN=true`, `ENABLE_LUSHA_FALLBACK=false`
  - Actors: `APIFY_COMPANY_ACTOR_ID`, `APIFY_EMPLOYEES_ACTOR_ID`, `APIFY_LINKEDIN_ACTOR_ID`, optional `APIFY_SEARCH_ACTOR_ID`
  - Other env: `APIFY_SYNC_TIMEOUT_S=600`, `APIFY_DATASET_FORMAT=json`, `APIFY_EMPLOYEES_SCRAPER_MODE`, `CONTACT_TITLES`
  - Caps: `APIFY_DAILY_CAP` per-tenant
- Tenant overrides stored in DB (policy/config table) to adjust titles, caps, and inclusion.

**Compliance & ToS**
- Respect LinkedIn Terms of Service and robots directives. This feature must only be used for public data, at responsible rates, and within the allowed legal framework of the operating region and platform.
- PDPA/Privacy: Do not store personal emails gathered from profiles unless explicit consent is documented; prefer business role data only. Apply suppression lists (PRD 9) and retention policies.

**Cost Guardrails (PRD 10 alignment)**
- Track vendor usage as `vendor='apify_linkedin'` in `run_vendor_usage`.
- Per-tenant daily caps (calls and item limits); degrade gracefully when budget is low: reduce titles set, reduce batch size, or defer to nightly.

**Observability (PRD 8 alignment)**
- Log stage `contact_discovery` with `vendor=apify_linkedin`, success/error counts, p50/p95, and cost estimates.
- Store `trace_id` for correlation if used in LLM post-processing.

**APIs & Internal Modules**
- Module: `src/vendors/apify_linkedin.py`
  - `run_sync_get_dataset_items(payload, dataset_format, timeout_s)` with payload variants + `run-sync` fallback
  - `_run_actor_items(actor_id, payload, ...)` helper for specific actors
  - `company_to_profile_urls(company_name, max_items)`
  - `contacts_via_company_chain(company_name, titles, max_items)`
  - `build_queries(company_name, titles)`, `normalize_contacts(items)`
- Integrated via `node_apify_contacts` in `src/enrichment.py` (prefers chain; falls back to queries path)

**Failure & Fallback Behavior**
- 4xx/5xx or quota: mark vendor event, skip contact addition for that company, continue scoring with reduced contact weight.
- Timeout: one retry with jitter; on second failure, skip gracefully.

**Logging & Audit**
- Console logs: company URL resolved, employees count, profile actor received/filtered; samples when `APIFY_DEBUG_LOG_ITEMS=true` (`APIFY_LOG_SAMPLE_SIZE`)
- DB: `run_vendor_usage` (vendor='apify_linkedin'), `run_event_logs` (stage='contact_discovery', event='vendor_call')

**Acceptance Criteria**
- With Lusha disabled and Apify enabled, nightly runs complete with no pipeline aborts.
- At least one role-qualified LinkedIn contact is found for ≥25% of companies where website content is thin (tunable by industry).
- ZeroBounce is only called when email is present; no guessed emails.
- Vendor usage and costs recorded; caps enforced.

**Implementation Plan**
- Phase 1: Add module + envs; integrate into contact discovery; feature-flagged rollout to dev tenant.
- Phase 2: Add observability (run_vendor_usage, stage stats) and caps; update QA sampling to include contact validity check (profile URL reachable).
- Phase 3: Tenant-tunable titles and per-tenant caps; docs and runbooks.
