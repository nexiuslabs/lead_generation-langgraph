**TODO — Dev Plan 16: Apify LinkedIn Fallback**

Source: featurePRD16.md and featureDevPlan16.md

Legend
- [ ] Pending
- [~] In progress
- [x] Done
- [!] Blocked / Needs decision

## Configuration & Flags
- [x] Add envs: `APIFY_TOKEN`, `APIFY_LINKEDIN_ACTOR_ID`, `APIFY_SYNC_TIMEOUT_S`, `APIFY_DATASET_FORMAT`, `APIFY_DAILY_CAP`
- [x] Feature flags: `ENABLE_APIFY_LINKEDIN=true`, `ENABLE_LUSHA_FALLBACK=false`
- [~] Tenant titles: read from `icp_rules.payload.preferred_titles` with fallback `CONTACT_TITLES` (fallback env wired; per-tenant titles pending)
- [x] Add actor envs: `APIFY_COMPANY_ACTOR_ID`, `APIFY_EMPLOYEES_ACTOR_ID`, optional `APIFY_SEARCH_ACTOR_ID`
- [x] Add chain/debug envs: `APIFY_USE_COMPANY_EMPLOYEE_CHAIN`, `APIFY_EMPLOYEES_SCRAPER_MODE`, `APIFY_DEBUG_LOG_ITEMS`, `APIFY_LOG_SAMPLE_SIZE`

## Vendor Module
- [x] Create `src/vendors/apify_linkedin.py` with:
  - [x] `run_sync_get_dataset_items(payload, dataset_format, timeout_s)` using httpx
  - [x] `build_queries(company_name, titles)`
  - [x] `normalize_contacts(items)`
  - [x] Chain helpers: `company_to_profile_urls`, `contacts_via_company_chain`
- [x] Unit tests with mocked HTTP responses

## Enrichment Integration
- [x] Add `fetch_contacts_via_apify(company_id, company_name, titles)`
- [x] Upsert normalized rows into `contacts`
- [x] Verify provided emails and upsert into `lead_emails` (ZeroBounce)
- [x] Replace Lusha fallback path with Apify under flag
- [x] Enforce per-tenant `APIFY_DAILY_CAP` (read from `run_vendor_usage`/helper)

## Observability & Caps
- [x] Call `obs.bump_vendor(run_id, tenant_id, 'apify_linkedin', calls, errors, cost)`
- [x] Log per-company `run_event_logs` entries for contact discovery
- [~] Add usage to dashboards (Feature 8 SQL: vendor usage by run) (vendor usage table updated; dashboard wiring pending)

## Compliance
- [ ] Document LinkedIn ToS/robots guidelines in runbook
- [ ] Ensure suppression lists/retention apply; no guessed emails

## Testing & Rollout
- [x] Fault injection: HTTP 429/5xx/timeouts; run continues with degraded reason
- [x] Staging tenant pilot: measure coverage and cost; tune titles and batch size
- [x] Enable globally after validation; keep Lusha path off by default
