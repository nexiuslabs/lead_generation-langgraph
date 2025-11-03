# TODO — Dev Plan 11: Error Handling & Fallbacks

Source: featurePRD11.md and featureDevPlan11.md

Legend
- [ ] Pending
- [~] In progress
- [x] Done
- [!] Blocked / Needs decision

## Configuration
- [x] Add envs: `RETRY_MAX_ATTEMPTS`, `RETRY_BASE_DELAY_MS`, `RETRY_MAX_DELAY_MS`
- [x] Add envs: `CB_ERROR_THRESHOLD`, `CB_COOL_OFF_S`, `CB_GLOBAL_EXEMPT_VENDORS`
- [x] Add flags: `ENABLE_TAVILY_FALLBACK`, `ENABLE_APIFY_LINKEDIN` (or `ENABLE_LUSHA_FALLBACK`)
- [x] Add `ODOO_EXPORT_SET_DEFAULTS`

## Utilities
- [x] Create `src/retry.py` with `with_retry`, `BackoffPolicy`, `CircuitBreaker`
- [ ] Unit tests for retry + breaker behavior (optional if infra exists)

## Enrichment Fallback Chain
- [x] Implement `enrich_company(cid, tenant_id, run_id)` orchestration with:
  - [x] Deterministic crawl (+ per-domain limiter already exists)
  - [x] Tavily fallback when thin/blocked (guarded by flag)
  - [x] Contact discovery via Apify LinkedIn (preferred) or Lusha when missing contacts (Apify placeholder; Lusha active)
  - [x] ZeroBounce verification try/catch → mark unknown on error
  - [x] Collect `degraded_reasons` and return
- [x] Persist `degraded_reasons` into `company_enrichment_runs`
- [x] Set run-level `degraded=true` if any company is degraded

## Vendor Wrappers
- [~] Wrap Tavily client calls with retry + breaker + obs (obs + degrade present; retry/breaker not applied)
- [x] Wrap contact discovery vendor with retry + breaker + obs (Lusha via `with_retry` + circuit breaker + obs)
- [~] Wrap ZeroBounce verify with retry + obs (obs + try/catch present; no retry)
- [x] Wrap OpenAI calls for rationale with vendor usage bumps

## Odoo Export Resilience
- [x] Add safe write (strip problematic fields on failure)
- [x] On DB missing/unreachable: disable export for tenant for this run; log and continue

## Observability (Feature 8 hooks)
- [x] Add `obs.log_event` at start/finish/error of stages with `error_code` and `extra.fallback_taken`
- [x] Add `obs.bump_vendor` for each vendor call (calls/errors/tokens/costs)
- [x] Call `obs.aggregate_percentiles` before finalizing run

## Testing
- [ ] Fault injection: simulate 429/5xx/timeouts for vendors and assert fallbacks/degradation
- [ ] End-to-end dry-run on staging tenant; verify non-abort and QA samples creation

## Documentation
- [ ] Update runbooks with fallback chain and remediation guidance
- [ ] Document envs and feature flags in README/ops docs
