**Feature PRD 11 — Error Handling & Fallbacks**

- **Objective:** Degrade gracefully across data and vendor failures to maximize usable output, preserve cost controls, and provide transparent status.
- **Primary Users:** Engineering, Ops, On-call; implicit benefit to end users via higher successful coverage.

**Error Taxonomy**
- `DATA_EMPTY` (ICP MV empty, missing domain, thin content)
- `CRAWL_*` (blocked, timeout, robots, DNS/TLS)
- `VENDOR_*` (TAVILY, LUSHA, ZEROBOUNCE — quota, 4xx, 5xx)
- `OPENAI_*` (rate_limit, timeout, server_error)
- `EXPORT_*` (Odoo constraints, connectivity)

**Fallback Chain (per company)**
- ICP MV empty → run industry-only fallback; mark run `degraded=true`.
- No domain found → query Lusha company search (if allowed); otherwise skip contacts and down-weight.
- Crawl blocked/thin → Tavily multi-URL; if still thin, attempt Lusha contacts.
- ZeroBounce API error → mark emails `unknown`, enqueue for retry next run.
- Lusha quota/HTTP errors → proceed without contacts; lower contact-weight in scoring to avoid bias.

**Execution Semantics**
- Retries: exponential backoff with jitter; limit attempts per vendor error class.
- Circuit breakers: trip vendor usage for a tenant when repeated 5xx/429; cool-off window.
- Idempotency: deterministic cache keys ensure re-runs do not duplicate work/charges.
- Degradation flags: annotate `enrichment_runs.status` and per-company result with `degraded_reason`.
- Partial success: persist features and provisional scores even when a stage fails (with provenance).

**Observability**
- Log structured events with `error_code`, `fallback_taken`, and `degraded` fields (see PRD 8).
- Stage-level counters for retries and fallbacks; alert thresholds for chronic degradation.

**Odoo Export Fallbacks**
- Missing DB or constraint failure (e.g., `autopost_bills`):
  - Set defaults server-side; skip non-essential fields; retry subset export.
  - If DB missing, disable export for tenant and notify Ops.

**Testing Strategy**
- Fault injection (mock vendor 4xx/5xx, timeouts) in unit/integration tests.
- End-to-end dry-runs on a small tenant with induced errors to verify degradation and alerting.

**Acceptance Criteria**
- For each error class above, the fallback is executed and logged; run completes with `degraded=true` where applicable.
- Vendor circuit breaker prevents >3 consecutive 429 retries within 1 minute.
- Odoo export failures do not abort the entire run; partial exports succeed.

**Open Questions**
- Tenant-configurable fallback order (e.g., allow skipping Lusha entirely).
- Global vs per-tenant circuit breaker thresholds.

