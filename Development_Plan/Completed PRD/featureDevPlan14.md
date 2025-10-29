**Dev Plan — Feature 14: Acceptance Criteria (Pre‑SDR)**

Source: featurePRD14.md

Objective: Implement measurable checks and tooling to verify the acceptance criteria end‑to‑end (pipeline, auth/isolation, observability, UX/exports, Odoo readiness, SSO). Provide code paths and scripts to compute metrics and produce a pass/fail report for pilot sign‑off.

**Strategy**
- Compute metrics directly from Postgres using authoritative tables (`companies`, `company_enrichment_runs`, `lead_emails`, `lead_scores`, `icp_candidate_companies`, `enrichment_runs`).
- Expose a lightweight API for a JSON acceptance report; provide a CLI script to run checks cross‑tenant.
- Leverage Feature 8 observability (if available) for run scoping; otherwise use timestamp heuristics.
- Exports acceptance: availability and correctness; redaction is out of scope per Feature 12 decision.

**Metrics & Queries**
- Window selection:
  - If `enrichment_runs` has recent `started_at`, use the latest run per tenant as the window.
  - Else, use a configurable time window (e.g., last 24h).
- SQL snippets:
  - Global candidates (MV): `SELECT COUNT(*) FROM icp_candidate_companies;`
  - Per‑tenant shortlist (scored): `SELECT COUNT(*) FROM lead_scores;`
  - Domain presence (≥70%): `SELECT SUM(CASE WHEN c.website_domain IS NOT NULL AND c.website_domain<>'' THEN 1 ELSE 0 END) AS with_domain, COUNT(*) AS total FROM companies c JOIN lead_scores s ON s.company_id=c.company_id;`
  - About text presence (≥60%): `SELECT SUM(CASE WHEN COALESCE(NULLIF(TRIM(about_text), ''), NULL) IS NOT NULL THEN 1 ELSE 0 END) AS with_about, COUNT(*) AS total FROM company_enrichment_runs r JOIN lead_scores s ON s.company_id=r.company_id;`
  - Verified/unknown email availability (≥40%): `SELECT SUM(CASE WHEN EXISTS (SELECT 1 FROM lead_emails e WHERE e.company_id=s.company_id AND COALESCE(e.verification_status,'unknown') IN ('valid','unknown')) THEN 1 ELSE 0 END) AS companies_with_email, COUNT(*) AS total FROM lead_scores s;`
  - Bucket distribution sanity (no bucket >70%): `SELECT bucket, COUNT(*) FROM lead_scores GROUP BY bucket;`
  - Rationale presence: `SELECT SUM(CASE WHEN NULLIF(TRIM(rationale),'') IS NOT NULL THEN 1 ELSE 0 END) AS with_rationale, COUNT(*) AS total FROM lead_scores;`

**Rationale Cache (optional for measurable reuse)**
- Migration: `CREATE TABLE IF NOT EXISTS rationale_cache (cache_key TEXT PRIMARY KEY, rationale TEXT NOT NULL, created_at TIMESTAMPTZ DEFAULT NOW());`
- lead_scoring integration: Look up `cache_key` before LLM call; insert on miss; metric via joining `lead_scores.cache_key` to `rationale_cache`.

**API: Acceptance Report**
- Add `GET /acceptance/report` to `app/main.py` returning `{ mv_candidates, shortlisted, domain_rate, about_rate, email_rate, bucket_counts, rationale_rate }` computed by queries above (RLS applies tenant scoping).

**CLI: scripts/acceptance_check.py**
- Async script using `asyncpg` that verifies thresholds and exits non‑zero on failure (usable in CI/cron).

**SSO, Isolation, and UX Checks**
- SSO & Isolation: Without cookies → `/info` returns 401; with cookie → 200. RLS smoke with `DEFAULT_TENANT_ID` confirms scoping.
- UX & streaming: Manual QA — chat loads, streams, errors toasts; export buttons download CSV/JSON.
- Odoo readiness: `/onboarding/verify_odoo` p95 ≤ 60s; error messages actionable.

**Observability Hooks**
- Ensure dashboards cover counts, durations, error rates, vendor usage, token spend per Feature 8. Alerts for MV failure, error rate, quotas, OpenAI 429s, low candidates.

**Rollout**
- Add acceptance endpoint and CLI script; (optional) add `rationale_cache` and integrate caching; create Metabase dashboard; run nightly across two tenants and capture three consecutive passes.

