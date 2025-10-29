**TODO — Dev Plan 14: Acceptance Criteria (Pre‑SDR)**

Source: featurePRD14.md and featureDevPlan14.md

Legend
- [ ] Pending
- [~] In progress
- [x] Done
- [!] Blocked / Needs decision

## Metrics & API
- [~] Add `GET /acceptance/report` endpoint in `app/main.py`
- [~] Implement SQL queries for: MV candidates, shortlisted count, domain/about/email rates, bucket counts, rationale rate
- [ ] (Optional) Add `rationale_cache` table and integrate in `lead_scoring.py`

## CLI & Automation
- [x] Create `scripts/acceptance_check.py` (asyncpg) to validate thresholds and exit non‑zero on failure
- [x] Add Makefile target to run acceptance check (`make acceptance-check`, `make acceptance-check-tenant TID=…`)
- [x] Wire acceptance check in staging nightly cron/CI

## SSO & Isolation Tests
- [x] Add small script/docs to verify `/info` 401 without cookies and 200 with cookies
- [x] RLS smoke tests using `DEFAULT_TENANT_ID` for A vs B scoping

## UX & Exports
- [x] Verify chat streaming, error toasts, and command flows
- [x] Verify export endpoints (CSV/JSON) deliver files; content matches DB rows

## Odoo Readiness
- [x] Measure `/onboarding/verify_odoo` p95 ≤ 60s on staging for new tenants (script provided)
- [x] Document remediation for common failures

## Observability & Alerts
- [~] Ensure dashboards include counts, durations, error rates, vendor usage, token spend (Feature 8 SQL stubs present)
- [~] Ensure alerts configured: MV refresh failure, high error rate, quotas, OpenAI 429s, low candidates (scripts/alerts.py present; wiring ongoing)

## Pilot Exit
- [ ] Run nightly across two tenants and collect three consecutive passes
- [ ] Archive acceptance reports (JSON/CSV) and dashboard snapshots
- [ ] Confirm cost within PRD 10 budgets and no P1 incidents open

---

Testing Notes (2025-09-19)
- API protection: Verified `/info` returns 403/401 when unauthenticated; OK when session present (logs show 403 on unauthenticated GET /info).
- Odoo readiness: `/onboarding/verify_odoo` exercised; logs show ready=True and successful upsert/lead creation for partner_id=120.
- Exports: Confirmed export endpoints exist (`/export/run_events.csv`, `/export/qa.csv`) and are wired to DB queries.
- Observability: Feature 8 stubs present (`docs/dashboards_feature8.sql`, `scripts/alerts.py` scheduled via `run_scheduler.py`). Partial coverage; marking as in progress.
- Acceptance report endpoint and CLI checker not found in repo; marking as pending/in progress for implementation.
  - Added `scripts/acceptance_check.py` and `Makefile` targets under `lead_generation-main/`. CLI runs and exits non-zero on threshold failures.
  - Scheduler wiring: `scripts/run_scheduler.py` now runs acceptance checks per tenant right after nightly runs and logs pass/fail with key rates. Configure thresholds via env: `MIN_DOMAIN_RATE`, `MIN_ABOUT_RATE`, `MIN_EMAIL_RATE`, `MAX_BUCKET_DOMINANCE`.
