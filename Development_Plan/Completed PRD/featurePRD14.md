**Feature PRD 14 — Acceptance Criteria (Pre‑SDR)**

- **Objective:** Define measurable acceptance criteria for the pre‑SDR pipeline and UI prior to pilot sign‑off.

**Pipeline Output**
- Nightly job produces a shortlist with ≥500 MV candidates available globally; per‑tenant daily cap delivered on schedule.
- For each shortlisted company:
  - Domain present for ≥70% of records.
  - About text present for ≥60%.
  - ≥1 verified/unknown business email for ≥40% of companies.
- Score buckets: no single bucket >70% of total; thresholds respected.
- Rationale present; cache reused (≥80% hit on repeated features).

**Isolation & Auth**
- Multi‑tenant isolation verified (RLS tests) and SSO works end‑to‑end.
- Lusha invoked only when site/contacts missing; costs & call counts logged.

**Observability**
- Dashboards show: counts, durations, error rates, vendor usage, token spend.
- Alerts configured for MV refresh failure, error rate spikes, vendor quotas, OpenAI rate limit, low candidate counts.

**UX & Exports**
- Streaming progress visible in chat; commands drive flows; role guard respected.
- Export shortlist (CSV/JSON) available and redacted per viewer role; ops can view unmasked where allowed.

**Odoo Readiness**
- First‑login to “ready” (Odoo connection verified) ≤60s p95; failures expose actionable error states.
- `odoo_connections` exists and passes read/write smoke test per new tenant.

**SSO (Nexius SSO across Chat UI + Odoo)**
- Sign‑in required: both apps blocked without valid session; no unauthenticated access in production.
- Single sign‑on: session works across Chat UI and Odoo in same browser context within lifetimes.
- Tenant resolution: JWT includes `tenant_id`; no tenant switcher in prod; switching requires re‑auth.
- Token validation: LangGraph validates JWT via JWKS; `X‑Tenant‑ID` ignored in production.
- Roles enforced: `roles[]` mapped to app permissions; unauthorized actions blocked.
- Token lifecycle: refresh/renew does not break active usage; streaming reconnects get fresh auth.
- Logout: terminates access to both apps; re‑login required.
- Auditing: server‑side writes tag `tenant_id` and user (sub/email) where applicable.
- Transport security: HTTPS enforced; secure cookies and appropriate SameSite/CSRF protections.

**Exit Criteria**
- Three consecutive nightly runs meet the above thresholds across two tenants.
- Cost within budgets (PRD 10); no P1 incidents open; runbooks validated by dry‑runs (PRD 8).

