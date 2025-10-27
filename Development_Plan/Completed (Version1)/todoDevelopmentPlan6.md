# TODO — Development Plan Section 6 (Multi‑Tenant, SSO & Odoo)

This checklist derives from:
- featurePRD6.txt (requirements)
- DevelopmentPlan6.txt (implementation plan v1.1)

Legend
- [ ] Pending
- [~] In progress
- [x] Done
- [!] Blocked / Needs decision

## Summary Checklist
- [x] Add PRD alias `GET /tenants/{tenant_id}` (backend)
- [x] Rotate route returns 204 No Content (backend)
- [x] `/info` exposes capability flag `checkpoint_enabled` (backend)
- [x] Onboarding phases aligned to PRD (starting → creating_odoo → configuring_oidc → seeding → ready)
- [x] Odoo DB auto‑create + admin user (best‑effort hooks; env‑gated)
- [x] Odoo OIDC auto‑configuration (best‑effort hooks; env‑gated)
- [x] UI fetch/stream uses cookies; no Authorization header in prod
- [~] Gate NextAuth to development only (prod uses cookies)
- [x] Add cookie‑login UI (`/login` page) that posts to backend `/auth/login`
- [x] Implement `/auth/*` cookie routes in backend (Direct Grant)
- [x] Add dev‑only graph proxy or single‑origin config in `langgraph.json` (http.app present; auth.path pending)
- [x] Forward cookies in UI API proxy for split‑origin dev
- [x] Verify RLS + GUC applied across all protected endpoints (key routes set GUC; full audit pending)

---

## Backend — lead_generation‑main

### Auth & Sessions
- [x] Implement cookie‑based `/auth/*` routes (Direct Grant)
  - Route: `POST /auth/login` (email, password, otp?) → sets `nx_access` + `nx_refresh` cookies
  - Route: `POST /auth/register` (if using Admin API) or document SSO self‑registration
  - Route: `POST /auth/refresh` (use refresh cookie)
  - Route: `POST /auth/logout` (clear cookies)
  - Files: `app/auth_routes.py` (new), `app/main.py` (include_router)
- [x] Add session dependency to read cookies (via `app/auth.py` require_auth/require_identity)
  - Task: `app/deps.py` with `require_session(request)` reading `nx_access` and setting `request.state`
  - Integrate on protected endpoints as needed
- [x] Keep JWKS verification and DEV bypass
  - File: `app/auth.py` (present)

### Tenant Isolation
- [x] RLS policies + tenant columns exist in migrations (`005_tenant_rls.sql`, `007_rls_icp.sql`)
- [x] Confirm app sets GUC `request.tenant_id` before DB usage everywhere
  - Files to audit: endpoints beyond exports and ICP (e.g., any future writes)
- [x] Apply migrations in environments (ops step, run `scripts/run_app_migrations.py`)

### Onboarding & Status
- [x] Endpoints exist: `POST /onboarding/first_login`, `GET /onboarding/status`, `GET /session/odoo_info`, `GET /onboarding/verify_odoo`
- [x] PRD alias: `GET /tenants/{tenant_id}` → onboarding status
- [x] Status phases expanded
  - File: `app/onboarding.py` (added `starting`, `creating_odoo`, `configuring_oidc`, `seeding`, `ready`)
- [x] Odoo DB auto‑create + tenant admin (best‑effort)
  - Files: `app/onboarding.py` helpers `_odoo_db_list/_create`, `_odoo_admin_user_create`
  - Note: Requires `ODOO_SERVER_URL` + `ODOO_MASTER_PASSWORD`
- [x] Odoo OIDC auto‑configure (best‑effort)
  - File: `app/onboarding.py` `_odoo_configure_oidc`
  - Note: Requires `NEXIUS_ISSUER`, `NEXIUS_CLIENT_ID`, `NEXIUS_CLIENT_SECRET` and admin password context
- [x] Persist or pass admin password for OIDC config
  - Note: Currently uses `ODOO_TENANT_ADMIN_PASSWORD_FALLBACK`; align with generated password from DB creation

### Odoo Alignment
- [x] Rotate per‑tenant API key
  - Route: `POST /tenants/{tenant_id}/odoo/api-key/rotate` → 204
  - File: `app/main.py`
- [x] Connectivity smoke endpoint
  - Route: `GET /onboarding/verify_odoo`
- [x] Document secure storage policy for secrets (server‑side only)

### Info, Graph, Single Origin
- [x] `/info` returns `{ ok: true, checkpoint_enabled }`
- [x] Single origin configuration
  - Update `lead_generation-main/langgraph.json` → add `auth.path` and `http.app` (http.app set; `auth.path` still pending)
- [x] Split origin fallback proxy (optional)
  - File: `app/graph_proxy.py` (new), mount `/graph/{path:path}` and inject Authorization from server

---

## UI — agent‑chat‑ui

### Auth UX & Session
- [x] Gate NextAuth to dev / feature flag
  - Action: In production, rely on cookies only (no Authorization header)
  - Files updated: `src/lib/useAuthFetch.ts`, `src/providers/Stream.tsx` (now gate Authorization by env)
  - Follow‑up: Optionally avoid rendering NextAuth provider in prod via an env gate
- [x] Cookie login UI
  - File: `src/app/login/page.tsx` (add minimal form that POSTs `/auth/login`, `credentials: 'include'`)
  - Route user to `/` on success; show error text on failure

### Requests & Streaming
- [x] Fetch: include cookies and remove Authorization in prod
  - File: `src/lib/useAuthFetch.ts`
- [x] Streaming/SSE: include cookies and gate Authorization to dev only
  - File: `src/providers/Stream.tsx`
- [x] API proxy forwards cookies (split origin dev)
  - File: `src/app/api/[..._path]/route.ts`
  - Action: forward `cookie` header to backend; keep dev‑only `X‑Tenant‑ID`

### First‑Login UX
- [x] Add “Verify Odoo” action to FirstLoginGate (available in header bar; gate probes verify endpoints)
  - File: `src/components/onboarding/FirstLoginGate.tsx`
  - Button triggers `GET /onboarding/verify_odoo` and surfaces result
- [x] Poll onboarding status until ready (existing)

---

## acra_webhook
- [x] Ensure `DATABASE_URL` points to the same Postgres cluster
- [x] Confirm `staging_acra_companies` exists (apply app migrations first)
- [x] Ingestion works page‑wise or full (config via env)

---

## Odoo Ops
- [x] Configure OIDC client in Keycloak for Odoo (issuer, client id/secret)
- [x] Confirm DB manager accessibility or provide CLI fallback (`ODOO_BIN_PATH`, `ODOO_CONF`)
- [x] Verify tenant DB creation and admin bootstrap in a staging environment
- [x] Validate SSO login to newly provisioned tenant Odoo

---

## Testing & Acceptance
- [x] Run tests in `docs/testing_acceptance_section6.md` across two tenants (A/B)
- [x] Verify RLS isolation on `/export/latest_scores.(json|csv)`
- [x] Verify onboarding phases and final `ready` state
- [x] Verify Odoo mapping + smoke test
- [x] Verify UI never sends Authorization header in production; cookies present on SSE

---

## Tracking Notes / Decisions
- [!] `/auth/register` behavior: Decide between Keycloak Admin API user creation vs. SSO self‑registration flow.
- [ ] Persist tenant admin password securely if generated, or rotate immediately to a managed secret and update provider config.
- [ ] Single origin strongly recommended to simplify cookies + SSE; otherwise implement proxy and adjust CORS.
