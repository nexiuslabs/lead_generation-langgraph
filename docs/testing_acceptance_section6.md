Testing & Acceptance — Section 6 (SSO, Tenancy, Odoo)

Smoke
- UI redirect: Open http://localhost:3000 → should redirect to Nexius SSO (or dev login) and return authenticated UI.
- API protection: curl -i http://localhost:2024/info without Authorization → 401. With a valid Bearer token → 200 {"ok": true}.

Tenant Isolation
1) Seed two tenants and rules (see scripts/seed_tenants.sql). Refresh MV.
2) Issue two tokens with different tenant_id claims (A and B). For each token:
   - curl -H "Authorization: Bearer <tokenA>" http://localhost:2024/export/latest_scores.json → returns only tenant A rows
   - curl -H "Authorization: Bearer <tokenB>" http://localhost:2024/export/latest_scores.json → returns only tenant B rows
3) Connect to DB and run: SELECT current_setting('request.tenant_id', true); during a request to confirm it is set.

Odoo Sync
- Insert `odoo_connections` for tenant A.
- Trigger enrichment/scoring for tenant A and confirm server writes to the configured Odoo DB.
- Re-auth as tenant B (different token), configure its DSN, and confirm writes go to B’s DB.

Security
- Serve UI/API over HTTPS in staging/prod.
- Confirm Authorization header is present on all API requests; NEXT_PUBLIC_ENABLE_TENANT_SWITCHER=false in prod builds.

Rollout
- Enable in staging with 1–2 tenants; exercise the flows above.
- Monitor logs/errors; then promote to production.

