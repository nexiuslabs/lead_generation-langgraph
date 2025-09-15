Odoo — OIDC (Nexius SSO) Setup

Objectives
- Configure Odoo as an OpenID Connect client of Nexius SSO so that users sign in once and access both Agent Chat UI and Odoo.

Prereqs
- Nexius SSO (OIDC) issuer URL: NEXIUS_ISSUER (e.g., https://sso.nexius.example)
- Admin access to Odoo and SSO console

Steps
1) Register Odoo in Nexius SSO
   - Create a new OIDC client.
   - Redirect URIs: follow Odoo docs for OAuth/OIDC (e.g., https://odoo.example.com/auth/openid/callback)
   - Assign custom claims to ID token: tenant_id (string/int), roles (array of strings)
   - Save client_id and client_secret.

2) Enable OAuth/OIDC in Odoo
   - Install/enable OAuth / OpenID Connect module
   - Configure provider:
     - Issuer: NEXIUS_ISSUER
     - Discovery: on (/.well-known/openid-configuration)
     - Client ID / Secret: from step 1
     - Scopes: openid email profile
     - Map email from token claim `email`
     - Optionally, set group mapping based on roles[] claim

3) Test
   - Visit Odoo login, choose Nexius SSO, authenticate
   - Verify session established
   - Inspect token (dev only) to confirm tenant_id and roles are present
   - Ensure audit fields in Odoo reflect your user identity (name/email)

Notes
- The Pre-SDR API never exposes Odoo credentials to the browser; server-side uses per-tenant DSN from `odoo_connections`.
- If you prefer, store a full Postgres DSN in `odoo_connections.db_name`. Otherwise, set `ODOO_BASE_DSN_TEMPLATE` and store only db_name.

Provisioning Notes
- Server-driven provisioning prefers two paths:
  - HTTP DB manager (when enabled): creates the database via JSON-RPC, sets admin credentials, installs base modules, and can auto-configure OIDC.
  - Template-clone fallback (supported): when the DB manager is disabled (`list_db=false`) or not reachable, the backend clones a configured `ODOO_TEMPLATE_DB` at the Postgres layer and then sets admin credentials via XML-RPC or directly in the DB. This is the supported fallback instead of invoking `odoo-bin`.
- We do not rely on `odoo-bin` CLI for provisioning. The template-clone approach is simpler to operate across environments and works over a secure Postgres connection (optionally via SSH tunnel) using `ODOO_BASE_DSN_TEMPLATE` or `ODOO_POSTGRES_DSN`.
