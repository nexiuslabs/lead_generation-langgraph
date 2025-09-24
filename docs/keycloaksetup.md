# Keycloak Setup (OIDC) for Nexius Apps

This guide configures Keycloak so the backend can authenticate users via OIDC, issue cookies, and (optionally) self‑register users through the admin API.

Prerequisites
- Keycloak base URL and admin access
- Planned realm name (e.g., `nexius`)
- Hostname(s) where the app is deployed

## 1) Create/Select Realm
- Realm: create or select your target realm (e.g., `nexius`).
- Keys: keep default RS256.
- Issuer URL: note the realm issuer (example)
  - `https://keycloak.example.com/realms/nexius`

## 2) App Client (Login/Refresh)
Create the OIDC client the backend uses for direct grant (password) and refresh.
- Client → Create
  - Client type: OpenID Connect
  - Client ID: `nexius-app` (example; must match `NEXIUS_CLIENT_ID`)
  - Access Type: Confidential
  - Client Authentication: ON (uses client secret)
  - Standard Flow: OFF (optional; backend does not need browser flow)
  - Direct Access Grants: ON (required for password grant)
  - Service Accounts: OFF
  - Valid Redirect URIs / Web Origins: not required for direct grant; leave empty
- Save → Credentials tab: copy Client Secret (for `NEXIUS_CLIENT_SECRET`).

### Protocol Mappers (on the App Client)
Add claims used by the backend.
- Roles → top‑level `roles` claim
  - Mapper Type: User Realm Role
  - Token Claim Name: `roles`
  - Claim JSON Type: String
  - Multivalued: ON
  - Add to ID token: ON; Add to Access token: ON
- Tenant → `tenant_id` claim
  - Mapper Type: User Attribute
  - User Attribute: `tenant_id`
  - Token Claim Name: `tenant_id`
  - Claim JSON Type: String (or Long if you prefer)
  - Add to ID token: ON; Add to Access token: ON
- Email
  - Ensure the built‑in “email”/“profile” mappers or scope are enabled so `email` is included.

## 3) Admin Client (Optional: /auth/register)
Only required if you will use the backend’s admin‑backed registration route.
- Client → Create
  - Client ID: `nexius-admin` (example; must match `NEXIUS_ADMIN_CLIENT_ID`)
  - Access Type: Confidential
  - Service Accounts Enabled: ON
- Credentials: copy Client Secret (for `NEXIUS_ADMIN_CLIENT_SECRET`).
- Service Account Roles:
  - Client Roles → realm‑management: assign at least `manage-users` and `view-users`.

## 4) Users
- Create users or sync from your IdP.
- Attributes: set `tenant_id` per user under the Attributes tab.
- Roles: assign realm roles as needed (will appear in `roles` claim).
- Email: ensure a valid email; set `emailVerified = true` if you want to skip required actions.

## 5) App Environment Variables
Set these on the backend to match your Keycloak realm and clients.
- `NEXIUS_ISSUER=https://keycloak.example.com/realms/nexius`
- `NEXIUS_CLIENT_ID=nexius-app`
- `NEXIUS_CLIENT_SECRET=<secret-from-credentials-tab>`
- `NEXIUS_AUDIENCE=nexius-app` (optional; enables audience validation)
- Cookies
  - `ACCESS_COOKIE_NAME=nx_access` (default)
  - `REFRESH_COOKIE_NAME=nx_refresh` (default)
  - `COOKIE_SECURE=true` (default; set false only for local HTTP)
- Admin registration (optional)
  - `NEXIUS_ADMIN_CLIENT_ID=nexius-admin`
  - `NEXIUS_ADMIN_CLIENT_SECRET=<admin-secret>`

Notes
- Token endpoint is derived from issuer unless overridden by `NEXIUS_TOKEN_URL`:
  - `<issuer>/protocol/openid-connect/token`
- Discovery: the app fetches `/.well-known/openid-configuration` from `NEXIUS_ISSUER`.

## 6) Endpoints Reference (derived from Issuer)
Assuming `ISSUER = https://keycloak.example.com/realms/nexius`:
- Discovery: `${ISSUER}/.well-known/openid-configuration`
- Auth: `${ISSUER}/protocol/openid-connect/auth` (used by Odoo UI login only)
- Token: `${ISSUER}/protocol/openid-connect/token`
- Userinfo: `${ISSUER}/protocol/openid-connect/userinfo`
- JWKS: from discovery `jwks_uri` (fallback `${ISSUER}/protocol/openid-connect/certs`)

## 7) Quick Test (Password Grant)
Replace placeholders and run:
```
curl -s -X POST "${ISSUER}/protocol/openid-connect/token" \
  -d grant_type=password \
  -d client_id="nexius-app" \
  -d client_secret="<secret>" \
  -d username="user@example.com" \
  -d password="<password>"
```
- Expect JSON with `id_token` or `access_token`. Decode and verify:
  - `iss` matches `NEXIUS_ISSUER`
  - `aud` contains `NEXIUS_CLIENT_ID` (when `NEXIUS_AUDIENCE` is set)
  - `email`, `tenant_id`, `roles` present per mappers

## 8) Troubleshooting
- 401 Invalid audience
  - Set `NEXIUS_AUDIENCE` to the client ID or ensure the token’s `aud` contains the client ID.
- 500 OIDC discovery/JWKS fetch failed
  - Verify `NEXIUS_ISSUER` is reachable from the backend; check TLS/Firewall.
- 401 Invalid credentials on password grant
  - Enable “Direct Access Grants” on the app client; confirm username/password.
- Missing `tenant_id` claim
  - Add the User Attribute mapper and set `tenant_id` on the user.
- Missing roles claim
  - Add “User Realm Role” mapper for `roles` and assign realm roles to users.
- Odoo SSO (optional) fails
  - Ensure Odoo OIDC provider uses the same `NEXIUS_ISSUER` and client ID/secret; verify callback URL in Odoo.

---
This setup aligns with the backend’s expectations in `app/auth.py` and `app/auth_routes.py`. Keep your issuer/client IDs consistent across Keycloak and the app’s environment.
