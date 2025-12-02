# To‑Do List — Auth + LangGraph Bugfix (Tracking)

Purpose: Track implementation of the multi‑tenant Keycloak SSO integration and tenant propagation into LangGraph runs per `Development_Plan/Auth_Langgraph_Bugfix.md`.

Status legend: [ ] pending · [~] in progress · [x] completed

## 0) Scope & Outcomes
- [x] Validate Keycloak JWTs on the LangGraph server (no more noop auth).
- [x] Ensure tenant is available inside nodes via `config.configurable.tenant_id` and thread `metadata.tenant_id`.
- [x] FastAPI fallback injects tenant when Keycloak token lacks `tenant_id`.
- [ ] UI always starts runs with tenant context; no first‑run `tenant_id=None` in node logs.

## 1) LangGraph Auth Plugin
- [x] Add Keycloak JWT auth plugin (JWKS) and expose `tenant_id` in auth context.
  - File(s): `src/auth.py`
  - Accept: Valid tokens decode; invalid tokens return 401 (unless dev bypass is enabled). `ctx.user.tenant_id` exists when claim/role is present.
- [x] Register auth plugin in server config.
  - File(s): `langgraph.json` — add `"auth": { "path": "src/auth.py:auth" }` and `"env": ".env"`.
  - Accept: LangGraph server starts without "auth type=noop"; uses custom auth.
- [x] Add `cryptography` dependency for PyJWT RSA verification.
  - File(s): `requirements.txt`
  - Accept: `pip install -r requirements.txt` succeeds.
- [x] Dev bypass for local flows (no hard 401 on missing/invalid token when `DEV_AUTH_BYPASS=true`).
  - File(s): `src/auth.py`
  - Accept: In dev, runs proceed and tenant can be injected via proxy/configurable.

## 2) FastAPI Graph Proxy — Tenant Injection (Fallback)
- [x] Inject tenant into run payload at `config.configurable.tenant_id`; mirror in `metadata/context` (best‑effort).
  - File(s): `app/graph_proxy.py`
  - Accept: Proxy logs show forward with tenant; runs created via `/graph` carry tenant in configurable.
- [x] Ensure proxy is mounted and used in dev.
  - File(s): `.env` — set `ENABLE_GRAPH_PROXY=true`, `LANGGRAPH_REMOTE_URL=http://127.0.0.1:8001`.
  - Accept: `app.main` logs " /graph proxy routes enabled"; UI targets Next proxy to FastAPI.

## 3) FastAPI Auth/Gateway
- [x] `require_auth` / `require_identity` set `request.state.tenant_id` from JWT or dev header.
  - File(s): `app/auth.py`
  - Accept: `request.state.tenant_id` populated for downstream proxy injection.

## 4) UI Streaming (Agent Chat)
- [x] Forward Authorization to LangGraph in dev when enabled.
  - File(s): `agent-chat-ui/src/providers/Stream.tsx` — already supports `NEXT_PUBLIC_USE_AUTH_HEADER=true`.
  - Accept: Requests contain `Authorization: Bearer <idToken>` in dev; no tokens required in prod when proxying via FastAPI.
- [x] Pass tenant in run config and pre‑create tenant‑scoped thread.
  - File(s): `agent-chat-ui/src/providers/Stream.tsx` — set `config.configurable.tenant_id`; pre‑create thread with `metadata.tenant_id`.
  - Accept: First run is attached to a thread carrying tenant metadata; configurable.tenant_id present.
- [x] Block first run until tenant is known and patch existing threads if needed.
  - File(s): `agent-chat-ui/src/providers/Stream.tsx` (guard + PATCH `/threads/{id}`)
  - Accept: No runs begin with `tenant_id=None`; LangGraph logs show `tenant_context ... tenant_id=<id>` for every first run.

## 5) Environment / Keycloak
- [ ] Configure Keycloak protocol mapper (tenant claim or role mapping).
  - Owner: Ops/IDP
  - Accept: Access tokens include `tenant_id` claim or a `realm_access.roles` entry like `tenant-1105`.
- [ ] Set environment variables.
  - Files: `.env`
  - Values: `KEYCLOAK_URL`, `KEYCLOAK_CLIENT_ID`, `ENABLE_GRAPH_PROXY=true`, `LANGGRAPH_REMOTE_URL`, `DEV_AUTH_BYPASS=true` (dev only), `NEXT_PUBLIC_USE_API_PROXY=true`, `NEXT_PUBLIC_USE_AUTH_HEADER=true`.
  - Accept: Services restart cleanly with new env.

## 6) Verification
- [ ] LangGraph logs show authenticated runs and tenant detection.
  - Accept: For a new chat, logs contain `tenant_context ... tenant_id=<id>`; no `tenant_id=None` in `return_user_probe`.
- [ ] Return‑user behavior observed when cache exists.
  - Accept: For tenant with prior snapshots/top‑10 preview, `return_user_probe` logs `use_cached=True` or `profiles_loaded_no_candidates` (not `cached=0` due to missing tenant).

## 7) Tests / Smoke
- [ ] Add smoke script for `/graph` proxy + stream with configurable tenant.
  - File(s): `scripts/smoke_sse_proxy.py` (reuse) — extend to print detected tenant from node logs.
  - Accept: Script runs end‑to‑end: UI → FastAPI → LangGraph; node prints tenant id.

## 8) Docs
- [x] Implementation plan captured.
  - File(s): `Development_Plan/Auth_Langgraph_Bugfix.md`
  - Accept: Document includes Keycloak mapper, auth plugin, proxy injection, UI config, and fallback path.

---

### Cross‑links
- Plan: `Development_Plan/Auth_Langgraph_Bugfix.md`
- Server auth: `src/auth.py`, `langgraph.json`
- Proxy: `app/graph_proxy.py`, `app/main.py`
- Gateway auth: `app/auth.py`
- UI stream: `agent-chat-ui/src/providers/Stream.tsx`
- Nodes: `my_agent/utils/nodes.py` (reads configurable/context/metadata)

### Open Risks / Notes
- If Keycloak tokens omit `tenant_id` and the proxy is bypassed (UI → LangGraph direct), runs can still lack tenant unless UI forwards config + Authorization.
- In prod, prefer UI → FastAPI `/graph` proxy; service‑to‑service auth avoids placing access tokens in the browser.
- Ensure DB snapshots exist per tenant for return‑user reuse; absence of cache will correctly report `cached=0` even with tenant present.
