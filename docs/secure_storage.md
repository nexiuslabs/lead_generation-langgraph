Secure Storage Policy — Secrets & Sessions

Scope
- Backend: lead_generation-main FastAPI app (cookies, Odoo, SSO, LangGraph).
- Frontend: agent-chat-ui (no secrets stored; cookie-based session only in prod).

Principles
- Server-side only: Store all sensitive credentials in environment variables on the server (never in the client or repo).
- Cookies for sessions: Use HTTP-only cookies (`nx_access`, `nx_refresh`) to hold tokens; do not send Authorization in production.
- Short-lived sessions: Access cookie max-age derives from token `exp`; refresh cookie is long-lived with rotation via `/auth/refresh`.
- Key injection at server: When proxying to a remote LangGraph, inject `X-Api-Key`/`Authorization` from server env (never expose to browsers).
- RLS enforcement: Set `request.tenant_id` GUC per request before DB access so row-level security isolates tenant data.

Configuration
- ACCESS_COOKIE_NAME, REFRESH_COOKIE_NAME: Customize cookie names if needed.
- COOKIE_SECURE=true: Secure cookies in production (HTTPS only).
- ENABLE_GRAPH_PROXY=true and LANGGRAPH_REMOTE_URL: Enable backend graph proxy; use LANGSMITH_API_KEY and/or LANGGRAPH_BEARER_TOKEN for server-side auth.
- ODOO_* vars: Keep Odoo DB credentials, template admin credentials, and master password only on the server.

Client Behavior
- agent-chat-ui sends no Authorization header in production; all requests include credentials: 'include' so cookies flow.
- For split-origin development, the UI edge proxy forwards the Cookie header but does not attach any API keys.

