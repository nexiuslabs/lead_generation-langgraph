---
owner: Backend – Lead Gen Platform
status: active
last_reviewed: 2025-10-28
---

# TODO — Jina MCP Server Integration (Reader + Search)

Progress tracker derived from `devplan/featureDevPlanPRD_JinaMCP.md` and current codebase.

- [x] Configure settings and flags — Owner: Backend – Lead Gen Platform
  - Add `ENABLE_MCP_READER`, `MCP_ENDPOINT`, `MCP_API_KEY`, `MCP_TIMEOUT_S`, `MCP_MAX_PARALLEL`, `MCP_DUAL_READ_SAMPLE_PCT` in `src/settings.py`.
- [ ] Document env vars — Owner: Backend – Lead Gen Platform
  - Update `.env`/`.env.example` and README/devplan with configuration guidance and defaults.
- [ ] Optional MCP client dependency — Owner: Backend – Lead Gen Platform
  - Evaluate adding `jina-mcp` to `requirements.txt` or keep current bridge/remote-only approach; document decision.

- [x] Implement Python MCP reader client — Owner: Backend – Lead Gen Platform
  - `src/services/mcp_reader.py` with retries, fallback behavior, and detailed invoke logs.
- [x] Add remote JSON‑RPC fallback — Owner: Backend – Lead Gen Platform
  - `src/services/mcp_remote.py` implements `read_url`, `search_web`, `parallel_search_web` against `MCP_REMOTE_URL`.
- [x] Add LangGraph server MCP bridge client — Owner: Backend – Lead Gen Platform
  - `src/services/mcp_server_bridge.py` calls `/mcp/servers/{server}/tools/{tool}/invoke` with cooldown and timeouts.

- [x] Wire MCP into URL reader — Owner: Backend – Lead Gen Platform
  - `src/jina_reader.py` prefers server bridge/client when enabled; falls back to `r.jina.ai` with cleaning.
- [x] Wire MCP into web search — Owner: Backend – Lead Gen Platform
  - `src/ddg_simple.py` prefers MCP `search_web` (bridge/client), falls back to DDG HTML parsing.

- [x] Expose MCP routes in backend — Owner: Backend – Lead Gen Platform
  - `app/mcp_routes.py` implements `/mcp/servers/*/tools/*/invoke` plus discovery; mounted in `app/main.py`.
- [x] Health probe for MCP — Owner: Backend – Lead Gen Platform
  - `GET /health/mcp` checks MCP path and returns 503 on failure.

- [x] Logging and resilience — Owner: Backend – Lead Gen Platform
  - Structured start/ok/fail logs, reason/body snippets; cooldown on bridge errors; local‑dev auth tolerance.
- [ ] Telemetry hooks — Owner: Backend – Lead Gen Platform
  - Integrate `src/obs.py` (`log_event`, `bump_vendor`) in MCP paths for per‑tenant/run metrics.

- [x] Backend route tests — Owner: Backend – Lead Gen Platform
  - `tests/test_mcp_backend_routes.py` validates servers/tools listing and tool invocations (patched remote client).
- [ ] Reader/client unit tests — Owner: Backend – Lead Gen Platform
  - Add tests for `mcp_reader`, `mcp_server_bridge`, and `mcp_remote` covering success, timeouts, 4xx, and cooldown.
- [ ] Log‑parity tests — Owner: Backend – Lead Gen Platform
  - Reproduce observed 404/406/skip flows and assert log sequences to prevent regressions.

- [ ] Docs and runbooks — Owner: Backend – Lead Gen Platform
  - Add rollout/enablement steps, troubleshooting (403 auth, 404 routes, 406 remote), and recommended dev flags.
- [ ] Production hardening — Owner: Backend – Lead Gen Platform
  - Gate MCP routes behind stricter auth in non‑dev, enforce valid JWT/audience, and rate‑limit tool calls.

Notes
- Bridge path prefers local LangGraph `/mcp` routes; remote JSON‑RPC is a last‑resort fallback.
- Optional `jina-mcp` dependency can remain excluded if bridge/remote paths meet requirements; revisit if SSE transport is needed.

