---
owner: Codex Agent – Frontend Generator
status: draft
last_reviewed: 2025-10-30
---

# To‑Do List — Jina MCP Read URL Migration

Tracking checklist derived from Development_Plan/featurePRDDevplan_JinaMCP.md. Scope is limited to adopting the Jina MCP `read_url` tool behind a feature flag and preserving legacy behavior. Search tools are out of scope.

## 0) Prerequisites
- [ ] Confirm JINA_API_KEY is provisioned and scoped for MCP usage (Owner: Ops)
- [ ] Set transport default: `MCP_TRANSPORT=remote` (mcp-remote). Python client deferred. (Owner: Backend)
- [ ] Align timeouts and baseline latency thresholds vs current HTTP reader (Owner: Backend)
- [ ] Ensure Node 20 LTS runtime available for `mcp-remote` (Owner: Ops)

## 1) Settings & Flags
- [x] Add `ENABLE_MCP_READER` to `src/settings.py` (default: false) (Owner: Backend)
- [x] Add `MCP_SERVER_URL` (default: `https://mcp.jina.ai/sse`) (Owner: Backend)
- [ ] Add `MCP_TRANSPORT` (default: `remote`; optional `python`) (Owner: Backend)
- [x] Add `MCP_TIMEOUT_S` (default: 12.0) (Owner: Backend)
- [x] Add `MCP_DUAL_READ_PCT` (default: 0) (Owner: Backend)
- [ ] Add `MCP_INIT_TIMEOUT_S` (default: 25.0) (Owner: Backend)
- [ ] Add `MCP_PROTOCOL_VERSION` (default: `2024-10-07`) (Owner: Backend)
- [ ] Add `MCP_NPX_PATH` (default: `npx`) and optional `MCP_EXEC` (use global `mcp-remote`) (Owner: Backend)
- [x] Document new env vars in README/ops notes (Owner: Docs)

## 2) MCP Reader Service (`src/services/mcp_reader.py`)
- [x] Implement `MCPReader` class: init, tool discovery, reconnect on error (Owner: Backend)
- [ ] Remote transport: spawn `mcp-remote` (via `MCP_EXEC` or `MCP_NPX_PATH=npx`) with `Authorization: Bearer $JINA_API_KEY` and `MCP_SERVER_URL` (Owner: Backend)
- [x] Implement `read_url(url: str, timeout_s: float|None) -> str|None` (Owner: Backend)
- [x] Result normalization: concatenate `content` parts of `{type: "text"}` (Owner: Backend)
- [x] Do NOT clean here; caller applies `src/jina_reader.clean_jina_text` to preserve legacy semantics (Owner: Backend)
- [ ] Add retries using `src/retry.py` and enforce timeout (Owner: Backend)
- [x] Thread-safety: protect session/tool cache; expose sync API (Owner: Backend)
- [x] Session/process reuse: maintain a pooled session per `(server_url, api_key)` to reuse a single `mcp-remote` process (Owner: Backend)
- [x] SSE/stderr watcher: detect "SSE error", "Body Timeout", "Connect Timeout"; invalidate session and auto-restart on next call (Owner: Backend)
- [ ] Wrap calls with `obs.stage_timer("mcp_read_url")` for latency measurement (Owner: Backend)

## 3) Integrate Into Legacy Reader (`src/jina_reader.py`)
- [x] Gate by `ENABLE_MCP_READER` (Owner: Backend)
- [x] Implement dual-read sampler using `MCP_DUAL_READ_PCT` (Owner: Backend)
- [x] Fallback to HTTP on MCP errors; update `_FAIL_CACHE` accordingly (Owner: Backend)
- [x] Preserve function signature and `clean_jina_text` behavior (Owner: Backend)
- [x] Apply `clean_jina_text` on the MCP result within the caller to match legacy behavior (Owner: Backend)

## 4) Telemetry & Alerts
- [x] Call `obs.bump_vendor(vendor="jina_mcp")` with calls/errors, duration (Owner: Backend)
- [x] Use `obs.log_event(stage="mcp_read_url", …)` around each call (Owner: Backend)
- [x] Optional Prometheus metrics (counters/histograms) behind feature check (Owner: Backend)
- [ ] Define alert thresholds (error rate >5%, p95 latency 3× baseline) (Owner: Ops)
- [ ] Add dashboard panels for MCP usage and latency (Owner: Ops)
- [ ] Use `obs.stage_timer("mcp_read_url")` to time end-to-end MCP calls (Owner: Backend)

## 5) Testing
- [ ] Unit: mock MCP transport; verify content concatenation and cleaning (Owner: QA)
- [ ] Unit: retry/circuit breaker behavior under injected exceptions (Owner: QA)
- [x] Unit: dual-read sampler returns HTTP while logging diffs (Owner: QA)
- [ ] Integration (conditional): with `JINA_API_KEY`, smoke test `read_url` on stable domain (Owner: QA)
- [x] Manual: add `scripts/check_mcp_read_url.py` to print config, run a read, and show result length/preview (Owner: Backend)

## 6) Docs & Runbooks
- [x] Add enable/disable instructions for `ENABLE_MCP_READER` and `MCP_DUAL_READ_PCT` (Owner: Docs)
- [x] Write brief rollback SOP (flip flag, verify dashboards) (Owner: Docs)
- [ ] Note key rotation procedure or link to master PRD follow-up (Owner: Docs)
- [ ] Document that DDG HTML snapshot reads remain on `r.jina.ai` (HTTP) by design (Owner: Docs)
- [ ] Recommend Node 20 LTS for `mcp-remote`; note 22.x may be sensitive to SSE timeouts (Owner: Docs)
- [x] Describe `MCP_PROTOCOL_VERSION` purpose and when to update (Owner: Docs)

## 7) Rollout Plan
- [x] Dev: land changes with flag off; unit tests green (Owner: Backend)
- [ ] Staging: enable `ENABLE_MCP_READER=true`, `MCP_DUAL_READ_PCT=50` (Owner: Ops)
- [ ] Review parity diffs daily; adjust cleaning if needed (Owner: QA/Backend)
- [ ] Canary: enable on low-risk tenant; target ≥95% success (Owner: Ops)
- [ ] Cutover: raise to 100% upon ≥98% success; keep HTTP fallback (Owner: Ops)
- [ ] Post-cutover monitoring for 2–4 weeks; decide on keeping fallback (Owner: Ops)

## 8) Risks & Mitigations (Actionables)
- [ ] Add explicit handling/logging of rate-limit errors; consider basic rate limiter (Owner: Backend)
- [ ] Implement per-tenant circuit breaker using `CircuitBreaker` (Owner: Backend)
- [x] Track variance of content length vs HTTP to detect drift (Owner: QA)

## 9) Validation & QA Sign-off
- [ ] Dual-read deltas within ≤5% variance for sampled set (Owner: QA)
- [ ] No Sev-1 incidents during staged enablement (Owner: Ops)
- [ ] Sign-off from data consumers on output quality (Owner: PM/QA)

## Nice-to-haves (Deferred)
- [ ] Prometheus exporter coverage if not already present (Owner: Backend)
- [ ] Structured diff artifacts for QA (store sampled pairs) (Owner: QA)
- [ ] Python transport using official `jina-mcp` client (`MCP_TRANSPORT=python`) once stable (Owner: Backend)

## Notes & Out‑of‑Scope
- DDG HTML snapshot reads remain on `r.jina.ai` (legacy HTTP path).
- Initial transport is remote via `mcp-remote`; Python client adoption is deferred.
- MCP search tools (`search_web`, `parallel_search_web`) remain out of scope for this migration.
