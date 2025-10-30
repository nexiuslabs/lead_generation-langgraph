---
owner: Codex Agent – Frontend Generator
status: draft
last_reviewed: 2025-10-30
---

# To‑Do List — Jina MCP Read URL Migration

Tracking checklist derived from Development_Plan/featurePRDDevplan_JinaMCP.md. Scope is limited to adopting the Jina MCP `read_url` tool behind a feature flag and preserving legacy behavior. Search tools are out of scope.

## 0) Prerequisites
- [ ] Confirm JINA_API_KEY is provisioned and scoped for MCP usage (Owner: Ops)
- [ ] Decide transport default: `MCP_TRANSPORT=python` (official client) vs `remote` (mcp-remote) (Owner: Backend)
- [ ] Align timeouts and baseline latency thresholds vs current HTTP reader (Owner: Backend)

## 1) Settings & Flags
- [ ] Add `ENABLE_MCP_READER` to `src/settings.py` (default: false) (Owner: Backend)
- [ ] Add `MCP_SERVER_URL` (default: `https://mcp.jina.ai/sse`) (Owner: Backend)
- [ ] Add `MCP_TRANSPORT` (default: `python`; optional `remote`) (Owner: Backend)
- [ ] Add `MCP_TIMEOUT_S` (default: 12.0) (Owner: Backend)
- [ ] Add `MCP_DUAL_READ_PCT` (default: 0) (Owner: Backend)
- [ ] Document new env vars in README/ops notes (Owner: Docs)

## 2) MCP Reader Service (`src/services/mcp_reader.py`)
- [ ] Implement `MCPReader` class: init, tool discovery, reconnect on error (Owner: Backend)
- [ ] Python transport: use official `jina-mcp` client (Owner: Backend)
- [ ] Remote transport: spawn `npx mcp-remote` with Authorization header (Owner: Backend)
- [ ] Implement `read_url(url: str, timeout_s: float|None) -> str|None` (Owner: Backend)
- [ ] Result normalization: concatenate `content` parts of `{type: "text"}` (Owner: Backend)
- [ ] Apply `src/jina_reader.clean_jina_text` before returning (Owner: Backend)
- [ ] Add retries using `src/retry.py` and enforce timeout (Owner: Backend)
- [ ] Thread-safety: protect session/tool cache; expose sync API (Owner: Backend)

## 3) Integrate Into Legacy Reader (`src/jina_reader.py`)
- [ ] Gate by `ENABLE_MCP_READER` (Owner: Backend)
- [ ] Implement dual-read sampler using `MCP_DUAL_READ_PCT` (Owner: Backend)
- [ ] Fallback to HTTP on MCP errors; update `_FAIL_CACHE` accordingly (Owner: Backend)
- [ ] Preserve function signature and `clean_jina_text` behavior (Owner: Backend)

## 4) Telemetry & Alerts
- [ ] Call `obs.bump_vendor(vendor="jina_mcp")` with calls/errors, duration (Owner: Backend)
- [ ] Use `obs.log_event(stage="mcp_read_url", …)` around each call (Owner: Backend)
- [ ] Optional Prometheus metrics (counters/histograms) behind feature check (Owner: Backend)
- [ ] Define alert thresholds (error rate >5%, p95 latency 3× baseline) (Owner: Ops)
- [ ] Add dashboard panels for MCP usage and latency (Owner: Ops)

## 5) Testing
- [ ] Unit: mock MCP transport; verify content concatenation and cleaning (Owner: QA)
- [ ] Unit: retry/circuit breaker behavior under injected exceptions (Owner: QA)
- [ ] Unit: dual-read sampler returns HTTP while logging diffs (Owner: QA)
- [ ] Integration (conditional): with `JINA_API_KEY`, smoke test `read_url` on stable domain (Owner: QA)

## 6) Docs & Runbooks
- [ ] Add enable/disable instructions for `ENABLE_MCP_READER` and `MCP_DUAL_READ_PCT` (Owner: Docs)
- [ ] Write brief rollback SOP (flip flag, verify dashboards) (Owner: Docs)
- [ ] Note key rotation procedure or link to master PRD follow-up (Owner: Docs)

## 7) Rollout Plan
- [ ] Dev: land changes with flag off; unit tests green (Owner: Backend)
- [ ] Staging: enable `ENABLE_MCP_READER=true`, `MCP_DUAL_READ_PCT=50` (Owner: Ops)
- [ ] Review parity diffs daily; adjust cleaning if needed (Owner: QA/Backend)
- [ ] Canary: enable on low-risk tenant; target ≥95% success (Owner: Ops)
- [ ] Cutover: raise to 100% upon ≥98% success; keep HTTP fallback (Owner: Ops)
- [ ] Post-cutover monitoring for 2–4 weeks; decide on keeping fallback (Owner: Ops)

## 8) Risks & Mitigations (Actionables)
- [ ] Add explicit handling/logging of rate-limit errors; consider basic rate limiter (Owner: Backend)
- [ ] Implement per-tenant circuit breaker using `CircuitBreaker` (Owner: Backend)
- [ ] Track variance of content length vs HTTP to detect drift (Owner: QA)

## 9) Validation & QA Sign-off
- [ ] Dual-read deltas within ≤5% variance for sampled set (Owner: QA)
- [ ] No Sev-1 incidents during staged enablement (Owner: Ops)
- [ ] Sign-off from data consumers on output quality (Owner: PM/QA)

## Nice-to-haves (Deferred)
- [ ] Prometheus exporter coverage if not already present (Owner: Backend)
- [ ] Structured diff artifacts for QA (store sampled pairs) (Owner: QA)

