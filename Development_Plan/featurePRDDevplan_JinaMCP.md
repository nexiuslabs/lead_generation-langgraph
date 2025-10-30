---
owner: Codex Agent – Frontend Generator
status: draft
last_reviewed: 2025-10-30
---

# Feature Dev Plan — Jina MCP Read URL Migration

This plan narrows scope from the broader MCP adoption PRD to a focused migration of the legacy Jina HTTP reader (`r.jina.ai`) to the Jina MCP server’s `read_url` tool. It references the prior PRD and the demo MCP agent to ensure implementation details are aligned and testable.

## Story & Context
- Source PRD: `Development_Plan/featurePRD_jinaMCP.md` — adopt MCP transport under a feature flag, preserve sync pipelines, and ship observability and rollback.
- Constraint: Only adopt the `read_url` tool at this time. Exclude MCP search and parallel search tools.
- Reference: `docs/demo_mcp_agent.md` — shows a working Python client that spawns `mcp-remote`, selects the `read_url` tool, invokes it, and concatenates the `content` parts of type `text`.

## Scope
- Replace legacy HTTP reader usage with MCP `read_url` behind a feature flag.
- Maintain legacy behavior, including text cleaning and function signatures, to avoid breaking downstream code.
- Add metrics, structured logs, and rollback toggles.

## Non‑Goals
- MCP `search_web` and `parallel_search_web` (explicitly out of scope).
- Broader async refactor; keep synchronous wrappers for compatibility.
- Cutting over by default before parity is demonstrated.

## Acceptance Criteria
- A reusable MCP client module exposes a synchronous `read_url(url, timeout_s)` wrapper that calls the Jina MCP server (`https://mcp.jina.ai/sse`) with `Authorization: Bearer $JINA_API_KEY`.
- `src/jina_reader.read_url` respects a feature flag (`ENABLE_MCP_READER`) to switch between legacy HTTP and the new MCP-backed client without code changes at call sites.
- Output shape and cleaning match legacy behavior (concat and sanitize content, reuse existing `clean_jina_text`).
- Structured logs and metrics record MCP usage, latency, errors, and rate-limits; dashboards and alert thresholds exist.
- A dual-read sampling mode enables parity verification without changing outputs.
- Operators can roll back to HTTP within one deploy (via env toggle) and a short runbook is documented.

## Architecture & Design
- Service module: `src/services/mcp_reader.py`
  - Encapsulates auth, connection/session, retries, timeouts, and telemetry.
  - Provides a synchronous `read_url(url: str, timeout_s: float | None = None) -> str | None` API.
  - Transport selection via env:
    - `MCP_TRANSPORT=python` (preferred): use official `jina-mcp` Python client when available.
    - `MCP_TRANSPORT=remote` (fallback): mirror `docs/demo_mcp_agent.md` by spawning `npx mcp-remote` with the Authorization header over stdio JSON‑RPC.
  - Tool discovery: prefer `read_url`; accept `jina_read_url` or `read` if present for compatibility (as per demo).
  - Response handling: when result contains `content` array, join parts of `{type: "text", text: ...}`; else stringify.
  - Cleaning: pass through `src/jina_reader.clean_jina_text` to preserve downstream assumptions.
  - Resilience: retry on transient MCP errors using `src/retry.py`; enforce a call timeout; thread-safe; caches tool list per process; reconnects on broken sessions.
- Feature flags & settings (extend `src/settings.py`):
  - `ENABLE_MCP_READER` (default: false) — gate MCP usage globally for `read_url`.
  - `MCP_SERVER_URL` (default: `https://mcp.jina.ai/sse`).
  - `MCP_TRANSPORT` (default: `python`; `remote` to force `mcp-remote`).
  - `MCP_TIMEOUT_S` (default: 12.0) — per-call timeout.
  - `MCP_DUAL_READ_PCT` (default: 0) — percentage to run MCP+HTTP in parallel for parity diff; return HTTP to callers while logging diffs.
  - Reuse `JINA_FAIL_TTL_S` and `_FAIL_CACHE` semantics from legacy reader for host cool‑off.
- Integration point:
  - Modify `src/jina_reader.read_url` to honor flags:
    1) If `ENABLE_MCP_READER=true` and dual‑read sampling triggers, invoke both MCP and HTTP, log parity metrics, return HTTP.
    2) Else if `ENABLE_MCP_READER=true`, call `mcp_reader.read_url` and return cleaned text.
    3) Else use legacy HTTP path.
  - No changes required at calling sites (`icp_pipeline`, `agents_icp`, etc.).
- Telemetry & health:
  - Use `src/obs.py` to log vendor usage: `vendor="jina_mcp"` with calls/errors, duration, and error codes.
  - Stage timers: wrap `read_url` with `obs.stage_timer("mcp_read_url")` where appropriate in the service module.
  - Optional Prometheus counters/histograms if the exporter is present; otherwise rely on DB‑backed stats.

## Implementation Steps
1) Settings & Flags
   - Add `ENABLE_MCP_READER`, `MCP_SERVER_URL`, `MCP_TRANSPORT`, `MCP_TIMEOUT_S`, `MCP_DUAL_READ_PCT` to `src/settings.py` with sensible defaults.

2) MCP Reader Service
   - Create `src/services/mcp_reader.py`:
     - `class MCPReader`: manages client init, tool cache, retries, and `read_url` wrapper.
     - Transport adapters: python client; remote subprocess (stdio JSON‑RPC) following `docs/demo_mcp_agent.md`.
     - Observability hooks: call `obs.bump_vendor` and `obs.log_event` with tool, duration, and status.

3) Integrate Into Legacy Reader
   - Update `src/jina_reader.read_url` to:
     - Check flags and dual‑read sampling percent.
     - On MCP path, call `MCPReader.read_url`, then `clean_jina_text`; on error, increment vendor error, mark `_FAIL_CACHE`, and fallback to HTTP if allowed.
     - Preserve function signature and behavior.

4) Dual‑Read Parity Sampling
   - Implement a lightweight sampler in `src/jina_reader.py` (e.g., `random.random() < pct`) to run both transports for a subset of calls.
   - Compare basic signals (len(text), token overlap) and ship diffs via `obs.log_event(stage="mcp_dual_read", extra={...})` for dashboards.

5) Tests
   - Unit tests (no network):
     - Mock MCP transport to return `content` arrays and assert concatenation + cleaning.
     - Retry/circuit breaker behavior via `src/retry.py`.
     - Dual‑read sampler returns HTTP content while logging diffs.
   - Conditional integration tests (skipped if no `JINA_API_KEY`):
     - Smoke test `read_url` on a stable site; assert non‑empty cleaned text.

6) Docs & Runbooks
   - Update `README`/ops notes with new env keys, enable/disable instructions, and dual‑read usage.
   - Add rollback steps and key rotation SOP references (align with master PRD decisions).

## File Changes
- Add: `src/services/mcp_reader.py` (new service module).
- Update: `src/settings.py` (new flags and defaults).
- Update: `src/jina_reader.py` (feature‑flagged MCP path, dual‑read sampler, telemetry hooks; preserve cleaner and fail cache).
- Add: `tests/test_mcp_reader.py` (unit tests with mocks; optional integration test gated by env).
- Docs: This plan (`Development_Plan/featurePRDJinaMCP.md`) plus minor updates to operational docs.

## Reference: Demo Agent Alignment
- From `docs/demo_mcp_agent.md`:
  - Start transport (in demo: `npx mcp-remote`) with `Authorization: Bearer $JINA_API_KEY` and server URL `https://mcp.jina.ai/sse`.
  - Initialize session, list tools, select `read_url` (or compatible alias), and call with `{"url": "..."}`.
  - Parse `result.content` list, concatenate items of `{type: "text", text: "..."}` to yield raw text.
  - Our service mirrors these semantics; when using `MCP_TRANSPORT=remote`, replicate the initialization and call flow shown in the demo.

## Observability & Alerts
- Metrics & logs:
  - `obs.bump_vendor(vendor="jina_mcp", calls/errors, duration)` on every call.
  - `obs.stage_timer("mcp_read_url")` around service calls where feasible.
  - Optional Prometheus: `mcp_calls_total{tool,status}` and `mcp_latency_seconds{tool}` if exporter enabled.
- Alert thresholds (staging and prod):
  - Error rate > 5% over 5 minutes.
  - p95 latency > 3× HTTP baseline.
  - Spike in rate‑limit error codes.

## Rollout Plan
1) Dev: land service, flags, tests; default off.
2) Staging: enable `ENABLE_MCP_READER=true`, set `MCP_DUAL_READ_PCT` to 50; collect parity stats.
3) Canary: enable on a low‑risk tenant; target ≥95% success, p95 close to HTTP.
4) Cutover: raise to 100% after ≥98% success sustained; keep HTTP as emergency fallback.
5) Rollback: set `ENABLE_MCP_READER=false`; document outage and perform key rotation if indicated.

## Risks & Mitigations
- Protocol/stream complexity: hide behind service module; prefer official client; fall back to `mcp-remote` if needed.
- Latency variance: connection reuse and tuned timeouts; staged rollout with p95 monitoring.
- Quota/rate limits: capture specific error codes; integrate circuit breaker; keep HTTP fallback.
- Content parity differences: dual‑read sampling, cleaner tweaks, QA samples.

## Validation & QA
- Dual‑read parity reports reviewed daily during staging.
- Manual QA on sampled domains; verify business‑critical fields downstream are unaffected.
- Confirm logs and dashboards reflect MCP usage and latency as expected.

## Open Questions
None. Future items (e.g., extending to search tools) will be captured in a separate plan.

