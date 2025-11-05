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
  - Encapsulates auth, connection/session, timeouts, and telemetry.
  - Exposes a synchronous `read_url(url: str, timeout_s: float | None = None) -> str | None` function.
  - Transport (current implementation): remote only via `mcp-remote` over stdio JSON‑RPC.
    - Spawns `mcp-remote` with `Authorization: Bearer $JINA_API_KEY` and `MCP_SERVER_URL`.
    - Uses a global session pool keyed by `(server_url, api_key)` to reuse a single `mcp-remote` process and avoid per‑call spawn.
    - Supports `MCP_EXEC` to run a globally installed `mcp-remote` binary (skips `npx`); else uses `MCP_NPX_PATH` (default `npx`).
    - Note: `MCP_TRANSPORT` exists for future options but is currently treated as remote under the hood.
  - Tool discovery & caching:
    - Lists tools once per session and caches the first matching of `jina_read_url`, `read_url`, or `read`.
  - Response handling: when result has a `content` list, concatenate `{type: "text", text: ...}` parts; else stringify the result.
  - Cleaning: caller (`src/jina_reader`) applies `clean_jina_text` to preserve legacy semantics.
  - Resilience:
    - Auto‑restart on transient SSE errors: stderr watcher detects “SSE error”, “Body Timeout”, “Connect Timeout”, etc., marks the session inactive, and closes it so the pool restarts cleanly on the next call.
    - Per‑call retry: on tool call failure, invalidate the session and retry once with a fresh session.
    - Per‑call timeouts: `MCP_TIMEOUT_S` (call), `MCP_INIT_TIMEOUT_S` (handshake).
- Feature flags & settings (extend `src/settings.py`):
  - `ENABLE_MCP_READER` (default: false) — gate MCP usage globally for `read_url`.
  - `MCP_SERVER_URL` (default: `https://mcp.jina.ai/sse`).
  - `MCP_TRANSPORT` (reserved): present for future transport options; current code uses the remote transport path.
  - `MCP_TIMEOUT_S` (default: 12.0) — per-call timeout.
  - `MCP_DUAL_READ_PCT` (default: 0) — percentage to run MCP+HTTP in parallel for parity diff; return HTTP to callers while logging diffs.
  - `MCP_INIT_TIMEOUT_S` (default: 25.0) — initialization handshake timeout.
  - `MCP_NPX_PATH` (default: `npx`) — override npx path when not using `MCP_EXEC`.
  - `MCP_EXEC` (optional) — use a globally installed `mcp-remote` binary to avoid `npx` overhead.
  - `MCP_PROTOCOL_VERSION` (default: `2024-10-07`) — protocol version sent during initialize.
  - Important: The service reads `JINA_API_KEY` (not `MCP_API_KEY`) and `MCP_SERVER_URL` (not `MCP_ENDPOINT`).
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
     - On MCP path, call `mcp_reader.read_url`, then `clean_jina_text`; on error, log and fallback to HTTP.
     - Preserve function signature and behavior.

4) Dual‑Read Parity Sampling
   - Implement a lightweight sampler in `src/jina_reader.py` (e.g., `random.random() < pct`) to run both transports for a subset of calls.
   - Compare basic signals (len(text), token overlap) and ship diffs via `obs.log_event(stage="mcp_dual_read", extra={...})` for dashboards.

5) Tests
   - Unit tests (no network):
     - Wrapper path and dual‑read sampler can be exercised by mocking `src.services.mcp_reader.read_url`.
   - Backend route tests: see `tests/test_mcp_backend_routes.py` for server/tool listing and invocation stubs.
   - Manual check: `scripts/check_mcp_read_url.py` prints config, runs a read, and shows result length/preview.

6) Docs & Runbooks
   - Update `README`/ops notes with new env keys, enable/disable instructions, and dual‑read usage.
   - Add rollback steps and key rotation SOP references (align with master PRD decisions).

## File Changes (as implemented)
- Add: `src/services/mcp_reader.py` (MCP client over `mcp-remote`, session pool, stderr SSE handling, retry).
- Update: `src/settings.py` (ENABLE_MCP_READER, MCP_* envs).
- Update: `src/jina_reader.py` (feature‑flagged MCP path, dual‑read sampler, fallback to HTTP, telemetry hooks).
- Add: `scripts/check_mcp_read_url.py` (CLI checker for config and read_url).
- Docs: `read.md` updated with envs, rollout, and CLI usage.

## Reference: Demo Agent Alignment
- From `docs/demo_mcp_agent.md`:
  - Start transport (in demo: `npx mcp-remote`) with `Authorization: Bearer $JINA_API_KEY` and server URL `https://mcp.jina.ai/sse`.
  - Initialize session, list tools, select `read_url` (or compatible alias), and call with `{"url": "..."}`.
  - Parse `result.content` list, concatenate items of `{type: "text", text: "..."}` to yield raw text.
  - Our service mirrors these semantics; when using `MCP_TRANSPORT=remote`, replicate the initialization and call flow shown in the demo.

## Observability & Alerts
- Metrics & logs:
  - `obs.bump_vendor(vendor="jina_mcp", calls/errors, duration)` and `obs.log_event(stage="mcp_read_url", …)` on every call.
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

## Notes & Out‑of‑Scope
- DDG HTML snapshot reads remain on `r.jina.ai` by design; only homepage/content reads moved to MCP.
- Recommended runtime for `mcp-remote`: Node 20 LTS; 22.x may be more sensitive to SSE timeouts.
- Future: adopt official Python MCP client when available; keep the `MCP_TRANSPORT` flag reserved.

## Open Questions
None currently.

## Phase 2 — PRD Alignment (Search + Metrics + Concurrency + Full Flow Migration)

Goals
- Implement MCP search tools (`search_web`, `parallel_search_web`).
- Add Prometheus metrics with explicit names/labels for MCP calls and latency.
- Introduce a thread‑pool abstraction to support controlled parallelism while keeping sync wrappers.
- Expand adoption beyond homepage reads to resolver/evidence/search flows.

Acceptance Criteria
- `src/services/mcp_reader.py` exposes:
  - `search_web(query: str, *, country: str | None = None, max_results: int = 20, timeout_s: float | None = None) -> list[str]`
  - `parallel_search_web(queries: list[str], *, per_query: int = 10, timeout_s: float | None = None) -> dict[str, list[str]]`
- Discovery and enrichment paths can use MCP search behind a feature flag with graceful fallback to DDG.
- Prometheus metrics exported: `mcp_calls_total{tool, status}` and `mcp_latency_seconds{tool}` with sensible buckets.
- A shared `ThreadPoolExecutor` is used for MCP calls that need parallelism (e.g., `parallel_search_web`), with bounded concurrency.
- Rollback: a single env toggle disables MCP search and restores DDG‑only behavior.

New Env Vars
- `ENABLE_MCP_SEARCH` (default: false): gate MCP search tools in discovery/resolver flows.
- `MCP_SEARCH_COUNTRY` (optional): default country/market hint passed to `search_web` when unset by caller.
- `MCP_POOL_MAX_WORKERS` (default: small integer, e.g., 4–8): caps the thread pool size for MCP parallel calls.
- `PROM_ENABLE` (default: false): enable Prometheus export if the app’s metrics endpoint is wired.

Design & Changes
- Service (`src/services/mcp_reader.py`):
  - Tool discovery caches `search_web` and `parallel_search_web` names (similar to `read_url`).
  - Add `search_web` and `parallel_search_web` functions that reuse the same pooled `mcp-remote` session, timeouts, stderr‑based SSE restart, and per‑call retry once semantics.
  - Add optional thread pool (`ThreadPoolExecutor`) used only when executing many MCP calls concurrently (e.g., fan‑out search). Size controlled by `MCP_POOL_MAX_WORKERS`.
  - Telemetry: `obs.log_event(stage="mcp_search_web"|"mcp_parallel_search_web", …)` and vendor bumps.
  - Prometheus: increment `mcp_calls_total{tool="read_url|search_web|parallel_search_web",status="ok|error"}` and observe `mcp_latency_seconds{tool=…}`.
- Discovery/Agents integration:
  - Add flag path in `src/agents_icp.py` (and any discovery helpers) to call MCP `search_web` instead of DDG when `ENABLE_MCP_SEARCH=true`, preserving DDG as fallback.
  - Keep DDG HTML snapshot behavior as a fallback/read‑only validation path during rollout.
- Resolver/Evidence flows:
  - Identify all read/search entry points used to build evidence and resolver cards; route through `jina_reader` and MCP search helpers with feature flags.
  - Maintain identical output shapes so prompts and scorers remain stable.

Prometheus Wiring
- Exporter: if the app already exposes a `/metrics` endpoint, extend the existing registry to include MCP metrics. Otherwise, add a basic exporter consistent with current infra.
- Metrics:
  - Counter: `mcp_calls_total` with labels `{tool, status}`.
  - Histogram/Summary: `mcp_latency_seconds` with label `{tool}`; buckets tuned to current p95 (e.g., 0.5, 1, 2, 3, 5, 8, 13s).
- Correlate with existing logs by including `run_id`/tenant in log events; metrics stay per‑process without PII.

Thread‑Pool Abstraction
- Add a module‑level `ThreadPoolExecutor` (lazy‑initialized) in `mcp_reader` for bulk operations.
- Use it only where parallel execution is beneficial (e.g., `parallel_search_web` or batched read_url parity tests), keeping single calls synchronous.
- Cap concurrency (`MCP_POOL_MAX_WORKERS`) to avoid server throttling and local oversubscription.

Implementation Steps
1) Extend `mcp_reader.py` with cached tool names and two new functions (search + parallel_search), mirroring existing retry/telemetry.
2) Add Prometheus metrics plumbing (guarded by `PROM_ENABLE`) and emit metrics in all MCP functions.
3) Introduce a small thread pool in `mcp_reader` and use it for parallel search fan‑out.
4) Add `ENABLE_MCP_SEARCH` gating in discovery/agents; wire fallback to DDG.
5) Update docs (`read.md`) with new env vars and rollout guidance for search tools.
6) Tests: stub MCP functions in discovery tests; unit test tool selection and result shaping for search functions.

Rollout
- Stage 1: Enable `ENABLE_MCP_SEARCH=true` for canary tenants with small `MCP_POOL_MAX_WORKERS` (e.g., 4). Compare MCP vs DDG coverage.
- Stage 2: Increase tenants and pool size if p95 latency and success rates meet thresholds.
- Rollback: set `ENABLE_MCP_SEARCH=false` to revert discovery back to DDG immediately.
