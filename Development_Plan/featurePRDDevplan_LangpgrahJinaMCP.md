---
owner: Codex Agent – Frontend Generator
status: draft
last_reviewed: 2025-10-31
---

# Feature Dev Plan — Switch Jina MCP to LangGraph MCP Adapters (Python)

This plan migrates our current Jina MCP integration from the custom `mcp-remote` subprocess client to the LangGraph/LangChain MCP adapters for Python, keeping call sites and behavior stable. It is informed by `docs/langgraph_mcp.md` and aligns with `Development_Plan/featurePRDDevplan_JinaMCP.md`.

## Story & Context
- Current: `src/services/mcp_reader.py` spawns `mcp-remote` (Node) and implements JSON‑RPC to list/call tools, with a feature flag in `src/jina_reader.py` and dual‑read parity.
- Target: Use the Python package `langchain-mcp-adapters` to connect directly to the Jina MCP server via streamable HTTP/SSE with Authorization headers, exposing a sync `read_url` wrapper.
- Why: Reduce subprocess overhead, rely on maintained adapters, simplify reconnections and tool management while preserving existing flags, telemetry, and rollbacks.

## Scope
- Implement a new transport in `src/services/mcp_reader.py` backed by `langchain_mcp_adapters` to call the Jina MCP server’s `read_url` tool.
- Keep `src/jina_reader.read_url` unchanged other than recognizing the new transport via settings. Cleaning and dual‑read remain in Python.
- Limit to `read_url` in this phase. Defer MCP `search_web`/`parallel_search_web` to Phase 2.

## Non‑Goals
- Rewriting downstream enrichment/discovery code to async.
- Changing cleaning behavior or response shape beyond extracting plain text.
- Hosting our own MCP server.

## Acceptance Criteria
- New transport (`MCP_TRANSPORT=adapters_http`) in Python connects to `MCP_SERVER_URL` with `Authorization: Bearer $JINA_API_KEY` using `langchain_mcp_adapters`.
- Provides a synchronous `read_url(url, timeout_s)` wrapper which returns concatenated plain text from MCP content blocks.
- `src/jina_reader.read_url` continues to gate via `ENABLE_MCP_READER`, supports dual‑read sampling, and falls back to HTTP path on errors.
- Observability parity: Python logs vendor usage, durations, error codes; no regression in metrics.
- Rollback by flipping transport to `remote` or disabling MCP entirely.

## Architecture & Design
- Client/session management (Python):
  - Use `langchain_mcp_adapters.client.MultiServerMCPClient` to configure a single server named `jina` with:
    - `transport: "streamable_http"` or `"sse"`
    - `url: MCP_SERVER_URL` (default `https://mcp.jina.ai/sse`)
    - `headers: { "Authorization": "Bearer " + JINA_API_KEY }` (per docs, headers supported for SSE/streamable HTTP)
  - Initialize the client once per process and reuse it (global singleton) to avoid per‑call session cost.
  - Discover tools on first use; cache the read tool name by alias preference: `read_url`, `jina_read_url`, `read`.
  - Text extraction:
    - Prefer `useStandardContentBlocks=True` semantics (when available) and concatenate text blocks.
    - Fallback: when content is the legacy list of blocks, join `{type: "text"}` blocks; else stringify.
- Sync wrapper:
  - Start an asyncio event loop in a daemon thread for the adapters client.
  - Execute async calls via `asyncio.run_coroutine_threadsafe(...)` to provide a blocking `read_url` function.
  - Enforce per‑call timeout with `timeout_s` (defaults to `MCP_TIMEOUT_S`).
- Resilience:
  - On call failure: attempt one invalidate/reconnect then retry once.
  - Keep behavior of stderr/SSE pattern handling conceptually replaced by adapters’ reconnect logic; surface errors to caller for HTTP fallback.

## Configuration & Flags
- Existing:
  - `ENABLE_MCP_READER` — feature‑flag MCP usage in `src/jina_reader.py`.
  - `MCP_SERVER_URL` — default `https://mcp.jina.ai/sse`.
  - `MCP_TIMEOUT_S` — per‑call timeout (default 12.0s).
  - `MCP_DUAL_READ_PCT` — 0..100 dual‑read sampling for parity.
- New:
  - `MCP_TRANSPORT=adapters_http` — select LangGraph MCP adapters path.
  - Optional tuning: `MCP_ADAPTER_USE_SSE=true|false` (default false → streamable_http),
    `MCP_ADAPTER_USE_STANDARD_BLOCKS=true|false` (default true).

## Implementation Steps
1) Settings & Defaults
   - Add `MCP_TRANSPORT` option `adapters_http` to `src/settings.py` (do not change default).
   - Add `MCP_ADAPTER_USE_SSE` (default false) and `MCP_ADAPTER_USE_STANDARD_BLOCKS` (default true).

2) Service: `src/services/mcp_reader.py`
   - Add a new transport branch `adapters_http`:
     - Initialize a module‑level `MultiServerMCPClient` singleton inside a background asyncio loop/thread.
     - On first call, get tools and cache the read tool.
     - Call the tool with `{"url": url}`; apply timeout; extract and return text.
     - On error: invalidate client/session and retry once, else raise to trigger HTTP fallback in `jina_reader`.
   - Keep existing `remote` (mcp-remote) branch intact for rollback.

3) Integration: `src/jina_reader.py`
   - No signature or behavior change.
   - Confirm dual‑read parity sampling still logs `http_len`/`mcp_len` and returns HTTP.

4) Observability
   - Preserve `obs.bump_vendor` and `obs.log_event(stage="mcp_read_url")` paths around service calls.
   - Optionally record adapter transport as `transport="adapters_http"` in `extra` for dashboards.

5) Tests
   - Unit: mock adapters client/tool to return content blocks; verify concatenation and cleaner application.
   - Unit: inject adapter exceptions to confirm retry/raise behavior and HTTP fallback.
   - Optional integration (if key present): smoke test `read_url` against stable domains.

6) Docs & Runbook
   - Update `read.md` with `MCP_TRANSPORT=adapters_http`, adapter notes, and headers requirement.
   - Rollback: flip `MCP_TRANSPORT=remote` or `ENABLE_MCP_READER=false`.

## File Changes (planned)
- Update: `src/settings.py` — add adapter flags; document `adapters_http` choice.
- Update: `src/services/mcp_reader.py` — implement `adapters_http` transport with thread‑backed asyncio and client singleton.
- No change: `src/jina_reader.py` beyond recognizing transport (existing service API unchanged).
- Docs: update `read.md` with adapter transport docs and examples.

## Output Handling Details
- With standard blocks (recommended): join text from StandardTextBlock entries.
- Without standard blocks: join items where block `{type: "text"}` to preserve legacy behavior.
- Ignore images/audio/resources for `read_url` (content‑only); log sizes if needed for QA parity.

## Rollout Plan
1) Dev: land transport, unit tests; default remains on `remote`.
2) Staging: enable `ENABLE_MCP_READER=true`, set `MCP_TRANSPORT=adapters_http`, `MCP_DUAL_READ_PCT=50`; review parity.
3) Canary: enable on a low‑risk tenant; target ≥95% success and p95 close to remote.
4) Cutover: keep adapters as default once ≥98% success sustained; retain `remote` for emergency fallback.
5) Post‑cutover: monitor error rate and latency over 2–4 weeks.

## Risks & Mitigations
- Async boundary in a sync codebase: encapsulate in a background event loop thread; provide strict timeouts.
- Output mapping drift: standardize blocks and text extraction; use dual‑read and QA to validate parity.
- Library updates: pin `langchain-mcp-adapters` and track upstream changes; keep remote fallback.
- Authentication change: ensure Authorization header is sent for all HTTP/SSE calls.

## Validation & QA
- Daily parity checks on staged traffic via dual‑read; auto‑summarize variance (length/token overlap).
- Manual spot‑checks for critical downstream fields; confirm no regressions.
- Confirm telemetry volume/latency lines up with expectations and alert thresholds.

## References
- docs/langgraph_mcp.md (Python adapters, MultiServerMCPClient, headers with SSE/streamable_http)
- Development_Plan/featurePRDDevplan_JinaMCP.md (flags, parity, observability, rollout)

