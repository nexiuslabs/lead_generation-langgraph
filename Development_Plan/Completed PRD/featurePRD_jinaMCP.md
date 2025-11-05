---
owner: Codex Agent – Frontend Generator
status: draft
last_reviewed: 2025-03-24
---

# Feature PRD — Adopt Jina MCP Server for Content & Search Acquisition

## Story
Sales operations wants to unlock richer data gathering and tooling from Jina by adopting the official Model Context Protocol (MCP) server instead of relying solely on HTTP proxies. As the lead generation platform maintainer, I need to introduce an MCP transport layer that lets discovery, enrichment, and evidence flows call MCP tools (read_url, parallel_search_web, etc.) while preserving compatibility with existing synchronous pipelines.

## Acceptance Criteria
- A reusable MCP client module connects to `https://mcp.jina.ai/sse` using our credentials, exposes tool invocation helpers, and gracefully handles reconnects, timeouts, and retries.
- A feature flag (e.g., `ENABLE_MCP_READER`) allows orchestrators to switch specific code paths between the legacy HTTP reader and the MCP-backed client without redeploying.
- Critical flows (`src/jina_reader.read_url`, resolver card builder, enrichment mergers, ICP agents) can fetch web content and search results via MCP and continue emitting prompts/evidence identical in structure to the legacy behavior.
- Observability dashboards and structured logs capture MCP request volume, latency, failure reasons, and tool usage counts alongside existing enrichment telemetry.
- Rollback documentation and automated health checks ensure operators can fall back to the HTTP reader within one deploy if MCP errors spike.

## Dependencies
- MCP server availability and authentication (API key issuance via the Jina dashboard).
- Python MCP client or transport abstractions compatible with our runtime (may require third-party library or in-house implementation).
- Feature flag/configuration management to store endpoint URLs, API keys, and rollout toggles.
- QA coordination with data consumers to validate that MCP responses meet parity thresholds for coverage and quality.

## Success Metrics
- ≥95% MCP tool call success rate during the dual-read rollout window, improving to ≥98% before full cutover.
- ≤5% variance in evidence counts and resolver card accuracy compared to the baseline HTTP reader on the same dataset.
- Zero Sev-1 incidents attributed to MCP transport faults in the first 30 days after enabling it for production tenants.

## Risks & Mitigations
- **Protocol complexity.** MCP introduces session management and streamed responses unfamiliar to the current codebase. *Mitigation:* encapsulate the protocol in a dedicated service module with integration tests and timeouts mirroring existing SLAs.
- **Throughput degradation.** MCP round-trips may increase latency relative to HTTP fetches. *Mitigation:* implement connection pooling, parallel tool invocation where allowed, and staged rollout guarded by latency monitors.
- **Operational blind spots.** Lack of telemetry could hide failures. *Mitigation:* ship Prometheus metrics and structured logs before enabling MCP in production; add alerts for sustained error rates.
- **Vendor quota surprises.** MCP tool limits may differ from legacy quotas. *Mitigation:* negotiate explicit quotas during onboarding and encode protective rate limiting in the client.

## Decisions
1. **Pursue MCP as an additive transport behind a feature flag.** Maintain the HTTP reader for fallback until MCP parity metrics are met.
2. **Centralize MCP interactions in `src/services/mcp_reader.py`.** This consolidates auth, retries, and tool invocation semantics.
3. **Leverage dual-read validation.** During rollout, run MCP and HTTP fetches in parallel for a sampled cohort to diff content quality before making MCP the default.
4. **Adopt Jina's official `jina-mcp` Python client as the foundation.** Delivers protocol compliance quickly while allowing upstream contributions if we need enhanced telemetry or retry hooks.
5. **Expose MCP through synchronous-friendly wrappers backed by thread pools.** Lets legacy pipelines upgrade with minimal refactors while we assess longer-term async adoption.
6. **Enable the full MCP tool suite (`read_url`, `parallel_search_web`, `search_web`) during initial rollout.** Provides comprehensive coverage to validate MCP's value despite the heavier QA lift.
7. **Credential scope for MCP sessions.** Use the shared workspace-wide Jina API key for all MCP traffic, leaning on in-app tenant routing while we validate value and hardening the approach with tenant-level monitoring and a rapid rotation playbook.
8. **Primary telemetry pipeline.** Continue publishing MCP metrics through the existing Prometheus scrape path to minimize rollout complexity, with a scheduled checkpoint to revisit OTLP expansion once usage stabilizes.

## Follow-up Actions

- Document the monitoring thresholds and rotation runbook that make the shared key safe for MCP traffic. <!-- TODO(Codex Agent – Frontend Generator, 2025-03-28) -->
- Capture the Prometheus metric additions required for MCP and define the checkpoint for revisiting OTLP expansion. <!-- TODO(Codex Agent – Frontend Generator, 2025-04-02) -->

