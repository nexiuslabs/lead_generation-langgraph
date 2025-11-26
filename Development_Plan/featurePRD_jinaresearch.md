---
owner: Codex Agent – Frontend Generator
status: draft
last_reviewed: 2025-05-14
---

# Feature PRD — Jina Deep Research Discovery & Hands-Free Enrichment

## Story
Our sales-assist orchestrator currently leans on DuckDuckGo heuristics plus Jina Reader snapshots and pauses for human confirmation before enrichment. We want to pivot discovery and enrichment to a Jina Deep Research-first strategy and automatically continue into enrichment once discovery results are available, eliminating the human-in-the-loop pause.

## Goals
- Replace Jina Reader homepage pulls with Jina Deep Research findings throughout ICP discovery, resolver card construction, and evidence gathering.
- Use Jina Deep Research outputs during enrichment crawls to feed extraction chains before falling back to existing r.jina/Tavily pathways.
- Remove manual enrichment confirmations so enrichment begins automatically after discovery completes successfully.

## Acceptance Criteria
- Resolver cards and fast-facts for seeds are generated from Jina Deep Research queries (seed + ICP context) with confidence/why fields citing Deep Research; only fall back to DDG/Jina Reader if Deep Research is unavailable.
- Mini-crawl evidence and `collect_evidence_for_domain` use Jina Deep Research summaries/content for candidate domains while preserving current evidence shapes and diagnostics.
- Enrichment crawl path consults Jina Deep Research for structured findings before r.jina/Tavily, merging results into existing page+summary structures with source metadata indicating Deep Research when used.
- Orchestrator `journey_guard` and `refresh_icp` no longer block on `awaiting_enrichment_confirmation`/`enrichment_confirmed`; enrichment starts automatically once discovery candidates are ready and persisted.
- Configuration toggles/env vars exist to enable/disable Jina Deep Research separately for discovery and enrichment, with default-safe fallbacks and logging for failures.
- Telemetry records Deep Research usage, success/failure rates, and fallback paths for discovery and enrichment nodes.

## Dependencies
- Jina Deep Research API credentials and rate limits; alignment with existing Jina MCP settings.
- Orchestrator state machine changes in `my_agent/utils/nodes.py` to bypass HITL enrichment gating.
- Downstream extractors (`agents_icp.evidence_extractor`, `extract_chain`) remain compatible with Deep Research output formats.
- Observation/logging pipeline to capture Deep Research performance and fallback behavior.

## Non-Goals
- Replacing DuckDuckGo heuristics entirely; they remain a backup when Deep Research cannot return candidates.
- Broader UI/UX changes to the chat surface beyond removing the enrichment confirmation prompt.
- New scoring heuristics; lead scoring remains unchanged.

## Open Questions (resolved with recommended choices)
1. **How should we position Deep Research relative to existing MCP reads?**
   - **Decision:** Option A — Make Deep Research the first call with a hard fallback to r.jina/Tavily only on errors/timeouts to keep costs predictable and behavior deterministic; revisit parallelism only if latency proves unacceptable.

2. **Should we auto-enrich partial candidate sets?**
   - **Decision:** Option A — Proceed with auto-enrichment when at least one candidate is available, logging per-domain failures while maintaining hands-free progress even for narrow ICPs.

## Risks & Mitigations
- **Latency spikes from Deep Research** → Set per-call timeouts and fallback to DDG/r.jina with clear telemetry.
- **Schema drift between Deep Research outputs and extractors** → Normalize Deep Research responses into current resolver/evidence schemas with validation and tests.
- **Accidental infinite loops if gating flags not cleared** → Add explicit state resets when discovery concludes and log transitions to enrichment.
