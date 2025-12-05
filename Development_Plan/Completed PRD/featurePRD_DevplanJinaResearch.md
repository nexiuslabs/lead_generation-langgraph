# Feature Dev Plan — Jina Deep Research Discovery & Hands‑Free Enrichment

Owner: Codex Agent – Frontend Generator
Status: draft
Last reviewed: 2025-05-14

This document translates the PRD into a concrete, code‑level implementation plan for discovery → enrichment with Jina Deep Research as the primary signal source. It also captures the updated requirement: Remove the old Top‑10/Next‑40 flows entirely. After the user confirms the ICP profile, a background job performs ICP discovery (up to 50 candidates) and full enrichment, then sends an email to the user. No discovery preview or enrichment happens in chat.


## Scope & Outcomes

- Replace Jina Reader homepage pulls with Jina Deep Research for ICP discovery, resolver card fast‑facts, and evidence collection.
- Use Deep Research outputs during enrichment (before r.jina/Tavily), merging into our existing page+summary corpus with provenance.
- Remove chat‑time discovery and enrichment. After ICP confirmation in chat, queue a background job that runs ICP discovery (50 candidates) and enriches them end‑to‑end.
- Add toggles to independently enable Deep Research for discovery and enrichment, with safe defaults and fallbacks.
- Add telemetry for Deep Research usage, latency, error/fallback rates, background queueing and run outcomes. Email delivery status is also logged.


## Architecture Changes

1) New service client: `src/services/jina_deep_research.py`
- Purpose: A thin adapter for Jina’s Deep Research (via MCP tool if available, else direct HTTP API), returning normalized objects used by discovery and enrichment.
- Key functions (sync wrappers + async where needed):
  - `deep_research_query(seed: str, icp_context: dict, *, timeout_s: float = 18.0) -> DeepResearchPack`
  - `deep_research_for_domain(domain: str, *, timeout_s: float = 12.0) -> ResearchSummaryPack`
- Shapes (normalized):
  - `DeepResearchPack = { "domains": ["example.com", ...], "snippets_by_domain": {domain: short_text}, "fast_facts": {domain: {industry_guess, size_band_guess, geo_guess, buyer_titles, integrations_mentions}}, "source": "jina_deep_research" }`
  - `ResearchSummaryPack = { "domain": "example.com", "summary": "...", "pages": [{"url": "...", "summary": "..."}], "source": "jina_deep_research" }`
- Transport selection:
  - Prefer MCP tool (if exposed as `deep_research`/`search_web`) using `src/services/mcp_reader.py` conventions.
  - Fallback to HTTPS with `JINA_API_KEY` and documented endpoint (request/response validated; normalization happens here).
  - Apply rate limiting, retries, and per‑call timeouts similar to MCP reader defaults.

Reference Samples (docs/Jina_deepresearch.md)
- The Deep Research client and its consumers MUST align with the reference sample code and response schema in `docs/Jina_deepresearch.md`:
  - Request shapes: search config array, query fields, limits.
  - Response mapping: `domains`, `snippets_by_domain`, `fast_facts`, and `pages` keys.
  - Tool invocation (MCP) vs HTTP behavior, including streaming vs non‑streaming notes.
  - Error/timeout handling patterns and normalization rules.
  - Add inline comments pointing to the relevant sections of `docs/Jina_deepresearch.md`.

Deep Research Protocol Details (HTTP)
- Endpoint: `POST https://deepsearch.jina.ai/v1/chat/completions`
- Headers: `Content-Type: application/json`, `Authorization: Bearer ${JINA_API_KEY}`
- Required fields (adapted from sample):
  - `model`: `"jina-deepsearch-v1"`
  - `messages`: array of chat messages. We will construct:
    - `{"role":"user","content": <seed + ICP context prompt>}`
  - `stream`: `true` preferred; support `false` fallback
  - `reasoning_effort`: `"low"` by default (configurable)
  - `max_returned_urls`: `"50"`
  - `bad_hostnames`: merge of `config_profiles.deny_host` and environment (e.g., `*.gov`, `*.gov.sg`, etc.)
  - `response_format` (json_schema): when non‑streaming, request a structured object for discovery:
    - Schema example (object):
      - `domains: string[]` (apex or FQDN)
      - `snippets_by_domain: { [domain: string]: string }`
      - `fast_facts: { [domain: string]: { industry_guess?: string, size_band_guess?: string, geo_guess?: string, buyer_titles?: string[], integrations_mentions?: string[] } }`
- Response parsing (streaming):
  - Aggregate `choices[].delta.content` into a single text; collect `choices[].delta.annotations[].url_citation.url` for source mapping.
  - Use `visitedURLs`/`readURLs` arrays to harvest domains.
  - Build `snippets_by_domain` from citations (title/exactQuote) when present; fallback to text windows around cited URLs.
- Response parsing (non‑streaming):
  - Prefer the JSON schema payload when present; otherwise, apply the same citation/URL parsing as above.
- Normalization:
  - Deduplicate/normalize to apex; apply deny‑host/path hygiene.
  - Fill `fast_facts` either from JSON schema (preferred) or via our fast‑facts LLM over the Deep Research text.

2) Settings, flags, and timeouts (src/settings.py)
- Add Deep Research feature toggles and tuning:
  - `ENABLE_JINA_DEEP_RESEARCH_DISCOVERY` (default: false)
  - `ENABLE_JINA_DEEP_RESEARCH_ENRICHMENT` (default: false)
  - `JINA_DEEP_RESEARCH_TIMEOUT_S` (default: 18.0)
  - Reuse existing `JINA_API_KEY` (already used by MCP) or add if missing.
- Keep existing MCP flags (`ENABLE_MCP_READER`, `ENABLE_MCP_SEARCH`) unchanged; Deep Research will use those when tool is available.

3) Discovery pipeline changes (now background‑only)
- `src/icp_pipeline.py`
  - `build_resolver_cards(...)`:
    - Before DDG/Jina Reader, call `deep_research_query(seed_name, icp_context)` when `ENABLE_JINA_DEEP_RESEARCH_DISCOVERY=true`.
    - If Deep Research returns candidate domains and/or domain‑scoped fast‑facts, build cards from those with `confidence` and `why` citing Deep Research.
    - Fallback to current DDG + r.jina flow when Deep Research is unavailable or times out.
  - `collect_evidence_for_domain(...)`:
    - When flag enabled, call `deep_research_for_domain(domain)` and pass the resulting `summary` to `agents_icp.evidence_extractor` instead of a raw homepage body.
    - Preserve existing evidence record shapes and set `source="jina_deep_research"`.

- `src/agents_icp.py`
  - Remove Top‑10 preview in chat entirely. Chat does not render a 50‑row table; it returns a queued job reference only.
  - `discovery_planner` unchanged in surface API; it may pass ICP context to the Deep Research seed query via the new client in the background flow.

4) Enrichment flow changes (src/enrichment.py)
- Background‑only discovery + enrichment:
  - Chat does not run discovery, enrichment, or scoring. A background job executes discovery (50 candidates) and enrichment end‑to‑end.
  - Within the job, use Deep Research first for each seed/domain, then MCP/Jina Reader and Tavily fallbacks. Preserve existing corpus shapes and provenance.
  - On completion, email the user a summary + CSV and perform Odoo tenant export.

5) Orchestrator gating removal + background trigger (my_agent/utils/nodes.py)
- In `journey_guard`:
  - After ICP confirmation, do not perform discovery or enrichment in chat. Enqueue a single background job that performs ICP discovery (50 candidates) and full enrichment, then email/export.
  - Drop discovery preview table, enrichment prompts, and any enrichment nodes from the chat graph.
- In chat path:
  - Remove edges to `plan_top10`, `enrich_batch`, `score_leads`, and any discovery nodes. Chat ends after queuing the job; it returns `job_id` and status.

6) Telemetry (src/obs.py integration)
- Vendor usage: record calls to Deep Research as `vendor="jina_deep_research"` with counts/errors.
- Stage logs: emit `stage="bg_discovery"`, `stage="bg_enrich_queue"`, `stage="bg_enrich_run"`, and `stage="email_notify"` with `start/finish/error` and durations.
- Fallbacks: When Deep Research fails → `obs.log_event(..., extra={"fallback":"ddg"})` in discovery; `{ "fallback": "jina_reader|tavily" }` in background enrichment.


## End‑to‑End Flow

1) Chat intake → profile capture (unchanged)
- User provides company website + ICP. `profile_builder` persists snapshot.
- Once ICP is confirmed, the assistant enqueues a single background job and returns `job_id`.
- Assistant reply in chat: confirm that discovery + enrichment will be processed in the background and results will be emailed to the resolved address. Include the `job_id`.
  - Example: "Thanks — I’ve queued background discovery and enrichment for your ICP. I’ll email the results to you at john@acme.com when it finishes. Job ID: 81237."

2) Background job: discovery → enrichment → export → email
- Planner produces seeds and ICP context.
- Deep Research query first (seed + ICP context) → candidate domains, snippets, fast‑facts (up to 50).
- Fallback to DDG + r.jina when Deep Research is slow/unavailable.
- Persist 50 candidates (`staging_global_companies`) and diagnostics.
- For each candidate, run enrichment: Deep Research summary → merge corpus → LLM extract → contacts via Apify LinkedIn → scoring → persist enrichment history.
- Export results to Odoo tenant DB and generate CSV.
- Send email notification with summary + CSV to the user.

3) Telemetry + diagnostics
- Vendor usage counters; fallback rates; latency percentiles; background queue/run metrics; email delivery outcome.

## Chat UI Messaging

- When the user confirms the ICP profile in chat, the assistant must:
  - Confirm that discovery (up to 50 candidates) and enrichment are running in the background.
  - Provide the resolved notification email address and the background `job_id`.
  - Set expectations on timing and how to retrieve results later (e.g., CSV via email and export endpoint).
- Recipient resolution uses the same policy as the email sender:
  - Header `X-Notify-Email` (dev override) → JWT email → `tenant_users.user_id` when dev guard allows → `DEFAULT_NOTIFY_EMAIL`.
- Do not mask the recipient in the user-facing message (masking applies to logs only).


## User Flow (Before vs After)

- Before: discovery preview → “Ready to enrich?” prompt → waits for ‘enrich 10’ → enrich Top‑10 in chat → background Next‑40.
- After: assistant immediately queues a background job for discovery (50) + enrichment and returns a job reference. No discovery preview table, no chat‑time enrichment, no progress ticks. Results arrive via email and are available for export.


## Module‑Level Changes

- New: `src/services/jina_deep_research.py`
  - Shared client for Deep Research via MCP or HTTP; rate‑limited; typed return shapes.

- Update: `src/settings.py`
  - Add flags: `ENABLE_JINA_DEEP_RESEARCH_DISCOVERY`, `ENABLE_JINA_DEEP_RESEARCH_ENRICHMENT`, `ENABLE_AUTO_ENRICH_AFTER_DISCOVERY`, `JINA_DEEP_RESEARCH_TIMEOUT_S`.
  - Document in `project_documentation.md` or `docs/agents_prompts.md` (see Docs below).

- Update: `src/icp_pipeline.py`
  - `build_resolver_cards`: call Deep Research first; construct `ResolverCard` from normalized `fast_facts`; fallback preserves current logic.
  - `collect_evidence_for_domain`: use Deep Research `summary` as input to `evidence_extractor`; set source to `jina_deep_research`.

- Update: `src/agents_icp.py`
  - Remove Top‑10 preview and discovery table from chat; keep planner utilities for background use only.

- Update: `src/enrichment.py`
  - Introduce an early node/step to fetch and merge Deep Research results into `deterministic_summary` and `extracted_pages` with provenance.
  - Keep all current fallback and extraction behavior intact.

- Update: `my_agent/utils/nodes.py`
  - `journey_guard`: drop enrichment confirmation prompts and flags; after discovery staging, enqueue background enrichment (50 IDs) and present a summary + job ID.
  - Remove chat edges to `plan_top10`, `enrich_batch`, and `score_leads`.

- Update: `src/obs.py`
  - `bump_vendor` usages for `jina_deep_research` in discovery and enrichment call sites.
  - `log_event` stage names for deep research steps.

- Optional: `src/services/mcp_reader.py`
  - If Jina exposes a `deep_research` MCP tool, add a cached resolver akin to `read_url` and call it from the Deep Research client.


## Prompts & Extraction Changes

- Fast‑facts (resolver card) extraction prompt (icp_pipeline.build_resolver_cards)
  - System: “Extract quick company fast‑facts from research text. Return JSON with keys: industry_guess, size_band_guess, geo_guess, buyer_titles (array), integrations_mentions (array). Keep values concise.”
  - Human: “Seed: {name}\nDomain: {domain}\n\n{deep_research_summary}”
  - Note: Replace `{body}` with Deep Research `summary` when available; keep the original prompt as fallback.

- Evidence extractor (src/agents_icp.evidence_extractor)
  - No prompt change required; inputs remain `{ "evidence": [{ "summary": text }] }`.
  - Ensure tests include Deep Research sourced evidence.

- Enrichment extract prompt (src/enrichment.extract_chain)
  - No schema change. Ensure Deep Research summary is included in the `raw_content` assembled corpus within the background job.

Reference Compliance
- Ensure all Deep Research usage (client, discovery, enrichment) complies with `docs/Jina_deepresearch.md`:
  - Match sample request/response structures and field names.
  - Mirror example tool names and parameters for MCP.
  - Keep normalization identical so downstream resolvers/evidence/scoring expect consistent shapes.


## Fallback Strategy

- Discovery (per seed):
  - Try Deep Research (timeout `JINA_DEEP_RESEARCH_TIMEOUT_S` per call) → normalize domains/fast‑facts.
  - On error/timeout/empty: fallback to DDG + r.jina homepage snapshot as today.
  - Emit telemetry with `fallback: ddg` and reason.

- Enrichment (per domain):
  - Try Deep Research → merge summary/page insights into corpus with `source=jina_deep_research`.
  - On error/timeout: continue with current MCP/Jina Reader → Tavily stack.
  - Emit telemetry with `fallback: jina_reader|tavily` and reason.


## Configuration

Environment variables (src/settings.py):
- `ENABLE_JINA_DEEP_RESEARCH_DISCOVERY` (bool, default false)
- `ENABLE_JINA_DEEP_RESEARCH_ENRICHMENT` (bool, default false)
- `ENABLE_AUTO_ENRICH_AFTER_DISCOVERY` (bool, default false initially; set true at rollout)
- `JINA_DEEP_RESEARCH_TIMEOUT_S` (float seconds; default 18.0)
- Uses `JINA_API_KEY` for auth; reuses `MCP_SERVER_URL` when using the MCP transport.

Docs update: add variable descriptions to `project_documentation.md` and `docs/agents_prompts.md`.


## Testing Plan

- Unit tests
  - New: `tests/test_deep_research_client.py`
    - Mocks MCP/HTTP responses; verifies normalization, timeouts, and fallback triggers.
  - Remove: `tests/test_top10_flow.py` and `tests/test_pre_sdr_top10.py` from chat path
    - Replace with background enrichment tests ensuring Deep Research summaries are consumed and fallbacks operate as expected.
  - Modify: `tests/test_orchestrator_nodes.py`
    - Update `journey_guard` expectations: when discovery candidates exist and auto‑enrich flag is enabled, no enrichment confirmation prompt is emitted and `enrichment_confirmed=True`.
  - Modify: `tests/test_orchestrator_regression.py`
    - Ensure `refresh_icp` does not block on `enrichment_confirmed` when auto‑enrich flag enabled.

- Integration tests
  - Gate with flags to avoid external calls. Provide a stubbed Deep Research client in tests to simulate success/failure and assert fallback paths.
  - Add telemetry assertions: vendor counters increment for `jina_deep_research`; fallback logs captured.

- Acceptance check
  - `ENABLE_JINA_DEEP_RESEARCH_DISCOVERY=true ENABLE_JINA_DEEP_RESEARCH_ENRICHMENT=true ENABLE_AUTO_ENRICH_AFTER_DISCOVERY=true make acceptance-check`
  - Verify end‑to‑end: resolver cards cite Deep Research, candidates staged, enrichment auto‑starts, summary/logs show Deep Research usage and fallbacks.


## Rollout Plan

1) Phase 0 — Land behind flags (default off)
- Ship client + plumbing with flags off. CI/acceptance remain green.

2) Phase 1 — Enable for discovery only in dev/staging
- Turn on `ENABLE_JINA_DEEP_RESEARCH_DISCOVERY=true`; confirm resolver cards/preview quality; monitor fallback and latency.

3) Phase 2 — Enable enrichment path
- Turn on `ENABLE_JINA_DEEP_RESEARCH_ENRICHMENT=true`; validate merges, extraction quality, stability; watch Tavily usage deltas.

4) Phase 3 — Remove enrichment confirmation gate
- Set `ENABLE_AUTO_ENRICH_AFTER_DISCOVERY=true` and update tests to default behavior.

5) Phase 4 — Default on
- Flip defaults to true; keep flags for quick rollback.


## Risk & Mitigations

- Latency spikes: Bound via per‑call timeouts and strict fallback; record latency percentiles.
- Response drift: Normalize Deep Research outputs into existing shapes; add validation and tests.
- Orchestrator loops: Explicitly set `enrichment_confirmed=True` once, guard against re‑prompting.
- Vendor caps/costs: Add usage counters; optionally cap Deep Research calls per run/tenant using existing breaker/cap patterns.


## Implementation Notes (per file)

- `src/services/jina_deep_research.py` (new)
  - Mirror `services/mcp_reader.py` rate limiting. Prefer MCP tool if present (e.g., `deep_research`, `search_web` with advanced mode), else HTTP.
  - Return normalized packs (see shapes above). Add `__all__` for exported helpers.

- `src/icp_pipeline.py`
  - In `build_resolver_cards`, call client first; when returning candidates, set `card.why = "deep_research"` and bump confidence when fast‑facts are rich.
  - In `collect_evidence_for_domain`, switch input body to Deep Research `summary` when enabled; set `source="jina_deep_research"`.

- `src/agents_icp.py`
  - `mini_crawl_worker`: replace per‑domain `jina_read(url, timeout=6)` with `deep_research_for_domain(domain)` summary when enabled. Keep 4k truncation.

- `src/enrichment.py`
  - Add early optional “Deep Research snapshot” step: set `state["deterministic_summary"]` if empty and push 1–3 synthetic pages to `extracted_pages` with source tag. Respect `LLM_MAX_CHUNKS`.

- `my_agent/utils/nodes.py`
  - `journey_guard`: remove “Ready for me to enrich?” prompt; set `enrichment_confirmed=True` upon staging persist; do not set `awaiting_enrichment_confirmation`.
  - `refresh_icp`: allow proceed without checking `enrichment_confirmed` when `ENABLE_AUTO_ENRICH_AFTER_DISCOVERY`.

- `src/obs.py`
  - Add `bump_vendor(..., vendor="jina_deep_research")` from discovery/enrichment call sites.
  - Add `log_event` calls with stage names and outcomes.


## Developer Ergonomics

- Local run examples
  - Discovery preview: `cd lead_generation-main && make acceptance-check-tenant TID=1034 ENABLE_JINA_DEEP_RESEARCH_DISCOVERY=true ENABLE_AUTO_ENRICH_AFTER_DISCOVERY=true`
  - Orchestrator debug: `cd lead_generation-main && python -m scripts.run_orchestrator --tenant-id 1034 --input samples/prompt.json`
  - Tail logs: `cd lead_generation-main && make logs-tail`

- Env setup
  - Export `JINA_API_KEY` and confirm `MCP_SERVER_URL` if using MCP transport. Set `POSTGRES_DSN`.


## Documentation & Comms

- Update `docs/agents_prompts.md` with Deep Research positioning, new flags, and prompt tweaks.
- Update `project_documentation.md` (env var descriptions, rollback strategy).
- Annotate `AGENTS.md` mermaid graph to remove the HITL enrichment gate once Phase 4 is live.


## Acceptance Criteria Mapping

- Discovery returns up to 50 candidate domains with resolver fast‑facts citing Deep Research; fallback only on failure.
- Chat does not perform enrichment; instead, a background job is queued for all discovered candidates and the job reference is returned.
- Background job runs Deep Research‑first enrichment per domain, merges findings, performs extraction, scoring, and export.
- Orchestrator removes chat gating and Top‑10/Next‑40 concepts from the chat path.
- Config toggles exist; safe defaults and logging on failures; telemetry covers discovery, queue, and background run stages.


## Implementation Delta Checklist

- Orchestrator graph (my_agent/utils/nodes.py)
  - Remove chat edges to `plan_top10`, `enrich_batch`, and `score_leads`.
  - In `journey_guard`, after persisting up to 50 candidates, enqueue a background enrichment job with those IDs and surface `job_id` in `status`.
  - Drop uses of `awaiting_enrichment_confirmation` and `enrichment_confirmed`; do not proceed to enrichment nodes in chat.
  - Ensure `refresh_icp` is not invoked in chat mode; it can remain for non‑chat/nightly paths if needed.

- Background jobs (src/jobs.py)
  - Promote a generic function to enqueue/process enrichment for an explicit list of candidate IDs (all 50). Reuse or rename `enqueue_web_discovery_bg_enrich` to reflect “all candidates” instead of “Next‑40”.
  - Update log/event payloads and docstrings to remove “Next‑40” semantics.
  - Ensure the worker reads staged candidates reliably and handles idempotency/retries.

- API/routers (app/icp_endpoints.py)
  - Remove calls to `plan_top10_with_reasons` from confirmation/preview endpoints.
  - Ensure discovery preview renders the 50‑row table and returns a queued background job reference when applicable.

- Agents (src/agents_icp.py)
  - Remove Top‑10 preview logic from chat flows entirely; keep discovery table building independent of chat‑time crawls.
  - Retain agent utilities for non‑chat use only (if required by other flows), or deprecate with clear comments.

- Enrichment (src/enrichment.py)
  - No chat entry points; ensure the module is only invoked by the background job.
  - Keep Deep Research‑first merge and existing fallbacks intact; confirm idempotent persistence.

- Settings (src/settings.py)
  - Add/confirm flags: `ENABLE_JINA_DEEP_RESEARCH_DISCOVERY`, `ENABLE_JINA_DEEP_RESEARCH_ENRICHMENT`, and `BG_DISCOVERY_AND_ENRICH=true`.
  - Optional: `CHAT_DISCOVERY_ENABLED=false` and `CHAT_ENRICHMENT_ENABLED=false` (defaults) to guard any legacy paths.

- Telemetry (src/obs.py)
  - Add stages `bg_enrich_queue` and `bg_enrich_run`; record job `job_id`, candidates count, success/failure counts.
  - Keep vendor counters; add Deep Research usage in background runs.

- Docs
  - Update `AGENTS.md` mermaid graph to end chat after job enqueue.
  - Remove Top‑10/Next‑40 and chat preview terminology from user‑facing docs and examples.

- Tests
  - Remove/refactor Top‑10 and chat discovery preview tests: `tests/test_top10_flow.py`, `tests/test_pre_sdr_top10.py`.
  - Update `tests/test_orchestrator_nodes.py` to expect job enqueue after ICP confirmation and no discovery/enrichment prompts.
  - Add tests for background job flow (discovery → enrichment → export → email) with mocks for vendor calls and email sender.
  - Update acceptance check to assert: background job runs full pipeline, persists rows, exports to Odoo, and sends email.

## Agentic Background Flow (Proposed)

- Define a LangGraph state machine `bg_icp_graph` (or reuse existing orchestration) with nodes:
  - `prepare_context` → `plan_discovery` (Deep Research first) → `persist_candidates` → `enrich_company` (map over 50) → `score` → `export_odoo` → `email_notify` → `finalize`.
- State includes: `tenant_id`, `icp_profile`, `seeds`, `candidate_ids`, `diagnostics`, `enrichment_runs`, `export_paths`, `email_recipient`.
- Idempotency: guard inserts/updates via upsert patterns; resume-safe checkpoints per stage.
- Observability: emit `obs.log_event` per node; bump vendor usage; mark run degraded on fallbacks.


## Open Questions & Defaults

- MCP vs HTTP transport: default to MCP when `ENABLE_MCP_SEARCH` is on and a `deep_research`/`search_web` tool is visible; otherwise use HTTP with `JINA_API_KEY`.
- Partial candidate sets: proceed with auto‑enrichment when ≥1 candidate, log per‑domain failures (matches PRD decision).
- Latency SLO: target p95 ≤ 7s for discovery seed call; revisit batching/parallelism if exceeded.


---
Appendix: Minimal client signature (sketch)

```
# src/services/jina_deep_research.py
from typing import Any, Dict, List, Optional, Tuple

def deep_research_query(seed: str, icp_context: Dict[str, Any], *, timeout_s: float = 18.0) -> Dict[str, Any]:
    """Return { domains, snippets_by_domain, fast_facts, source }."""
    ...

def deep_research_for_domain(domain: str, *, timeout_s: float = 12.0) -> Dict[str, Any]:
    """Return { domain, summary, pages, source }."""
    ...
```
