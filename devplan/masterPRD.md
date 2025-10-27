---
owner: Codex Agent – Frontend Generator
status: draft
last_reviewed: 2025-03-24
---

# Master PRD — Jina MCP Adoption Program

## Vision & Background
Our lead generation platform currently depends on Jina's r.jina.ai proxy for deterministic content retrieval. The Jina MCP server unlocks a richer toolset (content extraction, web search, query expansion) delivered over the Model Context Protocol. This initiative replaces the legacy HTTP-only integration with an MCP-based transport that maintains backwards compatibility for synchronous pipelines while widening the accessible tooling surface.

## Goals
- Ship a production-ready MCP client layer that can call Jina tools reliably and expose synchronous-friendly helpers for existing flows.
- Enable orchestrators, enrichment jobs, and agent planners to invoke MCP tools (`read_url`, `parallel_search_web`, `search_web`) without degrading quality or latency.
- Instrument the new transport with feature flags, telemetry, and rollback mechanisms that keep operations confident during rollout.

## Non-Goals
- A full async refactor of orchestration and agent pipelines (tracked separately once MCP adoption stabilizes).
- Broader MCP ecosystem tooling beyond the initial content and search suite unless required for parity.
- Immediate deprecation of the HTTP reader before dual-read parity metrics are satisfied.

## Scope & Deliverables
1. **MCP Client Service** — A dedicated module (planned `src/services/mcp_reader.py`) encapsulating authentication, retries, telemetry hooks, and thread-backed synchronous wrappers over the Jina MCP server.
   - **Credential scope decision.** Proceed with **Option A — Workspace-wide API key.** Reuse the shared Jina key for all tenants, leaning on our internal tenant routing for segmentation while MCP value is validated. Bolster the shared-secret posture with tenant-level monitoring and a documented key rotation procedure to uphold isolation expectations. <!-- TODO(Codex Agent – Frontend Generator, 2025-03-28): Draft the shared-key rotation SOP and monitoring thresholds that uphold tenant isolation expectations. -->
2. **Consumer Integrations** — Updates to `src/jina_reader.read_url`, resolver card builders, enrichment mergers, and ICP agents so they can toggle between HTTP and MCP transport under a feature flag.
3. **Configuration & Observability** — Environment variables, rollout toggles (`ENABLE_MCP_READER`), Prometheus/log metrics, and operational runbooks covering rollout, validation, and rollback steps.
   - **Telemetry sink alignment.** Continue with the Prometheus-first path, extending existing exporters and dashboards to cover MCP metrics so the rollout can iterate quickly, and schedule a checkpoint to reassess OTLP adoption once load patterns are well understood. <!-- TODO(Codex Agent – Frontend Generator, 2025-04-02): Outline the Prometheus metric extensions and define the criteria/timeline for the OTLP reassessment. -->

## User Flow
1. **Greeting & Intent Detection.** Sessions begin in a welcome state that only surfaces guidance until the user explicitly issues a lead-generation command such as “start lead gen,” “find leads,” or “run enrichment,” preventing surprise automation after greetings.【F:app/pre_sdr_graph.py†L5754-L5769】【F:src/conversation_agent.py†L37-L66】
2. **ICP Intake.** Once started, the chat sequence collects a website URL, a list of reference customers with domains, and optional firmographic qualifiers (industries, employee band, geography, buying triggers) before allowing confirmation, aligning with the intake script encoded in the conversation agent reference and the ICP node prompts.【F:src/conversation_agent.py†L41-L64】【F:app/pre_sdr_graph.py†L2991-L3113】
3. **Confirmation & Candidate Preview.** When the user confirms, the router pivots to the candidates node to assemble resolver cards, summarize discovery counts, and block enrichment until a micro-ICP is accepted or an explicit override is issued.【F:app/pre_sdr_graph.py†L5604-L5688】【F:app/pre_sdr_graph.py†L3462-L3703】
4. **Micro-ICP Selection.** Suggested segments can be reviewed and accepted via commands like “accept micro-icp 1,” which persist the rule payload and unlock the enrichment step for the session.【F:app/pre_sdr_graph.py†L5704-L5728】【F:app/pre_sdr_graph.py†L5872-L5933】
5. **Enrichment & Scoring.** After the user runs enrichment, the pipeline executes deterministic crawl plus planner-driven enrichment, persists updates, scores leads, and returns tabular results along with background job follow-ups for larger batches.【F:app/pre_sdr_graph.py†L4855-L5198】【F:src/enrichment.py†L3128-L3338】【F:src/lead_scoring.py†L300-L332】

## Agent Flows & Responsibilities
### Pre-SDR Chat Graph
- **Topology.** The LangGraph-defined chat workflow wires router, ICP intake, candidate prep, confirmation, enrichment, scoring, micro-ICP acceptance, welcome, and Q&A nodes, all looping through a central router to keep state transitions explicit.【F:app/pre_sdr_graph.py†L5698-L5795】
- **Routing Logic.** The router inspects user turns, gating confirm/enrich commands until prerequisites such as website + seeds are satisfied, handling pasted company lists, and diverting free-form questions to the Q&A node without advancing the pipeline prematurely.【F:app/pre_sdr_graph.py†L5604-L5688】

### Conversation Agent
- Provides closed-book Q&A answers and relevance checks grounded in the documented workflow, reinforcing available commands and surfacing next actions without invoking enrichment itself.【F:src/conversation_agent.py†L12-L120】

### ICP Discovery Subgraph
- A dedicated LangGraph orchestrates micro-ICP synthesis, discovery planning, mini-crawl evidence gathering, and scoring/guard rails before feeding results back to the main chat flow.【F:src/agents_icp.py†L1002-L1127】【F:src/agents_icp.py†L1684-L1721】
- Component agents include the `icp_synthesizer` for micro-ICP JSON extraction, the `discovery_planner` for DuckDuckGo queries with hygiene controls, the async `mini_crawl_worker` for homepage snippets, and deterministic evidence extraction plus compliance gating ahead of scoring.【F:src/agents_icp.py†L632-L720】【F:src/agents_icp.py†L720-L840】【F:src/agents_icp.py†L1002-L1109】【F:src/agents_icp.py†L960-L1037】

### Enrichment Planner Agent
- The enrichment module exposes an agentic planner that iteratively selects actions such as deterministic crawl, URL discovery, content extraction, contact lookup, and persistence until the state is complete, overriding loops and protecting vendor quotas along the way.【F:src/enrichment.py†L3128-L3338】

### Lead Scoring Agent
- Lead scoring compiles a LangGraph pipeline that fetches candidate features, scores them (logistic regression or heuristics), assigns A/B/C buckets with manual research bonuses, generates rationales, and persists results for downstream consumption.【F:src/lead_scoring.py†L12-L166】【F:src/lead_scoring.py†L300-L332】

## Success Metrics
- ≥95% MCP call success rate during dual-read trials; ≥98% before full cutover.
- ≤5% variance in resolver card and evidence outputs relative to the HTTP baseline.
- Zero Sev-1 incidents attributable to MCP transport issues in the first 30 production days.

## Milestones
- **Milestone 1:** Land the MCP client service with unit coverage and feature flag plumbing.
- **Milestone 2:** Roll dual-read into staging environments, collect telemetry, and compare against HTTP outputs.
- **Milestone 3:** Expand rollout to production tenants with alerting, concluding with HTTP reader deprecation after success metrics hold.

## Decisions & Rationale
- **Feature Flagged Rollout:** MCP remains optional until parity is demonstrated, ensuring rapid fallback.
- **Official `jina-mcp` Client Adoption:** Using the upstream client accelerates delivery and reduces protocol maintenance overhead while permitting targeted patches.
- **Synchronous Wrapper Strategy:** Thread-pool backed wrappers allow legacy synchronous code to adopt MCP without broad async rewrites.
- **Full Tool Suite at Launch:** Activating `read_url`, `parallel_search_web`, and `search_web` together validates MCP’s broader value proposition despite higher QA demands.

## Open Questions
None at this time. Future uncertainties will be documented here with inline TODO annotations naming owners and due dates.
