# Feature Dev Plan 19 — Implementation Plan for PRD19 Pre‑Enrichment Lead Generation

References: Development_Plan/Development/featurePRD19Revised.md (PRD), lead_generation-main backend, agent-chat-ui frontend, Odoo integration.

## Scope

Deliver the PRD19 pre‑enrichment layer: minimal intake (website + seeds), agent micro‑ICP synthesis, instant Top‑10 lookalikes (persisted preview), immediate Top‑10 enrichment (run‑now), Non‑SG background next‑40 enrichment with chat notification, ResearchOps Markdown import, tenant‑specific ACRA nightly discovery (prioritized queues), deterministic scoring/gating, and Odoo mirroring. Implement LangChain/LangGraph agents and tools that automate discovery and evidence extraction before heavy enrichment.

## Preferred User Story & Process

- A user submits their business website plus multiple customer websites via chat.
- The agent crawls and analyzes these sites to:
  - Understand the user’s business model from the user site.
  - Identify common patterns across customer sites (integrations, titles, operations, industry signals).
  - Build an ICP profile/pattern from these insights and persist it.
- With the ICP established, the agent:
  - Discovers additional ICP‑matching customers (instant Top‑10 preview and queued remainder).
  - Persists all discovered web candidates into `staging_global_companies` (source `web_discovery`) and shows the total discovery count in chat; Top‑10 rows carry `ai_metadata.preview=true`, `score`, `bucket`, `why`, `snippet`.
  - Enriches the Top‑10 immediately upon user action (strictly those 10). For Non‑SG ICPs, automatically enqueues the next 40 in the background and informs the user in chat; posts a completion message when done.
  - Triggers Odoo syncing for enriched companies.
- Nightly runner (existing) processes discovery + enrichment + Odoo sync.
- Country‑specific handling:
  - Singapore: leverage ACRA (UEN/SSIC) via existing staging tables and flows.
  - Other countries: persist to `companies` (with country fields), enrich, and include in nightly runs.
- Infrastructure: A unified, scalable design supports both SG ACRA and non‑SG workflows.

## User Journey Flow

1) Seeds → user submits 3–10 seed customers (domains/UENs) in UI or via API.
2) Instant results → micro‑ICP synthesized, candidates searched/crawled, Top‑10 displayed with badges/why and persisted preview.
3) ResearchOps import (DB‑first) → analyst submits research via `/icp/research/import` (CLI/UI). Markdown under `docs/` is optional input only; the system persists artifacts to DB and updates candidates/scores.
4) Nightly additions → ACRA tenant‑specific candidates appear by morning; A‑bucket only by default.
5) Shortlist → SDR filters A‑bucket and exports/syncs; enrichment unlocks for gated candidates.

## Detailed User Journey (Chat → Enrichment → Nightly)

1) Chat start (UI → Graph)
- Message 1: user pastes their business website.
- Message 2: user pastes multiple customer websites.
- UI forwards each to the graph; once both are present, the UI sends `{ user_website, customer_websites[] }` to the `icp_discovery` node in `app/pre_sdr_graph.py`.

2) Crawl + understand (Graph)
- Graph crawls the user site to infer business model and the customer sites to extract common signals (SSIC hints, integrations, buyer titles, ops signals, pricing/case studies, hiring roles).
- Smart inference first: agents infer industries, employees/size band, revenue bucket, incorporation years, and geos directly from site content, metadata, registry hints, LinkedIn signals, and TLD/address sections.
- Minimal questions: only if a field is low‑confidence or unreasonable, the agent asks a short, focused follow‑up for that specific gap; retries if needed. Clarifications persist via `save_icp_intake` into `icp_intake_responses`.
- State accumulates normalized signals and proposes micro‑ICP cards.

3) Persist ICP (Backend)
- On user confirmation, UI calls `POST /icp/accept` to persist the ICP payload to `icp_rules` for the tenant. This becomes the active profile used by discovery and nightly.
 - The ICP record is upserted (per-tenant, per-name). As confirm and enrichment derive new `icp_profile` fields (industries, integrations, buyer_titles, size_bands, triggers), those fields are merged into `icp_rules.payload` so the profile evolves over time and is reused by Top‑10 discovery and nightly jobs.

4) Instant discovery (Graph + Backend)
- Graph plans queries from the active ICP and discovers candidate domains (DDG HTML endpoint only), mini‑crawls, and extracts evidence to compute EvidenceScore. Seed domains are excluded from Top‑10. Strong domain validation drops images/files (e.g., .webp/.jpg/.html).
- The system also applies the ICP intake filters (industries, employees_min/max, revenue_bucket, year range, geos, signals) to the existing companies catalog (`_default_candidates` in `pre_sdr_graph`) to treat matches as ICP customers and include them in the shortlist.
- UI renders Top‑10 immediately; remainder is queued in `discovery_queue`. All discovered domains are also persisted into `staging_global_companies` for auditability with `ai_metadata` provenance; Top‑10 preview rows include `ai_metadata.preview=true`, `score`, `bucket`, `why`, `snippet`.

5) Top‑10 enrichment (strict) + Non‑SG background next‑40 (Backend)
- Top‑10: User clicks “Enrich Top‑10 now” or types `run enrichment`. Enrichment reuses the persisted Top‑10 preview for the tenant; it never re‑discovers nor mixes other sources into this immediate batch. Steps: mini‑crawl (4–6 pages) → LLM extraction merge (bounded) → Apify LinkedIn contacts (domain‑first; respect caps) → ZeroBounce verify → score/bucket → persist; then export to Odoo when ready.
  - Domain‑first Apify: When the candidate domain is known, build Apify queries anchored on the website domain combined with ICP `buyer_titles`; fall back to company‑name queries only when domain is missing. Log titles used per company for traceability.
- Next‑40 (Non‑SG only): After Top‑10 enrichment begins, enqueue background enrichment for the next 40 from the same preview list (excluding Top‑10). Return `{ job_id }` and have the agent inform the user it’s running in the background; the agent polls `/jobs/{job_id}` (or uses SSE) and posts a completion message with processed counts and A/B/C split. Respect vendor caps and throttle concurrency.

6) Optional ResearchOps import (Backend)
- Analysts submit artifacts via `POST /icp/research/import` or CLI. Importer writes `icp_research_artifacts`, `icp_evidence(research_*)`, enqueues `manual_research` candidates with a bounded bonus, and re‑scores. Drafts under `docs/` are optional and treated as provenance only.

7) Nightly + Background worker (Jobs)
- Dedicated worker: A separate background service (`scripts/run_bg_worker.py`) consumes `web_discovery_bg_enrich` jobs as soon as they are enqueued after Top‑10. It uses Postgres LISTEN/NOTIFY on channel `bg_jobs` (with a short polling sweep fallback), safe job claiming via `FOR UPDATE SKIP LOCKED`, and bounded concurrency (env `BG_WORKER_MAX_CONCURRENCY`). The chat/API only enqueues; it never runs the job inline.
- Nightly order: `web_discovery → manual_research → acra_ssic` (configurable via `SCHED_PRIORITY_SOURCE_ORDER`). Nightly does not process `web_discovery_bg_enrich`; the worker handles those.
- Nightly ACRA bootstrap (first run per tenant): In `scripts/run_nightly.py`, before consuming the queue, resolve ACRA terms from `icp_rules` (preferred_titles → industries → ssic_codes→titles). If the tenant has no ACRA jobs yet, enqueue one `staging_upsert(terms=…)` so candidates come from `staging_acra_companies`.
- Nightly ACRA daily cap: In `src/jobs.run_enrich_candidates`, enforce a per‑tenant daily cap (default 20 via `ACRA_DAILY_ENRICH_LIMIT`). Compute today’s already‑enriched count by joining `company_enrichment_runs`→`enrichment_runs` filtered by `started_at >= date_trunc('day', now())`, limit the SELECT by the remaining, and if exhausted re‑queue the job with `status='queued'` (deferred) so it resumes the next night. Finalize the run header per job.
- SG: pull SSIC candidates from `staging_acra_companies`, anchor by UEN, crawl, score, enrich.
- Non‑SG: persist companies with country fields, crawl, score, enrich.
- Odoo sync executes post‑enrichment; A‑bucket is presented by default.

## Process Flow (System)

- Seeds arrive → ICP Synthesizer (LLM) persists tenant ICP payload to `icp_rules` via `/icp/accept` → Discovery Planner (LLM) generates queries using the latest `icp_rules.payload` → Mini‑Crawl Worker fetches pages → Evidence Extractor (LLM) writes `icp_evidence` → Scoring & Gating computes Top‑10 and queues remainder.
- ResearchOps branch ingests submissions (optionally from a `docs/` root) → writes `icp_research_artifacts`, `icp_evidence(research_*)`, and enqueues `manual_research` candidates → scoring updates. No `.md` outputs are created by the system under `docs/`.
- Nightly ACRA branch: first run auto‑enqueues a `staging_upsert` from tenant ICP; then nightly processes queued `staging_upsert` → `enrich_candidates` with a per‑tenant daily cap (20 default). Each enrich job opens an `enrichment_runs` header and links per‑company history for accurate per‑day accounting.
- Odoo Sync mirrors seeds and active ICP profile; companies sync opportunistically.

Country routing during crawl and persistence:
- Attempt ACRA anchor for candidates with SG signals (domain TLD .sg, SSIC hint, or detected `hq_country = 'Singapore'`):
  - Use existing `staging_acra_companies` mapping to upsert companies by UEN; attach SSIC evidence.
  - Persist to `companies(uen,name,industry_code,...)` with ACRA provenance.
- For non‑SG candidates:
  - Persist or upsert in `companies(name, website_domain, hq_city?, hq_country?)`.
  - Persist discovered domains into `staging_global_companies` with `source='web_discovery'`.
  - Enrich via existing pipeline; include in nightly scoring and Odoo sync.

## Architecture & Components

- Backend (lead_generation-main): FastAPI + LangGraph + Postgres.
  - New: research import module, endpoints, migrations, scoring hook.
- Frontend (agent-chat-ui): Next.js app surfacing Top‑10, ACRA list, “Ingest docs/” action and upload fallback.
- Jobs/Scheduler:
  - Nightly harvesting and periodic research import.
  - New dedicated worker for next‑40 web discovery enrichment (`scripts/run_bg_worker.py`).
  - Nightly ACRA bootstrap + cap: `scripts/run_nightly.py` now bootstraps ACRA jobs per tenant once; `src/jobs.run_enrich_candidates` enforces `ACRA_DAILY_ENRICH_LIMIT` and re‑queues when quota is hit.
- Agents (LangChain/LangGraph): ICP Synthesizer, Discovery Planner, Evidence Extractor, Research Import Processor, Scoring/Gating, ACRA Orchestrator, Odoo Sync.

Implemented LLM agents (new):
- File: `lead_generation-main/src/agents_icp.py`
  - `icp_synthesizer` (ChatOpenAI): synthesizes micro‑ICP from seeds/prior.
  - `discovery_planner` (ChatOpenAI + DDG HTML endpoint): generates queries and deduped domains; enforces strict domain validation and excludes seeds. Paginates DDG result pages per query until 50 unique domains; early‑stops remaining queries when a single query reaches 50.
  - `mini_crawl_worker` (tool): crawls candidate domains via existing pipeline.
  - `evidence_extractor` (ChatOpenAI structured): normalizes crawl summaries to evidence fields.
  - `scoring_and_gating` (deterministic stub): computes A/B/C; integrate with `lead_scoring.py` in prod. Top‑10 is always filled to 10 via controlled backfills (DDG first, Jina‑only fallback last).
  - `build_icp_agents_graph()`: composes nodes into a LangGraph for plug‑in or testing.

## LangChain/LangGraph Integration Map (By File)

- `app/pre_sdr_graph.py` (chat runtime)
  - Use `icp_synthesizer` during ICP discovery to augment inferred ICP from pasted websites before proposing micro‑ICPs (low‑latency call; cacheable per thread).
  - In the discovery phase, call `discovery_planner` to generate queries and candidate domains; merge with existing Tavily flow; dedupe into `state["candidates"]`.
  - After mini‑crawl (existing `collect_evidence_for_domain`), apply `evidence_extractor` to normalize signals before persistence/scoring.
  - Persist web discovery candidates into `staging_global_companies` (with `ai_metadata` provenance) and record the web discovery total in chat state. For Top‑10 preview rows, store `ai_metadata.preview=true` with `score`, `bucket`, `why`, `snippet`.
  - Enrichment uses a strict Top‑10 path when available (build candidates from `agent_top10` domains and enrich those first). If Top‑10 is not available in memory, load last persisted Top‑10 by tenant; otherwise, prompt the user to confirm to regenerate — never enrich arbitrary fallbacks, and do not crawl seeds as a substitute during the immediate batch. For Non‑SG, trigger background next‑40 after starting the Top‑10 batch and surface `{ job_id }` to the chat agent for status updates.
  - Keep deterministic gating in DB; the in‑memory `scoring_and_gating` remains a safe fallback for preview lists only.

- `app/lg_entry.py` (graph entry / testing)
  - Expose a dev/test endpoint or CLI path to run `build_icp_agents_graph()` against a small state for smoke tests.
  - Optionally add a feature flag `ENABLE_AGENT_DISCOVERY` to route preview discovery through the agents graph for comparison logging.

- `src/icp_pipeline.py` (crawl + evidence)
  - After `crawl_site(...)` returns a summary, pass the merged corpus through `evidence_extractor` and persist normalized fields into `icp_evidence`.
  - Preserve current regex/heuristic signals as a fallback; prefer LLM extraction when confidence is adequate.

- `src/orchestrator.py` (ad‑hoc orchestration)
  - Add helpers to invoke `icp_synthesizer` and `discovery_planner` to generate candidate domains from a stored ICP rule and run a light mini‑crawl + extraction pass for batch experiments.

- `src/lead_scoring.py` (deterministic scoring)
  - Keep the source of truth for scoring in SQL/DB; map fields produced by `evidence_extractor` so they contribute to `EvidenceScore` and `reason_json`.

- `app/icp_endpoints.py` (APIs)
  - For fast preview routes (e.g., suggestions/top‑k), optionally use `icp_synthesizer` and `discovery_planner` when `ENABLE_AGENT_DISCOVERY=true` to improve first‑run relevance.

- `scripts/run_scheduler.py` / `scripts/run_nightly.py` (jobs)
  - Maintain queue order (web_discovery → manual_research → acra_ssic). For the `web_discovery` slice, allow an agent‑driven pass (`discovery_planner` + `mini_crawl_worker` + `evidence_extractor`) before DB scoring.

- `agent-chat-ui` (UI)
  - No direct LLM calls from the browser. Chat messages hit the server graph; ICP page fetches Top‑10 via APIs. Surface results that were produced by agents via standard endpoints.

## Feature Flags and Config

- `ENABLE_AGENT_DISCOVERY=true|false` — Use `discovery_planner` + `evidence_extractor` in preview/instant discovery flows (default: off; enable in staging).
- `AGENT_MODEL_DISCOVERY=gpt-4o-mini` — Override the model for planning/extraction agents.
- `DOCS_ROOT` — Path used by `/icp/research/import`.
- `STAGING_GLOBAL_TABLE` — Name of the staging table for web discovery candidates (default: `staging_global_companies`).

## Minimal Integration Steps (PR ready)

1) pre_sdr_graph: import from `src.agents_icp` and call `icp_synthesizer` right after ingesting websites; memoize result on the thread state.
2) pre_sdr_graph: when building preview candidates, call `discovery_planner` and merge with existing candidates; dedupe and cap.
3) icp_pipeline: after `crawl_site`, call `evidence_extractor` and persist normalized fields (where present) into `icp_evidence` with `source='extractor'`.
4) lead_scoring: ensure new fields (integrations, titles, counts) from extractor are included in EvidenceScore; add `research_sources` passthrough when present.
5) scheduler: behind feature flag, allow a limited `web_discovery` agent pass prior to DB selection for preview freshness.

Additional implementation steps:
- pre_sdr_graph: persist all discovered web candidates to `staging_global_companies`; show total count in chat.
- pre_sdr_graph: in enrichment path, if Top‑10 exists, construct candidates from Top‑10 domains and enrich strictly those (do not top‑up from ACRA in the run‑now batch).

## Migrations (added)

- 018_staging_global_companies.sql — create `staging_global_companies` with unique index on `(tenant_id, domain, source)` and a per‑tenant created_at index.

Chat intake specifics (graph):
- Extend `pre_sdr_graph.icp_discovery` to accept arrays of websites: `{ user_website, customer_websites[] }`.
- Crawl user_website for business model; crawl customer_websites for common signals; synthesize and persist ICP rule for tenant.
- Immediately run discovery using the persisted ICP; present Top‑10; queue remainder.

## Deliverables and Code Changes

### 1) Database Migrations

- File: `lead_generation-main/app/migrations/016_prd19_research_ops.sql`
- Content (from PRD): tables `icp_research_artifacts`, `research_import_runs`; GIN indexes on `fit_signals` and `ai_metadata`.
- Optional MV: `icp_features_web(...)` if not present; otherwise ensure it includes fields used by scoring/UX.

### 2) Backend Schemas (Pydantic)

- File: `lead_generation-main/schemas/research.py`
  - `ResearchArtifactIn { tenant_id, company_hint?, website?, path, snapshot_md, source_urls[], fit_signals{tags[],ops_signals[]} }`
  - `ResearchImportRequest { tenant_id, root?: str, files?: List[UploadFile] }`
  - `ResearchImportResult { files_scanned: int, leads_upserted: int, errors: List[str] }`

### 3) Research Import Core Module

- File: `lead_generation-main/src/research_import.py`
- Functions:
  - `parse_docs(root: Path) -> List[ResearchArtifactIn]` — parse `docs/ideal_customer_profile.md`, `docs/profiles/*`, `docs/leads_for_nexius.md` into artifacts; extract citations and signals.
    - Honor the preferred workflow and methodology:
      - Read baseline term stack from `docs/lead-search-keywords.md` (optional); store in `ai_metadata.search_keywords` for traceability.
      - Support snapshots collected via `curl -Ls https://r.jina.ai/<url>` and DuckDuckGo queries via `curl -Ls "https://r.jina.ai/https://duckduckgo.com/?q=<query>&ia=web"`.
  - `resolve_company(db, tenant_id, website, company_hint) -> company_id`
  - `upsert_artifact(db, artifact)` → insert into `icp_research_artifacts` and attach `ai_metadata.provenance`.
  - `write_research_evidence(db, tenant_id, company_id, artifact)` → `icp_evidence(research_fit|research_note)`.
  - `enqueue_manual_research(db, tenant_id, company_id, priority)` → `discovery_queue` with `source='manual_research'`.
  - `import_docs_for_tenant(tenant_id: int, root: str|Path) -> ResearchImportResult` (idempotent via content hash).
  - `manual_research_bonus(artifact, icp_profile) -> int` (capped by env `MANUAL_RESEARCH_BONUS_MAX`).
  - Persisted artifacts are “effectively used” by: (a) creating `icp_evidence` that feeds EvidenceScore, (b) adding ManualResearch bonus, and (c) enqueuing discovery candidates with appropriate priority.

### 4) API Endpoints

- File: `lead_generation-main/app/icp_endpoints.py` (extend existing router):
  - `POST /research/import` — body: `ResearchImportRequest`. Options: scan `root` on server, or accept uploaded files. Returns `ResearchImportResult`.
  - Ensure RLS/tenant guard based on auth; attach `ai_metadata` with user/agent for provenance.

Add/confirm support for chat intake of multiple websites:
- Graph entry (`app/pre_sdr_graph.py`): state accepts `{ user_website: str, customer_websites: List[str] }` from chat.
- If needed, add a thin endpoint to kick off discovery with websites payload (or rely on graph channel only).

### 5) Scoring Hook

- File: `lead_generation-main/src/lead_scoring.py`
  - Add `manual_research_bonus` to the EvidenceScore accumulator or at final compositing per PRD19 formula.
  - Persist `reason_json.research_sources` if present.

### 6) Scheduler / Jobs

- File: `lead_generation-main/scripts/run_scheduler.py`
  - Add research import periodic job (optional) to scan a configured `DOCS_ROOT` per tenant.
  - Preserve existing nightly order: `web_discovery → manual_research → acra_ssic` in dispatcher.
- File: `lead_generation-main/scripts/run_nightly.py`
  - Ensure queues process with the updated priority policy; log bonus contributions.

Country‑specific nightly handling:
- SG: ensure ACRA selection uses `staging_acra_companies` by SSIC; enrich by UEN anchoring when available.
- Non‑SG: ensure newly persisted `companies` (without UEN) are included in queue selection by evidence/ICP match and enriched similarly.

### 7) LangGraph Integration

- File: `lead_generation-main/app/pre_sdr_graph.py`
  - Register nodes: `import_docs` (tool), `plan_discovery` → `mini_crawl` → `extract_evidence` → `score_gate`.
  - Add edge from `import_docs` to `score_gate` to re‑score when research arrives.
- File: `lead_generation-main/src/orchestrator.py`
  - Expose helpers to call graph nodes or tools for ad‑hoc runs.

Country detection tool:
- Add a small utility in `src/enrichment.py` (or a shared util) to infer `hq_country` from signals and TLD; prefer explicit evidence.

ICP intake clarifications (reuse existing with confidence gating):
- Prefer automatic inference of industries, employees, revenue, incorporation years, geos, and buying signals.
- Gate `next_icp_question()` calls behind confidence checks; ask only when inference is low‑confidence or contradictory, and re‑ask if the user’s answer is unreasonable.
- Persist each clarification with `save_icp_intake()`; merge into the active ICP payload in `icp_rules`.
- Apply these filters within discovery using `_default_candidates()` and combine with web discovery evidence for Top‑10.

### 8) Frontend (agent-chat-ui)

- Top‑10: ensure page `agent-chat-ui/src/app/icp/page.tsx` displays Top‑10 cards with badges/why tooltips via `GET /icp/top10` (already present per PRD18/19 context). Persist accepted ICP by calling `POST /icp/accept` from the UI when the user confirms a suggestion.
- ACRA list: add/verify `agent-chat-ui/src/app/candidates/page.tsx` (or similar) showing nightly A‑bucket.
- ResearchOps import: add an action in `icp` page to POST `/research/import` with either:
  - `root` path (for server‑side docs) or
  - drag‑and‑drop upload fallback.
- Update `/api/backend/[..._path]` proxy if needed to allow multipart upload.

Lead Research UX notes:
- Provide a short helper panel linking to `docs/mycompanyprofile.md`, `docs/ideal_customer_profile.md`, `docs/lead-search-keywords.md`, and `docs/leads_for_nexius.md` conventions and examples.
- Show import results with counts of files scanned, leads upserted, and highlight any missing “Sources” sections for traceability.

Chat UX:
- In the conversation, prompt to paste the business website and multiple customer websites; pass `{ user_website, customer_websites[] }` to the graph; surface progress and Top‑10.

### 9) Odoo Sync

- File: `lead_generation-main/app/icp_endpoints.py` and `lead_generation-main/app/odoo_store.py`
  - Confirm seed mirroring and active ICP record sync; add `x_ai_metadata` where applicable.

## APIs (New/Updated)

- `POST /icp/accept`: persist micro‑ICP to `icp_rules` (active profile).
- `GET /icp/top10`: return persisted Top‑10 preview (why/snippets, preview=true rows).
- `POST /icp/enrich/top10`: run immediate enrichment strictly for the persisted Top‑10; return enriched shortlist.
- `POST /icp/enrich/next40` (Non‑SG only): enqueue background enrichment for the next 40 from the same preview; returns `{ job_id }`.
- `GET /jobs/{job_id}`: poll job status; agent posts a chat message when status becomes `done`.
- `POST /research/import`: ingest artifacts; enqueue `manual_research` candidates.

## Implementation Steps (Sequenced)

1) DB: add `016_prd19_research_ops.sql`, run migration; verify DDL and GIN indexes.
2) Backend: implement `schemas/research.py` and `src/research_import.py`; unit‑test parsing and upsert idempotency.
3) API: extend `app/icp_endpoints.py` with `POST /research/import`; wire to router in `app/main.py`. Add `POST /icp/enrich/top10` and `POST /icp/enrich/next40` (Non‑SG only). Confirm `/icp/accept` persists to `icp_rules` and that enrich‑now paths reuse the persisted Top‑10 preview.
4) Scoring: adjust `src/lead_scoring.py` to include manual research bonus; update `reason_json`.
5) Scheduler/Jobs: update `scripts/run_scheduler.py` and `scripts/run_nightly.py` for source priority and optional auto‑import. Add a new background job `web_discovery_bg_enrich` that accepts a tenant and the next‑40 list; ensure observability writes (run_event_logs, run_vendor_usage) and a small concurrency limit.
6) LangGraph: add `import_docs` tool node and edges in `app/pre_sdr_graph.py`; ensure the graph reads the latest tenant ICP from `icp_rules` (e.g., most recent row for tenant) to plan discovery; smoke test graph.
7) UI: add ResearchOps import action and confirm Top‑10/ACRA views; allow multipart upload; validate proxy behavior.
8) Telemetry: log `files_scanned`, `leads_upserted`, `citation_density`; expose basic metrics.
9) Docs: link featurePRD19.md from project docs; add short “What changed since PRD19” summary.
10) Country routing: implement SG vs non‑SG persistence path in `src/icp_pipeline.py` and `src/icp_intake.py` (reuse existing ACRA helpers; add non‑SG default path).

## Configuration

- `SCHED_PRIORITY_SOURCE_ORDER=web_discovery,manual_research,acra_ssic`
- `SCHED_DAILY_CAP_PER_TENANT=200`
- `ICP_DISCOVERY_TOPK=10`, `ICP_DISCOVERY_CAP=200`, `ICP_DISCOVERY_MAX_PAGES=3`
- `ICP_SCORE_A_MIN=65`, `ICP_MIN_EVIDENCE_TYPES=1`, `MANUAL_RESEARCH_BONUS_MAX=20`
- `DOCS_ROOT=/app/docs` (server path for default import)
- `RUN_NOW_LIMIT=10` (Top‑10 immediate batch), `BG_NEXT_COUNT=40` (Non‑SG background top‑up)
- `ENRICH_RECHECK_DAYS=7`, `ENRICH_SKIP_IF_ANY_HISTORY=false`
 - DDG pagination: `DDG_PAGINATION_MAX_PAGES=8`, `DDG_PAGINATION_SLEEP_MS=250`, `DDG_MAX_CALLS=2`, optional `DDG_KL=sg-en|us-en|...`

## Testing Plan

- Unit: research parser (signals, citations), idempotent upsert (hash), scoring bonus calculation, RLS/tenant guards.
- Integration: `POST /research/import` with root and with file uploads; verify artifacts, evidence, queue entries, and scores updated.
- E2E: chat submits `{ user_website, customer_websites[] }` → Top‑10 within seconds; accepting a suggestion persists `icp_rules`; user clicks “Enrich Top‑10 now” → enriched shortlist within ≤ 5 minutes p95; for Non‑SG, background next‑40 enqueued and completes with a chat notification (agent polls `/jobs/{id}`); ingest `docs/` → manual bonus reflected; nightly run → A‑bucket displayed.
- Country E2E: SG candidates resolve to ACRA/UEN and SSIC evidence; non‑SG candidates persist to `companies` with `hq_country`, are enriched, and appear in nightly A‑bucket.
- ResearchOps E2E: run the exact workflow — author `mycompanyprofile.md`, `profiles/yakin_profile.md`, `ideal_customer_profile.md`, `leads_for_nexius.md` with `curl -Ls` snapshots and DuckDuckGo results. Import and confirm: artifacts persisted, evidence rows created, research sources referenced in reason_json, candidate queued with ManualResearch bonus.
- UI: file upload success path, error states, and “why” panel citations.
 - DDG pagination quick test (Postman):
   1) GET `https://duckduckgo.com/html/` with `q=foodservice distributors`.
   2) Inspect response for a `?s=` next-page link; repeat GET on that URL to advance pages.
   3) Continue until unique domains reach 50 or pages exhaust. When running the backend, verify logs `[ddg] parsed domains=… page=N …` and summary `[plan] ddg domains found=… (uniq=…)`.

## Rollout & Ops

- Phase 0: migrate + deploy with feature flags off.
- Phase 1: enable `/research/import` for pilot tenants; monitor metrics.
- Phase 2: enable full priority order; increase nightly caps; enable Non‑SG background next‑40.
- Phase 3: hygiene tasks (MV backfill, prune snapshots, re‑score legacy).

## Code Stubs (for reference)

SQL — `lead_generation-main/app/migrations/016_prd19_research_ops.sql`
```sql
-- PRD19 research ops tables
CREATE TABLE IF NOT EXISTS icp_research_artifacts (
  id BIGSERIAL PRIMARY KEY,
  tenant_id BIGINT NOT NULL,
  company_hint TEXT,
  company_id BIGINT,
  path TEXT NOT NULL,
  source_urls TEXT[] NOT NULL,
  snapshot_md TEXT NOT NULL,
  fit_signals JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  ai_metadata JSONB NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_icp_ra_fit ON icp_research_artifacts USING GIN ((fit_signals));
CREATE INDEX IF NOT EXISTS idx_icp_ra_meta ON icp_research_artifacts USING GIN ((ai_metadata));

CREATE TABLE IF NOT EXISTS research_import_runs (
  id BIGSERIAL PRIMARY KEY,
  tenant_id BIGINT NOT NULL,
  run_started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  files_scanned INT NOT NULL DEFAULT 0,
  leads_upserted INT NOT NULL DEFAULT 0,
  errors JSONB NOT NULL DEFAULT '[]'::jsonb,
  ai_metadata JSONB NOT NULL DEFAULT '{}'
);
```

Python — `lead_generation-main/src/research_import.py`
```python
def import_docs_for_tenant(tenant_id: int, root: str) -> dict:
    # 1) Parse docs → artifacts
    # 2) Resolve/insert companies
    # 3) Upsert artifacts + write icp_evidence(research_*)
    # 4) Enqueue discovery_queue(source='manual_research') with bonus
    # 5) Return {files_scanned, leads_upserted, errors}
    ...
```

API — `lead_generation-main/app/icp_endpoints.py`
```python
@router.post("/research/import", response_model=ResearchImportResult)
async def research_import(req: ResearchImportRequest, user=Depends(auth.current_user)):
    guard_tenant(user, req.tenant_id)
    return await run_in_threadpool(import_docs_for_tenant, req.tenant_id, req.root or default_docs_root())
```

UI — `agent-chat-ui/src/app/icp/page.tsx`
```tsx
// Add a button to trigger /research/import and a dropzone for file uploads
```

---

This plan aligns with featurePRD19Revised.md and maps each requirement to specific code locations, ensuring rapid instant results, auditable research ingestion, safe gating before enrichment, strict Top‑10 immediate enrichment, and Non‑SG background next‑40 with chat notification.

## PRD 0–15 Alignment (Coverage in this Dev Plan)

0) Purpose
- Covered by Scope: pre‑enrichment instant discovery, research import, tenant ACRA nightly, explainable scoring, Odoo visibility.

1) Problem Statement
- Addressed by Process Flow and Detailed Journey: reduce waste by front‑loading discovery, ranking, and gating before enrichment.

2) Objectives
- Implemented via Deliverables and Code Changes + Testing Plan; mirrors PRD objectives (provenance, instant Top‑10, nightly ACRA, ResearchOps ingestion).

3) User Stories
- Captured in Preferred User Story & Process and Detailed User Journey; acceptance encoded in Testing Plan and Acceptance bullets below.

4) End‑to‑End Flow (Before Enrichment)
- Mapped to Detailed User Journey (Chat → Enrichment → Nightly) and Process Flow sections.

5) Lead Research Workflow
- Implemented by Research Import Core Module, new migration, and API. UX notes included; persistence to icp_research_artifacts and icp_evidence.

6) LangChain/LangGraph Agentic Design
- Reflected in Architecture & Components and LangGraph Integration steps (nodes, edges, discovery + scoring hooks, clarifications with confidence gating).

7) Data Model and Persistence
- New migration 016/017 (per repo) for research tables + existing companies/evidence/scores; ai_metadata stored for provenance.

8) APIs and CLI
- Added POST `/icp/research/import`; existing `/icp/accept`, `/icp/suggestions`, `/icp/top10` referenced. CLI optional follow‑up.

9) Scoring and Prioritization
- Scoring Hook step adds ManualResearch bonus and reason_json sources; deterministic EvidenceScore maintained.

10) UX Deliverables
- UI tasks under Frontend section: Top‑10 badges/why, ACRA list, ResearchOps import action and result view, chat prompts for two‑message intake.

11) Security and Compliance
- Guardrails: robots‑safe crawl, bounded snapshots, GIN indexes, no PII in research tables, tenant guard/X‑Tenant‑ID, RLS references.

12) Telemetry and KPIs
- Testing/Telemetry plan: files_scanned, leads_upserted, citation_density, queue priority processed, A‑bucket coverage; add logs around import and scoring.

13) Acceptance Criteria
- E2E tests in Testing Plan; includes: Top‑10 within seconds, accept persists ICP + head enrichment, research import persists evidence and re‑ranks, nightly order + SG vs non‑SG routing.

14) Rollout Plan
- Rollout & Ops section with phased enablement, feature flags, nightly caps.

15) Implementation Notes (Agents and Tools)
- Code Stubs, LangGraph Integration, and Operational safeguards enumerate tools, prompts, and protective limits.

## Step‑By‑Step Code (Concrete Snippets)

1) Settings: caps and flags (src/settings.py)
```python
# --- Enrichment batch sizes ---
RUN_NOW_LIMIT = int(os.getenv("RUN_NOW_LIMIT", "10") or 10)
BG_NEXT_COUNT = int(os.getenv("BG_NEXT_COUNT", "40") or 40)
ENRICH_RECHECK_DAYS = int(os.getenv("ENRICH_RECHECK_DAYS", "7") or 7)
ENRICH_SKIP_IF_ANY_HISTORY = os.getenv("ENRICH_SKIP_IF_ANY_HISTORY", "false").lower() in ("1","true","yes","on")
# Scheduler priority
SCHED_PRIORITY_SOURCE_ORDER = os.getenv(
    "SCHED_PRIORITY_SOURCE_ORDER", "web_discovery,manual_research,acra_ssic"
)
```

2) API: enrich Top‑10 now (app/icp_endpoints.py)
```python
@router.post("/enrich/top10")
async def enrich_top10(req: Request, user=Depends(_auth_dep), x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID")):
    tid = _resolve_tenant_id(req, x_tenant_id)
    if tid is None:
        raise HTTPException(status_code=400, detail="tenant_id is required")
    # Read persisted Top‑10 preview domains (ai_metadata.preview=true)
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT domain
            FROM staging_global_companies
            WHERE tenant_id=%s AND COALESCE((ai_metadata->>'preview')::boolean,false)=true
            ORDER BY COALESCE((ai_metadata->>'score')::float,0) DESC
            LIMIT %s
            """,
            (tid, RUN_NOW_LIMIT),
        )
        domains = [r[0] for r in (cur.fetchall() or []) if r and r[0]]
    if not domains:
        raise HTTPException(status_code=412, detail="top10 preview not found; please confirm to regenerate")
    # Map domains → company_ids
    company_ids: list[int] = []
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT company_id FROM companies WHERE LOWER(website_domain) = ANY(%s)",
            ([d.lower() for d in domains],),
        )
        company_ids = [int(r[0]) for r in (cur.fetchall() or []) if r and r[0]]
    # Enrich now (same function used by nightly)
    from src.enrichment import enrich_company_with_tavily
    processed = 0
    for cid in company_ids:
        try:
            await enrich_company_with_tavily(cid)
            processed += 1
        except Exception:
            pass
    return {"ok": True, "processed": processed, "requested": len(company_ids)}
```

3) Jobs: enqueue + NOTIFY (src/jobs.py)
```python
def enqueue_web_discovery_bg_enrich(tenant_id: int, company_ids: list[int]) -> dict:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO background_jobs(tenant_id, job_type, status, params) VALUES (%s,'web_discovery_bg_enrich','queued', %s) RETURNING job_id",
            (tenant_id, Json({"company_ids": ids})),
        )
        jid = int(cur.fetchone()[0])
        cur.execute("NOTIFY bg_jobs, %s", (json.dumps({"job_id": jid, "type": "web_discovery_bg_enrich"}),))
        return {"job_id": jid}
```

4) Worker: claim + run (scripts/run_bg_worker.py)
```python
async def _claim_one(conn, job_type: str) -> Optional[int]:
    sql = """
    WITH cte AS (
      SELECT job_id FROM background_jobs
       WHERE status='queued' AND job_type=$1
       ORDER BY job_id ASC
       FOR UPDATE SKIP LOCKED
       LIMIT 1)
    UPDATE background_jobs b
       SET status='running', started_at=now()
      FROM cte
     WHERE b.job_id = cte.job_id
 RETURNING b.job_id
    """
    row = await conn.fetchrow(sql, job_type)
    return int(row[0]) if row else None

async def _run_job(job_id: int) -> None:
    await run_web_discovery_bg_enrich(job_id)
```

5) Deployment
- Systemd unit: ExecStart=/usr/bin/env python -m scripts.run_bg_worker; Restart=always; env: POSTGRES_DSN, BG_WORKER_MAX_CONCURRENCY, BG_WORKER_SWEEP_INTERVAL.
- Docker/K8s: separate service/deployment running `python -m scripts.run_bg_worker`.

3) API: enqueue Non‑SG next‑40 background (app/icp_endpoints.py)
```python
@router.post("/enrich/next40")
async def enrich_next40(req: Request, user=Depends(_auth_dep), x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID")):
    tid = _resolve_tenant_id(req, x_tenant_id)
    if tid is None:
        raise HTTPException(status_code=400, detail="tenant_id is required")
    # Pick next 40 from the same preview, skipping Top‑10
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT domain
            FROM staging_global_companies
            WHERE tenant_id=%s AND COALESCE((ai_metadata->>'preview')::boolean,false)=true
            ORDER BY COALESCE((ai_metadata->>'score')::float,0) DESC
            OFFSET %s LIMIT %s
            """,
            (tid, RUN_NOW_LIMIT, BG_NEXT_COUNT),
        )
        domains = [r[0] for r in (cur.fetchall() or []) if r and r[0]]
    if not domains:
        raise HTTPException(status_code=404, detail="no next40 domains found")
    # Resolve to company_ids best‑effort
    company_ids: list[int] = []
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT company_id, website_domain FROM companies WHERE LOWER(website_domain) = ANY(%s)", ([d.lower() for d in domains],))
        rows = cur.fetchall() or []
        found = {str((r[1] or "").lower()): int(r[0]) for r in rows if r and r[0]}
        for d in domains:
            cid = found.get(str(d.lower()))
            if cid:
                company_ids.append(cid)
    # Enqueue background job
    from src.jobs import enqueue_web_discovery_bg_enrich
    job = enqueue_web_discovery_bg_enrich(tid, company_ids)
    return {"ok": True, "job_id": job.get("job_id")}
```

4) Jobs: background next‑40 (src/jobs.py)
```python
from typing import Iterable

try:
    from src.enrichment import enrich_company_with_tavily
except Exception:
    enrich_company_with_tavily = None  # pragma: no cover


def enqueue_web_discovery_bg_enrich(tenant_id: int, company_ids: list[int]) -> dict:
    if not company_ids:
        return {"job_id": 0}
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO background_jobs(tenant_id, job_type, status, params) VALUES (%s,'web_discovery_bg_enrich','queued', %s) RETURNING job_id",
            (tenant_id, Json({"company_ids": company_ids})),
        )
        row = cur.fetchone()
        return {"job_id": int(row[0]) if row and row[0] else 0}


async def run_web_discovery_bg_enrich(job_id: int) -> None:
    import time
    t0 = time.perf_counter()
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("UPDATE background_jobs SET status='running', started_at=now() WHERE job_id=%s", (job_id,))
        cur.execute("SELECT tenant_id, params FROM background_jobs WHERE job_id=%s", (job_id,))
        row = cur.fetchone()
        tenant_id = int(row[0]) if row and row[0] is not None else None
        params = (row and row[1]) or {}
        ids: Iterable[int] = [int(i) for i in (params.get('company_ids') or []) if str(i).strip()]
    processed = 0
    try:
        if enrich_company_with_tavily is None:
            raise RuntimeError("enrich unavailable")
        for cid in ids:
            try:
                await enrich_company_with_tavily(int(cid))
                processed += 1
            except Exception:
                pass
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("UPDATE background_jobs SET status='done', processed=%s, total=%s, ended_at=now() WHERE job_id=%s", (processed, processed, job_id))
    except Exception as e:  # pragma: no cover
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("UPDATE background_jobs SET status='error', error=%s, processed=%s, ended_at=now() WHERE job_id=%s", (str(e), processed, job_id))
    finally:
        dur_ms = int((time.perf_counter() - t0) * 1000)
        log.info('{"job":"web_discovery_bg_enrich","job_id":%s,"processed":%s,"duration_ms":%s}', job_id, processed, dur_ms)
```

5) Nightly dispatcher wiring (scripts/run_nightly.py)
```python
# Include web_discovery_bg_enrich in the dispatcher with high priority
cur.execute(
    """
    SELECT job_id, job_type
    FROM background_jobs
    WHERE status='queued' AND job_type IN ('web_discovery_bg_enrich','manual_research_enrich','staging_upsert','enrich_candidates','icp_intake_process')
    ORDER BY CASE job_type
      WHEN 'web_discovery_bg_enrich' THEN 0
      WHEN 'manual_research_enrich' THEN 1
      WHEN 'staging_upsert' THEN 2
      WHEN 'enrich_candidates' THEN 3
      WHEN 'icp_intake_process' THEN 4
    END, job_id
    """
)
...
elif jtype == 'web_discovery_bg_enrich':
    from src.jobs import run_web_discovery_bg_enrich
    await run_web_discovery_bg_enrich(int(jid))
```

6) UI/Agent integration (app/pre_sdr_graph.py) — on “Enrich Top‑10 now”
```python
# Pseudocode inside the node handling the CTA / command
resp = await http.post(f"{api_base}/icp/enrich/top10", headers=auth_headers)
if is_non_sg_active_profile():
    # Kick off background next‑40 and inform the user
    job = await http.post(f"{api_base}/icp/enrich/next40", headers=auth_headers)
    say(f"Enriching the next 40 in the background (job {job['job_id']}). I will reply here when it finishes. You can also check status later.")
```

7) Acceptance check (scripts/acceptance_check.py) additions
```python
# Add these helpers near compute_metrics()
async def compute_top10_enrich_metrics(conn: asyncpg.Connection) -> Dict[str, Any]:
    """Return metrics for the latest strict Top‑10 enrichment run.

    Uses enrichment_runs + run_summaries + run_manifests + run_event_logs.
    """
    out: Dict[str, Any] = {"run_id": None, "duration_ms": None, "stage_p95_ms": {}, "bucket_counts_top10": {}}
    row = await conn.fetchrow(
        """
        SELECT er.run_id, er.started_at, er.ended_at, rs.candidates, rs.processed
        FROM enrichment_runs er
        JOIN run_summaries rs USING(run_id)
        WHERE er.ended_at IS NOT NULL
          AND rs.candidates = 10
        ORDER BY er.ended_at DESC
        LIMIT 1
        """
    )
    if not row:
        return out
    out["run_id"] = int(row["run_id"]) if row["run_id"] else None
    try:
        dur_ms = int((row["ended_at"] - row["started_at"]).total_seconds() * 1000)
    except Exception:
        dur_ms = None
    out["duration_ms"] = dur_ms
    # p95 per stage for this run
    try:
        rows = await conn.fetch(
            """
            SELECT stage, percentile_cont(0.95) WITHIN GROUP (ORDER BY duration_ms) AS p95
            FROM run_event_logs
            WHERE run_id = $1 AND duration_ms IS NOT NULL
            GROUP BY stage
            """,
            out["run_id"],
        )
        out["stage_p95_ms"] = {str(r["stage"]): int(r["p95"] or 0) for r in rows}
    except Exception:
        out["stage_p95_ms"] = {}
    # A/B/C counts among the Top‑10 manifest
    try:
        ids = await conn.fetchval("SELECT selected_ids FROM run_manifests WHERE run_id=$1", out["run_id"])
        if ids:
            rows = await conn.fetch(
                """
                SELECT s.bucket, COUNT(*) AS c
                FROM lead_scores s
                WHERE s.company_id = ANY($1::bigint[])
                GROUP BY s.bucket
                """,
                ids,
            )
            out["bucket_counts_top10"] = {str(r["bucket"]): int(r["c"]) for r in rows}
    except Exception:
        out["bucket_counts_top10"] = {}
    return out

def evaluate_top10_sla(m: Dict[str, Any], max_ms: int = 300_000) -> Tuple[bool, int]:
    dur = int(m.get("duration_ms") or 0)
    return (dur <= max_ms and dur > 0), dur

# In main(), add argument and wire it into output
parser.add_argument("--max-top10-ms", type=int, default=int(os.getenv("MAX_TOP10_MS", "300000") or 300000))
...
metrics = await compute_metrics(conn)
top10 = await compute_top10_enrich_metrics(conn)
passed, results = evaluate(metrics, thresholds)
sla_ok, top10_ms = evaluate_top10_sla(top10, args.max_top10_ms)
results["top10_sla_ok"] = sla_ok
out = {
    "tenant_id": args.tenant,
    "metrics": metrics,
    "top10": top10,
    "thresholds": thresholds,
    "max_top10_ms": args.max_top10_ms,
    "results": results,
    "passed": passed and sla_ok,
}
...
print(f"  top10_sla: {top10_ms} ms (<= {args.max_top10_ms} ms) -> { 'OK' if sla_ok else 'FAIL' }")
print(f"  top10_stage_p95_ms: {top10.get('stage_p95_ms')}")
print(f"  top10_bucket_counts: {top10.get('bucket_counts_top10')}")
```

8) SQL: background_jobs table (if missing columns used)
```sql
-- Ensure background_jobs has fields we rely on
ALTER TABLE background_jobs
  ADD COLUMN IF NOT EXISTS processed INT DEFAULT 0,
  ADD COLUMN IF NOT EXISTS total INT DEFAULT 0,
  ADD COLUMN IF NOT EXISTS error TEXT;
CREATE INDEX IF NOT EXISTS idx_bg_jobs_status ON background_jobs(status);
CREATE INDEX IF NOT EXISTS idx_bg_jobs_type ON background_jobs(job_type);
```
## Configuration & Ops (added)
- `ACRA_DAILY_ENRICH_LIMIT`: Per‑tenant nightly ACRA cap (default 20). Effective nightly SELECT limit is `min(ACRA_DAILY_ENRICH_LIMIT - already_today, ENRICH_BATCH_SIZE)`.
- `SCHED_START_CRON`: Cron for nightly (default `0 1 * * *` Asia/Singapore).
- `BG_WORKER_MAX_CONCURRENCY`, `BG_WORKER_SWEEP_INTERVAL`: Next‑40 worker concurrency and sweep interval (supports ms/s/m/h suffixes).
