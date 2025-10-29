# Feature PRD 19 — Lead Generation Enhancements Before Enrichment (with LangChain/LangGraph Agents)

## 0) Purpose

This document enhances and modifies the existing lead generation project ICP flow and the enrichment pipeline. It incorporates the complete PRD19 scope and adds detailed user stories, a lead research workflow, an end-to-end user flow, and how LLM-powered LangChain and LangGraph intelligent agents automate the process using the provided tools (web crawl via r.jina.ai, DuckDuckGo search, ACRA nightly, Odoo sync, internal APIs, and Markdown ingestion).

What’s new vs. PRD19:
- Pre-enrichment planning and automation with LLM agents, ensuring quality candidates are generated, prioritized, and deduplicated before heavy enrichment runs.
- A clear human-in-the-loop ResearchOps workflow and how it integrates into automated discovery.
- A LangGraph-based orchestration blueprint with concrete nodes, tools, and control flow.

---

## 1) Problem Statement

The enrichment pipeline is compute- and IO-heavy. Running it on weak or low-signal candidates wastes cycles and delays value. We need a robust pre-enrichment lead generation layer that: (a) rapidly discovers high-likelihood candidates, (b) explains “why” with evidence, (c) allows analysts to inject verifiable research, and (d) prioritizes the best subset for enrichment and sales workflows.

---

## 2) Objectives

- Generate and prioritize high-quality leads before enrichment using automated signals and research artifacts.
- Blend automated Web Discovery (instant Top-10) with tenant-specific nightly ACRA candidates.
- Enrich strictly the Top‑10 web‑discovery domains immediately; do not mix with ACRA/top‑ups in the run‑now batch.
- Ingest human-curated Markdown research under `docs/` and convert it to structured evidence and queue entries.
- Maintain explainability and provenance via `ai_metadata` and source citations.
- Mirror core entities to Odoo for operational visibility.
- Orchestrate the process with LangGraph, leveraging LangChain tools for search, crawl, parsing, and persistence.

---

## 2.1) User Story & Process (Chat → Enrichment → Nightly)

1) Start chat (two messages): The user first pastes their business website. In the next message, they paste multiple customer websites. The agent validates URLs and starts lightweight crawls (robots-safe, small budgets) after each message.
2) Understand + pattern match: From the user site, the agent infers the business model; from customer sites, it extracts common signals (SSIC, integrations, buyer titles, ops signals like cold-chain, chat/WhatsApp, pricing/case studies, hiring roles). The agent also infers industries, size band, revenue bucket, incorporation years, and geos directly from content, metadata, and third‑party signals.
3) Minimal clarifications only, propose ICP, and persist: By default, the agent does not ask industries, employee range, revenue bucket, incorporation years, or geos — it infers these intelligently from the pasted websites. If any inferred field is low‑confidence or unreasonable, the agent asks a short, focused follow‑up for that specific gap (and retries if still unreasonable). It then proposes a micro‑ICP (cards + payload). On user confirmation, it persists the ICP to `icp_rules` via `/icp/accept` and saves any clarifications to `icp_intake_responses`. This becomes the active ICP profile.
4) Instant discovery + Top-10: Using the persisted ICP, the agent plans queries, discovers candidate domains, mini-crawls, extracts evidence, and surfaces Top‑10 immediately (persisted to staging with `ai_metadata` preview fields); queues the remainder.
5) Top‑10 immediate enrichment (run‑now batch): On user confirm/CTA, enrich exactly these 10 (no mixing with ACRA/queues): mini‑crawl 4–6 pages → LLM extraction merge (bounded corpus/timeouts) → Apify LinkedIn contacts (respect caps) → ZeroBounce verify → score/bucket → persist to canonical/enrichment tables. Optionally export to Odoo when ready. All writes include `ai_metadata`.
6) Optional ResearchOps: Analysts contribute Markdown artifacts (snapshots, leads); the importer persists them to `icp_research_artifacts` and `icp_evidence`, re-scoring candidates with a capped ManualResearch bonus.
7) Nightly runner: Processes discovery queues in order web_discovery → manual_research → acra_ssic; for SG, anchors on ACRA/UEN and SSIC; for non-SG, persists to companies with country fields. Enriches and syncs to Odoo; shows A-bucket by default.
8) Outcome: The user reviews the A-bucket shortlist with a clear “why,” iterates ICP or research as needed, and proceeds with outreach.

---

## 3) User Stories

- SDR/RevOps: As an SDR, I want instant Top-10 lookalikes from seeds so I can quickly shortlist and take action without waiting for overnight jobs.
- Analyst (ResearchOps): As an analyst, I want to drop reproducible artifacts into `docs/` and have the system ingest and credit my research so that it directly improves discovery and ranking.
- Tenant Admin: As an admin, I want nightly tenant-specific ACRA candidates gated by clear criteria and evidence so I can trust the quality and focus on A-bucket leads.
- RevOps/Manager: As a manager, I want all leads and ICPs visible in Odoo and traced to sources so teams can align on prioritization and handoff.
- Platform Engineer: As a platform engineer, I want the pre-enrichment orchestration to be auditable, deterministic, and scalable so operations remain stable and explainable.

Acceptance per story is captured in the combined Acceptance Criteria section below.

---

## 4) End-to-End Flow (Before Enrichment)

High-level stages and outcomes:
1) Seed Intake → Micro-ICP Synthesis → Web Discovery Top-10 (instant)
2) Queue Remaining Candidates → Human ResearchOps Import (DB-first) → Merge Evidence
3) Nightly ACRA Candidate Harvest (tenant-specific) → Queue
4) Scoring and Prioritization → A/B/C Buckets → Only A-bucket proceeds automatically to enrichment by default

Detailed flow:
- Seed Intake: User provides 3–10 seed customers (domains or UENs). A mini-crawl (robots-safe, small page budget) collects initial signals.
- Micro-ICP Synthesis (LLM): From seeds, the ICP Synthesizer Agent extracts SSICs, integrations, typical buyer titles, size bands, and trigger phrases; stores/updates `icp_profiles`.
- Instant Web Discovery: Using ICP features, the Discovery Planner Agent generates search queries, discovers candidate domains, and schedules mini-crawls. Evidence is extracted and scored.
- Top‑10 Presentation + Staging: Highest EvidenceScore candidates appear immediately with badges and source tooltips. Persist all discovered candidate domains to `staging_global_companies` (source = `web_discovery`) and surface the total count to the user. Persist per‑domain `ai_metadata` including preview fields (score, bucket, why, snippet) and provenance. The Top‑10 can be enriched immediately via a CTA; the remainder are queued for later processing. Enrichment strictly reuses this Top‑10; if missing on a later run, the user is prompted to regenerate instead of falling back to other sources.
- ResearchOps Import: Analysts submit research via `/icp/research/import`; Markdown under `docs/` is optional input. The importer adds `icp_research_artifacts`, creates `icp_evidence` entries, and applies a bounded ManualResearch bonus. Canonical data lives in DB, not under `docs/`.
- Nightly ACRA Harvest: Select ACRA candidates by SSIC and filters; resolve, crawl, and score for the tenant.
- Prioritization & Gating: Compute final tenant scores with both automated and manual-research signals; by default, show/gate A-bucket only for enrichment.
- Odoo Mirror: Seeds and active ICP mirror to Odoo; company mirror/updates as needed.

---

## 5) Lead Research Workflow (Human-in-the-Loop)

The following codifies the ResearchOps practice and how it integrates with automation. This section reflects the preferred workflow and methodology and is used by the importer to persist data and influence discovery and scoring.

Artifacts (submitted via `/icp/research/import`; optional local drafts allowed):
- Company positioning context — services, differentiators that anchor subsequent research.
- Seed customer write‑up — real‑world pains, workflows, automation wins.
- Company profile snapshots — snapshot text plus traceable source links per profiled company.
- Curated leads list — name, website, snapshot, fit signals, wedge, contacts, sources.
- Lead‑search keywords — baseline term stack used to craft DuckDuckGo queries.

CLI-friendly collection (collect locally, then submit via import):
- Primary content snapshot: `curl -Ls https://r.jina.ai/<prospect-url>` → capture rendered content snapshot with URL citations (e.g., product, about, distributorship pages).
- Universe mapping: `curl -Ls "https://r.jina.ai/https://duckduckgo.com/html/?q=<query>"` → capture candidate links and supporting context.

Lead Search Methodology (submit results to DB via `/icp/research/import`):
- Anchor DuckDuckGo queries to ICP signals such as “food distributor”, “Peppol”, “WhatsApp ordering”, “cold‑chain”, and “OEM” to surface qualified operators.
- Run web lookups via `curl -Ls "https://r.jina.ai/https://duckduckgo.com/html/?q=<query>"` while referencing the baseline stack (optional local doc).
- Iterate by swapping adjacent terms (e.g., “seafood processor”, “digital portal”, “B2B distributor”) and combining with `site:sgpbusiness.com` filters until high‑fit distributors cluster.
- Qualify each lead by opening the official site and SGPBusiness profile, then include snapshot, fit signals, wedge, and contacts in the import payload so it persists to the database.

Profile Generation Sequence (DB-first):
1. Draft Nexius positioning (services, differentiators) for analyst context (optional local doc).
2. Translate that context into a seed customer write‑up (optional local doc) outlining pains, workflows, and automation wins.
3. Persist the Ideal Customer Profile to the database (icp_rules per tenant) via `/icp/accept` (firmographics, operational signals, triggers). The active ICP is updated over time: when confirm/enrichment runs, newly inferred `icp_profile` fields (industries, integrations, buyer_titles, size_bands, triggers) are merged into `icp_rules.payload` (upsert).
4. Use the system ICP as the gating rubric while researching and documenting prospects (optional local doc for notes); submit research via ingest so it persists tenant‑scoped in DB.

Import & scoring (persistence and usage):
- Importer parses Markdown, resolves companies, writes research artifacts to `icp_research_artifacts` and evidence to `icp_evidence(research_fit|research_note)`, and enqueues candidates with a ManualResearch bonus (capped). All records include `ai_metadata` with provenance and hashes.
- The scoring pipeline reads these persisted artifacts to compute ManualResearch bonus and merges into final ranking and gating.
- The active ICP in `icp_rules` is the source of truth for discovery (Top‑10) and nightly scheduling; `/icp/top10` reconstructs `icp_profile` from `icp_rules.payload` including `size_bands`.

---

## 6) LangChain/LangGraph Agentic Design

We orchestrate pre-enrichment lead generation with a LangGraph state machine. Each node uses LangChain tools for deterministic, auditable steps.

Core state keys:
- `tenant_id`, `seeds[]`, `icp_profile`, `discovery_candidates[]`, `research_artifacts[]`, `evidence[]`, `scores[]`, `queue[]`, `errors[]`.

Agents and responsibilities:
- ICP Synthesizer Agent (LLM):
  - Inputs: seed crawl snippets, existing ICP.
  - Tools: Text splitter/loader, classification chain; writes `icp_profiles` with SSIC, integrations, titles, size bands, triggers.
  - Output: normalized micro-ICP used for search planning and scoring.
- Discovery Planner Agent (LLM):
  - Inputs: micro-ICP.
  - Tools: DuckDuckGo search tool, query generator, URL deduper.
  - Output: prioritized candidate domains, with query justification.
- Mini-Crawl Worker (Tool-based):
  - Inputs: candidate domains.
  - Tools: HTTP fetch via `r.jina.ai` snapshot, robots-aware throttled fetch, HTML-to-Markdown conversion.
  - Output: small content bundle (4–6 pages) per domain.
- Evidence Extractor Agent (LLM):
  - Inputs: content bundle.
  - Tools: Extraction chain prompting (regex-guarded), schema validation.
  - Output: `icp_evidence` signals (evidence_types, integrations, buyer_titles, hiring_open_roles, pricing/case studies, freshness).
- Research Import Processor (Tool + LLM assist):
  - Inputs: Markdown files from `docs/`.
  - Tools: Markdown loader, domain resolver, Postgres upserter.
  - Output: `icp_research_artifacts`, `icp_evidence(research_*)`, queue entries; adds provenance to `ai_metadata`.
- Scoring & Gating Node (Tool):
  - Inputs: evidence set, research bonus, tenant filters.
  - Tools: SQL queries or db tool, deterministic scoring functions.
  - Output: Final tenant score, A/B/C buckets, `reason_json`.
- ACRA Nightly Orchestrator (Tool + LLM assist optional):
  - Inputs: tenant SSIC filters, size/entity constraints.
  - Tools: ACRA dataset query, resolver, mini-crawl, evidence extractor, scoring.
  - Output: queued ACRA candidates, updated scores.
- Odoo Sync Agent (Tool):
  - Inputs: seed, company, ICP updates.
  - Tools: Odoo RPC/HTTP client wrapper.
  - Output: mirrored partners and ICP profile records.

Control flow (LangGraph):
1) Start → Seeds Loaded → ICP Synthesizer
2) ICP → Discovery Planner → Mini-Crawl → Evidence Extractor → Top-10 now, rest queued
3) Parallel branch: Research Import Processor monitors `docs/` → updates evidence and queue (idempotent)
4) Nightly branch: ACRA Orchestrator → Resolver → Mini-Crawl → Evidence Extractor → Scoring
5) Scoring & Gating Node computes `final_score` and buckets; A-bucket autosent to enrichment pipeline (toggleable)
6) Odoo Sync runs at appropriate persistence points
7) Errors routed to Retry/Dead-letter with ai_metadata diagnostics

Example pseudo: registering tools and nodes
```python
from langgraph.graph import StateGraph
from langchain_community.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()

graph = StateGraph()
graph.add_node("synthesize_icp", icp_synthesizer)
graph.add_node("plan_discovery", lambda s: plan_queries(s, search))
graph.add_node("mini_crawl", mini_crawl_worker)
graph.add_node("extract_evidence", evidence_extractor)
graph.add_node("import_docs", research_import_processor)
graph.add_node("score_gate", scoring_and_gating)
graph.add_node("acra_nightly", acra_orchestrator)
graph.add_node("odoo_sync", odoo_sync)

graph.set_entry_point("synthesize_icp")
graph.add_edge("synthesize_icp", "plan_discovery")
graph.add_edge("plan_discovery", "mini_crawl")
graph.add_edge("mini_crawl", "extract_evidence")
graph.add_edge("extract_evidence", "score_gate")
graph.add_edge("import_docs", "score_gate")
graph.add_edge("acra_nightly", "mini_crawl")
graph.add_edge("score_gate", "odoo_sync")
app = graph.compile()
```

Guardrails:
- Constrain page budgets and source domains; respect robots.txt via fetch tool.
- Use JSON schema validation for evidence extraction; discard hallucinated fields.
- Apply deterministic scoring outside the LLM; LLMs propose, tools verify.

---

## 7) Data Model (additions related to web discovery staging)

- `staging_global_companies` (NEW): staging table for non‑SG web discovery candidates with columns `(id, tenant_id, domain, source, created_at, ai_metadata)`, unique on `(tenant_id, domain, source)`.
- `global_company_registry`: canonical enrichment store for non‑SG (unchanged); SG continues to use `acra_enrichment`.

Environment/config:
- `STAGING_GLOBAL_TABLE` env var defaults to `staging_global_companies` and is used by the chat runtime to persist web discovery candidates.


## 7) Data Model and Persistence

We retain PRD19 structures, add `ai_metadata` for provenance, and incorporate research artifacts. See full PRD19 content below for exact DDL and MV concepts.

- Canonical: `companies`, `tenant_companies`, `icp_profiles`, `lead_scores`.
- Enrichment stores: `acra_enrichment` (SG), `global_company_registry` (non-SG).
- Discovery queue: `discovery_queue` with source, priority, status.
- Evidence roll-up MV: `icp_features_web(...)`.
- Research tables: `icp_research_artifacts`, `research_import_runs` with GIN indexes.

---

## 8) APIs and CLI (Pre-Enrichment Focus)

- `POST /icp/run` → Seeds in, returns `{top10, queued, micro_icp}` quickly.
- `GET /icp/top10?tenant_id=` → badges and sources for immediate action.
- `POST /icp/profile` → Upsert active micro-ICP for tenant.
- `POST /icp/enrich/top10` or chat `run enrichment` → Enrich the persisted Top‑10 preview (fail fast if missing) and return enriched shortlist with A/B/C buckets and why.
- `GET /leads?tenant_id=&bucket=A&limit=50` → Gated list with `reason_json`.
- `POST /research/import` → Scan `docs/` or accept files; returns `{files_scanned, leads_upserted, errors[]}`.
- CLI helper: `nexius icp research-import --tenant <id> --root ./docs`.

---

## 9) Scoring and Prioritization

We keep deterministic EvidenceScore and add a capped ManualResearch bonus. Tenant Final Score gates display and enrichment eligibility. Buckets: A ≥ 65, B 50–64, C < 50 (configurable).

Ranking order: EvidenceScore → LeadScore → Employees → Year → stable hash for tiebreaks.

Top‑10 run‑now batch specifics:
- Skip enrichment for companies with recent history when `ENRICH_RECHECK_DAYS > 0` (or skip any history if `ENRICH_SKIP_IF_ANY_HISTORY=true`).
- Respect `CRAWL_MAX_PAGES`, `EXTRACT_CORPUS_CHAR_LIMIT`, `LLM_MAX_CHUNKS`, and timeouts (`LLM_CHUNK_TIMEOUT_S`, `MERGE_DETERMINISTIC_TIMEOUT_S`).
- Respect vendor caps (`APIFY_DAILY_CAP`) and verify emails only when `ZEROBOUNCE_API_KEY` is set; degrade gracefully.
- Persist to `summaries`, `company_enrichment_runs`, `companies`, and lead/score tables with `ai_metadata` provenance.

---

## 10) UX Deliverables (Pre-Enrichment)

- Top-10 Lookalikes with badges and source tooltips; “why” panel shows signals.
- CTA: “Enrich Top‑10 now” → streams progress chips (discover → crawl → extract → verify → score) and ends with enriched shortlist and export buttons.
- ACRA nightly list (A-bucket default) with the same badges and explanations.
- Micro-ICP cards (SSIC + integrations + titles) with “Use as filter”.
- ResearchOps Import UI/CLI feedback: files scanned, leads upserted, conflicts, errors.

---

## 11) Security and Compliance

- Respect robots.txt; small page budgets; per-domain throttles.
- No PII in evidence or `ai_metadata`.
- Multi-tenant isolation on all reads/writes.
- Hash and bound Markdown snapshots; store provenance in `ai_metadata`.

---

## 12) Telemetry and KPIs

- Time-to-value: % sessions with action from Top-10.
- Precision proxy: Top-10 → shortlist rate.
- Coverage: % leads with `evidence_types ≥ 1`.
- Queue health: avg priority processed, age, daily failure rate.
- Research ingestion: files scanned, leads upserted, citation density.
- Odoo sync: seed mirror rate; duplicate rate.
 - Top‑10 SLA: p95 time-to-enriched‑shortlist ≤ 5 minutes.

---

## 13) Acceptance Criteria (Pre-Enrichment)

- After ≥5 seeds, micro-ICP saved; Top‑10 appears with ≥2 badges and source tooltips within seconds; Top‑10 enrichment completes ≤ 5 minutes p95 from CTA.
- Research ingestion creates `icp_research_artifacts`, writes `icp_evidence(research_*)`, enqueues `manual_research`, and links sources. Markdown in `docs/` is optional input only; the system does not create `.md` outputs.
- Nightly scheduler processes queues in order: web_discovery → manual_research → acra_ssic (configurable via `SCHED_PRIORITY_SOURCE_ORDER`).
- Default surfaced leads are A-bucket, have a website, and ≥1 evidence signal; “why” panel cites SSIC/Integrations/Titles/Freshness and research sources if present.
- Seeds mirrored to Odoo with `x_is_seed_customer=True`; ICP mirrored to `nexius.icp.profile`.
- All enriched tables contain `ai_metadata` with crawl/ingest agent + timestamps.

Top‑10 immediate enrichment (implementation notes):
- Input: reuse persisted Top‑10 from `staging_global_companies` where `ai_metadata.preview=true` (fail if not present; prompt user to `confirm`).
- Batching: strictly first 10; do not include ACRA or other queued sources in this run.
- Observability: write per‑stage `run_event_logs`, vendor usage in `run_vendor_usage`, and QA samples for the run.

---

## 14) Rollout Plan

Phased enablement, aligned with PRD19 migrations and feature flags:
1) Migrations → add research tables, MVs, indexes; Odoo module deploy.
2) Ingestion → ship `/research/import` + CLI; pilot `docs/` repo; verify mapping & scoring.
3) Default On → enable queue order including `manual_research`; raise nightly caps gradually.
4) Hygiene → backfill MV; prune large snapshots; re-score legacy rows; document analyst workflow.

Feature flags: `FF_ICP_WEB_DISCOVERY`, `FF_TOP10_SECTION`, `FF_TOP10_ENRICH_NOW`, `FF_ODOO_ICP_PROFILE`, `FF_RESEARCH_IMPORT`.

---

## 15) Implementation Notes (Agents and Tools)

LangChain tools registry (suggested):
- Search: `DuckDuckGoSearchRun` for query-time domain discovery.
- HTTP Snapshot: wrapper calling `r.jina.ai` for HTML→readable content.
- Markdown Loader: parse `docs/` files to structured objects.
- Resolver: domain→company and fuzzy name resolver.
- Database: Postgres read/write tool for upserts and scoring queries.
- Odoo: RPC/HTTP client wrapper tool with minimal scopes.
- Queue: enqueue/dequeue utility for `discovery_queue` with source and priority.

Prompting patterns:
- ICP Synthesizer: few-shot summaries of seed signals; output constrained JSON schema.
- Evidence Extractor: tool-verified JSON schema with strict enums; reject low-confidence extractions.
- Discovery Planner: generate queries grounded in micro-ICP fields; cite which fields drove each query.

Operational safeguards:
- Per-tenant and per-domain rate limits; retry with exponential backoff.
- Idempotent imports (hash snapshots); dedupe candidates by domain and company_id.
- Strict JSON schema validation; log violations into `errors[]` and `ai_metadata`.

---

## 16) Full Source Content — PRD19.md (Verbatim)

> The following section reproduces PRD19.md in full to ensure complete coverage and traceability for this enhanced plan.

# Feature PRD — ICP Finder vNext + ResearchOps (Evidence-Driven Discovery, Tenant-Specific ACRA Nightly, Odoo Sync, AI Metadata, and Human-in-the-Loop Lead Research)

## 0) Executive Summary

We extend the ICP Finder to combine:

1. **Automated signals** (seed-customer web crawl → micro-ICP → instant Top-10 lookalikes → tenant-specific ACRA nightly scoring), and
2. **ResearchOps workflow** (lightweight, CLI-friendly lead research using `r.jina.ai` and DuckDuckGo, with Markdown artifacts under `docs/`).

All enrichment persists to **SG → ACRA enrichment** or **non-SG → Global Registry**, seeds mirror to **Odoo Contacts**, tenant ICPs are stored in Postgres (and visible in Odoo), and every entity supports **`ai_metadata`** for agent provenance. Research artifacts are importable, traced to sources, and influence scoring and prioritization.

---

## 1) Goals / Non-Goals

### Goals

* Use crawled signals + human research to **rank, discover, and filter** leads.
* **Top-10** lookalikes shown immediately after seeds; nightly **ACRA** runs are **tenant-specific** and show **A-bucket** only by default.
* Persist to the right stores (ACRA/Global/Companies/Odoo) with **explainability** (“why this lead”).
* Add **ResearchOps**: reproducible CLI recipes + Markdown artifacts (`docs/`) that the system can **ingest** and **score**.
* Unify provenance via **`ai_metadata`**.

### Non-Goals

* Heavy ML stack (start with explainable rules).
* Full UI redesign (add badges/cards/why-panels only).
* PII capture/storage (explicitly excluded).

---

## 2) Modes & Users

* **Automated mode (default)**: Seeds → crawl → micro-ICP → instant Top-10 → discovery queue → nightly ACRA scoring. (SDR/RevOps)
* **ResearchOps mode (opt-in)**: Analysts use curl/Markdown to gather intel; system ingests `docs/` and turns it into evidence & candidates. (Analyst/SDR)

---

## 3) End-to-End Flow

### 3.1 Automated (recap)

1. **Seeds in** → resolver + mini-crawl (robots-safe, 4–6 pages).
2. Persist seed enrichment (SG→ACRA / non-SG→Global) + **companies**, link to tenant, mirror to **Odoo**.
3. **Micro-ICP** mined (top SSICs, integrations, buyer titles, size bands) and stored.
4. **Instant Web Discovery** from Micro-ICP → domains → mini-crawl → **EvidenceScore** → **Top-10 now**, enqueue rest.
5. **Nightly ACRA** selection (SSIC match, age/entity filters) → resolve/crawl → **TenantFinalScore** → show **A-bucket** (website + ≥1 evidence).

### 3.2 ResearchOps (new)

Analysts can run a documented, deterministic workflow to contribute research that the system ingests and uses. All tenant‑specific research is persisted in the database (multi‑tenant tables), not as long‑lived files. Storing drafts under `docs/` is optional for local iteration; production ingestion writes to Postgres.

**Lead Research Workflow (DB‑first)**

1. **Clarify ICP**: Refresh from the system ICP or author locally, then submit via the ingestion endpoint so it persists per tenant.
2. **Collect primary intel** (snapshots):

   ```bash
   curl -Ls https://r.jina.ai/<prospect-url>   # rendered content snapshot
   ```

   Submit snapshots to the ingest API/CLI (e.g., `POST /research/import`) with source URLs; the worker stores the artifact body and citations in DB.
3. **Map target universe** (candidate discovery):

   ```bash
   curl -Ls "https://r.jina.ai/https://duckduckgo.com/html/?q=<query>"
   ```

   Anchor queries to ICP signals (e.g., `food distributor peppol "whatsapp ordering" site:sg`); submit extracted candidates via the ingest API to persist.
4. **Profile each lead**: include snapshot, fit signals, “Nexius wedge” (why we win), contacts, links; submit via ingest so it’s tenant‑scoped in DB.
5. **Review & triage**: rank by operational complexity, hiring/expansion news; hand off with suggested talking points.
6. **Organize artifacts**: if using `docs/` locally, keep everything under `docs/` and re‑ingest when updated. In production, DB is the source of truth.

**Profile Generation Sequence**

1. Nexius positioning (services, differentiators)
2. Seed customer write‑up (e.g., “yakin” profile)
3. Synthesize the ideal customer profile (firmographics, operational signals, triggers)
4. Research + document prospects with traceable source links; submit to DB via ingest

**System ingestion (tenant‑scoped, DB persistence)**: a worker parses submitted payloads (and optionally Markdown files during imports), extracts entities, sources, and fit signals → writes to **icp_research_artifacts**, **icp_evidence**, and **discovery_queue** with a **manual_research bonus** in scoring. Any file path from `docs/` is captured only as provenance; the canonical data lives in the database per tenant.

---

## 4) UX Deliverables

* **Top-10 Lookalikes (Web Evidence)** list with badges (e.g., `HubSpot • RevOps • Pricing • 6 roles`) and source tooltip.
* **ACRA (Nightly)** list shows **A-bucket** only by default; same badges and “why” panel.
* **Micro-ICP cards** (3–5) like “SSIC 62012 + HubSpot + RevOps (9/12 seeds)” with “Use as filter”.
* **ResearchOps Import**: one-click “Ingest research” (or CLI, optionally pointing to a `docs/` root) with import results (new/updated leads, conflicts, errors). No system output is written to `docs/`.

---

## 5) Data Model Additions / Changes (Postgres)

### 5.1 Canonical & Enrichment (unchanged from prior PRD; now all have `ai_metadata`)

* `companies`, `acra_enrichment`, `global_company_registry`, `tenant_companies`, `icp_profiles`, `lead_scores`, `discovery_queue` — each includes `ai_metadata JSONB DEFAULT '{}'`.
* `staging_global_companies` — includes `ai_metadata JSONB` populated with provenance; for Top‑10 preview rows, `ai_metadata.preview=true` and includes `score`, `bucket`, `why`, `snippet`.

### 5.2 Web Evidence Roll-up (MV)

* `icp_features_web(company_id, evidence_types, hiring_open_roles, has_pricing, has_case_studies, integrations[], buyer_titles[], last_seen)`.

### 5.3 **NEW: Research artifacts & import log**

```sql
CREATE TABLE IF NOT EXISTS icp_research_artifacts (
  id             BIGSERIAL PRIMARY KEY,
  tenant_id      BIGINT NOT NULL,
  company_hint   TEXT,             -- name/domain as written in the doc
  company_id     BIGINT,           -- resolved later
  path           TEXT NOT NULL,    -- docs/... filename
  source_urls    TEXT[] NOT NULL,  -- traceable citations
  snapshot_md    TEXT NOT NULL,    -- raw markdown snapshot (bounded)
  fit_signals    JSONB,            -- {tags:["peppol","whatsapp"], ops_signals:["cold-chain"], ...}
  created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
  ai_metadata    JSONB NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS research_import_runs (
  id             BIGSERIAL PRIMARY KEY,
  tenant_id      BIGINT NOT NULL,
  run_started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  files_scanned  INT NOT NULL DEFAULT 0,
  leads_upserted INT NOT NULL DEFAULT 0,
  errors         JSONB NOT NULL DEFAULT '[]'::jsonb,
  ai_metadata    JSONB NOT NULL DEFAULT '{}'
);
```

**Indexes**: GIN on `icp_research_artifacts.fit_signals`, `ai_metadata`.

---

## 6) Ingestion & Mapping (ResearchOps → System)

### 6.1 Markdown conventions (ingestible)

Note: The Ideal Customer Profile (ICP) is stored in the database (icp_rules per tenant) and managed via API/UI. If an ICP Markdown draft is provided during import, it is parsed and upserted to `icp_rules`. The database remains the source of truth and is incrementally updated as the system learns (`icp_profile` merge on confirm/enrich).
* `docs/profiles/<company>_profile.md` (snapshot + links).
* `docs/leads_for_nexius.md` (list; each lead section includes: **Name**, **Website**, **Snapshot**, **Fit Signals**, **Nexius Wedge**, **Contacts**, **Sources**).

**Example (lead section):**

```md
## ACME Logistics
- Website: https://acme.sg
- Snapshot: mid-market cold-chain distributor with WhatsApp ordering portal
- Fit Signals: ["peppol", "cold-chain", "whatsapp ordering", "b2b portal"]
- Nexius Wedge: automate order capture + Peppol e-invoicing + Odoo ops
- Contacts: sales@acme.sg
- Sources: https://acme.sg/about, https://sgpbusiness.com/company/acme-logistics
```

### 6.2 Importer behavior

* Parse MD → create `icp_research_artifacts` rows (per lead/section).
* Resolve **company** (domain exact; fallback fuzzy by name) → **companies** row (create if missing).
* Write **icp_evidence** entries:

  * `signal_key='research_fit'` with `value.tags`, `value.ops_signals`.
  * `signal_key='research_note'` with a short normalized summary.
* Enqueue to `discovery_queue(source='manual_research', priority=manual_research_bonus)` (see §7) if not already queued.
* Merge **`ai_metadata`**: `{provenance:{doc_path, sources[], reviewer?, hash}}`.

---

## 7) Scoring & Ranking (with ResearchOps)

### 7.1 Evidence Score (0–100) — unchanged

```
+30  evidence_types ≥ 3
+25  integration ∩ icp.integrations ≠ ∅
+15  up to 2 champion title matches
+10  hiring_open_roles ≥ 3
+10  pricing OR case studies
+10  last_seen ≤ 90d
```

### 7.2 Manual Research bonus (additive, capped)

```
+10  if research_fit.tags contains any icp.integrations/ops_signals keyword
+10  if research artifact cites ≥2 source URLs
+5   if snapshot mentions ICP trigger phrase (from system ICP in icp_rules)
(Max +20)
```

### 7.3 Tenant Final Score (gate for display)

```
final = 0.45 * (EvidenceScore + ManualResearchBonus)
      + 0.30 * SSIC_fit(0/100)
      + 0.15 * Size_band_fit(0/100)
      + 0.10 * Entity_type_bonus(0/50)
```

**Buckets**: A ≥ 65, B 50–64, C < 50 (configurable).
**Ranking everywhere**: EvidenceScore→LeadScore→Employees→Year→stable hash.

---

## 8) Discovery & Scheduler

### 8.1 Sources (priority order)

1. `web_discovery` (instant lookalikes) — priority = EvidenceScore
2. `manual_research` (from docs) — priority = ManualResearchBonus + pre_relevance
3. `acra_ssic` (nightly SSIC candidates) — priority = pre_relevance

### 8.2 Nightly ACRA (tenant-specific) — unchanged behavior + Research bonus in final score

* Candidate selection from ACRA by SSIC + age/entity filters; enqueue with pre_relevance; resolve/crawl; score; present **A-bucket** only.

**Config knobs (env)**

```
SCHED_PRIORITY_SOURCE_ORDER=web_discovery,manual_research,acra_ssic
SCHED_DAILY_CAP_PER_TENANT=200
SCHED_COMPANY_BATCH_SIZE=10
SCHED_DISCOVERY_BATCH=100

ICP_DISCOVERY_TOPK=10
ICP_DISCOVERY_CAP=200
ICP_DISCOVERY_MAX_PAGES=3
ICP_SCORE_A_MIN=65
ICP_MIN_EVIDENCE_TYPES=1
MANUAL_RESEARCH_BONUS_MAX=20
```

---

## 9) Odoo Integration (recap)

* `res.partner` extra fields: `x_is_seed_customer`, `x_external_domain`, `x_uen`, `x_ssic_primary`, `x_ai_metadata`.
* New model `nexius.icp.profile` for visibility of active ICP payload.
* Seeds mirrored as partners; ICP mirrored as a single active record per tenant.

---

## 10) APIs & CLI

### 10.1 Internal APIs

* `POST /icp/run` → `{top10, queued, micro_icp}`
* `GET /icp/top10?tenant_id=` → cards with badges + sources
* `POST /icp/profile` → store + set active
* `GET /leads?tenant_id=&bucket=A&limit=50` → ranked gated list with `reason_json`
* **NEW** `POST /research/import` → scans `docs/` (or accepts files) and upserts artifacts; returns `{files_scanned, leads_upserted, errors[]}`

### 10.2 CLI (optional helper)

```bash
# Ingest local docs/ artifacts for a tenant
nexius icp research-import --tenant 1041 --root ./docs
```

---

## 11) Security, Compliance, Guardrails

* Respect **robots.txt**; timeouts; small page budgets; per-domain throttles.
* No **PII** in evidence/`ai_metadata`. Contacts, if captured, live in a separate, ACL-guarded table (out of scope here).
* Research artifacts store only **snapshots & links**; keep size bounded and hashed; provenance in `ai_metadata`.
* Multi-tenant isolation on all reads/writes; tenant notes stay in `tenant_companies.ai_metadata`.

---

## 12) Telemetry & KPIs

* **Time-to-value**: % sessions with click/export from Top-10.
* **Precision proxy**: Top-10 → shortlist rate.
* **Coverage**: % leads surfaced with `evidence_types ≥ 1`.
* **Queue health**: avg priority processed, age, failure rate (<3%/day).
* **Research ingestion**: files scanned, leads upserted, citation density.
* **Odoo sync**: seed → partner mirror rate; dupe rate <1%.

---

## 13) Acceptance Criteria

* After ≥5 seeds, **Micro-ICP saved**; **Top-10** appears with ≥2 badges and source tooltips.
* Research ingestion creates **icp_research_artifacts** rows, writes **icp_evidence(research_*)**, enqueues `manual_research`, and links sources. Markdown in `docs/` is optional input; DB is the source of truth.
* Nightly scheduler processes queues in the order: **web_discovery → manual_research → acra_ssic**.
* Presented leads (default) are **A-bucket**, have a **website**, and **≥1 evidence** signal; **“why”** panel cites SSIC/Integrations/Titles/Freshness and any **research sources**.
* Seeds mirrored to **Odoo** with `x_is_seed_customer=True`; ICP mirrored to `nexius.icp.profile`.
* All enriched tables contain **`ai_metadata`** with crawl/ingest agent + timestamps.

---

## 14) Rollout Plan

1. **Phase 0 — Migrations**: tables/MV/indexes; add `icp_research_artifacts` & `research_import_runs`; Odoo module deploy.
2. **Phase 1 — Ingestion**: ship `/research/import` + CLI; ingest a pilot `docs/` repo; verify mapping & scoring.
3. **Phase 2 — Default On**: enable queue priority order incl. `manual_research`; raise nightly caps gradually.
4. **Phase 3 — Hygiene**: backfill MV; prune large snapshots; re-score legacy rows; doc the workflow for analysts.

Feature flags: `FF_ICP_WEB_DISCOVERY`, `FF_TOP10_SECTION`, `FF_ODOO_ICP_PROFILE`, `FF_RESEARCH_IMPORT`.

---

## 15) Engineering Notes (snapshots)

**Manual Research bonus (SQL-ish)**

```sql
-- Additive bonus derived from research artifacts
WITH ra AS (
  SELECT company_id,
         COALESCE((fit_signals->'tags')::jsonb, '[]'::jsonb) AS tags,
         array_length(source_urls, 1) AS url_ct
  FROM icp_research_artifacts
  WHERE tenant_id = :tenant_id
)
SELECT c.company_id,
       LEAST(
         10 * (EXISTS (SELECT 1 FROM jsonb_array_elements_text(ra.tags) t
                       WHERE lower(t.value) = ANY(:icp_integrations OR '{}'))::int)
       +10 * (COALESCE(ra.url_ct,0) >= 2)::int
       , 20) AS manual_research_bonus
FROM companies c
JOIN ra ON ra.company_id = c.company_id;
```

**Reason JSON example**

```json
{
  "ssic_match": "62012",
  "integration_hit": ["hubspot"],
  "title_hits": ["revops"],
  "evidence_types": 4,
  "freshness_days": 23,
  "research_sources": [
    "https://acme.sg/about",
    "https://sgpbusiness.com/company/acme"
  ]
}
```

---

**Outcome:** ICP Finder now blends **automated signal mining** with a **repeatable ResearchOps workflow**. You get instant Top-10 lookalikes, tenant-specific nightly ACRA leads, full provenance, Odoo visibility, and the ability to **shape the pipeline** with documented human research that the system can ingest, score, and explain.
