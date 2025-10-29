# Feature PRD 19 — ICP Finder vNext (Revised)

## 0) Executive Summary
- Blend minimal ICP intake, agent micro‑ICP synthesis, instant Top‑10 discovery, and immediate enrichment of the Top‑10.
- Prefer Apify LinkedIn for contacts; verify emails via ZeroBounce; respect costs and caps.
- Queue remainder for nightly runs (web_discovery → manual_research → acra_ssic) with clear observability and acceptance.
- Keep provenance via `ai_metadata`; enforce multi‑tenant isolation end‑to‑end; mirror to Odoo when ready.

## 1) Scope
- In‑scope: minimal prompts ICP intake, micro‑ICP synthesis, Top‑10 staging and enrichment, discovery queues, ResearchOps ingest, scoring, scheduler priority, dashboards, acceptance.
- Out‑of‑scope: outbound sequencing/campaigns, non‑public personal data processing, heavy ML models.

## 2) Goals / Non‑Goals
- Goals: fast Top‑10 with explainability; enrich the Top‑10 immediately; nightly throughput for the rest; costs under control; reproducible logs and metrics.
- Non‑Goals: full UI redesign; complex ML; storing PII.

## 3) Users & Flows
- SDR/RevOps: confirm ICP → see Top‑10 → “Enrich Top‑10 now” → shortlist with why and contacts.
- Analyst (ResearchOps): submit Markdown artifacts; system ingests as evidence and queues leads.
- Admin/Platform: monitor runs, vendor usage, acceptance checks, and Odoo readiness.

## 4) Minimal ICP Intake
- Prompts: website + a few best‑customer seeds; no 10‑question wizard.
- Agent: infer micro‑ICP (SSIC hints, integrations, titles, size bands) from seeds + crawls.
- Persist: on confirm, upsert to `icp_rules` (active ICP); intake answers to `icp_intake_responses`.
- Backend: ACRA/SSIC mapping and `icp_patterns` materialization occur in jobs/nightly.

## 5) Instant Web Discovery (Top‑10 Preview)
- Plan: derive queries from micro‑ICP; discover domains; mini‑crawl for signals.
- Stage: persist to `staging_global_companies` with `source='web_discovery'` and `ai_metadata` preview fields (`score`, `bucket`, `why`, `snippet`, `preview=true`).
- Present: show Top‑10 table with badges and “why”; display total discovered count; queue remainder.
 - DDG Pagination & Early‑Stop: For each discovery query, paginate DuckDuckGo HTML/Lite result pages until 50 unique domains are collected for that query or pages are exhausted. If a single query yields 50 unique domains, skip executing any remaining queries. Logs clarify per‑page parsed counts and final parsed vs. uniq totals.

## 6) Top‑10 Immediate Enrichment (Run‑Now)
- Trigger: user clicks “Enrich Top‑10 now” or types `run enrichment` after confirm.
- Input: reuse persisted Top‑10 preview; fail fast if missing (ask to confirm again).
- Steps:
  - Crawl: 4–6 high‑signal pages; respect robots and per‑domain throttles.
  - Extract: merge deterministic signals with bounded LLM extraction.
  - Contacts: Apify LinkedIn chain (cap via `APIFY_DAILY_CAP`); verify any emails via ZeroBounce.
  - Score: compute EvidenceScore and final buckets; produce concise “why”.
  - Persist: `summaries`, `company_enrichment_runs`, `companies` updates, lead/score tables; tag `ai_metadata` provenance.
- Output: enriched Top‑10 table (A/B/C, why, contacts) + CSV/JSON export.
- Guards:
  - Skip if recent enrichment (`ENRICH_RECHECK_DAYS`) or any history (`ENRICH_SKIP_IF_ANY_HISTORY`).
  - Bound costs via `CRAWL_MAX_PAGES`, `EXTRACT_CORPUS_CHAR_LIMIT`, `LLM_MAX_CHUNKS`, vendor caps.

### 6.1) Non‑SG Background Top‑ups (Next 40)
- Behavior: For non‑SG ICPs, after enriching the Top‑10, automatically enqueue enrichment for the next 40 discovered domains in the background.
- UX: The agent informs the user that “I’m enriching the next 40 in the background; I’ll reply here when it’s done. You can also check the status anytime.”
- Mechanics:
  - Select next 40 from the same persisted preview (ordered by preview score), excluding the Top‑10.
  - Enqueue a background job (`web_discovery_bg_enrich`) carrying the 40 company_ids and the tenant id.
  - A dedicated worker service (separate process) listens for NOTIFY `bg_jobs` and immediately claims queued jobs; it is resilient to API/chat restarts and supports safe parallelism.
  - Post a follow‑up message in the originating chat when the job status transitions to `done` (agent polls `/jobs/{id}` or subscribes via SSE). Include a quick summary (processed, A/B/C counts, errors).
  - Respect all vendor caps; if caps are reached, process what’s available and report partial completion.
  - Do not mix ACRA or manual_research in this immediate top‑up batch.

### 6.2) Contacts (Domain‑First Apify)
- Strategy: Prefer domain‑first contact discovery. When a candidate’s website domain is known from discovery/enrichment, build Apify queries anchored on that domain (e.g., "example.com" "head of operations" site:linkedin.com/in) to surface employee profiles; fall back to company‑name queries only when the domain is unavailable.
- Titles: Use `buyer_titles` from the active ICP profile for search (fallback to `CONTACT_TITLES`, then defaults). Log titles used per company: `[apify_contacts] titles_source=<source> titles_used=[...] company_id=<id>`.
- Chain: The optional company→employees→profile chain runs only when domain is not available; this keeps the domain‑first approach primary.

## 7) Queue The Rest (Nightly)
- Sources (priority): `web_discovery` → `manual_research` → `acra_ssic` (config: `SCHED_PRIORITY_SOURCE_ORDER`).
- Nightly ACRA (Scheduled, SG‑only) — simple model:
  - First night per tenant (bootstrap): Read the tenant’s accepted ICP from `icp_rules` (preferred_titles → industries → ssic_codes→titles). Auto‑enqueue one `background_jobs` row of type `staging_upsert(terms=…)` so candidates are selected from `staging_acra_companies` at night. Then, `staging_upsert` best‑effort enqueues an `enrich_candidates(ssic_codes=…)` job.
  - Subsequent nights: The nightly dispatcher only consumes queued ACRA jobs (`staging_upsert` then `enrich_candidates`). It does not touch Next‑40 (handled by the background worker).
  - Daily per‑tenant cap: Enrichment is limited to 20 companies per tenant per day (configurable via `ACRA_DAILY_ENRICH_LIMIT`, default 20). If a job has more candidates, it processes up to the remaining daily quota and re‑queues itself for the next night.
  - Persistence & observability: Normal enrichment pipeline runs (domain→crawl→LLM→contacts→verify→persist) writing to `companies`, `company_enrichment_runs`, `contacts`, `lead_emails`, and observability tables (`enrichment_runs`, `run_event_logs`, `run_vendor_usage`, `run_stage_stats`, `run_summaries`).

## 8) ResearchOps Ingest
- Input: Markdown artifacts (optional local drafts) and/or JSON payloads; submit via `/research/import`.
- Persist: `icp_research_artifacts` (with citations), `icp_evidence` (research_* signals), enqueue `manual_research` with capped bonus.
- DB‑first: database is source of truth; file paths only for provenance.

## 9) Data Model (Additions/Usage)
- Staging: `staging_global_companies(tenant_id, domain, source, ai_metadata jsonb)` — preview Top‑10 with per‑domain metadata.
- Intake: `icp_intake_responses`, `customer_seeds`, `icp_evidence` (migrations 013/014) and MV `icp_patterns`.
- Research: `icp_research_artifacts`, `research_import_runs` with GIN indexes.
- Observability: `enrichment_runs`, `run_event_logs`, `run_vendor_usage`, `run_stage_stats`, `qa_samples`.

## 10) APIs
- `POST /icp/intake`: save answers + seeds; enqueue mapping/patterns.
- `GET /icp/patterns`: return tenant patterns.
- `GET /icp/top10`: return Top‑10 preview (why/snippets); same payload for chat preview reuse.
- `POST /icp/accept`: persist micro‑ICP to `icp_rules` (active profile).
- `POST /icp/enrich/top10`: enrich persisted Top‑10; return enriched shortlist.
- `POST /icp/enrich/next40`: enqueue background enrichment for the next 40 (non‑SG only). Returns `{ job_id }`. The agent will notify on completion.
- `POST /research/import`: ingest artifacts; enqueue manual research enrich.
- `GET /leads?bucket=A`: list enriched leads with reasons (tenant‑scoped).
 - `GET /jobs/{job_id}`: poll job status for background enrich; agent uses this to send a completion message in chat.

## 11) Scoring & Buckets
- EvidenceScore: deterministic site signals + LLM evidence.
- ManualResearchBonus: additive, capped; requires citations.
- TenantFinalScore: combine EvidenceScore (+bonus) with SSIC/size/entity fit; A ≥ 65, B 50–64, C < 50.

## 12) Scheduler & Throughput
- Orchestrator: APScheduler (`scripts/run_scheduler.py`), per‑tenant runs.
- Dedicated background worker for next‑40: `scripts/run_bg_worker.py` (async Postgres LISTEN/NOTIFY + polling fallback) processes `web_discovery_bg_enrich` jobs as soon as they are enqueued. The API/chat only enqueues; it never runs the job inline.
- Priority: `web_discovery` → `manual_research` → `acra_ssic` (configurable). Nightly remains focused on ACRA/SSIC and does not process next‑40 jobs.
- Nightly ACRA bootstrap: `scripts/run_nightly.py` performs a one‑time per‑tenant bootstrap using `icp_rules` to enqueue `staging_upsert`. Afterwards, it only processes queued ACRA jobs.
- Nightly ACRA cap: Enforce `ACRA_DAILY_ENRICH_LIMIT` (default 20) per tenant per day; re‑queue `enrich_candidates` when quota is hit.
- Caps: Apify daily cap; ZeroBounce batch size; LLM corpus/timeouts; per‑domain crawl budget.
- Non‑SG top‑ups: background batch of 40 runs immediately after Top‑10 via the worker; throttle via small concurrency and respect vendor caps.

## 13) Observability & QA
- Events: per‑stage logs with durations and error codes; vendor usage per tenant.
- Metrics: candidate counts, fallback rates, verified email rate, score distribution, token spend, vendor costs.
- QA: 10 random High/A‑bucket checks per run; store samples.
 - DDG logs: emit `[ddg] parsed domains=<count> page=<N> via <endpoint> for query=<q>` while paginating, and a summary `[plan] ddg domains found=<parsed> (uniq=<deduped>)`. Distinguish parsed (raw hits across pages) vs uniq (deduped apex domains).

## 14) Security & Compliance
- Robots and ToS respected; business contacts only; suppress personal emails when policy dictates.
- Multi‑tenant isolation (RLS/WHERE), role‑based access (admin/ops/viewer).
- Retention: short retention for event logs; bounded artifacts; DSAR purge planned per tenant.

## 15) Acceptance
- Top‑10 appears within seconds after confirm; enrichment for Top‑10 completes ≤ 5 minutes p95.
- Nightly processes queued web_discovery and ACRA candidates; acceptance checks pass per tenant (domain ≥ 70%, about ≥ 60%, email ≥ 40%).
- Contacts via Apify; emails verified when present; costs logged; Lusha disabled by default.
- Odoo mapping verified per tenant; export succeeds for enriched companies.
 - Non‑SG: Top‑10 shown immediately; next 40 run in background with a chat notification on completion. Users can query job status via `/jobs/{id}`; completion summary includes processed count and A/B/C split.

## 16) Rollout & Flags
- Phases: (1) Migrations + flags → (2) Top‑10 preview + enrich‑now → (3) ResearchOps ingest → (4) Prioritized nightly → (5) Acceptance dashboards.
- Flags: `FF_ICP_WEB_DISCOVERY`, `FF_TOP10_SECTION`, `FF_TOP10_ENRICH_NOW`, `FF_RESEARCH_IMPORT`.
- Env: `ACRA_DAILY_ENRICH_LIMIT` (per‑tenant nightly cap), `BG_WORKER_MAX_CONCURRENCY`, `BG_WORKER_SWEEP_INTERVAL`, `SCHED_START_CRON`.

## 17) Implementation Checklist (Code Touchpoints)
- Chat Flow: `app/pre_sdr_graph.py` (confirm → preview/top‑10 reuse → enrich).
- Endpoints: `app/icp_endpoints.py` (`/icp/top10`, `/icp/accept`, patterns); add `/icp/enrich/top10`.
- Intake/Patterns: `src/icp_intake.py`, `src/icp_tasks.py`; MV refresh in scheduler.
- Enrichment: `src/enrichment.py` (crawl/merge/score), `src/vendors/apify_linkedin.py`, ZeroBounce in `src/enrichment.py`.
- Jobs/Scheduler: `src/jobs.py`, `scripts/run_nightly.py`, `scripts/run_scheduler.py`.
 - Add job: `web_discovery_bg_enrich` (background next‑40) and wire agent notification on `/jobs/{id}` completion.
- ResearchOps: `src/research_import.py`, jobs `enqueue_manual_research_enrich`.
- Settings/Flags: `src/settings.py` (caps, toggles), `.env`.
- Observability: `src/obs.py`, `scripts/alerts.py`, `scripts/acceptance_check.py`.

## 18) Config Knobs (Env)
- Discovery: `CRAWL_MAX_PAGES`, `EXTRACT_CORPUS_CHAR_LIMIT`, `LLM_MAX_CHUNKS`, `LLM_CHUNK_TIMEOUT_S`.
- Vendors: `APIFY_TOKEN`, `APIFY_DAILY_CAP`, `ENABLE_APIFY_LINKEDIN`, `ZEROBOUNCE_API_KEY`.
- Enrichment: `ENRICH_RECHECK_DAYS`, `ENRICH_SKIP_IF_ANY_HISTORY`.
- Scheduler: `SCHED_START_CRON`, `SCHED_PRIORITY_SOURCE_ORDER`.
 - DDG: `DDG_PAGINATION_MAX_PAGES` (default 8), `DDG_PAGINATION_SLEEP_MS` (default 250), `DDG_MAX_CALLS` (min 2 effective), `DDG_KL` locale.

## 19) Risks & Mitigations
- Vendor quotas: cap and degrade gracefully; cache previews; skip contacts on quota errors.
- Preview mismatch: persist and reuse Top‑10; refuse enrichment if preview missing.
- Cost spikes: strict corpus/timeouts and daily caps; skip if recent history.
- Data quality: QA samples; acceptance checks; ResearchOps citations required for bonus.
