# To-Do List — Dev Plan 19 (Pre-Enrichment Lead Generation)

Source: Development_Plan/Development/featurePRD19Revised.md
Purpose: Track implementation of minimal ICP intake + agent micro‑ICP, Top‑10 preview and strict enrich‑now, Non‑SG background next‑40 with chat notification, ResearchOps import, prioritized nightly queues, deterministic scoring/gating, and Odoo sync.

Legend: [ ] Pending · [~] In Progress · [x] Done

## 1) Database & Migrations
- [x] Create research ops tables (`icp_research_artifacts`, `research_import_runs`) with GIN indexes and `ai_metadata`.
  - Implemented: `lead_generation-main/app/migrations/017_prd19_research_ops.sql`.
- [x] Ensure `icp_evidence` accommodates research_* evidence entries and provenance.
  - Implemented: `src/research_import.py` writes `research_fit` and `research_note` to `icp_evidence`.
- [~] Ensure canonical/enrichment tables include `ai_metadata` (companies, acra_enrichment, global_company_registry, tenant_companies, icp_rules, lead_scores, discovery_queue).
  - Note: `ai_metadata` present where needed for PRD19 flows; broader adoption optional.
- [x] Add staging table for global web discovery candidates: `staging_global_companies` with unique index and created_at index.
  - Implemented: `lead_generation-main/app/migrations/018_staging_global_companies.sql`.
  - Enhancement: staging writes include meaningful `ai_metadata` (provenance). Top‑10 preview rows store `ai_metadata.preview=true`, `score`, `bucket`, `why`, `snippet`.
  - RLS/GUC: `request.tenant_id` is set before staging upserts for consistent scoping.
- [ ] Add/refresh MV for web evidence roll-up: `icp_features_web(company_id, evidence_types, hiring_open_roles, has_pricing, has_case_studies, integrations[], buyer_titles[], last_seen)`.
- [~] Add helpful indexes (tenant_id, company_id, GIN on JSONB, queue status/source).

## 2) Agents & LangGraph Orchestration
- [x] ICP Synthesizer (LLM) → outputs micro-ICP.
  - Implemented: `lead_generation-main/src/agents_icp.py::icp_synthesizer`; invoked from `confirm_node` when missing profile.
- [x] Discovery Planner (LLM) → generates queries, candidate domains, dedupe/justify.
  - Implemented with single "perfect" LLM-composed query; Jina‑proxied DDG pagination (max 8 pages) until 50 domains; logs added.
- [x] Mini-Crawl Worker (tool) → robots-safe snapshots (small bundle per domain).
  - Implemented: `agents_icp.mini_crawl_worker` uses `collect_evidence_for_domain`.
- [x] Evidence Extractor (LLM) → normalized signals.
  - Implemented: `agents_icp.evidence_extractor` (integrations, buyer_titles, hiring, pricing, case studies).
- [x] Scoring & Gating node (tool) → A/B/C + reason_json.
  - Implemented: `agents_icp.scoring_and_gating` with PRD19 weights (integrations/titles/pricing emphasized).
- [x] Compose nodes and wire into LangGraph.
  - Implemented: Agent Top‑10 integrated into `app/pre_sdr_graph.py::confirm_node` with chat output.
- [x] Strict Top‑10 enrichment path (no ACRA/top‑ups in immediate batch).
  - Implemented: `app/pre_sdr_graph.py::run_enrichment` builds candidates from Top‑10 domains when present.
  - Enhancement: If Top‑10 is not found in memory or persisted for the tenant, the system asks the user to confirm to regenerate; it does not crawl seeds as a fallback during the immediate batch.
- [x] Always‑10 shortlist with resilient fallbacks.
  - Implemented: DDG→Jina pipeline with added resilience: prefers `duckduckgo.com/html` then `duckduckgo.com/lite`, falls back to `ddgs` library, and finally proxies via `r.jina.ai` if direct endpoints are blocked; also falls back to HTTP homepage snippets on 429/timeouts; backfills from later DDG candidates, seed‑based queries (apex‑label), and legacy heuristic until 10.
- [x] Exclude seeds from discovery Top‑10 candidates.
  - Implemented: `src/agents_icp.py` filters out `SEED_HINTS` from candidate sets and backfills.
- [x] DDG discovery via Jina proxy + single query; no re‑discovery at enrich.
  - Implemented: agents use one LLM‑composed query; fetch `https://r.jina.ai/https://html.duckduckgo.com/html/?q=…` with pagination (up to 8 pages via `s` offset) until 50 unique domains; enrich reuses persisted Top‑10 and never re‑plans.
- [x] Strong domain validation (reject images/files/shorteners).
  - Implemented: `_is_probable_domain()` guards DDG parsing, fallbacks, and candidate lists.
- [x] DDG pagination until 50 unique per query; early‑stop remaining queries when one query fills 50.
  - Implemented: Jina‑proxied DDG snapshots paginate up to 8 pages using the `s` offset; LLM extracts domains per page; planner stops queries as soon as 50 unique are collected.
- [~] Research Import Processor (tool+LLM assist).
  - Implemented as API/module (`/icp/research/import`, `src/research_import.py`); not wired into graph.
- [ ] ACRA Nightly Orchestrator (tool) → SSIC-based SG pass.
- [~] Odoo Sync (tool) → mirror seeds/companies and active ICP profile.
 - [x] Non‑SG next‑40 background enrichment flow.
   - Implemented: After Top‑10, select next 40 from persisted preview (ordered by preview score, excluding Top‑10), map to company_ids, enqueue `web_discovery_bg_enrich`, surface `{ job_id }`, store `pending_bg_jobs` in state. On subsequent user turns, agent posts completion summary (processed, A/B/C split, errors) by checking `/jobs/{id}`; SSE/poll-on-timer optional.

## 3) Backend APIs (Pre-Enrichment Focus)
- [x] `POST /icp/run` → returns Top‑10 with preview persisted to DB.
- [x] `GET /icp/top10?tenant_id=` → Top‑10 with why/snippets; persists preview evidence (`icp_evidence`) and scores (`lead_scores`). Enhancement: reconstruct `icp_profile` from `icp_rules.payload` (including `size_bands`) and use it for discovery.
  - Persistence hardening: set `request.tenant_id` GUC for RLS on write/read; enrich falls back to reading Top‑10 preview from `staging_global_companies.ai_metadata` when `icp_evidence` is empty.
- [x] `POST /icp/accept` → upsert active ICP; trigger head enrichment + nightly scheduling. On confirm/enrich, newly inferred `icp_profile` fields (industries, integrations, buyer_titles, size_bands, triggers) are merged into `icp_rules.payload`.
- [ ] `GET /leads?tenant_id=&bucket=A&limit=50` → gated A-bucket with `reason_json`.
- [x] `POST /icp/research/import` → ingests ResearchOps artifacts (optionally from a `docs/` root); returns `{files_scanned, leads_upserted, errors[]}`.
 - [x] `POST /icp/enrich/top10` → enrich persisted Top‑10; return enriched shortlist (A/B/C + why).
 - [x] `POST /icp/enrich/next40` (Non‑SG) → enqueue background enrichment for next 40; return `{ job_id }`.
- [x] `GET /jobs/{job_id}` → expose job status (queued/running/done/error) for agent polling.
 - [x] Document DDG Postman test steps in Dev Plan.

## 4) Scheduler & Queues
- [x] Maintain source priority: `web_discovery → manual_research → acra_ssic` (config `SCHED_PRIORITY_SOURCE_ORDER`).
- [x] Config caps: `SCHED_DAILY_CAP_PER_TENANT`, `SCHED_COMPANY_BATCH_SIZE`, `SCHED_DISCOVERY_BATCH`.
- [x] Nightly ACRA: select SSIC candidates by tenant filters; enqueue with `pre_relevance`.
- [x] Nightly ACRA bootstrap (first run per tenant): in `scripts/run_nightly.py` auto‑enqueue `staging_upsert(terms)` from `icp_rules` if no ACRA jobs exist for that tenant.
- [x] Nightly ACRA daily cap: enforce `ACRA_DAILY_ENRICH_LIMIT` (default 20) per tenant per day in `src/jobs.run_enrich_candidates`; re‑queue any remaining work for the next night. Finalize `enrichment_runs` headers per job.
- [x] Web discovery: enqueue remainder after Top‑10 with priority = EvidenceScore.
- [x] Persist all web discovery candidates to `staging_global_companies` for auditability.
  - Includes provenance in `ai_metadata` for planned candidates; per‑domain Top‑10 preview stored as well.
- [ ] Manual research: enqueue with ManualResearch bonus + pre_relevance.
 - [x] Non‑SG background next‑40: implement `web_discovery_bg_enrich` job; small concurrency; respect vendor caps; observability writes. Completion summary posted in chat on next user turn; timer/SSE polling optional.
- [x] Dedicated worker for next‑40: add `scripts/run_bg_worker.py` (async LISTEN/NOTIFY + polling fallback), safe job claiming with `FOR UPDATE SKIP LOCKED`, bounded concurrency via `BG_WORKER_MAX_CONCURRENCY`.
 - [x] Enqueue‑only in app: remove inline `create_task` run; app enqueues and returns job_id with a chat notice; worker processes the job.
 - [x] Instant wakeup on enqueue: add `NOTIFY bg_jobs, '{"job_id":...,"type":"web_discovery_bg_enrich"}'` in `enqueue_web_discovery_bg_enrich`.
 - [~] Ops: add systemd unit (or Docker service) to run the worker in production; document env vars and restart policy.

## 5) Scoring & Gating
- [~] Deterministic EvidenceScore from web/research/ACRA signals.
  - Implemented for web preview in `agents_icp.scoring_and_gating`; DB-backed scoring pending.
- [ ] ManualResearch bonus (cap) merged into final score.
- [x] Buckets: A ≥ 70, B ≥ 50, C < 50 (tuned for PRD19 preview).
- [ ] Ranking: EvidenceScore → LeadScore → Employees → Year → stable hash.
- [~] Populate `reason_json`.
  - Preview reasons include integrations/titles/pricing/case studies; persistence pending.
  - Include research citations and contact/email verification when available.

-## 6) ResearchOps Ingestion (DB-first)
- [x] Importer ingests submitted artifacts (Markdown under `docs/` is optional input): profiles, leads.
- [x] Resolve/insert companies; write `icp_research_artifacts` with provenance in `ai_metadata`.
- [x] Write `icp_evidence(research_*)` entries and link sources.
- [ ] Enqueue `manual_research` candidates; re-score impacted leads.
- [ ] CLI helper (optional wrapper around API).

## 7) Odoo Integration
- [~] Mirror seeds and active ICP profile; enrichment sync exists.
  - Existing Odoo store in place; PRD19-specific mirrors pending.

## 8) Security & Compliance
- [~] Respect robots.txt/limits (mini-crawl is small); throttles in crawler modules.
- [x] Store no PII in research evidence/`ai_metadata`; bound MD snapshot length.
- [x] Enforce multi-tenant isolation on reads/writes (existing guards in DB helpers; review added paths).
  - Implemented: set `request.tenant_id` via `set_config` before Top‑10 preview writes/reads and staging upserts; strict reuse of tenant‑scoped Top‑10 at enrich.
- [x] Hash/bound Markdown snapshots; persist provenance in `ai_metadata` (research runs).
 - [x] Non‑SG background: ensure job status polling and summaries are tenant‑scoped; no cross‑tenant visibility (jobs queried by id; state carries job ids; GUC tenant set on DB reads).

## 9) Telemetry & KPIs
- [ ] Time-to-value: % sessions with action from Top‑10.
- [ ] Precision proxy: Top‑10 → shortlist rate.
- [ ] Coverage: % leads with `evidence_types ≥ 1`.
- [ ] Queue health: avg priority processed, age, daily failure rate.
- [~] Research ingestion: files scanned, leads upserted (returned by API today).
- [ ] Odoo sync: seed mirror rate; duplicate rate.
 - [ ] Top‑10 SLA: p95 time from CTA to enriched shortlist ≤ 5 minutes.
 - [ ] Background next‑40: completion counts, duration, A/B/C split; vendor cap hit ratio.
 - [x] DDG logs clarify parsed vs uniq counts per run.

## 10) UX Deliverables
- [x] Top‑10 Lookalikes presented as a pretty markdown table (rank • domain • score • why • snippet); snippets cleaned.
  - Implemented in chat output (LangGraph). UI badges/tooltips pending.
- [x] Show web discovery total in chat; persist domains to staging.
  - Implemented: web candidate total displayed; persisted via `staging_global_companies`.
  - UI polish: preview table shows reasons/snippets; stricter seed exclusion reduces noise.
- [x] ACRA nightly A-bucket list with badges/explanations.
- [ ] Micro‑ICP cards with “Use as filter”.
- [~] ResearchOps import feedback.
  - API returns files/leads/errors; UI integration pending.
 - [x] CTA: “Enrich Top‑10 now” → stream progress chips and render enriched shortlist.
 - [x] Non‑SG next‑40 chat messages: show queued job with link to status; post completion summary (processed, A/B/C split). Enhancement (optional): proactive polling to announce without user turn.

## 11) Feature Flags & Config
- [x] `ENABLE_ICP_INTAKE` (default true) — gate new ICP Finder.
- [x] `ENABLE_AGENT_DISCOVERY` (default true) — enable DDG/Jina agents.
- [x] `ENABLE_ACRA_IN_CHAT` (default false) — keep SSIC/ACRA out of chat.
- [x] `DOCS_ROOT` — server path for ResearchOps docs.
- [x] `STAGING_GLOBAL_TABLE` — staging table name for web discovery (default `staging_global_companies`).
- [~] Tuning knobs: `CHAT_ENRICH_LIMIT`/`RUN_NOW_LIMIT`, `CRAWL_*`, `LLM_*` present; additional knobs optional.
 - [ ] Flags: `FF_TOP10_ENRICH_NOW` (gate enrich‑now UI/API), optional `FF_NEXT40_BG_ENRICH`.
 - [ ] Config: `RUN_NOW_LIMIT=10`, `BG_NEXT_COUNT=40`, `ENRICH_RECHECK_DAYS`, `ENRICH_SKIP_IF_ANY_HISTORY`.

## 12) Rollout Plan
- [x] Phase 0 — Migrations: research tables + indexes added.
- [~] Phase 1 — Ingestion: `/icp/research/import` shipped; ingest/pilot ready.
- [x] Phase 2 — Default On: queue order + nightly caps.
- [ ] Phase 3 — Hygiene: MV backfill; prune snapshots; re-score legacy rows; docs.
- [x] Enable Non‑SG background next‑40 in Phase 2 after Top‑10 enrich‑now is stable.

## 13) Testing & QA
- [~] Unit: research parser + idempotent upsert (hash) covered by implementation; dedicated tests pending.
- [ ] Integration: `/icp/research/import` end-to-end including evidence/queue/score updates.
- [ ] E2E: seeds → micro‑ICP saved → Top‑10 in seconds → “Enrich Top‑10 now” → enriched shortlist ≤ 5 min p95 → Non‑SG next‑40 runs in background and posts chat completion → nightly ACRA A‑bucket → research import influences scores.
- [ ] Country E2E: SG (ACRA nightly) vs non‑SG path.
- [ ] UI: import action, Top‑10 badges, “why” panel, error states.

## 14) Docs & Ops
- [ ] Link PRD to project docs; add “What changed since PRD19” summary.
- [ ] Add runbooks for scheduler, research import, and Odoo sync.
- [ ] Add minimal on-call diagnostics (dead-letter queue, ai_metadata error logs).

---

Notes
- This file reflects current implementation status in `lead_generation-main` as of now.
- Align naming with actual schema (`icp_rules` is used for ICP storage).
## 9) Documentation Updates (added)
- [x] PRD: Clarified 3 enrichment paths (Top‑10 immediate, Next‑40 background worker, Nightly ACRA scheduled), first‑run bootstrap and daily cap details.
- [x] Dev Plan: Added Nightly ACRA bootstrap + per‑tenant daily cap, re‑queue semantics, env vars.
- [x] To‑Do: Reflected completion of bootstrap and daily cap items (this file).
