# PRD‑Opt: Singapore‑Focused ICP Discovery & Enrichment — Current vs New, Rationale, and Flows

This document translates the optimization spec (see `lead_generation-main/devplan/featurePRD_optimise.md`) into an actionable implementation plan. It explains:
- What we have today (current system behavior and code paths)
- What changes we will introduce and why
- How the code and process flows work end‑to‑end
- Data model, config, testing, and rollout


## 1) Context & Goals

We observed noisy Top‑10s and uneven enrichment for Singapore‑focused runs due to domain/category drift (expos, directories, standards), weak SG signals on `.com` domains, SaaS‑biased scoring, sparse evidence, contact coverage gaps, domain hygiene issues, and a UEN/ACRA normalization bug that marked foreign sites as `sg_registered` without high‑confidence UEN.

Optimization goals:
- Precision: ≥80% valid organizations for the chosen lead profile with Singapore presence.
- Actionability: Each shortlist has “why now” and at least one relevant buyer title or pattern.
- Transparency: Every score has explainable signals; failures include first‑drop reason.
- Configurability: Country/industry guardrails, allow/deny lists, lead profiles, weights.


## 2) Current Implementation Snapshot (What we have now)

The interactive ICP → Top‑10 → Enrichment flow is implemented across:
- `app/icp_endpoints.py`
  - `GET /icp/top10`: uses `src/agents_icp.plan_top10_with_reasons()` to plan candidates; persists Top‑10 preview into `icp_evidence` and `lead_scores`; now also persists Top‑10 and the “remainder” into `staging_global_companies` (preview vs non‑preview) so Next‑40 always has rows.
  - `POST /icp/enrich/top10`: enriches strictly the persisted Top‑10 by mapping domains → `companies.company_id`; emits SSE ticks and summary; selects “remainder” from `staging_global_companies` (preview=false) and enqueues Next‑40 background job via `src.jobs.enqueue_web_discovery_bg_enrich()`; emits `enrich:next40_enqueued` with `{job_id}`.
- `app/pre_sdr_graph.py`
  - Chat graph orchestration; persists Top‑10 preview via `_persist_top10_preview()` and (now) also stages the remainder as non‑preview; enqueues Next‑40 once per thread using `_enqueue_next40_if_applicable(state)`.
- `src/agents_icp.py`
  - Implements `plan_top10_with_reasons` using DDG/Jina; returns up to Top‑10 with “why/snippet/score”.
- `src/enrichment.py`
  - Jina‑first deterministic crawl + HTTP fallback for About/Contact; LinkedIn extraction/normalization; contact parsing; single Apify attempt guard; `search_policy='require_existing'` for Top‑10/Next‑40.
- `src/jobs.py` and `scripts/run_bg_worker.py`
  - Background jobs stored in `background_jobs`. Next‑40 uses job type `web_discovery_bg_enrich` with `params.company_ids`.
  - Worker claims `queued` jobs (`FOR UPDATE SKIP LOCKED`), runs `run_web_discovery_bg_enrich(job_id)`, and updates `processed/total/status`.
- SSE events (`src/chat_events`)
  - `icp:planning_start`, `icp:toplikes_ready`, `icp:profile_ready`, `icp:candidates_found`, `enrich:start_top10`, `enrich:company_tick`, `enrich:summary`, `enrich:next40_enqueued`.
- Tenant handling (no DEFAULT_TENANT_ID fallbacks) and strict Top‑10 reuse (persisted shortlist; re‑discovery gated).

Known fixes already landed:
- Persist “remainder” candidates to `staging_global_companies` (preview=false) so Next‑40 always enqueues.
- Jina‑first crawl with HTTP subpages fallback, timeouts, LinkedIn extraction normalization.
- Guard single Apify attempt; prevent vendor loops.
- SSE progress events throughout interactive flow.


## 3) New Changes (What will change and why)

Referencing the optimization spec, we’ll introduce:

3.1 SG Discovery Lock and Marker‑Based Gating
- Change: Constrain DDG queries to `kl=sg-en` and prefer `site:.sg`. Accept `.com` results only when page contains SG markers: `\bSingapore\b`, `+65`, `\b\d{6}\b` (SG postcode).
- Why: Reduce geo drift; `.com` domains often serve multiple regions; SG markers confirm local presence.
- Where: `src/agents_icp.plan_top10_with_reasons` discovery phase. We will add a `region_hint='sg'` and a “prefer .sg” switch; for `.com` hits, fetch/skim content and run `is_singapore_page(text)`.

3.2 Expanded Denylist and Path Filters
- Change: Block `.gov.sg`, `.edu.sg`, and apexes/paths that indicate standards, directories, expos, trade fairs, associations (`/standards|policy|directory|expo|tradefair|event|conference|exhibition/`).
- Why: Remove “non‑lead” categories early to boost precision and reduce noise in scoring.
- Where: Discovery normalization in `src/agents_icp` and page normalization in `src/enrichment`. Externalize the lists (YAML) for easy maintenance.

3.3 Multi‑Profile Scoring
- Change: Introduce `lead_profile` with distinct markers/weights:
  - `sg_employer_buyers` (employer presence + SG compliance triggers)
  - `sg_referral_partners` (HR consulting, recruitment, payroll, HRIS, EOR/PEO, corp‑sec, relocation)
  - `sg_generic_leads` (SME presence + growth)
- Why: Scoring cues differ by target; a single SaaS‑biased scorer underperforms for these lead types.
- Where: `src/agents_icp` scoring stage (or `src/lead_scoring.py`); a profile‑aware weight map and gating rules; outputs include a breakdown and “why” chips.

3.4 Evidence Extraction (SG + Profile Cues)
- Change: Extract HQ city (prefer “Singapore”), SG phone `+65`, SG postcode, hiring intensity, profile markers, compliance triggers (MOM/TAFEP/TADM/WFL/WICA/WSH/CPF/PDPA), HRIS hints.
- Why: Drive explainable scores and actionable insights (who to contact and why now).
- Where: `src/enrichment.py` deterministic extract; persist into `companies` and `icp_evidence` with compact summary (≤300 chars).

3.5 ACRA Normalization and `sg_registered`
- Change: Set `sg_registered=True` only when UEN resolves with high‑confidence name match; store `uen_confidence` and `acra_source`.
- Why: Prevent false SG registration based on TLD/markers; improve downstream trust.
- Where: `src/icp_pipeline.py`, `src/icp.py`, `src/acra_direct.py`: gate flips only with UEN+confidence.

3.6 Domain Hygiene
- Change: Accept only valid FQDNs; drop bare TLDs or suffix‑only candidates (e.g., `co.th`), normalize apex for dedupe.
- Why: Removes malformed inputs early; avoids wasted enrichment.
- Where: Candidate normalization in `src/agents_icp` and pre‑enrichment checks in `src/enrichment`.

3.7 Telemetry & Honest UX
- Change: Emit metrics for discovery precision, denylist drops, evidence completeness, shortlist yield, hygiene failures, UEN match rate. Keep “honest failures” (no vague promises) in UI.
- Why: Quantify improvements, guide further tuning, and keep user trust.
- Where: Logging in agents/enrichment/jobs; optional dashboard queries over `background_jobs`, `lead_scores`, `icp_evidence`.


## 4) Code Flow (Request → Discovery → Preview → Enrichment → Next‑40)

4.1 Interactive Chat / API (Top‑10)
- User confirms ICP; server emits `icp:planning_start`.
- `GET /icp/top10` (or chat node) calls `src/agents_icp.plan_top10_with_reasons(icp_profile, tenant_id)`:
  - Discovery: DDG constrained to SG; prefer `.sg`; skim `.com` pages for SG markers; apply denylist/hygiene.
  - Extraction: summaries/snippets; profile markers; preliminary scores with rationale.
  - Output: up to Top‑10 with `domain, why, snippet, score, bucket`.
- Persistence:
  - Top‑10 preview → `icp_evidence(top10_preview)` and `lead_scores`.
  - Staging: Top‑10 rows as preview (`ai_metadata.preview=true`) and remainder as non‑preview in `staging_global_companies` (source `web_discovery`).
- Events: `icp:toplikes_ready`, `icp:profile_ready`, `icp:candidates_found`.

4.2 Enrichment (Top‑10 strict)
- `POST /icp/enrich/top10` maps staged Top‑10 domains → `companies.company_id` and enriches each with `search_policy='require_existing'`.
- Deterministic crawl (`src/enrichment`): Homepage → About/Contact via Jina; HTTP fallback; SG/contact cues; LinkedIn normalization; single Apify attempt; persist to `company_enrichment_runs`, `contacts`, `lead_emails`, `companies`.
- Events: `enrich:start_top10`, `enrich:company_tick`, `enrich:summary`.

4.3 Next‑40 Background Enqueue
- After Top‑10, API/chat selects remainder from `staging_global_companies` where `preview=false` and enqueues a `web_discovery_bg_enrich` job with resolved `company_ids`.
- SSE: `enrich:next40_enqueued` includes `{ job_id }`.

4.4 Background Worker Execution
- `scripts/run_bg_worker.py` listens on `bg_jobs` and sweeps at `BG_WORKER_SWEEP_INTERVAL`.
- Claims one `queued` job via `FOR UPDATE SKIP LOCKED`, sets `running`, calls `src.jobs.run_web_discovery_bg_enrich(job_id)`:
  - Loads `params.company_ids`, enriches each with `search_policy='require_existing'`, logs per‑company results, updates `processed/total`, sets `done` or `error`.
- Chat graph periodically announces completion in later turns via `_announce_completed_bg_jobs(state)`.


## 5) Process Flow (High‑Level)

1) Confirm ICP → Plan Top‑10 (SG‑locked discovery, profile scoring) → Persist preview + remainder to staging.
2) Enrich Top‑10 strictly → SSE progress.
3) Enqueue Next‑40 from staged remainder (once per session) → Worker runs in background.
4) Results are persisted (`companies`, `contacts`, `lead_emails`, `lead_scores`, `company_enrichment_runs`); chat announces completion.


## 6) Data Model Changes (Additions)

Companies table (SG fields + hygiene):
- `uen TEXT`, `uen_confidence NUMERIC`, `hq_city TEXT`, `sg_phone TEXT`, `sg_postcode TEXT`,
- `sg_markers TEXT[]`, `employee_bracket TEXT`, `locations_est INT`, `domain_hygiene BOOLEAN DEFAULT TRUE`,
- `sg_registered BOOLEAN DEFAULT FALSE` (gated only with high‑confidence UEN).

Evidence and scoring:
- `icp_evidence`: `signal_key='top10_preview'` for preview; additional signals for SG markers, triggers.
- `lead_scores`: store `score`, `bucket`, `rationale`; keep profile id in rationale or side metadata.

Staging (already present):
- `staging_global_companies`: `ai_metadata.preview=true|false`, `score`, `bucket`, `why`, `snippet`, `provenance`.


## 7) Configuration

YAML config (example):
```yaml
region: sg
profiles:
  sg_employer_buyers:
    include_markers: ["careers","jobs","people & culture","human resources","employee relations","industrial relations"]
    deny_host_suffix: ["gov.sg","edu.sg","mil","int"]
    weights: { employer_presence:20, sg_compliance_triggers:25, hr_ir_presence:20, hiring_intensity:10, hq_singapore:10, evidence_completeness:15 }
  sg_referral_partners:
    include_markers: ["hr consulting","recruitment","payroll","hris","eor","peo","corporate secretarial","relocation","immigration","bookkeeping"]
    deny_host_suffix: ["gov.sg","mil","int"]
    weights: { services_match:30, sg_presence:20, partner_fit:30, evidence_completeness:20 }
  sg_generic_leads:
    include_markers: ["about","services","contact","clients","hiring","new outlet","expansion","tender"]
    deny_host_suffix: ["gov.sg","edu.sg","mil","int"]
    weights: { sg_presence:30, org_signals:30, hiring_growth:20, evidence_completeness:20 }
deny:
  apex: ["w3.org","ifrs.org","ilo.org","oecd.org","deloitte.com","grandviewresearch.com","umbrex.com","10times.com","tradefairdates.com","interpack.com","pack-print.de","exhibitorsvoice.com","expotobi.com","cantonfair.net"]
  host_suffix: ["gov.sg","edu.sg","mil","int"]
  path_regex: "(?i)/(standards?|regulations?|policy|association|directory|glossary|wiki|expo|tradefair|event|conference|exhibition|exhibitors)/"
sg_markers: ["\\bSingapore\\b","\\+65","\\b\\d{6}\\b"]
```

Env flags (selected):
- `RUN_NOW_LIMIT`, `BG_NEXT_COUNT`, `BG_WORKER_MAX_CONCURRENCY`, `BG_WORKER_SWEEP_INTERVAL`
- `ENABLE_AGENT_DISCOVERY`, `ENABLE_BG_WORKER`, `ENABLE_APIFY_LINKEDIN`


## 8) Implementation Plan by Module

- `src/agents_icp.py`
  - Add `region_hint='sg'` and SG marker gating; prefer `.sg` in queries and filter `.com` unless SG markers present.
  - Integrate denylist/host‑suffix/path regex and hygiene check before scoring.
  - Add `lead_profile` param and profile‑aware scoring with breakdown/rationale; emit “why” chips.

- `src/enrichment.py`
  - Extend extraction to SG cues (HQ city, `+65`, 6‑digit postcode), hiring counts, profile markers, compliance tokens.
  - Persist compact summary and SG markers into `companies`/`icp_evidence`.

- `src/icp_pipeline.py`, `src/icp.py`, `src/acra_direct.py`
  - Gate `sg_registered=True` only with UEN+confidence; store `uen_confidence` and `acra_source`.

- `src/lead_scoring.py`
  - Optional: centralize weights; expose helpers for A/B/C buckets and max‑cap behavior on sparse evidence.

- `app/icp_endpoints.py`, `app/pre_sdr_graph.py`
  - Already persist Top‑10 preview and remainder; ensure profile id is carried (preview metadata) to guide Next‑40 runs consistently.

- `scripts/run_bg_worker.py`
  - Keep as is for Next‑40; ensure logs include per‑company results and totals.

- Migrations
  - Add columns to `companies` for SG fields and hygiene; add indexes as needed.


## 9) Testing & QA

Unit:
- Regexes for SG markers and deny paths; profile marker detection.
- Scoring by profile (weights, gating, bucket thresholds).
- Domain hygiene validator (reject suffix‑only, malformed hosts).
- ACRA gate: `sg_registered` flips only with UEN+confidence.

Integration:
- Mixed `.sg` and `.com` inputs → only `.com` with SG markers pass discovery.
- Expos/standards/directories are denied at discovery and normalization.
- Persisted preview and remainder present in `staging_global_companies`; Next‑40 enqueue selects from remainder.

E2E (staging):
- For each profile, seed with 10 known targets → expect ≥8 in Bucket A/B with 2+ “why” chips.


## 10) Rollout

Phased rollout (Dev → Staging → Prod):
- Dev: implement SG lock, denylist/hygiene, profile scoring, SG evidence, ACRA gate; unit green.
- Staging: 50‑domain sample per profile; validate acceptance criteria; tune weights.
- Prod: behind `icp_sg_profiles` feature flag for 1 week; monitor telemetry; GA after two consecutive green runs.


## 11) Risks & Mitigations

- Over‑filtering: add UI toggle to allow `.com` with SG markers and relax markers per profile.
- Sparse SME evidence: lean on ACRA/press/careers; cap max score on sparse evidence.
- Keyword drift: externalize lists; schedule weekly review; telemetry to signal drift.


## 12) Appendix — Key Helpers

```python
SG_MARKERS = [r"\bSingapore\b", r"\+65", r"\b\d{6}\b"]
DENY_PATH = re.compile(r"/(standards?|regulations?|policy|association|directory|glossary|wiki|expo|tradefair|event|conference|exhibition|exhibitors)/", re.I)

def is_singapore_page(text: str) -> bool:
    return any(re.search(p, text or "", re.I) for p in SG_MARKERS)

def is_valid_fqdn(host: str) -> bool:
    # Must contain at least one label before a known public suffix; no bare TLDs/suffix-only.
    return bool(re.match(r"^(?!-)[A-Za-z0-9-]{1,63}(?<!-)\.(?:[A-Za-z0-9-]+\.)*[A-Za-z]{2,}$", host or ""))
```

References
- Optimization spec: `lead_generation-main/devplan/featurePRD_optimise.md`
- Current flow code: `app/icp_endpoints.py`, `app/pre_sdr_graph.py`, `src/agents_icp.py`, `src/enrichment.py`, `src/jobs.py`, `scripts/run_bg_worker.py`

