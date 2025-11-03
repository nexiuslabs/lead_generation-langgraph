---
owner: Codex Agent – Pipeline Engineer
status: draft
last_reviewed: 2025-03-24
---

# Feature PRD Opt‑2 — Discovery Hygiene, Firmographics Recovery, and Scoring Guardrails

This plan translates PRDOpt2 into concrete, incremental changes across discovery, enrichment, and scoring. It emphasizes precision (valid, resolvable domains), completeness (industry and headcount captured), and honest prioritization (no “high” bucket without core firmographics).


## 1) Current System Snapshot

- Discovery (domains): `src/agents_icp.py`
  - Domain helpers: `_normalize_host`, `_apex_domain`, `_is_probable_domain`.
  - DuckDuckGo HTML parsing via `_ddg_search_domains` with site filter support; r.jina fallback.
  - Hygiene/denylist hooks from `src/config_profiles.py` (`is_denied_host`, `deny_path_regex`).
  - Planner ranks evidence for Top‑N; fills list with heuristic backfill when Jina fails.

- Enrichment (firmographics): `src/enrichment.py`
  - Deterministic crawl + LLM extract on merged corpus; homepage fallback parser for title/description.
  - LinkedIn via Apify where available; single‑attempt cap and vendor usage counters.
  - Persists to `companies`, `contacts`, `company_enrichment_runs`, `lead_emails`.

- Scoring: `src/lead_scoring.py`
  - Feature fetch from `companies` (employees_est, revenue_bucket, sg_registered, incorporation_year).
  - Probability + manual research bonus (ev_count) up to `MANUAL_RESEARCH_BONUS_MAX` (default 20).
  - Buckets: A/B/C → high/medium/low.

Gaps observed (from PRDOpt2):
- Malformed domains survive (`2f…` artifacts, pathy or percent‑encoded remnants) causing enrichment failures.
- Top‑10 fallback can include portals/directories (e.g., `gov.sg`, `w3.org`).
- Industry/Employees often `None`; yet research bonus can push such leads to high.

## Multi‑Agentic LLM + Tools Workflow

We standardize on a modular, multi‑agent workflow where every agent uses both LLMs and tools, collaborating via a shared state in LangGraph. Each agent must (a) invoke at least one tool to gather or transform evidence, and (b) invoke an LLM (often with structured outputs) to decide, summarize, or transform. Deterministic fallbacks keep progress moving when an LLM or vendor times out.

- Global state (subset)
  - `run_id`, `tenant_id`, `icp_profile`, `seeds`, `seed_hints`
  - `discovery_candidates[]`, `evidence[]`
  - `firmographics{ industry_code, industry_norm, employees_est }`, `contacts[]`
  - `scores[]`, `errors[]`, `metrics{}`

- Agents (LLM + Tools for all)
  - DiscoveryPlannerAgent (LLM + Tools):
    - Tools: Jina Reader to snapshot tenant website; DDG HTML fetch/parse (`web_search`).
    - LLM: Builds a single, strict industry‑only query from tenant website text (or primary industry) using a constrained prompt; outputs exactly one query string. The agent then calls DDG tool to retrieve candidates and emits `discovery_candidates[]` with tool‑derived drop reasons.
  - EvidenceCollectorAgent (LLM + Tools):
    - Tools: Jina Reader to fetch homepage snippets for candidates.
    - LLM: Structured extraction into micro‑ICP signals (industries, titles, triggers) and normalized evidence records in `evidence[]`.
  - ComplianceGuardAgent (LLM + Tools):
    - Tools: Config profile functions (`is_valid_fqdn`, `is_denied_host`, `deny_path_regex`, `is_sg_page`) and optional additional Jina reads on borderline cases.
    - LLM: Summarizes ambiguous candidates (short textual justification) and decides keep/drop when rules are inconclusive; generates a short “why dropped/kept” note attached to `guard_report[]`. Can request a re‑plan (loopback) with a one‑line LLM suggestion (e.g., “relax site:.sg” when `ENABLE_DDG_RELAX=true`).
  - EnrichmentAgent (LLM + Tools):
    - Tools: deterministic crawl, Tavily (if configured), Jina snapshots, JSON‑LD parser, Apify LinkedIn company/employees chains.
    - LLM: Extracts firmographics and signals from merged corpus; merges tool‑derived JSON‑LD/LinkedIn/headcount heuristics; produces consolidated `data{}`.
  - ScoringAgent (LLM + Tools):
    - Tools: DB feature fetch; optional sklearn baseline.
    - LLM: Generates concise rationales; optionally computes a profile‑aware score via structured output that is ensembled with the baseline (fallback to baseline if LLM unavailable). Post‑rules demote for missing firmographics.

- Collaboration and loopbacks (LangGraph)
  - Primary flow: `plan_discovery → collect_evidence → guard_compliance → enrich → score → persist`.
  - Guard loopback: If guard prunes all candidates, the guard asks the planner (LLM) for a minimal re‑plan (e.g., relax `site:` once if `ENABLE_DDG_RELAX=true`) and re‑runs discovery → evidence → guard.
  - Enrichment feedback: When enrichment recovers firmographics for only a small subset, it can request more candidates from guard or planner (bounded one‑shot retry) with a short LLM summary (“need more in-industry domains”).
  - Scoring feedback: For batches where a large fraction is demoted for missing firmographics, scoring posts a signal to enrichment to prioritize JSON‑LD and LinkedIn backfills in the next pass.

- Operational safeguards
  - Vendor caps/circuit breaker; per‑node timeouts; JSON‑schema prompts; deterministic regex and heuristic fallbacks.


## 2) Objectives & Non‑Goals

- Objectives
  - Persist only resolvable, normalized apex domains; dedupe URL‑encoded and variant forms.
  - Tighten “Top‑N” fallback so irrelevant directories/portals are filtered before persistence.
  - Recover firmographics (industry, employees) ≥90% on staging set via layered extraction.
  - Align scoring to demote leads missing core firmographics, regardless of research events.

- Non‑Goals
  - No new external data vendors; reuse Tavily/Jina/Apify/LLM.
  - No UI overhaul; document rules for maintainers and tests for regressions.


## 3) Changes — Discovery Hygiene

3.1 URL/Domain Normalization Hardening (agents)
- Add `_clean_possible_percent_encoded(url_or_text: str) -> str` that:
  - Attempts `unquote` once, then extracts `netloc` if an URL, else tokenizes on whitespace and `/`.
  - Strips spurious `"www."` and leading artifacts like `"2f"` when they directly prefix a valid label (common from `%2F`).
  - Normalizes to lowercase; pass through `_normalize_host` and collapse to `_apex_domain`.
- Apply in:
  - `_ddg_search_domains` before evaluating `_is_probable_domain`.
  - `discovery_planner` when building `uniq` and before merging `cand` into Top‑N.

3.2 Strict FQDN Validation and Dedupe (agents)
- Replace/augment `_is_probable_domain` with `config_profiles.is_valid_fqdn` for final acceptance.
- Dedupe across apex: treat `sub.example.com` and `example.com` as the same candidate, keep apex.
- Drop tokens that are actually file extensions or paths (already partially covered; expand list).

3.3 Denylist and Path Filters (agents)
- Always load cfg = `config_profiles.load_profiles()`; for each candidate domain:
  - Drop if `is_denied_host(domain, cfg)`.
  - For snippet‑based evidence, drop if `deny_path_regex(cfg)` matches prominent URLs (DDG anchors often include path slugs like `/directory`, `/expo`).
- Expand defaults (if not already present) via YAML to include portals/directories noted in PRD (keep code data‑driven).

3.4 Counters & Logging (agents)
- Track and log counts: `DOMAIN_HYGIENE`, `DENY_HOST`, `DENY_PATH`, `ACCEPTED` per discovery call.
- Emit first‑drop reason when candidates are excluded to aid tuning.


## 4) Changes — Enrichment Firmographics Recovery

4.1 Deterministic Industry Recovery
- Add `_infer_industry_from_corpus(text) -> dict`:
  - Keyword → SSIC title hints; pass tokens to `orchestrator._find_ssic_codes_by_terms` to resolve probable `industry_code` and `industry_norm`.
  - If multiple, select highest score, store `industry_norm` (lowercase title) and `industry_code`.
  - Persist as soft inference with `industry_confidence` float in `company_enrichment_runs` payload (no schema change; store under JSON where available, else in `icp_evidence`).

4.2 LinkedIn‑First Headcount Backfill
- When `apify_company_url_from_domain(domain)` yields a company URL, call a lightweight chain:
  - `apify_contacts_via_domain_chain` or `apify_contacts_via_chain` extended to also fetch company summary payload (industry/category and `employees_on_linkedin` bucket if exposed by the actor).
  - Map buckets to `employees_est` using mid‑point heuristics (e.g., 11–50 → 30, 51–200 → 120). Keep mapping centralized.
  - Respect `_apify_cap_ok(tenant_id)` and vendor circuit breaker.

4.3 Schema.org and Footer Heuristics
- Parse `application/ld+json` Organization blocks for `employee`, `numberOfEmployees`, `foundingDate`, `address` fields; map `address.addressCountry` and `address.addressLocality`.
- Footer/company overview pages: regex for “employees”, “team of”, “over X staff” to derive approximate headcount.

4.4 Completeness Guard and Retry
- If both `industry_*` and `employees_est` are missing after the first pass:
  - Attempt a targeted two‑page fallback: About + LinkedIn (if URL known) with shorter timeouts.
  - Mark run as `degraded_reasons += ['firmographics_missing_initial']` in `company_enrichment_runs`.

4.5 Persistence Notes
- Prefer writing firmographic fields into `companies` when available: `industry_code`, `industry_norm`, `employees_est`.
- Store intermediate confidence and raw source in `company_enrichment_runs` JSON payload.


## 5) Changes — Scoring Guardrails

5.1 Firmographics‑Required for High Bucket
- Introduce a demotion rule post‑score:
  - If `industry_code` is NULL OR `employees_est` is NULL, then:
    - Apply penalty: `final = max(0, final - MISSING_FIRMO_PENALTY)`; default `MISSING_FIRMO_PENALTY=30` via env.
    - Enforce `bucket != 'high'` regardless of numeric score after penalty.
- Make manual research bonus conditional:
  - Only apply `research_ev_count` bonus if at least one firmographic is present (industry OR employees); else cap bonus at 5.

5.2 Align A/B/C to Buckets
- Keep current mapping but ensure the “high” label maps only when firmographic rule passes.

5.3 Explainability
- Append rationale note when demoted: “demoted due to missing firmographics (industry/employees)”.


## 6) Config & Flags

- `ENABLE_STRICT_DOMAIN_HYGIENE` (default true)
- `DISCOVERY_ALLOW_PORTALS` (default false) — bypass denylist for debug.
- `MISSING_FIRMO_PENALTY` (default 30)
- `FIRMO_MIN_COMPLETENESS_FOR_BONUS` (default 1 field present)
- YAML at `config/sg_profiles.yaml` continues to control denylist/markers; extend `deny.apex` and `deny.host_suffix` as needed without code changes.
  
- Agentic toggles (existing; document and enforce)
  - `ENABLE_AGENT_DISCOVERY`: drive DiscoveryPlannerAgent; else heuristic fallback.
  - `ENRICH_AGENTIC`: enable EnrichmentAgent planner vs fixed graph.
  - `ENRICH_AGENTIC_MAX_STEPS`: cap agentic steps (e.g., 8–12).
  - `AGENT_MODEL_DISCOVERY`: planner model; default to `LANGCHAIN_MODEL`.


## 7) Testing Plan

7.1 Discovery Unit Tests (`tests/`)
- Normalize/acceptance:
  - Inputs: `https://%2F%2Ffinestservices.com.sg`, `2ffinestservices.com.sg`, `www.example.com/path?x=1` → `finestservices.com.sg`, `example.com`.
  - Rejects: `gov.sg`, `w3.org`, `directory.gov.sg`, `example.webp`, `pdfhost.com`.
- Dedupe to apex: `a.b.example.com` and `example.com` → one candidate.
- Site filter: preserve `.sg` requirements when industries imply SG profile.

7.2 Planner Integration Test
- Given an ICP industries list and synthetic DDG HTML page with mixed anchors:
  - Assert that denylisted hosts are excluded; relevant `.sg` companies remain.

7.3 Enrichment Extraction Tests
- Given merged corpus with:
  - SSIC‑mappable tokens → `_infer_industry_from_corpus` sets `industry_code/norm`.
  - Schema.org JSON‑LD snippet with `numberOfEmployees` → sets `employees_est`.
  - LinkedIn payload fixture → mapping fills `employees_est` and `industry_norm`.

7.4 Scoring Tests
- No firmographics + high research_ev_count → bucket is not `high` and penalty applied; rationale includes demotion note.
- One firmographic present + some research → bonus applied normally; bucket can be `high` if threshold reached.

Notes: Existing smoke tests avoid network; monkeypatch extract chain and Apify helpers with fixtures as done in `tests/test_enrichment_smoke.py`.


## 8) Implementation Sketch (File‑level)

- `src/agents_icp.py`
  - Add `_clean_possible_percent_encoded` and use it inside `_ddg_search_domains` and `discovery_planner` when building `uniq` and `cand`.
  - Replace final `_is_probable_domain` check with `config_profiles.is_valid_fqdn` when `ENABLE_STRICT_DOMAIN_HYGIENE`.
  - Apply `is_denied_host` and `deny_path_regex` during candidate filtering and Top‑N backfill; increment drop counters.

- `src/enrichment.py`
  - Add `_infer_industry_from_corpus` and call it in the fixed and agentic paths after `build_chunks`/`llm_extract` to backfill `industry_*` if missing.
  - Extend Apify chain result normalization to map `employees_on_linkedin`/`company_size` buckets to `employees_est`.
  - Parse JSON‑LD organization blocks when merging corpus and update firmographics.
  - Implement limited retry path with degraded reason tracking.

- `src/lead_scoring.py`
  - After computing `final`, apply penalty/demotion rules based on missing firmographics and conditional research bonus.
  - Append rationale string for demoted cases.

- `src/settings.py`
  - Add flags: `ENABLE_STRICT_DOMAIN_HYGIENE`, `DISCOVERY_ALLOW_PORTALS`, `MISSING_FIRMO_PENALTY`, `FIRMO_MIN_COMPLETENESS_FOR_BONUS` with sane defaults.

- `src/config_profiles.py`
  - Ensure defaults include deny apexes and `host_suffix` for portals/directories cited; keep path regex flexible.
  
- `src/agents_icp.py` (agents exposure)
  - Expose DiscoveryPlannerAgent and EvidenceCollectorAgent as LangGraph nodes (`plan_discovery`, `collect_evidence`) returning structured dicts for reuse by chat/API flows.

- `src/enrichment.py` (agentic tools)
  - Flesh out EnrichmentAgent tool registry with explicit tool functions and unify planner step names with tests.

- `src/orchestrator.py`
  - Provide a multi‑agent runner wiring `plan_discovery → collect_evidence → guard_compliance → enrich → score → persist` under flags; reuse existing `agent_discovery_domains` helper.


## 9) Observability & QA

- Metrics (via `src.obs`):
  - `discovery.domain_hygiene_drops`, `discovery.deny_host_drops`, `discovery.accepted`.
  - `enrichment.firmographics_recovered` (industry only, employees only, both).
  - `scoring.demotions_missing_firmo`.
- Logs include first‑drop reason per candidate and summary counts per run.
- Staging validation: compare before/after on a fixed seed list; target ≥90% non‑empty `industry_*` or `employees_est`.


## 10) Rollout Plan

- Phase 1 (Discovery hygiene): behind `ENABLE_STRICT_DOMAIN_HYGIENE=true`; ship tests; monitor drop rates.
- Phase 2 (Firmographics recovery): enable LinkedIn + schema.org + SSIC inference; monitor completeness and latency.
- Phase 3 (Scoring guardrails): enable penalty/demotion; review bucket distribution impacts with archived runs.
- Document rules and toggles in `project_documentation.md` and `Development_Plan/PRDOpt2.md` cross‑reference.


## 11) Risks & Mitigations

- Over‑filtering true prospects: keep `DISCOVERY_ALLOW_PORTALS=true` for quick bypass in QA; log candidates dropped with reasons.
- Vendor data gaps: SSIC inference and schema.org parsing provide non‑vendor fallbacks; retries capped.
- Latency increase: LinkedIn/company page pass only when firmographics missing; short timeouts; counters to cap calls.


## 12) Timeline (estimates)

- Discovery hygiene hardening: 1–2 days incl. unit tests.
- Firmographics recovery layers: 2–3 days incl. fixtures and integration tests.
- Scoring guardrails + rationale: 0.5–1 day.
- QA & tuning: 1–2 days with staging datasets.


## 13) Acceptance Checklist (maps to PRD)

- Deduped, normalized, resolvable domains only persisted (tests pass for `%2F` artifacts).
- Top‑N excludes directories/government portals; integration test green.
- ≥90% firmographic completeness on staging set; metrics recorded.
- Leads lacking firmographics cannot be `high` even with research bonus; tests green.
- Documentation updated and flags/toggles documented.
