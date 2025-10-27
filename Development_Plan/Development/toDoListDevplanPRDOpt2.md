---
owner: Codex Agent – Pipeline Engineer
status: active
last_reviewed: 2025-10-26
source_prd: Development_Plan/Development/FeaturePRDOpt2.md
---

**To‑Do — PRD Opt‑2 Implementation Tracker**

Discovery Hygiene (agents_icp)
- [x] Add `_clean_possible_percent_encoded` utility in `lead_generation-main/src/agents_icp.py`.
- [x] Apply cleaning in `_ddg_search_domains` and `discovery_planner` before validation and apex collapse.
- [x] Enforce `is_valid_fqdn` (from `src/config_profiles.py`) in final acceptance.
- [x] Apex dedupe: prefer apex when subdomain and apex both found.
- [x] Denylist integration: drop candidates via `is_denied_host` with counters.
- [ ] Path regex filter: drop directory/portal slugs via `deny_path_regex` with counters.
- [ ] Log counters per call: kept, DOMAIN_HYGIENE, DENY_HOST, DENY_PATH (first drop reason per host).
- [x] Prioritize `.sg` domains in final list; keep pagination (1–15 pages) and cache.

Enrichment Firmographics (enrichment.py)
- [x] Implement `_infer_industry_from_corpus` (keyword→SSIC mapping) with confidence.
- [ ] Extend Apify LinkedIn chain to pull company summary (industry/category, employees bucket).
- [ ] Map LinkedIn employees bucket → `employees_est` (centralized mapping).
- [x] Parse JSON‑LD Organization for `numberOfEmployees`, `employee`, `address`.
- [x] Footer heuristics: regex for headcount phrases; backfill `employees_est` when confident.
- [ ] Completeness guard: trigger targeted retry (About + LinkedIn) when both firmographics missing.
- [ ] Persist firmographics to `companies`; store confidence/source in `company_enrichment_runs` JSON.

Scoring Guardrails (lead_scoring.py)
- [x] Add envs in `src/settings.py`: `MISSING_FIRMO_PENALTY`, `FIRMO_MIN_COMPLETENESS_FOR_BONUS`.
- [x] Demote when industry or employees missing (apply penalty and block high bucket).
- [x] Gate manual research bonus: require ≥1 firmographic; else cap at 5.
- [x] Append demotion rationale to `lead_scores.rationale`.

Config & Flags
- [x] Add `ENABLE_STRICT_DOMAIN_HYGIENE` (default true) and `DISCOVERY_ALLOW_PORTALS` (default false) to `src/settings.py`.
- [ ] Extend YAML denylist (`config/sg_profiles.yaml`) with portals/directories; keep code purely data‑driven.
- [ ] Update docs with envs and behavior: `docs/icp_to_enrichment_flow.md`, `project_documentation.md`.

Testing
- [ ] Unit: domain cleaning/normalization incl. `2f` and percent‑encoding cases.
- [ ] Unit: apex dedupe; reject invalid TLDs or file extensions; site filter honored.
- [ ] Integration: planner parses synthetic DDG HTML, applies deny/path filters, logs counters.
- [ ] Enrichment fixtures: JSON‑LD, LinkedIn payload, SSIC tokens; assert firmographic recovery.
- [ ] Scoring unit: penalty and rationale when missing firmographics; bonus gating.
 - [ ] Agents LLM+Tools checks: mock LLM/tool counters and assert each agent makes ≥1 LLM and ≥1 tool call; guard’s LLM path fires only on ambiguous cases and re‑plan is bounded to one retry.

Observability
- [x] Ensure logs for discovery counters: `[plan] host_filter kept=.. drop_hygiene=.. drop_deny=.. drop_path=..`.
- [ ] Optional: emit `run_event_logs` counters for discovery/enrichment/scoring (sampled).

Rollout
- [ ] Phase 1: Enable flags in dev; validate 10–50 clean candidates and counter logs on sample runs.
- [ ] Phase 2: Stage firmographics + scoring guardrails; compare bucket distribution vs baseline.
- [ ] Phase 3: Prod enablement; monitor drops and adjust YAML denylist.
- [ ] Backout plan documented; flags allow immediate revert.

Acceptance Checks
- [x] Discovery output free of `2f…` artifacts and directories/portals; `.sg` prioritized.
- [ ] ≥90% firmographics (industry + employees) populated on staging sample.
- [x] No lead without firmographics reaches “high”; rationales include demotion note.
 - [ ] Every agent uses both an LLM and at least one tool in a typical run; guard/planner loopback executed at most once and visible in logs.

Agents LLM + Tools (implementation tasks)
- [x] DiscoveryPlannerAgent: enable constrained LLM query composer (single strict, industry‑only query from tenant website text or primary industry) before running DDG tool; log `planner.llm=1`, `planner.tools.ddg>=1`.
- [x] EvidenceCollectorAgent: ensure Jina snapshot is called and structured LLM extraction runs; log `evidence.tools.jina>=1`, `evidence.llm=1`.
- [ ] ComplianceGuardAgent: add optional LLM tie‑breaker for ambiguous candidates with short “why” lines; integrate one‑shot re‑plan (when `ENABLE_DDG_RELAX=true`); log `guard.llm_tiebreak<=N`, `guard.replan<=1`.
- [x] EnrichmentAgent: confirm LLM extraction runs and tool calls (deterministic crawl/Jina/JSON‑LD/Apify) occur when available; log `enrich.llm>=1`, `enrich.tools.*` counters.
- [ ] ScoringAgent: keep LLM rationales and (optional) LLM structured scoring ensemble behind a flag; log `score.llm.rationale=1`, `score.llm.structured<=1`.
