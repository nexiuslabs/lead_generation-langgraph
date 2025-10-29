---
owner: Codex Agent – Pipeline Engineer
status: proposed
last_reviewed: 2025-10-26
source_prd: Development_Plan/Development/FeaturePRDOpt2.md
---

**Feature Dev Plan — PRD Opt‑2: Discovery Hygiene, Firmographics Recovery, Scoring Guardrails**

**Goals**
- Ensure discovered domains are normalized, resolvable apexes; remove URL‑encoded/“2f” artifacts and directories/portals.
- Recover firmographics (industry + employees) ≥90% on staging sample without new vendors.
- Enforce scoring guardrails so leads missing core firmographics cannot reach “high”.

**Scope**
- Discovery hygiene in `lead_generation-main/src/agents_icp.py` with config‑driven deny rules from `src/config_profiles.py`.
- Enrichment backfills in `lead_generation-main/src/enrichment.py` (deterministic + LinkedIn actor payloads + schema.org + footer heuristics).
- Scoring demotions in `lead_generation-main/src/lead_scoring.py` with rationale.
- Flags in `lead_generation-main/src/settings.py`; YAML in `config/` remains source of denylist/markers.

**Non‑Goals**
- No new data vendors; reuse existing Jina/Apify.
- No DB schema changes required; store intermediate confidences in existing JSON payloads.

**Architecture Touchpoints**
- Agents: `agents_icp.discovery_planner`, `_ddg_search_domains`, `plan_top10_with_reasons`.
- Enrichment: deterministic crawl/extract helpers; Apify LinkedIn chain.
- Profiles/deny rules: `src/config_profiles.py` (`is_valid_fqdn`, `is_denied_host`, `deny_path_regex`, `is_sg_page`).
- Scoring: post‑score demotions and bonus gating.

**Current Setup (from code)**
- Discovery flags (`lead_generation-main/src/settings.py`):
  - `ENABLE_DDG_DISCOVERY` on; `STRICT_DDG_ONLY` on; `STRICT_INDUSTRY_QUERY_ONLY` on; SG bias via `ICP_SG_PROFILES`; region hint via `DDG_KL`.
  - Timeouts and caps: `DDG_TIMEOUT_S` (min enforced 8s in agent), `DDG_MAX_CALLS`, per‑process cache TTL 300s (in `agents_icp`).
- Discovery implementation (`lead_generation-main/src/agents_icp.py`):
  - `_ddg_search_domains`: fetches 4 DDG HTML endpoints; paginates up to 15 pages (`DDG_MAX_PAGES`); parses anchors and DDG redirects (`/l/?uddg=`); enforces `site:` token when present; normalizes to apex via `_apex_domain`; strips `2f` artifacts; filters obvious search/CDN/wiki hosts; applies `_is_probable_domain`.
  - Fallback to r.jina.ai DDG HTML snapshot and regex extraction when direct endpoints fail; maintains a 5‑minute in‑process cache keyed by `(query,country)`.
  - `discovery_planner`: composes a single DDG query from tenant website‑derived terms or first industry; appends `site:.sg` when SG is inferred; logs `[plan] ddg-only query …`; calls `_ddg_search_domains`; applies `is_valid_fqdn` and `is_denied_host`; prioritizes `.sg`; excludes seed apexes; caps to 50; can optionally relax `site:` via `ENABLE_DDG_RELAX`.
  - Mini‑crawl (planning and Top‑10): uses r.jina.ai to snapshot homepages; gates by industry tokens and SG markers when `ICP_SG_PROFILES` is enabled.
- Profiles and deny rules (`lead_generation-main/src/config_profiles.py`):
  - Default config includes deny apex list (directories/associations/expo sites), deny `host_suffix` (e.g., `gov.sg`, `edu.sg`, `mil`, `int`), and `deny.path_regex` (directories, expo, events, etc.).
  - `is_valid_fqdn`, `is_denied_host`, `deny_path_regex()` available; `DENY_PATH` compiled for reuse.
- Enrichment (`lead_generation-main/src/enrichment.py`):
  - Deterministic onsite link discovery uses `deny_path_regex()` to avoid portal/directory slugs; merges Tavily/deterministic pages; LLM extracts fields (`industry_norm`, `employees_est`, `industry_code`, etc.) with chunking and fallbacks.
  - Apify LinkedIn company and employees chains exist; contacts path active; explicit mapping from LinkedIn employees buckets to `employees_est` is not centralized today.
  - Persists enriched fields into `companies` and keeps run payloads; also writes evidence and contacts.
- Scoring (`lead_generation-main/src/lead_scoring.py`):
  - Builds features from `companies`, applies logistic/heuristic scoring; adds manual research bonus (env `MANUAL_RESEARCH_BONUS_MAX`), maps to A/B/C and then high/medium/low; no demotion for missing firmographics.

**Required Changes (delta)**
- Discovery: add percent‑encoded cleaner; integrate `is_valid_fqdn` consistently; deny path filtering on DDG anchors/snippets; drop‑reason counters; keep strict single‑query and seed exclusion.
- Enrichment: implement `_infer_industry_from_corpus`; add JSON‑LD Organization and footer heuristics for employees; wire LinkedIn employees bucket→`employees_est`; add firmographics‑missing quick retry.
- Scoring: add `MISSING_FIRMO_PENALTY`/`FIRMO_MIN_COMPLETENESS_FOR_BONUS`; demote and add rationale when firmographics missing.
- Config/Docs: expand YAML denies; document flags and logs.

**Multi‑Agent Orchestration and Compliance Guard (explicit)**
- Agent mapping (PRD → code):
  - DiscoveryPlannerAgent → `src/agents_icp.discovery_planner` (single DDG query + hygiene) and `_ddg_search_domains`.
  - EvidenceCollectorAgent → `src/agents_icp.mini_crawl_worker` (r.jina snapshots) and `evidence_extractor` (micro‑ICP cues).
  - ComplianceGuardAgent → NEW node (see below) applied after evidence collection and before enrichment.
  - EnrichmentAgent → `src/enrichment.py` LangGraph (deterministic crawl, chunks, LLM extract, Apify tools).
  - ScoringAgent → `src/lead_scoring.py` LangGraph (features, score, rationale, persist).

- ComplianceGuardAgent (new):
  - Placement: between "collect_evidence" and "enrich" phases (in chat preview, after `mini_crawl_worker`/`evidence_extractor`; in full run, before `enrichment` graph entry).
  - Inputs: `state['discovery_candidates']`, optional snippets in `state['jina_snippets']`, micro‑ICP tokens, and loaded `cfg = load_profiles()`.
  - Checks:
    - Host hygiene: `is_valid_fqdn`, apex normalization, seed exclusion.
    - Deny host/suffix: `is_denied_host(d, cfg)`.
    - Path deny: apply `deny_path_regex(cfg)` to prominent URLs/snippet anchors if available.
    - SG markers (when `ICP_SG_PROFILES`): accept `.sg` or `is_sg_page(snippet, cfg)`.
  - Outputs: pruned `state['discovery_candidates']`, `state['guard_drops'] = {DOMAIN_HYGIENE, DENY_HOST, DENY_PATH, NO_SG_MARKERS}` and `state['guard_kept']` counts; log single summary line.
  - Loopback behavior: if all candidates drop, trigger a re‑plan condition:
    - If `ENABLE_DDG_RELAX=true` and original query had `site:` → re‑invoke planner with relaxed query (no `site:`) once; otherwise, surface a user‑visible note in chat explaining why zero candidates passed guard.
    - Keep strict single‑query constraint otherwise (no seed/competitor queries).

- Orchestration wiring:
  - Chat preview path (confirm stage): `discovery_planner → mini_crawl_worker → evidence_extractor → ComplianceGuardAgent → (optional) plan_top10_with_reasons`.
  - Full enrichment path: after candidates persisted, apply ComplianceGuardAgent just before enqueue/enrich to avoid wasting vendor calls on portals/invalid leads.
  - Logging: `[guard] kept=X drop_hygiene=Y drop_deny=Z drop_path=W drop_sg=Q`.

**Implementation Plan**

1) Discovery Hygiene
- Add `_clean_possible_percent_encoded(s: str) -> str` in `agents_icp.py`:
  - `unquote` once, extract `netloc` if URL, else split tokens, strip `www.`, remove leading `2f` sequences, lowercase.
- Apply cleaning pipeline in `_ddg_search_domains` and `discovery_planner` before `_is_probable_domain` and apex collapse:
  - Clean → `_normalize_host` → `_apex_domain` → `is_valid_fqdn` → site filter → deny host/path.
- Apex dedupe: prefer apex when both `sub.example.com` and `example.com` appear.
- Deny rules: load cfg and drop on `is_denied_host`; for DDG anchors/snippets, drop when `deny_path_regex` matches.
- Counters + logs per call: `kept`, `DOMAIN_HYGIENE`, `DENY_HOST`, `DENY_PATH`. Log first drop reason per host for tuning.
- Prioritize `.sg` domains in final list; keep cache and pagination (1–15 pages) intact.

2) Enrichment Firmographics Recovery
- Add `_infer_industry_from_corpus(text) -> {industry_code, industry_norm, confidence}`:
  - Tokenize corpus; map via existing SSIC/title resolver (orchestrator helper); store best match with confidence.
- LinkedIn‑first backfill:
  - Extend Apify chain to collect company summary when available; map `employees_on_linkedin` bucket → `employees_est` using mid‑point table.
  - Respect vendor caps/circuit breaker; one attempt per company.
- Schema.org + footer heuristics:
  - Parse JSON‑LD Organization for `numberOfEmployees`, `employee`, `address`; fallback regex on pages for “team of”, “over X staff”.
- Completeness guard + retry:
  - If both industry and employees missing after primary pass, attempt two quick reads (About + LinkedIn) with short timeouts; mark run `degraded_reasons += ['firmographics_missing_initial']`.
- Persistence: write `industry_code`, `industry_norm`, `employees_est` to `companies` if available; store confidences and sources in `company_enrichment_runs` JSON.

3) Scoring Guardrails
- Add envs to `settings.py`:
  - `MISSING_FIRMO_PENALTY` (default 30)
  - `FIRMO_MIN_COMPLETENESS_FOR_BONUS` (default 1)
- In `lead_scoring.py` post‑processing:
  - If `industry_code` is NULL OR `employees_est` is NULL → apply penalty and ensure bucket != high.
  - Only apply manual research bonus when at least one firmographic present; else cap at 5.
  - Append rationale: “demoted due to missing firmographics (industry/employees)”.

3b) Ensure LLM + Tools usage in every agent
- Discovery: enable constrained LLM query composer (single strict query from tenant website text or primary industry) before calling DDG tool. Keep strict seed‑free behavior.
- Evidence: keep Jina tool + LLM structured extraction.
- Guard: add optional LLM tie‑breaker to decide keep/drop for ambiguous cases and generate short “why” lines.
- Enrichment: keep LLM extraction + deterministic/Apify/JSON‑LD tools; reconcile tool and LLM outputs.
- Scoring: keep LLM rationales; optionally add LLM structured scoring ensembling with baseline (feature‑flagged).

4) Config, Flags, Docs
- `settings.py`: add `ENABLE_STRICT_DOMAIN_HYGIENE` (default true), `DISCOVERY_ALLOW_PORTALS` (default false).
- YAML: extend deny apex/suffix lists (directories/portals) in `config/sg_profiles.yaml`.
- Docs: update `docs/icp_to_enrichment_flow.md` and `project_documentation.md` with hygiene/guardrails; add runbook notes and envs.

5) Testing
- Unit tests (new `tests/` set):
  - Domain cleaning/normalization: inputs like `https://%2F%2Ffinestservices.com.sg`, `2ffinestservices.com.sg` → `finestservices.com.sg`.
  - Rejects: `gov.sg`, `w3.org`, `directory.gov.sg`, `example.webp`.
  - Apex dedupe and site filter adherence.
- Planner integration fixture: synthetic DDG HTML with mixed anchors; assert drops/keeps per deny/counters.
- Enrichment fixtures: JSON‑LD example, LinkedIn payload sample, corpus with SSIC tokens; assert recovered firmographics.
- Scoring unit: assert demotion and rationale when firmographics missing.
 - Agents LLM+Tools checks: assert each agent invokes at least one LLM and one tool call (via mock counters) in a happy‑path run; assert guard LLM path only fires on ambiguous cases and is bounded.

6) Observability & Metrics
- Log keys: `[plan] host_filter kept=.. drop_hygiene=.. drop_deny=.. drop_path=..`, `[enrich] firmo_recovery`, `[score] demotion_missing_firmo`.
- Optional: emit counters to `run_event_logs` with stage tags `discovery`, `enrichment`, `scoring` for sampled runs.

7) Rollout
- Phase 1 (dev): flags on, sample 50 companies; validate ≥10–50 discovery candidates, cleanliness, counters.
- Phase 2 (staging): enable firmo recovery + scoring demotions; compare bucket distribution vs baseline.
- Phase 3 (prod): enable `ENABLE_STRICT_DOMAIN_HYGIENE=true` and penalties; monitor adverse drops; tune YAML.
- Backout: flip flags off; revert to prior scoring without penalties and relaxed hygiene.

**Milestones**
- M1: Discovery hygiene code + tests pass; counters visible in logs.
- M2: Firmographics recovery integrated with enrichment; sample accuracy ≥90% on staging set.
- M3: Scoring guardrails shipped; no lead without firmographics lands in high.
- M4: Documentation and runbooks updated; rollout complete.

**Risks & Mitigations**
- Over‑filtering valid domains: Start with logging‑only denylist expansion; ship with `DISCOVERY_ALLOW_PORTALS=false`, toggleable.
- Latency increase: Keep per‑node timeouts; cap LinkedIn attempts; reuse Jina fallback snippets where needed.
- False firmographics: Use confidence thresholds; prefer deterministic sources (JSON‑LD/LinkedIn) over LLM inference.

**Acceptance Criteria**
- Discovery emits 10–50 clean candidates per run with `.sg` prioritization; no `2f..` artifacts present.
- ≥90% of enriched companies have `industry_norm` and `employees_est` populated on staging set.
- Leads missing industry or employees never appear as “high”; rationales include demotion reason.
 - Every agent uses both an LLM and at least one tool during typical runs; collaboration loopbacks operate within one bounded retry and are observable in logs.
