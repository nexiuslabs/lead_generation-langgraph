**Feature Dev Plan 17 — Implementation Plan (ICP Intake & Pattern Mining)**

- **Purpose:** Translate featurePRD17 into concrete database, backend, agents, jobs, and UI work. Reuse the existing enrichment pipeline tools and introduce LangChain/LangGraph agents for crawl → extract → analyze.
- **References:** Development/featurePRD17.md (PRD), existing enrichment modules (`src/enrichment.py`, `src/icp.py`, `app/pre_sdr_graph.py`).

**Scope & Goals**
- **Deliver:**
  - Fast-start wizard + advanced intake (A–H) stored server-side.
  - Seeds→companies→ACRA mapping with pg_trgm fuzzy fallback.
  - Deterministic crawls of user + seed sites; evidence persisted.
  - Pattern materialization (`icp_patterns`) and suggestion generation.
  - UI flow: suggestions with evidence; adopt/refine; write to `icp_rules`.
  - Finder flow updates: show SSIC+industry names, ACRA totals and samples; accept micro‑ICP; enrich 10 now and queue remainder nightly.
- **Non-goals:** Outreach/sequencing; scraping personal emails; replacing `icp_rules` — we derive/augment it.

**Architecture Overview**
- **Components:**
  - Backend API: ICP intake endpoints and suggestions service.
  - DB & Migrations: new tables, MV, indexes, RLS.
  - Jobs & Scheduler: queue tasks (seeds mapping, crawl, pattern mining), nightly refresh.
  - Agents: LangChain/LangGraph based crawlers/extractors using existing fetchers.
  - UI: wizard + suggestions + evidence drawer.
  - **Data Flow:**
  - Intake → persist responses + seeds → map seeds to companies/ACRA → crawl evidence → materialize `icp_patterns` → generate suggestions → user adopts → write `icp_rules`.

**Mermaid — Pipeline**
```mermaid
flowchart LR
  A[Intake (Fast-Start + A–H)] --> B[Persist responses + seeds]
  B --> C[Seeds→Companies domain match]
  C --> D[Fuzzy match to ACRA (pg_trgm)]
  D --> E[Agents crawl user + seed sites]
  E --> F[Extract signals → icp_evidence]
  F --> G[Materialize icp_patterns]
  G --> H[Generate micro-ICP suggestions]
  H --> I[UI: adopt/refine]
  I --> J[Write accepted → icp_rules]
```

**Auth & Tenancy Alignment**
- Reuse SSO and tenancy model from existing app. Endpoints must accept `Authorization: Bearer <JWT>` and use tenant context.
- For browser calls from the chat UI, support `X-Tenant-ID` header (already adopted in Feature 18) to avoid tenant_id-less tokens.
- FE calls should use `useAuthFetch` with proxy support and include `X-Tenant-ID`.

**Database Changes (Migrations)**
- **001_icp_intake.sql**
  - `icp_intake_responses`
    - `id bigserial primary key`
    - `tenant_id uuid not null`
    - `submitted_by uuid not null`
    - `submitted_at timestamptz not null default now()`
    - `answers_jsonb jsonb not null`  // stores fast-start + A–H
    - Indexes: `idx_icp_intake_tenant_time (tenant_id, submitted_at desc)`; GIN on `answers_jsonb`
    - RLS: tenant isolation; roles: viewer read-only, ops/admin write
  - `customer_seeds`
    - `id bigserial primary key`
    - `tenant_id uuid not null`
    - `seed_name text not null`
    - `domain text`  // normalized lower no scheme/www
    - `created_at timestamptz not null default now()`
    - Indexes: `(tenant_id, lower(seed_name))`, `(tenant_id, domain)`
    - RLS: tenant isolation
- **002_icp_evidence.sql**
  - `icp_evidence`
    - `id bigserial primary key`
    - `tenant_id uuid not null`
    - `company_id uuid not null references companies(company_id)`
    - `signal_key text not null`  // e.g., industry_label, integration, headcount_hint, buyer_title
    - `value jsonb not null`       // raw or normalized value + source url + confidence
    - `source text not null`       // user_site | seed_site | acra | crawl
    - `observed_at timestamptz not null default now()`
    - Indexes: `(tenant_id, company_id)`, `(tenant_id, signal_key)`, GIN on `value`
    - RLS: tenant isolation
- **003_icp_patterns_mv.sql**
  - Materialized view `icp_patterns` (tenant-scoped aggregates):
    - Top SSICs per tenant, common integrations, median headcount, frequent buyer titles, recurring themes.
    - Example (sketch):
```
create materialized view if not exists icp_patterns as
with e as (
  select tenant_id, company_id, signal_key, value
  from icp_evidence
)
select tenant_id,
  jsonb_object_agg('top_ssics', top_ssics) as aggregates
from (
  select tenant_id,
    (select jsonb_agg(x order by cnt desc)
     from (
       select (value->>'ssic') as code, count(*) cnt
       from e where signal_key = 'ssic'
       group by 1
     ) x) as top_ssics
  from e group by tenant_id
) t
group by tenant_id;
```
  - Index: `(tenant_id)`
  - Refresh policy: nightly and on-demand after evidence jobs
- **004_supporting.sql**
  - Enable `pg_trgm` if not present; GIN/GIST indexes for fuzzy name matching on `staging_acra_companies.entity_name`
  - Optional: `companies.primary_domain` index if missing
  - RLS policies for all new objects

**Seeds→Companies→ACRA Mapping SQL (reference)**
- As in PRD: domain match to `companies.primary_domain`, fuzzy join against `staging_acra_companies` with `similarity(...) > 0.35`. Prefer exact domain matches.

**Backend APIs**
- Request/Response contracts (v1)
  - POST `/icp/intake` — Body: `{ answers: json, seeds: [{ seed_name, domain }...] }` → `{ status, response_id, queued_job_id }`
  - GET `/icp/suggestions` — Query: `tenant_id` (server-resolved by token when absent) → `{ items: SuggestionCard[] }`
  - POST `/icp/accept` — Body: `{ tenant_id?, suggestion_id? | suggestion_payload }` → `{ ok: true }`
  - GET `/icp/patterns` (optional ops) — `{ patterns: {...} }`
- `POST /icp/intake`
  - Auth: SSO; roles ops/admin write; viewer forbidden
  - Body: `answers_jsonb` + `customer_seeds[]`
  - Behavior: upsert response row and bulk-insert seeds in one transaction; enqueue `icp_intake_process` job
  - Returns: `{status, response_id, queued_job_id}`
- `GET /icp/suggestions`
  - Auth: all roles (viewer read-only)
  - Query: `tenant_id`
  - Returns: 3–5 suggestions with evidence counts and rationale summaries
- `POST /icp/accept`
  - Auth: ops/admin
  - Body: `{tenant_id, suggestion_id | suggestion_payload}`
  - Behavior: normalize and write to `icp_rules` (per-tenant); emit audit log
- `GET /icp/patterns` (optional debug)
  - Auth: ops/admin
  - Returns: materialized aggregates for inspection

**Agents & Jobs (LangChain/LangGraph)**
- **Job Orchestration:**
  - Queue: `icp_intake_process` → fan-out to:
    - `map_seeds_to_companies` (domain + fuzzy ACRA)
    - `crawl_user_site_evidence`
    - `crawl_seed_sites_evidence`
  - Follow-ups: `refresh_icp_patterns`, `generate_suggestions`
- **Tools (reuse + new):**
  - Reuse: deterministic fetcher from enrichment (`src/enrichment.py`), domain finder with blacklist, caching layer, HTML→text extractor, robots.txt respect.
  - New (optional): Playwright-based fetch for JS-heavy pages; sitemap parser; simple rate limiter; `trafilatura`/`readability-lxml` content cleaner.
- **LLM Config:**
  - LangChain Runnable chain with an extraction prompt per page-type: Industries, Customers/Case Studies, Integrations, Pricing, Careers, About.
  - Parsers: Pydantic output schemas for `EvidenceItem` (key, value, url, confidence, snippet).
  - Token/Cost control: chunk pages, cap pages per domain (reuse existing caps), early-stop on saturation.
- **Agent Graph (LangGraph):**
  - Nodes: `SeedMapper` → `UserSiteCrawler` → `SeedSitesCrawler` → `EvidenceConsolidator` → `PatternMiner` → `SuggestionGenerator`.
  - Memory: per-tenant ephemeral context, deduping extracted signals.
  - Output: writes to `icp_evidence`; triggers MV refresh; emits suggestions to cache.

**Codebase Changes (by path)**
- `src/icp.py`
  - Add: `save_icp_intake(answers, seeds)`, `generate_suggestions(tenant_id)`
  - Add: SQL helpers for mapping (domain/fuzzy), normalization utilities
  - Add: `refresh_icp_patterns()` function (runs `refresh materialized view concurrently icp_patterns`)
  - Add: `_find_ssic_codes_by_terms`, `_select_acra_by_ssic_codes`, `_count_acra_by_ssic_codes` for accurate ACRA totals and samples by SSIC.
- `src/enrichment.py`
  - Expose content fetcher as a tool usable by agents (respect blacklist, caching, robots)
  - Add page-type detectors (industries/customers/integrations/pricing/careers/about)
- `app/main.py` (or API service)
  - Add endpoints: `/icp/intake`, `/icp/suggestions`, `/icp/accept`, `/icp/patterns`
  - Wire RBAC (viewer vs ops/admin)
- `app/pre_sdr_graph.py`
  - Add an entry node to kick off `icp_intake_process` for tenants; progress chips via run_event_logs
  - Finder routing: remove explicit “Industry” prompt (infer from evidence), show early micro‑ICPs with SSIC and industry names, display ACRA totals using `_count_acra_by_ssic_codes` and samples via `_select_acra_by_ssic_codes`.
  - Accept + Enrich: support `accept micro-icp N`; proceed when user types “run enrichment”. Enrich up to 10 now (`CHAT_ENRICH_LIMIT`/`RUN_NOW_LIMIT`), enqueue remainder for nightly.
  - Safety: add boot‑guard to prevent auto‑run on restart; router only advances on explicit human messages; avoid recursion/loops by centralized router.
- `jobs/icp_tasks.py` (new)
  - Task functions for mapping, crawl, evidence store, MV refresh, suggestion gen
- `schemas/icp.py` (new)
  - Pydantic models: IntakePayload, Seed, EvidenceItem, SuggestionCard
- `config/settings.py`
  - Flags: `ENABLE_ICP_INTAKE`, `ICP_WIZARD_FAST_START_ONLY`; crawl caps; LLM model names
  - Batch: `CHAT_ENRICH_LIMIT`/`RUN_NOW_LIMIT` (immediate enrichment cap, default 10)

**Ops & Runbooks**
- Add operator steps: rerun mapping, inspect patterns, troubleshoot crawl blocks.
- Nightly: ensure `icp_patterns` refresh is in scheduler and document `scripts/refresh_icp_patterns.py`.
- Alerts: warn when seeds→ACRA success < 0.80 or mapping > 300s.

**Security & Compliance**
- Enforce RLS on all new tables; tenant isolation on reads/writes.
- Respect robots.txt; do not collect personal emails; PII masking in viewer UI.
- Audit logs on accept; redact snippets if necessary for viewer role.

**Observability**
- Metrics: intake completion rate, seeds mapped %, evidence items per company, patterns count, suggestions per tenant, job durations/errors.
- Logs: stages `icp_intake`, `mapping`, `crawl`, `pattern_mining`, `suggestions`; include decisions like domain vs fuzzy path.

**Agent & UI Integration Notes**
- Minimal intake, evidence‑first: Collect only website URL and seeds (best customers + lost/churned). Do not prompt for industry/employee band; infer from website + seeds and ACRA mappings. Accept optional geos/integrations/ACV/cycle/price floor/champions/triggers when provided.
- Conversational path: Add a light intake node in `pre_sdr_graph` to capture website + seeds quickly; parse/normalize; persist on confirm. Use resolver confirmations only when a seed match is ambiguous.
- Persistence & jobs: On `confirm`, call `save_icp_intake(...)` then fan‑out mapping + crawl + patterns refresh (non‑blocking). Stream progress chips.
- Frontend: Reuse proxy/auth infra (Feature 18). Provide a compact intake form as an alternative to chat for website + seeds, and a small resolver modal when needed.

**Status Sync — Finder Flow (Implemented)**
- Industry prompt removed; industries inferred from evidence + ACRA.
- Early suggestions show SSIC code + industry name; ACRA totals counted via `_count_acra_by_ssic_codes`, with sample rows via `_select_acra_by_ssic_codes`.
- Accept + Run pattern supported in chat: `accept micro-icp N` then `run enrichment`.
- Enrichment: immediate batch size capped by `CHAT_ENRICH_LIMIT`/`RUN_NOW_LIMIT` (default 10); remainder queued as `background_jobs(staging_upsert)` for nightly.
- Boot‑guard in router prevents auto‑run after restart; last‑speaker check ensures only explicit user messages advance.
- Evidence NOT NULL guard: `map_seeds_to_evidence` and `store_intake_evidence` ensure `company_id` before inserting into `icp_evidence`.

**Updated Question Set & Setup (aligned to icp_finder_enchancement.md)**
- Fast‑Start (minimal)
  - Website URL (required)
  - Best customers (5–15; company + website)
  - Lost/churned (≈3; company + website + 1‑line reason)
  - Optional: geos/languages/TZ, must‑have integrations, ACV & cycle, price floor, champion titles, 3 win triggers
- Resolver Confirmations (only when needed)
  - Display detected ICP keys for a seed (SSIC, size band, geo, stack, champion titles) and offer: ✅ Confirm, 🔄 Pick another match, ✏️ Edit details
- Advanced Sections (A–H) (optional, augments evidence)
  - A) Your Business; B) Seed Customers; C) Anti‑ICP; D) Firmographics; E) Technographics; F) Buying Motion; G) Triggers; H) Outcomes & Proof

Implementation notes
- Store all answers in `icp_intake_responses.answers_jsonb`; keep seeds in `customer_seeds`.
- Normalize domains from URLs; dedupe titles/tags; validate required counts (e.g., ≥5 seeds for high confidence).
- Inference replaces prompts for industry/employee band; ask only via resolver when ambiguous or conflicting.

Setup (UI + Backend)
- UI: Add a compact intake form for website + seeds + lost/churned; optional fields collapsed by default. Add a resolver modal component for ambiguous matches.
- Backend: Implement `/icp/intake`, `/icp/suggestions`, `/icp/accept` as specified; wire `save_icp_intake(...)`, mapping, crawl, `refresh_icp_patterns`, suggestion generation.
- Flags: `ENABLE_ICP_INTAKE=true`; optionally `ICP_WIZARD_FAST_START_ONLY=true` to hide A–H and rely entirely on inference unless user expands.

Implementation notes
- Store all answers in `icp_intake_responses.answers_jsonb` keyed by section/question to allow incremental extension.
- Validate: URL formats; required counts (e.g., ≥5 seeds for high‑confidence suggestions); enums for bands.
- Normalize: domains from URLs; title casing; tag deduplication.
 - Conversational parsing: Track last asked focus (`icp_last_focus`) and parse inputs for website/seeds/lost/integrations/acv/cycle/price_floor/champions/events; fall back to LLM extraction for generic ICP slots.

**Suggestions & Targeting (enhanced per PRD examples)**
- Micro‑ICP cards include:
  - SSIC cluster (e.g., 62012/62019), headcount band, key integrations, champion titles
  - Evidence count summary and rationale
- Negative ICP list:
  - 3+ red‑flag themes derived from Anti‑ICP input and churn/loss reasons
- Targeting pack helper:
  - ssic_filters, technographic_filters, short segment pitch
- Data sources:
  - Combine `icp_evidence` (ACRA SSIC + crawled signals) with intake answers (bands/integrations/champions) when evidence is sparse

**Acceptance Mapping (from PRD examples)**
- With ≥5 seeds, system proposes ≥3 micro‑ICPs with evidence counts
- Nightly refresh materializes patterns used by suggestions; suggestions cite evidence items
- Negative ICP presents ≥3 themes; pack builder outputs SSIC + technographic filters and pitch

**User Flows & Journeys**
- **Flow States:** `intake_saved → mapping → crawl → pattern_mining → suggestions_ready`.
- **Journey 1 (Happy path):**
  - Maya (ops) opens ICP wizard, completes fast-start (10) and submits.
  - API saves intake + seeds; jobs queue. Status chip shows `mapping`.
  - 4/5 seeds resolve to ACRA; crawlers extract industries, integrations, careers from all seeds + user site.
  - MV refresh runs; SuggestionGenerator prepares 4 cards. UI shows `suggestions_ready`.
  - Maya opens a card, reviews evidence drawer, clicks Adopt. Backend writes normalized payload to `icp_rules`.
- **Journey 2 (Low data):**
  - Only 3 seeds provided; UI warns lower confidence. 2 resolve; crawl returns few pages.
  - Suggestions still generated (2 cards) with “reduced evidence” badge.
- **Journey 3 (Crawl blocked):**
  - Several seed sites block bots; agents fallback to ACRA + public metadata; cards show reduced evidence; UI recommends adding more seeds.
- **Journey 4 (Refine & Re-run):**
  - User adds 4 more seeds and excludes a red-flag SSIC from Anti‑ICP. Intake updated; lightweight re-materialization produces new cards; comparison history available.

**Validation & Testing**
- Unit: domain normalization; fuzzy matcher threshold behavior; evidence parsers; suggestion generator scoring.
- Integration: `/icp/intake` transactionality; jobs fan-out; evidence persistence; MV refresh.
- E2E: simulate intake→suggestions via fixtures (seed pages) and assert cards + evidence counts.
- Performance: crawl caps respected; job concurrency safe; MV refresh times.

**Rollout Plan**
- Phase 1: DB migrations + endpoints + mapping + minimal crawl → suggestions.
- Phase 2: Richer extraction, observability, UI polish, evidence drawer.
- Phase 3: Tenant knobs; docs/runbooks; acceptance tests.
- Feature flags: `ENABLE_ICP_INTAKE`, optional `ICP_WIZARD_FAST_START_ONLY`.
- Backward compatibility: No changes to existing pipelines; suggestions only used when tenant adopts.

**Step-by-Step Process Flow (Implementation)**
- 1) Intake persistence:
  - Add `POST /icp/intake` to accept `{ answers, seeds }` (chat-driven capture):
    - answers: minimal chat fields (website, best customers, lost/churned, optional details)
    - seeds: normalized array `{ seed_name, domain }`
  - Persist to `icp_intake_responses` and `customer_seeds` in one transaction; return `response_id` and enqueue job `icp_intake_process`.
- 2) Mapping job:
  - `map_seeds_to_companies`: domain exact match to `companies.primary_domain`; fuzzy fallback to `staging_acra_companies.entity_name` (pg_trgm, threshold ~0.35); attach UEN/SSIC.
- 3) Crawl + evidence:
  - Reuse `src/enrichment.py` fetch/crawl utilities; extract page-type signals (industries, customers, integrations, pricing hints, careers, partners) for user site + seeds; write `icp_evidence`.
- 4) Patterns + suggestions:
  - `refresh_icp_patterns` (MV refresh), then `generate_suggestions(tenant_id)` to compute micro‑ICPs and Negative ICP from evidence + Anti‑ICP inputs.
  - Cache suggestions or serve on demand via `GET /icp/suggestions`.
- 5) Adopt:
  - `POST /icp/accept` writes normalized payload to `icp_rules` for tenant; triggers optional small enrichment batch using existing pipeline.

Code changes (by file)
- `app/main.py`:
  - Add routes `/icp/intake`, `/icp/suggestions`, `/icp/accept` and optional `/icp/patterns` (ops).
  - In `normalize_input`, keep industry inference for legacy threads but prefer ICP Finder path when `ENABLE_ICP_INTAKE=true` and conversation mentions ICP.
- `app/pre_sdr_graph.py`:
  - Add `icp_finder_intake_node` to collect website + seeds conversationally; post-confirm, call `/icp/intake`; display progress chips; render suggestion cards when available.
  - Suppress old “Ask Industry first” branch when Finder is enabled; route to resolver when matching is ambiguous.
- `src/icp.py`:
  - Implement `save_icp_intake`, `map_seeds_to_companies`, `refresh_icp_patterns`, `generate_suggestions` helpers.
  - Add normalization utils for domains/titles and fuzzy name cleaner (strip Pte/Ltd/etc.).
- `jobs/icp_tasks.py` (new):
  - Task functions invoked by background job runner (`icp_intake_process`).
- `lead_generation-main/docs`:
  - Add runbook for ICP Finder ops: rerun mapping, inspect patterns, troubleshoot blocks.

RBAC and tenancy
- Enforce viewer read-only (can see suggestions), ops/admin can intake/accept.
- Use per-request GUC for tenant (`set_config('request.tenant_id', ...)`) mirroring existing endpoints.

User Journey Flow (Chat Example)
- Agent: “I can infer your ICP from your website + best/worst customers. What’s your website?”
- User: “https://nexiuslabs.com”
- Agent: “Please list 5–15 best customers (Company — website).”
- User: “Acme — acme.com; Beta Logistics — betalogistics.sg; …”
- Agent: “Any 2–3 lost/churned with a short reason?”
- User: “Globex — globex.sg (budget < $10k); LegacySoft — legacysoft.com (on‑prem only)”
- Agent: “Thanks! I’ll crawl your site + seed sites, map to ACRA/SSIC, and mine patterns. I’ll be back with draft micro‑ICPs.”
- Agent (later): “Here are 4 micro‑ICPs with evidence counts. Adopt/refine?”
- User: “Adopt #1 and #3; exclude companies <10 HC.”
- Agent: “Updated ICP and Negative‑ICP. Queue a small enrichment batch now?”

Example payloads
- POST `/icp/intake` body (chat-composed):
  - answers: `{ website: "https://acmeanalytics.io", lost: [{ name, url, reason }], optional: { geos, integrations, acv, cycle, price_floor, champions, triggers } }`
  - seeds: `[{ seed_name: "Supply59", domain: "supply59.com" }, ...]`
- GET `/icp/suggestions` response:
  - `[{ id, title, ssic_cluster: ["62012","62019"], headcount_band: "50–200", integrations: ["HubSpot"], champions: ["RevOps"], evidence_count: 12, rationale: "..." }, ...]`
- POST `/icp/accept` body:
  - `{ suggestion_id: "..." }` or `{ suggestion_payload: {...normalized icp_rules payload...} }`

**Migration Steps**
1) Apply migrations 001–004 (create tables/MV; indexes; RLS; pg_trgm).
2) Deploy API endpoints gated by `ENABLE_ICP_INTAKE=false` initially.
3) Roll out jobs + limited tenants; monitor metrics; enable flag per tenant.
4) Nightly MV refresh added to scheduler; add on-demand refresh after evidence jobs.

**Risk & Mitigations**
- Crawl blocks/403s: aggregator blacklist; fallback to ACRA + public metadata; optional Playwright fetcher.
- Fuzzy false-positives: conservative threshold; manual review via evidence drawer; logs with similarity scores.
- Cost overruns: strict crawl caps; caching; stop conditions; batch limits per tenant/day.
- Data leakage: RLS enforcement; PII masking; robots respect; audit logs.

**Timeline (T-shirt sizing)**
- Week 1: Migrations + endpoints + mapping SQL + skeleton jobs.
- Week 2: Agents (user + seed crawlers), evidence store, MV refresh, initial suggestions.
- Week 3: UI integration (wizard + suggestions + evidence drawer), observability, flag-on for pilot tenant.
- Week 4: Hardening, docs/runbooks, acceptance test sign-off.

**Appendix — Payload Schemas (examples)**
- Intake payload
```
{
  "website": "https://acmeanalytics.io",
  "best_customers": [
    {"name": "Supply59", "website": "https://supply59.com"},
    {"name": "GreenPack Pte Ltd", "website": "https://greenpack.sg"}
  ],
  "lost_or_churned": [
    {"name": "TinyMart", "website": "https://tinymart.sg", "reason": "budget too low"}
  ],
  "employee_band": "50–200",
  "geo_language_tz": {"geo": "SG", "language": "en", "tz": "SGT"},
  "must_have_integrations": ["HubSpot", "Shopify"],
  "deal": {"acv_usd": 25000, "cycle_days": 60, "price_floor_usd": 10000},
  "champion_titles": ["Head of Sales", "RevOps"],
  "triggers": ["hiring RevOps", "opening new market", "Shopify migration"],
  "advanced": {"A_to_H": "..."}
}
```
- Suggestion card (response)
```
{
  "id": "sugg_01",
  "ssic_cluster": ["62012", "62019"],
  "headcount_band": "50–200",
  "integrations": ["HubSpot"],
  "champions": ["RevOps", "Head of Sales"],
  "evidence_count": 12,
  "rationale": "Observed across 4/5 seeds and user site",
  "negative_flags": ["budget < 10k"]
}
```

**Developer Notes**
- Keep changes minimal and consistent with existing style; reuse enrichment utilities rather than introducing new frameworks where possible.
- Add concise docstrings and error handling (logger.exception) where agents touch IO.
- Prefer parameterized SQL; ensure `%` in trigram ops is escaped when used via Python drivers.

**Required Implementation Code (copy/paste-ready)**
- The following code blocks provide ready-to-adapt migrations and Python modules consistent with the existing FastAPI + LangChain/LangGraph stack under `lead_generation-main/`.

- SQL — 001_icp_intake.sql
```
-- Enable extension for trigram if not present
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- icp_intake_responses: stores the wizard answers (fast-start + A–H)
CREATE TABLE IF NOT EXISTS icp_intake_responses (
  id BIGSERIAL PRIMARY KEY,
  tenant_id INTEGER NOT NULL,
  submitted_by TEXT NOT NULL,
  submitted_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  answers_jsonb JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_icp_intake_tenant_time ON icp_intake_responses(tenant_id, submitted_at DESC);
CREATE INDEX IF NOT EXISTS idx_icp_intake_answers_gin ON icp_intake_responses USING GIN (answers_jsonb);

-- customer_seeds: normalized list from intake
CREATE TABLE IF NOT EXISTS customer_seeds (
  id BIGSERIAL PRIMARY KEY,
  tenant_id INTEGER NOT NULL,
  seed_name TEXT NOT NULL,
  domain TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_customer_seeds_tenant_name ON customer_seeds(tenant_id, lower(seed_name));
CREATE INDEX IF NOT EXISTS idx_customer_seeds_tenant_domain ON customer_seeds(tenant_id, domain);
```

- SQL — 002_icp_evidence.sql
```
CREATE TABLE IF NOT EXISTS icp_evidence (
  id BIGSERIAL PRIMARY KEY,
  tenant_id INTEGER NOT NULL,
  company_id BIGINT NOT NULL REFERENCES companies(company_id),
  signal_key TEXT NOT NULL,
  value JSONB NOT NULL,
  source TEXT NOT NULL,
  observed_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_icp_evidence_tenant_company ON icp_evidence(tenant_id, company_id);
CREATE INDEX IF NOT EXISTS idx_icp_evidence_tenant_key ON icp_evidence(tenant_id, signal_key);
CREATE INDEX IF NOT EXISTS idx_icp_evidence_value_gin ON icp_evidence USING GIN (value);
```

- SQL — 003_icp_patterns_mv.sql
```
DROP MATERIALIZED VIEW IF EXISTS icp_patterns;
CREATE MATERIALIZED VIEW icp_patterns AS
WITH e AS (
  SELECT tenant_id, company_id, signal_key, value
  FROM icp_evidence
)
SELECT
  tenant_id,
  -- Top SSIC codes
  (
    SELECT jsonb_agg(x ORDER BY cnt DESC)
    FROM (
      SELECT (value->>'ssic') AS code, COUNT(*) cnt
      FROM e WHERE signal_key = 'ssic'
      GROUP BY 1
    ) x
  ) AS top_ssics,
  -- Common integrations
  (
    SELECT jsonb_agg(x ORDER BY cnt DESC)
    FROM (
      SELECT (value->>'integration') AS name, COUNT(*) cnt
      FROM e WHERE signal_key = 'integration'
      GROUP BY 1
    ) x
  ) AS common_integrations,
  -- Frequent buyer titles
  (
    SELECT jsonb_agg(x ORDER BY cnt DESC)
    FROM (
      SELECT (value->>'buyer_title') AS title, COUNT(*) cnt
      FROM e WHERE signal_key = 'buyer_title'
      GROUP BY 1
    ) x
  ) AS frequent_buyer_titles
FROM e
GROUP BY tenant_id;

CREATE INDEX IF NOT EXISTS idx_icp_patterns_tenant ON icp_patterns(tenant_id);
```

- SQL — 004_rls_and_perf.sql (optional hardening)
```
-- Example RLS templates (adapt to your auth model)
-- ALTER TABLE icp_intake_responses ENABLE ROW LEVEL SECURITY;
-- CREATE POLICY icp_intake_tenant_isolation ON icp_intake_responses
--   USING (tenant_id::text = current_setting('app.tenant_id', true));

-- Perf aids
CREATE INDEX IF NOT EXISTS idx_ssic_ref_title_trgm ON ssic_ref USING GIN (title gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_ssic_ref_desc_trgm ON ssic_ref USING GIN (coalesce(description,'') gin_trgm_ops);
```

- Python — schemas: `lead_generation-main/schemas/icp.py`
```
from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Dict, Any

class Seed(BaseModel):
    name: str = Field(..., alias='seed_name')
    website: Optional[HttpUrl] = None

class IntakePayload(BaseModel):
    website: Optional[HttpUrl] = None
    best_customers: List[Seed] = []
    lost_or_churned: List[Dict[str, Any]] = []
    employee_band: Optional[str] = None
    geo_language_tz: Optional[Dict[str, str]] = None
    must_have_integrations: List[str] = []
    deal: Optional[Dict[str, Any]] = None
    champion_titles: List[str] = []
    triggers: List[str] = []
    advanced: Optional[Dict[str, Any]] = None

class EvidenceItem(BaseModel):
    key: str = Field(..., alias='signal_key')
    value: Dict[str, Any]
    url: Optional[str] = None
    confidence: Optional[float] = None

class SuggestionCard(BaseModel):
    id: str
    ssic_cluster: List[str] = []
    headcount_band: Optional[str] = None
    integrations: List[str] = []
    champions: List[str] = []
    evidence_count: int = 0
    rationale: str = ""
    negative_flags: List[str] = []
```

- Python — backend endpoints: `lead_generation-main/app/icp_endpoints.py`
```
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import List
from schemas.icp import IntakePayload
from src.database import get_conn, get_pg_pool
from app.auth import require_optional_identity, require_auth
from src.icp_intake import (
    normalize_domain,
    save_icp_intake,
    enqueue_icp_jobs,
    get_suggestions_for_tenant,
    accept_suggestion,
)

router = APIRouter(prefix="/icp", tags=["icp"])

@router.post("/intake")
async def icp_intake(payload: IntakePayload, background: BackgroundTasks, claims: dict = Depends(require_optional_identity)):
    tid = claims.get("tenant_id")
    uid = claims.get("sub")
    if not tid or not uid:
        raise HTTPException(status_code=401, detail="Missing tenant or user identity")
    # Persist intake + seeds
    resp_id = save_icp_intake(tenant_id=int(tid), submitted_by=str(uid), payload=payload.model_dump(by_alias=True))
    # Kick off mapping + crawl + patterns
    background.add_task(enqueue_icp_jobs, int(tid))
    return {"status": "queued", "response_id": resp_id}

@router.get("/suggestions")
async def icp_suggestions(claims: dict = Depends(require_auth)):
    tid = claims.get("tenant_id")
    if not tid:
        raise HTTPException(status_code=401, detail="Missing tenant")
    return {"tenant_id": tid, "suggestions": get_suggestions_for_tenant(int(tid))}

@router.post("/accept")
async def icp_accept(body: dict, claims: dict = Depends(require_auth)):
    tid = claims.get("tenant_id")
    if not tid:
        raise HTTPException(status_code=401, detail="Missing tenant")
    suggestion = body.get("suggestion") or {}
    ok = accept_suggestion(int(tid), suggestion)
    return {"ok": bool(ok)}

@router.get("/patterns")
async def icp_patterns(claims: dict = Depends(require_auth)):
    tid = claims.get("tenant_id")
    if not tid:
        raise HTTPException(status_code=401, detail="Missing tenant")
    # Best-effort read
    from src.database import get_conn
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT top_ssics, common_integrations, frequent_buyer_titles FROM icp_patterns WHERE tenant_id=%s", (int(tid),))
        row = cur.fetchone()
        if not row:
            return {"tenant_id": tid, "patterns": None}
        return {
            "tenant_id": tid,
            "patterns": {
                "top_ssics": row[0],
                "common_integrations": row[1],
                "frequent_buyer_titles": row[2],
            },
        }
```

- Python — core logic: `lead_generation-main/src/icp_intake.py`
```
import re
import uuid
import time
import os
from typing import Any, Dict, List, Tuple
from psycopg2.extras import Json
from src.database import get_conn
from src.icp import _find_ssic_codes_by_terms
from src.crawler import crawl_site

def normalize_domain(url_or_domain: str | None) -> str | None:
    if not url_or_domain:
        return None
    u = (url_or_domain or "").strip().lower()
    u = re.sub(r"^https?://", "", u)
    u = re.sub(r"^www\.", "", u)
    u = u.split("/")[0]
    return u or None

def save_icp_intake(tenant_id: int, submitted_by: str, payload: Dict[str, Any]) -> int:
    seeds = payload.get("best_customers") or []
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO icp_intake_responses(tenant_id, submitted_by, answers_jsonb) VALUES (%s,%s,%s) RETURNING id",
            (int(tenant_id), submitted_by, Json(payload)),
        )
        resp_id = int(cur.fetchone()[0])
        if seeds:
            for s in seeds:
                name = (s.get("name") or s.get("seed_name") or "").strip()
                domain = normalize_domain(s.get("website") or s.get("domain"))
                if not name:
                    continue
                cur.execute(
                    "INSERT INTO customer_seeds(tenant_id, seed_name, domain) VALUES (%s,%s,%s)",
                    (int(tenant_id), name, domain),
                )
        return resp_id

def _company_id_by_domain(domain: str) -> int | None:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT company_id FROM companies WHERE primary_domain=%s OR website_domain=%s LIMIT 1", (domain, domain))
        row = cur.fetchone()
        return int(row[0]) if row else None

def _company_row_by_name(name: str) -> Tuple[int | None, str | None]:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT company_id, name FROM companies WHERE LOWER(name)=LOWER(%s) LIMIT 1", (name,))
        row = cur.fetchone()
        return (int(row[0]), str(row[1])) if row else (None, None)

def _ensure_company_for_seed(name: str, domain: str | None) -> int:
    cid, _ = _company_row_by_name(name)
    if cid:
        return int(cid)
    with get_conn() as conn, conn.cursor() as cur:
        if domain:
            cur.execute("SELECT company_id FROM companies WHERE primary_domain=%s OR website_domain=%s LIMIT 1", (domain, domain))
            row = cur.fetchone()
            if row:
                return int(row[0])
        cols = ["name"] + (["website_domain"] if domain else [])
        ph = ",".join(["%s"] * len(cols))
        sql = f"INSERT INTO companies({', '.join(cols)}) VALUES({ph}) RETURNING company_id"
        vals = [name] + ([domain] if domain else [])
        cur.execute(sql, vals)
        return int(cur.fetchone()[0])

def map_seeds_to_companies(tenant_id: int) -> List[Tuple[int, str]]:
    mapped: List[Tuple[int, str]] = []
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT seed_name, domain FROM customer_seeds WHERE tenant_id=%s ORDER BY id DESC LIMIT 200", (int(tenant_id),))
        for name, domain in cur.fetchall():
            cid = None
            if domain:
                cid = _company_id_by_domain(domain)
            if not cid and name:
                cid, _ = _company_row_by_name(name)
            if cid:
                mapped.append((int(cid), name))
    return mapped

def fuzzy_map_seeds_to_acra(tenant_id: int, seeds: List[Tuple[str, str | None]]) -> int:
    inserted = 0
    with get_conn() as conn, conn.cursor() as cur:
        for name, domain in seeds:
            cur.execute(
                """
                SELECT entity_name, uen, primary_ssic_code
                FROM staging_acra_companies a
                WHERE similarity(
                       regexp_replace(lower(%s), '(pte|ltd|private|limited|singapore|inc)\\b', '', 'gi'),
                       regexp_replace(lower(a.entity_name), '(pte|ltd|private|limited|singapore|inc)\\b', '', 'gi')
                ) > 0.35
                ORDER BY similarity(
                       regexp_replace(lower(%s), '(pte|ltd|private|limited|singapore|inc)\\b', '', 'gi'),
                       regexp_replace(lower(a.entity_name), '(pte|ltd|private|limited|singapore|inc)\\b', '', 'gi')
                ) DESC
                LIMIT 1
                """,
                (name, name),
            )
            row = cur.fetchone()
            if not row:
                continue
            matched_name, uen, ssic = row
            company_id = _company_id_by_domain(domain) if domain else None
            if not company_id:
                company_id = _ensure_company_for_seed(name, domain)
            cur.execute(
                "INSERT INTO icp_evidence(tenant_id, company_id, signal_key, value, source) VALUES (%s,%s,%s,%s,%s)",
                (int(tenant_id), int(company_id), 'ssic', Json({"ssic": str(ssic or "").strip(), "uen": (uen or "").strip(), "matched_name": matched_name}), 'acra'),
            )
            inserted += 1
    return inserted

async def crawl_company_site_to_evidence(company_id: int, url: str, tenant_id: int) -> int:
    out = await crawl_site(url)
    signals = out.get("signals") or {}
    count = 0
    with get_conn() as conn, conn.cursor() as cur:
        for bucket, arr in (signals.get("tech") or {}).items():
            for item in arr:
                cur.execute(
                    "INSERT INTO icp_evidence(tenant_id, company_id, signal_key, value, source) VALUES (%s,%s,%s,%s,%s)",
                    (int(tenant_id), int(company_id), 'integration', Json({"integration": item, "bucket": bucket}), 'crawl'),
                )
                count += 1
        titles = []
        for val in out.get("content_summary", "").split("|"):
            v = val.strip()
            if v and any(k in v.lower() for k in ["sales", "marketing", "ops", "revops", "cto", "cfo", "founder"]):
                titles.append(v)
        for t in titles[:6]:
            cur.execute(
                "INSERT INTO icp_evidence(tenant_id, company_id, signal_key, value, source) VALUES (%s,%s,%s,%s,%s)",
                (int(tenant_id), int(company_id), 'buyer_title', Json({"buyer_title": t}), 'crawl'),
            )
            count += 1
    return count

def refresh_icp_patterns():
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("REFRESH MATERIALIZED VIEW CONCURRENTLY icp_patterns")

# Optional: LLM-assisted extraction to complement deterministic crawler
from typing import Iterable
def _llm_client():
    try:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=os.getenv("LANGCHAIN_MODEL", "gpt-4o-mini"), temperature=float(os.getenv("TEMPERATURE", "0")))
    except Exception:
        return None

def llm_extract_structured(html: str) -> Iterable[Dict[str, Any]]:
    llm = _llm_client()
    if not llm:
        return []
    from langchain_core.prompts import ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You extract structured ICP evidence from HTML text. Return JSON lines: key, value, confidence."),
        ("user", "HTML text (truncated):\n{html}\nExtract up to 10 items across keys: industry_label, integration, buyer_title, location.")
    ])
    chain = prompt | llm
    try:
        resp = chain.invoke({"html": html[:20000]})
        text = getattr(resp, "content", "") if resp else ""
        items: list[Dict[str, Any]] = []
        for line in (text or "").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                import json
                obj = json.loads(line)
                if isinstance(obj, dict) and obj.get("key") and obj.get("value"):
                    items.append(obj)
            except Exception:
                continue
        return items[:10]
    except Exception:
        return []

def _load_latest_intake_answers(tenant_id: int) -> Dict[str, Any]:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT answers_jsonb FROM icp_intake_responses WHERE tenant_id=%s ORDER BY submitted_at DESC LIMIT 1", (int(tenant_id),))
        row = cur.fetchone()
        return row[0] if row and isinstance(row[0], dict) else {}

def _derive_negative_icp_flags(ans: Dict[str, Any]) -> List[str]:
    flags: List[str] = []
    try:
        deal = ans.get("deal") or {}
        floor = deal.get("price_floor_usd") or deal.get("price_floor")
        if floor:
            flags.append(f"budget < {floor}")
    except Exception:
        pass
    try:
        reasons = [ (x or {}).get('reason') for x in (ans.get('lost_or_churned') or []) ]
        reasons = [ (r or '').strip().lower() for r in reasons if (r or '').strip() ]
        top = []
        seen = set()
        for r in reasons:
            if r not in seen:
                seen.add(r)
                top.append(r)
        flags.extend(top[:3])
    except Exception:
        pass
    return flags[:5]

def generate_suggestions(tenant_id: int) -> List[Dict[str, Any]]:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT top_ssics, common_integrations, frequent_buyer_titles FROM icp_patterns WHERE tenant_id=%s", (int(tenant_id),))
        row = cur.fetchone()
        if not row:
            return []
        top_ssics, integrations, buyers = row
        cards: List[Dict[str, Any]] = []
        ssic_list = [x.get('code') for x in (top_ssics or []) if isinstance(x, dict) and x.get('code')]
        clusters = [ssic_list[i:i+2] for i in range(0, min(len(ssic_list), 6), 2)] or [ssic_list[:2]]
        ints = [i.get('name') for i in (integrations or []) if isinstance(i, dict) and i.get('name')]
        champs = [b.get('title') for b in (buyers or []) if isinstance(b, dict) and b.get('title')]
        neg = _derive_negative_icp_flags(_load_latest_intake_answers(tenant_id))
        for cl in clusters[:5]:
            cards.append({
                "id": f"sugg_{uuid.uuid4().hex[:6]}",
                "ssic_cluster": cl,
                "headcount_band": None,
                "integrations": ints[:3],
                "champions": champs[:3],
                "evidence_count": len(cl),
                "rationale": "ACRA + crawl evidence cluster",
                "negative_flags": neg[:3],
            })
        if len(cards) < 3:
            while len(cards) < 3:
                cards.append({
                    "id": f"sugg_{uuid.uuid4().hex[:6]}",
                    "ssic_cluster": ssic_list[:2],
                    "headcount_band": None,
                    "integrations": ints[:2],
                    "champions": champs[:2],
                    "evidence_count": 0,
                    "rationale": "Heuristic fill",
                    "negative_flags": neg[:2],
                })
        return cards

def get_suggestions_for_tenant(tenant_id: int) -> List[Dict[str, Any]]:
    return generate_suggestions(int(tenant_id))

def accept_suggestion(tenant_id: int, suggestion: Dict[str, Any]) -> bool:
    payload = {
        "ssic_codes": suggestion.get("ssic_cluster") or [],
        "integrations": suggestion.get("integrations") or [],
        "champions": suggestion.get("champions") or [],
    }
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO icp_rules(tenant_id, name, payload) VALUES (%s,%s,%s)",
            (int(tenant_id), 'icp_auto', Json(payload)),
        )
    return True

def enqueue_icp_jobs(tenant_id: int):
    start = time.perf_counter()
    mapped = map_seeds_to_companies(int(tenant_id))
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT answers_jsonb->>'website' FROM icp_intake_responses WHERE tenant_id=%s ORDER BY submitted_at DESC LIMIT 1", (int(tenant_id),))
        row = cur.fetchone()
        website = row[0] if row else None
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT seed_name, domain FROM customer_seeds WHERE tenant_id=%s ORDER BY id DESC LIMIT 200", (int(tenant_id),))
            seeds = [(r[0], r[1]) for r in cur.fetchall()]
        fuzzy_inserted = fuzzy_map_seeds_to_acra(int(tenant_id), seeds)
    except Exception:
        fuzzy_inserted = 0
    import asyncio
    async def _run():
        if website:
            dom = normalize_domain(website)
            cid = _company_id_by_domain(dom) or _ensure_company_for_seed((dom or "").split(".")[0].title(), dom)
            try:
                await crawl_company_site_to_evidence(int(cid), f"https://{dom}", int(tenant_id))
            except Exception:
                pass
        for cid, _name in mapped:
            with get_conn() as conn, conn.cursor() as cur:
                cur.execute("SELECT COALESCE(primary_domain, website_domain) FROM companies WHERE company_id=%s", (cid,))
                r = cur.fetchone()
                dom = r[0] if r and r[0] else None
            if dom:
                try:
                    await crawl_company_site_to_evidence(int(cid), f"https://{dom}", int(tenant_id))
                except Exception:
                    continue
        refresh_icp_patterns()
    asyncio.run(_run())
    try:
        total = 0
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM customer_seeds WHERE tenant_id=%s", (int(tenant_id),))
            total = int(cur.fetchone()[0] or 0)
        resolved = len(mapped)
        elapsed = time.perf_counter() - start
        rate = (resolved / total) if total else 0.0
        import logging
        logging.getLogger("icp_intake").info(
            "mapping_sla tenant=%s total=%d resolved=%d rate=%.2f elapsed_s=%.1f fuzzy_evidence=%d",
            tenant_id, total, resolved, rate, elapsed, fuzzy_inserted,
        )
    except Exception:
        pass
```

- Python — wiring the router in `lead_generation-main/app/main.py`
```
# near other include_router calls
try:
    from app.icp_endpoints import router as icp_router
    app.include_router(icp_router)
    logger.info("/icp routes enabled")
except Exception as _e:
    logger.warning("ICP routes not mounted: %s", _e)
```

- Python — optional targeting pack builder: `lead_generation-main/src/icp_intake.py`
```
def build_targeting_pack(suggestion: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ssic_filters": suggestion.get("ssic_cluster") or [],
        "technographic_filters": suggestion.get("integrations") or [],
        "pitch": "Short pitch: {} + {} segment".format(
            ",".join(suggestion.get("ssic_cluster") or []),
            "/".join(suggestion.get("integrations") or []),
        )
    }
```

**User Journey Examples (request/response)**
- Save intake (fast-start):
```
POST /icp/intake
{
  "website": "https://acmeanalytics.io",
  "best_customers": [
    {"name": "Supply59", "website": "https://supply59.com"},
    {"name": "GreenPack Pte Ltd", "website": "https://greenpack.sg"}
  ],
  "lost_or_churned": [
    {"name": "TinyMart", "website": "https://tinymart.sg", "reason": "budget too low"}
  ],
  "employee_band": "50–200",
  "geo_language_tz": {"geo": "SG", "language": "en", "tz": "SGT"},
  "must_have_integrations": ["HubSpot", "Shopify"],
  "deal": {"acv_usd": 25000, "cycle_days": 60, "price_floor_usd": 10000},
  "champion_titles": ["Head of Sales", "RevOps"],
  "triggers": ["hiring RevOps", "opening new market", "Shopify migration"]
}
-->
{ "status": "queued", "response_id": 42 }
```
- Get suggestions:
```
GET /icp/suggestions --> { "tenant_id": "...", "suggestions": [ { "id": "sugg_abc123", "ssic_cluster": ["62012","62019"], "integrations": ["HubSpot"], "champions": ["RevOps"], "evidence_count": 12, "rationale": "Derived from aggregated evidence" } ] }
```
- Accept a suggestion:
```
POST /icp/accept { "suggestion": { "ssic_cluster": ["62012","62019"], "integrations": ["HubSpot"], "champions": ["RevOps"] } }
-->
{ "ok": true }
```

**Scheduler Integration (Nightly Refresh + SLA Checks)**
- Python — `lead_generation-main/scripts/refresh_icp_patterns.py`
```
import os
import sys
import logging
from src.database import get_conn

def refresh_icp_patterns_mv():
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("REFRESH MATERIALIZED VIEW CONCURRENTLY icp_patterns;")
    logging.getLogger("nightly").info("icp_patterns MV refreshed")

if __name__ == "__main__":
    refresh_icp_patterns_mv()
```

- Wiring — `lead_generation-main/scripts/run_scheduler.py` (add job after nightly)
```
    async def job():
        try:
            await run_all()
            # ICP patterns refresh (decoupled from nightly flow)
            try:
                from scripts.refresh_icp_patterns import refresh_icp_patterns_mv
                await asyncio.to_thread(refresh_icp_patterns_mv)
            except Exception as exc:
                logging.getLogger("nightly").warning("icp_patterns refresh failed: %s", exc)
            # Acceptance checks (existing)
            ...
```

- SLA Metrics Targeting (≥80% seeds→ACRA within 5 minutes)
  - Mapping and fuzzy evidence insertion in `enqueue_icp_jobs` log: `mapping_sla tenant=<id> total=<n> resolved=<m> rate=<m/n> elapsed_s=<t>`.
  - Add alerting in your existing alerts script to warn when `rate < 0.80` or `elapsed_s > 300`.
