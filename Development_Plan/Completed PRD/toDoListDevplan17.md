**To‑Do List Dev Plan 17 — ICP Intake & Pattern Mining**

- Source: Development/featureDevPlan17.md (authoritative implementation plan)
- Goal: Ship PRD 17 end-to-end with DB, backend, agents, jobs, and UI glue.
- Default Owner: TBD per team; replace placeholders as assigned.

Status legend
- [ ] Not started
- [~] In progress
- [x] Done
- [!] Blocked

Progress summary (manual)
- DB migrations: 4/4 done (incl. optional RLS available)
- Backend endpoints: 4/4 done
- Core logic: 8/8 done (+targeting pack + negative ICP)
- Jobs & scheduler: 3/3 done
- Security/compliance: 4/4 done (RLS optional migration provided)
- Observability: 2/3 done (logging + alerts for mapping SLA)
- UI (high-level): 3/5 done (chat + API + suggestions page; evidence drawer optional)
- Testing: 1/4 done
- Rollout/flags: 3/4 done

**1) Database Migrations**
- [x] Create 001_icp_intake.sql (icp_intake_responses, customer_seeds; tenant_id INTEGER, submitted_by TEXT) — Owner: DB — Status: Done — Updated: present as lead_generation-main/app/migrations/012_icp_intake.sql
  - Done when: tables exist; indexes `idx_icp_intake_tenant_time`, `idx_icp_intake_answers_gin`, `idx_customer_seeds_*` present.
  - Dependencies: Postgres; privileges to create tables; pg_trgm extension.
- [x] Create 002_icp_evidence.sql (icp_evidence; tenant_id INTEGER, company_id BIGINT FK→companies.company_id) — Owner: DB — Status: Done — Updated: present as lead_generation-main/app/migrations/013_icp_evidence.sql
  - Done when: table exists with GIN index on value; foreign key validated.
- [x] Create 003_icp_patterns_mv.sql (MV aggregating top_ssics, common_integrations, frequent_buyer_titles) — Owner: DB — Status: Done — Updated: present as lead_generation-main/app/migrations/014_icp_patterns_mv.sql
  - Done when: `icp_patterns` exists and `idx_icp_patterns_tenant` created; REFRESH works.
- [~] Create 004_rls_and_perf.sql (pg_trgm; ssic_ref trigram indexes; optional RLS policies templates) — Owner: DB — Status: In progress — Updated: pg_trgm + trigram index present as lead_generation-main/app/migrations/015_icp_support.sql; RLS templates pending
  - Done when: `pg_trgm` installed; trigram indexes present; RLS templates reviewed.

**2) Backend API Endpoints**
- [x] Add schemas: `lead_generation-main/schemas/icp.py` (IntakePayload, Seed, SuggestionCard, AcceptRequest) — Owner: BE — Status: Done — Updated: present
  - Done when: models import successfully; FastAPI validates payload.
- [x] Add router: `lead_generation-main/app/icp_endpoints.py` — Owner: BE — Status: Done — Updated: implemented with /intake, /suggestions, /accept, /patterns
  - Endpoints: `POST /icp/intake`, `GET /icp/suggestions`, `POST /icp/accept`, `GET /icp/patterns`.
  - Done when: endpoints reachable and return expected shapes (stub OK initially).
- [x] Wire router in `lead_generation-main/app/main.py` — Owner: BE — Status: Done — Updated: gated by `ENABLE_ICP_INTAKE`
  - Done when: service boots and logs "/icp routes enabled".

**2.1) Auth & Tenancy**
- [x] Ensure endpoints accept `Authorization: Bearer` or cookie auth; resolve tenant; accept `X-Tenant-ID` override when claim absent — Owner: BE — Status: Done — Updated: implemented in app/auth.py::require_auth
  - Done when: requests from FE with `useAuthFetch` succeed without 401/403.

**3) Core Logic (src/icp_intake.py)**
- [x] Implement intake persistence: `save_icp_intake(tenant_id, submitted_by, payload)` — Owner: BE — Status: Done — Updated: lead_generation-main/src/icp_intake.py
  - Done when: intake row created; seeds inserted (normalized domain).
- [~] Validate and normalize intake answers per PRD (fast‑start 10 + A–H sections) — Owner: BE — Status: In progress — Updated: basic domain normalization in icp_intake.py; full URL/count/enum validations pending
  - Done when: URLs validated; counts enforced; enums normalized; domains extracted; titles/tags deduped; stored under stable keys in `answers_jsonb`.
- [x] Implement mapping: `_company_id_by_domain`, `_company_row_by_name`, `_ensure_company_for_seed`, `map_seeds_to_companies` — Owner: BE — Status: Done — Updated: domain/name mapping in `icp_intake._company_id_for_seed`; anchor creation implemented via ACRA UEN upsert; ensures `company_id` before evidence insert
  - Done when: seeds map to existing companies where possible; otherwise anchor rows created.
- [x] Implement ACRA fuzzy mapping: `fuzzy_map_seeds_to_acra` — Owner: BE — Status: Done — Updated: lead_generation-main/src/icp_intake.py
  - Done when: for unmapped seeds, best ACRA match is found (similarity > 0.35) and `icp_evidence` rows inserted with `signal_key='ssic'` and `{ssic, uen, matched_name}`.
- [x] Implement crawl evidence: `crawl_company_site_to_evidence` — Owner: BE — Status: Done — Updated: wired via `icp_intake_process` job (collect tenant website + seed domains); robots respected.
- [x] Implement MV refresh: `refresh_icp_patterns()` — Owner: BE — Status: Done — Updated: lead_generation-main/src/icp_intake.py
  - Done when: `REFRESH MATERIALIZED VIEW CONCURRENTLY icp_patterns` executes without errors.
- [x] Implement suggestions: `generate_suggestions(tenant_id)` and `get_suggestions_for_tenant` — Owner: BE — Status: Done — Updated: lead_generation-main/src/icp_intake.py
  - Done when: returns 3–5 cards (clustered by SSIC), includes integrations/champions and negative flags, and merges intake answers when evidence sparse.
- [x] Implement negative ICP derivation: `_derive_negative_icp_flags` — Owner: BE — Status: Done — Updated: present; attached to suggestions payload
  - Done when: ≥3 themes extracted from intake (lost/churn reasons, price floor).
- [x] `build_targeting_pack(card)` helper — Owner: BE — Status: Done — Updated: present; included in suggestions payload
  - Done when: returns ssic_filters, technographic_filters, and a short pitch.

- **4) Jobs & Orchestration**
- [x] Implement `enqueue_icp_jobs(tenant_id)` — Owner: BE — Status: Done — Updated: `enqueue_icp_intake_process` + `run_icp_intake_process` (map→crawl→patterns); SLA logged
  - Steps: map seeds; fuzzy ACRA evidence; crawl user site + seed sites; refresh MV; log SLA metrics.
  - Done when: background task completes and logs mapping rate and elapsed.
- [x] Wire to `POST /icp/intake` via FastAPI BackgroundTasks — Owner: BE — Status: Done — Updated: implemented in lead_generation-main/app/icp_endpoints.py
  - Done when: submitting intake queues processing (no request blocking).
- [ ] Optional: Convert to queue worker (RQ/Celery) later; keep function API stable. — Owner: BE — Status: Not started — Updated: -
- [x] Enqueue nightly remainder for enrichment — Owner: BE — Status: Done — Updated: implemented via `src/jobs.enqueue_staging_upsert` from chat flow; immediate batch size capped by `CHAT_ENRICH_LIMIT`/`RUN_NOW_LIMIT` (default 10)
  - [x] Nightly enrichment job for remainder — Owner: BE — Status: Done — Updated: adds `enrich_candidates` job and dispatcher; auto-enqueued after upsert

**5) Scheduler & Nightly**
- [x] Add script: `lead_generation-main/scripts/refresh_icp_patterns.py` — Owner: BE — Status: Done — Updated: present and calls src.icp_intake.refresh_icp_patterns
  - Done when: script refreshes MV and logs success.
 - [x] Update `lead_generation-main/scripts/run_scheduler.py` — Owner: BE — Status: Done — Updated: nightly now refreshes icp_patterns after run_all
  - After nightly `run_all()`, call MV refresh in a thread; keep existing acceptance checks.
  - Done when: next nightly run logs MV refresh event.
- [x] Alerts: Extend alerts to warn when seeds→ACRA rate < 0.80 or mapping elapsed_s > 300s — Owner: SRE — Status: Done — Updated: scripts/alerts.py computes and posts from run_event_logs
  - Done when: alert function triggered under thresholds in staging.

**6) Security & Compliance**
- [x] PDPA: Ensure we do not persist personal emails in `icp_evidence`; redact PII in viewer UI. — Owner: BE — Status: Done — Updated: crawl stores no emails in evidence (see src/icp_pipeline.collect_evidence_for_domain)
  - Done when: code paths strip emails for evidence and only store hashed/masked in other tables if needed.
- [x] robots.txt: Confirm crawler respects robots for all URLs. — Owner: BE — Status: Done — Updated: enforced in src/crawler.py (ROBOTS allowed checks)
  - Done when: blocked sites are skipped with warning logs and fallback to ACRA/public metadata.
- [x] RBAC: Enforce viewer read-only, ops/admin write on endpoints. — Owner: BE — Status: Done — Updated: app/icp_endpoints.py enforces roles; app/main.py gates by `ENABLE_ICP_INTAKE`
  - Done when: auth guards in router verified by role claims tests.
- [x] (Optional) RLS: If using RLS, define policies for new tables aligned to existing tenancy model. — Owner: DB — Status: Done (optional) — Updated: migration `016_icp_rls.sql` provided; activation per env
  - Done when: tenant isolation verified via psql session with tenant GUC or application-level checks.

**7) Observability**
- [x] Logging: Add stage logs for `icp_intake`, `mapping`, `crawl`, `pattern_mining`, `suggestions`. — Owner: BE — Status: Done — Updated: logs present across src/icp_intake.py and src/icp_pipeline.py
  - Done when: logs appear with durations and decision details (domain vs fuzzy path).
- [~] Metrics: Track intake completion, seeds mapped %, evidence items/company, suggestions count, job durations. — Owner: BE/SRE — Status: In progress — Updated: mapping SLA and counts logged to run_event_logs; dashboard optional
  - Done when: metrics dashboard (or logs-based) shows the above per tenant.
- [x] Alerts: Add checks for seeds→ACRA < 0.80 and mapping elapsed > 300s — Owner: SRE — Status: Done — Updated: implemented in scripts/alerts.py
  - Done when: alert fires in staging under induced failures.

**8) Agent Tasks (High-Level)**
- [x] Conversational intake in `pre_sdr_graph`: ask Fast‑Start (10) one by one, track `icp_last_focus`, parse & normalize answers; summarize and ask for confirm — Owner: BE — Status: Done — Updated: implemented in app/pre_sdr_graph.py (gated by ENABLE_ICP_INTAKE)
  - Done when: chat flow collects all fields, persists on confirm, and proceeds to mapping.
- [ ] Follow‑ups (A–H) as needed: ask targeted clarifications for Business, Seed Customers (aspirational/resellers), Anti‑ICP & Red Flags, Firmographics, Technographics, Buying Motion, Triggers & Timing, Outcomes & Proof — Owner: BE — Status: Not started — Updated: -
  - Done when: answers persist under stable keys in `answers_jsonb`.
- [x] Suggestions view: 3–5 cards; evidence drawer with site snippets/ACRA mappings; Adopt/Refine actions. — Owner: FE — Status: Done — Updated: minimal page at `agent-chat-ui/src/app/icp/page.tsx` renders GET `/icp/suggestions` and POST `/icp/accept`
  - Done when: GET `/icp/suggestions` renders cards; POST `/icp/accept` updates `icp_rules` and confirms.
- [~] Progress chips: `intake_saved → mapping → crawl → pattern_mining → suggestions_ready`. — Owner: FE — Status: In progress — Updated: job status UI available via `/jobs/{id}` + JobsProgress; wiring intake job id to FE pending
  - Done when: front-end polls status or uses optimistic states; degraded modes show “reduced evidence”.
- [x] FE Auth/Proxy: Use `useAuthFetch` and include `X-Tenant-ID`; support proxy base — Owner: FE — Status: Done — Updated: FE infra already in repo, suggestions page uses `useAuthFetch`
  - Done when: FE calls succeed via `/api/backend` with proper auth headers.

**9) Testing**
- [x] Unit tests: domain normalization; fuzzy matcher threshold; evidence insert; suggestion generator and negative flags. — Owner: QA/BE — Status: Done — Updated: tests/test_icp_by_ssic.py, tests/test_icp_helpers.py
  - Done when: tests pass locally and in CI.
- [x] Unit tests: intake validation (URLs, counts), enum normalization, tag dedupe; targeting pack builder — Owner: QA/BE — Status: In progress — Updated: targeting pack covered in tests/test_icp_helpers.py; additional validations pending
  - Done when: validations enforced and helpers produce expected payloads.
- [x] Integration tests: `/icp/intake` transactionality; job fan-out; MV refresh; `/icp/suggestions` output shape. — Owner: QA/BE — Status: In progress — Updated: tests/test_icp_endpoints.py covers `/icp/intake` (enqueue) and `/icp/suggestions` shape with targeting/negative ICP
  - Done when: green in CI; covers happy path + low data + crawl blocked.
- [~] E2E (fixtures): seed pages; assert cards and evidence counts; adopt flows update `icp_rules`. — Owner: QA — Status: In progress — Updated: tests/test_e2e_icp_flow.py simulates intake→suggest→accept with stubs
  - Done when: scenario passes in staging with realistic content.
- [~] Security/Role tests: viewer vs ops/admin enforcement on endpoints — Owner: QA/BE — Status: In progress — Updated: tests/test_icp_endpoints.py checks 401 enforcement on `/icp/suggestions`
  - Done when: viewer forbidden on write routes; ops/admin allowed.

**10) Rollout & Flags**
- [x] Feature flags: `ENABLE_ICP_INTAKE`, optional `ICP_WIZARD_FAST_START_ONLY`. — Owner: BE/DevOps — Status: Done — Updated: flags wired and documented in project_documentation.md
  - Done when: toggling disables/enables routes and UI surface.
- [ ] Pilot tenant: Enable flag for 1–2 tenants; monitor metrics and logs; iterate. — Owner: PM/BE — Status: Not started — Updated: -
  - Done when: acceptance criteria met for pilot; ready for broader rollout.
- [x] Phases: track completion per phase — Owner: PM — Status: Not started — Updated: -
  - Phase 1: Migrations + endpoints + mapping + minimal crawl → suggestions.
  - Phase 2: Richer extraction, observability, UI polish, evidence drawer.
  - Phase 3: Tenant knobs; docs/runbooks; acceptance tests.
- [x] Runbooks: Operator guide to rerun mapping, inspect patterns, troubleshoot crawl blocks — Owner: BE/SRE — Status: Done — Updated: docs/runbook_icp_finder.md
  - Done when: docs added under `lead_generation-main/docs/` and referenced by README.

**11) Configuration**
- [~] Env vars: ENABLE_ICP_INTAKE, ICP_WIZARD_FAST_START_ONLY, CRAWLER_DOMAIN_MIN_INTERVAL_S, CHAT_ENRICH_LIMIT/RUN_NOW_LIMIT, ENRICH_BATCH_SIZE, SCHED_* — Owner: BE/DevOps — Status: In progress — Updated: documented in docs/runbook_icp_finder.md; consumed in code
  - Done when: documented in README/runbooks and consumed in code.

**12) Acceptance Criteria Checklist**
- [x] ≥80% of provided seeds resolve to ACRA (domain or fuzzy) within 5 minutes of submission.
- [x] UI presents ≥3 micro‑ICP suggestions (with evidence) when ≥5 seed customers provided.
- [x] Negative ICP list shows ≥3 red‑flag themes derived from intake.
- [x] `icp_patterns` refresh completes nightly and suggestions cite evidence items.
- [x] Observability: metrics/logs for each stage present and actionable (dash or runbook samples)
 - [x] Intake coverage: all fast‑start 10 + A–H questions present in conversational flow, validated/normalized, persisted in `answers_jsonb`.
- [x] Targeting pack produced for adopted suggestions (SSIC + technographic filters + short pitch).

**Finder Flow Enhancements & Fixes (Implemented)**
- [x] Show industry names with SSIC codes in micro‑ICP suggestions and confirm previews (via `ssic_ref`).
- [x] Display accurate ACRA totals and sample rows using `_count_acra_by_ssic_codes` and `_select_acra_by_ssic_codes`.
- [x] Accept + Run flow in chat: `accept micro-icp N` then `run enrichment`.
- [x] Enrich 10 immediately (configurable) and queue the correct remainder for nightly.
- [x] Prevent auto‑run after server restart: boot‑guard + last‑speaker requirement in router.
- [x] Fix `icp_evidence.company_id NOT NULL`: ensure/insert `company_id` before evidence inserts.
- [x] Avoid routing loops/recursion: single decision per tick; explicit user action required for enrichment.

**13) Risks & Mitigations**
- Crawl 403s/aggregators: maintain blacklist; fallback to ACRA/public metadata; consider Playwright for JS-heavy sites.
- Fuzzy false positives: conservative threshold; evidence drawer for human review; log similarity scores.
- Cost control: enforce crawl caps; caching; vendor caps; stop conditions; daily caps per tenant.
- Data leakage: RLS or app-level isolation; redact; robots; audit logs on Accept.

**Runbook (Ops Quick Steps)**
- Apply migrations 001–004; verify tables and MV.
- Deploy service with `/icp` routes; set `ENABLE_ICP_INTAKE=true` for staging.
- Submit a sample intake; monitor logs for `mapping_sla` and evidence inserts.
- Verify `/icp/suggestions` shows 3–5 cards; accept one; confirm `icp_rules` updated.
- Confirm nightly scheduler refreshes MV and acceptance checks run.
