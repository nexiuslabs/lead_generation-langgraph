**Feature PRD 8 — Observability, QA & Runbooks**

- **Objective:** End-to-end visibility of enrichment runs (per-tenant, per-run) with actionable alerts, fast diagnosis, and consistent QA, while honoring privacy/compliance and cost tracking.
- **Primary Users:** Ops/SRE, Data/ML engineers, Product, On-call responders.
- **Out of Scope:** Vendor-side dashboards; long-term BI modeling; replacing LangSmith traces.

**Outcomes**
- **Reliable Metrics:** Per-tenant/per-run counts, rates, costs, and latencies readily available within minutes of a run completing.
- **Actionable Alerts:** Clear, low-noise alerting on critical failures and leading indicators (quotas/rate limits) with runbooks linked.
- **Fast Triage:** One-click trace correlation (DB run_id ↔ LangSmith trace), stage-level timings, and error context.
- **QA Confidence:** Automated sample and checklist ensure quality stays within thresholds; deviations surface to Ops.

**Key Concepts**
- **Run:** A nightly or ad-hoc enrichment execution scoped to a tenant. Identified by `enrichment_runs.run_id` and annotated with `tenant_id`, `started_at`, `ended_at`, `status`.
- **Stage:** Major steps within a run (e.g., `mv_refresh`, `crawl`, `tavily`, `lusha`, `zerobounce`, `score`, `export_odoo`).
- **Event:** Fine-grain log inside a stage (start/finish, error, retry, vendor call result) correlated by `run_id` and `company_id` where applicable.

**Instrumentation**
- **Where:**
  - `scripts/run_nightly.py`, `scripts/run_scheduler.py`: run lifecycle, stage start/stop, counts, error rates, daily caps.
  - `src/enrichment.py`, `src/lead_scoring.py`: per-company stage outcomes, timings, cache hits, content hash.
  - `app/main.py` exports: Odoo export counts and error classes.
  - LangSmith traces: LLM/tool steps remain traced; store trace/run correlation.
- **How (logging):**
  - Use structured JSON logs with fields: `ts`, `level`, `tenant_id`, `run_id`, `stage`, `company_id`, `event`, `status`, `duration_ms`, `error_code`, `error_msg`, `vendor`, `cost_usd`, `tokens_input`, `tokens_output`, `trace_id`.
  - Logger names: `obs.run`, `obs.stage`, `obs.vendor`, `obs.qa`.
- **How (metrics):**
  - Emit counters/gauges via DB tables (below) and optionally Prometheus/OpenTelemetry for runtime dashboards.
  - Correlate with LangSmith using `trace_id` and keep in DB for joins.

**Data Model (Postgres)**
- `enrichment_runs` (already present)
  - Ensure columns: `run_id (PK)`, `tenant_id`, `started_at`, `ended_at`, `status`, `langsmith_trace_url TEXT NULL`.
- `run_stage_stats`
  - Columns: `run_id`, `tenant_id`, `stage VARCHAR`, `count_total INT`, `count_success INT`, `count_error INT`, `p50_ms INT`, `p95_ms INT`, `p99_ms INT`, `cost_usd NUMERIC(12,4) DEFAULT 0`, `tokens_input INT`, `tokens_output INT`.
  - Index: `(tenant_id, run_id, stage)`.
- `run_vendor_usage`
  - Columns: `run_id`, `tenant_id`, `vendor VARCHAR` (e.g., `tavily`, `lusha`, `zerobounce`, `openai`), `calls INT`, `errors INT`, `cost_usd NUMERIC(12,4)`, `rate_limit_hits INT`, `quota_exhausted BOOL DEFAULT FALSE`.
- `run_event_logs`
  - Columns: `run_id`, `tenant_id`, `stage`, `company_id NULL`, `event VARCHAR`, `status VARCHAR`, `error_code NULL`, `duration_ms INT NULL`, `trace_id TEXT NULL`, `extra JSONB NULL`, `ts TIMESTAMPTZ DEFAULT now()`.
  - Retention: 14–30 days rolling; compact long messages.
- `qa_samples`
  - Columns: `run_id`, `tenant_id`, `company_id`, `bucket VARCHAR`, `checks JSONB`, `result VARCHAR` (`pass`/`fail`/`needs_review`), `notes TEXT`, `created_at TIMESTAMPTZ DEFAULT now()`.

**Metrics (Per Tenant, Per Run)**
- **Candidates:**
  - **Pre-filter:** count of rows from `icp_candidate_companies` considered.
  - **Post-filter/Selected:** companies chosen for this run (respecting caps/backpressure).
- **Crawl:**
  - **Success Rate:** `crawl_success / crawl_attempted`.
  - **Fallback Rate (Tavily):** `tavily_used / crawl_attempted`.
- **Contacts:**
  - **Lusha Usage:** lookups attempted/succeeded; fallback share.
  - **Verified Email Rate:** `emails_verified / emails_total` (ZeroBounce statuses).
- **Scoring:**
  - **Bucket Distribution:** share in `High/Medium/Low`.
- **Costs:**
  - **OpenAI Spend:** `$` via tokens×pricing per model.
  - **Vendor Spend:** ZeroBounce/Lusha spend from responses or price cards.
- **Latency:**
  - **Stage Percentiles:** `p50/p95/p99` per stage.
  - **Run Duration:** `ended_at - started_at`.

**Dashboards**
- **Tenant Overview:**
  - Candidates pre/post, shortlist delivered, success/error rates, bucket distribution, run duration.
- **Vendor Usage & Cost:**
  - Lusha/ZeroBounce/OpenAI calls, error rates, rate-limit/quota flags, estimated cost.
- **Quality & Verification:**
  - Verified email rate, QA sample pass rate, rationale length histogram.
- **Latency:**
  - Stage p95 charts; drill-down by stage and by run.

**Alerts**
- **MV Refresh Failure:**
  - Condition: `run_stage_stats.stage=mv_refresh AND count_error>0` or status not recorded within window.
  - Action: page on-call; see runbook MV-01.
- **Crawl Error Rate > X%:**
  - Condition: `count_error / count_total > 0.3` for `crawl`.
  - Action: warn Slack; if >50% escalate; see CR-02.
- **Vendor Quota/Rate Limit:**
  - Condition: `run_vendor_usage.quota_exhausted=TRUE` OR `rate_limit_hits>0`.
  - Action: alert Slack with tenant/vendor; see VQ-03.
- **OpenAI Rate Limit:**
  - Condition: `run_vendor_usage.vendor='openai' AND rate_limit_hits>0`.
  - Action: throttle batch size; see OA-04.
- **Candidate Count Below Floor:**
  - Condition: post-filter candidates < configured floor (e.g., 50) for a tenant.
  - Action: create Jira ticket; see CC-05.

**QA Spot Checks**
- **Sampling:**
  - Random 10 companies from `High` bucket per run (`ORDER BY random() LIMIT 10`).
- **Checklist:**
  - **Domain:** correct and resolvable.
  - **About Text:** non-empty, relevant to company.
  - **Contacts:** at least one verified/unknown business email; no personal webmail if policy prohibits.
  - **Rationale:** coherent, references extracted signals; not generic boilerplate.
- **Workflow:**
  - Auto-generate QA sample rows in `qa_samples` with placeholders; Ops fills `result`/`notes`.
  - Flag if `pass rate < 80%` → open follow-up.

**Runbooks**
- **MV-01: MV Refresh Failure**
  - **Symptom:** Alert fired; no `mv_refresh` success; `/logs` show SQL errors.
  - **Checks:** connection to DB, locks on MV, recent DDL; rerun `REFRESH MATERIALIZED VIEW CONCURRENTLY icp_candidate_companies;`.
  - **Mitigation:** run `scripts/refresh_icp_mv.py`; if schema drift, re-apply migrations.
- **CR-02: Crawl Error Rate High**
  - **Symptom:** `crawl` error rate >30%.
  - **Checks:** robots blocks, DNS failures, TLS errors; domain min interval config `CRAWLER_DOMAIN_MIN_INTERVAL_S`.
  - **Mitigation:** increase delay, reduce concurrency, enable Tavily fallback earlier; re-run affected batch.
- **VQ-03: Lusha/ZeroBounce Quota**
  - **Symptom:** quota_exhausted or HTTP 402/429.
  - **Checks:** key validity, current quotas; recent usage spikes.
  - **Mitigation:** skip vendor gracefully, reduce contacts weighting in score, retry next night; if urgent, top-up quota.
- **OA-04: OpenAI Rate Limit**
  - **Symptom:** Rate limit hits; latency spikes.
  - **Checks:** model usage, token bursts, parallelism.
  - **Mitigation:** backoff with jitter, lower concurrency, switch to fallback model temporarily.
- **CC-05: Candidate Count Low**
  - **Symptom:** post-filter below floor.
  - **Checks:** `icp_rules` validity; MV contents; industry resolution fallback path engaged.
  - **Mitigation:** widen filters, refresh staging data, enable industry-only fallback, verify `icp_rules` persisted.
- **OE-06: Odoo Export Errors**
  - **Symptom:** export stage errors or `ready=false`.
  - **Checks:** `odoo_connections` mapping, DB availability, `autopost_bills` default constraints.
  - **Mitigation:** set sane defaults, skip non-essential fields, verify credentials; re-run export only.

**SLIs & SLOs**
- **Run Success Rate:** ≥ 98% of nightly runs complete without critical errors per tenant.
- **Run Duration:** p95 < 90 minutes for 1k shortlist.
- **Crawl Success Rate:** p50 ≥ 70% (tenant-dependent).
- **Verified Email Rate:** ≥ 40% companies have ≥1 verified/unknown business email.
- **Alert MTTA:** < 10 minutes; **MTTR:** < 2 hours for P1 incidents.

**Privacy & Compliance**
- **Redaction:** Never log raw email addresses or LLM prompts containing PII; store email hashes and vendor identifiers where needed.
- **Content Hash:** Persist content hash only (not full text) for reproducibility unless retention policy allows; if retained, cap window (e.g., 90 days).
- **Access Control:** All observability tables include `tenant_id` and honor RLS/filters; Ops accounts only.

**APIs & Surfaces**
- **Status Endpoint:** `GET /shortlist/status` already surfaces counts; extend to include `last_run_id`, `last_run_status`, `last_run_started_at`, `last_run_ended_at`.
- **Run Detail:** `GET /runs/{run_id}/stats` (optional) returns aggregated `run_stage_stats` and `run_vendor_usage` for UI/ops.
- **CSV Exports:** `GET /export/latest_scores.csv` unchanged; add `GET /export/run_events.csv?run_id=...` for ad-hoc forensics.

**Tooling**
- **Tracing:** Continue using LangSmith for LLM/tool steps (env: `LANGCHAIN_TRACING_V2`, `LANGSMITH_PROJECT`). Save top-level trace URL per run.
- **Dashboards:** Start with SQL-based charts (Metabase/Grafana over Postgres); consider Prometheus later for real-time.
- **Log Files:** Keep `.logs/*.log` for tailing; prefer DB-backed metrics for durable aggregation.

**Implementation Plan**
- **Phase 1 (Schema + Minimal Aggregates):**
  - Add tables: `run_stage_stats`, `run_vendor_usage`, `run_event_logs`, `qa_samples`.
  - Annotate `enrichment_runs` with `langsmith_trace_url`.
  - Instrument nightly/scheduler and per-company stages to write aggregates at stage end; write per-event only on errors initially.
- **Phase 2 (Dashboards + Alerts):**
  - Create tenant overview and vendor dashboards via SQL.
  - Implement alert workers checking last run stats; send Slack/email.
- **Phase 3 (QA Workflow):**
  - Auto-create `qa_samples` per run; add lightweight UI surface or CSV export; track pass rate.
- **Phase 4 (Hygiene + Retention):**
  - Add retention jobs (DELETE old `run_event_logs` >30d); compact large payloads.

**Acceptance Criteria**
- **Data Completeness:** For 3 consecutive nightly runs across 2 tenants, all metrics populated; `run_stage_stats` and `run_vendor_usage` totals match logs within 2%.
- **Dashboards Live:** Ops can see per-tenant overview, vendor usage, and latency charts.
- **Alerts Working:** Each alert type fires in simulation (dry-run); runbooks referenced and followed successfully.
- **QA Logged:** Each run generates 10 `High` QA samples; pass rate tracked; failures create follow-ups.
- **Privacy:** No raw emails or PII in logs; access restricted by role.

**Open Questions**
- **Tooling Choice:** Start with Postgres + Metabase or invest in Prometheus now?
- **Retention Windows:** Finalize event log retention (14 vs 30 days) and raw content retention policy.
- **Granularity:** Do we need per-URL crawl metrics or keep at company level?

