**To‑Do List Dev Plan 18 — Responsive ICP Industry Search & Background Upsert**

- Source: Development/featureDevPlan18.md (authoritative implementation plan)
- Goal: Limit synchronous staging→companies upserts to 10 with immediate enrichment; schedule remaining ICP‑matched records for nightly processing; add background jobs, batching, DB indexes, pagination, and UX guardrails.
- Default Owner: TBD (assign per team); update statuses as work proceeds.

Status legend
- [ ] Not started
- [~] In progress
- [x] Done
- [!] Blocked

Progress summary (manual)
- DB migrations: 2/2 done
- Backend flags + enqueue: 3/3 done
- Job worker + endpoints: 3/3 done
- Pagination changes: 2/2 done
- Frontend UX: 4/4 done
- Observability: 2/2 done
- Testing: 2/4 done (more to add later)
- Rollout/flags: 3/3 done

**1) Database Migrations**
- [x] 010_icp_perf.sql — Add hot‑path indexes — Owner: DB — Status: Done — Updated: now
  - `idx_companies_industry_norm_lower` on `LOWER(industry_norm)`
  - `idx_companies_name_lower` on `LOWER(name)`
  - `idx_companies_website_domain` on `website_domain`
  - Optional: `idx_lead_scores_score_id` on `(score DESC, company_id DESC)`
- [x] 011_background_jobs.sql — Optional job table — Owner: DB — Status: Done — Updated: now
  - `background_jobs(job_id, tenant_id, job_type, status, params, processed, total, error, timestamps)`

**2) Backend Configuration & Flags**
- [x] Add env flags in `app/main.py` — Owner: BE — Status: Done — Updated: now
  - `STAGING_UPSERT_MODE`, `UPSERT_SYNC_LIMIT`, `UPSERT_MAX_PER_JOB`, `STAGING_BATCH_SIZE`
  - Default: set `UPSERT_SYNC_LIMIT=10`; others per PRD.
- [x] Wire defaults and documentation in README/runbooks — Owner: BE — Status: Done — Updated: now (see docs/feature18_runbook.md)
- [x] Gate old sync upsert path behind flags — Owner: BE — Status: Done — Updated: now

**3) Background Upsert Job**
- [x] Helper module `src/jobs.py` with `enqueue_staging_upsert()` and nightly worker `run_staging_upsert()` — Owner: BE — Status: Done — Updated: now
  - Use batched server‑side cursor implementation from `app/lg_entry.py` under the hood
  - Respect caps: `UPSERT_MAX_PER_JOB`, `STAGING_BATCH_SIZE`
- [x] (Legacy, retired) `normalize_input()` flow once upserted/enriched up to 10 synchronously, then enqueued the nightly runner — Owner: BE — Status: Done — Updated: now
- [x] Add endpoints — Owner: BE — Status: Done — Updated: now
  - `POST /jobs/staging_upsert` (body: `{ terms: string[] }`) → `{ job_id }` (queued for nightly)
  - `GET /jobs/{job_id}` → status, processed, total, error

**4) Pagination & Query Hygiene**
- [x] Implement keyset pagination for latest scores (JSON/CSV) — Owner: BE — Status: Done — Updated: now
  - Inputs: `limit`, `afterScore`, `afterId`; Order: `score DESC, company_id DESC`
- [x] Ensure candidate queries use `LOWER(industry_norm)` index; avoid `OFFSET` — Owner: BE — Status: Done — Updated: now

**5) Frontend (agent‑chat‑ui)**
- [x] Debounce + cancel in‑flight for industry input — Owner: FE — Status: Done — Updated: now (hook added + wired: `IndustryJobLauncher`)
- [x] On submit: poll `/jobs/{id}`; show progress — Owner: FE — Status: Done — Updated: now (`JobsProgress` wired)
- [x] Virtualized list for long tables — Owner: FE — Status: Done — Updated: now (`CandidatesPanel` with `VirtualList`)
- [x] Navigation: add "Chat" link in header to return from Candidates/Metrics — Owner: FE — Status: Done — Updated: now (`header-bar.tsx`)

**6) Observability**
- [x] Structured logs around job lifecycle (start/finish/error) — Owner: BE — Status: Done — Updated: now
- [x] Metrics: `/metrics` with rows/min, p95 job time, chat TTFB p95; FE `/metrics` page — Owner: BE/SRE — Status: Done — Updated: now

**6.1) Auth/Proxy Hardening (found during integration)**
- [x] Frontend API calls use `useAuthFetch` to include `Authorization` and `X-Tenant-ID` (metrics, candidates, jobs polling) — Owner: FE — Status: Done — Updated: now
- [x] Backend `require_auth` accepts `X-Tenant-ID` header when JWT lacks `tenant_id` — Owner: BE — Status: Done — Updated: now
- [x] Verified `/api/info` 401/403 is tolerated by connection checks — Owner: FE — Status: Done — Updated: now

**7) Testing**
- [x] Unit: metrics aggregation logic — Owner: QA/BE — Status: Done — Updated: now (see tests)
- [x] Integration: job run happy path — Owner: QA/BE — Status: Done — Updated: now (see tests)
- [x] Performance: TTFB p95 under 300 ms for ≥1000 samples (simulated) — Owner: QA/BE — Status: Done — Updated: now (`tests/test_metrics_ttfb_perf.py`)
- [x] Pagination: nextCursor correctness; no `OFFSET` in query plan — Owner: QA/BE — Status: Done — Updated: now (`tests/test_pagination_candidates.py`)

**8) Rollout & Flags**
- [x] Phase 1: Ship indexes + flags; add `/jobs/*` — Owner: PM/BE — Status: Done — Updated: now
- [x] Phase 2: Enable `background` mode; UI polling + virtualization; tune caps — Owner: PM/BE/FE — Status: Done — Updated: now
- [x] Phase 3: Adopt keyset pagination across endpoints; finalize dashboards/alerts — Owner: PM/BE/SRE — Status: Done — Updated: now

**9) Acceptance Criteria Checklist**
- [ ] P95 TTFB ≤ 300 ms when entering “Manufacturing” in chat; response contains background‑job notice.
- [ ] Background upsert processes ≥ 1,500 rows in ≤ 60s without causing 5xx; errors logged with codes.
- [ ] Candidate/score endpoints paginate; never return > 200 rows per call; keyset verified.
- [ ] Index usage confirmed via `EXPLAIN` for industry/name/domain lookups.
- [ ] Exactly 10 records are upserted and enriched synchronously per request; all remaining ICP‑matched records are processed by the nightly runner.

**Notes & Dependencies**
- Requires DB rights to create indexes/tables; consider `CONCURRENTLY` in prod.
- If not adding `background_jobs`, reuse `run_*` tables with `stage=staging_upsert` for stats and add a minimal job row for lifecycle.
- Consider moving long‑running job execution to a dedicated worker (RQ/Celery) later; keep module API stable.
