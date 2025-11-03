**Feature PRD 18 — Responsive ICP Industry Search & Background Upsert (Aligned with Dev Plan 18)**

- **Objective:** Eliminate UI stalls when users enter broad industries (e.g., “Manufacturing”) by removing heavy staging→companies upserts from the request path, batching DB work, adding the right indexes, and guiding users to refine or run bulk tasks asynchronously.
- **Primary Users:** SDRs/ops using the ICP chat; anyone filtering candidates by industry; admins monitoring throughput.
- **Non‑Goals:** Replacing the enrichment/scoring pipeline; changing shortlist ranking; redesigning the chat UI beyond small guardrails.

**Problem Statement**
- Typing an industry in chat triggers `normalize_input()` → `_upsert_companies_from_staging_by_industries()` synchronously.
- For “Manufacturing”, this selects up to 1000 staging rows and performs per‑row lookups/updates, blocking the HTTP response.
- Missing supporting indexes (LOWER(name), website_domain, LOWER(industry_norm)) and use of `ILIKE ANY(...)` exacerbate latency.

**Outcomes**
- Chat remains responsive (TTFB) when users provide or edit “Industry” values, regardless of match volume.
- Bulk ingestion from staging runs asynchronously with visible progress and safe caps.
- ICP filtering and list views use indexed queries and keyset pagination.

**Scope**
- Backend: Move staging→companies upsert off the request path; add caps; adopt batched/streaming upsert; introduce feature flags.
- Database: Add hot‑path indexes; optional trigram support if fuzzy matching is kept.
- API/UX: Guardrails for very broad results; debounce, cancel in‑flight; progressive loading and virtualization where list output is shown.

**Functional Requirements**
- FR1: When a message includes an industry term, the HTTP response must not perform more than 10 synchronous upserts; those 10 also trigger enrichment immediately; all remaining ICP‑matched records are deferred to the nightly runner. (Implemented)
- FR2: Provide an async job to upsert staging rows by resolved SSIC codes or description patterns with batching (server‑side cursor, batch size 500). (Implemented)
- FR3: Expose background job status (queued, running, processed/total) in a lightweight endpoint for UI polling. (Implemented)
- FR4: Cap background processing per trigger (default 2,000 rows) at a global level. Per‑tenant caps/rate limiting are out of scope for v1 and may be added later. (Implemented w/ note)
- FR5: ICP candidate queries and exports must use keyset pagination (typical page sizes 25–200). (Implemented)
- FR6: “Too many results” refinement hints when estimated results > 500 are deferred for a future iteration. For v1, endpoints rely on keyset pagination and limits. (Deferred)

**Non‑Functional Requirements**
- NFR1: P95 TTFB for chat messages that include “industry” stays under 300 ms on typical dataset sizes.
- NFR2: Background upsert sustains ≥ 2k rows/minute without causing p95 DB CPU > 70%.
- NFR3: Zero data loss/duplication: idempotent upserts; safe retries.

**Design**
- Request path
  - Replace in‑request call to `_upsert_companies_from_staging_by_industries()` with:
    - “sync preview” mode: resolve SSIC codes and fetch only a tiny sample (≤ 20 names) for UX feedback.
    - Upsert and enrich up to 10 matching records synchronously for immediate usability.
    - Schedule the remaining ICP‑matched records for the nightly upsert + enrichment runner; return immediately with notice.
- Background upsert (nightly)
  - Use the batched server‑side cursor approach from `app/lg_entry.py` (itersize 500, batch upserts per connection).
  - Limits: stop after `UPSERT_MAX_PER_JOB` (default 2000) to avoid run‑away scans.
  - Idempotency: match by `uen`, fallback to `LOWER(name)` when safe; always set `last_seen = NOW()`.
  - Observability: write run stats (processed, batches, errors) to `run_summaries`/`run_stage_stats` with stage `staging_upsert`.
  - Scheduling: triggered work is queued for the nightly runner window rather than executed immediately during request handling.
- Querying candidates
  - Prefer indexed equality on `LOWER(industry_norm)`; default sort on `(score DESC, company_id DESC)` when available.
  - Use keyset pagination via `(score, company_id)` or `(updated_at, company_id)` tuples.

**Database Changes**
- Add indexes (safe to create concurrently in prod):
  - `CREATE INDEX IF NOT EXISTS idx_companies_industry_norm_lower ON companies (LOWER(industry_norm));`
  - `CREATE INDEX IF NOT EXISTS idx_companies_name_lower ON companies (LOWER(name));`
  - `CREATE INDEX IF NOT EXISTS idx_companies_website_domain ON companies (website_domain);`
  - Optional (for frequent sort): `CREATE INDEX IF NOT EXISTS idx_lead_scores_score_id ON lead_scores (score DESC, company_id DESC);`
- Optional: ensure `pg_trgm` is enabled if fuzzy joins remain in fallback paths.

**API Changes**
- New: `POST /jobs/staging_upsert` → body: `{ terms: string[] }` → `{ job_id }` (schedules work for the nightly runner). Optional `limit` is not supported in v1.
- New: `GET /jobs/{job_id}` → `{ status, processed, total, errors? }`.
- Change: `normalize_input()` respects flags and schedules nightly processing; the synchronous head (≤10) is upserted + enriched. A dedicated “preview sample” message is deferred; current UI uses job progress and candidates pages.
- Pagination: endpoints that list candidates or scores accept `limit`, `after` cursor; respond with `{ items, nextCursor }`.

**Configuration Flags**
- `STAGING_UPSERT_MODE`: `background` | `off` | `sync_preview` (default: `background`).
- `UPSERT_SYNC_LIMIT`: max rows allowed in request path (default: 10; 0 disables).
- `UPSERT_MAX_PER_JOB`: cap per background job (default: 2000).
- `STAGING_BATCH_SIZE`: server‑side cursor batch size (default: 500).
- `CHAT_DEBOUNCE_MS`: recommended 300–500 for client.

**UX Changes**
- Debounce and cancel in‑flight requests while typing. (Implemented)
- Virtualized lists for any table rendering large sets. (Implemented)
- Job progress via polling `/jobs/{id}`. (Implemented)
- “Too many results” chips and explicit chat preview message are deferred for a future iteration. (Deferred)

**Acceptance Criteria**
- AC1: P95 TTFB ≤ 300 ms for chat messages that include an industry term; measured via client-reported `/metrics/ttfb` aggregated to p95. (Implemented)
- AC2: Background upsert can process ≥ 1,500 rows in ≤ 60s on staging without 5xx; validation is operational (runbook), not an automated test in v1. (Partially implemented)
- AC3: Candidate/score list endpoints paginate correctly and never return > 200 rows per call. (Implemented)
- AC4: With the new indexes applied, `EXPLAIN` shows index usage for `industry_norm`, `LOWER(name)`, and `website_domain` lookups. (Implemented)
- AC5: Exactly 10 records are upserted and enriched synchronously per request; remaining ICP‑matched records are processed by the nightly runner. (Implemented)

**Observability**
- Metrics (v1): job queue depth, jobs processed total, lead scores total, rows/min (avg recent), p95 job duration, chat TTFB p95. p99 and per‑batch/per‑stage timings are deferred.
- Logs: structured job lifecycle events (start/finish/error) with `job_id`, `processed`, `duration_ms`. Batch‑level timings may be added later.

**Rollout Plan**
- Phase 1: Ship indexes and feature flags; switch request path to `sync_preview` and enqueue job.
- Phase 2: Add job status endpoints and UI polling; enable virtualization; ship debounce/cancel.
- Phase 3: Switch default pagination to keyset; tune caps; finalize alerts/dashboards.

**Alignment Notes & Known Gaps (v1)**
- FR6 guardrails (refinement chips, “too_many_results” messaging) are deferred; rely on keyset pagination for v1.
- “Sync preview” explicit sample in chat is deferred; UI surfaces progress via jobs and candidates view instead.
- Observability currently includes rows/min avg, p95 job duration, chat TTFB p95; p99 and per‑stage timings are future work.
- AC2 throughput is validated operationally; an automated perf harness can be added later.
- Per‑tenant caps/rate limiting are not in v1; global caps are enforced by flags.
- `POST /jobs/staging_upsert` does not accept an optional `limit` parameter in v1.

**Risks & Mitigations**
- Risk: Job backlog grows for very broad terms → Mitigate with caps, per‑tenant rate limits, and nightly continuation.
- Risk: Lock contention during heavy upserts → Keep batches small, avoid long‑running transactions, rely on index lookups.
- Risk: User confusion when results don’t appear instantly → Provide clear “started/importing…” status and sample preview.

**Open Questions**
- Should we disable description‑pattern fallback entirely unless SSIC codes resolve? (Reduces scans.)
- Do we need tenant‑specific caps and schedules for import runs?
- Which endpoint needs pagination first (scores vs candidates) for quickest UX win?
