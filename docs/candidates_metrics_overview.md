# Candidates & Metrics Feature Overview

## Backend Data Flow (Technical)
- Visiting `/candidates/latest` fetches recent companies for the signed-in tenant. It filters on `industry_norm`, orders by `last_seen`/`company_id`, and returns a keyset `nextCursor` so the UI can paginate (`app/main.py:927`). The query joins `icp_evidence` (tenant-scoped) to resolve the company ids and calls `set_config('request.tenant_id', ...)` before hitting Postgres so RLS policies remain effective.
- `/metrics` aggregates operational stats from Postgres: job queue depth and processed totals from `background_jobs`, lead-score counts from `lead_scores`, and latency metrics from `run_event_logs` (`app/main.py:1081`). The handler now sets the tenant GUC and adds explicit `tenant_id` predicates before executing each query, so callers only see their own data. 
- Tests keep pagination honest: `tests/test_pagination_candidates.py:59` ensures no `OFFSET` use, and metric p95 logic matches test expectations (`app/main.py:1056`).
- `IndustryJobLauncher` queues work through `/jobs/staging_upsert`, inserting a queued row in `background_jobs` (`app/main.py:1099`, `src/jobs.py:32`). The worker later maps SSIC codes in `ssic_ref`, streams rows from `staging_acra_companies`, and upserts or updates `companies` (including `last_seen`) (`app/lg_entry.py:211`). Job polling hits `/jobs/{id}` and reads `background_jobs` (`app/main.py:1113`).
- `/metrics/ttfb` allows the UI to persist chat-first-token timing by inserting into `run_event_logs` (`app/main.py:1085`).

## Frontend Overview (Technical)
- `AppShell` provides navigation, signed-in identity, export buttons, optional tenant override, Odoo check, and sign-out. It’s rendered on Chat/Candidates/Metrics pages (`agent-chat-ui/src/components/ui/app-shell.tsx:213`).
- `CandidatesPanel` calls `/candidates/latest`, stores cursor state, supports an industry filter, and renders rows through `VirtualList` for smooth scrolling (`agent-chat-ui/src/components/CandidatesPanel.tsx:8`).
- `IndustryJobLauncher` collects industries, POSTs to `/jobs/staging_upsert`, and hands the job id to `JobsProgress` (`agent-chat-ui/src/components/IndustryJobLauncher.tsx:7`).
- `JobsProgress` polls `/jobs/{job_id}` every 1.5 s to show processed/total counts (`agent-chat-ui/src/components/JobsProgress.tsx:1`).
- The Metrics page polls `/metrics` every 15 s and renders the six dashboard cards (`agent-chat-ui/src/app/metrics/page.tsx:15`).
- All data-fetching hooks go through `useAuthFetch`, which attaches bearer tokens, optional tenant overrides, retries once after `/auth/refresh`, and supports API proxying (`agent-chat-ui/src/lib/useAuthFetch.ts:5`).

## Plain-Language Summary
- The Candidates tab asks the server for fresh company leads in chunks, so the list can keep going as you scroll or press “Load more.”
- The Metrics tab checks in every 15 seconds to show how many jobs are waiting, how many finished, and how fast recent runs or chats were.
- You must be signed in; the helper that sends requests includes your token and tenant info automatically.

### Candidates Page UI Pieces
- **Header (`AppShell`)** – shared top strip with navigation, who’s logged in, export buttons, Odoo check, optional tenant override, and sign-out.
- **Industry Launcher** – text box where you paste industry names; it queues a background refresh job and shows its progress.
- **Candidates List** – searchable, scrollable list of companies showing ID, name, industry, website, and last-seen time. You can filter by industry and manually refresh or load more.
- **Virtual Scroll** – keeps the list snappy by rendering only visible rows, so large result sets don’t slow the browser.

### Metrics Page UI Pieces
- **Header (`AppShell`)** – same shared navigation and controls.
- **Metric Cards** – six tiles for queue depth, jobs processed, lead-score totals, recent rows/minute, p95 job duration, and chat response speed. Errors are surfaced inline if the fetch fails.

## Database Touchpoints by Action
- **Browse Candidates** (load page, filter, load more): reads `companies` via `/candidates/latest`, scoped through `icp_evidence` for the signed-in tenant.
- **Queue Industry Refresh** (submit industries): inserts into `background_jobs` via `/jobs/staging_upsert`; the downstream worker reads `ssic_ref` and `staging_acra_companies` and upserts/updates `companies` (updates `last_seen`).
- **Monitor Job Progress** (auto polling): reads `background_jobs` via `/jobs/{id}`.
- **View Metrics Dashboard** (poll every 15 s): reads `background_jobs`, `lead_scores`, and `run_event_logs` via `/metrics`.
- **Record Chat TTFB** (optional UI metric submission): inserts into `run_event_logs` via `/metrics/ttfb`.

## Tenant Scope Notes

- `/candidates/latest` (`app/main.py:927`) invokes `set_config('request.tenant_id', ...)` and joins tenant-scoped `icp_evidence`, so pagination is restricted to companies that belong to the caller’s workspace.
- `/metrics` (`app/main.py:1081`) also sets the tenant GUC and includes explicit `tenant_id = $1` predicates when reading `background_jobs`, `lead_scores`, and `run_event_logs`, ensuring each dashboard card reflects only the signed-in tenant.

## UI Diagram

![Candidates & Metrics architecture](./candidates_metrics_overview.svg)

**Callouts**
- `AppShell` has no direct DB effect; it frames the experience while the other widgets drive API traffic.
- `IndustryJobLauncher` writes to `background_jobs` immediately; later the worker reads `ssic_ref` and `staging_acra_companies` and updates `companies`.
- `CandidatesPanel` is a pure read against `companies` for the tenant.
- `JobsProgress` polls `background_jobs` to keep its bar current.
- `Metrics Dashboard` reads `background_jobs`, `lead_scores`, and `run_event_logs`, while optional TTFB submissions add rows to `run_event_logs`.

## Code Reference

- Backend FastAPI server: `app/main.py` (functions: `candidates_latest`, `metrics`, `metric_ttfb`, `jobs_staging_upsert`, `jobs_status`)
- Database utility: `src/database.py` (functions: `get_pg_pool`, `get_conn`)
- Auth helpers: `app/auth.py` (functions: `require_auth`, `require_identity`, `require_optional_identity`)
- Background job enqueuing & runner: `src/jobs.py` (functions: `_insert_job`, `enqueue_staging_upsert`, `run_staging_upsert`)
- Staging upsert implementation: `app/lg_entry.py` (function: `_upsert_companies_from_staging_by_industries`)
- ICP/SSIC helpers: `src/icp.py` (function: `_find_ssic_codes_by_terms`)
- Candidates pagination test: `tests/test_pagination_candidates.py` (test: `test_candidates_pagination_next_cursor_and_no_offset`)
- Frontend shell/navigation: `agent-chat-ui/src/components/ui/app-shell.tsx` (component: `AppShell`)
- Candidates page: `agent-chat-ui/src/app/candidates/page.tsx` (component: `CandidatesPage`)
- Candidates list component: `agent-chat-ui/src/components/CandidatesPanel.tsx` (component: `CandidatesPanel`)
- Industry job launcher: `agent-chat-ui/src/components/IndustryJobLauncher.tsx` (component: `IndustryJobLauncher`)
- Job progress widget: `agent-chat-ui/src/components/JobsProgress.tsx` (component: `JobsProgress`)
- Metrics page: `agent-chat-ui/src/app/metrics/page.tsx` (component: `MetricsPage`)
- Authenticated fetch hook: `agent-chat-ui/src/lib/useAuthFetch.ts` (hook: `useAuthFetch`)
