# Feature 18 Runbook — Responsive ICP Search + Background Upsert

This runbook summarizes env flags, migrations, nightly runner, and key endpoints.

## Env Flags
- `STAGING_UPSERT_MODE` (background|off|sync_preview) — default: background
- `UPSERT_SYNC_LIMIT` — default: 10 (synchronous upserts per request)
- `UPSERT_MAX_PER_JOB` — default: 2000 (cap per background job)
- `STAGING_BATCH_SIZE` — default: 500 (server-side cursor batch)
- `LLM_MAX_CHUNKS` — default: 2 (limit chunks per company for extraction)
- `LLM_CHUNK_TIMEOUT_S` — default: 30 (seconds per chunk)
- `MERGE_DETERMINISTIC_TIMEOUT_S` — default: 10 (seconds for deterministic merge)
- Frontend proxy: `NEXT_BACKEND_TIMEOUT_MS` — default: 600000 (10 min)

## Migrations (DB Indexes)
Apply `lead_generation-main/app/migrations/010_icp_perf.sql` in your migration pipeline.
- Creates hot-path indexes on `companies` and sorted index for `lead_scores`.
- CI/CD: run `make app-migrations` (calls `scripts/run_app_migrations.py`) before starting the app.

## Nightly Runner
- Script: `lead_generation-main/scripts/run_nightly.py`
  - Process all queued jobs: `python scripts/run_nightly.py`
  - Process a max number: `python scripts/run_nightly.py --limit 20`
- Cron example (1 AM daily):
```
0 1 * * * cd /path/to/lead_generation-main && /usr/bin/env python -m scripts.run_nightly >> /var/log/leadgen-nightly.log 2>&1
```
- Systemd timer (user): see `scripts/systemd/leadgen-nightly.service` and `scripts/systemd/leadgen-nightly.timer` then:
  - `systemctl --user enable --now leadgen-nightly.timer`

## Endpoints
- POST `/jobs/staging_upsert` → `{ job_id }`
- GET `/jobs/{job_id}` → `{ status, processed, total, error? }`
- GET `/scores/latest?limit&afterScore&afterId` — keyset pagination on scores
- GET `/candidates/latest?limit&afterUpdatedAt&afterId&industry=...` — keyset pagination on companies (optional industry filter)
- GET `/metrics` — metrics: job queue depth, jobs processed total, lead scores total, rows/min (avg recent), p95 job time, chat TTFB p95
- POST `/metrics/ttfb` — FE can report chat TTFB `{ ttfb_ms }`

## Frontend Hooks/Components
- Debounce hook: `agent-chat-ui/src/hooks/useDebouncedFetch.ts`
- Job polling: `agent-chat-ui/src/hooks/useJobPolling.ts`
- Progress widget: `agent-chat-ui/src/components/JobsProgress.tsx`
- Virtualized list: `agent-chat-ui/src/components/VirtualList.tsx`

## Notes
- Request path upserts up to 10 companies and immediately triggers enrichment for those 10 (non-blocking). Remaining work is queued for nightly.
- Use `/jobs/*` endpoints to show status and progress.
- For long lists, prefer the `VirtualList` component to keep UI responsive.
