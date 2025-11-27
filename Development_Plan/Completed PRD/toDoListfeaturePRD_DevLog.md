# To‑Do List — Troubleshooting‑Only Logging (Feature PRD Dev Log)

Reference: featurePRD_DevLog.md
Scope: Frontend (Next.js) + Backend (LangGraph/optional FastAPI) + Minimal ingestion

## Status Summary
- Frontend local ingestion via `/api/logs`: Completed
- Frontend client logger + global handlers: Completed
- Network/SSE failure hooks: Completed (minimal)
- Backend (FastAPI) correlation + `/v1/logs`: Completed (optional if not used)
- Frontend local file sink (`TROUBLESHOOT_FE_LOG_DIR`): Completed
- Diagnostic Mode (info/debug in prod): Completed
- JSON formatting for global server exceptions: Completed
- Unit tests for redaction/rate limits: Completed
- `/v1/logs/health` endpoint + monitoring checks: Completed
- DigitalOcean deployment logging (systemd, logrotate, shipping): Completed (shipping agent optional)

## Checklist — Frontend (agent‑chat‑ui)
- [x] Add lightweight client logger with rate‑limit + brief dedupe
  - File: `src/lib/troubleshoot-logger.ts`
  - Sends to `POST /api/logs` (UI server) to support LangGraph‑only backends
- [x] Install global error handlers once per session
  - File: `src/app/providers/ClientInit.tsx` (mounted in `src/app/layout.tsx`)
- [x] Hook network failures (timeouts, non‑2xx, 401 retries)
  - File: `src/lib/useAuthFetch.ts` — logs exceptions and final 401s with method/host/duration
- [x] Hook SSE errors (EventSource on chat progress)
  - File: `src/providers/ChatProgress.tsx` — logs `'SSE error'` (warn)
- [x] Add Next API route to ingest logs and emit JSONL to stdout
  - File: `src/app/api/logs/route.ts` — sanitizes allowlisted fields, prints one JSON line per event
- [x] Diagnostic Mode toggle (client) — enable info/debug for 60 min via signed cookie
  - Files: `src/components/ui/app-shell.tsx`, `src/lib/troubleshoot-logger.ts`
- [x] Local JSONL sink honoring `TROUBLESHOOT_FE_LOG_DIR` with rotation + docs update
  - File(s): `src/app/api/logs/route.ts`, docs in `docs/troubleshoot_logging.md`

## Checklist — Backend (optional FastAPI app)
- [x] Correlation middleware (`x-request-id`, `x-trace-id`) and request.state binding
  - File: `lead_generation-main/app/middleware_request_id.py`
- [x] `/v1/logs` router: accept single/batch, rate limit per IP, optional HMAC, sanitize, emit JSONL
  - File: `lead_generation-main/app/logs_routes.py`
- [x] Wire middleware + router into FastAPI app
  - File: `lead_generation-main/app/main.py`
- [x] Global exception handler that logs structured JSON for 5xx with correlation
  - File: `app/main.py`
- [x] `/v1/logs/health` endpoint for readiness/monitoring responses
  - File: `lead_generation-main/app/logs_routes.py`
- [ ] Optional gzip JSONL archive (S3/GCS) — Not Implemented

## Policy / Privacy / Config
- [x] FE: allowlisted `data` keys only; no bodies/headers; strip query strings
- [x] FE: rate‑limit (~2 ev/s avg, 10 burst) and 30s dedupe window
- [x] BE (FastAPI): per‑IP leaky‑bucket; prod rejects `info/debug`
- [x] Diagnostic Mode (JWT cookie `diag`) to allow info/debug in prod
- [x] Unit tests: sanitizer and policy gates
- [x] Environment variable matrix documented for FE (`NEXT_PUBLIC_*`, `TROUBLESHOOT_FE_LOG_DIR`) and BE (`LOG_INGEST_HMAC_SECRET`, `TROUBLESHOOT_API_LOG_DIR`, etc.)

## Deployment & Ops (DigitalOcean)
- [x] Convert FE/BE services to `systemd` units with journald + logrotate policy (`/var/log/agent-chat-ui`, `/var/log/lead-generation`)
- [ ] Install log shipping agent (`do-agent` or `td-agent-bit`) with environment/service tagging
- [x] Document retention policy (dev vs prod) and verification steps (`journalctl`, remote sink latency, `/v1/logs/health`)

## Developer Ergonomics & Tooling
- [x] Provide local tail scripts/tasks (e.g., `npm run logs:tail`) for `.log` / `.log_api`
- [x] Document how to enable/disable proxy routing (`NEXT_PUBLIC_USE_API_PROXY`) for log troubleshooting
## How To Verify (Local)
- Start UI: `pnpm build && pnpm start` (in `agent-chat-ui`)
  - Post a test event:
    - `fetch('/api/logs',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({events:[{timestamp:new Date().toISOString(),level:'warn',service:'web',environment:'dev',release:'',message:'manual test'}]})}).then(r=>r.json()).then(console.log)`
  - Expect `{ accepted: 1 }` in browser, JSON line in Next terminal
- Trigger client error: `setTimeout(()=>{throw new Error('boom')},0)` → JSON line
- Cause network error (bad path) → logged with method/host/duration
- SSE: temporarily disable backend → `'SSE error'` warn event

## Next Steps (Optional)
- [ ] Install log shipping agent (e.g., `td-agent-bit`) for centralized aggregation
- [ ] Offload JSONL archives to object storage (S3/GCS) for long-term retention
