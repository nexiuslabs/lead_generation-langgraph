# Feature PRD Dev Plan: Troubleshooting‑Only Logging (FE + BE)

Owner: Engineering (Platform/Infra)
Status: Implementation Plan (v1.0)
Scope: Frontend (Next.js) + Backend (FastAPI) + Minimal Ingestion

---

## 1) Architecture Overview

- Frontend (Next.js app `agent-chat-ui`)
  - Capture: unhandled errors, unhandled rejections, failed network calls (non‑2xx, timeout), SSE disconnects.
  - Context: `session_id`, hashed `user_id` (if available), `release`, `environment`, `route`, `component`, `device`.
  - Transport: `navigator.sendBeacon` (preferred) → `/v1/logs`; fallback single `POST` with `credentials: 'include'`.
  - Backpressure: in‑memory rate‑limit + small dedupe window to avoid storms.

- Backend (FastAPI app in `lead_generation-main`)
  - Ingestion endpoint: `POST /v1/logs` accepts single or batch; validates, sanitizes, enforces policy (warn/error only in prod unless Diagnostic Mode active), and writes structured JSON to stdout.
  - Request correlation middleware: assign `request_id`, parse `traceparent` → `trace_id` else generate.
  - Error logging: unhandled exceptions and HTTP 5xx with correlation.
  - Optional: HMAC signature header for FE requests.

- Sink/Retention
  - Keep stdout/stderr (12‑factor). Rely on platform logs; optional gzip JSONL archive path documented.

---

## 2) Frontend Implementation (Next.js `agent-chat-ui`)

### 2.1 Files & Modules

- `agent-chat-ui/src/lib/troubleshoot-logger.ts`
- `agent-chat-ui/src/app/providers/ClientInit.tsx` (or equivalent provider) to install listeners once.
- Use existing fetch wrappers: `src/lib/useAuthFetch.ts`, `src/providers/Stream.tsx`, `src/providers/ChatProgress.tsx` to hook into failures.

### 2.2 Minimal Logger Utility

```ts
// agent-chat-ui/src/lib/troubleshoot-logger.ts
/* Lightweight client logger strictly for troubleshooting */

const ENV = (process.env.NEXT_PUBLIC_ENVIRONMENT || process.env.NODE_ENV || 'dev');
const SERVICE = 'web';
const RELEASE = process.env.NEXT_PUBLIC_RELEASE || '';
const INGEST = process.env.NEXT_PUBLIC_USE_API_PROXY === 'true'
  ? '/api/backend/v1/logs'
  : (process.env.NEXT_PUBLIC_API_URL || '') + '/v1/logs';

// Simple leaky‑bucket rate limiter (≈2 ev/s avg, 10 burst)
let tokens = 10; let last = Date.now();
function allowed(): boolean {
  const now = Date.now();
  tokens = Math.min(10, tokens + ((now - last) / 1000) * 2);
  last = now; if (tokens < 1) return false; tokens -= 1; return true;
}

// 30s dedupe for identical signatures
const recent = new Set<string>();
function dedupeKey(e: any): string {
  return [e.level, e.message, e.component, e.error?.type, e.error?.stack?.[0]].join('|');
}
function keep(e: any): boolean {
  const key = dedupeKey(e);
  if (recent.has(key)) return false;
  recent.add(key); setTimeout(() => recent.delete(key), 30_000); return true;
}

// Prefer sendBeacon; fall back to fetch POST
async function send(events: any[]) {
  const payload = JSON.stringify({ events });
  const headerName = 'x-log-signature';
  const hmac = process.env.NEXT_PUBLIC_LOG_INGEST_HMAC || '';
  // best‑effort beacon first
  try {
    if (navigator.sendBeacon) {
      const blob = new Blob([payload], { type: 'application/json' });
      const ok = navigator.sendBeacon(INGEST, blob);
      if (ok) return;
    }
  } catch {}
  // fallback fetch
  try {
    await fetch(INGEST, {
      method: 'POST',
      credentials: 'include',
      headers: {
        'content-type': 'application/json',
        ...(hmac ? { [headerName]: hmac } : {}),
      },
      body: payload,
      keepalive: true,
    });
  } catch {}
}

export type FELog = {
  level: 'error' | 'warn';
  message: string;
  component?: string;
  session_id?: string;
  route?: string;
  http?: { method?: string; host?: string; status?: number; duration_ms?: number };
  error?: { type?: string; message?: string; stack?: string[] };
  data?: Record<string, unknown>;
};

export function logEvent(ev: FELog) {
  if (!allowed()) return;
  const event = {
    timestamp: new Date().toISOString(),
    level: ev.level,
    service: SERVICE,
    environment: ENV,
    release: RELEASE,
    message: String(ev.message || '').slice(0, 512),
    session_id: ev.session_id,
    component: ev.component,
    route: ev.route,
    http: ev.http,
    error: ev.error,
    data: ev.data,
  };
  if (!keep(event)) return;
  void send([event]);
}

export function installGlobalErrorHandlers(getCtx?: () => { session_id?: string; route?: string }) {
  window.addEventListener('error', (e) => {
    const ctx = getCtx?.() || {};
    logEvent({
      level: 'error',
      message: e?.error?.message || e?.message || 'Unhandled error',
      component: 'window.onerror',
      session_id: ctx.session_id,
      route: ctx.route,
      error: {
        type: e?.error?.name || 'Error',
        message: e?.error?.message || e?.message,
        stack: (e?.error?.stack || '').split('\n').slice(0, 6),
      },
    });
  });
  window.addEventListener('unhandledrejection', (e: PromiseRejectionEvent) => {
    const reason: any = (e && (e.reason ?? e)) || {};
    const ctx = getCtx?.() || {};
    logEvent({
      level: 'error',
      message: reason?.message || 'Unhandled rejection',
      component: 'window.unhandledrejection',
      session_id: ctx.session_id,
      route: ctx.route,
      error: {
        type: reason?.name || 'UnhandledRejection',
        message: reason?.message || String(reason),
        stack: String(reason?.stack || '').split('\n').slice(0, 6),
      },
    });
  });
}
```

### 2.3 Install Once (Client Provider)

```tsx
// agent-chat-ui/src/app/providers/ClientInit.tsx
'use client';
import { useEffect } from 'react';
import { installGlobalErrorHandlers, logEvent } from '@/lib/troubleshoot-logger';
import { usePathname } from 'next/navigation';

export default function ClientInit({ sessionId }: { sessionId?: string }) {
  const route = usePathname();
  useEffect(() => {
    installGlobalErrorHandlers(() => ({ session_id: sessionId, route }));
  }, [sessionId, route]);
  // Optionally capture EventSource errors from chat stream components by calling logEvent
  return null;
}
```

Add to your root layout or top‑level shell so it mounts once per session.

### 2.4 Network Failure Hooks

- `useAuthFetch` and stream providers already centralize calls; add `.catch()` to log failed requests with `host`, `status`, `duration_ms`.
- For SSE, catch `EventSource` error callbacks and call `logEvent({ level: 'warn', message: 'SSE error', component: 'chat.sse', ... })`.

---

## 3) Backend Implementation (FastAPI `lead_generation-main`)

### 3.1 Request Correlation Middleware

Add a middleware to assign `request_id` and parse W3C trace context.

```py
# lead_generation-main/app/middleware_request_id.py
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
import uuid, re

TRACE_RE = re.compile(r"traceparent: (?:00-)?([0-9a-f]{32})-([0-9a-f]{16})-(?:0[0-9])", re.I)

class CorrelationMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        req_id = request.headers.get('x-request-id') or f"r-{uuid.uuid4().hex[:16]}"
        trace_id = request.headers.get('traceparent')
        if trace_id and '-' in trace_id:
            try:
                trace_id = trace_id.split('-')[1]
            except Exception:
                trace_id = None
        request.state.request_id = req_id
        request.state.trace_id = trace_id or f"t-{uuid.uuid4().hex[:16]}"
        response: Response = await call_next(request)
        response.headers['x-request-id'] = req_id
        response.headers['x-trace-id'] = request.state.trace_id
        return response
```

Wire into `app/main.py` (after app creation):

```py
from app.middleware_request_id import CorrelationMiddleware
app.add_middleware(CorrelationMiddleware)
```

### 3.2 Structured Logging Setup

Configure a JSON formatter once (optional but recommended). If you keep current logging, ensure error logs include `request_id`/`trace_id` from `request.state`.

### 3.3 Ingestion Router `/v1/logs`

```py
# lead_generation-main/app/logs_routes.py
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Any
import os, hashlib, time, json

router = APIRouter(prefix="/v1/logs", tags=["logs"]) 

ALLOWED_LEVELS = {"error","warn","info","debug"}
ENV = (os.getenv("ENVIRONMENT") or os.getenv("PY_ENV") or os.getenv("NODE_ENV") or "dev").lower()
HMAC_SECRET = os.getenv("LOG_INGEST_HMAC_SECRET")

class ErrorModel(BaseModel):
    type: Optional[str] = Field(default=None, max_length=64)
    message: Optional[str] = Field(default=None, max_length=256)
    stack: Optional[List[str]] = None

class HttpModel(BaseModel):
    method: Optional[str] = Field(default=None, max_length=8)
    route: Optional[str] = Field(default=None, max_length=128)
    host: Optional[str] = Field(default=None, max_length=128)
    status: Optional[int] = None
    duration_ms: Optional[int] = None

class LogEvent(BaseModel):
    timestamp: str
    level: str
    service: str
    environment: str
    release: str
    message: str
    trace_id: Optional[str] = None
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    component: Optional[str] = None
    error: Optional[ErrorModel] = None
    http: Optional[HttpModel] = None
    data: Optional[dict[str, Any]] = None

class LogBatch(BaseModel):
    events: List[LogEvent]

# Simple leaky bucket per IP
_buckets: dict[str, tuple[float, float]] = {}  # ip -> (last_ts, tokens)

def _rate_limit(ip: str) -> bool:
    now = time.time()
    ts, tokens = _buckets.get(ip, (now, 20.0))
    refill = (now - ts) * 5.0  # 5 ev/s avg
    tokens = min(20.0, tokens + refill)
    if tokens < 1.0:
        _buckets[ip] = (now, tokens)
        return False
    tokens -= 1.0
    _buckets[ip] = (now, tokens)
    return True

def _sanitize_data(d: Optional[dict]) -> Optional[dict]:
    if not isinstance(d, dict):
        return None
    allow = {"route","route_template","host","method","status","duration_ms","component","pathname"}
    block_keys = ("authorization","cookie","token","secret","password","set-cookie")
    out: dict[str, Any] = {}
    for k, v in d.items():
        kl = str(k).lower()
        if any(b in kl for b in block_keys):
            continue
        if k in allow:
            if isinstance(v, str):
                vv = v
                vv = vv.rsplit('?', 1)[0]  # strip query
                out[k] = vv[:256]
            else:
                out[k] = v
    return out or None

@router.post("")
async def ingest(request: Request, body: Any):
    # Rate limit per IP
    ip = request.client.host if request.client else "unknown"
    if not _rate_limit(ip):
        raise HTTPException(status_code=429, detail="rate_limited")
    # Optional HMAC check (best effort)
    if HMAC_SECRET:
        sig = request.headers.get("x-log-signature")
        if not sig:
            raise HTTPException(status_code=401, detail="missing_signature")
        try:
            raw = await request.body()
            mac = hashlib.sha256()
            mac.update(raw)
            calc = mac.hexdigest()
            if sig != calc:
                raise HTTPException(status_code=401, detail="bad_signature")
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(status_code=400, detail="bad_request")
    # Normalize to batch
    try:
        events: List[LogEvent]
        if isinstance(body, list):
            events = [LogEvent.model_validate(e) for e in body]
        elif isinstance(body, dict) and isinstance(body.get("events"), list):
            events = [LogEvent.model_validate(e) for e in body.get("events")]
        else:
            events = [LogEvent.model_validate(body)]
    except Exception:
        raise HTTPException(status_code=400, detail="invalid_payload")
    # Prod policy: no info/debug unless Diagnostic Mode (per‑session) — stub: allow only warn/error
    if ENV == 'prod':
        for e in events:
            if e.level in ('info','debug'):
                raise HTTPException(status_code=400, detail="level_not_allowed")
    # Enrich & sanitize
    out = []
    for e in events:
        e.request_id = e.request_id or getattr(request.state, 'request_id', None)
        e.trace_id = e.trace_id or getattr(request.state, 'trace_id', None)
        e.data = _sanitize_data(e.data)
        out.append(json.loads(e.model_dump_json()))
    # Emit to stdout as structured lines (respecting existing logging setup)
    logger = __import__('logging').getLogger('troubleshoot')
    for rec in out:
        try:
            logger.warning(json.dumps(rec, ensure_ascii=False))
        except Exception:
            pass
    return {"accepted": len(out)}
```

Wire in router in `app/main.py`:

```py
try:
    from app.logs_routes import router as logs_router
    app.include_router(logs_router)
    logger.info("/v1/logs endpoint enabled")
except Exception as _e:
    logger.warning("Logs router not mounted: %s", _e)
```

### 3.4 Backend Exception Logging

Add a global exception handler (optional) to log unhandled exceptions with `request_id`/`trace_id` and route. Ensure 5xx responses are also logged by existing logger.

```py
from fastapi.responses import JSONResponse
from fastapi import Request
import logging, json, os

@app.exception_handler(Exception)
async def on_exception(request: Request, exc: Exception):
    log = logging.getLogger('api')
    try:
        log.error(
            json.dumps({
                'level': 'error',
                'service': 'api',
                'environment': os.getenv('ENVIRONMENT','dev'),
                'release': os.getenv('RELEASE',''),
                'message': f"Unhandled exception on {request.method} {request.url.path}",
                'trace_id': getattr(request.state,'trace_id',None),
                'request_id': getattr(request.state,'request_id',None),
                'error': {'type': type(exc).__name__, 'message': str(exc)},
            })
        )
    except Exception:
        pass
    return JSONResponse(status_code=500, content={"detail": "internal_error"})
```

### 3.5 LangGraph Run Instrumentation

Instrument all LangGraph orchestrations so every node checkpoint is persisted in the troubleshooting pipeline. Extend `/v1/logs` to accept `info` level events when `service == 'langgraph'` so successful transitions are retained in production while other `info/debug` events remain blocked.

```py
# lead_generation-main/app/langgraph_logging.py
import json, os
from datetime import datetime
from langgraph.callbacks.base import BaseCallbackHandler
from typing import Any, Callable

class LangGraphTroubleshootHandler(BaseCallbackHandler):
    def __init__(self, emit: Callable[[dict[str, Any]], None]):
        self.emit = emit

    def _record(self, run_id: str, node: str, status: str, payload: dict[str, Any]):
        self.emit({
            "timestamp": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
            "level": "info" if status == "end" else "warn",
            "service": "langgraph",
            "environment": os.getenv("ENVIRONMENT", "dev"),
            "run_id": run_id,
            "node": node,
            "status": status,
            "payload": payload,
        })

    def on_node_start(self, run_id: str, node: str, inputs: dict[str, Any], **kwargs):
        self._record(run_id, node, "start", {"inputs": inputs})

    def on_node_end(self, run_id: str, node: str, outputs: dict[str, Any], metadata: dict[str, Any] | None = None, **kwargs):
        payload = {"outputs": outputs}
        if metadata:
            payload["metadata"] = metadata
        self._record(run_id, node, "end", payload)

    def on_node_error(self, run_id: str, node: str, error: Exception, **kwargs):
        self._record(run_id, node, "error", {"error_type": type(error).__name__, "error_message": str(error)})
```

Register the handler where graphs are constructed so it forwards into `/v1/logs` with the current job metadata (`session_id`, `request_id`, `job_id`):

```py
from app.langgraph_logging import LangGraphTroubleshootHandler
from app.logging import log_event  # thin wrapper over /v1/logs

graph = build_graph(...)
graph.add_callback_handler(
    LangGraphTroubleshootHandler(lambda event: log_event(service="langgraph", **event))
)
```

---

## 4) Diagnostic Mode (Secure, Scoped, Optional)

- Token: HS256 JWT cookie `diag` signed by `LOG_DIAGNOSTIC_SECRET`, contains `session_id` and `exp` (≤ 60 min).
- Server gate: If cookie valid and matches incoming event `session_id`, allow `info/debug` levels; otherwise reject these in prod.
- For first iteration, ship without Diagnostic Mode enabled; add later with a small dependency (PyJWT) and a request dependency to set `request.state.diagnostic_mode = True` when valid.

---

## 5) Process Flows

- Frontend error flow:
  - Component throws → `window.onerror` captures → build event with `session_id` + route → rate‑limit + dedupe → sendBeacon to `/v1/logs` → server validates + enriches → writes JSON line to stdout.

- Frontend network error flow:
  - `useAuthFetch` call fails (timeout or non‑2xx) → compose event with method/host/status/duration → send via logger → `/v1/logs` → stdout.

- Backend 5xx flow:
  - Exception bubbles to handler → logs structured event with `request_id`/`trace_id` → returns 500.

- Correlation:
  - The FE includes `session_id` (already used for chat SSE). BE assigns `request_id` and `trace_id` for API calls. Troubleshooting joins by any of these IDs.

- LangGraph automation flow:
  - LangGraph handler records node start/end/error → forwards structured event (run_id, graph_id, node, status, duration, summaries) to `/v1/logs` → backend enriches with `request_id`/`job_id` → DigitalOcean pipeline ships for dev/prod inspection.

- ACRA Direct flow:
  - On-demand trigger seeds `job_id` + `customer_id` → LangGraph instrumentation emits node events + final summary → `/v1/logs` writes to stdout → droplet shipper persists and alerts on failures.

- Nightly ACRA flow:
  - Cron (02:00 UTC) kicks nightly aggregate job → start event logs cohort + snapshot hash → each LangGraph node produces telemetry → completion summary captures totals and warnings → shipping identical across environments.

- Background Next 40 flow:
  - Background service polls `next40` queue → when dispatching conversations it logs acceptance metadata, LangGraph node transitions, and completion metrics → success emits `info` summary, failures emit `warn/error` with correlation IDs for support.

---

## 6) Local Development Logging

- Default sinks: When running locally, the Next.js frontend persists structured troubleshooting logs to the relative `./.log` directory, and the FastAPI backend writes to `./.log_api`. Both folders are added to `.gitignore` to prevent accidental commits.

- Configurable paths: Expose `TROUBLESHOOT_FE_LOG_DIR` (frontend) and `TROUBLESHOOT_API_LOG_DIR` (backend) in the respective `.env.local` / `.env` files so developers can point logs to alternate directories (for example, `/tmp/logs/frontend`). The runtime should create the directory if it does not exist.

- File format & rotation: Frontend logger appends newline-delimited JSON (`*.jsonl`) per session, while the backend appends JSON to `api.log`. Lightweight local rotation can rely on `maxFileSize` (frontend) and `logging.handlers.RotatingFileHandler` (backend) to cap files at 10 MB with three backups.

- Developer ergonomics: Document VS Code tasks or npm scripts (`npm run logs:tail`) that tail the configured directories, enabling quick inspection without scraping console output.

---

## 7) DigitalOcean Deployment Logging

- Droplet split: Frontend (`do-fe-agent-chat-ui`) and backend (`do-be-lead-generation`) run on separate droplets; both processes log to stdout/stderr and mirror critical files under `/var/log/agent-chat-ui` and `/var/log/lead-generation` for quick SSH inspection in dev and prod.

- Local retention: Convert app processes to `systemd` units with `StandardOutput=journal` and use `logrotate` to keep mirrored files fresh:

```conf
/var/log/agent-chat-ui/*.log /var/log/lead-generation/*.log {
    daily
    rotate 14
    compress
    missingok
    copytruncate
}
```

- Log shipping: Install `do-agent` (or `td-agent-bit`) on each droplet to forward `journald` streams and the above log files to DigitalOcean Log Forwarding (Spaces/S3/Logtail). Tag records with `service=frontend|backend`, `environment=dev|prod`, and `droplet_id` for traceability.

- Retention policy: Dev droplets retain 7 days locally and 14 days remotely; prod droplets retain 14 days locally and 30 days remotely. Document SLAs and alerts for stalled shipping (DigitalOcean monitoring or Grafana Loki heartbeat).

- Verification: `journalctl -u agent-chat-ui --since "15 minutes ago"`, `journalctl -u lead-generation-api --since "15 minutes ago"`, confirm remote sink ingestion latency (<5 min), and hit `/v1/logs/health` to validate backend acceptance.

---

## 8) Environment & Config

- Frontend:
  - `NEXT_PUBLIC_USE_API_PROXY=true` (recommended; avoids CORS for logs)
  - `NEXT_PUBLIC_API_URL=https://api.example.com` (when not proxying)
  - `NEXT_PUBLIC_ENVIRONMENT=staging|prod`
  - `NEXT_PUBLIC_RELEASE=2025.11.03+abcd`
  - `NEXT_PUBLIC_LOG_INGEST_HMAC=<optional-hmac-hex>`
  - `TROUBLESHOOT_FE_LOG_DIR=.log` (development default; override in `.env.local`).

- Backend:
  - `ENVIRONMENT=dev|staging|prod`
  - `RELEASE=2025.11.03+abcd`
  - `LOG_INGEST_HMAC_SECRET` (optional)
  - `LOG_LEVEL=warn` (prod)
  - `EXTRA_CORS_ORIGINS` (only needed if not proxying)
  - `TROUBLESHOOT_API_LOG_DIR=.log_api` (development default; override in `.env`).

---

## 9) Testing & Validation

- Unit tests (BE):
  - Validate `LogEvent` schema; reject `info/debug` in prod; enforce size/length.
  - Sanitizer: emails/phones/tokens redacted; query strings removed.
  - Rate limiter hits 429 appropriately.

- Manual tests (FE):
  - Trigger `throw new Error()` in a component → event shows in server logs.
  - Simulate network error (point to bad host) → network failure logged.
  - SSE drop → `warn` event recorded.

- E2E: run locally with proxy (`/api/backend`) to ensure no CORS/cookie issues.

---

## 10) Rollout Plan

- Week 1: Backend router + middleware + exception handler. Deploy to staging.
- Week 2: Frontend logger + global handlers + fetch/SSE hooks. Gate by env; deploy to staging.
- Week 3: Add Diagnostic Mode gate and FE toggle; redaction unit tests.
- Week 4: Tune limits; author docs for support; enable in production.

---

## 11) Code Placement Summary

- FE:
  - `agent-chat-ui/src/lib/troubleshoot-logger.ts`
  - `agent-chat-ui/src/app/providers/ClientInit.tsx` (mounted from app shell)
  - small additions in `useAuthFetch.ts`, `Stream.tsx`, `ChatProgress.tsx` to log failures

- BE:
  - `lead_generation-main/app/middleware_request_id.py`
  - `lead_generation-main/app/logs_routes.py`
  - `app/main.py` include router and (optional) exception handler

---

## 12) Notes & Constraints

- No bodies or headers are ever ingested; only allowlisted metadata is stored (host, route, method, status, duration).
- Prod policy restricts levels (`warn`/`error`), preventing inadvertent verbose logging.
- HMAC signature is best‑effort to reduce spoofing; full auth is intentionally not required to keep troubleshooting friction low.
- Keep FE logger tiny and best‑effort. Dropped events are acceptable.

*** End of Plan ***
