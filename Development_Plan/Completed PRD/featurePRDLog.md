# Feature PRD: Troubleshooting-Only Logging

Owner: Engineering (Platform/Infra)
Stakeholders: Frontend, Backend, Support, Security
Status: Draft (v0.2)
Target Release: Lightweight rollout (see Rollout)

## Summary

Provide minimal, structured, privacy-safe logging across frontend and backend strictly for troubleshooting incidents and bugs. The scope focuses on capturing errors and critical warnings with enough context to reproduce issues, plus lightweight request correlation. No analytics, RUM, performance dashboards, or audit-compliance logging are included.

## Goals

- Capture errors and critical warnings in FE and BE with structured JSON.
- Correlate FE↔BE via `trace_id`/`request_id`/`session_id` for end-to-end debugging.
- Keep overhead low: no performance metrics, no high-volume info/debug in prod.
- Privacy-first: redact PII/secrets by default; scrub URLs and headers.
- Simple ingestion path that fits existing infrastructure.
- Optional on-demand Diagnostic Mode to temporarily increase verbosity per session.

## Non-Goals

- No product analytics, RUM, Core Web Vitals, or business event tracking.
- No dashboards/alerts or SRE observability features.
- No audit/SIEM or long-term compliance logging.
- No long-term retention beyond troubleshooting needs.

## Success Metrics

- 100% of backend 5xx and unhandled exceptions logged with correlation IDs.
- ≥ 95% of frontend unhandled errors captured with release and route context.
- Zero known PII/secret leaks in logs after rollout (automated checks pass).
- Mean time-to-diagnose (MTTD) for P1/P2 issues reduced measurably post-rollout.

## Scope

### Frontend (Web/App)

- Capture unhandled errors and unhandled promise rejections with component and route info.
- Capture failed network calls (non-2xx/timeout/DNS) with method, host, status, duration.
- Minimal context: `session_id`, anonymized `user_id` (hash), `release`, `env`, `route`, `device`.
- Transport: `navigator.sendBeacon` where available; fallback to batched POST to `/v1/logs`.
- Reliability: best-effort only (no offline queue). One retry max; drop on failure.
- Rate control: ≤ 2 events/sec average, burst ≤ 10 per session.
- Levels: `error` and `warn` only in production; `info/debug` available in Diagnostic Mode.
- Diagnostic Mode: opt-in per-session toggle (feature flag or signed token) active ≤ 1 hour.

### Backend (APIs/Services/Jobs)

- Structured JSON logs to stdout/stderr (12-factor). Use existing collector/agent if any.
- Middleware: generate `request_id`; parse/propagate W3C `traceparent`; create new trace if absent.
- Log on failures only: unhandled exceptions (with limited stack), 5xx responses, failed outbound calls.
- Outbound failures: log method, host, status, duration, retry count. Never log bodies/headers.
- DB issues: log slow query hits and errors with table and duration only (no SQL/params).

## Minimal Schema

JSON event with required base fields (others optional):

```json
{
  "timestamp": "2025-01-01T12:34:56.789Z",
  "level": "error",
  "service": "web|api|worker",
  "environment": "dev|staging|prod",
  "release": "2025.11.03+abcd",
  "message": "Human-readable summary",
  "trace_id": "<otel-trace-id>",
  "request_id": "<uuid>",
  "session_id": "<uuid>",
  "component": "<component/module>",
  "error": { "type": "ErrorName", "message": "...", "stack": ["frame1", "frame2"] },
  "http": { "method": "GET", "route": "/api/leads", "host": "api.example.com", "status": 500, "duration_ms": 123 },
  "data": { "k": "v" }
}
```

Conventions:
- ISO-8601 UTC timestamps; levels: `error`, `warn` (plus `info/debug` only in Diagnostic Mode).
- Keep `data` tiny and allowlisted. Do not include request/response bodies.
- If `traceparent` exists, derive `trace_id`; otherwise generate a new one.

## JSON Schema (v0.2)

Enforce structure and size; prod policy restricts levels (see below).

```json
{
  "$id": "https://nexius/troubleshoot.logevent.schema.json",
  "type": "object",
  "additionalProperties": false,
  "required": ["timestamp","level","service","environment","release","message"],
  "properties": {
    "timestamp": { "type": "string", "format": "date-time" },
    "level": { "enum": ["error","warn","info","debug"] },
    "service": { "enum": ["web","api","worker","ingestion"] },
    "environment": { "enum": ["dev","staging","prod"] },
    "release": { "type": "string", "maxLength": 64 },
    "message": { "type": "string", "maxLength": 512 },
    "trace_id": { "type": "string", "maxLength": 64 },
    "request_id": { "type": "string", "maxLength": 64 },
    "session_id": { "type": "string", "maxLength": 64 },
    "component": { "type": "string", "maxLength": 64 },
    "error": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "type": { "type": "string", "maxLength": 64 },
        "message": { "type": "string", "maxLength": 256 },
        "stack": {
          "type": "array",
          "maxItems": 6,
          "items": { "type": "string", "maxLength": 256 }
        }
      }
    },
    "http": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "method": { "type": "string", "maxLength": 8 },
        "route": { "type": "string", "maxLength": 128 },
        "host": { "type": "string", "maxLength": 128 },
        "status": { "type": "integer", "minimum": 100, "maximum": 599 },
        "duration_ms": { "type": "integer", "minimum": 0 }
      }
    },
    "data": { "type": "object", "maxProperties": 24, "additionalProperties": true }
  }
}
```

Prod gate (policy): when `environment = "prod"`, the server must reject events where `level ∈ {"info","debug"}` unless Diagnostic Mode is active.

## Privacy & Security

- Redact or hash: emails, phone numbers, tokens, cookies, auth headers, API keys.
- Scrub query strings; store only `host` and route templates (no full URLs).
- Hash IP addresses unless needed for security investigation, then store separately.
- Access to logs restricted to engineering/support. No customer-accessible logs.

## Ingestion (Simplified)

- Endpoint: `POST /v1/logs` (accepts array or single event). Only `error`/`warn` allowed in prod.
- Validation: required fields, size limits (FE ≤ 16 KB/event; BE ≤ 64 KB/event).
- Enrichment: server timestamp, hashed source IP, region, generated `request_id` if missing.
- Response: `202 Accepted` with counts; silently drop oversize or invalid events.

### Ingestion Hardening (Express Middleware)

```ts
// guards.ts
import crypto from "node:crypto";
import { Request, Response, NextFunction } from "express";

// 300KB body, FE 16KB/event, BE 64KB/event (already in PRD)
export function onlyWarnErrorInProd(req: Request, res: Response, next: NextFunction) {
  const env = process.env.NODE_ENV || "dev";
  const diag = Boolean((req as any).diagnostic_mode);
  const events = Array.isArray(req.body) ? req.body : req.body?.events;
  if (env === "prod" && !diag) {
    for (const e of events ?? []) {
      if (e?.level === "info" || e?.level === "debug") return res.status(400).json({ error: "level_not_allowed" });
    }
  }
  next();
}

// Optional HMAC to prevent spoofing/noise flooding
export function hmacVerify(req: Request, res: Response, next: NextFunction) {
  const secret = process.env.LOG_INGEST_HMAC_SECRET;
  if (!secret) return next();
  const sig = req.get("x-log-signature");
  if (!sig) return res.status(401).end();
  const mac = crypto.createHmac("sha256", secret).update(JSON.stringify(req.body)).digest("hex");
  try { crypto.timingSafeEqual(Buffer.from(sig), Buffer.from(mac)); } catch { return res.status(401).end(); }
  next();
}

// Simple per-IP leaky bucket
const buckets = new Map<string, { t: number; tokens: number }>();
export function rateLimit(req: Request, res: Response, next: NextFunction) {
  const key = req.ip || "unknown";
  const now = Date.now();
  const b = buckets.get(key) ?? { t: now, tokens: 20 };
  const refill = (now - b.t) / 1000 * 5; // 5 events/sec avg, 20 burst
  b.tokens = Math.min(20, b.tokens + refill);
  if (b.tokens < 1) return res.status(429).end();
  b.tokens -= 1; b.t = now; buckets.set(key, b); next();
}

// De-dup within a short window (storms)
const recent = new Set<string>();
export function dedupeBriefly(seconds = 30) {
  return (req: Request, _res: Response, next: NextFunction) => {
    const events = Array.isArray(req.body) ? req.body : req.body?.events;
    if (!Array.isArray(events)) return next();
    const keep = [] as any[];
    for (const e of events) {
      const key = hash(`${e.level}|${e.service}|${e.component}|${e.message}|${e.error?.type}|${e.error?.stack?.[0]}`);
      if (recent.has(key)) continue;
      recent.add(key); setTimeout(() => recent.delete(key), seconds * 1000);
      keep.push(e);
    }
    (req as any)._deduped_events = keep;
    next();
  };
}
function hash(s: string) { return crypto.createHash("sha256").update(s).digest("hex").slice(0,16); }
```

Wire in `/v1/logs` route: `rateLimit → hmacVerify → diagnosticModeCheck → onlyWarnErrorInProd → validate → enrich`.

## Config (env vars)

- `LOG_LEVEL` (default `warn` in prod)
- `LOG_INGEST_URL` (FE)
- `RELEASE`, `SERVICE`, `ENVIRONMENT`
- `LOG_DIAGNOSTIC_TOKEN_TTL_MIN` (default 60)
- `LOG_REDACTION_ALLOWLIST` (JSON string)
- `LOG_INGEST_HMAC_SECRET` (optional, enables HMAC verification)
- `LOG_DIAGNOSTIC_SECRET` (signs Diagnostic Mode JWT cookie)

## Diagnostic Mode — Secure & Scoped

- Trigger: feature flag in admin or a signed one-time link.
- Mechanism: `diag=1` cookie with JWT (HS256) containing `session_id`, optional `user_id`, and `exp = now + LOG_DIAGNOSTIC_TOKEN_TTL_MIN` (default 60 minutes).
- Server check: if JWT valid and matches incoming `session_id`, mark `req.diagnostic_mode = true` and allow `info/debug` for that session only.
- Audit: log security event “diagnostic_mode_enabled/disabled” with actor and scope.
- Auto-expire and provide an admin kill switch.

Verification stub:

```ts
import jwt from "jsonwebtoken";
export function diagnosticMode(req, _res, next) {
  const token = req.cookies?.diag;
  if (!token) return next();
  try {
    const payload = jwt.verify(token, process.env.LOG_DIAGNOSTIC_SECRET!) as any;
    if (payload.session_id && payload.session_id === req.body?.session_id) (req as any).diagnostic_mode = true;
  } catch { /* ignore invalid */ }
  next();
}
```

## Sanitizer (Allowlist; No Bodies/Queries)

```ts
// sanitizer.ts
const ALLOW = new Set(["route","route_template","host","method","status","duration_ms","component","pathname"]);
const BLOCK_KEYS = [/authorization/i, /cookie/i, /token/i, /secret/i, /password/i, /set-cookie/i];
const EMAIL = /\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b/gi;
const PHONE = /\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{4}\b/g;

export function sanitizeData(data?: Record<string, unknown>) {
  if (!data) return undefined;
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(data)) {
    if (!ALLOW.has(k)) continue;
    if (BLOCK_KEYS.some(rx => rx.test(k))) continue;
    out[k] = redact(v);
  }
  return out;
}
function redact(v: unknown): unknown {
  if (typeof v === "string") {
    return v.replace(/[?&][^#]*$/,"") // strip queries at tail if any slipped in
            .replace(EMAIL,"[email]")
            .replace(PHONE,"[phone]")
            .slice(0,256);
  }
  return v;
}
```

## Retention & Sink (Cost-Minimal)

- Default: 7 days hot in existing cloud logs (CloudWatch / Cloud Logging / Azure Monitor).
- Optional archive: Gzip JSONL to object storage (S3/GCS) under `logs/service=$SERVICE/dt=YYYY-MM-DD/hour=HH/*.json.gz`.
- Rehydration: ad-hoc via Athena/BigQuery/DuckDB only during incidents.

## Runbook (Troubleshooting)

- Ask affected user to click “Enable Diagnostics (1h)” (sets JWT cookie) and reproduce.
- Grab `request_id`/`trace_id` from error page or API response.
- In cloud logs, filter last 60 minutes by `request_id` OR `trace_id` OR `route_template`.
- If noisy, temporarily elevate service to `warn` level via feature flag, then revert.
- Closeout: disable Diagnostics; attach log excerpt to ticket; file postmortem if 5xx.

## Rollout

Week 1: Backend error logging + `request_id`/trace parsing middleware.
Week 2: Frontend unhandled error + failed network capture; sendBeacon transport.
Week 3: Diagnostic Mode (per-session toggle) + redaction tests.
Week 4: Tune rate limits; finalize docs.

## Open Questions (Troubleshooting Focus)

- Sink: Use existing cloud-native logs (e.g., CloudWatch/Cloud Logging) or current vendor? Recommendation: reuse what we already have to minimize setup.
- Diagnostic Mode trigger: feature flag in admin, signed link, or support-only header? Recommendation: feature flag + signed session cookie.
- Retention: 7 vs 14 days? Recommendation: 7 days hot is sufficient for troubleshooting; extend temporarily during incidents.
- Who can enable Diagnostic Mode? Recommendation: engineers and designated support only (audited).

## Acceptance Checklist

- FE: unhandled errors and failed network calls captured with release/route; rate limits enforced.
- BE: exceptions and 5xx logged with correlation; outbound failure logs in place.
- Redaction: unit tests for sanitizer; PII scan passes on sample events.
- Ingestion: validates and enriches events; rejects oversize or invalid payloads.
- Docs: quick-start for enabling Diagnostic Mode and reading correlated logs.

### Final Items

- Prod gate rejects `info/debug` unless Diagnostic Mode active.
- HMAC signing for FE POSTs enabled (optional but recommended).
- Dedup window active (≥ 30s) to handle client-side error storms.
- JWT diagnostics limited to `session_id`, audited, TTL ≤ 60 min.
- PII tests: CI runs sanitizer property tests (emails/phones/tokens never pass).
- 7-day retention configured in cloud logs; optional archive path validated.

## Appendix — Examples

Frontend error:

```json
{
  "timestamp": "2025-11-03T12:00:00.000Z",
  "level": "error",
  "service": "web",
  "environment": "prod",
  "release": "2025.11.03+abcd",
  "trace_id": "a1b2c3",
  "session_id": "s-123",
  "component": "LeadForm",
  "message": "Unhandled error in LeadForm",
  "error": { "type": "TypeError", "message": "Cannot read properties of undefined", "stack": ["LeadForm.tsx:42"] },
  "http": { "method": "POST", "host": "api.example.com", "status": 500, "duration_ms": 84 }
}
```

Backend exception:

```json
{
  "timestamp": "2025-11-03T12:00:05.000Z",
  "level": "error",
  "service": "api",
  "environment": "prod",
  "release": "2025.11.03+abcd",
  "trace_id": "a1b2c3",
  "request_id": "r-456",
  "message": "Unhandled exception on POST /api/leads",
  "error": { "type": "ValidationError", "message": "invalid payload", "stack": ["controllers/leads.js:77"] },
  "http": { "method": "POST", "route": "/api/leads", "status": 500, "duration_ms": 84 }
}
```
