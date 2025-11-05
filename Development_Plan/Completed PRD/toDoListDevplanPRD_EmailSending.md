---
owner: Codex Agent – Backend & Agents
status: active
last_reviewed: 2025-10-29
---

# TODOs — PRD Email Sending (LLM Agentic)

Scope: Track implementation progress for LLM‑based agentic email delivery (LangChain tool‑calling) of enrichment results across Top‑10 (chat) and Next‑40 (background) flows.

## Backend: Config & Settings
- [x] Add `ENABLE_EMAIL_RESULTS`, `SENDGRID_API_KEY`, `SENDGRID_FROM_EMAIL`, optional `SENDGRID_TEMPLATE_ID` to `src/settings.py`.
- [ ] Update `.env.example` and `read.md` with configuration notes.
\- [x] Add `DEFAULT_NOTIFY_EMAIL` fallback and `EMAIL_DEV_ACCEPT_TENANT_USER_ID_AS_EMAIL` guard.

## Backend: Notifications Adapter
- [x] Create `src/notifications/sendgrid.py` with `send_leads_email(...)` and structured logging.
- [x] Mask recipient emails in logs; include request id / http status.
\- [x] Support single CSV attachment (base64) with filename and content type.

## Backend: HTML Renderer
- [x] Create `src/notifications/render.py` with `render_summary_html(tenant_id, limit=200)` producing `(subject, html, csv_link)`.
- [ ] Validate bucket counts and column mapping match `/export/latest_scores.*`.
\- [x] Add `build_csv_bytes(tenant_id, limit)` to generate CSV attachment content consistent with exports.

## Backend: Agentic Orchestrator
- [x] Create `src/notifications/agentic_email.py` implementing:
  - [x] `@tool send_email_tool(to, subject, intro_html, tenant_id, limit=200)`
  - [x] `agentic_send_results(to, tenant_id)` with LLM prompt (subject + intro), tool‑calling, and fallback path.
- [x] Safety constraints: concise intro (1–3 sentences), subject <= 80 chars, no PII/hallucinations.
\- [x] Enforce recipient/tenant idempotently (LLM cannot override `to`/`tenant_id`).
\- [x] Attach CSV bytes by default (limit 500) in tool send path.

## Synchronous Top‑10 Integration (Chat)
 - [x] Capture `notify_email` (override→JWT) in `/icp/enrich/top10` endpoint.
- [x] After enrichment completes, call `agentic_send_results(to, tenant_id)`; emit `email:status` SSE event.
- [ ] Replace large markdown summary with concise acknowledgement on success.

## Background Next‑40 Integration
- [x] Include `notify_email` in `enqueue_web_discovery_bg_enrich` params when launched from chat.
- [x] On completion in `run_web_discovery_bg_enrich`, call `agentic_send_results(to, tenant_id)` once; set `params.email_sent_at` for idempotency.
- [x] Ensure tenant resolution for renderer is correct and robust.
\- [x] `/icp/enrich/next40` resolves and passes `notify_email` (override→JWT→tenant_users→DEFAULT_NOTIFY_EMAIL).
\- [x] Debug script `scripts/enqueue_next40_debug.py` accepts `--notify-email` and falls back to `DEFAULT_NOTIFY_EMAIL`.

## Frontend: SSE Acknowledgement
- [ ] Handle `email:status` in `agent-chat-ui/src/components/thread/ChatProgressFeed.tsx`.
 - [ ] Render compact card with destination and note CSV is attached; toast on failure.

## Observability & Error Handling
- [x] Add structured logs for sends; include `tenant_id`, `job_id?`, masked `to`, `status`, `http_status`.
- [x] Validate fallback behaviors: `SKIPPED_NO_CONFIG`, `FAILED` paths do not break flows.
\- [x] Log if an LLM attempts to override recipient (ignored).

## Testing
 - [x] Unit: mock httpx; adapter payload + error mapping.
- [x] Unit: recipient precedence (override→JWT) stored correctly.
- [x] Unit: idempotent background email based on `params.email_sent_at`.
 - [x] Unit: LLM agent/tool calling (`tests/test_agentic_email.py`) covering tool binding + fallback (dummy success/skip).
 - [x] Unit: CSV attachment present and named correctly when enabled.
- [ ] Integration: extend `tests/test_e2e_icp_flow.py` to assert `email:status` event.

## Docs & Rollout
- [ ] Document behavior and CSV auth requirements; add troubleshooting.
- [ ] Gate with `ENABLE_EMAIL_RESULTS`; plan staged enablement.
\- [ ] Update docs to reflect CSV file attachment (not just link) and `/email/test` endpoint usage.

## Utilities
- [x] Add `/email/test` endpoint for SendGrid smoke tests (simple/agentic modes).
