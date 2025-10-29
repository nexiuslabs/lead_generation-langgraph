---
owner: Codex Agent – Backend & Agents
status: active
last_reviewed: 2025-10-28
---

# TODOs — PRD Email Sending (LLM Agentic)

Scope: Track implementation progress for LLM‑based agentic email delivery (LangChain tool‑calling) of enrichment results across Top‑10 (chat) and Next‑40 (background) flows.

## Backend: Config & Settings
- [x] Add `ENABLE_EMAIL_RESULTS`, `SENDGRID_API_KEY`, `SENDGRID_FROM_EMAIL`, optional `SENDGRID_TEMPLATE_ID` to `src/settings.py`.
- [ ] Update `.env.example` and `read.md` with configuration notes.

## Backend: Notifications Adapter
- [x] Create `src/notifications/sendgrid.py` with `send_leads_email(...)` and structured logging.
- [x] Mask recipient emails in logs; include request id / http status.

## Backend: HTML Renderer
- [x] Create `src/notifications/render.py` with `render_summary_html(tenant_id, limit=200)` producing `(subject, html, csv_link)`.
- [ ] Validate bucket counts and column mapping match `/export/latest_scores.*`.

## Backend: Agentic Orchestrator
- [x] Create `src/notifications/agentic_email.py` implementing:
  - [x] `@tool send_email_tool(to, subject, intro_html, tenant_id, limit=200)`
  - [x] `agentic_send_results(to, tenant_id)` with LLM prompt (subject + intro), tool‑calling, and fallback path.
- [x] Safety constraints: concise intro (1–3 sentences), subject <= 80 chars, no PII/hallucinations.

## Synchronous Top‑10 Integration (Chat)
- [ ] Capture `notify_email` (override→JWT) in `/icp/enrich/top10` endpoint.
- [x] After enrichment completes, call `agentic_send_results(to, tenant_id)`; emit `email:status` SSE event.
- [ ] Replace large markdown summary with concise acknowledgement on success.

## Background Next‑40 Integration
- [x] Include `notify_email` in `enqueue_web_discovery_bg_enrich` params when launched from chat.
- [x] On completion in `run_web_discovery_bg_enrich`, call `agentic_send_results(to, tenant_id)` once; set `params.email_sent_at` for idempotency.
- [x] Ensure tenant resolution for renderer is correct and robust.

## Frontend: SSE Acknowledgement
- [ ] Handle `email:status` in `agent-chat-ui/src/components/thread/ChatProgressFeed.tsx`.
- [ ] Render compact card with destination + CSV link; toast on failure.

## Observability & Error Handling
- [x] Add structured logs for sends; include `tenant_id`, `job_id?`, masked `to`, `status`, `http_status`.
- [x] Validate fallback behaviors: `SKIPPED_NO_CONFIG`, `FAILED` paths do not break flows.

## Testing
- [ ] Unit: mock httpx; adapter payload + error mapping.
- [ ] Unit: recipient precedence (override→JWT) stored correctly.
- [ ] Unit: idempotent background email based on `params.email_sent_at`.
- [ ] Unit: LLM agent/tool calling (`tests/test_agentic_email.py`) covering tool binding + fallback.
- [ ] Integration: extend `tests/test_e2e_icp_flow.py` to assert `email:status` event.

## Docs & Rollout
- [ ] Document behavior and CSV auth requirements; add troubleshooting.
- [ ] Gate with `ENABLE_EMAIL_RESULTS`; plan staged enablement.
