---
owner: Codex Agent – Backend & Agents
status: draft
last_reviewed: 2025-10-28
---

# Dev Plan — Multi‑Agent Email Delivery for Enrichment Results

## Overview
Implements an LLM‑based agentic workflow (LangChain tool‑calling) to deliver SendGrid result emails for Top‑10 and Next‑40 enrichments, across chat and background runs. An LLM composes a concise subject + intro paragraph, then calls a tool to render a deterministic HTML table and send the email. The system decomposes into cooperating agents (RecipientResolver, ComposerAgent(LLM), EmailSender(tool), AckEmitter, TelemetryAgent) orchestrated via LangGraph/async hooks, preserving simple REST semantics while enabling future template and recipient expansion.

Goals (per PRD):
- Capture requester email (JWT with optional override) and persist in run state.
- On completion, send an HTML summary with a CSV download link to `/export/latest_scores.csv`.
- Trigger a single email for background runs; surface chat/SSE acknowledgement (success/failure).
- Provide observability for delivery outcomes.

Non‑Goals:
- Sequenced outreach or multi‑recipient routing; CC/BCC deferred.
- New persistent tables unless metrics demand them (use lightweight idempotency flags first).


## Multi‑Agent Workflow (LLM + tool‑calling)
- RecipientResolver Agent
  - Input: request context (JWT claims), optional `notify_email` override from payload or LangGraph input.
  - Output: `notify_email` state; validates and normalizes address.
  - Fallbacks: If missing, mark `notify_email=None` and do not block the run.

- ComposerAgent (LLM)
  - Input: tenant/run context; a light preview of top names and an auto subject fallback.
  - Behavior: Uses `ChatOpenAI` with a safety prompt to compose a concise subject (<=80 chars) and a 1–3 sentence HTML intro; avoids PII and hallucination.
  - Tool‑calling: Invokes `send_email_tool(to, subject, intro_html, tenant_id, limit)`.

- EmailSender Agent
  - Input: `notify_email`, subject, HTML content, CSV link.
  - Output: `DeliveryStatus` enum (`SENT`, `SKIPPED_NO_CONFIG`, `FAILED`) with metadata (SendGrid id, http status, error).
  - Implementation: SendGrid REST via internal adapter; feature‑flagged by `ENABLE_EMAIL_RESULTS`.

- AckEmitter Agent
  - Input: delivery status, session/tenant identifiers.
  - Output: SSE/chat event `email:status` with `{ status, to, link }`; in synchronous flows, replaces final big markdown with a concise acknowledgement.

- TelemetryAgent
  - Input: delivery attempt context.
  - Output: structured log events + counters; optional future webhook processing.

Graph sketch (logical):
State = { messages, candidates?, notify_email?, email_delivery? }
  └─ after scoring → RecipientResolver → ComposerAgent(LLM) → EmailSender(tool) → AckEmitter (+ Telemetry)

## Agentic Orchestration Details
- Files: `src/notifications/agentic_email.py`, `src/notifications/render.py`, `src/notifications/sendgrid.py`.
- Tool: `send_email_tool(to, subject, intro_html, tenant_id, limit=200)`
  - Renders summary via `render_summary_html(tenant_id, limit)` and sends via SendGrid.
  - Returns `{ status, http_status?, request_id?, csv_link?, error? }`.
- LLM: `ChatOpenAI` from `LANGCHAIN_MODEL` and `TEMPERATURE`, bound with the tool using LangChain tool‑calling.
- Prompt: Safety‑constrained system + human messages instructing concise, neutral intro; no hallucinations or PII; subject <= 80 chars. The table and CSV link are appended deterministically by the tool.
- Fallback: If the model does not invoke the tool, the orchestrator sends with the auto subject and a generic intro.


## Backend Implementation

### LangGraph Integration Touchpoints
- Files: `app/pre_sdr_graph.py`, `app/lg_entry.py`.
- State extension: add `notify_email: Optional[str]` and `email_delivery: { status: str, sent_at?: ts, to?: str }` to the graph state/checkpoint so recipient metadata survives resumes.
- Capture: set `notify_email` during input normalization/entry (override → JWT) in `app/lg_entry.py` (graph entry helpers) and maintain through `pre_sdr_graph` nodes.
- Invoke send: at the scoring/completion node in `pre_sdr_graph`, call renderer + sender when `notify_email` is present; on success, emit `email:status` and suppress large markdown table in chat.
- Background path: when the graph schedules Next‑40 (via jobs), include `notify_email` in `background_jobs.params` so the job runner can send exactly once post‑completion.
- Checkpointing: ensure these fields are included in any persisted state to avoid loss on restarts.

### 1) Configuration and Settings
- Files: `lead_generation-main/src/settings.py`, `.env.example`, `requirements.txt` (SendGrid SDK optional; REST via httpx is sufficient)
- Add envs:
  - `ENABLE_EMAIL_RESULTS` (default: true in dev, false in prod until ready)
  - `SENDGRID_API_KEY`
  - `SENDGRID_FROM_EMAIL`
  - Optional: `SENDGRID_TEMPLATE_ID`
- Expose via settings:
  - `EMAIL_ENABLED`, `SENDGRID_API_KEY`, `SENDGRID_FROM_EMAIL`, `SENDGRID_TEMPLATE_ID`

### 2) SendGrid Adapter
- New: `src/notifications/sendgrid.py`
  - `class DeliveryStatus(str, Enum): SENT = "sent"; SKIPPED_NO_CONFIG = "skipped_no_config"; FAILED = "failed"`
  - `def send_leads_email(to: str, subject: str, html: str, *, template_id: str|None=None, substitutions: dict|None=None) -> dict:`
    - Validates config; if missing, return `{ status: SKIPPED_NO_CONFIG }`.
    - Uses `httpx.AsyncClient` to POST to `https://api.sendgrid.com/v3/mail/send` with minimal payload.
    - Returns `{ status, request_id?, http_status?, error? }`.
  - Log with structured logger `notifications`.

### 3) Email Content Helpers
- New: `src/notifications/render.py`
  - `def render_summary_html(tenant_id: int, limit: int = 200) -> tuple[str, str, str]` → `(subject, html_body, csv_link)`
    - Queries `lead_scores` join `companies` similarly to `/export/latest_scores.*` endpoints.
    - Builds a small HTML: header (run timestamp, bucket counts) + table (name, domain, score, bucket, primary_email, contact_name/title).
    - Link: `/export/latest_scores.csv?limit=500` (UI/Docs instructs that user must be authenticated when clicking).

### 4) Capture Recipient in State
- Location: `app/main.py` normalize_input and ICP endpoints; plus LangGraph state if used.
- Precedence: override in request/graph input → JWT claim `email|preferred_username|sub`.
- Store in request‑local context and pass to enrichment routines via:
  - Synchronous chat path: carry `notify_email` through the handlers and finalization.
  - Background path: persist best‑effort into `background_jobs.params.notify_email` when enqueuing, if available; otherwise derive at send‑time via tenant→user mapping (last actor on run).

Concretely:
- In `app/icp_endpoints.py` for `/icp/enrich/top10` and enqueue of Next‑40:
  - Resolve `to_email` from claims; if body provides an override (future), prefer it.
  - Emit chat events up front: `email:intent` when email is set, e.g., “I’ll email your shortlist to <to> when done.”
  - Include `notify_email` in the in‑memory flow context and pass into completion handler.

### 5) Send at Completion — Synchronous Top‑10 (Agentic)
- After loop over Top‑10 (existing summary emit), call:
  - `res = agentic_send_results(to_email, tenant_id)` which internally prompts the LLM and calls the `send_email_tool`.
  - Emit SSE: `emit_chat_event(x_session_id, tid, "email:status", msg, { status: res.status, to: to_email, link })` where `msg` acknowledges success or failure.
- Replace any large markdown payload with a short confirmation when `res.status == SENT`.

### 6) Send at Completion — Background Next‑40 (Agentic)
- Hook in `src/jobs.py` in `run_web_discovery_bg_enrich` and `run_enrich_candidates` after marking job `done`:
  - Resolve `tenant_id` from job record.
  - Derive recipient:
    - Prefer `background_jobs.params.notify_email` (if we stored it when enqueuing from chat), else
    - Resolve via last mapped user for tenant (best‑effort), else skip.
  - Guard idempotency: write a volatile flag in `background_jobs.params` (in‑place update) under `email_sent_at` or keep a small in‑memory cache by `job_id` within process; prefer DB params update to survive restarts.
  - Compose and send via `agentic_send_results(to_email, tenant_id)`. No SSE here unless we also keep a `session_id` param; out of scope for first cut.

### 7) Export Endpoint Reuse
- Keep `/export/latest_scores.(json|csv)` unchanged; the email only links to it.
- Ensure tenant resolution path in export endpoints remains robust so the link serves the right tenant with the recipient’s session.

### 8) Observability and Telemetry
- Structured logs on logger `notifications` with fields: `{ tenant_id, to, status, http_status, request_id, job_id?, run_id?, duration_ms }`.
- Optional counters (future): Prometheus/StatsD. For now, include aggregate logs that can be parsed.


## Frontend (agent‑chat‑ui)

Minimal changes to acknowledge delivery in chat and to avoid rendering large tables when email is sent.

- SSE Handling
  - File: `agent-chat-ui/src/components/thread/ChatProgressFeed.tsx`
  - Add handler for `email:status` event; render a compact message card: “Sent results to <to>. Download CSV” with link button to `/export/latest_scores.csv`.
  - Optionally toast on failure with retry advice.

- No auth changes needed; the CSV link requires an authenticated session which the UI already maintains.


## Expected Output

- Email (user inbox)
  - Subject: "Your shortlist is ready — Top‑10/Next‑40 (Tenant <id>)"
  - Subject source: LLM‑generated (safety‑constrained) with deterministic fallback.
  - Header: run timestamp, tenant, job context (Top‑10 vs Next‑40), bucket counts summary (e.g., A: 3, B: 5, C: 2).
  - Intro: 1–3 sentences authored by the LLM per safety prompt.
  - Body: Deterministic HTML table with columns: Name, Domain, Score, Bucket, Primary Email, Contact Name, Contact Title, Contact LinkedIn.
  - Link: button and hyperlink to `/export/latest_scores.csv?limit=500`.
  - Footer: note that CSV requires being signed in; include brief support tip if link returns 401.

- SSE/chat acknowledgement (Top‑10 flow)
  - Event: `email:status`
  - Payload example:
    ```json
    {
      "event": "email:status",
      "message": "Sent results to ops@example.com",
      "context": {
        "status": "sent",
        "to": "ops@example.com",
        "link": "/export/latest_scores.csv?limit=500",
        "session_id": "<uuid>",
        "tenant_id": 42
      }
    }
    ```
  - Failure example (`status: "failed"`) shows a concise message and keeps the CSV link visible.

- REST response additions (Top‑10 endpoint)
  - Shape (augment current):
    ```json
    {
      "ok": true,
      "requested": 10,
      "processed": 10,
      "next40_job_id": 1234,
      "emailed": true,
      "email_status": "sent",
      "emailed_to": "ops@example.com"
    }
    ```

- Logs/telemetry (structured)
  - Logger: `notifications`
  - Example: `{ "tenant_id": 42, "job_id": 1234, "to": "o**@e**.com", "status": "sent", "http_status": 202, "duration_ms": 340 }`


## Expected New User Flow

- In Chat (Top‑10)
  - User signs in and runs ICP flow; clicks “Enrich Top‑10”.
  - The assistant posts: “I’ll email your shortlist to <their email> when done.”
  - Progress messages stream as usual; the final large table is no longer posted.
  - A new message appears: “Sent results to <email>. Download CSV” with a button linking to `/export/latest_scores.csv`.
  - Clicking the link downloads the CSV if the user is still authenticated; otherwise, the UI prompts to sign in and retry.

- Background (Next‑40)
  - After Top‑10 finishes, the system queues Next‑40 enrichment in the background.
  - When it completes, the user receives a single email summarizing bucket counts with the same CSV link.
  - No additional chat messages are required unless the session is active; duplicates are prevented via job param guards.

- REST‑only usage
  - Client calls `/icp/enrich/top10`; response JSON includes `emailed`, `email_status`, `emailed_to`.
  - Client may optionally subscribe to `/chat/stream/{session_id}` to receive the `email:status` event.
  - Background Next‑40 email is sent once using the captured recipient; no polling is required to receive results.

- Edge conditions
  - If email is not configured (`SKIPPED_NO_CONFIG`), the chat explicitly states it and still provides the CSV link.
  - If sending fails, the chat shows a failure notice with CSV link and suggests verifying email settings.


## Data Model & Idempotency
- Prefer lightweight idempotency aligned to existing artifacts:
  - For background jobs: update `background_jobs.params` JSON with `{ "email_sent_at": now(), "email_to": "..." }` once sending succeeds; check this before sending.
  - For synchronous runs: attach `email_delivery` to ephemeral graph state; no DB table required.
- If future auditing demanded: add `notification_events(job_id, tenant_id, recipient, status, created_at, payload_json)` table (deferred).


## Security & Compliance
- Validate email shape with simple regex; optional allowlist per domain.
- Mask emails in logs (`a**@e**.com`).
- Do not store recipient emails long‑term outside of job params (optional, non‑PII context).
- Keep SendGrid API key in env only; no source control artifacts.
 - LLM constraints: prompts prohibit PII and hallucination; intro text is short and neutral; deterministic renderer prevents data leakage; fallback path avoids sending if config missing.


## Testing Strategy
- Unit Tests (backend):
  - `tests/test_notifications_sendgrid.py`: mock httpx, assert payload, error mapping, SKIPPED when no config.
  - `tests/test_email_recipient_resolution.py`: precedence rules (override → JWT → None).
  - `tests/test_icp_enrich_email_ack.py`: monkeypatch agent `agentic_send_results` to return SENT/FAILED; verify SSE `email:status` emission.
  - `tests/test_jobs_bg_email_once.py`: simulate `run_web_discovery_bg_enrich` completion; ensure single send with `params.email_sent_at` guard.
  - `tests/test_agentic_email.py`: assert LLM tool‑call binding and fallback path; ensure output merges tool result fields and csv_link.

- Integration Tests:
  - Extend `tests/test_e2e_icp_flow.py`: simulate Top‑10 enrich; assert email ack event in stream.
  - Background job path: run job with params notify_email; assert email send called once.

- Manual QA:
  - Send with valid SendGrid creds to seed account; review HTML rendering.
  - Missing creds path → chat displays “skipped” and link to CSV; no crash.


## Rollout Plan
- Feature flag: `ENABLE_EMAIL_RESULTS=false` in production initially; enable per environment.
- Shadow mode (optional): Compose HTML and log only (no send) until confident; then switch on.
- Monitor logs for failure rates; verify no duplicate sends.


## Risks & Mitigations
- Deliverability issues: authenticate sender domain (SPF/DKIM); consider SendGrid webhook logging later.
- Duplicate notifications: guard by `background_jobs.params.email_sent_at` and careful chat flow control.
- Long HTML emails: cap rows (e.g., 200) and rely on CSV link for full detail.


## File/Code Touchpoints
- Backend
  - Add: `src/notifications/sendgrid.py`, `src/notifications/render.py`, `src/notifications/agentic_email.py`.
  - Edit: `src/settings.py` (envs), `app/icp_endpoints.py` (Top‑10 ack + agentic send + params propagation), `src/jobs.py` (bg completion email via agent), `app/main.py` (optional recipient capture on legacy path), `app/chat_stream.py` (no change; reuse emitter).
  - Tests: `tests/test_notifications_sendgrid.py`, `tests/test_icp_enrich_email_ack.py`, `tests/test_jobs_bg_email_once.py`.

- Frontend (agent‑chat‑ui)
  - Edit: `src/components/thread/ChatProgressFeed.tsx` to render `email:status`.


## Pseudocode Snippets

```python
# Agentic tool + LLM orchestration
@tool
async def send_email_tool(to: str, subject: str, intro_html: str, tenant_id: int, limit: int = 200) -> dict:
    subj_auto, table_html, csv_link = render_summary_html(tenant_id, limit)
    html = f"{intro_html}{table_html}"
    return await send_leads_email(to, subject or subj_auto, html)

llm = ChatOpenAI(model=LANGCHAIN_MODEL, temperature=TEMPERATURE).bind_tools([send_email_tool])
ai = await llm.ainvoke([SystemMessage(...), HumanMessage(...)])
for call in ai.tool_calls:
    if call["name"] == "send_email_tool":
        args = {**call["args"], "to": to_email, "tenant_id": tid}
        res = await send_email_tool.invoke(args)
        break
```


## Acceptance Alignment
- Meets PRD ACs: captures email, sends at completion via SendGrid, HTML table equivalent, link to CSV, background single email, chat acknowledgement.
- Success metrics supported by logs (95% under 2 min assumed OK given async send + render cost; optimize if needed).


## Next Steps
- Confirm acceptable HTML layout with stakeholders.
- Wire feature flag defaults per env.
- Implement in small PRs per checklist to ease review.
