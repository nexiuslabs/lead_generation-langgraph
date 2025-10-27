---
owner: Codex Agent – Frontend Generator
status: draft
last_reviewed: 2025-03-24
---

# Dev Plan — Email Delivery for Enrichment Results

## Summary
Implement SendGrid-backed email notifications that deliver enrichment results across synchronous chat runs and background jobs while keeping chat output lightweight. The solution must persist recipient context through LangGraph state, reuse existing scoring artifacts, and provide observability around delivery outcomes.

## Architecture & Design
- **Email Adapter (`src/notifications/sendgrid.py`).**
  - Expose `send_leads_email(state: NotifyContext, payload: EmailPayload)` which wraps SendGrid's REST API.
  - Provide helper methods for formatting HTML table content from scored leads and building CSV download links.
  - Handle missing credentials by logging and returning a `DeliveryStatus` enum (`SENT`, `SKIPPED_NO_CONFIG`, `FAILED`).
- **Graph State Extension.**
  - Augment `GraphState` dataclass (and any `TypedDict`) with `notify_email: Optional[str]` plus metadata for deduplication (e.g., `last_email_job_id`).
  - Update `_normalize` in `app/lg_entry.py` to capture recipient precedence: context override → JWT claim → `None`.
  - Ensure state persistence for background jobs (checkpoint serialization) includes the new fields.
- **Score Node Integration.**
  - After scoring completes, gather top results and invoke the email adapter when `notify_email` is available and emails enabled.
  - Replace chat Markdown table with acknowledgement text containing recipient and CSV link when email send returns `SENT`.
  - Surface failure or skip states in chat for transparency.
- **Background Job Hook.**
  - Extend `_announce_completed_bg_jobs` (or analogous callback) to trigger email send exactly once per job ID, using job metadata to guard duplicates.
  - Include summary counts and reuse the same HTML template.
- **REST Endpoint Update.**
  - Modify `/icp/enrich/top10` (and other relevant endpoints) to call the email adapter and return JSON flags (`emailed`, `email_status`).

## Data & State Considerations
- Persist `notify_email` and `last_email_job_id` (or timestamps) within graph storage to survive restarts.
- No new database tables anticipated; rely on existing job run records. If additional auditing is required, consider a lightweight table `notification_events` but defer unless metrics demand it.
- Ensure PII handling aligns with compliance: store only necessary email in volatile state, not long-term.

## Configuration & Secrets
- Add environment variables:
  - `SENDGRID_API_KEY`
  - `SENDGRID_FROM_EMAIL`
  - Optional `SENDGRID_TEMPLATE_ID`
- Update `src/settings.py` and documentation (`README.md`, `.env.example`).
- Provide feature flag (e.g., `ENABLE_EMAIL_RESULTS`) to allow staged rollout.

## Email Content & Template
- Base HTML on existing `_fmt_table` output; convert Markdown rows to HTML `<table>` using reusable helper.
- Include link to `/export/latest_scores.csv?tenant_id=...` with secure token or rely on authenticated session instructions.
- Add summary header listing job name, run timestamp, and bucket counts.

## Observability & Error Handling
- Log structured events for send attempts (recipient, job ID, status, latency).
- Capture SendGrid response codes for debugging.
- Optionally emit metrics via StatsD/Prometheus counters (`email_send_success_total`, `email_send_failure_total`).
- Integrate with existing alerting to flag sustained failure rates.

## Security & Compliance
- Validate override emails (simple regex + domain allowlist if required).
- Mask emails in logs where necessary.
- Ensure SendGrid API key is fetched from environment, not hardcoded.

## Testing Strategy
- **Unit Tests:**
  - Mock SendGrid client to assert payload shape and error handling.
  - Verify `_normalize` stores override/JWT email precedence.
  - Ensure `score_node` triggers email send and handles status mapping.
- **Integration Tests:**
  - Extend existing flow tests in `tests/test_e2e_icp_flow.py` to simulate email sends (mocked) and confirm chat acknowledgement.
  - Cover background job completion path.
- **Manual QA:**
  - Validate HTML output renders correctly via sample email preview.
  - Test fallback behaviour when credentials missing (chat should note skip).

## Risks
- SendGrid outages causing delayed notifications (mitigate with retries/backoff and clear chat messaging).
- Email template regressions impacting readability (mitigate with design review and storybook-like snapshot tests).
- Feature flag misconfiguration disabling notifications silently (mitigate by defaulting to acknowledgement even when skipped).

## Open Follow-Ups
- Evaluate need for multi-recipient support after initial launch.
- Assess telemetry granularity once metrics collected; may require new dashboards.

