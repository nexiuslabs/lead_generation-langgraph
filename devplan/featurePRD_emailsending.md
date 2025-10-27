---
owner: Codex Agent – Frontend Generator
status: draft
last_reviewed: 2025-03-21
---

# Feature PRD — Email Delivery for Enrichment Results

## Story
Revenue operations specialists running enrichment jobs in the pre-SDR assistant need results delivered even when they cannot keep the chat session open. As a requester launching top-10 and next-40 enrichments, I want the system to email the scored leads to me once processing completes so I can continue prospecting without waiting in the conversational UI.

## Acceptance Criteria
- Enrichment flows (chat and REST) capture the requester email from JWT claims with an optional override supplied in the LangGraph payload and store it in graph state.
- When scoring completes, the workflow sends a SendGrid email using the captured recipient information instead of streaming the full Markdown table into chat.
- Emails contain an HTML summary table equivalent to the current chat output and include a link to download the existing `/export/latest_scores.csv` file for detailed analysis.
- Background and next-40 enrichment jobs trigger a single follow-up email on completion that summarises bucket counts and links back to the export, even if the user disconnects from the chat.
- Chat/SSE responses acknowledge that an email was sent (or that email delivery failed) so users receive immediate status feedback in the conversation history.

## Dependencies
- SendGrid API access (API key, sender identity, optional dynamic template) configured via application settings.
- Existing LangGraph state management and normalization routines (`app/lg_entry.py`, `app/pre_sdr_graph.py`) updated to persist requester metadata.
- SendGrid SDK or lightweight HTTP client wrapper introduced into the backend runtime and covered by tests.
- Logging/observability tooling to capture email delivery success/failure for audit and troubleshooting.

## Success Metrics
- ≥95% of enrichment runs with a captured email successfully send within 2 minutes of score completion.
- Support volume for “enrichment took too long in chat” issues drops by at least 50% post-launch.
- <1% of jobs report duplicate emails or missing acknowledgements in chat transcripts over a rolling 30-day period.

## Risks & Mitigations
- **Email deliverability issues (spam, bounces).** *Mitigation:* Authenticate the sender domain (SPF, DKIM) and log SendGrid event webhooks to monitor bounces.
- **State propagation bugs causing missing recipients.** *Mitigation:* Add regression tests covering state normalization and ensure `notify_email` persists through background job resumes.
- **Duplicate or excessive notifications.** *Mitigation:* Record send attempts with job metadata and guard email triggers to fire only once per job run.
- **Increased dependency surface on SendGrid SDK.** *Mitigation:* Wrap the SDK with an internal adapter and implement graceful fallbacks when credentials are absent in non-production environments.

## Decisions
1. **Recipient source hierarchy.** Default to the authenticated user’s JWT `email` claim, allowing an optional override supplied via the LangGraph payload when explicitly provided.
2. **Email content structure.** Render an HTML table mirroring the existing chat summary and append a link to the `/export/latest_scores.csv` endpoint for full data access.
3. **Chat acknowledgement strategy.** Replace the final Markdown table message with a concise confirmation indicating the email destination while preserving error messaging for failed deliveries.
4. **Background job notifications.** Extend `_announce_completed_bg_jobs` (and similar hooks) to send a single completion email per job, sharing the same summary template as the synchronous flow.
5. **Team visibility scope.** Limit the initial launch to a single primary recipient derived from the requester context (JWT claim with optional override) to minimize complexity and compliance overhead; revisit CC/BCC support once stability is proven.
6. **Rapid-fire job safeguards.** Rely on SendGrid throughput controls while monitoring telemetry for anomalies, deferring custom batching/rate limiting until traffic patterns justify additional investment.

## Open Questions

- _None at this time; proceed with implementation per the decisions above._
