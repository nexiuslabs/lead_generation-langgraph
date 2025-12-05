# Mid-Run ICP Update and Cancellation

This feature lets users update the ICP while a discovery/enrichment job is active, with a cooperative cancel flow.

Key behaviors
- Orchestrator: A new `run_guard` node intercepts `update_icp` requests when an `icp_discovery_enrich` job is `queued|running` and asks to cancel or keep.
- Cancel request: Reply “yes” to cancel; the server sets `cancel_requested=true` and the worker stops at safe boundaries.
- Keep running: Reply “no” to continue without cancel. ICP changes can be applied after the job completes.

API
- POST `/jobs/{job_id}/cancel` → `{ ok, job_id, status: 'pending_cancel' }` (admin or same-tenant only)
- GET `/jobs/{job_id}` includes `cancel_requested`, `canceled_at` fields.

Observability
- Orchestrator emits `orchestrator/cancel_ack` when a user confirms cancel.
- Worker emits `background_worker/cancelled` when a job transitions to cancelled (phase: start/post_discovery/enrich_loop).

Migration
- Adds `cancel_requested boolean DEFAULT false` and `canceled_at timestamptz` to `background_jobs`.

Notes
- Cancellation happens at safe boundaries (after discovery and between company enrichment iterations) to avoid partial writes.
- The graph’s `progress_report` surfaces the pending/complete states so the UI remains responsive.
