# Orchestrator Dev Plan 05 — API & Observability

## Scope
- Provide REST endpoints to trigger and monitor orchestrations.
- Attach LangGraph troubleshooting callbacks / LangSmith traces to the new graph.
- Record metrics for orchestration runs.

## Tasks
1. **API endpoints**
   - Add POST `/api/orchestrations` to start a run (request payload → `ingest_message` input) and return the current status/message.
   - Add GET `/api/orchestrations/{thread_id}` (or similar) to fetch the latest checkpoint for a thread.
   - Ensure routes require auth (same as existing chat endpoints) and pass `configurable.thread_id` / tenant metadata into the orchestrator.
2. **Callbacks & tracing**
   - Register `LangGraphTroubleshootHandler` when compiling the orchestrator graph or when invoking `handle_turn`.
   - Configure LangSmith tracing (respect existing env flags `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT`).
3. **Metrics**
   - Emit logging/metrics (e.g., `log_json`) for runs started/completed/failed, including stage durations and candidate counts.
   - Optionally integrate with existing monitoring (e.g., Prometheus, CloudWatch) if available.
4. **Docs & tests**
   - Document the new endpoints (request/response shape, auth) in the API runbook.
   - Add a basic test (FastAPI client) to ensure POST/GET routes return expected data when orchestrator is invoked.
