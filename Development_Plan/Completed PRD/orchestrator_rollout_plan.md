# Orchestrator Rollout Plan

## Phase 1: CLI / Cron
- Replace `python src/orchestrator.py` (or `scripts/run_nightly.py`) with `python scripts/run_orchestrator.py --tenant-id <tid> --input "run full pipeline"` in all cron/supervisor jobs. Example entries live in `scripts/cron.example`.
- Systemd workers should be updated to use the same command (`scripts/systemd/leadgen-nightly.service` now ships with the new ExecStart value). Override the tenant ID or payload via drop-in unit files if you need multiple tenants.
- The CLI emits JSON status to stdout and LangGraph troubleshooting logs (`service=orchestrator`) for monitoring.
- Pass tenant/thread metadata via flags or environment; use `--icp-payload '{"industries":["software"]}'` for batch runs that need specific filters.

## Phase 2: Internal Chat
- Enable the orchestrator-backed chat entry (`app/lg_entry.py`) behind an environment flag (e.g., `ORCHESTRATOR_CHAT=1`) for internal users first.
- Monitor `/api/orchestrations` metrics (run_start/run_complete logs, LangSmith traces) and compare against legacy transcripts.

## Phase 3: Full Production
- Remove the flag once internal validation passes and make the orchestrator the default for all chat sessions.
- Deprecate the legacy Pre-SDR graph routes; delete unused helpers once metrics show parity for 1–2 weeks.

## Monitoring
- `/api/orchestrations` logs include `run_start` / `run_complete` with thread/tenant and duration.
- LangGraphTroubleshootHandler + LangSmith traces capture per-node telemetry; query by `thread_id` or `source` (api/cli/chat).
- Cron jobs should alert on non-zero exit codes or missing stdout updates.
- Systemd units should forward `journalctl -u leadgen-nightly.service` to the central log shipper; tail the log for `status.message` lines during verification.

## Testing
- Unit tests (`tests/test_orchestrator_nodes.py`) cover LLM prompt plumbing.
- API regression tests (`tests/test_orchestrator_api.py`, `tests/test_orchestrator_regression.py`) ensure repeated runs remain stable.
- For load/soak, run `python scripts/run_orchestrator.py` with mocked dependencies (or against staging) in a loop and observe duration/metrics.
- Use `python scripts/run_orchestrator_loadtest.py --iterations 10 --tenant-id <tid>` (set `ORCHESTRATOR_OFFLINE=1` locally) to capture run durations and success counts in one command.

## Load & Regression Playbook
1. **Baseline regression** — `python -m pytest tests/test_orchestrator_nodes.py tests/test_orchestrator_api.py tests/test_orchestrator_regression.py`
2. **CLI smoke** — `python scripts/run_orchestrator.py --tenant-id <tid> --input "run full pipeline"` (watch stdout for `status.message`).
3. **Load test** — run the CLI repeatedly (e.g., `for i in {1..10}; do ...; done`) or hit `/api/orchestrations` with representative payloads while LangSmith traces stay green.
4. **Latency budget** — capture start/end timestamps per run via `log_json` metrics and ensure each stage matches the thresholds listed in the PRD (normalize <30s, enrichment per-company <45s, export <20s).
5. **Vendor cap rehearsal** — simulate capped vendors by forcing mock responses so that `enrich_batch` surfaces pause/resume events; verify chat UI reflects `status.message`.

## Deployment Checklist
- [ ] Cron jobs updated (compare `/etc/cron.d/leadgen` with `scripts/cron.example`).
- [ ] Systemd unit or supervisor updated and reloaded (`systemctl daemon-reload && systemctl restart leadgen-nightly`).
- [ ] LangGraph checkpoints confirmed for at least one cron-triggered run (inspect `/api/orchestrations/{thread_id}`).
- [ ] Chat flag toggled for internal users after two consecutive cron successes.
- [ ] Legacy Pre-SDR endpoints disabled once the above steps hold for 1–2 weeks.
