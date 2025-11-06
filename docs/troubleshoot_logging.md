# Troubleshooting Logging Configuration

This feature exposes lightweight file sinks for both the backend (FastAPI) and the
frontend (Next.js) so developers and operators can capture JSONL logs locally or
on the droplet.

## Backend (`lead_generation-main`)

| Variable | Default | Purpose |
| --- | --- | --- |
| `TROUBLESHOOT_API_LOG_DIR` | `.log_api` (dev only) | Directory for `api.log`, `nightly.log`, `acra_direct.log`, and `ingest.jsonl`. Set to `/var/log/leadgen-api` in production. |
| `LOGS_DIR` | _unset_ | Fallback directory for background workers that call `src.troubleshoot_log.log_json`. |
| `LOG_DIAGNOSTIC_SECRET` | _unset_ | HS256 secret for signing the `diag` cookie that enables diagnostic logging in production. |
| `LOG_INGEST_HMAC_SECRET` | _unset_ | Optional SHA-256 signature expected in the `x-log-signature` header for `/v1/logs`. |
| `PG_CONNECT_TIMEOUT_S` | `3` | Connection timeout used by log-backed routes when emitting DB queries. |

## Frontend (`agent-chat-ui`)

| Variable | Default | Purpose |
| --- | --- | --- |
| `TROUBLESHOOT_FE_LOG_DIR` | `.log` (dev only) | Directory for `frontend-troubleshoot.jsonl`. Configure to `/var/log/agent-chat-ui` on the droplet. |
| `TROUBLESHOOT_FE_LOG_FILE` | `frontend-troubleshoot.jsonl` | Override filename used by the Next.js log sink API route. |
| `TROUBLESHOOT_FE_LOG_MAX_BYTES` | `10485760` (10 MB) | Rotate the frontend log file once it exceeds this size. |
| `TROUBLESHOOT_FE_LOG_BACKUPS` | `3` | Number of rotated copies to keep. |
| `NEXT_PUBLIC_LOG_TO_UI` | `false` in production | When `true`, the browser posts troubleshooting events to the UI sink (`/api/logs`) in addition to the backend. |
| `NEXT_PUBLIC_USE_API_PROXY` | `false` | When `true`, the frontend proxies API requests (including troubleshooting logs) through `/api/backend/...` instead of calling `NEXT_PUBLIC_API_URL` directly. |

## Helpful Commands

*Tail backend logs locally:*
```bash
LOGS_DIR=/var/log/leadgen-api pnpm run --filter lead_generation-main logs:tail
```

*Tail frontend logs on the droplet:*
```bash
sudo tail -F /var/log/agent-chat-ui/frontend-troubleshoot.jsonl
```

*Tail frontend logs locally:*
```bash
pnpm --filter agent-chat-ui logs:tail
```

*Enable diagnostic mode cookie (60 minutes):*
```python
import jwt, time
token = jwt.encode(
    {"session_id": "<client-session-id>", "exp": int(time.time()) + 3600},
    "LOG_DIAGNOSTIC_SECRET",
    algorithm="HS256",
)
print(token)
```
Set the `diag` cookie in the browser with this token to allow `info`/`debug` events in production for the matching session.

> Tip: The “Advanced” drawer in the UI now includes a Diagnostic Mode section that lets you paste the token above and clear it once you are done.

## Systemd & Logrotate Reference

Backend unit (`/etc/systemd/system/langgraph.service`):
```ini
[Unit]
Description=Lead Generation API
After=network.target

[Service]
WorkingDirectory=/opt/lead_generation
Environment="TROUBLESHOOT_API_LOG_DIR=/var/log/leadgen-api"
Environment="LOG_DIAGNOSTIC_SECRET=${LOG_DIAGNOSTIC_SECRET}"
ExecStart=/opt/lead_generation/.venv/bin/python -m app.main
Restart=on-failure
User=leadgen
Group=leadgen

[Install]
WantedBy=multi-user.target
```

Frontend unit (`/etc/systemd/system/agent-chat-ui.service`):
```ini
[Unit]
Description=Agent Chat UI
After=network.target

[Service]
WorkingDirectory=/opt/agent-chat-ui
Environment="TROUBLESHOOT_FE_LOG_DIR=/var/log/agent-chat-ui"
Environment="NEXT_PUBLIC_API_URL=https://api.example.com"
ExecStart=/opt/agent-chat-ui/.venv/bin/next start -p 3000
Restart=on-failure
User=agentui
Group=agentui

[Install]
WantedBy=multi-user.target
```

Logrotate snippet (`/etc/logrotate.d/leadgen-api`):
```
/var/log/leadgen-api/*.log /var/log/leadgen-api/*.jsonl {
    daily
    rotate 14
    compress
    missingok
    copytruncate
}
```

Adjust paths and user/group names to match your deployment. Remember to `sudo systemctl daemon-reload` after editing unit files.

## Proxy Routing Tips

`NEXT_PUBLIC_USE_API_PROXY=true` forces the browser to send troubleshooting payloads (and other API calls) through the Next.js `/api/backend` catch-all. Leave it `false` when the browser can reach `NEXT_PUBLIC_API_URL` directly (e.g., same origin). This variable can be toggled at runtime without rebuilding; restart the Next.js server after changing `.env.local`.
