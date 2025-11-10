## MCP Timeout Mitigation Playbook

This note captures the exact configuration used to stabilize the MCP/Jina fetch path when deploying both the **frontend** (Next.js) and **backend** (FastAPI/LangGraph) droplets. Follow the steps below whenever a fresh droplet is provisioned or when MCP rate-limit/timeout alarms fire.

---

### 1. Backend Droplet (LangGraph API)

1. **Pull latest code** (includes MCP throttling/backoff logic):
   ```bash
   cd /opt/lead-generation/backend
   git pull
   ```
2. **Update the `.env`** (or systemd Environment file) with the new MCP pacing knobs:
   ```dotenv
   ENABLE_MCP_READER=true
   MCP_TRANSPORT=adapters_http
   MCP_READ_MAX_CONCURRENCY=2        # keep <=2 to avoid bursts
   MCP_READ_MIN_INTERVAL_S=0.35
   MCP_READ_JITTER_S=0.15
   MCP_READ_MAX_ATTEMPTS=3
   MCP_READ_BACKOFF_BASE_S=0.6
   MCP_READ_BACKOFF_CAP_S=4.0
   MCP_READ_RATELIMIT_BACKOFF_S=2.0
   MCP_TIMEOUT_S=20                  # higher headroom for slow sites
   JINA_API_KEY=***                  # dedicated paid key preferred
   ```
3. **Restart the process** to apply settings:
   ```bash
   sudo systemctl restart leadgen-backend.service
   ```
4. **Verify logs** no longer show rapid-fire `POST https://mcp.jina.ai/sse/message` lines—calls should now be spaced ~400ms apart with jitter. Retries will back off aggressively on 429/Timeout errors instead of immediately canceling the run.

#### Nginx upstream (backend droplet)

Update `/etc/nginx/sites-available/leadgen-api` (or equivalent) so the proxy allows the longer MCP backoff window:
```nginx
upstream leadgen_api {
    server 127.0.0.1:8000;
    keepalive 32;
}

server {
    listen 443 ssl;
    server_name api.leadgen.example.com;

    location / {
        proxy_pass         http://leadgen_api;
        proxy_http_version 1.1;
        proxy_set_header   Connection "";

        proxy_connect_timeout 15s;
        proxy_send_timeout    120s;
        proxy_read_timeout    120s;
        send_timeout          120s;
    }
}
```
Then reload nginx:
```bash
sudo nginx -t && sudo systemctl reload nginx
```
This prevents nginx from killing long-lived MCP responses while still capping runaway connections.

---

### 2. Frontend Droplet (Next.js UI)

1. **Update login copy** (already merged) and rebuild to ensure users understand the product scope:
   ```bash
   cd /opt/lead-generation/frontend
   git pull
   pnpm install
   pnpm build
   sudo systemctl restart leadgen-frontend.service
   ```
2. **Increase API request timeout** for the frontend’s fetch layer (`NEXT_PUBLIC_API_TIMEOUT_MS` in `.env`) to accommodate the backend’s longer MCP backoff:
   ```dotenv
   NEXT_PUBLIC_API_TIMEOUT_MS=90000
   NEXT_PUBLIC_BACKEND_BASE_URL=https://api.leadgen.example.com
   ```
3. **Confirm nginx proxy** in front of the Next.js droplet matches the backend keep-alive/timeouts (e.g., `proxy_read_timeout 120s;`) so client-side requests do not abort before MCP completes.

#### Nginx reverse proxy (frontend droplet)

Sample `/etc/nginx/sites-available/leadgen-frontend`:
```nginx
upstream leadgen_frontend {
    server 127.0.0.1:3000;
    keepalive 16;
}

server {
    listen 443 ssl;
    server_name app.leadgen.example.com;

    location / {
        proxy_pass         http://leadgen_frontend;
        proxy_http_version 1.1;
        proxy_set_header   Host $host;
        proxy_set_header   X-Real-IP $remote_addr;
        proxy_set_header   X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;

        proxy_connect_timeout 10s;
        proxy_send_timeout    90s;
        proxy_read_timeout    90s;
    }
}
```

Commands to validate and confirm traffic is flowing through the proxy:
```bash
# syntax check + reload
sudo nginx -t && sudo systemctl reload nginx

# confirm the configured server block is live
sudo nginx -T | grep -A4 app.leadgen.example.com

# tail access logs to ensure requests hit nginx
sudo tail -f /var/log/nginx/app.leadgen.access.log
```
Hitting `https://app.leadgen.example.com/__nextjs_original_stack_frame` should emit a log entry in the above file, proving nginx is fronting the Next.js process.

---

### 3. Shared Monitoring Checklist

- **Prometheus / Loki dashboards**: watch `mcp_calls_total{status="error"}` and `Cancelled run` counts; they should trend down once throttling is active.
- **DigitalOcean metrics**: CPU should drop ~20% because the backend no longer floods the MCP reader threads.
- **Alerting**: keep existing 90s timeout alert but add a warning when MCP retries exceed 2 per minute (indicates quota saturation—consider upgrading the Jina key).

Document updated: _2025-05-01_.
