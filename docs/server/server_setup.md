# LeadGen Droplet Service Setup

## 1. Overview

This runbook captures the nginx and systemd configuration for both the chat frontend (`chat.nexiusagent.com`) and the LangGraph backend (`langgraphapi.nexiusagent.com`). The raw config snippets are preserved exactly so they can be copied into place; the surrounding notes explain why each block exists and the operational considerations when editing them.

## 2. Frontend Stack (chat.nexiusagent.com)

The chat UI runs on Next.js (port `3001`) behind nginx. The proxy terminates TLS, keeps long-lived SSE streams alive, and forwards NextAuth login/register flows directly to the backend auth host.

### 2.1 Nginx Proxy (`/etc/nginx/sites-available/chat.nexiusagent.com`)

# --- HTTP -> HTTPS ---
  server {
    listen 80;
    listen [::]:80;
    server_name chat.nexiusagent.com;
    return 301 https://$host$request_uri;
  }

  # --- HTTPS site ---
  server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name chat.nexiusagent.com;

    ssl_certificate     /etc/letsencrypt/live/chat.nexiusagent.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/chat.nexiusagent.com/privkey.pem;

    proxy_http_version 1.1;
    proxy_set_header X-Real-IP         $remote_addr;
    proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_set_header Connection        "";
    proxy_read_timeout   36000s;
    proxy_send_timeout   36000s;
    proxy_connect_timeout 300s;
    proxy_buffering off;
    chunked_transfer_encoding on;
    add_header X-Accel-Buffering no;

    # ---------- AUTH to backend (cookie rewrites stay here) ----------
    location = /api/auth/login {
      proxy_ssl_server_name on;
      proxy_set_header Host langgraphapi.nexiusagent.com;
      proxy_set_header X-Tenant-ID prod;
      proxy_pass https://langgraphapi.nexiusagent.com/auth/login;

      proxy_cookie_domain langgraphapi.nexiusagent.com .nexiusagent.com;
      proxy_cookie_path / /;
      proxy_cookie_flags nx_access  SameSite=None secure;
      proxy_cookie_flags nx_refresh SameSite=None secure;
    }

    location = /api/auth/register {
      proxy_ssl_server_name on;
      proxy_set_header Host langgraphapi.nexiusagent.com;
      proxy_set_header X-Tenant-ID prod;
      proxy_pass https://langgraphapi.nexiusagent.com/auth/register;

      proxy_cookie_domain langgraphapi.nexiusagent.com .nexiusagent.com;
      proxy_cookie_path / /;
      proxy_cookie_flags nx_access  SameSite=None secure;
      proxy_cookie_flags nx_refresh SameSite=None secure;
    }

    # ---------- NextAuth & everything else under /api/auth/* stays on Next.js ----------
    location ^~ /api/auth/ {
      proxy_set_header Host $host;
      proxy_pass http://127.0.0.1:3001/api/auth/;
    }
  # ---------- /api/backend/* -> Next.js proxy (applies SSE timeouts + logging) ----------
    location ^~ /api/backend/ {
      proxy_set_header Host              $host;
      proxy_set_header X-Forwarded-Host  $host;
      proxy_set_header X-Tenant-ID       prod;
      proxy_set_header Authorization     $auth_header;
      proxy_pass http://127.0.0.1:3001$request_uri;

      proxy_cookie_domain langgraphapi.nexiusagent.com .nexiusagent.com;
      proxy_cookie_path / /;
      proxy_cookie_flags nx_access  SameSite=None secure;
      proxy_cookie_flags nx_refresh SameSite=None secure;
    }

    # ---------- all other /api/* calls go through Next.js too ----------
    location ^~ /api/ {
      proxy_set_header Host              $host;
      proxy_set_header X-Forwarded-Host  $host;
      proxy_set_header X-Tenant-ID       prod;
      proxy_set_header Authorization     $auth_header;
      proxy_pass http://127.0.0.1:3001$request_uri;

      proxy_cookie_domain langgraphapi.nexiusagent.com .nexiusagent.com;
      proxy_cookie_path / /;
      proxy_cookie_flags nx_access  SameSite=None secure;
      proxy_cookie_flags nx_refresh SameSite=None secure;
    }

    # ---------- Static + app shell ----------
    location / {
      proxy_set_header Host $host;
      proxy_pass http://127.0.0.1:3001;
    }

    location = /healthz { return 200 "ok\n"; }
  }


*Why these settings?* The generous `proxy_read_timeout`/`send_timeout` values (up to 10 hours) stop nginx from killing streaming responses. Authentication routes (`/api/auth/login|register`) go straight to `langgraphapi.nexiusagent.com`, while every other `/api/*` path stays on the Next.js proxy so cookies and headers are managed consistently. After editing this file run `sudo nginx -t && sudo systemctl reload nginx` so changes take effect.

### 2.2 Chat Next.js systemd service

Chat-nextjs.service ( /etc/systemd/system/chat-nextjs.service)

[Unit]
Description=Next.js (chat.nexiusagent.com)
After=network.target

[Service]
Type=simple
WorkingDirectory=/var/www/chat.nexiusagent.com/app/langchain_chat_ui
User=www-data
Group=www-data
Environment=NODE_ENV=production
Environment=PORT=3001
Environment=NEXTAUTH_URL=https://chat.nexiusagent.com
Environment=NEXT_PUBLIC_API_URL=/api
Environment=NEXT_PUBLIC_API_BASE=/api
ExecStart=/usr/bin/node node_modules/next/dist/bin/next start -p 3001
Environment="TROUBLESHOOT_FE_LOG_DIR=.log"
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target



Use the same `.env` values locally and in production: the “PUBLIC” section feeds the browser bundle, while the server-side set keeps secrets on the droplet. Key rules:

- Keep `NEXT_PUBLIC_API_URL` and `NEXT_PUBLIC_API_BASE` pointing to the nginx proxy (`/api/backend`) so auth headers are injected consistently.
- `NEXTAUTH_URL` must always equal the public HTTPS origin, otherwise callbacks will fail.
- Timeouts (`NEXT_BACKEND_TIMEOUT_MS`, etc.) should stay aligned with the nginx proxy timeouts defined earlier.

### 2.3 Environment File Setup (`.env.local` / `.env.production`)

 ----------------- PUBLIC (browser) -----------------
# Point the UI to the chat proxy, NOT the backend host.
# Your app is using /api/backend/*, so keep that.
NEXT_PUBLIC_API_URL=/api/backend
NEXT_PUBLIC_API_BASE=/api/backend

NEXT_PUBLIC_ASSISTANT_ID=agent
NEXT_PUBLIC_ENABLE_TENANT_SWITCHER=false
NEXT_PUBLIC_USE_API_PROXY=true
NEXT_PUBLIC_USE_AUTH_HEADER=false   # nginx injects Authorization for you

# ----------------- NextAuth -----------------
# Must be your public site URL in prod:
NEXTAUTH_URL=https://chat.nexiusagent.com
NEXTAUTH_DEBUG=true

# ----------------- Backend / OIDC -----------------
# Server-side values (not exposed to browser)
BACKEND_API_BASE=https://langgraphapi.nexiusagent.com
LANGGRAPH_API_URL=https://langgraphapi.nexiusagent.com

NEXIUS_ISSUER=https://sso.nexiusagent.com/realms/prod
NEXIUS_CLIENT_ID=agent-chat-ui
NEXIUS_CLIENT_SECRET=
NEXTAUTH_SECRET=

# ----------------- Misc -----------------
# Cookies must be Secure when SameSite=None
COOKIE_SECURE=true

NODE_OPTIONS=--dns-result-order=ipv4first
LANGGRAPH_ALLOW_ANON=false

NEXT_PUBLIC_ENABLE_TENANT_SWITCHER=false
NEXTAUTH_DEBUG=true
NEXT_PUBLIC_USE_AUTH_HEADER=false
COOKIE_SECURE=false
NEXT_PUBLIC_USE_API_PROXY=true
LANGGRAPH_ALLOW_ANON=false

NEXT_BACKEND_TIMEOUT_MS=3600000
NEXT_HEADERS_TIMEOUT_MS=3720000
NEXT_BODY_TIMEOUT_MS=3720000
NEXT_BACKEND_CONNECTIONS=32
#NODE_ENV=production
NODE_ENV=staging
#ENVIRONMENT=staging
TROUBLESHOOT_FE_LOG_DIR=.log

#ENVIRONMENT=production
ENVIRONMENT=staging
#TROUBLESHOOT_FE_LOG_DIR=/var/log/chat-nextjs
TROUBLESHOOT_FE_LOG_FILE=frontend-troubleshoot.jsonl
TROUBLESHOOT_FE_LOG_MAX_BYTES=10485760
TROUBLESHOOT_FE_LOG_BACKUPS=1
NEXT_PUBLIC_LOG_TO_UI=true



---

## 3. Backend Stack (langgraphapi.nexiusagent.com)

### 3.1 Backend Nginx Setup

The LangGraph API listens on port `8001` (uvicorn) and sits behind nginx for TLS and CORS. The snippet below is stored in `/etc/nginx/sites-available/langgraphapi.nexiusagent.com` and exposes only `/healthz` plus the proxied API.

============ BACKEND Server Setup ==========

Nigix Setup

server {
      listen 80;
      listen [::]:80;
      server_name langgraphapi.nexiusagent.com;

      location ^~ /.well-known/acme-challenge/ { root /var/www/html; }
      return 301 https://$host$request_uri;
  }

  server {
      listen 443 ssl http2;
      listen [::]:443 ssl http2;
      server_name langgraphapi.nexiusagent.com;

      ssl_certificate     /etc/letsencrypt/live/langgraphapi.nexiusagent.com/fullchain.pem;
      ssl_certificate_key /etc/letsencrypt/live/langgraphapi.nexiusagent.com/privkey.pem;
      include             /etc/letsencrypt/options-ssl-nginx.conf;
      ssl_dhparam         /etc/letsencrypt/ssl-dhparams.pem;

      client_max_body_size 25m;

      location / {
          if ($request_method = OPTIONS) {
              add_header Access-Control-Allow-Origin "https://chat.nexiusagent.com" always;
              add_header Access-Control-Allow-Credentials "true" always;
              add_header Access-Control-Allow-Methods "GET,POST,PUT,PATCH,DELETE,OPTIONS" always;
              add_header Access-Control-Allow-Headers "Content-Type, Authorization, X-Requested-With" always;
              add_header Vary "Origin" always;
              add_header Content-Length 0;
              add_header Content-Type text/plain;
              return 204;
          }

          proxy_hide_header Access-Control-Allow-Origin;
          proxy_hide_header Access-Control-Allow-Credentials;
          proxy_hide_header Access-Control-Allow-Headers;
          proxy_hide_header Access-Control-Allow-Methods;
          proxy_hide_header Access-Control-Expose-Headers;
          proxy_hide_header Vary;

          add_header Access-Control-Allow-Origin "https://chat.nexiusagent.com" always;
          add_header Access-Control-Allow-Credentials "true" always;
          add_header Vary "Origin" always;

          proxy_http_version 1.1;
          proxy_set_header Host              $host;
          proxy_set_header X-Real-IP         $remote_addr;
          proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
          proxy_set_header X-Forwarded-Proto $scheme;
          proxy_set_header Connection        "";

          proxy_read_timeout   3600s;
          proxy_send_timeout   3600s;
          proxy_connect_timeout 300s;
          proxy_buffering off;
          chunked_transfer_encoding on;
          add_header X-Accel-Buffering no;

          proxy_pass http://127.0.0.1:8001;
      }

      location = /healthz { return 200 "ok\n"; }
  }



The `if ($request_method = OPTIONS)` block handles CORS preflight so browsers can call the API directly from the chat domain. Remember to reload nginx after edits: `sudo nginx -t && sudo systemctl reload nginx`.

### 3.2 Systemd Units

  LangGraph Service Setup
  ----------------------

  1. langgraph.service
  2. langgraph-scheduler.service
  3. langgraph-scheduler.timer
  4. leadgen-acra-direct.service
  5. leadgen-bg-worker.service
  6. odoo-tunnel.service


Each unit below can be dropped into `/etc/systemd/system/` and enabled with `sudo systemctl enable --now <service>`. They assume the repo lives at `/root/lead_generation`—adjust `WorkingDirectory`/`ExecStart` if you deploy elsewhere.

#### 3.2.1 langgraph.service

langgraph.service

[Unit]
Description=LangGraph API (dev)
After=network.target
Wants=network-online.target

[Service]
Type=simple
# **Adjust this path to where your project lives**
WorkingDirectory=/root/lead_generation
Environment="TROUBLESHOOT_API_LOG_DIR=/var/log/leadgen-api"
# If you keep your variables in a .env file, uncomment this:
# EnvironmentFile=/root/lead_generation/.env

# ---- Inline env (kept here so it's self-contained) ----
# Core dev/runtime knobs (keep memory footprint lower)
Environment=ENABLE_ENRICHMENT=false
Environment=BG_JOB_MAX_WORKERS=1
Environment=THREAD_TTL_MINUTES=2
# If present in your build; harmless if ignored:
Environment=CHECKPOINT_TTL_MINUTES=5

# Odoo DB direct connection pattern (used by your app)
Environment=ODOO_BASE_DSN_TEMPLATE=postgresql://odoo:odoo@127.0.0.1:25060/{db_name}

# SSH tunnel to reach Postgres in the Odoo droplet
Environment=SSH_HOST=188.166.183.13
Environment=SSH_PORT=22
Environment=SSH_USER=root
Environment=SSH_PASSWORD=Nexius-qt28-i8tc-azdi
Environment=DB_HOST_IN_DROPLET=172.18.0.2
Environment=DB_PORT=5432
Environment=LOCAL_PORT=25060
Environment=DB_USER=odoo
Environment=DB_PASSWORD=odoo

# Odoo server API (provisioning/admin)
Environment=ODOO_SERVER_URL=https://agent.nexiusagent.com
Environment=ODOO_MASTER_PASSWORD=Nexiuslabs2025
Environment=ODOO_TEMPLATE_DB=odoo_template
# Environment=ODOO_TEMPLATE_ADMIN_LOGIN=admin
# Environment=ODOO_TEMPLATE_ADMIN_PASSWORD=

# Your auth plugin in app/lg_auth.py expects custom auth:
# (keep whatever you already use; nothing special needed here)

# ---- Process command ----
# **Adjust venv path if different**
ExecStart=/root/lead_generation/.venv/bin/python3 -m langgraph_cli dev --port 8001 --allow-blocking --no-browser --no-reload
Environment="UVICORN_CMD_ARGS=--timeout-keep-alive 3600 --timeout-graceful-shutdown 3600"
# Graceful stop & restart policy
Restart=always
RestartSec=3
TimeoutStopSec=20
KillSignal=SIGINT

# Make the OOM killer less eager to kill us (not a guarantee)
OOMScoreAdjust=-500
# Useful hardening (optional)
NoNewPrivileges=yes
#ProtectSystem=full
#ProtectHome=yes

# Increase file descriptors a bit (optional)
LimitNOFILE=65535

[Install]
WantedBy=multi-user.target

_Note:_ For production, keep sensitive credentials out of the unit file by uncommenting `EnvironmentFile=/root/lead_generation/.env` and storing secrets there. After editing run `sudo systemctl daemon-reload && sudo systemctl restart langgraph`.

#### 3.2.2 leadgen-acra-direct.service

leadgen-acra-direct.service
------------------------

Runs the standalone ACRA enrichment script. Trigger it manually with `sudo systemctl start leadgen-acra-direct` whenever you need to backfill registry data.

[Unit]
Description=LeadGen ACRA Direct
After=network.target

[Service]
Type=oneshot
WorkingDirectory=/root/lead_generation
Environment="TROUBLESHOOT_API_LOG_DIR=/var/log/leadgen-api"
EnvironmentFile=/root/lead_generation/.env
ExecStart=/root/lead_generation/.venv/bin/python -m scripts.run_acra_direct

[Install]
WantedBy=multi-user.target



#### 3.2.3 langgraph-scheduler.service

langgraph-scheduler.service
------------------------

Executes `scripts/run_scheduler.py` (pattern discovery, nightly IC tasks). Usually fired via the timer below but can be run ad-hoc with `sudo systemctl start langgraph-scheduler`.

[Unit]
Description=LangGraph Scheduler Runner
After=network.target

[Service]
Type=oneshot
WorkingDirectory=/root/lead_generation
Environment="TROUBLESHOOT_API_LOG_DIR=/var/log/leadgen-api"
ExecStart=/root/lead_generation/.venv/bin/python scripts/run_scheduler.py
StandardOutput=journal
StandardError=journal


 langgraph-scheduler.timer
------------------------

Defines the cron schedule (daily 17:00 UTC / 01:00 SGT). Enable with `sudo systemctl enable --now langgraph-scheduler.timer`.

[Unit]
Description=Run LangGraph Scheduler daily at 01:00 SGT (17:00 UTC)

[Timer]
OnCalendar=*-*-* 17:00:00
AccuracySec=1min
Persistent=true

[Install]
WantedBy=timers.target


#### 3.2.4 leadgen-bg-worker.service

leadgen-bg-worker.service
------------------------

Keeps the “next 40” enrichment queue moving. It reads `.env` for db creds and restarts automatically on failure.

    [Unit]
    Description=LeadGen Next-40 Background Worker
    After=network.target

    [Service]
    Type=simple
    WorkingDirectory=/root/lead_generation
    Environment="TROUBLESHOOT_API_LOG_DIR=/var/log/leadgen-api"
    EnvironmentFile=/root/lead_generation/.env
    ExecStart=/root/lead_generation/.venv/bin/python -m scripts.run_bg_worker
    Restart=on-failure
    RestartSec=3

    [Install]
    WantedBy=multi-user.target


#### 3.2.5 odoo-tunnel.service

 odoo-tunnel.service
------------------------

Maintains the SSH port-forward so the API can reach the Odoo/Postgres database inside the ERP droplet. Make sure `/etc/odoo-tunnel.env` contains the same credentials referenced in `langgraph.service`.

    [Unit]
Description=Persistent SSH tunnel to Odoo Postgres via jump host
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
EnvironmentFile=/etc/odoo-tunnel.env
ExecStart=/usr/bin/sshpass -p ${SSH_PASSWORD} /usr/bin/ssh \
  -o PreferredAuthentications=password,keyboard-interactive \
  -o PubkeyAuthentication=no \
  -o StrictHostKeyChecking=accept-new \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -N -L 127.0.0.1:${LOCAL_PORT}:${DB_HOST_IN_DROPLET}:${DB_PORT} \
  -p ${SSH_PORT} ${SSH_USER}@${SSH_HOST}
Restart=always
RestartSec=5
User=root

[Install]
WantedBy=multi-user.target


## 4. Post-deploy Verification

1. `sudo systemctl status chat-nextjs langgraph leadgen-bg-worker odoo-tunnel` – all should be `active (running)`.
2. `curl -I https://chat.nexiusagent.com/healthz` and `curl -I https://langgraphapi.nexiusagent.com/healthz` – expect `200 OK`.
3. `sudo tail -f /var/log/nginx/*.log` – confirm no 502/504s immediately after a deploy; if you see them, double-check the proxy/read timeout alignment with the `.env` values above.
4. `journalctl -u langgraph -u chat-nextjs -f` – watch during a release to ensure services restart cleanly without permission errors.
