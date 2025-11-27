# Lead Management Agent — Enrichment & Orchestration

Tavily + LLM + deterministic crawler pipeline for company enrichment, ICP candidate selection, and lead scoring.

Highlights
- Tavily/LLM extraction merged with deterministic crawler signals
- Deterministic, robots-aware site crawler (httpx + BeautifulSoup)
- ZeroBounce email verification (optional)
- Async normalization from `staging_acra_companies` → `companies`
- Orchestrator uses ICP rules first, then industry-code-only fallback

## Requirements
- Python 3.11+ (tested on 3.12)
- PostgreSQL 13+
- Recommended: virtualenv

Install
- python -m venv .venv
- source .venv/bin/activate
- pip install --upgrade pip
- pip install -r requirements.txt

Dev tools (optional)
- pip install -r requirements-dev.txt

Dependencies (from requirements.txt)
- fastapi>=0.115,<0.116 and starlette>=0.38.6
- httpx>=0.26,<0.28, requests, beautifulsoup4
- langchain, langchain-core, langchain-openai, langchain_community
- langgraph, langgraph-prebuilt, langgraph-sdk
- asyncpg, psycopg2, scikit-learn
- python-dotenv, PyJWT, passlib, apscheduler, grandalf (diagramming)

Notes on environments
- Always install in a fresh virtualenv; avoid using system Python packages.
- If you previously installed conflicting global packages (e.g., embedchain, crewai, langgraph-api), recreate the venv: `rm -rf .venv && python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.

## Environment Variables (credentials)
Create a local `.env` (never commit). GitHub Push Protection will block pushes if secrets are committed.

Required
- POSTGRES_DSN: e.g. `postgres://user:password@host:5432/dbname`
- OPENAI_API_KEY: used by LLM extraction, scoring rationale, and local LangGraph runs
- TAVILY_API_KEY: used by Tavily search

Optional / Recommended
- ZEROBOUNCE_API_KEY: validates emails; if empty, verification is skipped/degraded
- ICP_RULE_NAME: default `default`
- LANGCHAIN_MODEL: default `gpt-4o-mini`
- TEMPERATURE: default `0.3`
- CRAWL_MAX_PAGES: default `6` (total site pages after homepage)
- EXTRACT_CORPUS_CHAR_LIMIT: default `35000`
- ODOO_POSTGRES_DSN: connection string for your Odoo database; keep separate from `POSTGRES_DSN`

.env example (MCP adapters)
Create `.env` at project root with at least these keys (see `.env.example`):
```
JINA_API_KEY=your_jina_api_key_here
MCP_SERVER_URL=https://mcp.jina.ai/sse
ENABLE_MCP_READER=true
MCP_TRANSPORT=adapters_http
MCP_TIMEOUT_S=12.0
MCP_DUAL_READ_PCT=0
MCP_ADAPTER_USE_SSE=false
MCP_ADAPTER_USE_STANDARD_BLOCKS=true
```

MCP Reader (Jina) — Read URL
- ENABLE_MCP_READER: set to `true` to use Jina MCP `read_url` instead of HTTP `r.jina.ai` (default `false`)
- JINA_API_KEY: required when MCP is enabled (Authorization: Bearer)
- MCP_SERVER_URL: MCP server endpoint (default `https://mcp.jina.ai/sse`)
- MCP_TRANSPORT: `python`, `remote`, or `adapters_http` (default `python`; `remote` uses `npx mcp-remote`; `adapters_http` uses Python LangGraph MCP adapters)
- MCP_TIMEOUT_S: per-call timeout in seconds (default `12.0`)
- MCP_DUAL_READ_PCT: `0..100`; when >0, runs MCP and HTTP in parallel and returns HTTP for parity checks (default `0`)
- MCP_INIT_TIMEOUT_S: initialize handshake timeout in seconds (default `25`)
- MCP_NPX_PATH: override `npx` executable path if needed (default `npx`)
- MCP_PROTOCOL_VERSION: override protocol version sent in `initialize` (default `2024-10-07`)
- MCP_EXEC: optional path/name of a globally installed `mcp-remote` binary. If set (e.g., `mcp-remote`), the backend skips `npx` and executes it directly to reduce spawn overhead.
- MCP_READ_MAX_CONCURRENCY: max concurrent MCP `read_url` calls (default `2`; hard-clamped `1..16`)
- MCP_READ_MIN_INTERVAL_S / MCP_READ_JITTER_S: enforce a minimum spacing (plus jitter) between MCP calls to avoid bursting; defaults `0.35s`/`0.15s`
- MCP_READ_MAX_ATTEMPTS: capped retry attempts per URL before falling back (default `3`)
- MCP_READ_BACKOFF_BASE_S / MCP_READ_BACKOFF_CAP_S: exponential backoff base and ceiling used between attempts (defaults `0.6s` / `4s`)
- MCP_READ_RATELIMIT_BACKOFF_S: floor delay applied when a 429/rate-limit is detected (default `2s`)
Adapters (Python) — Optional
- MCP_ADAPTER_USE_SSE: when `true`, use SSE transport; otherwise use streamable HTTP (default `false`)
- MCP_ADAPTER_USE_STANDARD_BLOCKS: when `true`, standardize tool outputs for robust text extraction (default `true`)


Rollout guidance
1) Set `ENABLE_MCP_READER=true` and `MCP_DUAL_READ_PCT=50` in staging; verify dashboards for success rate (≥95%), latency, and content variance.
2) Increase or tune cleaner if variance >5%. When stable, set `MCP_DUAL_READ_PCT=0` to return MCP responses end-to-end.
3) For production, canary on a low-risk tenant; target ≥98% success before full cutover. Keep HTTP path available for quick rollback by flipping `ENABLE_MCP_READER=false`.
4) If using `MCP_TRANSPORT=remote`, ensure Node.js and `npx` are available on the host. The client spawns `mcp-remote` with `Authorization: Bearer $JINA_API_KEY`.
   - First run may take longer while `npx` fetches `mcp-remote`; consider increasing `MCP_INIT_TIMEOUT_S` (e.g., `30`).
   - Sanity check: `npx -y mcp-remote --help` should print usage.
   - Performance: Install `mcp-remote` globally and set `MCP_EXEC=mcp-remote` to avoid `npx` startup cost per call.

Session reuse
- The backend now reuses a single `mcp-remote` stdio session per `(MCP_SERVER_URL, JINA_API_KEY)` pair and caches the selected `read_url` tool.
- This eliminates per-call process spawn and tool discovery costs during a run.

Quick MCP check
- Verify config and wiring with:
```
python scripts/check_mcp_read_url.py https://example.com --force-mcp
```
- Expected logs (info): lines starting with `[mcp]` such as `starting read_url`, `calling tool=read_url`, and `call done ... text_len=...`. The script prints config snapshot, result length, and a short preview.

Email results (SendGrid)
- ENABLE_EMAIL_RESULTS: set to `true` to enable emails
- SENDGRID_API_KEY: your SendGrid API key
- SENDGRID_FROM_EMAIL: verified sender (Single Sender or authenticated domain)
- SENDGRID_TEMPLATE_ID: optional dynamic template id (not required)
- DEFAULT_NOTIFY_EMAIL: optional fallback recipient when no JWT/header email is present (useful in dev)
- EMAIL_DEV_ACCEPT_TENANT_USER_ID_AS_EMAIL: default `true`; if `true`, accept `tenant_users.user_id` as an email when it contains `@` (dev convenience)

PRD‑Opt (SG Profiles)
- ICP_SG_PROFILES: set to `1` to enable Singapore‑focused discovery gating, denylists, and markers.
- SG_PROFILES_CONFIG: path to YAML (default `config/sg_profiles.yaml`) to override profiles/weights/deny rules.
- DDG_KL: optional region hint like `sg-en` (automatically applied when SG profiles are enabled).

Examples:
```
ICP_SG_PROFILES=1
SG_PROFILES_CONFIG=config/sg_profiles.yaml
DDG_KL=sg-en
```
 

Example `.env` (do not use real keys here)
```
POSTGRES_DSN=postgres://USER:PASSWORD@HOST:5432/DB
ODOO_POSTGRES_DSN=postgres://odoo:odoo@localhost:25060/demo
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
ZEROBOUNCE_API_KEY=zb-...
ICP_RULE_NAME=default
LANGCHAIN_MODEL=gpt-4o-mini
TEMPERATURE=0.3
CRAWL_MAX_PAGES=6
EXTRACT_CORPUS_CHAR_LIMIT=35000
ENABLE_EMAIL_RESULTS=true
SENDGRID_API_KEY=SG.xxxxxx
SENDGRID_FROM_EMAIL=no-reply@yourdomain.com
DEFAULT_NOTIFY_EMAIL=ops@yourdomain.com
EMAIL_DEV_ACCEPT_TENANT_USER_ID_AS_EMAIL=true
 
```

 

## Odoo Integration

To extend an Odoo instance with enrichment fields, connect directly to its PostgreSQL
database using a separate DSN. The migration script can open an SSH tunnel when the
following variables are provided (example droplet):

```
SSH_HOST=188.166.183.13
SSH_PORT=22
SSH_USER=root
SSH_PASSWORD=My_password
DB_HOST_IN_DROPLET=172.18.0.2
DB_PORT=5432
DB_NAME=demo
DB_USER=odoo
DB_PASSWORD=odoo
LOCAL_PORT=25060
```

If `SSH_PASSWORD` is set, the migration script relies on `sshpass` to feed the
password to `ssh`. Install it before running:

- Debian/Ubuntu: `sudo apt-get install sshpass`
- macOS (Homebrew): `brew install hudochenkov/sshpass/sshpass`

Alternatively, omit `SSH_PASSWORD` and use key-based authentication.

Run the migration; the script will forward `LOCAL_PORT` to the droplet and build the
DSN automatically if `ODOO_POSTGRES_DSN` isn't set:

```
python scripts/run_odoo_migration.py
```

This keeps your application database (`POSTGRES_DSN`) independent from Odoo's
database connection.

## Database Schema & Migrations
The code expects the following tables/columns. Adjust to your schema as needed.

1) summaries (crawler persistence)
```
CREATE TABLE IF NOT EXISTS summaries (
  id BIGSERIAL PRIMARY KEY,
  company_id BIGINT NOT NULL,
  url TEXT,
  title TEXT,
  description TEXT,
  content_summary TEXT,
  key_pages JSONB,
  signals JSONB,
  rule_score NUMERIC,
  rule_band TEXT,
  shortlist JSONB,
  crawl_metadata JSONB,
  created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_summaries_company_id ON summaries(company_id);
```

2) company_enrichment_runs (projection for downstream)
```
CREATE TABLE IF NOT EXISTS company_enrichment_runs (
  id BIGSERIAL PRIMARY KEY,
  company_id BIGINT NOT NULL,
  run_timestamp TIMESTAMPTZ DEFAULT now(),
  about_text TEXT,
  tech_stack JSONB,
  public_emails JSONB,
  jobs_count INT,
  linkedin_url TEXT
);
CREATE INDEX IF NOT EXISTS idx_enrich_runs_company_id ON company_enrichment_runs(company_id);
```

3) companies (canonical; add columns if missing)
```
ALTER TABLE companies
  ADD COLUMN IF NOT EXISTS industry_code TEXT,
  ADD COLUMN IF NOT EXISTS website_domain TEXT,
  ADD COLUMN IF NOT EXISTS about_text TEXT,
  ADD COLUMN IF NOT EXISTS tech_stack JSONB,
  ADD COLUMN IF NOT EXISTS email JSONB,
  ADD COLUMN IF NOT EXISTS phone_number JSONB,
  ADD COLUMN IF NOT EXISTS hq_city TEXT,
  ADD COLUMN IF NOT EXISTS hq_country TEXT,
  ADD COLUMN IF NOT EXISTS linkedin_url TEXT;
```
Note: Arrays are handled as JSONB in this pipeline. If your schema uses `text[]`, adapt the SQL write to cast accordingly.

## How It Works
1) Normalization (src/icp.py)
- Reads raw rows from `staging_acra_companies`
- Normalizes and upserts into `companies` (ensures `industry_code` is text)

2) ICP Refresh (src/icp.py)
- Produces candidate company_ids based on the configured ICP rules

3) Orchestrator (src/orchestrator.py)
- Runs normalization → ICP refresh
- Fallback if no candidates: derive industry codes from `icp_payload["industries"]` using
  - `fetch_industry_codes_by_names(industries)` (staging description match, fallback to existing `companies.industry_norm` only to DERIVE codes)
  - Fetches candidates strictly via `fetch_candidate_ids_by_industry_codes(codes)`
- Enrichment: Tavily/LLM merged with deterministic crawler signals
- Lead scoring and output

4) Enrichment (src/enrichment.py)
- Tavily search + LLM extraction (LangChain Runnable + `.invoke()`)
- Deterministic crawler merges signals (emails, phones, tech, pricing pages, etc.)
- Persist to `summaries`, project into `company_enrichment_runs`, update `companies`
- ZeroBounce email verification (if key provided)

### Domain Discovery Heuristics
- Exact-match search: tries `"<Company Name>" "official website"` then `site:.sg` before fallbacks.
- Brand/.sg filter: keeps only `.sg` domains or exact brand apex matches (e.g., `acme.com`).
- Marketplace/aggregator rejection: discards results from marketplaces, directories, or socials (e.g., `linkedin.com`, `shopee.sg`, `amazon.com`), unless the apex exactly equals the brand (e.g., Amazon).

## Running
- Ensure DB and `.env` are set
- source .venv/bin/activate
- python3 src/orchestrator.py

Logs & Output
- Candidate IDs, enrichment steps, and lead scores are printed to console
- Use `output_candidate_records()` for quick JSON snapshots of `companies`

## Scheduler & Cron
- Start the async scheduler: `python lead_generation-main/scripts/run_scheduler.py`

## PRD‑Opt Lead Profiles
- Default lead profile is `sg_employer_buyers`. Available profiles: `sg_employer_buyers`, `sg_referral_partners`, `sg_generic_leads`.
- To change the default globally, set `profile:` in `config/sg_profiles.yaml`:
```
profile: sg_referral_partners
```
- To override per‑run programmatically (advanced), pass a `lead_profile` key into the `icp_profile` you send to the agent Top‑10 planner (e.g., when invoking `src.agents_icp.plan_top10_with_reasons(icp_profile, tenant_id)`). The planner will carry `lead_profile` into `ai_metadata` for Top‑10 and the staged remainder.

## Email Results (SendGrid)

Overview
- When enabled, the backend emails a shortlist summary and attaches a CSV.
- Top‑10 (chat) sends immediately after enrichment; Next‑40 (background) sends once the job completes.
- Recipient resolution: header `X-Notify-Email` (dev override) → JWT email → `tenant_users.user_id` (when it contains `@` and the dev guard is enabled) → `DEFAULT_NOTIFY_EMAIL`.

Configuration
- Set `ENABLE_EMAIL_RESULTS=true` and provide `SENDGRID_API_KEY` and `SENDGRID_FROM_EMAIL`.
- `SENDGRID_FROM_EMAIL` must be a verified Single Sender or from an authenticated domain (SPF/DKIM) in SendGrid.
- Optional: `DEFAULT_NOTIFY_EMAIL` for dev/staging convenience.

Test endpoints
- Simple SendGrid smoke test (no DB/LLM):
  - `curl -X POST "http://localhost:8000/email/test?to=you@example.com&simple=true"`
- Agentic email (uses DB renderer + LLM intro/subject):
  - `curl -X POST "http://localhost:8000/email/test?to=you@example.com&simple=false&tenant_id=1104"`

Notes on delivery
- A 202 from SendGrid means “accepted”; delivery can still be delayed or filtered.
- If emails aren’t received:
  - Check Gmail Spam/Promotions.
  - Verify the sender in SendGrid (Single Sender or domain authentication).
  - Check SendGrid Email Activity for events (processed/delivered/deferred/dropped).
  - Inspect suppression lists (bounces/blocks/invalid/spam_reports) and remove entries if appropriate.

Attachment behavior
- Emails include a CSV attachment (`shortlist_tenant_<ID>.csv`).
- Attachment row cap defaults to 500 to keep size reasonable. If you need a different limit, update the sender tool to pass a different limit for `build_csv_bytes`.

## Running Tests
Run tests from the repo root, pointing `PYTHONPATH` at the backend package:
```
PYTHONPATH=lead_generation-main pytest -q
```
If your environment disallows writing to `/tmp`, configure a tmpdir for `pytest` via `--basetemp`.

- Control start time via `SCHED_START_CRON` (default `0 1 * * *` for 01:00 SGT)
- Limits & caps:
  - `SCHED_DAILY_CAP_PER_TENANT` (default 20)
  - `SCHED_COMPANY_BATCH_SIZE` (per-batch company count)
  - Tavily cap (coarse): `TAVILY_MAX_QUERIES` (units ≈ search + crawl + extract calls)
  - Contacts cap (coarse): `APIFY_DAILY_CAP`
  - ZeroBounce caps: `ZEROBOUNCE_MAX_VERIFICATIONS`, `ZEROBOUNCE_BATCH_SIZE`
- Admin kickoff HTTP:
  - Run all tenants: `POST /admin/runs/nightly`
  - One tenant: `POST /admin/runs/nightly?tenant_id=<id>`
  - Requires admin role; cookie-based auth (`nx_access`)

## Security & Secrets
- `.gitignore` excludes `.env` and secrets. Never commit keys.
- If a secret is accidentally committed, rewrite history (e.g., `git filter-repo --invert-paths --path src/.env`) and rotate keys.

## Troubleshooting
- Push blocked due to secrets: remove secret from history and rotate keys
- `asyncpg` DataError on industry code: ensure `industry_code` is TEXT
- Missing tables: run the SQL migrations above
- ZeroBounce errors: missing/invalid key; enrichment continues without verified status
- `GET /whoami 403`: the request lacks a valid Nexius SSO token or `tenant_id` claim. Sign in and include `Authorization: Bearer <JWT>` when calling `/whoami`.
- `Failed to connect to API`: ensure the API is running and `OPENAI_API_KEY` is set when using LLM-backed features.

## Notes
- Default OpenAI model is `gpt-4o-mini`; set `LANGCHAIN_MODEL` to change
- Crawler is robots-aware; respects `robots.txt` and uses a custom UA
- Industry fallback uses only industry codes for candidate selection (per project decision)
