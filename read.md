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
- pip install -r requirements.txt

Dependencies (from requirements.txt)
- python-dotenv, langchain-core, langchain-openai, langchain_community
- langgraph, langgraph-prebuilt
- asyncpg, psycopg2
- scikit-learn
- httpx, beautifulsoup4
- grandalf (diagramming)

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
- ENABLE_LANGSERVE_IN_APP: set to `true` to expose `/agent` routes for LangGraph development

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
ENABLE_LANGSERVE_IN_APP=true
```

## Local LangGraph Development

The API can expose a LangGraph at `/agent` for interactive testing.

1. Set `OPENAI_API_KEY` in your environment.
2. Set `ENABLE_LANGSERVE_IN_APP=true` to mount the `/agent` routes when running `uvicorn`.
3. Start the server: `uvicorn app.main:app --host 0.0.0.0 --port 2024`.
4. When using `npx langgraph dev` or LangGraph Studio, supply the same `OPENAI_API_KEY`; the key is used client-side for OpenAI calls.

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

## Security & Secrets
- `.gitignore` excludes `.env` and secrets. Never commit keys.
- If a secret is accidentally committed, rewrite history (e.g., `git filter-repo --invert-paths --path src/.env`) and rotate keys.

## Troubleshooting
- Push blocked due to secrets: remove secret from history and rotate keys
- `asyncpg` DataError on industry code: ensure `industry_code` is TEXT
- Missing tables: run the SQL migrations above
- ZeroBounce errors: missing/invalid key; enrichment continues without verified status
- `GET /whoami 403`: the request lacks a valid Nexius SSO token or `tenant_id` claim. Sign in and include `Authorization: Bearer <JWT>` when calling `/whoami`.
- `Failed to connect to LangGraph server`: ensure the API is running, `OPENAI_API_KEY` is set, and `ENABLE_LANGSERVE_IN_APP=true` to expose `/agent`.

## Notes
- Default OpenAI model is `gpt-4o-mini`; set `LANGCHAIN_MODEL` to change
- Crawler is robots-aware; respects `robots.txt` and uses a custom UA
- Industry fallback uses only industry codes for candidate selection (per project decision)
