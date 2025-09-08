Agents Guide — lead_generation-main

Purpose
- Pre-SDR pipeline: normalize → ICP candidates → deterministic crawl + Tavily/Lusha → ZeroBounce verify → scoring + rationale → export → optional Odoo sync.

Run API (LangGraph server)
- python -m venv .venv && source .venv/bin/activate
- pip install -r requirements.txt
- uvicorn app.main:app --host 0.0.0.0 --port 2024
- Endpoints: /agent (LangServe), /export/latest_scores.csv, /docs

Run Orchestrator (one-off)
- source .venv/bin/activate
- python -m src.orchestrator

Key Env Vars (src/settings.py)
- POSTGRES_DSN (required)
- OPENAI_API_KEY, LANGCHAIN_MODEL=gpt-4o-mini, TEMPERATURE=0.3
- TAVILY_API_KEY?, ZEROBOUNCE_API_KEY?, LUSHA_API_KEY?, ENABLE_LUSHA_FALLBACK=true
- ICP_RULE_NAME=default, CRAWL_MAX_PAGES=6, EXTRACT_CORPUS_CHAR_LIMIT=35000
- ODOO_POSTGRES_DSN (or resolve per-tenant via odoo_connections)

Migrations
- Apply multi-tenant + MV: app/migrations/004_multi_tenant_icp.sql
- Odoo columns: app/migrations/001_presdr_odoo.sql

Tenancy & Auth (Section 6)
- Production: Validate Nexius SSO JWT, set request.state.tenant_id and roles.
- Enforce RLS/filters on tenant-owned tables; set GUC request.tenant_id per request.
- Dev: X-Tenant-ID header may be accepted for local testing only.

Common Ops
- Refresh MV: REFRESH MATERIALIZED VIEW CONCURRENTLY icp_candidate_companies;
- Export shortlist: curl "http://localhost:2024/export/latest_scores.csv?limit=200" -o shortlist.csv
- Logs: tail -f .logs/*.log

Troubleshooting
- Postgres connect errors: verify POSTGRES_DSN and DB reachable.
- Tavily/Lusha/ZeroBounce: missing keys → fallbacks/pathways skip gracefully; check settings flags.

