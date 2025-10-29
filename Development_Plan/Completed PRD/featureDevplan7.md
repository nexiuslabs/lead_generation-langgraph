# Dev Plan — Feature 7: Scheduling & Throughput

Source: featurePRD7.md (Scheduling & Throughput)

Objective: Implement a nightly, per-tenant pipeline that refreshes data, enriches candidates, verifies emails, scores leads, and upserts to Odoo within a 90-minute SLA for 1k shortlisted companies, with concurrency, rate limits, and graceful degradation.

---

## Implementation Overview

- Orchestrate a per-tenant Directed Acyclic Graph (DAG) of tasks:
  1) Refresh companies from ACRA staging → upsert into `companies`
  2) Refresh ICP candidates (per tenant)
  3) Select daily target set (cap)
  4) Deterministic crawl (per-domain rate limit)
  5) Tavily fallback → merge raw → AI extraction (guard tokens)
  6) Lusha fallback (contacts)
  7) ZeroBounce verification (batched)
  8) Scoring + rationale + persist
  9) Odoo pre-create/update companies & contacts (idempotent)

- Scheduling options:
  - CLI: `scripts/run_nightly.py` (single-run or immediate kickoff with flags)
  - APScheduler: `scripts/run_scheduler.py` with cron trigger at 01:00 SGT
  - Either can iterate tenants with bounded concurrency.

- RLS and multi-tenancy:
  - For non-HTTP runs, set `DEFAULT_TENANT_ID` for each tenant worker so DB GUC `request.tenant_id` is applied (code already reads env in `lead_scoring.py`).
  - Writes to tenant-owned tables (lead_features, lead_scores, enrichment_runs) honor RLS when migrations are applied.

---

## New/Updated Environment Variables

- Core scheduler
  - `SCHED_ENABLED=true|false` (default: true)
  - `SCHED_START_CRON="0 1 * * *"` (01:00 SGT)
  - `SCHED_TENANT_CONCURRENCY=3` (run tenants in parallel)
  - `SCHED_COMPANY_CONCURRENCY=8` (per-tenant, per-company workers)
  - `SCHED_DAILY_CAP_PER_TENANT=1000`
  - `SCHED_TENANT_INCLUDE` / `SCHED_TENANT_EXCLUDE` (CSV of tenant_ids)

- Thresholds & budgets (align with PRD 10)
  - `SHORTLIST_DAILY_CAP` (alias of `SCHED_DAILY_CAP_PER_TENANT`)
  - `SCORE_MIN_EXPORT=0.66`
  - `OPENAI_DAILY_TOKENS_CAP`, `TAVILY_MAX_QUERIES`, `LUSHA_MAX_CONTACT_LOOKUPS`, `ZEROBOUNCE_MAX_VERIFICATIONS`

- Existing used settings
  - `CRAWL_MAX_PAGES`, `CRAWLER_TIMEOUT_S`, `CRAWLER_USER_AGENT`, `EXTRACT_CORPUS_CHAR_LIMIT`, `PERSIST_CRAWL_CORPUS`

---

## Data Model Touchpoints

- `companies`: upserted nightly using staging ACRA rows (if present)
- `lead_features`, `lead_scores`: scored outputs; ensure `tenant_id` columns exist where migrations applied
- `enrichment_runs`: run bookkeeping (id, started/finished); used by `company_enrichment_runs`
- `company_enrichment_runs`: per-company snapshots and metrics (best-effort dynamic insert)
- Optional: `icp_candidate_companies` MV (or cached set)

---

## Tenant Discovery

SQL (sync; Python snippet in runner below):

```
SELECT t.tenant_id
FROM tenants t
JOIN odoo_connections oc ON oc.tenant_id = t.tenant_id AND oc.active = TRUE
WHERE t.status = 'active'
ORDER BY t.tenant_id;
```

Filter by `SCHED_TENANT_INCLUDE`/`SCHED_TENANT_EXCLUDE` when provided.

---

## Candidate Refresh and Target Selection

Use `icp_refresh_agent` to compute candidate IDs, then prioritize those missing features/contacts and cap by daily limit:

```
-- Within the tenant GUC context
SELECT c.company_id
FROM companies c
LEFT JOIN lead_scores s ON s.company_id = c.company_id
LEFT JOIN contacts k ON k.company_id = c.company_id
WHERE c.company_id = ANY($1)
ORDER BY (s.company_id IS NULL) DESC,
         (k.company_id IS NULL) DESC,
         c.last_seen DESC NULLS LAST
LIMIT $2;
```

---

## Per-Domain Rate Limiting (Deterministic Crawl)

Simple in-process limiter without extra deps:

```python
import time
from urllib.parse import urlparse
from collections import defaultdict
import asyncio

class DomainLimiter:
    def __init__(self, min_interval_s: float = 0.5):
        self.min_interval = min_interval_s
        self.last: dict[str, float] = defaultdict(lambda: 0.0)
        self.locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def wait(self, url: str):
        host = urlparse("http://" + (url or "")).hostname or "_"
        lock = self.locks[host]
        async with lock:
            now = time.time()
            delta = now - self.last[host]
            if delta < self.min_interval:
                await asyncio.sleep(self.min_interval - delta)
            self.last[host] = time.time()
```

Use before each HTTP fetch in deterministic crawl step.

---

## Runner — Code Skeleton (CLI)

File: `scripts/run_nightly.py` (new)

```python
import os
import asyncio
import logging
from typing import List
from urllib.parse import urlparse

from src.icp import icp_refresh_agent
from src.enrichment import enrich_company_with_tavily
from src.lead_scoring import lead_scoring_agent
from src.database import get_conn, get_pg_pool
from app.odoo_store import OdooStore

LOG = logging.getLogger("nightly")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")

TENANT_CONC = int(os.getenv("SCHED_TENANT_CONCURRENCY", "3"))
COMPANY_CONC = int(os.getenv("SCHED_COMPANY_CONCURRENCY", "8"))
DAILY_CAP    = int(os.getenv("SCHED_DAILY_CAP_PER_TENANT", os.getenv("SHORTLIST_DAILY_CAP", "1000")))

async def list_active_tenants() -> List[int]:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT t.tenant_id
            FROM tenants t
            JOIN odoo_connections oc ON oc.tenant_id=t.tenant_id AND oc.active
            WHERE t.status='active'
            ORDER BY t.tenant_id
        """)
        return [int(r[0]) for r in cur.fetchall()]

def select_target_set(candidates: List[int], limit: int) -> List[int]:
    if not candidates:
        return []
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.company_id
            FROM companies c
            LEFT JOIN lead_scores s ON s.company_id=c.company_id
            LEFT JOIN contacts k ON k.company_id=c.company_id
            WHERE c.company_id = ANY(%s)
            ORDER BY (s.company_id IS NULL) DESC,
                     (k.company_id IS NULL) DESC,
                     c.last_seen DESC NULLS LAST
            LIMIT %s
            """,
            (candidates, limit),
        )
        return [r[0] for r in cur.fetchall()]

async def enrich_many(company_ids: List[int]):
    sem = asyncio.Semaphore(COMPANY_CONC)
    async def _one(cid: int):
        async with sem:
            # Best-effort: derive website URL from DB if available (not shown here)
            # Fallback to enrichment without URL (function handles)
            try:
                await enrich_company_with_tavily(cid, None)
            except Exception as e:
                LOG.warning("enrich failed company_id=%s err=%s", cid, e)
    await asyncio.gather(*[_one(cid) for cid in company_ids])

async def run_tenant(tenant_id: int):
    os.environ["DEFAULT_TENANT_ID"] = str(tenant_id)  # enable RLS for non-HTTP ops
    LOG.info("tenant=%s start", tenant_id)

    # 1) Refresh ICP candidates via graph
    icp_state = await icp_refresh_agent.ainvoke({
        "rule_name": os.getenv("ICP_RULE_NAME", "default"),
        "payload": {
            "industries": ["Technology"],
            "employee_range": {"min": 2, "max": 100},
            "incorporation_year": {"min": 2000, "max": 2025},
        },
    })
    candidates = icp_state.get("candidate_ids", [])
    targets = select_target_set(candidates, DAILY_CAP)
    LOG.info("tenant=%s candidates=%d targets=%d", tenant_id, len(candidates), len(targets))

    # 2) Enrichment pipeline
    await enrich_many(targets)

    # 3) Score + rationale + persist
    scoring_state = await lead_scoring_agent.ainvoke({
        "candidate_ids": targets,
        "lead_features": [],
        "lead_scores": [],
        "icp_payload": {
            "industries": ["Technology"],
            "employee_range": {"min": 2, "max": 100},
            "incorporation_year": {"min": 2000, "max": 2025},
        },
    })
    LOG.info("tenant=%s scored=%d", tenant_id, len(scoring_state.get("lead_scores", [])))

    # 4) Export to Odoo (idempotent upserts + leads over threshold)
    try:
        store = OdooStore(tenant_id=tenant_id)
        for s in scoring_state.get("lead_scores", []):
            # Best-effort upsert company; add contact if email exists (omitted here)
            # For high scores, create a lead
            pass
    except Exception as e:
        LOG.warning("tenant=%s odoo export skipped: %s", tenant_id, e)

async def run_all():
    tenants = await asyncio.to_thread(list_active_tenants)
    include = set([int(x) for x in (os.getenv("SCHED_TENANT_INCLUDE", "").split(",")) if x.strip().isdigit()])
    exclude = set([int(x) for x in (os.getenv("SCHED_TENANT_EXCLUDE", "").split(",")) if x.strip().isdigit()])
    if include:
        tenants = [t for t in tenants if t in include]
    if exclude:
        tenants = [t for t in tenants if t not in exclude]

    sem = asyncio.Semaphore(TENANT_CONC)
    async def _one(tid: int):
        async with sem:
            await run_tenant(tid)
    await asyncio.gather(*[_one(t) for t in tenants])

if __name__ == "__main__":
    asyncio.run(run_all())
```

Notes:
- The skeleton reuses existing agents (`icp_refresh_agent`, `lead_scoring_agent`) and `OdooStore`.
- Enrichment uses `enrich_company_with_tavily` (deterministic + fallbacks inside module). Add explicit Lusha/ZeroBounce batch helpers as needed.
- Ensure robust logging and error handling around each step.

---

## Optional — APScheduler Entrypoint

File: `scripts/run_scheduler.py` (new)

```python
import os
import asyncio
import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from zoneinfo import ZoneInfo
from run_nightly import run_all  # import function from the CLI file above

logging.basicConfig(level=logging.INFO)

async def job():
    try:
        await run_all()
    except Exception as e:
        logging.exception("nightly run failed: %s", e)

if __name__ == "__main__":
    tz = ZoneInfo("Asia/Singapore")
    cron = os.getenv("SCHED_START_CRON", "0 1 * * *")
    minute, hour, dom, month, dow = cron.split()
    sched = AsyncIOScheduler(timezone=tz)
    sched.add_job(job, CronTrigger(minute=minute, hour=hour, day=dom, month=month, day_of_week=dow))
    sched.start()
    try:
        asyncio.get_event_loop().run_forever()
    except KeyboardInterrupt:
        pass
```

---

## Enhancements to Existing Modules (as needed)

- `src/enrichment.py`
  - Add `verify_emails_batch(emails: List[str]) -> List[dict]` using ZeroBounce with cache guards (present helpers exist: `_ensure_email_cache_table`, `_cache_get/_cache_set`).
  - Expose `enrich_company_with_tavily(company_id: int, name: Optional[str])` (already used by `orchestrator.py`).

- `src/icp.py`
  - Ensure `_select_icp_candidates` respects tenant GUC via `DEFAULT_TENANT_ID` (DB connection layer handles GUC).

- `src/lead_scoring.py`
  - Already sets GUC from `DEFAULT_TENANT_ID`; keep as-is.

- `app/main.py` and `app/odoo_store.py`
  - Export logic is available; reuse `OdooStore` directly in the runner to avoid HTTP.

---

## Flow Summary (Text Diagram)

Tenant loop (≤ `SCHED_TENANT_CONCURRENCY` in parallel):
- Set `DEFAULT_TENANT_ID` → Refresh ICP → Select target set (≤ cap)
- For targets (≤ `SCHED_COMPANY_CONCURRENCY` workers):
  - Crawl site (respect robots, per-domain rate-limit) → Tavily fallback → AI extract
  - Lusha fallback for contacts → ZeroBounce batch verify (cache)
- Score → Bucket → Rationale → Persist
- Odoo upsert companies + contacts; create leads for score ≥ `SCORE_MIN_EXPORT`

---

## Testing & Acceptance

- Dry-run mode: add `--dry-run` to avoid vendor calls and Odoo writes; assert query sets and ordering logic.
- Two-tenant test: run with `SCHED_TENANT_INCLUDE=a,b` and cap=100; verify runtime and per-tenant isolation via DB logs.
- Verify that Authorization headers are not needed; cookies not used in CLI; RLS via env GUC works (reads/writes scoped by tenant).
- SLA check: log wall-clock start/finish; ensure ≤ 90 minutes for 1000 targets on reference hardware.

---

## Rollout Steps

1) Land runner scripts and env variables; update deployment docs.
2) Validate on staging with 2 tenants; measure timings; tune concurrency and caps.
3) Enable APScheduler in a worker process or external cron; stagger tenants if needed.
4) Monitor vendor usage and costs; adjust max caps per PRD 10.
5) Add additional observability (PRD 8) as follow-up.


---

## Current Code Notes & DB Alignment

- Existing components ready to reuse:
  - Enrichment (deterministic crawl + Tavily + Lusha) in `src/enrichment.py`.
  - Lead scoring graph with GUC support in `src/lead_scoring.py`.
  - One-off orchestrator for a single tenant in `src/orchestrator.py`.
  - Odoo upsert/export via `app/odoo_store.py`.
- Postgres schema (Postgres_DB_Schema.sql) aligns with the plan:
  - Tenant-scoped tables with RLS enabled: `enrichment_runs`, `lead_features`, `lead_scores`, `icp_rules`.
  - Company/contact data: `companies`, `contacts`, `lead_emails`, `company_enrichment_runs` (FKs to `companies`/`enrichment_runs`).
  - Tenancy + mapping: `tenants`, `tenant_users`, `odoo_connections`, `onboarding_status`.
  - ACRA staging inputs: `staging_acra_companies`, `staging_raw_acra`.
  - Reference: `ssic_ref` for ICP by SSIC mapping.

Optional performance indexes to consider (if selection becomes hot):
```
CREATE INDEX IF NOT EXISTS idx_contacts_company ON contacts(company_id);
CREATE INDEX IF NOT EXISTS idx_lead_scores_score ON lead_scores(score DESC NULLS LAST);
CREATE INDEX IF NOT EXISTS idx_companies_last_seen ON companies(last_seen DESC NULLS LAST);
```

## SLA & Security

- SLA: capture per-step timings (refresh, enrich, score, export) in logs; optional summary row keyed by `enrichment_runs.run_id`.
- Security/RLS: workers set `DEFAULT_TENANT_ID` before DB use; HTTP routes continue to set `request.tenant_id`.
