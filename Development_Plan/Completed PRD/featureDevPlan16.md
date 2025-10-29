**Dev Plan — Feature 16: Replace Lusha Fallback with Apify LinkedIn Actor**

Source: featurePRD16.md

Objective: Swap the contact-discovery fallback from Lusha to an Apify Actor that scrapes public LinkedIn profiles, integrate it into the enrichment fallback chain, keep costs within caps, and comply with ToS/PDPA. Roll out behind a feature flag.

**Architecture**
- Vendor module: `src/vendors/apify_linkedin.py` encapsulates Apify Actor calls (sync dataset endpoint preferred) with retries and timeouts.
- Integration: Enrichment contact-discovery step calls Apify when contacts are missing after crawl/Tavily.
- Data mapping: Normalize Actor items → `contacts` rows; only create `lead_emails` if email is present and verified.
- Config/flags: `ENABLE_APIFY_LINKEDIN=true`, `ENABLE_LUSHA_FALLBACK=false` to migrate traffic; `APIFY_TOKEN`, `APIFY_LINKEDIN_ACTOR_ID` for credentials.
- Caps/usage: Per-tenant daily cap (`APIFY_DAILY_CAP`) and batch sizing; surface usage in `run_vendor_usage` (Feature 8).

**Environment**
- `APIFY_TOKEN` (required; server-side secret)
- `APIFY_LINKEDIN_ACTOR_ID=dev_fusion~linkedin-profile-scraper`
- `APIFY_SYNC_TIMEOUT_S=600`
- `APIFY_DATASET_FORMAT=json` (accepted formats per Apify API)
- `APIFY_DAILY_CAP=50` (runs per day per tenant; adjust to your budget)
- `CONTACT_TITLES` (CSV fallback, or read from `icp_rules.payload.preferred_titles`)
- `ENABLE_APIFY_LINKEDIN=true`, `ENABLE_LUSHA_FALLBACK=false`

**DB touchpoints**
- `contacts` table (assumed present; unique index by (company_id, email) via migration 002).
- `lead_emails` (only if email present from Actor and verified via ZeroBounce).
- `contact_discovery_events` (optional new table to track vendor details; otherwise use `run_event_logs` from Feature 8).

**Vendor Module**
File: `lead_generation-main/src/vendors/apify_linkedin.py`
```python
from __future__ import annotations
import os, httpx, asyncio
from typing import List, Dict, Any, Optional

APIFY_BASE = "https://api.apify.com/v2"

def _token() -> str:
    t = os.getenv("APIFY_TOKEN")
    if not t:
        raise RuntimeError("Missing APIFY_TOKEN")
    return t

def _actor_id() -> str:
    return os.getenv("APIFY_LINKEDIN_ACTOR_ID", "dev_fusion~linkedin-profile-scraper")

async def run_sync_get_dataset_items(payload: Dict[str, Any], *, dataset_format: str = "json", timeout_s: int = 600) -> List[Dict[str, Any]]:
    url = f"{APIFY_BASE}/acts/{_actor_id().replace('/', '~')}/run-sync-get-dataset-items"
    params = {"token": _token(), "format": dataset_format}
    headers = {"Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        r = await client.post(url, params=params, json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and "items" in data:
            return data["items"]  # some actors wrap items
        if isinstance(data, list):
            return data
        return []

def build_queries(company_name: str, titles: List[str]) -> List[str]:
    t = " OR ".join([f'"{x}"' if ' ' in x else x for x in titles])
    q = f'"{company_name}" AND ({t})'
    return [q]

def normalize_contacts(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for it in items:
        out.append({
            "full_name": it.get("fullName") or it.get("name") or it.get("full_name"),
            "title": it.get("headline") or it.get("title"),
            "company_current": it.get("companyName") or it.get("company_current"),
            "linkedin_url": it.get("url") or it.get("profileUrl") or it.get("linkedin_url"),
            "location": it.get("locationName") or it.get("location"),
            # emails are rarely present in public profiles; keep only if provided explicitly
            "email": it.get("email") or None,
            "source_json": it,
        })
    return out
```

**Integration**
Entry points in enrichment (pseudocode diff):
```python
# lead_generation-main/src/enrichment.py
from src.vendors.apify_linkedin import run_sync_get_dataset_items, build_queries, normalize_contacts
from src.settings import ZEROBOUNCE_API_KEY

async def fetch_contacts_via_apify(company_id: int, company_name: str, titles: List[str]) -> List[dict]:
    payload = {"queries": build_queries(company_name, titles)}
    raw = await run_sync_get_dataset_items(payload, dataset_format=os.getenv("APIFY_DATASET_FORMAT","json"), timeout_s=int(os.getenv("APIFY_SYNC_TIMEOUT_S","600")))
    contacts = normalize_contacts(raw)
    # upsert into contacts; verify any provided emails via ZeroBounce
    upsert_contacts(company_id, contacts)
    verify_and_upsert_emails(company_id, [c["email"] for c in contacts if c.get("email")])
    return contacts

def upsert_contacts(company_id: int, contacts: List[dict]):
    from src.database import get_conn
    with get_conn() as conn, conn.cursor() as cur:
        for c in contacts:
            cols = ["company_id", "full_name", "email", "role_title", "linkedin_url", "location", "source", "source_json"]
            vals = [company_id, c.get("full_name"), c.get("email"), c.get("title"), c.get("linkedin_url"), c.get("location"), "apify_linkedin", Json(c.get("source_json"))]
            ph = ",".join(["%s"] * len(vals))
            cur.execute(f"INSERT INTO contacts ({','.join(cols)}) VALUES ({ph}) ON CONFLICT DO NOTHING", vals)

def verify_and_upsert_emails(company_id: int, emails: List[str]):
    if not emails or not ZEROBOUNCE_API_KEY:
        return
    # reuse existing ZeroBounce helpers (batch or per-email); on verification error, skip safely
    from src.enrichment import verify_email_batch  # example helper; implement if missing
    results = verify_email_batch(emails)
    from src.database import get_conn
    with get_conn() as conn, conn.cursor() as cur:
        for r in results:
            cur.execute(
                """
                INSERT INTO lead_emails (email, company_id, verification_status, smtp_confidence, source, last_verified_at)
                VALUES (%s,%s,%s,%s,%s,NOW())
                ON CONFLICT (email) DO UPDATE SET
                  verification_status = EXCLUDED.verification_status,
                  smtp_confidence = EXCLUDED.smtp_confidence,
                  source = EXCLUDED.source,
                  last_verified_at = EXCLUDED.last_verified_at
                """,
                (r["email"], company_id, r.get("status"), r.get("confidence"), "apify_linkedin"),
            )
```

Fallback wiring (where Lusha was used):
```python
if _env_true("ENABLE_APIFY_LINKEDIN") and not _has_contacts(cid):
    try:
        titles = icp_preferred_titles_for_tenant(tenant_id) or env_titles()
        contacts = await fetch_contacts_via_apify(cid, company_name, titles)
        obs.bump_vendor(run_id, tenant_id, "apify_linkedin", calls=1)
    except Exception as e:
        obs.bump_vendor(run_id, tenant_id, "apify_linkedin", calls=1, errors=1)
        degraded_reasons.append("APIFY_LINKEDIN_FAIL")
```

**Compliance**
- Only public LinkedIn data; respect ToS and robots; do not attempt to bypass protections.
- Do not guess emails; only store provided emails that are verified.
- Apply suppression lists and retention policies (PRD 9).

**Cost/Quotas**
- Enforce per-tenant daily calls via `APIFY_DAILY_CAP`; read historical count from `run_vendor_usage` for today and skip if exceeded.
- Bound payload per run: ≤ 25 company queries.

**Observability (Feature 8)**
- Log vendor usage via `obs.bump_vendor(…, 'apify_linkedin', …)`.
- Log per-company events (`stage='contact_discovery'` and `event='vendor_call'`).

**Testing**
- Mock httpx responses for Actor endpoints; assert normalization and DB upserts.
- Fault injection: HTTP 429/5xx/timeouts → fallbacks skip and run continues.

**Rollout**
- Phase 1: Add vendor module, envs, and enrich integration behind `ENABLE_APIFY_LINKEDIN` flag; leave Lusha disabled.
- Phase 2: Add caps and observability; QA on staging tenant; measure coverage and cost.
- Phase 3: Tenant-tunable titles; docs and runbooks; pilot enablement.

---

Implemented Integration Details (current code)
- Chain-first approach: company-by-name (`harvestapi~linkedin-company`) → employees (`harvestapi~linkedin-company-employees`) → profile details (`dev_fusion~linkedin-profile-scraper`).
- Helpers in `src/vendors/apify_linkedin.py`:
  - `company_to_profile_urls(company_name, max_items)` to resolve LinkedIn company URL and gather employee profile URLs.
  - `contacts_via_company_chain(company_name, titles, max_items)` to scrape details by profile URLs, then normalize and optionally title-filter.
  - `run_sync_get_dataset_items(payload, ...)` with payload schema variants and fallback to `run-sync` + dataset fetch.
  - `_run_actor_items(actor_id, payload, ...)` to call specific actors with unified behavior.
  - `build_queries(company_name, titles)` and `normalize_contacts(items)` for the queries-only fallback.
- Flags & env now supported:
  - `APIFY_COMPANY_ACTOR_ID`, `APIFY_EMPLOYEES_ACTOR_ID`, `APIFY_LINKEDIN_ACTOR_ID`, optional `APIFY_SEARCH_ACTOR_ID`.
  - `APIFY_USE_COMPANY_EMPLOYEE_CHAIN=true` (default), `APIFY_EMPLOYEES_SCRAPER_MODE` (optional), `APIFY_DEBUG_LOG_ITEMS`, `APIFY_LOG_SAMPLE_SIZE`.
- Logging & metrics:
  - Logs company URL resolution, employees count, profile actor received/filtered counts, and compact item samples when debug enabled.
  - Records vendor usage in `run_vendor_usage` (vendor='apify_linkedin') and per-company events in `run_event_logs` (stage='contact_discovery').
- Caps: Per-tenant `APIFY_DAILY_CAP` is checked before vendor calls; skip with a degraded reason when exceeded.
