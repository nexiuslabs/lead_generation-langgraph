#!/usr/bin/env python3
"""
Dev helper: Enrich a single company by domain using the interactive fast-path.

- Sets env to disable enrichment skips for this run unless overridden.
- Ensures a companies row exists for the provided domain.
- Invokes enrich_company_with_tavily(search_policy='require_existing')

Usage:
  python scripts/dev_enrich_one.py --domain example.com [--tenant-id 1234] [--company-name "Example, Inc."]

Environment you might set:
  POSTGRES_DSN=postgresql://user:pass@host:port/db
  DEFAULT_TENANT_ID=<id>   (or pass --tenant-id)
  ENRICH_SKIP_IF_ANY_HISTORY=false
  ENRICH_RECHECK_DAYS=0
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

# Ensure repository root is on sys.path so `src.*` can be imported
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Load .env files from repo root (POSTGRES_DSN, etc.) before importing settings
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(_ROOT / ".env")
    load_dotenv(_ROOT / "src/.env")
except Exception:
    pass


def _canonical_domain(raw: str) -> str:
    r = (raw or "").strip()
    if not r:
        return r
    if not r.startswith("http"):
        r = "https://" + r
    try:
        p = urlparse(r)
        host = (p.netloc or r).lower()
        # strip leading www.
        if host.startswith("www."):
            host = host[4:]
        # drop port if any
        if ":" in host:
            host = host.split(":", 1)[0]
        return host
    except Exception:
        return r.lower()


def _ensure_env(tenant_id: int | None):
    # Make sure skip rules are disabled for interactive verification
    os.environ.setdefault("ENRICH_SKIP_IF_ANY_HISTORY", "false")
    os.environ.setdefault("ENRICH_RECHECK_DAYS", "0")
    if tenant_id is not None:
        os.environ["DEFAULT_TENANT_ID"] = str(tenant_id)


def _ensure_company(domain: str, name: str | None) -> int:
    from src.database import get_conn
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT company_id FROM companies WHERE LOWER(website_domain)=LOWER(%s) LIMIT 1", (domain,))
        row = cur.fetchone()
        if row and row[0] is not None:
            return int(row[0])
        cur.execute(
            "INSERT INTO companies(name, website_domain, last_seen) VALUES (%s,%s, NOW()) RETURNING company_id",
            (name or domain, domain),
        )
        return int(cur.fetchone()[0])


async def _run(company_id: int, company_name: str | None):
    # Import after setting env so settings are read with overrides
    from src.enrichment import enrich_company_with_tavily, set_vendor_caps
    logging.getLogger("jina_reader").setLevel(logging.INFO)
    logging.getLogger("src.enrichment").setLevel(logging.INFO)
    logging.getLogger("enrichment").setLevel(logging.INFO)
    # Keep Apify to a single attempt per run in dev to avoid loops/costs
    try:
        set_vendor_caps(contact_lookups=1)
    except Exception:
        pass
    state = await enrich_company_with_tavily(company_id, company_name=company_name, search_policy="require_existing")
    print("--- Enrichment final state (summary) ---")
    print({
        "company_id": state.get("company_id"),
        "home": state.get("home"),
        "extracted_pages": len(state.get("extracted_pages") or []),
        "emails": len((state.get("data") or {}).get("email") or []),
        "phones": len((state.get("data") or {}).get("phone_number") or []),
        "linkedin_url": (state.get("data") or {}).get("linkedin_url"),
        "completed": state.get("completed"),
        "error": state.get("error"),
        "degraded_reasons": state.get("degraded_reasons"),
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True, help="Company website (domain or URL)")
    parser.add_argument("--tenant-id", type=int, default=None, help="Tenant id for run context (DEFAULT_TENANT_ID)")
    parser.add_argument("--company-name", default=None, help="Optional display name for company")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    domain = _canonical_domain(args.domain)
    _ensure_env(args.tenant_id)
    # Import after env to ensure settings are loaded post-override and .env is read
    from src.settings import POSTGRES_DSN  # noqa: F401

    cid = _ensure_company(domain, args.company_name)
    logging.info("Company ready id=%s domain=%s", cid, domain)
    asyncio.run(_run(cid, args.company_name))


if __name__ == "__main__":
    main()
