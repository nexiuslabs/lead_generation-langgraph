#!/usr/bin/env python3
"""
Run enrichment for a single company from a simple JSON payload.

Accepted payload shape (via --json or --file):
{
  "tenant_id": 123,                 # optional; also accepted via --tenant-id
  "domain": "example.com",         # required
  "company_name": "Example Pte Ltd" # optional; improves DR contacts
}

Examples:
  python scripts/enrich_from_payload.py --json '{"tenant_id":1,"domain":"foodempire.com","company_name":"Food Empire"}'
  python scripts/enrich_from_payload.py --file payload.json
  python scripts/enrich_from_payload.py --domain foodempire.com --company-name "Food Empire" --tenant-id 1

Behavior:
  - Ensures a row exists in companies for the domain (inserts if missing).
  - Disables enrichment skip guards for this run (override with env if needed).
  - Invokes enrich_company_with_tavily(search_policy='require_existing').
  - Prints a small JSON summary of results.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Load .env early so DB/keys are available
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(_ROOT / ".env")
    load_dotenv(_ROOT / "src/.env")
except Exception:
    pass


def _canonical_domain(raw: str) -> str:
    v = (raw or "").strip()
    if not v:
        return v
    if not v.startswith("http"):
        v = "https://" + v
    try:
        p = urlparse(v)
        host = (p.netloc or v).lower()
        if host.startswith("www."):
            host = host[4:]
        if ":" in host:
            host = host.split(":", 1)[0]
        return host
    except Exception:
        return v.lower()


def _ensure_env(tenant_id: Optional[int]):
    # Disable skip guards for ad-hoc testing
    os.environ.setdefault("ENRICH_SKIP_IF_ANY_HISTORY", "false")
    os.environ.setdefault("ENRICH_RECHECK_DAYS", "0")
    if tenant_id is not None:
        os.environ["DEFAULT_TENANT_ID"] = str(int(tenant_id))


def _ensure_company(domain: str, name: Optional[str]) -> int:
    from src.database import get_conn
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT company_id FROM companies WHERE LOWER(website_domain)=LOWER(%s) LIMIT 1",
            (domain,),
        )
        row = cur.fetchone()
        if row and row[0] is not None:
            return int(row[0])
        cur.execute(
            "INSERT INTO companies(name, website_domain, last_seen) VALUES (%s,%s,NOW()) RETURNING company_id",
            (name or domain, domain),
        )
        return int(cur.fetchone()[0])


async def _run(company_id: int, company_name: Optional[str]):
    from src.enrichment import enrich_company_with_tavily, set_vendor_caps
    # Be chatty for vendor/network I/O
    logging.getLogger("jina_reader").setLevel(logging.INFO)
    logging.getLogger("jina_deep_research").setLevel(logging.INFO)
    logging.getLogger("src.enrichment").setLevel(logging.INFO)
    logging.getLogger("enrichment").setLevel(logging.INFO)
    # Keep contact lookups capped (avoids surprises in dev)
    try:
        set_vendor_caps(contact_lookups=1)
    except Exception:
        pass
    state = await enrich_company_with_tavily(company_id, company_name=company_name, search_policy="require_existing")
    # Compact JSON summary
    d = state.get("data") or {}
    print(json.dumps({
        "company_id": state.get("company_id"),
        "home": state.get("home"),
        "pages": len(state.get("extracted_pages") or []),
        "emails": len(d.get("email") or []),
        "phones": len(d.get("phone_number") or []),
        "website_domain": d.get("website_domain"),
        "linkedin_url": d.get("linkedin_url"),
        "hq_city": d.get("hq_city"),
        "hq_country": d.get("hq_country"),
        "location_city": d.get("location_city"),
        "location_country": d.get("location_country"),
        "about_len": len((d.get("about_text") or "")),
        "completed": state.get("completed"),
        "error": state.get("error"),
        "degraded": state.get("degraded_reasons"),
    }, ensure_ascii=False))


def _load_payload(args) -> Dict[str, Any]:
    if args.json:
        return json.loads(args.json)
    if args.file:
        p = Path(args.file)
        return json.loads(p.read_text(encoding="utf-8"))
    # CLI args fallback
    return {
        "tenant_id": args.tenant_id,
        "domain": args.domain,
        "company_name": args.company_name,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="Inline JSON payload")
    ap.add_argument("--file", help="Path to JSON payload file")
    ap.add_argument("--tenant-id", type=int, default=None)
    ap.add_argument("--domain", help="Company website domain or URL")
    ap.add_argument("--company-name", default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    payload = _load_payload(args)
    domain = _canonical_domain(payload.get("domain") or args.domain or "")
    if not domain:
        print("Error: domain is required (in JSON or --domain)", file=sys.stderr)
        sys.exit(2)
    tenant_id = payload.get("tenant_id") or args.tenant_id
    company_name = payload.get("company_name") or args.company_name

    _ensure_env(tenant_id if tenant_id is not None else None)

    # Ensure DB connectivity is configured
    from src.settings import POSTGRES_DSN  # noqa: F401

    cid = _ensure_company(domain, company_name)
    logging.info("Company ready id=%s domain=%s name=%s", cid, domain, company_name or "")
    asyncio.run(_run(cid, company_name))


if __name__ == "__main__":
    main()
