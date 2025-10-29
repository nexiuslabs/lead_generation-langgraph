import os
import sys
import argparse
from typing import List

# Ensure project root on path so imports work when invoked as module
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _get_domains_for_next40(tenant_id: int, limit: int) -> List[str]:
    from src.database import get_conn
    doms: List[str] = []
    with get_conn() as conn, conn.cursor() as cur:
        # Try batch-bound remainder first
        cur.execute(
            """
            SELECT ai_metadata->>'batch_id'
              FROM staging_global_companies
             WHERE tenant_id=%s AND COALESCE((ai_metadata->>'preview')::boolean,false)=true
             ORDER BY created_at DESC
             LIMIT 1
            """,
            (tenant_id,),
        )
        row = cur.fetchone()
        batch_id = (row and row[0]) or None
        if batch_id:
            cur.execute(
                """
                SELECT domain
                  FROM staging_global_companies
                 WHERE tenant_id=%s
                   AND source='web_discovery'
                   AND COALESCE((ai_metadata->>'preview')::boolean,false)=false
                   AND ai_metadata->>'batch_id'=%s
                 ORDER BY created_at DESC
                 LIMIT %s
                """,
                (tenant_id, batch_id, limit),
            )
            rows = cur.fetchall() or []
            doms = [str(r[0]).strip().lower() for r in rows if r and r[0]]
        if not doms:
            # Fallback: non-preview latest, excluding current preview top 10
            cur.execute(
                """
                WITH preview AS (
                  SELECT LOWER(domain) AS d
                  FROM staging_global_companies
                  WHERE tenant_id=%s AND COALESCE((ai_metadata->>'preview')::boolean,false)=true
                  ORDER BY created_at DESC
                  LIMIT 10
                )
                SELECT domain
                  FROM staging_global_companies
                 WHERE tenant_id=%s
                   AND source='web_discovery'
                   AND COALESCE((ai_metadata->>'preview')::boolean,false)=false
                   AND LOWER(domain) NOT IN (SELECT d FROM preview)
                 ORDER BY created_at DESC
                 LIMIT %s
                """,
                (tenant_id, tenant_id, limit),
            )
            rows = cur.fetchall() or []
            doms = [str(r[0]).strip().lower() for r in rows if r and r[0]]
    return [d for d in doms if d]


def main() -> int:
    ap = argparse.ArgumentParser(description="Enqueue Next-40 web discovery enrichment for latest preview batch")
    ap.add_argument("--tenant-id", type=int, required=True, help="Tenant ID")
    ap.add_argument("--limit", type=int, default=int(os.getenv("BG_NEXT_COUNT", "40") or 40), help="Max domains to enqueue (default from BG_NEXT_COUNT or 40)")
    ap.add_argument("--notify-email", type=str, default=None, help="Optional recipient email to notify on completion")
    args = ap.parse_args()

    tenant_id: int = int(args.tenant_id)
    limit: int = max(1, int(args.limit))

    doms = _get_domains_for_next40(tenant_id, limit)
    if not doms:
        print("[next40-debug] no candidate domains found in staging for tenant", tenant_id)
        return 1

    # Map to company_ids, ensuring rows exist for missing domains
    from src.database import get_conn
    ids: List[int] = []
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT company_id, website_domain FROM companies WHERE LOWER(website_domain) = ANY(%s)",
            ([d.lower() for d in doms],),
        )
        found = {str((r[1] or "").lower()): int(r[0]) for r in (cur.fetchall() or []) if r and r[0] is not None}
        for d in doms:
            cid = found.get(d.lower())
            if cid is not None:
                ids.append(int(cid))
            else:
                cur.execute(
                    "INSERT INTO companies(name, website_domain, last_seen) VALUES (%s,%s,NOW()) RETURNING company_id",
                    (d, d),
                )
                ids.append(int(cur.fetchone()[0]))

    if not ids:
        print("[next40-debug] no company ids resolved; aborting")
        return 2

    # Enqueue the job (this will NOTIFY bg_jobs)
    from src.jobs import enqueue_web_discovery_bg_enrich
    to_email = (args.notify_email or "").strip() if isinstance(args.notify_email, str) else None
    if not to_email:
        # Fallback to env default when provided (dev convenience)
        from src.settings import DEFAULT_NOTIFY_EMAIL as _DEF_TO
        if _DEF_TO and ("@" in str(_DEF_TO)):
            to_email = str(_DEF_TO)
    job = enqueue_web_discovery_bg_enrich(tenant_id, ids, notify_email=to_email)
    print(f"[next40-debug] enqueued job_id={job.get('job_id')} count={len(ids)} tenant_id={tenant_id} notify_email={to_email}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
