#!/usr/bin/env python3
import asyncio
import os
import sys
import argparse
import json
from typing import Any, Dict, Tuple

import asyncpg
from dotenv import load_dotenv


DEFAULTS = {
    "min_domain_rate": 0.70,
    "min_about_rate": 0.60,
    "min_email_rate": 0.40,
    "max_bucket_dominance": 0.70,
}


async def table_has_column(conn: asyncpg.Connection, table: str, column: str) -> bool:
    try:
        val = await conn.fetchval(
            """
            SELECT 1
            FROM information_schema.columns
            WHERE table_name = $1 AND column_name = $2
            LIMIT 1
            """,
            table,
            column,
        )
        return bool(val)
    except Exception:
        return False


async def compute_metrics(conn: asyncpg.Connection) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "mv_candidates": None,
        "shortlisted": None,
        "domain_rate": None,
        "about_rate": None,
        "email_rate": None,
        "bucket_counts": {},
        "rationale_rate": None,
    }

    # Candidate MV count (if view exists)
    try:
        mv = await conn.fetchval("SELECT COUNT(*) FROM icp_candidate_companies")
        metrics["mv_candidates"] = int(mv or 0)
    except Exception:
        metrics["mv_candidates"] = None

    # Shortlisted (lead_scores)
    try:
        sl = await conn.fetchval("SELECT COUNT(*) FROM lead_scores")
        metrics["shortlisted"] = int(sl or 0)
    except Exception:
        metrics["shortlisted"] = 0

    # Domain rate among shortlisted companies
    try:
        row = await conn.fetchrow(
            """
            SELECT
              SUM(CASE WHEN NULLIF(TRIM(c.website_domain), '') IS NOT NULL THEN 1 ELSE 0 END) AS with_domain,
              COUNT(*) AS total
            FROM lead_scores s
            JOIN companies c ON c.company_id = s.company_id
            """
        )
        with_domain = int(row["with_domain"] or 0)
        total = int(row["total"] or 0)
        metrics["domain_rate"] = (with_domain / total) if total else 0.0
    except Exception:
        metrics["domain_rate"] = 0.0

    # About rate from company_enrichment_runs joined to lead_scores
    try:
        has_about_col = await table_has_column(conn, "company_enrichment_runs", "about_text")
        if has_about_col:
            row = await conn.fetchrow(
                """
                SELECT
                  SUM(CASE WHEN COALESCE(NULLIF(TRIM(r.about_text), ''), NULL) IS NOT NULL THEN 1 ELSE 0 END) AS with_about,
                  COUNT(*) AS total
                FROM lead_scores s
                JOIN company_enrichment_runs r ON r.company_id = s.company_id
                """
            )
            with_about = int(row["with_about"] or 0)
            total = int(row["total"] or 0)
            metrics["about_rate"] = (with_about / total) if total else 0.0
        else:
            metrics["about_rate"] = 0.0
    except Exception:
        metrics["about_rate"] = 0.0

    # Email availability among shortlisted companies
    try:
        # prefer verification_status if present
        has_ver_col = await table_has_column(conn, "lead_emails", "verification_status")
        if has_ver_col:
            row = await conn.fetchrow(
                """
                SELECT
                  SUM(CASE WHEN EXISTS (
                    SELECT 1 FROM lead_emails e
                    WHERE e.company_id = s.company_id
                      AND COALESCE(e.verification_status,'unknown') IN ('valid','unknown')
                  ) THEN 1 ELSE 0 END) AS companies_with_email,
                  COUNT(*) AS total
                FROM lead_scores s
                """
            )
        else:
            row = await conn.fetchrow(
                """
                SELECT
                  SUM(CASE WHEN EXISTS (
                    SELECT 1 FROM lead_emails e WHERE e.company_id = s.company_id
                  ) THEN 1 ELSE 0 END) AS companies_with_email,
                  COUNT(*) AS total
                FROM lead_scores s
                """
            )
        ok = int(row["companies_with_email"] or 0)
        total = int(row["total"] or 0)
        metrics["email_rate"] = (ok / total) if total else 0.0
    except Exception:
        metrics["email_rate"] = 0.0

    # Bucket distribution and dominance
    try:
        rows = await conn.fetch("SELECT bucket, COUNT(*) AS c FROM lead_scores GROUP BY bucket")
        counts = {str(r["bucket"]): int(r["c"]) for r in rows}
        metrics["bucket_counts"] = counts
    except Exception:
        metrics["bucket_counts"] = {}

    # Rationale presence
    try:
        row = await conn.fetchrow(
            """
            SELECT
              SUM(CASE WHEN NULLIF(TRIM(rationale), '') IS NOT NULL THEN 1 ELSE 0 END) AS with_rationale,
              COUNT(*) AS total
            FROM lead_scores
            """
        )
        ok = int(row["with_rationale"] or 0)
        total = int(row["total"] or 0)
        metrics["rationale_rate"] = (ok / total) if total else 0.0
    except Exception:
        metrics["rationale_rate"] = 0.0

    return metrics


def evaluate(metrics: Dict[str, Any], thresholds: Dict[str, float]) -> Tuple[bool, Dict[str, Any]]:
    # Compute dominance from bucket counts
    counts = metrics.get("bucket_counts") or {}
    total = sum(counts.values()) or 0
    dominance = max((c / total for c in counts.values()), default=0.0)
    results = {
        "domain_rate_ok": (metrics.get("domain_rate") or 0.0) >= thresholds["min_domain_rate"],
        "about_rate_ok": (metrics.get("about_rate") or 0.0) >= thresholds["min_about_rate"],
        "email_rate_ok": (metrics.get("email_rate") or 0.0) >= thresholds["min_email_rate"],
        "bucket_dominance_ok": dominance <= thresholds["max_bucket_dominance"],
    }
    passed = all(results.values())
    results["bucket_dominance"] = dominance
    return passed, results


async def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Acceptance check for Feature 14")
    parser.add_argument("--tenant", type=int, default=None, help="Tenant ID for RLS scoping (sets request.tenant_id)")
    parser.add_argument("--dsn", type=str, default=os.getenv("POSTGRES_DSN") or os.getenv("DATABASE_URL"), help="Postgres DSN")
    parser.add_argument("--min-domain-rate", type=float, default=float(os.getenv("MIN_DOMAIN_RATE", DEFAULTS["min_domain_rate"])))
    parser.add_argument("--min-about-rate", type=float, default=float(os.getenv("MIN_ABOUT_RATE", DEFAULTS["min_about_rate"])))
    parser.add_argument("--min-email-rate", type=float, default=float(os.getenv("MIN_EMAIL_RATE", DEFAULTS["min_email_rate"])))
    parser.add_argument("--max-bucket-dominance", type=float, default=float(os.getenv("MAX_BUCKET_DOMINANCE", DEFAULTS["max_bucket_dominance"])))
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    args = parser.parse_args()

    if not args.dsn:
        print("ERROR: POSTGRES_DSN or DATABASE_URL is required", file=sys.stderr)
        return 2

    thresholds = {
        "min_domain_rate": args.min_domain_rate,
        "min_about_rate": args.min_about_rate,
        "min_email_rate": args.min_email_rate,
        "max_bucket_dominance": args.max_bucket_dominance,
    }

    conn = await asyncpg.connect(dsn=args.dsn)
    try:
        if args.tenant is None:
            t_env = os.getenv("DEFAULT_TENANT_ID")
            if t_env and t_env.isdigit():
                args.tenant = int(t_env)
        if args.tenant is not None:
            try:
                await conn.execute("SELECT set_config('request.tenant_id', $1, true)", str(args.tenant))
            except Exception:
                pass

        metrics = await compute_metrics(conn)
        passed, results = evaluate(metrics, thresholds)

        out = {
            "tenant_id": args.tenant,
            "metrics": metrics,
            "thresholds": thresholds,
            "results": results,
            "passed": passed,
        }
        if args.json:
            print(json.dumps(out, indent=2, default=str))
        else:
            print("Acceptance Check (Feature 14)")
            print(f"  tenant_id: {out['tenant_id']}")
            print(f"  mv_candidates: {metrics['mv_candidates']}")
            print(f"  shortlisted: {metrics['shortlisted']}")
            print(f"  domain_rate: {metrics['domain_rate']:.2%} (>= {thresholds['min_domain_rate']:.0%}) -> { 'OK' if results['domain_rate_ok'] else 'FAIL' }")
            print(f"  about_rate: {metrics['about_rate']:.2%} (>= {thresholds['min_about_rate']:.0%}) -> { 'OK' if results['about_rate_ok'] else 'FAIL' }")
            print(f"  email_rate: {metrics['email_rate']:.2%} (>= {thresholds['min_email_rate']:.0%}) -> { 'OK' if results['email_rate_ok'] else 'FAIL' }")
            print(f"  bucket_counts: {metrics['bucket_counts']}")
            print(f"  bucket_dominance: {results['bucket_dominance']:.2%} (<= {thresholds['max_bucket_dominance']:.0%}) -> { 'OK' if results['bucket_dominance_ok'] else 'FAIL' }")
            print(f"  rationale_rate: {metrics['rationale_rate']:.2%}")
            print(f"  PASSED: {passed}")

        return 0 if passed else 1
    finally:
        try:
            await conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

