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

# Default SLA for Top-10 immediate enrichment (ms)
DEFAULT_MAX_TOP10_MS = 300_000


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


async def compute_top10_enrich_metrics(conn: asyncpg.Connection) -> Dict[str, Any]:
    """Return metrics for the latest strict Top‑10 enrichment run.

    Uses enrichment_runs + run_summaries + run_manifests + run_event_logs.
    """
    out: Dict[str, Any] = {"run_id": None, "duration_ms": None, "stage_p95_ms": {}, "bucket_counts_top10": {}}
    try:
        row = await conn.fetchrow(
            """
            SELECT er.run_id, er.started_at, er.ended_at, rs.candidates, rs.processed
            FROM enrichment_runs er
            JOIN run_summaries rs USING(run_id)
            WHERE er.ended_at IS NOT NULL
              AND rs.candidates = 10
            ORDER BY er.ended_at DESC
            LIMIT 1
            """
        )
    except Exception:
        row = None
    if not row:
        return out
    out["run_id"] = int(row["run_id"]) if row["run_id"] else None
    try:
        dur_ms = int((row["ended_at"] - row["started_at"]).total_seconds() * 1000)
    except Exception:
        dur_ms = None
    out["duration_ms"] = dur_ms
    # p95 per stage for this run
    try:
        rows = await conn.fetch(
            """
            SELECT stage, percentile_cont(0.95) WITHIN GROUP (ORDER BY duration_ms) AS p95
            FROM run_event_logs
            WHERE run_id = $1 AND duration_ms IS NOT NULL
            GROUP BY stage
            """,
            out["run_id"],
        )
        out["stage_p95_ms"] = {str(r["stage"]): int(r["p95"] or 0) for r in rows}
    except Exception:
        out["stage_p95_ms"] = {}
    # A/B/C counts among the Top‑10 manifest
    try:
        ids = await conn.fetchval("SELECT selected_ids FROM run_manifests WHERE run_id=$1", out["run_id"])
        if ids:
            rows = await conn.fetch(
                """
                SELECT s.bucket, COUNT(*) AS c
                FROM lead_scores s
                WHERE s.company_id = ANY($1::bigint[])
                GROUP BY s.bucket
                """,
                ids,
            )
            out["bucket_counts_top10"] = {str(r["bucket"]): int(r["c"]) for r in rows}
    except Exception:
        out["bucket_counts_top10"] = {}
    return out


def evaluate_top10_sla(m: Dict[str, Any], max_ms: int = DEFAULT_MAX_TOP10_MS) -> Tuple[bool, int]:
    dur = int(m.get("duration_ms") or 0)
    return (dur <= max_ms and dur > 0), dur


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

    parser = argparse.ArgumentParser(description="Acceptance check for Feature 14 + PRD19 Top-10 SLA")
    parser.add_argument("--tenant", type=int, default=None, help="Tenant ID for RLS scoping (sets request.tenant_id)")
    parser.add_argument("--dsn", type=str, default=os.getenv("POSTGRES_DSN") or os.getenv("DATABASE_URL"), help="Postgres DSN")
    parser.add_argument("--min-domain-rate", type=float, default=float(os.getenv("MIN_DOMAIN_RATE", DEFAULTS["min_domain_rate"])))
    parser.add_argument("--min-about-rate", type=float, default=float(os.getenv("MIN_ABOUT_RATE", DEFAULTS["min_about_rate"])))
    parser.add_argument("--min-email-rate", type=float, default=float(os.getenv("MIN_EMAIL_RATE", DEFAULTS["min_email_rate"])))
    parser.add_argument("--max-bucket-dominance", type=float, default=float(os.getenv("MAX_BUCKET_DOMINANCE", DEFAULTS["max_bucket_dominance"])))
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    parser.add_argument("--max-top10-ms", type=int, default=int(os.getenv("MAX_TOP10_MS", str(DEFAULT_MAX_TOP10_MS)) or DEFAULT_MAX_TOP10_MS), help="SLA for Top-10 enrich (ms)")
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
        top10 = await compute_top10_enrich_metrics(conn)
        passed, results = evaluate(metrics, thresholds)
        sla_ok, top10_ms = evaluate_top10_sla(top10, args.max_top10_ms)
        results["top10_sla_ok"] = sla_ok

        out = {
            "tenant_id": args.tenant,
            "metrics": metrics,
            "thresholds": thresholds,
            "top10": top10,
            "max_top10_ms": args.max_top10_ms,
            "results": results,
            "passed": passed and sla_ok,
        }
        if args.json:
            print(json.dumps(out, indent=2, default=str))
        else:
            print("Acceptance Check (Feature 14 + PRD19)")
            print(f"  tenant_id: {out['tenant_id']}")
            print(f"  mv_candidates: {metrics['mv_candidates']}")
            print(f"  shortlisted: {metrics['shortlisted']}")
            print(f"  domain_rate: {metrics['domain_rate']:.2%} (>= {thresholds['min_domain_rate']:.0%}) -> { 'OK' if results['domain_rate_ok'] else 'FAIL' }")
            print(f"  about_rate: {metrics['about_rate']:.2%} (>= {thresholds['min_about_rate']:.0%}) -> { 'OK' if results['about_rate_ok'] else 'FAIL' }")
            print(f"  email_rate: {metrics['email_rate']:.2%} (>= {thresholds['min_email_rate']:.0%}) -> { 'OK' if results['email_rate_ok'] else 'FAIL' }")
            print(f"  bucket_counts: {metrics['bucket_counts']}")
            print(f"  bucket_dominance: {results['bucket_dominance']:.2%} (<= {thresholds['max_bucket_dominance']:.0%}) -> { 'OK' if results['bucket_dominance_ok'] else 'FAIL' }")
            print(f"  rationale_rate: {metrics['rationale_rate']:.2%}")
            print(f"  top10_sla: {top10_ms} ms (<= {args.max_top10_ms} ms) -> { 'OK' if sla_ok else 'FAIL' }")
            print(f"  top10_stage_p95_ms: {top10.get('stage_p95_ms')}")
            print(f"  top10_bucket_counts: {top10.get('bucket_counts_top10')}")
            print(f"  PASSED: {passed and sla_ok}")

        return 0 if passed else 1
    finally:
        try:
            await conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
