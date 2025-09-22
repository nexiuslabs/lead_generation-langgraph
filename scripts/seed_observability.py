#!/usr/bin/env python3
"""
Seed synthetic observability events for a run and compute percentiles.

Usage:
  python -m scripts.seed_observability --tenant 1034

Environment:
  - POSTGRES_DSN must be set (.env is loaded by src.settings)
  - DEFAULT_TENANT_ID can be used if --tenant not provided
"""
import argparse
import os
import time
from typing import Dict, List

from src import obs
from src.database import get_conn


STAGE_DURATIONS: Dict[str, List[int]] = {
    "mv_refresh": [120, 140, 160],
    "select_targets": [20, 30, 50, 40],
    "enrich": [400, 650, 800, 500, 720, 600],
    "score": [100, 150, 210, 180],
    "verify_emails": [50, 75, 90],
    "export_odoo": [200, 240, 260],
}


def seed(run_id: int, tenant_id: int) -> None:
    # Ensure rows exist in run_stage_stats for each stage, and add a real duration via stage_timer
    for stage, samples in STAGE_DURATIONS.items():
        # Create a base row and one finish event duration using the context manager
        with obs.stage_timer(run_id, tenant_id, stage, total_inc=0):
            time.sleep(0.01)
        # Add additional synthetic finish events with specific durations
        for ms in samples:
            obs.log_event(
                run_id,
                tenant_id,
                stage,
                event="finish",
                status="ok",
                duration_ms=int(ms),
            )


def fetch_stats(run_id: int, tenant_id: int):
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT stage, count_total, count_success, count_error, COALESCE(p50_ms,0), COALESCE(p95_ms,0), COALESCE(p99_ms,0)
            FROM run_stage_stats
            WHERE run_id=%s AND tenant_id=%s
            ORDER BY stage
            """,
            (run_id, tenant_id),
        )
        return cur.fetchall()


def main() -> int:
    ap = argparse.ArgumentParser(description="Seed observability data and compute percentiles")
    ap.add_argument("--tenant", type=int, default=None, help="Tenant ID (falls back to DEFAULT_TENANT_ID)")
    args = ap.parse_args()

    tid = args.tenant
    if tid is None:
        env_tid = os.getenv("DEFAULT_TENANT_ID")
        if env_tid and env_tid.isdigit():
            tid = int(env_tid)
    if tid is None:
        print("ERROR: tenant id not provided and DEFAULT_TENANT_ID not set")
        return 2

    # Begin a synthetic run
    run_id = obs.begin_run(tid)
    print(f"Started synthetic run_id={run_id} tenant_id={tid}")
    try:
        obs.set_run_context(run_id, tid)
    except Exception:
        pass

    # Seed events per stage
    seed(run_id, tid)

    # Compute percentiles for this run
    obs.aggregate_percentiles(run_id, tid)
    obs.finalize_run(run_id, status="succeeded")

    # Show a quick summary
    rows = fetch_stats(run_id, tid)
    print("\nStage percentiles (p50/p95/p99 ms) and counters:")
    for stage, total, ok, err, p50, p95, p99 in rows:
        print(f"  - {stage}: p50={p50} p95={p95} p99={p99} | total={total} ok={ok} err={err}")
    print(f"\nRun complete. View via GET /runs/{run_id}/stats (requires auth)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

