"""
Simple runner to exercise the LangGraph orchestrator multiple times.

Usage:
    ORCHESTRATOR_OFFLINE=1 python scripts/run_orchestrator_loadtest.py --iterations 5 --tenant-id 1 --input "run full pipeline"
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from typing import Any, Dict, List
import json

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from scripts.run_orchestrator import _ainvoke


async def _run_once(payload: Dict[str, Any]) -> Dict[str, Any]:
    started = time.perf_counter()
    ok = True
    error = None
    status: Dict[str, Any] | None = None
    try:
        result = await _ainvoke(payload)
        status = (result or {}).get("status") or {}
    except Exception as exc:  # pragma: no cover - network failures
        ok = False
        error = str(exc)
    duration_ms = int((time.perf_counter() - started) * 1000)
    return {"ok": ok, "error": error, "status": status, "duration_ms": duration_ms}


async def main():
    parser = argparse.ArgumentParser(description="Loop the LangGraph orchestrator for load/soak testing.")
    parser.add_argument("--iterations", type=int, default=5, help="Number of sequential runs to execute.")
    parser.add_argument("--tenant-id", type=int, required=False, help="Tenant ID context passed to the graph.")
    parser.add_argument("--input", type=str, default="run full pipeline", help="User input text.")
    parser.add_argument("--icp-payload", type=str, default="{}", help="Optional JSON string for ICP payload.")
    args = parser.parse_args()

    try:
        icp_payload = json.loads(args.icp_payload or "{}")
    except json.JSONDecodeError:
        icp_payload = {}

    payload: Dict[str, Any] = {
        "tenant_id": args.tenant_id,
        "input": args.input,
        "role": "user",
        "icp_payload": icp_payload,
    }
    # Avoid json import to keep parity with CLI; simple eval is unsafe, so require JSON via CLI wrapper.
    # For quick runs, pass ORCHESTRATOR_OFFLINE=1 to skip external dependencies.

    results: List[Dict[str, Any]] = []
    for i in range(max(1, args.iterations)):
        print(f"[loadtest] Starting iteration {i + 1}/{args.iterations}")
        out = await _run_once(payload)
        status_msg = (out.get("status") or {}).get("message")
        if out["ok"]:
            print(f"[loadtest] ✓ iteration {i + 1} completed in {out['duration_ms']} ms — {status_msg}")
        else:
            print(f"[loadtest] ✗ iteration {i + 1} failed in {out['duration_ms']} ms — {out['error']}")
        results.append(out)

    durations = [r["duration_ms"] for r in results]
    successes = sum(1 for r in results if r["ok"])
    failures = len(results) - successes
    summary = {
        "runs": len(results),
        "successes": successes,
        "failures": failures,
        "min_ms": min(durations) if durations else 0,
        "max_ms": max(durations) if durations else 0,
        "avg_ms": int(statistics.fmean(durations)) if durations else 0,
    }
    print("[loadtest] summary:", summary)


if __name__ == "__main__":
    asyncio.run(main())
