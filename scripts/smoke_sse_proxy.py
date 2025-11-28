#!/usr/bin/env python3
"""
Smoke test for streaming through the FastAPI /graph proxy.

Prereqs:
  - LangGraph dev server running at http://127.0.0.1:8001
  - FastAPI running at http://127.0.0.1:8000

Usage:
  python scripts/smoke_sse_proxy.py --tenant 1105 --graph orchestrator --text "hello"

This creates a thread via FastAPI /graph, then streams a run via
POST /graph/threads/{id}/runs/stream and prints the first few SSE events.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys

import httpx


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000/graph")
    parser.add_argument("--tenant", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--text", default="hello")
    args = parser.parse_args()

    headers = {"X-Tenant-ID": args.tenant}
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=30.0)) as client:
        # Create a thread
        r = await client.post(
            f"{args.base.rstrip('/')}/threads",
            headers={**headers, "Content-Type": "application/json"},
            json={"metadata": {"tenant_id": args.tenant}, "graphId": args.graph},
        )
        r.raise_for_status()
        data = r.json()
        thread_id = data.get("thread_id") or data.get("threadId") or data.get("id")
        if not thread_id:
            print("Could not parse thread id from:", data)
            sys.exit(2)
        print("Thread:", thread_id)

        # Stream a run
        run_url = f"{args.base.rstrip('/')}/threads/{thread_id}/runs/stream"
        payload = {
            "messages": {
                "type": "human",
                "content": [{"type": "text", "text": args.text}],
            }
        }
        async with client.stream(
            "POST",
            run_url,
            headers={
                **headers,
                "Accept": "text/event-stream",
                "Content-Type": "application/json",
            },
            json=payload,
        ) as resp:
            print("Status:", resp.status_code, resp.headers.get("content-type"))
            resp.raise_for_status()
            n = 0
            async for chunk in resp.aiter_lines():
                if chunk:
                    print(chunk)
                n += 1
                if n > 40:
                    break


if __name__ == "__main__":
    asyncio.run(main())

