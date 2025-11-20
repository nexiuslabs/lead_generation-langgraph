"""
CLI wrapper to invoke the LangGraph orchestrator once.

Usage:
    python scripts/run_orchestrator.py --tenant-id 1 --input "run enrichment"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from typing import Any, Dict

# Ensure repo root on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from app.langgraph_logging import LangGraphTroubleshootHandler
from my_agent.agent import build_orchestrator_graph
from my_agent.utils.state import OrchestrationState


async def _ainvoke(payload: Dict[str, Any]) -> Dict[str, Any]:
    graph = build_orchestrator_graph()
    thread_id = payload.get("thread_id") or str(uuid.uuid4())
    state: OrchestrationState = {
        "messages": payload.get("messages") or [],
        "input": payload.get("input") or "",
        "input_role": payload.get("role") or "api",
        "entry_context": {
            "thread_id": thread_id,
            "tenant_id": payload.get("tenant_id"),
            "user_id": payload.get("user_id"),
            "source": "cli",
        },
        "icp_payload": payload.get("icp_payload") or {},
    }
    result = await graph.ainvoke(
        state,
        config={
            "configurable": {"thread_id": thread_id},
            "callbacks": [LangGraphTroubleshootHandler(context={"thread_id": thread_id, "source": "cli"})],
        },
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Run LangGraph orchestrator once.")
    parser.add_argument("--tenant-id", type=int, help="Tenant ID context", required=False)
    parser.add_argument("--input", type=str, help="User input text", default="run enrichment")
    parser.add_argument("--icp-payload", type=str, help="JSON string with ICP payload", default="{}")
    args = parser.parse_args()

    try:
        icp_payload = json.loads(args.icp_payload or "{}")
    except json.JSONDecodeError:
        icp_payload = {}

    payload = {
        "tenant_id": args.tenant_id,
        "input": args.input,
        "icp_payload": icp_payload,
        "role": "user",
    }
    result = asyncio.run(_ainvoke(payload))
    status = (result or {}).get("status") or {}
    print(json.dumps({"status": status, "output": status.get("message")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
