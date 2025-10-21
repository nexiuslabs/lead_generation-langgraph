"""
Quick verifier for the ICP Profile → discovery SSE flow.

Runs two graph invocations with a fixed `session_id` and collects SSE events
from the in-memory event bus to prove the sequence:
  1) icp:profile_ready
  2) (run ends; UI would refresh history)
  3) discovery:started
  4) discovery:done

Notes
- The script stubs `src.agents_icp.plan_top10_with_reasons` to avoid network.
- It also pre-populates `icp_profile` to skip LLM/web calls in confirm_node.

Usage
  PYTHONPATH=lead_generation-main ENABLE_AGENT_DISCOVERY=1 \
  python lead_generation-main/scripts/verify_icp_profile_sse.py
"""
from __future__ import annotations

import asyncio
import os
from typing import List

os.environ.setdefault("ENABLE_AGENT_DISCOVERY", "1")
os.environ.setdefault("ENABLE_ICP_INTAKE", "0")

from langchain_core.messages import HumanMessage
from app.pre_sdr_graph import build_graph
from app.event_bus import subscribe


async def collect_events(session_id: str, max_events: int = 6, timeout: float = 10.0) -> List[str]:
    labels: List[str] = []

    async def _runner():
        async for chunk in subscribe(session_id):
            try:
                s = chunk.decode("utf-8", errors="ignore")
            except Exception:
                s = str(chunk)
            ev = None
            for line in s.splitlines():
                if line.startswith("event: "):
                    ev = line[len("event: ") :].strip()
            if ev:
                labels.append(ev)
            if len(labels) >= max_events:
                break

    try:
        await asyncio.wait_for(_runner(), timeout=timeout)
    except asyncio.TimeoutError:
        pass
    return labels


async def main() -> None:
    g = build_graph()
    sid = "test-session-icp-001"

    # Minimal state to avoid LLM/web synthesis (buyer_titles/integrations present)
    state = {
        "messages": [HumanMessage(content="confirm")],
        "icp": {"industries": ["food service", "distribution"]},
        "icp_profile": {
            "industries": ["food service", "distribution"],
            "buyer_titles": ["Purchasing Manager"],
            "integrations": ["shopify"],
            "size_bands": ["11-50"],
            "triggers": ["ecommerce replatform"],
        },
        "session_id": sid,
    }

    # Start event collector
    events_task = asyncio.create_task(collect_events(sid, max_events=6, timeout=10.0))

    # Run 1: confirm -> profile emitted -> router defers to next run
    out1 = await g.ainvoke(state)

    # Stub discovery to avoid network calls; return empty candidates quickly
    import src.agents_icp as agents  # type: ignore

    def _stub_top10(icp_profile, tenant_id=None):
        return []

    agents.plan_top10_with_reasons = _stub_top10  # type: ignore[attr-defined]

    # Run 2: autostart discovery (started/done events should appear)
    out2 = await g.ainvoke(out1)

    labels = await events_task
    print("SSE events:", labels)
    print(
        "Flags:",
        {
            "profile_shown_after_run1": out1.get("profile_shown"),
            "autostart_after_run1": out1.get("autostart_discovery"),
            "discovery_done_after_run2": out2.get("agent_discovery_done"),
        },
    )


if __name__ == "__main__":
    asyncio.run(main())

