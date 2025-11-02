from fastapi import APIRouter, Request, Depends, HTTPException, Path
from fastapi.responses import StreamingResponse
import asyncio
import json
import os
from typing import AsyncIterator, Optional

from app.auth import require_optional_identity
from src.chat_events import subscribe, Event

router = APIRouter(prefix="/chat", tags=["chat"])


async def _sse(iter_events: AsyncIterator[Event]):
    try:
        async for ev in iter_events:
            # Server-Sent Events format: event: <type>\n data: <json>\n\n
            payload = {"event": ev.event, "message": ev.message, "context": ev.context}
            data = json.dumps(payload, ensure_ascii=False)
            yield f"event: {ev.event}\n".encode("utf-8")
            yield f"data: {data}\n\n".encode("utf-8")
            await asyncio.sleep(0)
    except asyncio.CancelledError:
        return


async def _merge_with_heartbeat(request: Request, iter_events: AsyncIterator[Event], *, interval_s: float = 30.0):
    """Yield SSE-formatted chunks, inserting keepalive comments every interval.

    This prevents intermediaries from treating the stream as idle and closing it.
    Uses asyncio tasks to race the next event against a heartbeat timer.
    """
    try:
        # obtain async iterator protocol
        ait = iter_events.__aiter__()
        next_ev_task = asyncio.create_task(ait.__anext__())
        hb_task = asyncio.create_task(asyncio.sleep(interval_s))
        while True:
            # Break early if client disconnected
            if await request.is_disconnected():
                break
            done, pending = await asyncio.wait({next_ev_task, hb_task}, return_when=asyncio.FIRST_COMPLETED)
            if hb_task in done:
                # Send SSE comment heartbeat
                yield b": keepalive\n\n"
                # schedule next heartbeat
                hb_task = asyncio.create_task(asyncio.sleep(interval_s))
            if next_ev_task in done:
                try:
                    ev = next_ev_task.result()
                except StopAsyncIteration:
                    break
                # format SSE event chunk
                payload = {"event": ev.event, "message": ev.message, "context": ev.context}
                data = json.dumps(payload, ensure_ascii=False)
                yield f"event: {ev.event}\n".encode("utf-8")
                yield f"data: {data}\n\n".encode("utf-8")
                # schedule next event wait
                next_ev_task = asyncio.create_task(ait.__anext__())
    except asyncio.CancelledError:
        return
    finally:
        # best-effort cleanup
        try:
            next_ev_task.cancel()  # type: ignore[name-defined]
        except Exception:
            pass
        try:
            hb_task.cancel()  # type: ignore[name-defined]
        except Exception:
            pass


@router.get("/stream/{session_id}")
async def stream_session(
    request: Request,
    session_id: str = Path(..., min_length=1),
    user=Depends(require_optional_identity),
):
    """SSE stream of chat events for a session.

    - Auth: accepts optional identity (dev bypass) but prefers JWT if present.
    - Tenant scoping: if a tenant_id is present on the request state, scope subscription to it.
    - Backpressure: bounded; drops oldest events when slow clients fall behind.
    """
    tenant_id: Optional[str] = getattr(request.state, "tenant_id", None)
    iter_events = subscribe(session_id, tenant_id)
    # Heartbeat interval configurable via env; default 30s
    try:
        hb = float(os.getenv("SSE_HEARTBEAT_INTERVAL_S", "30") or 30.0)
    except Exception:
        hb = 30.0
    # Stream with heartbeat to keep intermediaries from timing out idle periods
    async def _gen():
        async for chunk in _merge_with_heartbeat(request, iter_events, interval_s=hb):
            yield chunk
    return StreamingResponse(_gen(), media_type="text/event-stream")
