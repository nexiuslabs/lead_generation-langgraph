from fastapi import APIRouter, Request, Depends, HTTPException, Path
from fastapi.responses import StreamingResponse
import asyncio
import json
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
    # Fast disconnect check
    async def _gen():
        async for chunk in _sse(iter_events):
            # Stop if client disconnected
            if await request.is_disconnected():
                break
            yield chunk
    return StreamingResponse(_gen(), media_type="text/event-stream")

