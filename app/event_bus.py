from __future__ import annotations

import asyncio
import json
from contextvars import ContextVar
from typing import AsyncGenerator, Dict, Optional

# Simple in-proc event bus for chat progress streaming (SSE).
# - Per-session asyncio.Queue buffers stringified SSE event payloads.
# - Call set_current_session(session_id) to bind a context for deep calls.

_sessions: Dict[str, asyncio.Queue[str]] = {}
_current_session: ContextVar[Optional[str]] = ContextVar("chat_session_id", default=None)


def set_current_session(session_id: Optional[str]) -> None:
    try:
        _current_session.set(session_id or None)
    except Exception:
        pass


def _queue_for(session_id: str) -> asyncio.Queue[str]:
    q = _sessions.get(session_id)
    if q is None:
        q = asyncio.Queue(maxsize=200)
        _sessions[session_id] = q
    return q


async def emit(label: str, data: dict | str | None = None, *, session_id: Optional[str] = None) -> None:
    sid = session_id or _current_session.get() or None
    if not sid:
        return
    try:
        payload = data if isinstance(data, str) else json.dumps(data or {})
    except Exception:
        payload = json.dumps({"message": str(data) if data is not None else ""})
    # SSE framing: optional event name + data line + terminator
    msg = f"event: {label}\n" + f"data: {payload}\n\n"
    try:
        q = _queue_for(sid)
        # Drop oldest if full to avoid blocking
        if q.full():
            try:
                _ = q.get_nowait()
            except Exception:
                pass
        await q.put(msg)
    except Exception:
        pass


async def subscribe(session_id: str) -> AsyncGenerator[bytes, None]:
    """Yield SSE messages for a session until client disconnects."""
    q = _queue_for(session_id)
    # Send an initial keep-alive/comment to open the stream
    yield b":ok\n\n"
    try:
        while True:
            msg = await q.get()
            # Convert to bytes; ensure UTF-8
            try:
                yield msg.encode("utf-8")
            except Exception:
                try:
                    yield (msg or "").encode()
                except Exception:
                    yield b"\n"
    except asyncio.CancelledError:
        # Client disconnected; cleanup queue if empty
        try:
            if q.empty():
                _sessions.pop(session_id, None)
        except Exception:
            pass
        raise


async def emit_progress(message: str, *, label: str = "progress", extra: Optional[dict] = None, session_id: Optional[str] = None) -> None:
    data = {"message": message}
    if extra:
        try:
            data.update(extra)
        except Exception:
            pass
    await emit(label, data, session_id=session_id)

