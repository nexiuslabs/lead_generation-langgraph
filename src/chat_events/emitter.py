import asyncio
import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional

logger = logging.getLogger("chat_events")


@dataclass
class Event:
    event: str
    message: str
    context: Dict[str, Any]


_subscribers: Dict[str, List[asyncio.Queue]] = {}


def _get_key(session_id: Optional[str], tenant_id: Optional[str | int]) -> str:
    sid = (session_id or "").strip() or "global"
    tid = str(tenant_id) if tenant_id is not None else ""
    return f"{sid}:{tid}" if tid else sid


async def subscribe(session_id: Optional[str], tenant_id: Optional[str | int]) -> AsyncIterator[Event]:
    """Subscribe to events for a session (and optional tenant scope).

    Yields Event objects as they arrive. Caller must iterate until generator completes
    (e.g., client disconnect). Backpressure is bounded; when full, the newest event replaces the oldest.
    """
    key = _get_key(session_id, tenant_id)
    q: asyncio.Queue = asyncio.Queue(maxsize=100)
    _subscribers.setdefault(key, []).append(q)
    logger.info("[chat_events] subscriber added key=%s; total=%d", key, len(_subscribers.get(key, [])))
    try:
        while True:
            ev = await q.get()
            if isinstance(ev, Event):
                yield ev
    except asyncio.CancelledError:
        # Normal disconnect
        pass
    finally:
        try:
            _subscribers.get(key, []).remove(q)
        except Exception:
            pass
        logger.info("[chat_events] subscriber removed key=%s; total=%d", key, len(_subscribers.get(key, [])))


def unsubscribe(session_id: Optional[str], tenant_id: Optional[str | int]) -> None:
    try:
        key = _get_key(session_id, tenant_id)
        _subscribers.pop(key, None)
    except Exception:
        pass


def emit(session_id: Optional[str], tenant_id: Optional[str | int], event: str, message: str, context: Optional[Dict[str, Any]] = None) -> None:
    """Emit a compact event to all subscribers for the session/tenant.

    - Non-blocking; drops when no listeners are present.
    - Payload must be JSON-serializable by the caller.
    """
    key = _get_key(session_id, tenant_id)
    queues = _subscribers.get(key)
    if not queues:
        # No listeners; skip
        return
    ctx = context or {}
    try:
        ctx.setdefault("session_id", session_id)
        if tenant_id is not None:
            ctx.setdefault("tenant_id", tenant_id)
    except Exception:
        pass
    ev = Event(event=event, message=message, context=ctx)
    for q in list(queues or []):
        try:
            if q.full():
                # Drop oldest to keep recent progress visible
                _ = q.get_nowait()
            q.put_nowait(ev)
        except Exception:
            # Remove broken subscribers
            try:
                queues.remove(q)
            except Exception:
                pass

