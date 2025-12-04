from __future__ import annotations

from fastapi import APIRouter, Body, Depends, HTTPException, Request
from typing import Any, Dict, Optional

from app.auth import require_auth
from .threads_db import (
    context_key_from_payload,
    create_thread,
    lock_prior_open,
    auto_archive_stale_locked,
    resume_eligible,
    get_thread,
    list_threads,
)
from src.settings import (
    THREAD_RESUME_WINDOW_DAYS,
    THREAD_STALE_DAYS,
    AUTO_ARCHIVE_STALE_LOCKED,
)


router = APIRouter()


def _label_for_context(context_key: str, fallback: Optional[str] = None) -> str:
    if isinstance(context_key, str) and context_key.startswith("domain:"):
        return context_key.split(":", 1)[1]
    return fallback or "ICP session"


@router.post("/threads")
async def create_thread_route(
    request: Request,
    payload: Dict[str, Any] = Body(default_factory=dict),
    identity: Dict[str, Any] = Depends(require_auth),
):
    tenant_id = getattr(request.state, "tenant_id", None)
    user_id = identity.get("sub")
    context_key = context_key_from_payload(payload, tenant_id)
    label = _label_for_context(context_key, str(payload.get("label") or "").strip() or None)

    # Create new open thread and lock prior open threads for the same context
    tid = create_thread(tenant_id, user_id, "icp_finder", context_key, label)
    lock_prior_open(tenant_id, user_id, "icp_finder", context_key, tid)
    if AUTO_ARCHIVE_STALE_LOCKED:
        try:
            auto_archive_stale_locked(tenant_id, THREAD_STALE_DAYS)
        except Exception:
            pass

    return {"id": tid, "status": "open", "context_key": context_key, "label": label}


@router.post("/threads/{thread_id}/resume")
async def resume_thread_route(
    thread_id: str,
    request: Request,
    _: Dict[str, Any] = Depends(require_auth),
):
    tenant_id = getattr(request.state, "tenant_id", None)
    row = get_thread(thread_id, tenant_id)
    if not row:
        raise HTTPException(status_code=404, detail="thread_not_found")
    if (row.get("status") or "open") != "open":
        raise HTTPException(status_code=409, detail={"error": "thread_locked", "hint": "create_new"})
    return {"ok": True, "thread": row}


@router.post("/threads/resume-eligible")
async def resume_eligible_route(
    request: Request,
    payload: Dict[str, Any] = Body(default_factory=dict),
    identity: Dict[str, Any] = Depends(require_auth),
):
    tenant_id = getattr(request.state, "tenant_id", None)
    user_id = identity.get("sub")
    context_key = context_key_from_payload(payload, tenant_id)
    cands = resume_eligible(tenant_id, user_id, "icp_finder", context_key, THREAD_RESUME_WINDOW_DAYS)
    if not cands:
        # Create new when none
        tid = create_thread(tenant_id, user_id, "icp_finder", context_key, _label_for_context(context_key))
        lock_prior_open(tenant_id, user_id, "icp_finder", context_key, tid)
        return {"auto_created": True, "thread": get_thread(tid, tenant_id)}
    if len(cands) == 1:
        return {"auto_resumed": True, "thread": get_thread(cands[0]["id"], tenant_id)}
    # Multiple: return 2–3 options with labels
    out = []
    for c in cands:
        row = get_thread(c["id"], tenant_id)
        if not row:
            continue
        out.append({
            "id": row["id"],
            "label": row.get("label") or _label_for_context(row.get("context_key") or ""),
            "updated_at": row.get("last_updated_at") or row.get("created_at"),
        })
    return {"auto_resumed": False, "candidates": out[:3], "context_key": context_key}


@router.get("/threads")
async def list_threads_route(
    request: Request,
    show_archived: bool = False,
    identity: Dict[str, Any] = Depends(require_auth),
):
    tenant_id = getattr(request.state, "tenant_id", None)
    user_id = identity.get("sub")
    rows = list_threads(tenant_id, user_id, show_archived=show_archived)
    return {"items": rows}


@router.get("/threads/{thread_id}")
async def get_thread_route(
    thread_id: str,
    request: Request,
    _: Dict[str, Any] = Depends(require_auth),
):
    tenant_id = getattr(request.state, "tenant_id", None)
    row = get_thread(thread_id, tenant_id)
    if not row:
        raise HTTPException(status_code=404, detail="thread_not_found")
    return row

