from __future__ import annotations

from fastapi import APIRouter, Body, Depends, HTTPException, Request
from typing import Any, Dict, Optional
import os
import asyncio
import httpx
from .thread_runtime_map import get_runtime_id, set_runtime_id
import logging
from src.troubleshoot_log import log_json

from app.auth import require_auth
from .threads_db import (
    context_key_from_payload,
    create_thread,
    lock_prior_open,
    auto_archive_stale_locked,
    resume_eligible,
    get_thread,
    list_threads,
    archive_thread,
    hard_delete_thread,
)
from src.settings import (
    THREAD_RESUME_WINDOW_DAYS,
    THREAD_STALE_DAYS,
    AUTO_ARCHIVE_STALE_LOCKED,
)


router = APIRouter()
_lg = logging.getLogger("threads")


def _label_for_context(context_key: str, fallback: Optional[str] = None) -> str:
    if isinstance(context_key, str) and context_key.startswith("domain:"):
        return context_key.split(":", 1)[1]
    return fallback or "ICP session"


def _derive_label(context_key: str, payload: Dict[str, Any]) -> str:
    # 1) explicit label wins
    raw = (payload or {}).get("label")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    # 2) domain label
    if isinstance(context_key, str) and context_key.startswith("domain:"):
        return context_key.split(":", 1)[1]
    # 3) use input snippet
    txt = (payload or {}).get("input")
    if isinstance(txt, str) and txt.strip():
        s = " ".join(txt.strip().split())
        return s[:60]
    # 4) fall back to last user message
    msgs = (payload or {}).get("messages")
    if isinstance(msgs, list):
        for m in reversed(msgs):
            try:
                if str(m.get("role") or "").lower() == "user":
                    c = str(m.get("content") or "").strip()
                    if c:
                        return (" ".join(c.split()))[:60]
            except Exception:
                continue
    # 5) generic
    return "ICP session"


@router.post("/threads")
async def create_thread_route(
    request: Request,
    payload: Dict[str, Any] = Body(default_factory=dict),
    identity: Dict[str, Any] = Depends(require_auth),
):
    tenant_id = getattr(request.state, "tenant_id", None)
    user_id = identity.get("sub")
    context_key = context_key_from_payload(payload, tenant_id)
    label = _derive_label(context_key, payload)

    # Create new open thread and lock prior open threads for the same context
    tid = create_thread(tenant_id, user_id, "icp_finder", context_key, label)
    locked = 0
    try:
        locked = lock_prior_open(tenant_id, user_id, "icp_finder", context_key, tid)
    finally:
        try:
            _lg.info(
                "threads:create db_thread id=%s tenant=%s user=%s ctx=%s label=%s locked_prior=%s",
                tid,
                tenant_id,
                user_id,
                context_key,
                label,
                locked,
            )
            log_json(
                "threads",
                "info",
                "db_thread_create",
                {
                    "thread_id": tid,
                    "tenant_id": tenant_id,
                    "user_id": user_id,
                    "context_key": context_key,
                    "label": label,
                    "locked_prior": locked,
                },
            )
        except Exception:
            pass
    if AUTO_ARCHIVE_STALE_LOCKED:
        try:
            auto_archive_stale_locked(tenant_id, THREAD_STALE_DAYS)
        except Exception:
            pass

    # Best-effort: also create a matching LangGraph thread so SDK calls to
    # /threads/{id}/history or /threads/{id}/runs do not 404 after dev reloads.
    async def _create_remote_thread_if_configured(thread_id: str) -> Optional[str]:
        base = (os.getenv("LANGGRAPH_REMOTE_URL") or "").strip()
        if not base:
            return None
        try:
            url = base.rstrip("/") + "/threads"
            headers = {"content-type": "application/json"}
            api_key = (os.getenv("LANGSMITH_API_KEY") or "").strip()
            if api_key:
                headers["x-api-key"] = api_key
            tenant = getattr(request.state, "tenant_id", None)
            if tenant is not None:
                headers["x-tenant-id"] = str(tenant)
            log_json("threads", "info", "runtime_create_attempt", {"thread_id": thread_id, "base": base})
            async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0)) as client:
                # Create runtime thread using the same id as our DB thread when supported
                # Many LangGraph servers accept `thread_id` as an override on POST /threads
                resp = await client.post(url, headers=headers, json={"thread_id": thread_id})
                rid: Optional[str] = None
                if resp.status_code in (200, 201):
                    try:
                        data = resp.json()
                        rid = str(data.get("thread_id") or data.get("id") or "") or None
                    except Exception:
                        rid = None
                if rid:
                    set_runtime_id(thread_id, rid)
                    try:
                        same = (rid == thread_id)
                    except Exception:
                        same = False
                    log_json("threads", "info", "runtime_thread_bound", {"thread_id": thread_id, "runtime_thread_id": rid, "same_id": bool(same)})
                else:
                    log_json("threads", "warning", "runtime_create_failed", {"thread_id": thread_id, "status": getattr(resp, 'status_code', None)})
                return rid
        except Exception as e:
            # Ignore failures; thread will be auto-recreated on first 404 by UI proxy
            log_json("threads", "warning", "runtime_create_exception", {"thread_id": thread_id, "error": str(e)})
            return None

    try:
        # Fire-and-forget; do not delay response
        asyncio.create_task(_create_remote_thread_if_configured(tid))
    except Exception:
        pass

    # Return canonical DB row to reflect actual label/status
    row = get_thread(tid, tenant_id)
    return row or {"id": tid, "status": "open", "context_key": context_key, "label": label}


@router.get("/threads/{thread_id}/runtime")
async def get_runtime_thread_route(thread_id: str, request: Request, _: Dict[str, Any] = Depends(require_auth)):
    """Return or create the runtime thread id for a DB thread id.

    - If a mapping exists, return it.
    - Otherwise, attempt to create a runtime thread and store the mapping.
    """
    rid = get_runtime_id(thread_id)
    if rid:
        try:
            log_json("threads", "info", "runtime_mapping_exists", {"thread_id": thread_id, "runtime_thread_id": rid})
        except Exception:
            pass
        return {"thread_id": thread_id, "runtime_thread_id": rid}
    # Create runtime thread now
    # Attempt to create a new runtime thread now (same as on create)
    rid = None
    base = (os.getenv("LANGGRAPH_REMOTE_URL") or "").strip()
    if base:
        try:
            url = base.rstrip("/") + "/threads"
            headers = {"content-type": "application/json"}
            api_key = (os.getenv("LANGSMITH_API_KEY") or "").strip()
            if api_key:
                headers["x-api-key"] = api_key
            tenant = getattr(request.state, "tenant_id", None)
            if tenant is not None:
                headers["x-tenant-id"] = str(tenant)
            log_json("threads", "info", "runtime_create_attempt", {"thread_id": thread_id, "base": base})
            async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0)) as client:
                # Prefer a stable id equal to our DB thread id
                resp = await client.post(url, headers=headers, json={"thread_id": thread_id})
                if resp.status_code in (200, 201):
                    try:
                        data = resp.json()
                        rid = str(data.get("thread_id") or data.get("id") or "") or None
                    except Exception:
                        rid = None
        except Exception:
            rid = None
    if rid:
        set_runtime_id(thread_id, rid)
        try:
            log_json("threads", "info", "runtime_thread_created", {"thread_id": thread_id, "runtime_thread_id": rid, "same_id": bool(rid == thread_id)})
        except Exception:
            pass
    if not rid:
        try:
            log_json("threads", "warning", "runtime_thread_not_found", {"thread_id": thread_id})
        except Exception:
            pass
        raise HTTPException(status_code=404, detail="runtime_thread_not_found")
    return {"thread_id": thread_id, "runtime_thread_id": rid}


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


@router.patch("/threads/{thread_id}")
async def rename_thread_route(
    thread_id: str,
    body: Dict[str, Any],
    request: Request,
    _: Dict[str, Any] = Depends(require_auth),
):
    tenant_id = getattr(request.state, "tenant_id", None)
    new_label = (body or {}).get("label")
    if not isinstance(new_label, str) or not new_label.strip():
        raise HTTPException(status_code=400, detail="label required")
    from src.database import get_conn
    with get_conn() as conn, conn.cursor() as cur:
        try:
            if tenant_id is not None:
                cur.execute("SELECT set_config('request.tenant_id', %s, true)", (str(int(tenant_id)),))
        except Exception:
            pass
        cur.execute("UPDATE threads SET label=%s, last_updated_at=now() WHERE id=%s", (new_label.strip(), thread_id))
        if (cur.rowcount or 0) <= 0:
            raise HTTPException(status_code=404, detail="thread_not_found")
    return {"ok": True, "id": thread_id, "label": new_label.strip()}


@router.delete("/threads/{thread_id}")
async def delete_thread_route(
    thread_id: str,
    request: Request,
    hard: bool = False,
    claims: Dict[str, Any] = Depends(require_auth),
):
    """Delete a thread.

    - Default (hard=false): archive the thread (soft delete).
    - hard=true: permanently delete. Requires 'admin' role.
    """
    tenant_id = getattr(request.state, "tenant_id", None)
    if hard:
        roles = claims.get("roles", []) or []
        if "admin" not in roles:
            raise HTTPException(status_code=403, detail="admin role required for hard delete")
        n = hard_delete_thread(thread_id, tenant_id)
        if n <= 0:
            raise HTTPException(status_code=404, detail="thread_not_found")
        return {"ok": True, "deleted": True, "id": thread_id}
    # Soft delete (archive)
    n = archive_thread(thread_id, tenant_id, reason="user_archive")
    if n <= 0:
        raise HTTPException(status_code=404, detail="thread_not_found")
    return {"ok": True, "archived": True, "id": thread_id}
