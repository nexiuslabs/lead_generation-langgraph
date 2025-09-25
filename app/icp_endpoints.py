from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import Any, Dict
import os

from app.auth import require_auth
from src.icp_intake import save_icp_intake, map_seeds_to_evidence, refresh_icp_patterns, generate_suggestions

router = APIRouter(prefix="/icp", tags=["icp"])


def _require_role(claims: dict, allowed: set[str]) -> None:
    roles = set((claims or {}).get("roles", []) or [])
    # Allow all authenticated users in non-production to ease local development
    if not roles.intersection(allowed):
        env = (os.getenv("NODE_ENV") or os.getenv("PYTHON_ENV") or os.getenv("ENV") or "").lower()
        if env not in ("production", "prod"):
            return
        raise HTTPException(status_code=403, detail="forbidden: missing role")


@router.post("/intake")
async def icp_intake(body: Dict[str, Any], background: BackgroundTasks, claims: dict = Depends(require_auth)):
    _require_role(claims, {"ops", "admin"})
    tenant_id = int(claims.get("tenant_id"))
    submitted_by = str(claims.get("email") or claims.get("preferred_username") or claims.get("sub") or "unknown")
    resp_id = save_icp_intake(tenant_id, submitted_by, body or {})
    # Background mapping + patterns refresh
    def _job(tid: int):
        try:
            map_seeds_to_evidence(tid)
            refresh_icp_patterns()
        except Exception:
            pass
    background.add_task(_job, tenant_id)
    return {"status": "queued", "response_id": resp_id}


@router.get("/suggestions")
async def icp_suggestions(claims: dict = Depends(require_auth)):
    # viewer read is allowed
    tenant_id = int(claims.get("tenant_id"))
    items = generate_suggestions(tenant_id)
    return {"items": items}


@router.post("/accept")
async def icp_accept(body: Dict[str, Any], claims: dict = Depends(require_auth)):
    _require_role(claims, {"ops", "admin"})
    # v1 stub: we would normalize to icp_rules; for now, acknowledge.
    return {"ok": True}


@router.get("/patterns")
async def icp_patterns(claims: dict = Depends(require_auth)):
    # Optional ops view — fetch raw MV rows for the tenant
    from src.database import get_conn
    import json
    out = {}
    tid = int(claims.get("tenant_id")) if claims and claims.get("tenant_id") is not None else None
    try:
        with get_conn() as conn, conn.cursor() as cur:
            if tid is None:
                raise Exception("tenant_id missing")
            cur.execute("SELECT top_ssics FROM icp_patterns WHERE tenant_id=%s", (tid,))
            row = cur.fetchone()
            out = {"top_ssics": row[0] if row else None}
    except Exception:
        out = {"top_ssics": None}
    return out
