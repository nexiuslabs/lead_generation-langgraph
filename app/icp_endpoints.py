from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException, Header, Request
from typing import Optional, List, Dict, Any
from psycopg2.extras import Json

# Import the dependency and helpers at module scope so tests can monkeypatch ep.* symbols
from app.auth import require_auth  # noqa: F401  # exposed for monkeypatching in tests
from schemas.icp import IntakePayload, SuggestionCard, AcceptRequest
from src.database import get_conn
from src.icp_intake import (
    save_icp_intake,
    generate_suggestions,
    refresh_icp_patterns,
    build_targeting_pack,  # re-export for tests to monkeypatch
    _derive_negative_icp_flags,  # re-export for tests to monkeypatch
)
from src.jobs import (
    enqueue_icp_intake_process,  # re-export for tests to monkeypatch
    run_icp_intake_process,  # re-export for tests to monkeypatch
)

router = APIRouter(prefix="/icp", tags=["icp"])


def _resolve_tenant_id(req: Request, x_tenant_id: Optional[str]) -> Optional[int]:
    # Prefer X-Tenant-ID header if provided; else try cookie/context
    if x_tenant_id and str(x_tenant_id).strip().isdigit():
        return int(str(x_tenant_id).strip())
    try:
        ctx = getattr(req.state, "auth_ctx", {}) or {}
        tid = ctx.get("tenant_id")
        if tid is not None:
            return int(tid)
    except Exception:
        pass
    # Fallback to first active mapping in odoo_connections
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT tenant_id FROM odoo_connections WHERE active=TRUE LIMIT 1")
            row = cur.fetchone()
            return int(row[0]) if row and row[0] is not None else None
    except Exception:
        return None


def _save_icp_rule(tid: int, payload: Dict[str, Any], name: str = "Default ICP") -> None:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO icp_rules(tenant_id, name, payload) VALUES (%s,%s,%s)",
            (tid, name, Json(payload)),
        )


async def _auth_dep(request: Request):
    """Delegate to current module's require_auth so monkeypatching ep.require_auth works.

    FastAPI binds dependency callables at router registration time. By keeping this thin
    wrapper and calling our module-level symbol, tests can replace `require_auth`
    on this module and the wrapper will use the patched function at request time.
    """
    return await require_auth(request)  # type: ignore[misc]


@router.post("/intake")
async def post_intake(
    payload: IntakePayload,
    bg: BackgroundTasks,
    req: Request,
    user=Depends(_auth_dep),
    x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-ID"),
):
    tid = _resolve_tenant_id(req, x_tenant_id)
    if tid is None:
        raise HTTPException(status_code=400, detail="tenant_id is required")
    resp_id = save_icp_intake(tid, str(user.get("user_id") or "api"), payload.model_dump())

    # Enqueue full intake pipeline job and also attempt to run it immediately in background
    try:
        # Use module-scoped symbols so tests can monkeypatch ep.enqueue_icp_intake_process
        job = enqueue_icp_intake_process(tid)

        async def _run_now():
            try:
                await run_icp_intake_process(int(job.get("job_id") or 0))
            except Exception:
                pass

        bg.add_task(_run_now)
        return {"status": "queued", "response_id": resp_id, "job_id": job.get("job_id")}
    except Exception:
        # Fallback to old simple background path
        try:
            from src.icp_intake import map_seeds_to_evidence

            def _process():
                try:
                    map_seeds_to_evidence(tid)
                except Exception:
                    pass
                try:
                    refresh_icp_patterns()
                except Exception:
                    pass

            bg.add_task(_process)
        except Exception:
            pass
        return {"status": "queued", "response_id": resp_id}


@router.get("/suggestions", response_model=List[SuggestionCard])
async def get_suggestions(
    req: Request,
    user=Depends(_auth_dep),
    x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-ID"),
):
    tid = _resolve_tenant_id(req, x_tenant_id)
    if tid is None:
        raise HTTPException(status_code=400, detail="tenant_id is required")
    items = generate_suggestions(tid)
    # Optionally add targeting pack and negative ICP using module-scoped helpers
    try:
        # Derive negative ICP themes from latest intake answers
        neg: Optional[list] = None
        try:
            with get_conn() as conn:
                cur = conn.cursor()
                cur.execute(
                    "SELECT answers_jsonb FROM icp_intake_responses WHERE tenant_id=%s ORDER BY submitted_at DESC LIMIT 1",
                    (tid,),
                )
                row = cur.fetchone()
                answers = row[0] if row and row[0] is not None else {}
            neg = _derive_negative_icp_flags(answers)
        except Exception:
            neg = None
        for it in items:
            if isinstance(it, dict):
                it["targeting_pack"] = build_targeting_pack(it)
                if neg is not None:
                    it["negative_icp"] = neg
    except Exception:
        pass
    return items


@router.post("/accept")
async def post_accept(
    body: AcceptRequest,
    req: Request,
    user=Depends(_auth_dep),
    x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-ID"),
):
    tid = _resolve_tenant_id(req, x_tenant_id)
    if tid is None:
        raise HTTPException(status_code=400, detail="tenant_id is required")
    payload: Dict[str, Any] = {}
    name = "Accepted ICP"
    if body.suggestion_payload:
        payload = dict(body.suggestion_payload)
    elif body.suggestion_id:
        sid = body.suggestion_id.strip().lower()
        if sid.startswith("ssic:"):
            payload = {"ssic_codes": [sid.split(":", 1)[1]]}
            name = f"SSIC {payload['ssic_codes'][0]}"
    if not payload:
        raise HTTPException(status_code=400, detail="Invalid accept payload")
    _save_icp_rule(tid, payload, name=name)
    return {"ok": True}


@router.get("/patterns")
async def get_patterns(
    req: Request,
    user=Depends(require_auth),
    x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-ID"),
):
    tid = _resolve_tenant_id(req, x_tenant_id)
    if tid is None:
        raise HTTPException(status_code=400, detail="tenant_id is required")
    with get_conn() as conn, conn.cursor() as cur:
        try:
            cur.execute("SELECT aggregates FROM icp_patterns WHERE tenant_id=%s", (tid,))
            row = cur.fetchone()
            return row[0] if row and row[0] is not None else {}
        except Exception:
            return {}
