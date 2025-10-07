from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException, Header, Request
import asyncio
import os
from typing import Optional, List, Dict, Any
from psycopg2.extras import Json

# Import the dependency and helpers at module scope so tests can monkeypatch ep.* symbols
from app.auth import require_auth  # noqa: F401  # exposed for monkeypatching in tests
from schemas.icp import IntakePayload, SuggestionCard, AcceptRequest
from schemas.research import ResearchImportRequest, ResearchImportResult
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
            """
            INSERT INTO icp_rules(tenant_id, name, payload)
            VALUES (%s,%s,%s)
            ON CONFLICT (tenant_id, name) DO UPDATE SET
              payload = EXCLUDED.payload,
              created_at = NOW()
            """,
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

    # Enrich a small head now, and enqueue remainder for nightly
    try:
        # Resolve SSIC codes and human titles
        ssic_codes: list[str] = []
        titles: list[str] = []
        if payload.get("ssic_codes"):
            ssic_codes = [str(c).strip() for c in payload.get("ssic_codes") or [] if str(c).strip()]
            # Lookup titles for those codes
            with get_conn() as conn:
                cur = conn.cursor()
                cur.execute(
                    "SELECT title FROM ssic_ref WHERE regexp_replace(code::text,'\\D','','g') = ANY(%s::text[])",
                    (ssic_codes,),
                )
                titles = [r[0] for r in (cur.fetchall() or []) if r and r[0]]
        # Immediate head upsert + enrich
        head = 10
        try:
            import os as _os
            head = int(_os.getenv("RUN_NOW_LIMIT", "10") or 10)
        except Exception:
            head = 10
        try:
            from app.main import upsert_by_industries_head as _upsert_head  # uses staging by SSIC via resolver
            from app.main import _trigger_enrichment_async as _enrich_async
            inds = titles if titles else [f"SSIC {c}" for c in ssic_codes]
            if inds:
                ids = _upsert_head(inds, limit=head)
                if ids:
                    _enrich_async(ids)
        except Exception:
            pass
        # Enqueue remainder for nightly (staging upsert → enrich)
        try:
            from src.jobs import enqueue_staging_upsert as _enqueue_upsert
            if titles:
                _enqueue_upsert(tid, titles)
            elif ssic_codes:
                # Fallback: resolve titles from codes and enqueue
                with get_conn() as conn:
                    cur = conn.cursor()
                    cur.execute(
                        "SELECT DISTINCT title FROM ssic_ref WHERE regexp_replace(code::text,'\\D','','g') = ANY(%s::text[])",
                        (ssic_codes,),
                    )
                    trows = [r[0] for r in (cur.fetchall() or []) if r and r[0]]
                if trows:
                    _enqueue_upsert(tid, trows)
        except Exception:
            pass
    except Exception:
        # Non-blocking: acceptance persists even if scheduling fails
        pass
    return {"ok": True, "scheduled": True, "run_now": min(head, len(ssic_codes) if ssic_codes else head)}


@router.get("/top10")
async def get_top10(
    req: Request,
    user=Depends(_auth_dep),
    x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-ID"),
):
    """Return Top‑10 lookalikes with why/snippets using DDG+Jina discovery.

    Also persists lightweight preview evidence and scores to DB for auditability.
    """
    tid = _resolve_tenant_id(req, x_tenant_id)
    if tid is None:
        raise HTTPException(status_code=400, detail="tenant_id is required")
    # Build a minimal icp_profile from latest icp_rules payload when available
    icp_profile: Dict[str, Any] = {}
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT payload FROM icp_rules WHERE tenant_id=%s ORDER BY created_at DESC LIMIT 1",
                (tid,),
            )
            row = cur.fetchone()
            payload = (row and row[0]) or {}
            # Map simple keys if present
            if isinstance(payload, dict):
                if isinstance(payload.get("industries"), list):
                    icp_profile["industries"] = payload.get("industries")
                if isinstance(payload.get("integrations"), list):
                    icp_profile["integrations"] = payload.get("integrations")
                if isinstance(payload.get("buyer_titles"), list):
                    icp_profile["buyer_titles"] = payload.get("buyer_titles")
                if isinstance(payload.get("triggers"), list):
                    icp_profile["triggers"] = payload.get("triggers")
                if isinstance(payload.get("size_bands"), list):
                    icp_profile["size_bands"] = payload.get("size_bands")
    except Exception:
        icp_profile = {}
    # Run agents Top‑10
    try:
        from src.agents_icp import plan_top10_with_reasons as _top10
    except Exception:
        raise HTTPException(status_code=500, detail="agents unavailable")
    top = await asyncio.to_thread(_top10, icp_profile, tid)
    items: List[Dict[str, Any]] = []
    # Persist preview evidence and ensure company rows
    try:
        with get_conn() as conn, conn.cursor() as cur:
            for it in (top or [])[:10]:
                dom = (it.get("domain") or "").strip().lower()
                name = dom  # placeholder until crawl/enrichment fills real name
                if not dom:
                    continue
                # Ensure company row by domain if missing
                cur.execute(
                    "SELECT company_id, name FROM companies WHERE website_domain=%s",
                    (dom,),
                )
                r = cur.fetchone()
                if r and r[0] is not None:
                    cid = int(r[0])
                else:
                    cur.execute(
                        "INSERT INTO companies(name, website_domain, last_seen) VALUES (%s,%s, NOW()) RETURNING company_id",
                        (name, dom),
                    )
                    cid = int(cur.fetchone()[0])
                # Write preview evidence (why/snippet) for auditability
                why = it.get("why") or ""
                snip = it.get("snippet") or ""
                try:
                    cur.execute(
                        "INSERT INTO icp_evidence(tenant_id, company_id, signal_key, value, source) VALUES (%s,%s,%s,%s,'web_preview')",
                        (tid, cid, "top10_preview", Json({"why": why, "snippet": snip})),
                    )
                except Exception:
                    pass
                # Upsert into lead_scores with preview fields
                try:
                    score = float(it.get("score") or 0)
                except Exception:
                    score = 0.0
                bucket = it.get("bucket") or "C"
                rationale = why or snip
                cur.execute(
                    """
                    INSERT INTO lead_scores(company_id, score, bucket, rationale, cache_key)
                    VALUES (%s,%s,%s,%s,NULL)
                    ON CONFLICT (company_id) DO UPDATE SET
                      score = EXCLUDED.score,
                      bucket = EXCLUDED.bucket,
                      rationale = EXCLUDED.rationale
                    """,
                    (cid, score, bucket, rationale),
                )
                items.append({"company_id": cid, **it})
    except Exception:
        # Non-fatal: still return top list
        items = [{**it} for it in (top or [])]
    return {"items": items}


@router.post("/run")
async def post_icp_run(
    req: Request,
    user=Depends(_auth_dep),
    x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-ID"),
):
    """Shortcut endpoint: generate Top‑10 and persist preview artifacts.

    Returns the same payload as GET /icp/top10.
    """
    return await get_top10(req, user, x_tenant_id)  # type: ignore[misc]


@router.post("/research/import", response_model=ResearchImportResult)
async def post_research_import(
    body: ResearchImportRequest,
    req: Request,
    user=Depends(_auth_dep),
    x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-ID"),
):
    tid = _resolve_tenant_id(req, x_tenant_id)
    if tid is None:
        raise HTTPException(status_code=400, detail="tenant_id is required")
    if int(body.tenant_id) != int(tid):
        raise HTTPException(status_code=403, detail="tenant mismatch")
    # Only allow server-side scanning by path for now
    root = body.root or os.getenv("DOCS_ROOT") or "./docs"
    try:
        from src.research_import import import_docs_for_tenant

        result = await asyncio.to_thread(import_docs_for_tenant, tid, root)
        # Coerce to schema
        return ResearchImportResult(**result)
    except Exception as e:  # noqa: F841
        raise HTTPException(status_code=500, detail="import failed")


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
