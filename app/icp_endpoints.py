from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException, Header, Request
import logging
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
from src.enrichment import enrich_company_with_tavily  # async enrich by company_id
from src.chat_events import emit as emit_chat_event

router = APIRouter(prefix="/icp", tags=["icp"])
log = logging.getLogger("icp_endpoints")


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
    x_session_id: Optional[str] = Header(default=None, alias="X-Session-ID"),
):
    tid = _resolve_tenant_id(req, x_tenant_id)
    if tid is None:
        raise HTTPException(status_code=400, detail="tenant_id is required")
    resp_id = save_icp_intake(tid, str(user.get("user_id") or "api"), payload.model_dump())
    # Emit intake saved and confirm pending for interactive chat
    try:
        emit_chat_event(x_session_id, tid, "icp:intake_saved", "Received your ICP answers and seeds. Normalizing and saving…", {"response_id": resp_id})
        emit_chat_event(x_session_id, tid, "icp:confirm_pending", "I’ll crawl your site + seed sites… Reply ‘confirm’ to proceed.", {})
    except Exception:
        pass

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
                    try:
                        emit_chat_event(x_session_id, tid, "icp:seeds_mapped", "Anchoring seeds to company records and ACRA. Extracting SSIC codes…", {})
                    except Exception:
                        pass
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


@router.post("/enrich/top10")
async def enrich_top10(
    req: Request,
    user=Depends(_auth_dep),
    x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-ID"),
    x_session_id: Optional[str] = Header(default=None, alias="X-Session-ID"),
):
    """Enrich the persisted Top‑10 preview strictly by tenant.

    Looks up Top‑10 domains from staging_global_companies where ai_metadata.preview=true
    ordered by preview score, maps to company_ids, and enriches them one by one.
    """
    tid = _resolve_tenant_id(req, x_tenant_id)
    if tid is None:
        raise HTTPException(status_code=400, detail="tenant_id is required")
    try:
        run_now_limit = int(os.getenv("RUN_NOW_LIMIT", "10") or 10)
    except Exception:
        run_now_limit = 10
    domains: list[str] = []
    with get_conn() as conn, conn.cursor() as cur:
        try:
            cur.execute(
                """
                SELECT domain
                FROM staging_global_companies
                WHERE tenant_id=%s AND COALESCE((ai_metadata->>'preview')::boolean,false)=true
                ORDER BY COALESCE((ai_metadata->>'score')::float,0) DESC
                LIMIT %s
                """,
                (tid, run_now_limit),
            )
            rows = cur.fetchall() or []
            domains = [str(r[0]) for r in rows if r and r[0]]
        except Exception:
            domains = []
    # Fallback: if no persisted Top‑10 preview, compute a fresh Top‑10, persist, and use it
    if not domains:
        # Reconstruct a minimal icp_profile from latest icp_rules
        icp_profile: Dict[str, Any] = {}
        try:
            with get_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT payload FROM icp_rules WHERE tenant_id=%s ORDER BY created_at DESC LIMIT 1",
                    (tid,),
                )
                row = cur.fetchone()
                payload = (row and row[0]) or {}
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
        # Plan Top‑10 with reasons using agents helper
        try:
            from src.agents_icp import plan_top10_with_reasons as _top10  # type: ignore
        except Exception:
            _top10 = None  # type: ignore
        if _top10 is None:
            raise HTTPException(status_code=412, detail="top10 preview not found and agents unavailable")
        top = await asyncio.to_thread(_top10, icp_profile, tid)
        if not top:
            raise HTTPException(status_code=412, detail="top10 preview not found; please confirm to regenerate")
        # Persist preview rows and discovered candidates for auditability and reuse
        try:
            from app.pre_sdr_graph import _persist_top10_preview, _persist_web_candidates_to_staging  # type: ignore
            # Persist Top‑10 preview with why/snippet/score
            _persist_top10_preview(tid, top)
            # Persist any additional discovered candidates (beyond Top‑10) into staging without preview flag
            rest = [str(it.get("domain")).strip().lower() for it in (top[10:] if len(top) > 10 else []) if isinstance(it, dict) and it.get("domain")]
            if rest:
                _persist_web_candidates_to_staging(rest, tid, ai_metadata={"provenance": {"agent": "agents_icp.plan_top10", "stage": "staging"}})
        except Exception:
            # Best-effort: proceed with enrichment even if persistence fails
            pass
        domains = [str(it.get("domain")) for it in (top or [])[:run_now_limit] if isinstance(it, dict) and it.get("domain")]
    # Map to company_ids
    company_ids: list[int] = []
    with get_conn() as conn, conn.cursor() as cur:
        try:
            cur.execute(
                "SELECT company_id FROM companies WHERE LOWER(website_domain) = ANY(%s)",
                ([d.lower() for d in domains],),
            )
            rows = cur.fetchall() or []
            company_ids = [int(r[0]) for r in rows if r and r[0] is not None]
        except Exception:
            company_ids = []
    processed = 0
    # Emit enrichment start for interactive chat sessions
    try:
        emit_chat_event(x_session_id, tid, "enrich:start_top10", "Enriching Top‑10 now (require existing domains).", {"requested": len(company_ids)})
    except Exception:
        pass
    for cid in company_ids:
        try:
            # Top-10 enrichment expects domains already present; skip domain search
            await enrich_company_with_tavily(int(cid), search_policy="require_existing")
            processed += 1
            try:
                emit_chat_event(x_session_id, tid, "enrich:company_tick", f"Enriched company_id={cid}", {"company_id": int(cid)})
            except Exception:
                pass
        except Exception:
            # continue with best-effort behavior
            pass
    # After Top‑10 enrichment, enqueue the next 40 for background enrichment
    bg_job_id = None
    try:
        from src.jobs import enqueue_web_discovery_bg_enrich as _enqueue_bg
        next_domains: list[str] = []
        try:
            with get_conn() as conn, conn.cursor() as cur:
                # Prefer staged web discovery rows excluding preview to select the next set
                cur.execute(
                    """
                    WITH preview AS (
                      SELECT LOWER(domain) AS d
                      FROM staging_global_companies
                      WHERE tenant_id=%s AND COALESCE((ai_metadata->>'preview')::boolean,false)=true
                      ORDER BY COALESCE((ai_metadata->>'score')::float,0) DESC
                      LIMIT %s
                    )
                    SELECT domain
                      FROM staging_global_companies
                     WHERE tenant_id=%s
                       AND source='web_discovery'
                       AND COALESCE((ai_metadata->>'preview')::boolean,false)=false
                       AND LOWER(domain) NOT IN (SELECT d FROM preview)
                     ORDER BY created_at DESC
                     LIMIT %s
                    """,
                    (tid, run_now_limit, tid, int(os.getenv("BG_NEXT_COUNT", "40") or 40)),
                )
                rows2 = cur.fetchall() or []
                next_domains = [str(r[0]) for r in rows2 if r and r[0]]
        except Exception:
            next_domains = []
        if next_domains:
            # Map to company_ids (ensure rows exist)
            ids: list[int] = []
            with get_conn() as conn, conn.cursor() as cur:
                try:
                    cur.execute(
                        "SELECT company_id, website_domain FROM companies WHERE LOWER(website_domain) = ANY(%s)",
                        ([d.lower() for d in next_domains],),
                    )
                    found = {str((r[1] or "").lower()): int(r[0]) for r in (cur.fetchall() or []) if r and r[0] is not None}
                    for d in next_domains:
                        cid = found.get(str(d.lower()))
                        if cid is not None:
                            ids.append(int(cid))
                        else:
                            # ensure a row exists for this domain
                            cur.execute(
                                "INSERT INTO companies(name, website_domain, last_seen) VALUES (%s,%s,NOW()) RETURNING company_id",
                                (d, d),
                            )
                            ids.append(int(cur.fetchone()[0]))
                except Exception:
                    ids = []
            if ids:
                job = _enqueue_bg(int(tid), ids)
                bg_job_id = job.get("job_id") if isinstance(job, dict) else None
                try:
                    emit_chat_event(
                        x_session_id,
                        tid,
                        "enrich:next40_enqueued",
                        "Queued background enrichment for the next 40 candidates.",
                        {"job_id": bg_job_id, "count": len(ids)},
                    )
                except Exception:
                    pass
    except Exception:
        bg_job_id = None
    # Emit enrichment summary
    try:
        emit_chat_event(x_session_id, tid, "enrich:summary", f"Enriched results ({processed}/{len(company_ids)} completed). Remaining queued.", {"processed": processed, "requested": len(company_ids), "next40_job_id": bg_job_id})
    except Exception:
        pass
    return {"ok": True, "requested": len(company_ids), "processed": processed, "next40_job_id": bg_job_id}


@router.post("/enrich/next40")
async def enrich_next40(
    req: Request,
    user=Depends(_auth_dep),
    x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-ID"),
):
    """Enqueue background enrichment for the next 40 preview domains (Non‑SG only).

    Selects the next set of preview domains after the Top‑10 for the same tenant,
    resolves to company_ids, and enqueues a background job to enrich them.
    """
    tid = _resolve_tenant_id(req, x_tenant_id)
    if tid is None:
        raise HTTPException(status_code=400, detail="tenant_id is required")
    try:
        run_now_limit = int(os.getenv("RUN_NOW_LIMIT", "10") or 10)
        bg_next_count = int(os.getenv("BG_NEXT_COUNT", "40") or 40)
    except Exception:
        run_now_limit, bg_next_count = 10, 40
    domains: list[str] = []
    with get_conn() as conn, conn.cursor() as cur:
        try:
            cur.execute(
                """
                SELECT domain
                FROM staging_global_companies
                WHERE tenant_id=%s AND COALESCE((ai_metadata->>'preview')::boolean,false)=true
                ORDER BY COALESCE((ai_metadata->>'score')::float,0) DESC
                OFFSET %s LIMIT %s
                """,
                (tid, run_now_limit, bg_next_count),
            )
            rows = cur.fetchall() or []
            domains = [str(r[0]) for r in rows if r and r[0]]
        except Exception:
            domains = []
    if not domains:
        raise HTTPException(status_code=404, detail="no next40 preview domains found")
    # Resolve to company_ids
    company_ids: list[int] = []
    with get_conn() as conn, conn.cursor() as cur:
        try:
            cur.execute(
                "SELECT company_id, website_domain FROM companies WHERE LOWER(website_domain) = ANY(%s)",
                ([d.lower() for d in domains],),
            )
            rows = cur.fetchall() or []
            found = {str((r[1] or "").lower()): int(r[0]) for r in rows if r and r[0] is not None}
            for d in domains:
                cid = found.get(str(d.lower()))
                if cid:
                    company_ids.append(cid)
        except Exception:
            company_ids = []
    if not company_ids:
        raise HTTPException(status_code=404, detail="no company_ids resolved for next40")
    # Enqueue background job
    try:
        from src.jobs import enqueue_web_discovery_bg_enrich
        job = enqueue_web_discovery_bg_enrich(int(tid), company_ids)
        return {"ok": True, "job_id": job.get("job_id")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"enqueue failed: {e}")


@router.get("/top10")
async def get_top10(
    req: Request,
    user=Depends(_auth_dep),
    x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-ID"),
    x_session_id: Optional[str] = Header(default=None, alias="X-Session-ID"),
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
    # Planning start event (only when part of interactive chat session)
    try:
        emit_chat_event(x_session_id, tid, "icp:planning_start", "Confirmed. Gathering evidence and planning Top‑10…", {})
    except Exception:
        pass
    top = await asyncio.to_thread(_top10, icp_profile, tid)
    try:
        log.info("[top10] planned candidates count=%d tenant_id=%s", len(top or []), tid)
    except Exception:
        pass
    # Emit toplikes ready event
    try:
        emit_chat_event(x_session_id, tid, "icp:toplikes_ready", "Top‑listed lookalikes (with why) produced.", {"count": len(top or [])})
    except Exception:
        pass
    # Persist to staging so Next‑40 can always be enqueued later
    try:
        from app.pre_sdr_graph import _persist_web_candidates_to_staging  # type: ignore
        # Build per-domain preview metadata for Top‑10
        per_meta: Dict[str, Dict[str, Any]] = {}
        for it in (top or [])[:10]:
            try:
                dom = (it.get("domain") or "").strip().lower() if isinstance(it, dict) else ""
                if not dom:
                    continue
                per_meta[dom] = {
                    "preview": True,
                    "score": it.get("score"),
                    "bucket": it.get("bucket"),
                    "why": it.get("why"),
                    "snippet": (it.get("snippet") or "")[:200],
                    "provenance": {"agent": "agents_icp.plan_top10", "stage": "preview"},
                }
            except Exception:
                continue
        # Persist Top‑10 preview rows
        try:
            _persist_web_candidates_to_staging(
                [str(it.get("domain")).strip().lower() for it in (top or [])[:10] if isinstance(it, dict) and it.get("domain")],
                tid,
                ai_metadata={"provenance": {"agent": "agents_icp.plan_top10"}},
                per_domain_meta=per_meta,
            )
        except Exception:
            pass
        # Persist remainder (beyond Top‑10) as non‑preview rows
        rest = [
            str(it.get("domain")).strip().lower()
            for it in (top[10:] if isinstance(top, list) and len(top) > 10 else [])
            if isinstance(it, dict) and it.get("domain")
        ]
        if rest:
            try:
                _persist_web_candidates_to_staging(
                    rest,
                    tid,
                    ai_metadata={"provenance": {"agent": "agents_icp.plan_top10", "stage": "staging"}},
                )
            except Exception:
                pass
    except Exception:
        # Best-effort; do not block Top‑10 response
        pass
    items: List[Dict[str, Any]] = []
    # Persist preview evidence and ensure company rows (Top‑10 only)
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
    # Emit profile ready and candidates found events (profile computed from evidence stats)
    try:
        from src.icp_pipeline import winner_profile
        profile = winner_profile(int(tid)) if tid is not None else {}
        emit_chat_event(x_session_id, tid, "icp:profile_ready", "ICP Profile produced.", {})
        emit_chat_event(x_session_id, tid, "icp:candidates_found", f"Found {len(top or [])} ICP candidates. We can enrich 10 now…", {"count": len(top or [])})
    except Exception:
        pass
    return {"items": items}


@router.post("/chat/confirm")
async def post_chat_confirm(
    req: Request,
    user=Depends(_auth_dep),
    x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-ID"),
    x_session_id: Optional[str] = Header(default=None, alias="X-Session-ID"),
):
    """Confirm gating for interactive chat flow.

    Emits a planning start event and returns 202 to indicate the UI may proceed
    to call /icp/top10. This endpoint does not block on planning or enrichment.
    """
    tid = _resolve_tenant_id(req, x_tenant_id)
    if tid is None:
        raise HTTPException(status_code=400, detail="tenant_id is required")
    try:
        emit_chat_event(x_session_id, tid, "icp:planning_start", "Confirmed. Gathering evidence and planning Top‑10…", {})
    except Exception:
        pass
    return {"ok": True, "status": 202}


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
