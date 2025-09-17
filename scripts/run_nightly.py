import os
import sys
import asyncio
import logging
from typing import List, Optional, Dict, Any

from src.icp import icp_refresh_agent, normalize_agent
from src.enrichment import enrich_company_with_tavily
from src.lead_scoring import lead_scoring_agent
from src.database import get_conn
from app.odoo_store import OdooStore
from src.orchestrator import (
    fetch_industry_codes_by_names,
    fetch_candidate_ids_by_industry_codes,
)
from src import obs
from src import enrichment as enrich_mod

# Ensure project root is on sys.path so `src` and other top-level modules import
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

LOG = logging.getLogger("nightly")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


TENANT_CONC = _int_env("SCHED_TENANT_CONCURRENCY", 3)
COMPANY_CONC = _int_env("SCHED_COMPANY_CONCURRENCY", 8)
# Default daily cap now 20 per PRD-7
DAILY_CAP = _int_env("SCHED_DAILY_CAP_PER_TENANT", _int_env("SHORTLIST_DAILY_CAP", 20))
BATCH_SIZE = _int_env("SCHED_COMPANY_BATCH_SIZE", 1)

def _bool_env(name: str, default: bool) -> bool:
    val = (os.getenv(name, "") or "").strip().lower()
    if val in ("1", "true", "yes", "on"): return True
    if val in ("0", "false", "no", "off"): return False
    return default

AUTO_CONTINUE = _bool_env("SCHED_AUTO_BATCH_CONTINUE", False)
PAUSE_SECONDS = _int_env("SCHED_BATCH_PAUSE_SECONDS", 0)
TAVILY_MAX_QUERIES = _int_env("TAVILY_MAX_QUERIES", 0)  # 0 means no extra cap beyond DAILY_CAP
LUSHA_MAX_CONTACT_LOOKUPS = _int_env("LUSHA_MAX_CONTACT_LOOKUPS", 0)
ZB_MAX_VERIFY = _int_env("ZEROBOUNCE_MAX_VERIFICATIONS", 0)
ZB_BATCH_SIZE = _int_env("ZEROBOUNCE_BATCH_SIZE", 50)


def list_active_tenants() -> List[int]:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT t.tenant_id
            FROM tenants t
            JOIN odoo_connections oc ON oc.tenant_id=t.tenant_id AND oc.active
            WHERE t.status='active'
            ORDER BY t.tenant_id
            """
        )
        return [int(r[0]) for r in cur.fetchall()]


def _load_icp_payload_from_db(tenant_id: int, rule_name: Optional[str]) -> Optional[Dict[str, Any]]:
    """Load ICP payload for the tenant from icp_rules.

    Preference order:
    1) Match by name (ICP_RULE_NAME), newest first
    2) Latest rule for the tenant by created_at
    Returns a dict on success, otherwise None.
    """
    try:
        with get_conn() as conn, conn.cursor() as cur:
            # Ensure tenant RLS context (best-effort)
            try:
                cur.execute("SELECT set_config('request.tenant_id', %s, true)", (tenant_id,))
            except Exception:
                pass
            if rule_name:
                try:
                    cur.execute(
                        """
                        SELECT payload
                        FROM icp_rules
                        WHERE tenant_id=%s AND name=%s
                        ORDER BY created_at DESC
                        LIMIT 1
                        """,
                        (tenant_id, rule_name),
                    )
                    row = cur.fetchone()
                    if row and row[0]:
                        payload = row[0]
                        # psycopg2 returns dict for jsonb
                        if isinstance(payload, dict):
                            return payload
                        try:
                            import json as _json
                            return _json.loads(payload)
                        except Exception:
                            return None
                except Exception:
                    # fall through to latest
                    pass
            # Fallback: latest by created_at
            try:
                cur.execute(
                    """
                    SELECT payload
                    FROM icp_rules
                    WHERE tenant_id=%s
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (tenant_id,),
                )
                row = cur.fetchone()
                if row and row[0]:
                    payload = row[0]
                    if isinstance(payload, dict):
                        return payload
                    try:
                        import json as _json
                        return _json.loads(payload)
                    except Exception:
                        return None
            except Exception:
                return None
    except Exception:
        return None
    return None


def select_target_set(candidates: List[int], limit: int) -> List[int]:
    if not candidates:
        return []
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.company_id
            FROM companies c
            LEFT JOIN lead_scores s ON s.company_id=c.company_id
            LEFT JOIN contacts k ON k.company_id=c.company_id
            WHERE c.company_id = ANY(%s)
            ORDER BY (s.company_id IS NULL) DESC,
                     (k.company_id IS NULL) DESC,
                     c.last_seen DESC NULLS LAST
            LIMIT %s
            """,
            (candidates, limit),
        )
        return [int(r[0]) for r in cur.fetchall()]


async def enrich_many(company_ids: List[int]):
    sem = asyncio.Semaphore(COMPANY_CONC)

    async def _one(cid: int):
        async with sem:
            try:
                await enrich_company_with_tavily(cid, None)
            except Exception as e:
                LOG.warning("enrich failed company_id=%s err=%s", cid, e)

    await asyncio.gather(*[_one(cid) for cid in company_ids])


async def run_tenant(tenant_id: int):
    os.environ["DEFAULT_TENANT_ID"] = str(tenant_id)
    LOG.info("tenant=%s start", tenant_id)
    import time as _time
    _t_run_start = _time.perf_counter()
    # Begin run and set context for vendor accounting
    run_id = obs.begin_run(tenant_id)
    # Vendor caps: None means unlimited
    tav_units = TAVILY_MAX_QUERIES if TAVILY_MAX_QUERIES > 0 else None
    contact_cap = LUSHA_MAX_CONTACT_LOOKUPS if LUSHA_MAX_CONTACT_LOOKUPS > 0 else None
    try:
        enrich_mod.set_run_context(run_id, tenant_id)
        enrich_mod.set_vendor_caps(tavily_units=tav_units, contact_lookups=contact_cap)
        enrich_mod.reset_vendor_counters()
    except Exception:
        pass
    # 0) Optional normalization pass (seed companies from staging if available)
    try:
        run_norm = (os.getenv("RUN_NORMALIZATION_FIRST", "true").strip().lower() in ("1", "true", "yes", "on"))
        if run_norm:
            await normalize_agent.ainvoke({"raw_records": [], "normalized_records": []})
    except Exception as _e:
        LOG.info("tenant=%s normalization skipped: %s", tenant_id, _e)

    # Build ICP payload from env
    def _payload_from_env() -> dict:
        inds_env = (os.getenv("ICP_INDUSTRIES", "") or "").strip()
        industries = [s.strip() for s in inds_env.split(",") if s.strip()] or ["Technology"]
        def _int(name: str, default: int) -> int:
            try:
                return int(os.getenv(name, str(default)))
            except Exception:
                return default
        return {
            "industries": industries,
            "employee_range": {"min": _int("ICP_EMPLOYEES_MIN", 2), "max": _int("ICP_EMPLOYEES_MAX", 100)},
            "incorporation_year": {"min": _int("ICP_YEAR_MIN", 2000), "max": _int("ICP_YEAR_MAX", 2025)},
        }

    # Prefer DB-stored ICP per tenant; fallback to env payload
    rule_name = os.getenv("ICP_RULE_NAME", "default")
    icp_payload = _load_icp_payload_from_db(tenant_id, rule_name) or _payload_from_env()
    try:
        src = "db(icp_rules)" if icp_payload is not None and _load_icp_payload_from_db(tenant_id, rule_name) else "env"
        LOG.info("tenant=%s using ICP source=%s payload=%s", tenant_id, src, icp_payload)
    except Exception:
        LOG.info("tenant=%s using ICP payload (source unknown)", tenant_id)

    # 1) Refresh ICP candidates via graph (or resume from previous manifest)
    resume_run_id_env = os.getenv("RESUME_FROM_RUN_ID")
    candidates: list[int] = []
    if resume_run_id_env:
        try:
            resume_id = int(resume_run_id_env)
            with get_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT selected_ids FROM run_manifests WHERE run_id=%s AND tenant_id=%s",
                    (resume_id, tenant_id),
                )
                row = cur.fetchone()
                if row and row[0]:
                    candidates = list(row[0])
                    LOG.info("tenant=%s resuming from manifest run_id=%s candidates=%d", tenant_id, resume_id, len(candidates))
        except Exception as e:
            LOG.warning("tenant=%s resume manifest load failed: %s", tenant_id, e)
    if not candidates:
        with obs.stage_timer(run_id, tenant_id, "mv_refresh"):
            icp_state = await icp_refresh_agent.ainvoke(
                {
                    "rule_name": os.getenv("ICP_RULE_NAME", "default"),
                    "payload": icp_payload,
                }
            )
        candidates = icp_state.get("candidate_ids", [])

    # Fallback: derive industry codes and select by industry_code if none
    if not candidates and icp_payload.get("industries"):
        try:
            inds_norm = sorted({(s or "").strip().lower() for s in icp_payload["industries"] if isinstance(s, str) and s.strip()})
            codes = fetch_industry_codes_by_names(inds_norm)
            if codes:
                fb_ids = fetch_candidate_ids_by_industry_codes(codes)
                if fb_ids:
                    LOG.info("tenant=%s fallback industry-code candidates=%d", tenant_id, len(fb_ids))
                    candidates = fb_ids
        except Exception as _e:
            LOG.info("tenant=%s fallback selection skipped: %s", tenant_id, _e)
    processed = 0
    processed_ids: set[int] = set()
    total_candidates = len(candidates)
    batch_idx = 0
    selected_all: list[int] = []

    while True:
        remaining = [cid for cid in candidates if cid not in processed_ids]
        if not remaining:
            break
        # Enforce overall daily cap and vendor-specific caps (coarse: per-company proxy)
        effective_cap = DAILY_CAP
        if TAVILY_MAX_QUERIES > 0:
            effective_cap = min(effective_cap, TAVILY_MAX_QUERIES)
        if LUSHA_MAX_CONTACT_LOOKUPS > 0:
            # Approximate: assume ≤1 lookup per company
            effective_cap = min(effective_cap, LUSHA_MAX_CONTACT_LOOKUPS)
        if processed >= effective_cap:
            LOG.info("tenant=%s reached daily cap=%d", tenant_id, DAILY_CAP)
            break
        batch_limit = effective_cap - processed
        if BATCH_SIZE > 0:
            batch_limit = min(batch_limit, BATCH_SIZE)

        with obs.stage_timer(run_id, tenant_id, "select_targets"):
            targets = select_target_set(remaining, batch_limit)
        if not targets:
            LOG.info("tenant=%s no more targets after prioritization", tenant_id)
            break
        batch_idx += 1
        LOG.info(
            "tenant=%s batch=%d/%s candidates=%d targets=%d processed=%d",
            tenant_id,
            batch_idx,
            "?",
            total_candidates,
            len(targets),
            processed,
        )

        # 2) Enrichment pipeline for this batch
        with obs.stage_timer(run_id, tenant_id, "enrich", total_inc=len(targets)):
            await enrich_many(targets)

        # 3) Score + rationale + persist for this batch
        with obs.stage_timer(run_id, tenant_id, "score", total_inc=len(targets)):
            scoring_state = await lead_scoring_agent.ainvoke(
            {
                "candidate_ids": targets,
                "lead_features": [],
                "lead_scores": [],
                "icp_payload": icp_payload,
            }
            )
        LOG.info(
            "tenant=%s batch=%d scored=%d",
            tenant_id,
            batch_idx,
            len(scoring_state.get("lead_scores", [])),
        )

        # 3.5) Batched ZeroBounce verification (best-effort)
        try:
            if ZB_MAX_VERIFY != 0:  # 0 means unlimited; negative treated as unlimited
                from src.database import get_conn as _get_conn
                need_verify: list[str] = []
                with _get_conn() as conn, conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT e.email
                        FROM lead_emails e
                        WHERE e.company_id = ANY(%s)
                          AND (e.verification_status IS NULL OR e.verification_status = '' OR e.verification_status = 'unknown')
                        LIMIT %s
                        """,
                        (targets, max(ZB_MAX_VERIFY, 0) if ZB_MAX_VERIFY > 0 else 1000000),
                    )
                    need_verify = [r[0] for r in cur.fetchall() if r and r[0]]
                if need_verify:
                    from src.enrichment import verify_emails as _zb_verify
                    with obs.stage_timer(run_id, tenant_id, "verify_emails", total_inc=len(need_verify)):
                        # chunk sequentially to respect provider; verify_emails already throttles subtly
                        verified_total = 0
                        for i in range(0, len(need_verify), max(1, ZB_BATCH_SIZE)):
                            chunk = need_verify[i:i+ZB_BATCH_SIZE]
                            if not chunk:
                                break
                            try:
                                await asyncio.to_thread(_zb_verify, chunk)
                                verified_total += len(chunk)
                            except Exception:
                                LOG.warning("tenant=%s ZeroBounce batch failed (chunk size=%d)", tenant_id, len(chunk))
                            if ZB_MAX_VERIFY > 0 and verified_total >= ZB_MAX_VERIFY:
                                break
        except Exception:
            LOG.exception("tenant=%s batch=%d ZeroBounce verification step failed", tenant_id, batch_idx)

        # 4) Export to Odoo for this batch
        try:
            scores = {s["company_id"]: s for s in scoring_state.get("lead_scores", [])}
            features = {f["company_id"]: f for f in scoring_state.get("lead_features", [])}

            ids = list(scores.keys()) or targets
            exported = 0
            if ids:
                with get_conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            SELECT company_id, name, uen, industry_norm, employees_est, revenue_bucket,
                                   incorporation_year, website_domain
                            FROM companies WHERE company_id = ANY(%s)
                            """,
                            (ids,),
                        )
                        comp_rows = cur.fetchall()
                        comps = {
                            r[0]: {
                                "name": r[1],
                                "uen": r[2],
                                "industry_norm": r[3],
                                "employees_est": r[4],
                                "revenue_bucket": r[5],
                                "incorporation_year": r[6],
                                "website_domain": r[7],
                            }
                            for r in comp_rows
                        }
                        cur.execute(
                            "SELECT company_id, email FROM lead_emails WHERE company_id = ANY(%s)",
                            (ids,),
                        )
                        email_rows = cur.fetchall()
                        emails: dict[int, str] = {}
                        for cid, em in email_rows:
                            emails.setdefault(cid, em)

                try:
                    env_thr = os.getenv("SCORE_MIN_EXPORT", os.getenv("LEAD_THRESHOLD", "0.0"))
                    threshold = float(env_thr)
                except Exception:
                    threshold = 0.0

                with obs.stage_timer(run_id, tenant_id, "export_odoo", total_inc=len(ids)):
                    store = OdooStore(tenant_id=tenant_id)
                    for cid in ids:
                        comp = comps.get(cid)
                        if not comp:
                            continue
                        email = emails.get(cid)
                        s = scores.get(cid) or {}
                        try:
                            odoo_id = await store.upsert_company(
                                comp.get("name"),
                                comp.get("uen"),
                                industry_norm=comp.get("industry_norm"),
                                employees_est=comp.get("employees_est"),
                                revenue_bucket=comp.get("revenue_bucket"),
                                incorporation_year=comp.get("incorporation_year"),
                                website_domain=comp.get("website_domain"),
                            )
                            if email:
                                try:
                                    await store.add_contact(odoo_id, email)
                                except Exception:
                                    LOG.warning("odoo export: add_contact failed partner_id=%s", odoo_id)
                            try:
                                await store.merge_company_enrichment(odoo_id, {})
                            except Exception:
                                pass
                            sc = float(s.get("score", 0) or 0)
                            try:
                                await store.create_lead_if_high(
                                    odoo_id,
                                    comp.get("name") or "",
                                    sc,
                                    features.get(cid, {}),
                                    (s.get("rationale") or ""),
                                    email,
                                    threshold=threshold,
                                )
                            except Exception:
                                LOG.warning("odoo export: create_lead failed partner_id=%s", odoo_id)
                            exported += 1
                        except Exception:
                            LOG.exception("Odoo export failed for company_id=%s", cid)
                LOG.info("tenant=%s batch=%d odoo_exported=%d", tenant_id, batch_idx, exported)
            else:
                LOG.info("tenant=%s batch=%d no ids to export to Odoo", tenant_id, batch_idx)
        except Exception:
            LOG.exception("tenant=%s batch=%d Odoo export step failed", tenant_id, batch_idx)

        processed += len(targets)
        processed_ids.update(targets)
        selected_all.extend(targets)
        if not AUTO_CONTINUE:
            break
        if PAUSE_SECONDS > 0:
            try:
                await asyncio.sleep(PAUSE_SECONDS)
            except Exception:
                pass

    LOG.info(
        "tenant=%s done batches=%d processed=%d/%d (cap=%d)",
        tenant_id,
        batch_idx,
        processed,
        total_candidates,
        DAILY_CAP,
    )
    LOG.info("tenant=%s run_duration_s=%.2f", tenant_id, (_time.perf_counter()-_t_run_start))
    try:
        obs.persist_manifest(run_id, tenant_id, selected_all)
        obs.write_summary(run_id, tenant_id, candidates=total_candidates, processed=processed, batches=batch_idx)
        obs.finalize_run(run_id, status="succeeded")
    except Exception:
        pass


async def run_all():
    tenants = await asyncio.to_thread(list_active_tenants)
    include = set(
        int(x) for x in (os.getenv("SCHED_TENANT_INCLUDE", "").split(",")) if x.strip().isdigit()
    )
    exclude = set(
        int(x) for x in (os.getenv("SCHED_TENANT_EXCLUDE", "").split(",")) if x.strip().isdigit()
    )
    if include:
        tenants = [t for t in tenants if t in include]
    if exclude:
        tenants = [t for t in tenants if t not in exclude]

    sem = asyncio.Semaphore(TENANT_CONC)

    async def _one(tid: int):
        async with sem:
            await run_tenant(tid)

    await asyncio.gather(*[_one(t) for t in tenants])


async def run_tenant_partial(tenant_id: int, max_now: int = 10) -> int:
    """Run a single, prioritized batch up to `max_now` companies for the tenant.

    Leaves the remainder to the nightly scheduler (no auto-continue).
    Returns the number of companies processed in this ad-hoc batch.
    """
    os.environ["DEFAULT_TENANT_ID"] = str(tenant_id)
    try:
        limit = int(max_now)
    except Exception:
        limit = 10
    if limit <= 0:
        limit = 1

    LOG.info("tenant=%s partial-run start limit=%d", tenant_id, limit)

    # Optional normalization first (mirrors run_tenant)
    try:
        run_norm = (os.getenv("RUN_NORMALIZATION_FIRST", "true").strip().lower() in ("1", "true", "yes", "on"))
        if run_norm:
            await normalize_agent.ainvoke({"raw_records": [], "normalized_records": []})
    except Exception as _e:
        LOG.info("tenant=%s normalization skipped: %s", tenant_id, _e)

    # Build ICP payload same as run_tenant
    def _payload_from_env() -> dict:
        inds_env = (os.getenv("ICP_INDUSTRIES", "") or "").strip()
        industries = [s.strip() for s in inds_env.split(",") if s.strip()] or ["Technology"]
        def _int(name: str, default: int) -> int:
            try:
                return int(os.getenv(name, str(default)))
            except Exception:
                return default
        return {
            "industries": industries,
            "employee_range": {"min": _int("ICP_EMPLOYEES_MIN", 2), "max": _int("ICP_EMPLOYEES_MAX", 100)},
            "incorporation_year": {"min": _int("ICP_YEAR_MIN", 2000), "max": _int("ICP_YEAR_MAX", 2025)},
        }

    # Prefer DB-stored ICP per tenant; fallback to env payload
    rule_name = os.getenv("ICP_RULE_NAME", "default")
    icp_payload = _load_icp_payload_from_db(tenant_id, rule_name) or _payload_from_env()
    try:
        src = "db(icp_rules)" if icp_payload is not None and _load_icp_payload_from_db(tenant_id, rule_name) else "env"
        LOG.info("tenant=%s partial-run ICP source=%s payload=%s", tenant_id, src, icp_payload)
    except Exception:
        LOG.info("tenant=%s partial-run using ICP payload (source unknown)", tenant_id)

    # Refresh ICP candidates
    icp_state = await icp_refresh_agent.ainvoke(
        {
            "rule_name": os.getenv("ICP_RULE_NAME", "default"),
            "payload": icp_payload,
        }
    )
    candidates = icp_state.get("candidate_ids", [])

    # Fallback by industry codes if none
    if not candidates and icp_payload.get("industries"):
        try:
            inds_norm = sorted({(s or "").strip().lower() for s in icp_payload["industries"] if isinstance(s, str) and s.strip()})
            codes = fetch_industry_codes_by_names(inds_norm)
            if codes:
                fb_ids = fetch_candidate_ids_by_industry_codes(codes)
                if fb_ids:
                    LOG.info("tenant=%s fallback industry-code candidates=%d", tenant_id, len(fb_ids))
                    candidates = fb_ids
        except Exception as _e:
            LOG.info("tenant=%s fallback selection skipped: %s", tenant_id, _e)

    total = len(candidates)
    if not candidates:
        LOG.info("tenant=%s partial-run no candidates found", tenant_id)
        return 0

    # Prioritize and cap to `limit`
    targets = select_target_set(candidates, min(limit, total))
    if not targets:
        LOG.info("tenant=%s partial-run no prioritized targets", tenant_id)
        return 0

    LOG.info("tenant=%s partial-run targets=%d total_candidates=%d", tenant_id, len(targets), total)

    # Enrich
    await enrich_many(targets)

    # Score + rationale + persist
    scoring_state = await lead_scoring_agent.ainvoke(
        {
            "candidate_ids": targets,
            "lead_features": [],
            "lead_scores": [],
            "icp_payload": icp_payload,
        }
    )
    LOG.info("tenant=%s partial-run scored=%d", tenant_id, len(scoring_state.get("lead_scores", [])))

    # Export to Odoo (same logic as run_tenant)
    try:
        scores = {s["company_id"]: s for s in scoring_state.get("lead_scores", [])}
        features = {f["company_id"]: f for f in scoring_state.get("lead_features", [])}

        ids = list(scores.keys()) or targets
        exported = 0
        if ids:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT company_id, name, uen, industry_norm, employees_est, revenue_bucket,
                               incorporation_year, website_domain
                        FROM companies WHERE company_id = ANY(%s)
                        """,
                        (ids,),
                    )
                    comp_rows = cur.fetchall()
                    comps = {
                        r[0]: {
                            "name": r[1],
                            "uen": r[2],
                            "industry_norm": r[3],
                            "employees_est": r[4],
                            "revenue_bucket": r[5],
                            "incorporation_year": r[6],
                            "website_domain": r[7],
                        }
                        for r in comp_rows
                    }
                    cur.execute(
                        "SELECT company_id, email FROM lead_emails WHERE company_id = ANY(%s)",
                        (ids,),
                    )
                    email_rows = cur.fetchall()
                    emails: dict[int, str] = {}
                    for cid, em in email_rows:
                        emails.setdefault(cid, em)

            try:
                env_thr = os.getenv("SCORE_MIN_EXPORT", os.getenv("LEAD_THRESHOLD", "0.0"))
                threshold = float(env_thr)
            except Exception:
                threshold = 0.0

            store = OdooStore(tenant_id=tenant_id)
            for cid in ids:
                comp = comps.get(cid)
                if not comp:
                    continue
                email = emails.get(cid)
                s = scores.get(cid) or {}
                try:
                    odoo_id = await store.upsert_company(
                        comp.get("name"),
                        comp.get("uen"),
                        industry_norm=comp.get("industry_norm"),
                        employees_est=comp.get("employees_est"),
                        revenue_bucket=comp.get("revenue_bucket"),
                        incorporation_year=comp.get("incorporation_year"),
                        website_domain=comp.get("website_domain"),
                    )
                    if email:
                        try:
                            await store.add_contact(odoo_id, email)
                        except Exception:
                            LOG.warning("odoo export: add_contact failed partner_id=%s", odoo_id)
                    try:
                        await store.merge_company_enrichment(odoo_id, {})
                    except Exception:
                        pass
                    sc = float(s.get("score", 0) or 0)
                    try:
                        await store.create_lead_if_high(
                            odoo_id,
                            comp.get("name") or "",
                            sc,
                            features.get(cid, {}),
                            (s.get("rationale") or ""),
                            email,
                            threshold=threshold,
                        )
                    except Exception:
                        LOG.warning("odoo export: create_lead failed partner_id=%s", odoo_id)
                    exported += 1
                except Exception:
                    LOG.exception("Odoo export failed for company_id=%s", cid)
            LOG.info(
                "tenant=%s partial-run exported=%d remainder=%d",
                tenant_id,
                exported,
                max(0, total - len(targets)),
            )
        else:
            LOG.info("tenant=%s partial-run no ids to export", tenant_id)
    except Exception:
        LOG.exception("tenant=%s partial-run Odoo export step failed", tenant_id)

    LOG.info("tenant=%s partial-run done processed=%d/%d", tenant_id, len(targets), total)
    return len(targets)


if __name__ == "__main__":
    try:
        asyncio.run(run_all())
    except KeyboardInterrupt:
        pass
