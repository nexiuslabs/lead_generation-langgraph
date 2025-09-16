import os
import asyncio
import logging
from typing import List

from src.icp import icp_refresh_agent, normalize_agent
from src.enrichment import enrich_company_with_tavily
from src.lead_scoring import lead_scoring_agent
from src.database import get_conn
from app.odoo_store import OdooStore
from src.orchestrator import (
    fetch_industry_codes_by_names,
    fetch_candidate_ids_by_industry_codes,
)

LOG = logging.getLogger("nightly")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


TENANT_CONC = _int_env("SCHED_TENANT_CONCURRENCY", 3)
COMPANY_CONC = _int_env("SCHED_COMPANY_CONCURRENCY", 8)
DAILY_CAP = _int_env("SCHED_DAILY_CAP_PER_TENANT", _int_env("SHORTLIST_DAILY_CAP", 100))
BATCH_SIZE = _int_env("SCHED_COMPANY_BATCH_SIZE", 1)

def _bool_env(name: str, default: bool) -> bool:
    val = (os.getenv(name, "") or "").strip().lower()
    if val in ("1", "true", "yes", "on"): return True
    if val in ("0", "false", "no", "off"): return False
    return default

AUTO_CONTINUE = _bool_env("SCHED_AUTO_BATCH_CONTINUE", False)
PAUSE_SECONDS = _int_env("SCHED_BATCH_PAUSE_SECONDS", 0)


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

    icp_payload = _payload_from_env()

    # 1) Refresh ICP candidates via graph
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

    while True:
        remaining = [cid for cid in candidates if cid not in processed_ids]
        if not remaining:
            break
        if processed >= DAILY_CAP:
            LOG.info("tenant=%s reached daily cap=%d", tenant_id, DAILY_CAP)
            break
        batch_limit = DAILY_CAP - processed
        if BATCH_SIZE > 0:
            batch_limit = min(batch_limit, BATCH_SIZE)

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
        await enrich_many(targets)

        # 3) Score + rationale + persist for this batch
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

    icp_payload = _payload_from_env()

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
