import os
import asyncio
import logging
from typing import List

from src.icp import icp_refresh_agent
from src.enrichment import enrich_company_with_tavily
from src.lead_scoring import lead_scoring_agent
from src.database import get_conn
from app.odoo_store import OdooStore

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

    # 1) Refresh ICP candidates via graph
    icp_state = await icp_refresh_agent.ainvoke(
        {
            "rule_name": os.getenv("ICP_RULE_NAME", "default"),
            "payload": {
                "industries": ["Technology"],
                "employee_range": {"min": 2, "max": 100},
                "incorporation_year": {"min": 2000, "max": 2025},
            },
        }
    )
    candidates = icp_state.get("candidate_ids", [])
    targets = select_target_set(candidates, DAILY_CAP)
    LOG.info("tenant=%s candidates=%d targets=%d", tenant_id, len(candidates), len(targets))

    # 2) Enrichment pipeline
    await enrich_many(targets)

    # 3) Score + rationale + persist
    scoring_state = await lead_scoring_agent.ainvoke(
        {
            "candidate_ids": targets,
            "lead_features": [],
            "lead_scores": [],
            "icp_payload": {
                "industries": ["Technology"],
                "employee_range": {"min": 2, "max": 100},
                "incorporation_year": {"min": 2000, "max": 2025},
            },
        }
    )
    LOG.info(
        "tenant=%s scored=%d",
        tenant_id,
        len(scoring_state.get("lead_scores", [])),
    )

    # 4) Export to Odoo (idempotent upserts + leads over threshold)
    try:
        # Build lookups from scoring_state (company_id -> score/feature)
        scores = {s["company_id"]: s for s in scoring_state.get("lead_scores", [])}
        features = {f["company_id"]: f for f in scoring_state.get("lead_features", [])}

        ids = list(scores.keys()) or targets
        exported = 0
        if ids:
            # Fetch company core info and primary emails (sync DB access)
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

            # Threshold (env) — prefer SCORE_MIN_EXPORT, fallback LEAD_THRESHOLD
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
                except Exception as exc:
                    LOG.exception("Odoo export failed for company_id=%s", cid)
            LOG.info("tenant=%s odoo_exported=%d", tenant_id, exported)
        else:
            LOG.info("tenant=%s no ids to export to Odoo", tenant_id)
    except Exception:
        LOG.exception("tenant=%s Odoo export step failed", tenant_id)


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


if __name__ == "__main__":
    try:
        asyncio.run(run_all())
    except KeyboardInterrupt:
        pass
