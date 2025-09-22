import asyncio
import os
from src.icp import normalize_agent, icp_refresh_agent, _find_ssic_codes_by_terms
from src.enrichment import _should_skip_enrichment
from src.settings import ICP_RULE_NAME
from src.openai_client import generate_rationale
from src.lead_scoring import lead_scoring_agent
from app.odoo_store import OdooStore  # type: ignore
from app.odoo_connection_info import get_odoo_connection_info  # type: ignore
import logging
import sys


from src.enrichment import enrich_company_with_tavily
import psycopg2
import json
from src.settings import POSTGRES_DSN

def fetch_companies(company_ids):
    conn = psycopg2.connect(dsn=POSTGRES_DSN)
    with conn, conn.cursor() as cur:
        cur.execute(
            "SELECT company_id, name FROM companies "
            "WHERE company_id = ANY(%s)",
            (company_ids,),
        )
        rows = cur.fetchall()
    conn.close()
    return rows


def fetch_candidate_ids_by_industry_codes(industry_codes):
    """Fetch company_ids whose industry_code matches any of the provided codes."""
    if not industry_codes:
        return []
    conn = psycopg2.connect(dsn=POSTGRES_DSN)
    with conn, conn.cursor() as cur:
        cur.execute(
            "SELECT company_id FROM companies WHERE industry_code = ANY(%s)",
            (industry_codes,)
        )
        rows = cur.fetchall()
    conn.close()
    return [r[0] for r in rows]

def fetch_industry_codes_by_names(industries):
    """Resolve SSIC codes from free-text industry names via ssic_ref; fallback to companies.

    - Primary: use `ssic_ref` FTS/trigram (via `_find_ssic_codes_by_terms`).
    - Fallback: if none found, check `companies` by `industry_norm` to collect `industry_code`.
    """
    if not industries:
        return []
    normed = sorted({(s or '').strip().lower() for s in industries if isinstance(s, str) and s.strip()})
    if not normed:
        return []
    codes = {c for (c, _title, _score) in _find_ssic_codes_by_terms(normed)}
    if codes:
        return sorted(codes)
    # Fallback to companies table
    conn = psycopg2.connect(dsn=POSTGRES_DSN)
    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT industry_code
                FROM companies
                WHERE industry_norm = ANY(%s)
                  AND industry_code IS NOT NULL
                """,
                (normed,)
            )
            rows = cur.fetchall()
            for (code,) in rows:
                if code:
                    codes.add(str(code))
        return sorted(codes)
    finally:
        conn.close()

async def enrich_companies(company_ids):
    companies = fetch_companies(company_ids)
    print(f"▶️  Starting enrichment for {len(companies)} companies...")

    for idx, (company_id, name) in enumerate(companies, start=1):
        print(f"\n--- ({idx}/{len(companies)}) id={company_id}, name={name!r} ---")
        await enrich_company_with_tavily(company_id, name)

def output_candidate_records(company_ids):
    conn = psycopg2.connect(dsn=POSTGRES_DSN)
    with conn, conn.cursor() as cur:
        cur.execute(
            "SELECT * FROM companies WHERE company_id = ANY(%s)",
            (company_ids,)
        )
        columns = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
    conn.close()
    records = [dict(zip(columns, row)) for row in rows]
    print("candidate records JSON:")
    print(json.dumps(records, indent=2, default=str))

async def main():
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)
    # Suppress OpenAI and LangChain HTTP request logs at INFO level
    logging.getLogger('openai').setLevel(logging.ERROR)
    logging.getLogger('langchain').setLevel(logging.ERROR)
    logging.getLogger('langchain_openai').setLevel(logging.ERROR)
    # Normalize step
    norm_initial_state = { 'raw_records': [], 'normalized_records': [] }
    # Use the Runnable interface (async invoke) to run the normalization graph
    norm_result_state = await normalize_agent.ainvoke(norm_initial_state)

    # ICP refresh step
    # Industries: read from env ICP_INDUSTRIES (comma-separated). Default to single 'Technology'.
    inds_env = os.getenv("ICP_INDUSTRIES", "").strip()
    industries = [s.strip() for s in inds_env.split(",") if s.strip()] or ["Technology"]
    icp_payload = {
        "industries":      industries,
        "employee_range":  { "min": 2,  "max": 100 },
        "incorporation_year": {"min": 2000, "max": 2025}
    }
    logging.info(f"ICP criteria: {icp_payload}")
    icp_initial_state = { 'rule_name': ICP_RULE_NAME, 'payload': icp_payload, 'candidate_ids': [] }
    # Run the ICP refresh graph asynchronously
    logging.info("Refreshing ICP candidate view: 'icp_candidate_companies'")
    icp_result_state = await icp_refresh_agent.ainvoke(icp_initial_state)
    logging.info(f" ✅ Matched ICP candidate IDs: {icp_result_state['candidate_ids']} (count={len(icp_result_state['candidate_ids'])})")
    if icp_result_state['candidate_ids']:
        logging.info(" ✅ ICP rule matched candidates")
    else:
        logging.info(" ✅ ICP rule matched no candidates")

    # Fallback: derive industry codes from industries and fetch by industry_code ONLY
    candidate_ids = icp_result_state['candidate_ids']
    if not candidate_ids and icp_payload.get('industries'):
        industries_norm = sorted({(s or '').strip().lower() for s in icp_payload['industries'] if isinstance(s, str) and s.strip()})
        codes = fetch_industry_codes_by_names(industries_norm)
        logging.info(f"Derived industry codes from industries: {codes}")
        if codes:
            fallback_ids = fetch_candidate_ids_by_industry_codes(codes)
            logging.info(f"🔥 Fallback industry-code match IDs: {fallback_ids}")
        else:
            fallback_ids = []
            logging.info("No industry codes derived; no fallback candidates found via codes")
        candidate_ids = fallback_ids
    else:
        candidate_ids = icp_result_state['candidate_ids']

    # Output candidate IDs
    logging.info(f"Today's candidate IDs: {candidate_ids}")
    logging.info(f"Fetched {len(norm_result_state['raw_records'])} staging rows")
    logging.info(f"Normalized to {len(norm_result_state['normalized_records'])} companies")
   #print("Batch upsert complete") 

    # Demo: generate an LLM rationale for the first candidate
    if icp_result_state['candidate_ids']:
        prompt = f"Explain fit for company_id {icp_result_state['candidate_ids'][0]} based on features."
        rationale = await generate_rationale(prompt)
        #logging.info('LLM Rationale:', rationale)

    # Enrich ICP candidates
    await enrich_companies(candidate_ids)
    # Output enriched records JSON
    output_candidate_records(candidate_ids)
    # Execute lead scoring pipeline
    logging.info("\n\n▶️ Lead scoring pipeline:\n")
    scoring_initial_state = {'candidate_ids': candidate_ids, 'lead_features': [], 'lead_scores': [], 'icp_payload': icp_payload}
    scoring_state = await lead_scoring_agent.ainvoke(scoring_initial_state)
    logging.info("\n\n ✅ Lead scoring results:\n")
    logging.info(json.dumps(scoring_state['lead_scores'], indent=2, default=str))

    # --- Export to Odoo: upsert companies + contacts; create leads for high scores ---
    try:
        # Resolve tenant (prefer app DB mapping; fallback to ODOO_POSTGRES_DSN)
        tid = None
        try:
            info = await get_odoo_connection_info(email="exporter@nexiuslabs.local", claim_tid=None)
            tid = info.get("tenant_id")
        except Exception:
            tid = None
        if tid is None:
            # Fallback: pick the first active mapping
            try:
                conn_map = psycopg2.connect(dsn=POSTGRES_DSN)
                with conn_map, conn_map.cursor() as curm:
                    curm.execute("SELECT tenant_id FROM odoo_connections WHERE active=TRUE LIMIT 1")
                    rowm = curm.fetchone()
                    if rowm:
                        tid = int(rowm[0])
            except Exception:
                tid = None
            finally:
                try:
                    conn_map.close()
                except Exception:
                    pass
        store = OdooStore(tenant_id=int(tid)) if tid is not None else OdooStore()

        # Build lookups from scoring_state
        scores = {s["company_id"]: s for s in scoring_state.get("lead_scores", [])}
        features = {f["company_id"]: f for f in scoring_state.get("lead_features", [])}

        ids = candidate_ids
        if not ids:
            ids = [s["company_id"] for s in scoring_state.get("lead_scores", []) if s.get("company_id") is not None]

        # Skip Odoo export for companies where enrichment was skipped
        try:
            ids = [cid for cid in ids if not _should_skip_enrichment(int(cid))]
        except Exception:
            # Best-effort: if guard fails, proceed with original ids
            ids = ids

        if ids:
            # Fetch company core info and primary emails
            conn = psycopg2.connect(dsn=POSTGRES_DSN)
            try:
                with conn, conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT company_id, name, uen, industry_norm, employees_est, revenue_bucket,
                               incorporation_year, website_domain
                        FROM companies WHERE company_id = ANY(%s)
                        """,
                        (ids,),
                    )
                    comp_rows = cur.fetchall()
                    comps = {r[0]: {
                        "name": r[1],
                        "uen": r[2],
                        "industry_norm": r[3],
                        "employees_est": r[4],
                        "revenue_bucket": r[5],
                        "incorporation_year": r[6],
                        "website_domain": r[7],
                    } for r in comp_rows}
                    cur.execute("SELECT company_id, email FROM lead_emails WHERE company_id = ANY(%s)", (ids,))
                    email_rows = cur.fetchall()
                    emails = {}
                    for cid, em in email_rows:
                        emails.setdefault(cid, em)
            finally:
                conn.close()

            # Threshold (optional) from env — align with PRD7 SCORE_MIN_EXPORT, fallback to LEAD_THRESHOLD
            try:
                env_thr = os.getenv("SCORE_MIN_EXPORT", os.getenv("LEAD_THRESHOLD", "0.0"))
                threshold = float(env_thr)
            except Exception:
                threshold = 0.0

            exported = 0
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
                        await store.add_contact(odoo_id, email)
                    # Merge any enrichment features if available
                    await store.merge_company_enrichment(odoo_id, {})
                    sc = float(s.get("score", 0) or 0)
                    await store.create_lead_if_high(
                        odoo_id,
                        comp.get("name") or "",
                        sc,
                        features.get(cid, {}),
                        (s.get("rationale") or ""),
                        email,
                        threshold=threshold,
                    )
                    exported += 1
                except Exception as exc:
                    logging.exception("Odoo export failed for company_id=%s", cid)
            logging.info("✅ Exported %d companies to Odoo", exported)
        else:
            logging.info("No candidate IDs to export to Odoo")
    except Exception:
        logging.exception("Odoo export step failed")

if __name__ == '__main__':
    asyncio.run(main())
