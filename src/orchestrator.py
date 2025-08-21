import asyncio
from src.icp import normalize_agent, icp_refresh_agent
from src.settings import ICP_RULE_NAME
from src.openai_client import generate_rationale
from src.lead_scoring import lead_scoring_agent
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
    """Resolve SSIC industry codes from human-readable industry names.
    First checks staging_acra_companies by description, then falls back to companies.industry_norm.
    Returns a sorted list of distinct code strings.
    """
    if not industries:
        return []
    # Normalize and dedupe inputs
    normed = sorted({(s or '').strip().lower() for s in industries if isinstance(s, str) and s.strip()})
    if not normed:
        return []
    codes = set()
    conn = psycopg2.connect(dsn=POSTGRES_DSN)
    try:
        # Prefer authoritative mapping from staging
        with conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT CAST(primary_ssic_code AS TEXT)
                FROM public.staging_acra_companies
                WHERE LOWER(primary_ssic_description) = ANY(%s)
                  AND primary_ssic_code IS NOT NULL
                """,
                (normed,)
            )
            rows = cur.fetchall()
            for (code,) in rows:
                if code:
                    codes.add(str(code))
        # Fallback: resolve via companies table if staging yields nothing
        if not codes:
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
    icp_payload = {

        "industries":      ["Accounting","Technology"],
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

if __name__ == '__main__':
    asyncio.run(main())
