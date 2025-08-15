# ---------- src/langgraph_agents.py ----------
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph
from database import get_pg_pool
from settings import ICP_RULE_NAME

# --- Agent 1: Normalize staging_acra_companies into canonical companies table ---
class NormalizeState(TypedDict):
    raw_records: List[Dict[str, Any]]
    normalized_records: List[Dict[str, Any]]

async def fetch_raw_records(state: NormalizeState) -> NormalizeState:
    # Acquire pool and connection for fetching raw staging data
    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        # set search path to ensure public schema
        await conn.execute("SET search_path TO public, pg_catalog;")
        # select from fully-qualified table
        rows = await conn.fetch("SELECT * FROM public.staging_acra_companies;")
        # Convert to plain dicts so downstream code can use .get safely
        rows = [dict(r) for r in rows]
    state['raw_records'] = rows
    return state

async def normalize_records(state: NormalizeState) -> NormalizeState:
    normalized: List[Dict[str, Any]] = []
    for row in state['raw_records']:
        uen = row['uen']
        name = row['entity_name']
        uen_issue_date = row.get('uen_issue_date')
        if uen_issue_date:
            try:
                founded_year = int(str(uen_issue_date).split('-')[0])
            except:
                founded_year = None
        else:
            founded_year = None
        ownership_type = row.get('entity_type_description')
        industry_norm = (row.get('primary_ssic_description') or '').lower()
        employees_est = row.get('no_of_officers') or 0
        # revenue bucket categorization
        if employees_est < 10:
            revenue_bucket = 'small'
        elif employees_est <= 200:
            revenue_bucket = 'medium'
        else:
            revenue_bucket = 'large'
        # parse incorporation year
        inc_year = None
        inc_date = row.get('registration_incorporation_date')
        if inc_date:
            try:
                inc_year = int(str(inc_date).split('-')[0])
            except:
                inc_year = None
        sg_registered = (row.get('entity_status_description') == 'Live')
        # Ensure industry_code is text for DB insert
        ssic_code = row.get('primary_ssic_code')
        if ssic_code is not None:
            ssic_code = str(ssic_code)
        normalized.append({
            'uen': uen,
            'name': name,
            'founded_year': founded_year,
            'ownership_type': ownership_type,
            'industry_norm': industry_norm,
            'industry_code': ssic_code,
            'employees_est': employees_est,
            'revenue_bucket': revenue_bucket,
            'incorporation_year': inc_year,
            'sg_registered': sg_registered
        })
    state['normalized_records'] = normalized
    return state

async def upsert_companies(state: NormalizeState) -> NormalizeState:
    # Batch upsert normalized records into companies table for performance
    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        upsert_sql = '''
        INSERT INTO companies
          (uen, name, founded_year, ownership_type, industry_norm, industry_code, employees_est, revenue_bucket, incorporation_year, sg_registered)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        ON CONFLICT (uen) DO UPDATE SET
          name = EXCLUDED.name,
          founded_year = EXCLUDED.founded_year,
          ownership_type = EXCLUDED.ownership_type,
          industry_norm = EXCLUDED.industry_norm,
          industry_code = EXCLUDED.industry_code,
          employees_est = EXCLUDED.employees_est,
          revenue_bucket = EXCLUDED.revenue_bucket,
          incorporation_year = EXCLUDED.incorporation_year,
          sg_registered = EXCLUDED.sg_registered,
          last_seen = now();
        '''
        # Prepare batch arguments
        args = [(
            comp['uen'],
            comp['name'],
            comp['founded_year'],
            comp['ownership_type'],
            comp['industry_norm'],
            comp['industry_code'],
            comp['employees_est'],
            comp['revenue_bucket'],
            comp['incorporation_year'],
            comp['sg_registered']
        ) for comp in state['normalized_records']]
        # Execute as a batch
        await conn.executemany(upsert_sql, args)
        print(f"Upserted {len(args)} companies in batch")
    return state

# Build the graph for normalization
normalize_graph = StateGraph(NormalizeState)
normalize_graph.add_node('fetch_raw_records', fetch_raw_records)
normalize_graph.add_node('normalize_records', normalize_records)
normalize_graph.add_node('upsert_companies', upsert_companies)
normalize_graph.set_entry_point('fetch_raw_records')
normalize_graph.add_edge('fetch_raw_records', 'normalize_records')
normalize_graph.add_edge('normalize_records', 'upsert_companies')
normalize_agent = normalize_graph.compile()
try:
    normalize_agent.get_graph().draw_mermaid_png()
except Exception as e:
    print(f"normalize graph diagram generation skipped: {e}")


# --- Agent 2: Refresh ICP rules & candidate view ---
class ICPRefreshState(TypedDict):
    rule_name: str
    payload: Dict[str, Any]
    candidate_ids: List[int]

import json

async def refresh_icp_rules(state: ICPRefreshState) -> ICPRefreshState:
    """
    Ensure the ICP rule exists or update its payload.
    Serializes the payload dict to JSON string for JSONB insertion.
    """
    pool = await get_pg_pool()
    payload_json = json.dumps(state['payload'])
    async with pool.acquire() as conn:
        await conn.execute(
            '''
            INSERT INTO icp_rules (name, payload)
            VALUES ($1, $2::jsonb)
            ON CONFLICT (name) DO UPDATE SET payload = EXCLUDED.payload;
            ''',
            state['rule_name'], payload_json
        )
    return state

async def refresh_candidate_view(state: ICPRefreshState) -> ICPRefreshState:
    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        await conn.execute('REFRESH MATERIALIZED VIEW icp_candidate_companies;')
    return state

async def fetch_candidate_ids(state: ICPRefreshState) -> ICPRefreshState:
    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch('SELECT company_id FROM icp_candidate_companies;')
    state['candidate_ids'] = [r['company_id'] for r in rows]
    return state

# Build the graph for ICP refresh
icp_graph = StateGraph(ICPRefreshState)
icp_graph.add_node('refresh_icp_rules', refresh_icp_rules)
icp_graph.add_node('refresh_candidate_view', refresh_candidate_view)
icp_graph.add_node('fetch_candidate_ids', fetch_candidate_ids)
icp_graph.set_entry_point('refresh_icp_rules')
icp_graph.add_edge('refresh_icp_rules', 'refresh_candidate_view')
icp_graph.add_edge('refresh_candidate_view', 'fetch_candidate_ids')
icp_refresh_agent = icp_graph.compile()
#ICP FLOW
try:
    icp_refresh_agent.get_graph().draw_mermaid_png()
except Exception as e:
    print(f"ICP graph diagram generation skipped: {e}")

