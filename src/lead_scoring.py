from typing import TypedDict, List, Dict, Any
import hashlib
import json
from langgraph.graph import StateGraph
import os
from src.database import get_pg_pool
from src.settings import MISSING_FIRMO_PENALTY, FIRMO_MIN_COMPLETENESS_FOR_BONUS
from langchain_openai import ChatOpenAI
from sklearn.linear_model import LogisticRegression
from src.openai_client import generate_rationale

# Define state for lead scoring
class LeadScoringState(TypedDict):
    candidate_ids: List[int]
    lead_features: List[Dict[str, Any]]
    lead_scores: List[Dict[str, Any]]
    icp_payload: Dict[str, Any]  # optional ICP range criteria

async def fetch_features(state: LeadScoringState) -> LeadScoringState:
    """
    Fetch core features from companies table for given candidate IDs.
    """
    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT company_id, employees_est, revenue_bucket, sg_registered, incorporation_year, industry_code FROM companies WHERE company_id = ANY($1)",
            state['candidate_ids']
        )
        # ResearchOps evidence counts for manual bonus (DevPlan19)
        ev_counts = await conn.fetch(
            """
            SELECT company_id, COUNT(*) AS cnt
            FROM icp_evidence
            WHERE company_id = ANY($1) AND source IN ('research','web_preview')
            GROUP BY company_id
            """,
            state['candidate_ids']
        )
        by_ev = {r['company_id']: int(r['cnt']) for r in ev_counts}
    features: List[Dict[str, Any]] = []
    for row in rows:
        features.append({
            'company_id': row['company_id'],
            'employees_est': row['employees_est'],
            'revenue_bucket': row['revenue_bucket'],
            'sg_registered': row['sg_registered'],
            'incorporation_year': row['incorporation_year'],
            'research_ev_count': by_ev.get(row['company_id'], 0),
            'industry_code': row['industry_code'],
        })
    state['lead_features'] = features
    return state

async def train_and_score(state: LeadScoringState) -> LeadScoringState:
    """
    Train a logistic regression model on ICP data (balanced) and score candidates.
    """
    # Prepare feature matrix
    X: List[List[float]] = []
    for feat in state['lead_features']:
        # One-hot encode revenue_bucket: small, medium, large
        rb = feat['revenue_bucket']
        one_hot = [1 if rb == 'small' else 0,
                   1 if rb == 'medium' else 0,
                   1 if rb == 'large' else 0]
        X.append([feat['employees_est'], 1 if feat['sg_registered'] else 0] + one_hot)
    # Labels: treat all candidates as positive ICP signal
    y = [1] * len(X)
    # Handle single-class case by heuristic closeness to ICP criteria
    if len(set(y)) < 2:
        icp = state.get('icp_payload', {})
        probs = []
        for feat in state['lead_features']:
            scores = []
            # Employee range closeness
            er = icp.get('employee_range')
            val = feat.get('employees_est', 0)
            mn = er.get('min', 0) if er else None
            mx = er.get('max', mn) if er else None
            if er and None not in (val, mn, mx):
                width = mx - mn if mx > mn else max(mn, 1)
                if mn <= val <= mx:
                    scores.append(1.0)
                else:
                    dist = mn - val if val < mn else val - mx
                    scores.append(max(0.0, 1 - dist/width))
            # Incorporation year closeness
            iy = icp.get('incorporation_year')
            year = feat.get('incorporation_year') if feat.get('incorporation_year') is not None else 0
            mn = iy.get('min', year) if iy else None
            mx = iy.get('max', mn) if iy else None
            if iy and None not in (year, mn, mx):
                width = mx - mn if mx > mn else 1
                if mn <= year <= mx:
                    scores.append(1.0)
                else:
                    dist = mn - year if year < mn else year - mx
                    scores.append(max(0.0, 1 - dist/width))
            # Revenue bucket match
            rb = icp.get('revenue_bucket')
            if rb:
                scores.append(1.0 if feat.get('revenue_bucket') == rb else 0.0)
            # Compute average
            prob = sum(scores)/len(scores) if scores else 1.0
            probs.append(prob)
    else:
        # Train logistic regression with balanced classes
        model = LogisticRegression(class_weight='balanced', max_iter=1000)
        model.fit(X, y)
        # Predict probabilities
        probs = model.predict_proba(X)[:, 1]
    lead_scores: List[Dict[str, Any]] = []
    # DevPlan19: apply ManualResearch bonus (cap) and map to A/B/C buckets on 0-100 scale
    bonus_cap = 20
    try:
        bonus_cap = int(os.getenv("MANUAL_RESEARCH_BONUS_MAX", "20") or 20)
    except Exception:
        bonus_cap = 20
    # Optional LLM structured scoring ensemble
    use_llm_score = (os.getenv("ENABLE_LLM_STRUCTURED_SCORING", "").lower() in ("1","true","yes","on"))
    llm = ChatOpenAI(model=os.getenv("AGENT_MODEL_DISCOVERY", os.getenv("LANGCHAIN_MODEL", "gpt-4o-mini")), temperature=0) if use_llm_score else None

    def _llm_structured_score(_feat: Dict[str, Any], _icp: Dict[str, Any]) -> int:
        try:
            if not llm:
                return 0
            prompt = (
                "Given these company features and ICP preferences, return a single integer 0-100 representing fit. "
                "Company features: {feat}\nICP: {icp}\nOutput only the integer."
            )
            out = llm.invoke(prompt.format(feat=_feat, icp=_icp))
            txt = (getattr(out, 'content', None) or '').strip()
            import re as _re
            m = _re.search(r"\b(\d{1,3})\b", txt)
            if not m:
                return 0
            val = int(m.group(1))
            return max(0, min(100, val))
        except Exception:
            return 0

    for feat, p in zip(state['lead_features'], probs):
        base = max(0.0, min(1.0, float(p)))
        base100 = int(round(base * 100))
        # Optional ensemble: blend baseline with LLM score (30% weight)
        if use_llm_score:
            try:
                llm_score = _llm_structured_score(feat, state.get('icp_payload', {}))
                base100 = int(round(0.7 * base100 + 0.3 * llm_score))
            except Exception:
                pass
        ev_cnt = int(feat.get('research_ev_count') or 0)
        # Gate bonus by firmographics completeness (employees or industry present)
        firmo_present = 0
        if feat.get('employees_est') is not None:
            firmo_present += 1
        if feat.get('industry_code') is not None:
            firmo_present += 1
        if firmo_present >= max(1, int(FIRMO_MIN_COMPLETENESS_FOR_BONUS)):
            bonus = min(bonus_cap, ev_cnt * 5) if ev_cnt > 0 else 0
        else:
            bonus = min(5, ev_cnt * 5) if ev_cnt > 0 else 0
        final = max(0, min(100, base100 + bonus))
        # Apply penalty for missing firmographics and mark flag
        firmo_missing = (feat.get('industry_code') is None) or (feat.get('employees_est') is None)
        if firmo_missing:
            final = max(0, final - int(MISSING_FIRMO_PENALTY))
        # A/B/C thresholds per DevPlan19
        bucket = 'A' if final >= 70 else ('B' if final >= 50 else 'C')
        lead_scores.append({
            'company_id': feat['company_id'],
            'score': float(final),
            'bucket': bucket,
            'firmo_missing': bool(firmo_missing),
        })
    state['lead_scores'] = lead_scores
    return state

def assign_buckets(state: LeadScoringState) -> LeadScoringState:
    """
    Bucket scores into High, Medium, Low based on thresholds.
    """
    for s in state['lead_scores']:
        p = s['score']
        if p >= 0.66:
            bucket = 'high'
        elif p >= 0.33:
            bucket = 'medium'
        else:
            bucket = 'low'
        # Enforce demotion when firmographics missing: never allow 'high'
        try:
            if bucket == 'high' and bool(s.get('firmo_missing')):
                bucket = 'medium'
        except Exception:
            pass
        s['bucket'] = bucket
    return state

async def generate_rationales(state: LeadScoringState) -> LeadScoringState:
    """
    Generate concise 2-sentence rationale for each lead score and cache key.
    """
    for feat, score in zip(state['lead_features'], state['lead_scores']):
        # Create cache key based on sorted feature items
        items = sorted(feat.items())
        key_str = json.dumps(items, sort_keys=True)
        cache_key = hashlib.sha1(key_str.encode('utf-8')).hexdigest()
        prompt = (
            f"Given the features {feat} with score {score['score']:.2f}, "
            "provide a concise 2-sentence justification referencing the top signals and research evidence if present."
        )
        rationale = await generate_rationale(prompt)
        # Append demotion reason when firmographics missing
        try:
            if bool(score.get('firmo_missing')):
                rationale = (rationale or '')
                if rationale:
                    rationale += "\n"
                rationale += "demoted due to missing firmographics (industry/employees)"
        except Exception:
            pass
        score['rationale'] = rationale
        score['cache_key'] = cache_key
    return state

async def persist_results(state: LeadScoringState) -> LeadScoringState:
    """
    Persist lead_features and lead_scores into database tables.
    """
    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        # Try set tenant GUC for RLS from env (supports non-HTTP runs),
        # and also detect an existing tenant from the current session if present.
        try:
            import os as _os
            _tenant_env = _os.getenv('DEFAULT_TENANT_ID')
            if _tenant_env:
                await conn.execute("SELECT set_config('request.tenant_id', $1, true)", _tenant_env)
        except Exception:
            pass
        # Ensure tables exist
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS lead_features (
              company_id INT PRIMARY KEY,
              features JSONB
            );
            """
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS lead_scores (
              company_id INT PRIMARY KEY,
              score FLOAT,
              bucket TEXT,
              rationale TEXT,
              cache_key TEXT
            );
            """
        )
        # Detect tenant_id column presence
        has_tenant = bool(
            await conn.fetchval(
                "SELECT 1 FROM information_schema.columns WHERE table_name='lead_scores' AND column_name='tenant_id' LIMIT 1"
            )
        )
        tenant_val = None
        # Prefer an already-applied request.tenant_id if present; else env
        try:
            _current = await conn.fetchval("SELECT current_setting('request.tenant_id', true)")
            if _current and str(_current).strip().isdigit():
                tenant_val = str(_current).strip()
        except Exception:
            tenant_val = None
        if tenant_val is None:
            try:
                import os as _os
                tenant_val = _os.getenv('DEFAULT_TENANT_ID')
            except Exception:
                tenant_val = None
        for feat, score in zip(state['lead_features'], state['lead_scores']):
            # Upsert features
            if has_tenant and tenant_val is not None:
                await conn.execute(
                    """
                    INSERT INTO lead_features (company_id, features, tenant_id)
                    VALUES ($1, $2::jsonb, $3::int)
                    ON CONFLICT (company_id) DO UPDATE SET features = EXCLUDED.features, tenant_id = EXCLUDED.tenant_id;
                    """,
                    feat['company_id'], json.dumps(feat), int(tenant_val)
                )
            else:
                await conn.execute(
                    """
                    INSERT INTO lead_features (company_id, features)
                    VALUES ($1, $2::jsonb)
                    ON CONFLICT (company_id) DO UPDATE SET features = EXCLUDED.features;
                    """,
                    feat['company_id'], json.dumps(feat)
                )
            # Upsert scores
            if has_tenant and tenant_val is not None:
                await conn.execute(
                    """
                    INSERT INTO lead_scores (company_id, score, bucket, rationale, cache_key, tenant_id)
                    VALUES ($1, $2, $3, $4, $5, $6::int)
                    ON CONFLICT (company_id) DO UPDATE SET
                      score = EXCLUDED.score,
                      bucket = EXCLUDED.bucket,
                      rationale = EXCLUDED.rationale,
                      cache_key = EXCLUDED.cache_key,
                      tenant_id = EXCLUDED.tenant_id;
                    """,
                    score['company_id'], score['score'], score['bucket'], score['rationale'], score['cache_key'], int(tenant_val)
                )
            else:
                await conn.execute(
                    """
                    INSERT INTO lead_scores (company_id, score, bucket, rationale, cache_key)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (company_id) DO UPDATE SET
                      score = EXCLUDED.score,
                      bucket = EXCLUDED.bucket,
                      rationale = EXCLUDED.rationale,
                      cache_key = EXCLUDED.cache_key;
                    """,
                    score['company_id'], score['score'], score['bucket'], score['rationale'], score['cache_key']
                )
    return state

# Build and compile LangGraph pipeline
graph = StateGraph(LeadScoringState)
graph.add_node('fetch_features', fetch_features)
graph.add_node('train_and_score', train_and_score)
graph.add_node('assign_buckets', assign_buckets)
graph.add_node('generate_rationales', generate_rationales)
graph.add_node('persist_results', persist_results)
graph.set_entry_point('fetch_features')
graph.add_edge('fetch_features', 'train_and_score')
graph.add_edge('train_and_score', 'assign_buckets')
graph.add_edge('assign_buckets', 'generate_rationales')
graph.add_edge('generate_rationales', 'persist_results')

lead_scoring_agent = graph.compile()
# Only attempt network rendering of the graph if explicitly enabled
try:
    if os.getenv("DRAW_MERMAID", "").lower() in ("1", "true", "yes"):
        lead_scoring_agent.get_graph().draw_mermaid_png()
except Exception:
    # Safe to ignore rendering issues during runtime or tests without network
    pass
