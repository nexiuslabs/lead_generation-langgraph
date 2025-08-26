from typing import TypedDict, List, Dict, Any
import hashlib
import json
from langgraph.graph import StateGraph
import os
from src.database import get_pg_pool
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
            "SELECT company_id, employees_est, revenue_bucket, sg_registered, incorporation_year FROM companies WHERE company_id = ANY($1)",
            state['candidate_ids']
        )
    features: List[Dict[str, Any]] = []
    for row in rows:
        features.append({
            'company_id': row['company_id'],
            'employees_est': row['employees_est'],
            'revenue_bucket': row['revenue_bucket'],
            'sg_registered': row['sg_registered'],
            'incorporation_year': row['incorporation_year']
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
    for feat, p in zip(state['lead_features'], probs):
        lead_scores.append({
            'company_id': feat['company_id'],
            'score': float(p)
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
            "provide a concise 2-sentence justification referencing the top signals."
        )
        rationale = await generate_rationale(prompt)
        score['rationale'] = rationale
        score['cache_key'] = cache_key
    return state

async def persist_results(state: LeadScoringState) -> LeadScoringState:
    """
    Persist lead_features and lead_scores into database tables.
    """
    pool = await get_pg_pool()
    async with pool.acquire() as conn:
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
        for feat, score in zip(state['lead_features'], state['lead_scores']):
            # Upsert features
            await conn.execute(
                """
                INSERT INTO lead_features (company_id, features)
                VALUES ($1, $2::jsonb)
                ON CONFLICT (company_id) DO UPDATE SET features = EXCLUDED.features;
                """,
                feat['company_id'], json.dumps(feat)
            )
            # Upsert scores
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
