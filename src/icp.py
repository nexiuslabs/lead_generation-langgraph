# icp.py
import logging
from typing import Any, Dict, List, TypedDict, Optional

from langgraph.graph import StateGraph, END
from src.database import get_conn

log = logging.getLogger(__name__)

# ---------- State types ----------

class NormState(TypedDict, total=False):
    raw_records: List[Dict[str, Any]]
    normalized_records: List[Dict[str, Any]]  # what we upserted


class ICPState(TypedDict, total=False):
    rule_name: str
    payload: Dict[str, Any]
    candidate_ids: List[int]


# ---------- Helpers ----------

def _fetch_staging_rows(limit: int = 100) -> List[Dict[str, Any]]:
    """Fetch raw rows from a staging table; fall back if table is absent."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            try:
                # Prefer your staging table if it exists
                cur.execute(
                    """
                    SELECT
                        uen,
                        entity_name,
                        primary_ssic_description,
                        primary_ssic_code,
                        website,
                        incorporation_year,
                        entity_status_de
                    FROM staging_acra_companies
                    ORDER BY uen
                    LIMIT %s
                    """,
                    (limit,),
                )
            except Exception:
                # Fallback: use companies as a source of 'raw' rows
                cur.execute(
                    """
                    SELECT
                        company_id,
                        uen,
                        entity_name,
                        primary_ssic_description,
                        primary_ssic_code,
                        website,
                        incorporation_year,
                        sg_registered
                    FROM companies
                    ORDER BY uen
                    LIMIT %s
                    """,
                    (limit,),
                )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]


def _normalize_row(r: Dict[str, Any]) -> Dict[str, Any]:
    """Minimal normalization pass."""
    def _norm_str(x: Optional[str]) -> Optional[str]:
        if x is None:
            return None
        s = str(x).strip()
        return s or None

    norm = {
        "company_id": r.get("company_id"),
        "uen": _norm_str(r.get("uen")),
        "name": _norm_str(r.get("name")),
        "industry_norm": _norm_str(r.get("industry_norm")).lower() if r.get("industry_norm") else None,
        "industry_code": _norm_str(str(r.get("industry_code")) if r.get("industry_code") is not None else None),
        "website_domain": _norm_str(r.get("website_domain")),
        "incorporation_year": r.get("incorporation_year"),
        "sg_registered": r.get("sg_registered"),
    }
    return norm


def _upsert_companies_batch(rows: List[Dict[str, Any]]) -> int:
    """Upsert normalized rows into companies table."""
    if not rows:
        return 0
    with get_conn() as conn:
        with conn.cursor() as cur:
            for r in rows:
                cur.execute(
                    """
                    INSERT INTO companies (
                        company_id, uen, name, industry_norm, industry_code,
                        website_domain, incorporation_year, sg_registered, last_seen
                    )
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s, NOW())
                    ON CONFLICT (company_id) DO UPDATE SET
                        uen = EXCLUDED.uen,
                        name = EXCLUDED.name,
                        industry_norm = EXCLUDED.industry_norm,
                        industry_code = EXCLUDED.industry_code,
                        website_domain = EXCLUDED.website_domain,
                        incorporation_year = EXCLUDED.incorporation_year,
                        sg_registered = EXCLUDED.sg_registered,
                        last_seen = NOW()
                    """,
                    (
                        r.get("company_id"),
                        r.get("uen"),
                        r.get("name"),
                        r.get("industry_norm"),
                        r.get("industry_code"),
                        r.get("website_domain"),
                        r.get("incorporation_year"),
                        r.get("sg_registered"),
                    ),
                )
        conn.commit()
    return len(rows)


def _select_icp_candidates(payload: Dict[str, Any]) -> List[int]:
    """Build a simple WHERE from payload and fetch matching company_ids."""
    industries = [s.strip().lower() for s in payload.get("industries", []) if isinstance(s, str) and s.strip()]
    emp = payload.get("employee_range", {}) or {}
    inc = payload.get("incorporation_year", {}) or {}

    where = ["TRUE"]
    params: List[Any] = []

    if industries:
        where.append("LOWER(industry_norm) = ANY(%s)")
        params.append(industries)
    if "min" in emp:
        where.append("(employees_est IS NOT NULL AND employees_est >= %s)")
        params.append(emp["min"])
    if "max" in emp:
        where.append("(employees_est IS NOT NULL AND employees_est <= %s)")
        params.append(emp["max"])
    if "min" in inc:
        where.append("(incorporation_year IS NOT NULL AND incorporation_year >= %s)")
        params.append(inc["min"])
    if "max" in inc:
        where.append("(incorporation_year IS NOT NULL AND incorporation_year <= %s)")
        params.append(inc["max"])

    sql = f"""
        SELECT company_id
        FROM companies
        WHERE {' AND '.join(where)}
        ORDER BY company_id
        LIMIT 1000
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return [row[0] for row in cur.fetchall()]


# ---------- LangGraph nodes ----------

async def fetch_raw_records(state: NormState) -> NormState:
    rows = _fetch_staging_rows(limit=100)
    state["raw_records"] = rows
    log.info("Fetched %d staging rows", len(rows))
    return state


async def normalize_and_upsert(state: NormState) -> NormState:
    raw = state.get("raw_records", []) or []
    normalized = [_normalize_row(r) for r in raw]
    count = _upsert_companies_batch(normalized)
    log.info("Upserted %d companies in batch", count)
    state["normalized_records"] = normalized
    return state


async def refresh_icp_candidates(state: ICPState) -> ICPState:
    payload = state.get("payload", {}) or {}
    ids = _select_icp_candidates(payload)
    state["candidate_ids"] = ids
    return state


# ---------- Graphs ----------

# Normalization agent
_norm_graph = StateGraph(NormState)
_norm_graph.add_node("fetch_raw_records", fetch_raw_records)
_norm_graph.add_node("normalize_and_upsert", normalize_and_upsert)
_norm_graph.set_entry_point("fetch_raw_records")
_norm_graph.add_edge("fetch_raw_records", "normalize_and_upsert")
_norm_graph.add_edge("normalize_and_upsert", END)
normalize_agent = _norm_graph.compile()

# ICP refresh agent
_icp_graph = StateGraph(ICPState)
_icp_graph.add_node("refresh", refresh_icp_candidates)
_icp_graph.set_entry_point("refresh")
_icp_graph.add_edge("refresh", END)
icp_refresh_agent = _icp_graph.compile()

__all__ = ["normalize_agent", "icp_refresh_agent"]
