# icp.py
import logging
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

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
    """Fetch raw rows from staging_acra_companies with best-effort column mapping.

    Falls back to companies if staging is unavailable.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            # Introspect staging columns
            cur.execute(
                """
                SELECT LOWER(column_name) FROM information_schema.columns
                WHERE table_name = 'staging_acra_companies'
                """
            )
            staging_cols = {r[0] for r in cur.fetchall()}
            if staging_cols:
                # Pick available column names
                def pick(*names: str) -> str | None:
                    for n in names:
                        if n.lower() in staging_cols:
                            return n
                    return None

                src_uen = pick("uen", "uen_no", "uen_number") or "NULL"
                src_name = pick("entity_name", "name", "company_name") or "NULL"
                src_desc = (
                    pick(
                        "primary_ssic_description",
                        "ssic_description",
                        "industry_description",
                    )
                    or "NULL"
                )
                src_code = pick("primary_ssic_code", "industry_code") or "NULL"
                src_year = (
                    pick(
                        "incorporation_year",
                        "founded_year",
                        "registration_incorporation_date",
                    )
                    or "NULL"
                )
                src_status = (
                    pick(
                        "entity_status_description",
                        "entity_status",
                        "status",
                        "entity_status_de",
                    )
                    or "NULL"
                )

                sql = f"""
                    SELECT
                      {src_uen}   AS uen,
                      {src_name}  AS entity_name,
                      {src_desc}  AS primary_ssic_description,
                      {src_code}  AS primary_ssic_code,
                      ref.code    AS ssic_code,
                      {src_year}  AS raw_year,
                      {src_status} AS entity_status_description
                    FROM staging_acra_companies sc
                    LEFT JOIN ssic_ref ref

                      ON CAST({src_code} AS TEXT) = ref.code
                    ORDER BY 1
                    LIMIT %s
                """
                cur.execute(sql, (limit,))
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, r)) for r in cur.fetchall()]

            # Fallback: use companies as a source of 'raw' rows
            cur.execute(
                """
                SELECT
                    company_id,
                    uen,
                    name AS entity_name,
                    industry_norm AS primary_ssic_description,
                    industry_code AS primary_ssic_code,
                    industry_code AS ssic_code,
                    incorporation_year AS raw_year,
                    sg_registered
                FROM companies
                ORDER BY company_id
                LIMIT %s
                """,
                (limit,),
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]


def _parse_year(val: Any) -> Optional[int]:
    if val is None:
        return None
    try:
        if isinstance(val, int):
            return val
        s = str(val).strip()
        # Extract 4-digit year
        import re as _re

        m = _re.search(r"(19|20)\d{2}", s)
        if m:
            return int(m.group(0))
    except Exception:
        return None
    return None


def _normalize_row(r: Dict[str, Any]) -> Dict[str, Any]:
    """Minimal normalization pass with flexible source keys."""

    def _norm_str(x: Optional[str]) -> Optional[str]:
        if x is None:
            return None
        s = str(x).strip()
        return s or None

    name = r.get("name") or r.get("entity_name")
    ind_norm = r.get("industry_norm") or r.get("primary_ssic_description")
    ind_code = r.get("industry_code")
    if ind_code is None:
        ind_code = r.get("primary_ssic_code")
    # Normalize to text
    ind_code = str(ind_code) if ind_code is not None else None
    raw_year = r.get("incorporation_year")
    if raw_year is None:
        raw_year = (
            r.get("founded_year")
            or r.get("raw_year")
            or r.get("registration_incorporation_date")
        )
    year = _parse_year(raw_year)
    # Founded year mirrors incorporation if available
    founded = year
    # sg_registered heuristic if missing
    sg = r.get("sg_registered")
    if sg is None:
        status = (r.get("entity_status_description") or "").lower()
        sg = True if status and ("live" in status or "active" in status) else None

    norm = {
        "company_id": r.get("company_id"),
        "uen": _norm_str(r.get("uen")),
        "name": _norm_str(name),
        "industry_norm": _norm_str(ind_norm).lower() if ind_norm else None,
        "industry_code": _norm_str(ind_code),
        "website_domain": _norm_str(r.get("website_domain") or r.get("website")),
        "incorporation_year": year,
        "founded_year": founded,
        "sg_registered": sg,
    }
    return norm


def _table_columns(conn, table: str) -> set[str]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT LOWER(column_name) FROM information_schema.columns WHERE table_name=%s",
            (table,),
        )
        return {r[0] for r in cur.fetchall()}


def _upsert_companies_batch(rows: List[Dict[str, Any]]) -> int:
    """Upsert normalized rows into companies table (dynamic columns, robust conflicts)."""
    if not rows:
        return 0
    affected = 0
    with get_conn() as conn:
        cols = _table_columns(conn, "companies")
        # Resolve target column names for schema variants
        col_industry = (
            "industry_code"
            if "industry_code" in cols
            else ("industory_code" if "industory_code" in cols else None)
        )
        col_founded = (
            "founded_year"
            if "founded_year" in cols
            else ("incorporation_year" if "incorporation_year" in cols else None)
        )
        col_sg = "sg_registered" if "sg_registered" in cols else None

        for r in rows:
            # Build column map for this row
            insert_cols: List[str] = []
            params: List[Any] = []

            # Optional PK if provided
            if r.get("company_id") is not None:
                insert_cols.append("company_id")
                params.append(r.get("company_id"))
            if r.get("uen") is not None and "uen" in cols:
                insert_cols.append("uen")
                params.append(r.get("uen"))
            if r.get("name") is not None and "name" in cols:
                insert_cols.append("name")
                params.append(r.get("name"))
            if r.get("industry_norm") is not None and "industry_norm" in cols:
                insert_cols.append("industry_norm")
                params.append(r.get("industry_norm"))
            if col_industry and r.get("industry_code") is not None:
                insert_cols.append(col_industry)
                params.append(r.get("industry_code"))
            if r.get("website_domain") is not None and "website_domain" in cols:
                insert_cols.append("website_domain")
                params.append(r.get("website_domain"))
            # Years
            if col_founded and r.get("founded_year") is not None:
                insert_cols.append(col_founded)
                params.append(r.get("founded_year"))
            elif (
                "incorporation_year" in cols and r.get("incorporation_year") is not None
            ):
                insert_cols.append("incorporation_year")
                params.append(r.get("incorporation_year"))
            if col_sg and r.get("sg_registered") is not None:
                insert_cols.append(col_sg)
                params.append(r.get("sg_registered"))

            # Always set last_seen on update; insert via NOW()
            insert_cols_sql = (
                ", ".join([*insert_cols, "last_seen"]) if insert_cols else "last_seen"
            )
            placeholders = ",".join(["%s"] * len(params) + ["NOW()"])

            # Determine conflict target: prefer company_id, else uen if available
            conflict_col = None
            if "company_id" in insert_cols:
                conflict_col = "company_id"
            elif "uen" in insert_cols and "uen" in cols:
                conflict_col = "uen"

            # Build update assignments for upsert
            set_cols = [c for c in insert_cols if c not in (conflict_col or "")] + [
                "last_seen"
            ]
            set_sql = ", ".join([f"{c} = EXCLUDED.{c}" for c in set_cols])

            sql = f"INSERT INTO companies ({insert_cols_sql}) VALUES ({placeholders})"
            if conflict_col:
                sql += f" ON CONFLICT ({conflict_col}) DO UPDATE SET {set_sql}"

            with conn.cursor() as cur:
                cur.execute(sql, params)
                affected += cur.rowcount or 1
        conn.commit()
    return affected


def _select_icp_candidates(payload: Dict[str, Any]) -> List[int]:
    """Build a simple WHERE from payload and fetch matching company_ids."""
    industries = [
        s.strip().lower()
        for s in payload.get("industries", [])
        if isinstance(s, str) and s.strip()
    ]
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
