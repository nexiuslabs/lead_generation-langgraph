import logging
from typing import Any, Dict, List, Optional
from psycopg2.extras import Json
from src.database import get_conn

log = logging.getLogger("icp_intake")


def _norm_domain(domain: Optional[str]) -> Optional[str]:
    if not domain:
        return None
    d = (domain or "").strip().lower()
    d = d.replace("http://", "").replace("https://", "")
    if d.startswith("www."):
        d = d[4:]
    # strip path
    for sep in ["/", "?", "#"]:
        if sep in d:
            d = d.split(sep, 1)[0]
    return d or None


def save_icp_intake(tenant_id: int, submitted_by: str, payload: Dict[str, Any]) -> int:
    """Persist intake responses and seeds in a single transaction.

    payload: { answers: json, seeds: [{ seed_name, domain }...] }
    Returns response_id.
    """
    answers = payload.get("answers") or {}
    seeds = payload.get("seeds") or []
    resp_id = 0
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO icp_intake_responses(tenant_id, submitted_by, answers_jsonb) VALUES (%s,%s,%s) RETURNING id",
            (tenant_id, submitted_by, Json(answers)),
        )
        row = cur.fetchone()
        resp_id = int(row[0]) if row else 0
        # Insert seeds
        for s in seeds:
            name = (s.get("seed_name") or "").strip()
            dom = _norm_domain(s.get("domain"))
            if not name:
                continue
            cur.execute(
                "INSERT INTO customer_seeds(tenant_id, seed_name, domain) VALUES (%s,%s,%s)",
                (tenant_id, name, dom),
            )
    return resp_id


def _company_id_for_seed(cur, name: str, domain: Optional[str]) -> Optional[int]:
    # Try domain match first
    if domain:
        cur.execute("SELECT company_id FROM companies WHERE website_domain = %s LIMIT 1", (domain,))
        r = cur.fetchone()
        if r and r[0] is not None:
            return int(r[0])
    # Fallback: exact name match (case-insensitive)
    cur.execute("SELECT company_id FROM companies WHERE LOWER(name) = LOWER(%s) LIMIT 1", (name,))
    r = cur.fetchone()
    if r and r[0] is not None:
        return int(r[0])
    return None


def fuzzy_map_seed_to_acra(cur, seed_name: str, threshold: float = 0.35) -> Optional[Dict[str, Any]]:
    """Find best fuzzy match in staging_acra_companies by entity_name using pg_trgm similarity.
    Returns { uen, primary_ssic_code, entity_name } or None.
    """
    try:
        cur.execute(
            """
            SELECT entity_name, uen, primary_ssic_code
            FROM staging_acra_companies
            WHERE similarity(
                regexp_replace(%s, '(pte|ltd|private|limited|singapore|inc)\\b', '', 'gi'),
                regexp_replace(entity_name, '(pte|ltd|private|limited|singapore|inc)\\b', '', 'gi')
            ) > %s
            ORDER BY similarity(
                regexp_replace(%s, '(pte|ltd|private|limited|singapore|inc)\\b', '', 'gi'),
                regexp_replace(entity_name, '(pte|ltd|private|limited|singapore|inc)\\b', '', 'gi')
            ) DESC
            LIMIT 1
            """,
            (seed_name, threshold, seed_name),
        )
        row = cur.fetchone()
        if row:
            return {"entity_name": row[0], "uen": row[1], "primary_ssic_code": row[2]}
    except Exception as e:
        log.info("fuzzy ACRA lookup failed: %s", e)
    return None


def map_seeds_to_evidence(tenant_id: int) -> int:
    """Map seeds to companies and ACRA; insert evidence rows. Returns count of seeds processed.

    Ensures a concrete company_id before inserting evidence to satisfy NOT NULL constraints.
    """
    processed = 0
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT seed_name, domain FROM customer_seeds WHERE tenant_id=%s ORDER BY id ASC", (tenant_id,))
        rows = cur.fetchall() or []
        for row in rows:
            name = (row[0] or "").strip()
            dom = _norm_domain(row[1])
            if not name:
                continue
            # Resolve or create company_id
            company_id = _company_id_for_seed(cur, name, dom)
            # Fuzzy ACRA mapping to enrich and help ensure company row by UEN
            acra = fuzzy_map_seed_to_acra(cur, name)
            if company_id is None and acra and acra.get("uen"):
                try:
                    cur.execute(
                        """
                        INSERT INTO companies(uen, name, industry_code, last_seen)
                        VALUES (%s, %s, %s, NOW())
                        ON CONFLICT (uen) DO UPDATE SET name=EXCLUDED.name, industry_code=COALESCE(EXCLUDED.industry_code, companies.industry_code), last_seen=NOW()
                        RETURNING company_id
                        """,
                        (
                            (acra.get("uen") or "").strip(),
                            (acra.get("entity_name") or name)[:255],
                            str(acra.get("primary_ssic_code")) if acra.get("primary_ssic_code") is not None else None,
                        ),
                    )
                    r = cur.fetchone()
                    if r and r[0] is not None:
                        company_id = int(r[0])
                except Exception as e:
                    log.info("ensure company by ACRA failed: %s", e)
            if acra and acra.get("primary_ssic_code") and company_id is not None:
                try:
                    cur.execute(
                        """
                        INSERT INTO icp_evidence(tenant_id, company_id, signal_key, value, source)
                        VALUES (%s,%s,'ssic', %s, 'acra')
                        """,
                        (
                            tenant_id,
                            company_id,
                            Json(
                                {
                                    "ssic": acra.get("primary_ssic_code"),
                                    "uen": acra.get("uen"),
                                    "matched_name": acra.get("entity_name"),
                                }
                            ),
                        ),
                    )
                except Exception as e:
                    log.info("icp_evidence insert failed: %s", e)
            processed += 1
    return processed


def refresh_icp_patterns() -> None:
    with get_conn() as conn, conn.cursor() as cur:
        try:
            cur.execute("REFRESH MATERIALIZED VIEW CONCURRENTLY icp_patterns")
        except Exception:
            # Fallback when non-concurrent refresh is required
            cur.execute("REFRESH MATERIALIZED VIEW icp_patterns")


def generate_suggestions(tenant_id: int, limit: int = 5) -> List[Dict[str, Any]]:
    """Generate simple suggestions based on SSIC frequency in evidence."""
    items: List[Dict[str, Any]] = []
    with get_conn() as conn, conn.cursor() as cur:
        try:
            cur.execute(
                """
                SELECT (value->>'ssic') AS code, COUNT(*) AS cnt
                FROM icp_evidence
                WHERE tenant_id=%s AND signal_key='ssic' AND value ? 'ssic'
                GROUP BY 1
                ORDER BY cnt DESC
                LIMIT %s
                """,
                (tenant_id, limit),
            )
            rows = cur.fetchall() or []
            for code, cnt in rows:
                items.append({
                    "id": f"ssic:{code}",
                    "title": f"SSIC {code}",
                    "rationale": f"Top occurrence across mapped seeds ({cnt})",
                    "evidence_count": int(cnt),
                })
        except Exception as e:
            log.info("suggestion generation failed: %s", e)
    return items


def store_intake_evidence(tenant_id: int, answers: Dict[str, Any]) -> int:
    """Persist minimal evidence derived from Fast‑Start answers into icp_evidence.

    Inserts one row per signal type with source='intake'. Returns number of rows written.
    """
    written = 0
    with get_conn() as conn, conn.cursor() as cur:
        try:
            # Avoid transaction-wide aborts: autocommit small inserts so one failure does not poison the cursor
            setattr(conn, "autocommit", True)
        except Exception:
            pass
        # Ensure a concrete company_id to satisfy icp_evidence.company_id NOT NULL
        company_id: Optional[int] = None
        try:
            site_url = (answers.get("website") or "").strip()
            apex = None
            if site_url:
                try:
                    from urllib.parse import urlparse
                    apex = (urlparse(site_url).netloc or site_url).lower()
                except Exception:
                    apex = None
            if apex:
                cur.execute("SELECT company_id FROM companies WHERE website_domain=%s LIMIT 1", (apex,))
                row = cur.fetchone()
                if row and row[0] is not None:
                    company_id = int(row[0])
                else:
                    # Insert minimal row for the tenant website
                    cur.execute(
                        "INSERT INTO companies(name, website_domain, last_seen) VALUES (%s,%s,NOW()) RETURNING company_id",
                        (apex, apex),
                    )
                    rr = cur.fetchone()
                    if rr and rr[0] is not None:
                        company_id = int(rr[0])
        except Exception as e:
            log.info("ensure tenant website company failed: %s", e)

        def put(key: str, value: Any):
            nonlocal written
            try:
                if company_id is None:
                    return  # cannot insert due to NOT NULL schema
                cur.execute(
                    "INSERT INTO icp_evidence(tenant_id, company_id, signal_key, value, source) VALUES (%s, %s, %s, %s, 'intake')",
                    (tenant_id, company_id, key, Json(value)),
                )
                written += 1
            except Exception as e:
                log.info("intake evidence insert failed: %s", e)
                try:
                    # Best effort: clear error state if transaction aborted
                    conn.rollback()
                except Exception:
                    pass

        # Champion titles
        titles = answers.get("champion_titles") or []
        if isinstance(titles, list) and titles:
            put("champion_titles", {"titles": titles})
        # Integrations required
        integ = answers.get("integrations_required") or []
        if isinstance(integ, list) and integ:
            put("integrations_required", {"integrations": integ})
        # ACV USD
        acv = answers.get("acv_usd")
        if isinstance(acv, (int, float)) and acv > 0:
            put("acv", {"usd": float(acv)})
        # Deal cycle in weeks
        wmin = answers.get("cycle_weeks_min")
        wmax = answers.get("cycle_weeks_max")
        if (isinstance(wmin, (int, float)) and isinstance(wmax, (int, float)) and wmin > 0 and wmax >= wmin) or (
            isinstance(wmin, (int, float)) and wmin > 0
        ):
            rec = {}
            if isinstance(wmin, (int, float)):
                rec["min"] = float(wmin)
            if isinstance(wmax, (int, float)):
                rec["max"] = float(wmax)
            put("deal_cycle_weeks", rec)
        # Price floor USD
        pf = answers.get("price_floor_usd")
        if isinstance(pf, (int, float)) and pf > 0:
            put("price_floor", {"usd": float(pf)})
        # Triggers/events
        ev = answers.get("triggers") or []
        if isinstance(ev, list) and ev:
            put("triggers", {"events": ev})
        # Lost/churned summary
        lost = answers.get("lost_churned") or []
        if isinstance(lost, list) and lost:
            # Store reasons-only snapshot to help pattern mining later
            reasons = [r.get("reason") for r in lost if isinstance(r, dict) and r.get("reason")]
            if reasons:
                put("lost_churned_reasons", {"reasons": reasons})
    return written
