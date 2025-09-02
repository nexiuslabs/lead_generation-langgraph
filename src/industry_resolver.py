from __future__ import annotations

import logging
from typing import Any, Dict, List

from src.database import get_conn

logger = logging.getLogger(__name__)


def resolve_industry_terms(
    terms: List[str],
    limit_codes: int = 3,
    samples_per_code: int = 3,
) -> List[Dict[str, Any]]:
    """Resolve free-text industry terms to ranked SSIC codes with sample companies.

    Each term maps to a list of matches ordered by frequency of occurrence in
    ``staging_acra_companies`` (falling back to ``companies`` when staging data is
    unavailable). For every SSIC code, up to ``samples_per_code`` company names are
    returned.

    Parameters
    ----------
    terms:
        Free-text industry descriptions supplied by the user.
    limit_codes:
        Maximum number of SSIC codes to return per term.
    samples_per_code:
        Maximum number of sample company names per SSIC code.
    """
    cleaned = [t.strip().lower() for t in terms if isinstance(t, str) and t.strip()]
    if not cleaned:
        return []

    results: List[Dict[str, Any]] = []

    with get_conn() as conn:
        for term in cleaned:
            matches: List[Dict[str, Any]] = []
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT primary_ssic_code::text, primary_ssic_description,
                               COUNT(*) AS freq
                        FROM staging_acra_companies
                        WHERE LOWER(primary_ssic_description) LIKE %s
                          AND primary_ssic_code IS NOT NULL
                        GROUP BY 1,2
                        ORDER BY freq DESC
                        LIMIT %s
                        """,
                        (f"%{term}%", limit_codes),
                    )
                    rows = cur.fetchall()
                # Fallback to companies table if staging has no matches
                if not rows:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            SELECT industry_code::text, industry_norm, COUNT(*) AS freq
                            FROM companies
                            WHERE LOWER(industry_norm) LIKE %s
                              AND industry_code IS NOT NULL
                            GROUP BY 1,2
                            ORDER BY freq DESC
                            LIMIT %s
                            """,
                            (f"%{term}%", limit_codes),
                        )
                        rows = cur.fetchall()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("resolve_industry_terms failed for %s: %s", term, exc)
                rows = []

            for code, desc, *_ in rows:
                samples: List[str] = []
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT name FROM companies WHERE industry_code = %s LIMIT %s",
                            (code, samples_per_code),
                        )
                        samples = [r[0] for r in cur.fetchall()]
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug("Sampling companies for SSIC %s failed: %s", code, exc)
                matches.append(
                    {
                        "code": str(code),
                        "description": desc,
                        "companies": samples,
                    }
                )

            results.append({"term": term, "matches": matches})

    return results
