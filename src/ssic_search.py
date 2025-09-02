"""Helpers for resolving SSIC (Singapore Standard Industrial Classification) codes.

This module provides utilities to look up SSIC reference data and sample
companies.  The database is treated as read‑only so that concurrent SSIC load
jobs are not affected.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence, Tuple

import psycopg2

from src.settings import POSTGRES_DSN

# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------


_CODE_RE = re.compile(r"^\d{4,5}$")


def _norm_terms(terms: Iterable[str]) -> Tuple[List[str], List[str]]:
    """Split terms into text terms and SSIC codes.

    Returns (texts, codes). Codes are normalised to 5‑digit strings with
    leading zeros preserved.
    """

    texts: List[str] = []
    codes: List[str] = []
    for t in terms:
        if t is None:
            continue
        s = str(t).strip().lower()
        if not s:
            continue
        if _CODE_RE.fullmatch(s):
            codes.append(s.zfill(5))
        else:
            texts.append(s)
    return texts, codes


# ---------------------------------------------------------------------------
# SSIC search
# ---------------------------------------------------------------------------


def search_ssic_terms(
    terms: Sequence[str], limit: int = 20
) -> List[Tuple[str, str, float]]:
    """Search ``ssic_ref_latest`` for the provided terms.

    Matching uses a combination of trigram similarity and full‑text search
    ranking.  The view automatically targets the latest ``ssic_ref`` version.
    """

    texts, codes = _norm_terms(terms)
    if not texts and not codes:
        return []

    conn = psycopg2.connect(dsn=POSTGRES_DSN)
    conn.set_session(readonly=True, autocommit=True)
    try:
        with conn.cursor() as cur:
            results: dict[str, Tuple[str, str, float]] = {}
            if codes:
                cur.execute(
                    """
                    SELECT code, title, 1.0 AS score
                    FROM ssic_ref_latest
                    WHERE code = ANY(%s)
                    """,
                    (codes,),
                )
                for code, title, score in cur.fetchall():
                    results[str(code)] = (str(code), title, float(score))

            for term in texts:
                cur.execute(
                    """
                    SELECT code,
                           title,
                           GREATEST(
                               similarity(title || ' ' || COALESCE(description,''), %s),
                               ts_rank_cd(
                                   to_tsvector('english', title || ' ' || COALESCE(description,'')),
                                   websearch_to_tsquery('english', %s)
                               )
                           ) AS score
                    FROM ssic_ref_latest
                    WHERE (
                          similarity(title || ' ' || COALESCE(description,''), %s) >= 0.1 OR
                          to_tsvector('english', title || ' ' || COALESCE(description,'')) @@ websearch_to_tsquery('english', %s)
                      )
                    ORDER BY score DESC
                    LIMIT %s
                    """,
                    (term, term, term, term, limit),
                )
                for code, title, score in cur.fetchall():
                    existing = results.get(str(code))
                    if existing is None or score > existing[2]:
                        results[str(code)] = (str(code), title, float(score))

        return sorted(results.values(), key=lambda r: r[2], reverse=True)[:limit]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Companies lookup
# ---------------------------------------------------------------------------


def companies_by_ssic(codes: Sequence[str], limit: int = 100) -> List[dict]:
    """Return sample companies for the given SSIC codes.

    Results are pulled from ``staging_acra_companies`` and joined against the
    ``companies`` table to attach an existing ``company_id`` when present.
    """

    _, codes_norm = _norm_terms(codes)
    if not codes_norm:
        return []

    conn = psycopg2.connect(dsn=POSTGRES_DSN)
    conn.set_session(readonly=True, autocommit=True)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT s.uen,
                       s.entity_name,
                       s.primary_ssic_code,
                       c.company_id
                FROM staging_acra_companies AS s
                LEFT JOIN companies AS c ON c.uen = s.uen
                WHERE s.primary_ssic_code = ANY(%s)
                ORDER BY s.entity_name
                LIMIT %s
                """,
                (codes_norm, limit),
            )
            rows = cur.fetchall()
        return [
            {
                "uen": row[0],
                "entity_name": row[1],
                "primary_ssic_code": str(row[2]) if row[2] is not None else None,
                "company_id": row[3],
            }
            for row in rows
        ]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# High level helper
# ---------------------------------------------------------------------------


def resolve_industry_terms(text: str, limit: int = 20) -> List[Tuple[str, str, float]]:
    """Extract possible SSIC search terms from free text and resolve them."""

    if not text:
        return []
    tokens = re.split(r"[,\n;]+|\band\b|\bor\b|/|\\|\|", text, flags=re.IGNORECASE)
    terms = [t.strip() for t in tokens if t.strip()]
    return search_ssic_terms(terms, limit=limit)
