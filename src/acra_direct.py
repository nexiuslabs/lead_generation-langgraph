import os
import re
import time
import json
import logging
import asyncio
from typing import Dict, Any, Iterable, Optional, Tuple

from src.database import get_conn

log = logging.getLogger("acra_direct")


def _env_int(name: str, default: int) -> int:
    try:
        v = os.getenv(name)
        return int(v) if v is not None and str(v).strip() else default
    except Exception:
        return default


def _env_str(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return (v.strip() if isinstance(v, str) else default) if v is not None else default


def _table_has_column(table: str, column: str) -> bool:
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM information_schema.columns WHERE table_name=%s AND column_name=%s",
                (table, column),
            )
            return cur.fetchone() is not None
    except Exception:
        return False


def _normalize_ssic(code: Any) -> str:
    try:
        return re.sub(r"\D", "", str(code or ""))
    except Exception:
        return ""


def _resolve_industry_title(code_digits: str, fallback_desc: Optional[str]) -> str:
    if not code_digits:
        return (fallback_desc or "").strip()
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT title
                FROM ssic_ref
                WHERE regexp_replace(code::text,'\\D','','g') = %s
                LIMIT 1
                """,
                (code_digits,),
            )
            row = cur.fetchone()
            if row and row[0]:
                return str(row[0]).strip()
    except Exception:
        pass
    return (fallback_desc or "").strip()


def upsert_company_from_staging(row: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    """Upsert a single company row derived from a staging_acra_companies record.

    Returns (company_id, debug_info) where debug_info includes resolved fields.
    """
    name = (row.get("entity_name") or "").strip()
    uen = (row.get("uen") or "").strip() or None
    ssic_code = _normalize_ssic(row.get("primary_ssic_code"))
    ssic_desc = row.get("primary_ssic_description")
    try:
        year = None
        val = row.get("registration_incorporation_date")
        if val:
            y = str(val)[:4]
            year = int(y) if y.isdigit() else None
    except Exception:
        year = None
    status = (row.get("entity_status_description") or None)
    website_hint = (row.get("website") or None)

    industry_title = _resolve_industry_title(ssic_code, ssic_desc)
    has_industry_col = _table_has_column("companies", "industry")
    has_status_col = _table_has_column("companies", "status")
    has_web_col = _table_has_column("companies", "website_domain")
    has_acra_source_col = _table_has_column("companies", "acra_source")

    with get_conn() as conn, conn.cursor() as cur:
        cid: Optional[int] = None
        if uen:
            # Insert or update by UEN
            cols = ["uen", "name", "industry_code", "industry_norm", "incorporation_year"]
            vals = [uen, name, ssic_code or None, industry_title or None, year]
            if has_status_col:
                cols.append("status")
                vals.append(status)
            if has_web_col:
                cols.append("website_domain")
                vals.append(website_hint)
            if has_industry_col:
                cols.append("industry")
                vals.append(industry_title or None)
            if has_acra_source_col:
                cols.append("acra_source")
                vals.append("staging_acra_companies")
            ph = ",".join(["%s"] * len(cols))
            cur.execute(
                f"INSERT INTO companies({','.join(cols)}) VALUES ({ph}) "
                f"ON CONFLICT (uen) DO UPDATE SET "
                f"name=EXCLUDED.name, industry_code=EXCLUDED.industry_code, industry_norm=EXCLUDED.industry_norm, "
                f"incorporation_year=EXCLUDED.incorporation_year"
                + (", status=EXCLUDED.status" if has_status_col else "")
                + (", website_domain=COALESCE(companies.website_domain, EXCLUDED.website_domain)" if has_web_col else "")
                + (", industry=EXCLUDED.industry" if has_industry_col else "")
                + (", acra_source=EXCLUDED.acra_source" if has_acra_source_col else "")
                + " RETURNING company_id",
                tuple(vals),
            )
            r = cur.fetchone()
            cid = int(r[0]) if r and r[0] is not None else None
        else:
            # Fallback identity: by LOWER(name) and incorporation_year
            cur.execute(
                "SELECT company_id FROM companies WHERE LOWER(name)=LOWER(%s) AND (incorporation_year IS NOT DISTINCT FROM %s) LIMIT 1",
                (name, year),
            )
            rr = cur.fetchone()
            if rr and rr[0] is not None:
                cid = int(rr[0])
                # Update core fields best-effort
                sets = [
                    ("industry_code", ssic_code or None),
                    ("industry_norm", industry_title or None),
                    ("incorporation_year", year),
                ]
                if has_status_col:
                    sets.append(("status", status))
                if has_web_col and website_hint:
                    sets.append(("website_domain", website_hint))
                if has_industry_col:
                    sets.append(("industry", industry_title or None))
                if has_acra_source_col:
                    sets.append(("acra_source", "staging_acra_companies"))
                set_sql = ", ".join([f"{k}=%s" for k, _ in sets])
                cur.execute(
                    f"UPDATE companies SET {set_sql} WHERE company_id=%s",
                    tuple([v for _, v in sets] + [cid]),
                )
            else:
                cols = ["name", "industry_code", "industry_norm", "incorporation_year"]
                vals = [name, ssic_code or None, industry_title or None, year]
                if has_status_col:
                    cols.append("status")
                    vals.append(status)
                if has_web_col:
                    cols.append("website_domain")
                    vals.append(website_hint)
                if has_industry_col:
                    cols.append("industry")
                    vals.append(industry_title or None)
                if has_acra_source_col:
                    cols.append("acra_source")
                    vals.append("staging_acra_companies")
                ph = ",".join(["%s"] * len(cols))
                cur.execute(
                    f"INSERT INTO companies({','.join(cols)}) VALUES ({ph}) RETURNING company_id",
                    tuple(vals),
                )
                r = cur.fetchone()
                cid = int(r[0]) if r and r[0] is not None else None
    if not cid:
        raise RuntimeError("Failed to upsert company")
    return cid, {
        "uen": uen,
        "name": name,
        "industry_code": ssic_code,
        "industry_title": industry_title,
        "incorporation_year": year,
        "status": status,
        "website_hint": website_hint,
    }


def _recent_enrichment_exists(company_id: int) -> bool:
    from src.enrichment import ENRICH_RECHECK_DAYS, ENRICH_SKIP_IF_ANY_HISTORY  # type: ignore
    try:
        with get_conn() as conn, conn.cursor() as cur:
            if ENRICH_SKIP_IF_ANY_HISTORY:
                cur.execute("SELECT 1 FROM company_enrichment_runs WHERE company_id=%s LIMIT 1", (company_id,))
                return cur.fetchone() is not None
            try:
                days = int(ENRICH_RECHECK_DAYS)
            except Exception:
                days = 0
            if days and days > 0:
                cur.execute(
                    "SELECT 1 FROM company_enrichment_runs WHERE company_id=%s AND COALESCE(updated_at, now()) >= now() - (%s::text || ' days')::interval LIMIT 1",
                    (company_id, str(days)),
                )
                return cur.fetchone() is not None
    except Exception:
        return False
    return False


def _insert_run_header() -> Optional[int]:
    try:
        with get_conn() as conn, conn.cursor() as cur:
            # Try with tenant_id if column exists
            cur.execute(
                "SELECT 1 FROM information_schema.columns WHERE table_name='enrichment_runs' AND column_name='tenant_id'"
            )
            has_tenant = cur.fetchone() is not None
            tid_env = os.getenv("DEFAULT_TENANT_ID")
            tid = int(tid_env) if tid_env and tid_env.isdigit() else None
            if has_tenant and tid is not None:
                cur.execute("INSERT INTO enrichment_runs(tenant_id) VALUES (%s) RETURNING run_id", (tid,))
            else:
                cur.execute("INSERT INTO enrichment_runs DEFAULT VALUES RETURNING run_id")
            r = cur.fetchone()
            return int(r[0]) if r and r[0] is not None else None
    except Exception:
        return None


def _set_rls_tenant() -> None:
    try:
        tid_env = os.getenv("DEFAULT_TENANT_ID")
        if not (tid_env and tid_env.isdigit()):
            return
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT set_config('request.tenant_id', %s, true)", (str(int(tid_env)),))
    except Exception:
        pass


def stream_staging_candidates(
    limit: Optional[int] = None,
    start_after_id: Optional[int] = None,
    start_from_uen: Optional[str] = None,
    recheck_days: Optional[int] = None,
) -> Iterable[Dict[str, Any]]:
    """Yield staging rows without ICP filtering, with flexible column mapping.

    Notes:
    - Not all deployments have a numeric ID on staging; we avoid hardcoding it.
    - Resume by UEN if available; `start_after_id` is ignored when no such column exists.
    """
    with get_conn() as conn, conn.cursor() as cur:
        # Introspect available staging columns
        cur.execute(
            "SELECT LOWER(column_name) FROM information_schema.columns WHERE table_name = 'staging_acra_companies'"
        )
        cols = {r[0] for r in cur.fetchall()}
        if not cols:
            return

        def pick(*names: str) -> Optional[str]:
            for n in names:
                if n.lower() in cols:
                    return n
            return None

        c_id = pick("id", "row_id", "_id")
        c_uen = pick("uen", "uen_no", "uen_number")
        c_name = pick("entity_name", "name", "company_name")
        c_desc = pick("primary_ssic_description", "ssic_description", "industry_description")
        c_code = pick("primary_ssic_code", "ssic_code", "industry_code", "ssic")
        c_year = pick("registration_incorporation_date", "incorporation_year", "founded_year")
        c_status = pick("entity_status_description", "entity_status", "status", "entity_status_de")

        select_list = []
        alias_pairs = [
            (c_uen, "uen"),
            (c_name, "entity_name"),
            (c_code, "primary_ssic_code"),
            (c_desc, "primary_ssic_description"),
            (c_year, "registration_incorporation_date"),
            (c_status, "entity_status_description"),
        ]
        if c_id:
            alias_pairs.insert(0, (c_id, "staging_id"))
        for src, alias in alias_pairs:
            select_list.append(f"{src} AS {alias}" if src else f"NULL AS {alias}")

        where_clauses = []
        params: list[Any] = []
        if start_after_id is not None and c_id:
            where_clauses.append(f"{c_id} > %s")
            params.append(int(start_after_id))
        if start_from_uen and c_uen:
            where_clauses.append(f"{c_uen} > %s")
            params.append(start_from_uen)
        # Selection-time recency filter: when recheck_days is provided and UEN exists on staging,
        # skip rows whose corresponding company has a recent enrichment history.
        if recheck_days and recheck_days > 0 and c_uen:
            where_clauses.append(
                "NOT EXISTS (\n"
                "  SELECT 1\n"
                "  FROM companies c\n"
                "  JOIN company_enrichment_runs r ON r.company_id = c.company_id\n"
                f"  WHERE c.uen IS NOT DISTINCT FROM {c_uen}\n"
                "    AND COALESCE(r.updated_at, now()) >= now() - (%s::text || ' days')::interval\n"
                ")"
            )
            params.append(str(int(recheck_days)))
        where_sql = (" WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
        order_by = c_id or (c_uen if c_uen else c_name)
        lim_sql = f" LIMIT {int(limit)}" if isinstance(limit, int) and limit > 0 else ""
        sql = f"SELECT {', '.join(select_list)} FROM staging_acra_companies{where_sql} ORDER BY {order_by} ASC{lim_sql}"
        # Optional debug to aid diagnosing empty selections
        try:
            dbg = (os.getenv("ACRA_DIRECT_DEBUG", "").lower() in ("1", "true", "yes", "on"))
        except Exception:
            dbg = False
        if dbg:
            try:
                print(f"[acra_direct.debug] order_by={order_by} limit={limit}")
                print(f"[acra_direct.debug] where={where_clauses} params={params}")
                print(f"[acra_direct.debug] sql={sql}")
            except Exception:
                pass
        cur.execute(sql, tuple(params))
        out_cols = [d[0] for d in cur.description]
        for r in cur.fetchall() or []:
            yield dict(zip(out_cols, r))


def _ensure_service_progress_table() -> None:
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS service_progress (
                  service_key TEXT PRIMARY KEY,
                  last_id BIGINT,
                  last_uen TEXT,
                  updated_at TIMESTAMPTZ DEFAULT now()
                )
                """
            )
    except Exception:
        pass


def _get_progress(service_key: str) -> Tuple[Optional[int], Optional[str]]:
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT last_id, last_uen FROM service_progress WHERE service_key=%s",
                (service_key,),
            )
            row = cur.fetchone()
            if row:
                return (row[0], row[1])
    except Exception:
        return (None, None)
    return (None, None)


def _set_progress(service_key: str, last_id: Optional[int], last_uen: Optional[str]) -> None:
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO service_progress(service_key, last_id, last_uen, updated_at)
                VALUES (%s,%s,%s, now())
                ON CONFLICT (service_key) DO UPDATE SET
                  last_id=EXCLUDED.last_id,
                  last_uen=EXCLUDED.last_uen,
                  updated_at=now()
                """,
                (service_key, last_id, last_uen),
            )
    except Exception:
        pass


def run_once() -> Dict[str, int]:
    """Run one pass over selected ACRA companies (no scheduling)."""
    from src.enrichment import enrich_company_with_tavily, set_run_context  # type: ignore
    processed = 0
    skipped = 0
    failures = 0
    t0 = time.perf_counter()
    _set_rls_tenant()
    run_id = _insert_run_header()
    # Read config
    batch = _env_int("ACRA_DIRECT_BATCH_LIMIT", 0)
    start_after_id = _env_int("ACRA_DIRECT_START_AFTER_ID", 0) or None
    start_from_uen = _env_str("ACRA_DIRECT_START_FROM_UEN", None)
    recheck_days = _env_int("ACRA_DIRECT_RECHECK_DAYS", 0) or None
    # Optional per-run vendor caps (contact lookups)
    try:
        from src.enrichment import set_vendor_caps  # type: ignore
        cap_env = os.getenv("CONTACT_LOOKUPS_CAP") or os.getenv("ACRA_CONTACT_LOOKUPS_CAP")
        cap = int(cap_env) if cap_env and str(cap_env).isdigit() else None
        if cap:
            set_vendor_caps(contact_lookups=cap)
    except Exception:
        pass
    # Progress resume: merge explicit env with stored checkpoint by taking the furthest position
    try:
        _ensure_service_progress_table()
        pid, puen = _get_progress("acra_direct")
        if pid is not None:
            try:
                pid_i = int(pid)
                if (start_after_id is None) or (pid_i > int(start_after_id)):
                    start_after_id = pid_i
            except Exception:
                pass
        if puen is not None and str(puen).strip():
            try:
                puen_s = str(puen)
                if (start_from_uen is None) or (puen_s > str(start_from_uen)):
                    start_from_uen = puen_s
            except Exception:
                pass
    except Exception:
        pass
    # Iterate
    count = 0
    for row in stream_staging_candidates(
        limit=(batch if batch and batch > 0 else None),
        start_after_id=start_after_id,
        start_from_uen=start_from_uen,
        recheck_days=recheck_days,
    ):
        count += 1
        try:
            cid, info = upsert_company_from_staging(row)
            log.info(
                json.dumps(
                    {
                        "service": "acra_direct",
                        "event": "upsert",
                        "company_id": cid,
                        "uen": info.get("uen"),
                        "name": info.get("name"),
                        "industry_code": info.get("industry_code"),
                        "industry_title": info.get("industry_title"),
                    },
                    ensure_ascii=False,
                )
            )
            # Advance checkpoint early to avoid repeating the same row on interruption.
            try:
                _set_progress("acra_direct", row.get("staging_id"), row.get("uen"))
            except Exception:
                pass
            if _recent_enrichment_exists(cid):
                skipped += 1
                log.info(
                    json.dumps(
                        {
                            "service": "acra_direct",
                            "event": "skip_existing_enrichment",
                            "company_id": cid,
                        },
                        ensure_ascii=False,
                    )
                )
                try:
                    _set_progress("acra_direct", row.get("staging_id"), row.get("uen"))
                except Exception:
                    pass
                continue
            if run_id is not None:
                set_run_context(run_id, int(os.getenv("DEFAULT_TENANT_ID", "0") or 0))
            # enrich_company_with_tavily is async; run it to completion here
            state = asyncio.run(
                enrich_company_with_tavily(
                    cid, info.get("name"), info.get("uen"), search_policy="discover"
                )
            )
            # enrichment graph persists core + history; log summary
            try:
                pages = len((state or {}).get("extracted_pages") or []) if isinstance(state, dict) else None
                emails = len(((state or {}).get("data") or {}).get("email", [])) if isinstance(state, dict) else None
                log.info(
                    json.dumps(
                        {
                            "service": "acra_direct",
                            "event": "enriched",
                            "company_id": cid,
                            "pages": pages,
                            "emails": emails,
                        },
                        ensure_ascii=False,
                    )
                )
            except Exception:
                pass
            processed += 1
        except KeyboardInterrupt:
            # Ensure checkpoint saved, then break to allow clean resume next run
            try:
                _set_progress("acra_direct", row.get("staging_id"), row.get("uen"))
            except Exception:
                pass
            log.info(
                json.dumps(
                    {
                        "service": "acra_direct",
                        "event": "interrupted",
                        "message": "checkpoint saved, exiting early",
                    },
                    ensure_ascii=False,
                )
            )
            break
        except Exception as e:
            failures += 1
            sid = row.get("staging_id") or row.get("uen") or row.get("entity_name")
            log.warning(
                json.dumps(
                    {
                        "service": "acra_direct",
                        "event": "failure",
                        "staging_ref": sid,
                        "error": str(e),
                    },
                    ensure_ascii=False,
                )
            )
            continue
        finally:
            try:
                _set_progress("acra_direct", row.get("staging_id"), row.get("uen"))
            except Exception:
                pass
    dur_ms = int((time.perf_counter() - t0) * 1000)
    log.info(
        json.dumps(
            {
                "service": "acra_direct",
                "event": "summary",
                "processed": processed,
                "skipped": skipped,
                "failures": failures,
                "duration_ms": dur_ms,
            },
            ensure_ascii=False,
        )
    )
    return {"processed": processed, "skipped": skipped, "failures": failures}
