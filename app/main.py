# app/main.py
from fastapi import FastAPI, Request, Response, Depends, BackgroundTasks, HTTPException, Path, Query
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from app.onboarding import handle_first_login, get_onboarding_status
from app.odoo_connection_info import get_odoo_connection_info
from src.database import get_pg_pool, get_conn
from app.auth import require_auth, require_identity, require_optional_identity
from app.odoo_store import OdooStore
from src.settings import OPENAI_API_KEY
import os
import shutil
import csv
from io import StringIO
import logging
import re
import os
from datetime import datetime
import time
import threading
import asyncio
import math

fmt = logging.Formatter("[%(levelname)s] %(asctime)s %(name)s :: %(message)s", "%H:%M:%S")

def _ensure_logger(name: str, level: str = "INFO"):
    lg = logging.getLogger(name)
    if not lg.handlers:
        h = logging.StreamHandler()
        h.setFormatter(fmt)
        lg.addHandler(h)
    lg.setLevel(level)
    return lg

# Configure important app loggers so they are visible in Uvicorn output
logger = _ensure_logger("input_norm")
_ensure_logger("onboarding")
_ensure_logger("app.odoo_store")
# Reduce noise from upstream libraries during local_dev
try:
    logging.getLogger("langgraph_api.metadata").setLevel(logging.ERROR)
    # Keep server logs, but avoid spamming warnings for 401/403 on health checks
    if (os.getenv("LANGSMITH_LANGGRAPH_API_VARIANT", "") or "").strip().lower() == "local_dev":
        srv_logger = logging.getLogger("langgraph_api.server")
        srv_logger.setLevel(logging.INFO)

        # Suppress extremely chatty access logs for specific health/poll endpoints
        class _SkipAccessPath(logging.Filter):
            def __init__(self, paths):
                super().__init__()
                # Match by substring in rendered log line
                self.paths = tuple(paths)

            def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
                try:
                    msg = record.getMessage()
                    if not isinstance(msg, str):
                        msg = str(msg)
                    # Drop logs that contain any of the noisy paths
                    if any(p in msg for p in self.paths):
                        return False
                except Exception:
                    # Fail-open: keep the record if anything goes wrong
                    return True
                return True

        # Filter out access lines for hot-poll endpoints (frontend polls frequently)
        srv_logger.addFilter(_SkipAccessPath(["/session/odoo_info", "/shortlist/status"]))
except Exception:
    pass

# Ensure LangGraph checkpoint directory exists to prevent FileNotFoundError
# e.g., '.langgraph_api/.langgraph_checkpoint.*.pckl.tmp'
CHECKPOINT_DIR = os.environ.get("LANGGRAPH_CHECKPOINT_DIR", ".langgraph_api")
try:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
except Exception as e:
    logger.warning("Failed to ensure checkpoint dir %s: %s", CHECKPOINT_DIR, e)

# Optional: clear any persisted LangGraph checkpoint/runs on server boot in local dev
try:
    _variant = (os.getenv("LANGSMITH_LANGGRAPH_API_VARIANT") or "local_dev").strip().lower()
    _clear_flag = (os.getenv("LANGGRAPH_CLEAR_ON_BOOT") or "").strip().lower()
    # Default to clearing on boot in local_dev unless explicitly disabled
    _should_clear = (_variant == "local_dev" and _clear_flag not in {"0", "false", "no", "off"}) or _clear_flag in {"1", "true", "yes", "on"}
    if _should_clear and os.path.isdir(CHECKPOINT_DIR):
        removed = 0
        for name in os.listdir(CHECKPOINT_DIR):
            p = os.path.join(CHECKPOINT_DIR, name)
            try:
                if os.path.isdir(p) and not os.path.islink(p):
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    os.unlink(p)
                removed += 1
            except Exception:
                # Best-effort; continue
                continue
        if removed:
            logger.info("Cleared %d checkpoint items from %s on boot", removed, CHECKPOINT_DIR)
except Exception as _e:
    logger.warning("Unable to clear checkpoint dir on boot: %s", _e)

app = FastAPI(title="Pre-SDR LangGraph Server")

# CORS allowlist (env-extensible)
extra_origins = []
try:
    raw = os.getenv("EXTRA_CORS_ORIGINS", "")
    if raw:
        extra_origins = [o.strip() for o in raw.split(",") if o.strip()]
except Exception:
    extra_origins = []

allow_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
] + extra_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Note: LangServe routes removed; chat/graph execution is handled internally without mounting /agent

# Mount auth cookie routes
try:
    from app.auth_routes import router as auth_router
    app.include_router(auth_router)
except Exception as _e:
    logger.warning("Auth routes not mounted: %s", _e)

# Optional: mount split-origin graph proxy if configured
try:
    if (os.getenv("ENABLE_GRAPH_PROXY") or "").strip().lower() in ("1", "true", "yes", "on"):
        from app.graph_proxy import router as graph_router
        app.include_router(graph_router)
        logger.info("/graph proxy routes enabled")
except Exception as _e:
    logger.warning("Graph proxy not mounted: %s", _e)

# Mount ICP Finder endpoints when enabled
try:
    from src.settings import ENABLE_ICP_INTAKE  # type: ignore
    if ENABLE_ICP_INTAKE:
        from app.icp_endpoints import router as icp_router  # type: ignore
        app.include_router(icp_router)
        logger.info("/icp endpoints enabled")
    else:
        logger.info("Skipping /icp endpoints (ENABLE_ICP_INTAKE is false)")
except Exception as _e:
    logger.warning("ICP endpoints not mounted: %s", _e)

# Mount chat SSE stream routes
try:
    from app.chat_stream import router as chat_router
    app.include_router(chat_router)
    logger.info("/chat SSE routes enabled")
except Exception as _e:
    logger.warning("Chat SSE routes not mounted: %s", _e)

def _role_to_type(role: str) -> str:
    r = (role or "").lower()
    if r in ("user", "human"): return "human"
    if r in ("assistant", "ai"): return "ai"
    if r == "system": return "system"
    return "human"

def _to_message(msg: dict) -> BaseMessage:
    # Accepts {"role":"human","content":"..."} or {"type":"ai","content":"..."}
    mtype = msg.get("type") or _role_to_type(msg.get("role", "human"))
    content = msg.get("content", "")
    if mtype == "human":
        return HumanMessage(content=content)
    if mtype == "system":
        return SystemMessage(content=content)
    return AIMessage(content=content)  # default to AI


def _last_human_text(messages: list[BaseMessage] | None) -> str:
    if not messages:
        return ""
    # Scan from end to find last HumanMessage
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return (m.content or "").strip()
    # Fallback to last message content
    try:
        return (messages[-1].content or "").strip()
    except Exception:
        return ""


def _extract_industry_terms(text: str) -> list[str]:
    """Best-effort extraction of industry terms from free text.

    Heuristics:
    - Split on commas/newlines/semicolons and common connectors
    - Keep alpha tokens (2+ chars), drop obvious geo/size words
    - Lowercase for DB equality against staging primary_ssic_description
    """
    if not text:
        return []
    # Split into candidate chunks
    chunks = re.split(r"[,\n;]+|\band\b|\bor\b|/|\\\\|\|", text, flags=re.IGNORECASE)
    terms: list[str] = []
    stop = {
        "sg",
        "singapore",
        "sea",
        "apac",
        "global",
        "worldwide",
        "us",
        "usa",
        "uk",
        "eu",
        "emea",
        "asia",
        "startup",
        "startups",
        "smb",
        "sme",
        "enterprise",
        "b2b",
        "b2c",
        "confirm",
        "run enrichment",
        # conversational fillers
        "start",
        "which",
        "which industries",
        "problem spaces",
        "should we target",
        "e.g.",
        "eg",
        # revenue buckets that might slip into industries
        "small",
        "medium",
        "large",
    }
    for c in chunks:
        s = (c or "").strip()
        if not s or len(s) < 2:
            continue
        # Thin out non-alpha heavy tokens
        if not re.search(r"[a-zA-Z]", s):
            continue
        # Drop obvious question/parenthetical fragments and bullet lines
        if any(ch in s for ch in ("?", "(", ")", ":")):
            continue
        if s.strip().startswith("-"):
            continue
        sl = s.lower()
        if sl in stop:
            continue
        # Common formatting artifacts
        sl = re.sub(r"\s+", " ", sl)
        terms.append(sl)
    # Dedupe while preserving order and prefer multi-word phrases
    seen = set()
    out: list[str] = []
    for t in terms:
        if t not in seen:
            seen.add(t)
            out.append(t)
    multi = [t for t in out if " " in t]
    if multi:
        singles = {t for t in out if " " not in t}
        singles = {s for s in singles if any(s in m.split() for m in multi)}
        out = [t for t in out if not (" " not in t and t in singles)]
    return out[:10]


"""Feature 18 flags: sync-head limit and mode"""
STAGING_UPSERT_MODE = os.getenv("STAGING_UPSERT_MODE", "background").strip().lower()
try:
    UPSERT_SYNC_LIMIT = int(os.getenv("UPSERT_SYNC_LIMIT", "10") or 10)
except Exception:
    UPSERT_SYNC_LIMIT = 10

def upsert_by_industries_head(industries: list[str], limit: int = UPSERT_SYNC_LIMIT) -> list[int]:
    """Upsert up to `limit` companies from staging for the given industries and return their IDs.

    No enrichment is triggered here; enrichment runs only after ICP confirmation.
    """
    if not industries or limit <= 0:
        return []
    upserted_ids: list[int] = []
    try:
        from src.icp import _find_ssic_codes_by_terms
        with get_conn() as conn, conn.cursor() as cur:
            # Introspect staging columns we need
            cur.execute(
                """
                SELECT LOWER(column_name)
                FROM information_schema.columns
                WHERE table_name = 'staging_acra_companies'
                """
            )
            cols = {r[0] for r in cur.fetchall()}
            def pick(*names: str) -> str | None:
                for n in names:
                    if n.lower() in cols:
                        return n
                return None
            src_uen = pick('uen','uen_no','uen_number') or 'NULL'
            src_name = pick('entity_name','name','company_name') or 'NULL'
            src_desc = pick('primary_ssic_description','ssic_description','industry_description')
            src_code = pick('primary_ssic_code','ssic_code','industry_code','ssic') or 'NULL'
            raw_year = pick('registration_incorporation_date','incorporation_year','year_incorporated','inc_year','founded_year') or 'NULL'
            if isinstance(raw_year, str) and raw_year.lower() == 'registration_incorporation_date':
                src_year = f"NULLIF(substring({raw_year} from '\\d{{4}}'), '')::int"
            else:
                src_year = raw_year
            src_stat = pick('entity_status_de','entity_status','status','entity_status_description') or 'NULL'
            src_owner = pick('business_constitution_description','company_type_description','entity_type_description','paf_constitution_description','ownership_type') or 'NULL'

            if not src_desc:
                return []

            lower_terms = [((t or '').strip().lower()) for t in industries if (t or '').strip()]
            like_patterns = [f"%{t}%" for t in lower_terms]
            codes_rows = _find_ssic_codes_by_terms(lower_terms)
            code_list = [c for (c, _title, _score) in codes_rows]

            if code_list:
                select_sql = f"""
                    SELECT
                      {src_uen} AS uen,
                      {src_name} AS entity_name,
                      {src_desc} AS primary_ssic_description,
                      {src_code} AS primary_ssic_code,
                      {src_year} AS incorporation_year,
                      {src_stat} AS entity_status_de,
                      {src_owner} AS ownership_type
                    FROM staging_acra_companies
                    WHERE regexp_replace({src_code}::text, '\\D', '', 'g') = ANY(%s::text[])
                    LIMIT %s
                """
                cur.execute(select_sql, (code_list, int(limit)))
            else:
                select_sql = f"""
                    SELECT
                      {src_uen} AS uen,
                      {src_name} AS entity_name,
                      {src_desc} AS primary_ssic_description,
                      {src_code} AS primary_ssic_code,
                      {src_year} AS incorporation_year,
                      {src_stat} AS entity_status_de,
                      {src_owner} AS ownership_type
                    FROM staging_acra_companies
                    WHERE LOWER({src_desc}) = ANY(%s)
                       OR {src_desc} ILIKE ANY(%s)
                    LIMIT %s
                """
                cur.execute(select_sql, (lower_terms, like_patterns, int(limit)))

            rows = cur.fetchall()
            try:
                col_aliases = [
                    getattr(d, 'name', None) or (d[0] if isinstance(d, (list, tuple)) and d else None)
                    for d in (cur.description or [])
                ]
            except Exception:
                col_aliases = []

            def row_to_map(row: object) -> dict[str, object]:
                try:
                    if not isinstance(row, (list, tuple)):
                        return {}
                    limitn = min(len(col_aliases), len(row)) if col_aliases else 0
                    out: dict[str, object] = {}
                    for i in range(limitn):
                        key = col_aliases[i]
                        if key:
                            out[key] = row[i]
                    return out
                except Exception:
                    return {}

            for r in rows:
                m = row_to_map(r)
                uen = m.get('uen')
                entity_name = m.get('entity_name')
                ssic_desc = m.get('primary_ssic_description')
                ssic_code = m.get('primary_ssic_code')
                inc_year = m.get('incorporation_year')
                status_de = m.get('entity_status_de')
                ownership_type = m.get('ownership_type')

                name = (entity_name or "").strip() or None  # type: ignore[arg-type]
                desc_lower = (ssic_desc or "").strip().lower()  # type: ignore[arg-type]
                match_term = None
                for t in industries:
                    tl = (t or '').strip().lower()
                    if desc_lower == tl or (tl and tl in desc_lower):
                        match_term = tl
                        break
                industry_norm = (match_term or desc_lower) or None
                industry_code = str(ssic_code) if ssic_code is not None else None
                sg_registered = None
                try:
                    sg_registered = ((status_de or "").strip().lower() in {"live", "registered", "existing"})  # type: ignore[arg-type]
                except Exception:
                    pass

                # Locate existing company
                company_id = None
                if uen:
                    cur.execute("SELECT company_id FROM companies WHERE uen = %s LIMIT 1", (uen,))
                    rw = cur.fetchone()
                    if rw and isinstance(rw, (list, tuple)) and len(rw) >= 1:
                        company_id = rw[0]
                if company_id is None and name:
                    cur.execute("SELECT company_id FROM companies WHERE LOWER(name) = LOWER(%s) LIMIT 1", (name,))
                    rw = cur.fetchone()
                    if rw and isinstance(rw, (list, tuple)) and len(rw) >= 1:
                        company_id = rw[0]

                fields = {
                    "uen": uen,
                    "name": name,
                    "industry_norm": industry_norm,
                    "industry_code": industry_code,
                    "incorporation_year": inc_year,
                    "sg_registered": sg_registered,
                    "ownership_type": (ownership_type or None),
                }
                if company_id is not None:
                    set_parts = []
                    params = []
                    for k, v in fields.items():
                        if v is not None:
                            set_parts.append(f"{k} = %s")
                            params.append(v)
                    set_sql = ", ".join(set_parts) + ", last_seen = NOW()" if set_parts else "last_seen = NOW()"
                    cur.execute(f"UPDATE companies SET {set_sql} WHERE company_id = %s", params + [company_id])
                    # Track updated IDs
                    if company_id is not None:
                        upserted_ids.append(int(company_id))
                else:
                    cols = [k for k, v in fields.items() if v is not None]
                    vals = [fields[k] for k in cols]
                    if not cols:
                        continue
                    cols_sql = ", ".join(cols)
                    ph = ",".join(["%s"] * len(vals))
                    cur.execute(f"INSERT INTO companies ({cols_sql}) VALUES ({ph}) RETURNING company_id", vals)
                    rw = cur.fetchone()
                    new_id = rw[0] if (rw and isinstance(rw, (list, tuple)) and len(rw) >= 1) else None
                    if new_id is not None:
                        cur.execute("UPDATE companies SET last_seen = NOW() WHERE company_id = %s", (new_id,))
                        upserted_ids.append(int(new_id))
        return upserted_ids
    except Exception:
        logger.exception("sync head upsert error")
        return []

def _trigger_enrichment_async(company_ids: list[int]) -> None:
    """Fire-and-forget enrichment for provided company IDs (non-blocking)."""
    if not company_ids:
        return
    try:
        from src.orchestrator import enrich_companies as _enrich_async
    except Exception:
        logger.info("Enrichment module unavailable; skipping async enrichment trigger")
        return
    import threading, asyncio as _asyncio
    def _runner(ids: list[int]):
        try:
            _asyncio.run(_enrich_async(ids))
        except Exception:
            logger.warning("Async enrichment failed for ids=%s", ids)
    try:
        threading.Thread(target=_runner, args=(list(company_ids),), daemon=True).start()
    except Exception:
        logger.info("Failed to start enrichment thread; skipping")

def _collect_industry_terms(messages: list[BaseMessage] | None) -> list[str]:
    if not messages:
        return []
    # Use only last human message to avoid assistant prompts
    text = _last_human_text(messages)
    # If the input looks like a URL or bare domain, do not treat it as industry terms
    try:
        t = (text or "").strip()
        if not t:
            return []
        # URL or www.*
        if re.match(r"^(https?://|www\.)", t, flags=re.IGNORECASE):
            return []
        # Bare domain like example.com or sub.example.co.uk
        if re.match(r"^[A-Za-z0-9.-]+\.[A-Za-z]{2,}$", t):
            return []
    except Exception:
        pass
    return _extract_industry_terms(text)


def _upsert_companies_from_staging_by_industries(industries: list[str]) -> int:
    """Resolve SSIC codes via ssic_ref, fetch staging companies by code, and upsert into companies.

    Flow:
      1) Resolve codes from `ssic_ref` using industry terms (title/description via FTS/trigram).
      2) If codes found: pull rows from staging where normalized primary SSIC code is in that set.
      3) Else fallback: match staging by LOWER(primary_ssic_description) (with ILIKE partials).
      4) Upsert results into companies.
    Returns number of affected rows (inserted + updated best-effort).
    """
    if not industries:
        return 0
    affected = 0
    try:
        with get_conn() as conn, conn.cursor() as cur:
            # Discover available columns to build a safe SELECT
            cur.execute(
                """
                SELECT LOWER(column_name)
                FROM information_schema.columns
                WHERE table_name = 'staging_acra_companies'
                """
            )
            cols = {r[0] for r in cur.fetchall()}
            def pick(*names: str) -> str | None:
                for n in names:
                    if n.lower() in cols:
                        return n
                return None
            src_uen = pick('uen','uen_no','uen_number') or 'NULL'
            src_name = pick('entity_name','name','company_name') or 'NULL'
            src_desc = pick('primary_ssic_description','ssic_description','industry_description')
            src_code = pick('primary_ssic_code','ssic_code','industry_code','ssic') or 'NULL'
            src_web  = pick('website','website_url','website_domain','url','homepage') or 'NULL'
            raw_year = pick('registration_incorporation_date','incorporation_year','year_incorporated','inc_year','founded_year') or 'NULL'
            if isinstance(raw_year, str) and raw_year.lower() == 'registration_incorporation_date':
                src_year = f"NULLIF(substring({raw_year} from '\\d{{4}}'), '')::int"
            else:
                src_year = raw_year
            src_stat = pick('entity_status_de','entity_status','status','entity_status_description') or 'NULL'
            src_owner = pick('business_constitution_description','company_type_description','entity_type_description','paf_constitution_description','ownership_type') or 'NULL'

            if not src_desc:
                return 0

            # Resolve SSIC codes from ssic_ref (FTS/trigram)
            lower_terms = [((t or '').strip().lower()) for t in industries if (t or '').strip()]
            like_patterns = [f"%{t}%" for t in lower_terms]
            codes_rows = _find_ssic_codes_by_terms(lower_terms)
            code_list = [c for (c, _title, _score) in codes_rows]
            if code_list:
                codes_preview = ", ".join(code_list[:50])
                if len(code_list) > 50:
                    codes_preview += f", ... (+{len(code_list)-50} more)"
                logger.info("ssic_ref resolved %d SSIC codes from industries=%s: %s", len(code_list), lower_terms, codes_preview)

            if code_list:
                select_sql = f"""
                    SELECT
                      {src_uen} AS uen,
                      {src_name} AS entity_name,
                      {src_desc} AS primary_ssic_description,
                      {src_code} AS primary_ssic_code,
                      {src_web}  AS website,
                      {src_year} AS incorporation_year,
                      {src_stat} AS entity_status_de,
                      {src_owner} AS ownership_type
                    FROM staging_acra_companies
                    WHERE regexp_replace({src_code}::text, '\\D', '', 'g') = ANY(%s::text[])
                    LIMIT 1000
                """
                cur.execute(select_sql, (code_list,))
            else:
                select_sql = f"""
                    SELECT
                      {src_uen} AS uen,
                      {src_name} AS entity_name,
                      {src_desc} AS primary_ssic_description,
                      {src_code} AS primary_ssic_code,
                      {src_web}  AS website,
                      {src_year} AS incorporation_year,
                      {src_stat} AS entity_status_de,
                      {src_owner} AS ownership_type
                    FROM staging_acra_companies
                    WHERE LOWER({src_desc}) = ANY(%s)
                       OR {src_desc} ILIKE ANY(%s)
                    LIMIT 1000
                """
                cur.execute(select_sql, (lower_terms, like_patterns))

            rows = cur.fetchall()

            # Build alias list safely
            try:
                col_aliases = [
                    getattr(d, 'name', None) or (d[0] if isinstance(d, (list, tuple)) and d else None)
                    for d in (cur.description or [])
                ]
            except Exception:
                col_aliases = []

            def row_to_map(row: object) -> dict[str, object]:
                try:
                    if not isinstance(row, (list, tuple)):
                        return {}
                    limit = min(len(col_aliases), len(row)) if col_aliases else 0
                    out: dict[str, object] = {}
                    for i in range(limit):
                        key = col_aliases[i]
                        if key:
                            out[key] = row[i]
                    return out
                except Exception:
                    return {}

            # Preview names for SSIC path
            if code_list and rows:
                try:
                    names = []
                    for r in rows[:50]:
                        nm = (row_to_map(r).get('entity_name') or '').strip()  # type: ignore[arg-type]
                        if nm:
                            names.append(nm)
                    if names:
                        preview = ", ".join(names)
                        extra = f", ... (+{len(rows)-50} more)" if len(rows) > 50 else ""
                        logger.info("staging_acra_companies matched %d rows by SSIC code; names: %s%s", len(rows), preview, extra)
                except Exception:
                    pass

            if not rows:
                return 0

            for r in rows:
                m = row_to_map(r)
                uen = m.get('uen')
                entity_name = m.get('entity_name')
                ssic_desc = m.get('primary_ssic_description')
                ssic_code = m.get('primary_ssic_code')
                website = m.get('website')
                inc_year = m.get('incorporation_year')
                status_de = m.get('entity_status_de')
                ownership_type = m.get('ownership_type')

                name = (entity_name or "").strip() or None  # type: ignore[arg-type]
                desc_lower = (ssic_desc or "").strip().lower()  # type: ignore[arg-type]
                match_term = None
                for t in industries:
                    if desc_lower == t or (t in desc_lower):
                        match_term = t
                        break
                industry_norm = (match_term or desc_lower) or None
                industry_code = str(ssic_code) if ssic_code is not None else None
                website_domain = (website or "").strip() or None  # type: ignore[arg-type]
                sg_registered = None
                try:
                    sg_registered = ((status_de or "").strip().lower() in {"live", "registered", "existing"})  # type: ignore[arg-type]
                except Exception:
                    pass

                # Locate existing company by UEN, name, or website
                company_id = None
                if uen:
                    cur.execute("SELECT company_id FROM companies WHERE uen = %s LIMIT 1", (uen,))
                    rw = cur.fetchone()
                    if rw and isinstance(rw, (list, tuple)) and len(rw) >= 1:
                        company_id = rw[0]
                if company_id is None and name:
                    cur.execute("SELECT company_id FROM companies WHERE LOWER(name) = LOWER(%s) LIMIT 1", (name,))
                    rw = cur.fetchone()
                    if rw and isinstance(rw, (list, tuple)) and len(rw) >= 1:
                        company_id = rw[0]
                if company_id is None and website_domain:
                    cur.execute("SELECT company_id FROM companies WHERE website_domain = %s LIMIT 1", (website_domain,))
                    rw = cur.fetchone()
                    if rw and isinstance(rw, (list, tuple)) and len(rw) >= 1:
                        company_id = rw[0]

                fields = {
                    "uen": uen,
                    "name": name,
                    "industry_norm": industry_norm,
                    "industry_code": industry_code,
                    "website_domain": website_domain,
                    "incorporation_year": inc_year,
                    "sg_registered": sg_registered,
                    "ownership_type": (ownership_type or None),
                }

                if company_id is not None:
                    set_parts = []
                    params = []
                    for k, v in fields.items():
                        if v is not None:
                            set_parts.append(f"{k} = %s")
                            params.append(v)
                    set_sql = ", ".join(set_parts) + ", last_seen = NOW()" if set_parts else "last_seen = NOW()"
                    cur.execute(f"UPDATE companies SET {set_sql} WHERE company_id = %s", params + [company_id])
                    affected += cur.rowcount or 0
                else:
                    cols = [k for k, v in fields.items() if v is not None]
                    vals = [fields[k] for k in cols]
                    if not cols:
                        continue
                    cols_sql = ", ".join(cols)
                    ph = ",".join(["%s"] * len(vals))
                    cur.execute(f"INSERT INTO companies ({cols_sql}) VALUES ({ph}) RETURNING company_id", vals)
                    rw = cur.fetchone()
                    new_id = rw[0] if (rw and isinstance(rw, (list, tuple)) and len(rw) >= 1) else None
                    if new_id is not None:
                        cur.execute("UPDATE companies SET last_seen = NOW() WHERE company_id = %s", (new_id,))
                        affected += 1
        return affected
    except Exception:
        logger.exception("staging upsert error")
        return 0

from src.settings import ENABLE_ICP_INTAKE  # ensure available for gating


def normalize_input(payload: dict) -> dict:
    """
    Accept a variety of UI payloads and emit the graph state:
      {"messages": [BaseMessage, ...], "candidates": [...]}
    """
    data = payload.get("input", payload) or {}
    msgs = data.get("messages") or []
    if isinstance(msgs, dict):  # sometimes a single message object is sent
        msgs = [msgs]
    norm_msgs = [_to_message(m) if not isinstance(m, BaseMessage) else m for m in msgs]

    # Ensure we always have at least one message
    if not norm_msgs:
        norm_msgs = [HumanMessage(content="")]

    state = {"messages": norm_msgs}

    # pass-through optional fields you use (companies/candidates)
    if "candidates" in data:
        state["candidates"] = data["candidates"]
    elif "companies" in data:
        state["candidates"] = data["companies"]

    # NEW: Feature 18 — small synchronous head upsert+enrich.
    # When ICP Finder is enabled, defer upsert/enrichment until after ICP intake completes
    try:
        inds = _collect_industry_terms(state.get("messages"))
        if inds and STAGING_UPSERT_MODE != "off":
            if ENABLE_ICP_INTAKE:
                # Defer during ICP Finder; allow explicit override via keyword
                text = " ".join([(m.content or "") for m in norm_msgs if isinstance(m, HumanMessage)])
                if re.search(r"\brun enrichment\b", text, flags=re.IGNORECASE):
                    logger.info("ICP Finder override: running enrichment for inds=%s", inds)
                else:
                    # Enqueue nightly upsert of the full set while deferring immediate run
                    try:
                        from src.jobs import enqueue_staging_upsert
                        # Resolve tenant best-effort; OK to be None
                        tid = None
                        try:
                            info = asyncio.run(get_odoo_connection_info(email=None, claim_tid=None))
                            tid = info.get("tenant_id") if isinstance(info, dict) else None
                        except Exception:
                            tid = None
                        enqueue_staging_upsert(tid, inds)
                        logger.info("Queued nightly staging_upsert for inds=%s (Finder deferral)", inds)
                    except Exception as _qe:
                        logger.info("enqueue nightly staging_upsert failed: %s", _qe)
                    logger.info("Deferring staging upsert/enrich while ICP Finder is active; inds=%s", inds)
                    return state
            # Upsert only the first `head` records synchronously (no enrichment yet)
            head = max(0, int(UPSERT_SYNC_LIMIT))
            if head > 0:
                ids = upsert_by_industries_head(inds, limit=head)
                if ids:
                    logger.info("Upserted(head=%d) %d companies for industries=%s", head, len(ids), inds)
                    state["sync_head_company_ids"] = ids
                    # Immediate enrichment for the head set (non-blocking)
                    _trigger_enrichment_async(ids)
            # Always enqueue remaining for nightly processing
            try:
                from src.jobs import enqueue_staging_upsert
                # Resolve tenant best-effort; OK to be None
                tid = None
                try:
                    info = asyncio.run(get_odoo_connection_info(email=None, claim_tid=None))
                    tid = info.get("tenant_id") if isinstance(info, dict) else None
                except Exception:
                    tid = None
                enqueue_staging_upsert(tid, inds)
            except Exception as _qe:
                logger.info("enqueue nightly staging_upsert failed: %s", _qe)
    except Exception as _e:
        # Never block the chat flow; log and continue
        logger.warning("input-normalization staging sync failed: %s", _e)

    return state

# LangServe setup removed: previously mounted /agent when ENABLE_LANGSERVE_IN_APP was true.

@app.get("/info")
async def info(_: dict = Depends(require_optional_identity)):
    # Expose capability hints and current auth mode (no secrets)
    checkpoint_enabled = True if CHECKPOINT_DIR else False
    dev_bypass = os.getenv("DEV_AUTH_BYPASS", "false").lower() in ("1", "true", "yes", "on")
    issuer = (os.getenv("NEXIUS_ISSUER") or "").strip() or None
    audience = (os.getenv("NEXIUS_AUDIENCE") or "").strip() or None
    return {
        "ok": True,
        "checkpoint_enabled": checkpoint_enabled,
        "auth": {"dev_bypass": dev_bypass, "issuer": issuer, "audience": audience},
    }


# --- Keyset pagination endpoints ---
@app.get("/scores/latest")
async def scores_latest(limit: int = Query(50, ge=1, le=200), afterScore: float | None = None, afterId: int | None = None, _: dict = Depends(require_auth)):
    """List latest lead scores with keyset pagination.

    Returns { items: [...], nextCursor: { afterScore, afterId } | null }
    """
    from src.database import get_conn
    items: list[dict] = []
    with get_conn() as conn, conn.cursor() as cur:
        if afterScore is None or afterId is None:
            cur.execute(
                """
                SELECT s.company_id, s.score, s.bucket, s.rationale
                FROM lead_scores s
                ORDER BY s.score DESC, s.company_id DESC
                LIMIT %s
                """,
                (limit,),
            )
        else:
            cur.execute(
                """
                SELECT s.company_id, s.score, s.bucket, s.rationale
                FROM lead_scores s
                WHERE (s.score, s.company_id) < (%s, %s)
                ORDER BY s.score DESC, s.company_id DESC
                LIMIT %s
                """,
                (afterScore, afterId, limit),
            )
        rows = cur.fetchall() or []
        for r in rows:
            items.append({"company_id": r[0], "score": float(r[1]), "bucket": r[2], "rationale": r[3]})
    next_cursor = None
    if items and len(items) == limit:
        last = items[-1]
        next_cursor = {"afterScore": last["score"], "afterId": last["company_id"]}
    return {"items": items, "nextCursor": next_cursor}


@app.get("/candidates/latest")
async def candidates_latest(
    limit: int = Query(50, ge=1, le=200),
    afterUpdatedAt: datetime | None = None,
    afterId: int | None = None,
    industry: str | None = None,
    _: dict = Depends(require_auth),
):
    """List latest companies (optionally filtered by industry) with keyset pagination.

    Returns { items: [...], nextCursor: { afterUpdatedAt, afterId } | null }
    """
    items: list[dict] = []
    where = []
    params: list = []
    if industry and industry.strip():
        where.append("LOWER(industry_norm) = LOWER(%s)")
        params.append(industry.strip())
    where_sql = (" WHERE " + " AND ".join(where)) if where else ""
    with get_conn() as conn, conn.cursor() as cur:
        if afterUpdatedAt is None or afterId is None:
            cur.execute(
                f"""
                SELECT company_id, name, industry_norm, website_domain, last_seen
                FROM companies
                {where_sql}
                ORDER BY last_seen DESC NULLS LAST, company_id DESC
                LIMIT %s
                """,
                (*params, limit),
            )
        else:
            cur.execute(
                f"""
                SELECT company_id, name, industry_norm, website_domain, last_seen
                FROM companies
                {where_sql} {' AND ' if where_sql else ' WHERE '} (last_seen, company_id) < (%s, %s)
                ORDER BY last_seen DESC NULLS LAST, company_id DESC
                LIMIT %s
                """,
                (*params, afterUpdatedAt, afterId, limit),
            )
        rows = cur.fetchall() or []
        for r in rows:
            items.append(
                {
                    "company_id": r[0],
                    "name": r[1],
                    "industry_norm": r[2],
                    "website_domain": r[3],
                    "last_seen": r[4].isoformat() if isinstance(r[4], datetime) else None,
                }
            )
    next_cursor = None
    if items and len(items) == limit:
        last = items[-1]
        next_cursor = {"afterUpdatedAt": last["last_seen"], "afterId": last["company_id"]}
    return {"items": items, "nextCursor": next_cursor}


@app.get("/metrics")
async def metrics(_: dict = Depends(require_auth)):
    """Light metrics for ops and dashboards with richer stats."""
    out = {
        "job_queue_depth": 0,
        "jobs_processed_total": 0,
        "lead_scores_total": 0,
        "rows_per_min": None,
        "p95_job_ms": None,
        "chat_ttfb_p95_ms": None,
    }
    with get_conn() as conn, conn.cursor() as cur:
        try:
            cur.execute("SELECT COUNT(*) FROM background_jobs WHERE status='queued'")
            out["job_queue_depth"] = int((cur.fetchone() or [0])[0] or 0)
        except Exception:
            pass
        try:
            cur.execute("SELECT COALESCE(SUM(processed),0) FROM background_jobs WHERE job_type='staging_upsert' AND status='done'")
            out["jobs_processed_total"] = int((cur.fetchone() or [0])[0] or 0)
        except Exception:
            pass
        try:
            cur.execute("SELECT COUNT(*) FROM lead_scores")
            out["lead_scores_total"] = int((cur.fetchone() or [0])[0] or 0)
        except Exception:
            pass
        # rows/min and p95 job duration from recent completed jobs
        try:
            cur.execute(
                """
                SELECT processed, EXTRACT(EPOCH FROM (ended_at - started_at)) AS secs
                FROM background_jobs
                WHERE job_type='staging_upsert' AND status='done' AND started_at IS NOT NULL AND ended_at IS NOT NULL
                ORDER BY job_id DESC
                LIMIT 20
                """
            )
            rows = cur.fetchall() or []
            rates = []
            durs = []
            for r in rows:
                p = (r[0] or 0)
                s = float(r[1] or 0.0)
                if s > 0:
                    rates.append((p / s) * 60.0)
                    durs.append(s * 1000.0)
            if rates:
                out["rows_per_min"] = sum(rates) / len(rates)
            if durs:
                durs_sorted = sorted(durs)
                # Use nearest-rank method for small N: ceil(p*N)-1 (0-indexed)
                n = len(durs_sorted)
                k = max(0, math.ceil(0.95 * n) - 1)
                out["p95_job_ms"] = durs_sorted[k]
        except Exception:
            pass
        # Chat TTFB p95 from recent run_event_logs stage='chat'/event='ttfb'
        try:
            cur.execute(
                """
                SELECT duration_ms FROM run_event_logs
                WHERE stage='chat' AND event='ttfb' AND ts > NOW() - INTERVAL '7 days' AND duration_ms IS NOT NULL
                ORDER BY duration_ms
                LIMIT 1000
                """
            )
            vals = sorted(int(r[0]) for r in (cur.fetchall() or []) if r and r[0] is not None)
            if vals:
                # Keep p95 method consistent with test expectation: floor(0.95*(n-1))
                k = max(0, int(0.95 * (len(vals) - 1)))
                out["chat_ttfb_p95_ms"] = float(vals[k])
        except Exception:
            pass
    # Structured metrics log (best-effort)
    try:
        logging.getLogger("metrics").info("%s", out)
    except Exception:
        pass
    return out

@app.post("/metrics/ttfb")
async def metric_ttfb(body: dict, claims: dict = Depends(require_optional_identity)):
    """Allow FE to record chat TTFB (first token) as an event for p95 aggregation."""
    try:
        ttfb_ms = int((body or {}).get("ttfb_ms"))
    except Exception:
        raise HTTPException(status_code=400, detail="ttfb_ms integer required")
    # resolve tenant
    email = claims.get("email") or claims.get("preferred_username") or claims.get("sub")
    claim_tid = claims.get("tenant_id")
    tid = None
    try:
        info = await get_odoo_connection_info(email=email, claim_tid=claim_tid)
        tid = info.get("tenant_id")
    except Exception:
        tid = claim_tid
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO run_event_logs(run_id, tenant_id, stage, company_id, event, status, error_code, duration_ms, trace_id, extra)
                VALUES (0, %s, 'chat', NULL, 'ttfb', 'ok', NULL, %s, NULL, NULL)
                """,
                (tid if tid is not None else 0, ttfb_ms),
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"persist failed: {e}")
    return {"ok": True}


# Jobs API for staging_upsert (nightly queued)
@app.post("/jobs/staging_upsert")
async def jobs_staging_upsert(body: dict, claims: dict = Depends(require_optional_identity)):
    terms = (body or {}).get("terms") or []
    if not isinstance(terms, list) or not terms:
        raise HTTPException(status_code=400, detail="terms[] required")
    # resolve tenant best-effort
    email = claims.get("email") or claims.get("preferred_username") or claims.get("sub")
    claim_tid = claims.get("tenant_id")
    info = await get_odoo_connection_info(email=email, claim_tid=claim_tid)
    from src.jobs import enqueue_staging_upsert
    res = enqueue_staging_upsert(info.get("tenant_id"), terms)
    return res


@app.get("/jobs/{job_id}")
async def jobs_status(job_id: int, _: dict = Depends(require_optional_identity)):
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT job_id, job_type, status, processed, total, error, created_at, started_at, ended_at FROM background_jobs WHERE job_id=%s",
            (job_id,),
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="job not found")
        cols = [d[0] for d in cur.description]
        return dict(zip(cols, row))

@app.get("/whoami")
async def whoami(claims: dict = Depends(require_auth)):
    return {
        "sub": claims.get("sub"),
        "email": claims.get("email"),
        "tenant_id": claims.get("tenant_id"),
        "roles": claims.get("roles", []),
    }

@app.get("/onboarding/verify_odoo")
async def verify_odoo(claims: dict = Depends(require_identity)):
    """Verify Odoo mapping + connectivity for the current session.

    Aligns with PRD/DevPlan: does not require a tenant_id claim; resolves
    tenant via DSN→odoo_connections, claim, or email mapping. Uses
    odoo_connection_info for smoke test.
    """
    email = claims.get("email") or claims.get("preferred_username") or claims.get("sub")
    claim_tid = claims.get("tenant_id")

    # Use shared resolver + smoke test
    info = await get_odoo_connection_info(email=email, claim_tid=claim_tid)
    tid = info.get("tenant_id")

    # Determine whether an active mapping exists
    exists = False
    if tid is not None:
        try:
            with get_conn() as conn, conn.cursor() as cur:
                cur.execute("SELECT active FROM odoo_connections WHERE tenant_id=%s", (tid,))
                row = cur.fetchone()
                exists = bool(row and row[0])
        except Exception as e:
            logger.exception("Odoo verify DB lookup failed tenant_id=%s", tid)
            return {"tenant_id": tid, "exists": False, "ready": False, "error": str(e)}

    smoke = bool((info.get("odoo") or {}).get("ready"))
    error = (info.get("odoo") or {}).get("error")

    out = {
        "tenant_id": tid,
        "exists": exists,
        "smoke": smoke,
        "ready": bool(exists and smoke),
        "error": error,
    }
    try:
        if out.get("tenant_id") is not None and out.get("ready"):
            os.environ["DEFAULT_TENANT_ID"] = str(out.get("tenant_id"))
    except Exception:
        pass
    return out


@app.get("/debug/tenant")
async def debug_tenant(claims: dict = Depends(require_auth)):
    """Return current user identity, tenant mapping, and Odoo connectivity status."""
    email = claims.get("email") or claims.get("preferred_username")
    tid = claims.get("tenant_id")
    roles = claims.get("roles", [])

    db_name = None
    mapping_exists = False
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT db_name FROM odoo_connections WHERE tenant_id=%s", (tid,))
            row = cur.fetchone()
            if row:
                mapping_exists = True
                db_name = row[0]
    except Exception as e:
        return {
            "email": email,
            "tenant_id": tid,
            "roles": roles,
            "odoo": {"exists": False, "ready": False, "error": f"mapping fetch failed: {e}"},
        }

    # Try connectivity
    ready = False
    error = None
    try:
        store = OdooStore(tenant_id=int(tid))
        await store.connectivity_smoke_test()
        ready = True
    except Exception as e:
        error = str(e)

    return {
        "email": email,
        "tenant_id": tid,
        "roles": roles,
        "odoo": {"exists": mapping_exists, "db_name": db_name, "ready": ready, "error": error},
    }


@app.get("/session/odoo_info")
async def session_odoo_info(claims: dict = Depends(require_optional_identity)):
    email = claims.get("email") or claims.get("preferred_username") or claims.get("sub")
    claim_tid = claims.get("tenant_id")
    info = await get_odoo_connection_info(email=email, claim_tid=claim_tid)
    try:
        if info.get("tenant_id") is not None and (info.get("odoo") or {}).get("ready"):
            os.environ["DEFAULT_TENANT_ID"] = str(info.get("tenant_id"))
    except Exception:
        pass
    return info


@app.post("/onboarding/repair_admin")
async def onboarding_repair_admin(body: dict, _: dict = Depends(require_auth)):
    """Reset the admin login/password for a tenant's Odoo DB via XML-RPC.

    Body: { tenant_id: int, email: str, password: str }
    Requires: ODOO_SERVER_URL, ODOO_TEMPLATE_ADMIN_LOGIN, ODOO_TEMPLATE_ADMIN_PASSWORD
    """
    tenant_id = (body or {}).get("tenant_id")
    email = (body or {}).get("email")
    password = (body or {}).get("password")
    if not tenant_id or not email or not password:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="tenant_id, email, password required")
    server = (os.getenv("ODOO_SERVER_URL") or "").rstrip("/")
    admin_login = os.getenv("ODOO_TEMPLATE_ADMIN_LOGIN", "admin")
    admin_pw = os.getenv("ODOO_TEMPLATE_ADMIN_PASSWORD")
    if not server or not admin_pw:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail="Missing ODOO server or template admin credentials")
    # Resolve db_name for tenant
    db_name = None
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT db_name FROM odoo_connections WHERE tenant_id=%s", (tenant_id,))
        r = cur.fetchone()
        db_name = r[0] if r and r[0] else None
    if not db_name:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="No db_name for tenant")
    # XML-RPC update
    import xmlrpc.client
    common = xmlrpc.client.ServerProxy(f"{server}/xmlrpc/2/common")
    uid = common.authenticate(db_name, admin_login, admin_pw, {})
    if not uid:
        from fastapi import HTTPException
        raise HTTPException(status_code=403, detail="Template admin auth failed (check dbfilter/password)")
    models = xmlrpc.client.ServerProxy(f"{server}/xmlrpc/2/object")
    ids = models.execute_kw(db_name, uid, admin_pw, 'res.users', 'search', [[['login', '=', admin_login]]], {'limit': 1}) or [2]
    models.execute_kw(db_name, uid, admin_pw, 'res.users', 'write', [ids, {'login': email, 'password': password}])
    try:
        recs = models.execute_kw(db_name, uid, admin_pw, 'res.users', 'read', [ids, ['partner_id']])
        if recs and recs[0].get('partner_id'):
            models.execute_kw(db_name, uid, admin_pw, 'res.partner', 'write', [[recs[0]['partner_id'][0]], {'email': email}])
    except Exception:
        pass
    return {"ok": True, "db_name": db_name}


@app.post("/onboarding/verify_admin_login")
async def onboarding_verify_admin_login(body: dict, _: dict = Depends(require_auth)):
    """Verify that the provided email/password can authenticate to the tenant's Odoo DB.

    Body: { tenant_id?: int, email: str, password: str }
    Resolves tenant_id from body or from odoo_connections via email mapping when omitted.
    """
    email = (body or {}).get("email") or ""
    password = (body or {}).get("password") or ""
    tenant_id = (body or {}).get("tenant_id")
    if not email or not password:
        raise HTTPException(status_code=400, detail="email and password required")
    # Resolve tenant_id if not provided
    if tenant_id is None:
        try:
            with get_conn() as conn, conn.cursor() as cur:
                cur.execute("SELECT tenant_id FROM tenant_users WHERE user_id=%s LIMIT 1", (email,))
                r = cur.fetchone()
                if r:
                    tenant_id = int(r[0])
        except Exception:
            tenant_id = tenant_id
    if tenant_id is None:
        raise HTTPException(status_code=404, detail="tenant_id not found for email")
    # Resolve db_name for tenant
    db_name = None
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT db_name FROM odoo_connections WHERE tenant_id=%s", (tenant_id,))
        r = cur.fetchone()
        db_name = r[0] if r and r[0] else None
    if not db_name:
        raise HTTPException(status_code=404, detail="No db_name for tenant")
    server = (os.getenv("ODOO_SERVER_URL") or "").rstrip("/")
    if not server:
        raise HTTPException(status_code=500, detail="Missing ODOO_SERVER_URL")
    # XML-RPC authenticate
    try:
        import xmlrpc.client
        common = xmlrpc.client.ServerProxy(f"{server}/xmlrpc/2/common")
        uid = common.authenticate(db_name, email, password, {})
        ok = bool(uid)
        return {"ok": ok, "tenant_id": tenant_id, "db_name": db_name, "uid": uid or None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"XML-RPC auth failed: {e}")


@app.post("/onboarding/first_login")
async def onboarding_first_login(
    background: BackgroundTasks, claims: dict = Depends(require_optional_identity)
):
    email = claims.get("email") or claims.get("preferred_username")
    # Ignore tenant_id from token to avoid reliance on claim
    tenant_id_claim = None

    # If we already have an onboarding record for this tenant in progress or ready,
    # avoid enqueueing duplicate background tasks. Resolve candidate tenant ID from
    # DSN→odoo_connections mapping first, then fall back to existing user mapping by email.
    try:
        candidate_tid = None
        # DSN-based mapping: if ODOO_POSTGRES_DSN points at a specific DB, find the active mapping
        from src.settings import ODOO_POSTGRES_DSN
        try:
            inferred_db = None
            if ODOO_POSTGRES_DSN:
                from urllib.parse import urlparse
                u = urlparse(ODOO_POSTGRES_DSN)
                inferred_db = (u.path or "/").lstrip("/") or None
            if inferred_db:
                with get_conn() as conn, conn.cursor() as cur:
                    cur.execute("SELECT tenant_id FROM odoo_connections WHERE db_name=%s AND active=TRUE LIMIT 1", (inferred_db,))
                    row = cur.fetchone()
                    if row:
                        candidate_tid = int(row[0])
        except Exception:
            candidate_tid = candidate_tid  # keep any value already found

        # Fallback to email → tenant_users mapping
        if candidate_tid is None:
            with get_conn() as conn, conn.cursor() as cur:
                cur.execute("SELECT tenant_id FROM tenant_users WHERE user_id=%s LIMIT 1", (email,))
                row = cur.fetchone()
                if row:
                    candidate_tid = int(row[0])

        if candidate_tid is not None:
            current = get_onboarding_status(int(candidate_tid))
            # Dedup for all known in-progress/ready states
            if current and current.get("status") in {"provisioning", "syncing", "ready", "starting", "creating_odoo", "configuring_oidc", "seeding"}:
                logger.info(
                    "onboarding:first_login dedup tenant_id=%s inferred_db=%s email=%s status=%s",
                    candidate_tid,
                    locals().get("inferred_db"),
                    email,
                    current.get("status"),
                )
                return {"status": current.get("status"), "tenant_id": current.get("tenant_id"), "error": current.get("error")}
    except Exception:
        # Non-blocking; proceed to kickoff if status lookup fails
        pass

    async def _run():
        # No password available here; only register flow can pass one
        await handle_first_login(email, tenant_id_claim, user_password=None)

    background.add_task(_run)
    return {"status": "provisioning"}


@app.get("/onboarding/status")
async def onboarding_status(claims: dict = Depends(require_optional_identity)):
    # Resolve tenant id primarily via DSN→odoo_connections; fallback to claim or email mapping
    email = claims.get("email") or claims.get("preferred_username") or claims.get("sub")
    logger.info("onboarding_status: enter email=%s claim_tid=%s", email, claims.get("tenant_id"))
    tid = None
    try:
        from src.settings import ODOO_POSTGRES_DSN
        inferred_db = None
        if ODOO_POSTGRES_DSN:
            from urllib.parse import urlparse
            u = urlparse(ODOO_POSTGRES_DSN)
            inferred_db = (u.path or "/").lstrip("/") or None
        if inferred_db:
            with get_conn() as conn, conn.cursor() as cur:
                cur.execute("SELECT tenant_id FROM odoo_connections WHERE db_name=%s AND active=TRUE LIMIT 1", (inferred_db,))
                row = cur.fetchone()
                if row:
                    tid = int(row[0])
    except Exception:
        tid = None

    if tid is None:
        tid = claims.get("tenant_id") or getattr(getattr(app, "state", object()), "tenant_id", None)

    if tid is None:
        try:
            with get_conn() as conn, conn.cursor() as cur:
                cur.execute("SELECT tenant_id FROM tenant_users WHERE user_id=%s LIMIT 1", (email,))
                row = cur.fetchone()
                if row:
                    tid = int(row[0])
        except Exception:
            tid = tid

    # Guard: if tenant_id is still unknown, return a provisioning placeholder instead of 500
    if tid is None:
        logger.info("onboarding_status: no tenant yet for email=%s → returning provisioning placeholder", email)
        return {"tenant_id": None, "status": "provisioning", "error": None}

    try:
        res = get_onboarding_status(int(tid))
        # Back-compat: normalize legacy 'complete' to 'ready' for UI gate
        try:
            if (res or {}).get("status") == "complete":
                res["status"] = "ready"
        except Exception:
            pass
        logger.info("onboarding_status: tenant_id=%s status=%s", tid, res.get("status"))
        return res
    except Exception as e:
        logger.exception("Onboarding status failed tenant_id=%s", tid)
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=500, content={"tenant_id": tid, "status": "error", "error": str(e)})


@app.get("/tenants/{tenant_id}")
async def tenant_status(tenant_id: int, _: dict = Depends(require_optional_identity)):
    """PRD alias for onboarding status by explicit tenant id."""
    try:
        res = get_onboarding_status(int(tenant_id))
        # Back-compat normalization
        try:
            if (res or {}).get("status") == "complete":
                res["status"] = "ready"
        except Exception:
            pass
        return res
    except Exception as e:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=500, content={"tenant_id": tenant_id, "status": "error", "error": str(e)})


# --- Role-based access helpers and ICP endpoints ---
def require_roles(allowed: set[str]):
    async def _dep(request: Request):
        roles = getattr(request.state, "roles", []) or []
        if not any(r in allowed for r in roles):
            raise HTTPException(status_code=403, detail="Insufficient role")
        return True
    return _dep


@app.post("/api/icp/by-ssic")
async def api_icp_by_ssic(payload: dict):
    import src.icp as icp_module

    terms = (payload or {}).get("terms")
    if not isinstance(terms, list):
        terms = []
    norm_terms = [t.strip().lower() for t in terms if isinstance(t, str) and t.strip()]
    matched = icp_module._find_ssic_codes_by_terms(norm_terms)
    codes = [code for code, _title, _score in matched]
    acra = icp_module._select_acra_by_ssic_codes(codes)
    return {
        "matched_ssic": [{"code": c, "title": t, "score": s} for c, t, s in matched],
        "acra_candidates": acra,
    }


@app.get("/icp/rules")
async def list_icp_rules(_: dict = Depends(require_auth), request: Request = None):
    # List ICP rules for current tenant (viewer allowed)
    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        tid = getattr(request.state, "tenant_id", None)
        if tid:
            try:
                await conn.execute("SELECT set_config('request.tenant_id', $1, true)", tid)
            except Exception:
                pass
        rows = await conn.fetch(
            "SELECT rule_id, tenant_id, name, payload, created_at FROM icp_rules ORDER BY created_at DESC LIMIT 50"
        )
    return [dict(r) for r in rows]


@app.post("/icp/rules")
async def upsert_icp_rule(item: dict, _: dict = Depends(require_auth), __: bool = Depends(require_roles({"ops", "admin"})), request: Request = None):
    # Upsert ICP rule for tenant (ops/admin only)
    name = (item or {}).get("name") or "Default ICP"
    payload = (item or {}).get("payload") or {}
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="payload must be an object")
    tid = getattr(request.state, "tenant_id", None)
    if not tid:
        raise HTTPException(status_code=400, detail="missing tenant context")
    with get_conn() as conn, conn.cursor() as cur:
        try:
            cur.execute("SELECT set_config('request.tenant_id', %s, true)", (tid,))
        except Exception:
            pass
        cur.execute(
            """
            INSERT INTO icp_rules(tenant_id, name, payload)
            VALUES (%s, %s, %s)
            ON CONFLICT (rule_id) DO NOTHING
            """,
            (tid, name, payload),
        )
    return {"ok": True}

@app.get("/health")
def health():
    return {"ok": True}


# --- Export endpoints (JSON/CSV) ---
@app.get("/export/latest_scores.json")
async def export_latest_scores_json(limit: int = 200, request: Request = None, claims: dict = Depends(require_identity)):
    # Resolve tenant from identity and Odoo mapping (supports tokens without tenant_id claim)
    email = claims.get("email") or claims.get("preferred_username") or claims.get("sub")
    claim_tid = claims.get("tenant_id")
    from app.odoo_connection_info import get_odoo_connection_info
    info = await get_odoo_connection_info(email=email, claim_tid=claim_tid)
    tid = info.get("tenant_id")
    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        # Set per-request tenant GUC for RLS
        try:
            if tid is not None:
                await conn.execute("SELECT set_config('request.tenant_id', $1, true)", tid)
        except Exception:
            pass
        # Require a resolved tenant id to avoid cross-tenant leakage
        if tid is None:
            return []
        rows = await conn.fetch(
            """
            SELECT c.company_id,
                   c.name,
                   c.website_domain,
                   c.industry_norm,
                   c.employees_est,
                   s.score,
                   s.bucket,
                   s.rationale,
                   -- Primary email from discovered lead emails (best-effort)
                   (
                     SELECT e.email
                     FROM lead_emails e
                     WHERE e.company_id = s.company_id
                     ORDER BY e.left_company NULLS FIRST, e.smtp_confidence DESC NULLS LAST
                     LIMIT 1
                   ) AS primary_email,
                   -- Basic contact person details (best-effort)
                   (
                     SELECT c2.full_name FROM contacts c2
                     WHERE c2.company_id = s.company_id AND c2.email IS NOT NULL
                     LIMIT 1
                   ) AS contact_name,
                   (
                     SELECT c2.job_title FROM contacts c2
                     WHERE c2.company_id = s.company_id AND c2.email IS NOT NULL
                     LIMIT 1
                   ) AS contact_title,
                   (
                     SELECT c2.linkedin_profile FROM contacts c2
                     WHERE c2.company_id = s.company_id AND c2.email IS NOT NULL
                     LIMIT 1
                   ) AS contact_linkedin,
                   (
                     SELECT c2.phone_number FROM contacts c2
                     WHERE c2.company_id = s.company_id AND c2.email IS NOT NULL
                     LIMIT 1
                   ) AS contact_phone
            FROM companies c
            JOIN lead_scores s ON s.company_id = c.company_id
            WHERE s.tenant_id = $2
            ORDER BY s.score DESC NULLS LAST
            LIMIT $1
            """,
            limit,
            tid,
        )
    return [dict(r) for r in rows]


@app.get("/export/latest_scores.csv")
async def export_latest_scores_csv(limit: int = 200, request: Request = None, claims: dict = Depends(require_identity)):
    # Resolve tenant from identity and Odoo mapping (supports tokens without tenant_id claim)
    email = claims.get("email") or claims.get("preferred_username") or claims.get("sub")
    claim_tid = claims.get("tenant_id")
    from app.odoo_connection_info import get_odoo_connection_info
    info = await get_odoo_connection_info(email=email, claim_tid=claim_tid)
    tid = info.get("tenant_id")
    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        try:
            if tid is not None:
                await conn.execute("SELECT set_config('request.tenant_id', $1, true)", tid)
        except Exception:
            pass
        # Require a resolved tenant id to avoid cross-tenant leakage
        if tid is None:
            # Return an empty CSV with headers
            rows = []
        else:
            rows = await conn.fetch(
                """
                SELECT c.company_id,
                       c.name,
                       c.industry_norm,
                       c.employees_est,
                       s.score,
                       s.bucket,
                       s.rationale,
                       -- Primary email from discovered lead emails (best-effort)
                       (
                         SELECT e.email
                         FROM lead_emails e
                         WHERE e.company_id = s.company_id
                         ORDER BY e.left_company NULLS FIRST, e.smtp_confidence DESC NULLS LAST
                         LIMIT 1
                       ) AS primary_email,
                       -- Basic contact person details (best-effort)
                       (
                         SELECT c2.full_name FROM contacts c2
                         WHERE c2.company_id = s.company_id AND c2.email IS NOT NULL
                         LIMIT 1
                       ) AS contact_name,
                       (
                         SELECT c2.job_title FROM contacts c2
                         WHERE c2.company_id = s.company_id AND c2.email IS NOT NULL
                         LIMIT 1
                       ) AS contact_title,
                       (
                         SELECT c2.linkedin_profile FROM contacts c2
                         WHERE c2.company_id = s.company_id AND c2.email IS NOT NULL
                         LIMIT 1
                       ) AS contact_linkedin,
                       (
                         SELECT c2.phone_number FROM contacts c2
                         WHERE c2.company_id = s.company_id AND c2.email IS NOT NULL
                         LIMIT 1
                       ) AS contact_phone
                FROM companies c
                JOIN lead_scores s ON s.company_id = c.company_id
                WHERE s.tenant_id = $2
                ORDER BY s.score DESC NULLS LAST
                LIMIT $1
                """,
                limit,
                tid,
            )
    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()) if rows else [
        "company_id","name","industry_norm","employees_est","score","bucket","rationale",
        "primary_email","contact_name","contact_title","contact_linkedin","contact_phone"
    ])
    writer.writeheader()
    for r in rows:
        writer.writerow(dict(r))
    return Response(content=buf.getvalue(), media_type="text/csv")


@app.post("/export/odoo/sync")
async def export_odoo_sync(body: dict | None = None, request: Request = None, claims: dict = Depends(require_identity)):
    """Export scored companies to Odoo for the current tenant.

    Body (optional): { "min_score": float, "limit": int }
    - Selects top scored rows for the tenant and upserts company + primary contact; creates a lead when score ≥ min_score.
    """
    min_score = 0.0
    limit = 100
    try:
        if isinstance(body, dict):
            if isinstance(body.get("min_score"), (int, float)):
                min_score = float(body["min_score"])  # type: ignore[index]
            if isinstance(body.get("limit"), int):
                limit = int(body["limit"])  # type: ignore[index]
    except Exception:
        pass

    # Resolve tenant context even if token lacks tenant_id
    email = claims.get("email") or claims.get("preferred_username") or claims.get("sub")
    claim_tid = claims.get("tenant_id")
    from app.odoo_connection_info import get_odoo_connection_info
    info = await get_odoo_connection_info(email=email, claim_tid=claim_tid)
    tid = info.get("tenant_id")

    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        # Set per-request tenant for RLS
        try:
            if tid is not None:
                await conn.execute("SELECT set_config('request.tenant_id', $1, true)", tid)
        except Exception:
            pass
        rows = await conn.fetch(
            """
            SELECT s.company_id, s.score, s.rationale,
                   c.name, c.uen, c.industry_norm, c.employees_est, c.revenue_bucket,
                   c.incorporation_year, c.website_domain,
                   (SELECT e.email FROM lead_emails e WHERE e.company_id=s.company_id LIMIT 1) AS primary_email
            FROM lead_scores s
            JOIN companies c ON c.company_id = s.company_id
            WHERE s.tenant_id = $2
            ORDER BY s.score DESC NULLS LAST
            LIMIT $1
            """,
            limit,
            tid,
        )
    # Use tenant-scoped Odoo mapping
    store = OdooStore(tenant_id=int(tid)) if tid is not None else OdooStore()
    exported = 0
    errors: list[dict] = []
    for r in rows:
        try:
            odoo_id = await store.upsert_company(
                r["name"],
                r["uen"],
                industry_norm=r["industry_norm"],
                employees_est=r["employees_est"],
                revenue_bucket=r["revenue_bucket"],
                incorporation_year=r["incorporation_year"],
                website_domain=r["website_domain"],
            )
            try:
                import logging as _lg
                _lg.getLogger("onboarding").info("export: upsert company partner_id=%s name=%s", odoo_id, r["name"])
            except Exception:
                pass
            if r["primary_email"]:
                try:
                    await store.add_contact(odoo_id, r["primary_email"])
                    try:
                        import logging as _lg
                        _lg.getLogger("onboarding").info("export: add_contact email=%s partner_id=%s", r["primary_email"], odoo_id)
                    except Exception:
                        pass
                except Exception as _c_exc:
                    errors.append({"company_id": r["company_id"], "error": f"contact: {_c_exc}"})
            await store.merge_company_enrichment(odoo_id, {})
            sc = float(r["score"] or 0)
            try:
                await store.create_lead_if_high(
                    odoo_id,
                    r["name"],
                    sc,
                    {},
                    r["rationale"] or "",
                    r["primary_email"],
                    threshold=min_score,
                )
                try:
                    import logging as _lg
                    _lg.getLogger("onboarding").info("export: lead check partner_id=%s score=%.2f threshold=%.2f", odoo_id, sc, min_score)
                except Exception:
                    pass
            except Exception as _l_exc:
                errors.append({"company_id": r["company_id"], "error": f"lead: {_l_exc}"})
            exported += 1
        except Exception as e:
            errors.append({"company_id": r["company_id"], "error": str(e)})
    return {"exported": exported, "count": len(rows), "min_score": min_score, "errors": errors}
@app.middleware("http")
async def auth_guard(request: Request, call_next):
    # Always let CORS preflight through
    if request.method.upper() == "OPTIONS":
        return await call_next(request)
    # Allow unauthenticated for health and docs
    open_paths = {"/health", "/docs", "/openapi.json"}
    if request.url.path in open_paths:
        return await call_next(request)
    # In production, do not double-enforce here; route dependencies perform auth
    return await call_next(request)

# --- Admin/ops: rotate per-tenant Odoo API key ---
@app.post("/tenants/{tenant_id}/odoo/api-key/rotate")
async def rotate_odoo_key(tenant_id: int = Path(...), _: dict = Depends(require_auth)):
    import secrets
    new_secret = secrets.token_urlsafe(32)
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("UPDATE odoo_connections SET secret=%s WHERE tenant_id=%s", (new_secret, tenant_id))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"rotate failed: {e}")
    return Response(status_code=204)


# --- Shortlist status and ad-hoc scheduler trigger ---
@app.get("/shortlist/status")
async def shortlist_status(request: Request = None, claims: dict = Depends(require_optional_identity)):
    """Return shortlist freshness and size for the current tenant.

    Response: { last_refreshed_at: ISO8601|null, total_scored: int, tenant_id?: int }
    """
    # Resolve tenant from identity and Odoo mapping (same approach as export)
    email = claims.get("email") or claims.get("preferred_username") or claims.get("sub")
    claim_tid = claims.get("tenant_id")
    from app.odoo_connection_info import get_odoo_connection_info
    info = await get_odoo_connection_info(email=email, claim_tid=claim_tid)
    tid = info.get("tenant_id")

    # Simple per-tenant cache to avoid DB work and chatter; default TTL 5 minutes
    try:
        _SHORTLIST_CACHE  # type: ignore[name-defined]
    except NameError:  # first load
        _SHORTLIST_CACHE = {}
    try:
        _ttl = float(os.getenv("SHORTLIST_TTL_S", "300") or 300)
    except Exception:
        _ttl = 300.0

    # Serve cached value when fresh
    try:
        key = int(info.get("tenant_id")) if info.get("tenant_id") is not None else None
    except Exception:
        key = None
    if key in _SHORTLIST_CACHE:
        ts, cached = _SHORTLIST_CACHE.get(key, (0.0, None))
        if cached is not None and (time.time() - float(ts)) <= _ttl:
            return cached

    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        # Apply RLS tenant context if known
        try:
            if tid is not None:
                await conn.execute("SELECT set_config('request.tenant_id', $1, true)", tid)
        except Exception:
            pass
        # If we cannot resolve tenant, do not leak global counts
        if tid is None:
            total_scored = 0
            last_ts: datetime | None = None
            last_run_id = None
            last_run_status = None
            last_run_started_at = None
            last_run_ended_at = None
        else:
            # Count only this tenant's scored rows
            try:
                total_scored = int(await conn.fetchval("SELECT COUNT(*) FROM lead_scores WHERE tenant_id = $1", tid))
            except Exception:
                total_scored = 0

            # Last activity from this tenant's enrichment runs
            last_ts: datetime | None = None
            last_run_id = None
            last_run_status = None
            last_run_started_at = None
            last_run_ended_at = None

            try:
                row = await conn.fetchrow(
                    "SELECT run_id, status, started_at, ended_at FROM enrichment_runs WHERE tenant_id = $1 ORDER BY run_id DESC LIMIT 1",
                    tid,
                )
                if row:
                    last_run_id = row["run_id"]
                    last_run_status = row["status"]
                    last_run_started_at = row["started_at"]
                    last_run_ended_at = row["ended_at"]
                    if isinstance(last_run_started_at, datetime):
                        last_ts = last_run_started_at
            except Exception:
                last_ts = last_ts

    out = {
        "tenant_id": tid,
        "total_scored": total_scored,
        "last_refreshed_at": (last_ts.isoformat() if isinstance(last_ts, datetime) else None),
        "last_run_id": last_run_id,
        "last_run_status": last_run_status,
        "last_run_started_at": (last_run_started_at.isoformat() if isinstance(last_run_started_at, datetime) else None),
        "last_run_ended_at": (last_run_ended_at.isoformat() if isinstance(last_run_ended_at, datetime) else None),
    }
    # Update cache
    try:
        _SHORTLIST_CACHE[key] = (time.time(), out)
    except Exception:
        pass
    return out


@app.post("/scheduler/run_now")
async def scheduler_run_now(background: BackgroundTasks, claims: dict = Depends(require_auth)):
    """Trigger a background run for the current tenant (ad-hoc).

    Returns immediately with {status: "scheduled", tenant_id}.
    """
    # Resolve tenant from identity and Odoo mapping
    email = claims.get("email") or claims.get("preferred_username") or claims.get("sub")
    claim_tid = claims.get("tenant_id")
    from app.odoo_connection_info import get_odoo_connection_info
    info = await get_odoo_connection_info(email=email, claim_tid=claim_tid)
    tid = info.get("tenant_id")
    if tid is None:
        raise HTTPException(status_code=400, detail="Unable to resolve tenant for current session")

    try:
        # Import runner lazily to avoid import-time overhead unless used
        from scripts.run_nightly import run_tenant_partial  # type: ignore

        async def _run():
            try:
                import os
                # Process up to 10 now; leave remainder for nightly scheduler
                limit = 10
                try:
                    limit = int(os.getenv("RUN_NOW_LIMIT", "10") or 10)
                except Exception:
                    limit = 10
                await run_tenant_partial(int(tid), max_now=limit)
            except Exception as exc:
                logging.getLogger("nightly").exception("ad-hoc run failed tenant_id=%s: %s", tid, exc)

        background.add_task(_run)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"schedule failed: {e}")


# --- Observability endpoints (Feature 8) ---
@app.get("/runs/{run_id}/stats")
async def get_run_stats(run_id: int, _: dict = Depends(require_auth)):
    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        rows1 = await conn.fetch("SELECT * FROM run_stage_stats WHERE run_id=$1 ORDER BY stage", run_id)
        rows2 = await conn.fetch("SELECT * FROM run_vendor_usage WHERE run_id=$1 ORDER BY vendor", run_id)
    return {
        "run_id": run_id,
        "stage_stats": [dict(r) for r in rows1],
        "vendor_usage": [dict(r) for r in rows2],
    }


@app.get("/export/run_events.csv")
async def export_run_events(run_id: int, _: dict = Depends(require_auth)):
    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT run_id, tenant_id, stage, company_id, event, status, error_code, duration_ms, trace_id, extra, ts FROM run_event_logs WHERE run_id=$1 ORDER BY ts",
            run_id,
        )
    import csv as _csv, io as _io
    buf = _io.StringIO()
    w = _csv.writer(buf)
    w.writerow(["run_id","tenant_id","stage","company_id","event","status","error_code","duration_ms","trace_id","extra","ts"])
    for r in rows:
        w.writerow([
            r["run_id"], r["tenant_id"], r["stage"], r["company_id"], r["event"], r["status"],
            r["error_code"], r["duration_ms"], r["trace_id"], r["extra"], r["ts"]
        ])
    from fastapi.responses import Response
    return Response(content=buf.getvalue(), media_type="text/csv")


@app.get("/export/qa.csv")
async def export_qa(run_id: int, _: dict = Depends(require_auth)):
    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT run_id, tenant_id, company_id, bucket, checks, result, notes, created_at
            FROM qa_samples
            WHERE run_id=$1
            ORDER BY bucket, company_id
            """,
            run_id,
        )
    import csv as _csv, io as _io
    buf = _io.StringIO()
    w = _csv.writer(buf)
    w.writerow(["run_id","tenant_id","company_id","bucket","checks","result","notes","created_at"])
    for r in rows:
        w.writerow([
            r["run_id"], r["tenant_id"], r["company_id"], r["bucket"], r["checks"], r["result"], r["notes"], r["created_at"]
        ])
    from fastapi.responses import Response
    return Response(content=buf.getvalue(), media_type="text/csv")

    return {"status": "scheduled", "tenant_id": tid}


# --- Admin: kickoff full nightly run (optionally for a single tenant) ---
@app.post("/admin/runs/nightly")
async def admin_run_nightly(background: BackgroundTasks, request: Request, claims: dict = Depends(require_auth)):
    roles = claims.get("roles", []) or []
    if "admin" not in roles:
        raise HTTPException(status_code=403, detail="admin role required")
    # Optional tenant_id query param
    try:
        tenant_id = request.query_params.get("tenant_id")
        tenant_id = int(tenant_id) if tenant_id is not None else None
    except Exception:
        raise HTTPException(status_code=400, detail="invalid tenant_id")

    try:
        from scripts.run_nightly import run_all, run_tenant  # type: ignore

        async def _run_all():
            try:
                await run_all()
            except Exception as exc:
                logging.getLogger("nightly").exception("admin run_all failed: %s", exc)

        async def _run_one(tid: int):
            try:
                await run_tenant(tid)
            except Exception as exc:
                logging.getLogger("nightly").exception("admin run_tenant failed tenant_id=%s: %s", tid, exc)

        if tenant_id is None:
            background.add_task(_run_all)
        else:
            background.add_task(_run_one, tenant_id)
        return {"status": "scheduled", "tenant_id": tenant_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"schedule failed: {e}")
