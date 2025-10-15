# tools.py
import asyncio
import os
import json
import re
import time
import logging
from typing import Any, Dict, List, Optional, TypedDict
from urllib.parse import urljoin, urlparse

import httpx
import psycopg2
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool

# LangChain imports for AI-driven extraction
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilyCrawl, TavilyExtract
from langgraph.graph import END, StateGraph
from psycopg2.extras import Json
from tavily import TavilyClient
from src.obs import bump_vendor as _obs_bump, log_event as _log_obs_event
from src.retry import with_retry, RetryableError, CircuitBreaker, BackoffPolicy

from src.jina_reader import read_url as jina_read
from src.lusha_client import AsyncLushaClient, LushaError
from src.vendors.apify_linkedin import (
    run_sync_get_dataset_items as apify_run,
    build_queries as apify_build_queries,
    normalize_contacts as apify_normalize,
    contacts_via_company_chain as apify_contacts_via_chain,
)
from src.openai_client import get_embedding
from src.settings import (
    CRAWL_KEYWORDS,
    CRAWL_MAX_PAGES,
    CRAWLER_MAX_PAGES,
    CRAWLER_TIMEOUT_S,
    CRAWLER_USER_AGENT,
    ENABLE_TAVILY_FALLBACK,
    ENABLE_LUSHA_FALLBACK,
    ENABLE_APIFY_LINKEDIN,
    EXTRACT_CORPUS_CHAR_LIMIT,
    LANGCHAIN_MODEL,
    TEMPERATURE,
    LUSHA_API_KEY,
    LUSHA_PREFERRED_TITLES,
    PERSIST_CRAWL_CORPUS,
    POSTGRES_DSN,
    TAVILY_API_KEY,
    ZEROBOUNCE_API_KEY,
    RETRY_MAX_ATTEMPTS,
    RETRY_BASE_DELAY_MS,
    RETRY_MAX_DELAY_MS,
    CB_ERROR_THRESHOLD,
    CB_COOL_OFF_S,
    CB_GLOBAL_EXEMPT_VENDORS,
    APIFY_DATASET_FORMAT,
    APIFY_SYNC_TIMEOUT_S,
    CONTACT_TITLES,
    ICP_RULE_NAME,
    APIFY_DAILY_CAP,
)
from src.settings import (
    APIFY_USE_COMPANY_EMPLOYEE_CHAIN,
    LLM_MAX_CHUNKS,
    LLM_CHUNK_TIMEOUT_S,
    MERGE_DETERMINISTIC_TIMEOUT_S,
)
from src.settings import ENRICH_RECHECK_DAYS, ENRICH_SKIP_IF_ANY_HISTORY
from src.settings import ENRICH_AGENTIC, ENRICH_AGENTIC_MAX_STEPS

load_dotenv()

logger = logging.getLogger(__name__)
logger.info("🛠️  Initializing enrichment pipeline…")

# Simple in-memory cache for ZeroBounce to avoid duplicate calls per-run
ZB_CACHE: dict[str, dict] = {}

def _mask_email(e: str) -> str:
    try:
        import hashlib
        local, _, domain = (e or "").partition("@")
        if not domain:
            # Not an email shape; hash entire string
            h = hashlib.sha256((e or "").encode()).hexdigest()[:8]
            return f"hash:{h}"
        head = (local[:2] + "***") if local else "***"
        dom_parts = domain.split(".")
        if len(dom_parts) >= 2:
            dom = dom_parts[0][:1] + "***." + dom_parts[-1]
        else:
            dom = "***"
        return f"{head}@{dom}"
    except Exception:
        return "***redacted***"

def _redact_email_list(emails: list[str]) -> list[str]:
    return [_mask_email(e) for e in emails]

def _looks_like_domain(s: str | None) -> bool:
    try:
        v = (s or "").strip().lower()
        return bool(v and "." in v and not v.startswith("http"))
    except Exception:
        return False

def _company_query_name_from_state(state: dict) -> str:
    """Choose a human-ish company name to use for Apify queries (no domains).

    Priority:
      1) LLM-extracted name in state['data']['name'] if not domain-like
      2) state['company_name'] if not domain-like
      3) Derive label from website_domain/home apex (strip TLDs, www, split hyphens)
    """
    try:
        data = state.get("data") or {}
        nm = (data.get("name") or "").strip()
        if nm and not _looks_like_domain(nm):
            return nm
    except Exception:
        pass
    try:
        nm2 = (state.get("company_name") or "").strip()
        if nm2 and not _looks_like_domain(nm2):
            return nm2
    except Exception:
        pass
    # Fall back to deriving from domain/home
    dom = None
    try:
        dom = (data.get("website_domain") or state.get("home") or state.get("company_name") or "").strip()
    except Exception:
        dom = (state.get("home") or state.get("company_name") or "").strip()
    try:
        if dom and dom.startswith("http"):
            dom = urlparse(dom).netloc
        host = (dom or "").lower()
        if host.startswith("www."):
            host = host[4:]
        # take apex label (before first dot)
        label = host.split(".")[0]
        # de-hyphen, simple capitalization
        parts = [p for p in re.split(r"[-_]+", label) if p]
        if not parts:
            return label or ""
        # Title-case words of length > 2, keep acronyms as upper
        cleaned = " ".join([w.upper() if len(w) <= 2 else w.capitalize() for w in parts])
        return cleaned
    except Exception:
        return (dom or "").split(".")[0]

def _default_tenant_id() -> int | None:
    try:
        v = os.getenv("DEFAULT_TENANT_ID")
        return int(v) if v and v.isdigit() else None
    except Exception:
        return None

# Initialize Tavily clients (optional). If no API key, skip Tavily and rely on fallbacks.
if TAVILY_API_KEY:
    tavily_client = TavilyClient(TAVILY_API_KEY)
    tavily_crawl = TavilyCrawl(api_key=TAVILY_API_KEY)
    tavily_extract = TavilyExtract(api_key=TAVILY_API_KEY)
else:
    tavily_client = None  # type: ignore[assignment]
    tavily_crawl = None  # type: ignore[assignment]
    tavily_extract = None  # type: ignore[assignment]


def _fallback_extract_from_text(text: str) -> dict:
    """Lightweight, non-LLM extraction to salvage key fields when LLM times out.

    Extracts:
      - emails: simple regex
      - phone_number: loose international formats
      - website_domain: first URL/domain-like token
      - about_text: first ~2 sentences
    """
    out: Dict[str, Any] = {}
    try:
        # emails
        emails = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text or "")
        if emails:
            out["email"] = sorted(set(emails))[:5]
    except Exception:
        pass
    try:
        # phones (very loose)
        phones = re.findall(r"\+?\d[\d\s().-]{6,}\d", text or "")
        if phones:
            out["phone_number"] = sorted(set([p.strip() for p in phones]))[:5]
    except Exception:
        pass
    try:
        # domain or URL
        m = re.search(r"https?://[^\s]+", text or "")
        if not m:
            m = re.search(r"\b([a-z0-9-]+\.)+[a-z]{2,}\b", text or "", re.I)
        if m:
            out["website_domain"] = m.group(0)
    except Exception:
        pass
    try:
        # about: first 2 sentences (approx)
        s = (text or "").strip()
        if s:
            parts = re.split(r"(?<=[.!?])\s+", s)
            out["about_text"] = " ".join(parts[:2])[:500]
    except Exception:
        pass
    return out

# ---- Run context and vendor counters/caps ----
_RUN_CTX: dict[str, int | None] = {"run_id": None, "tenant_id": None}
_VENDOR_COUNTERS: dict[str, int] = {
    "tavily_queries": 0,
    "tavily_crawl_calls": 0,
    "tavily_extract_calls": 0,
    "lusha_lookups": 0,
    "apify_linkedin_calls": 0,
}
_VENDOR_CAPS: dict[str, int | None] = {"tavily_units": None, "contact_lookups": None}
_CB = CircuitBreaker(error_threshold=CB_ERROR_THRESHOLD, cool_off_s=CB_COOL_OFF_S)

# Default retry/backoff policy from env
_DEFAULT_RETRY_POLICY = BackoffPolicy(
    max_attempts=RETRY_MAX_ATTEMPTS,
    base_delay_ms=RETRY_BASE_DELAY_MS,
    max_delay_ms=RETRY_MAX_DELAY_MS,
)

_RUN_ANY_DEGRADED = False

def was_run_degraded() -> bool:
    return bool(_RUN_ANY_DEGRADED)


def _apify_calls_today(tenant_id: int | None) -> int:
    if not tenant_id:
        return 0
    try:
        with get_db_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT COALESCE(SUM(rv.calls),0)
                FROM run_vendor_usage rv
                JOIN enrichment_runs er USING(run_id, tenant_id)
                WHERE rv.tenant_id=%s AND rv.vendor='apify_linkedin'
                  AND er.started_at >= date_trunc('day', now())
                """,
                (int(tenant_id),),
            )
            row = cur.fetchone()
            return int(row[0] or 0) if row else 0
    except Exception:
        return 0


def _apify_cap_ok(tenant_id: int | None, need: int = 1) -> bool:
    try:
        cap = int(APIFY_DAILY_CAP)
    except Exception:
        cap = 50
    if cap <= 0:
        return True
    current = _apify_calls_today(tenant_id)
    return (current + need) <= cap


def icp_preferred_titles_for_tenant(tenant_id: int | None) -> List[str]:
    if not tenant_id:
        return []
    try:
        with get_db_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT payload->'preferred_titles'
                FROM icp_rules
                WHERE tenant_id=%s AND name=%s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (int(tenant_id), ICP_RULE_NAME),
            )
            row = cur.fetchone()
            if not row:
                return []
            val = row[0]
            if isinstance(val, list):
                return [str(x).strip() for x in val if (str(x) or "").strip()]
            import json as _json
            try:
                arr = _json.loads(val) if isinstance(val, str) else []
                if isinstance(arr, list):
                    return [str(x).strip() for x in arr if (str(x) or "").strip()]
            except Exception:
                pass
    except Exception:
        return []
    return []

def _cb_allows(tenant_id: int | None, vendor: str) -> bool:
    try:
        if (vendor or "").lower() in CB_GLOBAL_EXEMPT_VENDORS:
            return True
    except Exception:
        pass
    if not tenant_id:
        return True
    try:
        return _CB.allow(int(tenant_id), vendor)
    except Exception:
        return True

def set_run_context(run_id: int, tenant_id: int):
    _RUN_CTX["run_id"] = int(run_id)
    _RUN_CTX["tenant_id"] = int(tenant_id)

def set_vendor_caps(tavily_units: int | None = None, contact_lookups: int | None = None):
    _VENDOR_CAPS["tavily_units"] = tavily_units
    _VENDOR_CAPS["contact_lookups"] = contact_lookups

def get_vendor_counters() -> dict[str, int]:
    return dict(_VENDOR_COUNTERS)

def reset_vendor_counters():
    for k in _VENDOR_COUNTERS.keys():
        _VENDOR_COUNTERS[k] = 0

def _units_used(key: str) -> int:
    if key == "tavily_units":
        # Tightened definition: treat Tavily units as (search queries + extract calls).
        # Crawl discovery is not charged as an extra unit here since extract calls
        # usually dominate cost. Adjust if your billing model differs.
        return _VENDOR_COUNTERS["tavily_queries"] + _VENDOR_COUNTERS["tavily_extract_calls"]
    if key == "contact_lookups":
        return _VENDOR_COUNTERS["lusha_lookups"] + _VENDOR_COUNTERS["apify_linkedin_calls"]
    return 0

def _dec_cap(key: str, need: int = 1) -> bool:
    cap = _VENDOR_CAPS.get(key)
    if cap is None:
        return True
    return (_units_used(key) + need) <= cap

_warned_no_tavily = False

def _obs_vendor(vendor: str, calls: int = 1, errors: int = 0, *, rate_limit_hits: int = 0, quota_exhausted: bool = False):
    """Record vendor usage and emit relevant one-time warnings.

    Only warn about missing Tavily key when actually calling Tavily and key is not set,
    and do so at most once per process to avoid log spam.
    """
    rid = _RUN_CTX.get("run_id")
    tid = _RUN_CTX.get("tenant_id")
    if rid and tid:
        try:
            _obs_bump(int(rid), int(tid), vendor, calls=calls, errors=errors, rate_limit_hits=rate_limit_hits, quota_exhausted=quota_exhausted)
        except Exception:
            pass
    global _warned_no_tavily
    if (vendor.startswith("tavily")) and not TAVILY_API_KEY and not _warned_no_tavily:
        logger.warning(
            "⚠️  TAVILY_API_KEY not set; using deterministic/HTTP fallbacks for crawl/extract."
        )
        _warned_no_tavily = True

def _obs_log(stage: str, event: str, status: str, *, company_id: Optional[int] = None, error_code: Optional[str] = None, duration_ms: Optional[int] = None, extra: Optional[Dict[str, Any]] = None):
    try:
        rid = _RUN_CTX.get("run_id")
        tid = _RUN_CTX.get("tenant_id")
        if rid and tid:
            _log_obs_event(int(rid), int(tid), stage, event, status, company_id=company_id, error_code=error_code, duration_ms=duration_ms, trace_id=None, extra=extra)
    except Exception:
        pass

# Initialize LangChain LLM for AI extraction
# Use configured model; some models (e.g., gpt-5) do not accept an explicit
# temperature override, so omit the parameter in that case to avoid 400 errors.
def _make_chat_llm(model: str, temperature: float | None) -> ChatOpenAI:
    kwargs: dict = {"model": model}
    # Omit temperature for models that only support default behavior
    if temperature is not None and not model.lower().startswith("gpt-5"):
        kwargs["temperature"] = temperature
    return ChatOpenAI(**kwargs)

llm = _make_chat_llm(LANGCHAIN_MODEL, TEMPERATURE)
prompt_template = PromptTemplate(
    input_variables=["raw_content", "schema_keys", "instructions"],
    template=(
        "You are a data extraction agent.\n"
        "Given the following raw page content, extract the fields according to the schema keys and instructions,\n"
        "and return a JSON object with keys exactly matching the schema.\n\n"
        "Schema Keys: {schema_keys}\n"
        "Instructions: {instructions}\n\n"
        "Raw Content:\n{raw_content}\n"
    ),
)
extract_chain = prompt_template | llm | StrOutputParser()


def get_db_connection():
    return psycopg2.connect(dsn=POSTGRES_DSN)


def _ensure_email_cache_table(conn):
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS email_verification_cache (
                  email TEXT PRIMARY KEY,
                  status TEXT,
                  confidence FLOAT,
                  checked_at TIMESTAMPTZ DEFAULT now()
                );
                """
            )
    except Exception:
        pass


def _cache_get(conn, email: str) -> Optional[dict]:
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT status, confidence FROM email_verification_cache WHERE email=%s",
                (email,),
            )
            row = cur.fetchone()
            if row:
                return {
                    "email": email,
                    "status": row[0],
                    "confidence": float(row[1] or 0.0),
                    "source": "zerobounce-cache",
                }
    except Exception:
        return None
    return None


def _cache_set(conn, email: str, status: str, confidence: float) -> None:
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO email_verification_cache(email, status, confidence, checked_at)
                VALUES (%s,%s,%s, now())
                ON CONFLICT (email) DO UPDATE SET status=EXCLUDED.status, confidence=EXCLUDED.confidence, checked_at=now()
                """,
                (email, status, confidence),
            )
    except Exception:
        pass


# ---------- Contacts persistence helpers (DB-introspective) ----------
def _get_table_columns(conn, table_name: str) -> set:
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = %s
                """,
                (table_name,),
            )
            return {r[0] for r in cur.fetchall()}
    except Exception:
        return set()


def _insert_company_enrichment_run(conn, fields: dict) -> None:
    """Insert a row into company_enrichment_runs using only columns that exist.

    This guards against environments where certain columns (e.g., public_emails,
    verification_results, embedding) may be absent. It reads the live table
    columns and builds a minimal INSERT accordingly. Relies on DB defaults for
    run_timestamp, enrichment_id, etc.
    """
    try:
        cols = _get_table_columns(conn, "company_enrichment_runs")
        if not cols:
            return

        # Back-compat: some databases have a NOT NULL run_id on this table
        # that references enrichment_runs(run_id). If the column exists and
        # caller didn't provide one, create a new enrichment_runs row and use it.
        if "run_id" in cols and (
            "run_id" not in fields or fields.get("run_id") is None
        ):
            try:
                with conn.cursor() as cur:
                    # Ensure tenant context is set for RLS-aware inserts
                    try:
                        tid_guc = _default_tenant_id()
                        if tid_guc is not None:
                            cur.execute("SELECT set_config('request.tenant_id', %s, true)", (str(tid_guc),))
                    except Exception:
                        pass
                    # Ensure enrichment_runs table exists (idempotent create)
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS enrichment_runs (
                          run_id BIGSERIAL PRIMARY KEY,
                          started_at TIMESTAMPTZ DEFAULT now()
                        );
                        """
                    )
                    # Ensure tenant_id column exists if RLS/migrations are applied
                    try:
                        cur.execute("ALTER TABLE enrichment_runs ADD COLUMN IF NOT EXISTS tenant_id INT")
                    except Exception:
                        pass
                    # Insert a run row; include tenant_id when column exists
                    has_tenant_col = False
                    try:
                        cur.execute(
                            "SELECT 1 FROM information_schema.columns WHERE table_name='enrichment_runs' AND column_name='tenant_id'"
                        )
                        has_tenant_col = cur.fetchone() is not None
                    except Exception:
                        has_tenant_col = False

                    tid_val = _default_tenant_id()
                    if has_tenant_col and tid_val is not None:
                        cur.execute(
                            "INSERT INTO enrichment_runs(tenant_id) VALUES (%s) RETURNING run_id",
                            (tid_val,),
                        )
                    else:
                        cur.execute(
                            "INSERT INTO enrichment_runs DEFAULT VALUES RETURNING run_id"
                        )
                    rid = cur.fetchone()[0]
                    fields["run_id"] = rid
            except Exception:
                # If we fail to create a run_id, proceed; insert may still work
                pass

        keys = [k for k, v in fields.items() if k in cols and v is not None]
        if not keys:
            return
        placeholders = ",".join(["%s"] * len(keys))
        sql = f"INSERT INTO company_enrichment_runs ({', '.join(keys)}) VALUES ({placeholders})"
        with conn.cursor() as cur:
            cur.execute(sql, [fields[k] for k in keys])
    except Exception as e:
        # Surface but don't crash callers; they may have follow-up persistence
        logger.warning("insert company_enrichment_runs skipped", exc_info=True)


def _get_contact_stats(company_id: int):
    """Return (total_contacts, has_named_contact, founder_present).
    Uses best-effort checks based on available columns.
    """
    total = 0
    has_named = False
    founder_present = False
    conn = None
    try:
        conn = get_db_connection()
        cols = _get_table_columns(conn, "contacts")
        with conn, conn.cursor() as cur:
            # total contacts
            cur.execute(
                "SELECT COUNT(*) FROM contacts WHERE company_id=%s", (company_id,)
            )
            total = int(cur.fetchone()[0] or 0)

            # any named contact
            name_conds = []
            if "title" in cols:
                name_conds.append("(title IS NOT NULL AND title <> '')")
            if "full_name" in cols:
                name_conds.append("(full_name IS NOT NULL AND full_name <> '')")
            if "first_name" in cols:
                name_conds.append("(first_name IS NOT NULL AND first_name <> '')")
            if name_conds:
                q = (
                    "SELECT COUNT(*) FROM contacts WHERE company_id=%s AND ("
                    + " OR ".join(name_conds)
                    + ")"
                )
                cur.execute(q, (company_id,))
                has_named = int(cur.fetchone()[0] or 0) > 0

            # founder / leadership presence by title
            if "title" in cols:
                terms = [
                    (t or "").strip().lower()
                    for t in (LUSHA_PREFERRED_TITLES or "").split(",")
                    if (t or "").strip()
                ]
                # If titles list is empty, use a default set
                if not terms:
                    terms = [
                        "founder",
                        "co-founder",
                        "ceo",
                        "cto",
                        "owner",
                        "director",
                        "head of",
                        "principal",
                    ]
                like_clauses = ["LOWER(title) LIKE %s" for _ in terms]
                params = [f"%{t}%" for t in terms]
                q = (
                    "SELECT COUNT(*) FROM contacts WHERE company_id=%s AND ("
                    + " OR ".join(like_clauses)
                    + ")"
                )
                cur.execute(q, (company_id, *params))
                founder_present = int(cur.fetchone()[0] or 0) > 0
    except Exception:
        pass
    finally:
        try:
            if conn:
                conn.close()
        except Exception:
            pass
    logger.info(f"[enrichment] contact stats computed for company_id={company_id}")
    return total, has_named, founder_present


def _normalize_lusha_contact(c: dict) -> dict:
    """Flatten/normalize contact from Lusha enrich payload to a common schema."""
    out = {}
    out["lusha_contact_id"] = (
        c.get("lushaContactId") or c.get("contactId") or c.get("id")
    )
    out["first_name"] = c.get("firstName")
    out["last_name"] = c.get("lastName")
    name = c.get("name")
    if not name and (out["first_name"] or out["last_name"]):
        name = " ".join([p for p in [out["first_name"], out["last_name"]] if p])
    out["full_name"] = name
    out["title"] = c.get("jobTitle") or c.get("title")
    out["linkedin_url"] = (
        c.get("linkedinUrl") or c.get("linkedinProfileUrl") or c.get("linkedin")
    )
    out["company_name"] = c.get("companyName")
    out["company_domain"] = c.get("companyDomain")
    out["seniority"] = c.get("seniority")
    out["department"] = c.get("department")
    out["city"] = (
        c.get("city") or (c.get("location") or {}).get("city")
        if isinstance(c.get("location"), dict)
        else c.get("location")
    )
    out["country"] = (
        c.get("country") or (c.get("location") or {}).get("country")
        if isinstance(c.get("location"), dict)
        else None
    )

    # Emails
    emails = []
    src_emails = c.get("emailAddresses") or c.get("emails") or c.get("email_addresses")
    if isinstance(src_emails, list):
        for e in src_emails:
            if isinstance(e, dict):
                v = e.get("email") or e.get("value")
                if v:
                    emails.append(v)
            elif isinstance(e, str):
                emails.append(e)
    elif isinstance(src_emails, str):
        emails.append(src_emails)
    out["emails"] = [e for e in emails if e]

    # Phones
    phones = []
    src_phones = c.get("phoneNumbers") or c.get("phones") or c.get("phone_numbers")
    if isinstance(src_phones, list):
        for p in src_phones:
            if isinstance(p, dict):
                v = (
                    p.get("internationalNumber")
                    or p.get("number")
                    or p.get("value")
                    or p.get("e164")
                )
                if v:
                    phones.append(v)
            elif isinstance(p, str):
                phones.append(p)
    elif isinstance(src_phones, str):
        phones.append(src_phones)
    out["phones"] = [p for p in phones if p]
    return out


def upsert_contacts_from_apify(company_id: int, contacts: List[Dict[str, Any]]):
    """Upsert normalized Apify contacts into contacts table.

    Returns tuple (inserted_count, updated_count).
    """
    inserted, updated = 0, 0
    conn = None
    try:
        conn = get_db_connection()
        cols = _get_table_columns(conn, "contacts")
        has_updated_at = "updated_at" in cols
        with conn, conn.cursor() as cur:
            for c in contacts or []:
                row: Dict[str, Any] = {"company_id": company_id}
                if "full_name" in cols and c.get("full_name"):
                    row["full_name"] = c.get("full_name")
                if "title" in cols and c.get("title"):
                    row["title"] = c.get("title")
                if "linkedin_profile" in cols and c.get("linkedin_url"):
                    row["linkedin_profile"] = c.get("linkedin_url")
                if "linkedin_url" in cols and c.get("linkedin_url"):
                    row["linkedin_url"] = c.get("linkedin_url")
                if "location_city" in cols and c.get("location"):
                    row["location_city"] = c.get("location")
                if "contact_source" in cols:
                    row["contact_source"] = "apify_linkedin"
                email = c.get("email")
                has_email_col = "email" in cols
                if has_email_col and email:
                    row["email"] = email

                exists = False
                if has_email_col and email:
                    cur.execute(
                        "SELECT 1 FROM contacts WHERE company_id=%s AND email IS NOT DISTINCT FROM %s LIMIT 1",
                        (company_id, email),
                    )
                    exists = bool(cur.fetchone())
                else:
                    if c.get("linkedin_url") and ("linkedin_url" in cols or "linkedin_profile" in cols):
                        lk_col = "linkedin_url" if "linkedin_url" in cols else "linkedin_profile"
                        cur.execute(
                            f"SELECT 1 FROM contacts WHERE company_id=%s AND {lk_col} IS NOT DISTINCT FROM %s LIMIT 1",
                            (company_id, c.get("linkedin_url")),
                        )
                        exists = bool(cur.fetchone())
                if exists:
                    set_cols = [k for k in row.keys() if k not in ("company_id", "email")]
                    if set_cols:
                        assignments = ", ".join([f"{k}=%s" for k in set_cols])
                        params = [row[k] for k in set_cols]
                        if has_email_col and email:
                            where_clause = "company_id=%s AND email IS NOT DISTINCT FROM %s"
                            params.extend([company_id, email])
                        else:
                            lk_col = "linkedin_url" if "linkedin_url" in cols else "linkedin_profile"
                            where_clause = f"company_id=%s AND {lk_col} IS NOT DISTINCT FROM %s"
                            params.extend([company_id, c.get("linkedin_url")])
                        if has_updated_at:
                            assignments = assignments + ", updated_at=now()"
                        cur.execute(f"UPDATE contacts SET {assignments} WHERE {where_clause}", params)
                        updated += cur.rowcount or 0
                else:
                    cols_list = list(row.keys())
                    placeholders = ",".join(["%s"] * len(cols_list))
                    cur.execute(
                        f"INSERT INTO contacts ({', '.join(cols_list)}) VALUES ({placeholders}) ON CONFLICT DO NOTHING",
                        [row[k] for k in cols_list],
                    )
                    inserted += cur.rowcount or 0
    except Exception:
        return (inserted, updated)
    finally:
        try:
            if conn:
                conn.close()
        except Exception:
            pass
    return inserted, updated


def upsert_contacts_from_lusha(
    company_id: int, lusha_contacts: list[dict]
) -> tuple[int, int]:
    """Upsert contacts from Lusha into contacts table. Returns (inserted, updated)."""
    if not lusha_contacts:
        return (0, 0)
    inserted = 0
    updated = 0
    conn = get_db_connection()
    try:
        cols = _get_table_columns(conn, "contacts")
        has_email = "email" in cols
        has_updated_at = "updated_at" in cols
        for raw in lusha_contacts:
            c = _normalize_lusha_contact(raw)
            emails = c.get("emails") or [None]
            phone_primary = (c.get("phones") or [None])[0]
            for email in emails:
                # Build payload dynamically based on existing columns
                row = {"company_id": company_id, "contact_source": "lusha"}
                if "lusha_contact_id" in cols and c.get("lusha_contact_id"):
                    row["lusha_contact_id"] = c.get("lusha_contact_id")
                if "first_name" in cols and c.get("first_name"):
                    row["first_name"] = c.get("first_name")
                if "last_name" in cols and c.get("last_name"):
                    row["last_name"] = c.get("last_name")
                if "full_name" in cols and c.get("full_name"):
                    row["full_name"] = c.get("full_name")
                if "title" in cols and c.get("title"):
                    row["title"] = c.get("title")
                if "linkedin_url" in cols and c.get("linkedin_url"):
                    row["linkedin_url"] = c.get("linkedin_url")
                if "seniority" in cols and c.get("seniority"):
                    row["seniority"] = c.get("seniority")
                if "department" in cols and c.get("department"):
                    row["department"] = c.get("department")
                if "city" in cols and c.get("city"):
                    row["city"] = c.get("city")
                if "country" in cols and c.get("country"):
                    row["country"] = c.get("country")
                # phones
                if "phone_number" in cols and phone_primary:
                    row["phone_number"] = phone_primary
                elif "phone" in cols and phone_primary:
                    row["phone"] = phone_primary
                # email and verification placeholders
                if has_email:
                    row["email"] = email
                if "email_verified" in cols and email is not None:
                    row["email_verified"] = None
                if "verification_confidence" in cols and email is not None:
                    row["verification_confidence"] = None

                # Decide existence
                with conn, conn.cursor() as cur:
                    exists = False
                    if has_email:
                        cur.execute(
                            "SELECT 1 FROM contacts WHERE company_id=%s AND email IS NOT DISTINCT FROM %s LIMIT 1",
                            (company_id, email),
                        )
                        exists = bool(cur.fetchone())
                    # Build SQL dynamically
                    if exists:
                        set_cols = [
                            k for k in row.keys() if k not in ("company_id", "email")
                        ]
                        if set_cols:
                            assignments = ", ".join([f"{k}=%s" for k in set_cols])
                            params = [row[k] for k in set_cols]
                            where_clause = (
                                "company_id=%s AND email IS NOT DISTINCT FROM %s"
                                if has_email
                                else "company_id=%s"
                            )
                            params.extend(
                                [company_id, email] if has_email else [company_id]
                            )
                            if has_updated_at:
                                assignments = assignments + ", updated_at=now()"
                            cur.execute(
                                f"UPDATE contacts SET {assignments} WHERE {where_clause}",
                                params,
                            )
                            updated += cur.rowcount or 0
                    else:
                        cols_list = list(row.keys())
                        placeholders = ",".join(["%s"] * len(cols_list))
                        cur.execute(
                            f"INSERT INTO contacts ({', '.join(cols_list)}) VALUES ({placeholders}) ON CONFLICT DO NOTHING",
                            [row[k] for k in cols_list],
                        )
                        inserted += cur.rowcount or 0
                        # Also mirror into lead_emails if available
                        if has_email and email:
                            try:
                                cur.execute(
                                    """
                                    INSERT INTO lead_emails (email, company_id, first_name, last_name, role_title, source)
                                    VALUES (%s,%s,%s,%s,%s,%s)
                                    ON CONFLICT (email) DO UPDATE SET company_id=EXCLUDED.company_id,
                                      first_name=COALESCE(EXCLUDED.first_name, lead_emails.first_name),
                                      last_name=COALESCE(EXCLUDED.last_name, lead_emails.last_name),
                                      role_title=COALESCE(EXCLUDED.role_title, lead_emails.role_title),
                                      source=EXCLUDED.source
                                    """,
                                    (
                                        email,
                                        company_id,
                                        row.get("first_name"),
                                        row.get("last_name"),
                                        row.get("title"),
                                        "lusha",
                                    ),
                                )
                            except Exception:
                                pass
        return inserted, updated
    except Exception as e:
        print(f"       ↳ Lusha contacts upsert failed: {e}")
        return (inserted, updated)
    finally:
        try:
            conn.close()
        except Exception:
            pass


# -------------- Tavily merged-corpus helpers --------------


def _clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s


async def _fetch(client: httpx.AsyncClient, url: str) -> str:
    r = await client.get(url, follow_redirects=True, timeout=CRAWLER_TIMEOUT_S)
    r.raise_for_status()
    return r.text


async def _discover_relevant_urls(home_url: str, max_pages: int) -> list[str]:
    """Fetch homepage, parse same-domain links, keep only keyword-matching URLs."""
    parsed = urlparse(home_url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    urls: list[str] = [home_url]
    async with httpx.AsyncClient(headers={"User-Agent": CRAWLER_USER_AGENT}) as client:
        try:
            html = await _fetch(client, home_url)
        except Exception:
            return urls
        soup = BeautifulSoup(html, "html.parser")
        found = set()
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if (
                not href
                or href.startswith(("#", "mailto:", "tel:"))
                or "javascript:" in href
            ):
                continue
            full = urljoin(base, href)
            if urlparse(full).netloc != urlparse(base).netloc:
                continue
            label = (a.get_text(" ", strip=True) or href).lower()
            if any(k in label for k in CRAWL_KEYWORDS) or any(
                k in full.lower() for k in CRAWL_KEYWORDS
            ):
                found.add(full)
            if len(found) >= (max_pages - 1):
                break
        urls += sorted(found)[: max_pages - 1]
        return urls


def _combine_pages(pages: list[dict], char_limit: int) -> str:
    """Combine extracted pages (url, title, raw_content) into a single corpus."""
    blobs: list[str] = []
    for p in pages:
        url = p.get("url") or ""
        title = _clean_text(p.get("title") or "")
        body = p.get("raw_content") or p.get("content") or p.get("html") or ""
        if isinstance(body, dict):
            body = body.get("text") or ""
        body = _clean_text(body)
        if not body and title:
            body = title
        if not body:
            continue
        blobs.append(f"[URL] {url}\n[TITLE] {title}\n[BODY]\n{body}\n")
    combined = "\n\n".join(blobs)
    # Debug print can be noisy; keep minimal
    if len(combined) > char_limit:
        combined = combined[:char_limit] + "\n\n[TRUNCATED]"
    return combined


def _make_corpus_chunks(pages: list[dict], chunk_char_size: int) -> list[str]:
    """Build corpus chunks from pages, ensuring each chunk <= chunk_char_size.

    - Strips HTML to text when needed to reduce token bloat.
    - Splits any single oversized page into multiple blocks before packing.
    """
    # Clamp to a safe upper bound regardless of env configuration
    safe_size = max(10_000, min(chunk_char_size, 200_000))
    blocks: list[str] = []
    for p in pages:
        url = p.get("url") or ""
        title = _clean_text(p.get("title") or "")
        body = p.get("raw_content") or p.get("content") or p.get("html") or ""

        # Normalize body into plain text
        if isinstance(body, dict):
            body = body.get("text") or ""
        if isinstance(body, str) and ("</" in body or "<br" in body or "<p" in body):
            try:
                body = BeautifulSoup(body, "html.parser").get_text(" ", strip=True)
            except Exception:
                pass
        body = _clean_text(body)
        if not body and title:
            body = title
        if not body:
            continue

        header = f"[URL] {url}\n[TITLE] {title}\n[BODY]\n"
        max_body_len = max(1, safe_size - len(header) - 10)
        if len(body) <= max_body_len:
            blocks.append(f"{header}{body}\n")
        else:
            # Split a single large page into multiple pieces
            part = 1
            for i in range(0, len(body), max_body_len):
                piece = body[i : i + max_body_len]
                blocks.append(f"{header}(part {part})\n{piece}\n")
                part += 1

    # Pack blocks into chunks within size limit
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for blk in blocks:
        if cur and (cur_len + len(blk) > safe_size):
            chunks.append("\n\n".join(cur))
            cur = [blk]
            cur_len = len(blk)
        else:
            cur.append(blk)
            cur_len += len(blk)
    if cur:
        chunks.append("\n\n".join(cur))

    # Final hard cap just in case
    chunks = [c[:safe_size] for c in chunks]
    return chunks


def _merge_extracted_records(base: dict, new: dict) -> dict:
    """Merge two extraction results. Arrays are unioned; scalars prefer non-null; about_text prefers longer."""
    if not base:
        base = {}
    base = dict(base)
    array_keys = {"email", "phone_number", "tech_stack"}
    for k, v in (new or {}).items():
        if v is None:
            continue
        if k in array_keys:
            a = base.get(k) or []
            b = v if isinstance(v, list) else [v]
            base[k] = list({*a, *b})
        elif k == "about_text":
            prev = base.get(k) or ""
            nv = v or ""
            base[k] = nv if len(nv) > len(prev) else prev
        else:
            if base.get(k) in (None, ""):
                base[k] = v
    return base


def _ensure_list(v):
    if v is None:
        return None
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        parts = [p.strip() for p in re.split(r"[,\n;]+", v) if p.strip()]
        return parts or None
    return None


async def _merge_with_jina(data: dict, home: str) -> dict:
    logger.info("    🔁 Merging with r.jina reader")
    try:
        text = jina_read(home, timeout=8) or ""
    except Exception:
        text = ""
    if not text:
        return data
    # Reuse the LLM extraction chain for a small subset to enrich gaps
    subset_schema = ["about_text", "email", "phone_number", "tech_stack"]
    try:
        async def _do_llm(payload: dict) -> str:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: extract_chain.invoke(payload))
        payload = {
            "raw_content": f"{text[:4000]}",
            "schema_keys": subset_schema,
            "instructions": (
                "Return JSON with only the above keys. For email/phone_number/tech_stack return arrays. "
                "about_text should be a concise 1-2 sentence summary."
            ),
        }
        ai_output = await asyncio.wait_for(_do_llm(payload), timeout=float(LLM_CHUNK_TIMEOUT_S))
        import json as _json
        m = re.search(r"\{.*\}", ai_output, re.S)
        piece = _json.loads(m.group(0)) if m else _json.loads(ai_output)
        # normalize arrays
        for k in ["email", "phone_number", "tech_stack"]:
            piece[k] = _ensure_list(piece.get(k)) or []
        data = _merge_extracted_records(data, piece)
    except Exception:
        pass
    # Website and HQ quick guess from URL alone
    if not data.get("website_domain") and isinstance(home, str):
        data["website_domain"] = home
    try:
        if (not data.get("hq_city") or not data.get("hq_country")) and isinstance(home, str):
            if home.lower().endswith(".sg/") or ".sg" in home.lower():
                data.setdefault("hq_city", "Singapore")
                data.setdefault("hq_country", "Singapore")
    except Exception:
        pass
    return data


def update_company_core_fields(company_id: int, data: dict):
    """Update core scalar fields on companies table; arrays handled by store_enrichment."""
    conn = get_db_connection()
    try:
        with conn, conn.cursor() as cur:
            sql = """
                UPDATE companies SET
                  name = COALESCE(%s, name),

                  employees_est = %s,
                  revenue_bucket = %s,
                  incorporation_year = %s,

                  website_domain = COALESCE(%s, website_domain),

                  company_size = %s,
                  annual_revenue = %s,
                  hq_city = %s,
                  hq_country = %s,
                  linkedin_url = %s,
                  founded_year = %s,
                  ownership_type = %s,
                  funding_status = %s,
                  employee_turnover = %s,
                  web_traffic = %s,
                  location_city = %s,
                  location_country = %s,
                  last_seen = now()
                WHERE company_id = %s

            """
            params = [
                data.get("name"),
                data.get("employees_est"),
                data.get("revenue_bucket"),
                data.get("incorporation_year"),
                data.get("website_domain"),
                data.get("company_size"),
                data.get("annual_revenue"),
                data.get("hq_city"),
                data.get("hq_country"),
                data.get("linkedin_url"),
                data.get("founded_year"),
                data.get("ownership_type"),
                data.get("funding_status"),
                data.get("employee_turnover"),
                data.get("web_traffic"),
                data.get("location_city"),
                data.get("location_country"),
                company_id,
            ]
            assert sql.count("%s") == len(params), "placeholder mismatch"
            cur.execute(sql, params)


    except Exception as e:
        logger.exception("    ⚠️ companies core update failed")
    finally:
        conn.close()


async def _jina_snapshot_pages(company_id: int, url: str):
    """Fetch homepage and a few deterministic pages to seed initial data.

    Order of attempts:
      1) r.jina snapshot of homepage (fast, tolerant parser)
      2) Direct HTTP GET of homepage with crawler UA
      3) Direct HTTP GET of deterministic subpages (about/contact/careers, etc.)

    Returns a tuple (summary_dict, pages) where pages is a list of
    {url, html} or {url, raw_content} entries suitable for downstream
    chunking and extraction.
    """
    # Normalize and derive roots/variants
    try:
        base = url
        if not base:
            return None, []
        if not base.startswith("http"):
            base = f"https://{base}"
        u = urlparse(base)
        root = f"{u.scheme}://{u.netloc}"
        variants = [base]
        # Add www variant if missing
        try:
            host = u.netloc or ""
            if host and not host.lower().startswith("www."):
                variants.append(f"{u.scheme}://www.{host}{u.path or ''}")
        except Exception:
            pass
    except Exception:
        return None, []

    pages: list[dict] = []
    summary_text = ""

    # 1) Try r.jina for homepage
    text = ""
    try:
        for v in variants:
            text = jina_read(v, timeout=8) or ""
            if text:
                pages.append({"url": v, "html": text})
                summary_text = text
                break
    except Exception:
        text = ""

    # 2) Direct HTTP GET fallback for homepage
    if not text:
        try:
            async with httpx.AsyncClient(headers={"User-Agent": CRAWLER_USER_AGENT}) as client:
                resp = await client.get(variants[0], follow_redirects=True, timeout=CRAWLER_TIMEOUT_S)
                if getattr(resp, "text", ""):
                    body = resp.text
                    pages.append({"url": variants[0], "html": body})
                    summary_text = body
        except Exception:
            pass

    # 3) Deterministic subpages if homepage still empty or to augment thin pages
    need_more = not pages or len((summary_text or "").strip()) < 200
    if need_more:
        seeds = [
            "about", "about-us", "aboutus", "company", "who-we-are",
            "contact", "contact-us",
            "team", "leadership",
            "careers", "jobs",
            "services", "solutions", "products",
        ]
        cand_urls = []
        for p in seeds:
            cand_urls.append(f"{root}/{p}")
        # Fetch in parallel, keep first 2-3 that return content
        try:
            async with httpx.AsyncClient(headers={"User-Agent": CRAWLER_USER_AGENT}) as client:
                resps = await asyncio.gather(
                    *(
                        client.get(u, follow_redirects=True, timeout=CRAWLER_TIMEOUT_S)
                        for u in cand_urls
                    ),
                    return_exceptions=True,
                )
            added = 0
            for resp, u in zip(resps, cand_urls):
                if isinstance(resp, Exception):
                    continue
                body = getattr(resp, "text", "") or ""
                # Skip obvious soft-404s
                if not body or len(body.strip()) < 100:
                    continue
                pages.append({"url": u, "html": body})
                if not summary_text:
                    summary_text = body
                added += 1
                if added >= 3:
                    break
        except Exception:
            pass

    if not pages:
        return None, []

    # Project a minimal record into company_enrichment_runs
    try:
        conn = get_db_connection()
        with conn:
            fields = {
                "company_id": company_id,
                "about_text": (summary_text or "")[:1000],
                "tech_stack": [],
                "public_emails": [],
                "jobs_count": 0,
                "linkedin_url": None,
            }
            tid = _default_tenant_id()
            if tid is not None:
                fields["tenant_id"] = tid
            _insert_company_enrichment_run(conn, fields)
        conn.close()
    except Exception:
        pass

    # Legacy store for transparency
    try:
        legacy = {
            "about_text": (summary_text or "")[:1000],
            "tech_stack": [],
            "public_emails": [],
            "jobs_count": 0,
            "linkedin_url": None,
            "phone_number": [],
            "hq_city": None,
            "hq_country": None,
        }
        store_enrichment(company_id, root, legacy)
    except Exception:
        pass

    return {"url": root, "content_summary": (summary_text or "")[:1000], "signals": {}}, pages


async def enrich_company_with_tavily(
    company_id: int, company_name: str, uen: str | None = None
):
    """
    Orchestrated enrichment flow using LangGraph. This wrapper constructs
    the initial state and invokes the compiled enrichment_agent graph.
    """
    initial_state = {
        "company_id": company_id,
        "company_name": company_name,
        "uen": uen,
        "domains": [],
        "home": None,
        "filtered_urls": [],
        "page_urls": [],
        "extracted_pages": [],
        "chunks": [],
        "data": {},
        "lusha_used": False,
        "completed": False,
        "error": None,
        "degraded_reasons": [],
    }
    try:
        # Global skip logic to avoid re-enriching companies that already have
        # recent or any enrichment history (configurable).
        try:
            if _should_skip_enrichment(company_id):
                logger.info(
                    f"[enrichment] skip company_id={company_id} name={company_name!r} due to prior enrichment"
                )
                initial_state["completed"] = True
                initial_state["error"] = "skip_prior_enrichment"
                return initial_state
        except Exception:
            # Non-blocking: if the guard fails, proceed with enrichment to avoid false negatives
            pass
        logger.info(
            f"[enrichment] start company_id={company_id}, name={company_name!r}"
        )
        if ENRICH_AGENTIC:
            final_state = await run_enrichment_agentic(initial_state)  # planner-driven
        else:
            final_state = await enrichment_agent.ainvoke(initial_state)  # fixed graph
        logger.info(
            f"[enrichment] completed company_id={company_id}, extracted_pages={len(final_state.get('extracted_pages') or [])}, completed={final_state.get('completed')}"
        )
        return final_state
    except Exception:
        logger.exception("   ↳ Enrichment graph invoke failed")
        return initial_state


def _should_skip_enrichment(company_id: int) -> bool:
    """Return True if enrichment should be skipped based on prior history.

    Rules:
    - If ENRICH_SKIP_IF_ANY_HISTORY is true, skip when ANY row exists in company_enrichment_runs.
    - Else if ENRICH_RECHECK_DAYS > 0, skip when a row exists with updated_at within the window.
    - Else, do not skip.
    Best-effort: If company_enrichment_runs is absent or errors, do not skip.
    """
    try:
        conn = get_db_connection()
        with conn, conn.cursor() as cur:
            if ENRICH_SKIP_IF_ANY_HISTORY:
                cur.execute(
                    "SELECT 1 FROM company_enrichment_runs WHERE company_id=%s LIMIT 1",
                    (company_id,),
                )
                return cur.fetchone() is not None
            try:
                days = int(ENRICH_RECHECK_DAYS)
            except Exception:
                days = 0
            if days and days > 0:
                cur.execute(
                    """
                    SELECT 1
                    FROM company_enrichment_runs
                    WHERE company_id=%s
                      AND COALESCE(updated_at, now()) >= now() - (%s::text || ' days')::interval
                    LIMIT 1
                    """,
                    (company_id, str(days)),
                )
                return cur.fetchone() is not None
    except Exception:
        return False
    return False


class EnrichmentState(TypedDict, total=False):
    company_id: int
    company_name: str
    uen: Optional[str]
    domains: List[str]
    home: Optional[str]
    filtered_urls: List[str]
    page_urls: List[str]
    extracted_pages: List[Dict[str, Any]]
    chunks: List[str]
    data: Dict[str, Any]
    deterministic_summary: Dict[str, Any]
    lusha_used: bool
    completed: bool
    error: Optional[str]
    degraded_reasons: List[str]


async def node_find_domain(state: EnrichmentState) -> EnrichmentState:
    if state.get("completed"):
        return state
    name = state.get("company_name") or ""
    # 0) DB fallback: use existing website_domain for this company if present
    try:
        cid = state.get("company_id")
        if cid:
            conn = get_db_connection()
            with conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT website_domain FROM companies WHERE company_id=%s", (cid,)
                )
                row = cur.fetchone()
            try:
                conn.close()
            except Exception:
                pass
            if row and row[0]:
                dom = str(row[0])
                if not dom.startswith("http"):
                    dom = "https://" + dom
                domains = [dom]
            else:
                domains = []
        else:
            domains = []
    except Exception:
        domains = []

    # 1) Tavily search if available
    if not domains and ENABLE_TAVILY_FALLBACK and tavily_client is not None:
        try:
            t0 = time.perf_counter()
            domains = find_domain(name)
            _obs_vendor("tavily", calls=1)
            _obs_log("find_domain", "vendor_call", "ok", company_id=state.get("company_id"), duration_ms=int((time.perf_counter()-t0)*1000), extra={"query": name})
        except Exception as e:
            _obs_vendor("tavily", calls=1, errors=1)
            _obs_log("find_domain", "vendor_call", "error", company_id=state.get("company_id"), error_code=type(e).__name__)
            logger.warning("   ↳ Tavily find_domain failed", exc_info=True)
            try:
                (state.setdefault("degraded_reasons", [])) .append("TAVILY_FAIL")
            except Exception:
                pass
    # Lusha fallback if needed
    if (not domains) and ENABLE_LUSHA_FALLBACK and LUSHA_API_KEY:
        try:
            logger.info("   ↳ No domain via search; trying Lusha fallback…")
            tid = int(_RUN_CTX.get("tenant_id") or 0)
            lusha_domain = None
            if _cb_allows(tid, "lusha"):
                async with AsyncLushaClient() as lc:
                    async def _call():
                        return await lc.find_company_domain(name)
                    try:
                        lusha_domain = await with_retry(_call, retry_on=(Exception,), policy=_DEFAULT_RETRY_POLICY)
                        _VENDOR_COUNTERS["lusha_lookups"] += 1
                        _obs_vendor("lusha", calls=1)
                        if tid:
                            _CB.on_success(tid, "lusha")
                    except Exception:
                        if tid:
                            _CB.on_error(tid, "lusha")
                        lusha_domain = None
            if lusha_domain:
                normalized = (
                    lusha_domain
                    if lusha_domain.startswith("http")
                    else f"https://{lusha_domain}"
                )
                domains = [normalized]
                state["lusha_used"] = True
                logger.info(f"   ↳ Lusha provided domain: {normalized}")
        except Exception as e:
            logger.warning("   ↳ Lusha domain fallback failed", exc_info=True)
            try:
                (state.setdefault("degraded_reasons", [])) .append("LUSHA_FAIL")
            except Exception:
                pass
    if not domains:
        # Graceful termination: no domain available, nothing to crawl/extract.
        # Mark as completed so upstream pipeline can proceed to scoring/next steps.
        state["error"] = "no_domain"
        try:
            (state.setdefault("degraded_reasons", [])) .append("DATA_EMPTY:no_domain_name")
        except Exception:
            pass
        state["completed"] = True
        logger.info("   ↳ No domain found; marking enrichment as completed (no crawl)")
        return state
    home = domains[0]
    if not home.startswith("http"):
        home = "https://" + home
    state["domains"] = domains
    state["home"] = home
    return state


async def node_discover_urls(state: EnrichmentState) -> EnrichmentState:
    if state.get("completed") or not state.get("home"):
        return state
    home = state["home"]
    filtered_urls: List[str] = await _discover_relevant_urls(home, CRAWL_MAX_PAGES)
    if not filtered_urls and ENABLE_LUSHA_FALLBACK and LUSHA_API_KEY:
        try:
            tid = int(_RUN_CTX.get("tenant_id") or 0)
            lusha_domain = None
            if _cb_allows(tid, "lusha"):
                async with AsyncLushaClient() as lc:
                    async def _call():
                        return await lc.find_company_domain(state.get("company_name") or "")
                    try:
                        lusha_domain = await with_retry(_call, retry_on=(Exception,), policy=_DEFAULT_RETRY_POLICY)
                        _VENDOR_COUNTERS["lusha_lookups"] += 1
                        _obs_vendor("lusha", calls=1)
                        if tid:
                            _CB.on_success(tid, "lusha")
                    except Exception:
                        if tid:
                            _CB.on_error(tid, "lusha")
                        lusha_domain = None
            if lusha_domain:
                candidate_home = (
                    lusha_domain
                    if lusha_domain.startswith("http")
                    else f"https://{lusha_domain}"
                )
                if (
                    urlparse(candidate_home).netloc
                    and urlparse(candidate_home).netloc != urlparse(home).netloc
                ):
                    logger.info(
                        f"   ↳ Using Lusha-discovered domain for crawl: {candidate_home}"
                    )
                    state["home"] = candidate_home
                    state["lusha_used"] = True
                    filtered_urls = await _discover_relevant_urls(
                        candidate_home, CRAWL_MAX_PAGES
                    )
        except Exception as e:
            logger.warning("   ↳ Lusha fallback for filtered URLs failed", exc_info=True)
            try:
                (state.setdefault("degraded_reasons", [])) .append("LUSHA_FAIL")
            except Exception:
                pass
    if not filtered_urls:
        filtered_urls = [state["home"]]
        try:
            (state.setdefault("degraded_reasons", [])) .append("CRAWL_THIN")
        except Exception:
            pass
    state["filtered_urls"] = filtered_urls
    return state


async def node_expand_crawl(state: EnrichmentState) -> EnrichmentState:
    if state.get("completed") or not state.get("filtered_urls"):
        return state
    filtered_urls = state["filtered_urls"]
    home = state.get("home")
    page_urls: List[str] = []
    try:
        roots: List[str] = []
        for u in filtered_urls:
            parsed = urlparse(u)
            if not parsed.scheme:
                u = "https://" + u
                parsed = urlparse(u)
            roots.append(f"{parsed.scheme}://{parsed.netloc}")
        if home:
            roots.append(home)
        roots = list(dict.fromkeys(roots))

        # Seed About pages explicitly when we have filtered URLs
        if filtered_urls:
            for _root in roots:
                for _p in ("about", "aboutus"):
                    page_urls.append(f"{_root}/{_p}")

        if ENABLE_TAVILY_FALLBACK and tavily_crawl is not None and _dec_cap("tavily_units", 1):
            for root in roots[:3]:
                try:
                    crawl_input = {
                        "url": f"{root}/*",
                        "limit": CRAWL_MAX_PAGES,
                        "crawl_depth": 2,
                        "instructions": f"get all pages from {root}",
                        "enable_web_search": False,
                    }
                    t0 = time.perf_counter()
                    crawl_result = tavily_crawl.run(crawl_input)
                    _VENDOR_COUNTERS["tavily_crawl_calls"] += 1
                    _obs_vendor("tavily", calls=1)
                    _obs_log("crawl", "vendor_call", "ok", company_id=state.get("company_id"), duration_ms=int((time.perf_counter()-t0)*1000), extra={"root": root})
                    raw_urls = []
                    if isinstance(crawl_result, dict):
                        raw_urls = crawl_result.get("results") or crawl_result.get("urls") or []
                    elif isinstance(crawl_result, list):
                        raw_urls = crawl_result
                    for item in raw_urls:
                        if isinstance(item, dict) and item.get("url"):
                            page_urls.append(item["url"])
                        elif isinstance(item, str) and item.startswith("http"):
                            page_urls.append(item)
                    page_urls.append(root)
                except Exception as exc:
                    _obs_vendor("tavily", calls=1, errors=1)
                    _obs_log("crawl", "vendor_call", "error", company_id=state.get("company_id"), error_code=type(exc).__name__, extra={"root": root})
                    logger.warning(f"          ↳ TavilyCrawl error for {root}", exc_info=True)
                    page_urls.append(root)
                    try:
                        (state.setdefault("degraded_reasons", [])) .append("CRAWL_ERROR")
                    except Exception:
                        pass
        else:
            logger.info("       ↳ TavilyCrawl unavailable; using seeded URLs only")
        # Deduplicate and sanitize URLs (handle stray dicts from vendor responses)
        def _coerce_url(val):
            if isinstance(val, str):
                return val
            if isinstance(val, dict):
                u = val.get("url")
                return u if isinstance(u, str) else None
            return None

        seen = set()
        cleaned: List[str] = []
        for it in page_urls:
            u = _coerce_url(it)
            if not u or "*" in u or not u.startswith("http"):
                continue
            if u not in seen:
                seen.add(u)
                cleaned.append(u)
        page_urls = cleaned
        try:
            logger.info(
                f"       ↳ Seeded/Discovered {len(page_urls)} URLs (incl. about seeds)"
            )
            for _dbg in page_urls[:25]:
                logger.debug(f"          - {_dbg}")
        except Exception:
            pass
    except Exception as exc:
        logger.warning("          ↳ TavilyCrawl expansion skipped", exc_info=True)
        page_urls = []
    if not page_urls:
        page_urls = filtered_urls
    state["page_urls"] = page_urls
    return state


async def node_extract_pages(state: EnrichmentState) -> EnrichmentState:
    if state.get("completed") or not state.get("page_urls"):
        return state
    page_urls = state["page_urls"]
    extracted_pages: List[Dict[str, Any]] = []
    fallback_urls: List[str] = []

    def _extract_raw_from(obj: Any) -> Optional[str]:
        # Try common shapes from TavilyExtract
        if obj is None:
            return None
        if isinstance(obj, str):
            return obj
        if isinstance(obj, dict):
            for key in ("raw_content", "content", "text"):
                val = obj.get(key)
                if isinstance(val, str) and val.strip():
                    return val
                if isinstance(val, dict):
                    # nested content holder
                    for k2 in ("raw_content", "content", "text"):
                        v2 = val.get(k2)
                        if isinstance(v2, str) and v2.strip():
                            return v2
            # results list
            results = obj.get("results")
            if isinstance(results, list) and results:
                for item in results:
                    if isinstance(item, dict):
                        got = _extract_raw_from(item)
                        if got:
                            return got
        if isinstance(obj, list):
            for it in obj:
                got = _extract_raw_from(it)
                if got:
                    return got
        return None

    for u in page_urls:
        # Try TavilyExtract if configured
        raw_content: Optional[str] = None
        if ENABLE_TAVILY_FALLBACK and tavily_extract is not None:
            payload = {
                "urls": [u],
                "schema": {"raw_content": "str"},
                "instructions": "Retrieve the main textual content from this page.",
            }
            try:
                t0 = time.perf_counter()
                raw_data = tavily_extract.run(payload)
                _obs_vendor("tavily", calls=1)
                _obs_log("extract", "vendor_call", "ok", company_id=state.get("company_id"), duration_ms=int((time.perf_counter()-t0)*1000), extra={"url": u})
                raw_content = _extract_raw_from(raw_data)
            except Exception as exc:
                _obs_vendor("tavily", calls=1, errors=1)
                _obs_log("extract", "vendor_call", "error", company_id=state.get("company_id"), error_code=type(exc).__name__, extra={"url": u})
                logger.warning(f"          ↳ TavilyExtract error for {u}", exc_info=True)
                try:
                    (state.setdefault("degraded_reasons", [])) .append("TAVILY_FAIL")
                except Exception:
                    pass
        if raw_content and isinstance(raw_content, str) and raw_content.strip():
            extracted_pages.append({"url": u, "title": "", "raw_content": raw_content})
        else:
            fallback_urls.append(u)

    if fallback_urls:
        try:
            logger.info(
                f"       ↳ TavilyExtract empty for {len(fallback_urls)} URLs; attempting HTTP fallback"
            )
            async with httpx.AsyncClient(
                headers={"User-Agent": CRAWLER_USER_AGENT}
            ) as client:
                resps = await asyncio.gather(
                    *(
                        client.get(u, follow_redirects=True, timeout=CRAWLER_TIMEOUT_S)
                        for u in fallback_urls
                    ),
                    return_exceptions=True,
                )
            recovered = 0
            for resp, u in zip(resps, fallback_urls):
                if isinstance(resp, Exception):
                    continue
                body = getattr(resp, "text", "")
                if body:
                    extracted_pages.append({"url": u, "html": body})
                    recovered += 1
            logger.info(
                f"       ↳ HTTP fallback recovered {recovered}/{len(fallback_urls)} pages"
            )
        except Exception as _per_url_fb_exc:
            logger.warning("       ↳ Per-URL HTTP fallback failed", exc_info=True)

    if not extracted_pages:
        try:
            async with httpx.AsyncClient(
                headers={"User-Agent": CRAWLER_USER_AGENT}
            ) as client:
                resps = await asyncio.gather(
                    *(
                        client.get(u, follow_redirects=True, timeout=CRAWLER_TIMEOUT_S)
                        for u in page_urls
                    ),
                    return_exceptions=True,
                )
            for resp, u in zip(resps, page_urls):
                if isinstance(resp, Exception):
                    continue
                extracted_pages.append({"url": u, "html": getattr(resp, "text", "")})
        except Exception as e:
            logger.warning("   ↳ Fallback HTTP fetch failed", exc_info=True)
    # If still nothing, inject a Jina homepage snapshot and finish
    if not extracted_pages:
        try:
            if state.get("company_id") and state.get("home"):
                text = jina_read(state["home"], timeout=8) or ""
                if text:
                    state["extracted_pages"] = [
                        {"url": state["home"], "title": "", "raw_content": text}
                    ]
                    logger.info("   ↳ injected jina homepage content for extraction")
                else:
                    state["completed"] = True
                    state["extracted_pages"] = []
                    try:
                        (state.setdefault("degraded_reasons", [])) .append("DATA_EMPTY:jina_home")
                    except Exception:
                        pass
                    return state
        except Exception as exc:
            logger.warning("   ↳ jina fallback failed", exc_info=True)
    state["extracted_pages"] = extracted_pages
    try:
        logger.info(
            f"       ↳ Page extraction completed: extracted_pages={len(extracted_pages)}"
        )
    except Exception:
        pass
    return state


async def node_deterministic_crawl(state: EnrichmentState) -> EnrichmentState:
    if state.get("completed") or not state.get("home") or not state.get("company_id"):
        return state
    try:
        logger.info(f"[node_jina_snapshot] company_id={state['company_id']}, home={state['home']}")
        summary, pages = await _jina_snapshot_pages(state["company_id"], state["home"])
        if pages:
            state["extracted_pages"] = [
                {"url": p.get("url"), "title": "", "raw_content": p.get("html")}
                for p in pages
            ]
            logger.info(f"[node_jina_snapshot] extracted_pages from jina={len(pages)}")
        if summary:
            state["deterministic_summary"] = summary
            logger.info("[node_jina_snapshot] set deterministic_summary (jina)")
    except Exception as exc:
        logger.warning("   ↳ jina snapshot failed", exc_info=True)
    return state


async def node_build_chunks(state: EnrichmentState) -> EnrichmentState:
    if state.get("completed") or not state.get("extracted_pages"):
        return state
    chunks = _make_corpus_chunks(state["extracted_pages"], EXTRACT_CORPUS_CHAR_LIMIT)
    logger.info(
        f"       ↳ {len(state['extracted_pages'])} pages -> {len(chunks)} chunks for extraction"
    )
    # Persist merged corpus for transparency/audit
    try:
        if PERSIST_CRAWL_CORPUS:
            full_combined = "\n\n".join(chunks)
            _persist_corpus(
                state.get("company_id"),
                full_combined,
                len(state.get("extracted_pages") or []),
                source="tavily",
            )
    except Exception as _log_exc:
        logger.warning("       ↳ Failed to persist combined corpus", exc_info=True)
    state["chunks"] = chunks
    return state


async def node_llm_extract(state: EnrichmentState) -> EnrichmentState:
    if state.get("completed") or not state.get("chunks"):
        return state
    company_name = state.get("company_name") or ""
    schema_keys = [
        "name",
        "industry_norm",
        "employees_est",
        "revenue_bucket",
        "incorporation_year",
        "sg_registered",
        "last_seen",
        "website_domain",
        "industry_code",
        "company_size",
        "annual_revenue",
        "hq_city",
        "hq_country",
        "linkedin_url",
        "founded_year",
        "tech_stack",
        "ownership_type",
        "funding_status",
        "employee_turnover",
        "web_traffic",
        "email",
        "phone_number",
        "location_city",
        "location_country",
        "about_text",
    ]
    data: Dict[str, Any] = {}
    # Limit chunks to keep per-company latency bounded
    chunks = list(state["chunks"])[: max(1, int(LLM_MAX_CHUNKS))]
    for i, chunk in enumerate(chunks, start=1):
        try:
            async def _do_llm(payload: dict) -> str:
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(None, lambda: extract_chain.invoke(payload))
            payload = {
                "raw_content": f"Company: {company_name}\n\n{chunk}",
                "schema_keys": schema_keys,
                "instructions": (
                    "Return a single JSON object with only the above keys. Use null for unknown. "
                    "For tech_stack, email, and phone_number return arrays of strings. "
                    "Use integers for employees_est and incorporation_year when possible. "
                    "website_domain should be the official domain for the company. "
                    "about_text should be a concise 1-3 sentence summary of the company."
                ),
            }
            ai_output = await asyncio.wait_for(_do_llm(payload), timeout=float(LLM_CHUNK_TIMEOUT_S))
            m = re.search(r"\{.*\}", ai_output, re.S)
            piece = json.loads(m.group(0)) if m else json.loads(ai_output)
            data = _merge_extracted_records(data, piece)
        except Exception as e:
            # Best-effort recovery if context exceeded: retry with trimmed chunk
            msg = str(e) if e else ""
            if "context length" in msg.lower() or "maximum context length" in msg.lower():
                try:
                    trimmed = chunk[: int(len(chunk) * 0.6)]
                    payload_trim = {
                        "raw_content": f"Company: {company_name}\n\n{trimmed}",
                        "schema_keys": schema_keys,
                        "instructions": (
                            "Return a single JSON object with only the above keys. Use null for unknown. "
                            "For tech_stack, email, and phone_number return arrays of strings. "
                            "Use integers for employees_est and incorporation_year when possible. "
                            "website_domain should be the official domain for the company. "
                            "about_text should be a concise 1-3 sentence summary of the company."
                        ),
                    }
                    ai_output = await asyncio.wait_for(_do_llm(payload_trim), timeout=float(LLM_CHUNK_TIMEOUT_S))
                    m = re.search(r"\{.*\}", ai_output, re.S)
                    piece = json.loads(m.group(0)) if m else json.loads(ai_output)
                    data = _merge_extracted_records(data, piece)
                    logger.info(f"   ↳ Chunk {i} retried with trimmed content")
                    continue
                except Exception:
                    pass
            # Handle slow model / timeout / cancellation similarly by trimming once,
            # and fall back to regex extraction to salvage key fields.
            elif isinstance(e, asyncio.TimeoutError) or isinstance(e, asyncio.CancelledError) or "timeout" in msg.lower():
                try:
                    trimmed = chunk[: int(len(chunk) * 0.6)]
                    payload_trim2 = {
                        "raw_content": f"Company: {company_name}\n\n{trimmed}",
                        "schema_keys": schema_keys,
                        "instructions": (
                            "Return a single JSON object with only the above keys. Use null for unknown. "
                            "For tech_stack, email, and phone_number return arrays of strings. "
                            "Use integers for employees_est and incorporation_year when possible. "
                            "website_domain should be the official domain for the company. "
                            "about_text should be a concise 1-3 sentence summary of the company."
                        ),
                    }
                    ai_output = await asyncio.wait_for(_do_llm(payload_trim2), timeout=float(LLM_CHUNK_TIMEOUT_S))
                    m = re.search(r"\{.*\}", ai_output, re.S)
                    piece = json.loads(m.group(0)) if m else json.loads(ai_output)
                    data = _merge_extracted_records(data, piece)
                    logger.info(f"   ↳ Chunk {i} retried after timeout with trimmed content")
                    continue
                except Exception:
                    # Regex fallback to salvage some fields
                    try:
                        salvaged = _fallback_extract_from_text(chunk)
                        if salvaged:
                            data = _merge_extracted_records(data, salvaged)
                            (state.setdefault("degraded_reasons", [])) .append("LLM_TIMEOUT_FALLBACK")
                            logger.warning(f"   ↳ Chunk {i} timeout; applied regex fallback fields={list(salvaged.keys())}")
                            continue
                    except Exception:
                        pass
            logger.warning(f"   ↳ Chunk {i} extraction parse failed", exc_info=True)
            continue
    for k in ["email", "phone_number", "tech_stack"]:
        data[k] = _ensure_list(data.get(k)) or []
    try:
        if state.get("home"):
            data = await asyncio.wait_for(
                _merge_with_jina(data, state["home"]),
                timeout=float(MERGE_DETERMINISTIC_TIMEOUT_S),
            )
    except asyncio.TimeoutError:
        logger.warning("   ↳ jina merge timed out; skipping")
    except Exception:
        logger.warning("   ↳ jina merge skipped", exc_info=True)
    state["data"] = data
    return state


async def node_apify_contacts(state: EnrichmentState) -> EnrichmentState:
    # Allow Apify to run even when earlier steps marked the run completed due to no domain.
    # We can often discover LinkedIn contacts by company name alone.
    if state.get("completed") and state.get("error") != "no_domain":
        return state
    if state.get("completed") and state.get("error") == "no_domain":
        logger.info("[apify_contacts] No domain found; proceeding to Apify by company name")
    data = state.get("data") or {}
    company_id = state.get("company_id")
    if not company_id:
        return state
    try:
        need_emails = not (data.get("email") or [])
        need_phones = not (data.get("phone_number") or [])
        total_contacts, has_named, founder_present = _get_contact_stats(company_id)
        needs_contacts = total_contacts == 0
        missing_names = not has_named
        missing_founder = not founder_present
        trigger = (
            need_emails
            or need_phones
            or needs_contacts
            or missing_names
            or missing_founder
        )
        # Prefer Apify when enabled, or when Lusha fallback is disabled/missing
        tid = int(_RUN_CTX.get("tenant_id") or 0)
        rid = _RUN_CTX.get("run_id")
        prefer_apify = (
            ENABLE_APIFY_LINKEDIN or (not ENABLE_LUSHA_FALLBACK) or (not LUSHA_API_KEY)
        )
        if prefer_apify and trigger and _dec_cap("contact_lookups", 1):
            # Use Apify LinkedIn Actor for contact discovery when trigger conditions met
            titles_env = CONTACT_TITLES or []
            titles_tenant = icp_preferred_titles_for_tenant(tid if tid else None)
            titles = titles_tenant or titles_env or LUSHA_PREFERRED_TITLES
            try:
                titles_source = (
                    "icp_preferred_titles" if titles_tenant else ("CONTACT_TITLES" if titles_env else "default_titles")
                )
                logger.info(
                    f"[apify_contacts] titles_source={titles_source} titles_used={titles} company_id={company_id}"
                )
            except Exception:
                pass
            # Build company query name (avoid domains in queries); titles from ICP
            # Title-only queries (no company/domain) using ICP buyer titles
            company_query_name = _company_query_name_from_state(state)
            queries = apify_build_queries("", titles)
            if queries:
                logger.info(
                    f"[apify_contacts] Using Apify for contact discovery; company_id={company_id} query_mode=title_only titles={titles}"
                )
                # Daily cap enforcement
                if not _apify_cap_ok(tid, need=1):
                    try:
                        _obs_log(
                            "contact_discovery",
                            "vendor_call",
                            "disabled",
                            company_id=company_id,
                            extra={"reason": "apify_daily_cap"},
                        )
                        (state.setdefault("degraded_reasons", [])) .append("APIFY_CAP_EXCEEDED")
                    except Exception:
                        pass
                else:
                    try:
                        t0 = time.perf_counter()
                        # Prefer company -> employees -> profiles chain if enabled
                        # Force title-only search mode (no company/by-name chain) to avoid domain/name queries
                        mode_chain = False
                        if mode_chain:
                            # Reserved: chain path disabled for title-only mode
                            contacts_raw = []
                            raw = []
                        else:
                            raw = await apify_run(
                                {"queries": queries},
                                dataset_format=APIFY_DATASET_FORMAT,
                                timeout_s=APIFY_SYNC_TIMEOUT_S,
                            )
                            contacts_raw = apify_normalize(raw)
                        logger.info(
                            f"[apify_contacts] fetched={len(raw) if isinstance(raw, list) else 0} normalized={len(contacts_raw)} company_id={company_id}"
                        )
                        # Explicit success marker for operational confirmations
                        try:
                            duration_ms = int((time.perf_counter() - t0) * 1000)
                            logger.info(
                                f"[apify_contacts] success company_id={company_id} mode={'company_employee_chain' if mode_chain else 'direct_actor'} duration_ms={duration_ms}"
                            )
                        except Exception:
                            pass
                        # Optional: log a small sample of the Apify raw items and normalized contacts
                        try:
                            dbg = os.getenv("APIFY_DEBUG_LOG_ITEMS", "").lower() in ("1", "true", "yes", "on")
                            if dbg:
                                try:
                                    n = int(os.getenv("APIFY_LOG_SAMPLE_SIZE", "3") or 3)
                                except Exception:
                                    n = 3
                                # Build raw sample with selected fields to keep logs compact
                                def _pick_fields(it: dict) -> dict:
                                    keys = [
                                        "fullName",
                                        "name",
                                        "headline",
                                        "title",
                                        "companyName",
                                        "company",
                                        "profileUrl",
                                        "url",
                                        "linkedin_url",
                                        "locationName",
                                        "location",
                                        "email",
                                    ]
                                    out = {k: it.get(k) for k in keys if (isinstance(it, dict) and it.get(k) is not None)}
                                    if not out and isinstance(it, dict):
                                        for k in list(it.keys())[:6]:
                                            out[k] = it.get(k)
                                    return out
                                raw_sample = [_pick_fields(x) for x in (raw or [])[:n] if isinstance(x, dict)]
                                norm_sample = [
                                    {k: c.get(k) for k in ("full_name", "title", "company_current", "linkedin_url", "location", "email") if c.get(k) is not None}
                                    for c in (contacts_raw or [])[:n]
                                ]
                                logger.info(
                                    f"[apify_contacts] items_sample={raw_sample} company_id={company_id}"
                                )
                                logger.info(
                                    f"[apify_contacts] normalized_sample={norm_sample} company_id={company_id}"
                                )
                        except Exception:
                            pass
                        # Upsert into contacts table (best-effort)
                        try:
                            ins, upd = upsert_contacts_from_apify(company_id, contacts_raw)
                            logger.info(
                                f"[apify_contacts] upserted: inserted={ins}, updated={upd} company_id={company_id}"
                            )
                        except Exception:
                            logger.warning(
                                "[apify_contacts] upsert error; continuing", exc_info=True
                            )
                        # Verify any provided emails and upsert into lead_emails
                        try:
                            emails = [c.get("email") for c in contacts_raw if c.get("email")]
                            if emails:
                                verification = verify_emails(emails)
                                with get_db_connection() as conn:
                                    with conn.cursor() as cur:
                                        for ver in verification:
                                            email_verified = True if ver.get("status") == "valid" else False
                                            cur.execute(
                                                """
                                                INSERT INTO lead_emails (email, company_id, verification_status, smtp_confidence, source, last_verified_at)
                                                VALUES (%s,%s,%s,%s,%s, now())
                                                ON CONFLICT (email) DO UPDATE SET
                                                  company_id=EXCLUDED.company_id,
                                                  verification_status=EXCLUDED.verification_status,
                                                  smtp_confidence=EXCLUDED.smtp_confidence,
                                                  source=EXCLUDED.source,
                                                  last_verified_at=EXCLUDED.last_verified_at
                                                """,
                                                (
                                                    ver["email"],
                                                    company_id,
                                                    ver.get("status"),
                                                    ver.get("confidence"),
                                                    "apify_linkedin",
                                                ),
                                            )
                                logger.info(
                                    f"[apify_contacts] verified_emails={len(emails)} company_id={company_id}"
                                )
                        except Exception:
                            logger.warning(
                                "[apify_contacts] email verify/upsert skipped", exc_info=True
                            )
                        # Vendor usage + obs
                        _VENDOR_COUNTERS["apify_linkedin_calls"] = _VENDOR_COUNTERS.get("apify_linkedin_calls", 0) + 1
                        try:
                            if rid and tid:
                                _obs_vendor("apify_linkedin", calls=1)
                                _obs_log(
                                    "contact_discovery",
                                    "vendor_call",
                                    "ok",
                                    company_id=company_id,
                                    duration_ms=int((time.perf_counter() - t0) * 1000),
                                    extra={"queries": queries},
                                )
                        except Exception:
                            pass
                        state["apify_used"] = True
                        # Consider contacts added as mitigating trigger
                        if contacts_raw:
                            need_emails = need_emails and (len(emails or []) == 0)
                            needs_contacts = False
                            missing_names = False
                            missing_founder = False
                    except Exception as e:
                        try:
                            _obs_vendor("apify_linkedin", calls=1, errors=1)
                            _obs_log(
                                "contact_discovery",
                                "vendor_call",
                                "error",
                                company_id=company_id,
                                error_code=type(e).__name__,
                            )
                        except Exception:
                            pass
                        logger.warning(
                            "[apify_contacts] Apify LinkedIn call failed", exc_info=True
                        )
                        try:
                            (state.setdefault("degraded_reasons", [])) .append("APIFY_LINKEDIN_FAIL")
                        except Exception:
                            pass
        elif ENABLE_LUSHA_FALLBACK and LUSHA_API_KEY and trigger and _dec_cap("contact_lookups", 1):
            website_hint = data.get("website_domain") or state.get("home") or ""
            try:
                if website_hint.startswith("http"):
                    company_domain = urlparse(website_hint).netloc
                else:
                    company_domain = urlparse(f"https://{website_hint}").netloc
            except Exception:
                company_domain = None
            lusha_contacts: List[Dict[str, Any]] = []
            tid = int(_RUN_CTX.get("tenant_id") or 0)
            if not _cb_allows(tid, "lusha"):
                lusha_contacts = []
            else:
                async with AsyncLushaClient() as lc:
                    async def _call1():
                        return await lc.search_and_enrich_contacts(
                            company_name=state.get("company_name") or "",
                            company_domain=company_domain,
                            country=data.get("hq_country"),
                            titles=LUSHA_PREFERRED_TITLES,
                            limit=15,
                        )
                    try:
                        lusha_contacts = await with_retry(_call1, retry_on=(Exception,), policy=_DEFAULT_RETRY_POLICY)
                        _VENDOR_COUNTERS["lusha_lookups"] += 1
                        _obs_vendor("lusha", calls=1)
                        if tid:
                            _CB.on_success(tid, "lusha")
                    except Exception:
                        if tid:
                            _CB.on_error(tid, "lusha")
                        lusha_contacts = []
                    if not lusha_contacts:
                        async def _call2():
                            return await lc.search_and_enrich_contacts(
                                company_name=state.get("company_name") or "",
                                company_domain=company_domain,
                                country=data.get("hq_country"),
                                titles=None,
                                limit=15,
                            )
                        try:
                            lusha_contacts = await with_retry(_call2, retry_on=(Exception,), policy=_DEFAULT_RETRY_POLICY)
                            _VENDOR_COUNTERS["lusha_lookups"] += 1
                            _obs_vendor("lusha", calls=1)
                            if tid:
                                _CB.on_success(tid, "lusha")
                        except Exception:
                            if tid:
                                _CB.on_error(tid, "lusha")
                            lusha_contacts = []
            added_emails: List[str] = []
            added_phones: List[str] = []
            for c in lusha_contacts or []:
                for key in ("emails", "emailAddresses", "email_addresses"):
                    val = c.get(key)
                    if isinstance(val, list):
                        for e in val:
                            if isinstance(e, dict):
                                v = e.get("email") or e.get("value")
                                if v:
                                    added_emails.append(v)
                            elif isinstance(e, str):
                                added_emails.append(e)
                    elif isinstance(val, str):
                        added_emails.append(val)
                for key in ("phones", "phoneNumbers", "phone_numbers"):
                    val = c.get(key)
                    if isinstance(val, list):
                        for p in val:
                            if isinstance(p, dict):
                                v = (
                                    p.get("internationalNumber")
                                    or p.get("number")
                                    or p.get("value")
                                )
                                if v:
                                    added_phones.append(v)
                            elif isinstance(p, str):
                                added_phones.append(p)
                    elif isinstance(val, str):
                        added_phones.append(val)

            def _unique(seq: List[str]) -> List[str]:
                seen: set[str] = set()
                out: List[str] = []
                for x in seq:
                    if not x or x in seen:
                        continue
                    seen.add(x)
                    out.append(x)
                return out

            if added_emails or added_phones:
                data["email"] = _unique((data.get("email") or []) + added_emails)
                data["phone_number"] = _unique(
                    (data.get("phone_number") or []) + added_phones
                )
                logger.info(
                    f"       ↳ Lusha contacts fallback added {len(added_emails)} emails, {len(added_phones)} phones"
                )
            try:
                ins, upd = upsert_contacts_from_lusha(company_id, lusha_contacts or [])
                logger.info(
                    f"       ↳ Lusha contacts upserted: inserted={ins}, updated={upd}"
                )
            except Exception as _upsert_exc:
                logger.warning("       ↳ Lusha contacts upsert error", exc_info=True)
                try:
                    (state.setdefault("degraded_reasons", [])) .append("LUSHA_FAIL")
                except Exception:
                    pass
            state["lusha_used"] = True
    except Exception as _lusha_contacts_exc:
        logger.warning("       ↳ Lusha contacts fallback failed", exc_info=True)
        try:
            (state.setdefault("degraded_reasons", [])) .append("LUSHA_FAIL")
        except Exception:
            pass
    state["data"] = data
    return state


async def node_persist_core(state: EnrichmentState) -> EnrichmentState:
    if state.get("completed"):
        return state
    data = state.get("data") or {}
    company_id = state.get("company_id")
    if company_id and data:
        try:
            update_company_core_fields(company_id, data)
        except Exception as exc:
            logger.exception("   ↳ update_company_core_fields failed")
        # Best-effort projection of degradation reasons for this company
        try:
            reasons = ",".join(state.get("degraded_reasons") or []) or None
            if reasons:
                global _RUN_ANY_DEGRADED
                _RUN_ANY_DEGRADED = True
                conn = get_db_connection()
                with conn:
                    fields = {"company_id": company_id, "degraded_reasons": reasons}
                    tid = _default_tenant_id()
                    if tid is not None:
                        fields["tenant_id"] = tid
                    rid = _RUN_CTX.get("run_id")
                    if rid is not None:
                        fields["run_id"] = int(rid)
                    _insert_company_enrichment_run(conn, fields)
                try:
                    conn.close()
                except Exception:
                    pass
        except Exception:
            pass
    return state


async def node_persist_legacy(state: EnrichmentState) -> EnrichmentState:
    if state.get("completed"):
        return state
    data = state.get("data") or {}
    home = state.get("home") or ""
    company_id = state.get("company_id")
    if not (company_id and data and home):
        return state
    legacy = {
        "about_text": data.get("about_text") or "",
        "tech_stack": data.get("tech_stack") or [],
        "public_emails": data.get("email") or [],
        "jobs_count": 0,
        "linkedin_url": data.get("linkedin_url"),
        "phone_number": data.get("phone_number") or [],
        "hq_city": data.get("hq_city"),
        "hq_country": data.get("hq_country"),
        "website_domain": data.get("website_domain") or home,
        "email": data.get("email") or [],
        "products_services": data.get("products_services") or [],
        "value_props": data.get("value_props") or [],
        "pricing": data.get("pricing") or [],
    }
    try:
        # Run blocking store in a worker thread to avoid blocking the event loop
        await asyncio.to_thread(store_enrichment, company_id, home, legacy)
        logger.info(f"    💾 stored extracted fields for company_id={company_id}")
        state["completed"] = True
    except Exception as exc:
        logger.exception("   ↳ store_enrichment failed")
    return state


# Build the LangGraph for enrichment
enrichment_graph = StateGraph(EnrichmentState)
enrichment_graph.add_node("find_domain", node_find_domain)
 
enrichment_graph.add_node("deterministic_crawl", node_deterministic_crawl)
enrichment_graph.add_node("discover_urls", node_discover_urls)
enrichment_graph.add_node("expand_crawl", node_expand_crawl)
enrichment_graph.add_node("extract_pages", node_extract_pages)
enrichment_graph.add_node("build_chunks", node_build_chunks)
enrichment_graph.add_node("llm_extract", node_llm_extract)
enrichment_graph.add_node("apify_contacts", node_apify_contacts)
enrichment_graph.add_node("persist_core", node_persist_core)
enrichment_graph.add_node("persist_legacy", node_persist_legacy)

enrichment_graph.set_entry_point("find_domain")
enrichment_graph.add_edge("find_domain", "deterministic_crawl")


def _after_deterministic(state: EnrichmentState) -> str:
    return "build_chunks" if state.get("extracted_pages") else "discover_urls"


enrichment_graph.add_conditional_edges(
    "deterministic_crawl",
    _after_deterministic,
    {"build_chunks": "build_chunks", "discover_urls": "discover_urls"},
)
enrichment_graph.add_edge("discover_urls", "expand_crawl")
enrichment_graph.add_edge("expand_crawl", "extract_pages")
enrichment_graph.add_edge("extract_pages", "build_chunks")
enrichment_graph.add_edge("build_chunks", "llm_extract")
enrichment_graph.add_edge("llm_extract", "apify_contacts")
enrichment_graph.add_edge("apify_contacts", "persist_core")
enrichment_graph.add_edge("persist_core", "persist_legacy")

enrichment_agent = enrichment_graph.compile()
try:
    enrichment_agent.get_graph().draw_mermaid_png()
except Exception as e:
    logger.debug("enrichment graph diagram generation skipped", exc_info=True)


def _normalize_company_name(name: str) -> list[str]:
    n = (name or "").lower()
    # Replace & with 'and', remove punctuation
    n = n.replace("&", " and ")
    n = re.sub(r"[^a-z0-9\s-]", " ", n)
    parts = [p for p in re.split(r"\s+", n) if p]
    # Remove common suffixes
    SUFFIXES = {
        "pte",
        "pte.",
        "ltd",
        "ltd.",
        "inc",
        "inc.",
        "co",
        "co.",
        "company",
        "corp",
        "corp.",
        "llc",
        "plc",
        "limited",
        "holdings",
        "group",
        "singapore",
    }
    core = [p for p in parts if p not in SUFFIXES]
    # Keep first 2-3 tokens for matching
    return core[:3] or parts[:2]


# -------- Agentic planner (optional, feature-flagged) -------------------------

def _summarize_state_for_agent(state: "EnrichmentState") -> str:
    try:
        have_domain = bool(state.get("home"))
        pages = len(state.get("extracted_pages") or [])
        urls = len(state.get("filtered_urls") or [])
        data = state.get("data") or {}
        keys = [k for k, v in (data or {}).items() if v]
        parts = [
            f"have_domain={have_domain}",
            f"filtered_urls={urls}",
            f"extracted_pages={pages}",
            f"data_keys={keys[:8]}",
        ]
        return ", ".join(parts)
    except Exception:
        return "have_domain=?, filtered_urls=?, extracted_pages=?, data_keys=[]"


AGENT_ACTIONS = [
    # Domain
    "use_existing_domain",  # rely on DB/company state
    "search_domain",        # call node_find_domain
    # Crawl and content
    "deterministic_crawl",  # node_deterministic_crawl
    "discover_urls",        # node_discover_urls
    "expand_crawl",         # node_expand_crawl
    "extract_pages",        # node_extract_pages
    "build_chunks",         # node_build_chunks
    "llm_extract",          # node_llm_extract
    # Contacts
    "apify_contacts",       # node_apify_contacts
    # Company info via Apify
    # Persistence
    "persist_core",         # node_persist_core
    "persist_legacy",       # node_persist_legacy
    # Stop
    "finish",
]


async def _agent_execute(action: str, state: "EnrichmentState") -> "EnrichmentState":
    # Route to existing nodes to avoid duplicating logic.
    if action == "use_existing_domain":
        # node_find_domain already checks DB first, so reuse it
        return await node_find_domain(state)
    if action == "search_domain":
        return await node_find_domain(state)
    if action == "deterministic_crawl":
        return await node_deterministic_crawl(state)
    if action == "discover_urls":
        return await node_discover_urls(state)
    if action == "expand_crawl":
        return await node_expand_crawl(state)
    if action == "extract_pages":
        return await node_extract_pages(state)
    if action == "build_chunks":
        return await node_build_chunks(state)
    if action == "llm_extract":
        return await node_llm_extract(state)
    if action == "apify_contacts":
        return await node_apify_contacts(state)
    if action == "persist_core":
        return await node_persist_core(state)
    if action == "persist_legacy":
        return await node_persist_legacy(state)
    return state


def _agent_prompt(company_name: str, summary: str) -> str:
    return (
        "You are an enrichment planner agent. Your goal is to enrich a company with: "
        "website domain, key firmographics (about_text, tech_stack, linkedin_url, phones, HQ), and contact persons (prefer decision-makers). "
        "You have a set of actions. At each step, choose the best next action given the current summary, avoiding redundant work and respecting that vendor calls are capped.\n\n"
        f"Company: {company_name}\n"
        f"State: {summary}\n\n"
        "Available actions (JSON only):\n"
        "- use_existing_domain: rely on any existing domain (node_find_domain does this).\n"
        "- search_domain: search for domain if missing.\n"
        "- deterministic_crawl: fetch homepage + deterministic pages if domain exists.\n"
        "- discover_urls: pick relevant site URLs.\n"
        "- expand_crawl: add about/contact/careers pages.\n"
        "- extract_pages: fetch pages for extraction.\n"
        "- build_chunks: merge/trim content for LLM.\n"
        "- llm_extract: extract fields from chunks.\n"
        "- apify_contacts: discover contacts when emails/contacts are missing.\n"
        "- persist_core: upsert core fields to DB.\n"
        "- persist_legacy: write legacy projection and finish.\n"
        "- finish: stop when data is sufficient or after persistence.\n\n"
        "Return strictly JSON: {\"action\": <one of actions>, \"reason\": <short string>}\n"
        "Prefer to finish after persist_legacy or when state.completed is true."
    )


async def run_enrichment_agentic(state: "EnrichmentState") -> "EnrichmentState":
    steps = 0
    # Reuse global llm with low temperature for determinism
    while steps < int(ENRICH_AGENTIC_MAX_STEPS):
        # Stop if pipeline marked completed
        if state.get("completed"):
            logger.info("[agentic] state.completed=true; stopping")
            break
        summary = _summarize_state_for_agent(state)
        prompt = _agent_prompt(state.get("company_name") or "", summary)
        try:
            out = llm.invoke(prompt)
            # Extract content from ChatMessage if present
            content = None
            try:
                content = getattr(out, "content", None)
            except Exception:
                content = None
            if not content:
                content = str(out) if out is not None else ""
            # Try strict JSON parse first
            parsed = {}
            try:
                parsed = json.loads(content)
            except Exception:
                # Heuristic: find JSON substring
                s = content
                start = s.find("{")
                end = s.rfind("}")
                if start != -1 and end != -1 and end > start:
                    try:
                        parsed = json.loads(s[start : end + 1])
                    except Exception:
                        parsed = {}
            action = (parsed or {}).get("action")
            reason = (parsed or {}).get("reason") or ""
        except Exception:
            action = None
            reason = "planner_exception"

        # Choose safe default or override when parsing fails or action invalid/loops
        have_domain = bool(state.get("home"))
        have_pages = bool(state.get("extracted_pages") or [])
        have_chunks = bool(state.get("chunks") or [])
        have_data = bool(state.get("data") or {})
        # Prefer to run Apify contacts when emails/contacts are missing
        should_run_apify = False
        try:
            dat = state.get("data") or {}
            need_emails = not (dat.get("email") or [])
            need_phones = not (dat.get("phone_number") or [])
            needs_contacts = False
            cid = state.get("company_id")
            if cid:
                total, has_named, _founder = _get_contact_stats(int(cid))
                needs_contacts = (int(total) == 0) or (not has_named)
            prefer_apify = ENABLE_APIFY_LINKEDIN or (not ENABLE_LUSHA_FALLBACK) or (not LUSHA_API_KEY)
            should_run_apify = bool(prefer_apify and (need_emails or need_phones or needs_contacts))
        except Exception:
            should_run_apify = False
        if not action or action not in AGENT_ACTIONS:
            if not have_domain:
                action = "search_domain"
                reason = reason or "default_no_domain"
            elif not have_pages:
                action = "deterministic_crawl"
                reason = reason or "default_need_pages"
            elif not have_chunks:
                action = "build_chunks"
                reason = reason or "default_build_chunks"
            elif not have_data:
                action = "llm_extract"
                reason = reason or "default_llm_extract"
            else:
                # If we have core data but are missing contacts/emails, run Apify before persisting
                if should_run_apify:
                    action = "apify_contacts"
                    reason = reason or "default_contacts_missing"
                else:
                    action = "persist_legacy"
                    reason = reason or "default_persist_finish"
        else:
            # Override repetitive crawl/expand suggestions to progress the pipeline
            if have_pages and not have_chunks and action in ("expand_crawl", "deterministic_crawl"):
                action = "build_chunks"
                reason = "override_progress_build_chunks"
            elif have_chunks and not have_data and action in ("expand_crawl", "deterministic_crawl"):
                action = "llm_extract"
                reason = "override_progress_llm_extract"
            elif have_data and action in ("expand_crawl", "deterministic_crawl", "discover_urls"):
                if should_run_apify:
                    action = "apify_contacts"
                    reason = "override_contacts_missing"
                else:
                    action = "persist_legacy"
                    reason = "override_progress_persist"
            # If the planner suggests persisting/finishing but contacts are missing, run Apify first
            elif action in ("persist_core", "persist_legacy", "finish") and should_run_apify and not bool(state.get("apify_used")):
                action = "apify_contacts"
                reason = "override_contacts_before_persist"

        # keep routing overrides minimal

        logger.info(f"[agentic] step={steps+1} action={action} reason={reason}")
        if action == "finish":
            break
        try:
            state = await _agent_execute(action, state)
        except Exception:
            logger.warning("[agentic] action execution failed; continuing", exc_info=True)
        steps += 1
    return state


def find_domain(company_name: str) -> list[str]:
    print(f"    🔍 Search domain for '{company_name}'")
    if tavily_client is None:
        print("       ↳ Tavily client not initialized.")
        return []

    core = _normalize_company_name(company_name)
    normalized_query = " ".join(core)
    name_nospace = "".join(core)
    name_hyphen = "-".join(core)

    # 1) Use normalized name first, fall back to quoted variants
    try:
        queries = [
            f"{normalized_query} official website",
            f'"{company_name}" "official website"',
            f'"{company_name}" site:.sg',
            f"{company_name} official website",
        ]
        response = None
        for q in queries:
            if not _dec_cap("tavily_units", 1):
                break
            try:
                response = tavily_client.search(q)
                _VENDOR_COUNTERS["tavily_queries"] += 1
                _obs_vendor("tavily", calls=1)
            except Exception:
                response = None
            if isinstance(response, dict) and response.get("results"):
                break
        if not isinstance(response, dict) or not response.get("results"):
            print("       ↳ No results from Tavily search.")
            return []
    except Exception as exc:
        print(f"       ↳ Search error: {exc}")
        return []

    # Filter URLs to those containing the core company name (first two words)
    filtered_urls: list[str] = []
    AGGREGATORS = {
        "linkedin.com",
        "facebook.com",
        "twitter.com",
        "x.com",
        "instagram.com",
        "youtube.com",
        "tiktok.com",
        "glassdoor.com",
        "indeed.com",
        "jobsdb.com",
        "jobstreet.com",
        "mycareersfuture.gov.sg",
        "wikipedia.org",
        "crunchbase.com",
        "bloomberg.com",
        "reuters.com",
        "medium.com",
        "shopify.com",
        "lazada.sg",
        "shopee.sg",
        "shopee.com",
        "amazon.com",
        "ebay.com",
        "alibaba.com",
        "google.com",
        "maps.google.com",
        "goo.gl",
        "g2.com",
        "capterra.com",
        "tripadvisor.com",
        "expedia.com",
        "yelp.com",
        "recordowl.com",
        "sgpgrid.com",
        # Add common non-official/aggregator content domains observed
        "made-in-china.com",
        "morepaper.org",
        "artbarblog.com",
        "jumpfrompaper.com",
    }
    for h in response["results"]:
        url = h.get("url") if isinstance(h, dict) else None
        print("       ↳ Found URL:", url)
        if not url:
            continue
        parsed = urlparse(url)
        netloc = parsed.netloc.lower()
        if netloc.startswith("www."):
            netloc_stripped = netloc[4:]
        else:
            netloc_stripped = netloc
        apex = (
            ".".join(netloc_stripped.split(".")[-2:])
            if "." in netloc_stripped
            else netloc_stripped
        )
        apex_label = apex.split(".")[0]
        domain_label = netloc_stripped.split(".")[0]

        is_aggregator = apex in AGGREGATORS
        is_sg = netloc_stripped.endswith(".sg") or apex.endswith(".sg")
        is_brand_exact = (
            apex_label == name_nospace
            or domain_label.replace("-", "") == name_nospace
        )

        # page text signals
        title = (h.get("title") or "").lower()
        snippet = (h.get("content") or h.get("snippet") or "").lower()
        text = f"{title} {snippet}"
        label_match = (
            name_nospace in domain_label.replace("-", "")
            or name_hyphen in netloc_stripped
            or (core and core[0] in domain_label)
        )
        text_match = all(part in text for part in core)

        # Enforce heuristics:
        # - Reject marketplaces/aggregators (unless the brand name equals the aggregator apex e.g., Amazon)
        if is_aggregator and not is_brand_exact:
            continue
        # - Require name evidence in domain label or page text (or exact brand apex)
        if not (label_match or text_match or is_brand_exact):
            continue
        # Previously we forced .sg or exact brand; relax to accept legitimate non-.sg brand domains
        # as long as aggregator is excluded and name evidence is present.

        filtered_urls.append(url)

    # Rank: prefer .sg TLD, then shorter apex domains, then https
    def _rank(u: str) -> tuple:
        p = urlparse(u)
        host = p.netloc.lower()
        host_stripped = host[4:] if host.startswith("www.") else host
        labels = host_stripped.split(".")
        apex = ".".join(labels[-2:]) if len(labels) >= 2 else host_stripped
        apex_label = apex.split(".")[0]
        domain_label = host_stripped.split(".")[0]
        is_brand_exact_r = (
            apex_label == name_nospace or domain_label.replace("-", "") == name_nospace
        )
        tld_sg = host_stripped.endswith(".sg") or apex.endswith(".sg")
        return (
            0 if is_brand_exact_r else 1,
            0 if tld_sg else 1,
            len(labels),
            0 if p.scheme == "https" else 1,
            u,
        )

    if filtered_urls:
        filtered_urls = sorted(set(filtered_urls), key=_rank)
        # Only keep top 2 that most likely represent the official company website
        filtered_urls = filtered_urls[:2]
        print(f"       ↳ Filtered URLs: {filtered_urls}")
        return filtered_urls
    print("       ↳ No matching URLs found after heuristics.")
    return []


def qualify_pages(pages: list[dict], threshold: int = 4) -> list[dict]:
    print(f"    🔍 Qualifying {len(pages)} pages")
    prompt = PromptTemplate(
        input_variables=["url", "title", "content"],
        template=(
            "You are a qualifier agent. Given the following page, score 1–5 whether this is our official website or About Us page.\n"
            'Return JSON {{"score":<int>,"reason":"<reason>"}}.\n\n'
            "URL: {url}\n"
            "Title: {title}\n"
            "Content: {content}\n"
        ),
    )
    chain = prompt | llm | StrOutputParser()
    accepted = []
    for p in pages:
        url = p.get("url") or ""
        title = p.get("title") or ""
        content = p.get("content") or ""
        try:
            output = chain.invoke({"url": url, "title": title, "content": content})
            result = json.loads(output)
            score = result.get("score", 0)
            reason = result.get("reason", "")
            if score >= threshold:
                p["qualifier_reason"] = reason
                p["score"] = score
                accepted.append(p)
        except Exception as exc:
            print(f"       ↳ Qualify error for {url}: {exc}")
    return accepted


def extract_website_data(url: str) -> dict:
    print(f"    🌐 extract_website_data('{url}')")
    schema = {
        "about_text": "str",
        "tech_stack": "list[str]",
        "public_emails": "list[str]",
        "jobs_count": "int",
        "linkedin_url": "str",
        "hq_city": "str",
        "hq_country": "str",
        "phone_number": "str",
    }

    # 1) Crawl starting from the root of the given URL
    parsed_url = urlparse(url)
    root = f"{parsed_url.scheme}://{parsed_url.netloc}"
    # Crawl root to get subpage URLs
    try:
        print("       ↳ Crawling for subpages…")
        crawl_input = {
            "url": f"{root}/*",
            "limit": 20,
            "crawl_depth": 2,
            "enable_web_search": False,
        }
        if tavily_crawl is not None and _dec_cap("tavily_units", 1):
            crawl_result = tavily_crawl.run(crawl_input)
            _VENDOR_COUNTERS["tavily_crawl_calls"] += 1
            _obs_vendor("tavily", calls=1)
            raw_urls = crawl_result.get("results") or crawl_result.get("urls") or []
        else:
            raw_urls = []
    except Exception as exc:
        print(f"       ↳ Crawl error: {exc}")
        raw_urls = []

    # normalize to unique URLs
    page_urls = []
    for u in raw_urls:
        if isinstance(u, dict) and u.get("url"):
            page_urls.append(u["url"])
        elif isinstance(u, str) and u.startswith("http"):
            page_urls.append(u)
    # Ensure the original URL (or root) is processed first
    page_urls.insert(0, url)
    page_urls = list(dict.fromkeys(page_urls))
    print(f"       ↳ {len(page_urls)} unique pages discovered")

    aggregated = {k: None for k in schema}

    # 2) For each page: extract raw_content, then refine via AI Agent
    for url in page_urls:
        print(f"       ↳ Processing page: {url}")

        # a) Extract raw_content via TavilyExtract
        payload = {
            "urls": [url],
            "schema": {"raw_content": "str"},
            "instructions": "Retrieve the main textual content from this page.",
        }
        try:
            if tavily_extract is not None and _dec_cap("tavily_units", 1):
                tid = int(_RUN_CTX.get("tenant_id") or 0)
                if tid and not _CB.allow(tid, "tavily"):
                    raise RuntimeError("tavily_circuit_open")
                import time as _t, random as _rnd
                attempts = 0
                raw_data = None
                while attempts < 3:
                    try:
                        raw_data = tavily_extract.run(payload)
                        _VENDOR_COUNTERS["tavily_extract_calls"] += 1
                        _obs_vendor("tavily", calls=1)
                        if tid:
                            _CB.on_success(tid, "tavily")
                        break
                    except Exception:
                        attempts += 1
                        if tid:
                            _CB.on_error(tid, "tavily")
                        if attempts >= 3:
                            raise
                        delay = 0.25 * (2 ** (attempts - 1)) * (0.8 + 0.4 * _rnd.random())
                        _t.sleep(delay)
            else:
                raise RuntimeError("tavily_units_cap_reached")
        # print("          ↳ Tavily raw_data:", raw_data)
        except Exception as exc:
            print(f"          ↳ TavilyExtract error: {exc}")
            continue

        # b) Pull raw_content (top-level or nested)
        raw_content = None
        if isinstance(raw_data, dict):
            # top-level
            raw_content = raw_data.get("raw_content")
            # nested under results
            if (
                raw_content is None
                and isinstance(raw_data.get("results"), list)
                and raw_data["results"]
            ):
                raw_content = raw_data["results"][0].get("raw_content")
        if (
            not raw_content
            or not isinstance(raw_content, str)
            or not raw_content.strip()
        ):
            print("          ↳ No or empty raw_content found, skipping AI extraction.")
            continue
        print(f"          ↳ raw_content length: {len(raw_content)} characters")

        # 3) AI extraction
        try:
            print("          ↳ AI extraction:")
            ai_output = extract_chain.invoke(
                {
                    "raw_content": raw_content,
                    "schema_keys": list(schema.keys()),
                    "instructions": (
                        "Extract the About Us text, list of technologies, public business emails, "
                        "open job listing count, LinkedIn URL, HQ city & country, and phone number."
                    ),
                }
            )
            # Raw AI output string
            print("          ↳ AI output string:")
            print(ai_output)
            # Pretty-print AI output JSON
            try:
                parsed = json.loads(ai_output)
                print("          ↳ AI output JSON:")
                print(json.dumps(parsed, indent=2))
                page_data = parsed
            except json.JSONDecodeError as exc:
                print(f"          ↳ AI extraction JSON parse error: {exc}")
                continue
            page_data = json.loads(ai_output)
        except Exception as exc:
            print(f"          ↳ AI extraction error: {exc}")
            continue

        # 4) Merge into aggregated
        for key in schema:
            val = page_data.get(key)
            if val is None:
                continue
            if isinstance(val, list):
                base = aggregated[key] or []
                aggregated[key] = list({*base, *val})
            else:
                aggregated[key] = val

    print(f"       ↳ Final aggregated data: {aggregated}")
    return aggregated


def verify_emails(emails: list[str]) -> list[dict]:
    """
    2.4 Email Verification via ZeroBounce adapter.
    Adapter returns dicts: {email, status, confidence, source}.
    """
    print(f"    🔒 ZeroBounce Email Verification for {_redact_email_list(emails)}")
    results: list[dict] = []
    if not emails:
        return results
    # Skip verification entirely if no API key configured
    if not ZEROBOUNCE_API_KEY:
        return results
    conn = None
    try:
        conn = get_db_connection()
        _ensure_email_cache_table(conn)
        conn.commit()
    except Exception:
        pass
    for e in emails:
        try:
            # In-memory cache
            if e in ZB_CACHE:
                results.append(ZB_CACHE[e])
                continue
            # DB cache
            cached = _cache_get(conn, e) if conn else None
            if cached:
                ZB_CACHE[e] = cached
                results.append(cached)
                continue
            # Throttle to respect credits (simple delay)
            time.sleep(0.75)
            t0 = time.perf_counter()
            resp = requests.get(
                "https://api.zerobounce.net/v2/validate",
                params={"api_key": ZEROBOUNCE_API_KEY, "email": e, "ip_address": ""},
                timeout=10,
            )
            status_code = getattr(resp, "status_code", 200)
            data = resp.json()
            status = data.get("status", "unknown")
            confidence = float(data.get("confidence", 0.0))
            rec = {
                "email": e,
                "status": status,
                "confidence": confidence,
                "source": "zerobounce",
            }
            try:
                # Vendor usage: bump one call per verify; include rate-limit/quota flags; record duration
                rl = 1 if status_code == 429 else 0
                quota = True if status_code == 402 else False
                _obs_vendor("zerobounce", calls=1, rate_limit_hits=rl, quota_exhausted=quota)
                # Optional cost per check
                try:
                    cost_per = float(os.getenv("ZEROBOUNCE_COST_PER_CHECK", "0.0") or 0.0)
                except Exception:
                    cost_per = 0.0
                if cost_per > 0:
                    rid = _RUN_CTX.get("run_id"); tid = _RUN_CTX.get("tenant_id")
                    if rid and tid:
                        _obs_bump(int(rid), int(tid), "zerobounce", calls=0, errors=0, cost_usd=cost_per)
                _obs_log("verify_emails", "vendor_call", "ok", duration_ms=int((time.perf_counter()-t0)*1000), extra={"email": _mask_email(e), "status": status})
            except Exception:
                pass
            if conn:
                _cache_set(conn, e, status, confidence)
                try:
                    conn.commit()
                except Exception:
                    pass
            ZB_CACHE[e] = rec
            print(
                f"       ✅ ZeroBounce result for {_mask_email(e)}: status={status}, confidence={confidence}"
            )
        except Exception as exc:
            print(f"       ⚠️ ZeroBounce API error for {_mask_email(e)}: {exc}")
            status = "unknown"
            confidence = 0.0
            try:
                _obs_vendor("zerobounce", calls=1, errors=1)
                _obs_log("verify_emails", "vendor_call", "error", error_code=type(exc).__name__, extra={"email": _mask_email(e)})
            except Exception:
                pass
            results.append(
                {
                    "email": e,
                    "status": status,
                    "confidence": confidence,
                    "source": "zerobounce",
                }
            )
    return results


def _persist_corpus(
    company_id: Optional[int], corpus: str, page_count: int, source: str = "tavily"
) -> None:
    if not company_id or not corpus:
        return
    conn = get_db_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS crawl_corpus (
                      id BIGSERIAL PRIMARY KEY,
                      company_id BIGINT NOT NULL,
                      page_count INT,
                      source TEXT,
                      corpus TEXT,
                      created_at TIMESTAMPTZ DEFAULT now()
                    );
                    """
                )
                cur.execute(
                    """
                    INSERT INTO crawl_corpus (company_id, page_count, source, corpus)
                    VALUES (%s,%s,%s,%s)
                    """,
                    (company_id, page_count, source, corpus),
                )
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _normalize_phone_list(values: list[str]) -> list[str]:
    out: list[str] = []
    for v in values or []:
        s = (v or "").strip()
        if not s:
            continue
        # Keep leading + and digits only
        if s.startswith("+"):
            num = "+" + "".join(ch for ch in s if ch.isdigit())
        else:
            digits = "".join(ch for ch in s if ch.isdigit())
            # Heuristic: 8 digits -> assume Singapore local, prefix +65
            if len(digits) == 8:
                num = "+65" + digits
            elif len(digits) >= 9:
                num = "+" + digits
            else:
                num = digits
        if num and num not in out:
            out.append(num)
    return out


def store_enrichment(company_id: int, domain: str, data: dict):
    print(f"    💾 store_enrichment({company_id}, {domain})")
    conn = get_db_connection()
    embedding = get_embedding(data.get("about_text", "") or "")
    verification = verify_emails(data.get("public_emails") or [])

    # Normalize domain (apex, lowercase) and phone list
    try:
        apex = urlparse(domain).netloc.lower() or domain.lower()
    except Exception:
        apex = (domain or "").lower()
    phones_norm = _normalize_phone_list(data.get("phone_number") or [])

    with conn:
        fields2 = {
            "company_id": company_id,
            "about_text": data.get("about_text"),
            "tech_stack": (data.get("tech_stack") or []),
            "public_emails": (data.get("public_emails") or []),
            "jobs_count": data.get("jobs_count"),
            "linkedin_url": data.get("linkedin_url"),
            "verification_results": Json(verification),
            "embedding": embedding,
        }
        tid2 = _default_tenant_id()
        if tid2 is not None:
            fields2["tenant_id"] = tid2
        _insert_company_enrichment_run(
            conn,
            fields2,
        )
        print("       ↳ history saved")
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE companies SET
                  website_domain=%s, linkedin_url=%s, tech_stack=%s,
                  email=%s, phone_number=%s, hq_city=%s, hq_country=%s,
                  last_seen=now()
                WHERE company_id=%s
                """,
                (
                    apex,
                    data.get("linkedin_url"),
                    (
                        data.get("tech_stack")
                        if isinstance(data.get("tech_stack"), list)
                        else (
                            [data.get("tech_stack")] if data.get("tech_stack") else None
                        )
                    ),
                    (
                        data.get("public_emails")
                        if isinstance(data.get("public_emails"), list)
                        else (
                            [data.get("public_emails")]
                            if data.get("public_emails")
                            else None
                        )
                    ),
                    phones_norm,
                    data.get("hq_city"),
                    data.get("hq_country"),
                    company_id,
                ),
            )
            print("       ↳ companies updated")

            for ver in verification:
                email_verified = True if ver.get("status") == "valid" else False
                contact_source = ver.get("source", "zerobounce")
                cur.execute(
                    """
                    INSERT INTO contacts
                      (company_id,email,email_verified,verification_confidence,
                       contact_source,created_at,updated_at)
                    VALUES (%s,%s,%s,%s,%s,now(),now())
                    ON CONFLICT DO NOTHING
                    """,
                    (
                        company_id,
                        ver["email"],
                        email_verified,
                        ver["confidence"],
                        contact_source,
                    ),
                )
                # Also write to lead_emails if table exists
                try:
                    cur.execute(
                        """
                        INSERT INTO lead_emails (email, company_id, verification_status, smtp_confidence, source, last_verified_at)
                        VALUES (%s,%s,%s,%s,%s, now())
                        ON CONFLICT (email) DO UPDATE SET
                          company_id=EXCLUDED.company_id,
                          verification_status=EXCLUDED.verification_status,
                          smtp_confidence=EXCLUDED.smtp_confidence,
                          source=EXCLUDED.source,
                          last_verified_at=now()
                        """,
                        (
                            ver["email"],
                            company_id,
                            ver.get("status"),
                            ver.get("confidence"),
                            contact_source,
                        ),
                    )
                except Exception:
                    pass
            print("       ↳ contacts inserted")

    conn.close()
    print(f"    ✅ Done enrichment for company_id={company_id}\n")


async def enrich_company(company_id: int, company_name: str):
    # 1) find domain (your current method)
    urls = [u for u in find_domain(company_name) if u]  # filter out None/empty
    if not urls:
        print("   ↳ No domain found; skipping")
        return
    url = urls[0]

    # 2) jina reader snapshot first
    try:
        text = jina_read(url, timeout=8) or ""
        about_text = text[:1000]
        tech_stack: list[str] = []
        public_emails: list[str] = []
        jobs_count = 0

        print("about_text:", about_text, "tech_stack:", tech_stack, "public_emails:", public_emails, "jobs_count:", jobs_count)

        conn = get_db_connection()
        with conn:
            fields3 = {
                "company_id": company_id,
                "about_text": about_text,
                "tech_stack": tech_stack,
                "public_emails": public_emails,
                "jobs_count": jobs_count,
                "linkedin_url": None,
            }
            tid3 = _default_tenant_id()
            if tid3 is not None:
                fields3["tenant_id"] = tid3
            _insert_company_enrichment_run(
                conn,
                fields3,
            )
        conn.close()

        # Prepare data dict for store_enrichment (best-effort for all fields)
        # Heuristics for city/country: use 'Singapore' if '.sg' TLD; else None
        def guess_city_country(url):
            city = None
            country = None
            if url.lower().endswith(".sg/") or ".sg" in url.lower():
                city = country = "Singapore"
            return city, country

        hq_city, hq_country = guess_city_country(url)
        website_domain = (
            urlparse(url).netloc.lower()
            if url.startswith("http")
            else (url or "").lower()
        )
        data = {
            "about_text": about_text,
            "tech_stack": tech_stack,
            "public_emails": public_emails,
            "jobs_count": jobs_count,
            "linkedin_url": None,
            "phone_number": [],
            "hq_city": hq_city,
            "hq_country": hq_country,
            "website_domain": website_domain,
            "email": public_emails,  # all emails
            "products_services": [],
            "value_props": [],
            "pricing": [],
            # You can add more fields here as needed
        }
        print(
            "DEBUG: Data dict to store_enrichment:",
            json.dumps(data, indent=2, default=str),
        )
        store_enrichment(company_id, url, data)
        return  # success; skip LLM/Tavily path

    except Exception as exc:
        import traceback

        print(f"   ↳ Jina snapshot failed: {exc}. Falling back to Tavily/LLM.")
        traceback.print_exc()

    # 4) fallback to your existing Tavily + LLM extraction (current code path)
    data = extract_website_data(url)  # your existing function
    # …persist as you already do
    print(f"▶️  Enriching company_id={company_id}, name='{company_name}'")
    domains = find_domain(company_name)
    if not domains:
        print(f"   ⚠️ Skipping {company_id}: no domains found\n")
        return
    # Extract and store enrichment for each domain URL
    for idx, domain_url in enumerate(domains, start=1):
        print(f"    🌐 Processing domain ({idx}/{len(domains)}): {domain_url}")
        data = extract_website_data(domain_url)
        print(data)
        store_enrichment(company_id, domain_url, data)
