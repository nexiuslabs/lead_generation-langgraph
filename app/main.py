# app/main.py
from fastapi import FastAPI, Request, Response, Depends, BackgroundTasks, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from app.onboarding import handle_first_login, get_onboarding_status
from app.odoo_connection_info import get_odoo_connection_info
from src.database import get_pg_pool, get_conn
from app.auth import require_auth, require_identity, require_optional_identity
from src.icp import _find_ssic_codes_by_terms
from app.odoo_store import OdooStore
from src.settings import OPENAI_API_KEY
import csv
from io import StringIO
import logging
import re
import os

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

# Ensure LangGraph checkpoint directory exists to prevent FileNotFoundError
# e.g., '.langgraph_api/.langgraph_checkpoint.*.pckl.tmp'
CHECKPOINT_DIR = os.environ.get("LANGGRAPH_CHECKPOINT_DIR", ".langgraph_api")
try:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
except Exception as e:
    logger.warning("Failed to ensure checkpoint dir %s: %s", CHECKPOINT_DIR, e)

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

# Lazily enable /agent routes only when LLM is configured to avoid import-time errors
graph = None

# Mount auth cookie routes
try:
    from app.auth_routes import router as auth_router
    app.include_router(auth_router)
except Exception as _e:
    logger.warning("Auth routes not mounted: %s", _e)

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
    }
    for c in chunks:
        s = (c or "").strip()
        if not s or len(s) < 2:
            continue
        # Thin out non-alpha heavy tokens
        if not re.search(r"[a-zA-Z]", s):
            continue
        sl = s.lower()
        if sl in stop:
            continue
        # Common formatting artifacts
        sl = re.sub(r"\s+", " ", sl)
        terms.append(sl)
    # Dedupe while preserving order
    seen = set()
    out: list[str] = []
    for t in terms:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out[:10]


def _collect_industry_terms(messages: list[BaseMessage] | None) -> list[str]:
    if not messages:
        return []
    seen = set()
    out: list[str] = []
    for m in messages:
        if isinstance(m, HumanMessage):
            for t in _extract_industry_terms((m.content or "")):
                if t not in seen:
                    seen.add(t)
                    out.append(t)
                    if len(out) >= 20:
                        return out
    return out


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
            src_year = pick('incorporation_year','year_incorporated','inc_year','founded_year') or 'NULL'
            src_stat = pick('entity_status_de','entity_status','status','entity_status_description') or 'NULL'

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
                      {src_stat} AS entity_status_de
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
                      {src_stat} AS entity_status_de
                    FROM staging_acra_companies
                    WHERE LOWER({src_desc}) = ANY(%s)
                       OR LOWER({src_desc}) ILIKE ANY(%s)
                    LIMIT 1000
                """
                cur.execute(select_sql, (lower_terms, like_patterns))
            rows = cur.fetchall()
            if code_list and rows:
                # Log matched names for visibility (preview up to 50)
                try:
                    name_idx = 1  # entity_name is second selected column
                    names = [(r[name_idx] or "").strip() for r in rows]
                    names = [n for n in names if n]
                    names_preview = ", ".join(names[:50])
                    extra = f", ... (+{len(names)-50} more)" if len(names) > 50 else ""
                    logger.info("staging_acra_companies matched %d rows by SSIC code; names: %s%s", len(names), names_preview, extra)
                except Exception:
                    pass
            if not rows:
                return 0
            for (
                uen,
                entity_name,
                ssic_desc,
                ssic_code,
                website,
                inc_year,
                status_de,
            ) in rows:
                name = (entity_name or "").strip() or None
                desc_lower = (ssic_desc or "").strip().lower()
                # Prefer setting industry_norm to the user's term if it appears in the description
                match_term = None
                for t in industries:
                    if desc_lower == t or (t in desc_lower):
                        match_term = t
                        break
                industry_norm = (match_term or desc_lower) or None
                industry_code = str(ssic_code) if ssic_code is not None else None
                website_domain = (website or "").strip() or None
                sg_registered = None
                try:
                    sg_registered = (
                        (status_de or "").strip().lower() in {"live", "registered", "existing"}
                    )
                except Exception:
                    pass

                # Try locate existing company
                company_id = None
                if uen:
                    cur.execute(
                        "SELECT company_id FROM companies WHERE uen = %s LIMIT 1",
                        (uen,),
                    )
                    row = cur.fetchone()
                    if row:
                        company_id = row[0]
                if company_id is None and name:
                    cur.execute(
                        "SELECT company_id FROM companies WHERE LOWER(name) = LOWER(%s) LIMIT 1",
                        (name,),
                    )
                    row = cur.fetchone()
                    if row:
                        company_id = row[0]
                if company_id is None and website_domain:
                    cur.execute(
                        "SELECT company_id FROM companies WHERE website_domain = %s LIMIT 1",
                        (website_domain,),
                    )
                    row = cur.fetchone()
                    if row:
                        company_id = row[0]

                fields = {
                    "uen": uen,
                    "name": name,
                    "industry_norm": industry_norm,
                    "industry_code": industry_code,
                    "website_domain": website_domain,
                    "incorporation_year": inc_year,
                    "sg_registered": sg_registered,
                }

                if company_id is not None:
                    # Build dynamic update for non-null values
                    set_parts = []
                    params = []
                    for k, v in fields.items():
                        if v is not None:
                            set_parts.append(f"{k} = %s")
                            params.append(v)
                    set_sql = ", ".join(set_parts) + ", last_seen = NOW()" if set_parts else "last_seen = NOW()"
                    cur.execute(
                        f"UPDATE companies SET {set_sql} WHERE company_id = %s",
                        params + [company_id],
                    )
                    affected += cur.rowcount or 0
                else:
                    cols = [k for k, v in fields.items() if v is not None]
                    vals = [fields[k] for k in cols]
                    cols_sql = ", ".join(cols)
                    ph = ",".join(["%s"] * len(vals))
                    cur.execute(
                        f"INSERT INTO companies ({cols_sql}) VALUES ({ph}) RETURNING company_id",
                        vals,
                    )
                    new_id = cur.fetchone()[0]
                    cur.execute(
                        "UPDATE companies SET last_seen = NOW() WHERE company_id = %s",
                        (new_id,),
                    )
                    affected += 1
        return affected
    except Exception as e:
        logger.warning("staging upsert skipped: %s", e)
        return 0

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

    # NEW: best-effort staging→companies upsert for industries mentioned in the latest user message.
    try:
        inds = _collect_industry_terms(state.get("messages"))
        if inds:
            affected = _upsert_companies_from_staging_by_industries(inds)
            if affected:
                logger.info("Upserted %d companies from staging by industries=%s", affected, inds)
    except Exception as _e:
        # Never block the chat flow; log and continue
        logger.warning("input-normalization staging sync failed: %s", _e)

    return state

ENABLE_LANGSERVE_IN_APP = os.getenv("ENABLE_LANGSERVE_IN_APP", "false").lower() in ("1", "true", "yes", "on")
try:
    if ENABLE_LANGSERVE_IN_APP and OPENAI_API_KEY:
        # Import inside conditional to avoid loading langserve when mounted into LangGraph Server
        from langserve import add_routes  # type: ignore
        from app.pre_sdr_graph import build_graph  # type: ignore

        graph = build_graph()
        ui_adapter = RunnableLambda(normalize_input) | graph
        add_routes(app, ui_adapter, path="/agent")
        logger.info("/agent routes enabled (LLM configured)")
    else:
        logger.info("Skipping /agent routes (ENABLE_LANGSERVE_IN_APP is false or OPENAI_API_KEY missing)")
except Exception as e:
    # Never block API/docs if LangServe wiring fails; just log and continue
    logger.warning("Skipping /agent routes due to initialization error: %s", e)

@app.get("/info")
async def info(_: dict = Depends(require_auth)):
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
        await handle_first_login(email, tenant_id_claim)

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
async def export_latest_scores_json(limit: int = 200, request: Request = None, _: dict = Depends(require_auth)):
    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        # Set per-request tenant GUC for RLS
        try:
            if request and getattr(request.state, "tenant_id", None):
                await conn.execute("SELECT set_config('request.tenant_id', $1, true)", request.state.tenant_id)
        except Exception:
            pass
        rows = await conn.fetch(
            """
            SELECT c.company_id, c.name, c.website_domain, c.industry_norm, c.employees_est,
                   s.score, s.bucket, s.rationale
            FROM companies c
            JOIN lead_scores s ON s.company_id = c.company_id
            ORDER BY s.score DESC NULLS LAST
            LIMIT $1
            """,
            limit,
        )
    return [dict(r) for r in rows]


@app.get("/export/latest_scores.csv")
async def export_latest_scores_csv(limit: int = 200, request: Request = None, _: dict = Depends(require_auth)):
    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        try:
            if request and getattr(request.state, "tenant_id", None):
                await conn.execute("SELECT set_config('request.tenant_id', $1, true)", request.state.tenant_id)
        except Exception:
            pass
        rows = await conn.fetch(
            """
            SELECT c.company_id, c.name, c.industry_norm, c.employees_est,
                   s.score, s.bucket, s.rationale
            FROM companies c
            JOIN lead_scores s ON s.company_id = c.company_id
            ORDER BY s.score DESC NULLS LAST
            LIMIT $1
            """,
            limit,
        )
    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()) if rows else [
        "company_id","name","industry_norm","employees_est","score","bucket","rationale"
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
            ORDER BY s.score DESC NULLS LAST
            LIMIT $1
            """,
            limit,
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
