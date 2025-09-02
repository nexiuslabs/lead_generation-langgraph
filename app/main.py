# app/main.py
import csv
import logging
import os
import re
from io import StringIO

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langserve import add_routes

from app.pre_sdr_graph import build_graph
from src.database import get_conn, get_pg_pool

logger = logging.getLogger("input_norm")
if not logger.handlers:
    h = logging.StreamHandler()
    fmt = logging.Formatter(
        "[%(levelname)s] %(asctime)s %(name)s :: %(message)s", "%H:%M:%S"
    )
    h.setFormatter(fmt)
    logger.addHandler(h)
logger.setLevel("INFO")

# Ensure LangGraph checkpoint directory exists to prevent FileNotFoundError
# e.g., '.langgraph_api/.langgraph_checkpoint.*.pckl.tmp'
CHECKPOINT_DIR = os.environ.get("LANGGRAPH_CHECKPOINT_DIR", ".langgraph_api")
try:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
except Exception as e:
    logger.warning("Failed to ensure checkpoint dir %s: %s", CHECKPOINT_DIR, e)

app = FastAPI(title="Pre-SDR LangGraph Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

graph = build_graph()


def _role_to_type(role: str) -> str:
    r = (role or "").lower()
    if r in ("user", "human"):
        return "human"
    if r in ("assistant", "ai"):
        return "ai"
    if r == "system":
        return "system"
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


def _upsert_companies_from_staging_by_industries_old(industries: list[str]) -> int:
    """Fetch rows from staging_acra_companies matching the provided industries and upsert into companies.

    Matching is based on LOWER(primary_ssic_description) equality to any provided term.
    Upsert strategy:
      - Try to locate an existing companies row by UEN, or by (lower(name) OR website_domain)
      - If found: UPDATE set core fields and last_seen
      - Else: INSERT a new row (omit company_id so default/sequence can assign)
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

            src_uen = pick("uen", "uen_no", "uen_number") or "NULL"
            src_name = pick("entity_name", "name", "company_name") or "NULL"
            src_desc = pick(
                "primary_ssic_description", "ssic_description", "industry_description"
            )
            src_code = pick("primary_ssic_code", "industry_code") or "NULL"
            src_web = (
                pick("website", "website_url", "website_domain", "url", "homepage")
                or "NULL"
            )
            src_year = (
                pick(
                    "incorporation_year",
                    "year_incorporated",
                    "inc_year",
                    "founded_year",
                )
                or "NULL"
            )
            src_stat = (
                pick(
                    "entity_status_de",
                    "entity_status",
                    "status",
                    "entity_status_description",
                )
                or "NULL"
            )

            if not src_desc:
                return 0

            like_patterns = [f"%{t}%" for t in industries]
            select_sql = f"""
                SELECT
                  {src_uen} AS uen,
                  {src_name} AS entity_name,
                  {src_desc} AS primary_ssic_description,
                  {src_code} AS primary_ssic_code,
                  CAST({src_code} AS TEXT) AS ssic_code,

                  {src_web}  AS website,
                  {src_year} AS incorporation_year,
                  {src_stat} AS entity_status_de
                FROM staging_acra_companies sc
                WHERE LOWER({src_desc}) = ANY(%s)
                   OR LOWER({src_desc}) ILIKE ANY(%s)
                LIMIT 1000
            """
            cur.execute(select_sql, (industries, like_patterns))
            rows = cur.fetchall()
            if not rows:
                return 0
            for (
                uen,
                entity_name,
                ssic_desc,
                primary_ssic_code,
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
                    sg_registered = (status_de or "").strip().lower() in {
                        "live",
                        "registered",
                        "existing",
                    }
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
                    set_sql = (
                        ", ".join(set_parts) + ", last_seen = NOW()"
                        if set_parts
                        else "last_seen = NOW()"
                    )
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


def _upsert_companies_from_staging_by_industries(industries: list[str]) -> int:
    """Fetch staging rows by SSIC code and upsert into companies.

    Industry terms are resolved to SSIC codes via ``ssic_ref_latest``. Matching rows
    are pulled from ``staging_acra_companies`` by ``primary_ssic_code`` and
    inserted or updated into ``companies``.
    """
    if not industries:
        return 0
    affected = 0
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT LOWER(column_name)
                FROM information_schema.columns
                WHERE table_name = 'staging_acra_companies'
                """,
            )
            cols = {r[0] for r in cur.fetchall()}

            def pick(*names: str) -> str | None:
                for n in names:
                    if n.lower() in cols:
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
            src_web = (
                pick("website", "website_url", "website_domain", "url", "homepage")
                or "NULL"
            )
            src_year = (
                pick(
                    "incorporation_year",
                    "year_incorporated",
                    "inc_year",
                    "founded_year",
                )
                or "NULL"
            )
            src_stat = (
                pick(
                    "entity_status_de",
                    "entity_status",
                    "status",
                    "entity_status_description",
                )
                or "NULL"
            )

            if src_code == "NULL":
                return 0

            norm_inds = [
                (t or "").strip().lower() for t in industries if (t or "").strip()
            ]
            like_patterns = [f"%{t}%" for t in norm_inds]
            cur.execute(
                """
                SELECT DISTINCT code AS ssic_code, LOWER(description)
                FROM ssic_ref

                WHERE LOWER(description) = ANY(%s)
                   OR LOWER(title) = ANY(%s)
                   OR LOWER(description) LIKE ANY(%s)
                   OR LOWER(title) LIKE ANY(%s)
                """,
                (norm_inds, norm_inds, like_patterns, like_patterns),
            )
            code_rows = cur.fetchall()
            code_map = {
                str(code): (desc or "") for code, desc in code_rows if code is not None
            }
            codes = list(code_map.keys())
            if not codes:
                return 0

            select_sql = f"""
                SELECT
                  {src_uen} AS uen,
                  {src_name} AS entity_name,
                  {src_desc} AS primary_ssic_description,
                  {src_code} AS primary_ssic_code,

                  CAST({src_code} AS TEXT) AS ssic_code,

                  {src_web}  AS website,
                  {src_year} AS incorporation_year,
                  {src_stat} AS entity_status_de
                FROM staging_acra_companies sc

                WHERE CAST({src_code} AS TEXT) = ANY(%s)
                LIMIT 1000
            """
            cur.execute(select_sql, (codes,))
            rows = cur.fetchall()
            if not rows:
                return 0
            for (
                uen,
                entity_name,
                ssic_desc,
                primary_ssic_code,
                ssic_code,
                website,
                inc_year,
                status_de,
            ) in rows:
                name = (entity_name or "").strip() or None
                desc_lower = (ssic_desc or "").strip().lower()
                industry_norm = code_map.get(str(ssic_code)) or desc_lower or None
                industry_code = str(ssic_code) if ssic_code is not None else None
                website_domain = (website or "").strip() or None
                sg_registered = None
                try:
                    sg_registered = (status_de or "").strip().lower() in {
                        "live",
                        "registered",
                        "existing",
                    }
                except Exception:
                    pass

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
                    set_parts = []
                    params = []
                    for k, v in fields.items():
                        if v is not None:
                            set_parts.append(f"{k} = %s")
                            params.append(v)
                    set_sql = (
                        ", ".join(set_parts) + ", last_seen = NOW()"
                        if set_parts
                        else "last_seen = NOW()"
                    )
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
                logger.info(
                    "Upserted %d companies from staging by industries=%s",
                    affected,
                    inds,
                )
    except Exception as _e:
        # Never block the chat flow; log and continue
        logger.warning("input-normalization staging sync failed: %s", _e)

    return state


ui_adapter = RunnableLambda(normalize_input) | graph

# expose adapted runnable so /agent accepts role-based payloads
add_routes(app, ui_adapter, path="/agent")


@app.get("/health")
def health():
    return {"ok": True}


# --- Lightweight tenant middleware (optional header-based) ---
@app.middleware("http")
async def tenant_middleware(request: Request, call_next):
    # Extract tenant_id from header (e.g., set by SSO/edge); no validation here
    tenant_id = request.headers.get("X-Tenant-ID") or None
    request.state.tenant_id = tenant_id
    return await call_next(request)


# --- Export endpoints (JSON/CSV) ---
@app.get("/export/latest_scores.json")
async def export_latest_scores_json(limit: int = 200):
    pool = await get_pg_pool()
    async with pool.acquire() as conn:
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
async def export_latest_scores_csv(limit: int = 200):
    pool = await get_pg_pool()
    async with pool.acquire() as conn:
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
    writer = csv.DictWriter(
        buf,
        fieldnames=(
            list(rows[0].keys())
            if rows
            else [
                "company_id",
                "name",
                "industry_norm",
                "employees_est",
                "score",
                "bucket",
                "rationale",
            ]
        ),
    )
    writer.writeheader()
    for r in rows:
        writer.writerow(dict(r))
    return Response(content=buf.getvalue(), media_type="text/csv")
