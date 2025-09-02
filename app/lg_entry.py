# app/lg_entry.py
import logging
import re
from typing import Any, Dict, List, Union

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, StateGraph

from app.pre_sdr_graph import GraphState, build_graph  # new dynamic builder
from src.database import get_conn

logger = logging.getLogger("input_norm")
if not logger.handlers:
    h = logging.StreamHandler()
    fmt = logging.Formatter(
        "[%(levelname)s] %(asctime)s %(name)s :: %(message)s", "%H:%M:%S"
    )
    h.setFormatter(fmt)
    logger.addHandler(h)
logger.setLevel("INFO")

Content = Union[str, List[dict], dict, None]


def _role_to_type(role: str) -> str:
    r = (role or "").lower()
    if r in ("user", "human"):
        return "human"
    if r in ("assistant", "ai"):
        return "ai"
    if r == "system":
        return "system"
    return "human"


def _flatten_content(content: Content) -> str:
    """
    Accepts UI message content in various shapes and returns a plain string.
    Examples:
      - "hello"
      - [{"type":"input_text","text":"hello"}, {"type":"image_url",...}]
      - {"text": "..."}
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        # Common shape from SDKs
        if "text" in content and isinstance(content["text"], str):
            return content["text"]
        return str(content)
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                if "text" in item and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif item.get("type") in ("input_text", "text") and isinstance(
                    item.get("text"), str
                ):
                    parts.append(item["text"])
                elif "image_url" in item:
                    parts.append("[image]")
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return "\n".join([p for p in parts if p])
    # Fallback stringify
    return str(content)


def _to_message(msg: dict | BaseMessage) -> BaseMessage:
    if isinstance(msg, BaseMessage):
        # Ensure content is a string
        if not isinstance(msg.content, str):
            # Best-effort conversion
            text = _flatten_content(msg.content)  # type: ignore[arg-type]
            # Recreate message with string content to avoid mutating internals
            if isinstance(msg, HumanMessage):
                return HumanMessage(content=text)
            if isinstance(msg, SystemMessage):
                return SystemMessage(content=text)
            return AIMessage(content=text)
        return msg
    mtype = msg.get("type") or _role_to_type(msg.get("role", "human"))
    content = _flatten_content(msg.get("content"))
    if mtype == "human":
        return HumanMessage(content=content)
    if mtype == "system":
        return SystemMessage(content=content)
    return AIMessage(content=content)


def _extract_industry_terms(text: str) -> List[str]:
    if not text:
        return []
    chunks = re.split(r"[,\n;:=]+|\band\b|\bor\b|/|\\\\|\|", text, flags=re.IGNORECASE)
    terms: List[str] = []
    # Extract explicit key-value patterns like "industry = technology" or "industries: fintech"
    for m in re.findall(
        r"\b(?:industry|industries|sector|sectors)\s*[:=]\s*([^\n,;|/\\]+)",
        text,
        flags=re.IGNORECASE,
    ):
        s = (m or "").strip()
        if s:
            terms.append(s.lower())
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
        "industry",
        "industries",
        "sector",
        "sectors",
    }
    for c in chunks:
        s = (c or "").strip()
        if not s or len(s) < 2:
            continue
        if not re.search(r"[a-zA-Z]", s):
            continue
        sl = s.lower()
        if sl in stop:
            continue
        sl = re.sub(r"\s+", " ", sl)
        terms.append(sl)
    seen = set()
    out: List[str] = []
    for t in terms:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out[:20]


def _collect_industry_terms(messages: List[BaseMessage] | None) -> List[str]:
    if not messages:
        return []
    seen = set()
    out: List[str] = []
    for m in messages:
        # Parse terms from all message roles; filtering is handled inside extractor
        for t in _extract_industry_terms((m.content or "")):
            if t not in seen:
                seen.add(t)
                out.append(t)
                if len(out) >= 20:
                    return out
    return out


def _upsert_companies_from_staging_by_industries_old(industries: List[str]) -> int:
    if not industries:
        return 0
    affected = 0
    try:
        with get_conn() as conn, conn.cursor() as cur:
            # Normalize and log incoming terms
            lower_terms = [
                ((t or "").strip().lower()) for t in industries if (t or "").strip()
            ]
            lower_terms = [t for t in lower_terms if t]
            like_patterns = [f"%{t}%" for t in lower_terms]
            logger.info("Upsert from staging: industry terms=%s", lower_terms)
            # Introspect available columns to build a safe SELECT
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
            # Include broader variants for description and code columns
            src_desc = pick(
                "primary_ssic_description",
                "ssic_description",
                "industry_description",
                "industry",
                "industry_name",
                "primary_industry",
                "primary_industry_desc",
                "industry_desc",
                "sector",
                "primary_sector",
                "sector_description",
            )
            src_code = pick(
                "primary_ssic_code",
                "industry_code",
                "primary_ssic",
                "primary_industry_code",
            )
            # Prefer registration_incorporation_date but extract YEAR when present
            src_year = (
                pick(
                    "registration_incorporation_date",
                    "incorporation_year",
                    "year_incorporated",
                    "inc_year",
                    "founded_year",
                )
                or "NULL"
            )
            # Build an expression that yields a numeric year
            if (
                isinstance(src_year, str)
                and src_year.lower() == "registration_incorporation_date"
            ):
                src_year_expr = f"EXTRACT(YEAR FROM CAST({src_year} AS date))::int"
            else:
                src_year_expr = src_year
            src_stat = (
                pick(
                    "entity_status_de",
                    "entity_status",
                    "status",
                    "entity_status_description",
                )
                or "NULL"
            )

            if not src_desc or not src_code:
                logger.warning(
                    "staging_acra_companies missing required columns. desc=%s code=%s (available=%s)",
                    src_desc,
                    src_code,
                    sorted(list(cols))[:20],
                )
                return 0

            logger.info(
                "Staging columns used -> desc=%s, code=%s, name=%s, uen=%s, year=%s, status=%s",
                src_desc,
                src_code,
                src_name,
                src_uen,
                src_year,
                src_stat,
            )

            # Step 1: Resolve SSIC codes from industry description matches (exact or partial)
            codes_sql = f"""
                SELECT DISTINCT CAST({src_code} AS TEXT) AS code
                FROM staging_acra_companies
                WHERE LOWER({src_desc}) = ANY(%s::text[])
                   OR {src_desc} ILIKE ANY(%s::text[])
            """
            cur.execute(codes_sql, (lower_terms, like_patterns))
            code_rows = cur.fetchall()
            code_list = [r[0] for r in code_rows if r and r[0] is not None]

            if code_list:
                # Log resolved SSIC codes for traceability (preview up to 50)
                codes_preview = ", ".join([str(c) for c in code_list[:50]])
                if len(code_list) > 50:
                    codes_preview += f", ... (+{len(code_list)-50} more)"
                logger.info(
                    "Resolved %d SSIC codes from industries=%s: %s",
                    len(code_list),
                    lower_terms,
                    codes_preview,
                )

                # Step 2: Fetch all companies by resolved SSIC codes and upsert
                select_sql = f"""
                    SELECT
                      {src_uen} AS uen,
                      {src_name} AS entity_name,
                      {src_desc} AS primary_ssic_description,
                      {src_code} AS primary_ssic_code,

                      CAST({src_code} AS TEXT) AS ssic_code,
                      {src_year_expr} AS incorporation_year,
                      {src_stat} AS entity_status_de
                    FROM staging_acra_companies sc

                    WHERE CAST({src_code} AS TEXT) = ANY(%s::text[])
                """
                select_params = (code_list,)
                source_mode = "ssic"
            else:
                # Fallback: select by description patterns directly
                logger.warning(
                    "No SSIC codes resolved for industries=%s. Falling back to description match.",
                    lower_terms,
                )
                select_sql = f"""
                    SELECT
                      {src_uen} AS uen,
                      {src_name} AS entity_name,
                      {src_desc} AS primary_ssic_description,
                      {src_code} AS primary_ssic_code,
                      CAST({src_code} AS TEXT) AS ssic_code,
                      {src_year_expr} AS incorporation_year,
                      {src_stat} AS entity_status_de
                    FROM staging_acra_companies sc

                    WHERE LOWER({src_desc}) = ANY(%s::text[])
                       OR {src_desc} ILIKE ANY(%s::text[])
                """
                select_params = (lower_terms, like_patterns)
                source_mode = "description"

            # Pre-count for visibility
            if source_mode == "ssic":
                count_sql = f"SELECT COUNT(*) FROM staging_acra_companies WHERE CAST({src_code} AS TEXT) = ANY(%s::text[])"
                count_params = (code_list,)
            else:
                count_sql = f"SELECT COUNT(*) FROM staging_acra_companies WHERE LOWER({src_desc}) = ANY(%s::text[]) OR {src_desc} ILIKE ANY(%s::text[])"
                count_params = (lower_terms, like_patterns)
            cur.execute(count_sql, count_params)
            total_matches = cur.fetchone()[0] or 0
            logger.info("Matched %d staging rows by %s", total_matches, source_mode)

            # Stream rows using a server-side cursor; use a separate cursor for upserts
            cur_sel = conn.cursor(name="staging_upsert_sel")
            cur_sel.itersize = 500
            cur_sel.execute(select_sql, select_params)
            batch_size = 500
            logger.info(
                "Upserting staging companies by %s in batches of %d",
                source_mode,
                batch_size,
            )
            processed = 0
            while True:
                rows = cur_sel.fetchmany(batch_size)
                if not rows:
                    break
                logger.info(
                    "Processing batch of %d rows (processed=%d/%d)",
                    len(rows),
                    processed,
                    total_matches,
                )
                with conn.cursor() as cur_up:
                    for (
                        uen,
                        entity_name,
                        ssic_desc,
                        primary_ssic_code,
                        ssic_code,
                        inc_year,
                        status_de,
                    ) in rows:
                        name = (entity_name or "").strip() or None
                        desc_lower = (ssic_desc or "").strip().lower()
                        match_term = None
                        for t in lower_terms:
                            if desc_lower == t or (t in desc_lower):
                                match_term = t
                                break
                        industry_norm = (match_term or desc_lower) or None
                        industry_code = (
                            str(ssic_code) if ssic_code is not None else None
                        )
                        sg_registered = None
                        try:
                            sg_registered = (status_de or "").strip().lower() in {
                                "live",
                                "registered",
                                "existing",
                            }
                        except Exception:
                            pass

                        # Locate existing company
                        company_id = None
                        if uen:
                            cur_up.execute(
                                "SELECT company_id FROM companies WHERE uen = %s LIMIT 1",
                                (uen,),
                            )
                            row = cur_up.fetchone()
                            if row:
                                company_id = row[0]
                        if company_id is None and name:
                            cur_up.execute(
                                "SELECT company_id FROM companies WHERE LOWER(name) = LOWER(%s) LIMIT 1",
                                (name,),
                            )
                            row = cur_up.fetchone()
                            if row:
                                company_id = row[0]

                        fields = {
                            "uen": uen,
                            "name": name,
                            "industry_norm": industry_norm,
                            "industry_code": industry_code,
                            # Set both incorporation_year and founded_year from the same source year
                            "incorporation_year": inc_year,
                            "founded_year": inc_year,
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
                            cur_up.execute(
                                f"UPDATE companies SET {set_sql} WHERE company_id = %s",
                                params + [company_id],
                            )
                            affected += cur_up.rowcount or 0
                        else:
                            cols = [k for k, v in fields.items() if v is not None]
                            vals = [fields[k] for k in cols]
                            cols_sql = ", ".join(cols)
                            ph = ",".join(["%s"] * len(vals))
                            cur_up.execute(
                                f"INSERT INTO companies ({cols_sql}) VALUES ({ph}) RETURNING company_id",
                                vals,
                            )
                            new_id = cur_up.fetchone()[0]
                            cur_up.execute(
                                "UPDATE companies SET last_seen = NOW() WHERE company_id = %s",
                                (new_id,),
                            )
                            affected += 1
                processed += len(rows)
            logger.info(
                "Finished upserting by %s (%d rows processed, %d affected)",
                source_mode,
                processed,
                affected,
            )
        return affected
    except Exception as e:
        logger.warning("staging upsert skipped: %s", e)
        return 0


def _upsert_companies_from_staging_by_industries(industries: List[str]) -> int:
    """Fetch staging rows by SSIC code and upsert into companies.

    Industry terms are mapped to SSIC codes via ``ssic_ref_latest``; matching
    staging rows (by ``primary_ssic_code``) are then inserted or updated in
    ``companies``.
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
            src_year = (
                pick(
                    "registration_incorporation_date",
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

            # resolve codes
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
            rows = cur.fetchall()
            code_map = {
                str(code): (desc or "") for code, desc in rows if code is not None
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
                inc_year,
                status_de,
            ) in rows:
                name = (entity_name or "").strip() or None
                desc_lower = (ssic_desc or "").strip().lower()
                industry_norm = code_map.get(str(ssic_code)) or desc_lower or None
                industry_code = str(ssic_code) if ssic_code is not None else None
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

                fields = {
                    "uen": uen,
                    "name": name,
                    "industry_norm": industry_norm,
                    "industry_code": industry_code,
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


def _normalize(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Agent Chat UI will call /threads/.../runs with a body like:
      {"assistant_id":"agent","input":{"messages":[{"role":"human","content":"start"}]}}
    We map it to the graph state: {"messages": [BaseMessage,...], "candidates": ...}
    """
    data = payload.get("input", payload) or {}
    msgs = data.get("messages") or []
    if isinstance(msgs, dict):  # sometimes a single message object is sent
        msgs = [msgs]

    norm_msgs = [_to_message(m) for m in msgs] or [HumanMessage(content="")]
    state: Dict[str, Any] = {"messages": norm_msgs}

    # optional “companies”/“candidates” passthrough for your graph
    if "candidates" in data:
        state["candidates"] = data["candidates"]
    elif "companies" in data:
        state["candidates"] = data["companies"]

    # Best-effort staging→companies upsert based on any human-mentioned industries across the message history.
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
        logger.warning("input-normalization staging sync failed: %s", _e)

    return state


def make_graph(config: Dict[str, Any] | None = None):
    """Called by `langgraph dev` to get a valid compiled Graph.

    We wrap the existing compiled pre-SDR graph with a tiny outer graph that
    normalizes Chat UI payloads into the expected PreSDRState. Returning a
    compiled StateGraph ensures the dev server's graph validation passes.
    """
    inner = build_graph()  # compiled inner graph (dynamic Pre-SDR pipeline)

    def normalize_node(payload: Dict[str, Any]) -> GraphState:
        # Accept raw UI payload and coerce into graph state
        state = _normalize(payload)
        # type: ignore[return-value] — runtime shape matches PreSDRState
        return state  # type: ignore

    outer = StateGraph(GraphState)
    outer.add_node("normalize", normalize_node)
    outer.add_node("presdr", inner)
    outer.set_entry_point("normalize")
    outer.add_edge("normalize", "presdr")
    outer.add_edge("presdr", END)
    return outer.compile()
