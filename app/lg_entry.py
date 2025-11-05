# app/lg_entry.py
from typing import Dict, Any, List, Union
import os
import asyncio
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END
from app.pre_sdr_graph import build_graph, GraphState  # new dynamic builder
from app.langgraph_logging import LangGraphTroubleshootHandler
from src.database import get_conn
from src.icp import _find_ssic_codes_by_terms
import logging
import re

logger = logging.getLogger("input_norm")
if not logger.handlers:
    h = logging.StreamHandler()
    fmt = logging.Formatter("[%(levelname)s] %(asctime)s %(name)s :: %(message)s", "%H:%M:%S")
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
                elif item.get("type") in ("input_text", "text") and isinstance(item.get("text"), str):
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
    # Strip URLs to avoid misclassifying them as industry tokens
    try:
        text = re.sub(r"https?://\S+", " ", text)
    except Exception:
        pass
    chunks = re.split(r"[,\n;:=]+|\band\b|\bor\b|/|\\\\|\|", text, flags=re.IGNORECASE)
    terms: List[str] = []
    # Extract explicit key-value patterns like "industry = technology" or "industries: fintech"
    for m in re.findall(r"\b(?:industry|industries|sector|sectors)\s*[:=]\s*([^\n,;|/\\]+)", text, flags=re.IGNORECASE):
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
        # chat commands / control phrases
        "accept micro-icp",
        "accept micro‑icp",
        "accept micro icp",
        "accept a micro-icp",
        "accept a micro‑icp",
        "accept a micro icp",
        "industry",
        "industries",
        "sector",
        "sectors",
        # conversational fillers
        "start",
        "which",
        "which industries",
        "problem spaces",
        "should we target",
        "e.g.",
        "eg",
        # revenue buckets that shouldn't be treated as industries
        "small",
        "medium",
        "large",
        # URL/common web tokens
        "http",
        "https",
        "www",
    }
    for c in chunks:
        s = (c or "").strip()
        if not s or len(s) < 2:
            continue
        if not re.search(r"[a-zA-Z]", s):
            continue
        # Skip bare domains (e.g., example.com) — not industries
        try:
            if re.search(r"\b([a-z0-9-]+\.)+[a-z]{2,}\b", s, flags=re.IGNORECASE):
                continue
        except Exception:
            pass
        # Drop bullets/parentheticals and obvious question fragments
        if s.strip().startswith("-"):
            continue
        if any(ch in s for ch in ("?", "(", ")", ":")):
            continue
        sl = s.lower()
        if sl in stop:
            continue
        sl = re.sub(r"\s+", " ", sl)
        terms.append(sl)
    # Dedupe, prefer multi-word phrases
    seen = set()
    out: List[str] = []
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


def _collect_industry_terms(messages: List[BaseMessage] | None) -> List[str]:
    if not messages:
        return []
    # Use only the most recent Human message to avoid including assistant prompts
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return _extract_industry_terms(m.content or "")
    return []


# Feature 18 flags (defaults)
STAGING_UPSERT_MODE = os.getenv("STAGING_UPSERT_MODE", "background").strip().lower()
try:
    UPSERT_SYNC_LIMIT = int(os.getenv("UPSERT_SYNC_LIMIT", "10") or 10)
except Exception:
    UPSERT_SYNC_LIMIT = 10
def _upsert_companies_from_staging_by_industries(industries: List[str]) -> int:
    if not industries:
        return 0
    affected = 0
    try:
        with get_conn() as conn, conn.cursor() as cur:
            # Normalize and log incoming terms
            lower_terms = [((t or "").strip().lower()) for t in industries if (t or "").strip()]
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
            src_uen = pick('uen','uen_no','uen_number') or 'NULL'
            src_name = pick('entity_name','name','company_name') or 'NULL'
            # Include broader variants for description and code columns
            src_desc = pick(
                'primary_ssic_description', 'ssic_description', 'industry_description',
                'industry', 'industry_name', 'primary_industry', 'primary_industry_desc',
                'industry_desc', 'sector', 'primary_sector', 'sector_description'
            )
            src_code = pick(
                'primary_ssic_code', 'ssic_code', 'industry_code', 'ssic', 'primary_ssic',
                'primary_industry_code'
            )
            # Prefer registration_incorporation_date but robustly extract a 4-digit YEAR from text
            src_year = pick('registration_incorporation_date','incorporation_year','year_incorporated','inc_year','founded_year') or 'NULL'
            # Use regex substring to be resilient to text-formatted dates
            if isinstance(src_year, str):
                src_year_expr = f"NULLIF(substring({src_year}::text from '\\d{{4}}'), '')::int"
            else:
                src_year_expr = 'NULL'
            src_stat = pick('entity_status_de','entity_status','status','entity_status_description') or 'NULL'
            src_owner = pick('business_constitution_description','company_type_description','entity_type_description','paf_constitution_description','ownership_type') or 'NULL'

            if not src_desc or not src_code:
                logger.warning(
                    "staging_acra_companies missing required columns. desc=%s code=%s (available=%s)",
                    src_desc, src_code, sorted(list(cols))[:20],
                )
                return 0

            logger.info(
                "Staging columns used -> desc=%s, code=%s, name=%s, uen=%s, year=%s, status=%s",
                src_desc, src_code, src_name, src_uen, src_year, src_stat,
            )

            # Step 1: Resolve SSIC codes via ssic_ref using free-text industry terms
            ssic_matches = _find_ssic_codes_by_terms(lower_terms)
            code_list = [c for (c, _title, _score) in ssic_matches]
            if code_list:
                codes_preview = ", ".join([str(c) for c in code_list[:50]])
                if len(code_list) > 50:
                    codes_preview += f", ... (+{len(code_list)-50} more)"
                logger.info("ssic_ref resolved %d SSIC codes from industries=%s: %s", len(code_list), lower_terms, codes_preview)

            if code_list:
                # Log resolved SSIC codes for traceability (preview up to 50)
                codes_preview = ", ".join([str(c) for c in code_list[:50]])
                if len(code_list) > 50:
                    codes_preview += f", ... (+{len(code_list)-50} more)"
                logger.info("Resolved %d SSIC codes from industries=%s: %s", len(code_list), lower_terms, codes_preview)

                # Step 2: Fetch all companies by resolved SSIC codes and upsert
                select_sql = f"""
                    SELECT
                      {src_uen} AS uen,
                      {src_name} AS entity_name,
                      {src_desc} AS primary_ssic_description,
                      {src_code} AS primary_ssic_code,
                      {src_year_expr} AS incorporation_year,
                      {src_stat} AS entity_status_de,
                      {src_owner} AS ownership_type
                    FROM staging_acra_companies
                    WHERE regexp_replace({src_code}::text, '\\D', '', 'g') = ANY(%s::text[])
                """
                select_params = (code_list,)
                source_mode = 'ssic'
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
                      {src_year_expr} AS incorporation_year,
                      {src_stat} AS entity_status_de,
                      {src_owner} AS ownership_type
                    FROM staging_acra_companies
                    WHERE LOWER({src_desc}) = ANY(%s::text[])
                       OR {src_desc} ILIKE ANY(%s::text[])
                """
                select_params = (lower_terms, like_patterns)
                source_mode = 'description'

            # Pre-count for visibility
            if source_mode == 'ssic':
                count_sql = f"SELECT COUNT(*) FROM staging_acra_companies WHERE regexp_replace({src_code}::text, '\\D', '', 'g') = ANY(%s::text[])"
                count_params = (code_list,)
            else:
                count_sql = f"SELECT COUNT(*) FROM staging_acra_companies WHERE LOWER({src_desc}) = ANY(%s::text[]) OR {src_desc} ILIKE ANY(%s::text[])"
                count_params = (lower_terms, like_patterns)
            cur.execute(count_sql, count_params)
            _row = cur.fetchone()
            try:
                total_matches = int(_row[0]) if (_row and len(_row) >= 1 and _row[0] is not None) else 0
            except Exception:
                total_matches = 0
            logger.info("Matched %d staging rows by %s", total_matches, source_mode)

            # Stream rows using a server-side cursor; use a separate cursor for upserts
            cur_sel = conn.cursor(name="staging_upsert_sel")
            cur_sel.itersize = 500
            cur_sel.execute(select_sql, select_params)
            batch_size = 500
            logger.info("Upserting staging companies by %s in batches of %d", source_mode, batch_size)
            processed = 0
            names_preview_list: List[str] = []  # collect first ~50 names that matched codes

            # Defer alias discovery until after first fetch to guarantee description is populated
            col_aliases: List[str | None] = []

            def row_to_map(row: object, aliases: List[str | None]) -> Dict[str, Any]:
                try:
                    if not isinstance(row, (list, tuple)):
                        return {}
                    limit = min(len(aliases), len(row)) if aliases else 0
                    out: Dict[str, Any] = {}
                    for i in range(limit):
                        key = aliases[i]
                        if key:
                            out[key] = row[i]
                    return out
                except Exception:
                    return {}

            first_batch = True
            while True:
                rows = cur_sel.fetchmany(batch_size)
                if not rows:
                    break
                logger.info("Processing batch of %d rows (processed=%d/%d)", len(rows), processed, total_matches)
                # Build alias list on first batch
                if first_batch:
                    try:
                        desc = cur_sel.description or []
                        col_aliases = [
                            getattr(d, 'name', None) or (d[0] if isinstance(d, (list, tuple)) and d else None)
                            for d in desc
                        ]
                    except Exception:
                        col_aliases = []
                    # If still empty, default to the expected aliases from SELECT
                    if not col_aliases:
                        col_aliases = [
                            'uen', 'entity_name', 'primary_ssic_description', 'primary_ssic_code',
                            'incorporation_year', 'entity_status_de', 'ownership_type'
                        ]
                    first_batch = False
                with conn.cursor() as cur_up:
                    for r in rows:
                        m = row_to_map(r, col_aliases)
                        uen = m.get('uen')
                        entity_name = m.get('entity_name')
                        ssic_desc = m.get('primary_ssic_description')
                        ssic_code = m.get('primary_ssic_code')
                        inc_year = m.get('incorporation_year')
                        status_de = m.get('entity_status_de')
                        ownership_type = m.get('ownership_type')

                        # capture names for preview if SSIC-based selection
                        if source_mode == 'ssic' and len(names_preview_list) < 50:
                            nm = (entity_name or "").strip() if isinstance(entity_name, str) else ""
                            if nm:
                                names_preview_list.append(nm)

                        name = (entity_name or "").strip() if isinstance(entity_name, str) else None
                        if name == "":
                            name = None
                        desc_lower = (ssic_desc or "").strip().lower() if isinstance(ssic_desc, str) else ""
                        match_term = None
                        for t in lower_terms:
                            if desc_lower == t or (t in desc_lower):
                                match_term = t
                                break
                        industry_norm = (match_term or desc_lower) or None
                        industry_code = str(ssic_code) if ssic_code is not None else None
                        sg_registered = None
                        try:
                            sg_registered = (
                                (status_de or "").strip().lower() in {"live", "registered", "existing"}
                            ) if isinstance(status_de, str) else None
                        except Exception:
                            pass

                        # Locate existing company
                        company_id = None
                        if uen:
                            cur_up.execute("SELECT company_id FROM companies WHERE uen = %s LIMIT 1", (uen,))
                            row = cur_up.fetchone()
                            if row and isinstance(row, (list, tuple)) and len(row) >= 1:
                                company_id = row[0]
                        if company_id is None and name:
                            cur_up.execute("SELECT company_id FROM companies WHERE LOWER(name) = LOWER(%s) LIMIT 1", (name,))
                            row = cur_up.fetchone()
                            if row and isinstance(row, (list, tuple)) and len(row) >= 1:
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
                            "ownership_type": ownership_type,
                        }

                        if company_id is not None:
                            set_parts: List[str] = []
                            params: List[Any] = []
                            for k, v in fields.items():
                                if v is not None:
                                    set_parts.append(f"{k} = %s")
                                    params.append(v)
                            set_sql = ", ".join(set_parts) + ", last_seen = NOW()" if set_parts else "last_seen = NOW()"
                            cur_up.execute(
                                f"UPDATE companies SET {set_sql} WHERE company_id = %s",
                                params + [company_id],
                            )
                            affected += cur_up.rowcount or 0
                        else:
                            cols = [k for k, v in fields.items() if v is not None]
                            vals = [fields[k] for k in cols]
                            if not cols:
                                continue
                            cols_sql = ", ".join(cols)
                            ph = ",".join(["%s"] * len(vals))
                            cur_up.execute(
                                f"INSERT INTO companies ({cols_sql}) VALUES ({ph}) RETURNING company_id",
                                vals,
                            )
                            rw_new = cur_up.fetchone()
                            new_id = rw_new[0] if (rw_new and isinstance(rw_new, (list, tuple)) and len(rw_new) >= 1) else None
                            if new_id is not None:
                                cur_up.execute("UPDATE companies SET last_seen = NOW() WHERE company_id = %s", (new_id,))
                                affected += 1
                processed += len(rows)
            if source_mode == 'ssic' and names_preview_list:
                extra = f", ... (+{total_matches - len(names_preview_list)} more)" if total_matches > len(names_preview_list) else ""
                logger.info("staging_acra_companies matched %d rows by SSIC code; names: %s%s", total_matches, ", ".join(names_preview_list), extra)
            logger.info("Finished upserting by %s (%d rows processed, %d affected)", source_mode, processed, affected)
        return affected
    except Exception:
        logger.exception("staging upsert error")
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

    # Propagate tenant_id from context or input for multi-user runs
    try:
        ctx = payload.get("context") or data.get("context") or {}
        tid = ctx.get("tenant_id") or data.get("tenant_id")
        if tid is not None:
            state["tenant_id"] = tid
    except Exception:
        pass

    # Best-effort: if we have a tenant_id (or can infer one), pre-load the last saved ICP rule
    # and reuse it to prime the chat state so users don't need to re-enter ICP each session.
    try:
        tenant_id = state.get("tenant_id")
        if tenant_id is None:
            # Fall back to DSN→odoo_connections mapping like other helpers do
            try:
                from app.odoo_connection_info import get_odoo_connection_info
                info = asyncio.run(get_odoo_connection_info(email=None, claim_tid=None))
                tenant_id = info.get("tenant_id") if isinstance(info, dict) else None
                if tenant_id is not None:
                    state["tenant_id"] = tenant_id
            except Exception:
                tenant_id = None
        if tenant_id is not None:
            from src.database import get_conn as _get_conn
            with _get_conn() as _c, _c.cursor() as _cur:
                _cur.execute(
                    "SELECT payload FROM icp_rules WHERE tenant_id=%s ORDER BY created_at DESC LIMIT 1",
                    (int(tenant_id),),
                )
                row = _cur.fetchone()
                payload_rule = row[0] if row and row[0] is not None else None
            if isinstance(payload_rule, dict) and payload_rule:
                # Map rule payload into chat state icp fields and an icp_profile for agent discovery
                icp_ic = {}
                prof = {}
                # Lists
                for k in ("industries", "integrations", "buyer_titles", "triggers", "geos"):
                    v = payload_rule.get(k)
                    if isinstance(v, list) and any(isinstance(x, str) and x.strip() for x in v):
                        if k in ("integrations", "buyer_titles", "triggers"):
                            prof[k] = v
                        else:
                            icp_ic[k] = v
                # Ranges
                er = payload_rule.get("employee_range") or {}
                if isinstance(er, dict):
                    if er.get("min") is not None:
                        icp_ic["employees_min"] = er.get("min")
                    if er.get("max") is not None:
                        icp_ic["employees_max"] = er.get("max")
                yr = payload_rule.get("incorporation_year") or {}
                if isinstance(yr, dict):
                    if yr.get("min") is not None:
                        icp_ic["year_min"] = yr.get("min")
                    if yr.get("max") is not None:
                        icp_ic["year_max"] = yr.get("max")
                # Other direct fields (optional)
                for k in ("revenue_bucket",):
                    if k in payload_rule:
                        icp_ic[k] = payload_rule.get(k)
                # Persist into state if not already set by this input
                if icp_ic and not state.get("icp"):
                    state["icp"] = icp_ic
                if prof:
                    state["icp_profile"] = prof
    except Exception:
        # Non-fatal; proceed without priming
        pass

    # optional “companies”/“candidates” passthrough for your graph
    if "candidates" in data:
        state["candidates"] = data["candidates"]
    elif "companies" in data:
        state["candidates"] = data["companies"]

    # Feature 18: Upsert up to 10 synchronously and kick off enrichment immediately; enqueue remainder for nightly
    try:
        inds = _collect_industry_terms(state.get("messages"))
        if inds and STAGING_UPSERT_MODE != "off":
            # Gate staging upsert/enrichment while ICP Finder intake is active to prevent early runs
            try:
                from src.settings import ENABLE_ICP_INTAKE as _FINDER_ON  # type: ignore
            except Exception:
                _FINDER_ON = False  # type: ignore
            if _FINDER_ON:
                # Only proceed if the user explicitly asked to run enrichment
                txt = " ".join([(m.content or "") for m in norm_msgs if isinstance(m, HumanMessage)])
                if not re.search(r"\brun enrichment\b", txt, flags=re.IGNORECASE):
                    logger.info("Deferring staging upsert/enrich while ICP Finder is active; inds=%s", inds)
                    return state
            head = max(0, int(UPSERT_SYNC_LIMIT))
            if head > 0:
                try:
                    # Reuse helper from app.main to perform head upsert and enrichment
                    from app.main import upsert_by_industries_head, _trigger_enrichment_async
                    ids = upsert_by_industries_head(inds, limit=head)
                    if ids:
                        logger.info(
                            "Upserted+enriching(head=%d) %d companies for industries=%s",
                            head, len(ids), inds,
                        )
                        state["sync_head_company_ids"] = ids
                        _trigger_enrichment_async(ids)
                except Exception as e:
                    logger.info("sync head upsert/enrich skipped: %s", e)
            # Enqueue remainder for nightly processing (best-effort)
            try:
                from src.icp import _find_ssic_codes_by_terms as _resolve_ssic
                resolved = _resolve_ssic(inds) if inds else []
                if not resolved:
                    logger.info(
                        "Skip enqueue nightly: no SSIC codes resolved for terms=%s",
                        inds,
                    )
                else:
                    from src.jobs import enqueue_staging_upsert
                    # Resolve tenant best-effort; allow None
                    tid = None
                    try:
                        from app.odoo_connection_info import get_odoo_connection_info
                        info = asyncio.run(
                            get_odoo_connection_info(email=None, claim_tid=None)
                        )
                        tid = info.get("tenant_id") if isinstance(info, dict) else None
                    except Exception:
                        tid = None
                    enqueue_staging_upsert(tid, inds)
            except Exception as _qe:
                logger.info("enqueue nightly staging_upsert failed: %s", _qe)
    except Exception as _e:
        logger.warning("input-normalization staging handling failed: %s", _e)

    return state


def _extract_log_context(config: Dict[str, Any] | None) -> Dict[str, Any]:
    ctx: Dict[str, Any] = {}
    if not isinstance(config, dict):
        return ctx
    for key in ("tenant_id", "session_id", "thread_id", "job_id", "graph_id", "request_id"):
        val = config.get(key)
        if isinstance(val, (str, int)):
            ctx[key] = val
    configurable = config.get("configurable")
    if isinstance(configurable, dict):
        for key, value in configurable.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                ctx[f"config_{key}"] = value
    metadata = config.get("metadata")
    if isinstance(metadata, dict):
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                ctx[f"meta_{key}"] = value
    return ctx


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
    compiled = outer.compile()
    handler = LangGraphTroubleshootHandler(context=_extract_log_context(config))
    return compiled.with_config(callbacks=[handler])
