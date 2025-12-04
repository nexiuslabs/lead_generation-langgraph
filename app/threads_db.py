from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from src.database import get_conn


def _domain_from_value(url: str) -> str:
    """Extract apex domain from URL or host (see app.main for rationale)."""
    try:
        from urllib.parse import urlparse as _parse
        u = url if url.startswith("http") else ("https://" + url)
        p = _parse(u)
        host = (p.netloc or p.path or "").lower()
    except Exception:
        host = (url or "").strip().lower()
    if not host:
        return ""
    if "@" in host:
        host = host.split("@", 1)[-1]
    for sep in ("/", "?", "#"):
        if sep in host:
            host = host.split(sep, 1)[0]
    if host.startswith("www."):
        host = host[4:]
    if ":" in host:
        host = host.split(":", 1)[0]
    if all(ch.isdigit() or ch == "." for ch in host) or ":" in host:
        return host
    labels = [l for l in host.split(".") if l]
    if len(labels) <= 2:
        return host
    multi = {
        "co.uk","org.uk","gov.uk","ac.uk","sch.uk","ltd.uk","plc.uk",
        "com.sg","net.sg","org.sg","gov.sg","edu.sg",
        "com.au","net.au","org.au","gov.au","edu.au",
        "co.nz","org.nz","govt.nz","ac.nz",
        "co.jp","ne.jp","or.jp","ac.jp","go.jp",
        "com.my","com.ph","com.id","co.id","or.id","ac.id",
        "com.hk","org.hk","edu.hk","gov.hk","idv.hk",
        "com.br","net.br","org.br","gov.br","edu.br",
        "co.kr","ne.kr","or.kr","go.kr","ac.kr",
        "com.cn","net.cn","org.cn","gov.cn","edu.cn",
    }
    sfx2 = ".".join(labels[-2:])
    if sfx2 in multi and len(labels) >= 3:
        return ".".join(labels[-3:])
    return sfx2


def _first_url_from_messages(messages: List[Dict[str, str]] | None, text: str | None) -> Optional[str]:
    import re as _re

    url_re = _re.compile(r"https?://[^\s)]+|\b([a-z0-9-]+\.)+[a-z]{2,}\b", _re.IGNORECASE)
    scan: List[str] = []
    if isinstance(text, str) and text.strip():
        scan.append(text)
    for m in (messages or [])[:8]:
        try:
            c = (m.get("content") or "").strip()
            if c:
                scan.append(c)
        except Exception:
            continue
    for chunk in scan:
        m = url_re.search(chunk or "")
        if not m:
            continue
        raw = m.group(0)
        raw = raw.rstrip(".,);]")
        return raw
    return None


def _icp_fp(icp_payload: dict) -> str:
    try:
        import json as _json, hashlib

        blob = _json.dumps(
            {k: icp_payload.get(k) for k in (
                "industries",
                "buyer_titles",
                "company_sizes",
                "size_bands",
                "geos",
                "signals",
                "triggers",
                "integrations",
                "seed_urls",
                "summary",
            ) if icp_payload.get(k) is not None},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()
    except Exception:
        return ""


def context_key_from_payload(payload: Dict[str, Any], tenant_id: Optional[int]) -> str:
    messages = payload.get("messages") or []
    input_text = str(payload.get("input") or "")
    url = _first_url_from_messages(messages, input_text)
    dom = _domain_from_value(url or "") if url else ""
    if dom:
        return f"domain:{dom}"
    icp = payload.get("icp_payload") or {}
    fp = _icp_fp(icp) if isinstance(icp, dict) else ""
    try:
        from src.settings import ICP_RULE_NAME as _RN

        rn = _RN
    except Exception:
        rn = None
    rule = rn or (icp.get("rule_name") if isinstance(icp, dict) else None) or "default"
    return f"icp:{rule}#{fp or 'none'}"


def _row_to_dict(row: tuple, cols: List[str]) -> dict:
    return {cols[i]: row[i] for i in range(len(cols))}


def create_thread(tenant_id: Optional[int], user_id: Optional[str], agent: str, context_key: str, label: Optional[str] = None) -> str:
    tid = str(uuid.uuid4())
    with get_conn() as conn, conn.cursor() as cur:
        if tenant_id is not None:
            cur.execute("SELECT set_config('request.tenant_id', %s, true)", (str(int(tenant_id)),))
        cur.execute(
            """
            INSERT INTO threads (id, tenant_id, user_id, agent, context_key, label, status)
            VALUES (%s, %s, %s, %s, %s, %s, 'open')
            """,
            (tid, tenant_id, user_id, agent, context_key, label),
        )
    return tid


def lock_prior_open(tenant_id: Optional[int], user_id: Optional[str], agent: str, context_key: str, exclude_id: str) -> int:
    with get_conn() as conn, conn.cursor() as cur:
        if tenant_id is not None:
            cur.execute("SELECT set_config('request.tenant_id', %s, true)", (str(int(tenant_id)),))
        cur.execute(
            """
            UPDATE threads
               SET status = 'locked', locked_at = now(), reason = 'new_thread_created'
             WHERE status = 'open'
               AND agent = %s
               AND context_key = %s
               AND id <> %s
               AND (%s IS NULL OR tenant_id = %s)
               AND (%s IS NULL OR user_id = %s)
            """,
            (agent, context_key, exclude_id, tenant_id, tenant_id, user_id, user_id),
        )
        return cur.rowcount or 0


def auto_archive_stale_locked(tenant_id: Optional[int], stale_days: int) -> int:
    with get_conn() as conn, conn.cursor() as cur:
        if tenant_id is not None:
            cur.execute("SELECT set_config('request.tenant_id', %s, true)", (str(int(tenant_id)),))
        cur.execute(
            """
            UPDATE threads
               SET status = 'archived', archived_at = now(), reason = COALESCE(reason, 'auto_archive_stale')
             WHERE status = 'locked'
               AND (%s IS NULL OR tenant_id = %s)
               AND last_updated_at < now() - make_interval(days => %s)
            """,
            (tenant_id, tenant_id, int(stale_days)),
        )
        return cur.rowcount or 0


def get_thread(thread_id: str, tenant_id: Optional[int]) -> Optional[dict]:
    with get_conn() as conn, conn.cursor() as cur:
        if tenant_id is not None:
            cur.execute("SELECT set_config('request.tenant_id', %s, true)", (str(int(tenant_id)),))
        cur.execute(
            """
            SELECT id, tenant_id, user_id, agent, context_key, label, status, locked_at, archived_at, reason, last_updated_at, created_at
              FROM threads
             WHERE id = %s
            """,
            (thread_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        cols = [
            "id",
            "tenant_id",
            "user_id",
            "agent",
            "context_key",
            "label",
            "status",
            "locked_at",
            "archived_at",
            "reason",
            "last_updated_at",
            "created_at",
        ]
        return _row_to_dict(row, cols)


def resume_eligible(tenant_id: Optional[int], user_id: Optional[str], agent: str, context_key: str, window_days: int) -> List[dict]:
    with get_conn() as conn, conn.cursor() as cur:
        if tenant_id is not None:
            cur.execute("SELECT set_config('request.tenant_id', %s, true)", (str(int(tenant_id)),))
        cur.execute(
            """
            SELECT id, label, last_updated_at, created_at, status
              FROM threads
             WHERE status = 'open'
               AND agent = %s
               AND context_key = %s
               AND (%s IS NULL OR tenant_id = %s)
               AND (%s IS NULL OR user_id = %s)
               AND last_updated_at >= now() - make_interval(days => %s)
             ORDER BY last_updated_at DESC
             LIMIT 3
            """,
            (agent, context_key, tenant_id, tenant_id, user_id, user_id, int(window_days)),
        )
        rows = cur.fetchall() or []
        return [
            {
                "id": r[0],
                "label": r[1],
                "last_updated_at": r[2],
                "created_at": r[3],
                "status": r[4],
            }
            for r in rows
        ]


def update_last_updated(thread_id: str, tenant_id: Optional[int]) -> None:
    with get_conn() as conn, conn.cursor() as cur:
        if tenant_id is not None:
            cur.execute("SELECT set_config('request.tenant_id', %s, true)", (str(int(tenant_id)),))
        cur.execute("UPDATE threads SET last_updated_at = now() WHERE id = %s", (thread_id,))


def list_threads(tenant_id: Optional[int], user_id: Optional[str], show_archived: bool = False) -> List[dict]:
    with get_conn() as conn, conn.cursor() as cur:
        if tenant_id is not None:
            cur.execute("SELECT set_config('request.tenant_id', %s, true)", (str(int(tenant_id)),))
        where_arch = "status IN ('open','locked','archived')" if show_archived else "status IN ('open','locked')"
        cur.execute(
            f"""
            SELECT id, agent, context_key, label, status, locked_at, archived_at, reason, last_updated_at, created_at
              FROM threads
             WHERE {where_arch}
               AND (%s IS NULL OR tenant_id = %s)
               AND (%s IS NULL OR user_id = %s)
             ORDER BY last_updated_at DESC
            """,
            (tenant_id, tenant_id, user_id, user_id),
        )
        rows = cur.fetchall() or []
        cols = [
            "id",
            "agent",
            "context_key",
            "label",
            "status",
            "locked_at",
            "archived_at",
            "reason",
            "last_updated_at",
            "created_at",
        ]
        return [_row_to_dict(r, cols) for r in rows]


def archive_thread(thread_id: str, tenant_id: Optional[int], reason: Optional[str] = None) -> int:
    """Soft-delete a thread by marking it archived.

    Returns number of affected rows (0 if not found or blocked by RLS).
    """
    with get_conn() as conn, conn.cursor() as cur:
        if tenant_id is not None:
            cur.execute("SELECT set_config('request.tenant_id', %s, true)", (str(int(tenant_id)),))
        cur.execute(
            """
            UPDATE threads
               SET status = 'archived', archived_at = now(), last_updated_at = now(),
                   reason = COALESCE(%s, reason)
             WHERE id = %s
            """,
            (reason, thread_id),
        )
        return cur.rowcount or 0


def hard_delete_thread(thread_id: str, tenant_id: Optional[int]) -> int:
    """Hard-delete a thread row. Subject to RLS.

    Returns number of affected rows (0 if not found or blocked by RLS).
    """
    with get_conn() as conn, conn.cursor() as cur:
        if tenant_id is not None:
            cur.execute("SELECT set_config('request.tenant_id', %s, true)", (str(int(tenant_id)),))
        cur.execute("DELETE FROM threads WHERE id = %s", (thread_id,))
        return cur.rowcount or 0
