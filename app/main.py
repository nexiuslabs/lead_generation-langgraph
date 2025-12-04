# app/main.py
from fastapi import FastAPI, Request, Response, Depends, BackgroundTasks, HTTPException, Path, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.requests import ClientDisconnect
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from app.onboarding import handle_first_login, get_onboarding_status
from app.odoo_connection_info import get_odoo_connection_info
from src.database import get_pg_pool, get_conn
from app.auth import require_auth, require_identity, require_optional_identity
from app.middleware_request_id import CorrelationMiddleware
from app.odoo_store import OdooStore
from src.settings import OPENAI_API_KEY
from src.troubleshoot_log import log_json
from app.langgraph_logging import LangGraphTroubleshootHandler
from my_agent.agent import build_orchestrator_graph
from my_agent.utils.state import OrchestrationState
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path as PathlibPath
import os
import shutil
import csv
from io import StringIO
import logging
import json
import re
from datetime import datetime, timezone
import time
import threading
import asyncio
import math
import uuid
from typing import Any, Dict, List, Optional
import re
from datetime import datetime, timezone, timedelta

fmt = logging.Formatter("[%(levelname)s] %(asctime)s %(name)s :: %(message)s", "%H:%M:%S")

APP_ENV = (
    os.getenv("ENVIRONMENT")
    or os.getenv("PY_ENV")
    or os.getenv("NODE_ENV")
    or "dev"
).strip().lower()


def _configure_troubleshoot_file_logging() -> None:
    """Attach a rotating file handler for API/background troubleshooting logs."""

    log_dir = os.getenv("TROUBLESHOOT_API_LOG_DIR")
    if not log_dir and APP_ENV in {"dev", "development", "local", "localhost"}:
        log_dir = ".log_api"
    if not log_dir:
        return
    try:
        path = PathlibPath(log_dir).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        file_path = path / "api.log"
        root = logging.getLogger()
        # Avoid duplicate handlers when module re-imports (uvicorn reload)
        for handler in root.handlers:
            if isinstance(handler, TimedRotatingFileHandler) and getattr(handler, "baseFilename", None) == str(file_path):
                return
        handler = TimedRotatingFileHandler(
            file_path,
            when="midnight",
            interval=1,
            backupCount=14,
            encoding="utf-8",
            utc=True,
        )
        handler.suffix = "%Y-%m-%d"
        handler.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}$")  # type: ignore[attr-defined]
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S"))
        handler.setLevel(logging.INFO)
        root.addHandler(handler)
        if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, TimedRotatingFileHandler) for h in root.handlers):
            stream = logging.StreamHandler()
            stream.setFormatter(fmt)
            root.addHandler(stream)
        if root.level == logging.NOTSET:
            root.setLevel(logging.INFO)
        # Surface path for other modules/helpers
        os.environ.setdefault("TROUBLESHOOT_API_LOG_DIR", str(path))
    except Exception as exc:  # pragma: no cover - best effort
        logging.getLogger("startup").warning("Failed to configure troubleshoot file logging: %s", exc)


_configure_troubleshoot_file_logging()

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

# Attach correlation middleware so request_id/trace_id propagate across logs
app.add_middleware(CorrelationMiddleware)

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

# Troubleshooting logs ingestion router
try:
    from app.logs_routes import router as logs_router
    app.include_router(logs_router)
    logging.getLogger("startup").info("/v1/logs endpoint enabled")
except Exception as _e:
    logging.getLogger("startup").warning("Logs router not mounted: %s", _e)


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception):
    """Capture unhandled exceptions and emit structured troubleshoot logs."""

    payload = {
        "timestamp": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
        "level": "error",
        "service": "api",
        "environment": APP_ENV,
        "release": os.getenv("RELEASE", ""),
        "message": f"Unhandled exception on {getattr(request, 'method', '?')} {getattr(getattr(request, 'url', None), 'path', '')}",
        "trace_id": getattr(request.state, "trace_id", None),
        "request_id": getattr(request.state, "request_id", None),
        "error": {
            "type": type(exc).__name__,
            "message": str(exc),
        },
    }
    try:
        logging.getLogger("troubleshoot").error(json.dumps(payload, ensure_ascii=False))
    except Exception:
        logging.getLogger("api").error("Failed to emit troubleshoot error log", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "internal_error"})

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

# Mount DB-backed threads API
try:
    from app.threads_routes import router as threads_router

    app.include_router(threads_router)
    logger.info("/threads routes enabled")
except Exception as _e:
    logger.warning("Threads routes not mounted: %s", _e)

# Background worker is managed exclusively by scripts/run_bg_worker.py. We do not
# auto-start any worker in the API process to avoid lifespan conflicts and to
# keep concerns separated.

# ---------------------------------------------------------------
# Orchestrator API
# ---------------------------------------------------------------

# In-memory thread registry to enforce simple policy hooks when embedded server
# is not mounted. This provides minimal behavior: single active thread per
# context, auto-resume when exactly one match exists, and 409 on locked.
_THREADS: dict[str, dict] = {}
_PENDING_DISAMBIG: dict[str, list[dict]] = {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _first_url_from_messages(messages: List[Dict[str, str]] | None, text: str | None) -> Optional[str]:
    url_re = re.compile(r"https?://[^\s)]+|\b([a-z0-9-]+\.)+[a-z]{2,}\b", re.IGNORECASE)
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
        # strip trailing punctuation
        raw = raw.rstrip(".,);]")
        return raw
    return None


def _domain_from_value(url: str) -> str:
    """Extract apex domain from URL or host.

    Heuristics (no external deps):
    - strip scheme, path, port, and leading www.
    - collapse subdomains to registrable (apex) domain.
    - handle common multi-part public suffixes (e.g., co.uk, com.sg, com.au).
    """
    try:
        from urllib.parse import urlparse as _parse
        u = url if url.startswith("http") else ("https://" + url)
        p = _parse(u)
        host = (p.netloc or p.path or "").lower()
    except Exception:
        host = (url or "").strip().lower()
    if not host:
        return ""
    # Trim auth/path/port fragments if present
    if "@" in host:
        host = host.split("@", 1)[-1]
    for sep in ("/", "?", "#"):
        if sep in host:
            host = host.split(sep, 1)[0]
    if host.startswith("www."):
        host = host[4:]
    if ":" in host:
        host = host.split(":", 1)[0]
    # If looks like IP, return as-is
    if all(ch.isdigit() or ch == "." for ch in host) or ":" in host:
        return host
    labels = [l for l in host.split(".") if l]
    if len(labels) <= 2:
        return host
    # Common multi-part public suffixes (non-exhaustive, covers SG/UK/AU/JP/HK/BR/NZ/ID/KR/CN)
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


def _context_key(payload: Dict[str, Any], tenant_id: Optional[int]) -> str:
    # Prefer domain context
    url = _first_url_from_messages(
        _normalize_message_payload(payload.get("messages")),
        str(payload.get("input") or ""),
    )
    dom = _domain_from_value(url or "") if url else ""
    if dom:
        return f"domain:{dom}"
    # Else hash ICP payload for icp:* key
    icp = payload.get("icp_payload") or {}
    fp = _icp_fp(icp) if isinstance(icp, dict) else ""
    rule = None
    try:
        from src.settings import ICP_RULE_NAME as _RN

        rule = _RN
    except Exception:
        rule = None
    rn = rule or icp.get("rule_name") or "default"
    return f"icp:{rn}#{fp or 'none'}"


def _thread_find_candidates(tenant_id: Optional[int], user_id: Optional[str], agent: str, context_key: str) -> list[dict]:
    out: list[dict] = []
    for tid, meta in _THREADS.items():
        try:
            if meta.get("agent") != agent:
                continue
            if tenant_id is not None and meta.get("tenant_id") != tenant_id:
                continue
            if user_id is not None and meta.get("user_id") != user_id:
                continue
            if meta.get("context_key") != context_key:
                continue
            out.append({"id": tid, **meta})
        except Exception:
            continue
    return out


def _thread_store_path() -> str:
    try:
        base = os.getenv("LANGGRAPH_CHECKPOINT_DIR", ".langgraph_api").rstrip("/")
    except Exception:
        base = ".langgraph_api"
    try:
        os.makedirs(base, exist_ok=True)
    except Exception:
        pass
    return os.path.join(base, "threads.json")


def _load_threads() -> None:
    path = _thread_store_path()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            _THREADS.update(data)
    except Exception:
        pass


def _save_threads() -> None:
    path = _thread_store_path()
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_THREADS, f)
    except Exception:
        pass


def _normalize_message_payload(messages: Any) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    if not isinstance(messages, list):
        return normalized
    for item in messages:
        try:
            role = str(item.get("role") or "user")
            content = str(item.get("content") or "")
            if content:
                normalized.append({"role": role, "content": content})
        except Exception:
            continue
    return normalized


def _label_from_payload(ctx_key: str, payload: Dict[str, Any]) -> str:
    # 1) explicit label in payload
    try:
        raw = payload.get("label")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    except Exception:
        pass
    # 2) domain label
    try:
        if isinstance(ctx_key, str) and ctx_key.startswith("domain:"):
            return ctx_key.split(":", 1)[1]
    except Exception:
        pass
    # 3) input snippet
    try:
        txt = payload.get("input")
        if isinstance(txt, str) and txt.strip():
            return (" ".join(txt.strip().split()))[:60]
    except Exception:
        pass
    # 4) last user message snippet
    try:
        msgs = payload.get("messages")
        if isinstance(msgs, list):
            for m in reversed(msgs):
                if str(m.get("role") or "").lower() == "user":
                    c = str(m.get("content") or "").strip()
                    if c:
                        return (" ".join(c.split()))[:60]
    except Exception:
        pass
    # 5) fallback
    return "ICP session"


@app.post("/api/orchestrations")
async def start_orchestration(
    request: Request,
    payload: Dict[str, Any] = Body(default_factory=dict),
    identity: Dict[str, Any] = Depends(require_auth),
):
    tenant_id = getattr(request.state, "tenant_id", None)

    # If DB-backed threads are enabled, resolve/create the thread in Postgres
    use_db_threads = False
    try:
        from src.settings import USE_DB_THREADS as _USE_DB_THREADS

        use_db_threads = bool(_USE_DB_THREADS)
    except Exception:
        use_db_threads = False

    if use_db_threads:
        # Branch: DB-backed thread lifecycle
        from app.threads_db import (
            context_key_from_payload as _ctx_key_from_payload,
            resume_eligible as _resume_eligible,
            create_thread as _create_thread,
            lock_prior_open as _lock_prior_open,
            get_thread as _get_thread,
            update_last_updated as _update_last_updated,
        )
        try:
            from src.settings import THREAD_RESUME_WINDOW_DAYS as _WIN
        except Exception:
            _WIN = 7

        ctx_key = _ctx_key_from_payload(payload, tenant_id)
        provided_tid = str(payload.get("thread_id") or "").strip()

        # If a thread was explicitly provided, ensure it's open
        if provided_tid:
            row = _get_thread(provided_tid, tenant_id)
            if not row:
                raise HTTPException(status_code=404, detail="thread_not_found")
            if (row.get("status") or "open") != "open":
                raise HTTPException(status_code=409, detail={"error": "thread_locked", "hint": "create_new"})
            thread_id = provided_tid
        else:
            # Auto-resume when exactly one eligible open thread exists
            cands = _resume_eligible(tenant_id, identity.get("sub"), "icp_finder", ctx_key, _WIN)
            if len(cands) == 1:
                thread_id = cands[0]["id"]
            elif len(cands) > 1:
                # Disambiguation flow (mirror legacy in-memory behavior)
                top = cands[:3]
                key = f"{tenant_id}|{identity.get('sub')}|{ctx_key}"
                _PENDING_DISAMBIG[key] = top

                def _label(c: dict) -> str:
                    from app.threads_db import get_thread as __get_thread

                    r = __get_thread(c["id"], tenant_id)
                    if r and r.get("label"):
                        return str(r.get("label"))
                    ck = (r or {}).get("context_key") or ctx_key
                    return ck.split(":", 1)[1] if str(ck).startswith("domain:") else (str(ck) or "ICP session")

                lines = []
                for i, c in enumerate(top, start=1):
                    label = _label(c)
                    ts = c.get("last_updated_at") or c.get("created_at") or "recently"
                    lines.append(f"{i}) {label} — last updated {ts}")
                msg = "You have multiple ongoing sessions for this ICP. Which should we continue?\n" + "\n".join(lines) + "\nReply with 1, 2, or 3."
                placeholder_id = str(uuid.uuid4())
                # Return a locked placeholder; clients can show status and call /threads to choose
                return {
                    "thread_id": placeholder_id,
                    "status": {"phase": "disambiguation", "message": msg},
                    "status_history": [{"phase": "disambiguation", "message": msg, "timestamp": _now_iso()}],
                    "output": msg,
                    "disambiguation_required": True,
                    "candidates": [
                        {
                            "id": c["id"],
                            "label": _label(c),
                            "updated_at": c.get("last_updated_at") or c.get("created_at"),
                        }
                        for c in top
                    ],
                }
            else:
                # Create fresh thread and lock any unexpected prior opens
                label = _label_from_payload(ctx_key, payload)
                thread_id = _create_thread(tenant_id, identity.get("sub"), "icp_finder", ctx_key, label)
                try:
                    _lock_prior_open(tenant_id, identity.get("sub"), "icp_finder", ctx_key, thread_id)
                except Exception:
                    pass

        # Build state and run orchestrator
        _msgs = _normalize_message_payload(payload.get("messages"))
        _input_raw = str(payload.get("input") or "")
        if not _input_raw and _msgs:
            for _m in reversed(_msgs):
                if (_m.get("role") or "").lower() == "user":
                    _input_raw = str(_m.get("content") or "")
                    break
        state: OrchestrationState = {
            "messages": _msgs,
            "input": _input_raw,
            "input_role": str(payload.get("role") or "user"),
            "entry_context": {
                "thread_id": thread_id,
                "tenant_id": tenant_id,
                "user_id": identity.get("sub"),
                "run_mode": str(payload.get("run_mode") or "chat_top10"),
            },
            "icp_payload": payload.get("icp_payload") or {},
        }
        context = {"thread_id": thread_id, "tenant_id": tenant_id, "source": "api", "user_id": identity.get("sub")}
        log_json("orchestrator", "info", "run_start", context)
        t0 = time.perf_counter()
        result = await ORCHESTRATOR.ainvoke(
            state,
            config={"configurable": {"thread_id": thread_id}, "callbacks": _orchestrator_callbacks(context)},
        )
        duration_ms = int((time.perf_counter() - t0) * 1000)
        log_json("orchestrator", "info", "run_complete", {"thread_id": thread_id, "tenant_id": tenant_id, "output_status": (result or {}).get("status"), "duration_ms": duration_ms})
        try:
            _update_last_updated(thread_id, tenant_id)
        except Exception:
            pass
        status = (result or {}).get("status") or {}
        status_history = (result or {}).get("status_history") or []
        return {"thread_id": thread_id, "status": status, "status_history": status_history, "output": status.get("message")}

    # Legacy path: in-memory thread registry
    # Compute context key to enforce single-active-thread policy per context
    ctx_key = _context_key(payload, tenant_id)

    # If no thread_id provided, try auto-resume an existing open thread for this context
    provided_tid = str(payload.get("thread_id") or "").strip()
    auto_resume_tid: Optional[str] = None
    # Check if user responded with a numeric selection for pending disambiguation
    user_input = str(payload.get("input") or "").strip()
    key = f"{tenant_id}|{identity.get('sub')}|{ctx_key}"
    if not provided_tid and user_input:
        try:
            import re as _re

            m = _re.search(r"\b(\d{1,2})\b", user_input)
            if m and key in _PENDING_DISAMBIG:
                idx = int(m.group(1)) - 1
                cand_list = _PENDING_DISAMBIG.get(key) or []
                if 0 <= idx < len(cand_list):
                    sel = cand_list[idx]
                    sel_id = str(sel.get("id") or "").strip()
                    if sel_id and sel_id in _THREADS:
                        provided_tid = sel_id
                        _PENDING_DISAMBIG.pop(key, None)
        except Exception:
            pass
    if not provided_tid:
        candidates = _thread_find_candidates(tenant_id, identity.get("sub"), "icp_finder", ctx_key)
        open_candidates = [c for c in candidates if (c.get("status") or "open") == "open"]
        # Enforce resume window
        try:
            from src.settings import THREAD_RESUME_WINDOW_DAYS as _WIN
        except Exception:
            _WIN = 7
        if open_candidates:
            eligible: list[dict] = []
            for c in open_candidates:
                try:
                    ts = c.get("updated_at") or c.get("created_at")
                    if not ts:
                        eligible.append(c)
                        continue
                    dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                    if dt >= datetime.now(timezone.utc) - timedelta(days=int(_WIN)):
                        eligible.append(c)
                except Exception:
                    eligible.append(c)
            open_candidates = eligible
        if len(open_candidates) == 1:
            auto_resume_tid = open_candidates[0]["id"]
        elif len(open_candidates) > 1 and not provided_tid:
            # Prepare disambiguation message; do not mutate existing threads
            open_candidates.sort(key=lambda c: (c.get("updated_at") or c.get("created_at") or ""), reverse=True)
            top = open_candidates[:3]
            _PENDING_DISAMBIG[key] = top
            def _label(meta: dict) -> str:
                ck = str(meta.get("context_key") or "")
                if ck.startswith("domain:"):
                    return ck.split(":", 1)[1]
                return ck or "ICP session"
            lines = []
            for i, c in enumerate(top, start=1):
                label = _label(c)
                ts = c.get("updated_at") or c.get("created_at") or "recently"
                lines.append(f"{i}) {label} — last updated {ts}")
            msg = "You have multiple ongoing sessions for this ICP. Which should we continue?\n" + "\n".join(lines) + "\nReply with 1, 2, or 3."
            # Create a placeholder locked thread id so the client can poll status safely
            placeholder_id = str(uuid.uuid4())
            _THREADS[placeholder_id] = {
                "tenant_id": tenant_id,
                "user_id": identity.get("sub"),
                "agent": "icp_finder",
                "context_key": ctx_key,
                "status": "locked",
                "locked_at": _now_iso(),
                "reason": "awaiting_disambiguation",
                "created_at": _now_iso(),
                "updated_at": _now_iso(),
            }
            _save_threads()
            return {
                "thread_id": placeholder_id,
                "status": {"phase": "disambiguation", "message": msg},
                "status_history": [{"phase": "disambiguation", "message": msg, "timestamp": _now_iso()}],
                "output": msg,
                "disambiguation_required": True,
                "candidates": [{"id": c["id"], "label": _label(c), "updated_at": c.get("updated_at") or c.get("created_at")} for c in top],
            }
    thread_id = provided_tid or auto_resume_tid or str(uuid.uuid4())

    # Reject requests to a locked thread (policy)
    meta = _THREADS.get(thread_id)
    if meta and meta.get("status") == "locked":
        raise HTTPException(status_code=409, detail={"error": "thread_locked", "hint": "create_new"})

    # If this is a new thread, register it and lock prior open threads for the same context
    if not meta:
        _THREADS[thread_id] = {
            "tenant_id": tenant_id,
            "user_id": identity.get("sub"),
            "agent": "icp_finder",
            "context_key": ctx_key,
            "status": "open",
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
        }
        try:
            from src.settings import SINGLE_THREAD_PER_CONTEXT as _STPC, AUTO_ARCHIVE_STALE_LOCKED as _AUTO_ARCH, THREAD_STALE_DAYS as _STALE

            if _STPC:
                for tid, meta2 in list(_THREADS.items()):
                    if tid == thread_id:
                        continue
                    if meta2.get("tenant_id") != tenant_id or meta2.get("user_id") != identity.get("sub"):
                        continue
                    if meta2.get("agent") != "icp_finder" or meta2.get("context_key") != ctx_key:
                        continue
                    # Lock prior open threads for the same context
                    if meta2.get("status") == "open":
                        meta2["status"] = "locked"
                        meta2["locked_at"] = _now_iso()
                        meta2["reason"] = "new_thread_same_context"
                    # option: auto-archive stale locked
                    if meta2.get("status") == "locked" and _AUTO_ARCH:
                        try:
                            ts = meta2.get("updated_at") or meta2.get("locked_at")
                            if ts:
                                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                                if dt < datetime.now(timezone.utc) - timedelta(days=int(_STALE)):
                                    meta2["status"] = "archived"
                                    meta2["archived_at"] = _now_iso()
                        except Exception:
                            pass
        except Exception:
            pass
        _save_threads()
    else:
        # Update last-used timestamp for existing threads
        try:
            meta["updated_at"] = _now_iso()
        except Exception:
            pass
        _save_threads()
    # Normalize incoming messages and derive a sensible fallback for `input` if FE omitted it
    _msgs = _normalize_message_payload(payload.get("messages"))
    _input_raw = str(payload.get("input") or "")
    if not _input_raw and _msgs:
        try:
            # Use last user message as input when present
            for _m in reversed(_msgs):
                if (_m.get("role") or "").lower() == "user":
                    _input_raw = str(_m.get("content") or "")
                    break
        except Exception:
            _input_raw = ""

    state: OrchestrationState = {
        "messages": _msgs,
        "input": _input_raw,
        "input_role": str(payload.get("role") or "user"),
        "entry_context": {
            "thread_id": thread_id,
            "tenant_id": tenant_id,
            "user_id": identity.get("sub"),
            "run_mode": str(payload.get("run_mode") or "chat_top10"),
        },
        "icp_payload": payload.get("icp_payload") or {},
    }
    context = {
        "thread_id": thread_id,
        "tenant_id": tenant_id,
        "source": "api",
        "user_id": identity.get("sub"),
    }
    log_json("orchestrator", "info", "run_start", context)
    t0 = time.perf_counter()
    result = await ORCHESTRATOR.ainvoke(
        state,
        config={
            "configurable": {"thread_id": thread_id},
            "callbacks": _orchestrator_callbacks(context),
        },
    )
    duration_ms = int((time.perf_counter() - t0) * 1000)
    log_json(
        "orchestrator",
        "info",
        "run_complete",
        {
            "thread_id": thread_id,
            "tenant_id": tenant_id,
            "output_status": (result or {}).get("status"),
            "duration_ms": duration_ms,
        },
    )
    status = (result or {}).get("status") or {}
    status_history = (result or {}).get("status_history") or []
    return {"thread_id": thread_id, "status": status, "status_history": status_history, "output": status.get("message")}


@app.get("/api/orchestrations/{thread_id}")
async def get_orchestration_status(
    thread_id: str,
    request: Request,
    identity: Dict[str, Any] = Depends(require_auth),
):
    """Return latest orchestrator state for a thread.

    Guard against missing state by seeding an idle snapshot so callers never see
    a 404 solely because the process restarted or a run hasn't started yet.
    """
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = ORCHESTRATOR.get_state(config)
    values = getattr(snapshot, "values", None) if snapshot else None

    if not values:
        # Seed a minimal idle state so the UI does not see a 404 for fresh or
        # restored threads. Best-effort: tolerate older LangGraph versions
        # missing update_state by catching AttributeError.
        idle = {
            "thread_id": thread_id,
            "status": {
                "phase": "idle",
                "message": "Thread created. No runs yet.",
                "updated_at": _now_iso(),
            },
            "status_history": [],
        }
        try:
            upd = getattr(ORCHESTRATOR, "update_state", None)
            if callable(upd):
                upd(config, idle)
            # Re-fetch to return normalized structure
            snapshot = ORCHESTRATOR.get_state(config)
            values = getattr(snapshot, "values", None) if snapshot else None
        except Exception:
            # Fall back to returning the idle structure directly
            values = idle

    status = values.get("status") or {}
    status_history = values.get("status_history") or []
    return {
        "thread_id": thread_id,
        "status": status,
        "output": status.get("message"),
        "status_history": status_history,
        "state": values,
    }

# ---------------------------------------------------------------
# Utilities: SendGrid email smoke test (no enrichment required)
# ---------------------------------------------------------------
@app.post("/email/test")
async def email_test(
    to: Optional[str] = Query(default=None, description="Recipient email"),
    subject: Optional[str] = Query(default="Test: Lead shortlist delivery", description="Email subject"),
    simple: bool = Query(default=True, description="Use simple payload (no DB/LLM)"),
    tenant_id: Optional[int] = Query(default=None, description="Tenant id (only used when simple=false)"),
    body: Optional[str] = Body(default=None, description="Optional HTML body for simple mode"),
):
    """Send a minimal test email via SendGrid to verify configuration.

    - simple=true (default): sends a static HTML payload; no DB or LLM involved.
    - simple=false: uses the agentic sender with render+LLM; requires tenant_id and DB data.
    """
    try:
        from src.settings import EMAIL_ENABLED, DEFAULT_NOTIFY_EMAIL, SENDGRID_API_KEY, SENDGRID_FROM_EMAIL
        if not EMAIL_ENABLED:
            raise HTTPException(status_code=400, detail="email disabled: ENABLE_EMAIL_RESULTS is false")
        if not SENDGRID_API_KEY or not SENDGRID_FROM_EMAIL:
            raise HTTPException(status_code=400, detail="sendgrid not configured: missing API key or from email")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"email config error: {e}")

    to_final = (to or "").strip() if isinstance(to, str) else None
    if not to_final:
        try:
            from src.settings import DEFAULT_NOTIFY_EMAIL as _DEF_TO
            if _DEF_TO and ("@" in str(_DEF_TO)):
                to_final = str(_DEF_TO)
        except Exception:
            pass
    if not to_final:
        raise HTTPException(status_code=400, detail="missing 'to' and DEFAULT_NOTIFY_EMAIL not set")

    if not simple:
        if tenant_id is None:
            raise HTTPException(status_code=400, detail="tenant_id is required when simple=false")
        try:
            from src.notifications.agentic_email import agentic_send_results
            res = await agentic_send_results(to_final, int(tenant_id))
            return {"ok": True, **(res or {})}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"agentic send failed: {e}")

    # Simple path: no DB, send minimal HTML
    html = body or (
        "<p>This is a SendGrid configuration test from Lead Generation backend.</p>"
        "<p>If you received this email, your SendGrid credentials are valid.</p>"
    )
    try:
        from src.notifications.sendgrid import send_leads_email
        res = await send_leads_email(to_final, subject or "Test", html)
        return {"ok": True, **(res or {})}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"sendgrid send failed: {e}")

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


def _require_tenant_id(request: Request) -> int:
    tenant_id = getattr(request.state, "tenant_id", None)
    if tenant_id is None:
        raise HTTPException(status_code=403, detail="tenant_id required")
    try:
        return int(tenant_id)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=400, detail="Invalid tenant_id") from exc


@app.get("/candidates/latest")
async def candidates_latest(
    request: Request,
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
    tenant_id = _require_tenant_id(request)
    filters: list[str] = []
    filter_params: list = []
    if industry and industry.strip():
        filters.append("LOWER(c.industry_norm) = LOWER(%s)")
        filter_params.append(industry.strip())
    where_sql = ""
    if filters:
        where_sql = " AND " + " AND ".join(filters)
    order_expr = "COALESCE(c.last_seen, '1970-01-01'::timestamptz)"
    with get_conn() as conn, conn.cursor() as cur:
        try:
            cur.execute("SELECT set_config('request.tenant_id', %s, true)", (str(tenant_id),))
        except Exception:
            pass
        cursor_clause = ""
        cursor_params: list = []
        if afterUpdatedAt is not None and afterId is not None:
            cursor_clause = f" AND ({order_expr}, c.company_id) < (%s, %s)"
            cursor_params = [afterUpdatedAt, afterId]
        sql = f"""
            SELECT c.company_id, c.name, c.industry_norm, c.website_domain, c.last_seen
            FROM companies c
            JOIN (
                SELECT DISTINCT company_id
                FROM icp_evidence
                WHERE tenant_id = %s
            ) ie ON ie.company_id = c.company_id
            WHERE 1=1{where_sql}
            {cursor_clause}
            ORDER BY {order_expr} DESC, c.company_id DESC
            LIMIT %s
        """
        params = [tenant_id, *filter_params, *cursor_params, limit]
        cur.execute(sql, params)
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
        cursor_last_seen = last["last_seen"]
        if cursor_last_seen is None:
            cursor_last_seen = datetime.fromtimestamp(0, timezone.utc).isoformat()
        next_cursor = {"afterUpdatedAt": cursor_last_seen, "afterId": last["company_id"]}
    return {"items": items, "nextCursor": next_cursor}


@app.get("/metrics")
async def metrics(request: Request, _: dict = Depends(require_auth)):
    """Light metrics for ops and dashboards with richer stats."""
    out = {
        "job_queue_depth": 0,
        "jobs_processed_total": 0,
        "lead_scores_total": 0,
        "rows_per_min": None,
        "p95_job_ms": None,
        "chat_ttfb_p95_ms": None,
    }
    tenant_id = _require_tenant_id(request)
    with get_conn() as conn, conn.cursor() as cur:
        try:
            cur.execute("SELECT set_config('request.tenant_id', %s, true)", (str(tenant_id),))
        except Exception:
            pass
        try:
            cur.execute("SELECT COUNT(*) FROM background_jobs WHERE tenant_id = %s AND status='queued'", (tenant_id,))
            out["job_queue_depth"] = int((cur.fetchone() or [0])[0] or 0)
        except Exception:
            pass
        try:
            cur.execute(
                "SELECT COALESCE(SUM(processed),0) FROM background_jobs WHERE tenant_id = %s AND job_type='staging_upsert' AND status='done'",
                (tenant_id,),
            )
            out["jobs_processed_total"] = int((cur.fetchone() or [0])[0] or 0)
        except Exception:
            pass
        try:
            cur.execute("SELECT COUNT(*) FROM lead_scores WHERE tenant_id = %s", (tenant_id,))
            out["lead_scores_total"] = int((cur.fetchone() or [0])[0] or 0)
        except Exception:
            pass
        # rows/min and p95 job duration from recent completed jobs
        try:
            cur.execute(
                """
                SELECT processed, EXTRACT(EPOCH FROM (ended_at - started_at)) AS secs
                FROM background_jobs
                WHERE tenant_id = %s AND job_type='staging_upsert' AND status='done' AND started_at IS NOT NULL AND ended_at IS NOT NULL
                ORDER BY job_id DESC
                LIMIT 20
                """,
                (tenant_id,),
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
                WHERE tenant_id = %s AND stage='chat' AND event='ttfb' AND ts > NOW() - INTERVAL '7 days' AND duration_ms IS NOT NULL
                ORDER BY duration_ms
                LIMIT 1000
                """,
                (tenant_id,),
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
async def whoami(request: Request, claims: dict = Depends(require_auth)):
    """Return identity info including the effective tenant_id.

    Shows tenant_id from the JWT claim when present, otherwise falls back to
    the value resolved by require_auth (e.g., X-Tenant-ID header).
    """
    effective_tid = claims.get("tenant_id")
    if effective_tid is None:
        # require_auth sets request.state.tenant_id from X-Tenant-ID when claim is missing
        effective_tid = getattr(request.state, "tenant_id", None)
    return {
        "sub": claims.get("sub"),
        "email": claims.get("email"),
        "tenant_id": effective_tid,
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

# --- Global error handler to emit structured troubleshoot logs ---
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, ClientDisconnect):
        return JSONResponse(status_code=499, content={"detail": "client_disconnect"})
    if isinstance(exc, HTTPException):
        if exc.status_code >= 500:
            log_json(
                "api",
                "error",
                "HTTPException",
                {
                    "status": exc.status_code,
                    "detail": exc.detail,
                    "path": request.url.path,
                    "method": request.method,
                    "request_id": getattr(request.state, "request_id", None),
                    "trace_id": getattr(request.state, "trace_id", None),
                },
            )
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    log_json(
        "api",
        "error",
        "Unhandled exception",
        {
            "error_type": type(exc).__name__,
            "detail": str(exc),
            "path": request.url.path,
            "method": request.method,
            "request_id": getattr(request.state, "request_id", None),
            "trace_id": getattr(request.state, "trace_id", None),
        },
    )
    return JSONResponse(status_code=500, content={"detail": "internal_server_error"})

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

    # Try to talk to the app DB, but fail fast and return a safe fallback when offline.
    try:
        pool = await get_pg_pool()
        try:
            _acq_timeout = float(os.getenv("PG_CONNECT_TIMEOUT_S", "3") or 3)
        except Exception:
            _acq_timeout = 3.0
        async with pool.acquire(timeout=_acq_timeout) as conn:
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
    except Exception as _db_exc:
        # DB offline/unresolvable host. Return a safe, non-failing fallback that the UI can handle.
        total_scored = 0
        last_ts = None
        last_run_id = None
        last_run_status = "unknown"
        last_run_started_at = None
        last_run_ended_at = None

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
# Build orchestrator graph once for API + chat entry
ORCHESTRATOR = build_orchestrator_graph()


def _maybe_mount_embedded_lg_server() -> None:
    """Optionally mount an embedded LangGraph server if available.

    This is best-effort only and safely no-ops when the library is absent. The
    existing /api/orchestrations route remains the primary entry in this app.
    """
    try:
        from src.settings import ENABLE_EMBEDDED_LG_SERVER as _ENABLE
    except Exception:
        _ENABLE = False
    if not _ENABLE:
        return
    try:
        # Attempt a few known mounting patterns; ignore failures
        try:
            # Newer server interface providing a FastAPI router
            from langgraph_api.fastapi import mount as _lg_mount  # type: ignore

            _lg_mount(app)  # mounts /threads and /runs/stream if configured
            logging.getLogger("startup").info("Embedded LangGraph server mounted via fastapi.mount")
            return
        except Exception:
            pass
        try:
            # Alternate pattern: module exposes a mount_app helper
            from langgraph_api.server import mount_app as _mount_app  # type: ignore

            _mount_app(app)
            logging.getLogger("startup").info("Embedded LangGraph server mounted via server.mount_app")
            return
        except Exception:
            pass
        logging.getLogger("startup").info("Embedded LangGraph server not available; skipping mount")
    except Exception as exc:
        logging.getLogger("startup").warning("Embedded LangGraph server mount failed: %s", exc)


_maybe_mount_embedded_lg_server()
_load_threads()


def _orchestrator_callbacks(context: Dict[str, Any] | None = None):
    ctx = context or {}
    return [LangGraphTroubleshootHandler(context=ctx)]


# --- Admin: backfill legacy .langgraph_api/threads.json into DB threads ---
@app.post("/admin/threads/backfill_legacy")
async def admin_backfill_legacy_threads(request: Request, claims: dict = Depends(require_auth)):
    roles = claims.get("roles", []) or []
    if "admin" not in roles:
        raise HTTPException(status_code=403, detail="admin role required")

    base = os.getenv("LANGGRAPH_CHECKPOINT_DIR", ".langgraph_api").rstrip("/")
    path = os.path.join(base, "threads.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"legacy threads file not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("invalid threads.json (expected object)")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"failed to read legacy file: {e}")

    from app.threads_db import get_thread as _get_thread
    created = 0
    skipped = 0
    exists = 0

    def _parse_ts(val: Any):
        if not val:
            return None
        try:
            s = str(val)
            if s.endswith("Z"):
                s = s.replace("Z", "+00:00")
            return datetime.fromisoformat(s)
        except Exception:
            return None

    with get_conn() as conn, conn.cursor() as cur:
        for tid, meta in data.items():
            try:
                if not isinstance(meta, dict):
                    skipped += 1
                    continue
                tenant_id_row = meta.get("tenant_id")
                # Skip if already present for this tenant
                row = _get_thread(tid, getattr(request.state, "tenant_id", None))
                if row:
                    exists += 1
                    continue
                tenant_id = tenant_id_row
                user_id = meta.get("user_id")
                agent = (meta.get("agent") or "icp_finder").strip()
                context_key = (meta.get("context_key") or "").strip()
                if not context_key:
                    skipped += 1
                    continue
                label = meta.get("label") or (context_key.split(":", 1)[1] if context_key.startswith("domain:") else "ICP session")
                status = (meta.get("status") or "open").strip()
                locked_at = _parse_ts(meta.get("locked_at"))
                archived_at = _parse_ts(meta.get("archived_at"))
                reason = meta.get("reason")
                last_updated_at = _parse_ts(meta.get("updated_at")) or _parse_ts(meta.get("last_updated_at")) or datetime.now(timezone.utc)
                created_at = _parse_ts(meta.get("created_at")) or datetime.now(timezone.utc)

                # Apply tenant GUC for RLS
                try:
                    if tenant_id is not None:
                        cur.execute("SELECT set_config('request.tenant_id', %s, true)", (str(int(tenant_id)),))
                except Exception:
                    pass

                cur.execute(
                    """
                    INSERT INTO threads (id, tenant_id, user_id, agent, context_key, label, status, locked_at, archived_at, reason, last_updated_at, created_at)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    (
                        tid,
                        tenant_id,
                        user_id,
                        agent,
                        context_key,
                        label,
                        status,
                        locked_at,
                        archived_at,
                        reason,
                        last_updated_at,
                        created_at,
                    ),
                )
                if cur.rowcount and int(cur.rowcount) > 0:
                    created += 1
                else:
                    exists += 1
            except Exception:
                skipped += 1
                continue

    return {"ok": True, "imported": created, "skipped": skipped, "exists": exists, "source": path}
