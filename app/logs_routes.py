from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Any, Dict
import time
import hashlib
import json
import logging
import os
from pathlib import Path
from datetime import datetime, timezone
import jwt
from jwt import PyJWTError


router = APIRouter(prefix="/v1/logs", tags=["logs"]) 

ALLOWED_LEVELS = {"error", "warn", "info", "debug"}
ENV = (os.getenv("ENVIRONMENT") or os.getenv("PY_ENV") or os.getenv("NODE_ENV") or "dev").lower()
HMAC_SECRET = os.getenv("LOG_INGEST_HMAC_SECRET")
DIAGNOSTIC_SECRET = os.getenv("LOG_DIAGNOSTIC_SECRET")

LOG_DIR = os.getenv("TROUBLESHOOT_API_LOG_DIR")
if not LOG_DIR and ENV in {"dev", "development", "local", "localhost"}:
    LOG_DIR = ".log_api"
LOG_DIR_PATH = Path(LOG_DIR).expanduser() if LOG_DIR else None

def _append_jsonl(line: str) -> None:
    if not LOG_DIR_PATH:
        return
    try:
        LOG_DIR_PATH.mkdir(parents=True, exist_ok=True)
        file_path = LOG_DIR_PATH / "ingest.jsonl"
        with file_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        # best effort only
        pass


class ErrorModel(BaseModel):
    type: Optional[str] = Field(default=None, max_length=64)
    message: Optional[str] = Field(default=None, max_length=256)
    stack: Optional[List[str]] = None


class HttpModel(BaseModel):
    method: Optional[str] = Field(default=None, max_length=8)
    route: Optional[str] = Field(default=None, max_length=128)
    host: Optional[str] = Field(default=None, max_length=128)
    status: Optional[int] = None
    duration_ms: Optional[int] = None


class LogEvent(BaseModel):
    timestamp: str
    level: str
    service: str
    environment: str
    release: str
    message: str
    trace_id: Optional[str] = None
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    component: Optional[str] = None
    error: Optional[ErrorModel] = None
    http: Optional[HttpModel] = None
    data: Optional[Dict[str, Any]] = None


# Simple per-IP leaky bucket
_buckets: Dict[str, List[float]] = {}  # ip -> [last_ts, tokens]


def _rate_limit(ip: str) -> bool:
    now = time.time()
    if ip not in _buckets:
        _buckets[ip] = [now, 20.0]
    ts, tokens = _buckets[ip]
    refill = (now - ts) * 5.0  # 5 ev/s avg
    tokens = min(20.0, tokens + refill)
    if tokens < 1.0:
        _buckets[ip] = [now, tokens]
        return False
    tokens -= 1.0
    _buckets[ip] = [now, tokens]
    return True


def _sanitize_data(d: Optional[dict]) -> Optional[dict]:
    if not isinstance(d, dict):
        return None
    allow = {"route", "route_template", "host", "method", "status", "duration_ms", "component", "pathname"}
    block_keys = ("authorization", "cookie", "token", "secret", "password", "set-cookie")
    out: Dict[str, Any] = {}
    for k, v in d.items():
        kl = str(k).lower()
        if any(b in kl for b in block_keys):
            continue
        if k in allow:
            if isinstance(v, str):
                vv = v.rsplit('?', 1)[0]
                out[k] = vv[:256]
            else:
                out[k] = v
    return out or None


def _decode_diagnostic_cookie(request: Request) -> Optional[Dict[str, Any]]:
    if not DIAGNOSTIC_SECRET:
        return None
    token = request.cookies.get("diag")
    if not token:
        return None
    try:
        payload = jwt.decode(token, DIAGNOSTIC_SECRET, algorithms=["HS256"])
        exp = payload.get("exp")
        if exp is not None:
            try:
                if datetime.now(timezone.utc).timestamp() > float(exp):
                    return None
            except Exception:
                return None
        return payload
    except PyJWTError:
        return None


@router.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@router.post("")
async def ingest(request: Request):
    # Rate limit per IP
    ip = request.client.host if request.client else "unknown"
    if not _rate_limit(ip):
        raise HTTPException(status_code=429, detail="rate_limited")

    raw = await request.body()

    diag_payload = _decode_diagnostic_cookie(request)
    diag_session = ""
    diag_allow_all = False
    if diag_payload:
        diag_session = str(
            diag_payload.get("session_id")
            or diag_payload.get("sid")
            or diag_payload.get("session")
            or ""
        ).strip()
        diag_allow_all = bool(diag_payload.get("allow_all") or diag_payload.get("allow"))
        setattr(request.state, "diagnostic_mode", True)

    # Optional HMAC verification (best-effort)
    if HMAC_SECRET:
        sig = request.headers.get("x-log-signature")
        if not sig:
            raise HTTPException(status_code=401, detail="missing_signature")
        try:
            mac = hashlib.sha256()
            mac.update(raw)
            calc = mac.hexdigest()
            if sig != calc:
                raise HTTPException(status_code=401, detail="bad_signature")
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(status_code=400, detail="bad_request")

    # Normalize to list of events
    try:
        body = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw or "")
        parsed: Any = json.loads(body) if body else {}
    except Exception:
        raise HTTPException(status_code=400, detail="invalid_payload")

    try:
        events: List[LogEvent]
        if isinstance(parsed, list):
            events = [LogEvent.model_validate(e) for e in parsed]
        elif isinstance(parsed, dict) and isinstance(parsed.get("events"), list):
            events = [LogEvent.model_validate(e) for e in parsed.get("events")]  # type: ignore[arg-type]
        elif isinstance(parsed, dict):
            events = [LogEvent.model_validate(parsed)]
        else:
            raise HTTPException(status_code=400, detail="invalid_payload")
    except HTTPException:
        raise
    except Exception:
            raise HTTPException(status_code=400, detail="invalid_payload")

    # Enforce prod policy: only warn/error unless langgraph service explicitly emits info
    if ENV == "prod":
        for e in events:
            lvl = (e.level or "").lower()
            svc = (e.service or "").lower()
            if lvl in {"warn", "error"}:
                continue
            if diag_payload:
                if diag_allow_all:
                    continue
                if diag_session and diag_session == (e.session_id or ""):
                    continue
            if svc == "langgraph" and lvl in {"info", "warn", "error"}:
                continue
            raise HTTPException(status_code=400, detail="level_not_allowed")

    tenant_id = getattr(request.state, "tenant_id", None)
    out = []
    for e in events:
        lvl = (e.level or "").lower()
        if lvl not in ALLOWED_LEVELS:
            lvl = "error"
        svc = (e.service or "web").lower()
        e.level = lvl
        e.service = svc
        e.request_id = e.request_id or getattr(request.state, "request_id", None)
        e.trace_id = e.trace_id or getattr(request.state, "trace_id", None)
        e.data = _sanitize_data(e.data)
        rec = json.loads(e.model_dump_json())
        if tenant_id is not None:
            rec.setdefault("tenant_id", tenant_id)
        rec.setdefault("client_ip", ip)
        out.append(rec)

    logger = logging.getLogger("troubleshoot")
    level_map = {
        "error": logging.ERROR,
        "warn": logging.WARNING,
        "info": logging.INFO,
        "debug": logging.DEBUG,
    }
    for rec in out:
        try:
            line = json.dumps(rec, ensure_ascii=False)
            logger.log(level_map.get(rec.get("level", "warn"), logging.WARNING), line)
            _append_jsonl(line)
        except Exception:
            # swallow logging errors to keep ingestion responsive
            pass

    return {"accepted": len(out)}
