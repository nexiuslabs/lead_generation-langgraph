import os
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional


def _resolve_dir() -> Optional[Path]:
  log_dir = os.getenv("TROUBLESHOOT_API_LOG_DIR") or os.getenv("LOGS_DIR")
  env = (os.getenv("ENVIRONMENT") or os.getenv("PY_ENV") or os.getenv("NODE_ENV") or "dev").lower()
  if not log_dir and env in {"dev", "development", "local", "localhost"}:
    log_dir = ".log_api"
  if not log_dir:
    return None
  return Path(log_dir).expanduser()


def _file_for(service: str) -> Optional[Path]:
  base = _resolve_dir()
  if not base:
    return None
  d = time.gmtime()
  fname = f"{service}-{d.tm_year:04d}-{d.tm_mon:02d}-{d.tm_mday:02d}.jsonl"
  p = base / fname
  try:
    p.parent.mkdir(parents=True, exist_ok=True)
  except Exception:
    return None
  return p


def log_json(service: str, level: str, message: str, data: Optional[Dict[str, Any]] = None) -> None:
  try:
    rec = {
      "timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
      "level": level,
      "service": service,
      "message": message,
      **({"data": data} if isinstance(data, dict) and data else {}),
    }
    # stdout for process manager capture
    print(json.dumps(rec, ensure_ascii=False))
    # file append when LOGS_DIR configured
    fp = _file_for(service)
    if fp:
      with fp.open('a', encoding='utf-8') as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
  except Exception:
    # best-effort only
    pass
