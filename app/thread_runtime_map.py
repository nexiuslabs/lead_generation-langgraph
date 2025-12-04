from __future__ import annotations

import json
import os
import threading
from typing import Optional

_LOCK = threading.Lock()


def _store_path() -> str:
    base = os.getenv("LANGGRAPH_CHECKPOINT_DIR", ".langgraph_api").rstrip("/")
    try:
        os.makedirs(base, exist_ok=True)
    except Exception:
        pass
    return os.path.join(base, "runtime_thread_map.json")


def get_runtime_id(db_thread_id: str) -> Optional[str]:
    path = _store_path()
    try:
        with _LOCK:
            if not os.path.exists(path):
                return None
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
            rid = data.get(db_thread_id)
            return str(rid) if rid else None
    except Exception:
        return None


def set_runtime_id(db_thread_id: str, runtime_thread_id: str) -> None:
    path = _store_path()
    try:
        with _LOCK:
            data = {}
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f) or {}
                except Exception:
                    data = {}
            data[db_thread_id] = str(runtime_thread_id)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(tmp, path)
    except Exception:
        # best-effort only
        pass

