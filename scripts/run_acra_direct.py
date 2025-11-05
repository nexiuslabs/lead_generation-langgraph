import os
import sys
import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
import re

# Ensure project root on path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _configure_logging() -> None:
    env = (
        os.getenv("ENVIRONMENT")
        or os.getenv("PY_ENV")
        or os.getenv("NODE_ENV")
        or "dev"
    ).strip().lower()
    log_dir = os.getenv("TROUBLESHOOT_API_LOG_DIR")
    if not log_dir and env in {"dev", "development", "local", "localhost"}:
        log_dir = ".log_api"
    if not log_dir:
        logging.basicConfig(level=logging.INFO)
        return
    try:
        path = Path(log_dir).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        handler = TimedRotatingFileHandler(
            path / "api.log",
            when="midnight",
            interval=1,
            backupCount=14,
            encoding="utf-8",
            utc=True,
        )
        handler.suffix = "%Y-%m-%d"
        handler.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}$")  # type: ignore[attr-defined]
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S"))
        root = logging.getLogger()
        if not root.handlers:
            root.addHandler(handler)
        else:
            existing = [h for h in root.handlers if isinstance(h, TimedRotatingFileHandler) and getattr(h, "baseFilename", None) == str(path / "api.log")]
            if not existing:
                root.addHandler(handler)
        stream_present = any(isinstance(h, logging.StreamHandler) and not isinstance(h, TimedRotatingFileHandler) for h in root.handlers)
        if not stream_present:
            sh = logging.StreamHandler()
            sh.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S"))
            root.addHandler(sh)
        if root.level == logging.NOTSET:
            root.setLevel(logging.INFO)
    except Exception:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def main():
    _configure_logging()
    try:
        from src.acra_direct import run_once
    except Exception as e:
        print(f"acra_direct import failed: {e}")
        sys.exit(2)
    try:
        res = run_once()
        print(f"acra_direct: {res}")
        sys.exit(0)
    except Exception as e:
        print(f"acra_direct fatal: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
