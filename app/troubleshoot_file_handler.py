from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


class DailyPersistHandler(logging.Handler):
    """Persist log records into per-day files with ISO timestamps and bounded retention."""

    def __init__(self, directory: Path, retention_hours: int = 168, prefix: str = "api"):
        super().__init__()
        self.directory = Path(directory).expanduser()
        self.retention = timedelta(hours=max(retention_hours, 1))
        self.prefix = prefix or "api"
        self._current_date: Optional[str] = None
        self._file = None
        self._lock = threading.Lock()

    def _ensure_file(self, now: datetime) -> None:
        date_str = now.strftime("%Y-%m-%d")
        if self._current_date == date_str and self._file:
            return
        if self._file:
            try:
                self._file.close()
            except Exception:
                pass
        self.directory.mkdir(parents=True, exist_ok=True)
        self._current_date = date_str
        file_path = self.directory / f"{self.prefix}-{date_str}.log"
        self._file = file_path.open("a", encoding="utf-8")

    def _prune(self, now: datetime) -> None:
        cutoff = now - self.retention
        threshold = cutoff.timestamp()
        pattern = f"{self.prefix}-*.log"
        for candidate in self.directory.glob(pattern):
            try:
                if candidate.stat().st_mtime < threshold:
                    candidate.unlink(missing_ok=True)
            except Exception:
                continue

    def emit(self, record: logging.LogRecord) -> None:  # type: ignore[override]
        try:
            msg = self.format(record)
            now = datetime.utcnow()
            line = f"{now.isoformat(timespec='milliseconds')}Z {msg}"
            with self._lock:
                self._ensure_file(now)
                if not self._file:
                    return
                self._file.write(line + "\n")
                self._file.flush()
                self._prune(now)
        except Exception:
            self.handleError(record)

    def close(self) -> None:  # type: ignore[override]
        with self._lock:
            if self._file:
                try:
                    self._file.close()
                except Exception:
                    pass
                self._file = None
        super().close()


__all__ = ["DailyPersistHandler"]
