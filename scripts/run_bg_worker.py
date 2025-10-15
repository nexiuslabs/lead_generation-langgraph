import asyncio
import json
import os
import signal
from typing import Optional

import asyncpg
import re

from src.settings import POSTGRES_DSN
from src.jobs import run_web_discovery_bg_enrich


class BGWorker:
    def __init__(self, dsn: str, max_concurrency: int = 2, sweep_interval: float = 15.0):
        self._dsn = dsn
        self._max = max_concurrency
        self._interval = sweep_interval
        self._stop = asyncio.Event()
        self._notify_event = asyncio.Event()
        self._tasks: set[asyncio.Task] = set()

    async def _listen(self) -> None:
        try:
            conn = await asyncpg.connect(dsn=self._dsn)
        except Exception:
            # Fallback to polling only
            return
        try:
            await conn.add_listener("bg_jobs", self._on_notify)  # type: ignore[attr-defined]
        except Exception:
            # Some asyncpg versions use different API; fallback to raw SQL LISTEN loop
            await conn.execute("LISTEN bg_jobs")
        try:
            while not self._stop.is_set():
                # Wait for notifications or timeout to keep connection alive
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=self._interval)
                except asyncio.TimeoutError:
                    continue
        finally:
            try:
                await conn.close()
            except Exception:
                pass

    def _on_notify(self, *_args) -> None:
        # Wake the sweeper loop to pick up newly queued jobs
        try:
            self._notify_event.set()
        except Exception:
            pass

    async def _claim_one(self, conn: asyncpg.Connection, job_type: str) -> Optional[int]:
        sql = (
            """
            WITH cte AS (
              SELECT job_id
              FROM background_jobs
              WHERE status='queued' AND job_type=$1
              ORDER BY job_id ASC
              FOR UPDATE SKIP LOCKED
              LIMIT 1
            )
            UPDATE background_jobs b
               SET status='running', started_at=now()
              FROM cte
             WHERE b.job_id = cte.job_id
         RETURNING b.job_id
            """
        )
        row = await conn.fetchrow(sql, job_type)
        return int(row[0]) if row and row[0] is not None else None

    async def _spawn_until_full(self, pool: asyncpg.Pool) -> None:
        async with pool.acquire() as conn:
            while len(self._tasks) < self._max and not self._stop.is_set():
                jid = await self._claim_one(conn, "web_discovery_bg_enrich")
                if not jid:
                    break
                t = asyncio.create_task(self._run_job(jid))
                self._tasks.add(t)
                t.add_done_callback(lambda tt: self._tasks.discard(tt))

    async def _run_job(self, job_id: int) -> None:
        try:
            await run_web_discovery_bg_enrich(int(job_id))
        except Exception:
            # Errors are handled inside the job call; ensure task finishes cleanly
            pass

    async def run(self) -> None:
        pool = await asyncpg.create_pool(dsn=self._dsn, min_size=1, max_size=max(2, self._max))
        listen_task = asyncio.create_task(self._listen())
        try:
            while not self._stop.is_set():
                await self._spawn_until_full(pool)
                # Wait for either notify, a task to finish, or timeout
                waiters = []
                # notify event
                waiters.append(asyncio.create_task(self._notify_event.wait()))
                # any existing task completion
                if self._tasks:
                    waiters.append(asyncio.create_task(asyncio.wait(self._tasks, return_when=asyncio.FIRST_COMPLETED)))
                try:
                    await asyncio.wait(waiters, timeout=self._interval, return_when=asyncio.FIRST_COMPLETED)
                except Exception:
                    pass
                finally:
                    for w in waiters:
                        try:
                            w.cancel()
                        except Exception:
                            pass
                self._notify_event.clear()
        finally:
            try:
                listen_task.cancel()
            except Exception:
                pass
            try:
                await pool.close()
            except Exception:
                pass

    def stop(self) -> None:
        try:
            self._stop.set()
        except Exception:
            pass


async def main() -> None:
    def _parse_int(env: str, default: int) -> int:
        val = os.getenv(env)
        if val is None:
            return default
        try:
            return int(val.strip())
        except Exception:
            # strip non-digits and retry
            m = re.search(r"\d+", val)
            return int(m.group(0)) if m else default

    def _parse_interval(env: str, default: float) -> float:
        raw = os.getenv(env)
        if not raw:
            return default
        s = raw.strip().lower()
        # Support suffix units: s, m, h (seconds/minutes/hours)
        try:
            if s.endswith("ms"):
                return max(0.01, float(s[:-2]) / 1000.0)
            if s.endswith("s"):
                return max(0.01, float(s[:-1]))
            if s.endswith("m"):
                return max(0.01, float(s[:-1]) * 60.0)
            if s.endswith("h"):
                return max(0.01, float(s[:-1]) * 3600.0)
            # Plain float
            return max(0.01, float(s))
        except Exception:
            # Last resort: extract first float-like token
            m = re.search(r"\d+(?:\.\d+)?", s)
            try:
                return max(0.01, float(m.group(0))) if m else default
            except Exception:
                return default

    max_c = _parse_int("BG_WORKER_MAX_CONCURRENCY", 2)
    interval = _parse_interval("BG_WORKER_SWEEP_INTERVAL", 15.0)
    worker = BGWorker(POSTGRES_DSN, max_concurrency=max_c, sweep_interval=interval)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, worker.stop)
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
