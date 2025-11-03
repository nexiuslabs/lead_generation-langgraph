import os
import asyncpg
import psycopg2
from psycopg2.pool import SimpleConnectionPool
from contextlib import contextmanager
from src.settings import POSTGRES_DSN

_pool = None
_sync_pool: SimpleConnectionPool | None = None

async def get_pg_pool():
    global _pool
    if _pool is None:
        # Initialize a very small async pool to avoid exhausting DB connections
        # Add a short connection timeout to avoid 60s+ stalls on DNS/host issues.
        try:
            _timeout = float(os.getenv("PG_CONNECT_TIMEOUT_S", "3") or 3)
        except Exception:
            _timeout = 3.0
        _pool = await asyncpg.create_pool(
            dsn=POSTGRES_DSN,
            min_size=0,
            max_size=1,
            timeout=_timeout,
            init=lambda conn: conn.execute("SET search_path TO public;")
        )
    return _pool


def _get_sync_pool() -> SimpleConnectionPool:
    global _sync_pool
    if _sync_pool is None:
        max_conn = int(os.getenv("DB_MAX_CONN", "4"))
        # Keep the sync pool tiny to prevent connection slot exhaustion
        # Add a short connection timeout so DNS/host issues fail fast in dev.
        try:
            _timeout = int(float(os.getenv("PG_CONNECT_TIMEOUT_S", "3") or 3))
        except Exception:
            _timeout = 3
        _sync_pool = SimpleConnectionPool(1, max_conn, dsn=POSTGRES_DSN, connect_timeout=_timeout)
    return _sync_pool


@contextmanager
def get_conn():
    """Context-managed pooled psycopg2 connection.

    Usage:
        with get_conn() as conn, conn.cursor() as cur:
            ...

    Behavior:
    - Commits if the block exits without exception.
    - Rolls back on exception, then re-raises.
    - Returns the connection to a small shared pool to avoid exhausting slots.
    """
    pool = _get_sync_pool()
    conn = pool.getconn()
    try:
        try:
            yield conn
            try:
                if not conn.closed:
                    conn.commit()
            except Exception:
                # If commit fails, best-effort rollback before returning to pool
                try:
                    if not conn.closed:
                        conn.rollback()
                except Exception:
                    pass
        except Exception:
            try:
                if not conn.closed:
                    conn.rollback()
            except Exception:
                pass
            raise
    finally:
        pool.putconn(conn, close=False)
