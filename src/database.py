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
        _pool = await asyncpg.create_pool(
            dsn=POSTGRES_DSN,
            min_size=0,
            max_size=1,
            init=lambda conn: conn.execute("SET search_path TO public;")
        )
    return _pool


def _get_sync_pool() -> SimpleConnectionPool:
    global _sync_pool
    if _sync_pool is None:
        max_conn = int(os.getenv("DB_MAX_CONN", "4"))
        # Keep the sync pool tiny to prevent connection slot exhaustion
        _sync_pool = SimpleConnectionPool(1, max_conn, dsn=POSTGRES_DSN)
    return _sync_pool


@contextmanager
def get_conn():
    """Context-managed pooled psycopg2 connection.

    Usage:
        with get_conn() as conn, conn.cursor() as cur:
            ...
    Ensures we reuse a small shared pool instead of opening new connections
    per request.
    """
    pool = _get_sync_pool()
    conn = pool.getconn()
    try:
        yield conn
    finally:
        try:
            # Best-effort rollback any open transaction before returning to pool
            if not conn.closed:
                conn.rollback()
        except Exception:
            pass
        pool.putconn(conn, close=False)
