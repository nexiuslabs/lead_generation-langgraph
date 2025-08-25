import asyncpg
import psycopg2
from src.settings import POSTGRES_DSN

_pool = None

async def get_pg_pool():
    global _pool
    if _pool is None:
        # Initialize connection pool with public schema search_path
        _pool = await asyncpg.create_pool(
            dsn=POSTGRES_DSN,
            min_size=0,
            max_size=1,
            # init should be a coroutine callback to set the search_path
            init=lambda conn: conn.execute("SET search_path TO public;")
        )
    return _pool


def get_conn():
    """Synchronous psycopg2 connection for code that expects a blocking conn."""
    return psycopg2.connect(dsn=POSTGRES_DSN)
