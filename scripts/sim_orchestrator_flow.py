import os
import sys
import types
from contextlib import contextmanager


# ----- Stub external deps before importing app code -----
_asyncpg_stub = types.SimpleNamespace()
class _AsyncpgPool:
    pass
class _AsyncpgConnection:
    pass
_asyncpg_stub.Pool = _AsyncpgPool
_asyncpg_stub.Connection = _AsyncpgConnection
_asyncpg_stub.create_pool = lambda *a, **k: None
sys.modules.setdefault("asyncpg", _asyncpg_stub)

class _DummyCursor:
    def execute(self, *args, **kwargs):
        return None
    @property
    def rowcount(self):
        return 0
    def fetchone(self):
        return None
    def fetchall(self):
        return []
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        return False

class _DummyConn:
    closed = False
    def cursor(self):
        return _DummyCursor()
    def commit(self):
        return None
    def rollback(self):
        return None

class _PsycoDummy:
    class Error(Exception):
        pass
    class extras:
        class Json(dict):
            pass
    class pool:
        class SimpleConnectionPool:
            def __init__(self, *a, **k):
                pass
            def getconn(self):
                return _DummyConn()
            def putconn(self, *a, **k):
                return None

sys.modules.setdefault("psycopg2", _PsycoDummy())
sys.modules.setdefault("psycopg2.extras", _PsycoDummy.extras)
sys.modules.setdefault("psycopg2.pool", _PsycoDummy.pool)


@contextmanager
def _dummy_get_conn():
    yield _DummyConn()


# Environment guards to avoid network paths
os.environ.setdefault("ORCHESTRATOR_OFFLINE", "1")
os.environ.setdefault("BG_DISCOVERY_AND_ENRICH", "false")
os.environ.setdefault("LANGGRAPH_CHECKPOINT_DIR", ".langgraph_api")


def main():
    # Patch DB helpers inside pre_sdr_graph before nodes import
    import app.pre_sdr_graph as presdr
    presdr.get_conn = _dummy_get_conn  # type: ignore

    # Collect saved icp_rules payloads for inspection
    saved_payloads = []
    _orig_save = getattr(presdr, "_save_icp_rule_sync", None)

    def _collector(tid: int, payload: dict, name: str = "Default ICP"):
        saved_payloads.append({"tenant_id": tid, "name": name, "payload": dict(payload)})
        # no-op instead of real DB write
        return None

    presdr._save_icp_rule_sync = _collector  # type: ignore

    from my_agent.utils.state import OrchestrationState
    from my_agent.utils import nodes

    # Scenario 1: company confirmed, no customer URLs yet → should ask for 5 URLs
    state1: OrchestrationState = {
        "messages": [
            {"role": "user", "content": "hello there"},
            {"role": "user", "content": "https://nexiuslabs.com"},
            {"role": "user", "content": "look great"},
        ],
        "entry_context": {"tenant_id": 1222, "intent": "confirm_company", "last_user_command": "ok confirmed"},
        "profile_state": {
            "company_profile": {
                "name": "Nexius Labs",
                "website": "https://nexiuslabs.com",
                "summary": "Test summary",
            },
            "company_profile_confirmed": True,
            "icp_profile": {},
            "icp_profile_generated": False,
            "icp_profile_confirmed": False,
            "customer_websites": [],
            "outstanding_prompts": [],
        },
    }

    import asyncio

    async def run_s1():
        out = await nodes.journey_guard(state1)
        prompts = (out.get("profile_state") or {}).get("outstanding_prompts") or []
        print("SCENARIO 1 OUTSTANDING:")
        for p in prompts:
            print(p)

    asyncio.run(run_s1())

    # Scenario 2: have 5 customer URLs + generated ICP → confirm ICP and persist with seed_urls
    state2: OrchestrationState = {
        "messages": [
            {"role": "user", "content": "confirm icp"},
        ],
        "entry_context": {"tenant_id": 1222, "intent": "confirm_icp", "last_user_command": "confirm icp"},
        "profile_state": {
            "company_profile": {
                "name": "Nexius Labs",
                "website": "https://nexiuslabs.com",
                "summary": "Test summary",
            },
            "company_profile_confirmed": True,
            "customer_websites": [
                "https://a.com",
                "https://b.com",
                "https://c.com",
                "https://d.com",
                "https://e.com",
            ],
            "icp_profile": {
                "summary": "ICP from customers",
                "industries": ["AI", "SaaS"],
                "company_sizes": ["1-10", "11-50"],
                "persona_titles": ["Founder"],
                "buying_triggers": ["Scaling"],
                "regions": ["SG"],
                "proof_points": ["45% efficiency"],
            },
            "icp_profile_generated": True,
            "icp_profile_confirmed": True,
            "outstanding_prompts": [],
        },
    }

    async def run_s2():
        saved_payloads.clear()
        out = await nodes.journey_guard(state2)
        print("SCENARIO 2 SAVED ICP PAYLOADS:")
        for rec in saved_payloads:
            payload = rec.get("payload") or {}
            seeds = payload.get("seed_urls") or []
            print({
                "tenant_id": rec.get("tenant_id"),
                "name": rec.get("name"),
                "keys": sorted(list(payload.keys()))[:10],
                "seed_urls_count": len(seeds),
                "has_summary": bool(payload.get("summary")),
            })

    asyncio.run(run_s2())


if __name__ == "__main__":
    # Add repo root to Python path
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, os.pardir))
    if root not in sys.path:
        sys.path.insert(0, root)
    main()
