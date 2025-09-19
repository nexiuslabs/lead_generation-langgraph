import pytest
from src.vendors import apify_linkedin as api


def test_build_queries_basic():
    q = api.build_queries("Acme Pte Ltd", ["founder", "head of sales"])  # mixed spacing
    assert len(q) == 1
    s = q[0]
    assert '"Acme Pte Ltd"' in s
    # titles joined with OR; quotes applied to those with spaces
    assert 'founder' in s and '"head of sales"' in s


def test_normalize_contacts():
    items = [
        {"fullName": "Jane Doe", "headline": "CTO", "url": "https://linkedin.com/in/jane", "locationName": "SG"},
        {"name": "John R", "title": "Head of Eng", "profileUrl": "https://ln/xyz", "location": "US"},
    ]
    out = api.normalize_contacts(items)
    assert len(out) == 2
    assert out[0]["full_name"] == "Jane Doe"
    assert out[0]["title"] == "CTO"
    assert out[0]["linkedin_url"].startswith("https://")
    assert out[0]["location"] == "SG"
    assert out[1]["full_name"] == "John R"
    assert out[1]["title"] == "Head of Eng"


@pytest.mark.asyncio
async def test_run_sync_get_dataset_items(monkeypatch):
    class FakeResp:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, params=None, json=None, headers=None):
            # return wrapped items
            return FakeResp({"items": [{"fullName": "A"}, {"fullName": "B"}]})

    monkeypatch.setattr(api.httpx, "AsyncClient", FakeClient)

    items = await api.run_sync_get_dataset_items({"queries": ["foo"]})
    assert isinstance(items, list)
    assert items and items[0].get("fullName") == "A"

