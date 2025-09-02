import os
import sys

os.environ.setdefault("OPENAI_API_KEY", "test")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import enrichment


class DummyTavilyClient:
    def __init__(self, results):
        self._results = results
        self.last_query = None

    def search(self, query):
        self.last_query = query
        return {"results": self._results}


def test_find_domain_allows_aggregator_when_name_matches_apex(monkeypatch):
    dummy_results = [
        {
            "url": "https://www.amazon.com",
            "title": "Amazon.com: Official Site",
            "content": "Shop online at Amazon",
        }
    ]
    monkeypatch.setattr(enrichment, "tavily_client", DummyTavilyClient(dummy_results))

    assert enrichment.find_domain("Amazon") == ["https://www.amazon.com"]


def test_find_domain_uses_title_or_snippet_when_domain_missing_name(monkeypatch):
    dummy_results = [
        {
            "url": "https://www.fairprice.com.sg",
            "title": "NTUC FairPrice - Home",
            "content": "Part of NTUC Enterprise",
        }
    ]
    monkeypatch.setattr(enrichment, "tavily_client", DummyTavilyClient(dummy_results))

    assert enrichment.find_domain("NTUC Enterprise") == ["https://www.fairprice.com.sg"]


def test_find_domain_normalizes_name_in_query(monkeypatch):
    dummy_results = [
        {
            "url": "https://nexiuslabs.com",
            "title": "Nexius Labs",
            "content": "Home",
        }
    ]
    dummy = DummyTavilyClient(dummy_results)
    monkeypatch.setattr(enrichment, "tavily_client", dummy)

    assert enrichment.find_domain("NEXIUS LABS PTE LTD") == ["https://nexiuslabs.com"]
    assert "nexius labs" in dummy.last_query.lower()
    assert "official website" in dummy.last_query.lower()
