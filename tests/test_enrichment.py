import os
import sys

os.environ.setdefault("OPENAI_API_KEY", "test")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import enrichment


class DummySearch:
    def __init__(self, results):
        self._results = {"results": results}

    def run(self, query):
        return self._results


def test_find_domain_allows_aggregator_when_name_matches_apex(monkeypatch):
    dummy_results = [
        {
            "url": "https://www.amazon.com",
            "title": "Amazon.com: Official Site",
            "content": "Shop online at Amazon",
        }
    ]
    monkeypatch.setattr(enrichment, "tavily_search", DummySearch(dummy_results))

    assert enrichment.find_domain("Amazon") == ["https://www.amazon.com"]


def test_find_domain_uses_title_or_snippet_when_domain_missing_name(monkeypatch):
    dummy_results = [
        {
            "url": "https://www.fairprice.com.sg",
            "title": "NTUC FairPrice - Home",
            "content": "Part of NTUC Enterprise",
        }
    ]
    monkeypatch.setattr(enrichment, "tavily_search", DummySearch(dummy_results))

    assert enrichment.find_domain("NTUC Enterprise") == ["https://www.fairprice.com.sg"]
