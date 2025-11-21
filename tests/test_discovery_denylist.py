def test_discovery_planner_applies_denylist(monkeypatch):
    import importlib
    from src import agents_icp as mod

    # Ensure a clean module state
    importlib.reload(mod)

    # Monkeypatch DDG discovery to return a controlled set
    def fake_ddg_search_domains(query: str, max_results: int = 25, country: str | None = None, lang: str | None = None):
        return [
            "w3.org",               # apex deny
            "foo.edu.sg",           # host suffix deny
            "good.com.sg",          # allowed
            "www.bar.gov.sg",       # host suffix deny
            "sub.w3.org",           # subdomain of apex deny
        ]

    monkeypatch.setattr(mod, "_ddg_search_domains", fake_ddg_search_domains)

    # Minimal ICP profile; SG region hint not required for deny to apply
    state = {
        "icp_profile": {"industries": ["digital marketing"]}
    }

    out = mod.discovery_planner(state)
    cands = out.get("discovery_candidates") or []
    assert "good.com.sg" in cands
    assert "w3.org" not in cands
    assert "sub.w3.org" not in cands
    assert "foo.edu.sg" not in cands
    assert "www.bar.gov.sg" not in cands


def test_mcp_search_domains_falls_back_to_ddg(monkeypatch):
    import importlib
    from src import agents_icp as mod
    from src.services import mcp_reader

    importlib.reload(mod)

    # Force MCP search to return no URLs
    monkeypatch.setattr(mcp_reader, "search_web", lambda query, country=None, max_results=None: [])

    fallback_domains = ["fallback.com", "extra.sg"]

    def fake_ddg(query: str, max_results: int = 25, country: str | None = None, lang: str | None = None):
        return list(fallback_domains)

    monkeypatch.setattr(mod, "_ddg_search_domains", fake_ddg)

    result = mod._mcp_search_domains("test query site:.sg", max_results=10, country="sg")
    assert result == fallback_domains
