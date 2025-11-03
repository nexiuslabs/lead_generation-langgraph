import types


def test_dual_read_sampling_returns_http_and_logs(monkeypatch):
    from src import jina_reader, settings

    # Enable MCP and set dual-read to 100%
    monkeypatch.setattr(settings, "ENABLE_MCP_READER", True, raising=False)
    monkeypatch.setattr(settings, "MCP_DUAL_READ_PCT", 100, raising=False)

    # Force HTTP branch to return deterministic text
    monkeypatch.setattr(jina_reader, "_read_url_http", lambda url, timeout=12.0: "HTTP_TEXT")

    # Stub MCP read_url to return some text (won't be returned to caller in dual-read)
    import src.services.mcp_reader as mcp_reader
    monkeypatch.setattr(mcp_reader, "read_url", lambda url, timeout_s=None: "MCP_TEXT")

    # Capture parity logging
    calls = {"count": 0, "last": None}

    def _capture(url, t0, **kwargs):
        calls["count"] += 1
        calls["last"] = kwargs

    monkeypatch.setattr(jina_reader, "_log_mcp_dual_event", _capture)

    out = jina_reader.read_url("https://example.com", timeout=1)
    assert out == "HTTP_TEXT"
    assert calls["count"] == 1
    # Ensure logged fields exist
    assert "http_txt" not in calls["last"]  # not passed directly
    # lengths present when texts are strings
    assert "http_len" in calls["last"]
    assert "mcp_len" in calls["last"]


def test_mcp_enabled_prefers_mcp_when_no_dual_and_applies_cleaner(monkeypatch):
    from src import jina_reader, settings

    monkeypatch.setattr(settings, "ENABLE_MCP_READER", True, raising=False)
    monkeypatch.setattr(settings, "MCP_DUAL_READ_PCT", 0, raising=False)

    # Stub MCP read_url to return raw text
    import src.services.mcp_reader as mcp_reader
    monkeypatch.setattr(mcp_reader, "read_url", lambda url, timeout_s=None: "RAW_MCP_TEXT")

    # Ensure cleaner is applied
    cleaned_holder = {"called": False}

    def fake_clean(txt: str) -> str:
        cleaned_holder["called"] = True
        return f"CLEANED::{txt}"

    monkeypatch.setattr(jina_reader, "clean_jina_text", fake_clean)

    out = jina_reader.read_url("https://example.com", timeout=1)
    assert out == "CLEANED::RAW_MCP_TEXT"
    assert cleaned_holder["called"] is True


def test_mcp_error_falls_back_to_http(monkeypatch):
    from src import jina_reader, settings

    monkeypatch.setattr(settings, "ENABLE_MCP_READER", True, raising=False)
    monkeypatch.setattr(settings, "MCP_DUAL_READ_PCT", 0, raising=False)

    # MCP raises
    import src.services.mcp_reader as mcp_reader
    def _raise(url, timeout_s=None):
        raise RuntimeError("boom")
    monkeypatch.setattr(mcp_reader, "read_url", _raise)

    # HTTP fallback returns
    monkeypatch.setattr(jina_reader, "_read_url_http", lambda url, timeout=12.0: "HTTP_TEXT_FALLBACK")

    out = jina_reader.read_url("https://example.com", timeout=1)
    assert out == "HTTP_TEXT_FALLBACK"

