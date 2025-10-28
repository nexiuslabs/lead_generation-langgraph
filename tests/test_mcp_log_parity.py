import logging
import types
import time as _time

import pytest


def _enable_flags(monkeypatch):
    import src.settings as settings

    # Enable both bridge and reader for the flow
    monkeypatch.setattr(settings, "ENABLE_SERVER_MCP_BRIDGE", True, raising=False)
    monkeypatch.setattr(settings, "ENABLE_MCP_READER", True, raising=False)
    # Ensure auth header is NOT forced in local dev to avoid 403
    monkeypatch.setattr(settings, "MCP_BRIDGE_FORCE_AUTH", False, raising=False)
    monkeypatch.setattr(settings, "MCP_SERVER_NAME", "jina", raising=False)
    monkeypatch.setattr(settings, "LGS_BASE_URL", "http://127.0.0.1:8001", raising=False)
    monkeypatch.setattr(
        settings,
        "MCP_BRIDGE_INVOKE_URL",
        "http://127.0.0.1:8001/mcp/servers/{server}/tools/{tool}/invoke",
        raising=False,
    )
    # Shorten cooldown/timeouts for tests
    monkeypatch.setattr(settings, "MCP_BRIDGE_COOL_OFF_S", 10.0, raising=False)
    monkeypatch.setattr(settings, "MCP_BRIDGE_CONNECT_TIMEOUT_S", 0.05, raising=False)
    monkeypatch.setattr(settings, "MCP_BRIDGE_READ_TIMEOUT_S", 0.1, raising=False)


def _fake_response(status=200, json_obj=None, text=""):
    class R:
        def __init__(self):
            self.status_code = status
            self._json = json_obj
            self.text = text
            self.reason = "OK" if status == 200 else "ERR"

        def json(self):
            if self._json is None:
                raise ValueError("no json")
            return self._json

    return R()


def test_logs_read_url_timeout_to_cooldown_and_remote_406(monkeypatch, caplog):
    _enable_flags(monkeypatch)
    import src.services.mcp_server_bridge as bridge
    # Simulate the 406 pattern seen in logs (not a timeout)
    monkeypatch.setattr(bridge.requests, "post", lambda *a, **k: _fake_response(406, text="Not Acceptable"))

    # Remote MCP returns 406 Not Acceptable (log parity with provided logs)
    import src.services.mcp_remote as mremote

    monkeypatch.setattr(mremote.requests, "post", lambda *a, **k: _fake_response(406, text="Not Acceptable"))

    # Capture logs
    caplog.set_level(logging.INFO)
    import src.jina_reader as jr

    out = jr.read_url("https://example.com", timeout=0.1)
    assert out is None

    text = caplog.text
    # Verify key log messages from the provided logs
    assert "[mcp-bridge] selected read_url url=https://example.com" in text
    assert "[mcp-bridge] invoke start tool=read_url" in text
    assert "invoke fail tool=read_url status=406" in text

    # Fallback path logs
    assert "[mcp-bridge] no content; falling back" in text
    assert "[mcp] selected read_url url=https://example.com" in text
    assert "[mcp-remote] http status=406" in text
    assert "[mcp-remote] read_url empty" in text


def test_logs_search_web_cooldown_skip_and_empty(monkeypatch, caplog):
    _enable_flags(monkeypatch)
    import src.services.mcp_server_bridge as bridge

    # Force cooldown by setting the internal marker in the past-future window
    base = _time.time()
    monkeypatch.setattr(bridge, "_BRIDGE_DOWN_UNTIL", base + 120.0, raising=False)

    # Remote MCP returns 406 for search as well
    import src.services.mcp_remote as mremote
    monkeypatch.setattr(mremote.requests, "post", lambda *a, **k: _fake_response(406, text="Not Acceptable"))

    # Capture logs
    caplog.set_level(logging.INFO)
    import src.services.mcp_reader as reader

    out = reader.search_web("food distribution companies site:.sg", country="sg", max_results=10)
    assert out == []

    text = caplog.text
    # Matches the sequence in the logs: selected -> skipped -> empty
    assert "[mcp-bridge] search_web empty" in text
    assert "[mcp] invoke start tool=search_web" in text
    assert "[mcp] search_web empty" in text
