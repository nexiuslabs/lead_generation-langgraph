import json
import types
import time as _time

import pytest


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


def enable_bridge(monkeypatch):
    import src.settings as settings

    monkeypatch.setattr(settings, "ENABLE_SERVER_MCP_BRIDGE", True, raising=False)
    monkeypatch.setattr(settings, "MCP_SERVER_NAME", "jina", raising=False)
    monkeypatch.setattr(settings, "LGS_BASE_URL", "http://127.0.0.1:8001", raising=False)
    monkeypatch.setattr(
        settings,
        "MCP_BRIDGE_INVOKE_URL",
        "http://127.0.0.1:8001/mcp/servers/{server}/tools/{tool}/invoke",
        raising=False,
    )
    monkeypatch.setattr(settings, "MCP_API_KEY", "test_key", raising=False)
    # Tighten timeouts for tests
    monkeypatch.setattr(settings, "MCP_BRIDGE_CONNECT_TIMEOUT_S", 0.1, raising=False)
    monkeypatch.setattr(settings, "MCP_BRIDGE_READ_TIMEOUT_S", 0.1, raising=False)
    monkeypatch.setattr(settings, "MCP_BRIDGE_COOL_OFF_S", 5.0, raising=False)


def test_mcp_bridge_read_url_success(monkeypatch):
    enable_bridge(monkeypatch)

    # capture the last request
    seen = {}

    def fake_post(url, headers=None, json=None, timeout=None):  # noqa: A002
        seen["url"] = url
        seen["headers"] = dict(headers or {})
        seen["json"] = json
        seen["timeout"] = timeout
        return _fake_response(200, {"content": " Hello world "})

    import src.services.mcp_server_bridge as bridge

    monkeypatch.setattr(bridge.requests, "post", fake_post)
    out = bridge.read_url("https://example.com", timeout=1.0)
    assert out == "Hello world"
    assert "authorization" in {k.lower() for k in seen["headers"].keys()}
    assert seen["url"].endswith("/mcp/servers/jina/tools/read_url/invoke")
    assert seen["json"] == {"arguments": {"url": "https://example.com", "timeout": 1.0}}


def test_mcp_bridge_search_variants(monkeypatch):
    enable_bridge(monkeypatch)
    import src.services.mcp_server_bridge as bridge

    # Variant 1: response is list of dicts/strings
    def fake_post_list(url, headers=None, json=None, timeout=None):  # noqa: A002
        return _fake_response(200, [
            {"url": "https://a"}, "https://b", {"ignored": True},
        ])

    monkeypatch.setattr(bridge.requests, "post", fake_post_list)
    out1 = bridge.search_web("q", country="sg", max_results=5)
    assert out1 == ["https://a", "https://b"]

    # Variant 2: response is dict with results list
    def fake_post_dict(url, headers=None, json=None, timeout=None):  # noqa: A002
        return _fake_response(200, {"results": [
            {"url": "https://c"}, "https://d",
        ]})

    monkeypatch.setattr(bridge.requests, "post", fake_post_dict)
    out2 = bridge.search_web("q2", max_results=5)
    assert out2 == ["https://c", "https://d"]


def test_mcp_bridge_timeout_then_cooldown_skip(monkeypatch):
    enable_bridge(monkeypatch)
    import src.services.mcp_server_bridge as bridge
    from requests import exceptions as req_exc

    # Freeze time so cooldown math is deterministic
    base = _time.time()
    tbox = {"now": base}

    def fake_time():
        return tbox["now"]

    monkeypatch.setattr(bridge, "time", types.SimpleNamespace(time=fake_time))

    def fake_post_raise(url, headers=None, json=None, timeout=None):  # noqa: A002
        raise req_exc.ReadTimeout("boom")

    monkeypatch.setattr(bridge.requests, "post", fake_post_raise)
    # First call errors and enters cooldown
    assert bridge.read_url("https://x") is None
    # Advance less than cooldown, next should skip quickly
    tbox["now"] = base + 1.0
    assert bridge.read_url("https://y") is None


def test_mcp_reader_uses_bridge_fallback_when_client_missing(monkeypatch):
    # Enable both flags so reader path is active
    import src.settings as settings

    monkeypatch.setattr(settings, "ENABLE_SERVER_MCP_BRIDGE", True, raising=False)
    monkeypatch.setattr(settings, "ENABLE_MCP_READER", True, raising=False)

    # Ensure bridge returns content without doing network
    import src.services.mcp_server_bridge as bridge

    monkeypatch.setattr(bridge, "read_url", lambda url, timeout=12.0: "from-bridge", raising=False)

    # Import reader and call read_url; it should synthesize a bridge-backed session
    import importlib
    mcp_reader = importlib.import_module("src.services.mcp_reader")

    out = mcp_reader.read_url("https://example.com", timeout=1.0)
    assert out == "from-bridge"


def test_jina_reader_prefers_bridge_then_client(monkeypatch):
    # Force bridge enabled
    import src.settings as settings
    monkeypatch.setattr(settings, "ENABLE_SERVER_MCP_BRIDGE", True, raising=False)
    monkeypatch.setattr(settings, "ENABLE_MCP_READER", True, raising=False)

    # Bridge returns None first -> forces client path
    import src.services.mcp_server_bridge as bridge
    calls = {"bridge": 0}

    def _br(url, timeout=12.0):
        calls["bridge"] += 1
        return None

    monkeypatch.setattr(bridge, "read_url", _br, raising=False)

    # Client path returns value (simulate by patching reader service)
    import src.services.mcp_reader as reader
    monkeypatch.setattr(reader, "read_url", lambda url, timeout=None: "client-ok", raising=False)

    import src.jina_reader as jr
    out = jr.read_url("https://example.com", timeout=1.0)
    assert out == "client-ok"
    assert calls["bridge"] >= 1

