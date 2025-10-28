import types
import time as _time


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


def _enable_bridge(monkeypatch, *, force_auth=False):
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
    monkeypatch.setattr(settings, "MCP_BRIDGE_FORCE_AUTH", force_auth, raising=False)
    monkeypatch.setattr(settings, "MCP_API_KEY", "test_key", raising=False)
    # Small cooldowns for tests
    monkeypatch.setattr(settings, "MCP_BRIDGE_COOL_OFF_S", 5.0, raising=False)
    monkeypatch.setattr(settings, "MCP_BRIDGE_CONNECT_TIMEOUT_S", 0.05, raising=False)
    monkeypatch.setattr(settings, "MCP_BRIDGE_READ_TIMEOUT_S", 0.1, raising=False)


def test_bridge_403_when_force_auth(monkeypatch):
    _enable_bridge(monkeypatch, force_auth=True)
    import src.services.mcp_server_bridge as bridge

    seen = {}

    def fake_post(url, headers=None, json=None, timeout=None):  # noqa: A002
        seen["headers"] = dict(headers or {})
        return _fake_response(403, {"error": "forbidden"})

    monkeypatch.setattr(bridge.requests, "post", fake_post)
    out = bridge.read_url("https://example.com", timeout=0.1)
    # 403 yields None and auth header was attached due to force_auth
    assert out is None
    assert "authorization" in {k.lower() for k in seen["headers"].keys()}


def test_bridge_404_not_found(monkeypatch):
    _enable_bridge(monkeypatch, force_auth=False)
    import src.services.mcp_server_bridge as bridge

    monkeypatch.setattr(bridge.requests, "post", lambda *a, **k: _fake_response(404, {"error": "not found"}))
    assert bridge.read_url("https://example.com", timeout=0.1) is None


def test_reader_fallbacks_on_404_and_remote_406(monkeypatch):
    # Enable reader and bridge
    import src.settings as settings
    monkeypatch.setattr(settings, "ENABLE_SERVER_MCP_BRIDGE", True, raising=False)
    monkeypatch.setattr(settings, "ENABLE_MCP_READER", True, raising=False)

    # Bridge returns 404 -> causes empty, and we simulate cooldown skip next call
    import src.services.mcp_server_bridge as bridge

    base = _time.time()
    tbox = {"now": base}

    def fake_time():
        return tbox["now"]

    monkeypatch.setattr(bridge, "time", types.SimpleNamespace(time=fake_time))

    def fake_post_404(*a, **k):
        return _fake_response(404, {"error": "not found"})

    monkeypatch.setattr(bridge.requests, "post", fake_post_404)

    # Also make remote JSON-RPC return 406
    import src.services.mcp_remote as mremote

    monkeypatch.setattr(mremote.requests, "post", lambda *a, **k: _fake_response(406, text="Not Acceptable"))

    # Invoke reader -> should end up returning None due to both failures
    import src.jina_reader as jr
    out = jr.read_url("https://example.com", timeout=0.1)
    assert out is None


def test_reader_search_web_fallback_empty(monkeypatch):
    # Enable reader and bridge
    import src.settings as settings
    monkeypatch.setattr(settings, "ENABLE_SERVER_MCP_BRIDGE", True, raising=False)
    monkeypatch.setattr(settings, "ENABLE_MCP_READER", True, raising=False)

    # Bridge returns 404
    import src.services.mcp_server_bridge as bridge
    monkeypatch.setattr(bridge.requests, "post", lambda *a, **k: _fake_response(404, {"error": "not found"}))

    # Remote returns 406 (no content extracted)
    import src.services.mcp_remote as mremote
    monkeypatch.setattr(mremote.requests, "post", lambda *a, **k: _fake_response(406, text="Not Acceptable"))

    # Reader search -> empty list
    import src.services.mcp_reader as reader
    out = reader.search_web("some query", country="sg", max_results=5)
    assert out == []

