import asyncio
import types
import pytest


@pytest.mark.asyncio
async def test_agentic_send_results_success(monkeypatch):
    """Agentic email flow succeeds with dummy LLM/tool + renderer + sender.

    This test does not require DB or network; it stubs all external calls.
    """
    from src.notifications import agentic_email as ag

    # Ensure email feature is enabled
    monkeypatch.setattr(ag, "EMAIL_ENABLED", True, raising=False)

    # Stub renderer to avoid DB
    dummy_subject = "Your shortlist is ready — 2 leads (Tenant 123)"
    dummy_table = "<table><tr><th>Name</th></tr><tr><td>Acme</td></tr></table>"
    dummy_link = "/export/latest_scores.csv?limit=500"

    def _fake_render(tenant_id: int, limit: int = 200):
        assert tenant_id == 123
        assert limit == 2
        return dummy_subject, dummy_table, dummy_link

    monkeypatch.setattr("src.notifications.agentic_email.render_summary_html", _fake_render)

    # Stub sender to avoid network
    async def _fake_send(to: str, subject: str, html: str):
        assert to == "user@example.com"
        assert subject in ("Subject from LLM", dummy_subject)
        assert dummy_table in html  # intro + deterministic table concatenated
        return {"status": "sent", "http_status": 202, "request_id": "req-123"}

    monkeypatch.setattr("src.notifications.agentic_email.send_leads_email", _fake_send)

    # Fake LLM that issues a tool call with minimal args
    class FakeLLM:
        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages):
            # Return object with tool_calls to trigger send_email_tool
            obj = types.SimpleNamespace()
            obj.tool_calls = [
                {
                    "name": "send_email_tool",
                    "args": {
                        "to": "user@example.com",
                        "subject": "Subject from LLM",
                        "intro_html": "<p>Here is your shortlist.</p>",
                        "tenant_id": 123,
                        "limit": 2,
                    },
                }
            ]
            return obj

    monkeypatch.setattr(ag, "_make_llm", lambda: FakeLLM())

    # Run agentic flow
    res = await ag.agentic_send_results("user@example.com", 123, limit=2)
    assert res["status"] == "sent"
    assert res["to"] == "user@example.com"
    # csv_link propagated by tool wrapper
    assert res.get("csv_link") == dummy_link


@pytest.mark.asyncio
async def test_agentic_send_results_skipped_when_disabled(monkeypatch):
    from src.notifications import agentic_email as ag
    monkeypatch.setattr(ag, "EMAIL_ENABLED", False, raising=False)
    res = await ag.agentic_send_results("user@example.com", 123)
    assert res["status"] == "skipped_no_config"


@pytest.mark.asyncio
async def test_sendgrid_adapter_success_and_failure(monkeypatch):
    from src.notifications import sendgrid as sg

    # Enable feature and set dummy creds
    monkeypatch.setattr(sg, "EMAIL_ENABLED", True, raising=False)
    monkeypatch.setattr(sg, "SENDGRID_API_KEY", "SG.dummy", raising=False)
    monkeypatch.setattr(sg, "SENDGRID_FROM_EMAIL", "no-reply@example.com", raising=False)

    # Fake httpx client
    class DummyResp:
        def __init__(self, code: int, text: str = "", headers=None):
            self.status_code = code
            self.text = text
            self.headers = headers or {"X-Message-Id": "msg-1"}

    class FakeClient:
        _calls = 0
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json=None, headers=None):
            # 1st call returns success, 2nd call returns failure (class-level counter across instances)
            FakeClient._calls += 1
            if FakeClient._calls == 1:
                return DummyResp(202, "accepted")
            return DummyResp(400, "bad request")

    monkeypatch.setattr("src.notifications.sendgrid.httpx.AsyncClient", FakeClient)

    ok = await sg.send_leads_email("user@example.com", "Subject", "<p>Hi</p>")
    assert ok["status"] == "sent"
    assert ok["http_status"] == 202
    assert ok.get("request_id")

    bad = await sg.send_leads_email("user@example.com", "Subject", "<p>Hi</p>")
    assert bad["status"] == "failed"
    assert bad["http_status"] == 400
