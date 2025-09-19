import os
import sys
import types
from typing import Any, Dict, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if "jwt" not in sys.modules:
    jwt_stub = types.ModuleType("jwt")

    class _StubJWTError(Exception):
        pass

    def _stub_get_unverified_header(token: str) -> Dict[str, Any]:
        return {}

    class _StubRSAAlgorithm:
        @staticmethod
        def from_jwk(data: str) -> str:
            return "key"

    jwt_stub.PyJWTError = _StubJWTError
    jwt_stub.get_unverified_header = _stub_get_unverified_header
    jwt_stub.decode = lambda *args, **kwargs: {}
    jwt_stub.algorithms = types.SimpleNamespace(RSAAlgorithm=_StubRSAAlgorithm)

    sys.modules["jwt"] = jwt_stub

import pytest
from starlette.authentication import AuthenticationError
from starlette.requests import Request

from app import lg_auth


async def _empty_receive() -> Dict[str, Any]:
    return {"type": "http.request", "body": b"", "more_body": False}


def _make_request(headers: Optional[Dict[str, str]] = None, cookies: Optional[Dict[str, str]] = None) -> Request:
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "path": "/",
        "raw_path": b"/",
        "query_string": b"",
        "headers": [],
    }
    headers = headers or {}
    for key, value in headers.items():
        scope["headers"].append((key.lower().encode(), value.encode()))
    if cookies:
        cookie_header = "; ".join(f"{k}={v}" for k, v in cookies.items())
        scope["headers"].append((b"cookie", cookie_header.encode()))
    return Request(scope, _empty_receive)


@pytest.mark.anyio("asyncio")
async def test_authenticate_uses_verified_tenant_claim(monkeypatch):
    monkeypatch.setenv("LANGGRAPH_ALLOW_ANON", "false")
    monkeypatch.setenv("LANGSMITH_LANGGRAPH_API_VARIANT", "")

    request = _make_request(headers={"x-tenant-id": "spoofed"})

    observed = {}

    def fake_verify(token: str) -> Dict[str, Any]:
        observed["token"] = token
        return {"tenant_id": "verified", "roles": ["user"]}

    monkeypatch.setattr(lg_auth, "verify_jwt", fake_verify)

    identity = await lg_auth.authenticate(request, authorization="Bearer good-token")

    assert identity == "tenant:verified"
    assert observed["token"] == "good-token"
    assert getattr(request.state, "tenant_id") == "verified"
    assert getattr(request.state, "roles") == ["user"]


@pytest.mark.anyio("asyncio")
async def test_authenticate_rejects_missing_tenant_claim(monkeypatch):
    monkeypatch.setenv("LANGGRAPH_ALLOW_ANON", "false")
    monkeypatch.setenv("LANGSMITH_LANGGRAPH_API_VARIANT", "")

    request = _make_request()

    def fake_verify(token: str) -> Dict[str, Any]:
        return {"sub": "user"}

    monkeypatch.setattr(lg_auth, "verify_jwt", fake_verify)

    with pytest.raises(AuthenticationError) as exc:
        await lg_auth.authenticate(request, authorization="Bearer missing-tenant")

    assert "tenant" in str(exc.value).lower()


@pytest.mark.anyio("asyncio")
async def test_authenticate_ignores_spoofed_tenant_header(monkeypatch):
    monkeypatch.setenv("LANGGRAPH_ALLOW_ANON", "false")
    monkeypatch.setenv("LANGSMITH_LANGGRAPH_API_VARIANT", "")

    request = _make_request(headers={"x-tenant-id": "spoofed"})

    def fake_verify(token: str) -> Dict[str, Any]:
        return {"tenant_id": "legit"}

    monkeypatch.setattr(lg_auth, "verify_jwt", fake_verify)

    identity = await lg_auth.authenticate(request, authorization="Bearer actual")

    assert identity == "tenant:legit"

@pytest.fixture
def anyio_backend():
    return "asyncio"

