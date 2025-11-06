import types

from app import logs_routes as lr


def test_sanitize_data_blocks_sensitive_fields():
    payload = {
        "route": "/foo",
        "authorization": "Bearer secret",
        "token": "abc",
        "status": 403,
        "method": "GET",
        "component": "useAuthFetch",
        "pathname": "/foo?secret=yes",
        "custom": "ignored",
    }
    sanitized = lr._sanitize_data(payload)
    assert sanitized == {
        "route": "/foo",
        "status": 403,
        "method": "GET",
        "component": "useAuthFetch",
        "pathname": "/foo",
    }


def test_rate_limit_consumes_bucket(monkeypatch):
    lr._buckets.clear()
    fake = types.SimpleNamespace(now=1000.0)

    def _fake_time():
        return fake.now

    monkeypatch.setattr(lr.time, "time", _fake_time)

    ip = "10.0.0.1"
    for _ in range(20):
        assert lr._rate_limit(ip)
    assert not lr._rate_limit(ip)

    fake.now += 5.0
    assert lr._rate_limit(ip)
