import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.industry_resolver as industry_resolver


class DummyCursor:
    def __init__(self, data):
        self.data = data
        self.result = []

    def execute(self, sql, params):
        if "staging_acra_companies" in sql:
            term = params[0].strip("%")
            self.result = self.data.get("staging", {}).get(term, [])
        elif "industry_norm" in sql and "GROUP BY" in sql:
            term = params[0].strip("%")
            self.result = self.data.get("codes_by_norm", {}).get(term, [])
        else:
            code = params[0]
            self.result = self.data.get("samples", {}).get(code, [])

    def fetchall(self):
        return self.result

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class DummyConn:
    def __init__(self, data):
        self.data = data

    def cursor(self):
        return DummyCursor(self.data)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_resolve_industry_terms_found(monkeypatch):
    data = {
        "staging": {
            "software": [("62019", "Software development", 10)],
        },
        "samples": {
            "62019": [("Acme Soft",), ("BetaWare",)],
        },
    }
    monkeypatch.setattr(industry_resolver, "get_conn", lambda: DummyConn(data))
    out = industry_resolver.resolve_industry_terms(["software"])
    assert out[0]["matches"][0]["code"] == "62019"
    assert out[0]["matches"][0]["companies"] == ["Acme Soft", "BetaWare"]


def test_resolve_industry_terms_no_data(monkeypatch):
    data = {"staging": {}, "codes_by_norm": {}, "samples": {}}
    monkeypatch.setattr(industry_resolver, "get_conn", lambda: DummyConn(data))
    out = industry_resolver.resolve_industry_terms(["unknown"])
    assert out[0]["matches"] == []


def test_resolve_industry_terms_fallback(monkeypatch):
    data = {
        "staging": {},
        "codes_by_norm": {"retail": [("47110", "Retail sale", 2)]},
        "samples": {"47110": [("ShopCo",)]},
    }
    monkeypatch.setattr(industry_resolver, "get_conn", lambda: DummyConn(data))
    out = industry_resolver.resolve_industry_terms(["retail"])
    assert out[0]["matches"][0]["code"] == "47110"
