import os
import sys

# Ensure env and import path for app and src modules
os.environ.setdefault("OPENAI_API_KEY", "test")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient

import src.icp as icp
from app.main import app


def test_norm_ssic_variants():
    # Basic passthrough 5-digit code
    assert icp._norm_ssic("62010") == "62010"
    # Pad shorter codes to 5 digits
    assert icp._norm_ssic("6201") == "06201"
    # Strip non-digits and truncate to 5
    assert icp._norm_ssic("62-010") == "62010"
    assert icp._norm_ssic("62 010") == "62010"
    # None and non-digit strings
    assert icp._norm_ssic(None) is None
    assert icp._norm_ssic("abc") is None


def test_find_ssic_codes_by_terms_empty_returns_empty():
    # Should not hit DB when no usable query terms
    assert icp._find_ssic_codes_by_terms([]) == []
    assert icp._find_ssic_codes_by_terms(["", "   "]) == []


def test_select_acra_by_ssic_codes_short_circuit():
    # Empty set
    assert icp._select_acra_by_ssic_codes(set()) == []
    # Non-digit-only codes normalize to None -> should short-circuit
    assert icp._select_acra_by_ssic_codes({"abc"}) == []


def test_api_icp_by_ssic_returns_shape_and_uses_terms(monkeypatch):
    captured = {}

    def fake_find_ssic(terms):
        # Capture terms after endpoint normalization (strip/empty removal)
        captured["terms"] = terms
        return [
            ("62010", "Computer programming activities", 0.95),
            ("62020", "Information technology consultancy activities", 0.87),
        ]

    def fake_select_acra(codes, limit=1000):
        # Expect codes derived from matched_ssic
        assert set(codes) == {"62010", "62020"}
        return [
            {
                "uen": "201234567A",
                "entity_name": "Acme Tech Pte Ltd",
                "primary_ssic_code": "62010",
                "entity_status_description": "Live",
            },
            {
                "uen": "201998765Z",
                "entity_name": "Beta IT Consulting",
                "primary_ssic_code": "62020",
                "entity_status_description": "Live",
            },
        ]

    monkeypatch.setattr(icp, "_find_ssic_codes_by_terms", fake_find_ssic)
    monkeypatch.setattr(icp, "_select_acra_by_ssic_codes", fake_select_acra)

    client = TestClient(app)
    # Avoid None/int elements to prevent Pydantic 422 on request parsing
    payload = {"terms": ["  software development  ", "it consulting", "", "   "]}
    resp = client.post("/api/icp/by-ssic", json=payload)

    assert resp.status_code == 200
    data = resp.json()

    # Endpoint shape
    assert "matched_ssic" in data
    assert "acra_candidates" in data
    assert isinstance(data["matched_ssic"], list)
    assert isinstance(data["acra_candidates"], list)

    # Normalization of terms (strip empties/whitespace and trim)
    assert captured["terms"] == ["software development", "it consulting"]

    # matched_ssic item shape
    row0 = data["matched_ssic"][0]
    assert set(row0.keys()) == {"code", "title", "score"}

    # acra_candidates minimal shape
    cand0 = data["acra_candidates"][0]
    assert {"uen", "entity_name", "primary_ssic_code"}.issubset(set(cand0.keys()))


def test_api_icp_by_ssic_empty_terms(monkeypatch):
    captured = {}

    def fake_find_ssic(terms):
        captured["terms"] = terms
        return []

    def fake_select_acra(codes, limit=1000):
        # Should not be called with any codes when matched_ssic is empty
        assert not codes
        return []

    monkeypatch.setattr(icp, "_find_ssic_codes_by_terms", fake_find_ssic)
    monkeypatch.setattr(icp, "_select_acra_by_ssic_codes", fake_select_acra)

    client = TestClient(app)

    # No terms provided
    resp = client.post("/api/icp/by-ssic", json={})
    assert resp.status_code == 200
    data = resp.json()
    assert data["matched_ssic"] == []
    assert data["acra_candidates"] == []

    # Ensure endpoint passes an empty list to the agent when no terms provided
    assert captured["terms"] == []
