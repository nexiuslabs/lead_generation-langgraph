import os
import sys

os.environ.setdefault("OPENAI_API_KEY", "test")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.icp_intake import _derive_negative_icp_flags, build_targeting_pack, _norm_domain


def test_norm_domain_basic():
    assert _norm_domain("https://www.example.com/path?q=1") == "example.com"
    assert _norm_domain("http://EXAMPLE.com") == "example.com"
    assert _norm_domain("example.com") == "example.com"
    assert _norm_domain(None) is None


def test_derive_negative_icp_flags_from_lost_and_anti():
    answers = {
        "lost_or_churned": [
            {"name": "Tiny", "website": "tiny.com", "reason": "budget < $5k"},
            {"name": "LegacySoft", "website": "legacy.com", "reason": "on-prem only, heavy security"},
            {"name": "NoFit", "website": "nofit.com", "reason": "poor fit / bad timing"},
        ],
        "anti_icp": ["solo founders", "government tenders"],
    }
    themes = _derive_negative_icp_flags(answers)
    keys = {t["theme"] for t in themes}
    # Expect at least three distinct themes
    assert len(themes) >= 3
    assert {"budget_too_low", "on_prem_only"}.issubset(keys)


def test_build_targeting_pack_from_ssic_card():
    card = {"id": "ssic:46900", "title": "SSIC 46900", "evidence_count": 2}
    pack = build_targeting_pack(card)
    assert pack["ssic_filters"] == ["46900"]
    assert isinstance(pack.get("technographic_filters"), list)
    assert isinstance(pack.get("pitch"), str)

