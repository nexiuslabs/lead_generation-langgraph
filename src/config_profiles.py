import os
import re
from typing import Any, Dict, List, Optional

"""
Lightweight profile/config loader and SG helpers.

This module provides:
- load_profiles(): load YAML config if present; otherwise return sane defaults
- is_singapore_page(text): detect SG presence in page text
- is_valid_fqdn(host): conservative domain hygiene check
- is_denied_host(host, cfg): deny by apex or host suffix
- DENY_PATH regex compiled from config (path indicators like /directory, /expo, etc.)
- score_profile(text, dom, profile_name, cfg): profile-weighted scoring with breakdown
- bucket(score): map numeric score → A/B/C
"""

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


DEFAULT_CFG: Dict[str, Any] = {
    "sg_markers": [r"\bSingapore\b", r"\+65", r"\b\d{6}\b"],
    "deny": {
        "apex": [
            "w3.org",
            "ifrs.org",
            "ilo.org",
            "oecd.org",
            "deloitte.com",
            "grandviewresearch.com",
            "umbrex.com",
            "10times.com",
            "tradefairdates.com",
            "interpack.com",
            "pack-print.de",
            "exhibitorsvoice.com",
            "expotobi.com",
            "cantonfair.net",
        ],
        "host_suffix": ["gov.sg", "edu.sg", "mil", "int"],
        "path_regex": r"(?i)/(standards?|regulations?|policy|association|directory|glossary|wiki|expo|tradefair|event|conference|exhibition|exhibitors)/",
    },
    # Minimal built-in profiles and weights, can be overridden by YAML
    "profile": "sg_employer_buyers",
    "profiles": {
        "sg_employer_buyers": {
            "include_markers": [
                "careers",
                "jobs",
                "people & culture",
                "human resources",
                "employee relations",
                "industrial relations",
            ],
            "deny_host_suffix": ["gov.sg", "edu.sg", "mil", "int"],
            "weights": {
                "employer_presence": 20,
                "sg_compliance_triggers": 25,
                "hr_ir_presence": 20,
                "hiring_intensity": 10,
                "hq_singapore": 10,
                "evidence_completeness": 15,
            },
        },
        "sg_referral_partners": {
            "include_markers": [
                "hr consulting",
                "recruitment",
                "payroll",
                "hris",
                "eor",
                "peo",
                "corporate secretarial",
                "relocation",
                "immigration",
                "bookkeeping",
            ],
            "deny_host_suffix": ["gov.sg", "mil", "int"],
            "weights": {
                "services_match": 30,
                "sg_presence": 20,
                "partner_fit": 30,
                "evidence_completeness": 20,
            },
        },
        "sg_generic_leads": {
            "include_markers": [
                "about",
                "services",
                "contact",
                "clients",
                "hiring",
                "new outlet",
                "expansion",
                "tender",
            ],
            "deny_host_suffix": ["gov.sg", "edu.sg", "mil", "int"],
            "weights": {
                "sg_presence": 30,
                "org_signals": 30,
                "hiring_growth": 20,
                "evidence_completeness": 20,
            },
        },
    },
}


def load_profiles(path: Optional[str] = None) -> Dict[str, Any]:
    p = path or os.getenv("SG_PROFILES_CONFIG") or os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "config", "sg_profiles.yaml"
    )
    if not yaml:
        return DEFAULT_CFG
    try:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                # Merge minimal keys with defaults
                cfg = dict(DEFAULT_CFG)
                if isinstance(data, dict):
                    # Only copy keys we use in this module
                    if isinstance(data.get("sg_markers"), list):
                        cfg["sg_markers"] = [str(x) for x in data["sg_markers"] if str(x).strip()]
                    if isinstance(data.get("deny"), dict):
                        d = data.get("deny") or {}
                        dd = dict(cfg.get("deny") or {})
                        if isinstance(d.get("apex"), list):
                            dd["apex"] = [str(x).lower() for x in d["apex"] if str(x).strip()]
                        if isinstance(d.get("host_suffix"), list):
                            dd["host_suffix"] = [
                                str(x).lower() for x in d["host_suffix"] if str(x).strip()
                            ]
                        if isinstance(d.get("path_regex"), str) and d["path_regex"].strip():
                            dd["path_regex"] = d["path_regex"].strip()
                        cfg["deny"] = dd
                    # Profiles and default selection
                    if isinstance(data.get("profile"), str) and data.get("profile").strip():
                        cfg["profile"] = str(data.get("profile")).strip()
                    if isinstance(data.get("profiles"), dict):
                        # Shallow merge of profiles tree
                        profs = dict(cfg.get("profiles") or {})
                        for k, v in (data.get("profiles") or {}).items():
                            if isinstance(v, dict):
                                profs[str(k)] = v
                        cfg["profiles"] = profs
                return cfg
    except Exception:
        pass
    return DEFAULT_CFG


def is_singapore_page(text: str, cfg: Optional[Dict[str, Any]] = None) -> bool:
    cfg = cfg or DEFAULT_CFG
    try:
        pats = cfg.get("sg_markers") or []
        low = str(text or "")
        for p in pats:
            if re.search(p, low, flags=re.I):
                return True
    except Exception:
        pass
    return False


def is_valid_fqdn(host: str) -> bool:
    try:
        return bool(
            re.match(
                r"^(?!-)[A-Za-z0-9-]{1,63}(?<!-)\.(?:[A-Za-z0-9-]+\.)*[A-Za-z]{2,}$",
                (host or "").strip(),
            )
        )
    except Exception:
        return False


def is_denied_host(host: str, cfg: Optional[Dict[str, Any]] = None) -> bool:
    cfg = cfg or DEFAULT_CFG
    try:
        h = (host or "").strip().lower()
        if not h:
            return True
        deny = cfg.get("deny") or {}
        apex_list = [str(x).strip().lower() for x in (deny.get("apex") or [])]
        if any(h == a or h.endswith("." + a) for a in apex_list):
            return True
        suf_list = [str(x).strip().lower() for x in (deny.get("host_suffix") or [])]
        if any(h.endswith("." + s) or h.endswith(s) for s in suf_list):
            return True
    except Exception:
        return False
    return False


def deny_path_regex(cfg: Optional[Dict[str, Any]] = None):
    cfg = cfg or DEFAULT_CFG
    try:
        pat = (cfg.get("deny") or {}).get("path_regex") or DEFAULT_CFG["deny"]["path_regex"]
        return re.compile(str(pat), re.I)
    except Exception:
        return re.compile(DEFAULT_CFG["deny"]["path_regex"], re.I)


# Convenience compiled regex
DENY_PATH = deny_path_regex(DEFAULT_CFG)


def _normalize_text(text: str) -> str:
    try:
        return " ".join((text or "").split()).lower()
    except Exception:
        return str(text or "").lower()


_COMPLIANCE_TOKENS = [
    "mom",  # Ministry of Manpower
    "tafep",
    "tadm",
    "wfl",  # Workfare
    "wica",
    "wsh",  # Workplace Safety & Health
    "cpf",
    "pdpa",
]
_HRIS_TOKENS = [
    "hris",
    "payroll",
    "leave management",
    "claims",
    "timesheets",
    "workday",
    "successfactors",
    "bamboohr",
    "rippling",
    "deel",
    "gusto",
]


def score_profile(text: str, dom: str, profile_name: Optional[str], cfg: Optional[Dict[str, Any]] = None) -> tuple[int, list[str], Dict[str, int]]:
    """Compute a profile-weighted score using simple textual cues.

    Returns (score, why_chips, breakdown_dict).
    """
    cfg = cfg or DEFAULT_CFG
    profiles = cfg.get("profiles") or {}
    pname = profile_name or cfg.get("profile") or "sg_employer_buyers"
    prof = profiles.get(pname) or {}
    weights = prof.get("weights") or {}
    t = _normalize_text(text)
    why: list[str] = []
    br: Dict[str, int] = {}
    score = 0
    # SG presence markers
    try:
        sg_present = any(re.search(p, text or "", re.I) for p in (cfg.get("sg_markers") or []))
    except Exception:
        sg_present = False
    # Generic signals
    if pname == "sg_employer_buyers":
        # employer presence
        emp = any(m in t for m in ["careers", "jobs", "people & culture", "human resources"])
        if emp:
            w = int(weights.get("employer_presence", 0))
            score += w
            br["employer_presence"] = w
            why.append("employer presence")
        # HR/IR presence
        hr_ir = any(m in t for m in ["human resources", "employee relations", "industrial relations"])
        if hr_ir:
            w = int(weights.get("hr_ir_presence", 0))
            score += w
            br["hr_ir_presence"] = w
            why.append("hr/ir content")
        # hiring intensity (count of terms)
        count = sum(t.count(k) for k in ["hiring", "jobs", "careers"])
        if count:
            w = int(weights.get("hiring_intensity", 0))
            score += w
            br["hiring_intensity"] = w
            why.append(f"hiring x{count}")
        # compliance triggers (from tokens)
        comp = [tok for tok in _COMPLIANCE_TOKENS if tok in t]
        if comp:
            w = int(weights.get("sg_compliance_triggers", 0))
            score += w
            br["sg_compliance_triggers"] = w
            why.append("sg compliance")
        if sg_present:
            w = int(weights.get("hq_singapore", 0))
            score += w
            br["hq_singapore"] = w
            why.append("hq singapore")
        # evidence completeness (length heuristic)
        if len(t) >= 200:
            w = int(weights.get("evidence_completeness", 0))
            score += w
            br["evidence_completeness"] = w
    elif pname == "sg_referral_partners":
        # services match
        inc = prof.get("include_markers") or []
        services = any(m in t for m in [s.lower() for s in inc])
        if services:
            w = int(weights.get("services_match", 0))
            score += w
            br["services_match"] = w
            why.append("services match")
        # partner fit (heuristic: presence of keywords 'partner', 'reseller')
        if any(k in t for k in ["partner", "reseller", "channel"]):
            w = int(weights.get("partner_fit", 0))
            score += w
            br["partner_fit"] = w
            why.append("partner fit")
        if sg_present:
            w = int(weights.get("sg_presence", 0))
            score += w
            br["sg_presence"] = w
            why.append("sg presence")
        if len(t) >= 200:
            w = int(weights.get("evidence_completeness", 0))
            score += w
            br["evidence_completeness"] = w
    else:  # sg_generic_leads
        # SG presence and org signals
        if sg_present:
            w = int(weights.get("sg_presence", 0))
            score += w
            br["sg_presence"] = w
            why.append("sg presence")
        if any(k in t for k in ["about", "services", "contact", "clients"]):
            w = int(weights.get("org_signals", 0))
            score += w
            br["org_signals"] = w
            why.append("org signals")
        count = sum(t.count(k) for k in ["hiring", "jobs", "new outlet", "expansion", "tender"])
        if count:
            w = int(weights.get("hiring_growth", 0))
            score += w
            br["hiring_growth"] = w
            why.append("growth")
        if len(t) >= 200:
            w = int(weights.get("evidence_completeness", 0))
            score += w
            br["evidence_completeness"] = w
    return int(score), why, br


def bucket(score: int | float) -> str:
    try:
        s = float(score)
    except Exception:
        s = 0.0
    return "A" if s >= 70 else ("B" if s >= 50 else "C")

