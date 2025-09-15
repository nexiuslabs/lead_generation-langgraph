"""
LangGraph Auth adapter (cookie-based) for local dev and production.

Compatible with langgraph_api >= 0.2.137. Exposes an Auth instance named
`auth` that the LangGraph server will import via langgraph.json.

Primary: CookieAuth using our access cookie (default: nx_access).
Fallbacks: If CookieAuth is unavailable in this version, fall back to
Bearer token auth. As a last resort, install a permissive Auth to avoid
import-time crashes (not recommended for production).
"""
from __future__ import annotations

import importlib
import inspect
import os
from typing import Any


def _candidate_modules() -> list[str]:
    return [
        "langgraph_api.auth",
        "langgraph_api.auth.backends",
        # keep list minimal to avoid import errors; add more if needed
    ]


def _scan_auth_classes() -> tuple[list[type], list[type]]:
    cookies: list[type] = []
    tokens: list[type] = []
    for modname in _candidate_modules():
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        for name, obj in inspect.getmembers(mod):
            if not inspect.isclass(obj):
                continue
            lname = name.lower()
            if "cookie" in lname:
                cookies.append(obj)
            elif any(k in lname for k in ("bearer", "jwt", "token")):
                tokens.append(obj)
    return cookies, tokens


def _try_new(cls: type, **kw: Any):
    try:
        return cls(**kw)
    except Exception:
        return None


def _mk_auth_instance():
    cookie_name = os.getenv("ACCESS_COOKIE_NAME", "nx_access")
    cookies, tokens = _scan_auth_classes()
    # Try cookie-based classes first with common signatures
    for cls in cookies:
        for kw in (
            {"cookie_name": cookie_name},
            {"cookie": cookie_name},
            {"cookie_names": [cookie_name]},
            {"names": [cookie_name]},
        ):
            inst = _try_new(cls, **kw)
            if inst is not None:
                return inst
        inst = _try_new(cls)
        if inst is not None:
            return inst
    # Fall back to bearer/JWT header classes
    for cls in tokens:
        for kw in (
            {"header_name": "Authorization"},
            {},
        ):
            inst = _try_new(cls, **kw)
            if inst is not None:
                return inst
    # Try explicit known names in minimal set
    for modpath, name, kw in [
        ("langgraph_api.auth", "CookieAuth", {"cookie_name": cookie_name}),
        ("langgraph_api.auth", "BearerTokenAuth", {"header_name": "Authorization"}),
    ]:
        try:
            cls = getattr(importlib.import_module(modpath), name)
            inst = _try_new(cls, **kw)
            if inst is not None:
                return inst
        except Exception:
            pass
    raise ImportError(
        "LangGraph Auth setup failed: no compatible Cookie/Bearer auth class found in langgraph_api.auth.*"
    )


# Construct the exported Auth instance used by LangGraph
auth = _mk_auth_instance()
