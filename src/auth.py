"""
Keycloak-auth plugin for LangGraph server.

Validates Keycloak JWTs via JWKS and extracts tenant context from claims.
If tenant is absent in the token, allows a FastAPI/gateway-supplied fallback
via thread metadata or run configurable in local/dev.
"""
from __future__ import annotations

import os
from functools import lru_cache
import httpx
import jwt

# Resolve Auth class from LangGraph API, or fall back to a minimal stub in dev
AuthClass = None
try:  # Preferred import path (may vary by version)
    from langgraph_api.auth import Auth as _Auth  # type: ignore
    AuthClass = _Auth
except Exception:  # pragma: no cover
    try:
        from langgraph_api.auth.types import Auth as _Auth  # type: ignore
        AuthClass = _Auth
    except Exception:
        AuthClass = None  # type: ignore


KEYCLOAK_URL = (os.getenv("KEYCLOAK_URL") or "").rstrip("/")
CLIENT_ID = os.getenv("KEYCLOAK_CLIENT_ID") or ""
JWKS_URL = f"{KEYCLOAK_URL}/protocol/openid-connect/certs" if KEYCLOAK_URL else ""


def _truthy(v: str | None) -> bool:
    return (v or "").strip().lower() in {"1", "true", "yes", "on"}


class _StubHTTPException(Exception):
    def __init__(self, status_code: int = 401, detail: str | None = None):
        super().__init__(detail or "HTTPException")
        self.status_code = status_code
        self.detail = detail or ""


class _StubAuth:
    class exceptions:  # type: ignore[override]
        HTTPException = _StubHTTPException

    def authenticate(self, fn):  # pass-through decorator
        return fn

    def on(self, fn):  # pass-through decorator
        return fn


if AuthClass is not None:
    auth = AuthClass()

    @lru_cache(maxsize=1)
    def _get_jwks_cached() -> dict:
        if not JWKS_URL:
            raise RuntimeError("KEYCLOAK_URL not configured")
        resp = httpx.get(JWKS_URL, timeout=10.0)
        resp.raise_for_status()
        return resp.json()

    def _decode_token(token: str) -> dict:
        jwks = _get_jwks_cached()
        # Prefer RSA keys; in most setups Keycloak uses a single key
        key = next((k for k in jwks.get("keys", []) if k.get("kty") == "RSA"), None)
        if not key:
            raise jwt.InvalidTokenError("No RSA key in JWKS")
        public_key = jwt.algorithms.RSAAlgorithm.from_jwk(key)
        return jwt.decode(
            token,
            public_key,
            algorithms=["RS256"],
            audience=CLIENT_ID or None,
            options={"verify_aud": bool(CLIENT_ID)},
        )

    @auth.authenticate
    async def authenticate(authorization: str | None):
        if not authorization:
            # Dev/local: allow missing bearer and rely on gateway-supplied tenant
            if _truthy(os.getenv("DEV_AUTH_BYPASS")):
                return {
                    "identity": "dev-bypass",
                    "email": None,
                    "username": None,
                    "tenant_id": None,
                    "is_authenticated": True,
                    "claims": {},
                }
            raise auth.exceptions.HTTPException(status_code=401, detail="Missing Authorization")
        try:
            scheme, token = authorization.split(" ", 1)
        except ValueError:
            raise auth.exceptions.HTTPException(status_code=401, detail="Malformed Authorization header")
        if scheme.lower() != "bearer" or not token:
            raise auth.exceptions.HTTPException(status_code=401, detail="Invalid auth scheme")
        try:
            claims = _decode_token(token)
        except jwt.ExpiredSignatureError:
            raise auth.exceptions.HTTPException(status_code=401, detail="Token expired")
        except jwt.PyJWTError as e:  # pragma: no cover - reported cleanly to client
            if _truthy(os.getenv("DEV_AUTH_BYPASS")):
                # Accept invalid/missing tokens in dev; rely on proxy to provide tenant
                return {
                    "identity": "dev-bypass",
                    "email": None,
                    "username": None,
                    "tenant_id": None,
                    "is_authenticated": True,
                    "claims": {},
                }
            raise auth.exceptions.HTTPException(status_code=401, detail=f"Invalid token: {e}")

        # Extract tenant from claim or realm role (tenant-<id>)
        tenant_id = claims.get("tenant_id")
        if not tenant_id:
            roles = (claims.get("realm_access") or {}).get("roles", [])
            if isinstance(roles, (list, tuple)):
                for r in roles:
                    if isinstance(r, str) and r.startswith("tenant-"):
                        tenant_id = r.replace("tenant-", "", 1)
                        break

        # Do not hard-fail here; allow gateway to supply tenant via metadata/configurable in dev
        return {
            "identity": claims.get("sub"),
            "email": claims.get("email"),
            "username": claims.get("preferred_username"),
            "tenant_id": str(tenant_id) if tenant_id is not None else None,
            "is_authenticated": True,
            "claims": claims,
        }

    @auth.on
    async def attach_tenant_metadata(ctx: "Auth.types.AuthContext", value: dict):  # type: ignore[name-defined]
        """Ensure threads/objects are tagged with tenant and owner.

        Accepts tenant from:
        - JWT (ctx.user.tenant_id)
        - value.metadata.tenant_id (gateway-supplied)
        In strict environments, require tenant; in local/dev allow missing.
        """
        tid = (ctx.user or {}).get("tenant_id")
        owner = (ctx.user or {}).get("identity")
        if not tid:
            # Gateway (FastAPI proxy) may provide tenant via metadata/configurable
            tid = (value.get("metadata") or {}).get("tenant_id")
        if not tid and not _truthy(os.getenv("DEV_AUTH_BYPASS")):
            raise auth.exceptions.HTTPException(status_code=401, detail="No tenant in auth context")
        if tid:
            md = value.setdefault("metadata", {})
            md.setdefault("tenant_id", str(tid))
            if owner:
                md.setdefault("owner", owner)
            return {"tenant_id": str(tid)}
        # Dev bypass: return empty filters
        return {}
else:
    # Fallback placeholder so import path is valid even if API lacks Auth
    auth = _StubAuth()  # type: ignore
