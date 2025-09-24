import os
from functools import lru_cache

import httpx
import json
import jwt
from fastapi import HTTPException, Request

# Raw envs (may contain whitespace); access via helpers below
_ISSUER_RAW = os.getenv("NEXIUS_ISSUER")
_AUD_RAW = os.getenv("NEXIUS_AUDIENCE")


def _is_truthy(val: str | None) -> bool:
    if val is None:
        return False
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _issuer() -> str:
    if not _ISSUER_RAW or not _ISSUER_RAW.strip():
        raise HTTPException(status_code=500, detail="SSO issuer not configured")
    return _ISSUER_RAW.strip()


def _audience() -> str | None:
    return (_AUD_RAW or "").strip() or None


@lru_cache(maxsize=1)
def _openid_config() -> dict:
    # Keycloak and other OIDC providers expose discovery here
    url = f"{_issuer()}/.well-known/openid-configuration"
    try:
        resp = httpx.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OIDC discovery failed: {e}")


@lru_cache(maxsize=1)
def _jwks() -> dict:
    jwks_uri = _openid_config().get("jwks_uri") or f"{_issuer()}/protocol/openid-connect/certs"
    try:
        resp = httpx.get(jwks_uri, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        # Surface a clean error to the API layer
        raise HTTPException(status_code=500, detail=f"JWKS fetch failed: {e}")


def _public_key_for_token(token: str):
    try:
        header = jwt.get_unverified_header(token)
    except jwt.PyJWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token header: {e}")
    kid = header.get("kid")
    if not kid:
        raise HTTPException(status_code=401, detail="Missing kid in token header")
    keys = (_jwks() or {}).get("keys", [])
    for jwk in keys:
        if jwk.get("kid") == kid:
            try:
                return jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(jwk))
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Invalid JWK: {e}")
    raise HTTPException(status_code=401, detail="No matching JWK for kid")


def verify_jwt(token: str) -> dict:
    key = _public_key_for_token(token)
    aud = _audience()
    try:
        return jwt.decode(
            token,
            key=key,
            algorithms=["RS256"],
            audience=aud,
            issuer=_issuer(),
            options={
                "verify_aud": bool(aud),
                # issuer is verified when issuer param is provided
            },
        )
    except jwt.PyJWTError as e:
        raise HTTPException(status_code=401, detail=str(e))


async def require_auth(request: Request) -> dict:
    """Require authenticated session via JWT.

    Order of precedence:
    1) Cookie `nx_access`
    2) Authorization: Bearer <token>
    """
    token = request.cookies.get("nx_access")
    if not token:
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            token = auth[7:]
    if not token:
        raise HTTPException(status_code=401, detail="Missing credentials")
    claims = verify_jwt(token)
    request.state.tenant_id = claims.get("tenant_id")
    request.state.roles = claims.get("roles", [])
    # Allow explicit X-Tenant-ID header to satisfy tenant requirement when claim is absent
    if not request.state.tenant_id:
        hdr_tid = request.headers.get("x-tenant-id") or request.headers.get("X-Tenant-ID")
        if hdr_tid and str(hdr_tid).strip():
            request.state.tenant_id = str(hdr_tid).strip()
        else:
            raise HTTPException(status_code=403, detail="Missing tenant_id (claim or X-Tenant-ID header)")
    return claims


async def require_identity(request: Request) -> dict:
    """Authenticate the request but do not require a tenant_id claim.

    Useful for endpoints like onboarding where we can resolve or create
    the tenant mapping server-side based on the user identity (email)
    if the SSO token does not include a tenant_id claim.
    """
    token = request.cookies.get("nx_access")
    if not token:
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            token = auth[7:]
    if not token:
        raise HTTPException(status_code=401, detail="Missing credentials")
    claims = verify_jwt(token)
    request.state.tenant_id = claims.get("tenant_id")
    request.state.roles = claims.get("roles", [])
    return claims


async def require_optional_identity(request: Request) -> dict:
    """Return identity from bearer token when present; otherwise allow dev-style headers.

    - Does not require a tenant_id claim.
    - If no Authorization header, checks cookie `nx_access`.
    - Intended for onboarding endpoints to avoid 401 loops when frontend has not
      attached the token yet, while still verifying when a token is present.
    """
    token = request.cookies.get("nx_access")
    if not token:
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            token = auth[7:]
    if token:
        claims = verify_jwt(token)
        request.state.tenant_id = claims.get("tenant_id")
        request.state.roles = claims.get("roles", [])
        return claims
    # No credentials; return 401 to enforce production behavior
    raise HTTPException(status_code=401, detail="Missing credentials")
