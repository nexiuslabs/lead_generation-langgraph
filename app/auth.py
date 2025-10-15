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


def _audiences() -> list[str]:
    raw = (_AUD_RAW or "").strip()
    if not raw:
        return []
    parts = [p.strip() for p in raw.split(",") if p and p.strip()]
    return parts


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
    """Verify a JWT against issuer and (optionally) audience.

    - Supports comma-separated audiences in NEXIUS_AUDIENCE.
    - In local_dev variant, if audience verification fails, fallback to no-aud verification to ease dev.
    """
    key = _public_key_for_token(token)
    audiences = _audiences()
    opts = {"verify_aud": bool(audiences)}
    try:
        return jwt.decode(
            token,
            key=key,
            algorithms=["RS256"],
            audience=audiences if audiences else None,
            issuer=_issuer(),
            options=opts,
        )
    except jwt.InvalidAudienceError as e:
        # Dev fallback when audience mismatch: allow decode without audience if running local_dev
        variant = (os.getenv("LANGSMITH_LANGGRAPH_API_VARIANT", "") or "").strip().lower()
        if variant == "local_dev":
            try:
                return jwt.decode(
                    token,
                    key=key,
                    algorithms=["RS256"],
                    issuer=_issuer(),
                    options={"verify_aud": False},
                )
            except jwt.PyJWTError as e2:
                raise HTTPException(status_code=401, detail=str(e2))
        raise HTTPException(status_code=401, detail=str(e))
    except jwt.PyJWTError as e:
        # Final dev fallback: decode without verifying signature/audience in local_dev to unblock UI login flows
        variant = (os.getenv("LANGSMITH_LANGGRAPH_API_VARIANT", "") or "").strip().lower()
        if variant == "local_dev":
            try:
                return jwt.decode(token, options={"verify_signature": False})
            except jwt.PyJWTError as e2:
                raise HTTPException(status_code=401, detail=str(e2))
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
    """Best-effort identity extraction.

    Behavior:
    - If a JWT is present (cookie or Authorization header), verify and return claims.
    - If no JWT:
      - In `local_dev` or when `DEV_AUTH_BYPASS` is truthy, accept an optional
        `X-Tenant-ID` header and return a minimal dev claim. This unblocks local
        health checks and status polling without requiring full SSO.
      - Otherwise, raise 401 (production default).
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
    # No token present: allow dev bypass with optional X-Tenant-ID
    variant = (os.getenv("LANGSMITH_LANGGRAPH_API_VARIANT", "") or "").strip().lower()
    dev_bypass = _is_truthy(os.getenv("DEV_AUTH_BYPASS")) or (variant == "local_dev")
    if dev_bypass:
        hdr_tid = request.headers.get("x-tenant-id") or request.headers.get("X-Tenant-ID")
        tid_val = hdr_tid.strip() if isinstance(hdr_tid, str) and hdr_tid.strip() else None
        request.state.tenant_id = tid_val
        request.state.roles = []
        return {"sub": "dev-bypass", "tenant_id": tid_val, "roles": []}
    raise HTTPException(status_code=401, detail="Missing credentials")
