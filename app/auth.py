import os
from functools import lru_cache

import httpx
import jwt
from fastapi import HTTPException, Request

ISSUER = os.getenv("NEXIUS_ISSUER")
AUD = os.getenv("NEXIUS_AUDIENCE")


@lru_cache(maxsize=1)
def _jwks():
    if not ISSUER:
        raise HTTPException(status_code=500, detail="SSO issuer not configured")
    url = f"{ISSUER}/.well-known/jwks.json"
    try:
        return httpx.get(url, timeout=5).json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"JWKS fetch failed: {e}")


def verify_jwt(token: str) -> dict:
    try:
        return jwt.decode(
            token,
            key=_jwks(),
            algorithms=["RS256"],
            audience=AUD,
            options={"verify_aud": bool(AUD)},
        )
    except jwt.PyJWTError as e:
        raise HTTPException(status_code=401, detail=str(e))


async def require_auth(request: Request) -> dict:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    claims = verify_jwt(auth[7:])
    request.state.tenant_id = claims.get("tenant_id")
    request.state.roles = claims.get("roles", [])
    if not request.state.tenant_id:
        raise HTTPException(status_code=403, detail="Missing tenant_id claim")
    return claims
