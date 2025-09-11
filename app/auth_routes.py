from fastapi import APIRouter, Response, Request, HTTPException
import os
import time
import httpx

from app.auth import verify_jwt

router = APIRouter(prefix="/auth", tags=["auth"]) 


ACCESS_COOKIE = os.getenv("ACCESS_COOKIE_NAME", "nx_access")
REFRESH_COOKIE = os.getenv("REFRESH_COOKIE_NAME", "nx_refresh")


def _cookie_secure() -> bool:
    val = (os.getenv("COOKIE_SECURE", "true") or "").strip().lower()
    return val in ("1", "true", "yes", "on")


def _token_url() -> str:
    token_url = os.getenv("NEXIUS_TOKEN_URL")
    if token_url:
        return token_url
    issuer = (os.getenv("NEXIUS_ISSUER") or "").strip()
    if not issuer:
        raise HTTPException(status_code=500, detail="SSO not configured (missing NEXIUS_ISSUER/NEXIUS_TOKEN_URL)")
    return issuer.rstrip("/") + "/protocol/openid-connect/token"


CLIENT_ID = os.getenv("NEXIUS_CLIENT_ID")
CLIENT_SECRET = os.getenv("NEXIUS_CLIENT_SECRET")


def _set_session(resp: Response, access: str, refresh: str | None) -> None:
    # Derive max-age from token exp to avoid stale cookies
    max_age = 15 * 60
    try:
        import jwt as pyjwt

        claims = pyjwt.decode(access, options={"verify_signature": False})
        exp = int(claims.get("exp", 0))
        if exp:
            max_age = max(60, exp - int(time.time()))
    except Exception:
        pass
    resp.set_cookie(
        ACCESS_COOKIE,
        access,
        httponly=True,
        secure=_cookie_secure(),
        samesite="lax",
        max_age=max_age,
        path="/",
    )
    if refresh:
        resp.set_cookie(
            REFRESH_COOKIE,
            refresh,
            httponly=True,
            secure=_cookie_secure(),
            samesite="lax",
            max_age=30 * 24 * 3600,
            path="/",
        )


async def _direct_grant(email: str, password: str, otp: str | None = None) -> dict:
    if not CLIENT_ID:
        raise HTTPException(status_code=500, detail="SSO client not configured (NEXIUS_CLIENT_ID)")
    data = {
        "grant_type": "password",
        "username": email,
        "password": password,
        "client_id": CLIENT_ID,
    }
    # Ensure ID token is returned (so audience matches the OIDC client)
    data["scope"] = "openid email profile"
    if CLIENT_SECRET:
        data["client_secret"] = CLIENT_SECRET
    if otp:
        data["totp"] = otp
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(_token_url(), data=data)
        if r.status_code != 200:
            # Log response body for diagnosis
            try:
                err = r.text
            except Exception:
                err = f"status={r.status_code}"
            raise HTTPException(status_code=401, detail=f"Invalid credentials: {err}")
        return r.json()


@router.post("/login")
async def login(body: dict, response: Response):
    email = (body.get("email") or "").strip()
    password = body.get("password") or ""
    otp = body.get("otp")
    if not email or not password:
        raise HTTPException(status_code=400, detail="email and password required")
    tok = await _direct_grant(email, password, otp)
    access = tok.get("id_token") or tok.get("access_token")
    refresh = tok.get("refresh_token")
    if not access:
        raise HTTPException(status_code=500, detail="No token in SSO response")
    claims = verify_jwt(access)
    _set_session(response, access, refresh)
    return {"tenant_id": claims.get("tenant_id"), "roles": claims.get("roles", []), "email": claims.get("email")}


@router.post("/refresh")
async def refresh(request: Request, response: Response):
    rt = request.cookies.get(REFRESH_COOKIE)
    if not rt:
        raise HTTPException(status_code=401, detail="No refresh cookie")
    data = {
        "grant_type": "refresh_token",
        "refresh_token": rt,
        "client_id": CLIENT_ID,
    }
    if CLIENT_SECRET:
        data["client_secret"] = CLIENT_SECRET
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(_token_url(), data=data)
        if r.status_code != 200:
            raise HTTPException(status_code=401, detail="Refresh failed")
        tok = r.json()
        access = tok.get("id_token") or tok.get("access_token")
        refresh_token = tok.get("refresh_token")
        if not access:
            raise HTTPException(status_code=500, detail="No token in refresh response")
        _set_session(response, access, refresh_token)
    return {"ok": True}


@router.post("/logout")
async def logout(response: Response):
    response.delete_cookie(ACCESS_COOKIE, path="/")
    response.delete_cookie(REFRESH_COOKIE, path="/")
    return Response(status_code=204)


# --- Optional: Admin-backed Registration via Keycloak Admin API ---
def _issuer_parts():
    issuer = (os.getenv("NEXIUS_ISSUER") or "").rstrip("/")
    if not issuer:
        raise HTTPException(status_code=500, detail="Missing NEXIUS_ISSUER for admin API")
    # issuer example: https://host/realms/<realm>
    parts = issuer.split("/realms/")
    if len(parts) != 2:
        raise HTTPException(status_code=500, detail="Unexpected issuer format; expected .../realms/<realm>")
    base, realm = parts[0], parts[1]
    return base, realm


async def _admin_token() -> str:
    admin_cid = os.getenv("NEXIUS_ADMIN_CLIENT_ID")
    admin_secret = os.getenv("NEXIUS_ADMIN_CLIENT_SECRET")
    if not admin_cid or not admin_secret:
        raise HTTPException(status_code=501, detail="Admin registration not configured")
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(
            _token_url(),
            data={
                "grant_type": "client_credentials",
                "client_id": admin_cid,
                "client_secret": admin_secret,
            },
        )
        if r.status_code != 200:
            raise HTTPException(status_code=500, detail="Admin token request failed")
        return r.json().get("access_token")


@router.post("/register")
async def register(body: dict, response: Response):
    email = (body.get("email") or "").strip()
    password = body.get("password") or ""
    first = (body.get("first_name") or "").strip() or None
    last = (body.get("last_name") or "").strip() or None
    email_verified = bool(body.get("email_verified") or False)
    if not email or not password:
        raise HTTPException(status_code=400, detail="email and password required")
    base, realm = _issuer_parts()
    admin_base = base + "/admin/realms/" + realm
    token = await _admin_token()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    # Create user
    payload = {
        "username": email,
        "email": email,
        "firstName": first,
        "lastName": last,
        "enabled": True,
        # Default to verified to avoid required actions blocking Direct Grant
        "emailVerified": True if email_verified is None else email_verified,
    }
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(f"{admin_base}/users", headers=headers, json=payload)
        if r.status_code not in (201, 204):
            # If already exists, continue to set password and login
            if r.status_code != 409:
                raise HTTPException(status_code=500, detail=f"User create failed: {r.status_code}")
        user_id = None
        if r.status_code == 201:
            loc = r.headers.get("Location", "")
            if "/users/" in loc:
                user_id = loc.rsplit("/", 1)[-1]
        if not user_id:
            # lookup by username
            q = await client.get(f"{admin_base}/users", headers=headers, params={"username": email})
            if q.status_code != 200 or not q.json():
                raise HTTPException(status_code=500, detail="Cannot resolve created user id")
            user_id = q.json()[0]["id"]
        # Set password
        rp = await client.put(
            f"{admin_base}/users/{user_id}/reset-password",
            headers=headers,
            json={"type": "password", "value": password, "temporary": False},
        )
        if rp.status_code not in (204, 201):
            raise HTTPException(status_code=500, detail="Failed to set password")
    # Log in the new user and set cookies
    tok = await _direct_grant(email, password, None)
    access = tok.get("id_token") or tok.get("access_token")
    refresh = tok.get("refresh_token")
    if not access:
        raise HTTPException(status_code=500, detail="No token after register")
    claims = verify_jwt(access)
    _set_session(response, access, refresh)
    return {"tenant_id": claims.get("tenant_id"), "roles": claims.get("roles", []), "email": claims.get("email")}


@router.post("/exchange")
async def exchange_token(body: dict, response: Response):
    """Exchange an ID token (from SSO) for a server session cookie.

    Body: { id_token: string }
    Verifies the JWT and sets the HttpOnly access cookie. No refresh cookie is set.
    """
    token = (body or {}).get("id_token") or (body or {}).get("token")
    if not token or not isinstance(token, str):
        raise HTTPException(status_code=400, detail="id_token required")
    # Verify and set cookie
    claims = verify_jwt(token)
    _set_session(response, token, None)
    return {"ok": True, "email": claims.get("email"), "tenant_id": claims.get("tenant_id")}
