- Login (dev-bypass enabled, any creds work):
    - curl -i -X POST http://localhost:8000/auth/login -H 'Content-Type: application/json' -d
'{"email":"dev@example.com","password":"anything"}' -c cookies.txt
- Verify session:
    - curl -s http://localhost:8000/whoami -b cookies.txt
- If tenant_id is missing (common in dev), include it when calling protected endpoints:
    - curl -s http://localhost:8000/whoami -b cookies.txt -H 'X-Tenant-ID: 123'
- Refresh cookies:
    - curl -i -X POST http://localhost:8000/auth/refresh -b cookies.txt -c cookies.txt
- Logout:
    - curl -i -X POST http://localhost:8000/auth/logout -b cookies.txt

Notes

- Your .env has DEV_AUTH_BYPASS=true and COOKIE_SECURE=false, so local HTTP cookie auth works even if SSO is unreachable.
- Endpoints using require_auth need tenant_id from the token or an X-Tenant-ID header; otherwise you’ll get 403.
- If logging in from the Next.js app, call /api/backend/auth/login (proxied) so cookies are set for localhost:3000. Ensure
NEXT_PUBLIC_API_BASE=http://localhost:8000.