import logging
from typing import Optional, Dict, Any

import httpx

from src.settings import EMAIL_ENABLED, SENDGRID_API_KEY, SENDGRID_FROM_EMAIL

log = logging.getLogger("notifications")


def _mask_email(e: Optional[str]) -> str:
    try:
        if not e:
            return ""
        local, _, domain = e.partition("@")
        local_mask = (local[:2] + "***") if local else "***"
        dom_parts = domain.split(".") if domain else []
        if len(dom_parts) >= 2:
            dom_mask = dom_parts[0][:1] + "***." + dom_parts[-1]
        else:
            dom_mask = "***"
        return f"{local_mask}@{dom_mask}"
    except Exception:
        return "***redacted***"


async def send_leads_email(
    to: str,
    subject: str,
    html: str,
    *,
    template_id: Optional[str] = None,
    substitutions: Optional[Dict[str, Any]] = None,
    attachment_bytes: Optional[bytes] = None,
    attachment_filename: Optional[str] = None,
    attachment_content_type: str = "text/csv",
) -> Dict[str, Any]:
    """Send an email via SendGrid.

    Returns a dict: { status: sent|failed|skipped_no_config, http_status?, request_id?, error? }
    """
    if not EMAIL_ENABLED or not SENDGRID_API_KEY or not SENDGRID_FROM_EMAIL:
        return {"status": "skipped_no_config"}

    payload: Dict[str, Any] = {
        "personalizations": [
            {
                "to": [{"email": to}],
            }
        ],
        "from": {"email": SENDGRID_FROM_EMAIL},
        "subject": subject,
    }

    if template_id:
        payload["template_id"] = template_id
        if substitutions:
            payload["personalizations"][0]["dynamic_template_data"] = substitutions
        else:
            # still send template without substitutions
            payload["personalizations"][0]["dynamic_template_data"] = {}
    else:
        payload["content"] = [{"type": "text/html", "value": html}]

    # Optional single attachment (CSV)
    if attachment_bytes and attachment_filename:
        try:
            import base64
            b64 = base64.b64encode(attachment_bytes).decode("ascii")
            payload["attachments"] = [
                {
                    "content": b64,
                    "type": attachment_content_type,
                    "filename": attachment_filename,
                    "disposition": "attachment",
                }
            ]
        except Exception:
            # If encoding fails, continue without attachment to avoid blocking the email
            pass

    headers = {"Authorization": f"Bearer {SENDGRID_API_KEY}"}

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post("https://api.sendgrid.com/v3/mail/send", json=payload, headers=headers)
        status = resp.status_code
        if 200 <= status < 300:
            out = {"status": "sent", "http_status": status, "request_id": resp.headers.get("X-Message-Id")}
            try:
                log.info(
                    "email sent to=%s http_status=%s request_id=%s",
                    _mask_email(to),
                    status,
                    out.get("request_id"),
                )
            except Exception:
                pass
            return out
        else:
            out = {"status": "failed", "http_status": status, "error": resp.text[:500]}
            try:
                log.warning(
                    "email failed to=%s http_status=%s error=%s",
                    _mask_email(to),
                    status,
                    out.get("error"),
                )
            except Exception:
                pass
            return out
    except Exception as e:
        try:
            log.exception("email exception to=%s err=%s", _mask_email(to), e)
        except Exception:
            pass
        return {"status": "failed", "error": str(e)}
