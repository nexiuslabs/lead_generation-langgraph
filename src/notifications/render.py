from __future__ import annotations

import datetime as _dt
from typing import Tuple, List, Dict, Any
import csv as _csv
from io import StringIO as _StringIO

from src.database import get_conn


def _bucket_counts(rows: List[dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for r in rows:
        b = (r.get("bucket") or "").strip() or "-"
        counts[b] = counts.get(b, 0) + 1
    return counts


def _render_table(rows: List[dict]) -> str:
    cols = [
        ("name", "Name"),
        ("website_domain", "Domain"),
        ("score", "Score"),
        ("bucket", "Bucket"),
        ("primary_email", "Primary Email"),
        ("contact_name", "Contact"),
        ("contact_title", "Title"),
        ("contact_linkedin", "LinkedIn"),
    ]
    th = "".join([f"<th style='text-align:left;padding:6px 8px;'>{label}</th>" for _, label in cols])
    trs = []
    for r in rows:
        tds = []
        for key, _label in cols:
            val = r.get(key)
            if key == "contact_linkedin" and val:
                cell = f"<a href='{val}' target='_blank' rel='noopener'>{val}</a>"
            else:
                cell = ("" if val is None else str(val))
            tds.append(f"<td style='padding:6px 8px;border-top:1px solid #eee;'>{cell}</td>")
        trs.append(f"<tr>{''.join(tds)}</tr>")
    return f"""
    <table cellspacing='0' cellpadding='0' style='border-collapse:collapse;width:100%;font-family:system-ui,Segoe UI,Arial,Sans-Serif;font-size:14px;'>
      <thead><tr>{th}</tr></thead>
      <tbody>
        {''.join(trs)}
      </tbody>
    </table>
    """


def render_summary_html(tenant_id: int, limit: int = 200) -> Tuple[str, str, str]:
    """Return (subject, html, csv_link) for the tenant’s latest scores.

    Uses the same selection as /export/latest_scores.*, scoped by tenant via request.tenant_id GUC.
    """
    # Fetch rows and set tenant context for RLS
    rows: List[dict] = []
    with get_conn() as conn, conn.cursor() as cur:
        try:
            cur.execute("SELECT set_config('request.tenant_id', %s, true)", (str(int(tenant_id)),))
        except Exception:
            # If set_config fails (e.g., permissions), rollback to clear aborted tx
            try:
                conn.rollback()
            except Exception:
                pass
        cur.execute(
            """
            SELECT c.company_id,
                   c.name,
                   c.website_domain,
                   c.industry_norm,
                   c.employees_est,
                   s.score,
                   s.bucket,
                   s.rationale,
                   (SELECT e.email FROM lead_emails e WHERE e.company_id = s.company_id ORDER BY e.left_company NULLS FIRST, e.smtp_confidence DESC NULLS LAST LIMIT 1) AS primary_email,
                   (SELECT c2.full_name FROM contacts c2 WHERE c2.company_id = s.company_id AND c2.email IS NOT NULL LIMIT 1) AS contact_name,
                   (SELECT c2.job_title FROM contacts c2 WHERE c2.company_id = s.company_id AND c2.email IS NOT NULL LIMIT 1) AS contact_title,
                   (SELECT c2.linkedin_profile FROM contacts c2 WHERE c2.company_id = s.company_id AND c2.email IS NOT NULL LIMIT 1) AS contact_linkedin
            FROM companies c
            JOIN lead_scores s ON s.company_id = c.company_id
            WHERE s.tenant_id = %s
            ORDER BY s.score DESC NULLS LAST
            LIMIT %s
            """,
            (int(tenant_id), int(limit)),
        )
        for rec in cur.fetchall() or []:
            rows.append({
                "company_id": rec[0],
                "name": rec[1],
                "website_domain": rec[2],
                "industry_norm": rec[3],
                "employees_est": rec[4],
                "score": float(rec[5]) if rec[5] is not None else None,
                "bucket": rec[6],
                "rationale": rec[7],
                "primary_email": rec[8],
                "contact_name": rec[9],
                "contact_title": rec[10],
                "contact_linkedin": rec[11],
            })

    now = _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    counts = _bucket_counts(rows)
    counts_txt = ", ".join([f"{k}: {v}" for k, v in sorted(counts.items())]) if counts else "none"
    subject = f"Your shortlist is ready — {len(rows)} leads (Tenant {tenant_id})"
    csv_link = "/export/latest_scores.csv?limit=500"

    header = f"""
    <p style='font-family:system-ui,Segoe UI,Arial,Sans-Serif;font-size:14px;'>
      <strong>Generated:</strong> {now}<br/>
      <strong>Tenant:</strong> {tenant_id}<br/>
      <strong>Bucket counts:</strong> {counts_txt}<br/>
      <em>Note: Full CSV is attached to this email.</em>
    </p>
    <hr style='border:none;border-top:1px solid #eee;margin:12px 0;' />
    """

    html = header + _render_table(rows)
    return subject, html, csv_link


def build_csv_bytes(tenant_id: int, limit: int = 500) -> Tuple[bytes, str]:
    """Build CSV bytes for the tenant’s latest scores.

    Returns (csv_bytes, filename)
    """
    # Query the same columns as export_latest_scores_csv
    with get_conn() as conn, conn.cursor() as cur:
        try:
            cur.execute("SELECT set_config('request.tenant_id', %s, true)", (str(int(tenant_id)),))
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
        cur.execute(
            """
            SELECT c.company_id,
                   c.name,
                   c.industry_norm,
                   c.employees_est,
                   s.score,
                   s.bucket,
                   s.rationale,
                   (
                     SELECT e.email
                     FROM lead_emails e
                     WHERE e.company_id = s.company_id
                     ORDER BY e.left_company NULLS FIRST, e.smtp_confidence DESC NULLS LAST
                     LIMIT 1
                   ) AS primary_email,
                   (
                     SELECT c2.full_name FROM contacts c2
                     WHERE c2.company_id = s.company_id AND c2.email IS NOT NULL
                     LIMIT 1
                   ) AS contact_name,
                   (
                     SELECT c2.job_title FROM contacts c2
                     WHERE c2.company_id = s.company_id AND c2.email IS NOT NULL
                     LIMIT 1
                   ) AS contact_title,
                   (
                     SELECT c2.linkedin_profile FROM contacts c2
                     WHERE c2.company_id = s.company_id AND c2.email IS NOT NULL
                     LIMIT 1
                   ) AS contact_linkedin,
                   (
                     SELECT c2.phone_number FROM contacts c2
                     WHERE c2.company_id = s.company_id AND c2.email IS NOT NULL
                     LIMIT 1
                   ) AS contact_phone
            FROM companies c
            JOIN lead_scores s ON s.company_id = c.company_id
            WHERE s.tenant_id = %s
            ORDER BY s.score DESC NULLS LAST
            LIMIT %s
            """,
            (int(tenant_id), int(limit)),
        )
        rows = cur.fetchall() or []
        headers = [
            "company_id","name","industry_norm","employees_est","score","bucket","rationale",
            "primary_email","contact_name","contact_title","contact_linkedin","contact_phone"
        ]
        buf = _StringIO()
        writer = _csv.writer(buf)
        writer.writerow(headers)
        for rec in rows:
            writer.writerow([rec[0], rec[1], rec[2], rec[3], rec[4], rec[5], rec[6], rec[7], rec[8], rec[9], rec[10], rec[11]])
        data = buf.getvalue().encode("utf-8")
    fname = f"shortlist_tenant_{int(tenant_id)}.csv"
    return data, fname
