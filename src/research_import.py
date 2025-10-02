from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from psycopg2.extras import Json

from src.database import get_conn


MAX_SNAPSHOT_LEN = 20_000  # guardrail: keep markdown snapshot bounded


def _hash_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8", errors="ignore")).hexdigest()


def _normalize_url(u: str) -> str:
    u = (u or "").strip()
    if not u:
        return u
    # best-effort cleanup of \'r.jina.ai/https://\' style wrappers
    if "/https://" in u:
        u = u.split("/https://", 1)[1]
        u = "https://" + u
    if "/http://" in u:
        u = u.split("/http://", 1)[1]
        u = "http://" + u
    return u


def _parse_leads_file(md: str) -> List[Dict[str, Any]]:
    """Very lightweight parser for docs/leads_for_nexius.md style sections.
    Expects sections like:
      ## Company Name
      - Website: https://acme.sg
      - Snapshot: ...
      - Fit Signals: ["peppol", ...]
      - Nexius Wedge: ...
      - Contacts: ...
      - Sources: url1, url2
    Returns a list of dicts with minimal normalized fields.
    """
    out: List[Dict[str, Any]] = []
    if not md:
        return out
    lines = [l.rstrip() for l in md.splitlines()]
    cur: Dict[str, Any] = {}
    for ln in lines:
        if ln.startswith("## "):
            if cur.get("company_hint"):
                out.append(cur)
            cur = {"company_hint": ln[3:].strip()}
        elif ln.lower().startswith("- website:"):
            cur["website"] = _normalize_url(ln.split(":", 1)[1].strip())
        elif ln.lower().startswith("- snapshot:"):
            cur.setdefault("snapshot", "")
            cur["snapshot"] = (cur["snapshot"] + "\n" + ln.split(":", 1)[1].strip()).strip()
        elif ln.lower().startswith("- fit signals:"):
            sig = ln.split(":", 1)[1].strip()
            # tolerate both [\"a\",\"b\"] format and comma-separated
            sig = sig.strip()
            tags: List[str] = []
            if sig.startswith("[") and sig.endswith("]"):
                # strip quotes crudely
                sig = sig.strip("[] ")
                tags = [s.strip().strip("\"'") for s in sig.split(",") if s.strip()]
            else:
                tags = [s.strip() for s in sig.split(",") if s.strip()]
            cur.setdefault("fit_signals", {})
            cur["fit_signals"]["tags"] = tags
        elif ln.lower().startswith("- nexius wedge:"):
            cur.setdefault("fit_signals", {})
            cur["fit_signals"]["wedge"] = ln.split(":", 1)[1].strip()
        elif ln.lower().startswith("- contacts:"):
            cur.setdefault("fit_signals", {})
            cur["fit_signals"]["contacts"] = ln.split(":", 1)[1].strip()
        elif ln.lower().startswith("- sources:"):
            # split on commas and whitespace
            rest = ln.split(":", 1)[1]
            urls = [
                _normalize_url(p.strip())
                for p in rest.replace(" ", "").split(",")
                if p.strip()
            ]
            cur.setdefault("source_urls", [])
            cur["source_urls"].extend(urls)
    if cur.get("company_hint"):
        out.append(cur)
    return out


def _resolve_company_id(tenant_id: int, name: Optional[str], website: Optional[str]) -> Optional[int]:
    name = (name or "").strip() or None
    website = (website or "").strip() or None
    with get_conn() as conn, conn.cursor() as cur:
        # 1) by website_domain
        if website:
            dom = website
            dom = dom.replace("https://", "").replace("http://", "").replace("/", "").strip()
            if dom:
                cur.execute("SELECT company_id FROM companies WHERE website_domain=%s", (dom,))
                row = cur.fetchone()
                if row and row[0] is not None:
                    return int(row[0])
        # 2) by name
        if name:
            cur.execute("SELECT company_id FROM companies WHERE name=%s", (name,))
            row = cur.fetchone()
            if row and row[0] is not None:
                return int(row[0])
        # 3) insert minimal
        if name or website:
            dom = None
            if website:
                dom = website.replace("https://", "").replace("http://", "").replace("/", "").strip()
            cur.execute(
                "INSERT INTO companies(name, website_domain, last_seen) VALUES (%s,%s,NOW()) RETURNING company_id",
                (name, dom),
            )
            row = cur.fetchone()
            if row and row[0] is not None:
                return int(row[0])
    return None


def _write_evidence(tenant_id: int, company_id: int, artifact: Dict[str, Any]) -> None:
    srcs = artifact.get("source_urls") or []
    fit = artifact.get("fit_signals") or {}
    note = (artifact.get("snapshot") or "").strip()[:800]
    with get_conn() as conn, conn.cursor() as cur:
        # research_fit
        try:
            cur.execute(
                "INSERT INTO icp_evidence(tenant_id, company_id, signal_key, value, source) VALUES (%s,%s,%s,%s,'research')",
                (tenant_id, company_id, "research_fit", Json(fit)),
            )
        except Exception:
            pass
        # research_note
        if note:
            try:
                cur.execute(
                    "INSERT INTO icp_evidence(tenant_id, company_id, signal_key, value, source) VALUES (%s,%s,%s,%s,'research')",
                    (tenant_id, company_id, "research_note", Json({"note": note})),
                )
            except Exception:
                pass


def import_docs_for_tenant(tenant_id: int, root: str) -> Dict[str, Any]:
    root_path = Path(root).resolve()
    if not root_path.exists() or not root_path.is_dir():
        return {"files_scanned": 0, "leads_upserted": 0, "errors": [f"root not found: {root}"]}

    files_scanned = 0
    leads_upserted = 0
    errors: List[str] = []

    # Track run row
    run_id: Optional[int] = None
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO research_import_runs(tenant_id, ai_metadata) VALUES (%s,%s) RETURNING id",
            (tenant_id, Json({"root": str(root_path)})),
        )
        r = cur.fetchone()
        run_id = int(r[0]) if r and r[0] is not None else None

    # Parse leads_for_nexius.md (primary) and profiles/* as secondary artifacts
    leads_md: Optional[str] = None
    leads_file = root_path / "leads_for_nexius.md"
    if leads_file.exists():
        leads_md = leads_file.read_text(encoding="utf-8", errors="ignore")
        files_scanned += 1
    artifacts: List[Dict[str, Any]] = []
    if leads_md:
        for item in _parse_leads_file(leads_md):
            item["path"] = str(leads_file)
            artifacts.append(item)

    # Ingest minimal profiles as artifacts too (optional)
    profiles_dir = root_path / "profiles"
    if profiles_dir.exists() and profiles_dir.is_dir():
        for p in profiles_dir.glob("*_profile.md"):
            try:
                md = p.read_text(encoding="utf-8", errors="ignore")
                files_scanned += 1
                # crude extraction of first line as name
                company_hint = p.name.replace("_profile.md", "").replace("-", " ").title()
                srcs: List[str] = []
                for ln in md.splitlines():
                    if ln.lower().startswith("- sources:"):
                        rest = ln.split(":", 1)[1]
                        srcs = [s.strip() for s in rest.replace(" ", "").split(",") if s.strip()]
                        break
                artifacts.append(
                    {
                        "company_hint": company_hint,
                        "website": None,
                        "path": str(p),
                        "snapshot": md[:2000],
                        "source_urls": srcs,
                        "fit_signals": {},
                    }
                )
            except Exception as e:  # noqa: F841
                errors.append(f"profile parse failed: {p.name}")

    # Upsert artifacts and write evidence
    for art in artifacts:
        try:
            company_id = _resolve_company_id(tenant_id, art.get("company_hint"), art.get("website"))
            if not company_id:
                errors.append(f"resolve failed: {art.get('company_hint') or art.get('website') or art.get('path')}")
                continue
            snapshot = (art.get("snapshot") or "").strip()[:MAX_SNAPSHOT_LEN]
            fit = art.get("fit_signals") or {}
            srcs = art.get("source_urls") or []
            ahash = _hash_text(snapshot + "|" + ",".join(srcs))
            with get_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    (
                        "INSERT INTO icp_research_artifacts(tenant_id, company_hint, company_id, path, source_urls, snapshot_md, fit_signals, ai_metadata) "
                        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s)"
                    ),
                    (
                        tenant_id,
                        art.get("company_hint"),
                        company_id,
                        art.get("path"),
                        srcs,
                        snapshot,
                        Json(fit),
                        Json({"provenance": {"hash": ahash, "run_id": run_id}}),
                    ),
                )
            _write_evidence(tenant_id, company_id, art)
            leads_upserted += 1
        except Exception as e:  # noqa: F841
            errors.append(f"artifact upsert failed: {art.get('company_hint')}")

    # Update run row
    if run_id is not None:
        try:
            with get_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    "UPDATE research_import_runs SET files_scanned=%s, leads_upserted=%s, errors=%s WHERE id=%s",
                    (files_scanned, leads_upserted, Json(errors), run_id),
                )
        except Exception:
            pass

    return {"files_scanned": files_scanned, "leads_upserted": leads_upserted, "errors": errors}

