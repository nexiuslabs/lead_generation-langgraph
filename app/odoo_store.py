# app/odoo_store.py
import json
import logging
import os
import shutil
import socket
import subprocess
import time
from typing import Any, Dict, Optional

import asyncpg

from src.settings import ODOO_POSTGRES_DSN, ODOO_EXPORT_SET_DEFAULTS
from src.database import get_conn

logger = logging.getLogger(__name__)


class OdooStore:
    def __init__(self, tenant_id: int | None = None, dsn: str | None = None):
        # Prefer explicit DSN if provided
        resolved_dsn = dsn
        if not resolved_dsn and tenant_id is not None:
            try:
                # Resolve per-tenant DSN from app DB mapping (odoo_connections) using pooled connection
                with get_conn() as c, c.cursor() as cur:
                    cur.execute(
                        "SELECT db_name FROM odoo_connections WHERE tenant_id=%s AND active",
                        (tenant_id,),
                    )
                    row = cur.fetchone()
                    if row and row[0]:
                        base_tpl = os.getenv("ODOO_BASE_DSN_TEMPLATE", "")
                        if base_tpl:
                            resolved_dsn = base_tpl.format(db_name=row[0])
                        else:
                            # Fallback: allow storing full DSN in db_name column
                            resolved_dsn = row[0]
            except Exception as e:
                logger.exception("Per-tenant Odoo DSN resolution failed")
        self.dsn = resolved_dsn or ODOO_POSTGRES_DSN
        if not self.dsn:
            raise ValueError(
                "Odoo DSN not provided; set ODOO_POSTGRES_DSN or map via odoo_connections."
            )
        # Log target host:port and db for observability (mask user/pass)
        try:
            from urllib.parse import urlparse
            u = urlparse(self.dsn)
            host = u.hostname or "?"
            port = u.port or 5432
            db = (u.path or "/").lstrip("/") or "?"
            logger.info("odoo:target host=%s port=%s db=%s", host, port, db)
        except Exception:
            pass
        # Lazy SSH tunnel (password or key) via env if configured
        self._ensure_tunnel_once()

    # --- Optional SSH tunnel management (no-op if env not provided) ---
    _tunnel_opened: bool = False
    _tunnel_proc: Optional[subprocess.Popen] = None

    @staticmethod
    def _port_open(host: str, port: int, timeout: float = 0.3) -> bool:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except Exception:
            return False

    def _ensure_tunnel_once(self) -> None:
        if OdooStore._tunnel_opened:
            return
        ssh_host = os.getenv("SSH_HOST")
        ssh_port = int(os.getenv("SSH_PORT", "22"))
        ssh_user = os.getenv("SSH_USER")
        ssh_password = os.getenv("SSH_PASSWORD")
        db_host_in_droplet = os.getenv("DB_HOST_IN_DROPLET")
        db_port = int(os.getenv("DB_PORT", "5432"))
        local_port = int(os.getenv("LOCAL_PORT", "25060"))

        # If local port is already open, assume a user-managed tunnel
        if self._port_open("127.0.0.1", local_port) or self._port_open("::1", local_port):
            OdooStore._tunnel_opened = True
            return

        if not (ssh_host and ssh_user and db_host_in_droplet):
            # No SSH configuration provided; skip silently.
            OdooStore._tunnel_opened = True
            return

        # Build ssh command; prefer password auth via sshpass when provided
        if ssh_password:
            if shutil.which("sshpass") is None:
                logger.error(
                    "sshpass not found but SSH_PASSWORD is set; cannot auto-open tunnel."
                )
                # Do not mark as opened; leave it false so future attempts can retry
                OdooStore._tunnel_opened = False
                return
            cmd = [
                "sshpass",
                "-p",
                ssh_password,
                "ssh",
                "-4",
                "-fN",
                "-L",
                f"127.0.0.1:{local_port}:{db_host_in_droplet}:{db_port}",
                f"{ssh_user}@{ssh_host}",
                "-p",
                str(ssh_port),
                "-o",
                "ExitOnForwardFailure=yes",
                "-o",
                "StrictHostKeyChecking=no",
            ]
        else:
            cmd = [
                "ssh",
                "-4",
                "-fN",
                "-L",
                f"127.0.0.1:{local_port}:{db_host_in_droplet}:{db_port}",
                f"{ssh_user}@{ssh_host}",
                "-p",
                str(ssh_port),
                "-o",
                "ExitOnForwardFailure=yes",
                "-o",
                "StrictHostKeyChecking=no",
            ]

        try:
            logger.info(
                "Opening SSH tunnel local 127.0.0.1:%s -> %s:%s via %s@%s",
                local_port,
                db_host_in_droplet,
                db_port,
                ssh_user,
                ssh_host,
            )
            OdooStore._tunnel_proc = subprocess.Popen(cmd)
            # Wait briefly for the local port to become available
            for _ in range(20):  # ~5s total
                if self._port_open("127.0.0.1", local_port) or self._port_open("::1", local_port):
                    OdooStore._tunnel_opened = True
                    break
                time.sleep(0.25)
            else:
                # Port did not open in time; check if process already exited
                rc = OdooStore._tunnel_proc.poll()
                if rc is not None:
                    logger.error("SSH tunnel process exited early with code %s; port %s not open", rc, local_port)
                else:
                    logger.error("SSH tunnel did not open port %s within timeout", local_port)
                OdooStore._tunnel_opened = False
        except Exception as exc:
            logger.exception("SSH tunnel start failed")
            OdooStore._tunnel_opened = False

    async def _acquire(self):
        # Re-check/ensure tunnel in case previous attempt failed or was terminated
        self._ensure_tunnel_once()
        # Best-effort visibility if local port is not open
        try:
            from urllib.parse import urlparse
            u = urlparse(self.dsn)
            host = u.hostname or "127.0.0.1"
            port = u.port or 5432
            if not self._port_open(host, port):
                logger.warning("Odoo DSN target %s:%s not reachable before connect()", host, port)
        except Exception:
            pass
        return await asyncpg.connect(self.dsn)

    async def upsert_company(self, name: str, uen: str | None = None, **fields) -> int:

        logger.info("Upserting company name=%s uen=%s", name, uen)
        conn = await self._acquire()
        try:
            try:
                row = await conn.fetchrow(
                    """
                  UPDATE res_partner
                     SET name=$1,
                         complete_name=$1,
                         x_uen=COALESCE($2::varchar,x_uen),
                         x_industry_norm=$3,
                         x_employees_est=$4,
                         x_revenue_bucket=$5,
                         x_incorporation_year=$6,
                         x_website_domain=COALESCE($7,x_website_domain),
                         write_date=now()
                   WHERE ($2::varchar IS NOT NULL AND x_uen=$2::varchar)
                     AND is_company = TRUE
                   RETURNING id
                """,
                    name,
                    uen,
                    fields.get("industry_norm"),
                    fields.get("employees_est"),
                    fields.get("revenue_bucket"),
                    fields.get("incorporation_year"),
                    fields.get("website_domain"),
                )
            except Exception:
                # Fallback for templates lacking custom x_* columns
                row = await conn.fetchrow(
                    """
                  UPDATE res_partner
                     SET name=$1,
                         complete_name=$1,
                         write_date=now()
                   WHERE is_company = TRUE AND lower(name)=lower($1)
                   RETURNING id
                """,
                    name,
                )
            if row:
                logger.info(
                    "Updated Odoo company id=%s name=%s uen=%s", row["id"], name, uen
                )
                return row["id"]

            try:
                row = await conn.fetchrow(
                    """
                  INSERT INTO res_partner (name, complete_name, type, is_company, active, commercial_company_name, x_uen, x_industry_norm,
                                           x_employees_est, x_revenue_bucket, x_incorporation_year, x_website_domain, create_date)
                  VALUES ($1, $1, 'contact', TRUE, TRUE, $2, $3, $4, $5, $6, $7, $8, now())
                  RETURNING id
                """,
                    name,
                    name,
                    uen,
                    fields.get("industry_norm"),
                    fields.get("employees_est"),
                    fields.get("revenue_bucket"),
                    fields.get("incorporation_year"),
                    fields.get("website_domain"),
                )
            except Exception as _insert_exc:
                try:
                    row = await conn.fetchrow(
                        """
                      INSERT INTO res_partner (name, complete_name, type, is_company, active, commercial_company_name, create_date)
                      VALUES ($1, $1, 'contact', TRUE, TRUE, $1, now())
                      RETURNING id
                    """,
                        name,
                    )
                except Exception as _minimal_exc:
                    # As a last resort on templates with strict NOT NULLs (e.g., autopost_bills), set safe defaults
                    if ODOO_EXPORT_SET_DEFAULTS:
                        try:
                            row = await conn.fetchrow(
                                """
                              INSERT INTO res_partner (name, complete_name, type, is_company, active, commercial_company_name, create_date, autopost_bills)
                              VALUES ($1, $1, 'contact', TRUE, TRUE, $1, now(), FALSE)
                              RETURNING id
                            """,
                                name,
                            )
                        except Exception:
                            # Give up; re-raise minimal exception
                            raise _minimal_exc
                    else:
                        raise _minimal_exc
            logger.info(
                "Inserted Odoo company id=%s name=%s uen=%s", row["id"], name, uen
            )
            return row["id"]

        finally:
            await conn.close()

    async def add_contact(
        self, company_id: int, email: str, full_name: str | None = None
    ) -> Optional[int]:
        if not email:
            logger.info(
                "skipping contact without email", extra={"partner_id": company_id}
            )
            return None

        logger.info("Adding contact email=%s company_id=%s", email, company_id)
        conn = await self._acquire()
        try:
            row = await conn.fetchrow(
                """
              SELECT id FROM res_partner
               WHERE parent_id=$1 AND lower(email)=lower($2) LIMIT 1
            """,
                company_id,
                email,
            )
            if row:
                logger.info(
                    "Contact exists id=%s company_id=%s email=%s",
                    row["id"],
                    company_id,
                    email,
                )
                return row["id"]
            row = await conn.fetchrow(
                """
              INSERT INTO res_partner (parent_id, type, is_company, active, name, complete_name, email, create_date)
              VALUES (
                $1,
                'contact',
                FALSE,
                TRUE,
                COALESCE($3, split_part($2,'@',1)),
                COALESCE($3, split_part($2,'@',1)),
                $2,
                now()
              )
              RETURNING id
            """,
                company_id,
                email,
                full_name,
            )
            logger.info(
                "Inserted contact id=%s company_id=%s email=%s",
                row["id"],
                company_id,
                email,
            )
            return row["id"]

        finally:
            await conn.close()

    async def connectivity_smoke_test(self) -> None:
        conn = await self._acquire()
        try:
            # Basic connectivity check
            await conn.execute("SELECT 1")
            # Optionally check res_partner exists
            try:
                await conn.fetchrow(
                    "SELECT 1 FROM information_schema.tables WHERE table_name='res_partner'"
                )
            except Exception:
                pass
        finally:
            await conn.close()

    async def seed_baseline_entities(self, tenant_id: int, email: str | None = None) -> None:
        """Create minimal baseline entities in Odoo for a new tenant context.

        - Create a company (res_partner, is_company=TRUE) if a placeholder does not exist
        - Optionally add a contact row with the user's email under that company
        """
        # Use a deterministic tenant-scoped company name
        company_name = f"Tenant {tenant_id} Company"
        cid = await self.upsert_company(name=company_name, uen=None)
        if email:
            try:
                await self.add_contact(company_id=cid, email=email, full_name=None)
            except Exception:
                # Best-effort; do not fail onboarding if contact insert fails
                pass

    async def merge_company_enrichment(
        self, company_id: int, enrichment: Dict[str, Any]
    ):

        logger.info("Merging enrichment for company_id=%s", company_id)
        conn = await self._acquire()
        try:
            await conn.execute(
                """
              UPDATE res_partner
                 SET x_enrichment_json = COALESCE(x_enrichment_json,'{}'::jsonb) || $1::jsonb,
                     x_jobs_count = COALESCE($2, x_jobs_count),
                     x_tech_stack = COALESCE(x_tech_stack,'[]'::jsonb) || to_jsonb(COALESCE($3,'[]'::jsonb)),
                     write_date=now()
               WHERE id=$4 AND is_company = TRUE
            """,
                json.dumps(enrichment),
                enrichment.get("jobs_count"),
                json.dumps(enrichment.get("tech_stack") or []),
                company_id,
            )
            logger.info("Merged enrichment for company_id=%s", company_id)

        finally:
            await conn.close()

    async def create_lead_if_high(
        self,
        company_id: int,
        title: str,
        score: float,
        features: Dict[str, Any],
        rationale: str,
        primary_email: str | None,
        threshold: float = 0,
    ) -> Optional[int]:
        if score < threshold:
            logger.info(

                "Skipping lead creation company_id=%s score=%.2f < %.2f",
                company_id,
                score,
                threshold,

            )
            return None
        logger.info(
            "creating lead",
            extra={"partner_id": company_id, "score": score},
        )
        conn = await self._acquire()
        try:

            row = await conn.fetchrow(
                """
              INSERT INTO crm_lead (
                name, partner_id, type, active, user_id, stage_id,
                x_pre_sdr_score, x_pre_sdr_bucket, x_pre_sdr_features, x_pre_sdr_rationale,
                email_from, create_date
              )
              VALUES (
                $1,
                $2,
                'opportunity',
                TRUE,
                2,
                1,
                $3,
                CASE WHEN $3>=0.66 THEN 'High' WHEN $3>=0.33 THEN 'Medium' ELSE 'Low' END,
                $4::jsonb,
                $5,
                $6,
                now()
              )
              RETURNING id
            """,
                title,
                company_id,
                score,
                json.dumps(features),
                rationale,
                primary_email,
            )
            logger.info(
                "Created lead id=%s company_id=%s score=%.2f",
                row["id"],
                company_id,
                score,
            )
            return row["id"]

        finally:
            await conn.close()
