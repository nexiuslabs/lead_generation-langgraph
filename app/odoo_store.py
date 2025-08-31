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

from src.settings import ODOO_POSTGRES_DSN

logger = logging.getLogger(__name__)


class OdooStore:
    def __init__(self, dsn: str | None = None):
        self.dsn = dsn or ODOO_POSTGRES_DSN
        if not self.dsn:
            raise ValueError(
                "Odoo DSN not provided; set ODOO_POSTGRES_DSN or pass dsn."
            )
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
                logger.warning(
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
                    logger.warning("SSH tunnel process exited early with code %s; port %s not open", rc, local_port)
                else:
                    logger.warning("SSH tunnel did not open port %s within timeout", local_port)
                OdooStore._tunnel_opened = False
        except Exception as exc:
            logger.warning("SSH tunnel start failed: %s", exc)
            OdooStore._tunnel_opened = False

    async def _acquire(self):
        # Re-check/ensure tunnel in case previous attempt failed or was terminated
        self._ensure_tunnel_once()
        return await asyncpg.connect(self.dsn)

    async def upsert_company(self, name: str, uen: str | None = None, **fields) -> int:

        logger.info("Upserting company name=%s uen=%s", name, uen)
        conn = await self._acquire()
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
            if row:
                logger.info(
                    "Updated Odoo company id=%s name=%s uen=%s", row["id"], name, uen
                )
                return row["id"]

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
