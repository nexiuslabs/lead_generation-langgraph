# app/odoo_store.py
from typing import Any, Dict, Optional
import asyncpg, json, os
from src.settings import ODOO_POSTGRES_DSN

class OdooStore:
    def __init__(self, dsn: str | None = None):
        self.dsn = dsn or ODOO_POSTGRES_DSN

    async def _acquire(self):
        return await asyncpg.connect(self.dsn)

    async def upsert_company(self, name:str, uen:str|None=None, **fields) -> int:
        conn = await self._acquire()
        try:
            # Try update by UEN; else insert as company
            row = await conn.fetchrow("""
              UPDATE res_partner
                 SET name=$1,
                     x_uen=COALESCE($2,x_uen),
                     x_industry_norm=$3,
                     x_employees_est=$4,
                     x_revenue_bucket=$5,
                     x_incorporation_year=$6,
                     x_website_domain=COALESCE($7,x_website_domain),
                     write_date=now()
               WHERE ($2 IS NOT NULL AND x_uen=$2)
                 AND company_type='company'
               RETURNING id
            """, name, uen, fields.get("industry_norm"), fields.get("employees_est"),
                 fields.get("revenue_bucket"), fields.get("incorporation_year"),
                 fields.get("website_domain"))
            if row: return row["id"]

            row = await conn.fetchrow("""
              INSERT INTO res_partner (name, company_type, x_uen, x_industry_norm,
                                       x_employees_est, x_revenue_bucket, x_incorporation_year, x_website_domain, create_date)
              VALUES ($1,'company',$2,$3,$4,$5,$6,$7, now())
              RETURNING id
            """, name, uen, fields.get("industry_norm"), fields.get("employees_est"),
                 fields.get("revenue_bucket"), fields.get("incorporation_year"), fields.get("website_domain"))
            return row["id"]
        finally:
            await conn.close()

    async def add_contact(self, company_id:int, email:str, full_name:str|None=None) -> Optional[int]:
        if not email: return None
        conn = await self._acquire()
        try:
            # dedupe by (parent_id, lower(email))
            row = await conn.fetchrow("""
              SELECT id FROM res_partner
               WHERE parent_id=$1 AND lower(email)=lower($2) LIMIT 1
            """, company_id, email)
            if row: return row["id"]
            row = await conn.fetchrow("""
              INSERT INTO res_partner (parent_id, company_type, name, email, create_date)
              VALUES ($1, 'person', COALESCE($3, split_part($2,'@',1)), $2, now())
              RETURNING id
            """, company_id, email, full_name)
            return row["id"]
        finally:
            await conn.close()

    async def merge_company_enrichment(self, company_id:int, enrichment:Dict[str,Any]):
        conn = await self._acquire()
        try:
            await conn.execute("""
              UPDATE res_partner
                 SET x_enrichment_json = COALESCE(x_enrichment_json,'{}'::jsonb) || $1::jsonb,
                     x_jobs_count = COALESCE($2, x_jobs_count),
                     x_tech_stack = COALESCE(x_tech_stack,'[]'::jsonb) || to_jsonb(COALESCE($3,'[]'::jsonb)),
                     write_date=now()
               WHERE id=$4 AND company_type='company'
            """, json.dumps(enrichment),
                 enrichment.get("jobs_count"),
                 json.dumps(enrichment.get("tech_stack") or []),
                 company_id)
        finally:
            await conn.close()

    async def create_lead_if_high(self, company_id:int, title:str, score:float, features:Dict[str,Any], rationale:str,
                                  primary_email:str|None, threshold:float=0.66) -> Optional[int]:
        if score < threshold: return None
        conn = await self._acquire()
        try:
            row = await conn.fetchrow("""
              INSERT INTO crm_lead (name, partner_id, type,
                                    x_pre_sdr_score, x_pre_sdr_bucket, x_pre_sdr_features, x_pre_sdr_rationale,
                                    email_from, create_date)
              VALUES ($1,$2,'lead',$3, CASE WHEN $3>=0.66 THEN 'High' WHEN $3>=0.33 THEN 'Medium' ELSE 'Low' END,
                      $4::jsonb, $5, $6, now())
              RETURNING id
            """, title, company_id, score, json.dumps(features), rationale, primary_email)
            return row["id"]
        finally:
            await conn.close()
