-- app/migrations/001_presdr_odoo.sql
-- Idempotent migration to extend Odoo tables for pre-SDR enrichment

-- Extend res_partner for company/contact enrichment fields
ALTER TABLE res_partner
  ADD COLUMN IF NOT EXISTS name varchar,
  ADD COLUMN IF NOT EXISTS complete_name varchar,
  ADD COLUMN IF NOT EXISTS type varchar,
  ADD COLUMN IF NOT EXISTS is_company boolean,
  ADD COLUMN IF NOT EXISTS active boolean,
  ADD COLUMN IF NOT EXISTS commercial_company_name varchar,
  ADD COLUMN IF NOT EXISTS create_date timestamp without time zone,
  ADD COLUMN IF NOT EXISTS x_uen varchar,
  ADD COLUMN IF NOT EXISTS x_industry_norm varchar,
  ADD COLUMN IF NOT EXISTS x_employees_est integer,
  ADD COLUMN IF NOT EXISTS x_revenue_bucket varchar,
  ADD COLUMN IF NOT EXISTS x_incorporation_year smallint,
  ADD COLUMN IF NOT EXISTS x_enrichment_json jsonb,
  ADD COLUMN IF NOT EXISTS x_jobs_count integer,
  ADD COLUMN IF NOT EXISTS x_tech_stack jsonb,
  ADD COLUMN IF NOT EXISTS x_website_domain text;

CREATE UNIQUE INDEX IF NOT EXISTS idx_res_partner_x_uen ON res_partner (x_uen);

-- Extend crm_lead for pre-SDR scoring and provenance
-- Ensure crm_lead core columns referenced by inserts exist
ALTER TABLE crm_lead
  ADD COLUMN IF NOT EXISTS active boolean,
  ADD COLUMN IF NOT EXISTS user_id integer,
  ADD COLUMN IF NOT EXISTS stage_id integer;

ALTER TABLE crm_lead
  ADD COLUMN IF NOT EXISTS x_pre_sdr_score numeric,
  ADD COLUMN IF NOT EXISTS x_pre_sdr_bucket varchar,
  ADD COLUMN IF NOT EXISTS x_pre_sdr_features jsonb,
  ADD COLUMN IF NOT EXISTS x_pre_sdr_rationale text,
  ADD COLUMN IF NOT EXISTS x_source_urls jsonb;
