-- app/migrations/021_prd_opt_companies.sql
-- PRD-Opt: add SG-specific and hygiene columns to companies

ALTER TABLE companies
  ADD COLUMN IF NOT EXISTS uen TEXT,
  ADD COLUMN IF NOT EXISTS uen_confidence NUMERIC,
  ADD COLUMN IF NOT EXISTS hq_city TEXT,
  ADD COLUMN IF NOT EXISTS sg_phone TEXT,
  ADD COLUMN IF NOT EXISTS sg_postcode TEXT,
  ADD COLUMN IF NOT EXISTS sg_markers TEXT[],
  ADD COLUMN IF NOT EXISTS employee_bracket TEXT,
  ADD COLUMN IF NOT EXISTS locations_est INT,
  ADD COLUMN IF NOT EXISTS domain_hygiene BOOLEAN DEFAULT TRUE,
  ADD COLUMN IF NOT EXISTS sg_registered BOOLEAN DEFAULT FALSE;

CREATE INDEX IF NOT EXISTS idx_companies_uen ON companies(uen);
CREATE INDEX IF NOT EXISTS idx_companies_sg_registered ON companies(sg_registered);

