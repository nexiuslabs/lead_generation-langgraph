-- Top‑10 enrichment alignment (idempotent)
-- Do NOT modify Development_Plan/Development/DBschema19Revised.sql; apply as a migration.

-- 1) companies: uniqueness + fast lookups on website_domain
CREATE UNIQUE INDEX IF NOT EXISTS ux_companies_website_domain
  ON companies(website_domain)
  WHERE website_domain IS NOT NULL;

-- 2) company_enrichment_runs: required columns for pipeline
ALTER TABLE IF EXISTS company_enrichment_runs
  ADD COLUMN IF NOT EXISTS tenant_id INT,
  ADD COLUMN IF NOT EXISTS public_emails TEXT[],
  ADD COLUMN IF NOT EXISTS verification_results JSONB,
  ADD COLUMN IF NOT EXISTS embedding DOUBLE PRECISION[];

-- FKs (added NOT VALID to avoid blocking on legacy rows)
DO $$ BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conname = 'company_enrichment_runs_tenant_id_fkey'
  ) THEN
    ALTER TABLE company_enrichment_runs
      ADD CONSTRAINT company_enrichment_runs_tenant_id_fkey
      FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id) ON DELETE SET NULL NOT VALID;
  END IF;
EXCEPTION WHEN undefined_table THEN NULL; END $$;

DO $$ BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conname = 'company_enrichment_runs_company_id_fkey'
  ) THEN
    ALTER TABLE company_enrichment_runs
      ADD CONSTRAINT company_enrichment_runs_company_id_fkey
      FOREIGN KEY (company_id) REFERENCES companies(company_id) ON DELETE CASCADE NOT VALID;
  END IF;
EXCEPTION WHEN undefined_table THEN NULL; END $$;

DO $$ BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conname = 'company_enrichment_runs_run_id_fkey'
  ) THEN
    ALTER TABLE company_enrichment_runs
      ADD CONSTRAINT company_enrichment_runs_run_id_fkey
      FOREIGN KEY (run_id) REFERENCES enrichment_runs(run_id) NOT VALID;
  END IF;
EXCEPTION WHEN undefined_table THEN NULL; END $$;

-- 3) contacts: FK + helpful indexes
DO $$ BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conname = 'contacts_company_id_fkey'
  ) THEN
    ALTER TABLE contacts
      ADD CONSTRAINT contacts_company_id_fkey
      FOREIGN KEY (company_id) REFERENCES companies(company_id) NOT VALID;
  END IF;
EXCEPTION WHEN undefined_table THEN NULL; END $$;

CREATE INDEX IF NOT EXISTS idx_contacts_company ON contacts(company_id);
CREATE INDEX IF NOT EXISTS idx_contacts_email_lower
  ON contacts(LOWER(email))
  WHERE email IS NOT NULL;

-- 4) lead_emails: FK + lookup indexes
DO $$ BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conname = 'lead_emails_company_id_fkey'
  ) THEN
    ALTER TABLE lead_emails
      ADD CONSTRAINT lead_emails_company_id_fkey
      FOREIGN KEY (company_id) REFERENCES companies(company_id) NOT VALID;
  END IF;
EXCEPTION WHEN undefined_table THEN NULL; END $$;

CREATE INDEX IF NOT EXISTS idx_lead_emails_company ON lead_emails(company_id);
CREATE INDEX IF NOT EXISTS idx_lead_emails_verified_at ON lead_emails(last_verified_at DESC);
CREATE INDEX IF NOT EXISTS idx_lead_emails_status ON lead_emails(verification_status);

-- 5) staging_global_companies: JSON probing for previews
CREATE INDEX IF NOT EXISTS idx_staging_global_companies_ai_metadata_gin
  ON staging_global_companies USING GIN (ai_metadata);

-- 6) icp_evidence: ensure loader indexes exist (if 013 not applied)
DO $$ BEGIN
  PERFORM 1 FROM pg_class WHERE relname='idx_icp_evidence_tenant_company';
  IF NOT FOUND THEN
    CREATE INDEX IF NOT EXISTS idx_icp_evidence_tenant_company ON icp_evidence(tenant_id, company_id);
  END IF;
EXCEPTION WHEN undefined_table THEN NULL; END $$;

DO $$ BEGIN
  PERFORM 1 FROM pg_class WHERE relname='idx_icp_evidence_tenant_signal';
  IF NOT FOUND THEN
    CREATE INDEX IF NOT EXISTS idx_icp_evidence_tenant_signal ON icp_evidence(tenant_id, signal_key);
  END IF;
EXCEPTION WHEN undefined_table THEN NULL; END $$;

DO $$ BEGIN
  PERFORM 1 FROM pg_class WHERE relname='idx_icp_evidence_value_gin';
  IF NOT FOUND THEN
    CREATE INDEX IF NOT EXISTS idx_icp_evidence_value_gin ON icp_evidence USING GIN (value);
  END IF;
EXCEPTION WHEN undefined_table THEN NULL; END $$;

-- Notes:
-- - icp_candidate_companies MV already created in 004_multi_tenant_icp.sql
-- - 018_staging_global_companies.sql already adds uniqueness + created_at index

