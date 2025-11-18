-- app/migrations/032_company_profile_storage.sql
-- Persist tenant-specific company profiles separately from ICP data

CREATE TABLE IF NOT EXISTS tenant_company_profiles (
  tenant_id INT PRIMARY KEY REFERENCES tenants(tenant_id) ON DELETE CASCADE,
  profile JSONB NOT NULL DEFAULT '{}'::jsonb,
  source_url TEXT,
  confirmed BOOLEAN DEFAULT FALSE,
  confirmed_at TIMESTAMPTZ,
  updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_tenant_company_profiles_confirmed
  ON tenant_company_profiles(tenant_id, confirmed);
