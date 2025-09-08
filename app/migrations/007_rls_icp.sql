-- Add RLS for icp_rules on tenant_id (idempotent)
ALTER TABLE IF EXISTS icp_rules ADD COLUMN IF NOT EXISTS tenant_id INT;

DO $$ BEGIN
  ALTER TABLE icp_rules
    ADD CONSTRAINT fk_icp_rules_tenant
    FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id) ON DELETE CASCADE;
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

ALTER TABLE icp_rules ENABLE ROW LEVEL SECURITY;

-- Ensure tenant_id_guc() exists for policy evaluation
CREATE OR REPLACE FUNCTION tenant_id_guc() RETURNS INT AS $$
BEGIN
  RETURN NULLIF(current_setting('request.tenant_id', true), '')::INT;
END;
$$ LANGUAGE plpgsql STABLE;

DO $$ BEGIN
  CREATE POLICY icp_rules_tenant_isolation ON icp_rules
  USING (tenant_id = tenant_id_guc());
EXCEPTION WHEN duplicate_object THEN NULL; END $$;
