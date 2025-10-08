-- Ensure icp_rules has a unique key on (tenant_id, name) so ON CONFLICT works
-- Idempotent: safe to run multiple times

DO $$ BEGIN
  CREATE UNIQUE INDEX uq_icp_rules_tenant_name ON icp_rules(tenant_id, name);
EXCEPTION WHEN duplicate_table OR duplicate_object THEN
  -- Index already exists
  NULL;
END $$;

