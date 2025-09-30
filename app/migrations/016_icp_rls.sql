-- Optional RLS policies for ICP tables (enable if your deployment uses DB-level isolation)
DO $$ BEGIN
  EXECUTE 'ALTER TABLE icp_intake_responses ENABLE ROW LEVEL SECURITY';
EXCEPTION WHEN OTHERS THEN NULL; END $$;
DO $$ BEGIN
  EXECUTE 'ALTER TABLE customer_seeds ENABLE ROW LEVEL SECURITY';
EXCEPTION WHEN OTHERS THEN NULL; END $$;
DO $$ BEGIN
  EXECUTE 'ALTER TABLE icp_evidence ENABLE ROW LEVEL SECURITY';
EXCEPTION WHEN OTHERS THEN NULL; END $$;

-- Create a simple policy that matches tenant_id to current_setting('request.tenant_id')
DO $$ BEGIN
  EXECUTE $$CREATE POLICY icp_intake_responses_tenant_rls ON icp_intake_responses
           USING (tenant_id::text = current_setting('request.tenant_id', true))$$;
EXCEPTION WHEN OTHERS THEN NULL; END $$;
DO $$ BEGIN
  EXECUTE $$CREATE POLICY customer_seeds_tenant_rls ON customer_seeds
           USING (tenant_id::text = current_setting('request.tenant_id', true))$$;
EXCEPTION WHEN OTHERS THEN NULL; END $$;
DO $$ BEGIN
  EXECUTE $$CREATE POLICY icp_evidence_tenant_rls ON icp_evidence
           USING (tenant_id::text = current_setting('request.tenant_id', true))$$;
EXCEPTION WHEN OTHERS THEN NULL; END $$;

