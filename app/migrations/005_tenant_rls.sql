-- Tenant columns and RLS policies for tenant-owned tables

ALTER TABLE IF EXISTS lead_scores   ADD COLUMN IF NOT EXISTS tenant_id INT;
ALTER TABLE IF EXISTS lead_features ADD COLUMN IF NOT EXISTS tenant_id INT;
ALTER TABLE IF EXISTS enrichment_runs ADD COLUMN IF NOT EXISTS tenant_id INT;

-- Enable RLS
ALTER TABLE IF EXISTS lead_scores ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS lead_features ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS enrichment_runs ENABLE ROW LEVEL SECURITY;

-- Policies use request-scoped GUC set via set_config('request.tenant_id', ...)
DO $$ BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies WHERE schemaname = 'public' AND tablename = 'lead_scores' AND policyname = 'lead_scores_isolation'
  ) THEN
CREATE POLICY lead_scores_isolation ON lead_scores
      USING (tenant_id::text = current_setting('request.tenant_id', true))
      WITH CHECK (tenant_id::text = current_setting('request.tenant_id', true));
  END IF;
END $$;

DO $$ BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies WHERE schemaname = 'public' AND tablename = 'lead_features' AND policyname = 'lead_features_isolation'
  ) THEN
CREATE POLICY lead_features_isolation ON lead_features
      USING (tenant_id::text = current_setting('request.tenant_id', true))
      WITH CHECK (tenant_id::text = current_setting('request.tenant_id', true));
  END IF;
END $$;

DO $$ BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies WHERE schemaname = 'public' AND tablename = 'enrichment_runs' AND policyname = 'enrichment_runs_isolation'
  ) THEN
CREATE POLICY enrichment_runs_isolation ON enrichment_runs
      USING (tenant_id::text = current_setting('request.tenant_id', true))
      WITH CHECK (tenant_id::text = current_setting('request.tenant_id', true));
  END IF;
END $$;
