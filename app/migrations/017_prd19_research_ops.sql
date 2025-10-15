-- PRD19: ResearchOps artifacts and import runs
CREATE TABLE IF NOT EXISTS icp_research_artifacts (
  id BIGSERIAL PRIMARY KEY,
  tenant_id BIGINT NOT NULL,
  company_hint TEXT,
  company_id BIGINT,
  path TEXT NOT NULL,
  source_urls TEXT[] NOT NULL,
  snapshot_md TEXT NOT NULL,
  fit_signals JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  ai_metadata JSONB NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_icp_ra_tenant ON icp_research_artifacts(tenant_id);
CREATE INDEX IF NOT EXISTS idx_icp_ra_fit ON icp_research_artifacts USING GIN ((fit_signals));
CREATE INDEX IF NOT EXISTS idx_icp_ra_meta ON icp_research_artifacts USING GIN ((ai_metadata));

CREATE TABLE IF NOT EXISTS research_import_runs (
  id BIGSERIAL PRIMARY KEY,
  tenant_id BIGINT NOT NULL,
  run_started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  files_scanned INT NOT NULL DEFAULT 0,
  leads_upserted INT NOT NULL DEFAULT 0,
  errors JSONB NOT NULL DEFAULT '[]'::jsonb,
  ai_metadata JSONB NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_research_runs_tenant ON research_import_runs(tenant_id, run_started_at DESC);

