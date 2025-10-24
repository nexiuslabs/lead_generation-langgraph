-- app/migrations/031_bg_next40_unique.sql
-- Enforce one queued/running Next‑40 job per tenant+batch_id

-- Background: we gate duplicates in code, but add a DB partial unique index
-- to hard-prevent races creating multiple queued/running jobs for the same
-- (tenant_id, batch_id) for job_type 'web_discovery_bg_enrich'.

-- Note: This is a partial unique index scoped to queued/running only and
-- requires params to carry 'batch_id'.

CREATE UNIQUE INDEX IF NOT EXISTS uq_bg_next40_batch_unique
  ON background_jobs (tenant_id, (params->>'batch_id'))
  WHERE job_type = 'web_discovery_bg_enrich'
    AND status IN ('queued','running')
    AND params ? 'batch_id';

