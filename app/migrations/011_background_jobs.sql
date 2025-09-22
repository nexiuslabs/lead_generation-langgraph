-- Feature 18: Lightweight background job tracking

CREATE TABLE IF NOT EXISTS background_jobs (
  job_id     bigserial PRIMARY KEY,
  tenant_id  int,
  job_type   text NOT NULL,          -- e.g., 'staging_upsert'
  status     text NOT NULL,          -- queued | running | done | error
  created_at timestamptz NOT NULL DEFAULT now(),
  started_at timestamptz,
  ended_at   timestamptz,
  params     jsonb,
  processed  int DEFAULT 0,
  total      int DEFAULT 0,
  error      text
);

CREATE INDEX IF NOT EXISTS idx_background_jobs_status ON background_jobs (status);

