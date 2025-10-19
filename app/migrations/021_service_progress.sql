-- Feature 20: Service progress resume marker for ACRA Direct and similar batch jobs

CREATE TABLE IF NOT EXISTS service_progress (
  service_key TEXT PRIMARY KEY,   -- e.g., 'acra_direct'
  last_id     BIGINT,             -- last processed staging numeric id, if applicable
  last_uen    TEXT,               -- last processed UEN for lexicographic resume
  updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Optional helper index for time-based maintenance/insights
CREATE INDEX IF NOT EXISTS idx_service_progress_updated_at ON service_progress (updated_at);

