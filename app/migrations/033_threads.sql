-- 033_threads.sql — tenant-scoped threads metadata table for return-user flow

CREATE TABLE IF NOT EXISTS threads (
  id UUID PRIMARY KEY,
  tenant_id INT NULL,
  user_id TEXT NULL,
  agent TEXT NOT NULL DEFAULT 'icp_finder',
  context_key TEXT NOT NULL,
  label TEXT NULL,
  status TEXT NOT NULL DEFAULT 'open', -- 'open'|'locked'|'archived'
  locked_at TIMESTAMPTZ NULL,
  archived_at TIMESTAMPTZ NULL,
  reason TEXT NULL,
  last_updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Helpful indexes
CREATE INDEX IF NOT EXISTS idx_threads_tenant ON threads(tenant_id);
CREATE INDEX IF NOT EXISTS idx_threads_lookup ON threads(tenant_id, user_id, agent, context_key, status);
CREATE INDEX IF NOT EXISTS idx_threads_updated ON threads(last_updated_at DESC);

-- One open thread per (tenant,user,agent,context)
CREATE UNIQUE INDEX IF NOT EXISTS uq_open_thread_per_context
  ON threads(tenant_id, user_id, agent, context_key)
  WHERE status = 'open';

-- Enable RLS and (re)create policies idempotently
ALTER TABLE threads ENABLE ROW LEVEL SECURITY;

-- Recreate read/update policies (DROP IF EXISTS is supported; CREATE POLICY lacks IF NOT EXISTS)
DROP POLICY IF EXISTS tenant_threads_read ON threads;
CREATE POLICY tenant_threads_read ON threads
  USING (
    tenant_id IS NULL OR tenant_id = current_setting('request.tenant_id', true)::int
  );

DROP POLICY IF EXISTS tenant_threads_write ON threads;
CREATE POLICY tenant_threads_write ON threads
  FOR UPDATE TO PUBLIC
  USING (
    tenant_id IS NULL OR tenant_id = current_setting('request.tenant_id', true)::int
  )
  WITH CHECK (
    tenant_id IS NULL OR tenant_id = current_setting('request.tenant_id', true)::int
  );
