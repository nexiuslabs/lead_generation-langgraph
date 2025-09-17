-- Observability and run bookkeeping (Feature 8 + PRD-7 extras)

-- Optional column on enrichment_runs for correlation
ALTER TABLE IF EXISTS enrichment_runs
  ADD COLUMN IF NOT EXISTS tenant_id INT,
  ADD COLUMN IF NOT EXISTS started_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS ended_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS status TEXT,
  ADD COLUMN IF NOT EXISTS langsmith_trace_url TEXT;

-- Stage-level stats per run
CREATE TABLE IF NOT EXISTS run_stage_stats (
  run_id           BIGINT      NOT NULL,
  tenant_id        INT         NOT NULL,
  stage            VARCHAR(64) NOT NULL,
  count_total      INT         DEFAULT 0,
  count_success    INT         DEFAULT 0,
  count_error      INT         DEFAULT 0,
  p50_ms           INT         DEFAULT 0,
  p95_ms           INT         DEFAULT 0,
  p99_ms           INT         DEFAULT 0,
  PRIMARY KEY (run_id, tenant_id, stage)
);
CREATE INDEX IF NOT EXISTS idx_rss_tenant_run_stage ON run_stage_stats(tenant_id, run_id, stage);

-- Vendor usage per run
CREATE TABLE IF NOT EXISTS run_vendor_usage (
  run_id           BIGINT      NOT NULL,
  tenant_id        INT         NOT NULL,
  vendor           VARCHAR(64) NOT NULL,
  calls            INT         DEFAULT 0,
  errors           INT         DEFAULT 0,
  tokens_input     INT         DEFAULT 0,
  tokens_output    INT         DEFAULT 0,
  cost_usd         NUMERIC(12,4) DEFAULT 0,
  PRIMARY KEY (run_id, tenant_id, vendor)
);
CREATE INDEX IF NOT EXISTS idx_rvu_tenant_run_vendor ON run_vendor_usage(tenant_id, run_id, vendor);

-- Event log (short retention)
CREATE TABLE IF NOT EXISTS run_event_logs (
  run_id       BIGINT      NOT NULL,
  tenant_id    INT         NOT NULL,
  stage        VARCHAR(64) NOT NULL,
  company_id   INT         NULL,
  event        VARCHAR(64) NOT NULL,
  status       VARCHAR(32) NOT NULL,
  error_code   VARCHAR(64) NULL,
  duration_ms  INT         NULL,
  trace_id     TEXT        NULL,
  extra        JSONB       NULL,
  ts           TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_rel_run_stage_ts ON run_event_logs(run_id, stage, ts DESC);
CREATE INDEX IF NOT EXISTS idx_rel_tenant_run ON run_event_logs(tenant_id, run_id);

-- QA samples
CREATE TABLE IF NOT EXISTS qa_samples (
  run_id     BIGINT      NOT NULL,
  tenant_id  INT         NOT NULL,
  company_id INT         NOT NULL,
  bucket     VARCHAR(16) NOT NULL,
  checks     JSONB       NOT NULL,
  result     VARCHAR(16) NOT NULL DEFAULT 'needs_review',
  notes      TEXT        NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (run_id, tenant_id, company_id)
);
CREATE INDEX IF NOT EXISTS idx_qa_tenant_run ON qa_samples(tenant_id, run_id);

-- Run manifest (selection) and summary (bookkeeping)
CREATE TABLE IF NOT EXISTS run_manifests (
  run_id        BIGINT      PRIMARY KEY,
  tenant_id     INT         NOT NULL,
  selected_ids  BIGINT[]    NOT NULL,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS run_summaries (
  run_id        BIGINT      PRIMARY KEY,
  tenant_id     INT         NOT NULL,
  candidates    INT         DEFAULT 0,
  processed     INT         DEFAULT 0,
  batches       INT         DEFAULT 0,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

