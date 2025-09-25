-- PRD 17: Evidence and patterns

CREATE TABLE IF NOT EXISTS icp_evidence (
  id           bigserial PRIMARY KEY,
  tenant_id    integer NOT NULL,
  company_id   integer NOT NULL,
  signal_key   text    NOT NULL,
  value        jsonb   NOT NULL,
  source       text    NOT NULL,
  observed_at  timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_icp_evidence_tenant_company
  ON icp_evidence (tenant_id, company_id);

CREATE INDEX IF NOT EXISTS idx_icp_evidence_tenant_signal
  ON icp_evidence (tenant_id, signal_key);

CREATE INDEX IF NOT EXISTS idx_icp_evidence_value_gin
  ON icp_evidence USING GIN (value);

