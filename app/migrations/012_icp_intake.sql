-- PRD 17: ICP Intake (responses + seeds)

CREATE TABLE IF NOT EXISTS icp_intake_responses (
  id           bigserial PRIMARY KEY,
  tenant_id    integer NOT NULL,
  submitted_by text    NOT NULL,
  submitted_at timestamptz NOT NULL DEFAULT now(),
  answers_jsonb jsonb NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_icp_intake_tenant_time
  ON icp_intake_responses (tenant_id, submitted_at DESC);

CREATE INDEX IF NOT EXISTS idx_icp_intake_answers_gin
  ON icp_intake_responses USING GIN (answers_jsonb);

CREATE TABLE IF NOT EXISTS customer_seeds (
  id           bigserial PRIMARY KEY,
  tenant_id    integer NOT NULL,
  seed_name    text    NOT NULL,
  domain       text,
  created_at   timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_customer_seeds_tenant_name
  ON customer_seeds (tenant_id, lower(seed_name));

CREATE INDEX IF NOT EXISTS idx_customer_seeds_tenant_domain
  ON customer_seeds (tenant_id, domain);

