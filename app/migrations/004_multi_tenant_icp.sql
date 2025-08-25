-- Core multi-tenant scaffolding and ICP rules + candidates MV

CREATE TABLE IF NOT EXISTS tenants (
  tenant_id SERIAL PRIMARY KEY,
  name TEXT NOT NULL,
  status TEXT DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS tenant_users (
  tenant_id INT REFERENCES tenants(tenant_id) ON DELETE CASCADE,
  user_id TEXT NOT NULL,
  roles TEXT[],
  PRIMARY KEY (tenant_id, user_id)
);

CREATE TABLE IF NOT EXISTS odoo_connections (
  tenant_id INT PRIMARY KEY REFERENCES tenants(tenant_id) ON DELETE CASCADE,
  base_url TEXT,
  db_name TEXT,
  auth_type TEXT,
  secret TEXT,
  active BOOLEAN DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS icp_rules (
  rule_id SERIAL PRIMARY KEY,
  tenant_id INT REFERENCES tenants(tenant_id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  payload JSONB NOT NULL,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- Materialized view of candidate companies per rule
CREATE MATERIALIZED VIEW IF NOT EXISTS icp_candidate_companies AS
SELECT
  r.rule_id,
  r.tenant_id,
  c.company_id
FROM icp_rules r
JOIN companies c ON TRUE
WHERE
  -- industries[] optional
  (
    NOT (r.payload ? 'industries') OR
    LOWER(c.industry_norm) = ANY (
      SELECT LOWER(x) FROM jsonb_array_elements_text(r.payload->'industries') AS x
    )
  )
  AND (
    NOT (r.payload ? 'employee_range') OR (
      (NOT (r.payload->'employee_range' ? 'min') OR c.employees_est IS NOT NULL AND c.employees_est >= ((r.payload->'employee_range'->>'min')::INT))
      AND
      (NOT (r.payload->'employee_range' ? 'max') OR c.employees_est IS NOT NULL AND c.employees_est <= ((r.payload->'employee_range'->>'max')::INT))
    )
  )
  AND (
    NOT (r.payload ? 'incorporation_year') OR (
      (NOT (r.payload->'incorporation_year' ? 'min') OR c.incorporation_year IS NOT NULL AND c.incorporation_year >= ((r.payload->'incorporation_year'->>'min')::INT))
      AND
      (NOT (r.payload->'incorporation_year' ? 'max') OR c.incorporation_year IS NOT NULL AND c.incorporation_year <= ((r.payload->'incorporation_year'->>'max')::INT))
    )
  );

CREATE INDEX IF NOT EXISTS idx_icp_cc_rule ON icp_candidate_companies(rule_id);
CREATE INDEX IF NOT EXISTS idx_icp_cc_tenant ON icp_candidate_companies(tenant_id);

