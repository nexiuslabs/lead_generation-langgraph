CREATE TABLE IF NOT EXISTS onboarding_status (
  tenant_id INT PRIMARY KEY REFERENCES tenants(tenant_id) ON DELETE CASCADE,
  status TEXT NOT NULL,
  error TEXT,
  updated_at TIMESTAMPTZ DEFAULT now()
);

