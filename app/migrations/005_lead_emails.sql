-- Lead emails table per PRD
CREATE TABLE IF NOT EXISTS lead_emails (
  email TEXT PRIMARY KEY,
  company_id INT NOT NULL,
  first_name TEXT,
  last_name TEXT,
  role_title TEXT,
  verification_status TEXT,
  smtp_confidence FLOAT,
  left_company BOOLEAN,
  role_last_seen TIMESTAMPTZ,
  source TEXT,
  source_json JSONB,
  last_verified_at TIMESTAMPTZ,
  bounce_count INT DEFAULT 0
);

