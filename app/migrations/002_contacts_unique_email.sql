-- Ensure unique emails per company; allow multiple NULLs via partial index
CREATE UNIQUE INDEX IF NOT EXISTS ux_contacts_company_email
  ON contacts(company_id, email)
  WHERE email IS NOT NULL;

