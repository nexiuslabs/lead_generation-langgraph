-- Hot-path indexes for Feature PRD 18 (create concurrently where possible)
-- Note: CONCURRENTLY cannot run inside a transaction; ensure your migration runner handles this accordingly.

CREATE INDEX IF NOT EXISTS idx_companies_industry_norm_lower
  ON companies (LOWER(industry_norm));

CREATE INDEX IF NOT EXISTS idx_companies_name_lower
  ON companies (LOWER(name));

CREATE INDEX IF NOT EXISTS idx_companies_website_domain
  ON companies (website_domain);

-- Optional: frequent sort for latest scores
CREATE INDEX IF NOT EXISTS idx_lead_scores_score_id
  ON lead_scores (score DESC, company_id DESC);

