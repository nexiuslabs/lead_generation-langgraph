-- Feature 18: Performance indexes for ICP flows

-- Industry norm lookups
CREATE INDEX IF NOT EXISTS idx_companies_industry_norm_lower
  ON companies (LOWER(industry_norm));

-- Company name equality match (normalize_input upsert; dedupe)
CREATE INDEX IF NOT EXISTS idx_companies_name_lower
  ON companies (LOWER(name));

-- Website-domain equality match
CREATE INDEX IF NOT EXISTS idx_companies_website_domain
  ON companies (website_domain);

-- Optional: score-sorted shortlist browsing
CREATE INDEX IF NOT EXISTS idx_lead_scores_score_id
  ON lead_scores (score DESC, company_id DESC);

