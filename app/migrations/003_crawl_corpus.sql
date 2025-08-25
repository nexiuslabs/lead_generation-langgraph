-- Optional table to persist merged crawl corpus for transparency
CREATE TABLE IF NOT EXISTS crawl_corpus (
  id BIGSERIAL PRIMARY KEY,
  company_id BIGINT NOT NULL,
  page_count INT,
  source TEXT,
  corpus TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);

