-- PRD 17: Supporting extensions and indexes

-- Enable pg_trgm for fuzzy matching if not present
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Optional: trigram index to speed fuzzy searches on ACRA names
DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.tables WHERE table_name = 'staging_acra_companies'
  ) THEN
    CREATE INDEX IF NOT EXISTS idx_staging_acra_name_trgm
      ON staging_acra_companies USING gin ((entity_name) gin_trgm_ops);
  END IF;
END $$;

