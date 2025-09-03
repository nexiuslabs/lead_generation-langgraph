-- SSIC reference table with FTS and trigram search

-- Extensions (safe if already installed)
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Versioned reference table
CREATE TABLE IF NOT EXISTS ssic_ref (
  code              TEXT PRIMARY KEY,              -- normalized, 5 chars (e.g., '62010')
  title             TEXT NOT NULL,                 -- official SSIC title
  description       TEXT,                          -- detailed definition
  version           TEXT NOT NULL,                 -- e.g., 'SSIC 2025A'
  effective_date    DATE NOT NULL DEFAULT CURRENT_DATE,
  source_file_hash  TEXT NOT NULL,                 -- SHA256 of the Excel/CSV
  created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Fast exact/IN lookups
CREATE INDEX IF NOT EXISTS idx_ssic_ref_code ON ssic_ref (code);

-- Full-text + trigram for free-text queries (e.g., “logistics”, “marine repair”)
ALTER TABLE ssic_ref
  ADD COLUMN IF NOT EXISTS fts tsvector;

-- Backfill/update fts for any existing rows
UPDATE ssic_ref
SET fts = to_tsvector('english',
          coalesce(title,'') || ' ' || coalesce(description,''));

CREATE INDEX IF NOT EXISTS idx_ssic_ref_fts
  ON ssic_ref USING GIN (fts);

CREATE INDEX IF NOT EXISTS idx_ssic_ref_title_trgm
  ON ssic_ref USING GIN (title gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_ssic_ref_desc_trgm
  ON ssic_ref USING GIN (description gin_trgm_ops);

-- Keep fts updated
CREATE OR REPLACE FUNCTION ssic_ref_fts_refresh()
RETURNS trigger AS $$
BEGIN
  NEW.fts := to_tsvector('english',
                 coalesce(NEW.title,'') || ' ' || coalesce(NEW.description,''));
  RETURN NEW;
END $$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_ssic_ref_fts ON ssic_ref;
CREATE TRIGGER trg_ssic_ref_fts
  BEFORE INSERT OR UPDATE ON ssic_ref
  FOR EACH ROW EXECUTE FUNCTION ssic_ref_fts_refresh();

-- Helpful expression index for staging lookups by SSIC code (text)
DO $$
BEGIN
  -- Create only if the table exists and the index does not
  IF EXISTS (
    SELECT 1 FROM information_schema.tables
    WHERE table_name = 'staging_acra_companies'
  ) THEN
    IF NOT EXISTS (
      SELECT 1
      FROM pg_class c
      JOIN pg_namespace n ON n.oid = c.relnamespace
      WHERE c.relkind = 'i'
        AND c.relname = 'idx_staging_acra_companies_ssic_norm'
    ) THEN
      EXECUTE 'CREATE INDEX idx_staging_acra_companies_ssic_norm ON staging_acra_companies ((regexp_replace(primary_ssic_code::text, ''\D'', '''', ''g'')));';
    END IF;
  END IF;
END $$;
