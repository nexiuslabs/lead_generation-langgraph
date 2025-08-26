-- Ensure columns exist on company_enrichment_runs without failing if already present
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'company_enrichment_runs' AND column_name = 'public_emails'
  ) THEN
    ALTER TABLE company_enrichment_runs ADD COLUMN public_emails TEXT[];
  END IF;
END $$;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'company_enrichment_runs' AND column_name = 'verification_results'
  ) THEN
    ALTER TABLE company_enrichment_runs ADD COLUMN verification_results JSONB;
  END IF;
END $$;
