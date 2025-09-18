-- Feature 11: Degradation signals

-- Run-level marker
ALTER TABLE IF EXISTS enrichment_runs
  ADD COLUMN IF NOT EXISTS degraded BOOL DEFAULT FALSE;

-- Per-company projection "reason" store (as free-form text CSV)
ALTER TABLE IF EXISTS company_enrichment_runs
  ADD COLUMN IF NOT EXISTS degraded_reasons TEXT;

