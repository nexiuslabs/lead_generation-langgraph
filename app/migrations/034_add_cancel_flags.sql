-- Add cooperative cancellation flags to background_jobs
ALTER TABLE background_jobs
  ADD COLUMN IF NOT EXISTS cancel_requested boolean DEFAULT false;

ALTER TABLE background_jobs
  ADD COLUMN IF NOT EXISTS canceled_at timestamptz NULL;

