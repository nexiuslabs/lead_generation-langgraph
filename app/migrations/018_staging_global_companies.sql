-- app/migrations/018_staging_global_companies.sql
-- Staging table for global (non‑SG) web discovery candidates

CREATE TABLE IF NOT EXISTS public.staging_global_companies (
  id BIGSERIAL PRIMARY KEY,
  tenant_id BIGINT NULL,
  domain TEXT NOT NULL,
  source TEXT NOT NULL DEFAULT 'web_discovery',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  ai_metadata JSONB NOT NULL DEFAULT '{}'
);

-- Ensure idempotent inserts per (tenant, domain, source)
CREATE UNIQUE INDEX IF NOT EXISTS idx_staging_global_companies_tenant_domain_source
  ON public.staging_global_companies (tenant_id, domain, source);

-- Convenience index for per-tenant recents and counts
CREATE INDEX IF NOT EXISTS idx_staging_global_companies_tenant_created_at
  ON public.staging_global_companies (tenant_id, created_at DESC);

