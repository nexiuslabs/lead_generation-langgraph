-- Create enrichment backlog table for deterministic nightly drain
CREATE TABLE IF NOT EXISTS enrichment_backlog (
    tenant_id INT NOT NULL,
    company_id INT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, company_id)
);

CREATE INDEX IF NOT EXISTS idx_enrichment_backlog_tenant_status
    ON enrichment_backlog(tenant_id, status, created_at);

