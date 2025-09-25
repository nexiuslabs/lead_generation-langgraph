-- PRD 17: Materialized view for ICP patterns (tenant-scoped)

CREATE MATERIALIZED VIEW IF NOT EXISTS icp_patterns AS
WITH e AS (
  SELECT tenant_id, company_id, signal_key, value
  FROM icp_evidence
)
SELECT tenant_id,
  -- Example aggregate: top SSIC codes by frequency
  (
    SELECT jsonb_agg(x ORDER BY cnt DESC)
    FROM (
      SELECT (value->>'ssic') AS code, count(*) AS cnt
      FROM e
      WHERE signal_key = 'ssic'
      GROUP BY 1
    ) x
  ) AS top_ssics
FROM e
GROUP BY tenant_id;

CREATE INDEX IF NOT EXISTS idx_icp_patterns_tenant ON icp_patterns (tenant_id);

