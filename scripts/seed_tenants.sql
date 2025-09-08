-- Seed example tenants, rules, and odoo connections (adjust IDs/names)

INSERT INTO tenants(name) VALUES ('Tenant A') ON CONFLICT DO NOTHING;
INSERT INTO tenants(name) VALUES ('Tenant B') ON CONFLICT DO NOTHING;

-- Discover assigned IDs
-- SELECT * FROM tenants;

-- Example: assume tenant_id 1 and 2
INSERT INTO icp_rules(tenant_id, name, payload)
VALUES (1, 'default', '{"industries":["technology"],"employee_range":{"min":2,"max":100}}')
ON CONFLICT DO NOTHING;
INSERT INTO icp_rules(tenant_id, name, payload)
VALUES (2, 'default', '{"industries":["services"],"employee_range":{"min":2,"max":100}}')
ON CONFLICT DO NOTHING;

-- Odoo connections: use a template or full DSN
INSERT INTO odoo_connections(tenant_id, db_name, auth_type, secret, active)
VALUES (1, 'odoo_tenant_a', 'service', NULL, TRUE)
ON CONFLICT (tenant_id) DO UPDATE SET db_name=EXCLUDED.db_name, active=EXCLUDED.active;

INSERT INTO odoo_connections(tenant_id, db_name, auth_type, secret, active)
VALUES (2, 'odoo_tenant_b', 'service', NULL, TRUE)
ON CONFLICT (tenant_id) DO UPDATE SET db_name=EXCLUDED.db_name, active=EXCLUDED.active;

-- Refresh MV for candidates
REFRESH MATERIALIZED VIEW CONCURRENTLY icp_candidate_companies;
