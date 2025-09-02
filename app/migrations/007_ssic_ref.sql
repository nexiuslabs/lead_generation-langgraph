-- Create ssic_ref table and indexes for SSIC reference data
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS ssic_ref (
    code text NOT NULL,
    title text NOT NULL,
    description text,
    version text NOT NULL,
    source_file_hash text NOT NULL,
    updated_at timestamp without time zone DEFAULT now(),
    PRIMARY KEY (code, version)
);

-- Indexes to support title/description search
CREATE INDEX IF NOT EXISTS ssic_ref_title_desc_trgm_idx
    ON ssic_ref USING gin ((title || ' ' || COALESCE(description,'')) gin_trgm_ops);

CREATE INDEX IF NOT EXISTS ssic_ref_title_desc_tsv_idx
    ON ssic_ref USING gin (to_tsvector('english', title || ' ' || COALESCE(description,'')));

-- View exposing only the latest ssic_ref version
CREATE OR REPLACE VIEW ssic_ref_latest AS
SELECT * FROM ssic_ref
WHERE version = (SELECT MAX(version) FROM ssic_ref);
