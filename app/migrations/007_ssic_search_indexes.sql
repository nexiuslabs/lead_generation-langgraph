-- Indexes to support title/description search in ssic_ref
CREATE INDEX IF NOT EXISTS ssic_ref_title_desc_trgm_idx
    ON ssic_ref USING gin ((title || ' ' || COALESCE(description,'')) gin_trgm_ops);

CREATE INDEX IF NOT EXISTS ssic_ref_title_desc_tsv_idx
    ON ssic_ref USING gin (to_tsvector('english', title || ' ' || COALESCE(description,'')));
