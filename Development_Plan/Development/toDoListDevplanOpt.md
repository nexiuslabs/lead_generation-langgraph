# To‑Do Tracker — PRD‑Opt (SG‑Focused Discovery & Enrichment)

Status legend: [ ] Not Started · [~] In Progress · [x] Done · [!] Blocked

## Milestones
- [~] Dev complete behind `icp_sg_profiles` flag
- [ ] Staging sample runs (≥50 domains/profile)
- [ ] Prod gated rollout (selected tenants)
- [ ] GA (two consecutive green cycles)

## Implementation Tasks

1) Config & Migrations
- [x] Add `config/sg_profiles.yaml` defaults
- [x] Add loader `src/config_profiles.py`
- [x] Migration `021_prd_opt_companies.sql` (SG fields + hygiene)

2) Discovery Filters & Hygiene (`src/agents_icp.py`)
- [x] Region lock (`kl=sg-en`), prefer `site:.sg`
- [x] SG markers gating for `.com` pages
- [x] Deny apex/host‑suffix/path filters (YAML‑driven)
- [x] FQDN validator (`is_valid_fqdn`)

3) Multi‑Profile Scoring (`src/agents_icp.py` or `src/lead_scoring.py`)
- [x] Profile switch: `sg_employer_buyers`, `sg_referral_partners`, `sg_generic_leads`
- [x] Weight maps + gating rules
- [x] Score breakdown + "why" chips per candidate

4) Evidence Extraction (`src/enrichment.py`)
- [x] Extract SG cues: `hq_city`, `+65`, 6‑digit postcode
- [x] Profile markers and compliance triggers (MOM/TAFEP/TADM/WFL/WICA/WSH/CPF/PDPA)
- [x] HRIS hints; hiring intensity
- [x] Persist summary to `companies` / `icp_evidence`

5) ACRA Normalization Gate
- [x] UEN + confidence required for `sg_registered=true`
- [x] Store `uen_confidence`, `acra_source`
- [x] Legal name normalization for match confidence

6) Staging & Next‑40 (already improved)
- [x] Persist Top‑10 preview and remainder to `staging_global_companies`
- [x] Ensure Next‑40 always enqueues; emit `enrich:next40_enqueued`

7) API/Chat integration
- [x] Carry `lead_profile` and provenance in `ai_metadata`
- [x] Keep SSE milestone order and messaging

8) Tests & QA
- [~] Unit: markers/deny/hygiene/scoring/ACRA gate
- [ ] Integration: discovery filters; staging persistence; Next‑40
- [ ] E2E: per‑profile seed runs; “why” chips sanity; UEN gating

9) Observability & Docs
- [~] Add precision/deny/evidence/UEN metrics (logs/queries)
- [x] Update README and runbook sections

## Notes
- Keep changes behind `icp_sg_profiles` until staging validation
- Weekly review of lists/weights; ensure config is externalized (YAML)
- Current status notes:
  - SG discovery gating and hygiene implemented behind `ICP_SG_PROFILES`; deny apex/host suffix applied at discovery; deny path applied during crawl/snippet selection.
  - Multi‑profile scoring implemented with weights, breakdowns, and `lead_profile` propagation; Top‑N items include `breakdown` and `lead_profile`.
  - ACRA gating writes `sg_registered` only when UEN is present and computes `uen_confidence` via name similarity; `acra_source` persisted when column exists.
  - SG cues, compliance triggers, HRIS hints, and hiring intensity extracted; SG cues persisted to companies; compliance snapshot persisted to `icp_evidence`.
  - Tests: added unit tests for profile scoring, compliance/HRIS extraction, and ACRA confidence; integration/E2E remain.
