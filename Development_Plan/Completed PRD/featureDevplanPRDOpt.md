# Dev Plan — PRD‑Opt (Singapore‑Focused ICP Discovery & Enrichment)

This is the concrete implementation plan for the optimization spec captured in `Development_Plan/Development/featurePRDOpt.md`.
It enumerates code changes, migrations, config surfaces, testing, rollout, and risks.


## 1) Scope
- Lock discovery to Singapore and add marker‑based gating for `.com` results.
- Expand deny/allow filters and enforce domain hygiene.
- Add multi‑profile scoring (employer buyers, referral partners, generic leads).
- Extract/persist SG evidence and compliance triggers; explainable scores and “why” chips.
- Correct ACRA normalization (UEN+confidence gate for `sg_registered`).
- Keep Top‑10 strictness and ensure Next‑40 always enqueues (already addressed), while improving precision.


## 2) Module Changes

**2.1 `src/agents_icp.py`**
- Add loader for SG config (markers, deny lists, profile weights) via `src/config_profiles.py`.
- Update `plan_top10_with_reasons(icp_profile, tenant_id)` to:
  - Inject `region_hint='sg'`; prefer `site:.sg` queries.
  - For `.com` hits, fetch skim text and require one SG marker.
  - Apply apex/host‑suffix/path deny filters and `is_valid_fqdn()`.
  - Compute profile‑aware scores and produce `why` chips and `score_breakdown`.
- Pseudocode:
```
from src.config_profiles import load_profiles
CFG = load_profiles()

def plan_top10_with_reasons(icp_profile: dict, tenant_id: int | None = None):
    profile = (icp_profile or {}).get("lead_profile") or CFG.default_profile
    q = build_queries(icp_profile, region_hint="sg", prefer_sg=True)
    hits = ddg_search(q, kl="sg-en")
    ranked = []
    for h in hits:
        dom = to_apex(h.url)
        if not is_valid_fqdn(dom) or is_denied(h, CFG):
            continue
        text = skim(h.url)
        if not is_singapore_page(text) and not dom.endswith(".sg"):
            continue
        s, why, br = score_profile(text, dom, profile, CFG)
        ranked.append({"domain": dom, "score": s, "bucket": bucket(s), "why": why, "breakdown": br, "snippet": h.snippet})
    return top_n(ranked, 50)  # UI shows Top‑10; we persist remainder for Next‑40
```

**2.2 `src/config_profiles.py` (NEW)**
- Load YAML at startup (env `SG_PROFILES_CONFIG` default `config/sg_profiles.yaml`).
- Provide helpers: `is_denied(url, cfg)`, `is_singapore_page(text)`, `is_valid_fqdn(host)`, `score_profile(text, dom, profile, cfg)`.
- Pseudocode:
```
import os, re, yaml

def load_profiles(path: str | None = None):
    p = path or os.getenv("SG_PROFILES_CONFIG") or "config/sg_profiles.yaml"
    with open(p, "r") as f:
        return yaml.safe_load(f)

SG_MARKERS = [r"\\bSingapore\\b", r"\\+65", r"\\b\\d{6}\\b"]
DENY_PATH = re.compile(r"/(standards?|regulations?|policy|association|directory|glossary|wiki|expo|tradefair|event|conference|exhibition|exhibitors)/", re.I)

def is_singapore_page(text: str) -> bool:
    return any(re.search(p, text or "", re.I) for p in SG_MARKERS)

def is_valid_fqdn(host: str) -> bool:
    return bool(re.match(r"^(?!-)[A-Za-z0-9-]{1,63}(?<!-)\\.(?:[A-Za-z0-9-]+\\.)*[A-Za-z]{2,}$", host or ""))
```

**2.3 `src/enrichment.py`**
- Extend deterministic extract to collect SG cues and profile markers:
  - `hq_city`, `sg_phone` (`+65`), `sg_postcode` (6 digits), `sg_markers[]`, `hiring_count`, `triggers[]` (MOM/TAFEP/TADM/WFL/WICA/WSH/CPF/PDPA), HRIS hints.
- Persist into `companies` and `icp_evidence` with a ≤300‑char summary; keep LinkedIn normalization.
- Enforce single Apify attempt (already in place) and respect `search_policy='require_existing'` for Top‑10/Next‑40.

**2.4 `src/icp.py` / `src/icp_pipeline.py` / `src/acra_direct.py`**
- Gate `sg_registered=True` only with high‑confidence UEN match; store `uen_confidence` and `acra_source`.
- Normalize legal names (strip punctuation, handle `Pte Ltd` variants) before confidence calc.

**2.5 `src/lead_scoring.py` (optional centralization)**
- Add profile weight maps and helpers `bucket(score)`, `cap_on_sparse_evidence(score, fields_present)`.

**2.6 `app/icp_endpoints.py` / `app/pre_sdr_graph.py`**
- Already persist Top‑10 and remainder to staging; ensure profile id travels in `ai_metadata` for consistent Next‑40.
- No breaking API changes; keep SSE event contract.
 - Emit and verify `enrich:next40_enqueued` after enqueue, including `{job_id, count}` payload.
 - Ensure Next‑40 is enqueued once per thread/session via `_enqueue_next40_if_applicable` and that chat announces completion when background jobs finish.
 - Maintain strict tenant scoping (no DEFAULT_TENANT_ID fallbacks); pass `tenant_id` through discovery, staging, and enrichment selectors.

**2.7 `scripts/run_bg_worker.py` / `src/jobs.py`**
- No functional change; retain logs for per‑company results and job lifecycle.
 - Claim jobs with `FOR UPDATE SKIP LOCKED`; guard against empty `params.company_ids` and record a clear error if encountered.
 - Record per‑candidate first‑drop reason (denylist, hygiene, no SG markers) to aid telemetry.

**2.8 Flow Parity & Events**
- Preserve the PRD flow: ICP confirm → SG‑locked discovery → preview Top‑10 + stage remainder → strict Top‑10 enrichment → enqueue Next‑40 → background completion announcement.
- Keep SSE sequence: `icp:planning_start`, `icp:toplikes_ready`, `icp:profile_ready`, `icp:candidates_found`, `enrich:start_top10`, `enrich:company_tick`, `enrich:summary`, `enrich:next40_enqueued`.
- Chat graph periodically surfaces completed background jobs in subsequent turns.


## 3) Migrations

Add SG fields and hygiene to `companies` (new SQL file `app/migrations/021_prd_opt_companies.sql`):
```
ALTER TABLE companies
  ADD COLUMN IF NOT EXISTS uen TEXT,
  ADD COLUMN IF NOT EXISTS uen_confidence NUMERIC,
  ADD COLUMN IF NOT EXISTS hq_city TEXT,
  ADD COLUMN IF NOT EXISTS sg_phone TEXT,
  ADD COLUMN IF NOT EXISTS sg_postcode TEXT,
  ADD COLUMN IF NOT EXISTS sg_markers TEXT[],
  ADD COLUMN IF NOT EXISTS employee_bracket TEXT,
  ADD COLUMN IF NOT EXISTS locations_est INT,
  ADD COLUMN IF NOT EXISTS domain_hygiene BOOLEAN DEFAULT TRUE,
  ADD COLUMN IF NOT EXISTS sg_registered BOOLEAN DEFAULT FALSE;

CREATE INDEX IF NOT EXISTS idx_companies_uen ON companies(uen);
CREATE INDEX IF NOT EXISTS idx_companies_sg_registered ON companies(sg_registered);
```

Notes
- No schema changes required for `icp_evidence` or `lead_scores`; continue persisting `signal_key='top10_preview'`, rationale, and profile metadata as per current flow.


## 4) Configuration

Create `config/sg_profiles.yaml` (defaults can match the spec):
```
profile: sg_employer_buyers
sg_markers: ["\\bSingapore\\b","\\+65","\\b\\d{6}\\b"]
profiles:
  sg_employer_buyers:
    include_markers: ["careers","jobs","people & culture","human resources","employee relations","industrial relations"]
    deny_host_suffix: ["gov.sg","edu.sg","mil","int"]
    weights: { employer_presence:20, sg_compliance_triggers:25, hr_ir_presence:20, hiring_intensity:10, hq_singapore:10, evidence_completeness:15 }
  sg_referral_partners:
    include_markers: ["hr consulting","recruitment","payroll","hris","eor","peo","corporate secretarial","relocation","immigration","bookkeeping"]
    deny_host_suffix: ["gov.sg","mil","int"]
    weights: { services_match:30, sg_presence:20, partner_fit:30, evidence_completeness:20 }
  sg_generic_leads:
    include_markers: ["about","services","contact","clients","hiring","new outlet","expansion","tender"]
    deny_host_suffix: ["gov.sg","edu.sg","mil","int"]
    weights: { sg_presence:30, org_signals:30, hiring_growth:20, evidence_completeness:20 }
deny:
  apex: ["w3.org","ifrs.org","ilo.org","oecd.org","deloitte.com","grandviewresearch.com","umbrex.com","10times.com","tradefairdates.com","interpack.com","pack-print.de","exhibitorsvoice.com","expotobi.com","cantonfair.net"]
  host_suffix: ["gov.sg","edu.sg","mil","int"]
  path_regex: "(?i)/(standards?|regulations?|policy|association|directory|glossary|wiki|expo|tradefair|event|conference|exhibition|exhibitors)/"
```

Env:
- `SG_PROFILES_CONFIG=config/sg_profiles.yaml`
- Reuse existing: `RUN_NOW_LIMIT`, `BG_NEXT_COUNT`, `ENABLE_AGENT_DISCOVERY`, etc.
 - Telemetry toggles (optional): `ENABLE_SG_TELEMETRY=1` to emit precision/deny/hygiene/uen gauges in logs.


## 5) Testing

Unit
- SG markers detection; deny path regex; FQDN hygiene; profile marker detection; scoring weights/buckets; cap on sparse evidence; UEN confidence gate.

Integration
- Discovery filters: mix of `.sg` and `.com` → only `.com` with SG markers pass; deny apex/path filtered out.
- Staging persistence: preview (10) + remainder (≥0) present; Next‑40 query finds remainder.
- Enrichment: SG evidence persisted; LinkedIn normalization; single Apify attempt; SSE events order.
- Events & tenancy: `enrich:next40_enqueued` present with job id; background completion surfaced; tenant scoping preserved throughout.

E2E
- Each profile seeded with 10 known targets → ≥8 in A/B; “why” chips align with extracted evidence; `sg_registered` only with UEN.
 - Honest failures visible (clear first‑drop reason available to UI) when a run yields few/no viable leads.

Observability (assert via logs/queries)
- Precision ≥80%, denylist drop %, evidence completeness %, shortlist yield, domain hygiene failures, UEN match rate.
- Per‑candidate first‑drop reason captured (denylist/hygiene/no‑SG‑marker/other).


## 6) Rollout
- Dev: implement features behind `icp_sg_profiles` flag; unit tests green.
- Staging: sample 50 domains per profile; tune weights.
- Prod: enable flag for select tenants; monitor precision, deny %, evidence completeness, hygiene failures, UEN rates; GA after two green cycles.


## 7) Risks
- Over‑filtering valid prospects → UI toggle to relax `.com` SG marker gating; tune lists weekly.
- Sparse evidence for SMEs → rely on ACRA/careers/press; cap scores on sparse evidence.
- Keyword drift → externalize lists in YAML; telemetry to surface drift.

## 9) Observability & Honest UX
- Metrics: discovery precision, denylist drop %, evidence completeness %, shortlist yield, domain hygiene failure count, UEN match rate.
- First‑drop reason: log and expose a concise reason when a candidate is filtered out.
- UI copy: avoid vague promises; surface honest failure messages when few/no leads pass gating.
- Docs: update `docs/runbook_icp_finder.md` and `docs/testing_acceptance_section6.md` with telemetry queries and troubleshooting tips.


## 8) Task Checklist
- See `Development_Plan/Development/toDoListDevplanOpt.md` for itemized status tracking.
