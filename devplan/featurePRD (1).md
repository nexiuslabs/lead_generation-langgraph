# featurePRD.md — ICP Discovery & Enrichment Fixes (Singapore focus, multi‑profile)

## 1) Background
Recent runs show noisy “Top-10”s and unusable enrichment due to:
- Domain/category drift (regulators, standards bodies, **expos/directories**, consultancies).
- Geographical mismatch and weak Singapore signals.
- Scoring tuned to SaaS cues rather than our **lead types**.
- Sparse/irrelevant evidence and low contact coverage.
- **Domain hygiene issues** (e.g., invalid apex like `co.th` treated as a candidate).
- **ACRA mapping bug**: `sg_registered=True` set for foreign/non‑company domains without UEN resolution.

This PRD updates the design to support **multiple lead profiles** and harden discovery, evidence, scoring, and enrichment for Singapore.

## 2) Goals
- **Precision:** ≥80% of candidates are valid organizations for the chosen **lead profile** with Singapore presence or markers.
- **Actionability:** Each shortlisted company has a clear **“why now”** and at least one **relevant buyer title** identified or pattern-ready.
- **Transparency:** Every score includes **explanations** (matched signals + rule hits); failures report first‑drop reason.
- **Configurability:** Country/industry guardrails, allow/deny lists, **lead profiles**, and weights are configurable.

## 3) Non-Goals
- Building a full contact-harvesting system from scratch.
- Advanced de-duplication across heterogeneous CRMs.
- Browser automation or login-gated data sources.

## 4) Primary Users
- **Partner/Associate / Practice Lead:** Needs a confident shortlist + context to craft outreach.
- **BD/SDR:** Needs enriched details (HQ, headcount bracket, buyer titles) to act immediately.

---

## 5) Solution Overview (Fixes)
1. **SG discovery lock:** DDG/Jina constrained to `kl=sg-en`; prefer `site:.sg`; allow `.com` when **SG markers** are present (e.g., “Singapore”, `+65`, 6‑digit SG postcode).
2. **Denylist & path filters:** Exclude `.gov.sg`, `.edu.sg`, **expos/directories/standards/associations**, and paths like `/regulations|standards|policy|association|directory|expo|tradefair|event|conference/`.
3. **Multi‑profile scoring:** Add `lead_profile` switch with distinct **markers/weights** for:
   - `sg_employer_buyers` (e.g., HR/IR/WSH/WFL/MOM/TADM/TAFEP signals).
   - `sg_referral_partners` (e.g., HR consulting/recruitment/payroll/HRIS/EOR/PEO/CorpSec/relocation).
   - `sg_generic_leads` (SMEs with SG presence and growth signals).
4. **Evidence extraction (SG cues):** HQ city, SG phone, postcode, hiring intensity, employer/partner markers, compliance triggers; concise evidence summary (≤300 chars).
5. **ACRA normalization:** Only flag `sg_registered=True` when a **UEN** is confidently resolved; store `uen_confidence`.
6. **Domain hygiene:** Validate candidates are real FQDNs (no bare TLDs, no invalid apex), drop malformed or suffix‑only candidates early.
7. **UX & telemetry:** “Why” chips, honest errors (no async promises), and metrics to track precision/recall by stage.

---

## 6) Detailed Requirements

### 6.1 Discovery (Search + Domain Harvest)
**Inputs:** seed domains (client & lookalikes), optional keywords.  
**Rules:**
- Set **`country_hint="sg"`** → DDG `kl=sg-en`.  
- Prefer `site:.sg`. Accept `.com` if page contains ≥1 **SG marker**:
  - Regexes: `\bSingapore\b`, `\+65`, `\b\d{6}\b` (SG postcode).
- Candidate gating:
  - **Domain hygiene** passes (see 6.8).
  - Host TLD/host suffix not in denylist (see 6.2).
  - Page text includes **profile markers** for the selected `lead_profile` (see 6.4).
- De-echo seeds: exclude exact seed apex and obvious mirrors.

**Outputs:** `discovery_candidates[] = {domain, url, title, snippet, sg_markers[], profile_markers[]}`

### 6.2 Deny/Allow Lists
- **Apex deny (expanded):** `w3.org, ifrs.org, ilo.org, oecd.org, deloitte.com, grandviewresearch.com, umbrex.com,`
  `10times.com, tradefairdates.com, interpack.com, pack-print.de, exhibitorsvoice.com, expotobi.com, cantonfair.net`
- **Host-suffix deny:** `{gov.sg, edu.sg, mil, int}`.
- **Path pattern deny (expanded):** `/(standards?|regulations?|policy|association|directory|glossary|wiki|expo|tradefair|event|conference|exhibition|exhibitors)/` (case‑insensitive).
- **Allowlist** overrides deny for known employer/SME conglomerates with atypical domains.

**Config (YAML):**
```yaml
region: sg
deny:
  apex:
    - w3.org
    - ifrs.org
    - ilo.org
    - oecd.org
    - deloitte.com
    - grandviewresearch.com
    - umbrex.com
    - 10times.com
    - tradefairdates.com
    - interpack.com
    - pack-print.de
    - exhibitorsvoice.com
    - expotobi.com
    - cantonfair.net
  host_suffix: [gov.sg, edu.sg, mil, int]
  path_regex: "(?i)/(standards?|regulations?|policy|association|directory|glossary|wiki|expo|tradefair|event|conference|exhibition|exhibitors)/"
allow:
  apex: []
```

### 6.3 Evidence Extraction (SG + Profile cues)
For each candidate domain:
- **Company:** normalized legal name (prefer ACRA), domain, UEN (if resolved).
- **Location:** HQ city (prefer “Singapore”), **SG phone** (`+65`), **SG postcode** (`\b\d{6}\b`).
- **Scale:** Employee count **bracket**, # locations/stores (approx.), growth/hiring signals (# open roles).
- **Profile signals:** based on selected `lead_profile` (see 6.4).
- **Triggers (SG):** `MOM`, `TAFEP`, `TADM`, `WFL`, `WICA`, `WSH`, `CPF`, `PDPA` (HR data).
- **HRIS hints:** Workday / SuccessFactors / UKG (lightweight inference).
- **Summary:** ≤300 chars with **top 2–3 matched signals**.

**Outputs:** `evidence[] = {..., sg_markers[], profile_signals[], triggers[], summary}`

### 6.4 Lead Profiles & Scoring

#### Profiles
- **`sg_employer_buyers`** (law firm B2B):  
  *Markers:* `careers|jobs|our stores|locations|people & culture|human resources|employee relations|industrial relations`, SG compliance tokens (`mom|tafep|tadm|wfl|wica|wsh|cpf|pdpa`).  
  *Buyer titles:* `HR Director`, `Head of People & Culture`, `IR/ER Manager`, `HRBP Lead`, `Legal Counsel (Employment)`.

- **`sg_referral_partners`** (partner ecosystem):  
  *Markers:* `hr consulting|recruitment|payroll|hris|eor|peo|corporate secretarial|relocation|immigration|bookkeeping`.
  *Targets:* HR consultancies, recruiters, payroll/HRIS vendors, EOR/PEO, corp‑sec, relocation/immigration firms, accelerators/coworking.

- **`sg_generic_leads`** (SMEs for services like accounting/ops/commerce):  
  *Markers:* generic org presence (`about|services|contact|clients`), SG markers, growth cues (`hiring|new outlet|grand opening|expansion|tender`).  
  *Buyer titles:* `Founder/Director/Owner`, `Finance/Accounts Manager`, `Operations Manager`.

#### Scoring Weights (100 pts)
- **Employer buyers (default example):**
  - Employer presence (careers/locations/people & culture): **20**
  - SG compliance triggers (MOM/TAFEP/TADM/WFL/WICA/WSH/CPF/PDPA): **25**
  - HR/IR org presence (HRBP, ER/LR): **20**
  - Hiring intensity (≥3 roles in SG): **10**
  - HQ=Singapore or clear SG ops: **10**
  - Evidence completeness (≥5 key fields): **15**

- **Referral partners (example):**
  - Services match (consulting/recruitment/payroll/HRIS/EOR/PEO/corp‑sec/relocation): **30**
  - SG presence/markers: **20**
  - Complementarity to our offerings (clear referral fit): **30**
  - Evidence completeness: **20**

- **Generic leads (example):**
  - SG presence/markers: **30**
  - Org/Commerce signals (multi‑site, ecommerce, wholesale): **30**
  - Hiring/growth: **20**
  - Evidence completeness: **20**

**Gating rules:**
- Any denylist hit → **drop** (not scored).
- Score ≥70 → **Bucket A** (shortlist), 50–69 → **B**, else **C**.
- No “100” unless **both** a core trigger and core presence marker are detected for the selected profile.
- Cap scores when employees/size are **Unknown** (no 100s on sparse evidence).

**Outputs:** `scores[] = {domain, score, bucket, score_breakdown{...}, why[] }`

### 6.5 Enrichment
- **Buyer titles** per profile (examples above).  
- **Contacts:** If name/email not found, produce **email pattern guess** and LinkedIn URL when available.
- Mark missing fields as `"Unknown"` (never `"None"`).

**Outputs:** `enriched[] = {..., buyer_titles[], contact{name?, title, linkedin?, email?, pattern?}}`

### 6.6 UI/UX
- Table (min): Company | Domain | HQ | Employees (bracket) | SG markers | Profile | Triggers/Signals | Buyer title | Why (chips) | Score.
- **Why chips:** e.g., `MOM mention`, `Careers page`, `SG +65`, `6‑digit postcode`, `HR/IR page`, `Recruitment/Payroll`, `New outlet`.
- **Honest failures:** “Could not enrich `<domain>` due to `<reason>` (rate limit / blocked / no profile signals).” **Never** say “queued in background.”
- **Filters:** `Lead profile`, `Only .sg`, `Include .com with SG markers`, `Min score`, `Include/Exclude industries`.

### 6.7 Telemetry & Quality
- Metrics: discovery precision @ top‑20; % dropped by denylist; evidence completeness; shortlist yield; % domains failing hygiene; ACRA UEN match rate & confidence.
- Log **first‑drop reason** per rejected candidate (denylist, no SG markers, no profile markers, domain hygiene fail, etc.).
- Weekly 30‑domain review; target **≥80% precision** and **≥70% SG presence** for each profile.

### 6.8 Domain Hygiene & Validation
- Accept only **valid FQDNs** (must contain a registrable label + known public suffix).  
- Drop bare TLDs or suffix‑only strings (e.g., `co.th`) and malformed hosts.  
- Normalize domains to apex for dedupe; keep source URL for context.

### 6.9 ACRA Normalization & `sg_registered`
- `sg_registered=True` **only** when a **UEN** is resolved with **high‑confidence name match** (normalize “Pte Ltd” variants, strip punctuation).  
- Store `uen`, `uen_confidence` (0–1), and `acra_source`.  
- Never infer SG registration from TLD/markers alone.  
- If UEN not found, set `sg_registered=False` and keep `hq_city`/SG markers as soft evidence.

---

## 7) Data Model Changes

### Tables / Schemas
- `companies` (SG fields + hygiene):
  - `uen TEXT`, `uen_confidence NUMERIC`, `hq_city TEXT`, `sg_phone TEXT`, `sg_postcode TEXT`,
  - `sg_markers TEXT[]`, `employee_bracket TEXT`, `locations_est INT`,
  - `domain_hygiene BOOLEAN DEFAULT TRUE`
- `icp_evidence`:
  - `domain TEXT`, `summary TEXT`, `profile TEXT`, `signals TEXT[]`, `triggers TEXT[]`, `hris_hints TEXT[]`, `source_urls TEXT[]`
- `icp_scores`:
  - `domain TEXT`, `profile TEXT`, `score INT`, `bucket TEXT`, `score_breakdown JSONB`, `why TEXT[]`
- `icp_config`:
  - `region TEXT`, `profiles JSONB`, `deny JSONB`, `allow JSONB`, `weights JSONB`
- `icp_drops`:
  - `domain TEXT`, `reason TEXT`, `stage TEXT`, `extra JSONB`, `created_at TIMESTAMPTZ DEFAULT now()`

**Indexes:** `companies(domain)`, `icp_evidence(domain)`, `icp_scores(domain, profile)`, `icp_drops(domain)`, GIN on arrays/JSONB.

---

## 8) API / Agent Contracts

### Node: `discovery_planner`
**Input:** `{ seeds[], keywords?, country_hint="sg", lead_profile, config_ref }`  
**Output:** `{ plan, kl:"sg-en", site_filter:".sg|.com", sg_markers_regex[], profile_markers[], deny, allow }`

### Node: `_ddg_search_domains`
**Input:** `{ plan }`  
**Output:** `{ discovery_candidates[] }`

### Node: `snapshot_and_extract`
**Input:** `{ discovery_candidates[], config, lead_profile }`  
**Output:** `{ evidence[] }`

### Node: `scoring_and_gating`
**Input:** `{ evidence[], lead_profile }`  
**Output:** `{ scores[], shortlist[] }`

### Node: `enrichment_runner`
**Input:** `{ shortlist[], buyer_titles[] }`  
**Output:** `{ enriched[] }`

---

## 9) Config Example (multi‑profile)
```yaml
profile: sg_employer_buyers  # options: sg_employer_buyers | sg_referral_partners | sg_generic_leads

sg_markers:
  - "\\bSingapore\\b"
  - "\\+65"
  - "\\b\\d{6}\\b"

profiles:
  sg_employer_buyers:
    include_markers: ["careers","jobs","people & culture","human resources","employee relations","industrial relations"]
    deny_host_suffix: ["gov.sg","edu.sg","mil","int"]
    deny_path_regex: "(?i)/(standards?|regulations?|policy|association|directory|glossary|wiki|expo|tradefair|event|conference|exhibition|exhibitors)/"
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
  apex: ["w3.org","ifrs.org","ilo.org","oecd.org","deloitte.com","grandviewresearch.com","umbrex.com",
         "10times.com","tradefairdates.com","interpack.com","pack-print.de","exhibitorsvoice.com","expotobi.com","cantonfair.net"]
  host_suffix: ["gov.sg","edu.sg","mil","int"]
  path_regex: "(?i)/(standards?|regulations?|policy|association|directory|glossary|wiki|expo|tradefair|event|conference|exhibition|exhibitors)/"
```

---

## 10) Acceptance Criteria (AC)
1. **AC‑1 (Discovery lock):** With only `.com` seeds, top‑20 includes ≥14 domains with SG markers or `.sg` for the chosen profile.
2. **AC‑2 (Denylist):** Pages under `*.gov.sg` / `*.edu.sg`, expos/directories/standards domains **never** appear as candidates.
3. **AC‑3 (Employer/Partner/Generic gating):** ≥80% of candidates have ≥1 marker **for the active profile**.
4. **AC‑4 (Scoring integrity):** No candidate reaches 100 unless a **core trigger** and **core presence** marker are both detected; cap scores when employees/size are Unknown.
5. **AC‑5 (Why chips):** Each shortlisted row shows ≥2 chips tied to evidence.
6. **AC‑6 (ACRA):** `sg_registered=True` only when UEN is resolved with high‑confidence; store `uen_confidence`.
7. **AC‑7 (Telemetry):** Dashboard shows discovery precision, denylist drop %, evidence completeness, shortlist yield, hygiene failures, and UEN match rate.
8. **AC‑8 (Hygiene):** Invalid apexes/suffix‑only (e.g., `co.th`) are dropped at discovery with reason logged.

---

## 11) Test Plan

### Unit Tests
- `test_sg_markers_regex()` — detects `Singapore`, `+65`, 6‑digit postcode.
- `test_deny_path_regex()` — blocks `/policy`, `/standards`, `/expo`, `/tradefair`, `/directory`.
- `test_profile_markers()` — profile‑specific markers are recognized.
- `test_scoring_profiles()` — weight accumulation & bucket thresholds per profile.
- `test_domain_hygiene()` — rejects invalid apex/suffix‑only (e.g., `co.th`).
- `test_acra_gate()` — `sg_registered` flips only with UEN + high‑confidence match.

### Integration Tests (mocked fetch)
- Mixed `.sg` and `.com` pages → only SG‑marked `.com` pass gating.
- Expos/directories/standards pages are denied both at discovery and normalization.
- ACRA name normalization resolves UEN for known SG companies; non‑SG directories do not set `sg_registered`.

### E2E (staging)
- For each profile, seed with 10 known targets → expect ≥8 in Bucket A/B with clear “why” chips.

---

## 12) Rollout Plan
1. **Dev (Day 1–2):** Implement SG markers, expanded denylist/path filters, multi‑profile scorer, evidence fields, hygiene validator, ACRA gate. Unit tests green.
2. **Staging (Day 3–4):** Run 50‑domain sample per profile; validate AC‑1..AC‑8. Tweak weights.
3. **Prod (Day 5):** Enable behind feature flag `icp_sg_profiles`. Monitor telemetry for 1 week.
4. **General Availability:** Remove flag once precision targets are met for 2 consecutive runs.

---

## 13) Risks & Mitigations
- **Over‑filtering valid prospects** via strict SG markers → toggle “Include `.com` with SG markers only / also include likely SG pages”.
- **Sparse evidence** for SMEs → lean on ACRA/press/careers; avoid over‑scoring; cap max without size.
- **Keyword drift** → externalize lists in config; weekly review.

---

## 14) Open Questions
- Do we want industry allowlists per profile (e.g., Retail/F&B/Hospitality/Transport/Healthcare/Construction) to boost precision?
- Should we infer **email patterns** from MX records or keep simple heuristics?
- What’s the minimum “evidence completeness” threshold to display a domain?

---

## 15) Appendix — Key Regex/Heuristics

```python
SG_MARKERS = [r"\bSingapore\b", r"\+65", r"\b\d{6}\b"]
EMPLOYER_MARKERS = [r"careers", r"jobs", r"people & culture", r"human resources", r"employee relations", r"industrial relations"]
REFERRAL_MARKERS = [r"hr consulting", r"recruitment", r"payroll", r"hris", r"eor", r"peo", r"corporate secretarial", r"relocation", r"immigration", r"bookkeeping"]
GENERIC_MARKERS = [r"about", r"services", r"contact", r"clients", r"hiring", r"new outlet", r"expansion", r"tender"]
DENY_PATH = re.compile(r"/(standards?|regulations?|policy|association|directory|glossary|wiki|expo|tradefair|event|conference|exhibition|exhibitors)/", re.I)

def is_singapore_page(text: str) -> bool:
    return any(re.search(p, text or "", re.I) for p in SG_MARKERS)

def has_profile_markers(text: str, profile: str) -> bool:
    lists = {
        "sg_employer_buyers": EMPLOYER_MARKERS,
        "sg_referral_partners": REFERRAL_MARKERS,
        "sg_generic_leads": GENERIC_MARKERS,
    }
    return any(re.search(p, text or "", re.I) for p in lists.get(profile, []))

def is_valid_fqdn(host: str) -> bool:
    # Must contain at least one label before a known public suffix; no bare TLDs/suffix-only.
    return bool(re.match(r"^(?!-)[A-Za-z0-9-]{1,63}(?<!-)\.(?:[A-Za-z0-9-]+\.)*[A-Za-z]{2,}$", host or ""))
```
