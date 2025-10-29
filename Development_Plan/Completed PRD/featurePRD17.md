**Feature PRD 17 — Enhanced ICP Intake & Pattern Mining (Aligned with Dev Plan 17)**

Status legend (for alignment)
- (Implemented): delivered in v1 scope or existing modules
- (Deferred): not in v1; candidate for future iteration
- (Optional): nice-to-have; implement as needed

- **Objective:** Deliver an agent-led conversational ICP intake (chat, not forms) that drives deterministic crawls of the user's site and seed-customer sites, cross-checks winners against ACRA/SSIC, surfaces data-backed patterns, and proposes micro‑ICPs and a Negative ICP list with evidence.
- **Primary Users:** Founders/GTMs, Ops, SDR enablement, ICP collection operators.
- **Non-Goals:** Automated outreach/sequencing; collecting personal emails from seed customer sites; replacing existing `icp_rules` persistence (we complement/derive it).

**UX & Flow (ICP Finder Enhancement)**
- Agent-led, conversational intake (no forms): The agent asks short, structured questions in chat and confirms understanding inline. No multi‑field form or wizard is used.
- Minimal input, evidence‑first: instead of asking “Industry?” or “Employee range?” upfront, the system learns from your website and best customer samples. The explicit “Industry” question is removed; industries are inferred from evidence and ACRA/SSIC mapping.
  - Provide your website URL (required), 5–15 best customers (company + website), and ~3 lost/churned (with a one‑line reason). Optional: geos, must‑have integrations, ACV/cycle, price floor, champion titles, win triggers.
- Automated understanding:
  - Learn from your website (Industries, Customers/Case Studies, Integrations, Pricing, Careers, Partners) and from seed customer sites (About, Industries, Products, Careers, Integrations, Locations).
  - Anchor to ACRA/SSIC by mapping seeds to UEN/SSIC for precise peer grouping.
  - Build an evidence bank of signals (SSIC, headcount hints, buyer titles, stacks, regions, use cases) with timestamp and source.
- Pattern mining → Micro‑ICPs:
  - Surface common SSICs, typical sizes, champion titles, integrations, and recurring themes across winners.
  - Shape 3–5 Micro‑ICPs (e.g., “SSIC 62012 + HubSpot + 50–200 HC + RevOps champion”) with segment size and rationale. Show industry names with SSIC codes (from `ssic_ref`) in suggestion titles.
  - Produce a Negative‑ICP list (SSICs/sizes/stacks/geos/budgets to exclude) with concise reasons.
- Review & tweak:
  - Show brief resolver confirmations only when needed (ambiguous matches). One‑click confirm/edit.
  - Nudge knobs (tighten headcount, require SOC2, deprioritize Salesforce) and recalc immediately.
- UI Output: 3–5 micro‑ICP suggestions with evidence; Negative ICP list; Targeting pack (SSIC + technographic filters + short pitch per segment). Adopt → writes `icp_rules`.

**Advanced Intake Details (A–H)**
- **A) Your Business:**
  - Problem statement: one sentence on the problem you solve.
  - Top 3 products/services: short labels.
  - Primary buyers: job titles/teams.
  - Best customer profile: plain words (e.g., “SG B2B SaaS, 50–200 employees, uses HubSpot”).
  - Must-have integrations: non‑negotiable.
  - Regulated industries: specialize in or must avoid.
  - Preferred geographies/languages/time zones to serve.
  - Capacity limits: onboarding bandwidth; SLAs you can actually meet.
- **B) Seed Customers:**
  - 5–15 best customers: company name | website.
  - Why they’re “best”: ACV, fast close, low support, strategic logo, etc.
  - Aspirational logos: not yet customers but ideal fits.
  - Resellers/partners you sell through.
- **C) Anti‑ICP & Red Flags:**
  - 3–10 lost/churned: company name | website + 1‑line reason.
  - Who should never be targeted: company types, sizes, tech stacks, budgets.
  - Deal killers: repeated blockers (security reviews, procurement rules, compliance gaps).
  - Price sensitivity: budget threshold where win‑rate collapses.
- **D) Firmographic Targets:**
  - Employee bands you win most (e.g., 10–50, 50–200, 200–1k).
  - Revenue bands (if applicable).
  - Entity type preferences (Pte Ltd vs Sole Prop vs Public).
  - Company age sweet spot (e.g., 2–10 years since incorporation).
  - Number of locations/stores/sites if relevant.
- **E) Technographic Fit:**
  - Stacks that increase win‑rate (e.g., HubSpot, Salesforce, Shopify, Snowflake).
  - Stacks that decrease/block win‑rate (e.g., legacy on‑prem X, competitor Y).
  - Security/compliance requirements you must meet (SOC2/ISO/PDPA specifics).
- **F) Buying Motion & Team:**
  - Typical champion titles and economic buyer title.
  - Security/procurement stakeholders who appear and when.
  - Implementation time customers accept (e.g., <2 weeks, <1 month).
  - Top use cases customers buy for (rank 1–3).
- **G) Triggers & Timing:**
  - Events that correlate with wins (e.g., hiring RevOps, opening new markets, migrating to cloud).
  - Public signals that suggest pain (compliance deadline, new regulation, press release).
  - Seasonality: when deals cluster.
- **H) Outcomes & Proof:**
  - Headline ROI claim (quantified + timeframe).
  - 2–3 case studies (links) that best represent ideal wins.
  - Top 3 objections with short rebuttals.

**Crawling & Evidence**
- **From user site:** Industries served; Customers/Case Studies; Integrations; Pricing (ACV hints); Careers (roles/scale); Partners; Blog topics.
- **From customer sites:** Industry labels; product lines; About text; Careers; Integrations pages; Locations.
- Collected observations are stored as `icp_evidence` and later summarized in `icp_patterns`.

**ACRA Cross-Check & Seeds Mapping**
- Build `customer_seeds(name, domain)` from intake.
- Resolve to `companies` by domain (prefer exact `primary_domain`), fallback to fuzzy legal-name match, then join to `staging_acra_companies` for UEN + SSIC.
- Requires Postgres `pg_trgm` (for `similarity(...)`). Prefer exact-domain matches; fuzzy threshold ≈ 0.35.
- **SQL sketch (reference):**
```
WITH seeds AS (
  SELECT lower(trim(company_name)) AS seed_name,
         lower(regexp_replace(website, 'https?://|www\.', '', 'gi')) AS domain
  FROM customer_seeds
),
resolved AS (
  SELECT s.seed_name, s.domain, c.company_id, c.legal_name
  FROM seeds s
  LEFT JOIN companies c ON c.primary_domain = s.domain
),
fuzzy AS (
  SELECT s.seed_name, s.domain, a.entity_name, a.uen, a.primary_ssic_code
  FROM seeds s
  JOIN staging_acra_companies a
    ON similarity(
         regexp_replace(s.seed_name, '(pte|ltd|private|limited|singapore|inc)\\b', '', 'gi'),
         regexp_replace(a.entity_name, '(pte|ltd|private|limited|singapore|inc)\\b', '', 'gi')
       ) > 0.35
)
SELECT DISTINCT ON (seed_name)
  seed_name, domain,
  COALESCE(r.legal_name, f.entity_name) AS matched_name,
  f.uen, f.primary_ssic_code
FROM resolved r
LEFT JOIN fuzzy f USING (seed_name, domain)
ORDER BY seed_name, (r.company_id IS NULL);
```
 - We’ll map each customer to ACRA (UEN, SSIC) to learn which SSIC codes and bands dominate your winners.

**Finder Flow Output (Implemented)**
- Resolver preview: After intake/confirm, show domain resolver cards when needed.
- Early micro‑ICPs: Present draft micro‑ICP suggestions with industry names and SSIC codes.
- ACRA totals: Display accurate ACRA total for suggested SSICs and 1–2 sample rows (UEN, entity name, SSIC, status).
- Accept + Run: User can “accept micro‑ICP N” and then type “run enrichment” to proceed.
- Enrichment cadence: Enrich up to 10 immediately (configurable via `CHAT_ENRICH_LIMIT`/`RUN_NOW_LIMIT`) and schedule the remainder for nightly (ACRA total minus the immediate batch).

**Stability & Safety (Implemented)**
- Prevent auto‑run after restarts: Router only progresses on explicit user messages; enrichment does not resume automatically on boot.
- Evidence NOT NULL fix: Before inserting into `icp_evidence`, ensure a valid `company_id` exists (create by UEN/domain if necessary).
- Routing loops avoided: Single‑decision router with guards against recursion and unintended progression.

**Data Model & Storage**
- `icp_intake_responses(tenant_id, submitted_by, submitted_at, answers_jsonb)`
- `customer_seeds(seed_name, domain, tenant_id, created_at)` (source = user input)
- `icp_evidence(company_id, signal_key, value, source, observed_at, tenant_id)`
- Materialized view `icp_patterns` (top SSICs, common integrations, median headcount, frequent buyer titles, recurring themes)
- Accepted suggestion(s) are normalized into `icp_rules.payload` for downstream scoring.

**Pipeline Integration**
- Intake stored → seeds mapped to ACRA → deterministic crawls populate `icp_evidence` → refresh `icp_patterns` → UI presents suggestions → user accepts → write `icp_rules` (per tenant) → existing enrichment pipeline proceeds using the adopted ICP.
- Locations:
  - `src/icp.py` (normalization + persistence)
  - nightly orchestration for `icp_patterns` refresh

**Configuration & Flags**
Status: (Implemented)
- `ENABLE_ICP_INTAKE=true` (gate wizard and backend endpoints)
- `ICP_WIZARD_FAST_START_ONLY` (optional; when true, hide advanced sections and omit industry/employee prompts; those are inferred from website + seeds and only confirmed when ambiguous)
- Crawl caps reused from existing settings (per-site page limits, timeouts)

**Observability**
Status: (Implemented core; deeper timings Deferred)
- Metrics: intake completion rate; % of seeds resolved to ACRA; evidence items per company; patterns materialized count; suggestion count per tenant.
- Logs: stages `icp_intake`, `pattern_mining`; include durations and mapping decisions.

**Compliance & Guardrails (PRD 9–10 alignment)**
- Public data only; respect robots.txt. PDPA: no personal emails from customer sites; apply suppression lists; retention per policy.
- Cost control: cap crawl pages per seed and per tenant; reuse cached content.

**Acceptance Criteria**
- ≥80% of provided seed customers resolve to ACRA (exact domain or fuzzy) within 5 minutes of submission.
- With only website + seeds provided (no industry/employee prompts), UI presents ≥3 micro‑ICP suggestions (with evidence) when ≥5 seeds are present.
- Negative ICP list shows ≥3 red‑flag themes derived from Anti‑ICP input and churn/loss reasons.
- `icp_patterns` refresh completes nightly and suggestions cite evidence items.
- Micro‑ICP suggestions display industry names with SSIC codes; confirm view shows accurate ACRA totals and 1–2 sample rows.
- After “accept micro‑ICP” and “run enrichment”, system enriches 10 immediately (configurable) and schedules the correct remainder for nightly; no enrichment auto‑runs after server restart.

**Implementation Plan**
- Phase 1: DB schema (responses, seeds, evidence) + chat-led minimal intake + ACRA mapping; basic `icp_patterns` view; minimal UI suggestions.
- Phase 2: Crawl/evidence enrichment; observability (metrics + logs); polish suggestion cards and Negative ICP explanations.
- Phase 3: Tenant-tunable knobs; docs/runbooks; finalize acceptance tests.

- **APIs & Internal Modules**
- Agent‑driven intake: `pre_sdr_graph` collects Fast‑Start answers and, on confirm, saves `icp_intake_responses` and `customer_seeds` (server-side; SSO required).
- Batch job: seeds→companies→ACRA mapping and evidence extraction; materialize `icp_patterns`.
- UI: chat components and suggestions view; accept decision writes to `icp_rules`.

**User Journey Flow**
- Entry: Authenticated user (ops/admin) types “Start ICP Finder” in Chat. Viewer can see existing ICP and evidence but cannot edit.
- Minimal Intake (chat-only): The agent prompts one item at a time for website URL and 5–15 seed customers (+ lost/churned). Employee band/industry are inferred from evidence and ACRA mappings; optional fields (geos, integrations, ACV/cycle, price floor, champions, triggers) are accepted when volunteered or via targeted follow‑ups.
- Resolver (only when needed): If a seed name is ambiguous, the agent presents a one‑click confirm/edit inline with detected ICP keys (SSIC, size, geo, stack) and proceeds.
- **On confirm:**
  - Backend persists `icp_intake_responses` and `customer_seeds`.
  - Background mapping runs (seeds→ACRA SSIC evidence) and patterns refresh.
  - Agent indicates planned enrichment and when suggestions are ready.
- **Advanced (optional):** User expands A–H sections to add context; these augment evidence but do not require answering industry/employee upfront.
- **Suggestions:** When ready, agent shows 3–5 micro‑ICP cards:
  - Each card displays SSIC cluster, headcount band, key integrations, champion titles, and an evidence count.
  - Actions: `View evidence` (opens evidence drawer with site snippets and ACRA mappings), `Adopt`, `Refine`.
- **Adopt ICP:** Selecting `Adopt` writes a normalized payload into `icp_rules` for the tenant. Agent confirms: “ICP updated. Nightly runs will use this ICP.” Optional: `Run a small batch now` to queue a limited run.
- **Refine & Re-run:** User tweaks inputs (e.g., remove a red-flag SSIC, add an integration), resubmits, and a lightweight re‑materialization runs. UI keeps history of prior suggestions for comparison.
- **Recovery & Edge Cases:**
  - < 5 seeds provided: UI warns that suggestions may be lower confidence; still proceeds.
  - 0 seeds mapped: falls back to domain‑exact matches only; if still none, shows industry‑only baseline with a prompt to add more seeds.
  - Crawl blocked: degrades gracefully; uses available ACRA + public metadata; cards show “reduced evidence”.
  - Privacy flags: personal emails on customer sites are ignored; UI explains policy.
- **Outcomes:** User exits with an adopted ICP, a Negative ICP list (with reasons), and a downloadable targeting pack (SSIC + technographic filters + short pitch per segment).

**Simple Example**
- **Inputs (Fast-Start):**
  - Website: `https://acmeanalytics.io`
  - 5 best customers: Supply59 (`supply59.com`), GreenPack Pte Ltd (`greenpack.sg`), PortPro Logistics (`portpro.sg`), SoftFlow ERP (`softflow.io`), Hudson Foods Asia (`hudsonfoods.com.sg`)
  - 3 lost/churned: TinyMart (`tinymart.sg`, budget too low), LegacySoft (`legacysoft.com`, on‑prem only), BudgetShip (`budgetship.sg`, security review failed)
  - Employee band: 50–200
  - Geo/language/time zone: Singapore; English; SGT
  - Must‑have integrations: HubSpot, Shopify
  - Deal size & cycle: ACV ≈ USD 25k; 60 days
  - Price floor: lose >50% when budget < USD 10k
  - Champion titles: Head of Sales, RevOps
  - Triggers (3): hiring RevOps; opening new market; Shopify migration

- **What happens under the hood:**
  - Build `customer_seeds` from the five companies and resolve to `companies.primary_domain` when possible; fuzzy‑match legal names to `staging_acra_companies` for UEN + SSIC when domain isn’t in `companies`.
  - Crawl user and seed sites for evidence: “Industries”, “Customers”, “Integrations”, “Careers”, “About”. Store into `icp_evidence` and refresh `icp_patterns`.
  - Aggregate patterns: SSIC frequency, headcount band, integrations, buyer titles; combine with Anti‑ICP constraints.

- **Example suggestions (UI):**
  - Micro‑ICP 1: SSIC cluster “Software consulting / integration” (e.g., 62012/62019); headcount 50–200; uses HubSpot; champions: RevOps/Head of Sales; Evidence: 4/5 seeds + site signals.
  - Micro‑ICP 2: SSIC “Logistics support / fulfillment tech”; headcount 50–500; uses Shopify; champions: Ops Lead; Evidence: 3/5 seeds.

- **Negative ICP (derived):**
  - Exclude companies with <10 employees or budget < USD 10k.
  - Exclude on‑prem only stacks; prefer cloud‑ready.
  - Flag strict security procurement as high risk.

- **Targeting pack (example):**
  - SSIC filters: include top 2–3 SSIC codes from winners; exclude codes seen in churned list.
  - Technographic filters: has HubSpot OR Shopify.
  - Pitch (sample): “Analytics + RevOps automations for 50–200 HC B2B teams on HubSpot/Shopify; live in 30–60 days.”

**User Journey Example (Chat Transcript)**
- Agent: “Hi! I’ll infer your ICP from your website + your best/worst customers. What’s your website URL?”
- User: “https://acmeanalytics.io”
- Agent: “Great. Share 5–15 of your best customers (Company — website).”
- User: “Supply59 — supply59.com; GreenPack Pte Ltd — greenpack.sg; PortPro Logistics — portpro.sg; SoftFlow ERP — softflow.io; Hudson Foods Asia — hudsonfoods.com.sg”
- Agent: “Thanks. Any 2–3 lost/churned with a short reason?”
- User: “TinyMart — tinymart.sg (budget < $10k); LegacySoft — legacysoft.com (on‑prem only)”
- Agent: “Got it. I’ll crawl your site + seed customer sites, map seeds to ACRA/SSIC, and mine patterns. This takes ~2–5 minutes. I’ll show draft micro‑ICPs when ready.”
- (background) Evidence gathering: user + seeds crawled; seeds→ACRA mapping; patterns materialized.
- Agent: “Here are 4 suggested micro‑ICPs (with evidence counts). 1) SSIC 62012/62019 + HubSpot + 50–200 HC + RevOps champion (Evidence: 4/5 seeds). 2) Logistics tech (SSIC cluster) + Shopify + 50–500 HC (Evidence: 3/5). …”
- User: “Adopt #1 and #2. Also exclude companies <10 HC.”
- Agent: “Done. ICP updated for this tenant and Negative‑ICP includes <10 HC. Want me to run a small enrichment batch now?”
- User: “Yes, run a small batch.”
- Agent: “Queued. I’ll notify you when enrichment + scoring is done. You can export top scores to Odoo anytime.”

**New Changes (Delta)**
- **Data Model:** Adds `icp_intake_responses`, `customer_seeds`, `icp_evidence`; introduces MV `icp_patterns`. Enforce RLS on all tenant-owned objects.
- **Endpoints:**
  - `POST /icp/intake` (create/update responses + seeds)
  - `GET /icp/suggestions` (returns micro‑ICPs + evidence counts)
  - `POST /icp/accept` (write accepted suggestion into `icp_rules`)
  - `GET /icp/patterns` (optional debug/ops view of MV aggregates)
- **Jobs & Orchestration:** New queue task for seeds→ACRA mapping and evidence crawl; nightly refresh of `icp_patterns` incorporated into scheduler.
- **UI:** Chat-first flow (no separate wizard); suggestions view with evidence drawer; resolver micro-modal from chat when needed; progress chips; draft state; comparison history.
- **Config/Flags:** `ENABLE_ICP_INTAKE`, `ICP_WIZARD_FAST_START_ONLY`; reuse crawl caps from existing settings.
- **Observability:** New stages `icp_intake`, `pattern_mining`; metrics for intake completion, seed mapping %, evidence items, suggestions per tenant; logs include mapping decisions and durations.
- **Security:** Role gating (viewer read-only; ops/admin edit); SSO claims enforced; PII masking in evidence views for viewer role.
- **Docs/Runbooks:** Add operator steps to rerun mapping, interpret evidence, and troubleshoot crawl blocks.

**Alignment Notes & Known Gaps (v1)**
- Chat-led intake + persistence are in scope; advanced A–H polish can ship incrementally. (Implemented core, deeper polish Deferred)
- Seeds→ACRA mapping supports domain exact + `pg_trgm` fuzzy fallback; threshold tuning and multilingual normalization are Deferred.
- Crawlers reuse deterministic fetchers with caps; JS-heavy rendering via Playwright is Optional/Deferred.
- `icp_patterns` MV and suggestions endpoint are in scope; per-tenant weighting knobs are Deferred.
- Observability covers key rates/counts and job lifecycle logs; per-stage timings/p99 metrics Deferred.
- Security/RLS and robots.txt compliance are in scope; PII redaction for viewer UI may need additional polish (Deferred).

Here’s the clean, non-technical, step-by-step flow your ICP Finder will run end-to-end.

Intake (5-minute wizard)
– You answer ~10 focused questions (website, 5 best customers, 3 lost, employee band, geo, must-have integrations, ACV/cycle, price floor, champion titles, win triggers).
– Light checks make sure nothing critical is missing; if so, you get a quick prompt to fix it.

Seed your “ground truth”
– Your 5–15 best customers become the training seeds.
– Your lost/churned list and “never target” notes become the guardrails.

Resolve who those customers actually are
– The system standardizes each company (domain/name cleanup).
– If a company name is ambiguous, you get a one-click confirm to pick the right one (human-in-the-loop).

Learn from your website
– It scans your site for “Industries,” “Customers/Case Studies,” “Integrations,” “Pricing signals,” “Careers,” and “Partners” to understand what you sell, to whom, and in what context.

Learn from your customers’ sites
– For each seed customer, it looks at “About,” “Industries,” “Products,” “Careers,” “Integrations,” and “Locations” to capture patterns that your winners share.

Anchor to official industry codes (ACRA/SSIC)
– Each customer is mapped to ACRA so the system knows their UEN/SSIC and comparable peers.
– This makes your targeting filters precise (not just fuzzy “industry” labels).

Build an evidence bank
– All signals (industry code, headcount hints, buyer titles, tech stacks, regions, use cases) are attached to each company with a timestamp and source.
– You get an audit trail for “why this company fits.”

Discover patterns from your winners
– It surfaces the most common SSIC codes, typical team sizes, frequent buyer titles, integrations that correlate with wins, and recurring themes from “About” pages and case studies.

Define a universe of potential lookalikes
– Using your geos, size bands, entity types, and the winner patterns, it assembles a list of candidate companies that could be a great fit.

Score fit (transparent, explainable)
– Each candidate is scored on:
• Pattern fit (looks like your winners)
• Firmographic fit (size, geo, age, entity)
• Technographic fit (must-have vs blocker stacks)
• Timing triggers (hiring signals, expansions, public moves)
• Negative signals (budget below floor, past deal-killers)
– Scores come with “because” reasons, not just a number.

Shape 3–5 “Micro-ICPs”
– The top candidates cluster into clear segments like:
“SSIC 62012 + HubSpot + 50–200 employees + RevOps/Head of Sales champion.”
– Each segment includes how large it is, why it fits, and how many of your best customers back it.

Produce a Negative-ICP list
– Explicit exclusions (SSICs, sizes, stacks, geos, budgets) with concise reasons to avoid time-wasters.

Deliver a Targeting Pack you can act on
– Practical filters (SSIC codes, headcount bands, geos, entity types, required integrations).
– A short pitch per segment + the top objection and best rebuttal for each.

Review & tweak (tight feedback loop)
– You can nudge the system (e.g., “tighten to 100–300 employees,” “must include SOC2,” “deprioritize Salesforce”).
– It instantly recalculates segments and scores.


Learn continuously
– As new deals win/lose, the system updates patterns, re-weights signals, and refines your micro-ICPs—so your targeting improves over time.

That’s it: ask smart → learn from winners → normalize with ACRA → mine patterns → score the market → propose 3–5 micro-ICPs with evidence → give you filters, pitches, and a clear “do-not-target” list → learn from results.


Perfect—here’s a clean, ready-to-use sample question set that follows the exact step-by-step flow. It’s split into: (1) 5-minute Wizard, (2) Full Intake (A–H), (3) Resolver confirmations, (4) Review & Tweak, (5) Activation preferences. Short, unambiguous, and UI-friendly.

1) Fast-Start Wizard (10 questions, ~5 mins)

Your website URL
e.g., https://nexiuslabs.com

List 5 best customers (name + website)
e.g., “Acme Pte Ltd – acme.com”

List 3 lost/churned (name + website + 1-line reason)
e.g., “Globex – globex.sg — budget < $10k”

Employee band you win most
10–50 / 50–200 / 200–1k / 1k+

Geos & languages you serve
e.g., “Singapore, Malaysia; English, Mandarin”

Must-have integrations
e.g., HubSpot, Salesforce, Shopify

Average deal size (ACV)
e.g., “$18k”

Typical deal cycle length
e.g., “4–8 weeks”

Price floor (deals usually fail when budget < …)
e.g., “<$8k ACV”

Champion title(s)
e.g., “RevOps Lead, Head of Sales”

3 events that predict a good fit (bonus if you have time)
e.g., “Hiring RevOps, opening SG office, migrating to HubSpot”

2) Full Intake (A–H)
A) Your Business (ground truth)

In one sentence, what problem do you solve?

Top 3 products/services (short labels).

Primary buyers (titles/teams).

Best customer profile (plain words).

Non-negotiable integrations you support.

Regulated industries you specialise in or must avoid.

Preferred geos/languages/time zones.

Capacity limits (onboarding bandwidth, SLAs you can truly meet).

B) Seed Customers (pattern mining)

List 5–15 best customers: Company | website.

Why are they “best”? (ACV, fast close, low support, strategic logo…).

Aspirational logos (not yet customers).

Resellers/partners you sell through.

C) Anti-ICP & Red Flags (avoid false positives)

3–10 lost/churned: Company | website | 1-line reason.

Who should never be targeted? (types, sizes, stacks, budgets).

Repeat deal-killers (security, procurement, compliance).

Price sensitivity: “We lose >50% when budget < $X.”

D) Firmographic Targets (align to ACRA/SSIC)

Employee bands you win most.

Revenue bands (if used).

Entity type preferences (Pte Ltd, Sole Prop, Public).

Company age sweet spot (e.g., 2–10 years since incorporation).

# of locations/stores/sites if relevant.

E) Technographic Fit (compatibility)

Stacks that increase win-rate (HubSpot, Salesforce, Shopify…).

Stacks that decrease/block win-rate (legacy X, competitor Y).

Security/compliance you must meet (SOC2/ISO/PDPA specifics).

F) Buying Motion & Team

Typical champions and economic buyer title.

Security/procurement stakeholders and when they appear.

Implementation time customers accept (<2 weeks, <1 month…).

Top 3 use cases (ranked).

G) Triggers & Timing (intent rules)

Events that correlate with wins (hiring, expansion, migrations).

Public pain signals (regulations, deadlines, press).

Seasonality (when deals cluster).

H) Outcomes & Proof (messaging)

Headline ROI claim (quant + timeframe).

2–3 case study links (your ideal wins).

Top 3 objections and your best short rebuttals.

3) Resolver Confirmations (brief human-in-the-loop)

Only shown if needed, right after seeding.

ICP Keys Detected:

Industry: Bags Manufacturing (SSIC 14101)

Employee Band: 10–50 employees

Geo Focus: Global (HQ: Singapore, offices in Malaysia, Vietnam)

Entity Type: Private Limited (Pte Ltd)

Company Age: 8 years since incorporation (2017)

Tech Stack: Shopify, HubSpot

Champion Titles Found: Head of Sales, Supply Chain Manager

Question to user:
➡️ Does this match your “best customer” seed?

✅ Yes, confirm

🔄 No, pick another match

✏️ Edit details (override industry, size, etc.)
