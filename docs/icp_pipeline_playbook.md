ICP Finder: Intake → Candidate Generation (Tools & Quality Gates)

> **Note:** The intake flow now relies on three lightweight guardrails instead of multi-step confirmation loops:
> 1. Company profile snapshot shared → `company_ready=True`
> 2. ICP profile snapshot shared once ≥5 normalized seeds exist → `icp_ready=True`
> 3. Single discovery approval prompt → `discovery_ready=True`
>
> Snapshots are always available via “show company profile / show ICP profile,” and edits are applied immediately without reopening any gating state. Discovery/enrichment commands are the only actions blocked until the final approval is granted.

1) Intake (5‑Minute Wizard)
- Input: Website; 5 best seeds; 3 lost (reason); industries; employee band; geos; integrations; ACV; deal cycle; price floor; champion titles; triggers.
- Actions: Ask‑then‑parse (chat). Normalize seeds/domains; clean list/tokens; accept “skip/none” for optional fields.
- Output: Canonical answers (JSON); normalized seed list (name + domain).
- Quality gates:
  - Domain normalization (strip protocol/www/path).
  - Deduplicate seeds by (name, domain).
  - Require minimal viable set before proceeding (website + ≥5 best seeds).

2) Seed Normalization (Find the Right Domain)
- Tools:
  - Tavily: “{Company} official site”, “{Company} {geo} careers/press” → candidate domains + why.
  - Deterministic crawler: skim homepage for footer/company line.
- Actions: Rank domains by evidence; discard vendor/directory pages.
- Output: Company → domain mapping with confidence and rationale.
- Quality gates:
  - Confidence: High (legal/footer match), Medium (press+careers corroborate), Low (directory mention only).
  - Require ≥2 corroborating signals for High.

3) Resolver Confirmation (Human‑in‑the‑Loop)
- Tools:
  - Deterministic crawler: Pull /about, /careers, /integrations key facts.
  - LLM (light, optional): Extract ICP keys (industry guess, employee band, geo, titles, stacks).
- Actions: Build confirmation cards; show top match + keys; allow Confirm & Enrich, Pick Another, or Edit.
- Output: Confirmed domains + draft ICP keys for each seed.
- Quality gates:
  - Only show cards with ≥2 agreeing sources (About + Careers) as strong.
  - If confidence < threshold, recommend “Pick Another” or edit.

4) Evidence Collection (Your Site + Seeds)
- Tools:
  - Deterministic crawler: Robots.txt‑aware pull of allowed sections; page caps.
  - Tavily: Fill gaps (press coverage, integrations mentions, case studies).
  - Apify: LinkedIn Company/Jobs for headcount/HQ/roles when HTML is dynamic.
  - LLM (optional): Normalize raw text to signals (SSIC guess, titles, stacks, regions).
- Actions: Normalize, dedupe, timestamp; attribute source; attach confidence and “why”.
- Output: Evidence records per company (signal_key, value, source, timestamp, confidence).
- Quality gates:
  - Robots.txt compliance; domain‑level rate limits; page budget.
  - Triangulation: mark “usable” if signal appears in ≥2 places (e.g., Careers + Press).
  - Confidence scaling by source hierarchy (legal/footer > careers > press > directory).

5) ACRA/SSIC Anchoring
- Tools:
  - ACRA join (pg_trgm): Map name variants → UEN → SSIC (strip Pte/Ltd/Inc/Singapore).
  - LLM notes (optional): Explain mismatches.
- Actions: Attach “ACRA‑anchored industry” to each seed; write SSIC evidence.
- Output: Verified SSIC per seed + audit fields (matched_name, UEN).
- Quality gates:
  - Similarity threshold; stopword stripping; flag low‑similarity matches.

6) Pattern Mining (Winners Profile)
- Tools:
  - Simple stats over icp_evidence (top SSIC, common stacks, frequent titles, median headcount, geos).
  - LLM theme detection on About/Case Studies (optional).
- Actions: Build compact profile; include negatives from lost/churned.
- Output: “What winners share” snapshot + negative flags.
- Quality gates:
  - Minimum evidence density (e.g., ≥3 seeds share same SSIC/stack).
  - Separate positive/negative supports; highlight contradictions.

7) Candidate Universe (Lookalikes)
- Tools:
  - Tavily: “{SSIC/industry} company {geo} site”, “{integration} partners directory {geo}”, “hiring {champion title} {geo}”.
  - Deterministic crawler: Verify About/Careers/Integrations.
  - LLM: Compute feature‑wise fit scores (pattern, firmographic, technographic, trigger).
- Actions: Collect, verify, score; explain “why” succinctly.
- Output: Candidate list with per‑feature scores and short rationales.
- Quality gates:
  - Require on‑site verification for inclusion (no directory‑only candidates).
  - Penalize conflicting negatives (budget floor, on‑prem only, cycle > threshold).

8) Micro‑ICP Suggestions (First Pass)
- Input: Patterns + early candidate set.
- Actions: Propose 3–5 micro‑ICPs; show evidence counts; flag low‑density segments.
- Output: Segments ready to confirm/tweak or drive enrichment batch.
