ICP Finder: Intake → Candidate Generation (Tools & Quality Gates)

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

### Chat Flow: Intake → Confirmation → Enrichment

The LangGraph chat orchestrator (`app/pre_sdr_graph.py`) wires these stages into the
end-to-end user flow:

1. **Kick-off / Intake questions.** `start lead gen` (or similar intent) routes the
   turn into the ICP node, which gathers website and seed customers before allowing a
   `confirm`. (`router` guards around lines 4360–4510) 【F:app/pre_sdr_graph.py†L4365-L4526】
2. **`confirm` processing.** When the user responds `confirm`, the graph
   - saves the intake payload and evidence (if persistence helpers are available),
   - crawls the tenant site + seeds for structured signals, and
   - calls `plan_top10_with_reasons` to generate the Top‑10 table while caching it on
     `state["agent_top10"]` and persisting a preview per tenant. 【F:app/pre_sdr_graph.py†L3008-L3329】【F:app/pre_sdr_graph.py†L1048-L1108】
3. **Micro‑ICP gating.** If Finder mode is on, the router keeps the conversation on
   hold until micro‑ICP suggestions are generated and the user accepts one with
   `accept micro-icp N`. This unlocks enrichment commands while preventing premature
   runs. 【F:app/pre_sdr_graph.py†L4439-L4499】
4. **`run enrichment`.** The enrichment node persists the active ICP rule, then loads
   the shortlist to enrich. It first prefers the in-memory Top‑10 (`state["agent_top10"]`),
   falls back to the persisted preview, and finally reuses any synchronous head
   upserts from intake normalization. If no shortlist is available, it refuses to run
   and asks the user to rerun `confirm` to regenerate it. 【F:app/pre_sdr_graph.py†L1239-L1336】

#### Why "run enrichment" can fail after showing a Top‑10

`run_enrichment` performs strict guarding before it touches vendor credits. Even if a
Top‑10 table was displayed earlier, the shortlist must still be present in the current
state (or reloadable from persistence) so the agent knows exactly which domains to
enrich. If the conversation state was reset (e.g., reconnect, tenant change, or
`agent_top10` cleared) and the persisted preview could not be read, the node returns
the message observed in the chat transcript:

> “I couldn’t find the last Top‑10 shortlist to enrich. Please type ‘confirm’ to
> regenerate and lock a Top‑10, then try 'run enrichment' again.”

This protects against accidentally enriching stale or unrelated companies. The remedy
is to rerun `confirm` so the pipeline rebuilds evidence, persists a fresh Top‑10, and
then retry `run enrichment` once the shortlist is back in scope. 【F:app/pre_sdr_graph.py†L1239-L1336】

