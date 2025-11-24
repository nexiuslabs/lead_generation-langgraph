# ICP Discovery Flow (Full Detail)

## 1. Chat Entry & State Setup
- Every user turn goes through `app/lg_entry.handle_turn`.
- History is replayed via `_append_message`, filtering JSON/system blobs.
- The new input is recorded in state (`input`, `input_role`), then the LangGraph orchestrator runs with this state.

## 2. Message Ingestion & Intent Detection
- `ingest_message` appends the latest user text to `messages`.
- `_simple_intent` (LLM) labels the intent (confirm company, run enrichment, etc.).
- `entry_context` stores `last_user_command` and `intent` for downstream nodes.

## 3. Profile Builder
- `_ensure_profile_state` seeds a default company profile if needed.
- `_summarize_company_site` uses Jina MCP (`jina_read`) to fetch the website and an LLM to convert that snippet into summary/industries/offerings/ideal customers/proof points.
- `_maybe_explain_question` answers user questions inline.
- Tracks outstanding prompts (e.g., “share 5 customer URLs”) until the user responds; sets `company_profile_confirmed` / `icp_profile_confirmed` on confirmation.

## 4. Journey Guard
- Verifies prerequisites: company profile, ICP profile, discovery confirmation, enrichment confirmation.
- Helpers:
  - `_needs_company_website` enforces that a URL was provided.
  - `_looks_like_discovery_confirmation` and `_wants_enrichment` parse the latest text (“start discovery”, “enrich 10”) to auto-flip flags.
- If inputs are missing, it prompts the user again.
- Once company + ICP + discovery are confirmed, it outputs “Prerequisites satisfied. Proceeding to normalization.”

## 5. Seed Normalization
- User-provided customer URLs go through `src.icp.normalize_agent` to extract canonical domains and metadata for the pipeline.

## 6. ICP Refresh (Seeds → Candidates)
- `refresh_icp` checks that enrichment is not pending; if all confirmations are in place it calls `icp_refresh_agent`.
- The agent:
  - Maps customer seeds to ACRA/SSIC (company registry codes, UENs).
  - Builds “winner” patterns (common stacks, hiring signals).
  - Runs DuckDuckGo searches (`src.ddg_simple`). The query text is generated from your ICP profile: industries, regions, and buyer hints are combined into phrases like `"b2b distributors" + geo + "site:.sg"` or `"food service wholesalers" + city`. Each query is logged (e.g., `[ddg] r.jina snapshot via html.duckduckgo.com ok for query=b2b distributors site:.sg`).
  - For each domain, fetches the homepage via `jina_read` (MCP snapshot). These snapshots power the “ICP Match” reasons in the Top-50 table.
  - Issues deterministic crawl instructions (the JSON `{"action":"deterministic_crawl",...}` you saw) to fetch about/contact/careers pages for richer evidence.
  - Outputs up to ~50 unique candidates with domain, bucket, score, and rationale.
- The agent returns `discovery["web_candidates"]`, `web_candidate_details`, `candidate_ids`, and diagnostics.

## 7. Strategy Decision
- `decide_strategy` inspects discovery state:
  - If cached Top-10/Top-50 details exist, it sets `strategy="use_cached"` and reuses them.
  - Otherwise it prompts an LLM to pick `use_cached`, `regenerate`, or `ssic_fallback` based on candidate counts and last SSIC attempt.

## 8. SSIC Fallback (Optional)
- If selected, `ssic_fallback` calls `icp_by_ssic_agent` to fetch companies matching the dominant SSIC codes, filling `discovery["candidate_ids"]` when the web search came up short.

## 9. Plan Top-10 + Next-40
- `plan_top10` either reuses cached results (when `strategy="use_cached"`) or calls `plan_top10_with_reasons` to score and rank candidates.
- `_cache_discovery_details` deduplicates by domain, stores the top 10 rows in `top10_details`, the next 40 in `next40_details`, plus `top10_domains`/`next40_domains`.
- `_ensure_company_ids_for_domains` guarantees each domain has a `companies` row and records those IDs in `top10_ids`/`next40_ids` (for a total of up to 50 unique candidates).

## 10. Progress Report to the User
- `progress_report` reads `discovery` and produces a Markdown table of candidates (e.g., “I found 14 ICP candidate websites…”).
- Sets `profile_state["awaiting_enrichment_confirmation"] = True` and tells the user to reply “enrich 10” or “retry discovery.”

## 11. User Confirmation
- When the user replies “enrich 10,” `journey_guard` marks `enrichment_confirmed = True` and clears outstanding prompts.
- The orchestrator can now move to `enrich_batch`, `score_leads`, and `export` (post-discovery steps) using the cached Top-10/Next-40 data.

## 12. Role of Jina MCP & DDG Throughout
- **Jina MCP** supplies fast, deterministic snapshots of every domain (both seeds and candidates), including homepage/about/contact/careers. These snapshots fuel the initial “ICP Match” text and later enrichment evidence.
- **DDG** provides breadth—multiple pages of candidate domains per query. Queries are synthesized from the ICP profile (industry + buyer + geo) so the crawler focuses on relevant verticals. The agent parses results, deduplicates them, and hands them to MCP for deeper inspection.
- When MCP can’t reach a site (e.g., HTTP 402), the system falls back to `https://r.jina.ai/...` or direct HTTP to keep the flow alive.

## 13. Scoring, Buckets, ICP Match & Deny Lists
- **ICP Match column** – The rationale shown in the discovery table (“hq singapore”, “signal match”) comes from the planner’s `why`/`reason` fields. Those are derived from the Jina MCP snapshot and preserved by `_format_candidate_table`; enrichment doesn’t rewrite them.
- **Lead scoring pipeline** (`src/lead_scoring.py`):
  1. `fetch_features` pulls firmographics (employees_est, revenue_bucket, sg_registered, incorporation_year, industry_code) plus manual-research counts from `icp_evidence` gathered during enrichment.
  2. `train_and_score` fits a logistic regression (balanced classes). When only positives exist it falls back to heuristics based on your ICP payload (employee range, revenue, incorporation year).
  3. `assign_buckets` maps the probability to “High/Medium/Low” buckets and demotes leads missing firmographics.
  4. `generate_rationales` produces concise explanations, noting penalties.
  5. `persist_results` writes the features/scores back to Postgres.
  `score_leads` then fetches primary emails (`_lookup_primary_emails`) and emits the “Lead Score summary” table (Company, Domain, Score 0–100, final Bucket, Email) in chat.
- **Bucket definitions** – Discovery’s Top‑50 list includes the planner’s heuristic bucket, but the Lead Score table shows the post-enrichment bucket from `assign_buckets`, reflecting actual firmographics + evidence.
- **Deny domain/path hygiene** – `src.config_profiles` defines apex, host-suffix, and path regex deny lists (e.g., `w3.org`, `.gov.sg`, `/login`). `src.agents_icp` filters every DDG/Jina candidate through these lists before caching them, and `src.enrichment` reuses the deny path regex to avoid crawling blocked URLs.

## 14. Transition to Enrichment/Scoring
- After discovery is confirmed and `enrichment_confirmed = True`, the orchestrator uses `top10_details`/`top10_ids` to run deterministic enrichment (Jina MCP crawl with Tavily fallback), lead scoring, and exports.
- `score_leads` now produces a “Lead Score summary” table (company, domain, score, bucket, email) based on the enriched evidence.

This workflow ensures you always see a clean Top-50 list derived from your seeds, with full reuse of cached candidates, deterministic snapshots via Jina MCP, and DDG-sourced breadth before you ever trigger Top-10 enrichment.
