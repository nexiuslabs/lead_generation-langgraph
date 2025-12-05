# PRD: Bugfix – Enrichment Flow (Contacts‑First, Remove DR Summary, LLM Agents, Full Company Enrichment)

- Status: Draft for review
- Owner: Platform/Agents
- Date: 2025‑11‑30
- Affected area: lead_generation-main/src/enrichment.py, lead_generation-main/src/services/jina_deep_research.py, scheduler/worker wiring and related docs

## 1) Summary
Shift enrichment to a contacts‑first strategy and remove per‑domain Deep Research summary usage. Enrichment must populate company‑level data (not only contacts) per onboarding.sql while using LLM‑driven agents for structured extraction and normalization.

New order:
- Primary: Jina Deep Research contacts -> extract decision‑makers -> ZeroBounce verify -> upsert.
- Secondary: Apify LinkedIn chain (by domain/company) -> verify -> upsert.
- Final fallback: Deterministic Crawl + Jina MCP read -> LLM extraction of firmographics and cues -> verify any emails found -> upsert all.

Remove deep_research_for_domain entirely and stop seeding deterministic_summary or synthetic pages from DR summary. Continue using DR contacts and any returned content/URLs to aid extraction.

## 2) Goals
- Maximize verified contact discovery quickly while still enriching full company records.
- Reduce vendor cost by removing per‑domain DR summaries.
- Use LLM agents to extract firmographics and cues from crawled corpus or DR returned content/URLs.

## 3) Non‑Goals
- Do not change discovery (Top‑50) logic or SSIC fallback.
- Do not change lead scoring formulas beyond consuming enriched fields.

## 4) Required Company Data (from onboarding.sql)
Company record should be enriched wherever available and reasonable:
- Identity and web: website_domain, linkedin_url, email, phone_number, domain_hygiene.
- Classification: industry_norm, industry_code, ownership_type, employee_bracket, company_size, locations_est.
- Size and revenue: employees_est, annual_revenue, revenue_bucket, employee_turnover, web_traffic.
- Incorporation and status: incorporation_year, founded_year, sg_registered, uen_confidence, sg_phone, sg_postcode, sg_markers.
- Geography: hq_city, hq_country, location_city, location_country.
- Tech and funding: tech_stack[], funding_status (jsonb).
- Run snapshot: company_enrichment_runs.about_text, tech_stack, jobs_count, linkedin_url, public_emails[], verification_results, source_json, embedding.
- Evidence: write key signals to icp_evidence (e.g., integrations, buyer_titles, hiring_open_roles, has_pricing, has_case_studies) where detected.
- Contacts: contacts table (full_name, email, job_title, department, linkedin_profile, seniority, location, source, verification fields) and lead_emails with ZeroBounce results.

## 5) Current vs New Flow
Current (simplified):
- find_domain -> deterministic_crawl (may seed with DR summary) -> discover_urls -> expand_crawl -> extract_pages -> build_chunks -> llm_extract -> apify_contacts (calls DR contacts first) -> persist

New (contacts‑first, LLM‑enabled):
- find_domain -> contacts_primary (DR contacts) -> contacts_secondary (Apify)
- If still insufficient -> deterministic_crawl -> discover_urls -> expand_crawl -> extract_pages -> build_chunks -> firmographics_llm_extract -> persist

Key deltas:
- Remove DR summary seeding. No early deep_research_for_domain call.
- Begin with deep_research_contacts before any crawling; Apify second.
- Deterministic crawl + MCP/Tavily is the final fallback path for both contacts (best‑effort) and company fields.
- Ensure enriched outputs cover required company fields listed above.

## 6) Deep Research API usage (enrichment only)
- Remaining: 1 API – deep_research_contacts(company_name, domain) (primary for contacts). Also harvest returned content/visited/read URLs (if any) to support extraction.
- Removed: deep_research_for_domain(domain).

## 7) LLM‑Driven Agent Design (LangChain + LangGraph)
- ContactDiscoveryAgent: consumes DR contacts output; parses names, titles, emails, profile links; tool: deep_research_contacts. Deterministic email verification via ZeroBounce tool.
- ContactsFallbackAgent: uses Apify domain/company chain; parses and verifies; toolset: Apify actors.
- FirmographicsAgent: structured LLM extraction over crawled corpus and/or DR returned content/URLs; outputs a Pydantic‑typed payload mapping to companies fields plus evidence signals for icp_evidence.
- NormalizationAgent: merges FirmographicsAgent output with existing DB; handles idempotent updates and provenance; maintains company_enrichment_runs snapshot fields and embeddings (via existing embedding tool).

Implementation rules:
- Agents created via langchain.agents.create_agent with explicit system prompts (role, outputs, failure rules).
- Tools decorated with @tool and typed args; deterministic retries and timeouts.
- LangGraph state as TypedDict; nodes idempotent with clear inputs/outputs.

## 8) Detailed Node/Gates
Node order (LangGraph):
- find_domain -> contacts_primary -> contacts_secondary -> deterministic_crawl -> discover_urls -> expand_crawl -> extract_pages -> build_chunks -> firmographics_llm_extract -> persist_core -> persist_legacy.

Contacts sufficiency gate (after contacts_primary and contacts_secondary):
- Sufficient when: at least one named contact and at least one ZeroBounce verified email (status valid or acceptable per config).
- If sufficient after primary or secondary, proceed to persistence (contacts + any readily available company fields) and skip heavy crawl; else continue to crawl pipeline.

Firmographics extraction:
- Input: crawled pages (Jina MCP, optional Tavily) and any DR returned content/URLs when present.
- Output: structured mapping for companies columns listed in section 4, plus icp_evidence signals and optional public_emails.

Verification rules:
- Run ZeroBounce on all discovered emails across paths; store verification status + confidence; only verified statuses count toward sufficiency gate.

Persistence:
- Upsert contacts with provenance (jina_deep_research or apify_*), and lead_emails with verification.
- Upsert company columns; write company_enrichment_runs snapshot (about_text, tech_stack, jobs_count, linkedin_url, public_emails, verification_results, source_json, embedding).
- Write icp_evidence for extracted signals with confidence and why.

## 9) Implementation Plan
Code changes
- Remove DR summary:
  - Delete deep_research_for_domain from src/services/jina_deep_research.py and remove from exports.
  - Remove import and usage of _dr_for_domain in src/enrichment.py (node_deterministic_crawl seeding of deterministic_summary and synthetic extracted_pages).
- Re‑order graph / early contacts:
  - Ensure the contacts node runs immediately after find_domain and before crawl; keep DR first inside that node, Apify as secondary.
  - Add/confirm early‑exit guard when contacts become sufficient (skip crawl).
- Add FirmographicsAgent:
  - New LLM step firmographics_llm_extract consuming extracted pages and DR returned content/URLs if available; produce typed struct mapping to companies columns and evidence signals.
  - Reuse existing embedding tool for about_text; write to company_enrichment_runs.
- Persistence and gates:
  - Persist contacts as soon as verified; persist company fields after firmographics_llm_extract or from lightweight data available pre‑crawl when present (e.g., LinkedIn URL discovered early).

Docs/config
- Update docs to remove DR summary seeding from enrichment and describe contacts‑first + LLM firmographics.
- Discovery flags unchanged; any enrichment summary flags removed from docs.

Observability
- Vendor usage bump for jina_deep_research only on contacts call.
- Stage events: contacts_primary, contacts_secondary, crawl_fallback, firmographics_llm_extract, persist.

## 10) Acceptance Criteria
- No references to deep_research_for_domain anywhere; function removed.
- Enrichment begins with DR contacts; when a verified email is found, contacts are upserted, and heavy crawl can be skipped unless firmographics are missing.
- If DR contacts insufficient, Apify runs; on sufficiency, persist without crawling.
- If both contact paths insufficient, the crawl pipeline and FirmographicsAgent run; company fields are populated per section 4 insofar as discoverable.
- ZeroBounce verification runs in all paths; verification results stored.
- Observability logs show stage timings and vendor usage for DR contacts and Apify; nightly metrics unaffected.
- All tests pass or are updated accordingly; smoke tests still show at least one extracted page when crawl path executes.

## 11) Test Plan
- Unit: mock DR contacts -> verify early persist; mock DR failure + Apify success -> verify secondary path; mock both fail -> ensure crawl + firmographics extraction executes and populates company fields.
- Integration: end‑to‑end run across tenants with varying availability; assert contacts upserted and company fields (industry, size, HQ, LinkedIn, tech_stack) populated; ensure emails verified and recorded in lead_emails.
- Regression: ensure extracted_pages populated when content fallback runs; enrich_company_with_tavily returns completed state; lead scoring unaffected.

## 12) Risks & Mitigations
- Risk: Skipping DR summary may reduce quick firmographics. Mitigation: FirmographicsAgent over MCP/Tavily corpus; DR contacts content/URLs leveraged when present.
- Risk: Contact parsing variability. Mitigation: structured parsers + multiple sources (DR content, visited/read URLs, MCP pages, Apify outputs).
- Risk: Vendor caps. Mitigation: respect existing daily caps and observability quotas.

## 13) Rollout
- Implement behind code changes (no new flags required).
- Monitor contact sufficiency rate, ZeroBounce pass rate, average enrichment duration, and firmographics coverage rate.
- Rollback plan: re‑enable prior node order if needed (not preferred).

## 14) Open Questions
- Which ZeroBounce statuses count as acceptable? Defaults: valid; consider accept_all configurable.
- If contacts are sufficient but firmographics are thin, run a minimal firmographics_llm_extract pass on lightweight sources (e.g., About/LinkedIn) without full crawl?

## 15) Affected Files (expected)
- src/services/jina_deep_research.py (remove deep_research_for_domain)
- src/enrichment.py (remove DR summary seeding; reorder graph edges; add firmographics_llm_extract; enforce gates)
- project_documentation.md, AGENTS.md (doc updates)
- tests/ (update enrichment order/behavior tests)

---

This PRD describes the intended changes only. Implementation will follow in a dedicated patch once approved.

