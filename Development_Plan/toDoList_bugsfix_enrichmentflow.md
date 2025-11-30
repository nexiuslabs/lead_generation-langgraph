# TODO – Bugfix Enrichment Flow (contacts‑first, LLM firmographics)

- [x] Remove Deep Research per‑domain summary API `deep_research_for_domain` from codebase.
- [x] Remove imports/usages of `deep_research_for_domain` in `src/enrichment.py` (node_deterministic_crawl seeding).
- [x] Move contacts discovery ahead of crawl: `find_domain → apify_contacts` (DR contacts primary, Apify secondary).
- [x] Add conditional routing: if `contacts_sufficient` then `persist_core`, else `deterministic_crawl` → `discover_urls` → `expand_crawl` → `extract_pages` → `build_chunks` → `llm_extract` → `persist_core` → `persist_legacy`.
- [x] Compute and set `state["contacts_sufficient"]` when named contact(s) + at least one ZeroBounce‑verified email present.
- [x] Skip early MCP/visited‑URL page reads inside contacts node (reserve Jina MCP for final fallback crawl).
- [x] Change post‑crawl path: `llm_extract` goes directly to `persist_core` (do not call `apify_contacts` again).
- [x] Keep LLM firmographics extraction (existing `node_llm_extract`) to populate company fields required by onboarding.sql.
- [ ] Update docs (project_documentation.md, AGENTS.md) to reflect contacts‑first flow and DR summary removal.
- [ ] Update/adjust tests impacted by node ordering and sufficiency gating.
- [ ] Optional: add minimal firmographics pass when contacts sufficient but key fields are missing (future flag).

---

Implementation owner: Platform/Agents
Tracking: This list will be updated as docs/tests are finalized.

