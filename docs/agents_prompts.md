# Agents & Prompts Reference

A quick reference of LLM-backed "agents" in this codebase: purpose, location, prompts, and where they are used or wired into the flow/graph.

Note: Paths are clickable in this workspace. This document reflects actual code in this repo (not just PRD plans).

## ICP Discovery Agents
### deep_research_query
- Purpose: Call Jina DeepResearch to fetch up to `{JINA_DEEP_RESEARCH_DISCOVERY_MAX_URLS}` candidate domains for a given seed and ICP context.
- Location: `src/services/jina_deep_research.py`
- Prompt (user message): see the inline template in that module. It instructs Jina to return ONLY a JSON array of `{company_name, domain_url}` objects and includes the ICP industries + geo summary plus the seed.
- Used by: `run_icp_discovery_enrich` (background job) before persisting domains to staging.

### deep_research_for_domain
- Purpose: Fetch a summary and up to `{JINA_DEEP_RESEARCH_SUMMARY_MAX_URLS}` page URLs for a single domain using Jina DeepResearch (HTTP).
- Location: `src/services/jina_deep_research.py`
- Used by: `node_deterministic_crawl` in `src/enrichment.py` to seed `deterministic_summary`/`extracted_pages` prior to MCP/Tavily.


### icp_synthesizer
- Purpose: Synthesize a micro‑ICP (industries, integrations, buyer titles, size bands, triggers) from seed snippets and prior ICP.
- Location: `lead_generation-main/src/agents_icp.py` (`icp_synthesizer`)
- System prompt:
```
You extract micro-ICP from short evidence. Return JSON with keys:
industries[], integrations[], buyer_titles[], size_bands[], triggers[].
Keep values short, lowercase, and deduplicated.
```
- Human template:
```
Existing ICP (optional): {prior}

Seed evidence:
{seeds}
```
- Used by: Helper for ICP planning (not hard-wired in the chat graph today). Can be called before discovery to enrich `state["icp_profile"]`.

### discovery_planner
- Purpose: Generate 3 concise web search queries from a micro‑ICP and collect candidate domains (DuckDuckGo/DDGS), optionally fetching r.jina.ai snippets.
- Location: `lead_generation-main/src/agents_icp.py` (`discovery_planner`)
- System prompt:
```
You write 3 web search queries to find B2B companies matching a micro‑ICP.
Keep concise. Cite which ICP fields influenced each query.
Return as lines: query — because ...
```
- Human template:
```
Industries: {inds}
Signals: {sigs}
Titles: {titles}
```
- Used by: `plan_top10_with_reasons()` in the same module; indirectly invoked from the chat graph to preview Top‑10 (see below).

### ensure_icp_enriched_with_jina
- Purpose: If the micro‑ICP is sparse, enrich it using r.jina.ai page snippets (DDG → r.jina.ai → LLM extraction).
- Location: `lead_generation-main/src/agents_icp.py` (`ensure_icp_enriched_with_jina`)
- System prompt:
```
Extract micro-ICP lists from web page snippets. Return arrays for industries,
integrations, buyer_titles, size_bands, triggers. Keep items concise but
meaningful; dedupe and lowercase.
```
- Human: concatenated r.jina.ai snippets (`evidence` string).
- Used by: Optional enrichment step; safe to call before or after `discovery_planner` if ICP fields are thin.

### evidence_extractor
- Purpose: Normalize raw crawl summaries into structured evidence fields (e.g., integrations, titles, pricing flags) for scoring/gating.
- Location: `lead_generation-main/src/agents_icp.py` (`evidence_extractor`)
- System prompt:
```
Extract ICP evidence fields.
```
- Human: the per-domain crawl/summary text.
- Used by: `plan_top10_with_reasons()` orchestration to produce Top‑10 with reasons; can be used after mini‑crawl.

### plan_top10_with_reasons (orchestrator helper)
- Purpose: Orchestrates `discovery_planner` → mini‑crawl → `evidence_extractor` → scoring stub; returns Top‑10 with "why".
- Location: `lead_generation-main/src/agents_icp.py` (`plan_top10_with_reasons`)
- Prompts: uses the prompts from the two agents above.
- Used by (graph): See references in `app/pre_sdr_graph.py` where it is imported and invoked to preview Top‑10.

References in graph:
- `lead_generation-main/app/pre_sdr_graph.py` (search for `plan_top10_with_reasons`):
  - During confirmation and candidates rendering, the graph attempts to call `plan_top10_with_reasons` to attach `agent_top10` into state.

## Pre‑SDR Graph Extraction

### extract_update_from_text (EXTRACT_SYS)
- Purpose: Convert the user’s latest chat message into a structured ICP update (industries, employees, revenue bucket, years, geos, signals, etc.).
- Location: `lead_generation-main/app/pre_sdr_graph.py` (`extract_update_from_text`, `EXTRACT_SYS`)
- System prompt (`EXTRACT_SYS`):
```
You extract ICP details from user messages.
Return JSON ONLY with industries (list[str]), employees_min/max (ints if present),
revenue_bucket (one of 'small','medium','large' if present), year_min/year_max (ints for incorporation year range if present),
geos (list[str]), signals (list[str]), confirm (bool), pasted_companies (list[str]), and signals_done (bool).
If the user indicates no preference for buying signals (e.g., 'none', 'any', 'skip'), set signals_done=true and signals=[].
If the user pasted company names (comma or newline separated), put them into pasted_companies.
```
- Human: latest user message text.
- Used by (graph): Core node in the chat graph to keep `state.icp` up to date.

Note: `QUESTION_SYS` is defined but not used; dynamic follow‑up questions in the graph are constructed heuristically.

## Conversation Agents (Helpers)

### answer_leadgen_question
- Purpose: Closed‑book Q&A about this system only (explains workflow/commands from an embedded reference and runtime context).
- Location: `lead_generation-main/src/conversation_agent.py`
- System prompt:
```
You answer ONLY from the provided Reference and Context. Do not use external/web knowledge.
If asked beyond what the Reference/Context covers, say what’s available in this system and what isn’t.
Be specific, concise, and recommend the next command the user can run (e.g., 'start lead gen', 'confirm', 'accept micro-icp N', 'run enrichment').

Reference:
{SYSTEM_REFERENCE}
```
- Human template:
```
Question: {q}
Context (state): {ctx}

Answer strictly from the Reference and Context.
```

### check_answer_relevance
- Purpose: Judge if a user answer addresses the agent’s question (strict relevance classifier).
- Location: `lead_generation-main/src/conversation_agent.py`
- System prompt:
```
You are a strict grader. Determine if the user's answer directly addresses the agent's question.
Only consider topical relevance, not writing style. Be fair but firm.
```
- Human template:
```
Question: {q}
Answer: {a}

Return a strict JSON object with keys: is_relevant (boolean), reason (short), missing_elements (string list).
```

## Enrichment Extractors

### extract_chain (page field extraction)
- Purpose: Extract schema‑guided fields from raw page content.
- Location: `lead_generation-main/src/enrichment.py`
- Prompt template:
```
You are a data extraction agent.
Given the following raw page content, extract the fields according to the schema keys and instructions,
and return a JSON object with keys exactly matching the schema.

Schema Keys: {schema_keys}
Instructions: {instructions}

Raw Content:
{raw_content}
```

### qualify_pages (is this the official site/About page?)
- Purpose: Score whether a page is likely the official website/About page, used to filter candidates.
- Location: `lead_generation-main/src/enrichment.py`
- Prompt template:
```
You are a qualifier agent. Given the following page, score 1–5 whether this is our official website or About Us page.
Return JSON {"score":<int>,"reason":"<reason>"}.

URL: {url}
Title: {title}
Content: {content}
```

## Rationale Generator

### generate_rationale
- Purpose: 2‑sentence justification for a lead score (used by scoring pipeline).
- Location: `lead_generation-main/src/openai_client.py`
- System prompt:
```
You are an SDR strategist.
```
- Human: dynamically constructed scoring prompt (includes features and score).

---

## Where the Graph Calls ICP Agents

- `lead_generation-main/app/pre_sdr_graph.py` (search for `plan_top10_with_reasons`):
  - On confirm/candidates stages, the graph attempts to import and call `src.agents_icp.plan_top10_with_reasons` to compute a quick Top‑10 with reasons and attach it to state (`agent_top10`). This helper internally uses `discovery_planner` and `evidence_extractor` prompts above.

## How Domain Search Queries Are Built

1) User pastes business website + customer websites → crawled into short seed snippets.
2) `icp_synthesizer` converts those snippets into a micro‑ICP (industries, integrations, buyer titles, triggers, size bands).
3) `discovery_planner` feeds the micro‑ICP fields to the LLM using its system/human prompts to produce 3 concise web queries; results are fetched via DDGS (DuckDuckGo) and deduped to candidate domains.
4) Optional: r.jina.ai snapshots are collected to provide evidence snippets and to enrich sparse micro‑ICP via `ensure_icp_enriched_with_jina`.

