# To‑Do List — Jina Deep Research (Background‑Only Discovery + Enrichment)

Owner: Codex Agent – Frontend Generator
Status: draft
Last updated: 2025-05-14

Legend: [ ] pending, [~] in‑progress, [x] done

## Planning & Scope
- [x] Update feature dev plan to background‑only flow (no Top‑10/Next‑40)
- [x] Add Implementation Delta Checklist to the plan
- [x] Define Chat UI messaging for post‑ICP confirmation (job + email)

## Settings & Config
- [x] Add flags: `ENABLE_JINA_DEEP_RESEARCH_DISCOVERY`, `ENABLE_JINA_DEEP_RESEARCH_ENRICHMENT`
- [x] Add flag: `BG_DISCOVERY_AND_ENRICH=true` (background runs discovery+enrichment)
- [x] Optional guards: `CHAT_DISCOVERY_ENABLED=false`, `CHAT_ENRICHMENT_ENABLED=false`
- [ ] Document new env vars in `project_documentation.md` and `docs/agents_prompts.md`

## Service Client (Deep Research)
- [x] New module: `src/services/jina_deep_research.py`
  - [ ] Implement MCP/HTTP transport selection
  - [x] `deep_research_query(seed, icp_context)` → {domains, snippets_by_domain, fast_facts}
  - [x] `deep_research_for_domain(domain)` → {domain, summary, pages}
  - [x] Normalize outputs, timeouts, retries, rate limiting
  - [x] Telemetry hooks (`obs.bump_vendor`, `obs.log_event`)
  - [x] Align request/response exactly with `docs/Jina_deepresearch.md` (endpoint, headers, fields, streaming/non‑streaming)

## Orchestrator (Chat Path)
- [x] `my_agent/utils/nodes.py`: After ICP confirmation
  - [x] Enqueue background job (discovery 50 + enrichment) and return `job_id`
  - [x] Send chat message: background processing + email target + `job_id`
  - [x] Remove edges to `plan_top10`, `enrich_batch`, `score_leads` and discovery nodes in chat
  - [x] Remove enrichment confirmation gating and preview prompts

## Background Job (Unified)
- [x] Introduce/rename unified job (e.g., `icp_discovery_enrich`)
  - [x] Enqueue function: `enqueue_icp_discovery_enrich(tenant_id, notify_email)`
  - [x] Runner: discovery → persist candidates (50) → per‑candidate enrichment → scoring → export → email
  - [x] Idempotency, retries, and per‑stage status updates in `background_jobs`
  - [x] Telemetry: `bg_discovery`, `bg_enrich_queue`, `bg_enrich_run`, `email_notify`

## Discovery Pipeline (Background)
- [x] Unified in `src/jobs.run_icp_discovery_enrich`
  - [x] `deep_research_query` seeds candidates; fallback MCP/DDG when needed
  - [x] `deep_research_for_domain` feeds enrichment summaries when enabled
  - [x] Persist domains to `staging_global_companies` with `ai_metadata` provenance

## Enrichment Pipeline (Background)
- [x] `src/enrichment.py`
  - [x] Inject Deep Research summary/pages into corpus prior to MCP/Jina Reader + Tavily
  - [x] Maintain existing shapes: `extracted_pages`, `deterministic_summary`, provenance
  - [x] Contacts via Apify LinkedIn; use tenant/env `CONTACT_TITLES`
  - [x] Persist: companies core fields, `company_enrichment_runs`, `summaries` (if used)

## Scoring & Export
- [x] `src/lead_scoring.py`: run scoring after enrichment; persist `lead_features`, `lead_scores`
- [x] Export to Odoo tenant DB when configured; generate CSV for email

## Email Notification
- [x] Resolve recipient: header → JWT email → `tenant_users.user_id` (dev guard) → `DEFAULT_NOTIFY_EMAIL`
- [x] Compose summary + attach CSV; send via SendGrid
- [x] Log delivery status; surface errors in job record

## Telemetry & Observability
- [x] Use `obs.begin_run`/`finalize_run` for per‑run context
- [x] Emit `run_event_logs`, `run_vendor_usage`, `run_stage_stats`, `run_summaries`
- [x] Mark degraded when fallbacks occur; persist run manifest of candidate IDs

## Docs
- [x] Clean Lusha mentions in AGENTS.md/read.md
- [x] Update `AGENTS.md` mermaid to end chat after job enqueue
- [x] Update `docs/agents_prompts.md` for Deep Research and flags
- [x] Update `project_documentation.md` with new flow, env vars, and endpoints

## Tests
- [ ] Remove/refactor Top‑10 and chat discovery preview tests
- [ ] Add tests: journey_guard enqueues job + chat message with email + id
- [ ] Add tests: background job end‑to‑end (mock vendors): discovery→enrich→score→export→email
- [ ] Acceptance check: background pipeline runs; tables updated; email queued/sent

## DB & Migrations (reuse existing schema)
- [ ] Verify writes: `staging_global_companies`, `companies`, `icp_evidence`, `contacts`, `lead_emails`
- [ ] Verify run tracking: `enrichment_runs`, `company_enrichment_runs`
- [ ] Verify scoring: `lead_features`, `lead_scores`
- [ ] Verify telemetry: `run_event_logs`, `run_vendor_usage`, `run_stage_stats`, `run_summaries`, `run_manifests`
- [ ] Verify Odoo export with `ODOO_POSTGRES_DSN`

## Cleanup (Deprecated Paths)
- [x] Remove Lusha implementation (code + flags) and doc references
- [x] Remove Top‑10/Next‑40 code paths from chat; migrate any remaining helpers to background‑only
- [x] Remove/rename any `web_discovery_bg_enrich` semantics to unified job naming

## Rollout
- [ ] Phase flags: discovery on → enrichment on → background‑only default
- [ ] Monitor telemetry + email deliverability
- [ ] Update runbooks and troubleshooting docs
