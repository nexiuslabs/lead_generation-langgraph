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
- [ ] Add flags: `ENABLE_JINA_DEEP_RESEARCH_DISCOVERY`, `ENABLE_JINA_DEEP_RESEARCH_ENRICHMENT`
- [ ] Add flag: `BG_DISCOVERY_AND_ENRICH=true` (background runs discovery+enrichment)
- [ ] Optional guards: `CHAT_DISCOVERY_ENABLED=false`, `CHAT_ENRICHMENT_ENABLED=false`
- [ ] Document new env vars in `project_documentation.md` and `docs/agents_prompts.md`

## Service Client (Deep Research)
- [ ] New module: `src/services/jina_deep_research.py`
  - [ ] Implement MCP/HTTP transport selection
  - [ ] `deep_research_query(seed, icp_context)` → {domains, snippets_by_domain, fast_facts}
  - [ ] `deep_research_for_domain(domain)` → {domain, summary, pages}
  - [ ] Normalize outputs, timeouts, retries, rate limiting
  - [ ] Telemetry hooks (`obs.bump_vendor`, `obs.log_event`)
  - [ ] Align request/response exactly with `docs/Jina_deepresearch.md` (endpoint, headers, fields, streaming/non‑streaming)

## Orchestrator (Chat Path)
- [ ] `my_agent/utils/nodes.py`: After ICP confirmation
  - [ ] Enqueue background job (discovery 50 + enrichment) and return `job_id`
  - [ ] Send chat message: background processing + email target + `job_id`
  - [ ] Remove edges to `plan_top10`, `enrich_batch`, `score_leads` and discovery nodes in chat
  - [ ] Remove enrichment confirmation gating and preview prompts

## Background Job (Unified)
- [ ] Introduce/rename unified job (e.g., `icp_discovery_enrich`)
  - [ ] Enqueue function: `enqueue_icp_discovery_enrich(tenant_id, notify_email)`
  - [ ] Runner: discovery → persist candidates (50) → per‑candidate enrichment → scoring → export → email
  - [ ] Idempotency, retries, and per‑stage status updates in `background_jobs`
  - [ ] Telemetry: `bg_discovery`, `bg_enrich_queue`, `bg_enrich_run`, `email_notify`

## Discovery Pipeline (Background)
- [ ] `src/icp_pipeline.py`
  - [ ] `build_resolver_cards`: use Deep Research first; fallback to DDG+r.jina
  - [ ] `collect_evidence_for_domain`: use Deep Research summary for extraction when enabled
  - [ ] Persist candidates to `staging_global_companies` with ai_metadata

## Enrichment Pipeline (Background)
- [ ] `src/enrichment.py`
  - [ ] Inject Deep Research summary/pages into corpus prior to MCP/Jina Reader + Tavily
  - [ ] Maintain existing shapes: `extracted_pages`, `deterministic_summary`, provenance
  - [ ] Contacts via Apify LinkedIn; use tenant/env `CONTACT_TITLES`
  - [ ] Persist: companies core fields, `company_enrichment_runs`, `summaries` (if used)

## Scoring & Export
- [ ] `src/lead_scoring.py`: run scoring after enrichment; persist `lead_features`, `lead_scores`
- [ ] Export to Odoo tenant DB when configured; generate CSV for email

## Email Notification
- [ ] Resolve recipient: header → JWT email → `tenant_users.user_id` (dev guard) → `DEFAULT_NOTIFY_EMAIL`
- [ ] Compose summary + attach CSV; send via SendGrid
- [ ] Log delivery status; surface errors in job record

## Telemetry & Observability
- [ ] Use `obs.begin_run`/`finalize_run` for per‑run context
- [ ] Emit `run_event_logs`, `run_vendor_usage`, `run_stage_stats`, `run_summaries`
- [ ] Mark degraded when fallbacks occur; persist run manifest of candidate IDs

## Docs
- [x] Clean Lusha mentions in AGENTS.md/read.md
- [ ] Update `AGENTS.md` mermaid to end chat after job enqueue
- [ ] Update `docs/agents_prompts.md` for Deep Research and flags
- [ ] Update `project_documentation.md` with new flow, env vars, and endpoints

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
- [ ] Remove Top‑10/Next‑40 code paths from chat; migrate any remaining helpers to background‑only
- [ ] Remove/rename any `web_discovery_bg_enrich` semantics to unified job naming

## Rollout
- [ ] Phase flags: discovery on → enrichment on → background‑only default
- [ ] Monitor telemetry + email deliverability
- [ ] Update runbooks and troubleshooting docs
