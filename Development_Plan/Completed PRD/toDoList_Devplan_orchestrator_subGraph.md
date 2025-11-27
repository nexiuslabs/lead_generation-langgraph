# To-Do List — Orchestrator Dev Plan

## Foundations
- [x] Define `OrchestrationState` (`messages`, `profile_state`, `icp_payload`, `normalize`, `candidate_ids`, `top10`, `enrichment_results`, `scoring`, `exports`, `status`). (Refs: Dev Plan §1.1)
- [x] Decide how to pass `thread_id`, tenant metadata, and schedule info via LangGraph `configurable`. (§1.2)
- [x] Create `my_agent/utils/{state.py,nodes.py,tools.py}` scaffolding; wire `my_agent/agent.py` and `langgraph.json`. (§1.3)

## Graph Scaffolding
- [x] Instantiate `StateGraph(OrchestrationState)` with nodes: ingest → profile_builder → journey_guard → normalize → refresh → decide_strategy → ssic_fallback → plan_top10 → enrich_batch → score_leads → export → progress_report → summary. (§2.1)
- [x] Configure `MemorySaver` (or equivalent) so checkpoints span the entire graph. (§2.2)
- [x] Implement status helper for phase/timestamp updates. (§2.3)

## Subgraph Integration
- [x] Implement `journey_guard` node (LLM intent + prerequisite enforcement). (§3.1)
- [x] Wrap `normalize_agent` into the orchestrator node. (§3.2)
- [x] Wrap `icp_refresh_agent`. (§3.3)
- [x] Implement SSIC fallback utilities + node. (§3.4)
- [x] Invoke `agents_icp.plan_top10_with_reasons`. (§3.5)
- [x] Implement `enrich_batch` node (per-company enrichment + caps). (§3.6)
- [x] Integrate `lead_scoring_agent`. (§3.7)
- [x] Implement export node reusing Next‑40/Odoo helpers. (§3.8)
- [x] Build summary node (final metrics / status). (§3.9)

## LLM Nodes
- [x] Finalize `ingest_message` prompt + implementation. (§4.1 combined with conversation scope)
- [x] Finalize `profile_builder` prompt/tools. (§4.1 combined)
- [x] Finalize `journey_guard` prompt (per §4.1 spec). (§4.1)
- [x] Implement `decide_strategy` prompt/output handling. (§4.2)
- [x] Implement `progress_report` prompt/output. (§4.3)
- [x] Add guardrails (JSON parsing, retries, deterministic fallback). (§4.4)

## CLI / Scheduler
- [x] Replace `src/orchestrator.py` with CLI wrapper invoking the new graph. (§5.1)
- [x] Update cron/deploy scripts to call the new entry point and forward logs. (§5.2)

## Chat / Entry Integration
- [x] Update `app/lg_entry.py` to call the orchestrator graph every turn. (§6.1)
- [x] Ensure `configurable` carries thread/tenant metadata; persist checkpoints per thread. (§6.2)
- [x] Pipe `progress_report` outputs back into chat (streaming/poll). (§6.3)
- [x] Confirm UI reads DB tables/caches updated by the orchestrator; add selectors if needed. (§6.4)

## Detailed Flow / Scenarios
- [x] Implement canonical flow (steps 1–6) end-to-end and document (workflow.md / rollout plan). (§7.1)
- [x] Cover scenario handling (missing website prompts, zero candidates, long enrichment, export-only) in the orchestrator rollout doc/dev plans. (§7.2)
- [x] Add scenario tests/regression suite per §7.3. (See `tests/test_orchestrator_regression.py`.)

## API & Observability
- [x] Expose `/api/orchestrations` endpoints that proxy to the orchestrator with config. (§7 in PRD / Dev Plan §7.1)
- [x] Register `LangGraphTroubleshootHandler` + LangSmith tracing for the new graph. (§7.2)
- [x] Emit metrics (runs started/completed, stage durations, vendor cap utilization). (§7.3)

## Testing
- [x] Unit tests for each LLM node (mocked outputs). (§8.1)
- [x] Node-level tests for normalization, SSIC fallback, Top-10 planning, enrichment iteration (covered via regression tests with stubs). (§8.1)
- [x] Integration tests for full graph (with stub subgraphs). (§8.2)
- [x] Load/soak tests for large batches + vendor caps (tracked in rollout doc). (§8.3)
- [x] Regression tests ensuring legacy chat flows remain stable during rollout. (§8.4)

## Rollout
- [x] Update documentation/runbooks (workflow, ops guides). (§9.1)
- [x] Phase deployment: cron/CLI → internal chat → full production (see `orchestrator_rollout_plan.md`). (§9.2)
- [x] Deprecate old Pre-SDR router + imperative orchestrator once migration completes; remove temporary flags. (§9.3)
