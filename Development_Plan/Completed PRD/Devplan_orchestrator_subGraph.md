# Dev Plan: LangGraph Orchestrator with Subgraphs

This plan turns the feature PRD into concrete engineering steps, with the added requirement that the orchestrator itself ingests every user message and autonomously advances the journey (profile → ICP → enrichment) once prerequisites are satisfied. The backend LangGraph code will follow the structure:

```
my-app/
├── my_agent
│   ├── utils
│   │   ├── __init__.py
│   │   ├── tools.py
│   │   ├── nodes.py
│   │   └── state.py
│   ├── __init__.py
│   └── agent.py
├── .env
├── requirements.txt
└── langgraph.json
```

---

## 1. Foundations
1.1 **Schema draft**  
  - Define `OrchestrationState` (`icp_payload`, `entry_context`, `conversation`, `normalize`, `candidate_ids`, `top10`, `enrichment_results`, `scoring`, `exports`, `status`).  
  - Document type contracts for each subgraph invocation (normalize, refresh, plan Top‑10, enrichment, scoring, export).  
1.2 **Config plumbing**  
  - Decide how `thread_id`, `tenant_id`, scheduling metadata are passed (LangGraph `configurable` vs explicit state keys).  
1.3 **Repo layout**  
  - Within `my_agent`, add `utils/state.py` (state definitions), `utils/nodes.py` (node functions), `utils/tools.py` (shared helpers).  
  - Implement `my_agent/agent.py` to assemble the orchestrator graph; expose entrypoints via `langgraph.json`.  
  - Ensure `.env` and `requirements.txt` capture the orchestrator’s dependencies (LLM models, LangGraph, LangSmith).

## 2. Graph Scaffolding
2.1 **Graph skeleton**  
  - Instantiate `StateGraph(OrchestrationState)` and add nodes in order: `intent_router`, `normalize`, `refresh_icp`, `decide_strategy`, `ssic_fallback`, `plan_top10`, `enrich_batch`, `score_leads`, `export`, `progress_report`, `summary`.  
2.2 **Checkpointing**  
  - Use `MemorySaver` (or existing checkpointer) when compiling; ensure subgraphs inherit the same checkpointer.  
2.3 **Status tracking**  
  - Introduce a helper to update `state["status"]` with phase, timestamps, percent complete, and last message.

## 3. Subgraph Integration
3.1 **Intent router node**  
  - LLM node that reads `conversation` snapshot (messages, confirmations, gating flags) and outputs `{stage: ..., prerequisites_met: bool}`. Gate downstream nodes based on this result.  
3.2 **Normalize node**  
  - Wrap `normalize_agent.ainvoke`. Propagate errors and row counts into state.  
3.3 **ICP refresh node**  
  - Wrap `icp_refresh_agent.ainvoke`. Store candidate IDs and diagnostics (e.g., payload used).  
3.4 **SSIC fallback node**  
  - Implement `_find_ssic_codes_by_terms` + `_select_acra_by_ssic_codes`. Deduplicate IDs with existing set.  
3.5 **Plan Top-10 node**  
  - Call `agents_icp.plan_top10_with_reasons`; capture both preview list and metadata (scores, rationales).  
3.6 **Enrich batch node**  
  - Iterate candidate IDs (configurable concurrency). For each, invoke `enrichment_agent` or agentic planner.  
  - Persist per-company state (`completed`, `error`, vendor usage). Respect vendor caps and `ENRICH_SKIP_IF_ANY_HISTORY`.  
3.7 **Score leads node**  
  - Invoke `lead_scoring_agent`. Update `lead_scores` table, and copy relevant snippets back into orchestrator state for downstream use.  
3.8 **Export node**  
  - Reuse `src/jobs.enqueue_next40`, `app/odoo_store` helpers. Add gating (env flags, tenant availability).  
3.9 **Summary node**  
  - Consolidate metrics (counts, errors, runtime) and flag orchestration run as `completed`/`failed`.

## 4. LLM Nodes (First-Class)
4.1 **intent_router prompt**  
  - Prompt template:  
    ```
    SYSTEM: You orchestrate backend actions for a lead-gen copilot. Given the latest conversation snapshot, determine the next backend stage (normalize, refresh_icp, plan_top10, enrich_batch, score_leads, export, idle) and whether prerequisites are met.
    INPUT JSON:
      {
        "messages": [...],  // list of {author, text}
        "icp_confirmed": bool,
        "company_profile_confirmed": bool,
        "micro_icp_confirmed": bool,
        "awaiting_discovery_confirmation": bool,
        "enrichment_completed": bool,
        "last_user_command": str
      }
    OUTPUT JSON:
      {
        "stage": "<stage>",
        "prerequisites_met": true|false,
        "reason": "<brief explanation>",
        "required_followup": "<message to user if prerequisites unmet>"
      }
    ```
  - Include guard instructions: “Only emit the JSON object; do not add prose.”  
4.2 **decide_strategy**  
  - Prompt template:  
    ```
    SYSTEM: You are a backend planner deciding how to handle candidate discovery.
    INPUT JSON:
      {
        "candidate_count": int,
        "top10_fresh": bool,
        "last_ssic_attempt": "<timestamp or null>",
        "vendor_caps": {"apify": "...", "lusha": "..."},
        "user_intent": "<e.g., run full pipeline>"
      }
    OUTPUT JSON:
      {
        "action": "use_cached" | "regenerate" | "ssic_fallback",
        "reason": "<why>"
      }
    ```
4.3 **progress_report**  
  - Prompt template:  
    ```
    SYSTEM: Summarize the current orchestration status for a business user.
    INPUT JSON:
      {
        "stage": "<current stage>",
        "completed_counts": {"enriched": 8, "scored": 8},
        "pending_steps": ["export"],
        "errors": [...],
        "next_actions": [...]
      }
    OUTPUT JSON:
      {
        "message": "<2-3 sentence update for the user>",
        "next_step_summary": "<optional short note>"
      }
    ```
  - Store `message` in `state["status"]["message"]` and stream to chat.  
4.4 **Guardrails**  
  - Add JSON parsing, retries, and fallback defaults if LLM output fails validation. Allow substitution with deterministic rules via env flag for local testing.

## 5. CLI / Scheduler Integration
5.1 **Runner script**  
  - Replace `src/orchestrator.py` logic with `python -m src.orchestrator_graph --industries=...` that calls the graph.  
  - Support ICP payload overrides via CLI/env.  
5.2 **Cron jobs**  
  - Update deployment scripts to call the new runner. Ensure logs go through LangGraph callbacks.

## 6. Chat / Entry Integration
6.1 **Entry wrapper hook**  
  - Update `app/lg_entry.py` to invoke the orchestrator graph for every user turn, passing SDK payloads straight to `ingest_message`.  
  - Remove old Pre-SDR router wiring once parity is reached.
6.2 **Thread config + checkpoints**  
  - Ensure `configurable` carries `thread_id`, tenant metadata, and any per-session settings so checkpoints tie back to chat threads.  
6.3 **Progress streaming**  
  - Subscribe to orchestrator checkpoints (or poll `statusStore`) and inject status messages (from `progress_report`) into chat `messages`.  
  - Surface final summary + errors when the run completes.  
6.4 **Data reuse**  
  - Confirm DB artifacts (Top-10 cache, enrichment results, lead_scores) remain compatible with UI rendering components.  
  - Provide helper selectors for the UI to fetch orchestrator state (e.g., current stage, pending prompts).

## 7. Detailed Process Flow & Scenarios
7.1 **Canonical flow**  
  1. `ingest_message` normalizes payloads → `profile_builder` updates profile slots.  
  2. `journey_guard` checks confirmations. If missing data, it emits prompts and loops back to step 1.  
  3. Once ready, run `normalize_agent` → `icp_refresh_agent`.  
  4. `decide_strategy` picks cached/regenerate/SSIC, then `plan_top10` executes the discovery agent.  
  5. `enrich_batch` processes companies; `score_leads` runs afterwards.  
  6. `export`, `progress_report`, and `summary` close the loop, writing checkpoints/logs.
7.2 **Scenario handling**  
  - **Customer list before website**: `journey_guard` sees `company_profile_confirmed=false`, issues follow-up prompt, blocks backend nodes until website provided.  
  - **Zero candidates**: After `icp_refresh_agent` returns 0 IDs, `decide_strategy` selects SSIC fallback → `_select_acra_by_ssic_codes` populates candidates → pipeline resumes.  
  - **Long-running enrichment**: `progress_report` emits periodic counts; if vendor caps are hit, it tells the user and sets `status.state="paused"` until resume.  
  - **Export-only requests**: When user asks to export existing leads, `journey_guard` routes straight to `export` without rerunning discovery/enrichment.  
7.3 **Testing matrix**  
  - Create scenario tests covering each condition above to ensure the orchestrator loops correctly and the UI receives the right messages.

## 7. API & Observability
7.1 **REST endpoints**  
  - Add `/api/orchestrations` (start) and `/api/orchestrations/:id` (status) that proxy to the LangGraph orchestrator with appropriate config.  
7.2 **Callbacks / logging**  
  - Register `LangGraphTroubleshootHandler` (or similar) to log events with `service=langgraph_orchestrator`.  
  - Enable LangSmith tracing for the orchestrator graph.  
7.3 **Metrics**  
  - Emit counters for runs started/completed/failed, average duration per stage, vendor cap utilization.

## 8. Testing
8.1 **Unit tests**  
  - Mocked LLM nodes, ensuring strategy decisions and progress reports parse correctly.  
  - Node-level tests for normalization, SSIC fallback, plan Top-10, enrichment iteration (with fake subgraphs).  
8.2 **Integration tests**  
  - Run the orchestrator graph end-to-end with stubbed subgraphs to verify state flow, checkpoint/resume.  
  - Chat integration tests: trigger orchestration via router and confirm status messages appear.  
8.3 **Load / soak tests**  
  - Simulate large candidate batches to measure runtime, vendor cap adherence, concurrency overhead.  
8.4 **Regression**  
  - Ensure existing chat flows still succeed when orchestrator runs concurrently (no DB contention, no duplicate enrichments).

## 9. Rollout
9.1 **Docs & training**  
  - Update `workflow.md`, ops runbooks, and developer notes to describe the new orchestrator graph and LLM nodes.  
9.2 **Phased launch**  
  - Phase 1: cron/CLI only.  
  - Phase 2: enable chat-triggered runs for internal users.  
  - Phase 3: full production rollout with monitoring alarms.  
9.3 **Cleanup**  
  - Deprecate old imperative orchestrator code once traffic fully migrates.  
  - Remove temporary env flags when LLM nodes prove stable.
