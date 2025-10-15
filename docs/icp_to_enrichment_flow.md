# ICP to Enrichment Flow

This document summarizes how the chat agent guides a user from the ICP intake stage through enrichment and scoring, and why enrichment can be blocked when the Top-10 shortlist is missing.

## 1. ICP Intake Wizard
- When the user issues `start lead gen`, the agent enters the ICP wizard. With **ICP Finder** enabled, it collects the website URL first, then requires at least five best-customer seeds before proceeding. The agent acknowledges these prompts explicitly in chat and stores them in `state["icp"]`.【F:app/pre_sdr_graph.py†L842-L877】
- Once the required inputs are present, optional synthesizers generate an `icp_profile` (industries, titles, size bands, integrations, triggers) that gets shown back to the user before requesting confirmation.【F:app/pre_sdr_graph.py†L878-L914】

## 2. Confirming the ICP and Persisting Context
- The `confirm` command persists the intake payload (website + seeds) and the synthesized profile for reuse. It also triggers discovery planning so subsequent stages have the same data even if the chat session restarts.【F:app/pre_sdr_graph.py†L924-L1108】
- During confirmation, the agent previews candidate counts and the Top-10 lookalike table, storing it in `state["agent_top10"]` and in persistent storage via `_persist_top10_preview` so enrichment can reuse the same list later.【F:app/pre_sdr_graph.py†L1054-L1108】

## 3. Micro-ICP Suggestions and Gating
- After confirmation the router holds at the candidates stage until Finder-generated micro-ICP suggestions are available and the user accepts one (e.g., `accept micro-icp 1`). This selection persists a focused rule and sets `state["micro_icp_selected"] = True`, unlocking enrichment.【F:app/pre_sdr_graph.py†L4431-L4442】【F:app/pre_sdr_graph.py†L4584-L4620】

## 4. Enrichment Command Flow
- When the user types `run enrichment`, the router ensures candidates exist and then invokes the enrichment node. The immediate batch is limited (≈10) while additional candidates are queued for nightly processing.【F:src/conversation_agent.py†L58-L64】【F:app/pre_sdr_graph.py†L4410-L4420】【F:app/pre_sdr_graph.py†L1453-L1487】
- Enrichment updates company rows, extracts evidence, runs lead scoring, and prepares contact data for downstream systems, with fallbacks for Odoo export and nightly queues.【F:app/pre_sdr_graph.py†L1453-L1500】【F:app/pre_sdr_graph.py†L1422-L1448】
## 5. Why Enrichment Fails Without a Stored Top-10
- The enrichment node first tries to reuse the in-memory `agent_top10` list or reloads a persisted Top-10 preview. If neither is available (for example, the user skipped confirmation or the state was reset), it refuses to run enrichment and prompts the user to type `confirm` to regenerate the shortlist.【F:app/pre_sdr_graph.py†L1240-L1343】
- This safeguard prevents the system from enriching arbitrary companies without the curated shortlist that discovery produced, ensuring consistency between what was shown in chat and what gets enriched.【F:app/pre_sdr_graph.py†L1327-L1343】

## 6. Why the Top-10 Might Disappear After Being Displayed
- `agent_top10` lives in the conversation state. Each new user message hydrates state from the checkpoint; if the checkpointer does not retain the list (for example, on a cold start or storage reset), the next run begins without the in-memory shortlist even though the user saw it in chat.【F:app/pre_sdr_graph.py†L1048-L1108】
- When that happens the enrichment node falls back to `_load_persisted_top10`, but this loader is a no-op if no tenant identifier can be resolved (`DEFAULT_TENANT_ID`, JWT, or Odoo mapping).【F:app/pre_sdr_graph.py†L355-L420】【F:app/pre_sdr_graph.py†L555-L607】
- In local testing where no tenant is configured, the preview is shown but cannot be reloaded on the next turn, so enrichment reports that the Top-10 is missing even though it was previously displayed. Setting a tenant id (or running in an environment where one is inferred) lets `_persist_top10_preview` save the list and `_load_persisted_top10` retrieve it for enrichment, eliminating the mismatch.【F:app/pre_sdr_graph.py†L1048-L1108】【F:app/pre_sdr_graph.py†L555-L607】

## 7. End-to-End User Flow Summary
1. `start lead gen` → Agent asks for website, then 5–15 seed customers.【F:app/pre_sdr_graph.py†L842-L877】
2. Agent synthesizes the ICP profile and instructs the user to `confirm`.【F:app/pre_sdr_graph.py†L878-L914】
3. `confirm` → Persist intake, show candidates, Top-10 table, and micro-ICP suggestions.【F:app/pre_sdr_graph.py†L924-L1108】
4. `accept micro-icp N` → Persist micro-ICP, unlock enrichment.【F:app/pre_sdr_graph.py†L4584-L4620】
5. `run enrichment` → Enrich the stored Top-10 candidates, score leads, and queue nightly follow-ups. Enrichment is blocked if the Top-10 shortlist is missing until the user re-confirms.【F:app/pre_sdr_graph.py†L1240-L1487】【F:app/pre_sdr_graph.py†L1327-L1343】

This flow matches the behaviour observed in the provided conversation: after running discovery, the agent could not find the locked Top-10 because the state was reset before enrichment, so it asked the user to `confirm` again before allowing `run enrichment` to proceed.【F:app/pre_sdr_graph.py†L1327-L1343】
