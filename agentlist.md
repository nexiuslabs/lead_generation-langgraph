# LLM Agents & Prompts

| # | Agent | Purpose | Prompt |
|---|-------|---------|--------|
| 1 | `_structure_company_profile` | Extract structured company info from a website snippet and user notes | ```text
You extract structured company profile data for a lead-generation workflow.
Website: <url>
Website snippet: <text>
Recent user snippets: [...]
Return JSON with keys:
  summary (<=2 sentences string)
  industries (list of up to 3 industry labels)
  offerings (list of up to 4 products/services)
  ideal_customers (list of up to 4 buyer/company descriptors)
  proof_points (list of up to 4 metrics/awards/quotes)
If a field is unknown, return an empty string or empty list.
``` |
| 2 | `_summarize_company_site` | Summarize a company website (and infer name) from scratch | ```text
You draft concise company profiles for a lead-generation assistant.
Website: <url>
Recent user snippets: [...]
Website content snippet: <text>
Return JSON {"summary": "<=2 sentence description", "name": "Short name if mentioned"}.
If you lack info, infer based on the domain (e.g., Nexius Labs) and invite the user to refine.
``` |
| 3 | `_generate_icp_from_customers` | Produce an ICP profile from five customer websites | ```text
You are an ICP strategist. Analyze these existing customer websites and summarize the ideal customer profile for a lead-generation campaign.
Customers: [...]
Return JSON with keys: summary, industries, company_sizes, regions, pains, buying_triggers, persona_titles, proof_points.
``` |
| 4 | `_simple_intent` (fallback) | Classify intent when the main ingest call fails | ```text
Classify the user's intention for the lead-generation orchestrator.
Return JSON {"intent": one of [run_enrichment, confirm_icp, confirm_company, accept_micro_icp, question, chat]}.
Text: <user message>
``` |
| 5 | `_maybe_explain_question` | Detect and answer inline user questions | ```text
You are a lead-generation assistant. Determine if the user is asking a question and, if so, answer it in ≤2 sentences.
Return JSON {"is_question": true/false, "answer": "..."}.
User text: <normalized question>
``` |
| 6 | `ingest_message` | Normalize chat text and classify intent/tags | ```text
You normalize chat inputs for a lead-generation orchestrator.
Respond with JSON {"normalized_text": "...", "intent": "...", "tags": [...]}.
Intent options: run_enrichment, confirm_company, confirm_icp, accept_micro_icp, question, chat, idle.
Input: <user message>
``` |
| 7 | `profile_builder` | Maintain company + ICP confirmations and summaries | ```text
You maintain company + ICP profile confirmations and draft summaries from chat snippets.
Return JSON with keys:
  - company_profile_confirmed (bool)
  - icp_profile_confirmed (bool)
  - micro_icp_selected (bool)
  - icp_discovery_confirmed (bool)
  - company_profile (object with website/name/summary fields)
  - icp_profile (object summarizing the ICP when provided)
History: [...]
Current profile: [...]
``` |
| 8 | `decide_strategy` | Choose discovery strategy (`use_cached`, `regenerate`, `ssic_fallback`) | ```text
Choose discovery action (use_cached, regenerate, ssic_fallback).
Return JSON {"action": "...", "reason": "..."}.
Candidate count: <n>
Last SSIC attempt: <timestamp or None>
``` |
| 9 | `progress_report` | Summarize backend progress for the user | ```text
Summarize the orchestration status for the user in ≤2 sentences using plain language.
Focus on next steps instead of internal phase names.
State: {"phase": ..., "candidates": ..., "web_candidates": ..., ...}
Return JSON {"message": "..."}
``` |
