---
owner: Codex Agent – Frontend Generator
status: draft
last_reviewed: 2025-04-06
---

# Feature PRD — Pre-SDR Onboarding Conversation Enhancements

## Summary
Improve the opening and early-intake portions of the pre-SDR chat agent so that users receive a proactive greeting, can validate the captured company profile, and confirm the accuracy of generated micro-ICPs (including identifying anti-ICP segments) before enrichment begins. The experience must stay approachable for non-technical sales operators by explaining any domain jargon inline.

## Background & Motivation
Early user feedback highlights uncertainty about where to start, whether the agent understood their business, and how to avoid pursuing off-target lead lists. The existing workflow waits for a user-issued command before providing structure, and confirmation moments are limited to the ICP lock-in step. By adding an opening greeting, a company-profile confirmation gate, and a micro-ICP accuracy check (plus anti-ICP capture), we can increase user confidence and reduce downstream rework.

## Goals
- Surface an automatic greeting message when a session starts, outlining the ICP intake flow and supported commands.
- After collecting the company URL and inferred profile, prompt users to verify or correct the summary before proceeding.
- Once micro-ICPs are generated, confirm the relevance with the user, capture anti-ICP guidance, and remind them how to continue.
- Maintain a clear, jargon-light tone with inline explanations when specialized terms are unavoidable.

## Non-Goals
- Reworking enrichment logic or scoring models beyond conversational prompts.
- Implementing multilingual copy or locale detection.
- Adding new commands beyond the existing lead-gen controls.

## Users & Personas
- **Revenue operations managers** who curate ICP definitions for SDR teams.
- **SDR/BDR leads** who need confidence that generated lead lists exclude poor-fit companies.

## User Stories
1. *As an SDR lead,* when I open the chat I want an immediate greeting that explains the process so I know what to do next.
2. *As a RevOps manager,* after the agent drafts a company profile from my website I want to confirm or adjust it so that downstream recommendations stay on target.
3. *As an SDR lead,* when micro-ICPs appear I want to approve the fit, disqualify segments, and specify anti-ICP traits before enrichment runs.

## Functional Requirements & Acceptance Criteria
1. **Greeting Banner**
   - Display the provided greeting message on initial session load before any user input.
   - Ensure the message includes command reminders exactly as specified.
   - Acceptance: QA transcript shows greeting appears without user prompt; no duplicate greeting within the same session.
2. **Company Profile Verification**
   - After the agent ingests the company URL and generates a profile summary, present the summary back to the user with a verification prompt.
   - Allow the user to confirm accuracy or provide corrections/steering instructions (free text or structured follow-up questions).
   - Acceptance: Test conversation demonstrates the agent waiting for confirmation or edits before proceeding to micro-ICP synthesis.
3. **Micro-ICP Confirmation & Anti-ICP Capture**
   - Once micro-ICPs are proposed, ask the user to confirm correctness of the segments and highlight any misses.
   - Prompt explicitly for anti-ICP characteristics or example companies to exclude.
   - Acceptance: Transcript includes an explicit yes/no (or feedback) exchange plus capture of anti-ICP notes prior to enabling enrichment commands.
4. **Tone & Jargon Guidance**
   - System prompts and assistant messages should default to plain language; whenever terms like “ICP” or “enrichment” are used, include a short explanation or parenthetical definition.
   - Acceptance: Content review of prompt templates shows clarifying phrasing accompanying unavoidable jargon.

## Conversation Flow Updates
1. Session start → Greeting message renders → Agent waits for user input.
2. User provides website URL (plus optional context) → Agent synthesizes company profile → Agent displays summary with verification question → Branches to collect corrections as needed.
3. Agent generates micro-ICPs → Agent presents segments with explanations → Agent asks for accuracy confirmation and anti-ICP details → On confirmation, unlocks enrichment commands.

## Metrics & Success Criteria
- Increase completion rate of ICP intake sessions by ≥15% relative to baseline week.
- Reduce percentage of enrichment runs canceled or restarted due to wrong-fit segments by ≥20% within two weeks of launch.
- Qualitative feedback: ≥80% of surveyed users agree that onboarding guidance is “clear and easy to follow.”

## Dependencies
- Existing pre-SDR chat graph prompt templates and state transitions (`app/pre_sdr_graph.py`, `src/conversation_agent.py`).
- Copy review from product marketing for tone validation.
- Analytics instrumentation to track new confirmation events (if not already available).

## Risks & Mitigations
- **Risk:** Greeting fires repeatedly on reconnects during the same session.
  - **Mitigation:** Guard the greeting behind a session flag persisted in conversation state.
- **Risk:** Additional confirmation steps slow power users.
  - **Mitigation:** Provide concise confirmation prompts with quick “confirm” shortcuts and allow users to skip by affirming immediately.
- **Risk:** Anti-ICP capture increases prompt length and token usage.
  - **Mitigation:** Summarize anti-ICP notes before storing; reuse structured fields where possible.

## Rollout Strategy
- Launch behind a feature flag targeting internal testers first.
- Collect sample transcripts for tone review and iterate on copy.
- Expand to a subset of production tenants after copy sign-off, then roll out globally once success metrics trend positively.

## Decisions
1. **Anti-ICP Data Persistence** — Persist anti-ICP guidance alongside micro-ICP rules in the existing datastore so future sessions automatically inherit exclusions, even though this requires a schema update.
2. **Greeting Content Scope** — Keep the greeting text-only and command-focused, reserving documentation links for later tooltips or help commands to avoid cluttering the opening exchange.

