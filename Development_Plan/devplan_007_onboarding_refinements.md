---
owner: Codex Agent – Frontend Generator
status: draft
last_reviewed: 2025-04-06
---

# Dev Plan — Pre-SDR Onboarding Conversation Enhancements

## 1. Scope Alignment
- **Feature PRD Reference:** `Development_Plan/featurePRD_007_onboarding_refinements.md`.
- **Master PRD Threads:** Greeting activation, company profile verification checkpoint, micro-ICP confirmation with anti-ICP capture.
- **Out of Scope:** Downstream enrichment scoring, new commands, or multi-language support.

## 2. System Architecture & Touchpoints
1. **LangGraph Workflow (`src/conversation/lead_gen_graph.py`, `src/flows/pre_sdr_flow.py`)**
   - Insert greeting node triggered on session bootstrap when no prior turns exist.
   - Extend graph edges to branch through company profile verification and micro-ICP confirmation stages.
2. **Prompt Templates (`src/prompts/system/pre_sdr.yaml`, `src/prompts/tooling/*.yaml`)**
   - Update system prompt to enforce friendly tone and jargon explanations.
   - Add reusable prompt snippets for verification and anti-ICP capture.
3. **State Management (`src/state/session.py`)**
   - Persist flags: `greeting_sent`, `company_profile_confirmed`, `micro_icp_confirmed`, `anti_icp_notes`.
   - Ensure persistence across reconnects within same conversation.
4. **API Surface (`app/api/routes/conversation.py`)**
   - No new endpoints; ensure payload includes new state fields when serializing conversation context.
5. **Analytics / Telemetry (`src/metrics/events.py`)**
   - Emit events for `company_profile_verified`, `company_profile_adjusted`, `micro_icp_confirmed`, `anti_icp_logged`.

## 3. Detailed Design
### 3.1 Greeting Injection
- On session start, check `state.greeting_sent`. If false, queue greeting message before awaiting user input.
- Guard against duplicate greetings by setting flag immediately after emission.
- Ensure conversation transcript stores greeting as system message for auditing.

### 3.2 Company Profile Verification Loop
- After company URL ingestion and profile synthesis node completes, present summary + verification prompt.
- Capture user response; if corrections are supplied, feed them into profile refinement tool and loop until user confirms.
- Store sanitized profile delta for telemetry.

### 3.3 Micro-ICP Confirmation & Anti-ICP Capture
- Following micro-ICP generation, assemble confirmation message enumerating segments with plain-language descriptors.
- Await explicit confirmation (`confirm`, `looks good`, etc.) or feedback to adjust segments.
- Introduce follow-up question for anti-ICP: request examples or traits; normalize into structured dict before storing.
- Block enrichment commands until confirmation flag true.

### 3.4 Tone & Copy Governance
- Expand prompt scaffolding to remind agent to define jargon inline (e.g., "ICP (ideal customer profile)").
- Provide copy variants for verification prompts with simple language and helper text.

### 3.5 Error Handling & Edge Cases
- If company URL missing or invalid, skip verification step but log warning event.
- For returning sessions with existing anti-ICP data, surface a reminder to confirm or amend stored exclusions.
- Rate-limit repeated clarification loops to avoid infinite cycles; after 3 unsuccessful attempts, escalate with fallback message directing user to human support.

## 4. Data & Schema Considerations
- Extend conversation state schema (likely JSONB) to include new fields; confirm compatibility with persistence layer defined in `app/models/conversation_state.py`.
- For anti-ICP persistence decision, add column or nested property in `icp_rules` table / document as determined by existing storage.
- Ensure migrations update Supabase/Postgres where relevant; add default values to avoid null-handling issues.

## 5. Tooling & Integration Updates
- Update any unit tests that assume immediate ICP flow without greeting (`tests/conversation/test_flow.py`).
- Adjust mock transcripts in docs or fixtures to include new checkpoints.
- Coordinate with marketing for copy review; track as TODO if not finalized before implementation.

## 6. Testing Strategy
- **Unit Tests:**
  - Validate graph transitions enforce greeting + verification gates.
  - Test state flag persistence and gating logic for enrichment commands.
- **Integration Tests:**
  - Simulate full conversation transcripts covering confirmation loops and anti-ICP capture.
  - Verify telemetry events emitted with correct payloads.
- **Regression Tests:**
  - Ensure legacy flows (users who skip URL or micro-ICP stage) still progress with guardrails.
- **Manual QA:**
  - Scripted walkthrough in staging verifying tone guidelines and single-greeting behavior.

## 7. Risks & Mitigations
- **Risk:** Additional steps frustrate advanced users.
  - *Mitigation:* Provide quick-confirm paths and highlight `confirm` command.
- **Risk:** Anti-ICP storage conflicts with existing schema.
  - *Mitigation:* Prototype migration in feature branch; validate against staging data snapshot.
- **Risk:** Copy drifts from tone requirements post-review.
  - *Mitigation:* Document approved phrases and include lint check or test snapshot on prompt templates.

## 8. Open Questions & Follow-Ups
1. **Anti-ICP Data Sync Scope**
   - **Option A — Keep data internal to onboarding flow:** Simpler implementation; avoids propagating partially vetted exclusions to external systems. *Recommendation.*
   - **Option B — Sync anti-ICP to CRM exports:** Ensures downstream users see exclusions but increases schema coupling and compliance review.
   - **Next Step:** Confirm with RevOps stakeholders whether CRM visibility is required before build freeze.
2. **Telemetry Payload Content**
   - **Option A — Store sanitized summaries:** Reduces risk of capturing PII while still allowing trend analysis. *Recommendation.*
   - **Option B — Store raw free-text corrections:** Provides full fidelity for qualitative review but raises privacy concerns and storage costs.
   - **Next Step:** Align with data governance lead on sanitization rules and update analytics schema accordingly.

## 9. Timeline & Milestones
- **Day 1:** Implement greeting node + state flags.
- **Day 2:** Build verification and confirmation loops with anti-ICP capture.
- **Day 3:** Update prompts, telemetry, and regression tests; run full suite.
- **Day 4:** Conduct copy review, finalize documentation, prep release notes.

