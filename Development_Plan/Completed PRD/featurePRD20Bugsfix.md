# PRD-20: ICP Finder Conversational Alignment, Fast-Path Enrichment, Realtime Chat Updates, and Bugfixes

## Summary
- Align the ICP finder to the provided chat script while preserving existing conversation agents and background services (Nightly ACRA, BG worker Next‑40, ACRA Direct).
- Deliver a fast, predictable flow: Intake → Evidence → ICP Profile → Top‑likes (≤50) → Enrich Top‑10 → Live progress in chat.
- Add realtime chat progress updates, harden the pipeline, and resolve minor bugs that affect speed and stability.

## Goals
- Conversational alignment: mirror the sample prompts/responses and intermediate confirmations.
- Performance: reduce end‑to‑end latency for the interactive path to under 35s P50 for Top‑10 planning; under 90s P50 for Top‑10 enrichment kickoff (first results streaming).
- Reliability: make the interactive flow resilient; never block on non‑critical vendor failures.
- Non‑intrusive: DO NOT alter existing conversation agents’ prompts/logic or background services (Nightly ACRA, BG Next‑40, ACRA Direct). Only add thin controllers, hooks, and status events.

## Non‑Goals / Out of Scope
- No changes to the underlying ICP agents (discovery_planner, evidence_extractor) beyond wiring status events.
- No changes to nightly jobs, Next‑40 scheduler, or ACRA Direct paths.
- No front‑end redesign; only add SSE/WebSocket event consumption and a small chat transcript adapter.

## User Experience & Chat Script Alignment
Script to reproduce verbatim (system messages originate from server events):

1) Agent → “Let’s infer your ICP from evidence. What’s your website URL?”
2) User → “https://www.theblackknight.org/”
3) Agent → “List 5–15 best customers (Company — website). Optionally 2–3 lost/churned with a short reason.”
4) User → provides list (5–15)
5) Agent → “I’ll crawl your site + seed sites, run discovery, then propose a Top‑10 lookalikes with evidence. ACRA runs later in the nightly SG pass. Reply ‘confirm’ to proceed.”
6) User → “confirm”
7) Agent (progressive updates):
   - “Confirmed. Gathering evidence and planning Top‑10…”
   - “Top‑listed lookalikes (with why) produced.”
   - “ICP Profile produced.”
   - “Progress: Intake saved → Evidence → Domain resolve → Evidence → Top‑10 ✓”
   - “Found 12 ICP candidates. We can enrich 10 now; nightly runner will process the rest. Accept a micro‑ICP, then type ‘run enrichment’.”
8) User → “run enrichment”
9) Agent → “Enriched results (8/10 completed). Remaining queued.” + present ICP Profile + ranked list (≤50) and the Top‑10 selection.

Notes:
- After user submits best customers, the flow is: ICP Profile → Top‑likes (max 50) → enrich Top‑10. ACRA evidence for SG runs nightly; do not trigger ACRA Direct here.

## System Flow (Mapped to Existing Code)
- Intake save: `app/icp_endpoints.py:post_intake` + `src/icp_intake.py.save_icp_intake` (answers + seeds) [unchanged logic].
- Evidence mapping: `icp_intake.map_seeds_to_evidence` and `icp_pipeline.collect_evidence_for_domain` (Jina homepage text) [interactive path only for user‑provided seeds, bounded].
- Top‑likes (≤50): `src/agents_icp.plan_top10_with_reasons` (discovery → Jina snippets → extract → score)
- ICP Profile: `src/icp_pipeline.winner_profile` (frequency across evidence)
- Top‑10 enrichment: `app/icp_endpoints.enrich_top10` → `src/enrichment.enrich_company_with_tavily(search_policy='require_existing')`
- Contacts (Top‑10): `node_apify_contacts` domain→company→employees chain (immediate `companies.linkedin_url` upsert) [unchanged algorithm]
- Background Next‑40: maintained as‑is after Top‑10 kick.

## Enrichment: Data Crawl & Extraction (Priority Order)
- Objective: Extract firmographic details quickly and consistently from candidate websites using a Jina‑first strategy, then fallback when needed; if contact persons remain missing, use Apify to discover people.

Priority 1 — Jina (deterministic content retrieval):
- Always fetch key pages via Jina Reader proxy first to avoid heavy crawling and JavaScript rendering issues.
- For a candidate domain (e.g., `https://www.theblackknight.org/`), construct and fetch:
  - `https://r.jina.ai/https://www.theblackknight.org/`
  - `https://r.jina.ai/https://www.theblackknight.org/contact-us`
  - `https://r.jina.ai/https://www.theblackknight.org/about`
- Parse returned text (strip boilerplate) and feed into the existing LLM extraction chain to populate: `about_text`, `hq_city`, `hq_country`, `website_domain`, `linkedin_url` (if present), `email[]`, `phone_number[]`, `tech_stack[]`, and other schema keys.
- Limit total characters per page and cap the number of pages to keep latency bounded. Use reflection to reduce the set when content density is low.

Fallback — HTTP (lightweight) then vendor (optional):
- If Jina returns empty or times out, fetch the same URLs via direct HTTP GET with strict timeouts (connect ≈2–3s, read ≈6–8s), then run the same extraction chain.
- Optional vendor extract (Tavily) only if enabled and within session budget; batch small (≤3 in parallel) with short timeouts.

Contacts resolution — Apify when still missing:
- After Jina/HTTP extraction, if contacts are still missing (e.g., no named contacts and zero verified emails):
  - Resolve LinkedIn company URL from the known domain via Apify domain→company actor and upsert immediately to `companies.linkedin_url`.
  - Fetch employees via `harvestapi~linkedin-company-employees` (interactive `maxItems` ≈ 25–35), normalize and upsert contacts, verify emails.
  - If employees return 0, reflect once to try the profile‑search variant; stop after a single fallback attempt.
- Keep attempts single‑shot in chat sessions to preserve responsiveness; expand only in background jobs.

Notes:
- This priority scheme does not alter background flows; it only changes the interactive chat path to be Jina‑first, fast, and deterministic.

## Agentic Workflow & Reflection (Self‑Improvement)
- Pattern: Sense → Plan → Act → Reflect → Adapt, without modifying core agent prompts/logic.
- Controllers orchestrate agent calls and collect outcomes; reflection tunes strategy per session/tenant via configuration knobs (no schema changes, no background job changes).

Agentic loop per interactive session:
- Plan: choose next milestone given state (intake saved, seeds mapped, evidence density, candidate count). Prefer fast‑path actions (Jina head, capped crawl, single vendor attempt) when caps or latency budgets constrain.
- Act: invoke existing agents/pipeline nodes (discovery_planner, evidence_extractor, enrichment nodes) with bounded parameters (HEAD sizes, timeouts, max pages) and emit progress events.
- Reflect: evaluate results using simple, objective signals:
  - Evidence quality: snippet length, presence of integrations/titles, ICP token matches.
  - Candidate yield: number of domains found, score distribution, percent gated out.
  - Enrichment completeness: pages extracted, keys filled (about_text, linkedin_url), contact count, named decision‑makers, email verification rate.
  - Vendor health: timeouts, rate limits, cap exhaustion, retries taken.
- Adapt: adjust subsequent steps for the same session:
  - Increase/Reduce Jina HEAD (e.g., 8↔12), lower timeouts when latency budget is tight.
  - Toggle Apify chain variant when employees=0 (fallback to profile queries) once per company.
  - Thin crawl: reduce `CRAWL_MAX_PAGES` and `LLM_MAX_CHUNKS` when prior chunks yielded little signal.
  - Early stop: skip redundant steps when reflection shows sufficient data (e.g., linkedin_url present + enough contacts).

Reflection memory (no schema change):
- Use existing `company_enrichment_runs.degraded_reasons` and vendor counters to aggregate session outcomes.
- Persist minimal session summary in logs and emit a compact `progress_summary` event for UI and telemetry.
- Optional (future): if needed, add a lightweight in‑memory cache keyed by `tenant_id` to bias HEAD/timeout defaults on subsequent sessions (no DB schema change).

Acceptance hooks:
- Gate each major phase with a reflection check (pass/fail + rationale) to decide next action or early finish.
- Emit reflection summaries as chat events, e.g., “Reduced HEAD to 8 due to vendor timeouts; proceeding with scoring”.

## Realtime Chat Updates (Server → UI)
- Add a lightweight event stream for the interactive session without altering agent logic:
  - Event bus: wrap existing observability points to emit progress events (labels below).
  - Transport: Server‑Sent Events (SSE) endpoint `GET /chat/stream/:session_id` (simple and browser‑friendly). Optional WS in future.
  - Correlation: include `session_id` and `tenant_id` on requests; store minimal session context in memory (or ephemeral KV).

Event labels and when to emit:
- `icp:intake_saved` (after `save_icp_intake`) → “Received your ICP answers and seeds. Normalizing and saving…”
- `icp:seeds_mapped` (after `map_seeds_to_evidence`) → “Anchoring seeds… Added SSIC evidence where available.”
- `icp:confirm_pending` (after seeds submitted, before heavy work) → “Reply ‘confirm’ to proceed.”
- `icp:planning_start` (before `plan_top10_with_reasons`) → “Confirmed. Gathering evidence and planning Top‑10…”
- `icp:toplikes_ready` (after planner) → “Top‑listed lookalikes (with why) produced.”
- `icp:profile_ready` (after `winner_profile`) → “ICP Profile produced.”
- `icp:progress_summary` → “Progress: Intake saved → Evidence → Domain resolve → Evidence → Top‑10 ✓”
- `icp:candidates_found` (with N) → “Found N ICP candidates. We can enrich 10 now…”
- `enrich:start_top10` (before Top‑10) → “Enriching Top‑10 now (require existing domains).”
- `enrich:company_tick` (per company) → short per‑company status (pages fetched, LinkedIn resolved, contacts saved)
- `enrich:summary` → “Enriched results (X/10 completed). Remaining queued.”

Implementation details:
- Implement an `EventEmitter` (in‑proc) tied to existing `_obs_log`/logger hooks; emit a compact JSON payload `{event, message, context}`.
- Back‑pressure: drop events if no listener; do not persist long‑term.
- Security: require auth; restrict sessions by tenant.

## Latency Targets & Fast‑Path Controls
- Discovery/Top‑likes (interactive):
  - HEAD size for Jina reads: 12
  - Jina timeout per domain: 5–6s with single retry
  - Skip long crawl; rely on Jina snippets for planning (already implemented)
- Enrichment (interactive Top‑10 head):
  - Deterministic snapshot first; cap Tavily usage and retries
  - LLM extraction: reduce chunk count (config) and enforce strict timeouts
  - Apify employees: run only once per company; log small samples; fallback chain behind a flag

Configuration knobs (envs, no logic change):
- `INTERACTIVE_HEAD=12`, `JINA_TIMEOUT_S=6`, `LLM_MAX_CHUNKS=3`, `CRAWL_MAX_PAGES=6`
- `APIFY_LOG_SAMPLE_SIZE=3`, `APIFY_DEBUG_LOG_ITEMS=true` (dev‑only)

## Bugfix/Hardening Items (without altering agents/background logic)
- Guard duplicate vendor calls in Top‑10 path with `state["apify_attempted"]`.
- Initialize locals before fallbacks in Apify branch (avoid `UnboundLocalError` on `raw`/`contacts_raw`).
- Accept regional LinkedIn hosts (`*.linkedin.com/company/...`) when resolving and normalize to `www`.
- Ensure DB connections closed in all code paths (double‑close safe guards already in place; audit critical nodes).
- Timeouts and retries: standardize `httpx` timeouts and prevent runaway awaits; log degrads not exceptions.
- Sanitize URL list (drop `*`, javascript:, mailto:, tel:) during expansion.
- Clamp LLM chunk size and count to avoid context blowups; enable trimmed retry path.
- Make daily/vendor caps visible in progress messages when throttled.

## Data & Persistence
- No schema changes. Existing tables suffices: `icp_intake_responses`, `customer_seeds`, `icp_evidence`, `staging_global_companies`, `companies`, `contacts`, `lead_emails`, `company_enrichment_runs`.
- Optional: ephemeral in‑memory map `{session_id → tenant_id}` for SSE authorization.

## API Additions (thin controllers)
- `POST /icp/chat/confirm` → sets a session flag that allows `plan_top10_with_reasons` to start and emits `icp:planning_start`.
- `GET /chat/stream/:session_id` → SSE stream of `{event, message, context}`.
- Note: Do NOT change existing `/icp/intake`, `/icp/suggestions`, `/icp/accept`, `/icp/enrich/top10` semantics.

## Acceptance Criteria
- Chat flow renders the exact sample prompts and confirmations.
- After “confirm”, events appear in near‑real time (≤300ms server emit → client display) for each milestone.
- Top‑likes list (≤50) produced within ≤35s P50; includes why + snippet.
- “run enrichment” streams per‑company ticks; at least first 3 companies complete within 90s P50.
- No regressions in Nightly ACRA, BG Next‑40, or ACRA Direct.
- Agentic reflection is active: logs and chat events show at least one adaptive decision (e.g., HEAD/timeout/crawl reduction or chain fallback) when conditions warrant.

## Observability
- Emit counts and durations at each stage into existing logs and as summarized `progress_summary` messages.
- Capture vendor cap exhaust as user‑visible degraded notes.

## Rollout Plan
- Phase 1 (Dev): implement SSE endpoint, wire event emission, gated via `ENABLE_CHAT_EVENTS`.
- Phase 2 (Staging): enable for internal tenants; collect latency P50/P90.
- Phase 3 (Prod): enable per‑tenant; keep fallback to legacy chat (no streaming) if SSE not connected.

## Risks & Mitigations
- Vendor throttling → Show degraded messages; continue best‑effort with fallbacks.
- Long‑running enrichment → Stream progress and allow user to exit while Next‑40 runs in background (unchanged).
- SSE disconnects → auto‑reconnect; server keeps emitting without storing (stateless).

## Implementation Tasks
1) Event plumbing
- Add `EventEmitter` (simple pub/sub) and a `chat_events` module with `emit(event, message, context)`.
- Wrap key points with emits (see Event labels).

2) SSE endpoint
- Add `GET /chat/stream/:session_id` to FastAPI; keep one writer per client; flush as events occur.
- Auth middleware to validate tenant and session.

3) Controllers for confirm/enrichment
- `POST /icp/chat/confirm` to gate the planner start; return 202 and emit `icp:planning_start`.
- Reuse existing `/icp/enrich_top10` for enrichment; emit per‑company ticks in node boundaries.

3a) Agentic reflection hooks (interactive path only)
- Add a small `reflect(state, metrics) -> decisions` helper that:
  - Reads evidence density, candidate counts, vendor caps/timeouts.
  - Returns tuned parameters for the next step (HEAD, timeouts, crawl/pages, apify_chain_variant).
- Apply decisions only to the live session (in‑memory), never altering agent prompts or background runners.

4) Performance pins
- Ensure config defaults for interactive path: `INTERACTIVE_HEAD`, `JINA_TIMEOUT_S`, `LLM_MAX_CHUNKS`, `CRAWL_MAX_PAGES`.
- Verify early Apify call in Top‑10 path and single attempt guard.

5) Bug fixes and audits
- Verify locals initialized in Apify branch; regional LinkedIn normalization preserved.
- Confirm connection close paths; standardize httpx timeouts.

6) QA & docs
- Add runbook: how to tail SSE in dev and validate milestones.
- Update README section “PRD‑20 Interactive ICP Flow”.

## Testing Plan
- Unit: event emitter, SSE endpoint auth/stream, controller gating.
- Integration: full chat flow on a test tenant; assert event order and timing; assert Top‑likes length ≤50 with why and snippet.
- Non‑regression: run nightly job locally (unchanged), Next‑40 enqueue path, ACRA Direct script.

---

## Sample Event Sequence (for the provided conversation)
- icp:intake_saved → “Received your ICP answers and seeds. Normalizing and saving…”
- icp:seeds_mapped → “Anchoring seeds to company records and ACRA. Extracting SSIC codes…”
- icp:confirm_pending → “I’ll crawl your site + seed sites… Reply ‘confirm’ to proceed.”
- icp:planning_start → “Confirmed. Gathering evidence and planning Top‑10…”
- icp:toplikes_ready → “Top‑listed lookalikes (with why) produced.”
- icp:profile_ready → “ICP Profile produced.”
- icp:candidates_found (N=12) → “Found 12 ICP candidates. We can enrich 10 now…”
- enrich:start_top10 → “Enriching Top‑10 now (require existing domains).”
- enrich:company_tick (×10) → per company updates
- enrich:summary → “Enriched results (8/10 completed). Remaining queued.”

This design aligns the process to the sample conversation, accelerates interactive latency, and adds realtime chat updates without modifying existing agent logic or background services.
