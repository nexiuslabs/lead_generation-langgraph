# Smart Multiple AI Agentic Workflows Implementation — PRD‑20 Interactive ICP Flow

## Scope & Objectives
- Align ICP conversational flow to PRD‑20 script without altering existing agents or background jobs.
- Add realtime chat progress via SSE and thin controllers.
- Speed up interactive path with fast‑path controls and bounded vendor usage.
- Harden enrichment path and fix listed bugs while preserving reliability.

## Deliverables
- Event plumbing (`EventEmitter` + emit hooks) with compact JSON payloads.
- SSE endpoint (`GET /chat/stream/:session_id`) with auth and tenant scoping.
- Controllers for confirm/enrichment: `POST /icp/chat/confirm` and reused `/icp/enrich_top10` with emits.
- Agentic reflection helper applied only to live interactive sessions.
- Performance controls via config knobs (env defaults documented below).
- Bug fixes: vendor call guard, locals init, LinkedIn normalization, httpx timeout standardization, URL sanitization, LLM chunk bounds.
- QA runbook and README updates for the interactive ICP flow.

## Architecture & Integration Points
- Event emission: in‑proc pub/sub attached near existing logging hooks; back‑pressure is best‑effort (drop if no listeners).
- SSE streaming: FastAPI streaming response; one writer per client; no persistence; auto‑reconnect supported by client.
- Controllers: thin, non‑intrusive wrappers that gate planning and emit progress; no changes to core agents.
- Reflection: pure function `reflect(state, metrics) -> decisions` to tune session‑scoped parameters only.

## Rollout Phases
- Phase 1 (Dev): implement SSE + emits; behind `ENABLE_CHAT_EVENTS`.
- Phase 2 (Staging): enable for internal tenants; collect latency P50/P90 and adjust knobs.
- Phase 3 (Prod): gradual per‑tenant enablement; fallback to legacy chat if SSE not connected.

## Implementation Breakdown (by Task)

1) Event plumbing
- Add `chat_events` module with `emit(event, message, context)` and in‑proc `EventEmitter`.
- Wire emits at milestones:
  - `icp:intake_saved`, `icp:seeds_mapped`, `icp:confirm_pending`, `icp:planning_start`, `icp:toplikes_ready`, `icp:profile_ready`, `icp:candidates_found`.
  - Enrichment: `enrich:start_top10`, `enrich:company_tick`, `enrich:summary`.
- Files to touch:
  - `src/chat_events/__init__.py`, `src/chat_events/emitter.py` (new)
  - Wrap emits within: `app/icp_endpoints.py`, `src/agents_icp.py`, `src/icp_pipeline.py`, `src/enrichment.py` (non‑invasive)
- Acceptance:
  - Events log in dev; payload `{event, message, context}` stable and JSON‑serializable.

2) SSE endpoint
- Add `GET /chat/stream/:session_id` that streams server‑sent events with `text/event-stream`.
- Auth: validate session/tenant; limit to single active writer per client; drop silently on disconnect.
- Files to touch:
  - `app/routes/chat_stream.py` (new), `app/main.py` (register route), `app/auth.py` (reuse middleware)
- Acceptance:
  - SSE client can connect and receive sample events within ≤300ms from emit.

3) Controllers for confirm/enrichment
- Add `POST /icp/chat/confirm` to flip a session flag and emit `icp:planning_start`.
- Reuse `/icp/enrich_top10`; add emits around company boundaries; ensure head‑only enrichment in interactive path.
- Files to touch:
  - `app/icp_endpoints.py` (controller), `src/agents_icp.py` (planner entry), `src/enrichment.py` (ticks/summary emits)
- Acceptance:
  - “confirm” triggers planning; “run enrichment” streams per‑company ticks.

3a) Agentic reflection hooks (interactive only)
- Implement `reflect(state, metrics) -> decisions` to tune: `HEAD`, `JINA_TIMEOUT_S`, `CRAWL_MAX_PAGES`, `LLM_MAX_CHUNKS`, `apify_chain_variant`.
- Apply per‑session; never persist; do not alter prompts or background runners.
- Files to touch: `src/reflect.py` (new), light calls from controllers before invoking nodes.
- Acceptance: logs show at least one adaptive decision under constrained conditions.

4) Performance pins
- Ensure env defaults are honored for interactive path:
  - `INTERACTIVE_HEAD=12`, `JINA_TIMEOUT_S=6`, `LLM_MAX_CHUNKS=3`, `CRAWL_MAX_PAGES=6`.
- Verify single Apify attempt per company in Top‑10 interactive path.
- Acceptance: Top‑likes P50 ≤35s; enrichment first 3 companies ≤90s P50.

5) Bug fixes and audits
- Guard duplicate vendor calls with `state["apify_attempted"]`.
- Initialize locals in Apify branch to avoid `UnboundLocalError`.
- Normalize LinkedIn hosts to `www.linkedin.com/company/...` while accepting regional variants.
- Standardize `httpx` timeouts and ensure all connection close paths are covered.
- Sanitize URL expansions (drop `*`, `javascript:`, `mailto:`, `tel:`).
- Clamp LLM chunk sizes/count and enable trimmed retry path.
- Acceptance: no regressions; vendor errors render as degraded notes not hard failures.

6) QA & docs
- Runbook: how to tail SSE in dev and verify milestone ordering.
- README: add “PRD‑20 Interactive ICP Flow”.
- Test plan alignment (see below).

## Config Knobs (env)
- `ENABLE_CHAT_EVENTS=true`
- `INTERACTIVE_HEAD=12`
- `JINA_TIMEOUT_S=6`
- `LLM_MAX_CHUNKS=3`
- `CRAWL_MAX_PAGES=6`
- `APIFY_LOG_SAMPLE_SIZE=3` (dev)
- `APIFY_DEBUG_LOG_ITEMS=true` (dev)

## Testing Plan (from PRD)
- Unit: event emitter, SSE auth/streaming, controller gating.
- Integration: full chat flow (test tenant); assert event order; Top‑likes length ≤50 including why + snippet.
- Non‑regression: nightly job local run (unchanged), Next‑40 enqueue path, ACRA Direct.

## Observability
- Emit counts/durations per stage; summarize as `progress_summary` events.
- Expose vendor cap exhaust as degraded user‑visible notes.

## Risks & Mitigations
- Vendor throttling → degrade gracefully; fallbacks; communicate via events.
- Long enrichment → stream progress; allow exit while Next‑40 continues.
- SSE disconnects → client auto‑reconnect; server stateless emit continues.

## Timeline & Milestones (suggested)
- Day 1–2: EventEmitter + SSE endpoint basic; dev behind flag.
- Day 3: Controllers + emits for planning and toplikes/profile events.
- Day 4: Enrichment ticks + reflection helper; performance pins.
- Day 5: Bugfix sweep, QA runbook, README updates; staging enable.

## Out of Scope (per PRD)
- No changes to core agent prompts/logic or background jobs (Nightly ACRA, Next‑40, ACRA Direct).

## Smart Agentic Workflow

### State Machine (Sense → Plan → Act → Reflect → Adapt)
- States:
  - `intake_saved`: user ICP answers + seeds persisted; emit `icp:intake_saved`.
  - `seeds_mapped`: seeds bound to companies/ACRA; emit `icp:seeds_mapped`.
  - `confirm_pending`: user prompt to “confirm”; emit `icp:confirm_pending`.
  - `planning`: discovery + toplikes/profile planning; emit `icp:planning_start`.
  - `toplikes_ready`: ≤50 candidates with why/snippets; emit `icp:toplikes_ready`.
  - `profile_ready`: ICP profile produced; emit `icp:profile_ready`.
  - `candidates_found`: N≥1 candidates; emit `icp:candidates_found` with N.
  - `enrichment_running`: Top‑10 enrichment streaming; emits `enrich:*` ticks.
  - `summary_emitted`: enrichment summary; emit `enrich:summary`.

- Transitions:
  - `intake_saved → seeds_mapped` (auto when mapping completes)
  - `seeds_mapped → confirm_pending` (auto)
  - `confirm_pending → planning` (on `POST /icp/chat/confirm`)
  - `planning → toplikes_ready → profile_ready → candidates_found` (sequential milestones)
  - `candidates_found → enrichment_running` (on user "run enrichment")
  - `enrichment_running → summary_emitted` (after Top‑10 ticks)

### Session State (ephemeral, per interactive chat)
- Keys:
  - `session_id`, `tenant_id`, `user_id`
  - `confirmed` (bool), `enrichment_requested` (bool)
  - `budgets`: `{vendor_caps, time_budget_ms}`
  - `perf`: `{head, jina_timeout_s, llm_max_chunks, crawl_max_pages}`
  - `metrics`: evidence density, candidate_count, contact_fill_rate, vendor_timeouts, retries
  - `flags`: `{apify_attempted: bool, apify_variant: "domain"|"profile"}`
  - `events_emitted`: set for idempotent emit

### Event Payload
- Shape: `{ event, message, context }`
- `context` includes: `session_id`, `tenant_id`, `state`, minimal counters (durations, counts), optional IDs.

### Decision Policies (heuristics)
- Evidence density low (< 1.2 tokens/char in first 2 pages) → increase `head` 8→12 once; if still low → reduce `llm_max_chunks` to 2 and skip long crawl.
- Candidate yield low (< 10) → allow 1 extra domain expansion pass from best evidence; cap at ≤50.
- Vendor health degraded (timeouts ≥2 or rate‑limit) → halve `jina_timeout_s`; skip Tavily; attempt Apify only once.
- Contacts missing and `linkedin_url` present → try Apify employees (maxItems 25–35) once; else reflect to profile variant once.
- Early stop: if `linkedin_url` + ≥2 decision‑maker contacts found → short‑circuit further contact discovery in interactive path.

### Orchestration Pseudocode
```python
def orchestrate_session(state):
    emit("icp:intake_saved", msg("Received your ICP answers and seeds. Normalizing and saving…"))
    map_seeds(state)  # sets seeds_mapped
    emit("icp:seeds_mapped", msg("Anchoring seeds and extracting SSIC codes…"))
    emit("icp:confirm_pending", msg("I’ll crawl your site + seed sites… Reply ‘confirm’ to proceed."))

def on_confirm(state):
    state.confirmed = True
    emit("icp:planning_start", msg("Confirmed. Gathering evidence and planning Top‑10…"))
    plan_toplikes_and_profile(state)

def plan_toplikes_and_profile(state):
    decisions = reflect(state.metrics)
    apply_perf(state, decisions)
    toplikes = agents_icp.plan_top10_with_reasons(head=state.perf.head, jina_timeout=state.perf.jina_timeout_s)
    emit("icp:toplikes_ready", msg("Top‑listed lookalikes (with why) produced."), {"count": len(toplikes)})
    profile = icp_pipeline.winner_profile(evidence=state.evidence)
    emit("icp:profile_ready", msg("ICP Profile produced."))
    emit("icp:candidates_found", msg(f"Found {len(toplikes)} ICP candidates. We can enrich 10 now…"), {"count": len(toplikes)})
    state.toplikes = toplikes
    state.profile = profile

def run_enrichment(state):
    emit("enrich:start_top10", msg("Enriching Top‑10 now (require existing domains)."))
    for idx, company in enumerate(state.toplikes[:10]):
        tick = enrich_company_fast(company, state)
        emit("enrich:company_tick", {"index": idx, "result": tick.status, "company": company.domain})
    emit("enrich:summary", msg(summary_from_ticks(state)))

def enrich_company_fast(company, state):
    # Jina-first deterministic pages, then light HTTP, optional Tavily (flagged), Apify once if contacts missing
    pages = jina_pages(company.domain, head=state.perf.head, timeout=state.perf.jina_timeout_s)
    info = extract_llm(pages, max_chunks=state.perf.llm_max_chunks)
    if missing_contacts(info):
        if not state.flags.apify_attempted:
            info = try_apify(company, variant=state.flags.apify_variant)
            state.flags.apify_attempted = True
    return info

def reflect(metrics):
    decisions = {}
    decisions["head"] = 12 if metrics.evidence_density_low else 8
    decisions["jina_timeout_s"] = 6 if metrics.vendor_timeouts else 5
    decisions["llm_max_chunks"] = 3 if metrics.evidence_density_ok else 2
    decisions["crawl_max_pages"] = 6 if metrics.candidate_yield_low else 4
    decisions["apify_variant"] = "profile" if metrics.employees_zero else "domain"
    return decisions
```

### Guardrails & Failure Handling
- Single‑shot vendor attempts in interactive flow; further expansion deferred to background jobs.
- Timeouts standardized; retries at most 1; emit degraded notes instead of raising.
- Idempotent emits: dedupe by `{session_id, event}`.
- URL sanitation for expansion; drop `javascript:`, `mailto:`, `tel:`, wildcards.

### Interfaces (existing code touchpoints)
- `app/icp_endpoints.py`: add confirm controller; wrap emits around intake, mapping, enrichment.
- `src/agents_icp.py`: accept perf params; leave core logic unchanged.
- `src/icp_pipeline.py`: expose `winner_profile` as used; no logic changes.
- `src/enrichment.py`: emit per‑company ticks; enforce interactive caps and single Apify attempt.
- `src/chat_events/*`: new emitter + simple pub/sub.
- `src/reflect.py`: new lightweight policy engine.

## Multi‑Agent Topology & Coordination

### Agent Roles (cooperating agents)
- `discovery_planner`: plans domains to evaluate; consumes intake and seed evidence; outputs candidate domains with rationale.
- `evidence_extractor`: retrieves Jina pages + light HTTP, normalizes text, scores density; outputs snippets/features.
- `profile_synthesizer`: aggregates evidence to produce ICP profile (winner_profile wrapper).
- `ranker_scorer`: merges planner + evidence signals to rank ≤50 lookalikes with “why”.
- `enrichment_agent`: performs deterministic company snapshot + optional Tavily enrichment.
- `contacts_resolver`: resolves LinkedIn URL and fetches employees via Apify, normalizes contacts, verifies emails.
- `reflection_governor`: computes decisions and budgets from metrics; feeds per‑agent param overrides.
- `orchestrator_controller`: session brain; routes tasks, enforces state machine, emits chat events.
- `watchdog`: monitors vendor timeouts/rate limits; triggers degraded mode and guardrails.

### Coordination Pattern
- Blackboard/event‑bus hybrid:
  - Blackboard (shared session store) holds `state`, `metrics`, `artifacts` (evidence, candidates, profile, enrichment ticks).
  - Event bus carries fine‑grained progress and hand‑off signals (`agent:*` and `icp:*` events), streamed to chat via SSE.
- Arbitration: `reflection_governor` adjusts agent parameters per step; `watchdog` can preempt with degraded mode.

### Task Graph (DAG) per Session
- Nodes: Intake → SeedMap → Plan → Extract → Rank → Profile → Candidates → (Enrich Top‑10 → Contacts) → Summary
- Parallelism:
  - `Extract` can fan‑out across candidate domains with concurrency cap.
  - `Enrich Top‑10` runs per company with strict concurrency (e.g., 2–3 in flight) to stay under budgets.
- Cutpoints (milestones → chat events): after SeedMap, after Rank (Top‑likes), after Profile, after Candidates, per enrichment tick, final summary.

### Inter‑Agent Protocol
- Message schema: `{type, agent, payload, session_id, ts, correlations:{company_id?, domain?}}`.
- Types: `task.start`, `task.done`, `task.error`, `handoff.ready`, `metrics.update`, `decision.update`.
- Reliability: at‑least‑once on the internal bus; idempotent consumers via correlation keys.

### Scheduling & Budgets
- Global per‑session caps: `{head, jina_timeout_s, llm_max_chunks, crawl_max_pages, max_concurrency_extract=3, max_concurrency_enrich=2}`.
- Budget policies:
  - If `watchdog.vendor_throttled`: set `max_concurrency_enrich=1`, disable Tavily, keep Apify single‑shot only.
  - If `metrics.evidence_density_low`: temporarily increase `head` to 12; cap `llm_max_chunks` at 2 for speed.
  - If `candidate_count<10`: allow one light expansion iteration in `discovery_planner`.

### Multi‑Agent Orchestrator (pseudocode)
```python
def run_session(session: State):
    post(icp_intake_saved())
    blackboard.write("state", session)

    # 1) Seed mapping (serial)
    seeds = discovery_planner.map_seeds(session)
    blackboard.write("seeds", seeds)
    bus.emit(ev("icp:seeds_mapped"))

    # 2) Wait for confirm
    wait_for_confirm(session)
    bus.emit(ev("icp:planning_start"))

    # 3) Plan + Extract + Rank + Profile (hybrid parallel)
    decisions = reflection_governor.decide(metrics=metrics.snapshot())
    apply_overrides(decisions)

    candidates = discovery_planner.plan(session, head=cfg.head)
    # Fan-out evidence extraction with concurrency cap
    evid = parallel_map(candidates, evidence_extractor.fetch, max_concurrency=cfg.max_concurrency_extract)
    ranked = ranker_scorer.rank(candidates, evid)
    bus.emit(ev("icp:toplikes_ready", {"count": len(ranked)}))

    profile = profile_synthesizer.summarize(evid)
    bus.emit(ev("icp:profile_ready"))
    bus.emit(ev("icp:candidates_found", {"count": len(ranked)}))

    blackboard.write("toplikes", ranked[:50])
    blackboard.write("profile", profile)

    # 4) Enrichment Top-10 (controlled concurrency)
    if session.enrichment_requested:
        bus.emit(ev("enrich:start_top10"))
        enrich_results = parallel_map(ranked[:10], enrichment_agent.snapshot,
                                      max_concurrency=cfg.max_concurrency_enrich)
        # Contacts resolution only when still missing
        pending_contacts = [r for r in enrich_results if missing_contacts(r)]
        contacts = parallel_map(pending_contacts, contacts_resolver.resolve, max_concurrency=1)
        summarize_and_emit(enrich_results, contacts)
```

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

### Failure Isolation & Degradation
- Isolate slow/failed vendors per agent; continue best‑effort using available artifacts.
- Convert exceptions to `task.error` with degraded notes; orchestrator never blocks the whole session.
- Apply “once per session” constraints for vendor chains (Apify) to keep latency predictable.

### Observability for Multi‑Agent Runs
- Correlate spans by `session_id` and `domain`; count per‑agent durations and success rates.
- Emit `progress_summary` periodically with aggregate metrics for UI and telemetry.
