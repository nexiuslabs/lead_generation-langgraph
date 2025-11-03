# To‑Do Tracker — PRD‑20 Interactive ICP + Bug Fixes

Status legend: [ ] Not Started · [~] In Progress · [x] Done · [!] Blocked

## Milestones
- [~] Phase 1 (Dev): SSE + events wiring (flag gating pending)
- [ ] Phase 2 (Staging): internal tenants enabled, collect P50/P90
- [ ] Phase 3 (Prod): per‑tenant rollout, legacy fallback if no SSE

## Implementation Tasks

1) Event plumbing
- [x] Create `src/chat_events/emitter.py` (in‑proc pub/sub)
- [x] Add `src/chat_events/__init__.py` with `emit(event, message, context)`
- [x] Emit `icp:intake_saved`
- [x] Emit `icp:seeds_mapped`
- [x] Emit `icp:confirm_pending`
- [x] Emit `icp:planning_start`
- [x] Emit `icp:toplikes_ready`
- [x] Emit `icp:profile_ready`
- [x] Emit `icp:candidates_found`
- [x] Emit `enrich:start_top10`
- [x] Emit `enrich:company_tick` (×10)
- [x] Emit `enrich:summary`

2) SSE endpoint
- [x] Add `app/chat_stream.py` (GET `/chat/stream/{session_id}`)
- [x] Register route in `app/main.py`
- [x] Auth guard (optional identity; tenant scope on request state)
- [x] Handle disconnects and bounded backpressure
- [ ] Dev sanity check with simple SSE client

3) Controllers for confirm/enrichment
- [x] Add `POST /icp/chat/confirm` (sets session gate, emits planning_start)
- [x] Reuse `/icp/enrich/top10` and add per‑company emits
- [~] Ensure interactive path uses fast‑head enrichment only (policy in place; additional caps optional)

3a) Reflection hooks (interactive only)
- [ ] Implement `src/reflect.py` with `reflect(state, metrics)`
- [ ] Apply tuned decisions (HEAD, timeouts, crawl/pages, apify variant)
- [ ] Ensure session‑scoped only, no persistence

4) Performance pins
- [ ] Honor `INTERACTIVE_HEAD=12`
- [~] Honor `JINA_TIMEOUT_S=6` (subpages 6s; homepage 8s)
- [x] Honor `LLM_MAX_CHUNKS=3`
- [ ] Honor `CRAWL_MAX_PAGES=6`
- [x] Single Apify attempt per company (interactive Top‑10)

5) Bug fixes & audits
- [x] Guard duplicate vendor calls (state `apify_attempts` >= 1)
- [ ] Initialize locals in Apify branch (avoid `UnboundLocalError`)
- [x] Normalize LinkedIn regional hosts to `www.linkedin.com`
- [~] Standardize `httpx` timeouts and close paths (fallbacks capped; audit globals pending)
- [ ] Sanitize URL expansions (drop `*`, `javascript:`, `mailto:`, `tel:`)
- [x] Clamp LLM chunk size/count; add trimmed retry path
- [x] Extract LinkedIn company URL from site pages (home/about/contact) and persist
- [x] Site contacts: parse About/Contact/home, upsert into `contacts`, mirror emails to `lead_emails`, verify
- [x] Use login tenant id (no `DEFAULT_TENANT_ID`) in chat flow
- [x] Stash Top‑10 shortlist in thread memory to avoid re‑discovery on `run enrichment`
- [x] Next‑40 always enqueues after Top‑10 (persist remainder to staging; emit enqueued event)

6) QA & docs
- [ ] Write dev runbook: tail SSE and validate milestones
- [ ] Update README: “PRD‑20 Interactive ICP Flow”

## Testing Checklist
- [ ] Unit: EventEmitter pub/sub
- [ ] Unit: SSE auth/stream handler
- [ ] Unit: Confirm controller gating
- [ ] Integration: Full chat flow event ordering
- [ ] Integration: Top‑likes ≤50 with why + snippet
- [ ] Integration: Enrichment ticks stream and summary
- [ ] Non‑regression: Nightly ACRA local
- [ ] Non‑regression: Next‑40 enqueue path
- [ ] Non‑regression: ACRA Direct script

## Observability
- [ ] Emit per‑stage counts/durations
- [ ] Emit `progress_summary` messages
- [~] Degraded vendor notes visible to user (reasons persisted; UI wiring pending)

## Config & Flags
- [ ] `ENABLE_CHAT_EVENTS=true` (wire gating in endpoints)
- [ ] `APIFY_LOG_SAMPLE_SIZE=3` (dev)
- [ ] `APIFY_DEBUG_LOG_ITEMS=true` (dev)

## Acceptance Criteria (tracking)
- [ ] Scripted chat matches PRD‑20 wording
- [ ] Event latency ≤300ms emit→display
- [ ] Top‑likes P50 ≤35s; Top‑10 first 3 enrich ≤90s P50
- [ ] No regressions in Nightly ACRA, Next‑40, ACRA Direct
- [ ] Reflection adapts at least once when constrained
