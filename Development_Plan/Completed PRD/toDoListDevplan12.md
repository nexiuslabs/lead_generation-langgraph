# TODO — Dev Plan 12: Agent Chat UI (Next.js)

Source: featurePRD12.md and featureDevPlan12.md

Legend
- [ ] Pending
- [~] In progress
- [x] Done
- [!] Blocked / Needs decision

## Core Plumbing (already present)
- [x] Stream provider with health-check and toast de-dup (`Stream.tsx`)
- [x] Thread provider and history panel
- [x] Login/Signup pages (cookie auth) and token exchange
- [x] FirstLoginGate onboarding probe and Verify Odoo
- [x] useAuthFetch with silent refresh and X-Tenant-ID

## New UX (MVP)
- [x] Add ShortlistStatusBadge to `header-bar.tsx` (poll `/shortlist/status`)
- [x] Add ExportButtons (CSV/JSON) in `header-bar.tsx`
- [x] Add ConnectionBadge in `header-bar.tsx`
- [x] Wire settings form defaults and persistence (already in `StreamProvider`)

## Multi‑Tenant Threads & History
- [x] Require known tenant to fetch history in `ThreadProvider` (no tenant ⇒ skip fetch)
- [x] Use top‑level `{ graph_id | assistant_id }` + `metadata.tenant_id` in `threads.search`
- [x] Pre‑create tenant‑scoped thread before first submit (metadata.tenant_id + graphId)
- [x] On `onThreadId`, PATCH thread to set `metadata.tenant_id` if missing
- [x] Persist `tenant_id` to `localStorage` on login, remove on sign‑out
- [x] Switch Next.js API proxies (`/api`, `/api/backend`) to Node runtime for dev stability
- [!] Optional (server): set `LANGGRAPH_ALLOW_ANON=false` in prod; validate `X-Tenant-ID` and thread metadata server‑side

## Optional (Post-MVP)
- [ ] Thread actions: delete/export transcript via SDK/API
- [ ] Role-aware UI gating (viewer vs ops) for exports and ICP edits
- [ ] Inline status panel component instead of compact badge

## Testing
- [ ] Dev: proxy ON, LangGraph on 8001 — verify connection badge, status, exports
- [ ] Dev: proxy OFF, direct URL — verify cookies still sent (same-origin assumption)
- [ ] Auth: silent refresh on 401 path exercised (e.g., expire access cookie)
- [ ] SSE: error toasts dedup and stream reconnection
- [ ] Multi‑tenant: verify isolation with two tenants (A cannot see B’s threads)
- [ ] Logout/Login: created threads remain visible for the same tenant across sessions
- [ ] No double thread creation: at most one POST /api/threads per new chat; metadata present

## Docs
- [ ] Add a short “Using Status & Exports” section to agent-chat-ui README
- [ ] Note envs: `NEXT_PUBLIC_API_URL`, `NEXT_PUBLIC_USE_API_PROXY`, `NEXT_PUBLIC_ASSISTANT_ID`
- [ ] Document multi‑tenant behavior: storage, isolation, and troubleshooting (legacy threads without `tenant_id`)

## Open Items
- [ ] Placement of badges vs a mini status panel (product call)
- [ ] Export limit defaults; add query UI or keep coded?
- [ ] Consider surfacing last run status and run_id in the badge tooltip
