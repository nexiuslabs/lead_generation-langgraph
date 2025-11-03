**Feature PRD 12 — UX: Agent Chat UI (Next.js)**

- **Objective:** Provide a responsive, role-aware chat interface to drive enrichment, visualize progress, and export results — backed by SSO and tenant isolation.
- **Primary Users:** Ops (edit ICP, trigger enrichments, export), Viewers (inspect shortlist, rationales).

**Architecture**
- **Framework:** Next.js (app router) with client components, Tailwind UI.
- **LangGraph SDK:** Streaming via `@langchain/langgraph-sdk/react` `useStream`; thread search via SDK client.
- **Providers:**
  - `StreamProvider` (`src/providers/Stream.tsx`): sets API URL, assistant ID, API key; manages health-check and stream context.
  - `ThreadProvider` (`src/providers/Thread.tsx`): loads threads with `client.threads.search()` using `graph_id` or `assistant_id`.
- **Proxy:** Edge routes forward to LangGraph/FastAPI with cookies and optional `X-Api-Key`:
  - `/api/[..._path]` and `/api/backend/[..._path]` → base from `LANGGRAPH_API_URL` or `NEXT_PUBLIC_API_URL`.

**Configuration**
- **Environment:**
  - `NEXT_PUBLIC_API_URL`: LangGraph base (e.g., `http://localhost:8001`).
  - `NEXT_PUBLIC_API_BASE`: Optional UI base; falls back to `NEXT_PUBLIC_API_URL` when unset.
  - `NEXT_PUBLIC_ASSISTANT_ID`: Assistant/graph ID (e.g., `agent`).
  - `NEXT_PUBLIC_USE_API_PROXY`: If `true`, client requests are rewritten to `/api/backend/*` to share cookies.
  - `NEXT_PUBLIC_USE_AUTH_HEADER`: Dev-only; if `true`, attach `Authorization: Bearer <idToken>`.
  - `NEXT_PUBLIC_ENABLE_TENANT_SWITCHER`: Dev-only; enables local tenant override.
  - `LANGSMITH_API_KEY`: Injected server-side by proxy for deployed graphs.
- **Settings Form:** When `apiUrl` or `assistantId` missing, `StreamProvider` renders a form to collect Deployment URL, Assistant/Graph ID, and optional LangSmith API key. Values persist in the URL query (`apiUrl`, `assistantId`) and localStorage (`lg:chat:apiKey`).

**Auth & Tenant Context**
- **Cookies:** Backend sets `nx_access`/`nx_refresh`; UI uses cookies by default. Dev-only auth header guarded by `NEXT_PUBLIC_USE_AUTH_HEADER`.
- **Silent Refresh:** `useAuthFetch` auto-calls `/auth/refresh` on 401 once, then retries the original request.
- **Tenant Propagation:**
  - Session-derived tenant ID (from NextAuth if present) or dev override `lg:chat:tenantId` (when enabled) is sent via `X-Tenant-ID` for both chat streams and REST calls.
  - Header bar shows “Signed in as <email> (tenant: <id>[, override])”.

**Login/Signup & Onboarding**
- **Pages:** `/login`, `/signup` post to backend `/auth/*` (proxied when enabled) and rely on cookies; no passwords stored in UI.
- **First-Login Gate:** `FirstLoginGate` kicks off `/onboarding/first_login`, then polls `/onboarding/status` and probes `/session/odoo_info` and `/onboarding/verify_odoo` until ready, showing progress text per stage.
- **Verify Odoo:** Header bar button calls `/onboarding/verify_odoo` and surfaces results.
- **Logout:** Keycloak logout with `post_logout_redirect_uri` and `id_token_hint`; fallback to `/login`.

**Connection Health & Resilience**
- **Health-Check:** `checkGraphStatus` first calls `/api/info` (or `{apiUrl}/info`), then falls back to `/assistants?limit=1`. Results are cached in-session to prevent duplicate toasts.
- **Toast De-dup:** SessionStorage key `lg:chat:connToast:<apiUrl>` avoids double alerts from React Strict Mode.
- **Stream Errors:** `Thread` listens to `stream.error` and shows a single toast per unique error message.
- **SSE:** Stream reconnects handled by SDK; UI shows “AssistantMessageLoading” until first token.
 - **Connection Indicator:** Header displays a compact badge (`Connected`/`Offline`) based on the health-check.

**Chat & Threads**
- **Thread State:** `threadId` in query string; history panel via `ThreadHistory` toggled with slide animation.
- **Submit:** `Thread` composes a human message plus any uploaded content blocks and calls `stream.submit({ messages, context })` where `context` includes `tenant_id`.
- **Tool Responses:** `ensureToolCallsHaveResponses` pre-fills missing tool results to keep the UI stable.
- **File Upload:** Drag-and-drop and paste supported; content blocks preview rendered before submit.

**Roles**
- **Viewer:** Read-only permissions in the chat and exports.
- **Ops/Admin:** ICP edits (via chat commands), ad-hoc enrich, costs visibility. UI passes through commands; backend enforces authorization.

**Routing & Params**
- **Query Keys:** `apiUrl`, `assistantId`, `threadId`, `chatHistoryOpen`, `hideToolCalls`.
- **Local Storage:** `lg:chat:apiKey`, `lg:chat:tenantId`.

**Status & Export Surfaces**
- **Status Panel:** Polls `GET /shortlist/status` to display `{ tenant_id, total_scored, last_refreshed_at }` (when enabled in the header area/panel).
- **Exports:** CSV/JSON export triggered via backend endpoints (UI wiring planned); include `tenant_id` and run metadata.

**Chat History Storage**
- **Persistence Layer:**
  - Use LangGraph Server’s built‑in thread/checkpoint storage. In local/dev, persist to the filesystem directory specified by `LANGGRAPH_CHECKPOINT_DIR` (default `.langgraph_api`). In production, run LangGraph Server with a durable store (same API), or back up the directory on a persistent volume.
- **Scope & Security:**
  - Custom auth (`app/lg_auth.py`) binds identity to tenant (e.g., `tenant:<id>`). LangGraph isolates threads by identity; cross‑tenant access is blocked server‑side.
- **Thread Metadata & Search:**
  - Threads are created by the SDK and tagged so they can be searched via `client.threads.search`. The UI filters by `{ graph_id: <assistantId> }` or `{ assistant_id: <uuid> }` (see `providers/Thread.tsx`).
  - The UI shows up to the last 100 threads in the history panel.
- **Message Retention:**
  - Default retention 90 days for threads/messages in dev; production retention configurable by deployment policy. Provide a periodic cleanup job (server‑side) that deletes threads older than the retention window.
- **User Operations:**
  - View past threads in the sidebar; clicking loads messages via the SDK.
  - “New thread” clears `threadId` to start a new conversation; a fresh thread is created on next submit.
  - Future (optional): Add “Delete thread” and “Export transcript (JSON/TXT)” actions calling the LangGraph threads API.
  - **Resilience & Backups:**
  - Ensure the checkpoint directory/volume is included in backups. On restore, threads reappear in history for the same identity.

**Multi‑Tenant History**
- **Tenant Isolation:** Each tenant sees only threads with `thread.metadata.tenant_id == currentTenantId`. The UI:
  - Sends `X-Tenant-ID: <tenantId>` on all SDK/REST calls (stream + history).
  - Searches with top‑level `{ graph_id | assistant_id }` AND `metadata: { tenant_id: <tenantId> }`.
  - Requires a known tenant to fetch history; when unknown, shows only the active thread optimistically and avoids cross‑tenant queries.
- **Thread Creation (No Duplicates):**
  - Pre‑create a thread with `metadata.tenant_id` and `graphId=<assistant>` before the first submit to avoid SDK auto‑create race.
  - If the SDK still emits a new `threadId`, immediately PATCH the thread’s `metadata.tenant_id` to the current tenant (best‑effort) so it remains visible in history.
- **Persistence Across Login:**
  - Persist `tenant_id` in localStorage (`lg:chat:tenantId`) on login and read it on app load; clear it on sign‑out.
  - History must show the tenant’s threads immediately after logout/login cycles.
- **Storage Backing:**
  - Dev: LangGraph local_dev runtime is in‑memory unless configured; use `LANGGRAPH_CHECKPOINT_DIR` to persist to disk (default `.langgraph_api`).
  - Prod: Run LangGraph Server with a durable checkpointer (filesystem volume/DB) so chat history survives restarts.
- **Security (Server‑Side):**
  - Custom auth binds identity to tenant (e.g., `tenant:<id>`).
  - Recommended: server middleware validates that read/write requests include `X-Tenant-ID` matching the authenticated tenant and that thread access respects `metadata.tenant_id`.
- **Edge Cases:**
  - Legacy threads missing `metadata.tenant_id` will not appear under tenant filters. Provide a one‑off backfill to set `tenant_id` on those threads if needed.

**Proxy Stability (Dev)**
- Use Node runtime for Next.js API proxies (`/api` and `/api/backend`) to prevent intermittent “fetch failed” errors observed on Edge in local dev.

**Accessibility & Performance**
- **Keyboard:** Press Enter to send; Shift+Enter for newline.
- **A11y:** Labels on switches/inputs; icons wrapped in tooltips where appropriate.
- **Perf:** Avoid heavy work on render; content virtualization not required for current message volumes.

**Acceptance Criteria**
- **Connection:** Health-check falls back to `/assistants` when `/info` requires auth; no duplicate alerts in dev.
- **Auth:** Cookie-mode works end-to-end with silent refresh; dev auth header path functional when enabled.
- **Tenant:** Header shows correct email/tenant; `X-Tenant-ID` forwarded on stream and REST requests; dev override works only when enabled.
- **Onboarding:** First-login gate advances through states and unblocks when Odoo is ready; Verify Odoo button reports accurate state.
- **Chat:** Submits messages and streams responses; thread history loads; error toasts are deduped.
- **Settings:** Missing config shows settings form; values persist in query/local storage.
- **Export/Status:** Status reads from `/shortlist/status`; exports include run metadata and are downloadable.
- **History:** Threads persist across reloads; last 100 threads listed; selecting a thread restores its messages. In dev, data survives process restarts if `.langgraph_api` persists.
- **Multi‑Tenant History:** New threads appear immediately in the sidebar and remain visible after logout/login. History shows only the current tenant’s threads; no cross‑tenant leakage. SDK double‑create is prevented (pre‑create) or healed (PATCH metadata). Proxies do not flake in dev.

**Open Questions**
- **Dashboards:** Inline basic charts vs link to Metabase.
- **ICP Edits:** Dedicated form alongside chat vs chat-only commands.

**User Journey**
- **1) Sign up / Sign in**
  - User visits `/signup` to create an account or `/login` to sign in. The UI posts to backend `/auth/*` (proxied if enabled) and stores session cookies. Header shows “Signed in as <email> (tenant: <id>)”.
- **2) First login provisioning**
  - `FirstLoginGate` automatically calls `/onboarding/first_login`, then polls `/onboarding/status` and probes Odoo endpoints until ready. While provisioning, the UI displays messages like “Creating Odoo DB…” and “Seeding baseline entities…”. Once ready, the chat unlocks.
- **3) Configure connection (if needed)**
  - If `NEXT_PUBLIC_API_URL` or assistant ID is missing, the settings form appears. The user enters Deployment URL (e.g., `http://localhost:8001`) and Assistant/Graph ID (e.g., `agent`) and clicks Continue.
- **4) Start a chat**
  - The chat loads. A connectivity check runs (using `/api/info` then `/assistants`). If reachable, the user can type commands. Thread history can be opened via the sidebar toggle; “New thread” resets the conversation.
  - **5) Command examples**
  - Example A: “Show today’s shortlist (top 10).” The UI sends the message with `tenant_id` in context. The backend returns a streaming response with the list; the UI renders items as the assistant message.
  - Example B: “Refresh ICP to software, 10–200 employees; process 10 now and schedule the rest nightly.” The backend starts an ad-hoc partial run. The UI shows stage chips as updates arrive (crawl → extract → verify → score). On completion, the assistant confirms: “Processed 10 now; remaining scheduled for tonight’s run.”
  - **6) Export and verify**
  - The user can trigger an export (CSV/JSON) from the export control (when wired). The header’s “Verify Odoo” button checks tenant Odoo connectivity and shows a brief status note.
  - Past chats remain available under Thread History (left sidebar). Selecting a past thread restores the conversation; “New thread” starts a fresh one.

**Implementation Highlights (from Dev Plan)**
- **Header components:**
  - `ShortlistStatusBadge` in `agent-chat-ui/src/components/ui/header-bar.tsx` polls `/shortlist/status` every 30s.
  - `ExportButtons` provides CSV/JSON downloads via `/export/latest_scores.{csv,json}?limit=500`.
  - `ConnectionBadge` shows `Connected`/`Offline` using the same health-check as `StreamProvider`.
- **Providers & Proxies:**
  - `StreamProvider` uses LangGraph SDK `useStream`, manages API URL/assistant ID and optional LangSmith API key, and emits de-duped connection toasts.
  - Edge routes `/api/[..._path]` and `/api/backend/[..._path]` forward to the configured base and attach cookies and optional `X-Api-Key`.

**Testing Plan**
- Dev with `NEXT_PUBLIC_USE_API_PROXY=true`, LangGraph on 8001: verify connection badge, status polling, and downloads.
- Proxy OFF (direct URL): ensure cookies flow and chat streams.
- Auth: silent refresh path triggers on 401 and recovers request.
- SSE: error toasts are deduped; stream reconnects and shows loading until first token.

**Rollout Steps**
1) Land header status components and wire into `HeaderBar`.
2) Verify in dev with proxy ON and standalone OFF.
3) Add brief docs to README (envs and where to find status/exports).
4) Optionally add role gating for exports (feature flag).

**Optional Enhancements**
- Thread actions: Delete thread, Export transcript via SDK/API.
- Role-aware UI (hide export for viewer, etc.).
- Inline mini-dashboards or links to Metabase.
