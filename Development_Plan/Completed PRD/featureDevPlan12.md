# Dev Plan — Feature 12: Agent Chat UI (Next.js)

Source: featurePRD12.md

Objective: Deliver a robust, tenant-aware chat UI backed by SSO and the LangGraph SDK, with clear connection status, shortlist status, export actions, onboarding gate, and thread history. Keep cookies as the primary auth mechanism; support dev-only overrides.

---

## Architecture & Surfaces

- Providers
  - `StreamProvider` (LangGraph SDK `useStream`): streams messages to/from LangGraph; sets API URL and Assistant/Graph ID; optionally accepts a LangSmith API key; manages a connectivity health-check (with fallback) and stream context; de‑dupes connection toasts.
  - `ThreadProvider` (SDK client): loads recent threads by `graph_id`/`assistant_id` via `client.threads.search()` and provides a history list.
- Proxies
  - `/api/[..._path]` and `/api/backend/[..._path]`: edge forwarders that attach cookies and optional `X-Api-Key`.
- Auth
  - Cookie mode by default; dev-only Authorization header via `NEXT_PUBLIC_USE_AUTH_HEADER=true`.
  - `useAuthFetch` injects `X-Tenant-ID` and attempts a silent `/auth/refresh` on 401.
- UI
  - Header bar: email/tenant, Verify Odoo, dev tenant switcher (optional), Sign out.
  - Chat area: threads sidebar, messages, file upload, send, basic error toasts.
  - Settings form: appears when `apiUrl`/`assistantId` missing.

---

## Configuration

- Environment
  - `NEXT_PUBLIC_API_URL`: LangGraph base (e.g., `http://localhost:8001`).
  - `NEXT_PUBLIC_API_BASE`: Optional alternate base used by UI helpers; if absent, falls back to `NEXT_PUBLIC_API_URL`.
  - `NEXT_PUBLIC_ASSISTANT_ID`: Assistant/graph ID (e.g., `agent`).
  - `NEXT_PUBLIC_USE_API_PROXY`: If `true`, client requests are rewritten to `/api/backend/*` to share cookies.
  - `NEXT_PUBLIC_USE_AUTH_HEADER`: Dev-only; if `true`, attach `Authorization: Bearer <idToken>`.
  - `NEXT_PUBLIC_ENABLE_TENANT_SWITCHER`: Dev-only; enables local tenant override in the header.
  - `LANGSMITH_API_KEY`: Injected server-side by proxy for deployed graphs (optional).
- Settings Form
  - When `apiUrl` or `assistantId` are missing, `StreamProvider` renders an inline form to collect Deployment URL, Assistant/Graph ID, and optional LangSmith API key.
  - Values persist in the URL query (`apiUrl`, `assistantId`) and in `localStorage` under `lg:chat:apiKey`.

---

## Auth & Tenant Context

- Cookies
  - Backend sets `nx_access`/`nx_refresh`; UI relies on cookies by default. Dev-only auth header path is guarded by `NEXT_PUBLIC_USE_AUTH_HEADER`.
- Silent Refresh
  - `useAuthFetch` auto-calls `/auth/refresh` once on 401 and retries the original request.
- Tenant Propagation
  - Session-derived tenant ID (or dev override `lg:chat:tenantId` when enabled) is sent via `X-Tenant-ID` for chat streams and REST calls.
  - Header bar shows “Signed in as <email> (tenant: <id>[, override])”.

---

## Login/Signup & Onboarding

- Pages: `/login`, `/signup` post to backend `/auth/*` (proxied when enabled) and rely on cookies; no passwords stored in UI.
- First-Login Gate: `FirstLoginGate` kicks off `/onboarding/first_login`, then polls `/onboarding/status` and probes `/session/odoo_info` and `/onboarding/verify_odoo` until ready, showing progress text per stage.
- Verify Odoo: Header bar button calls `/onboarding/verify_odoo` and surfaces results.
- Logout: Keycloak logout with `post_logout_redirect_uri` and `id_token_hint`; fallback to `/login`.

---

## Connection Health & Resilience

- Health-Check: `checkGraphStatus` first calls `/api/info` (or `{apiUrl}/info`), then falls back to `/assistants?limit=1`. Results are cached in-session to prevent duplicate toasts.
- Toast De-dup: `sessionStorage` key `lg:chat:connToast:<apiUrl>` prevents double alerts from React Strict Mode.
- Connection Badge: Header shows a small badge reflecting current connectivity (`Connected`/`Offline`) using the same health-check.
- Stream Errors: `Thread` listens to `stream.error` and shows a single toast per unique error message.
- SSE: Stream reconnects are handled by the SDK; UI shows “AssistantMessageLoading” until first token.

---

## Chat & Threads

- Thread State: `threadId` stored in the query string; history panel toggled via `ThreadHistory` with slide animation.
- Submit: `Thread` composes the user message plus any uploaded content blocks and calls `stream.submit({ messages, context })` where `context` includes `tenant_id`.
- Tool Responses: `ensureToolCallsHaveResponses` pre-fills missing tool results to keep the UI stable.
- File Upload: Drag-and-drop and paste supported; content blocks preview rendered before submit.

---

## Routing & Params

- Query Keys: `apiUrl`, `assistantId`, `threadId`, `chatHistoryOpen`, `hideToolCalls`.
- Local Storage: `lg:chat:apiKey`, `lg:chat:tenantId`.

---

## New UX Additions (from PRD 12)

1) Shortlist Status widget (polls `/shortlist/status`)
2) Export controls (CSV/JSON) using backend export endpoints
3) Connection indicator (badge) leveraging the existing health-check logic
4) Optional: delete/export transcript actions in thread history (scoped after MVP)

---

## Code Changes

### 1) Shortlist Status widget

File: `agent-chat-ui/src/components/ui/header-bar.tsx` — add a small component to poll `/shortlist/status` and render a badge.

```tsx
// below existing imports
import { useMemo, useEffect, useState } from "react";
import { useAuthFetch } from "@/lib/useAuthFetch";

function ShortlistStatusBadge() {
  const authFetch = useAuthFetch();
  const apiBase = useMemo(
    () => process.env.NEXT_PUBLIC_API_BASE || process.env.NEXT_PUBLIC_API_URL || "http://localhost:2024",
    []
  );
  const [data, setData] = useState<{ total_scored: number; last_refreshed_at: string | null } | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    let iv: any;
    async function poll() {
      try {
        const res = await authFetch(`${apiBase}/shortlist/status`);
        if (!res.ok) throw new Error(`status ${res.status}`);
        const j = await res.json();
        if (!cancelled) setData({ total_scored: j.total_scored ?? 0, last_refreshed_at: j.last_refreshed_at ?? null });
      } catch (e: any) {
        if (!cancelled) setErr(String(e));
      }
    }
    void poll();
    iv = setInterval(poll, 30000); // 30s
    return () => { cancelled = true; if (iv) clearInterval(iv); };
  }, [apiBase, authFetch]);

  if (!data) return null;
  const ts = data.last_refreshed_at ? new Date(data.last_refreshed_at) : null;
  const when = ts ? new Intl.DateTimeFormat(undefined, { hour: '2-digit', minute: '2-digit' }).format(ts) : "n/a";
  return (
    <span className="text-xs text-muted-foreground" title={`Last refreshed: ${ts?.toLocaleString() || "n/a"}`}>
      Shortlist: {data.total_scored} • {when}
    </span>
  );
}
```

Usage: render `<ShortlistStatusBadge />` inside the right-hand button group in `HeaderBar`.

### 2) Export controls

Provide CSV/JSON download buttons that call backend exports with cookies included.

```tsx
function ExportButtons() {
  const authFetch = useAuthFetch();
  const apiBase = useMemo(
    () => process.env.NEXT_PUBLIC_API_BASE || process.env.NEXT_PUBLIC_API_URL || "http://localhost:2024",
    []
  );

  async function dl(path: string, filename: string) {
    const res = await authFetch(`${apiBase}${path}`);
    if (!res.ok) return;
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="flex items-center gap-2">
      <button className="text-xs px-2 py-1 border rounded" onClick={() => dl(`/export/latest_scores.csv?limit=500`, `shortlist.csv`)}>CSV</button>
      <button className="text-xs px-2 py-1 border rounded" onClick={() => dl(`/export/latest_scores.json?limit=500`, `shortlist.json`)}>JSON</button>
    </div>
  );
}
```

Usage: add `<ExportButtons />` in `HeaderBar` (visible for all roles; server enforces any restrictions).

### 3) Connection indicator

Option A: reuse the `checkGraphStatus` logic in `Stream.tsx` by duplicating the lightweight check in the header (call `/api/info` then fallback to `/assistants?limit=1`).

```tsx
function useConnectionStatus() {
  const [ok, setOk] = useState<boolean | null>(null);
  const apiUrl = useMemo(() => new URLSearchParams(window.location.search).get('apiUrl') || process.env.NEXT_PUBLIC_API_URL || '', []);
  useEffect(() => {
    let cancelled = false;
    async function check() {
      try {
        const useProxy = (process.env.NEXT_PUBLIC_USE_API_PROXY || '').toLowerCase() === 'true';
        const infoUrl = useProxy ? '/api/info' : `${apiUrl}/info`;
        const altUrl  = useProxy ? '/api/assistants?limit=1' : `${apiUrl}/assistants?limit=1`;
        const r1 = await fetch(infoUrl, { credentials: 'include' });
        if (r1.ok) { if (!cancelled) setOk(true); return; }
        const r2 = await fetch(altUrl, { credentials: 'include' });
        if (!cancelled) setOk(r2.ok);
      } catch { if (!cancelled) setOk(false); }
    }
    void check();
    const id = setInterval(check, 30000);
    return () => { cancelled = true; clearInterval(id); };
  }, [apiUrl]);
  return ok;
}

function ConnectionBadge() {
  const ok = typeof window !== 'undefined' ? useConnectionStatus() : null;
  if (ok === null) return null;
  return (
    <span className={"text-xs px-2 py-1 rounded " + (ok ? "bg-green-100 text-green-700" : "bg-red-100 text-red-700")}> {ok ? 'Connected' : 'Offline'} </span>
  );
}
```

Usage: add `<ConnectionBadge />` into the header button group.

---

## Chat History Storage

- Persistence Layer
  - Use LangGraph Server’s built-in thread/checkpoint storage. In local/dev, persist to the filesystem directory specified by `LANGGRAPH_CHECKPOINT_DIR` (default `.langgraph_api`). In production, run LangGraph Server with a durable store (same API), or back up the directory on a persistent volume.
- Scope & Security
  - Custom auth binds identity to tenant (e.g., `tenant:<id>`). LangGraph isolates threads by identity; cross-tenant access is blocked server-side.
- Thread Metadata & Search
  - Threads are created by the SDK and tagged so they can be searched via `client.threads.search`. The UI filters by `{ graph_id: <assistantId> }` or `{ assistant_id: <uuid> }`.
  - The UI shows up to the last 100 threads in the history panel.
- Message Retention
  - Default retention 90 days for threads/messages in dev; production retention configurable by deployment policy. Provide a periodic cleanup job (server-side) that deletes threads older than the retention window.
- User Operations
  - View past threads in the sidebar; clicking loads messages via the SDK. “New thread” clears `threadId` to start a new conversation; a fresh thread is created on next submit.

---

## Accessibility & Performance

- Keyboard: Enter to send; Shift+Enter for newline.
- A11y: Labels on switches/inputs; icons wrapped in tooltips where appropriate.
- Performance: Avoid heavy work on render; content virtualization not required for current message volumes.

---

## Roles

- Viewer: Read-only permissions in the chat and exports.
- Ops/Admin: ICP edits (via chat commands), ad-hoc enrich, costs visibility. UI passes through commands; backend enforces authorization.

---

## Acceptance Criteria

- Connection: Health-check falls back to `/assistants` when `/info` requires auth; no duplicate alerts in dev.
- Auth: Cookie-mode works end-to-end with silent refresh; dev auth header path functional when enabled.
- Tenant: Header shows correct email/tenant; `X-Tenant-ID` forwarded on stream and REST requests; dev override works only when enabled.
- Onboarding: First-login gate advances through states and unblocks when Odoo is ready; Verify Odoo button reports accurate state.
- Chat: Submits messages and streams responses; thread history loads; error toasts are deduped.
- Settings: Missing config shows settings form; values persist in query/local storage.
- Export/Status: Status reads from `/shortlist/status`; exports include run metadata and are downloadable.
- History: Threads persist across reloads; last 100 threads listed; selecting a thread restores its messages. In dev, data survives process restarts if `.langgraph_api` persists.

---

## Open Questions

- Dashboards: Inline basic charts vs link to Metabase.
- ICP Edits: Dedicated form alongside chat vs chat-only commands.

---

## User Journey

1) Sign up / Sign in
   - User visits `/signup` to create an account or `/login` to sign in. The UI posts to backend `/auth/*` (proxied if enabled) and stores session cookies. Header shows “Signed in as <email> (tenant: <id>)”.
2) First login provisioning
   - `FirstLoginGate` automatically calls `/onboarding/first_login`, then polls `/onboarding/status` and probes Odoo endpoints until ready. While provisioning, the UI displays messages like “Creating Odoo DB…” and “Seeding baseline entities…”. Once ready, the chat unlocks.
3) Configure connection (if needed)
   - If `NEXT_PUBLIC_API_URL` or assistant ID is missing, the settings form appears. The user enters Deployment URL (e.g., `http://localhost:8001`) and Assistant/Graph ID (e.g., `agent`) and clicks Continue.
4) Start a chat
   - The chat loads. A connectivity check runs (using `/api/info` then `/assistants`). If reachable, the user can type commands. Thread history can be opened via the sidebar toggle; “New thread” resets the conversation.
5) Command examples
   - Example A: “Show today’s shortlist (top 10).” The UI sends the message with `tenant_id` in context. The backend returns a streaming response with the list; the UI renders items as the assistant message.
   - Example B: “Refresh ICP to software, 10–200 employees; process 10 now and schedule the rest nightly.” The backend starts an ad-hoc partial run. The UI shows stage chips as updates arrive (crawl → extract → verify → score). On completion, the assistant confirms: “Processed 10 now; remaining scheduled for tonight’s run.”
6) Export and verify
   - The user can trigger an export (CSV/JSON) from the export control (when wired). The header’s “Verify Odoo” button checks tenant Odoo connectivity and shows a brief status note. Past chats remain available under Thread History (left sidebar). Selecting a past thread restores the conversation; “New thread” starts a fresh one.

## Optional Enhancements (post-MVP)

- Thread actions: Delete thread, Export transcript (SDK/API based)
- Role-aware UI (hide export for viewer, etc.)
- Inline mini-dashboards or links to Metabase

---

## Testing Plan

- Local dev with `NEXT_PUBLIC_USE_API_PROXY=true`, LangGraph on 8001. Verify:
  - Successful health-check without duplicate toasts; Connection badge toggles on/off if server down.
  - Shortlist status updates every 30s and matches DB.
  - CSV/JSON downloads include rows; CSV opens in spreadsheet.
  - Onboarding gate unlocks when `/onboarding/verify_odoo` returns ready.
- Auth flows: login/signup, cookie exchange, silent refresh on 401.
- Threads: new thread, sidebar toggling, file uploads, SSE streams.

---

## Rollout Steps

1) Land header status components and wire into `HeaderBar`.
2) Verify in dev with proxy ON and standalone OFF.
3) Add basic docs to README (envs, where to find status/exports).
4) Optionally add role gating (feature flag).

---

## Multi‑Tenant Thread & Chat History

- Client Enforcement
  - Always send `X-Tenant-ID` for stream + REST calls (derived from localStorage `lg:chat:tenantId` first, else session).
  - Require a known tenant to fetch thread history; when tenant is unknown, avoid history fetch (prevents cross‑tenant leakage) and show only the current active thread optimistically.
  - Search shape: top‑level `{ graph_id | assistant_id }` + `metadata: { tenant_id }`.
- Thread Creation & Races
  - Pre‑create a thread with `metadata.tenant_id` and `graphId=<assistant>` before first submit so the SDK does not auto‑create an untagged thread.
  - On `onThreadId`, best‑effort PATCH the thread to set `metadata.tenant_id` if missing (heals SDK auto‑create). Add a brief delay (~25ms) after pre‑create so the stream picks up the `threadId` before submit.
- Persistence Across Login
  - On `/auth/login` success, persist `tenant_id` in `localStorage` (`lg:chat:tenantId`). On sign‑out, remove it to avoid cross‑user leakage.
- Proxies & Stability (Dev)
  - Use Node runtime for Next.js API proxies (`/app/api/[..._path]` and `/app/api/backend/[..._path]`) for reliable local dev networking (avoid intermittent Edge fetch failures).
- Storage & Envs
  - Dev: local_dev runtime is in‑memory unless `LANGGRAPH_CHECKPOINT_DIR` is set. Set it (e.g., `.langgraph_api`) to persist threads across backend restarts.
  - Prod: deploy LangGraph Server with durable checkpointer (filesystem volume/DB). Consider `LANGGRAPH_ALLOW_ANON=false` to forbid anonymous thread creation.
- Edge Cases
  - Legacy threads without `metadata.tenant_id` won’t appear under tenant filters; run a one‑off backfill if needed.

### Acceptance Criteria — Multi‑Tenant
- Creating a new chat creates exactly one thread tagged with `metadata.tenant_id` and visible in sidebar immediately.
- After logout/login, sidebar lists the tenant’s threads (no cross‑tenant threads, no disappearance).
- History search uses `{ graph_id | assistant_id }` plus `metadata.tenant_id` filter and returns up to 100 results.
- Proxies remain stable in dev; no intermittent fetch failures block history.
