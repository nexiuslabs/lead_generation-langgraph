---
owner: Backend – Lead Gen Platform
status: draft
linked_prd: devplan/featurePRD_jinaMCP.md
last_updated: 2025-10-27
---

# Dev Plan — Jina MCP Server Integration (Reader + Search)

Goal: Add a feature‑flagged MCP transport to power web reading and search for discovery, evidence, and enrichment while retaining HTTP fallbacks and identical downstream structures.

This plan details new configs, modules, wiring changes, code samples, step‑by‑step agent flow, telemetry, rollout, and tests.

## 1) Config and Dependencies

- requirements.txt (optional dependency):
  - Add `jina-mcp>=0.1.0` (or the official MCP Python client name). Keep optional: import lazily and guard with flag.
- Settings (new env vars in `src/settings.py`):
  - `ENABLE_MCP_READER` (bool, default false)
  - `MCP_ENDPOINT` (default `https://mcp.jina.ai/sse`)
  - `MCP_API_KEY`
  - `MCP_TIMEOUT_S` (default 15)
  - `MCP_MAX_PARALLEL` (default 4) — used by parallel search wrapper
  - `MCP_DUAL_READ_SAMPLE_PCT` (default 0) — percent of calls to dual‑read for parity checks

Code (settings additions):

```python
# src/settings.py (add near other feature flags)
ENABLE_MCP_READER = os.getenv("ENABLE_MCP_READER", "false").lower() in ("1","true","yes","on")
MCP_ENDPOINT = os.getenv("MCP_ENDPOINT", "https://mcp.jina.ai/sse")
MCP_API_KEY = os.getenv("MCP_API_KEY", "")
try:
    MCP_TIMEOUT_S = float(os.getenv("MCP_TIMEOUT_S", "15") or 15)
except Exception:
    MCP_TIMEOUT_S = 15.0
try:
    MCP_MAX_PARALLEL = int(os.getenv("MCP_MAX_PARALLEL", "4") or 4)
except Exception:
    MCP_MAX_PARALLEL = 4
try:
    MCP_DUAL_READ_SAMPLE_PCT = int(os.getenv("MCP_DUAL_READ_SAMPLE_PCT", "0") or 0)
except Exception:
    MCP_DUAL_READ_SAMPLE_PCT = 0
```

requirements.txt change:

```diff
# lead_generation-main/requirements.txt
@@
 # HTTP clients / scraping
 httpx>=0.26,<0.28
 beautifulsoup4
 requests
+# MCP client (optional; guarded by ENABLE_MCP_READER)
+jina-mcp>=0.1.0
```

## 2) New Module: MCP Reader Client

File: `src/services/mcp_reader.py`

Responsibilities:
- Manage MCP session to `MCP_ENDPOINT` with API key.
- Provide synchronous wrappers: `read_url(url)`, `search_web(query)`, `parallel_search_web(queries)`.
- Retries, timeouts, and small connection pool.
- Optional dual‑read parity check with legacy reader.
- Telemetry via `src/obs.py.bump_vendor` and `log_event`.

Code (skeleton):

```python
# src/services/mcp_reader.py
from __future__ import annotations
import os, time, random, logging
from typing import Optional, List, Dict, Any

from src.settings import (
    ENABLE_MCP_READER, MCP_ENDPOINT, MCP_API_KEY, MCP_TIMEOUT_S,
    MCP_MAX_PARALLEL, MCP_DUAL_READ_SAMPLE_PCT,
)
from src.obs import bump_vendor, log_event, get_run_context

log = logging.getLogger("mcp_reader")

_SESSION = None  # lazily initialized MCP client/session

def _ensure_session():
    global _SESSION
    if _SESSION is not None:
        return _SESSION
    try:
        # Prefer official client if installed, else raise for graceful fallback upstream
        from jina_mcp import Client  # type: ignore
    except Exception as e:
        log.warning("[mcp] jina_mcp client not installed: %s", e)
        _SESSION = None
        return None
    _SESSION = Client(base_url=MCP_ENDPOINT, api_key=MCP_API_KEY, timeout=MCP_TIMEOUT_S)
    return _SESSION


def _telemetry(vendor: str, *, ok: bool, start: float, extra: Dict[str, Any] | None = None):
    run_id, tenant_id = get_run_context()
    dur_ms = int((time.perf_counter() - start) * 1000)
    if run_id and tenant_id:
        try:
            log_event(run_id, tenant_id, stage="mcp", event=vendor, status=("ok" if ok else "error"), duration_ms=dur_ms, extra=extra)
            bump_vendor(run_id, tenant_id, vendor="mcp", calls=1, errors=(0 if ok else 1))
        except Exception:
            pass


def read_url(url: str, timeout: Optional[float] = None) -> Optional[str]:
    if not ENABLE_MCP_READER:
        return None
    session = _ensure_session()
    if session is None:
        return None
    t0 = time.perf_counter()
    try:
        # Tool name and shape follow Jina MCP contract
        res = session.tools.invoke("read_url", {"url": url, "timeout": timeout or MCP_TIMEOUT_S})
        # Expected shape: { "content": "..." } or a stream aggregator
        text = (res.get("content") or "").strip()
        _telemetry("read_url", ok=True, start=t0, extra={"bytes": len(text)})
        return text or None
    except Exception as e:
        log.info("[mcp] read_url error for %s: %s", url, e)
        _telemetry("read_url", ok=False, start=t0, extra={"error": type(e).__name__})
        return None


def search_web(query: str, *, country: Optional[str] = None, max_results: int = 20) -> List[str]:
    if not ENABLE_MCP_READER:
        return []
    session = _ensure_session()
    if session is None:
        return []
    t0 = time.perf_counter()
    try:
        payload = {"query": query, "limit": max_results, "country": country}
        res = session.tools.invoke("search_web", payload)
        # Expect list of URLs; normalize to apex domains upstream
        items = res.get("results") if isinstance(res, dict) else res
        out = []
        for it in items or []:
            if isinstance(it, str):
                out.append(it)
            elif isinstance(it, dict) and it.get("url"):
                out.append(str(it["url"]))
        _telemetry("search_web", ok=True, start=t0, extra={"count": len(out)})
        return out[:max_results]
    except Exception as e:
        log.info("[mcp] search_web error: %s", e)
        _telemetry("search_web", ok=False, start=t0, extra={"error": type(e).__name__})
        return []


def parallel_search_web(queries: List[str], *, per_query: int = 10) -> Dict[str, List[str]]:
    if not ENABLE_MCP_READER:
        return {}
    session = _ensure_session()
    if session is None:
        return {}
    t0 = time.perf_counter()
    try:
        payload = {"queries": queries[:MCP_MAX_PARALLEL], "per_query": per_query}
        res = session.tools.invoke("parallel_search_web", payload)
        # Expect mapping {query: [urls]}
        out: Dict[str, List[str]] = {}
        if isinstance(res, dict):
            for k, v in res.items():
                if isinstance(v, list):
                    out[str(k)] = [str(x.get("url") if isinstance(x, dict) else x) for x in v]
        _telemetry("parallel_search_web", ok=True, start=t0, extra={"queries": len(out)})
        return out
    except Exception as e:
        log.info("[mcp] parallel_search_web error: %s", e)
        _telemetry("parallel_search_web", ok=False, start=t0, extra={"error": type(e).__name__})
        return {}
```

Notes:
- The above expects the official `jina_mcp.Client`. If the package name differs, update imports accordingly.
- `tools.invoke` is a placeholder for the client’s tool call API. Adjust to the actual method as per client docs.
- We intentionally return `None`/empty on any MCP error so callers can fall back to HTTP when doing dual‑read.

## 3) Wire MCP into existing readers/search

### 3.1 `src/jina_reader.py` — conditionally use MCP

Behavior: If `ENABLE_MCP_READER` and MCP returns content, use it. Else, use legacy `r.jina.ai` HTTP. Keep cleaning and variants.

Code (patch excerpt):

```python
# at top
from src.settings import ENABLE_MCP_READER, MCP_DUAL_READ_SAMPLE_PCT
import random

def read_url(url: str, timeout: float = 12.0) -> Optional[str]:
    # Optional MCP path
    if ENABLE_MCP_READER:
        try:
            from src.services.mcp_reader import read_url as mcp_read_url
            # Optional dual-read sampling for parity
            if MCP_DUAL_READ_SAMPLE_PCT and random.randint(1,100) <= MCP_DUAL_READ_SAMPLE_PCT:
                mcp_txt = mcp_read_url(url, timeout=timeout) or ""
                legacy_txt = None
                try:
                    legacy_txt = _legacy_read(url, timeout=timeout)  # see wrapper below
                except Exception:
                    legacy_txt = None
                chosen = mcp_txt or (legacy_txt or "")
                return clean_jina_text(chosen)[:10000] if chosen else None
            # Normal MCP-first path
            txt = mcp_read_url(url, timeout=timeout)
            if txt:
                return clean_jina_text(txt)[:10000]
        except Exception:
            pass
    # existing legacy HTTP reader below…
    return _legacy_read(url, timeout=timeout)

def _legacy_read(url: str, timeout: float = 12.0) -> Optional[str]:
    # existing legacy code body here (variants + r.jina.ai requests)
    # return clean_jina_text(txt) or None
    ...
```

### 3.2 `src/ddg_simple.py` — add MCP search fallback

Behavior: Prefer MCP `search_web` when enabled. Fall back to current DDG HTML scraping via `r.jina`.

Code (patch excerpt):

```python
 from src.settings import ENABLE_MCP_READER
from urllib.parse import urlparse

def search_domains(query: str, max_results: int = 20, country: Optional[str] = None) -> List[str]:
    if ENABLE_MCP_READER:
        try:
            from src.services.mcp_reader import search_web
            urls = search_web(query, country=country, max_results=max_results)
            # normalize to apex domains using existing helpers
            hosts: List[str] = []
            for u in urls:
                try:
                    host = (urlparse(u).netloc or "").lower()
                    if host:
                        hosts.append(host)
                except Exception:
                    continue
            # dedupe via _apex logic
            seen: set[str] = set()
            out: List[str] = []
            for h in hosts:
                a = _apex(h)
                if a and a not in seen:
                    seen.add(a)
                    out.append(a)
            return out[:max_results]
        except Exception:
            pass
     # existing legacy r.jina scraping below…
 ```

### 3.3 `src/agents_icp.py` — use MCP search in DiscoveryPlannerAgent

Context: The discovery agent currently implements its own `_ddg_search_domains` that pulls DuckDuckGo HTML directly (and falls back to r.jina snapshots). To ensure the multi‑agent system leverages Jina MCP uniformly, add an MCP‑first path inside this function before the direct DDG logic.

Patch excerpt:

```python
# at top of src/agents_icp.py (imports section)
from src.settings import ENABLE_MCP_READER  # reuse global flag

...

def _ddg_search_domains(query: str, max_results: int = 25, country: str | None = None, lang: str | None = None) -> List[str]:
    """Perform domain discovery..."""
    # MCP-first path (Jina MCP search_web) to unify multi-agent usage
    if ENABLE_MCP_READER:
        try:
            from urllib.parse import urlparse as _urlparse
            from src.services.mcp_reader import search_web as _mcp_search
            urls = _mcp_search(query, country=country, max_results=max_results)
            hosts: List[str] = []
            for u in urls:
                try:
                    h = (_urlparse(u).netloc or "").lower()
                    if h:
                        hosts.append(h)
                except Exception:
                    continue
            # Normalize to apex and dedupe using local helpers
            uniq: List[str] = []
            seen: set[str] = set()
            for h in hosts:
                a = _apex_domain(h)
                if a and a not in seen and _is_probable_domain(a):
                    seen.add(a)
                    uniq.append(a)
                if len(uniq) >= max_results:
                    break
            if uniq:
                return uniq[:max_results]
        except Exception:
            pass
    # existing direct DDG + r.jina snapshot logic continues below…
```

Result: DiscoveryPlannerAgent, Evidence enrichment bootstrap, and any other callers to `_ddg_search_domains` will automatically use MCP search when enabled, keeping output identical (apex domains, deduped) and falling back transparently.

### 3.4 `app/pre_sdr_graph.py` — no code changes

Pre‑chat orchestration calls `src/agents_icp.discovery_planner` and the enrichment graph. With the above changes to `src/agents_icp.py` and `src/jina_reader.py`, MCP is used consistently across the planner and evidence collector when enabled. No modifications are required in the graph wiring.

Helper to dedupe apex (reuse internal function or inline using `_apex`).

## 4) Agentic Flow: Step‑by‑Step with MCP

The LangGraph orchestration remains the same; only fetch/search tools change when `ENABLE_MCP_READER` is true.

1) DiscoveryPlannerAgent (src/agents_icp.py::discovery_planner)
   - LLM composes strict query.
   - Search: uses `src/ddg_simple.py::search_domains` which prefers MCP `search_web` when enabled; otherwise DDG HTML via r.jina.ai.
   - Evidence preview: snapshots homepages via `src/jina_reader.py::read_url` — MCP first, else r.jina.ai.
   - Output: candidates list with hygiene logs.

2) EvidenceCollectorAgent
   - mini_crawl_worker snapshots homepages with `src/jina_reader.py::read_url` → MCP first.
   - LLM `evidence_extractor` unchanged (input text format stays identical).

3) ComplianceGuardAgent
   - No change. For borderline SG checks that peek content, rely on `jina_reader.read_url` → may use MCP.

4) EnrichmentAgent (src/enrichment.py)
   - Nodes using `src/jina_reader.py::read_url` (e.g., homepage merges) automatically benefit from MCP.
   - Deterministic crawl, Tavily, Apify unaffected.
   - Persist and provenance unchanged.

5) ScoringAgent
   - No change.

## 5) Observability and Health

- Metrics: in `src/services/mcp_reader.py`, call `bump_vendor(run_id, tenant_id, vendor="mcp", calls/errors)` and `log_event(stage="mcp", event=tool)`.
- Add Grafana panels for MCP success rate, p95 latency by tool, error codes top‑N.
- Dual‑read sampling: if `MCP_DUAL_READ_SAMPLE_PCT > 0`, randomly run both MCP and legacy for a subset; diff byte length and simple Jaccard on tokens. Log into `run_event_logs` with status `parity_ok`/`parity_diff`.

Code (dual‑read example sketch inside `jina_reader.read_url`):

```python
from src.settings import MCP_DUAL_READ_SAMPLE_PCT
if ENABLE_MCP_READER and random.randint(1,100) <= MCP_DUAL_READ_SAMPLE_PCT:
    mcp_txt = mcp_read_url(url, timeout=timeout) or ""
    legacy_txt = _legacy_read(url, timeout=timeout) or ""
    # simple parity check
    ok = abs(len(mcp_txt) - len(legacy_txt)) / max(1, len(legacy_txt)) <= 0.2
    run_id, tenant_id = get_run_context()
    if run_id and tenant_id:
        log_event(run_id, tenant_id, "mcp", event="dual_read_parity", status=("ok" if ok else "diff"), extra={"m": len(mcp_txt), "h": len(legacy_txt)})
    return clean_jina_text(mcp_txt or legacy_txt)[:10000] or None
```

## 6) Rollout Plan

- Phase 0: Ship behind flag, disabled. Land telemetry, dashboards, and docs.
- Phase 1: Enable on staging with `MCP_DUAL_READ_SAMPLE_PCT=50`. Track success rate and parity.
- Phase 2: Enable for 5% of production tenants. Watch p95 latency and errors for 48 hours.
- Phase 3: Increase to 50% → 100% after parity ≥95% and success ≥98% for 7 days.
- Rollback: set `ENABLE_MCP_READER=false` and redeploy. No code rollback needed.

## 7) Testing

- Unit tests (new):
  - `tests/test_mcp_reader.py`: mocks `jina_mcp.Client` to return canned responses and raises; assert fallbacks and telemetry calls.
  - `tests/test_ddg_mcp_switch.py`: ensure `search_domains` uses MCP when flag enabled and normalizes apex domains.
- Integration tests:
  - Staging: run enrichment for a sample company list with flag ON/OFF; compare firmographics and evidence rows for variance ≤5%.
- Smoke tests:
  - Health probe that calls `mcp_reader.read_url("https://example.com")` and exposes OK/ERROR endpoint for operators.

## 8) Code Changes Summary (by file)

- `requirements.txt`: add optional `jina-mcp`.
- `src/settings.py`: add MCP envs/flags.
- `src/services/mcp_reader.py`: new client module.
- `src/jina_reader.py`: conditional MCP call before legacy HTTP; optional dual‑read sampling.
- `src/ddg_simple.py`: conditional MCP `search_web` usage before HTML scraping.
- `src/enrichment.py`: no direct change (benefits via `jina_reader`).
- `src/icp_pipeline.py`: no change (already uses `jina_reader`).
- `docs`: update runbooks and feature flags cheat‑sheet.

## 9) Example Env

```
ENABLE_MCP_READER=true
MCP_ENDPOINT=https://mcp.jina.ai/sse
MCP_API_KEY=sk-xxxxx
MCP_TIMEOUT_S=15
MCP_MAX_PARALLEL=4
MCP_DUAL_READ_SAMPLE_PCT=25
```

## 10) Upsert Steps (copy/paste ready)

- Edit `lead_generation-main/requirements.txt` and add the `jina-mcp` line as shown above. Then install deps in your venv.
- Add env vars to `lead_generation-main/.env` (or your runtime env) using the Example Env block.
- Create new module `lead_generation-main/src/services/mcp_reader.py` using the provided skeleton.
- Patch `lead_generation-main/src/jina_reader.py` per the MCP-first logic with dual-read sampling and `_legacy_read` wrapper.
- Patch `lead_generation-main/src/ddg_simple.py` to use MCP `search_web` when enabled and normalize to apex.
- No changes required to `src/enrichment.py` or `src/icp_pipeline.py` beyond benefiting from the reader changes.
- Restart the dev server and run a small enrichment batch with `ENABLE_MCP_READER=true` to verify end-to-end.

## 11) Developer Notes

- The MCP client import is wrapped to avoid hard dependency during local dev. If not installed, the system silently falls back to legacy methods.
- Keep output structure stable: all callers receive plain text for `read_url` and normalized list of apex domains for search to avoid ripple changes.
- Avoid aggressive parallelism initially; respect vendor quotas and adhere to PRD latency SLAs.

## 12) Acceptance Checklist Mapping

- Reusable MCP client with retries/timeouts: provided in `mcp_reader.py` with telemetry and lazy session.
- Feature flag switch: `ENABLE_MCP_READER` gates code paths in `jina_reader.py` and `ddg_simple.py`.
- Parity of outputs: cleaning and normalization preserve shapes; dual‑read sampling validates parity.
- Observability: `bump_vendor` and `log_event` instrumented; add dashboard panels post‑merge.
- Rollback: single flag flip; health probe can be disabled independently.

-- end --
 
---

Addenda: Concrete Snippets To Fully Satisfy PRD

1) Optional Prometheus Requirement

```diff
# lead_generation-main/requirements.txt
@@
+prometheus-client>=0.20
```

2) Prometheus Instrumentation (_telemetry)

```python
# In src/services/mcp_reader.py, inside _telemetry() after dur_ms is computed
try:
    from prometheus_client import Counter, Histogram  # type: ignore
    global _MCP_REQS, _MCP_DUR
    try:
        _MCP_REQS
    except NameError:
        _MCP_REQS = Counter("mcp_requests_total", "MCP tool requests", ["tool", "status"])  # type: ignore
    try:
        _MCP_DUR
    except NameError:
        _MCP_DUR = Histogram("mcp_request_duration_seconds", "MCP tool request duration (seconds)", ["tool"])  # type: ignore
    _MCP_REQS.labels(vendor, ("ok" if ok else "error")).inc()  # type: ignore
    _MCP_DUR.labels(vendor).observe(max(0.0, dur_ms / 1000.0))  # type: ignore
except Exception:
    pass
```

3) Retries/Circuit Breaker Wrapper

```python
# src/services/mcp_reader.py
from src.retry import with_retry, BackoffPolicy, CircuitBreaker, RetryableError
from src.settings import CB_ERROR_THRESHOLD, CB_COOL_OFF_S

_BREAKERS = {
    "read_url": CircuitBreaker(CB_ERROR_THRESHOLD, CB_COOL_OFF_S),
    "search_web": CircuitBreaker(CB_ERROR_THRESHOLD, CB_COOL_OFF_S),
    "parallel_search_web": CircuitBreaker(CB_ERROR_THRESHOLD, CB_COOL_OFF_S),
}

def _invoke_with_resilience(tool: str, payload: dict) -> dict | list | None:
    br = _BREAKERS.get(tool)
    if br and br.is_open():
        raise RetryableError(f"breaker-open:{tool}")

    def _call():
        sess = _ensure_session()
        if sess is None:
            raise RetryableError("no-session")
        try:
            return sess.tools.invoke(tool, payload)
        except Exception:
            # reset session to force reconnect on next attempt
            try:
                global _SESSION
                _SESSION = None
            except Exception:
                pass
            raise

    res = with_retry(_call, BackoffPolicy(max_attempts=3, base_delay_ms=250, max_delay_ms=1500))
    if br:
        try:
            br.success()
        except Exception:
            pass
    return res

# Use _invoke_with_resilience in read_url/search_web/parallel_search_web
```

4) Health Probe Endpoint

```python
# app/main.py
@app.get("/health/mcp")
def mcp_health():
    try:
        from src.services.mcp_reader import read_url as _r
        txt = _r("https://example.com", timeout=1.0)
        if not txt:
            raise RuntimeError("empty")
        return {"ok": True}
    except Exception:
        raise HTTPException(status_code=503, detail="mcp_unhealthy")
```

5) Rollout Gates and Alerts (PromQL)

```text
- Phase 1 → Phase 2 gate: success_rate ≥ 95% and parity variance ≤ 5% over 48h.
- Phase 3 (full) gate: success_rate ≥ 98% for 7d and read_url p95 ≤ 3s.

Alerts:
- Error rate >5% over 10m:
  sum(rate(mcp_requests_total{status="error"}[10m])) / sum(rate(mcp_requests_total[10m])) > 0.05
- p95 > 5s over 10m (read_url):
  histogram_quantile(0.95, sum(rate(mcp_request_duration_seconds_bucket{tool="read_url"}[10m])) by (le)) > 5
```
