import os
import time
import json
import random
import threading
import logging
from typing import Optional, Any, Dict, Tuple, List
import subprocess
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from src import obs  # type: ignore
except Exception:  # pragma: no cover - obs is optional in import time
    obs = None  # type: ignore

# Optional Python adapters (LangGraph/LangChain MCP Adapters)
try:  # pragma: no cover
    # Python package per docs/langgraph_mcp.md
    from langchain_mcp_adapters.client import MultiServerMCPClient as _AdaptersClient
    _ADAPTERS_AVAILABLE = True
except Exception:  # pragma: no cover
    _ADAPTERS_AVAILABLE = False
    _AdaptersClient = None  # type: ignore

log = logging.getLogger("mcp_reader")
if not log.handlers:
    h = logging.StreamHandler()
    log.addHandler(h)
log.setLevel(logging.INFO)

# Optional Prometheus metrics
_PROM_ENABLE = os.getenv("PROM_ENABLE", "false").lower() in ("1", "true", "yes", "on")
try:  # pragma: no cover - metrics optional
    if _PROM_ENABLE:
        from prometheus_client import Counter as _PCounter, Histogram as _PHistogram  # type: ignore
        _MCP_CALLS = _PCounter("mcp_calls_total", "Total MCP tool calls", ["tool", "status"])  # type: ignore
        _MCP_LAT = _PHistogram("mcp_latency_seconds", "MCP tool latency (seconds)", ["tool"])  # type: ignore
    else:
        _MCP_CALLS = None  # type: ignore
        _MCP_LAT = None  # type: ignore
except Exception:  # pragma: no cover
    _PROM_ENABLE = False
    _MCP_CALLS = None  # type: ignore
    _MCP_LAT = None  # type: ignore

def _metrics_record(tool: str, status: str, duration_s: Optional[float]) -> None:
    try:
        if not _PROM_ENABLE or _MCP_CALLS is None or _MCP_LAT is None:
            return
        try:
            _MCP_CALLS.labels(tool=tool, status=status).inc()  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            if isinstance(duration_s, (int, float)):
                _MCP_LAT.labels(tool=tool).observe(float(duration_s))  # type: ignore[attr-defined]
        except Exception:
            pass
    except Exception:
        pass


class MCPClientNotImplemented(RuntimeError):
    pass


def _load_int_setting(name: str, default: int, *, min_value: int, max_value: int) -> int:
    try:
        from src import settings as _settings  # type: ignore
        raw = getattr(_settings, name, None)
    except Exception:
        raw = None
    if raw is None:
        raw = os.getenv(name)
    try:
        val = int(raw if raw not in (None, "") else default)
    except Exception:
        val = default
    if val < min_value:
        return min_value
    if val > max_value:
        return max_value
    return val


def _load_float_setting(name: str, default: float, *, min_value: float, max_value: Optional[float] = None) -> float:
    try:
        from src import settings as _settings  # type: ignore
        raw = getattr(_settings, name, None)
    except Exception:
        raw = None
    if raw is None:
        raw = os.getenv(name)
    try:
        val = float(raw if raw not in (None, "") else default)
    except Exception:
        val = default
    if val < min_value:
        val = min_value
    if max_value is not None and val > max_value:
        val = max_value
    return val


_MCP_RATE_MAX_CONCURRENCY = _load_int_setting("MCP_READ_MAX_CONCURRENCY", 2, min_value=1, max_value=16)
_MCP_RATE_MIN_INTERVAL_S = _load_float_setting("MCP_READ_MIN_INTERVAL_S", 0.35, min_value=0.0)
_MCP_RATE_JITTER_S = _load_float_setting("MCP_READ_JITTER_S", 0.15, min_value=0.0)
_MCP_READ_MAX_ATTEMPTS = _load_int_setting("MCP_READ_MAX_ATTEMPTS", 3, min_value=1, max_value=6)
_MCP_READ_BACKOFF_BASE_S = _load_float_setting("MCP_READ_BACKOFF_BASE_S", 0.6, min_value=0.0)
_MCP_READ_BACKOFF_CAP_S = _load_float_setting("MCP_READ_BACKOFF_CAP_S", 4.0, min_value=0.1)
_MCP_READ_RATELIMIT_BACKOFF_S = _load_float_setting("MCP_READ_RATELIMIT_BACKOFF_S", 2.0, min_value=0.0)


def _looks_like_rate_limit(exc: Exception) -> bool:
    try:
        msg = str(exc or "")
    except Exception:
        msg = ""
    msg = msg.lower()
    if not msg:
        return False
    return ("429" in msg) or ("rate" in msg and "limit" in msg)


def _compute_backoff_delay(attempt: int, rate_limited: bool) -> float:
    base = _MCP_READ_BACKOFF_BASE_S
    if base <= 0:
        base = 0.0
    delay = base * (2 ** max(0, attempt - 1))
    delay = min(delay if delay > 0 else base, _MCP_READ_BACKOFF_CAP_S)
    if rate_limited:
        delay = max(delay, _MCP_READ_RATELIMIT_BACKOFF_S)
    jitter_cap = min(max(delay * 0.25, 0.05), 0.5) if delay > 0 else 0.05
    jitter = random.uniform(0.0, jitter_cap) if jitter_cap > 0 else 0.0
    return max(0.0, delay + jitter)


class _RateLimiterToken:
    def __init__(self, limiter: "_ReadRateLimiter") -> None:
        self._limiter = limiter

    def __enter__(self) -> None:
        self._limiter._acquire()

    def __exit__(self, exc_type, exc, tb) -> None:
        self._limiter._release()


class _ReadRateLimiter:
    def __init__(self, max_concurrency: int, min_interval_s: float, jitter_s: float) -> None:
        self._sem = threading.BoundedSemaphore(max(1, max_concurrency))
        self._min_interval = max(0.0, min_interval_s)
        self._jitter = max(0.0, jitter_s)
        self._lock = threading.Lock()
        self._next_ready = 0.0

    def token(self) -> _RateLimiterToken:
        return _RateLimiterToken(self)

    def _acquire(self) -> None:
        self._sem.acquire()
        self._throttle()

    def _release(self) -> None:
        try:
            self._sem.release()
        except Exception:
            pass

    def _throttle(self) -> None:
        if self._min_interval <= 0:
            return
        while True:
            with self._lock:
                now = time.perf_counter()
                wait = self._next_ready - now
                if wait <= 0:
                    jitter = random.uniform(0.0, self._jitter) if self._jitter > 0 else 0.0
                    self._next_ready = now + self._min_interval + jitter
                    return
            time.sleep(min(wait, 0.05) if wait > 0 else 0.0)


_READ_RATE_LIMITER = _ReadRateLimiter(_MCP_RATE_MAX_CONCURRENCY, _MCP_RATE_MIN_INTERVAL_S, _MCP_RATE_JITTER_S)


def _get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    try:
        return os.getenv(key, default)
    except Exception:
        return default


def _get_server_url() -> str:
    """Resolve server URL preferring settings.MCP_SERVER_URL, then env vars.

    Defaults to https://mcp.jina.ai/sse.
    """
    try:
        # Prefer settings override when present
        from src import settings as _settings  # type: ignore
        v = getattr(_settings, "MCP_SERVER_URL", None)
        if isinstance(v, str) and v.strip():
            return v.strip()
    except Exception:
        pass
    try:
        v = os.getenv("MCP_SERVER_URL") or os.getenv("MCP_ENDPOINT")
        return (v or "https://mcp.jina.ai/sse").strip()
    except Exception:
        return "https://mcp.jina.ai/sse"


def _get_env_timeout_default() -> float:
    try:
        return float(os.getenv("MCP_TIMEOUT_S", "12.0") or 12.0)
    except Exception:
        return 12.0


def _effective_timeout(timeout_s: Optional[float], default_s: float) -> float:
    """Use env MCP_TIMEOUT_S as a floor so ops can raise deadlines centrally.

    - If caller passes a timeout, return max(caller, env_default)
    - If caller omits, return max(default_s, env_default)
    """
    env_default = _get_env_timeout_default()
    base = default_s if (timeout_s is None) else float(timeout_s)
    try:
        return max(base, env_default)
    except Exception:
        return env_default


class _RPCError(RuntimeError):
    pass


# -----------------------
# Adapters transport (Python)
# -----------------------
_ADAPTORS_LOOP = None  # type: ignore
_ADAPTORS_THREAD = None  # type: ignore
_ADAPTORS_CLIENT = None  # type: ignore
_ADAPTORS_READ_TOOL = None  # type: ignore
_ADAPTORS_TOOL_CACHE: Dict[str, Any] = {}
_ADAPTORS_LOCK = threading.Lock()

def _ensure_adapters_client() -> None:
    """Initialize a background asyncio loop and a MultiServerMCPClient singleton.

    This function is idempotent. It creates a background event loop in a thread and
    initializes a MultiServerMCPClient connected to the configured Jina MCP server
    using streamable HTTP (or SSE) with Authorization header.
    """
    global _ADAPTORS_LOOP, _ADAPTORS_THREAD, _ADAPTORS_CLIENT
    if _ADAPTERS_AVAILABLE is False:
        raise MCPClientNotImplemented("langchain_mcp_adapters not installed")
    if _ADAPTORS_CLIENT is not None:
        return
    import asyncio

    with _ADAPTORS_LOCK:
        if _ADAPTORS_CLIENT is not None:
            return

        # Create background loop
        loop = asyncio.new_event_loop()

        def _run_loop(l):
            asyncio.set_event_loop(l)
            l.run_forever()

        th = threading.Thread(target=_run_loop, args=(loop,), name="mcp-adapters-loop", daemon=True)
        th.start()

        # Build client config
        server_url = _get_server_url()
        api_key = _get_env("JINA_API_KEY") or ""
        # Transport selection: default to streamable_http; allow override to SSE
        try:
            from src import settings as _settings  # type: ignore
            use_sse = bool(getattr(_settings, "MCP_ADAPTER_USE_SSE", False))
            use_std_blocks = bool(getattr(_settings, "MCP_ADAPTER_USE_STANDARD_BLOCKS", True))
        except Exception:
            use_sse = False
            use_std_blocks = True

        transport = "sse" if use_sse else "streamable_http"
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

        config = {
            "use_standard_content_blocks": use_std_blocks,
            "servers": {
                "jina": {
                    "transport": transport,
                    "url": server_url,
                    "headers": headers,
                }
            },
        }

        # Instantiate client on loop
        async def _mk_client():
            # Python API uses dict positional per docs
            return _AdaptersClient(
                {"jina": {"transport": transport, "url": server_url, "headers": headers}}
            )

        fut = asyncio.run_coroutine_threadsafe(_mk_client(), loop)
        client = fut.result(timeout=10)

        _ADAPTORS_LOOP = loop
        _ADAPTORS_THREAD = th
        _ADAPTORS_CLIENT = client


def _adapters_run_coro(coro, timeout_s: float = 15.0):  # pragma: no cover
    import asyncio
    if _ADAPTORS_LOOP is None:
        raise RuntimeError("adapters loop not initialized")
    fut = asyncio.run_coroutine_threadsafe(coro, _ADAPTORS_LOOP)
    return fut.result(timeout=timeout_s)


def _adapters_get_tool(purpose: str, candidates: List[str], timeout_s: float = 15.0):  # pragma: no cover
    # General-purpose adapters tool resolver with per-purpose cache
    t = _ADAPTORS_TOOL_CACHE.get(purpose)
    if t is not None:
        return t
    _ensure_adapters_client()
    # MultiServerMCPClient.get_tools() is async; returns LangChain tools
    tools = _adapters_run_coro(_ADAPTORS_CLIENT.get_tools(), timeout_s=timeout_s)
    picked = None
    # Tools likely have .name attribute
    for name in candidates:
        picked = next((t for t in tools if getattr(t, "name", None) == name), None)
        if picked:
            break
    if picked is None and tools:
        # Last resort: first tool containing 'read'
        picked = next((x for x in tools if any(s in str(getattr(x, "name", "")).lower() for s in ("read", "search"))), None)
    if picked is None:
        raise RuntimeError(f"No compatible tool found for {purpose} via adapters")
    _ADAPTORS_TOOL_CACHE[purpose] = picked
    return picked


def _adapters_get_read_tool(timeout_s: float = 15.0):  # pragma: no cover
    global _ADAPTORS_READ_TOOL
    if _ADAPTORS_READ_TOOL is not None:
        return _ADAPTORS_READ_TOOL
    t = _adapters_get_tool("read_url", ["read_url", "jina_read_url", "read"], timeout_s=timeout_s)
    _ADAPTORS_READ_TOOL = t
    return t


def _adapters_call_read_url(url: str, timeout_s: float = 12.0) -> Optional[str]:  # pragma: no cover
    tool = _adapters_get_read_tool(timeout_s=timeout_s)
    # Tool may support .ainvoke (async) or .invoke (sync). Prefer async if present.
    if hasattr(tool, "ainvoke"):
        res = _adapters_run_coro(tool.ainvoke({"url": url}), timeout_s=timeout_s)
    elif hasattr(tool, "invoke"):
        res = tool.invoke({"url": url})
    else:
        raise RuntimeError("Adapter tool does not support invoke/ainvoke")

    # Extract text content: handle string or ToolMessage/content blocks
    if isinstance(res, str):
        return res
    # LangChain ToolMessage-like: try common fields
    content = None
    try:
        content = getattr(res, "content", None)
    except Exception:
        content = None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for c in content:
            if isinstance(c, dict):
                if c.get("type") == "text" and isinstance(c.get("text"), str):
                    parts.append(c["text"])  # standard
                elif c.get("type") == "text" and isinstance(c.get("data"), str):
                    parts.append(c["data"])  # legacy
        if parts:
            return "".join(parts)
    # Fallback to JSON dumps if needed
    try:
        import json
        return json.dumps(res, ensure_ascii=False)
    except Exception:
        return None


class _MCPRemote:
    """Minimal JSON-RPC client for `mcp-remote` over stdio.

    - Frames messages with LSP-style `Content-Length` headers.
    - Supports `initialize`/`initialized`, `tools/list`, and `tools/call`.
    - Not thread-safe per call; guard with a lock.
    """

    def __init__(self, server_url: str, api_key: str, npx_path: str = "npx", npx_package: Optional[str] = None) -> None:
        self.server_url = server_url
        self.api_key = api_key
        self.npx_path = npx_path
        # Package override unused when we call 'mcp-remote' directly, but kept for future use
        self.npx_package = npx_package or os.getenv("MCP_NPX_PACKAGE", "")
        self.proc: Optional[subprocess.Popen] = None
        self._id = 0
        self._lock = threading.Lock()
        self._cond = threading.Condition()
        self._responses: Dict[int, Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]] = {}
        self._stderr_thread: Optional[threading.Thread] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._write_lock = threading.Lock()
        self._active = False

    def _start_stderr_reader(self) -> None:
        assert self.proc and self.proc.stderr
        def _run() -> None:
            try:
                for ln in self.proc.stderr:  # type: ignore
                    try:
                        s = ln.decode(errors="ignore").rstrip()
                    except Exception:
                        s = str(ln).rstrip()
                    if s:
                        # escalate to info for visibility during integration
                        log.info("[mcp-remote stderr] %s", s)
                        # Proactive recovery on known SSE disconnect/timeout patterns
                        try:
                            line_l = s.lower()
                            if (
                                "sse error" in line_l
                                or "body timeout" in line_l
                                or "connect timeout" in line_l
                                or "econnreset" in line_l
                            ):
                                # Mark session inactive so the pool will restart on next use;
                                # also attempt a clean close to avoid process leaks.
                                log.info("[mcp] stderr indicates SSE disconnect; closing session for restart")
                                try:
                                    self._active = False
                                except Exception:
                                    pass
                                try:
                                    self.close()
                                except Exception:
                                    pass
                        except Exception:
                            pass
            except Exception:
                pass
        self._stderr_thread = threading.Thread(target=_run, daemon=True)
        self._stderr_thread.start()

    def _start_reader(self) -> None:
        assert self.proc and self.proc.stdout
        def _run() -> None:
            try:
                while True:
                    line = self.proc.stdout.readline()  # type: ignore
                    if not line:
                        return
                    try:
                        s = line.decode("utf-8", errors="ignore").strip()
                    except Exception:
                        s = str(line).strip()
                    if not s:
                        continue
                    try:
                        payload = json.loads(s)
                    except Exception:
                        # Ignore non-JSON noise
                        continue
                    if isinstance(payload, dict) and "id" in payload:
                        try:
                            rid = int(payload.get("id"))
                        except Exception:
                            continue
                        with self._cond:
                            self._responses[rid] = (
                                payload.get("result"),
                                payload.get("error"),
                            )
                            self._cond.notify_all()
            except Exception:
                return
        self._reader_thread = threading.Thread(target=_run, daemon=True)
        self._reader_thread.start()

    def start(self) -> None:
        if self.proc is not None:
            return
        env = os.environ.copy()
        # Prefer direct executable if provided; else call via npx
        exec_prog = os.getenv("MCP_EXEC", "").strip()
        # Optional: force SSE-only to avoid 'http-first' 404 then fallback
        force_sse = (os.getenv("MCP_SSE_ONLY", "false").lower() in ("1", "true", "yes", "on"))
        extra_args: list[str] = []
        # Allow power users to inject arbitrary args (e.g., "--transport sse-only")
        try:
            raw = os.getenv("MCP_REMOTE_ARGS", "").strip()
            if raw:
                extra_args.extend([x for x in raw.split() if x])
        except Exception:
            pass
        if force_sse and "--sse-only" not in extra_args:
            extra_args.append("--sse-only")

        if exec_prog:
            cmd = [
                exec_prog,
                self.server_url,
                "--header",
                f"Authorization: Bearer {self.api_key}",
            ] + extra_args
        else:
            # Call mcp-remote directly via npx (as used in demo_mcp_agent)
            cmd = [
                self.npx_path,
                "--yes",
                "mcp-remote",
                self.server_url,
                "--header",
                f"Authorization: Bearer {self.api_key}",
            ] + extra_args
        log.info("[mcp] spawning: %s", " ".join(cmd))
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        if not self.proc or not self.proc.stdin or not self.proc.stdout:
            raise RuntimeError("Failed to start mcp-remote process")
        try:
            log.info("[mcp] spawned pid=%s", self.proc.pid)
        except Exception:
            pass
        self._start_stderr_reader()
        self._start_reader()
        # Initialize session
        try:
            init_timeout = float(os.getenv("MCP_INIT_TIMEOUT_S", "25") or 25.0)
        except Exception:
            init_timeout = 25.0
        _ = self._rpc_request(
            method="initialize",
            params={
                # Include protocolVersion for stricter servers
                "protocolVersion": os.getenv("MCP_PROTOCOL_VERSION", "2024-10-07"),
                "capabilities": {"experimental": {}},
                "clientInfo": {"name": "leadgen-mcp-client", "version": "0.1"},
            },
            timeout_s=init_timeout,
        )
        self._rpc_notify("initialized", {})
        self._active = True

    def close(self) -> None:
        try:
            if self.proc and self.proc.stdin:
                try:
                    self._rpc_notify("shutdown", {})
                except Exception:
                    pass
                try:
                    self.proc.stdin.close()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            if self.proc:
                self.proc.terminate()
        except Exception:
            pass
        self.proc = None
        self._active = False

    def _send(self, obj: Dict[str, Any]) -> None:
        assert self.proc and self.proc.stdin
        data = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
        self.proc.stdin.write(data)
        self.proc.stdin.flush()

    def _rpc_request(self, *, method: str, params: Dict[str, Any], timeout_s: float = 15.0) -> Dict[str, Any]:
        with self._lock:
            self._id += 1
            rid = self._id
        req = {"jsonrpc": "2.0", "id": rid, "method": method, "params": params}
        with self._write_lock:
            self._send(req)
        # wait for response
        deadline = time.time() + timeout_s
        with self._cond:
            while True:
                if rid in self._responses:
                    result, error = self._responses.pop(rid)
                    if error:
                        raise _RPCError(str(error))
                    return result or {}
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise TimeoutError(f"RPC timeout for method {method}")
                self._cond.wait(timeout=remaining)

    def _rpc_notify(self, method: str, params: Dict[str, Any]) -> None:
        note = {"jsonrpc": "2.0", "method": method, "params": params}
        self._send(note)

    def list_tools(self, timeout_s: float = 15.0) -> Any:
        res = self._rpc_request(method="tools/list", params={}, timeout_s=timeout_s)
        return res.get("tools") if isinstance(res, dict) else res

    def call_tool(self, name: str, args: Dict[str, Any], timeout_s: float = 30.0) -> Any:
        res = self._rpc_request(
            method="tools/call",
            params={"name": name, "arguments": args},
            timeout_s=timeout_s,
        )
        return res

    def alive(self) -> bool:
        return self.proc is not None and self._active


class _ClientPool:
    def __init__(self) -> None:
        self._clients: Dict[Tuple[str, str], _MCPRemote] = {}
        self._tool_cache: Dict[Tuple[str, str], Dict[str, str]] = {}
        self._lock = threading.Lock()

    def get(self, server_url: str, api_key: str) -> _MCPRemote:
        key = (server_url, api_key)
        with self._lock:
            client = self._clients.get(key)
            if client is None or not client.alive():
                client = _MCPRemote(server_url, api_key, npx_path=os.getenv("MCP_NPX_PATH", "npx"))
                client.start()
                self._clients[key] = client
                self._tool_cache[key] = {}
            return client

    def get_cached_tool(self, server_url: str, api_key: str, purpose: str) -> Optional[str]:
        return self._tool_cache.get((server_url, api_key), {}).get(purpose)

    def set_cached_tool(self, server_url: str, api_key: str, purpose: str, name: str) -> None:
        key = (server_url, api_key)
        with self._lock:
            d = self._tool_cache.get(key) or {}
            d[purpose] = name
            self._tool_cache[key] = d

    def shutdown(self) -> None:
        with self._lock:
            for c in list(self._clients.values()):
                try:
                    c.close()
                except Exception:
                    pass
            self._clients.clear()
            self._tool_cache.clear()

    def invalidate(self, server_url: str, api_key: str) -> None:
        key = (server_url, api_key)
        with self._lock:
            c = self._clients.pop(key, None)
            if c is not None:
                try:
                    c.close()
                except Exception:
                    pass
            self._tool_cache.pop(key, None)


# Global pool instance
_POOL = _ClientPool()
import atexit as _atexit
_atexit.register(_POOL.shutdown)

# Thread pool for parallel operations (Phase 2)
_EXECUTOR: Optional[ThreadPoolExecutor] = None

def _get_executor() -> ThreadPoolExecutor:
    global _EXECUTOR
    if _EXECUTOR is None:
        try:
            max_workers = int(os.getenv("MCP_POOL_MAX_WORKERS", "4") or 4)
            max_workers = max(1, min(32, max_workers))
        except Exception:
            max_workers = 4
        _EXECUTOR = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="mcp-pool")
        try:
            _atexit.register(_EXECUTOR.shutdown, wait=False, cancel_futures=True)
        except Exception:
            _atexit.register(_EXECUTOR.shutdown)
    return _EXECUTOR


def _extract_text_from_result(result: Any) -> Optional[str]:
    if isinstance(result, dict):
        content = result.get("content")
        if isinstance(content, list):
            parts: list[str] = []
            for p in content:
                if isinstance(p, dict) and p.get("type") == "text" and isinstance(p.get("text"), str):
                    parts.append(p["text"]) 
            if parts:
                return "".join(parts)
    try:
        return json.dumps(result, ensure_ascii=False)
    except Exception:
        return None


def _extract_urls_from_text(text: str) -> List[str]:
    try:
        # Simple URL regex, http(s) only. Use regular string to avoid raw-string quoting pitfalls.
        pat = re.compile("https?://[^\\s\\)\\]\\}\\\"'>]+", flags=re.I)
        return [m.group(0) for m in pat.finditer(text or "")][:200]
    except Exception:
        return []


def _extract_urls_from_jsonlike(obj: Any) -> List[str]:
    urls: List[str] = []
    try:
        if isinstance(obj, dict):
            for key, val in obj.items():
                if key in {"url", "link", "href"} and isinstance(val, str):
                    urls.append(val)
                else:
                    urls.extend(_extract_urls_from_jsonlike(val))
        elif isinstance(obj, list):
            for item in obj:
                urls.extend(_extract_urls_from_jsonlike(item))
        elif isinstance(obj, str):
            urls.extend(_extract_urls_from_text(obj))
    except Exception:
        return []
    return urls


def _extract_list_from_result(result: Any) -> List[str]:
    # Accept multiple shapes: dict with results, list, or content text with URLs/JSON
    try:
        if isinstance(result, dict):
            if isinstance(result.get("results"), list):
                return [str(x) for x in result["results"] if isinstance(x, (str, bytes))][:200]
            content = result.get("content")
            if isinstance(content, list):
                texts: List[str] = []
                urls_from_blocks: List[str] = []
                for p in content:
                    if isinstance(p, dict) and p.get("type") == "text" and isinstance(p.get("text"), str):
                        texts.append(p["text"])  # standard text block
                    elif isinstance(p, dict) and p.get("type") in {"output_text", "observation"} and isinstance(p.get("text"), str):
                        texts.append(p["text"])
                    elif isinstance(p, dict) and p.get("type") in {"application/json", "json", "output_json"}:
                        payload = p.get("data")
                        if payload is None and isinstance(p.get("text"), str):
                            try:
                                payload = json.loads(p["text"])
                            except Exception:
                                payload = None
                        urls_from_blocks.extend(_extract_urls_from_jsonlike(payload))
                if texts:
                    joined = "\n".join(texts)
                    # Try parse JSON list
                    try:
                        j = json.loads(joined)
                        if isinstance(j, list):
                            return [str(x) for x in j if isinstance(x, (str, bytes))][:200]
                    except Exception:
                        pass
                    # Fallback: URL extraction
                    extracted = _extract_urls_from_text(joined)
                    if extracted:
                        return extracted
                if urls_from_blocks:
                    return urls_from_blocks[:200]
        elif isinstance(result, list):
            return [str(x) for x in result if isinstance(x, (str, bytes))][:200]
        # Fallback: treat as text and extract URLs
        try:
            txt = json.dumps(result, ensure_ascii=False)
        except Exception:
            txt = str(result)
        return _extract_urls_from_text(txt)
    except Exception:
        return []


def read_url(url: str, timeout_s: Optional[float] = None) -> Optional[str]:
    """Read page content via Jina MCP `read_url` tool using `mcp-remote` transport.

    Honors `MCP_SERVER_URL`, `MCP_TRANSPORT`, and `JINA_API_KEY`. If `MCP_TRANSPORT`
    is set to `python`, callers may inject an alternate implementation via tests;
    the default production path uses `mcp-remote`.
    """
    with _READ_RATE_LIMITER.token():
        return _read_url_inner(url, timeout_s=timeout_s)


def _read_url_inner(url: str, timeout_s: Optional[float] = None) -> Optional[str]:
    # Allow resetting adapter singletons on retry
    global _ADAPTORS_CLIENT, _ADAPTORS_READ_TOOL
    tenant: Optional[int] = None
    run_id: Optional[int] = None
    t0 = time.perf_counter()
    server_url = _get_server_url()
    api_key = _get_env("JINA_API_KEY")
    if not api_key:
        raise MCPClientNotImplemented("JINA_API_KEY not set; MCP client unavailable")
    # Resolve transport preferring settings.MCP_TRANSPORT (for in-process overrides),
    # then falling back to env MCP_TRANSPORT. Default remains 'python'.
    try:
        from src import settings as _settings  # type: ignore
        _trn = getattr(_settings, "MCP_TRANSPORT", None)
        if isinstance(_trn, str) and _trn.strip():
            transport_cfg = _trn.strip().lower()
        else:
            transport_cfg = (_get_env("MCP_TRANSPORT", "python") or "python").lower()
    except Exception:
        transport_cfg = (_get_env("MCP_TRANSPORT", "python") or "python").lower()
    if obs is not None:
        try:
            run_id, tenant = obs.get_run_context()
        except Exception:
            run_id, tenant = None, None
    tool = "read_url"
    try:
        # Adapters transport (Python): use langchain_mcp_adapters client
        if transport_cfg == "adapters_http":
            tcall0 = time.perf_counter()
            _use_timer = obs is not None and run_id is not None and tenant is not None
            if _use_timer:
                try:
                    ctx = obs.stage_timer(run_id, tenant, "mcp_read_url")  # type: ignore[attr-defined]
                except Exception:
                    ctx = None
            else:
                ctx = None
            text: Optional[str] = None
            attempts = max(1, _MCP_READ_MAX_ATTEMPTS)
            ctx_used = False
            for attempt in range(1, attempts + 1):
                try:
                    call_timeout = _effective_timeout(timeout_s, 12.0)
                    if ctx is not None and not ctx_used:
                        with ctx:
                            text = _adapters_call_read_url(url, timeout_s=call_timeout)
                            ctx_used = True
                    else:
                        text = _adapters_call_read_url(url, timeout_s=call_timeout)
                    break
                except Exception as _e1:
                    rate_limited = _looks_like_rate_limit(_e1)
                    try:
                        log.info(
                            "[mcp] adapters attempt=%s/%s err=%s url=%s",
                            attempt,
                            attempts,
                            type(_e1).__name__,
                            url,
                        )
                    except Exception:
                        pass
                    try:
                        _ADAPTORS_READ_TOOL = None
                        _ADAPTORS_CLIENT = None
                    except Exception:
                        pass
                    _ensure_adapters_client()
                    if attempt >= attempts:
                        raise
                    delay = _compute_backoff_delay(attempt, rate_limited)
                    if delay > 0:
                        time.sleep(delay)
                    ctx_used = True  # avoid reusing ctx after first failure
                    continue
            tcall1 = time.perf_counter()
            _metrics_record(tool, "ok", (tcall1 - tcall0))
            # Telemetry: success
            try:
                if obs is not None and run_id is not None and tenant is not None:
                    dur = int((time.perf_counter() - t0) * 1000)
                    obs.bump_vendor(run_id, tenant, vendor="jina_mcp", calls=1, errors=0)
                    obs.log_event(
                        run_id,
                        tenant,
                        stage="mcp_read_url",
                        event="finish",
                        status="ok",
                        duration_ms=dur,
                        extra={"transport": "adapters_http"},
                    )
            except Exception:
                pass
            return text

        log.info("[mcp] starting read_url transport=remote (cfg=%s) server=%s", transport_cfg, server_url)
        if transport_cfg not in ("remote", "python"):
            transport_cfg = "remote"
        tool_name = _POOL.get_cached_tool(server_url, api_key, purpose="read_url")
        text: Optional[str] = None
        attempts = max(1, _MCP_READ_MAX_ATTEMPTS)
        for attempt in range(1, attempts + 1):
            client = _POOL.get(server_url, api_key)
            if not tool_name:
                log.info("[mcp] session initialized; listing tools…")
                tools = client.list_tools(timeout_s=_effective_timeout(timeout_s, 15.0)) or []
                log.info("[mcp] tools discovered count=%s", len(tools) if isinstance(tools, list) else 0)
                for candidate in ("jina_read_url", "read_url", "read"):
                    if any(isinstance(t, dict) and t.get("name") == candidate for t in tools):
                        tool_name = candidate
                        break
                if not tool_name:
                    raise RuntimeError(
                        f"No compatible read tool found. Available: {[t.get('name') for t in tools if isinstance(t, dict)]}"
                    )
                _POOL.set_cached_tool(server_url, api_key, purpose="read_url", name=tool_name)
            try:
                log.info("[mcp] calling tool=%s url=%s attempt=%s", tool_name, url, attempt)
                tcall0 = time.perf_counter()
                result = client.call_tool(tool_name, {"url": url}, timeout_s=_effective_timeout(timeout_s, 30.0))
                tcall1 = time.perf_counter()
                text = _extract_text_from_result(result)
                log.info(
                    "[mcp] call done tool=%s url=%s text_len=%s",
                    tool_name,
                    url,
                    len(text) if isinstance(text, str) else None,
                )
                _metrics_record(tool, "ok", (tcall1 - tcall0))
                break
            except Exception as _e1:
                rate_limited = _looks_like_rate_limit(_e1)
                try:
                    log.info(
                        "[mcp] call failed attempt=%s/%s err=%s url=%s",
                        attempt,
                        attempts,
                        type(_e1).__name__,
                        url,
                    )
                except Exception:
                    pass
                if not isinstance(_e1, TimeoutError) and not rate_limited:
                    _POOL.invalidate(server_url, api_key)
                    tool_name = None
                if attempt >= attempts:
                    raise
                delay = _compute_backoff_delay(attempt, rate_limited)
                if delay > 0:
                    time.sleep(delay)
                continue
        # Telemetry: success
        try:
            if obs is not None and run_id is not None and tenant is not None:
                dur = int((time.perf_counter() - t0) * 1000)
                obs.bump_vendor(run_id, tenant, vendor="jina_mcp", calls=1, errors=0)
                obs.log_event(
                    run_id,
                    tenant,
                    stage="mcp_read_url",
                    event="finish",
                    status="ok",
                    duration_ms=dur,
                    extra={"transport": "remote", "transport_cfg": transport_cfg},
                )
        except Exception:
            pass
        return text
    except Exception as e:
        try:
            # Reflect the chosen transport for accurate diagnostics
            try:
                _t = transport_cfg
            except Exception:
                _t = "remote"
            log.info("[mcp] error transport=%s msg=%s", _t, (str(e) or type(e).__name__))
        except Exception:
            pass
        try:
            _metrics_record(tool, "error", None)
        except Exception:
            pass
        # Telemetry: record error
        try:
            if obs is not None and run_id is not None and tenant is not None:
                dur = int((time.perf_counter() - t0) * 1000)
                obs.bump_vendor(run_id, tenant, vendor="jina_mcp", calls=1, errors=1)
                obs.log_event(
                    run_id,
                    tenant,
                    stage="mcp_read_url",
                    event="error",
                    status="error",
                    duration_ms=dur,
                    error_code=type(e).__name__,
                    extra={"msg": str(e)[:200] if str(e) else None, "transport": "remote", "transport_cfg": transport_cfg},
                )
        except Exception:
            pass
        raise


def _adapters_call_search_web(query: str, *, country: Optional[str], max_results: int, timeout_s: float) -> List[str]:  # pragma: no cover
    tool = _adapters_get_tool("search_web", ["search_web", "jina_search_web", "search"], timeout_s=timeout_s)
    # Invoke
    if hasattr(tool, "ainvoke"):
        res = _adapters_run_coro(tool.ainvoke({"query": query, "limit": max_results, **({"country": country} if country else {})}), timeout_s=timeout_s)
    else:
        res = tool.invoke({"query": query, "limit": max_results, **({"country": country} if country else {})})
    # Extract URLs: try JSON list, else URL mining from text
    try:
        if isinstance(res, str):
            try:
                j = json.loads(res)
                if isinstance(j, list):
                    return [str(x) for x in j][:max_results]
            except Exception:
                pass
            return _extract_urls_from_text(res)[:max_results]
        content = getattr(res, "content", None)
        if isinstance(content, str):
            try:
                j = json.loads(content)
                if isinstance(j, list):
                    return [str(x) for x in j][:max_results]
            except Exception:
                pass
            return _extract_urls_from_text(content)[:max_results]
        if isinstance(content, list):
            texts: List[str] = []
            for c in content:
                if isinstance(c, dict) and c.get("type") == "text":
                    if isinstance(c.get("text"), str):
                        texts.append(c["text"])  # standard
                    elif isinstance(c.get("data"), str):
                        texts.append(c["data"])  # legacy
            if texts:
                joined = "\n".join(texts)
                try:
                    j = json.loads(joined)
                    if isinstance(j, list):
                        return [str(x) for x in j][:max_results]
                except Exception:
                    pass
                return _extract_urls_from_text(joined)[:max_results]
        # Fallback stringify
        try:
            s = json.dumps(res, ensure_ascii=False)
        except Exception:
            s = str(res)
        return _extract_urls_from_text(s)[:max_results]
    except Exception:
        return []


def search_web(query: str, *, country: Optional[str] = None, max_results: int = 20, timeout_s: Optional[float] = None) -> List[str]:
    """Search the web via Jina MCP `search_web` tool.

    Returns a list of URLs (strings). Extracts from multiple possible result shapes.
    """
    t0 = time.perf_counter()
    server_url = _get_server_url()
    api_key = _get_env("JINA_API_KEY")
    if not api_key:
        raise MCPClientNotImplemented("JINA_API_KEY not set; MCP client unavailable")
    tool = "search_web"
    try:
        # Adapters transport branch
        try:
            from src import settings as _settings  # type: ignore
            transport_cfg = str(getattr(_settings, "MCP_TRANSPORT", "") or "").lower()
        except Exception:
            transport_cfg = (os.getenv("MCP_TRANSPORT", "") or "").lower()
        if transport_cfg == "adapters_http":
            # Stage timer
            ctx = None
            if obs is not None:
                try:
                    run_id, tenant = obs.get_run_context()
                    if run_id is not None and tenant is not None:
                        ctx = obs.stage_timer(run_id, tenant, "mcp_search_web")  # type: ignore[attr-defined]
                except Exception:
                    ctx = None
            if ctx is None:
                urls = _adapters_call_search_web(query, country=country, max_results=max_results, timeout_s=_effective_timeout(timeout_s, 30.0))
            else:
                with ctx:
                    urls = _adapters_call_search_web(query, country=country, max_results=max_results, timeout_s=_effective_timeout(timeout_s, 30.0))
            # Telemetry
            try:
                if obs is not None:
                    run_id, tenant = obs.get_run_context()
                    if run_id is not None and tenant is not None:
                        dur = int((time.perf_counter() - t0) * 1000)
                        obs.bump_vendor(run_id, tenant, vendor="jina_mcp", calls=1, errors=0)
                        obs.log_event(run_id, tenant, stage="mcp_search_web", event="finish", status="ok", duration_ms=dur, extra={"transport": "adapters_http"})
            except Exception:
                pass
            return urls[:max_results]

        # Remote transport fallback (mcp-remote)
        client = _POOL.get(server_url, api_key)
        tool_name = _POOL.get_cached_tool(server_url, api_key, purpose=tool)
        if not tool_name:
            tools = client.list_tools(timeout_s=_effective_timeout(timeout_s, 15.0)) or []
            for candidate in ("search_web", "jina_search_web", "search"):
                if any(isinstance(t, dict) and t.get("name") == candidate for t in tools):
                    tool_name = candidate
                    break
            if not tool_name:
                raise RuntimeError("No compatible search_web tool found")
            _POOL.set_cached_tool(server_url, api_key, purpose=tool, name=tool_name)
        args: Dict[str, Any] = {"query": query, "limit": max_results}
        if country:
            args["country"] = country
        log.info("[mcp] calling tool=%s query=%s limit=%s", tool_name, query, max_results)
        tcall0 = time.perf_counter()
        res = client.call_tool(tool_name, args, timeout_s=_effective_timeout(timeout_s, 30.0))
        tcall1 = time.perf_counter()
        urls = _extract_list_from_result(res)
        log.info("[mcp] search done tool=%s urls=%s", tool_name, len(urls))
        _metrics_record(tool, "ok", (tcall1 - tcall0))
        # Telemetry
        try:
            if obs is not None:
                run_id, tenant = obs.get_run_context()
                if run_id is not None and tenant is not None:
                    dur = int((time.perf_counter() - t0) * 1000)
                    obs.bump_vendor(run_id, tenant, vendor="jina_mcp", calls=1, errors=0)
                    obs.log_event(run_id, tenant, stage="mcp_search_web", event="finish", status="ok", duration_ms=dur)
        except Exception:
            pass
        return urls[:max_results]
    except Exception as e:
        try:
            log.info("[mcp] error in search_web: %s", (str(e) or type(e).__name__))
        except Exception:
            pass
        _metrics_record(tool, "error", None)
        try:
            if obs is not None:
                run_id, tenant = obs.get_run_context()
                if run_id is not None and tenant is not None:
                    dur = int((time.perf_counter() - t0) * 1000)
                    obs.bump_vendor(run_id, tenant, vendor="jina_mcp", calls=1, errors=1)
                    obs.log_event(run_id, tenant, stage="mcp_search_web", event="error", status="error", duration_ms=dur, error_code=type(e).__name__)
        except Exception:
            pass
        # Retry once; invalidate only for non-timeout
        try:
            if not isinstance(e, TimeoutError):
                _POOL.invalidate(server_url, api_key)
            client = _POOL.get(server_url, api_key)
            tool_name = _POOL.get_cached_tool(server_url, api_key, purpose=tool)
            if not tool_name:
                tools = client.list_tools(timeout_s=_effective_timeout(timeout_s, 15.0)) or []
                for candidate in ("search_web", "jina_search_web", "search"):
                    if any(isinstance(t, dict) and t.get("name") == candidate for t in tools):
                        tool_name = candidate
                        break
                if not tool_name:
                    raise
                _POOL.set_cached_tool(server_url, api_key, purpose=tool, name=tool_name)
            tcall0 = time.perf_counter()
            res = client.call_tool(tool_name, {"query": query, "limit": max_results, **({"country": country} if country else {})}, timeout_s=_effective_timeout(timeout_s, 30.0))
            tcall1 = time.perf_counter()
            urls = _extract_list_from_result(res)
            _metrics_record(tool, "ok", (tcall1 - tcall0))
            return urls[:max_results]
        except Exception:
            raise


def _adapters_call_parallel_search_web(queries: List[str], *, per_query: int, timeout_s: float) -> Dict[str, List[str]]:  # pragma: no cover
    # Try parallel tool first, else fan out using adapters search_web
    try:
        tool = _adapters_get_tool("parallel_search_web", ["parallel_search_web", "jina_parallel_search_web"], timeout_s=timeout_s)
    except Exception:
        tool = None
    if tool is not None:
        if hasattr(tool, "ainvoke"):
            res = _adapters_run_coro(tool.ainvoke({"queries": queries, "per_query": per_query}), timeout_s=timeout_s)
        else:
            res = tool.invoke({"queries": queries, "per_query": per_query})
        # Expect dict-like mapping query->list[urls]
        if isinstance(res, dict):
            out: Dict[str, List[str]] = {}
            for q, v in res.items():
                out[str(q)] = [str(x) for x in (v or []) if isinstance(x, (str, bytes))][:per_query]
            return out
        # Fallback attempt via text
        try:
            s = json.dumps(res, ensure_ascii=False)
        except Exception:
            s = str(res)
        # Not ideal; return empty mapping when not parseable
        return {q: [] for q in queries}
    # Fan out
    out: Dict[str, List[str]] = {}
    for q in queries:
        out[q] = _adapters_call_search_web(q, country=os.getenv("MCP_SEARCH_COUNTRY") or None, max_results=per_query, timeout_s=timeout_s)
    return out


def parallel_search_web(queries: List[str], *, per_query: int = 10, timeout_s: Optional[float] = None) -> Dict[str, List[str]]:
    """Parallel search using MCP `parallel_search_web` if available, else fan out `search_web` with a thread pool.
    Returns a dict query -> list[URLs].
    """
    t0 = time.perf_counter()
    server_url = _get_server_url()
    api_key = _get_env("JINA_API_KEY")
    if not api_key:
        raise MCPClientNotImplemented("JINA_API_KEY not set; MCP client unavailable")
    tool = "parallel_search_web"
    try:
        # Adapters transport branch
        try:
            from src import settings as _settings  # type: ignore
            transport_cfg = str(getattr(_settings, "MCP_TRANSPORT", "") or "").lower()
        except Exception:
            transport_cfg = (os.getenv("MCP_TRANSPORT", "") or "").lower()
        if transport_cfg == "adapters_http":
            ctx = None
            if obs is not None:
                try:
                    run_id, tenant = obs.get_run_context()
                    if run_id is not None and tenant is not None:
                        ctx = obs.stage_timer(run_id, tenant, "mcp_parallel_search_web")  # type: ignore[attr-defined]
                except Exception:
                    ctx = None
            if ctx is None:
                out = _adapters_call_parallel_search_web(queries, per_query=per_query, timeout_s=_effective_timeout(timeout_s, 45.0))
            else:
                with ctx:
                    out = _adapters_call_parallel_search_web(queries, per_query=per_query, timeout_s=_effective_timeout(timeout_s, 45.0))
            try:
                if obs is not None:
                    run_id, tenant = obs.get_run_context()
                    if run_id is not None and tenant is not None:
                        dur = int((time.perf_counter() - t0) * 1000)
                        obs.bump_vendor(run_id, tenant, vendor="jina_mcp", calls=len(queries), errors=0)
                        obs.log_event(run_id, tenant, stage="mcp_parallel_search_web", event="finish", status="ok", duration_ms=dur, extra={"transport": "adapters_http"})
            except Exception:
                pass
            return out

        client = _POOL.get(server_url, api_key)
        tool_name = _POOL.get_cached_tool(server_url, api_key, purpose=tool)
        if not tool_name:
            tools = client.list_tools(timeout_s=timeout_s or 15.0) or []
            for candidate in ("parallel_search_web", "jina_parallel_search_web"):
                if any(isinstance(t, dict) and t.get("name") == candidate for t in tools):
                    tool_name = candidate
                    break
            # If no parallel tool, fallback to thread pool fan-out
            if tool_name:
                _POOL.set_cached_tool(server_url, api_key, purpose=tool, name=tool_name)
        if tool_name:
            log.info("[mcp] calling tool=%s queries=%d per_query=%d", tool_name, len(queries), per_query)
            tcall0 = time.perf_counter()
            res = client.call_tool(tool_name, {"queries": queries, "per_query": per_query}, timeout_s=_effective_timeout(timeout_s, 45.0))
            tcall1 = time.perf_counter()
            _metrics_record(tool, "ok", (tcall1 - tcall0))
            # Expect dict-like response
            if isinstance(res, dict):
                out: Dict[str, List[str]] = {}
                for q, v in res.items():
                    out[str(q)] = [str(x) for x in (v or []) if isinstance(x, (str, bytes))][:per_query]
                return out
            # Fallback attempt to parse content
            txt = _extract_text_from_result(res) or ""
            try:
                j = json.loads(txt)
                if isinstance(j, dict):
                    return {str(k): [str(x) for x in (v or []) if isinstance(x, (str, bytes))][:per_query] for k, v in j.items()}
            except Exception:
                pass
            # Last resort: run serially
        # Fallback: thread pool fan-out of search_web
        exec = _get_executor()
        futs = {exec.submit(search_web, q, country=os.getenv("MCP_SEARCH_COUNTRY") or None, max_results=per_query, timeout_s=timeout_s): q for q in queries}
        out: Dict[str, List[str]] = {}
        for f in as_completed(futs):
            q = futs[f]
            try:
                out[q] = list(f.result() or [])[:per_query]
            except Exception:
                out[q] = []
        # Telemetry
        try:
            if obs is not None:
                run_id, tenant = obs.get_run_context()
                if run_id is not None and tenant is not None:
                    dur = int((time.perf_counter() - t0) * 1000)
                    obs.bump_vendor(run_id, tenant, vendor="jina_mcp", calls=len(queries), errors=0)
                    obs.log_event(run_id, tenant, stage="mcp_parallel_search_web", event="finish", status="ok", duration_ms=dur)
        except Exception:
            pass
        return out
    except Exception as e:
        _metrics_record(tool, "error", None)
        try:
            if obs is not None:
                run_id, tenant = obs.get_run_context()
                if run_id is not None and tenant is not None:
                    dur = int((time.perf_counter() - t0) * 1000)
                    obs.bump_vendor(run_id, tenant, vendor="jina_mcp", calls=1, errors=1)
                    obs.log_event(run_id, tenant, stage="mcp_parallel_search_web", event="error", status="error", duration_ms=dur, error_code=type(e).__name__)
        except Exception:
            pass
        # Retry once after invalidate (parallel tool path only)
        try:
            if not isinstance(e, TimeoutError):
                _POOL.invalidate(server_url, api_key)
            client = _POOL.get(server_url, api_key)
            tool_name = _POOL.get_cached_tool(server_url, api_key, purpose=tool)
            if tool_name:
                tcall0 = time.perf_counter()
                res = client.call_tool(tool_name, {"queries": queries, "per_query": per_query}, timeout_s=_effective_timeout(timeout_s, 45.0))
                tcall1 = time.perf_counter()
                _metrics_record(tool, "ok", (tcall1 - tcall0))
                if isinstance(res, dict):
                    out2: Dict[str, List[str]] = {}
                    for q, v in res.items():
                        out2[str(q)] = [str(x) for x in (v or []) if isinstance(x, (str, bytes))][:per_query]
                    return out2
        except Exception:
            pass
        # Final fallback: thread pool
        exec = _get_executor()
        futs = {exec.submit(search_web, q, country=os.getenv("MCP_SEARCH_COUNTRY") or None, max_results=per_query, timeout_s=timeout_s): q for q in queries}
        out: Dict[str, List[str]] = {}
        for f in as_completed(futs):
            q = futs[f]
            try:
                out[q] = list(f.result() or [])[:per_query]
            except Exception:
                out[q] = []
        return out
