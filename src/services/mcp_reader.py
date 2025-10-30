import os
import time
import json
import threading
import logging
from typing import Optional, Any, Dict, Tuple
import subprocess

try:
    from src import obs  # type: ignore
except Exception:  # pragma: no cover - obs is optional in import time
    obs = None  # type: ignore

log = logging.getLogger("mcp_reader")
if not log.handlers:
    h = logging.StreamHandler()
    log.addHandler(h)
log.setLevel(logging.INFO)


class MCPClientNotImplemented(RuntimeError):
    pass


def _get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    try:
        return os.getenv(key, default)
    except Exception:
        return default


class _RPCError(RuntimeError):
    pass


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
        if exec_prog:
            cmd = [
                exec_prog,
                self.server_url,
                "--header",
                f"Authorization: Bearer {self.api_key}",
            ]
        else:
            # Call mcp-remote directly via npx (as used in demo_mcp_agent)
            cmd = [
                self.npx_path,
                "--yes",
                "mcp-remote",
                self.server_url,
                "--header",
                f"Authorization: Bearer {self.api_key}",
            ]
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


def read_url(url: str, timeout_s: Optional[float] = None) -> Optional[str]:
    """Read page content via Jina MCP `read_url` tool using `mcp-remote` transport.

    Honors `MCP_SERVER_URL`, `MCP_TRANSPORT`, and `JINA_API_KEY`. If `MCP_TRANSPORT`
    is set to `python`, callers may inject an alternate implementation via tests;
    the default production path uses `mcp-remote`.
    """
    tenant: Optional[int] = None
    run_id: Optional[int] = None
    t0 = time.perf_counter()
    server_url = _get_env("MCP_SERVER_URL", "https://mcp.jina.ai/sse") or "https://mcp.jina.ai/sse"
    api_key = _get_env("JINA_API_KEY")
    if not api_key:
        raise MCPClientNotImplemented("JINA_API_KEY not set; MCP client unavailable")
    # Respect env, but current implementation uses the remote transport under the hood
    transport_cfg = (_get_env("MCP_TRANSPORT", "python") or "python").lower()
    if obs is not None:
        try:
            run_id, tenant = obs.get_run_context()
        except Exception:
            run_id, tenant = None, None
    try:
        log.info("[mcp] starting read_url transport=remote (cfg=%s) server=%s", transport_cfg, server_url)
        if transport_cfg not in ("remote", "python"):
            transport_cfg = "remote"
        # For now, implement remote only (effective transport='remote'). Python client can be added later.
        client = _POOL.get(server_url, api_key)
        # Use cached tool name if available to avoid listing tools every call
        tool_name = _POOL.get_cached_tool(server_url, api_key, purpose="read_url")
        try:
            if not tool_name:
                log.info("[mcp] session initialized; listing tools…")
                tools = client.list_tools(timeout_s=timeout_s or 15.0) or []
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
            log.info("[mcp] calling tool=%s url=%s", tool_name, url)
            result = client.call_tool(tool_name, {"url": url}, timeout_s=timeout_s or 30.0)
            text = _extract_text_from_result(result)
            log.info("[mcp] call done tool=%s url=%s text_len=%s", tool_name, url, len(text) if isinstance(text, str) else None)
        except Exception as _e1:
            # Auto-recover on SSE disconnects/remote timeouts: restart session once and retry
            try:
                log.info("[mcp] call failed (%s); restarting session and retrying once", type(_e1).__name__)
            except Exception:
                pass
            _POOL.invalidate(server_url, api_key)
            client = _POOL.get(server_url, api_key)
            # Reacquire tool name if not cached
            tool_name = _POOL.get_cached_tool(server_url, api_key, purpose="read_url")
            if not tool_name:
                tools = client.list_tools(timeout_s=timeout_s or 15.0) or []
                for candidate in ("jina_read_url", "read_url", "read"):
                    if any(isinstance(t, dict) and t.get("name") == candidate for t in tools):
                        tool_name = candidate
                        break
                if not tool_name:
                    raise
                _POOL.set_cached_tool(server_url, api_key, purpose="read_url", name=tool_name)
            result = client.call_tool(tool_name, {"url": url}, timeout_s=timeout_s or 30.0)
            text = _extract_text_from_result(result)
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
            log.info("[mcp] error transport=remote msg=%s", (str(e) or type(e).__name__))
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
