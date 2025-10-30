## Python Jina MCP Client and Business Model Agent

This document explains how MCP is set up in this project and how the Business Model Agent uses it to read website content and produce an LLM-written business model explanation.

### Overview
- MCP transport: spawns `mcp-remote` (Node) as a subprocess and communicates over stdio using JSON-RPC.
- Tools discovered at runtime from the remote MCP server (`https://mcp.jina.ai/sse`).
- The agent uses the MCP `read_url` tool to fetch page content, then uses an LLM (OpenAI) to generate the explanation.

### Key Files
- `src/jina_mcp_client/proxy.py`: Low-level MCP client that starts `mcp-remote`, lists tools, and calls tools.
- `src/jina_mcp_client/agent.py`: High-level agent that reads a URL via MCP and summarizes with an LLM.
- `src/jina_mcp_client/llm.py`: OpenAI helper to summarize text; auto-loads `.env`.
- `scripts/agent_explain.py`: CLI entry to run the agent.
- `pyproject.toml`: Dependencies (`openai`, `python-dotenv`).

### Environment Variables
- `JINA_API_KEY` (required): Auth for Jina MCP remote.
- `OPENAI_API_KEY` (required): Auth for OpenAI models used by the agent.
- `.env` file is auto-loaded if present (via `python-dotenv`).

### MCP Client Details (`proxy.py`)
Responsibilities:
- Start `mcp-remote` with the correct server URL and Authorization header.
- Initialize the MCP session and list available tools.
- Invoke tools with the correct argument schema.

Important parts:

```1:28:src/jina_mcp_client/proxy.py
import json
import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class MCPResponse:
    id: int
    result: Any | None
    error: Any | None
```

```47:77:src/jina_mcp_client/proxy.py
    def start(self) -> None:
        if self.proc is not None:
            return
        env = os.environ.copy()
        cmd = [
            self.npx_path,
            "--yes",
            "mcp-remote",
            self.server_url,
            "--header",
            f"Authorization: Bearer {self.api_key}",
        ]
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        if not self.proc or not self.proc.stdin or not self.proc.stdout:
            raise RuntimeError("Failed to start mcp-remote process")
        self._start_stderr_reader()
        _ = self._rpc_request(
            method="initialize",
            params={
                "capabilities": {"experimental": {}},
                "clientInfo": {"name": "python-jina-mcp-proxy", "version": "0.1"},
            },
        )
        self._rpc_notify("initialized", {})
```

Tool invocation and schema handling for search:

```176:191:src/jina_mcp_client/proxy.py
def web_search_parallel(queries: List[str], max_results: int = 5, server_url: str = "https://mcp.jina.ai/sse") -> Any:
    client = MCPProxyClient(server_url, get_env_api_key())
    try:
        client.start()
        tools = client.list_tools()
        tool_name = None
        # Prefer true parallel search tool if available
        for candidate in ("parallel_search_web", "web_search_parallel", "search_web", "search"):
            if any(t.get("name") == candidate for t in tools):
                tool_name = candidate
                break
        if tool_name is None:
            raise RuntimeError(f"No compatible web search tool found. Available: {[t.get('name') for t in tools]}")
        # Shape arguments according to the selected tool's expected schema
        if tool_name == "parallel_search_web":
            # Jina MCP expects an array of search configs under 'searches'
            args = {"searches": [{"query": q, "num": max_results} for q in queries]}
        elif tool_name == "search_web":
            # Jina MCP expects 'query' to be a string or array; include 'num' for limit
            args = {"query": queries, "num": max_results}
        else:
            # Fallback: try common fields
            args = {"query": queries, "num": max_results}
        result = client.call_tool(tool_name, args)
        return result
    finally:
        client.close()
```

Reading URLs via MCP:

```150:173:src/jina_mcp_client/proxy.py
def jina_read_url(url: str, server_url: str = "https://mcp.jina.ai/sse") -> str:
    client = MCPProxyClient(server_url, get_env_api_key())
    try:
        client.start()
        tools = client.list_tools()
        tool_name = None
        for candidate in ("jina_read_url", "read_url", "read"):
            if any(t.get("name") == candidate for t in tools):
                tool_name = candidate
                break
        if tool_name is None:
            raise RuntimeError(f"No compatible read tool found. Available: {[t.get('name') for t in tools]}")
        result = client.call_tool(tool_name, {"url": url})
        content = result.get("content") if isinstance(result, dict) else None
        if isinstance(content, list):
            texts: List[str] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                    texts.append(part["text"])
            if texts:
                return "".join(texts)
        return json.dumps(result, ensure_ascii=False)
    finally:
        client.close()
```

### Agent (`agent.py`)
The agent always uses the LLM for explanation. It first reads the page via MCP, then summarizes with OpenAI.

```1:38:src/jina_mcp_client/agent.py
from .proxy import jina_read_url
from .llm import summarize_with_openai


class BusinessModelAgent:
    def __init__(self, server_url: str = "https://mcp.jina.ai/sse", model: str = "gpt-4o-mini") -> None:
        self.server_url = server_url
        self.model = model

    def read_url(self, url: str) -> str:
        return jina_read_url(url, server_url=self.server_url)

    def explain_business_model(self, url: str) -> str:
        content = self.read_url(url)
        return summarize_with_openai(content, model=self.model)
```

### LLM Helper (`llm.py`)
Loads `.env` automatically (if present), then calls OpenAI Chat Completions to summarize.

```1:33:src/jina_mcp_client/llm.py
import os
from typing import Optional


def _ensure_env_loaded() -> None:
    # Try to load from .env if available
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()  # loads .env from current working dir if present
    except Exception:
        # dotenv not installed or load failed; ignore and rely on existing env
        pass


def summarize_with_openai(text: str, model: str = "gpt-4o-mini") -> str:
    _ensure_env_loaded()
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError("openai package not installed. Run 'pip install openai'.") from e

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment.")

    client = OpenAI(api_key=api_key)
    system = (
        "You are a concise business analyst. Given raw website text, extract a crisp 'business model' "
        "summary: target customers, value proposition, revenue model, product/services, and any pricing hints."
    )
    prompt = (
        "Analyze the following website content and produce a concise business model explanation.\n\n" + text[:20000]
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=800,
    )
    return resp.choices[0].message.content.strip() if resp.choices else ""
```

### CLI (`scripts/agent_explain.py`)
Simple CLI wrapper that constructs the agent, reads the URL via MCP, and prints the LLM explanation.

```1:27:scripts/agent_explain.py
import argparse
import sys

from jina_mcp_client.agent import BusinessModelAgent


def main() -> int:
    parser = argparse.ArgumentParser(description="Read a URL via Jina MCP and explain its business model (LLM-only)")
    parser.add_argument("url", help="URL to read")
    parser.add_argument("--server", dest="server_url", default="https://mcp.jina.ai/sse", help="MCP server URL")
    parser.add_argument("--model", dest="model", default="gpt-4o-mini", help="LLM model name (OpenAI)")
    args = parser.parse_args()

    agent = BusinessModelAgent(server_url=args.server_url, model=args.model)
    try:
        explanation = agent.explain_business_model(args.url)
        print(explanation)
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
```

### End-to-End Flow
1) User runs the CLI with a URL and model.
2) Agent constructs `MCPProxyClient` and calls MCP `read_url` to fetch webpage content.
3) Agent passes the raw text to OpenAI via `summarize_with_openai`.
4) The LLM returns a concise business model explanation.
5) CLI prints the explanation.

### Setup and Running
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .

# Put keys into environment or .env (auto-loaded)
export JINA_API_KEY=...
export OPENAI_API_KEY=...

python3 scripts/agent_explain.py https://nexiuslabs.com --model gpt-4o-mini
```


