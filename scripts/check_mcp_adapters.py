#!/usr/bin/env python3
"""
Quick CLI to verify Jina MCP read_url via Python adapters transport.

Forces ENABLE_MCP_READER=true, MCP_TRANSPORT=adapters_http, and MCP_DUAL_READ_PCT=0
for the current process, then calls src.jina_reader.read_url and prints the
length and preview (or full text with --full).

Requires: pip install langchain-mcp-adapters langgraph "langchain[openai]"

Examples:
  python scripts/check_mcp_adapters.py https://example.com
  python scripts/check_mcp_adapters.py https://example.com --timeout 10 --full
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Jina MCP read_url via Python adapters transport")
    parser.add_argument("url", help="URL to read")
    parser.add_argument("--timeout", type=float, default=12.0, help="Timeout in seconds (default 12.0)")
    parser.add_argument("--full", action="store_true", help="Print full text instead of preview")
    args = parser.parse_args()

    # Ensure project root is on sys.path so `src` package resolves
    try:
        _ROOT = Path(__file__).resolve().parents[1]
        if str(_ROOT) not in sys.path:
            sys.path.insert(0, str(_ROOT))
    except Exception:
        pass

    # Import settings and force adapters transport for this run
    try:
        from src import settings  # type: ignore
    except Exception as e:
        print(f"Error importing settings: {e}", file=sys.stderr)
        return 2

    # Force MCP on for this process
    setattr(settings, "ENABLE_MCP_READER", True)
    setattr(settings, "MCP_TRANSPORT", "adapters_http")
    setattr(settings, "MCP_DUAL_READ_PCT", 0)

    # Snapshot relevant config
    srv = getattr(settings, "MCP_SERVER_URL", None)
    trn = getattr(settings, "MCP_TRANSPORT", None)
    use_sse = getattr(settings, "MCP_ADAPTER_USE_SSE", False)
    std_blocks = getattr(settings, "MCP_ADAPTER_USE_STANDARD_BLOCKS", True)
    api_set = bool(os.getenv("JINA_API_KEY"))
    print("Adapters MCP config:")
    print(f"  MCP_SERVER_URL                 = {srv}")
    print(f"  MCP_TRANSPORT                  = {trn}")
    print(f"  MCP_ADAPTER_USE_SSE            = {use_sse}")
    print(f"  MCP_ADAPTER_USE_STANDARD_BLOCKS= {std_blocks}")
    print(f"  JINA_API_KEY set               = {api_set}")

    # Attempt read via wrapper (honors flag and fallback)
    try:
        from src import jina_reader  # type: ignore
        txt = jina_reader.read_url(args.url, timeout=args.timeout)
    except Exception as e:
        print(f"Error: read_url failed: {e}", file=sys.stderr)
        return 1

    if not txt:
        print("Result: empty (None or empty string)")
        return 0

    if args.full:
        print(f"Result: length={len(txt)}")
        print("--- BEGIN TEXT ---")
        try:
            print(txt)
        except Exception:
            sys.stdout.write(str(txt) + "\n")
        print("--- END TEXT ---")
    else:
        preview = txt[:200].replace("\n", " ")
        print(f"Result: length={len(txt)} preview={preview!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

