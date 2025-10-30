#!/usr/bin/env python3
"""
Quick CLI to verify Jina MCP read_url wiring.

Prints current MCP config, attempts to read the provided URL via
src.jina_reader.read_url, and outputs the transport, length, and a short preview.

Examples:
  python scripts/check_mcp_read_url.py https://example.com --force-mcp
  python scripts/check_mcp_read_url.py https://example.com --timeout 10
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Jina MCP read_url wiring and output length/preview/full text")
    parser.add_argument("url", help="URL to read")
    parser.add_argument("--timeout", type=float, default=12.0, help="Timeout in seconds (default 12.0)")
    parser.add_argument("--force-mcp", action="store_true", help="Force-enable MCP and set dual-read to 0 for this run")
    parser.add_argument("--direct-mcp", action="store_true", help="Bypass wrapper and call src.services.mcp_reader.read_url directly (no HTTP fallback)")
    parser.add_argument("--require-mcp", action="store_true", help="Fail if MCP cannot be used (forces a direct MCP call test first)")
    parser.add_argument("--full", action="store_true", help="Print full text instead of preview")
    args = parser.parse_args()

    # Ensure project root is on sys.path so `src` package resolves when executed via file path
    try:
        _ROOT = Path(__file__).resolve().parents[1]
        if str(_ROOT) not in sys.path:
            sys.path.insert(0, str(_ROOT))
    except Exception:
        pass

    # Load settings and optionally force MCP on
    try:
        from src import settings  # type: ignore
    except Exception as e:
        print(f"Error importing settings: {e}", file=sys.stderr)
        print("Hint: Run with 'python -m scripts.check_mcp_read_url ...' or set PYTHONPATH=.")
        return 2

    if args.force_mcp:
        # Force-enable MCP in-process: jina_reader checks settings dynamically
        setattr(settings, "ENABLE_MCP_READER", True)
        setattr(settings, "MCP_DUAL_READ_PCT", 0)

    # Print config snapshot
    ena = getattr(settings, "ENABLE_MCP_READER", False)
    srv = getattr(settings, "MCP_SERVER_URL", None)
    trn = getattr(settings, "MCP_TRANSPORT", None)
    dual = getattr(settings, "MCP_DUAL_READ_PCT", 0)
    api = os.getenv("JINA_API_KEY")
    print("MCP config:")
    print(f"  ENABLE_MCP_READER = {ena}")
    print(f"  MCP_SERVER_URL    = {srv}")
    print(f"  MCP_TRANSPORT     = {trn}")
    print(f"  MCP_DUAL_READ_PCT = {dual}")
    print(f"  JINA_API_KEY set  = {bool(api)}")

    # Optionally require MCP by calling service directly
    if args.direct_mcp or args.require_mcp:
        try:
            from src.services import mcp_reader  # type: ignore
            txt = mcp_reader.read_url(args.url, timeout_s=args.timeout)
            # Also reflect in wrapper path (best-effort)
            if args.direct_mcp:
                pass
        except Exception as e:
            print(f"Error: direct MCP read_url failed: {e}", file=sys.stderr)
            return 1
    else:
        # Attempt read via wrapper (honors flag and fallback)
        from src import jina_reader  # type: ignore
        try:
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
            # Avoid double newlines if console already wraps
            print(txt)
        except Exception:
            # Fallback safe print
            sys.stdout.write(str(txt))
            sys.stdout.write("\n")
        print("--- END TEXT ---")
    else:
        preview = txt[:200].replace("\n", " ")
        print(f"Result: length={len(txt)} preview={preview!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
