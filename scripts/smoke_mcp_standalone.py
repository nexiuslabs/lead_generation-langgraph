import os
import sys
from urllib.parse import urlparse

def main():
    os.chdir(os.path.dirname(os.path.dirname(__file__)))
    url = sys.argv[1] if len(sys.argv) > 1 else "https://example.com"
    print(f"Testing MCP reader against {url}...")

    # Case 1: disabled flag → expect legacy fallback only
    os.environ['ENABLE_MCP_READER'] = 'false'
    from importlib import reload
    import src.settings as settings
    reload(settings)
    from src.jina_reader import read_url as legacy_or_mcp
    txt = legacy_or_mcp(url, timeout=2.0)
    print(f"flag=false: len={0 if not txt else len(txt)} (legacy path)")

    # Case 2: enabled flag → if MCP client unavailable, expect None (falls back handled by callers)
    os.environ['ENABLE_MCP_READER'] = 'true'
    reload(settings)
    from src.services.mcp_reader import read_url as mcp_read
    m = mcp_read(url, timeout=2.0)
    print(f"flag=true (mcp direct): {'ok' if m else 'unavailable'}")

if __name__ == '__main__':
    main()

