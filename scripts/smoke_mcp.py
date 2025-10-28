import os
import importlib
from fastapi.testclient import TestClient

def run_case(flag: str):
    os.environ['ENABLE_MCP_READER'] = flag
    # Reload settings and app to pick up the flag
    import src.settings as settings
    importlib.reload(settings)
    import app.main as main
    importlib.reload(main)
    app = getattr(main, 'app')
    client = TestClient(app)
    resp = client.get('/health/mcp')
    print(f'flag={flag} status={resp.status_code} body={resp.text}')

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(__file__)))
    run_case('false')
    run_case('true')

