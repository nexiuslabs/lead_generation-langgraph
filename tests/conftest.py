import os
import sys


def _ensure_repo_on_path() -> None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if root not in sys.path:
        sys.path.append(root)


_ensure_repo_on_path()
os.environ.setdefault("OPENAI_API_KEY", "test")
