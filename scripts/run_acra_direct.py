import os
import sys
import logging

# Ensure project root on path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main():
    logging.basicConfig(level=logging.INFO)
    try:
        from src.acra_direct import run_once
    except Exception as e:
        print(f"acra_direct import failed: {e}")
        sys.exit(2)
    try:
        res = run_once()
        print(f"acra_direct: {res}")
        sys.exit(0)
    except Exception as e:
        print(f"acra_direct fatal: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

