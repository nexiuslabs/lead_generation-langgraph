import logging
import sys
import os
from importlib import import_module

# Ensure package path
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

A = import_module('src.agents_icp')


class StubLLM:
    def __init__(self, *args, **kwargs):
        pass

    def invoke(self, msgs):
        class R:
            content = (
                '1. "B2B food service distribution companies"\n'
                '2. "food service suppliers for restaurants"\n'
                '3. "wholesale food distributors for food service industry"'
            )
        return R()

    def with_structured_output(self, *args, **kwargs):
        return self


def main():
    # Monkeypatch ChatOpenAI used inside discovery_planner
    A.ChatOpenAI = StubLLM  # type: ignore
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s :: %(message)s')
    state = {'icp_profile': {'industries': ['food service', 'distribution']}}
    out = A.discovery_planner(state)
    print("\n--- Discovery candidates (first 10) ---")
    print((out.get('discovery_candidates') or [])[:10])


if __name__ == '__main__':
    main()

