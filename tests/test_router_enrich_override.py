import os
import importlib

from langchain_core.messages import HumanMessage


def _import_graph_module():
    # Ensure Finder is enabled to exercise the gating code path
    os.environ.setdefault("ENABLE_ICP_INTAKE", "true")
    # Reload the module to pick up env flags for the router
    mod = importlib.import_module("app.pre_sdr_graph")
    importlib.reload(mod)
    return mod


def test_run_enrichment_bypasses_finder_hold():
    mod = _import_graph_module()

    # Minimal GraphState with last turn from human saying 'run enrichment'
    state = {
        "messages": [HumanMessage(content="run enrichment")],
        # Simulate candidates present but suggestions not done to trigger the hold path
        "candidates": [{"id": 1, "name": "Acme"}],
        "results": [],
        "finder_suggestions_done": False,
        # Ensure no duplicate-route guard triggers
        "last_routed_text": "",
    }

    dest = mod.router(state)  # type: ignore[attr-defined]
    assert dest == "enrich", f"router should route to enrich, got {dest}"

