from typing import List
from src.icp_intake import map_seeds_to_evidence, refresh_icp_patterns, generate_suggestions


def run_intake_pipeline(tenant_id: int) -> List[dict]:
    """Run a light pipeline: map seeds→evidence, refresh patterns, return suggestions.

    Intended for background execution from endpoints or jobs.
    """
    try:
        map_seeds_to_evidence(tenant_id)
    except Exception:
        pass
    try:
        refresh_icp_patterns()
    except Exception:
        pass
    try:
        return generate_suggestions(tenant_id)
    except Exception:
        return []

