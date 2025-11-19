from app import pre_sdr_graph as presdr
from langchain_core.messages import HumanMessage


def build_state(**overrides):
    base = {
        "messages": [],
        "icp": {},
        "candidates": [],
        "results": [],
        "confirmed": False,
        "icp_confirmed": False,
        "ask_counts": {},
        "scored": [],
    }
    base.update(overrides)
    return base


def test_profile_workflow_status_progression():
    state = build_state()
    status = presdr._profile_workflow_status(state)
    assert status.next_step == "company_profile"

    state["company_profile"] = {"summary": "Example"}
    status = presdr._profile_workflow_status(state)
    assert status.next_step == "icp_profile"

    state["icp_profile"] = {"industries": ["software"]}
    status = presdr._profile_workflow_status(state)
    assert status.next_step == "icp_discovery"

    state["icp_discovery_confirmed"] = True
    status = presdr._profile_workflow_status(state)
    assert status.next_step == "ready"


def test_sync_icp_seed_profiles_creates_snapshot_keys():
    seeds = [
        {"seed_name": "Acme", "domain": "acme.com"},
        {"seed_name": "Globex", "domain": "globex.com"},
    ]
    state = build_state(icp={"seeds_list": seeds})
    presdr._sync_icp_seed_profiles(state)
    store = state.get("icp_customer_profiles") or {}
    assert store["icp_profile_1"]["domain"] == "acme.com"
    assert store["icp_profile_2"]["seed_name"] == "Globex"
    assert store["icp_profile_1"]["confirmed"] is False


def test_router_does_not_loop_after_confirm_without_new_human():
    state = build_state(
        messages=[HumanMessage(content="confirm profile")],
        icp={},
        company_profile_confirmed=True,
        company_profile_newly_confirmed=True,
        company_profile={"summary": "Example"},
        greeting_sent=True,
        boot_init_token=presdr.BOOT_TOKEN,
        boot_seen_messages_len=1,
        last_user_boot_token=presdr.BOOT_TOKEN,
        prompt_best_customers_header=None,
    )
    result = presdr.router(state)
    assert result == "prompt_best_customers"
    presdr.prompt_best_customer_seeds_node(state)
    # Router should now wait for human input instead of re-emitting prompts
    assert presdr.router(state) == "end"
