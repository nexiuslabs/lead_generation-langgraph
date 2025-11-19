import app.pre_sdr_graph as presdr
from langchain_core.messages import AIMessage, HumanMessage


def _router_state(user_text: str):
    return {
        "messages": [
            AIMessage(content="Here’s what I captured from your site."),
            HumanMessage(content=user_text),
        ],
        "icp": {},
        "greeting_sent": True,
        "boot_init_token": presdr.BOOT_TOKEN,
    }


def test_router_routes_profile_update_back_to_icp_when_waiting_confirmation():
    state = _router_state("update industries to add automation")
    state["site_profile_summary_sent"] = True

    assert presdr.router(state) == "icp"


def test_router_routes_confirm_profile_even_without_persisted_snapshot():
    state = _router_state("confirm profile")
    state["site_profile_summary_sent"] = True
    state["company_profile"] = {}

    assert presdr.router(state) == "icp"


def test_router_handles_icp_profile_updates():
    state = _router_state("update icp profile to add automation buyers")
    state["icp_profile_summary_sent"] = True
    state["icp"] = {"seeds_list": [{}] * 5}

    assert presdr.router(state) == "icp"


def test_router_routes_profile_show_requests_back_to_icp():
    state = _router_state("show me the updated company profile")
    state["company_profile"] = {"industries": ["saas"]}
    state["site_profile_summary_sent"] = True

    assert presdr.router(state) == "icp"


def test_router_blocks_run_discovery_until_icp_confirmed():
    state = _router_state("please run discovery")
    state["icp_profile_confirmed"] = False

    assert presdr.router(state) == "end"
    last_ai = next((msg for msg in reversed(state["messages"]) if isinstance(msg, AIMessage)), None)
    assert last_ai is not None
    assert "confirm your icp profile" in (last_ai.content or "").lower()


def test_router_allows_run_discovery_once_icp_confirmed():
    state = _router_state("run discovery now")
    state["icp_profile_confirmed"] = True

    assert presdr.router(state) == "confirm"


def test_router_confirm_icp_profile_routes_to_discovery_prompt():
    state = _router_state("confirm icp profile")
    state["icp_profile"] = {"industries": ["saas"]}
    state["icp_profile_confirmed"] = True
    state["icp_discovery_confirmed"] = False

    assert presdr.router(state) == "icp_discovery_prompt"
