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
    state["awaiting_profile_confirmation"] = True
    state["site_profile_summary_sent"] = True

    assert presdr.router(state) == "icp"


def test_router_routes_confirm_profile_even_without_persisted_snapshot():
    state = _router_state("confirm profile")
    state["awaiting_profile_confirmation"] = True
    state["site_profile_summary_sent"] = True
    state["company_profile"] = {}

    assert presdr.router(state) == "icp"


def test_router_handles_icp_profile_updates_when_confirmation_pending():
    state = _router_state("update icp profile to add automation buyers")
    state["awaiting_icp_profile_confirmation"] = True
    state["icp_profile_summary_sent"] = True
    state["icp"] = {"seeds_list": [{}] * 5}

    assert presdr.router(state) == "icp"


def test_router_routes_profile_show_requests_back_to_icp():
    state = _router_state("show me the updated company profile")
    state["company_profile"] = {"industries": ["saas"]}
    state["site_profile_summary_sent"] = True

    assert presdr.router(state) == "icp"
