import os
import json

# Disable network-heavy discovery for a fast, deterministic check
os.environ["ENABLE_AGENT_DISCOVERY"] = "false"

from app import pre_sdr_graph as p  # noqa: E402


def main():
    state = {
        "messages": [],
        "icp": {
            "website_url": "https://example.com",
            "seeds_list": [
                {"seed_name": "A", "domain": "a.com"},
                {"seed_name": "B", "domain": "b.com"},
                {"seed_name": "C", "domain": "c.com"},
                {"seed_name": "D", "domain": "d.com"},
                {"seed_name": "E", "domain": "e.com"},
            ],
        },
        "icp_profile": {
            "industries": ["b2b distributors"],
            "buyer_titles": ["head of ops"],
            "size_bands": ["11-50"],
            "integrations": ["shopify"],
            "triggers": ["hiring"],
        },
    }
    out = p.icp_confirm(state)
    msg = out["messages"][-1]
    content = getattr(msg, "content", str(msg))
    print("ICP_FIRST=", content.strip().startswith("ICP Profile"))
    print("CONTENT:\n" + content)


if __name__ == "__main__":
    main()

