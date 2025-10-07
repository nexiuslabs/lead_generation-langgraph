#!/usr/bin/env python3
import os
import sys
import json
import logging

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src import agents_icp as icp


def main():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    # Basic ICP hint; adjust as needed
    icp_profile = {
        "industries": ["software"],
        "buyer_titles": ["operations"],
        "integrations": ["salesforce"],
        "triggers": ["hiring"],
    }

    # If no OpenAI key, stub LLM extractor to avoid failures
    if not os.getenv("OPENAI_API_KEY"):
        def _stub_evidence_extractor(state: dict) -> dict:
            # passthrough: keep evidence structure unchanged
            return state
        icp.evidence_extractor = _stub_evidence_extractor  # type: ignore
        print("[smoke] OPENAI_API_KEY not set — using stubbed evidence_extractor")

    # Always use fallback (DDG + Jina only) for smoke check to avoid LLM/network issues
    top = icp.plan_top10_with_reasons_fallback(icp_profile)
    if not top:
        print("[smoke] No Top-10 produced. Try broader ICP or check network.")
        return 1
    print("[smoke] Top-10 (domain | score | why | snippet..):")
    for i, row in enumerate(top, 1):
        dom = row.get("domain")
        score = row.get("score")
        why = row.get("why")
        snip = (row.get("snippet") or "")[:120]
        print(f"{i:02d}. {dom} | {score} | {why} | {snip}")
    print("\n[smoke] Confirm: evidence and snippets are sourced via r.jina Reader only (no deterministic crawl).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
