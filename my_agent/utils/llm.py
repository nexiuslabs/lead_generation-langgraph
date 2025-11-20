"""
Lightweight helper around LangChain chat models used by the orchestrator.

The helper intentionally tolerates missing API credentials by falling back to
rule-based heuristics. This lets developers run the graph locally without
OpenAI access while still providing a path to production-grade prompts.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

logger = logging.getLogger(__name__)

_model_name = os.getenv("ORCHESTRATOR_MODEL", os.getenv("LANGCHAIN_MODEL", "gpt-4o-mini"))


def _get_llm():
    try:
        from langchain_openai import ChatOpenAI

        temperature = float(os.getenv("ORCHESTRATOR_TEMPERATURE", "0") or 0)
        return ChatOpenAI(model=_model_name, temperature=temperature)
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("Falling back to heuristic mode (LLM unavailable): %s", exc)
        return None


def call_llm_json(prompt: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    """Call the configured LLM and parse JSON; return fallback on error."""
    llm = _get_llm()
    if llm is None:
        return fallback
    try:
        response = llm.invoke(prompt)
        content = getattr(response, "content", None) or str(response)
        return json.loads(content)
    except Exception as exc:  # pragma: no cover - robust mode
        logger.warning("LLM JSON parse failed: %s", exc)
        return fallback
