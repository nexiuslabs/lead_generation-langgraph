# ---------- src/openai_client.py ----------
import os
import json
from langchain_openai import ChatOpenAI  # install langchain-openai
from langchain_core.messages import SystemMessage, HumanMessage
from src.settings import OPENAI_API_KEY, LANGCHAIN_MODEL, TEMPERATURE
from src import obs

# Only set env when a real key is present to avoid TypeError
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# Initialize with no callback manager to avoid LangSmith errors
def _make_chat_client(model: str, temperature: float | None):
    kwargs = {
        "model": model,
        "callback_manager": None,
        "verbose": False,
    }
    # Some models (e.g., gpt-5) only support default temperature; omit override
    if temperature is not None and not (model or "").lower().startswith("gpt-5"):
        kwargs["temperature"] = temperature
    try:
        return ChatOpenAI(**kwargs)
    except Exception:
        # Defer failure until used
        return None


chat_client = _make_chat_client(LANGCHAIN_MODEL, TEMPERATURE)


async def generate_rationale(prompt: str) -> str:
    if not chat_client:
        # Graceful fallback when no API key configured
        return "LLM disabled: rationale unavailable."
    messages = [
        SystemMessage(content="You are an SDR strategist."),
        HumanMessage(content=prompt),
    ]
    # Use agenerate for chat models
    result = None
    try:
        result = await chat_client.agenerate([[msg for msg in messages]])
    except Exception as exc:
        # Rate-limit tracking for OpenAI
        try:
            rid, tid = obs.get_run_context()
            if rid and tid:
                rl = 1 if getattr(exc, "status_code", None) == 429 or "rate" in str(exc).lower() else 0
                obs.bump_vendor(int(rid), int(tid), "openai", calls=1, errors=1, rate_limit_hits=rl)
        except Exception:
            pass
        raise
    # Vendor usage bump (OpenAI): best-effort tokens/cost extraction
    try:
        rid, tid = obs.get_run_context()
        if rid and tid:
            # Try multiple shapes for token usage
            prompt_tokens = 0
            completion_tokens = 0
            llm_output = getattr(result, "llm_output", None)
            if isinstance(llm_output, dict):
                usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
                prompt_tokens = int(usage.get("prompt_tokens") or 0)
                completion_tokens = int(usage.get("completion_tokens") or 0)
            # Some versions store in `generations` metadata
            if (prompt_tokens + completion_tokens) == 0:
                try:
                    meta = result.generations[0][0].generation_info or {}
                    usage = meta.get("token_usage") or meta.get("usage") or {}
                    prompt_tokens = int(usage.get("prompt_tokens") or 0)
                    completion_tokens = int(usage.get("completion_tokens") or 0)
                except Exception:
                    pass
            # Estimate cost from env price card (USD per 1K tokens)
            try:
                import os as _os
                in_price = float(_os.getenv("OPENAI_PRICE_INPUT_PER_1K", "0.0") or 0.0)
                out_price = float(_os.getenv("OPENAI_PRICE_OUTPUT_PER_1K", "0.0") or 0.0)
            except Exception:
                in_price = 0.0
                out_price = 0.0
            cost = (prompt_tokens/1000.0)*in_price + (completion_tokens/1000.0)*out_price
            obs.bump_vendor(int(rid), int(tid), "openai", calls=1, tokens_in=prompt_tokens, tokens_out=completion_tokens, cost_usd=cost)
    except Exception:
        pass
    # The generations structure: List[List[ChatGeneration]]
    return result.generations[0][0].message.content.strip()

# enrichment
import os
from openai import OpenAI


def get_embedding(text: str) -> list[float]:
    """Best-effort embeddings; return [] when OpenAI is not configured."""
    try:
        client = OpenAI()
        response = client.embeddings.create(
            model=os.getenv("EMBED_MODEL", "text-embedding-ada-002"), input=text
        )
        try:
            rid, tid = obs.get_run_context()
            if rid and tid:
                usage = getattr(response, "usage", None) or {}
                prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or usage.get("prompt_tokens", 0) if isinstance(usage, dict) else 0)
                try:
                    import os as _os
                    in_price = float(_os.getenv("OPENAI_PRICE_INPUT_PER_1K", "0.0") or 0.0)
                except Exception:
                    in_price = 0.0
                cost = (prompt_tokens/1000.0)*in_price
                obs.bump_vendor(int(rid), int(tid), "openai", calls=1, tokens_in=prompt_tokens, tokens_out=0, cost_usd=cost)
        except Exception:
            pass
        return response.data[0].embedding
    except Exception:
        return []
