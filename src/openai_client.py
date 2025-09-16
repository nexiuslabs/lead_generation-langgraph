# ---------- src/openai_client.py ----------
import os
import json
from langchain_openai import ChatOpenAI  # install langchain-openai
from langchain_core.messages import SystemMessage, HumanMessage
from src.settings import OPENAI_API_KEY, LANGCHAIN_MODEL, TEMPERATURE

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
    result = await chat_client.agenerate([[msg for msg in messages]])
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
        return response.data[0].embedding
    except Exception:
        return []
