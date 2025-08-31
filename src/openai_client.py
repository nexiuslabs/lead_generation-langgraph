# ---------- src/openai_client.py ----------
import os
import json
from langchain_openai import ChatOpenAI        # install langchain-openai
from langchain_core.messages import SystemMessage, HumanMessage
from src.settings import OPENAI_API_KEY, LANGCHAIN_MODEL, TEMPERATURE

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# Initialize with no callback manager to avoid LangSmith errors
def _make_chat_client(model: str, temperature: float | None) -> ChatOpenAI:
    kwargs = {
        "model": model,
        "callback_manager": None,
        "verbose": False,
    }
    # Some models (e.g., gpt-5) only support default temperature; omit override
    if temperature is not None and not model.lower().startswith("gpt-5"):
        kwargs["temperature"] = temperature
    return ChatOpenAI(**kwargs)

chat_client = _make_chat_client(LANGCHAIN_MODEL, TEMPERATURE)

async def generate_rationale(prompt: str) -> str:
    messages = [
        SystemMessage(content="You are an SDR strategist."),
        HumanMessage(content=prompt)
    ]
    # Use agenerate for chat models
    result = await chat_client.agenerate([[msg for msg in messages]])
    # The generations structure: List[List[ChatGeneration]]
    return result.generations[0][0].message.content.strip()

# enrichment
import os
from openai import OpenAI

def get_embedding(text: str) -> list[float]:
    """
    Fetches a vector embedding for the given text using OpenAI.
    """
    client = OpenAI()
    response = client.embeddings.create(
        model=os.getenv("EMBED_MODEL", "text-embedding-ada-002"),
        input=text
    )
    return response.data[0].embedding
