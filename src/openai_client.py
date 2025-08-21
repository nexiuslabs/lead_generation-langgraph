# ---------- src/openai_client.py ----------
import os
import json
from langchain_openai import ChatOpenAI        # install langchain-openai
from langchain_core.messages import SystemMessage, HumanMessage
from src.settings import OPENAI_API_KEY, LANGCHAIN_MODEL, TEMPERATURE

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# Initialize with no callback manager to avoid LangSmith errors
chat_client = ChatOpenAI(
    model=LANGCHAIN_MODEL,
    temperature=TEMPERATURE,
    callback_manager=None,
    verbose=False
)

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