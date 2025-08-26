import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()  # default search
# Also load from project root and src/.env if present
_SRC_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SRC_DIR.parent
load_dotenv(_ROOT_DIR / ".env")
load_dotenv(_SRC_DIR / ".env")

# Database DSN (postgres://user:pass@host:port/db)
POSTGRES_DSN = os.getenv("POSTGRES_DSN")
# Single source: both “app” and “odoo” use the same DSN
ODOO_POSTGRES_DSN = POSTGRES_DSN
APP_POSTGRES_DSN = POSTGRES_DSN

# OpenAI / LangChain config
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ICP_RULE_NAME = os.getenv('ICP_RULE_NAME', 'default')
LANGCHAIN_MODEL = os.getenv('LANGCHAIN_MODEL', 'gpt-4o-mini')
TEMPERATURE = float(os.getenv('TEMPERATURE', '0.3'))


# Turn off all LangChain tracing/telemetry
os.environ["LANGCHAIN_TRACING"]   = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
# Remove any Smith API key so no telemetry is sent
os.environ.pop("LANGSMITH_API_KEY", None)

# Tavily API Key
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
# ZeroBounce API Key
ZEROBOUNCE_API_KEY = os.getenv('ZEROBOUNCE_API_KEY')

# Add new settings below this line if needed
CRAWLER_USER_AGENT = "ICPFinder-Bot/1.0 (+https://nexiuslabs.com)"
CRAWLER_TIMEOUT_S = 30
CRAWLER_MAX_PAGES = 50

# How many on-site pages to crawl after homepage (for Tavily + merged corpus flow)
CRAWL_MAX_PAGES = int(os.getenv("CRAWL_MAX_PAGES", str(CRAWLER_MAX_PAGES)))

# Keywords to pick high-signal pages from the site nav
CRAWL_KEYWORDS = [
    "pricing", "plans", "packages", "services", "solutions", "products",
    "about", "team", "contact", "industries", "sectors",
    "case studies", "success stories", "portfolio",
    "blog", "news", "insights",
    "careers", "jobs", "hiring",
]

# Max combined characters to send to LLM extraction (cost guard)
EXTRACT_CORPUS_CHAR_LIMIT = int(os.getenv("EXTRACT_CORPUS_CHAR_LIMIT", "35000"))

# --- Lusha configuration -------------------------------------------------------
# Minimal flags and keys for optional Lusha fallbacks
LUSHA_API_KEY = os.getenv("LUSHA_API_KEY", "")
LUSHA_BASE_URL = os.getenv("LUSHA_BASE_URL", "https://api.lusha.com")

# Toggle to enable/disable Lusha fallback without redeploying
ENABLE_LUSHA_FALLBACK = os.getenv("ENABLE_LUSHA_FALLBACK", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)

# Titles we prefer when using Lusha to search contacts
LUSHA_PREFERRED_TITLES = [
    t.strip()
    for t in os.getenv(
        "LUSHA_PREFERRED_TITLES",
        "founder,co-founder,ceo,cto,cfo,owner,director,head of,principal",
    ).split(",")
    if t.strip()
]

# Persist full merged crawl corpus (Tavily) for transparency (dev default off)
PERSIST_CRAWL_CORPUS = os.getenv("PERSIST_CRAWL_CORPUS", "false").lower() in (
    "1",
    "true",
    "yes",
    "on",
)

# Hint enrichment for Tavily domain search
FIND_DOMAIN_USE_UEN_HINT = os.getenv("FIND_DOMAIN_USE_UEN_HINT", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
FIND_DOMAIN_USE_INDUSTRY_HINT = os.getenv(
    "FIND_DOMAIN_USE_INDUSTRY_HINT",
    "true",
).lower() in (
    "1",
    "true",
    "yes",
    "on",
)
