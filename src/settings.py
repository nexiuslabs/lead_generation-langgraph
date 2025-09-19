import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()  # default search
# Also load from project root and src/.env if present
_SRC_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SRC_DIR.parent
load_dotenv(_ROOT_DIR / ".env")
load_dotenv(_SRC_DIR / ".env")

# Database DSN (postgres://user:pass@host:port/db)
POSTGRES_DSN = os.getenv("POSTGRES_DSN")

# Odoo has its own DSN and does not fall back to POSTGRES_DSN
ODOO_POSTGRES_DSN = os.getenv("ODOO_POSTGRES_DSN")
if not ODOO_POSTGRES_DSN:
    _local_port = os.getenv("LOCAL_PORT")
    _db_user = os.getenv("DB_USER")
    _db_password = os.getenv("DB_PASSWORD")
    _db_name = os.getenv("DB_NAME")
    if _local_port and _db_user and _db_password and _db_name:
        # Use IPv4 loopback explicitly to avoid systems preferring ::1
        ODOO_POSTGRES_DSN = (
            f"postgresql://{_db_user}:{_db_password}@127.0.0.1:{_local_port}/{_db_name}"
        )

APP_POSTGRES_DSN = POSTGRES_DSN

# OpenAI / LangChain config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ICP_RULE_NAME = os.getenv("ICP_RULE_NAME", "default")
LANGCHAIN_MODEL = os.getenv("LANGCHAIN_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))


# Turn off all LangChain tracing/telemetry
os.environ["LANGCHAIN_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
# Remove any Smith API key so no telemetry is sent
os.environ.pop("LANGSMITH_API_KEY", None)

# Tavily API Key
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
# ZeroBounce API Key
ZEROBOUNCE_API_KEY = os.getenv("ZEROBOUNCE_API_KEY")

# Add new settings below this line if needed
CRAWLER_USER_AGENT = "ICPFinder-Bot/1.0 (+https://nexiuslabs.com)"
CRAWLER_TIMEOUT_S = 30
CRAWLER_MAX_PAGES = 6

# How many on-site pages to crawl after homepage (for Tavily + merged corpus flow)
CRAWL_MAX_PAGES = int(os.getenv("CRAWL_MAX_PAGES", str(CRAWLER_MAX_PAGES)))

# Persist deterministic crawl pages for auditability
PERSIST_CRAWL_PAGES = os.getenv("PERSIST_CRAWL_PAGES", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)

# Keywords to pick high-signal pages from the site nav
CRAWL_KEYWORDS = [
    "pricing",
    "plans",
    "packages",
    "services",
    "solutions",
    "products",
    "about",
    "team",
    "contact",
    "industries",
    "sectors",
    "case studies",
    "success stories",
    "portfolio",
    "blog",
    "news",
    "insights",
    "careers",
    "jobs",
    "hiring",
]

# Max combined characters to send to LLM extraction (cost guard)
EXTRACT_CORPUS_CHAR_LIMIT = int(os.getenv("EXTRACT_CORPUS_CHAR_LIMIT", "35000"))

# --- Lusha configuration -------------------------------------------------------
# Minimal flags and keys for optional Lusha fallbacks
LUSHA_API_KEY = os.getenv("LUSHA_API_KEY", "")
LUSHA_BASE_URL = os.getenv("LUSHA_BASE_URL", "https://api.lusha.com")

# Toggle to enable/disable Lusha fallback without redeploying
# Default disabled now that Apify replaces Lusha in the pipeline
ENABLE_LUSHA_FALLBACK = os.getenv("ENABLE_LUSHA_FALLBACK", "false").lower() in (
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

# --- Feature 11: Retry/Breaker/Flags -----------------------------------------
# Retry/backoff policy (defaults align with DevPlan 11)
RETRY_MAX_ATTEMPTS = int(os.getenv("RETRY_MAX_ATTEMPTS", "3") or 3)
RETRY_BASE_DELAY_MS = int(os.getenv("RETRY_BASE_DELAY_MS", "250") or 250)
RETRY_MAX_DELAY_MS = int(os.getenv("RETRY_MAX_DELAY_MS", "4000") or 4000)

# Circuit breaker config (per-tenant, per-vendor)
CB_ERROR_THRESHOLD = int(os.getenv("CB_ERROR_THRESHOLD", "3") or 3)
CB_COOL_OFF_S = int(os.getenv("CB_COOL_OFF_S", "300") or 300)
CB_GLOBAL_EXEMPT_VENDORS = [
    v.strip().lower()
    for v in (os.getenv("CB_GLOBAL_EXEMPT_VENDORS", "") or "").split(",")
    if v.strip()
]

# Fallback toggles
ENABLE_TAVILY_FALLBACK = os.getenv("ENABLE_TAVILY_FALLBACK", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
# Prefer Apify LinkedIn by default
ENABLE_APIFY_LINKEDIN = os.getenv("ENABLE_APIFY_LINKEDIN", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)

# Odoo export defaults behavior
ODOO_EXPORT_SET_DEFAULTS = os.getenv("ODOO_EXPORT_SET_DEFAULTS", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)

# --- Feature 16: Apify LinkedIn configuration ---------------------------------
APIFY_TOKEN = os.getenv("APIFY_TOKEN")
APIFY_LINKEDIN_ACTOR_ID = os.getenv(
    "APIFY_LINKEDIN_ACTOR_ID", "dev_fusion~linkedin-profile-scraper"
)
APIFY_SEARCH_ACTOR_ID = os.getenv("APIFY_SEARCH_ACTOR_ID", "")
APIFY_SYNC_TIMEOUT_S = int(os.getenv("APIFY_SYNC_TIMEOUT_S", "600") or 600)
APIFY_DATASET_FORMAT = os.getenv("APIFY_DATASET_FORMAT", "json")
APIFY_COMPANY_ACTOR_ID = os.getenv("APIFY_COMPANY_ACTOR_ID", "harvestapi~linkedin-company")
APIFY_EMPLOYEES_ACTOR_ID = os.getenv("APIFY_EMPLOYEES_ACTOR_ID", "harvestapi~linkedin-company-employees")
APIFY_USE_COMPANY_EMPLOYEE_CHAIN = os.getenv("APIFY_USE_COMPANY_EMPLOYEE_CHAIN", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)

# Per-run/daily caps. We reuse 'contact_lookups' vendor cap for simplicity.
try:
    APIFY_DAILY_CAP = int(os.getenv("APIFY_DAILY_CAP", "50") or 50)
except Exception:
    APIFY_DAILY_CAP = 50

# Preferred contact titles (fallback); otherwise derive from ICP rules or env
CONTACT_TITLES = [
    t.strip()
    for t in (os.getenv("CONTACT_TITLES", "founder,co-founder,ceo,cto,cfo,owner,director,head of,principal") or "").split(",")
    if t.strip()
]
