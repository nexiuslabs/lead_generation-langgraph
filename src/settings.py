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
# Jina API key (used for MCP auth). Prefer MCP_API_KEY if set, else fall back to JINA_API_KEY
JINA_API_KEY = os.getenv("JINA_API_KEY", "")
ICP_RULE_NAME = os.getenv("ICP_RULE_NAME", "default")
LANGCHAIN_MODEL = os.getenv("LANGCHAIN_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))

# Agent discovery toggle (use LLM-based planning/extraction in preview flows)
# Default ON to activate PRD19 agent-driven discovery.
ENABLE_AGENT_DISCOVERY = os.getenv("ENABLE_AGENT_DISCOVERY", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
AGENT_MODEL_DISCOVERY = os.getenv("AGENT_MODEL_DISCOVERY", LANGCHAIN_MODEL)


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

# Limit how many chunks we pass to LLM per company (keeps latency bounded)
LLM_MAX_CHUNKS = int(os.getenv("LLM_MAX_CHUNKS", "2") or 2)

# LLM extraction timeout per chunk (seconds)
LLM_CHUNK_TIMEOUT_S = float(os.getenv("LLM_CHUNK_TIMEOUT_S", "30") or 30)

# Timeout for deterministic merge step during llm_extract (seconds)
MERGE_DETERMINISTIC_TIMEOUT_S = float(os.getenv("MERGE_DETERMINISTIC_TIMEOUT_S", "10") or 10)

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

# ResearchOps docs root (server path)
DOCS_ROOT = os.getenv("DOCS_ROOT", str((_ROOT_DIR / "docs").resolve()))

# PRD19: ACRA/SSIC usage only in nightly (SG) — do not run in chat
ENABLE_ACRA_IN_CHAT = os.getenv("ENABLE_ACRA_IN_CHAT", "false").lower() in (
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
ENABLE_TAVILY_FALLBACK = os.getenv("ENABLE_TAVILY_FALLBACK", "false").lower() in (
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
# Domain→LinkedIn company resolver (used for Top‑10/Next‑40)
APIFY_COMPANY_FINDER_BY_DOMAIN_ACTOR_ID = os.getenv(
    "APIFY_COMPANY_FINDER_BY_DOMAIN_ACTOR_ID",
    "s-r~free-linkedin-company-finder---linkedin-address-from-any-site",
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

# --- Enrichment recency guards -------------------------------------------------
# If > 0, skip enrichment when a company has an enrichment history row updated
# within this many days. Set to 0 to disable time-based skipping.
try:
    ENRICH_RECHECK_DAYS = int(os.getenv("ENRICH_RECHECK_DAYS", "7") or 7)
except Exception:
    ENRICH_RECHECK_DAYS = 7

# If true, skip enrichment whenever ANY history exists for the company,
# regardless of how old it is. Useful to eliminate duplicate vendor costs.
ENRICH_SKIP_IF_ANY_HISTORY = os.getenv("ENRICH_SKIP_IF_ANY_HISTORY", "false").lower() in (
    "1",
    "true",
    "yes",
    "on",
)

# --- Agentic enrichment toggle -------------------------------------------------
# When true, use a planner-driven agent loop instead of the fixed graph sequence.
ENRICH_AGENTIC = os.getenv("ENRICH_AGENTIC", "false").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
# Max number of planner steps per company to prevent runaway loops
try:
    ENRICH_AGENTIC_MAX_STEPS = int(os.getenv("ENRICH_AGENTIC_MAX_STEPS", "12") or 12)
except Exception:
    ENRICH_AGENTIC_MAX_STEPS = 12

# --- Feature 17/19: ICP Finder flags ----------------------------------------
# Gate the ICP Finder endpoints and chat flow; default ON to fully replace legacy.
ENABLE_ICP_INTAKE = os.getenv("ENABLE_ICP_INTAKE", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
# When true, keep intake to minimal website + seeds; advanced A–H captured only if volunteered.
ICP_WIZARD_FAST_START_ONLY = os.getenv("ICP_WIZARD_FAST_START_ONLY", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)

# DuckDuckGo discovery controls
ENABLE_DDG_DISCOVERY = os.getenv("ENABLE_DDG_DISCOVERY", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
# When true, domain discovery must use DuckDuckGo only (no non-DDG outlink mining for discovery)
STRICT_DDG_ONLY = os.getenv("STRICT_DDG_ONLY", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)

# When true, do not generate competitor/brand-based DDG queries from seeds
STRICT_INDUSTRY_QUERY_ONLY = os.getenv("STRICT_INDUSTRY_QUERY_ONLY", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
try:
    DDG_TIMEOUT_S = float(os.getenv("DDG_TIMEOUT_S", "5") or 5)
except Exception:
    DDG_TIMEOUT_S = 5.0
try:
    DDG_MAX_CALLS = int(os.getenv("DDG_MAX_CALLS", "1") or 1)
except Exception:
    DDG_MAX_CALLS = 2
DDG_KL = os.getenv("DDG_KL", "")  # e.g., 'sg-en', 'us-en'

# Feature flag: enable SG-focused profiles (discovery gating, deny lists, markers)
ICP_SG_PROFILES = os.getenv("ICP_SG_PROFILES", "false").lower() in (
    "1",
    "true",
    "yes",
    "on",
)

# PRD Opt-2: strict hygiene and scoring guards
ENABLE_STRICT_DOMAIN_HYGIENE = os.getenv("ENABLE_STRICT_DOMAIN_HYGIENE", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
DISCOVERY_ALLOW_PORTALS = os.getenv("DISCOVERY_ALLOW_PORTALS", "false").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
try:
    MISSING_FIRMO_PENALTY = int(os.getenv("MISSING_FIRMO_PENALTY", "30") or 30)
except Exception:
    MISSING_FIRMO_PENALTY = 30
try:
    FIRMO_MIN_COMPLETENESS_FOR_BONUS = int(os.getenv("FIRMO_MIN_COMPLETENESS_FOR_BONUS", "1") or 1)
except Exception:
    FIRMO_MIN_COMPLETENESS_FOR_BONUS = 1

# --- Jina MCP integration -----------------------------------------------------
# Feature flag to enable MCP-backed reader/search
ENABLE_MCP_READER = os.getenv("ENABLE_MCP_READER", "false").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
MCP_ENDPOINT = os.getenv("MCP_ENDPOINT", "https://mcp.jina.ai/sse")
MCP_API_KEY = os.getenv("MCP_API_KEY", "") or JINA_API_KEY
try:
    MCP_TIMEOUT_S = float(os.getenv("MCP_TIMEOUT_S", "15") or 15)
except Exception:
    MCP_TIMEOUT_S = 15.0
try:
    MCP_MAX_PARALLEL = int(os.getenv("MCP_MAX_PARALLEL", "4") or 4)
except Exception:
    MCP_MAX_PARALLEL = 4
try:
    MCP_DUAL_READ_SAMPLE_PCT = int(os.getenv("MCP_DUAL_READ_SAMPLE_PCT", "0") or 0)
except Exception:
    MCP_DUAL_READ_SAMPLE_PCT = 0

# Use LangGraph Server's MCP bridge (works on Python 3.12)
ENABLE_SERVER_MCP_BRIDGE = os.getenv("ENABLE_SERVER_MCP_BRIDGE", "false").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
# Name used in /mcp UI (e.g., 'jina')
MCP_SERVER_NAME = os.getenv("MCP_SERVER_NAME", "jina")
# Base URL for LangGraph Server (where /mcp is hosted)
LGS_BASE_URL = os.getenv("LGS_BASE_URL", "http://127.0.0.1:8001")
# Optional explicit invoke URL template; supports {server} and {tool}
# Example: http://127.0.0.1:8001/mcp/servers/{server}/tools/{tool}/invoke
MCP_BRIDGE_INVOKE_URL = os.getenv("MCP_BRIDGE_INVOKE_URL", "")
# Optional JSON string of extra headers for bridge calls (e.g., auth)
MCP_BRIDGE_HEADERS_JSON = os.getenv("MCP_BRIDGE_HEADERS_JSON", "")
MCP_BRIDGE_FORCE_AUTH = os.getenv("MCP_BRIDGE_FORCE_AUTH", "false").lower() in (
    "1",
    "true",
    "yes",
    "on",
)

# MCP bridge timeouts and cooldown controls
try:
    MCP_BRIDGE_COOL_OFF_S = float(os.getenv("MCP_BRIDGE_COOL_OFF_S", "180") or 180)
except Exception:
    MCP_BRIDGE_COOL_OFF_S = 180.0
try:
    MCP_BRIDGE_CONNECT_TIMEOUT_S = float(
        os.getenv("MCP_BRIDGE_CONNECT_TIMEOUT_S", "3.05") or 3.05
    )
except Exception:
    MCP_BRIDGE_CONNECT_TIMEOUT_S = 3.05
try:
    MCP_BRIDGE_READ_TIMEOUT_S = float(os.getenv("MCP_BRIDGE_READ_TIMEOUT_S", "12") or 12)
except Exception:
    MCP_BRIDGE_READ_TIMEOUT_S = 12.0

# Optional: direct remote MCP JSON-RPC endpoint (bypasses LangGraph bridge when needed)
MCP_REMOTE_URL = os.getenv("MCP_REMOTE_URL", "https://mcp.jina.ai/mcp")
