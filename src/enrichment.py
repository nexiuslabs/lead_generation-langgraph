# tools.py
from dotenv import load_dotenv
import json
import re
import psycopg2
from psycopg2.extras import Json
from langchain_core.tools import tool
from langchain_tavily import TavilySearch, TavilyCrawl, TavilyExtract
from openai_client import get_embedding
from settings import POSTGRES_DSN, TAVILY_API_KEY, ZEROBOUNCE_API_KEY, CRAWLER_USER_AGENT, CRAWLER_TIMEOUT_S, CRAWLER_MAX_PAGES, CRAWL_MAX_PAGES, CRAWL_KEYWORDS, EXTRACT_CORPUS_CHAR_LIMIT, ENABLE_LUSHA_FALLBACK, LUSHA_API_KEY, LUSHA_PREFERRED_TITLES
from urllib.parse import urlparse, urljoin
import requests
from crawler import crawl_site
import asyncio
import httpx
from bs4 import BeautifulSoup
from lusha_client import LushaClient, LushaError

# LangChain imports for AI-driven extraction
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

print("🛠️  Initializing enrichment pipeline...")

# Initialize Tavily clients
tavily_search = TavilySearch(api_key=TAVILY_API_KEY)
tavily_crawl = TavilyCrawl(api_key=TAVILY_API_KEY)
tavily_extract = TavilyExtract(api_key=TAVILY_API_KEY)

# Initialize LangChain LLM for AI extraction
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt_template = PromptTemplate(
    input_variables=["raw_content", "schema_keys", "instructions"],
    template=(
        "You are a data extraction agent.\n"
        "Given the following raw page content, extract the fields according to the schema keys and instructions,\n"
        "and return a JSON object with keys exactly matching the schema.\n\n"
        "Schema Keys: {schema_keys}\n"
        "Instructions: {instructions}\n\n"
        "Raw Content:\n{raw_content}\n"
    )
)
extract_chain = prompt_template | llm | StrOutputParser()


def get_db_connection():
    return psycopg2.connect(dsn=POSTGRES_DSN)

# ---------- Contacts persistence helpers (DB-introspective) ----------
def _get_table_columns(conn, table_name: str) -> set:
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = %s
                """,
                (table_name,)
            )
            return {r[0] for r in cur.fetchall()}
    except Exception:
        return set()

def _get_contact_stats(company_id: int):
    """Return (total_contacts, has_named_contact, founder_present).
    Uses best-effort checks based on available columns.
    """
    total = 0
    has_named = False
    founder_present = False
    conn = None
    try:
        conn = get_db_connection()
        cols = _get_table_columns(conn, "contacts")
        with conn, conn.cursor() as cur:
            # total contacts
            cur.execute("SELECT COUNT(*) FROM contacts WHERE company_id=%s", (company_id,))
            total = int(cur.fetchone()[0] or 0)

            # any named contact
            name_conds = []
            if "title" in cols:
                name_conds.append("(title IS NOT NULL AND title <> '')")
            if "full_name" in cols:
                name_conds.append("(full_name IS NOT NULL AND full_name <> '')")
            if "first_name" in cols:
                name_conds.append("(first_name IS NOT NULL AND first_name <> '')")
            if name_conds:
                q = "SELECT COUNT(*) FROM contacts WHERE company_id=%s AND (" + " OR ".join(name_conds) + ")"
                cur.execute(q, (company_id,))
                has_named = int(cur.fetchone()[0] or 0) > 0

            # founder / leadership presence by title
            if "title" in cols:
                terms = [(t or "").strip().lower() for t in (LUSHA_PREFERRED_TITLES or "").split(",") if (t or "").strip()]
                # If titles list is empty, use a default set
                if not terms:
                    terms = ["founder", "co-founder", "ceo", "cto", "owner", "director", "head of", "principal"]
                like_clauses = ["LOWER(title) LIKE %s" for _ in terms]
                params = [f"%{t}%" for t in terms]
                q = "SELECT COUNT(*) FROM contacts WHERE company_id=%s AND (" + " OR ".join(like_clauses) + ")"
                cur.execute(q, (company_id, *params))
                founder_present = int(cur.fetchone()[0] or 0) > 0
    except Exception:
        pass
    finally:
        try:
            if conn:
                conn.close()
        except Exception:
            pass
    return total, has_named, founder_present

def _normalize_lusha_contact(c: dict) -> dict:
    """Flatten/normalize contact from Lusha enrich payload to a common schema."""
    out = {}
    out["lusha_contact_id"] = c.get("lushaContactId") or c.get("contactId") or c.get("id")
    out["first_name"] = c.get("firstName")
    out["last_name"] = c.get("lastName")
    name = c.get("name")
    if not name and (out["first_name"] or out["last_name"]):
        name = " ".join([p for p in [out["first_name"], out["last_name"]] if p])
    out["full_name"] = name
    out["title"] = c.get("jobTitle") or c.get("title")
    out["linkedin_url"] = c.get("linkedinUrl") or c.get("linkedinProfileUrl") or c.get("linkedin")
    out["company_name"] = c.get("companyName")
    out["company_domain"] = c.get("companyDomain")
    out["seniority"] = c.get("seniority")
    out["department"] = c.get("department")
    out["city"] = c.get("city") or (c.get("location") or {}).get("city") if isinstance(c.get("location"), dict) else c.get("location")
    out["country"] = c.get("country") or (c.get("location") or {}).get("country") if isinstance(c.get("location"), dict) else None

    # Emails
    emails = []
    src_emails = c.get("emailAddresses") or c.get("emails") or c.get("email_addresses")
    if isinstance(src_emails, list):
        for e in src_emails:
            if isinstance(e, dict):
                v = e.get("email") or e.get("value")
                if v:
                    emails.append(v)
            elif isinstance(e, str):
                emails.append(e)
    elif isinstance(src_emails, str):
        emails.append(src_emails)
    out["emails"] = [e for e in emails if e]

    # Phones
    phones = []
    src_phones = c.get("phoneNumbers") or c.get("phones") or c.get("phone_numbers")
    if isinstance(src_phones, list):
        for p in src_phones:
            if isinstance(p, dict):
                v = p.get("internationalNumber") or p.get("number") or p.get("value") or p.get("e164")
                if v:
                    phones.append(v)
            elif isinstance(p, str):
                phones.append(p)
    elif isinstance(src_phones, str):
        phones.append(src_phones)
    out["phones"] = [p for p in phones if p]
    return out

def upsert_contacts_from_lusha(company_id: int, lusha_contacts: list[dict]) -> tuple[int, int]:
    """Upsert contacts from Lusha into contacts table. Returns (inserted, updated)."""
    if not lusha_contacts:
        return (0, 0)
    inserted = 0
    updated = 0
    conn = get_db_connection()
    try:
        cols = _get_table_columns(conn, "contacts")
        has_email = "email" in cols
        has_updated_at = "updated_at" in cols
        for raw in lusha_contacts:
            c = _normalize_lusha_contact(raw)
            emails = c.get("emails") or [None]
            phone_primary = (c.get("phones") or [None])[0]
            for email in emails:
                # Build payload dynamically based on existing columns
                row = {"company_id": company_id, "contact_source": "lusha"}
                if "lusha_contact_id" in cols and c.get("lusha_contact_id"):
                    row["lusha_contact_id"] = c.get("lusha_contact_id")
                if "first_name" in cols and c.get("first_name"):
                    row["first_name"] = c.get("first_name")
                if "last_name" in cols and c.get("last_name"):
                    row["last_name"] = c.get("last_name")
                if "full_name" in cols and c.get("full_name"):
                    row["full_name"] = c.get("full_name")
                if "title" in cols and c.get("title"):
                    row["title"] = c.get("title")
                if "linkedin_url" in cols and c.get("linkedin_url"):
                    row["linkedin_url"] = c.get("linkedin_url")
                if "seniority" in cols and c.get("seniority"):
                    row["seniority"] = c.get("seniority")
                if "department" in cols and c.get("department"):
                    row["department"] = c.get("department")
                if "city" in cols and c.get("city"):
                    row["city"] = c.get("city")
                if "country" in cols and c.get("country"):
                    row["country"] = c.get("country")
                # phones
                if "phone_number" in cols and phone_primary:
                    row["phone_number"] = phone_primary
                elif "phone" in cols and phone_primary:
                    row["phone"] = phone_primary
                # email and verification placeholders
                if has_email:
                    row["email"] = email
                if "email_verified" in cols and email is not None:
                    row["email_verified"] = None
                if "verification_confidence" in cols and email is not None:
                    row["verification_confidence"] = None

                # Decide existence
                with conn, conn.cursor() as cur:
                    exists = False
                    if has_email:
                        cur.execute(
                            "SELECT 1 FROM contacts WHERE company_id=%s AND email IS NOT DISTINCT FROM %s LIMIT 1",
                            (company_id, email),
                        )
                        exists = bool(cur.fetchone())
                    # Build SQL dynamically
                    if exists:
                        set_cols = [k for k in row.keys() if k not in ("company_id", "email")]
                        if set_cols:
                            assignments = ", ".join([f"{k}=%s" for k in set_cols])
                            params = [row[k] for k in set_cols]
                            where_clause = "company_id=%s AND email IS NOT DISTINCT FROM %s" if has_email else "company_id=%s"
                            params.extend([company_id, email] if has_email else [company_id])
                            if has_updated_at:
                                assignments = assignments + ", updated_at=now()"
                            cur.execute(
                                f"UPDATE contacts SET {assignments} WHERE {where_clause}",
                                params,
                            )
                            updated += cur.rowcount or 0
                    else:
                        cols_list = list(row.keys())
                        placeholders = ",".join(["%s"] * len(cols_list))
                        cur.execute(
                            f"INSERT INTO contacts ({', '.join(cols_list)}) VALUES ({placeholders}) ON CONFLICT DO NOTHING",
                            [row[k] for k in cols_list],
                        )
                        inserted += cur.rowcount or 0
        return inserted, updated
    except Exception as e:
        print(f"       ↳ Lusha contacts upsert failed: {e}")
        return (inserted, updated)
    finally:
        try:
            conn.close()
        except Exception:
            pass

# -------------- Tavily merged-corpus helpers --------------

def _clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s

async def _fetch(client: httpx.AsyncClient, url: str) -> str:
    r = await client.get(url, follow_redirects=True, timeout=CRAWLER_TIMEOUT_S)
    r.raise_for_status()
    return r.text

async def _discover_relevant_urls(home_url: str, max_pages: int) -> list[str]:
    """Fetch homepage, parse same-domain links, keep only keyword-matching URLs."""
    parsed = urlparse(home_url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    urls: list[str] = [home_url]
    async with httpx.AsyncClient(headers={"User-Agent": CRAWLER_USER_AGENT}) as client:
        try:
            html = await _fetch(client, home_url)
        except Exception:
            return urls
        soup = BeautifulSoup(html, "html.parser")
        found = set()
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if not href or href.startswith(("#", "mailto:", "tel:")) or "javascript:" in href:
                continue
            full = urljoin(base, href)
            if urlparse(full).netloc != urlparse(base).netloc:
                continue
            label = (a.get_text(" ", strip=True) or href).lower()
            if any(k in label for k in CRAWL_KEYWORDS) or any(k in full.lower() for k in CRAWL_KEYWORDS):
                found.add(full)
            if len(found) >= (max_pages - 1):
                break
        urls += sorted(found)[: max_pages - 1]
        return urls

def _combine_pages(pages: list[dict], char_limit: int) -> str:
    """Combine extracted pages (url, title, raw_content) into a single corpus."""
    blobs: list[str] = []
    for p in pages:
        url = p.get("url") or ""
        title = _clean_text(p.get("title") or "")
        body = p.get("raw_content") or p.get("content") or p.get("html") or ""
        if isinstance(body, dict):
            body = body.get("text") or ""
        body = _clean_text(body)
        if not body and title:
            body = title
        if not body:
            continue
        blobs.append(f"[URL] {url}\n[TITLE] {title}\n[BODY]\n{body}\n")
    combined = "\n\n".join(blobs)
    print('combined', combined)
    if len(combined) > char_limit:
        combined = combined[:char_limit] + "\n\n[TRUNCATED]"
    return combined

def _make_corpus_chunks(pages: list[dict], chunk_char_size: int) -> list[str]:
    """Build corpus chunks from pages without truncation. Each chunk length <= chunk_char_size."""
    blocks: list[str] = []
    for p in pages:
        url = p.get("url") or ""
        title = _clean_text(p.get("title") or "")
        body = p.get("raw_content") or p.get("content") or p.get("html") or ""
        if isinstance(body, dict):
            body = body.get("text") or ""
        body = _clean_text(body)
        if not body and title:
            body = title
        if not body:
            continue
        blocks.append(f"[URL] {url}\n[TITLE] {title}\n[BODY]\n{body}\n")

    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for blk in blocks:
        if cur and (cur_len + len(blk) > chunk_char_size):
            chunks.append("\n\n".join(cur))
            cur = [blk]
            cur_len = len(blk)
        else:
            cur.append(blk)
            cur_len += len(blk)
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks

def _merge_extracted_records(base: dict, new: dict) -> dict:
    """Merge two extraction results. Arrays are unioned; scalars prefer non-null; about_text prefers longer."""
    if not base:
        base = {}
    base = dict(base)
    array_keys = {"email", "phone_number", "tech_stack"}
    for k, v in (new or {}).items():
        if v is None:
            continue
        if k in array_keys:
            a = base.get(k) or []
            b = v if isinstance(v, list) else [v]
            base[k] = list({*a, *b})
        elif k == "about_text":
            prev = base.get(k) or ""
            nv = v or ""
            base[k] = nv if len(nv) > len(prev) else prev
        else:
            if base.get(k) in (None, ""):
                base[k] = v
    return base

def _ensure_list(v):
    if v is None:
        return None
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        parts = [p.strip() for p in re.split(r"[,\n;]+", v) if p.strip()]
        return parts or None
    return None

async def _merge_with_deterministic(data: dict, home: str) -> dict:
    print("    🔁 Merging with deterministic signals")
    try:
        summary = await crawl_site(home, max_pages=CRAWLER_MAX_PAGES)
    except Exception as exc:
        print(f"       ↳ deterministic crawl for merge failed: {exc}")
        return data
    signals = summary.get("signals") or {}
    contact = signals.get("contact") or {}
    sig_emails = (contact.get("emails") or [])
    sig_phones = (contact.get("phones") or [])
    # merge arrays
    base_emails = _ensure_list(data.get("email")) or []
    data["email"] = sorted(set([*base_emails, *sig_emails]))[:40]
    base_phones = _ensure_list(data.get("phone_number")) or []
    data["phone_number"] = sorted(set([*base_phones, *sig_phones]))[:40]
    # tech stack from detected tech signals
    tech_values = (signals.get("tech") or {}).values()
    tech_list: list[str] = []
    for sub in tech_values:
        if isinstance(sub, list):
            tech_list.extend(sub)
    base_tech = _ensure_list(data.get("tech_stack")) or []
    data["tech_stack"] = sorted(set([*base_tech, *tech_list]))[:40]
    # about_text if missing
    if not data.get("about_text"):
        val_props = (signals.get("value_props") or [])[:6]
        if val_props:
            data["about_text"] = " | ".join(val_props)
        else:
            title = signals.get("title") or ""
            desc = signals.get("meta_description") or ""
            data["about_text"] = (title + " - " + desc).strip(" -")
    # jobs_count from open roles
    if (data.get("jobs_count") in (None, 0)) and isinstance(signals.get("open_roles_count"), int):
        data["jobs_count"] = signals.get("open_roles_count", 0)
    # HQ guess if missing
    if not data.get("hq_city") or not data.get("hq_country"):
        text = ((signals.get("title") or "") + " " + (signals.get("meta_description") or "")).lower()
        if "singapore" in text or home.lower().endswith(".sg/") or ".sg" in home.lower():
            data.setdefault("hq_city", "Singapore")
            data.setdefault("hq_country", "Singapore")
    # website_domain
    if not data.get("website_domain"):
        data["website_domain"] = home
    return data

def update_company_core_fields(company_id: int, data: dict):
    """Update core scalar fields on companies table; arrays handled by store_enrichment."""
    conn = get_db_connection()
    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE companies SET
                  name = COALESCE(%s, name),
                  industry_norm = %s,
                  employees_est = %s,
                  revenue_bucket = %s,
                  incorporation_year = %s,
                  sg_registered = %s,
                  website_domain = COALESCE(%s, website_domain),
                  industry_code = %s,
                  company_size = %s,
                  annual_revenue = %s,
                  hq_city = %s,
                  hq_country = %s,
                  linkedin_url = %s,
                  founded_year = %s,
                  ownership_type = %s,
                  funding_status = %s,
                  employee_turnover = %s,
                  web_traffic = %s,
                  location_city = %s,
                  location_country = %s,
                  last_seen = now()
                WHERE company_id = %s
                """,
                (
                    data.get("name"),
                    data.get("industry_norm"),
                    data.get("employees_est"),
                    data.get("revenue_bucket"),
                    data.get("incorporation_year"),
                    data.get("sg_registered"),
                    data.get("website_domain"),
                    data.get("industry_code"),
                    data.get("company_size"),
                    data.get("annual_revenue"),
                    data.get("hq_city"),
                    data.get("hq_country"),
                    data.get("linkedin_url"),
                    data.get("founded_year"),
                    data.get("ownership_type"),
                    data.get("funding_status"),
                    data.get("employee_turnover"),
                    data.get("web_traffic"),
                    data.get("location_city"),
                    data.get("location_country"),
                    company_id,
                ),
            )
    except Exception as e:
        print(f"    ⚠️ companies core update failed: {e}")
    finally:
        conn.close()

async def _deterministic_crawl_and_persist(company_id: int, url: str):
    """Run the existing deterministic crawler and persist results to summaries and companies tables."""
    try:
        summary = await crawl_site(url, max_pages=CRAWLER_MAX_PAGES)
    except Exception as exc:
        print(f"   ↳ deterministic crawler failed: {exc}")
        return

    # Store in summaries table
    conn = psycopg2.connect(dsn=POSTGRES_DSN)
    with conn, conn.cursor() as cur:
        cur.execute(
            """
                INSERT INTO summaries (company_id, url, title, description, content_summary, key_pages, signals, rule_score, rule_band, shortlist, crawl_metadata)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT DO NOTHING
            """,
            (
                company_id,
                summary.get("url"), summary.get("title"), summary.get("description"),
                summary.get("content_summary"), Json(summary.get("key_pages")),
                Json(summary.get("signals")), summary.get("rule_score"),
                summary.get("rule_band"), Json(summary.get("shortlist")),
                Json(summary.get("crawl_metadata")),
            )
        )
    conn.close()

    # Project into company_enrichment_runs for downstream compatibility
    signals = summary.get("signals", {}) or {}
    about_text = summary.get("content_summary") or " ".join((signals.get("value_props") or [])[:6])
    tech_values = (signals.get("tech") or {}).values()
    tech_stack = sorted({t for sub in tech_values for t in (sub or [])})[:25]
    public_emails = ((signals.get("contact") or {}).get("emails") or [])[:10]
    jobs_count = signals.get("open_roles_count", 0)

    conn = get_db_connection()
    with conn, conn.cursor() as cur:
        cur.execute(
            """
                INSERT INTO company_enrichment_runs (company_id, run_timestamp, about_text, tech_stack, public_emails, jobs_count, linkedin_url)
                VALUES (%s, now(), %s, %s, %s, %s, %s)
            """,
            (company_id, about_text, tech_stack, public_emails, jobs_count, None),
        )
    conn.close()

    # Guess HQ city/country (simple heuristics)
    def _guess_city_country(sig: dict, url_: str):
        text = (sig.get("title") or "") + " " + (sig.get("meta_description") or "")
        if "singapore" in text.lower() or url_.lower().endswith(".sg/") or ".sg" in url_.lower():
            return ("Singapore", "Singapore")
        return (None, None)

    hq_city, hq_country = _guess_city_country(signals, url)
    phones = ((signals.get("contact") or {}).get("phones") or [])

    legacy = {
        "about_text": about_text or "",
        "tech_stack": tech_stack or [],
        "public_emails": public_emails or [],
        "jobs_count": jobs_count or 0,
        "linkedin_url": None,
        "phone_number": phones,
        "hq_city": hq_city,
        "hq_country": hq_country,
    }
    store_enrichment(company_id, url, legacy)

async def enrich_company_with_tavily(company_id: int, company_name: str, uen: str | None = None):
    """
    Tavily-first enrichment: discover high-signal pages, extract raw text, merge corpus,
    run existing LLM extraction on the combined text, then persist.
    Falls back to direct HTTP fetch if Tavily returns no bodies.
    """
    # Resolve homepage/domain
    domains = find_domain(company_name, uen=uen) if "find_domain" in globals() else []
    # If no domain discovered, try Lusha fallback before giving up
    if not domains and ENABLE_LUSHA_FALLBACK and LUSHA_API_KEY:
        try:
            print("   ↳ No domain found via search; trying Lusha fallback…")
            lc = LushaClient()
            lusha_domain = lc.find_company_domain(company_name)
            if lusha_domain:
                normalized = lusha_domain if lusha_domain.startswith("http") else f"https://{lusha_domain}"
                domains = [normalized]
                print(f"   ↳ Lusha provided domain: {normalized}")
        except Exception as _lusha_exc:
            print(f"   ↳ Lusha domain fallback failed: {_lusha_exc}")
    if not domains:
        print("   ↳ No domain found; skipping")
        return
    home = domains[0]
    if not home.startswith("http"):
        home = "https://" + home

    # Discover a small set of relevant URLs
    filtered_urls = await _discover_relevant_urls(home, CRAWL_MAX_PAGES)
    if not filtered_urls:
        # If discovery yielded nothing, attempt Lusha domain fallback and retry discovery
        if ENABLE_LUSHA_FALLBACK and LUSHA_API_KEY:
            try:
                lc = LushaClient()
                lusha_domain = lc.find_company_domain(company_name)
                if lusha_domain:
                    candidate_home = lusha_domain if lusha_domain.startswith("http") else f"https://{lusha_domain}"
                    if urlparse(candidate_home).netloc and urlparse(candidate_home).netloc != urlparse(home).netloc:
                        print(f"   ↳ Using Lusha-discovered domain for crawl: {candidate_home}")
                        home = candidate_home
                        filtered_urls = await _discover_relevant_urls(home, CRAWL_MAX_PAGES)
            except Exception as _lusha_exc:
                print(f"   ↳ Lusha fallback for filtered URLs failed: {_lusha_exc}")
        if not filtered_urls:
            filtered_urls = [home]

    # If we have filtered URLs, first Tavily-crawl their roots to expand coverage
    page_urls: list[str] = []
    try:
        # derive unique roots from filtered URLs
        roots: list[str] = []
        for u in filtered_urls:
            parsed = urlparse(u)
            if not parsed.scheme:
                u = "https://" + u
                parsed = urlparse(u)
            root = f"{parsed.scheme}://{parsed.netloc}"
            roots.append(root)
        # include home as a root as well
        roots.append(home)
        roots = list(dict.fromkeys(roots))

        # Ensure common About pages are included explicitly when filtered URLs exist
        if filtered_urls:
            for _root in roots:
                for _p in ("about", "aboutus"):
                    page_urls.append(f"{_root}/{_p}")

        # crawl each root/* to discover high-signal pages
        for root in roots[:3]:  # cap roots to control cost
            try:
                crawl_input = {
                    "url": f"{root}/*",
                    "limit": CRAWL_MAX_PAGES,
                    "crawl_depth": 2,
                    "instructions": f"get all pages from {root}",
                    "enable_web_search": False,
                }
                crawl_result = tavily_crawl.run(crawl_input)
                raw_urls = crawl_result.get("results") or crawl_result.get("urls") or []
                #print('raw_urls', raw_urls)
                for item in raw_urls:
                    if isinstance(item, dict) and item.get("url"):
                        page_urls.append(item["url"])
                    elif isinstance(item, str) and item.startswith("http"):
                        page_urls.append(item)
                # always ensure root is included
                page_urls.append(root)
            except Exception as exc:
                print(f"          ↳ TavilyCrawl error for {root}: {exc}")
                page_urls.append(root)
        # dedupe; do not clip to ensure we process all crawled URLs
        page_urls = list(dict.fromkeys(page_urls))
        # filter out wildcard pattern URLs like "https://domain/*" which are not fetchable
        page_urls = [u for u in page_urls if "*" not in u]
        # log discovered URLs (including about seeds)
        try:
            print(f"       ↳ Seeded/Discovered {len(page_urls)} URLs (incl. about seeds)")
            for _dbg in page_urls[:25]:
                print(f"          - {_dbg}")
        except Exception:
            pass
    except Exception as exc:
        print(f"          ↳ TavilyCrawl expansion skipped: {exc}")
        page_urls = []
    if not page_urls:
        page_urls = filtered_urls

    # Try TavilyExtract per URL
    extracted_pages: list[dict] = []
    fallback_urls: list[str] = []
    for u in page_urls:
        payload = {
            "urls": [u],
            "schema": {"raw_content": "str"},
            "instructions": "Retrieve the main textual content from this page."
        }
        try:
            raw_data = tavily_extract.run(payload)
        except Exception as exc:
            print(f"          ↳ TavilyExtract error for {u}: {exc}")
            continue
        raw_content = None
        if isinstance(raw_data, dict):
            raw_content = raw_data.get("raw_content")
            if raw_content is None and isinstance(raw_data.get("results"), list) and raw_data["results"]:
                raw_content = raw_data["results"][0].get("raw_content")
        if raw_content and isinstance(raw_content, str) and raw_content.strip():
            extracted_pages.append({"url": u, "title": "", "raw_content": raw_content})
        else:
            fallback_urls.append(u)

    # Per-URL HTTP fallback for pages where TavilyExtract returned empty
    if fallback_urls:
        try:
            print(f"       ↳ TavilyExtract empty for {len(fallback_urls)} URLs; attempting HTTP fallback")
            async with httpx.AsyncClient(headers={"User-Agent": CRAWLER_USER_AGENT}) as client:
                resps = await asyncio.gather(
                    *(client.get(u, follow_redirects=True, timeout=CRAWLER_TIMEOUT_S) for u in fallback_urls),
                    return_exceptions=True
                )
            recovered = 0
            for resp, u in zip(resps, fallback_urls):
                if isinstance(resp, Exception):
                    continue
                body = getattr(resp, "text", "")
                if body:
                    extracted_pages.append({"url": u, "html": body})
                    recovered += 1
            print(f"       ↳ HTTP fallback recovered {recovered}/{len(fallback_urls)} pages")
        except Exception as _per_url_fb_exc:
            print(f"       ↳ Per-URL HTTP fallback failed: {_per_url_fb_exc}")

    # If Tavily had no bodies, fall back to HTTP fetch
    if not extracted_pages:
        try:
            async with httpx.AsyncClient(headers={"User-Agent": CRAWLER_USER_AGENT}) as client:
                resps = await asyncio.gather(*(client.get(u, follow_redirects=True, timeout=CRAWLER_TIMEOUT_S) for u in page_urls), return_exceptions=True)
            for resp, u in zip(resps, page_urls):
                if isinstance(resp, Exception):
                    continue
                extracted_pages.append({"url": u, "html": resp.text})
        except Exception as e:
            print(f"   ↳ Fallback HTTP fetch failed: {e}")
            await _deterministic_crawl_and_persist(company_id, home)
            return
        # Still nothing? fall back to deterministic
        if not extracted_pages:
            await _deterministic_crawl_and_persist(company_id, home)
            return

    # Build combined corpus without truncation by chunking
    chunks = _make_corpus_chunks(extracted_pages, EXTRACT_CORPUS_CHAR_LIMIT)
    print(f"       ↳ {len(extracted_pages)} pages -> {len(chunks)} chunks for extraction")
    # Log the full combined content for debugging (no truncation)
    try:
        full_combined = "\n\n".join(chunks)
        print("===== BEGIN FULL COMBINED CORPUS =====")
        print(full_combined)
        print("===== END FULL COMBINED CORPUS =====")
    except Exception as _log_exc:
        print(f"       ↳ Failed to log full combined corpus: {_log_exc}")

    # Use existing extraction chain with expanded schema across chunks
    schema_keys = [
        "name","industry_norm","employees_est","revenue_bucket","incorporation_year","sg_registered",
        "last_seen","website_domain","industry_code","company_size","annual_revenue","hq_city","hq_country",
        "linkedin_url","founded_year","tech_stack","ownership_type","funding_status","employee_turnover",
        "web_traffic","email","phone_number","location_city","location_country","about_text"
    ]
    data = {}
    for i, chunk in enumerate(chunks, start=1):
        try:
            ai_output = extract_chain.invoke({
                "raw_content": f"Company: {company_name}\n\n{chunk}",
                "schema_keys": schema_keys,
                "instructions": (
                    "Return a single JSON object with only the above keys. Use null for unknown. "
                    "For tech_stack, email, and phone_number return arrays of strings. "
                    "Use integers for employees_est and incorporation_year when possible. "
                    "website_domain should be the official domain for the company. "
                    "about_text should be a concise 1-3 sentence summary of the company."
                )
            })
            m = re.search(r"\{.*\}", ai_output, re.S)
            piece = json.loads(m.group(0)) if m else json.loads(ai_output)
            data = _merge_extracted_records(data, piece)
        except Exception as e:
            print(f"   ↳ Chunk {i} extraction parse failed: {e}")
            continue

    # Normalize arrays
    for k in ["email", "phone_number", "tech_stack"]:
        data[k] = _ensure_list(data.get(k)) or []
    # Augment with deterministic crawler signals to fill missing fields
    try:
        data = await _merge_with_deterministic(data, home)
    except Exception as exc:
        print(f"   ↳ deterministic merge skipped: {exc}")

    # Lusha contacts fallback if missing contacts/names/phones/emails
    try:
        need_emails = not (data.get("email") or [])
        need_phones = not (data.get("phone_number") or [])
        total_contacts, has_named, founder_present = _get_contact_stats(company_id)
        needs_contacts = total_contacts == 0
        missing_names = not has_named
        missing_founder = not founder_present
        trigger = need_emails or need_phones or needs_contacts or missing_names or missing_founder
        if ENABLE_LUSHA_FALLBACK and LUSHA_API_KEY and trigger:
            lc = LushaClient()
            # Resolve a domain for Lusha contact search
            website_hint = data.get("website_domain") or home
            try:
                if website_hint.startswith("http"):
                    company_domain = urlparse(website_hint).netloc
                else:
                    company_domain = urlparse(f"https://{website_hint}").netloc
            except Exception:
                company_domain = None
            # First attempt with preferred titles
            lusha_contacts = lc.search_and_enrich_contacts(
                company_name=company_name,
                company_domain=company_domain,
                country=data.get("hq_country"),
                titles=LUSHA_PREFERRED_TITLES,
                limit=15,
            )
            # Retry without titles to maximize recall if none
            if not lusha_contacts:
                lusha_contacts = lc.search_and_enrich_contacts(
                    company_name=company_name,
                    company_domain=company_domain,
                    country=data.get("hq_country"),
                    titles=None,
                    limit=15,
                )
            added_emails: list[str] = []
            added_phones: list[str] = []
            for c in lusha_contacts or []:
                # Emails
                for key in ("emails", "emailAddresses", "email_addresses"):
                    val = c.get(key)
                    if isinstance(val, list):
                        for e in val:
                            if isinstance(e, dict):
                                v = e.get("email") or e.get("value")
                                if v:
                                    added_emails.append(v)
                            elif isinstance(e, str):
                                added_emails.append(e)
                    elif isinstance(val, str):
                        added_emails.append(val)
                # Phones
                for key in ("phones", "phoneNumbers", "phone_numbers"):
                    val = c.get(key)
                    if isinstance(val, list):
                        for p in val:
                            if isinstance(p, dict):
                                v = p.get("internationalNumber") or p.get("number") or p.get("value")
                                if v:
                                    added_phones.append(v)
                            elif isinstance(p, str):
                                added_phones.append(p)
                    elif isinstance(val, str):
                        added_phones.append(val)
            # Dedupe and assign to company-level arrays
            def _unique(seq: list[str]) -> list[str]:
                seen: set[str] = set()
                out: list[str] = []
                for x in seq:
                    if not x or x in seen:
                        continue
                    seen.add(x)
                    out.append(x)
                return out
            if added_emails or added_phones:
                data["email"] = _unique((data.get("email") or []) + added_emails)
                data["phone_number"] = _unique((data.get("phone_number") or []) + added_phones)
                print(f"       ↳ Lusha contacts fallback added {len(added_emails)} emails, {len(added_phones)} phones")
            # Upsert full contact rows into contacts table
            try:
                ins, upd = upsert_contacts_from_lusha(company_id, lusha_contacts or [])
                print(f"       ↳ Lusha contacts upserted: inserted={ins}, updated={upd}")
            except Exception as _upsert_exc:
                print(f"       ↳ Lusha contacts upsert error: {_upsert_exc}")
    except Exception as _lusha_contacts_exc:
        print(f"       ↳ Lusha contacts fallback failed: {_lusha_contacts_exc}")

    # Persist core fields
    update_company_core_fields(company_id, data)

    # Persist arrays and ZeroBounce via existing store_enrichment
    legacy = {
        "about_text": data.get("about_text") or "",
        "tech_stack": data.get("tech_stack") or [],
        "public_emails": data.get("email") or [],
        "jobs_count": 0,
        "linkedin_url": data.get("linkedin_url"),
        "phone_number": data.get("phone_number") or [],
        "hq_city": data.get("hq_city"),
        "hq_country": data.get("hq_country"),
        "website_domain": data.get("website_domain") or home,
        "email": data.get("email") or [],
        "products_services": [],
        "value_props": [],
        "pricing": [],
    }
    store_enrichment(company_id, home, legacy)
    print(f"    💾 stored extracted fields for company_id={company_id}")


def find_domain(company_name: str, sic_prefix: str = "", uen: str = None) -> list[str]:

    print(f"    🔍 Search domain for '{company_name}' with SIC prefix '{sic_prefix}' and UEN '{uen}'")
    try:
        query = f"{company_name} official website{' ' + sic_prefix if sic_prefix else ''}"
        results = tavily_search.run(query)
    except Exception as exc:
        print(f"       ↳ Search error: {exc}")
        return []

    hits = []
    if isinstance(results, dict) and "results" in results:
        hits = results["results"]
    elif isinstance(results, (list, tuple)):
        hits = results
    else:
        hits = [results]

    # Filter URLs to those containing the core company name (first two words)
    words = [w.strip('.,') for w in company_name.lower().split()]
    core = words[:2]
    name_nospace = "".join(core)
    name_hyphen = "-".join(core)
    filtered_urls = []
    for h in hits:
        url = h.get("url") if isinstance(h, dict) else None
        print("       ↳ URL:", url)
        if not url:
            continue
        parsed = urlparse(url)
        netloc = parsed.netloc.lower()
        if netloc.startswith("www."):
            netloc_stripped = netloc[4:]
        else:
            netloc_stripped = netloc
        # match core company name in domain only (or first word)
        domain_label = netloc_stripped.split('.')[0]
        if (name_nospace in domain_label.replace("-", "") or
            name_hyphen in netloc_stripped or
            core[0] in domain_label):
            filtered_urls.append(url)
    if filtered_urls:
        print(f"       ↳ Filtered URLs: {filtered_urls}")
        return filtered_urls
    print("       ↳ No matching URLs found.")
    return []


def qualify_pages(pages: list[dict], threshold: int = 4) -> list[dict]:
    print(f"    🔍 Qualifying {len(pages)} pages")
    prompt = PromptTemplate(
        input_variables=["url","title","content"],
        template=(
            "You are a qualifier agent. Given the following page, score 1–5 whether this is our official website or About Us page.\n"
            "Return JSON {{\"score\":<int>,\"reason\":\"<reason>\"}}.\n\n"
            "URL: {url}\n"
            "Title: {title}\n"
            "Content: {content}\n"
        )
    )
    chain = prompt | llm | StrOutputParser()
    accepted = []
    for p in pages:
        url = p.get("url") or ""
        title = p.get("title") or ""
        content = p.get("content") or ""
        try:
            output = chain.invoke({"url":url, "title":title, "content":content})
            result = json.loads(output)
            score = result.get("score", 0)
            reason = result.get("reason","")
            if score >= threshold:
                p["qualifier_reason"] = reason
                p["score"] = score
                accepted.append(p)
        except Exception as exc:
            print(f"       ↳ Qualify error for {url}: {exc}")
    return accepted

def extract_website_data(url: str) -> dict:
    print(f"    🌐 extract_website_data('{url}')")
    schema = {
        "about_text":    "str",
        "tech_stack":    "list[str]",
        "public_emails": "list[str]",
        "jobs_count":    "int",
        "linkedin_url":  "str",
        "hq_city":       "str",
        "hq_country":    "str",
        "phone_number":  "str"
    }

    # 1) Crawl starting from the root of the given URL
    parsed_url = urlparse(url)
    root = f"{parsed_url.scheme}://{parsed_url.netloc}"
    # Crawl root to get subpage URLs
    try:
        print("       ↳ Crawling for subpages…")
        crawl_input = {
            "url": f"{root}/*",
            "limit": 20,
            "crawl_depth": 2,
            "enable_web_search": False
        }
        crawl_result = tavily_crawl.run(crawl_input)
        raw_urls = crawl_result.get("results") or crawl_result.get("urls") or []
    except Exception as exc:
        print(f"       ↳ Crawl error: {exc}")
        raw_urls = []

    # normalize to unique URLs
    page_urls = []
    for u in raw_urls:
        if isinstance(u, dict) and u.get("url"):
            page_urls.append(u["url"])
        elif isinstance(u, str) and u.startswith("http"):
            page_urls.append(u)
    # Ensure the original URL (or root) is processed first
    page_urls.insert(0, url)
    page_urls = list(dict.fromkeys(page_urls))
    print(f"       ↳ {len(page_urls)} unique pages discovered")

    aggregated = {k: None for k in schema}

    # 2) For each page: extract raw_content, then refine via AI Agent
    for url in page_urls:
        print(f"       ↳ Processing page: {url}")

        # a) Extract raw_content via TavilyExtract
        payload = {
            "urls": [url],
            "schema": {"raw_content": "str"},
            "instructions": "Retrieve the main textual content from this page."
        }
        try:
            raw_data = tavily_extract.run(payload)
           # print("          ↳ Tavily raw_data:", raw_data)
        except Exception as exc:
            print(f"          ↳ TavilyExtract error: {exc}")
            continue

        # b) Pull raw_content (top-level or nested)
        raw_content = None
        if isinstance(raw_data, dict):
            # top-level
            raw_content = raw_data.get("raw_content")
            # nested under results
            if raw_content is None and isinstance(raw_data.get("results"), list) and raw_data["results"]:
                raw_content = raw_data["results"][0].get("raw_content")
        if not raw_content or not isinstance(raw_content, str) or not raw_content.strip():
            print("          ↳ No or empty raw_content found, skipping AI extraction.")
            continue
        print(f"          ↳ raw_content length: {len(raw_content)} characters")

        # 3) AI extraction
        try:
            print("          ↳ AI extraction:")
            ai_output = extract_chain.invoke({
                "raw_content": raw_content,
                "schema_keys": list(schema.keys()),
                "instructions": (
                    "Extract the About Us text, list of technologies, public business emails, "
                    "open job listing count, LinkedIn URL, HQ city & country, and phone number."
                )
            })
            # Raw AI output string
            print("          ↳ AI output string:")
            print(ai_output)
            # Pretty-print AI output JSON
            try:
                parsed = json.loads(ai_output)
                print("          ↳ AI output JSON:")
                print(json.dumps(parsed, indent=2))
                page_data = parsed
            except json.JSONDecodeError as exc:
                print(f"          ↳ AI extraction JSON parse error: {exc}")
                continue
            page_data = json.loads(ai_output)
        except Exception as exc:
            print(f"          ↳ AI extraction error: {exc}")
            continue

        # 4) Merge into aggregated
        for key in schema:
            val = page_data.get(key)
            if val is None:
                continue
            if isinstance(val, list):
                base = aggregated[key] or []
                aggregated[key] = list({*base, *val})
            else:
                aggregated[key] = val

    print(f"       ↳ Final aggregated data: {aggregated}")
    return aggregated


def verify_emails(emails: list[str]) -> list[dict]:
    """
    2.4 Email Verification via ZeroBounce adapter.
    Adapter returns dicts: {email, status, confidence, source}.
    """
    print(f"    🔒 ZeroBounce Email Verification for {emails}")
    results: list[dict] = []
    for e in emails:
        try:
            resp = requests.get(
                "https://api.zerobounce.net/v2/validate",
                params={
                    "api_key": ZEROBOUNCE_API_KEY,
                    "email": e,
                    "ip_address": ""
                },
                timeout=10
            )
            data = resp.json()
            status = data.get("status", "unknown")
            confidence = float(data.get("confidence", 0.0))
            print(f"       ✅ ZeroBounce result for {e}: status={status}, confidence={confidence}, raw={data}")
        except Exception as exc:
            print(f"       ⚠️ ZeroBounce API error for {e}: {exc}")
            status = "unknown"
            confidence = 0.0
        results.append({
            "email": e,
            "status": status,
            "confidence": confidence,
            "source": "zerobounce"
        })
    return results


def store_enrichment(company_id: int, domain: str, data: dict):
    print(f"    💾 store_enrichment({company_id}, {domain})")
    conn = get_db_connection()
    embedding = get_embedding(data.get("about_text", "") or "")
    verification = verify_emails(data.get("public_emails") or [])

    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO company_enrichment_runs
                  (company_id, about_text, tech_stack, public_emails,
                   jobs_count, linkedin_url, verification_results, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    company_id,
                    data.get("about_text"),
                    (data.get("tech_stack") or []),
                    (data.get("public_emails") or []),
                    data.get("jobs_count"),
                    data.get("linkedin_url"),
                    Json(verification),
                    embedding
                )
            )
            print("       ↳ history saved")

            cur.execute(
                """
                UPDATE companies SET
                  website_domain=%s, linkedin_url=%s, tech_stack=%s,
                  email=%s, phone_number=%s, hq_city=%s, hq_country=%s,
                  last_seen=now()
                WHERE company_id=%s
                """,
                (
                    domain,
                    data.get("linkedin_url"),
                    (data.get("tech_stack") if isinstance(data.get("tech_stack"), list) else [data.get("tech_stack")] if data.get("tech_stack") else None),
                    (data.get("public_emails") if isinstance(data.get("public_emails"), list) else [data.get("public_emails")] if data.get("public_emails") else None),
                    (data.get("phone_number") if isinstance(data.get("phone_number"), list) else [data.get("phone_number")] if data.get("phone_number") else None),
                    data.get("hq_city"),
                    data.get("hq_country"),
                    company_id
                )
            )
            print("       ↳ companies updated")

            for ver in verification:
                email_verified = True if ver.get("status") == "valid" else False
                contact_source = ver.get("source", "zerobounce")
                cur.execute(
                    """
                    INSERT INTO contacts
                      (company_id,email,email_verified,verification_confidence,
                       contact_source,created_at,updated_at)
                    VALUES (%s,%s,%s,%s,%s,now(),now())
                    ON CONFLICT DO NOTHING
                    """,
                    (
                        company_id,
                        ver["email"],
                        email_verified,
                        ver["confidence"],
                        contact_source
                    )
                )
            print("       ↳ contacts inserted")

    conn.close()
    print(f"    ✅ Done enrichment for company_id={company_id}\n")


async def enrich_company(company_id: int, company_name: str):
    # 1) find domain (your current method)
    urls = [u for u in find_domain(company_name) if u]  # filter out None/empty
    if not urls: 
        print("   ↳ No domain found; skipping")
        return
    url = urls[0]

    # 2) deterministic crawl first
    try:
        summary = await crawl_site(url, max_pages=CRAWLER_MAX_PAGES)
        # Store in summaries table
        conn = psycopg2.connect(dsn=POSTGRES_DSN)
        with conn, conn.cursor() as cur:
            cur.execute("""
                INSERT INTO summaries (company_id, url, title, description, content_summary, key_pages, signals, rule_score, rule_band, shortlist, crawl_metadata)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT DO NOTHING
            """, (
                company_id,
                summary["url"], summary.get("title"), summary.get("description"),
                summary.get("content_summary"), Json(summary.get("key_pages")),
                Json(summary.get("signals")), summary.get("rule_score"),
                summary.get("rule_band"), Json(summary.get("shortlist")),
                Json(summary.get("crawl_metadata")),
            ))
        conn.close()

        # Also project into company_enrichment_runs for downstream compatibility
        signals = summary.get("signals", {})
        about_text = summary.get("content_summary") or " ".join(signals.get("value_props", [])[:6])
        tech_stack = sorted(set(sum(signals.get("tech", {}).values(), [])))[:25]
        public_emails = (signals.get("contact") or {}).get("emails", [])[:10]
        jobs_count = signals.get("open_roles_count", 0)

        print("signals: ", signals, "about_text: ", about_text, "tech_stack: ", tech_stack, "public_emails: ", public_emails, "jobs_count: ", jobs_count)

        conn = get_db_connection()
        with conn, conn.cursor() as cur:
            cur.execute("""
                INSERT INTO company_enrichment_runs (company_id, run_timestamp, about_text, tech_stack, public_emails, jobs_count, linkedin_url)
                VALUES (%s, now(), %s, %s, %s, %s, %s)
            """, (company_id, about_text, tech_stack, public_emails, jobs_count, None))
        conn.close()

        # Prepare data dict for store_enrichment (best-effort for all fields)
        # Heuristics for city/country: use 'Singapore' if '.sg' TLD or city/country in signals, else None
        def guess_city_country(signals, url):
            # Try from signals, else guess from TLD
            city = None
            country = None
            text = (signals.get("title") or "") + " " + (signals.get("meta_description") or "")
            if "singapore" in text.lower() or url.lower().endswith(".sg/") or ".sg" in url.lower():
                city = country = "Singapore"
            # TODO: Add more heuristics as needed
            return city, country

        hq_city, hq_country = guess_city_country(signals, url)
        website_domain = url.split("/")[2] if url.startswith("http") else url
        email = public_emails[0] if public_emails else None
        phones = (signals.get("contact") or {}).get("phones", [])
        phone_number = phones[0] if phones else None
        data = {
            "about_text": about_text,
            "tech_stack": tech_stack,
            "public_emails": public_emails,
            "jobs_count": jobs_count,
            "linkedin_url": None,
            "phone_number": (signals.get("contact") or {}).get("phones", []),  # all phones
            "hq_city": hq_city,
            "hq_country": hq_country,
            "website_domain": website_domain,
            "email": public_emails,  # all emails
            "products_services": signals.get("products_services", []),
            "value_props": signals.get("value_props", []),
            "pricing": signals.get("pricing", []),
            # You can add more fields here as needed
        }
        print("DEBUG: Data dict to store_enrichment:", json.dumps(data, indent=2, default=str))
        store_enrichment(company_id, url, data)
        return  # success; skip LLM/Tavily path

    except Exception as exc:
        import traceback
        print(f"   ↳ deterministic crawler failed: {exc}. Falling back to Tavily/LLM.")
        traceback.print_exc()

    # 4) fallback to your existing Tavily + LLM extraction (current code path)
    data = extract_website_data(url)  # your existing function
    # …persist as you already do
    print(f"▶️  Enriching company_id={company_id}, name='{company_name}'")
    domains = find_domain(company_name)
    if not domains:
        print(f"   ⚠️ Skipping {company_id}: no domains found\n")
        return
    # Extract and store enrichment for each domain URL
    for idx, domain_url in enumerate(domains, start=1):
        print(f"    🌐 Processing domain ({idx}/{len(domains)}): {domain_url}")
        data = extract_website_data(domain_url)
        print(data)
        store_enrichment(company_id, domain_url, data)
