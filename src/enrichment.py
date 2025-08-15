# tools.py
from dotenv import load_dotenv
import json
import re
import psycopg2
from psycopg2.extras import Json
from langchain_core.tools import tool
from langchain_tavily import TavilySearch, TavilyCrawl, TavilyExtract
from openai_client import get_embedding
from settings import POSTGRES_DSN, TAVILY_API_KEY, ZEROBOUNCE_API_KEY, CRAWLER_USER_AGENT, CRAWLER_TIMEOUT_S, CRAWLER_MAX_PAGES, CRAWL_MAX_PAGES, CRAWL_KEYWORDS, EXTRACT_CORPUS_CHAR_LIMIT
from urllib.parse import urlparse, urljoin
import requests
from crawler import crawl_site
import asyncio
import httpx
from bs4 import BeautifulSoup

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

# -------------- Tavily merged-corpus helpers --------------

def _clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s

async def _fetch(client: httpx.AsyncClient, url: str) -> str:
    r = await client.get(url, follow_redirects=True, timeout=12)
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
    if len(combined) > char_limit:
        combined = combined[:char_limit] + "\n\n[TRUNCATED]"
    return combined

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
    if not domains:
        print("   ↳ No domain found; skipping")
        return
    home = domains[0]
    if not home.startswith("http"):
        home = "https://" + home

    # Discover a small set of relevant URLs
    filtered_urls = await _discover_relevant_urls(home, CRAWL_MAX_PAGES)
    if not filtered_urls:
        filtered_urls = [home]

    # Try TavilyExtract per URL
    extracted_pages: list[dict] = []
    for u in filtered_urls:
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

    # If Tavily had no bodies, fall back to HTTP fetch
    if not extracted_pages:
        try:
            async with httpx.AsyncClient(headers={"User-Agent": CRAWLER_USER_AGENT}) as client:
                resps = await asyncio.gather(*(client.get(u, follow_redirects=True, timeout=12) for u in filtered_urls), return_exceptions=True)
            for resp, u in zip(resps, filtered_urls):
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

    # Build combined corpus
    combined = _combine_pages(extracted_pages, EXTRACT_CORPUS_CHAR_LIMIT)

    # Use existing extraction chain with expanded schema
    schema_keys = [
        "name","industry_norm","employees_est","revenue_bucket","incorporation_year","sg_registered",
        "last_seen","website_domain","industry_code","company_size","annual_revenue","hq_city","hq_country",
        "linkedin_url","founded_year","tech_stack","ownership_type","funding_status","employee_turnover",
        "web_traffic","email","phone_number","location_city","location_country","about_text"
    ]
    try:
        ai_output = extract_chain.invoke({
            "raw_content": f"Company: {company_name}\n\n{combined}",
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
        data = json.loads(m.group(0)) if m else json.loads(ai_output)
    except Exception as e:
        print(f"   ↳ Extraction failed to parse JSON: {e}\nRAW:\n{ai_output if 'ai_output' in locals() else ''}")
        await _deterministic_crawl_and_persist(company_id, home)
        return

    # Normalize arrays
    for k in ["email", "phone_number", "tech_stack"]:
        data[k] = _ensure_list(data.get(k)) or []
    # Augment with deterministic crawler signals to fill missing fields
    try:
        data = await _merge_with_deterministic(data, home)
    except Exception as exc:
        print(f"   ↳ deterministic merge skipped: {exc}")

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
