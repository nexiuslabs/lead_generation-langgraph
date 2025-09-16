import asyncio
import re
import time
import urllib.robotparser as robotparser
from typing import Any, Dict, List, Tuple
from urllib.parse import urljoin, urlparse

import httpx
import os
from collections import defaultdict
from bs4 import BeautifulSoup

DEFAULT_UA = "ICPFinder-Bot/1.0 (+https://nexiuslabs.com)"
TIMEOUT_S = 12
MAX_PAGES = 6
TARGET_KEYWORDS = [
    "pricing",
    "plans",
    "packages",
    "cost",
    "services",
    "solutions",
    "products",
    "about",
    "about us",
    "team",
    "contact",
    "contact us",
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
SKIP_EXT = re.compile(r"\.(pdf|jpg|jpeg|png|gif|zip|doc|docx)$", re.I)


class RobotsCache:
    def __init__(self):
        self._cache: Dict[str, robotparser.RobotFileParser] = {}

    async def allowed(
        self, client: httpx.AsyncClient, url: str, ua: str = DEFAULT_UA
    ) -> bool:
        base = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
        if base not in self._cache:
            rp = robotparser.RobotFileParser()
            robots_url = urljoin(base, "/robots.txt")
            try:
                resp = await client.get(
                    robots_url, headers={"User-Agent": ua}, timeout=TIMEOUT_S
                )
                rp.parse([] if resp.status_code >= 400 else resp.text.splitlines())
            except Exception:
                rp.parse([])
            self._cache[base] = rp
        return self._cache[base].can_fetch(ua, url)


ROBOTS = RobotsCache()


def _extract_emails(text: str) -> List[str]:
    return sorted(
        set(re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text))
    )


def _extract_phones(text: str) -> List[str]:
    return sorted(
        set(
            re.findall(
                r"(?:(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)?\d{3,4}[\s.-]?\d{4})",
                text,
            )
        )
    )


TECH_PATTERNS = {
    "analytics": [r"gtag\(", r"googletagmanager", r"hotjar", r"mixpanel", r"clarity"],
    "crm": [r"hubspot", r"salesforce", r"zoho", r"pipedrive", r"closeio"],
    "cms": [r"wp-content", r"wordpress", r"squarespace", r"wix", r"ghost"],
    "ecommerce": [r"shopify", r"woocommerce", r"magento", r"bigcommerce"],
    "messaging": [r"intercom", r"drift", r"tawk\.to", r"crisp"],
}


async def _fetch(client: httpx.AsyncClient, url: str) -> Tuple[str, str]:
    # Optional global per-domain limiter across concurrent tasks
    await _maybe_wait_domain(url)
    r = await client.get(
        url,
        headers={"User-Agent": DEFAULT_UA, "Accept-Language": "en"},
        timeout=TIMEOUT_S,
        follow_redirects=True,
    )
    r.raise_for_status()
    return url, r.text


# ----------------------
# Optional domain limiter
# ----------------------

class DomainLimiter:
    def __init__(self, min_interval_s: float = 0.0):
        self.min_interval = max(0.0, float(min_interval_s))
        self._last: dict[str, float] = defaultdict(lambda: 0.0)
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def wait(self, url: str):
        if self.min_interval <= 1e-9:
            return
        host = urlparse(url).hostname or "_"
        lock = self._locks[host]
        async with lock:
            now = time.time()
            delta = now - self._last[host]
            if delta < self.min_interval:
                await asyncio.sleep(self.min_interval - delta)
            self._last[host] = time.time()


_limiter = None


def _domain_limiter() -> DomainLimiter:
    global _limiter
    if _limiter is None:
        try:
            v = float(os.getenv("CRAWLER_DOMAIN_MIN_INTERVAL_S", "0.0") or "0.0")
        except Exception:
            v = 0.0
        _limiter = DomainLimiter(v)
    return _limiter


async def _maybe_wait_domain(url: str):
    try:
        await _domain_limiter().wait(url)
    except Exception:
        # Never block crawl on limiter errors
        return


def _discover_links(html: str, base_url: str) -> List[str]:
    found = set()
    for a in BeautifulSoup(html, "html.parser").find_all("a", href=True):
        href = a["href"].strip()
        if (
            not href
            or href.startswith(("#", "mailto:", "tel:"))
            or "javascript:" in href
        ):
            continue
        full = urljoin(base_url, href)
        if urlparse(full).netloc != urlparse(base_url).netloc:
            continue
        if SKIP_EXT.search(urlparse(full).path):
            continue
        label = (a.get_text(" ", strip=True) or href).lower()
        if any(k in label for k in TARGET_KEYWORDS) or any(
            k in full.lower() for k in TARGET_KEYWORDS
        ):
            found.add(full)
        if len(found) >= 6:
            break
    return list(found)


def _extract_signals(html: str, base_url: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    title = (soup.title.string or "").strip() if soup.title else ""
    meta_desc = ""
    mdesc = soup.find("meta", attrs={"name": "description"}) or soup.find(
        "meta", attrs={"property": "og:description"}
    )
    if mdesc and mdesc.get("content"):
        meta_desc = mdesc["content"].strip()

    headings = [h.get_text(" ", strip=True) for h in soup.find_all(["h1", "h2", "h3"])]
    bullets = [li.get_text(" ", strip=True) for li in soup.find_all("li")]

    value_props = [t for t in headings[:8] if 3 <= len(t.split()) <= 18]
    products_services = [t for t in bullets if 2 <= len(t.split()) <= 10][:20]

    text = soup.get_text(" ", strip=True)
    emails = _extract_emails(text)
    phones = _extract_phones(text)

    srcs = " ".join(
        [s.get("src", "") for s in soup.find_all("script")]
        + [l.get("href", "") for l in soup.find_all("link")]
    )
    inline = " ".join([s.get_text(" ", strip=True) for s in soup.find_all("script")])
    tech: Dict[str, List[str]] = {k: [] for k in TECH_PATTERNS}
    blob = (srcs + " " + inline).lower()
    for bucket, patterns in TECH_PATTERNS.items():
        for patt in patterns:
            if re.search(patt, blob):
                tech[bucket].append(patt.strip(r"\\").strip("r"))

    def has_kw(*kws):
        t = (
            title + " " + meta_desc + " " + " ".join(headings) + " ".join(bullets)
        ).lower()
        return any(k in t for k in kws)

    has_case = has_kw("case studies", "success stories", "portfolio", "clients")
    has_testimonials = has_kw("testimonials", "reviews")
    has_careers = has_kw("careers", "jobs", "hiring")

    open_roles = 0
    for m in re.finditer(r"(\d+)\s+(open roles|jobs|positions)", text.lower()):
        open_roles = max(open_roles, int(m.group(1)))

    lang = (soup.find("html") or {}).get("lang") if soup.find("html") else None
    languages = [lang] if lang else []

    pricing = []
    for li in soup.find_all("li"):
        t = li.get_text(" ", strip=True)
        if re.search(r"(plan|tier|price)", t.lower()) and re.search(
            r"(\$|\bsgd\b|\busd\b|\bper\b)", t.lower()
        ):
            pricing.append(t)
            if len(pricing) >= 10:
                break

    return {
        "title": title,
        "meta_description": meta_desc,
        "value_props": value_props,
        "products_services": products_services,
        "pricing": pricing,
        "industry_candidates": [],
        "contact": {"emails": emails, "phones": phones, "address": None},
        "tech": tech,
        "has_case_studies": has_case,
        "has_testimonials": has_testimonials,
        "has_careers_page": has_careers,
        "open_roles_count": open_roles,
        "languages": languages,
        "hiring": {"careers_page": has_careers, "open_roles": open_roles},
    }


def _derive_features(signals: Dict[str, Any]) -> Dict[str, Any]:
    text = " ".join(
        [
            signals.get("title", ""),
            signals.get("meta_description", ""),
            " ".join(signals.get("value_props", [])),
            " ".join(signals.get("products_services", [])),
        ]
    ).lower()
    b2b_terms = [
        "clients",
        "enterprise",
        "b2b",
        "solutions",
        "consulting",
        "services",
        "agency",
    ]
    b2c_terms = ["customers", "shop", "store", "retail", "buy", "purchase"]
    b2x = (
        "b2b"
        if any(t in text for t in b2b_terms)
        else ("b2c" if any(t in text for t in b2c_terms) else "unknown")
    )
    size_guess = "unknown"
    if "enterprise" in text or "global" in text:
        size_guess = "51-200"
    elif "startup" in text or "small team" in text:
        size_guess = "2-10"
    urgency_hint = (
        "1-3_months"
        if signals.get("open_roles_count", 0) > 0
        else ("3-6_months" if signals.get("has_case_studies") else "none")
    )
    buying = "multi_step_process" if size_guess == "51-200" else "single_decision_maker"
    return {
        "b2x": b2x,
        "company_size_guess": size_guess,
        "buyer_role_guess": "Unknown",
        "budget_hint": "unknown",
        "urgency_hint": urgency_hint,
        "buying_process_guess": buying,
        "triggers": ["hiring_for_growth"] if urgency_hint == "1-3_months" else [],
        "risk_flags": [],
        "problem_terms_ranked": [],
    }


def _rule_score(signals: Dict[str, Any], derived: Dict[str, Any]) -> Dict[str, Any]:
    total = 0
    total += 3 if derived["b2x"] == "b2b" else 1 if derived["b2x"] == "unknown" else 0
    total += 2 if derived["company_size_guess"] in ["2-10", "11-50", "51-200"] else 0
    total += 2 if signals.get("has_case_studies") else 0
    total += 2 if signals.get("has_careers_page") else 0
    total += 2 if signals.get("open_roles_count", 0) > 0 else 0
    if signals.get("tech", {}).get("crm") or signals.get("tech", {}).get("analytics"):
        total += 2
    if signals.get("contact", {}).get("emails"):
        total += 2
    total += 2 if signals.get("pricing") else 0
    raw = max(0, min(total, 48))
    score = round((raw / 48) * 100)
    if score >= 75:
        band = "Ideal ICP"
    elif score >= 50:
        band = "Good fit"
    else:
        band = "Poor fit"
    shortlist = []
    if derived["company_size_guess"] == "solo" and derived["b2x"] == "b2b":
        shortlist.append("solo_services")
    if signals.get("tech", {}).get("ecommerce"):
        shortlist.append("smb_ecom")
    if derived["b2x"] == "b2b":
        shortlist.append("b2b_agency")
    if not shortlist:
        shortlist.append("other")
    return {"rule_score": score, "rule_band": band, "shortlist": shortlist}


async def crawl_site(url: str, max_pages: int = MAX_PAGES) -> Dict[str, Any]:
    base = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
    async with httpx.AsyncClient() as client:
        if not await ROBOTS.allowed(client, url):
            raise RuntimeError("Blocked by robots.txt")
        _, html = await _fetch(client, url)
        signals = _extract_signals(html, base)
        links = _discover_links(html, base)[: max_pages - 1]
        allowed = []
        for u in links:
            try:
                if await ROBOTS.allowed(client, u):
                    allowed.append(u)
            except Exception:
                continue
        # Respect basic rate limiting between internal page fetches as a floor
        pages: List[Tuple[str, str]] = []
        for u in allowed:
            try:
                # keep a small intra-site delay even if global limiter is disabled
                await asyncio.sleep(0.2)
                pages.append(await _fetch(client, u))
            except Exception:
                continue
        for res in pages:
            _, h = res
            s2 = _extract_signals(h, base)
            for k in ["value_props", "products_services", "pricing", "languages"]:
                signals[k] = sorted(set((signals.get(k) or []) + (s2.get(k) or [])))[
                    :40
                ]
            c = signals.get("contact", {"emails": [], "phones": []})
            c2 = s2.get("contact", {"emails": [], "phones": []})
            c["emails"] = sorted(set(c.get("emails", []) + c2.get("emails", [])))[:40]
            c["phones"] = sorted(set(c.get("phones", []) + c2.get("phones", [])))[:40]
            signals["contact"] = c
            for bucket, arr in (s2.get("tech") or {}).items():
                signals["tech"].setdefault(bucket, [])
                signals["tech"][bucket] = sorted(set(signals["tech"][bucket] + arr))[
                    :40
                ]
            signals["has_case_studies"] |= s2["has_case_studies"]
            signals["has_testimonials"] |= s2["has_testimonials"]
            signals["has_careers_page"] |= s2["has_careers_page"]
            signals["open_roles_count"] = max(
                signals["open_roles_count"], s2["open_roles_count"]
            )

        derived = _derive_features(signals)
        scored = _rule_score(signals, derived)

        return {
            "url": url,
            "title": signals.get("title") or "",
            "description": signals.get("meta_description") or "",
            "content_summary": " | ".join(signals.get("value_props", [])[:6]),
            "key_pages": allowed,
            "signals": signals,
            **scored,
            "crawl_metadata": {
                "fetched_pages": 1 + len(allowed),
                "ts": int(time.time()),
            },
            "pages": [{"url": u, "html": h} for u, h in pages],
        }
