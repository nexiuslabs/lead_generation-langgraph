import os
import re
from typing import List, Optional
from urllib.parse import urlparse, urljoin, parse_qs, unquote, quote

import requests
from bs4 import BeautifulSoup


def _apex(host: str) -> str:
    try:
        h = (host or "").strip().lower()
        if not h:
            return ""
        parts = h.split(".")
        if len(parts) <= 2:
            return h
        # naive public suffix handling (good enough for .com.sg/.co.uk)
        last2 = ".".join(parts[-2:])
        last3 = ".".join(parts[-3:])
        multi = {"com.sg", "co.uk", "com.au", "com.my", "com.hk"}
        if last2 in multi and len(parts) >= 3:
            return ".".join(parts[-3:])
        if last3 in multi and len(parts) >= 4:
            return ".".join(parts[-4:])
        return last2
    except Exception:
        return (host or "").strip().lower()


def _extract_domains_from_html(raw_html: str) -> List[str]:
    text = raw_html or ""
    out: List[str] = []
    try:
        soup = BeautifulSoup(text, "html.parser")
        for a in soup.find_all("a"):
            href = (a.get("href") or "").strip()
            if not href:
                continue
            try:
                href_abs = urljoin("https://duckduckgo.com", href)
                u = urlparse(href_abs)
                host = (u.netloc or "").lower()
                if not host:
                    continue
                # DDG redirect pattern
                if (host.endswith("duckduckgo.com") or host.endswith("r.duckduckgo.com")) and u.path.startswith("/l/"):
                    q = parse_qs(u.query)
                    target = q.get("uddg", [None])[0]
                    if target:
                        target_url = unquote(str(target))
                        host = (urlparse(target_url).netloc or "").lower()
                        if not host:
                            continue
                        out.append(host)
                        continue
                out.append(host)
            except Exception:
                continue
    except Exception:
        # Regex fallback
        for m in re.findall(r"/l/\?[^\s\"']*uddg=([^&\"']+)", text):
            try:
                target_url = unquote(m)
                host = (urlparse(target_url).netloc or "").lower()
                if host:
                    out.append(host)
            except Exception:
                continue
        for href in re.findall(r'href=[\"\']([^\"\']+)[\"\']', text):
            try:
                href_abs = urljoin("https://duckduckgo.com", href.strip())
                u = urlparse(href_abs)
                host = (u.netloc or "").lower()
                if host:
                    out.append(host)
            except Exception:
                continue
    # Filter obvious non-targets
    bad = ("duckduckgo.", "google.", "bing.", "brave.", "yahoo.", "yandex.", "wikipedia.", "wikimedia.")
    uniq: List[str] = []
    seen: set[str] = set()
    for h in out:
        if any(b in h for b in bad):
            continue
        a = _apex(h)
        if not a:
            continue
        if a in seen:
            continue
        seen.add(a)
        uniq.append(a)
    return uniq


def search_domains(query: str, max_results: int = 20, country: Optional[str] = None) -> List[str]:
    """Fetch DuckDuckGo results (via r.jina) and extract apex domains.

    - country: hint to set kl (e.g., 'sg' → 'sg-en')
    """
    if not query or not isinstance(query, str):
        return []
    kl = None
    try:
        if country:
            c = country.lower()
            if c in {"sg", "singapore"}:
                kl = "sg-en"
            elif c in {"us", "usa", "united states"}:
                kl = "us-en"
            elif c in {"uk", "gb", "united kingdom"}:
                kl = "uk-en"
    except Exception:
        kl = None
    # Try DDG HTML via r.jina
    session = requests.Session()
    urls = [
        f"https://r.jina.ai/https://duckduckgo.com/html/?q={quote(query)}",
        f"https://r.jina.ai/https://html.duckduckgo.com/html/?q={quote(query)}",
        f"https://r.jina.ai/https://duckduckgo.com/lite/?q={quote(query)}",
    ]
    if kl:
        urls = [u + f"&kl={quote(kl)}" for u in urls]
    collected: List[str] = []
    for u in urls:
        try:
            r = session.get(u, timeout=6)
            if r.status_code < 400 and (r.text or "").strip():
                hosts = _extract_domains_from_html(r.text)
                for h in hosts:
                    if h not in collected:
                        collected.append(h)
                    if len(collected) >= max_results:
                        break
        except Exception:
            continue
        if len(collected) >= max_results:
            break
    return collected[:max_results]

