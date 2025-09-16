# lusha_client.py
# Minimal Lusha API wrapper used as a fallback data provider.
# Docs: https://docs.lusha.com/apis/openapi.md

from __future__ import annotations
import os
import time
import logging
from typing import Any, Dict, List, Optional
import requests
import asyncio
import httpx

logger = logging.getLogger(__name__)

class LushaError(RuntimeError):
    pass

def _sanitize_company_text(text: Optional[str]) -> Optional[str]:
    """Sanitize company name text to Lusha-allowed characters.

    Lusha error message: "Text can only contain alphanumeric characters, spaces,
    and basic punctuation (.,&-_())".
    We strip any other characters (e.g., '@', '/', '\\', quotes).
    """
    if text is None:
        return None
    try:
        allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,&-_()")
        cleaned = "".join(ch if ch in allowed else " " for ch in str(text))
        # Collapse repeated whitespace and trim
        cleaned = " ".join(cleaned.split()).strip()
        return cleaned or None
    except Exception:
        return text

class AsyncLushaClient:
    BASE_URL = os.getenv("LUSHA_BASE_URL", "https://api.lusha.com")

    def __init__(self, api_key: Optional[str] = None, client: Optional[httpx.AsyncClient] = None, timeout: float = 30):
        self.api_key = api_key or os.getenv("LUSHA_API_KEY")
        if not self.api_key:
            raise LushaError("Missing LUSHA_API_KEY. Add it to your environment or settings.py.")
        self.timeout = timeout
        self._client = client or httpx.AsyncClient()
        self._own_client = client is None
        # Stores the last requestId returned by contact search, used by enrich
        self._last_request_id: Optional[str] = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.aclose()

    async def aclose(self):
        if getattr(self, "_own_client", False) and self._client:
            await self._client.aclose()

    # --- Low-level HTTP helpers -------------------------------------------------
    def _headers(self) -> Dict[str, str]:
        # Lusha uses "api_key" header (not Bearer).
        # Ref: docs.lusha.com → Security → ApiKeyAuth
        return {"api_key": self.api_key, "accept": "application/json", "content-type": "application/json"}

    async def _request(self, method: str, path: str, *, params: Optional[Dict[str, Any]] = None, json: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.BASE_URL}{path}"
        # one retry on 429
        for attempt in range(2):
            resp = await self._client.request(method, url, headers=self._headers(), params=params, json=json, timeout=self.timeout)
            if resp.status_code == 429 and attempt == 0:
                retry_after = int(resp.headers.get("Retry-After", "2"))
                logger.warning("Lusha rate-limit hit. Retrying in %ss…", retry_after)
                await asyncio.sleep(retry_after)
                continue
            return self._handle(resp)
        return self._handle(resp)

    def _handle(self, resp: httpx.Response) -> Dict[str, Any]:
        if 200 <= resp.status_code < 300:
            try:
                return resp.json()
            except Exception as e:
                raise LushaError(f"Lusha non-JSON response: {e} body={resp.text[:500]!r}")
        try:
            payload = resp.json()
        except Exception:
            payload = {"message": resp.text}
        raise LushaError(f"Lusha API error {resp.status_code}: {payload}")

    # --- Company ----------------------------------------------------------------
    async def get_company(self, *, domain: Optional[str] = None, name: Optional[str] = None, company_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        params: Dict[str, Any] = {}
        if domain:
            params["domain"] = domain
        if name:
            # API expects 'company' (not 'companyName')
            params["company"] = _sanitize_company_text(name) or name
        if company_id:
            params["companyId"] = company_id
        data = await self._request("GET", "/v2/company", params=params)
        if not data:
            return None
        return data.get("company", data)

    async def prospect_companies(self, *, name: Optional[str] = None, country: Optional[str] = None, domain: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
        """
        POST /prospecting/company/search — search by name/location/etc., returns 'results'.
        """
        # Some accounts reject pagination parameters; send only filters.
        body: Dict[str, Any] = {}
        filters: Dict[str, Any] = {}
        if name:
            s = _sanitize_company_text(name)
            if s:
                filters["companyNames"] = [s]
        if country:
            filters["locations"] = [{"country": country}]
        if domain:
            filters["companyDomains"] = [domain]
        if filters:
            body["filters"] = filters
        data = await self._request("POST", "/prospecting/company/search", json=body)
        results = data.get("results") or data.get("companies") or []
        return results[: max(0, limit)] if isinstance(results, list) else []

    # --- Contacts ---------------------------------------------------------------
    async def prospect_contacts(self, *, company_domain: Optional[str] = None, company_name: Optional[str] = None, country: Optional[str] = None, titles: Optional[List[str]] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        POST /prospecting/contact/search — user-provided schema with pages + nested filters
        """
        # Lusha requires pages.size >= 10
        size = max(10, min(50, int(limit or 10)))
        body: Dict[str, Any] = {
            "pages": {"page": 0, "size": size},
            "filters": {
                "contacts": {"include": {}, "exclude": {}},
                "companies": {"include": {}, "exclude": {}},
            },
        }
        # Contacts filters
        contacts_inc = body["filters"]["contacts"]["include"]
        if country:
            contacts_inc["locations"] = [{"country": country}]
        if titles:
            contacts_inc["titles"] = titles
        # Companies filters — include company names variations
        companies_inc = body["filters"]["companies"]["include"]
        name_variants: List[str] = []
        if company_name:
            name_variants.append(company_name)
            # Add a normalized variant without common suffixes
            ul = company_name.upper()
            for sfx in [" PTE LTD", " PTE. LTD.", " PRIVATE LIMITED", " LTD", " LIMITED"]:
                if ul.endswith(sfx):
                    name_variants.append(company_name[: -len(sfx)].strip())
                    break
            # Add dotted/non-dotted Pte Ltd variant
            if "Pte Ltd" in company_name:
                name_variants.append(company_name.replace("Pte Ltd", "Pte. Ltd"))
            if "Pte. Ltd" in company_name:
                name_variants.append(company_name.replace("Pte. Ltd", "Pte Ltd"))
        if name_variants:
            # dedupe while preserving order
            seen: set[str] = set()
            nv: List[str] = []
            for n in name_variants:
                sn = _sanitize_company_text(n)
                if sn and sn not in seen:
                    seen.add(n)
                    nv.append(sn)
            if nv:
                companies_inc["names"] = nv
        # Execute
        data = await self._request("POST", "/prospecting/contact/search", json=body)
        # Persist requestId for enrich
        try:
            self._last_request_id = data.get("requestId")  # type: ignore[union-attr]
        except Exception:
            self._last_request_id = None
        results = []
        if isinstance(data, dict):
            results = data.get("data") or data.get("results") or data.get("contacts") or []  # type: ignore[assignment]
        res_list = results[: size] if isinstance(results, list) else []
        return res_list

    async def enrich_contacts(self, *, contact_ids: Optional[List[str]] = None, contacts: Optional[List[Dict[str, Any]]] = None, request_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        POST /prospecting/contact/enrich — enrich contacts returned from search.
        Accepts contactIds and optional requestId (recommended by user-provided schema).
        """
        body: Dict[str, Any] = {}
        ids: List[str] = []
        if contact_ids:
            ids = [i for i in contact_ids if i]
        elif contacts:
            for c in contacts:
                cid = c.get("contactId") or c.get("id") or c.get("lushaContactId")
                if cid:
                    ids.append(cid)
        else:
            raise ValueError("Provide contact_ids or contacts")
        if not ids:
            return []
        body["contactIds"] = ids
        rid = request_id or self._last_request_id
        if rid:
            body["requestId"] = rid
        data = await self._request("POST", "/prospecting/contact/enrich", json=body)
        # Flatten each contact's data payload for downstream consumers
        raw_list = []
        if isinstance(data, dict):
            raw_list = data.get("contacts") or data.get("results") or []  # type: ignore[assignment]
        out: List[Dict[str, Any]] = []
        for item in raw_list or []:
            if isinstance(item, dict) and item.get("data"):
                out.append(item["data"])  # flatten
            elif isinstance(item, dict):
                out.append(item)
        return out

    async def get_person(self, *, email: Optional[str] = None, linkedin_url: Optional[str] = None, first_name: Optional[str] = None, last_name: Optional[str] = None, company_name: Optional[str] = None, company_domain: Optional[str] = None, reveal_phones: bool = False, reveal_emails: bool = False) -> Optional[Dict[str, Any]]:
        """
        GET /v2/person — single person enrich lookup.
        Requires: email OR linkedinUrl OR (firstName & lastName & (companyName OR companyDomain)).
        """
        params: Dict[str, Any] = {}
        if email: params["email"] = email
        if linkedin_url: params["linkedinUrl"] = linkedin_url
        if first_name: params["firstName"] = first_name
        if last_name: params["lastName"] = last_name
        if company_name: params["company"] = company_name
        if company_domain: params["companyDomain"] = company_domain
        if reveal_phones: params["revealPhones"] = "true"
        if reveal_emails: params["revealEmails"] = "true"
        data = await self._request("GET", "/v2/person", params=params)
        return data or None

    # --- Convenience ------------------------------------------------------------
    async def find_company_domain(self, company_name: str, country: Optional[str] = None) -> Optional[str]:
        """
        Use user-provided endpoint:
        POST /prospecting/filters/companies/names with {"text": company_name}
        Fallbacks:
        - try dotted/undotted Pte Ltd variants
        - try normalized name without common suffixes
        - finally, GET /v2/company by name
        """
        async def _from_filters(text: str) -> Optional[str]:
            try:
                resp = await self._request("POST", "/prospecting/filters/companies/names", json={"text": text})
                if isinstance(resp, list) and resp:
                    first = resp[0] or {}
                    return first.get("domains_homepage") or first.get("fqdn") or first.get("companyDomain") or first.get("website")
            except Exception as e:
                logger.info("Lusha companies/names failed for %r: %s", text, e)
            return None

        # Try sanitized original
        sanitized = _sanitize_company_text(company_name) or company_name
        d = await _from_filters(sanitized)
        if d:
            return d
        # Try dotted/undotted variants
        if "Pte Ltd" in company_name:
            v = _sanitize_company_text(company_name.replace("Pte Ltd", "Pte. Ltd")) or company_name.replace("Pte Ltd", "Pte. Ltd")
            d = await _from_filters(v)
            if d:
                return d
        if "Pte. Ltd" in company_name:
            v = _sanitize_company_text(company_name.replace("Pte. Ltd", "Pte Ltd")) or company_name.replace("Pte. Ltd", "Pte Ltd")
            d = await _from_filters(v)
            if d:
                return d
        # Try normalized (strip common suffixes)
        ul = company_name.upper()
        norm = ul
        for sfx in [" PTE LTD", " PTE. LTD.", " PRIVATE LIMITED", " LTD", " LIMITED"]:
            if ul.endswith(sfx):
                norm = ul[: -len(sfx)].strip()
                break
        norm_t = (_sanitize_company_text(norm.title()) or norm.title())
        if norm_t and norm_t != company_name:
            d = await _from_filters(norm_t)
            if d:
                return d
        # Last resort: GET /v2/company
        try:
            c = await self.get_company(name=company_name)
            domain = (c or {}).get("domain") or (c or {}).get("website")
            if domain:
                return domain
        except Exception as e:
            logger.info("Lusha get_company fallback failed: %s", e)
        return None

    async def search_and_enrich_contacts(self, *, company_name: Optional[str], company_domain: Optional[str], country: Optional[str], titles: Optional[List[str]] = None, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Utility to search then enrich. Returns contacts with emails/phones when available.
        """
        contacts = await self.prospect_contacts(company_domain=company_domain, company_name=company_name, country=country, titles=titles, limit=limit)
        if not contacts:
            try:
                logger.info(
                    "Lusha search_and_enrich_contacts: prospect returned 0 contacts for company=%r domain=%r country=%r titles=%s",
                    company_name,
                    company_domain,
                    country,
                    titles or [],
                )
            except Exception:
                pass
            return []
        ids = [
            (c.get("contactId") or c.get("id") or c.get("lushaContactId"))
            for c in contacts
            if c.get("contactId") or c.get("id") or c.get("lushaContactId")
        ]
        try:
            enriched = await (self.enrich_contacts(contact_ids=ids, request_id=self._last_request_id) if ids else self.enrich_contacts(contacts=contacts, request_id=self._last_request_id))
        except Exception as e:
            # Fallback to single enrich calls if bulk enrich isn't available on your plan
            logger.warning("Lusha enrich_contacts failed, falling back to GET /v2/person per contact: %s", e)
            enriched = []
            for c in contacts:
                fn = c.get("firstName")
                ln = c.get("lastName")
                if (not fn or not ln) and c.get("name") and isinstance(c.get("name"), str):
                    parts = c["name"].split()
                    if len(parts) >= 2:
                        fn, ln = parts[0], " ".join(parts[1:])
                person = await self.get_person(first_name=fn, last_name=ln,
                                         company_name=c.get("companyName") or company_name,
                                         company_domain=company_domain or c.get("companyDomain"),
                                         reveal_emails=True, reveal_phones=True)
                if person:
                    enriched.append(person)
        return enriched

class LushaClient:
    BASE_URL = os.getenv("LUSHA_BASE_URL", "https://api.lusha.com")

    def __init__(self, api_key: Optional[str] = None, session: Optional[requests.Session] = None, timeout: int = 30):
        self.api_key = api_key or os.getenv("LUSHA_API_KEY")
        if not self.api_key:
            raise LushaError("Missing LUSHA_API_KEY. Add it to your environment or settings.py.")
        self.session = session or requests.Session()
        self.timeout = timeout
        # Stores the last requestId returned by contact search, used by enrich
        self._last_request_id: Optional[str] = None

    # --- Low-level HTTP helpers -------------------------------------------------
    def _headers(self) -> Dict[str, str]:
        # Lusha uses "api_key" header (not Bearer).
        # Ref: docs.lusha.com → Security → ApiKeyAuth
        return {"api_key": self.api_key, "accept": "application/json", "content-type": "application/json"}

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.BASE_URL}{path}"
        resp = self.session.get(url, headers=self._headers(), params=params or {}, timeout=self.timeout)
        return self._handle(resp)

    def _post(self, path: str, json: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.BASE_URL}{path}"
        resp = self.session.post(url, headers=self._headers(), json=json or {}, timeout=self.timeout)
        return self._handle(resp)

    def _handle(self, resp: requests.Response) -> Dict[str, Any]:
        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", "2"))
            logger.warning("Lusha rate-limit hit. Retrying in %ss…", retry_after)
            time.sleep(retry_after)
            # After sleeping we just raise with the original body (keeps code simple)
        if 200 <= resp.status_code < 300:
            try:
                return resp.json()
            except Exception as e:
                raise LushaError(f"Lusha non-JSON response: {e} body={resp.text[:500]!r}")
        try:
            payload = resp.json()
        except Exception:
            payload = {"message": resp.text}
        raise LushaError(f"Lusha API error {resp.status_code}: {payload}")

    # --- Company ----------------------------------------------------------------
    def get_company(self, *, domain: Optional[str] = None, name: Optional[str] = None, company_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        GET /v2/company — retrieve a single company by domain, name, or company_id.
        At least one of domain/name/company_id is required. (See docs.)
        """
        params: Dict[str, Any] = {}
        if domain:
            params["domain"] = domain
        if name:
            # API expects 'company' (not 'companyName')
            params["company"] = _sanitize_company_text(name) or name
        if company_id:
            params["companyId"] = company_id
        data = self._get("/v2/company", params=params)
        if not data:
            return None
        return data.get("company", data)

    def prospect_companies(self, *, name: Optional[str] = None, country: Optional[str] = None, domain: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
        """
        POST /prospecting/company/search — search by name/location/etc., returns 'results'.
        """
        # Some accounts reject pagination parameters; send only filters.
        body: Dict[str, Any] = {}
        filters: Dict[str, Any] = {}
        if name:
            s = _sanitize_company_text(name)
            if s:
                filters["companyNames"] = [s]
        if country:
            filters["locations"] = [{"country": country}]
        if domain:
            filters["companyDomains"] = [domain]
        if filters:
            body["filters"] = filters
        data = self._post("/prospecting/company/search", json=body)
        results = data.get("results") or data.get("companies") or []
        return results[: max(0, limit)] if isinstance(results, list) else []

    # --- Contacts ---------------------------------------------------------------
    def prospect_contacts(self, *, company_domain: Optional[str] = None, company_name: Optional[str] = None, country: Optional[str] = None, titles: Optional[List[str]] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        POST /prospecting/contact/search — user-provided schema with pages + nested filters
        """
        # Lusha requires pages.size >= 10
        size = max(10, min(50, int(limit or 10)))
        body: Dict[str, Any] = {
            "pages": {"page": 0, "size": size},
            "filters": {
                "contacts": {"include": {}, "exclude": {}},
                "companies": {"include": {}, "exclude": {}},
            },
        }
        # Contacts filters
        contacts_inc = body["filters"]["contacts"]["include"]
        if country:
            contacts_inc["locations"] = [{"country": country}]
        if titles:
            contacts_inc["titles"] = titles
        # Companies filters — include company names variations
        companies_inc = body["filters"]["companies"]["include"]
        name_variants: List[str] = []
        if company_name:
            name_variants.append(company_name)
            # Add a normalized variant without common suffixes
            ul = company_name.upper()
            for sfx in [" PTE LTD", " PTE. LTD.", " PRIVATE LIMITED", " LTD", " LIMITED"]:
                if ul.endswith(sfx):
                    name_variants.append(company_name[: -len(sfx)].strip())
                    break
            # Add dotted/non-dotted Pte Ltd variant
            if "Pte Ltd" in company_name:
                name_variants.append(company_name.replace("Pte Ltd", "Pte. Ltd"))
            if "Pte. Ltd" in company_name:
                name_variants.append(company_name.replace("Pte. Ltd", "Pte Ltd"))
        if name_variants:
            # dedupe while preserving order
            seen: set[str] = set()
            nv: List[str] = []
            for n in name_variants:
                sn = _sanitize_company_text(n)
                if sn and sn not in seen:
                    seen.add(n)
                    nv.append(sn)
            if nv:
                companies_inc["names"] = nv
        # Execute
        try:
            logger.info(
                "Lusha prospect_contacts: company_name=%r, company_domain=%r, country=%r, titles=%s, size=%d",
                company_name,
                company_domain,
                country,
                titles or [],
                size,
            )
        except Exception:
            pass
        data = self._post("/prospecting/contact/search", json=body)
        # Persist requestId for enrich
        try:
            self._last_request_id = data.get("requestId")  # type: ignore[union-attr]
        except Exception:
            self._last_request_id = None
        results = []
        if isinstance(data, dict):
            results = data.get("data") or data.get("results") or data.get("contacts") or []  # type: ignore[assignment]
        res_list = results[: size] if isinstance(results, list) else []
        try:
            sample_ids = []
            for r in res_list[:5]:
                if isinstance(r, dict):
                    sample_ids.append(r.get("contactId") or r.get("id") or r.get("lushaContactId"))
            logger.info(
                "Lusha prospect_contacts: requestId=%s results=%d sample_ids=%s",
                self._last_request_id,
                len(res_list),
                sample_ids,
            )
        except Exception:
            pass
        return res_list

    def enrich_contacts(self, *, contact_ids: Optional[List[str]] = None, contacts: Optional[List[Dict[str, Any]]] = None, request_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        POST /prospecting/contact/enrich — enrich contacts returned from search.
        Accepts contactIds and optional requestId (recommended by user-provided schema).
        """
        body: Dict[str, Any] = {}
        ids: List[str] = []
        if contact_ids:
            ids = [i for i in contact_ids if i]
        elif contacts:
            for c in contacts:
                cid = c.get("contactId") or c.get("id") or c.get("lushaContactId")
                if cid:
                    ids.append(cid)
        else:
            raise ValueError("Provide contact_ids or contacts")
        if not ids:
            return []
        body["contactIds"] = ids
        rid = request_id or self._last_request_id
        if rid:
            body["requestId"] = rid
        try:
            logger.info(
                "Lusha enrich_contacts: requestId=%s ids_count=%d sample_ids=%s",
                rid,
                len(ids),
                ids[:5],
            )
        except Exception:
            pass
        data = self._post("/prospecting/contact/enrich", json=body)
        # Flatten each contact's data payload for downstream consumers
        raw_list = []
        if isinstance(data, dict):
            raw_list = data.get("contacts") or data.get("results") or []  # type: ignore[assignment]
        out: List[Dict[str, Any]] = []
        for item in raw_list or []:
            if isinstance(item, dict) and item.get("data"):
                out.append(item["data"])  # flatten
            elif isinstance(item, dict):
                out.append(item)
        # Post-enrich logging: counts and samples
        try:
            def _count_vals(v: Any) -> int:
                if isinstance(v, list):
                    return len(v)
                return 1 if v else 0

            emails_total = 0
            phones_total = 0
            for c in out:
                if not isinstance(c, dict):
                    continue
                ev = c.get("emails") or c.get("emailAddresses") or c.get("email_addresses")
                pv = c.get("phones") or c.get("phoneNumbers") or c.get("phone_numbers")
                emails_total += _count_vals(ev)
                phones_total += _count_vals(pv)
            sample_names = []
            for c in out[:3]:
                if isinstance(c, dict):
                    nm = c.get("fullName") or c.get("name") or " ".join(filter(None, [c.get("firstName"), c.get("lastName")]))
                    sample_names.append(nm)
            logger.info(
                "Lusha enrich_contacts: returned=%d emails~%d phones~%d sample_names=%s",
                len(out),
                emails_total,
                phones_total,
                sample_names,
            )
        except Exception:
            pass
        return out

    def get_person(self, *, email: Optional[str] = None, linkedin_url: Optional[str] = None, first_name: Optional[str] = None, last_name: Optional[str] = None, company_name: Optional[str] = None, company_domain: Optional[str] = None, reveal_phones: bool = False, reveal_emails: bool = False) -> Optional[Dict[str, Any]]:
        """
        GET /v2/person — single person enrich lookup.
        Requires: email OR linkedinUrl OR (firstName & lastName & (companyName OR companyDomain)).
        """
        params: Dict[str, Any] = {}
        if email: params["email"] = email
        if linkedin_url: params["linkedinUrl"] = linkedin_url
        if first_name: params["firstName"] = first_name
        if last_name: params["lastName"] = last_name
        if company_name: params["company"] = company_name
        if company_domain: params["companyDomain"] = company_domain
        if reveal_phones: params["revealPhones"] = "true"
        if reveal_emails: params["revealEmails"] = "true"
        data = self._get("/v2/person", params=params)
        return data or None

    # --- Convenience ------------------------------------------------------------
    def find_company_domain(self, company_name: str, country: Optional[str] = None) -> Optional[str]:
        """
        Use user-provided endpoint:
        POST /prospecting/filters/companies/names with {"text": company_name}
        Fallbacks:
        - try dotted/undotted Pte Ltd variants
        - try normalized name without common suffixes
        - finally, GET /v2/company by name
        """
        def _from_filters(text: str) -> Optional[str]:
            try:
                resp = self._post("/prospecting/filters/companies/names", json={"text": text})
                if isinstance(resp, list) and resp:
                    first = resp[0] or {}
                    return first.get("domains_homepage") or first.get("fqdn") or first.get("companyDomain") or first.get("website")
            except Exception as e:
                logger.info("Lusha companies/names failed for %r: %s", text, e)
            return None

        # Try sanitized original
        sanitized = _sanitize_company_text(company_name) or company_name
        d = _from_filters(sanitized)
        if d:
            return d
        # Try dotted/undotted variants
        if "Pte Ltd" in company_name:
            v = _sanitize_company_text(company_name.replace("Pte Ltd", "Pte. Ltd")) or company_name.replace("Pte Ltd", "Pte. Ltd")
            d = _from_filters(v)
            if d:
                return d
        if "Pte. Ltd" in company_name:
            v = _sanitize_company_text(company_name.replace("Pte. Ltd", "Pte Ltd")) or company_name.replace("Pte. Ltd", "Pte Ltd")
            d = _from_filters(v)
            if d:
                return d
        # Try normalized (strip common suffixes)
        ul = company_name.upper()
        norm = ul
        for sfx in [" PTE LTD", " PTE. LTD.", " PRIVATE LIMITED", " LTD", " LIMITED"]:
            if ul.endswith(sfx):
                norm = ul[: -len(sfx)].strip()
                break
        norm_t = (_sanitize_company_text(norm.title()) or norm.title())
        if norm_t and norm_t != company_name:
            d = _from_filters(norm_t)
            if d:
                return d
        # Last resort: GET /v2/company
        try:
            c = self.get_company(name=company_name)
            domain = (c or {}).get("domain") or (c or {}).get("website")
            if domain:
                return domain
        except Exception as e:
            logger.info("Lusha get_company fallback failed: %s", e)
        return None

    def search_and_enrich_contacts(self, *, company_name: Optional[str], company_domain: Optional[str], country: Optional[str], titles: Optional[List[str]] = None, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Utility to search then enrich. Returns contacts with emails/phones when available.
        """
        contacts = self.prospect_contacts(company_domain=company_domain, company_name=company_name, country=country, titles=titles, limit=limit)
        if not contacts:
            try:
                logger.info(
                    "Lusha search_and_enrich_contacts: prospect returned 0 contacts for company=%r domain=%r country=%r titles=%s",
                    company_name,
                    company_domain,
                    country,
                    titles or [],
                )
            except Exception:
                pass
            return []
        ids = [
            (c.get("contactId") or c.get("id") or c.get("lushaContactId"))
            for c in contacts
            if c.get("contactId") or c.get("id") or c.get("lushaContactId")
        ]
        try:
            enriched = self.enrich_contacts(contact_ids=ids, request_id=self._last_request_id) if ids else self.enrich_contacts(contacts=contacts, request_id=self._last_request_id)
        except Exception as e:
            # Fallback to single enrich calls if bulk enrich isn't available on your plan
            logger.warning("Lusha enrich_contacts failed, falling back to GET /v2/person per contact: %s", e)
            enriched = []
            for c in contacts:
                fn = c.get("firstName")
                ln = c.get("lastName")
                if (not fn or not ln) and c.get("name") and isinstance(c.get("name"), str):
                    parts = c["name"].split()
                    if len(parts) >= 2:
                        fn, ln = parts[0], " ".join(parts[1:])
                person = self.get_person(first_name=fn, last_name=ln,
                                         company_name=c.get("companyName") or company_name,
                                         company_domain=company_domain or c.get("companyDomain"),
                                         reveal_emails=True, reveal_phones=True)
                if person:
                    enriched.append(person)
        return enriched
