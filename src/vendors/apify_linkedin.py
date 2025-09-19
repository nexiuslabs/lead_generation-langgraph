from __future__ import annotations
import os
from typing import List, Dict, Any, Optional
import httpx
import logging
from urllib.parse import quote
import json as _json

APIFY_BASE = "https://api.apify.com/v2"
logger = logging.getLogger(__name__)


def _token() -> str:
    t = os.getenv("APIFY_TOKEN")
    if not t:
        raise RuntimeError("Missing APIFY_TOKEN")
    return t


def _actor_id() -> str:
    return os.getenv("APIFY_LINKEDIN_ACTOR_ID", "dev_fusion~linkedin-profile-scraper")

def _search_actor_id() -> Optional[str]:
    v = os.getenv("APIFY_SEARCH_ACTOR_ID")
    return v if v else None

def _company_actor_id() -> Optional[str]:
    v = os.getenv("APIFY_COMPANY_ACTOR_ID", "harvestapi~linkedin-company")
    return v or None

def _employees_actor_id() -> Optional[str]:
    v = os.getenv("APIFY_EMPLOYEES_ACTOR_ID", "harvestapi~linkedin-company-employees")
    return v or None


async def run_sync_get_dataset_items(
    payload: Dict[str, Any], *, dataset_format: str = "json", timeout_s: int = 600
) -> List[Dict[str, Any]]:
    """Run an Apify Actor with run-sync-get-dataset-items and return dataset items.

    Notes:
    - Actor input schemas vary. We first try the given payload verbatim.
    - On 400 responses, we try a few common variants (query/searchTerms/keywords).
    - Returns a list of items (possibly empty) and never raises on parse errors.
    """
    url = f"{APIFY_BASE}/acts/{_actor_id().replace('/', '~')}/run-sync-get-dataset-items"
    params = {"token": _token(), "format": dataset_format}
    headers = {"Content-Type": "application/json"}

    def _log_items_sample(items: List[Dict[str, Any]], label: str) -> None:
        try:
            dbg = os.getenv("APIFY_DEBUG_LOG_ITEMS", "").lower() in ("1", "true", "yes", "on")
            if not dbg:
                return
            try:
                n = int(os.getenv("APIFY_LOG_SAMPLE_SIZE", "3") or 3)
            except Exception:
                n = 3
            sample: List[Dict[str, Any]] = []
            fields = [
                "fullName",
                "name",
                "headline",
                "title",
                "companyName",
                "company",
                "profileUrl",
                "url",
                "linkedin_url",
                "locationName",
                "location",
                "email",
            ]
            for it in (items or [])[:n]:
                if not isinstance(it, dict):
                    continue
                rec = {k: it.get(k) for k in fields if it.get(k) is not None}
                if not rec:
                    # fallback to first few keys to aid debugging without huge dumps
                    keys = list(it.keys())[:6]
                    rec = {k: it.get(k) for k in keys}
                sample.append(rec)
            logger.info("Apify items sample label=%s n=%d sample=%s", label, len(sample), sample)
        except Exception:
            # Never block the pipeline on logging
            pass

    async def _post(_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            r = await client.post(url, params=params, json=_payload, headers=headers)
            r.raise_for_status()
            try:
                data = r.json()
            except Exception:
                return []
            if isinstance(data, dict) and "items" in data:
                items = data.get("items")
                out = items if isinstance(items, list) else []
                _log_items_sample(out, "run-sync-get-dataset-items")
                return out
            if isinstance(data, list):
                _log_items_sample(data, "run-sync-get-dataset-items[list]")
                return data
            return []

    async def _post_with_actor(actor_id: str, _payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Call a specific actor's run-sync-get-dataset-items with payload."""
        aurl = f"{APIFY_BASE}/acts/{actor_id.replace('/', '~')}/run-sync-get-dataset-items"
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            r = await client.post(aurl, params=params, json=_payload, headers=headers)
            r.raise_for_status()
            try:
                data = r.json()
            except Exception:
                return []
            if isinstance(data, dict) and "items" in data:
                items = data.get("items")
                out = items if isinstance(items, list) else []
                _log_items_sample(out, f"run-sync-get-dataset-items[{actor_id}]")
                return out
            if isinstance(data, list):
                _log_items_sample(data, f"run-sync-get-dataset-items[list][{actor_id}]")
                return data
            return []

    async def _run_sync_then_fetch(_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fallback: run actor via run-sync, then fetch dataset items explicitly.

        This helps when run-sync-get-dataset-items rejects input due to schema
        but actor still runs fine with run-sync, or when actor returns a custom
        structure where dataset items need to be fetched separately.
        """
        run_url = f"{APIFY_BASE}/acts/{_actor_id().replace('/', '~')}/run-sync"
        ds_items_url = f"{APIFY_BASE}/datasets/{{ds_id}}/items"
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            rr = await client.post(run_url, params={"token": _token()}, json=_payload, headers=headers)
            rr.raise_for_status()
            run_data = rr.json()
            # dataset references may appear under defaultDatasetId or datasetId
            ds_id = (
                run_data.get("defaultDatasetId")
                or (run_data.get("data") or {}).get("defaultDatasetId")
                or run_data.get("datasetId")
            )
            if not ds_id:
                # Some actors return items directly
                items = run_data.get("items") or (run_data.get("data") or {}).get("items")
                return items if isinstance(items, list) else []
            ir = await client.get(ds_items_url.format(ds_id=ds_id), params={"token": _token(), "format": dataset_format})
            ir.raise_for_status()
            try:
                data = ir.json()
            except Exception:
                return []
            if isinstance(data, list):
                _log_items_sample(data, "dataset-items[run-sync]")
                return data
            if isinstance(data, dict) and "items" in data:
                out = data.get("items") if isinstance(data.get("items"), list) else []
                _log_items_sample(out, "dataset-items[run-sync][dict]")
                return out
            return []

    # Try original payload first
    try:
        return await _post(payload)
    except httpx.HTTPStatusError as e:
        status = getattr(e.response, "status_code", None)
        body_snippet = None
        try:
            body_snippet = e.response.text[:400]
        except Exception:
            body_snippet = None
        # Try to extract a structured error message if available
        err_msg = None
        try:
            j = e.response.json()
            err_msg = (
                (j.get("error") or {}).get("message")
                or (j.get("message"))
                or None
            )
        except Exception:
            err_msg = None
        logger.warning(
            "Apify actor returned %s; retrying with alt inputs (msg=%r, snippet=%r)",
            status,
            err_msg,
            body_snippet,
        )
        # Build alternative payload variants for common actor schemas
        variants: List[Dict[str, Any]] = []
        # Prepare queries from original payload early (used in template below)
        queries = payload.get("queries")
        # 0) Env-specified template override (APIFY_INPUT_JSON)
        try:
            raw_tpl = os.getenv("APIFY_INPUT_JSON")
            if raw_tpl:
                tpl = _json.loads(raw_tpl)
                # Helper to deep-replace %%QUERY%% tokens in a structure
                def _replace(obj, query_str: str):
                    if isinstance(obj, dict):
                        return {k: _replace(v, query_str) for k, v in obj.items()}
                    if isinstance(obj, list):
                        return [_replace(x, query_str) for x in obj]
                    if isinstance(obj, str):
                        return obj.replace("%%QUERY%%", query_str)
                    return obj
                if isinstance(tpl, dict):
                    # If the template uses %%QUERIES%% placeholder as a string value, replace with list
                    t_clone = _json.loads(raw_tpl)
                    def _replace_queries(obj):
                        if isinstance(obj, dict):
                            return {k: _replace_queries(v) for k, v in obj.items()}
                        if isinstance(obj, list):
                            return [_replace_queries(x) for x in obj]
                        if isinstance(obj, str) and obj == "%%QUERIES%%":
                            return queries
                        return obj
                    t_clone = _replace_queries(t_clone)
                    if isinstance(queries, list) and queries and "%%QUERY%%" in _json.dumps(tpl):
                        # Generate per-query variants by replacing %%QUERY%%
                        for q in queries:
                            variants.append(_replace(tpl, str(q)))
                    else:
                        variants.append(t_clone)
        except Exception:
            pass
        if isinstance(queries, list) and queries:
            first = queries[0]
            base = {"maxResults": 10, "maxItems": 10}
            variants.extend([
                {"query": first, **base},
                {"searchTerms": queries, **base},
                {"search": queries, **base},
                {"search": first, **base},
                {"keywords": queries, **base},
                {"keywords": first, **base},
                {"queries": [{"query": q} for q in queries], **base},
                {"limit": 10, "queries": [{"query": first}]},
                {"searchQueries": queries, **base},
                {"searchQueries": [{"query": q} for q in queries], **base},
                {"searchQueryOrUrl": first, **base},
                # Direct LinkedIn people search URL variant expected by some actors
                {
                    "startUrls": [
                        {"url": f"https://www.linkedin.com/search/results/people/?keywords={quote(str(first))}"}
                    ],
                    **base,
                },
            ])
        elif isinstance(queries, str) and queries:
            base = {"maxResults": 10, "maxItems": 10}
            variants.extend([
                {"query": queries, **base},
                {"search": queries, **base},
                {"keywords": queries, **base},
                {"searchQueryOrUrl": queries, **base},
                {
                    "startUrls": [
                        {"url": f"https://www.linkedin.com/search/results/people/?keywords={quote(str(queries))}"}
                    ],
                    **base,
                },
            ])
        # If no queries key, give up quickly
        for i, v in enumerate(variants, start=1):
            try:
                logger.info("Apify retry variant #%d payload keys=%s", i, list(v.keys()))
                items = await _post(v)
                if not items:
                    # Try run-sync + dataset fetch as a secondary path
                    try:
                        items = await _run_sync_then_fetch(v)
                    except httpx.HTTPStatusError:
                        pass
                return items
            except httpx.HTTPStatusError:
                continue
            except Exception:
                continue
        # If the actor requires profileUrls, attempt to resolve with a search actor (if configured)
        try:
            if isinstance(err_msg, str) and "profileUrls" in err_msg and _search_actor_id():
                # Build a simple search payload and collect LinkedIn profile URLs
                sq = None
                if isinstance(queries, list) and queries:
                    sq = queries[0]
                elif isinstance(queries, str) and queries:
                    sq = queries
                if sq:
                    logger.info("Apify: resolving profileUrls via search actor %s", _search_actor_id())
                    search_variants = [
                        {"query": sq, "maxItems": 10},
                        {"search": sq, "maxItems": 10},
                        {"keywords": sq, "maxItems": 10},
                        {"searchQueryOrUrl": sq, "maxItems": 10},
                        {"queries": [{"query": sq}], "maxItems": 10},
                    ]
                    profile_urls: List[str] = []
                    for sv in search_variants:
                        try:
                            sitems = await _post_with_actor(_search_actor_id(), sv)
                            for it in sitems or []:
                                url = (it.get("url") or it.get("profileUrl") or it.get("linkedin_url") or "")
                                if isinstance(url, str) and "/linkedin.com/in/" in url:
                                    profile_urls.append(url)
                            if profile_urls:
                                break
                        except httpx.HTTPStatusError:
                            continue
                        except Exception:
                            continue
                    profile_urls = list(dict.fromkeys(profile_urls))[:10]
                    if profile_urls:
                        try:
                            logger.info("Apify: retrying profile actor with %d profileUrls", len(profile_urls))
                            return await _post({"profileUrls": profile_urls, "maxItems": 10})
                        except Exception:
                            pass
        except Exception:
            pass
        # Final attempt: run-sync against original payload
        try:
            return await _run_sync_then_fetch(payload)
        except Exception:
            return []
    except Exception:
        # Non-HTTP exception; return empty list
        logger.warning("Apify actor call failed unexpectedly", exc_info=True)
        return []


def build_queries(company_name: str, titles: List[str]) -> List[str]:
    titles = [t for t in (titles or []) if (t or "").strip()]
    if not company_name:
        return []
    company_q = f'"{company_name}"'
    queries: List[str] = []
    # Prefer one query per title to maximize compatibility with actors that
    # expect simple keyword strings rather than boolean logic.
    for t in titles:
        t_q = f'"{t}"' if (" " in t) else t
        queries.append(f"{company_q} {t_q}")
    # Also include a combined boolean query for actors that support it
    if titles:
        t_bool = " OR ".join([f'"{x}"' if " " in x else x for x in titles])
        queries.append(f"{company_q} AND ({t_bool})")
    # And a plain company-only query as a last resort
    queries.append(company_q)
    # De-duplicate while preserving order
    seen = set()
    out: List[str] = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out


def normalize_contacts(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for it in items or []:
        out.append(
            {
                "full_name": it.get("fullName")
                or it.get("name")
                or it.get("full_name"),
                "title": it.get("headline") or it.get("jobTitle") or it.get("title"),
                "company_current": it.get("companyName") or it.get("company_current"),
                "linkedin_url": it.get("url")
                or it.get("profileUrl")
                or it.get("linkedinUrl")
                or it.get("linkedin_url"),
                "location": it.get("locationName") or it.get("addressWithCountry") or it.get("location"),
                # emails rarely present
                "email": it.get("email") or None,
                "source_json": it,
            }
        )
    return out


async def _run_actor_items(actor_id: str, payload: Dict[str, Any], *, dataset_format: str = "json", timeout_s: int = 600) -> List[Dict[str, Any]]:
    """Run a specific Apify actor and return dataset items with logging.

    Uses run-sync-get-dataset-items; falls back to run-sync + dataset fetch.
    """
    def _log_sample(items: List[Dict[str, Any]], label: str) -> None:
        try:
            dbg = os.getenv("APIFY_DEBUG_LOG_ITEMS", "").lower() in ("1", "true", "yes", "on")
            if not dbg:
                return
            sample = []
            keys = ("fullName", "name", "headline", "jobTitle", "profileUrl", "linkedinUrl")
            for it in (items or [])[: int(os.getenv("APIFY_LOG_SAMPLE_SIZE", "3") or 3)]:
                if isinstance(it, dict):
                    rec = {k: it.get(k) for k in keys if it.get(k) is not None}
                    if not rec:
                        for k in list(it.keys())[:6]:
                            rec[k] = it.get(k)
                    sample.append(rec)
            logger.info("Apify items sample label=%s n=%d sample=%s", label, len(sample), sample)
        except Exception:
            pass

    aurl = f"{APIFY_BASE}/acts/{actor_id.replace('/', '~')}/run-sync-get-dataset-items"
    headers = {"Content-Type": "application/json"}
    params = {"token": _token(), "format": dataset_format}
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        try:
            r = await client.post(aurl, params=params, json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list):
                _log_sample(data, f"{actor_id}:run-sync-get-dataset-items[list]")
                return data
            if isinstance(data, dict) and isinstance(data.get("items"), list):
                _log_sample(data.get("items"), f"{actor_id}:run-sync-get-dataset-items")
                return data.get("items")
        except httpx.HTTPStatusError:
            # Fallback: run-sync then fetch dataset items
            run_url = f"{APIFY_BASE}/acts/{actor_id.replace('/', '~')}/run-sync"
            ds_items_url = f"{APIFY_BASE}/datasets/{{ds_id}}/items"
            try:
                rr = await client.post(run_url, params={"token": _token()}, json=payload, headers=headers)
                rr.raise_for_status()
                run_data = rr.json()
                ds_id = (
                    run_data.get("defaultDatasetId")
                    or (run_data.get("data") or {}).get("defaultDatasetId")
                    or run_data.get("datasetId")
                )
                if not ds_id:
                    items = run_data.get("items") or (run_data.get("data") or {}).get("items")
                    out = items if isinstance(items, list) else []
                    _log_sample(out, f"{actor_id}:run-sync[data->items]")
                    return out
                ir = await client.get(ds_items_url.format(ds_id=ds_id), params={"token": _token(), "format": dataset_format})
                ir.raise_for_status()
                data2 = ir.json()
                if isinstance(data2, list):
                    _log_sample(data2, f"{actor_id}:dataset-items[list]")
                    return data2
                if isinstance(data2, dict) and isinstance(data2.get("items"), list):
                    _log_sample(data2.get("items"), f"{actor_id}:dataset-items")
                    return data2.get("items")
            except Exception:
                pass
        except Exception:
            pass
    return []


async def company_to_profile_urls(company_name: str, *, max_items: int = 50, timeout_s: int = 600) -> List[str]:
    """Resolve a company's LinkedIn URL from name, then list employee profile URLs.

    Uses harvestapi~linkedin-company and harvestapi~linkedin-company-employees.
    """
    comp_actor = _company_actor_id()
    emp_actor = _employees_actor_id()
    if not (comp_actor and emp_actor):
        return []
    # 1) Company lookup
    comp_items = await _run_actor_items(comp_actor, {"companies": [company_name]}, timeout_s=timeout_s)
    company_url = None
    if comp_items:
        first = comp_items[0] or {}
        company_url = first.get("linkedinUrl") or (f"https://www.linkedin.com/company/{first.get('universalName')}/" if first.get("universalName") else None)
    logger.info("Apify company actor: company=%s company_url=%s", company_name, company_url)
    if not company_url:
        return []
    # 2) Employees listing
    payload = {"companies": [company_url], "maxItems": max_items}
    mode = os.getenv("APIFY_EMPLOYEES_SCRAPER_MODE")
    if mode:
        payload["profileScraperMode"] = mode
    emp_items = await _run_actor_items(emp_actor, payload, timeout_s=timeout_s)
    urls: List[str] = []
    for it in emp_items or []:
        url = it.get("linkedinUrl") or it.get("profileUrl") or it.get("url")
        if isinstance(url, str) and ".linkedin.com/in/" in url:
            urls.append(url)
    urls = list(dict.fromkeys(urls))
    logger.info("Apify employees actor: company_url=%s employees=%d", company_url, len(urls))
    return urls


def _title_matches(text: str | None, titles: List[str] | None) -> bool:
    if not titles:
        return True
    t = (text or "").lower()
    return any((x or "").strip().lower() in t for x in (titles or []))


async def contacts_via_company_chain(company_name: str, titles: List[str] | None = None, *, max_items: int = 50, timeout_s: int = 600) -> List[Dict[str, Any]]:
    """Chain: company-by-name -> employees -> profile details -> normalize.

    Filters employee profiles by title keywords when provided, then enriches via
    the profile actor configured in APIFY_LINKEDIN_ACTOR_ID.
    """
    profile_actor = _actor_id()
    profile_urls = await company_to_profile_urls(company_name, max_items=max_items, timeout_s=timeout_s)
    if not profile_urls:
        return []
    # If we can fetch employee details, optionally pre-filter by title using employees endpoint again to get headlines
    # For now, we do a quick pre-filter by using the same employees actor payload above that already returned list items.
    # As we only have URLs, we will filter after profile scrape using jobTitle/headline.
    items = await _run_actor_items(profile_actor, {"profileUrls": profile_urls[:max_items], "maxItems": max_items}, timeout_s=timeout_s)
    # Filter by titles if provided
    if titles:
        filtered = []
        for it in items or []:
            txt = (it.get("jobTitle") or it.get("headline") or "")
            if _title_matches(txt, titles):
                filtered.append(it)
        items = filtered
    logger.info("Apify profile actor: requested=%d received=%d filtered=%d", min(len(profile_urls), max_items), len(items or []), len(items or []))
    return normalize_contacts(items or [])
