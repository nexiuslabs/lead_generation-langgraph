# Dev Plan — Feature 11: Error Handling & Fallbacks

Source: featurePRD11.md

Objective: Implement robust retry/backoff, per-tenant per-vendor circuit breakers, a clear fallback chain for enrichment, graceful degradation with provenance, and resilient Odoo export — instrumented with Feature 8 observability.

---

## Architecture Overview

- Centralize retry/backoff and circuit-breaker logic in a small utility so all vendor calls (Tavily, OpenAI, ZeroBounce, Lusha/Apify) behave consistently.
- Express the fallback chain in enrichment as a sequence of guarded steps that short-circuit on success and annotate degradation on misses.
- Persist degradation signals on `enrichment_runs` (run-level) and in per-company projection `company_enrichment_runs` (reason).
- Ensure Odoo export errors never abort the run: default/omit problematic fields and continue.

---

## Configuration (env)

- Retries / Backoff
  - `RETRY_MAX_ATTEMPTS=3`
  - `RETRY_BASE_DELAY_MS=250`
  - `RETRY_MAX_DELAY_MS=4000`
- Circuit Breaker
  - `CB_ERROR_THRESHOLD=3` (consecutive)
  - `CB_COOL_OFF_S=300` (per vendor per tenant)
  - `CB_GLOBAL_EXEMPT_VENDORS=` (CSV; e.g., `openai`)
- Fallback Switches
  - `ENABLE_TAVILY_FALLBACK=true`
  - `ENABLE_APIFY_LINKEDIN=true` (see Feature 16) or `ENABLE_LUSHA_FALLBACK=false`
- Odoo Defaults
  - `ODOO_EXPORT_SET_DEFAULTS=true`

---

## Utilities: Retry + Circuit Breaker

File: `lead_generation-main/src/retry.py`

```python
import asyncio, random, time
from dataclasses import dataclass
from typing import Callable, Awaitable, Type, Sequence, Optional

class RetryableError(Exception):
    pass

@dataclass
class BackoffPolicy:
    max_attempts: int = 3
    base_delay_ms: int = 250
    max_delay_ms: int = 4000

async def with_retry(fn: Callable[[], Awaitable],
                     retry_on: Sequence[Type[BaseException]] = (RetryableError,),
                     policy: BackoffPolicy = BackoffPolicy()):
    attempt = 0
    while True:
        try:
            return await fn()
        except BaseException as e:
            retryable = any(isinstance(e, t) for t in retry_on)
            attempt += 1
            if (not retryable) or attempt >= policy.max_attempts:
                raise
            # exp backoff with jitter
            delay = min(policy.max_delay_ms, policy.base_delay_ms * (2 ** (attempt - 1)))
            delay = delay * (0.8 + 0.4 * random.random())
            await asyncio.sleep(delay / 1000.0)

class CircuitOpen(Exception):
    pass

class CircuitBreaker:
    def __init__(self, error_threshold: int = 3, cool_off_s: int = 300):
        self.error_threshold = error_threshold
        self.cool_off_s = cool_off_s
        # key: (tenant_id, vendor) -> (consec_errors, opened_at_ts)
        self._state: dict[tuple[int,str], tuple[int,float|None]] = {}

    def _key(self, tenant_id: int, vendor: str):
        return (int(tenant_id), vendor)

    def allow(self, tenant_id: int, vendor: str) -> bool:
        k = self._key(tenant_id, vendor)
        errors, opened_at = self._state.get(k, (0, None))
        if opened_at is None:
            return True
        if time.time() - opened_at >= self.cool_off_s:
            self._state[k] = (0, None)
            return True
        return False

    def on_success(self, tenant_id: int, vendor: str):
        self._state[self._key(tenant_id, vendor)] = (0, None)

    def on_error(self, tenant_id: int, vendor: str):
        k = self._key(tenant_id, vendor)
        errors, opened_at = self._state.get(k, (0, None))
        errors += 1
        if errors >= self.error_threshold:
            self._state[k] = (errors, time.time())
        else:
            self._state[k] = (errors, opened_at)
```

Integration point: instantiate a single `CircuitBreaker` per runner process and consult before vendor calls.

---

## Fallback Chain Implementation (Enrichment)

Modify `lead_generation-main/src/enrichment.py` to orchestrate per-company enrichment as follows:

```python
from src.retry import with_retry, BackoffPolicy, RetryableError, CircuitBreaker, CircuitOpen
from src import obs

CB = CircuitBreaker()

async def enrich_company(cid: int, tenant_id: int, run_id: int) -> dict:
    degraded_reasons: list[str] = []

    # 0) Preconditions: ensure we have a target domain or name
    domain, name = _get_company_domain_name(cid)
    if not domain and not name:
        degraded_reasons.append("DATA_EMPTY:no_domain_name")

    # 1) Deterministic crawl
    try:
        async def _crawl():
            return await crawl_site(domain or name)
        res = await with_retry(_crawl)
        if res and res.ok:
            obs.log_event(run_id, tenant_id, "crawl", "finish", "ok", company_id=cid)
        else:
            degraded_reasons.append("CRAWL_THIN")
    except Exception as e:
        obs.log_event(run_id, tenant_id, "crawl", "error", "error", company_id=cid, error_code=type(e).__name__)
        degraded_reasons.append("CRAWL_ERROR")

    # 2) Tavily fallback (if thin)
    if ("CRAWL_THIN" in degraded_reasons or "CRAWL_ERROR" in degraded_reasons) and _env_true("ENABLE_TAVILY_FALLBACK"):
        try:
            async def _tav():
                return await tavily_enrich(domain or name)
            await with_retry(_tav)
            obs.bump_vendor(run_id, tenant_id, "tavily", calls=1)
        except Exception as e:
            obs.bump_vendor(run_id, tenant_id, "tavily", calls=1, errors=1)
            degraded_reasons.append("TAVILY_FAIL")

    # 3) Contact discovery fallback (Apify LinkedIn or Lusha)
    contact_vendor = "apify_linkedin" if _env_true("ENABLE_APIFY_LINKEDIN") else "lusha"
    try:
        if not _has_contacts(cid) and CB.allow(tenant_id, contact_vendor):
            async def _contacts():
                return await fetch_contacts_via_vendor(cid, vendor=contact_vendor)
            await with_retry(_contacts, retry_on=(RetryableError,))
            CB.on_success(tenant_id, contact_vendor)
            obs.bump_vendor(run_id, tenant_id, contact_vendor, calls=1)
    except Exception as e:
        CB.on_error(tenant_id, contact_vendor)
        obs.bump_vendor(run_id, tenant_id, contact_vendor, calls=1, errors=1)
        degraded_reasons.append(f"{contact_vendor.upper()}_FAIL")

    # 4) Email verification (batch elsewhere; here mark pending on error)
    try:
        verify_new_emails_for_company(cid)
    except Exception as e:
        obs.bump_vendor(run_id, tenant_id, "zerobounce", errors=1)
        mark_emails_unknown(cid)
        degraded_reasons.append("ZEROBOUNCE_DEFER")

    # 5) Return per-company status; scoring downstream adjusts contact weight automatically
    return {"company_id": cid, "degraded_reasons": ",".join(degraded_reasons) or None}
```

Persist `degraded_reasons` into `company_enrichment_runs` when projecting, and set a `degraded=true` run flag on `enrichment_runs` if any company in run has a degradation reason.

---

## Odoo Export Resilience

Within `app/odoo_store.py` or the export phase in runner:

```python
def safe_write_partner(models, db_name, uid, pw, model, ids, vals):
    try:
        return models.execute_kw(db_name, uid, pw, model, 'write', [ids, vals])
    except Exception:
        # Strip non-essential fields that can violate constraints (e.g., autopost_bills)
        vals.pop('autopost_bills', None)
        vals.pop('website', None)
        try:
            return models.execute_kw(db_name, uid, pw, model, 'write', [ids, vals])
        except Exception:
            return False
```

If tenant Odoo DB is missing/unreachable: set a per-tenant export-disabled flag for the run and proceed.

---

## Observability Hooks (Feature 8)

- Use `obs.log_event` for every stage error/finish with `error_code` and set `fallback_taken` in `extra` JSON when applicable.
- Use `obs.bump_vendor` for each vendor call (calls/errors/tokens/costs when available).
- On run finalize: call `obs.aggregate_percentiles`.

---

## Testing Strategy

- Unit tests for `with_retry` and `CircuitBreaker` (inject fake retryable errors; assert attempts and cool-off behavior).
- Integration tests with monkeypatched vendor clients (`tavily_enrich`, `fetch_contacts_via_vendor`, `verify_new_emails_for_company`) to throw HTTP 429/5xx/timeouts and verify fallbacks and degradation flags.
- E2E dry-run on a staging tenant with induced failures to ensure run doesn’t abort and QA samples still generate.

---

## Rollout Plan

1) Land `src/retry.py` utilities and wire into vendor wrappers.
2) Implement degradation capture and per-company projection update.
3) Harden Odoo export with safe write and field defaults.
4) Add observability calls around fallback chain.
5) Staged rollout: enable on dev tenant; validate; then enable globally.

