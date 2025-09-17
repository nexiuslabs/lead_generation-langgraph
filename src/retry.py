import asyncio, random, time
from dataclasses import dataclass
from typing import Callable, Awaitable, Type, Sequence

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
    last_exc: BaseException | None = None
    while attempt < policy.max_attempts:
        try:
            return await fn()
        except BaseException as e:
            last_exc = e
            if not any(isinstance(e, t) for t in retry_on):
                break
            attempt += 1
            if attempt >= policy.max_attempts:
                break
            # exponential backoff with jitter
            delay = min(policy.max_delay_ms, policy.base_delay_ms * (2 ** (attempt - 1)))
            delay = delay * (0.8 + 0.4 * random.random())
            await asyncio.sleep(delay / 1000.0)
    if last_exc:
        raise last_exc

class CircuitOpen(Exception):
    pass

class CircuitBreaker:
    def __init__(self, error_threshold: int = 3, cool_off_s: int = 300):
        self.error_threshold = error_threshold
        self.cool_off_s = cool_off_s
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

