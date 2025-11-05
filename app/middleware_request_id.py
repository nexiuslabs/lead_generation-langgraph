from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
import uuid


class CorrelationMiddleware(BaseHTTPMiddleware):
    """Assigns request_id and parses/derives a trace_id per request.

    - request_id: from X-Request-Id header or generated as r-<hex>
    - trace_id: parse W3C traceparent header when present; else t-<hex>
    Adds both to request.state and echoes as response headers for easy correlation.
    """

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        req_id = request.headers.get("x-request-id") or f"r-{uuid.uuid4().hex[:16]}"
        traceparent = request.headers.get("traceparent")
        trace_id = None
        if traceparent and isinstance(traceparent, str):
            # traceparent format: 00-<trace-id>-<span-id>-<flags>
            try:
                parts = traceparent.split("-")
                if len(parts) >= 3:
                    trace_id = parts[1]
            except Exception:
                trace_id = None
        if not trace_id:
            trace_id = f"t-{uuid.uuid4().hex[:16]}"

        # attach to request context
        request.state.request_id = req_id
        request.state.trace_id = trace_id

        response: Response = await call_next(request)
        # echo headers for correlation across hops
        try:
            response.headers["x-request-id"] = req_id
            response.headers["x-trace-id"] = trace_id
        except Exception:
            pass
        return response

