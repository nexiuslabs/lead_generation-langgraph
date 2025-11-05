from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

BaseCallbackHandler = None
try:  # langgraph>=0.2.35
    from langgraph.callbacks.base import BaseCallbackHandler as _BaseHandler
    BaseCallbackHandler = _BaseHandler
except ModuleNotFoundError:
    pass

if BaseCallbackHandler is None:
    try:
        from langgraph_sdk.callbacks.base import BaseCallbackHandler as _SDKBaseHandler  # type: ignore
        BaseCallbackHandler = _SDKBaseHandler
    except ModuleNotFoundError:
        pass

if BaseCallbackHandler is None:
    try:
        from langchain_core.callbacks.base import (  # type: ignore
            BaseCallbackHandler as _LangChainBaseHandler,
        )

        BaseCallbackHandler = _LangChainBaseHandler
    except ModuleNotFoundError:
        pass


if BaseCallbackHandler is None:  # pragma: no cover
    class BaseCallbackHandler:  # type: ignore
        """Fallback no-op callback handler when LangGraph callbacks are unavailable."""

        run_inline = True
        raise_error = False

        @property
        def ignore_llm(self) -> bool:
            return True

        @property
        def ignore_retry(self) -> bool:
            return True

        @property
        def ignore_chain(self) -> bool:
            return True

        @property
        def ignore_agent(self) -> bool:
            return True

        @property
        def ignore_retriever(self) -> bool:
            return True

        @property
        def ignore_chat_model(self) -> bool:
            return True

        @property
        def ignore_custom_event(self) -> bool:
            return True

        def on_node_start(self, *args, **kwargs):  # noqa: D401 - noop
            return None

        def on_node_end(self, *args, **kwargs):
            return None

        def on_node_error(self, *args, **kwargs):
            return None

        # langchain compatibility hooks
        def on_llm_new_token(self, *args, **kwargs):
            return None

        def on_llm_end(self, *args, **kwargs):
            return None

        def on_llm_error(self, *args, **kwargs):
            return None

        def on_chain_start(self, *args, **kwargs):
            return None

        def on_chain_end(self, *args, **kwargs):
            return None

        def on_chain_error(self, *args, **kwargs):
            return None

        def on_tool_start(self, *args, **kwargs):
            return None

        def on_tool_end(self, *args, **kwargs):
            return None

        def on_tool_error(self, *args, **kwargs):
            return None

        def on_agent_action(self, *args, **kwargs):
            return None

        def on_agent_finish(self, *args, **kwargs):
            return None

        def on_custom_event(self, *args, **kwargs):
            return None



def _env() -> str:
    return (
        os.getenv("ENVIRONMENT")
        or os.getenv("PY_ENV")
        or os.getenv("NODE_ENV")
        or "dev"
    ).strip().lower()


def _sanitize(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for k, v in value.items():
            if len(out) >= 25:
                break
            out[str(k)] = _sanitize(v)
        return out
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in list(value)[:25]]
    return str(value)


class LangGraphTroubleshootHandler(BaseCallbackHandler):
    """Emit LangGraph node lifecycle events to the troubleshoot logger."""

    run_inline = True
    ignore_chain = False
    raise_error = False

    def __init__(self, context: Optional[Dict[str, Any]] = None):
        self.context = context or {}
        self.logger = logging.getLogger("troubleshoot")

    def _emit(self, run_id: str, node: str, status: str, payload: Dict[str, Any]) -> None:
        record = {
            "timestamp": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
            "level": "warn" if status == "error" else "info",
            "service": "langgraph",
            "environment": _env(),
            "run_id": run_id,
            "node": node,
            "status": status,
            "payload": _sanitize(payload),
        }
        if self.context:
            record.update({f"context_{k}": _sanitize(v) for k, v in self.context.items()})
        line = json.dumps(record, ensure_ascii=False)
        level = logging.WARNING if status == "error" else logging.INFO
        try:
            self.logger.log(level, line)
        except Exception:
            # best-effort only; never break graph execution
            pass

    def on_node_start(self, run_id: str, node: str, inputs: Dict[str, Any], **kwargs: Any) -> None:  # type: ignore[override]
        self._emit(run_id, node, "start", {"inputs": _sanitize(inputs)})

    def on_node_end(
        self,
        run_id: str,
        node: str,
        outputs: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:  # type: ignore[override]
        payload: Dict[str, Any] = {"outputs": _sanitize(outputs)}
        if metadata:
            payload["metadata"] = _sanitize(metadata)
        self._emit(run_id, node, "end", payload)

    def on_node_error(self, run_id: str, node: str, error: Exception, **kwargs: Any) -> None:  # type: ignore[override]
        self._emit(
            run_id,
            node,
            "error",
            {
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
        )
