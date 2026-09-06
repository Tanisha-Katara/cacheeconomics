"""LiteLLM proxy callback that exports only prompt-free telemetry."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from cacheeconomics.plugin import CachePlugin, litellm_handler
from cacheeconomics.segment import usage_from_response
from cacheeconomics.trace import Request, read_field

from .events import event_from_request


_log = logging.getLogger("cacheeconomics.collector.litellm")


def _safe_error_type(value) -> str:
    if isinstance(value, TimeoutError):
        return "timeout"
    if isinstance(value, ConnectionError):
        return "network_error"
    return "provider_error"


def _as_datetime(value, fallback: datetime) -> datetime:
    if not isinstance(value, datetime):
        return fallback
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def live_litellm_handler(
    plugin: CachePlugin,
    uploader,
    *,
    base=None,
    mutate: bool = False,
    target_id: str | None = None,
):
    """Compose the existing cache observer with a durable service uploader."""

    observed = litellm_handler(
        plugin,
        base=base,
        mutate=mutate,
        target_id=target_id,
    )
    parent = type(observed)

    class CacheEconomicsCollector(parent):
        def __init__(self):
            # The already-created observer owns the state and pending decisions.
            self.__dict__ = observed.__dict__
            self.uploader = uploader
            self.delivery_failures = 0

        async def _deliver(self, event):
            try:
                await asyncio.to_thread(self.uploader.submit, event)
            except Exception as error:
                self.delivery_failures += 1
                _log.warning(
                    "cacheeconomics collector failed open (%s)", type(error).__name__
                )

        def _safe_event(self, *args, **kwargs):
            try:
                return self._event(*args, **kwargs)
            except Exception as error:
                # A malformed callback shape is telemetry loss, not a reason
                # to interfere with the caller's completed LLM request.
                self.delivery_failures += 1
                _log.warning(
                    "cacheeconomics collector rejected telemetry (%s)",
                    type(error).__name__,
                )
                return None

        def _event(self, kwargs, response_obj, end_time, *, outcome, error_type=None):
            decision = self._pending.get(kwargs.get("litellm_call_id"))
            if decision is None:
                return None
            usage = usage_from_response(response_obj) or {}
            scope = decision.scope or (None, target_id or "unknown/unattributed", kwargs.get("model", ""))
            status = 200 if outcome == "success" else 500
            request = Request(
                request_id=str(
                    read_field(response_obj, "id")
                    or kwargs.get("litellm_call_id")
                    or f"live-{decision.sent_at.isoformat()}"
                ),
                sent_at=decision.sent_at,
                first_token_at=None,
                model=scope[2] or kwargs.get("model", ""),
                target_id=scope[1],
                agent=decision.agent,
                session=decision.session,
                tenant=scope[0],
                ttl_requested=_single_ttl(decision.segments),
                status=status,
                usage=usage,
                segments=list(decision.segments),
            )
            return event_from_request(
                request,
                source_type="litellm",
                tokens_counted=False,
                completed_at=_as_datetime(end_time, datetime.now(timezone.utc)),
                error_type=error_type,
            )

        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            event = self._safe_event(
                kwargs,
                response_obj,
                end_time,
                outcome="success",
            )
            await super().async_log_success_event(kwargs, response_obj, start_time, end_time)
            if event is not None:
                await self._deliver(event)

        async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
            event = self._safe_event(
                kwargs,
                response_obj,
                end_time,
                outcome="error",
                error_type=_safe_error_type(kwargs.get("exception")),
            )
            self._pending.pop(kwargs.get("litellm_call_id"), None)
            if event is not None:
                await self._deliver(event)

    return CacheEconomicsCollector()


def _single_ttl(segments) -> str | None:
    values = {segment.ttl or "5m" for segment in segments if segment.cache_marked}
    return next(iter(values)) if len(values) == 1 else None
