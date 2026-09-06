"""Privacy-safe metrics, structured logging, and optional tracing helpers."""

from __future__ import annotations

import json
import logging
import math
import threading
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator


HTTP_DURATION_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)


def _label(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


class MetricsRegistry:
    """Small in-process registry with no tenant or user labels."""

    def __init__(self):
        self._lock = threading.Lock()
        self._http: Counter[tuple[str, str, int]] = Counter()
        self._duration_count: Counter[tuple[str, str]] = Counter()
        self._duration_sum: defaultdict[tuple[str, str], float] = defaultdict(float)
        self._duration_buckets: Counter[tuple[str, str, float]] = Counter()
        self._in_flight = 0
        self._ingest: Counter[str] = Counter()

    def request_started(self) -> None:
        with self._lock:
            self._in_flight += 1

    def request_finished(
        self,
        *,
        method: str,
        route: str,
        status_code: int,
        duration_seconds: float,
    ) -> None:
        duration = max(0.0, duration_seconds)
        key = (method, route)
        with self._lock:
            self._in_flight = max(0, self._in_flight - 1)
            self._http[(method, route, status_code)] += 1
            self._duration_count[key] += 1
            self._duration_sum[key] += duration
            for boundary in HTTP_DURATION_BUCKETS:
                if duration <= boundary:
                    self._duration_buckets[(method, route, boundary)] += 1

    def ingest_batch(self, *, accepted: int, duplicates: int, rejected: int) -> None:
        with self._lock:
            self._ingest["accepted"] += accepted
            self._ingest["duplicate"] += duplicates
            self._ingest["rejected"] += rejected

    def render(self) -> str:
        with self._lock:
            http = self._http.copy()
            counts = self._duration_count.copy()
            sums = dict(self._duration_sum)
            buckets = self._duration_buckets.copy()
            ingest = self._ingest.copy()
            in_flight = self._in_flight

        lines = [
            "# HELP cacheeconomics_http_requests_total HTTP requests handled.",
            "# TYPE cacheeconomics_http_requests_total counter",
        ]
        for (method, route, status_code), count in sorted(http.items()):
            labels = (
                f'method="{_label(method)}",route="{_label(route)}",'
                f'status="{status_code}"'
            )
            lines.append(f"cacheeconomics_http_requests_total{{{labels}}} {count}")
        lines.extend(
            [
                "# HELP cacheeconomics_http_request_duration_seconds Request duration.",
                "# TYPE cacheeconomics_http_request_duration_seconds histogram",
            ]
        )
        for method, route in sorted(counts):
            base = f'method="{_label(method)}",route="{_label(route)}"'
            for boundary in HTTP_DURATION_BUCKETS:
                count = buckets[(method, route, boundary)]
                lines.append(
                    "cacheeconomics_http_request_duration_seconds_bucket"
                    f'{{{base},le="{boundary:g}"}} {count}'
                )
            lines.append(
                "cacheeconomics_http_request_duration_seconds_bucket"
                f'{{{base},le="+Inf"}} {counts[(method, route)]}'
            )
            lines.append(
                "cacheeconomics_http_request_duration_seconds_sum"
                f"{{{base}}} {sums[(method, route)]:.9f}"
            )
            lines.append(
                "cacheeconomics_http_request_duration_seconds_count"
                f"{{{base}}} {counts[(method, route)]}"
            )
        lines.extend(
            [
                "# HELP cacheeconomics_http_requests_in_flight Active HTTP requests.",
                "# TYPE cacheeconomics_http_requests_in_flight gauge",
                f"cacheeconomics_http_requests_in_flight {in_flight}",
                "# HELP cacheeconomics_ingest_events_total Ingest event outcomes.",
                "# TYPE cacheeconomics_ingest_events_total counter",
            ]
        )
        for outcome in ("accepted", "duplicate", "rejected"):
            lines.append(
                "cacheeconomics_ingest_events_total"
                f'{{outcome="{outcome}"}} {ingest[outcome]}'
            )
        return "\n".join(lines) + "\n"


class JsonEventFormatter(logging.Formatter):
    """Emit only a fixed event name and explicitly supplied safe fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname.lower(),
            "service": "cacheeconomics-control-plane",
            "event": str(record.msg),
        }
        fields = getattr(record, "event_fields", {})
        if isinstance(fields, dict):
            for key, value in fields.items():
                if isinstance(key, str) and isinstance(
                    value, (str, int, float, bool, type(None))
                ):
                    if not isinstance(value, float) or math.isfinite(value):
                        payload[key] = value
        return json.dumps(payload, separators=(",", ":"), allow_nan=False)


def event_logger(name: str, *, json_logs: bool) -> logging.Logger:
    logger = logging.getLogger(name)
    if getattr(logger, "_cacheeconomics_configured", False):
        return logger
    handler = logging.StreamHandler()
    if json_logs:
        handler.setFormatter(JsonEventFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger._cacheeconomics_configured = True
    return logger


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    logger.info(event, extra={"event_fields": fields})


@contextmanager
def trace_span(name: str, **attributes: Any) -> Iterator[None]:
    """Create a span when OpenTelemetry is installed; otherwise do nothing."""

    try:
        from opentelemetry import trace
    except ImportError:
        yield
        return
    tracer = trace.get_tracer("cacheeconomics-control-plane")
    safe_attributes = {
        key: value
        for key, value in attributes.items()
        if isinstance(value, (str, int, float, bool))
    }
    with tracer.start_as_current_span(
        name,
        attributes=safe_attributes,
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        try:
            yield
        except BaseException:
            # Mark the span without attaching the exception message or stack.
            # Stored ingest events are prompt-free, but fail closed here too so
            # future exception text cannot silently become telemetry content.
            span.set_status(trace.Status(trace.StatusCode.ERROR))
            raise
