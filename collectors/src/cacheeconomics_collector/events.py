"""Create the strict prompt-free event sent across the service boundary."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Iterable

from cacheeconomics.trace import Request, is_trusted_id

from . import __version__


FORBIDDEN_KEYS = frozenset(
    {
        "content",
        "completion",
        "completions",
        "body",
        "error_message",
        "messages",
        "prompt",
        "prompts",
        "request_body",
        "response_body",
        "tool_arguments",
        "tool_calls",
    }
)

EVENT_KEYS = frozenset(
    {
        "schema",
        "schema_version",
        "event_id",
        "request_id",
        "sent_at",
        "first_token_at",
        "completed_at",
        "model",
        "target_id",
        "agent",
        "session",
        "workload_tenant",
        "ttl_requested",
        "tokens_counted",
        "status",
        "usage",
        "segments",
        "collector",
    }
)
REQUIRED_EVENT_KEYS = frozenset(
    {
        "schema",
        "schema_version",
        "event_id",
        "sent_at",
        "model",
        "target_id",
        "tokens_counted",
        "status",
        "usage",
        "segments",
        "collector",
    }
)
STATUS_KEYS = frozenset({"outcome", "code", "error_type"})
USAGE_KEYS = frozenset(
    {
        "input_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
        "output_tokens",
        "cache_creation",
    }
)
REQUIRED_USAGE_KEYS = frozenset(
    {
        "input_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
        "output_tokens",
    }
)
CACHE_CREATION_KEYS = frozenset(
    {"ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens"}
)
SEGMENT_KEYS = frozenset(
    {"id", "role", "label", "tokens", "cache_marked", "index", "ttl"}
)
REQUIRED_SEGMENT_KEYS = frozenset(
    {"id", "role", "tokens", "cache_marked", "index", "ttl"}
)
COLLECTOR_KEYS = frozenset({"name", "version", "source_type"})
ERROR_TYPES = frozenset(
    {
        "authentication_error",
        "authorization_error",
        "bad_request",
        "cancelled",
        "network_error",
        "provider_error",
        "rate_limit",
        "timeout",
        "unavailable",
        "unknown",
    }
)
SOURCE_TYPES = frozenset(
    {"normalized_trace", "litellm", "request_bodies", "claude_code"}
)
HMAC_ID = re.compile(r"^hmac:[0-9a-f]{64}$")
SCHEMA_ERROR = "event does not match the prompt-free ingest schema"


def assert_prompt_free(value: Any, path: str = "event") -> None:
    """Reject common raw-content fields before an event reaches disk or network."""

    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in FORBIDDEN_KEYS:
                raise ValueError("event contains a forbidden field")
            assert_prompt_free(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_prompt_free(item, f"{path}[{index}]")


def _invalid() -> None:
    # Deliberately generic: a caller-controlled field name or value must not be
    # copied into logs by an exception handler.
    raise ValueError(SCHEMA_ERROR)


def _object_keys(
    value: Any,
    *,
    allowed: frozenset[str],
    required: frozenset[str],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        _invalid()
    keys = set(value)
    if (
        keys - allowed
        or required - keys
        or any(not isinstance(key, str) for key in keys)
    ):
        _invalid()
    return value


def _string(value: Any, *, minimum: int = 0, maximum: int) -> str:
    if not isinstance(value, str) or not minimum <= len(value) <= maximum:
        _invalid()
    return value


def _optional_string(value: Any, *, maximum: int) -> None:
    if value is not None:
        _string(value, maximum=maximum)


def _token_count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _invalid()
    return value


def _timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        _invalid()
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        _invalid()
    if parsed.tzinfo is None:
        _invalid()
    return parsed.astimezone(timezone.utc)


def _optional_timestamp(value: Any) -> datetime | None:
    return None if value is None else _timestamp(value)


def validate_event(value: Any) -> None:
    """Validate the complete allow-listed shape before disk or network I/O."""

    event = _object_keys(value, allowed=EVENT_KEYS, required=REQUIRED_EVENT_KEYS)
    if event["schema"] != "cacheeconomics.ingest-event":
        _invalid()
    if isinstance(event["schema_version"], bool) or event["schema_version"] != 1:
        _invalid()
    _string(event["event_id"], minimum=1, maximum=256)
    _optional_string(event.get("request_id"), maximum=256)
    sent_at = _timestamp(event["sent_at"])
    first_token_at = _optional_timestamp(event.get("first_token_at"))
    completed_at = _optional_timestamp(event.get("completed_at"))
    if first_token_at is not None and first_token_at < sent_at:
        _invalid()
    if completed_at is not None and completed_at < sent_at:
        _invalid()
    if (
        first_token_at is not None
        and completed_at is not None
        and completed_at < first_token_at
    ):
        _invalid()
    _string(event["model"], minimum=1, maximum=256)
    _string(event["target_id"], minimum=1, maximum=256)
    _string(event.get("agent", "unknown"), maximum=256)
    _optional_string(event.get("session"), maximum=256)
    _optional_string(event.get("workload_tenant"), maximum=256)
    _optional_string(event.get("ttl_requested"), maximum=64)
    if not isinstance(event["tokens_counted"], bool):
        _invalid()

    status = _object_keys(
        event["status"],
        allowed=STATUS_KEYS,
        required=STATUS_KEYS,
    )
    outcome = status["outcome"]
    if not isinstance(outcome, str) or outcome not in {
        "success",
        "error",
        "cancelled",
        "unknown",
    }:
        _invalid()
    code = status["code"]
    if code is not None and (
        isinstance(code, bool) or not isinstance(code, int) or not 100 <= code <= 599
    ):
        _invalid()
    error_type = status["error_type"]
    if error_type is not None and (
        not isinstance(error_type, str) or error_type not in ERROR_TYPES
    ):
        _invalid()
    if outcome == "success" and (
        error_type is not None or (code is not None and not 200 <= code <= 299)
    ):
        _invalid()
    if outcome != "success" and code is not None and 200 <= code <= 299:
        _invalid()
    if outcome == "error" and error_type is None:
        _invalid()

    usage = _object_keys(
        event["usage"],
        allowed=USAGE_KEYS,
        required=REQUIRED_USAGE_KEYS,
    )
    for key in REQUIRED_USAGE_KEYS:
        _token_count(usage[key])
    if "cache_creation" in usage:
        split = _object_keys(
            usage["cache_creation"],
            allowed=CACHE_CREATION_KEYS,
            required=frozenset(),
        )
        for count in split.values():
            _token_count(count)

    segments = event["segments"]
    if not isinstance(segments, list) or len(segments) > 256:
        _invalid()
    for expected_index, raw_segment in enumerate(segments):
        segment = _object_keys(
            raw_segment,
            allowed=SEGMENT_KEYS,
            required=REQUIRED_SEGMENT_KEYS,
        )
        segment_id = _string(segment["id"], minimum=1, maximum=69)
        if not HMAC_ID.fullmatch(segment_id):
            _invalid()
        _string(segment["role"], minimum=1, maximum=64)
        _string(segment.get("label", ""), maximum=256)
        _token_count(segment["tokens"])
        if not isinstance(segment["cache_marked"], bool):
            _invalid()
        if (
            isinstance(segment["index"], bool)
            or not isinstance(segment["index"], int)
            or segment["index"] != expected_index
        ):
            _invalid()
        _optional_string(segment["ttl"], maximum=64)

    collector = _object_keys(
        event["collector"],
        allowed=COLLECTOR_KEYS,
        required=COLLECTOR_KEYS,
    )
    _string(collector["name"], minimum=1, maximum=128)
    _string(collector["version"], minimum=1, maximum=64)
    if (
        not isinstance(collector["source_type"], str)
        or collector["source_type"] not in SOURCE_TYPES
    ):
        _invalid()


def _event_id(request: Request, source_type: str) -> str:
    identity = json.dumps(
        {
            "source_type": source_type,
            "request_id": request.request_id,
            "sent_at": request.sent_at.isoformat() if request.sent_at else None,
            "model": request.model,
            "target_id": request.target_id,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{source_type}:{hashlib.sha256(identity.encode()).hexdigest()}"


def _count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return 0
    return int(value)


def _usage(request: Request) -> dict[str, Any]:
    usage = request.usage if isinstance(request.usage, dict) else {}
    result = {
        "input_tokens": _count(usage.get("input_tokens")),
        "cache_read_input_tokens": _count(usage.get("cache_read_input_tokens")),
        "cache_creation_input_tokens": _count(usage.get("cache_creation_input_tokens")),
        "output_tokens": _count(usage.get("output_tokens")),
    }
    split = usage.get("cache_creation")
    if isinstance(split, dict):
        result["cache_creation"] = {
            key: _count(split[key])
            for key in ("ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens")
            if key in split
        }
    return result


def event_from_request(
    request: Request,
    *,
    source_type: str,
    tokens_counted: bool = False,
    completed_at: datetime | None = None,
    error_type: str | None = None,
) -> dict[str, Any]:
    if request.sent_at is None:
        raise ValueError("a request without sent_at cannot be ingested")
    if source_type not in SOURCE_TYPES:
        raise ValueError("unsupported source_type")

    # One untrusted id downgrades the whole row to usage-only. Replacing it or
    # uploading a bare digest would invent instrumented confidence.
    segments = []
    if request.segments and all(is_trusted_id(segment.id) for segment in request.segments):
        segments = [
            {
                "id": segment.id,
                "role": segment.role,
                "label": segment.label,
                "tokens": max(0, int(segment.tokens)),
                "cache_marked": bool(segment.cache_marked),
                "index": index,
                "ttl": segment.ttl,
            }
            for index, segment in enumerate(sorted(request.segments, key=lambda item: item.index))
        ]

    successful = request.status == 200
    code = request.status if 100 <= request.status <= 599 else None
    event = {
        "schema": "cacheeconomics.ingest-event",
        "schema_version": 1,
        "event_id": _event_id(request, source_type),
        "request_id": request.request_id or None,
        "sent_at": _as_utc(request.sent_at).isoformat(),
        "first_token_at": (
            _as_utc(request.first_token_at).isoformat() if request.first_token_at else None
        ),
        "completed_at": _as_utc(completed_at).isoformat() if completed_at else None,
        "model": request.model,
        "target_id": request.target_id,
        "agent": request.agent,
        "session": request.session,
        "workload_tenant": request.tenant,
        "ttl_requested": request.ttl_requested,
        "tokens_counted": bool(tokens_counted and segments),
        "status": {
            "outcome": "success" if successful else "error",
            "code": code,
            "error_type": None if successful else (error_type or "unknown"),
        },
        "usage": _usage(request),
        "segments": segments,
        "collector": {
            "name": "cacheeconomics-collector",
            "version": __version__,
            "source_type": source_type,
        },
    }
    validate_event(event)
    return event


def events_from_requests(
    requests: Iterable[Request],
    *,
    source_type: str,
) -> list[dict[str, Any]]:
    return [event_from_request(request, source_type=source_type) for request in requests]


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
