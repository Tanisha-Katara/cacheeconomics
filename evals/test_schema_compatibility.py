from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from cacheeconomics_collector.events import (
    CACHE_CREATION_KEYS,
    COLLECTOR_KEYS,
    EVENT_KEYS,
    REQUIRED_EVENT_KEYS,
    REQUIRED_SEGMENT_KEYS,
    REQUIRED_USAGE_KEYS,
    SEGMENT_KEYS,
    STATUS_KEYS,
    USAGE_KEYS,
    validate_event,
)
from cacheeconomics_control_plane.api_schemas import IngestEventV1


ROOT = Path(__file__).resolve().parents[1]
VALID = ROOT / "evals/fixtures/ingest-event-v1.valid.json"


def _valid() -> dict:
    return json.loads(VALID.read_text())


def _api_accepts(event: dict) -> bool:
    try:
        IngestEventV1.model_validate(event)
    except ValidationError:
        return False
    return True


def _collector_accepts(event: dict) -> bool:
    try:
        validate_event(event)
    except ValueError:
        return False
    return True


def test_checked_in_event_is_accepted_by_both_boundaries():
    event = _valid()
    assert _api_accepts(event)
    assert _collector_accepts(event)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda event: event.update(prompt="secret"),
        lambda event: event.pop("usage"),
        lambda event: event.update(schema_version=2),
        lambda event: event["status"].update(error_message="raw provider detail"),
        lambda event: event["status"].update(outcome="cancelled", code=200),
        lambda event: event["segments"][0].update(id="sha256:" + "a" * 64),
        lambda event: event["segments"][0].update(index=1),
        lambda event: event.update(completed_at="2026-01-02T03:04:04+00:00"),
        lambda event: event["usage"].update(input_tokens=-1),
        lambda event: event["collector"].update(source_type="fabricated-provider"),
    ],
)
def test_malformed_or_sensitive_events_are_rejected_before_disk(mutation):
    event = copy.deepcopy(_valid())
    mutation(event)
    assert not _collector_accepts(event)
    assert not _api_accepts(event)


@pytest.mark.parametrize("field", ["sent_at", "first_token_at", "completed_at"])
def test_timestamps_without_an_explicit_offset_are_rejected_by_both_boundaries(
    field,
):
    event = _valid()
    event[field] = "2026-01-02T03:04:05"

    assert not _collector_accepts(event)
    assert not _api_accepts(event)


def test_schema_property_sets_match_both_runtime_validators():
    schema = json.loads((ROOT / "schemas/ingest-event-v1.schema.json").read_text())
    assert set(schema["properties"]) == set(EVENT_KEYS)
    assert set(schema["required"]) == set(REQUIRED_EVENT_KEYS)
    assert set(schema["properties"]["status"]["properties"]) == set(STATUS_KEYS)
    assert set(schema["$defs"]["usage"]["properties"]) == set(USAGE_KEYS)
    assert set(schema["$defs"]["usage"]["required"]) == set(REQUIRED_USAGE_KEYS)
    assert set(schema["$defs"]["usage"]["properties"]["cache_creation"]["properties"]) == set(CACHE_CREATION_KEYS)
    assert set(schema["$defs"]["segment"]["properties"]) == set(SEGMENT_KEYS)
    assert set(schema["$defs"]["segment"]["required"]) == set(REQUIRED_SEGMENT_KEYS)
    assert set(schema["properties"]["collector"]["properties"]) == set(COLLECTOR_KEYS)

    api_names = {
        field.alias or name for name, field in IngestEventV1.model_fields.items()
    }
    api_required = {
        field.alias or name
        for name, field in IngestEventV1.model_fields.items()
        if field.is_required()
    }
    assert api_names == set(EVENT_KEYS)
    assert api_required == set(REQUIRED_EVENT_KEYS)


def test_published_schema_files_change_only_with_an_explicit_lock_update():
    locked = json.loads((ROOT / "evals/fixtures/schema-lock-v1.json").read_text())
    actual = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((ROOT / "schemas").glob("*.json"))
    }
    assert actual == locked
