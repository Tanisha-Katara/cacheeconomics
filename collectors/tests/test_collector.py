from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import sqlite3
import threading
import urllib.error
import urllib.request
from datetime import datetime, timezone

import pytest

from cacheeconomics.plugin import CachePlugin
from cacheeconomics.trace import Request, Segment
from cacheeconomics_collector.client import (
    DeliveryError,
    DurableUploader,
    HttpTransport,
    _RejectRedirectHandler,
)
from cacheeconomics_collector.cli import main as collector_cli_main
from cacheeconomics_collector.events import (
    CACHE_CREATION_KEYS,
    COLLECTOR_KEYS,
    ERROR_TYPES,
    EVENT_KEYS,
    REQUIRED_EVENT_KEYS,
    REQUIRED_SEGMENT_KEYS,
    REQUIRED_USAGE_KEYS,
    SEGMENT_KEYS,
    SOURCE_TYPES,
    STATUS_KEYS,
    USAGE_KEYS,
    assert_prompt_free,
    event_from_request,
    validate_event,
)
from cacheeconomics_collector.litellm import live_litellm_handler


def request_with_segments():
    return Request(
        request_id="request-1",
        sent_at=datetime.now(timezone.utc),
        model="claude-opus-5",
        target_id="anthropic/direct",
        usage={
            "input_tokens": 1_000,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "output_tokens": 10,
        },
        segments=[
            Segment(
                id="hmac:" + "a" * 64,
                role="system",
                label="policy",
                tokens=1_000,
                index=0,
                cache_marked=True,
                ttl="5m",
            )
        ],
        tenant="tenant-a",
    )


def test_event_is_stable_strict_and_prompt_free():
    request = request_with_segments()
    one = event_from_request(request, source_type="normalized_trace", tokens_counted=True)
    two = event_from_request(request, source_type="normalized_trace", tokens_counted=True)
    assert one["event_id"] == two["event_id"]
    assert one["tokens_counted"] is True
    assert one["segments"][0]["id"].startswith("hmac:")
    serialized = json.dumps(one)
    assert "messages" not in serialized
    assert "completion" not in serialized


def test_collector_allowlist_matches_the_versioned_public_schema():
    root = Path(__file__).resolve().parents[2]
    schema = json.loads((root / "schemas/ingest-event-v1.schema.json").read_text())
    status = schema["properties"]["status"]
    usage = schema["$defs"]["usage"]
    segment = schema["$defs"]["segment"]
    collector = schema["properties"]["collector"]

    assert EVENT_KEYS == set(schema["properties"])
    assert REQUIRED_EVENT_KEYS == set(schema["required"])
    assert STATUS_KEYS == set(status["properties"]) == set(status["required"])
    assert USAGE_KEYS == set(usage["properties"])
    assert REQUIRED_USAGE_KEYS == set(usage["required"])
    assert CACHE_CREATION_KEYS == set(
        usage["properties"]["cache_creation"]["properties"]
    )
    assert SEGMENT_KEYS == set(segment["properties"])
    assert REQUIRED_SEGMENT_KEYS == set(segment["required"])
    assert COLLECTOR_KEYS == set(collector["properties"]) == set(
        collector["required"]
    )
    assert SOURCE_TYPES == set(collector["properties"]["source_type"]["enum"])
    assert ERROR_TYPES == set(status["properties"]["error_type"]["enum"]) - {None}


def test_untrusted_segment_ids_are_not_uploaded_as_instrumented():
    request = request_with_segments()
    request.segments[0] = Segment(
        id="sha256:" + "a" * 64,
        role="system",
        label="policy",
        tokens=1_000,
        index=0,
    )
    event = event_from_request(request, source_type="normalized_trace", tokens_counted=True)
    assert event["segments"] == []
    assert event["tokens_counted"] is False


def test_prompt_fields_are_rejected_before_the_outbox(tmp_path):
    event = event_from_request(request_with_segments(), source_type="normalized_trace")
    event["messages"] = [{"content": "private"}]
    with pytest.raises(ValueError, match="forbidden field"):
        assert_prompt_free(event)

    uploader = DurableUploader(tmp_path / "outbox.sqlite3", AcceptingTransport())
    with pytest.raises(ValueError, match="prompt-free ingest schema"):
        uploader.enqueue(event)
    assert uploader.counts()["pending"] == 0


@pytest.mark.parametrize(
    "field",
    ["body", "error_message", "prompts", "completions", "unrecognized"],
)
def test_complete_allowlist_rejects_sensitive_or_unknown_fields_before_disk(
    tmp_path, field
):
    event = event_from_request(request_with_segments(), source_type="normalized_trace")
    event[field] = "private customer value"
    uploader = DurableUploader(tmp_path / f"{field}.sqlite3", AcceptingTransport())

    with pytest.raises(ValueError):
        validate_event(event)
    with pytest.raises(ValueError):
        uploader.enqueue(event)
    assert uploader.counts()["pending"] == 0


def test_complete_allowlist_applies_to_nested_objects(tmp_path):
    event = event_from_request(request_with_segments(), source_type="normalized_trace")
    event["status"]["debug"] = "private provider response"
    uploader = DurableUploader(tmp_path / "outbox.sqlite3", AcceptingTransport())

    with pytest.raises(ValueError, match="prompt-free ingest schema"):
        uploader.enqueue(event)
    assert uploader.counts()["pending"] == 0


@pytest.mark.parametrize("key_hex", ["", "00", "00" * 15])
def test_normalized_trace_cli_rejects_short_hmac_keys(tmp_path, capsys, key_hex):
    key_file = tmp_path / "hmac-key.hex"
    key_file.write_text(key_hex)

    with pytest.raises(SystemExit) as raised:
        collector_cli_main(
            [
                str(tmp_path / "trace.jsonl"),
                "--format",
                "normalized_trace",
                "--endpoint",
                "https://collector.example.test",
                "--token",
                "cec_test",
                "--hmac-key-file",
                str(key_file),
            ]
        )

    assert raised.value.code == 2
    assert "must contain at least 16 bytes" in capsys.readouterr().err


def test_http_transport_requires_tls_except_for_loopback():
    with pytest.raises(ValueError, match="HTTPS"):
        HttpTransport("http://collector.example.test", "cec_test")
    with pytest.raises(ValueError, match="without credentials"):
        HttpTransport("https://user:pass@collector.example.test", "cec_test")
    assert HttpTransport("https://collector.example.test", "cec_test").url.endswith(
        "/v1/ingest/events"
    )
    assert HttpTransport("http://127.0.0.1:8000", "cec_test").url.endswith(
        "/v1/ingest/events"
    )


@pytest.mark.parametrize("status_code", [301, 302, 303, 307, 308])
def test_http_transport_rejects_redirects_without_building_a_second_request(
    status_code,
):
    handler = _RejectRedirectHandler()
    request = urllib.request.Request(
        "https://collector.example.test/v1/ingest/events",
        headers={"Authorization": "Bearer cec_private"},
    )
    response = io.BytesIO(b"")

    with pytest.raises(urllib.error.HTTPError) as raised:
        getattr(handler, f"http_error_{status_code}")(
            request,
            response,
            status_code,
            "redirect",
            {"Location": "https://attacker.example.test/steal"},
        )
    assert raised.value.code == status_code
    assert response.closed


class AcceptingTransport:
    def __init__(self):
        self.batches = []

    def send(self, events):
        self.batches.append(events)
        return {
            "items": [
                {"event_id": event["event_id"], "status": "accepted"}
                for event in events
            ]
        }


class FlakyTransport:
    def __init__(self):
        self.calls = 0

    def send(self, events):
        self.calls += 1
        if self.calls == 1:
            raise DeliveryError("network_error")
        return AcceptingTransport().send(events)


class BlockingTransport(AcceptingTransport):
    def __init__(self):
        super().__init__()
        self.first_send_started = threading.Event()
        self.second_send_started = threading.Event()
        self.release = threading.Event()
        self._calls_lock = threading.Lock()

    def send(self, events):
        with self._calls_lock:
            call = len(self.batches) + 1
            self.batches.append(events)
        if call == 1:
            self.first_send_started.set()
        else:
            self.second_send_started.set()
        if not self.release.wait(timeout=2):
            raise RuntimeError("test transport was not released")
        return {
            "items": [
                {"event_id": event["event_id"], "status": "accepted"}
                for event in events
            ]
        }


def test_durable_outbox_retries_without_losing_the_event(tmp_path):
    transport = FlakyTransport()
    uploader = DurableUploader(
        tmp_path / "outbox.sqlite3",
        transport,
        retry_base_seconds=0,
    )
    event = event_from_request(request_with_segments(), source_type="normalized_trace")
    first = uploader.submit(event)
    assert first == {"sent": 0, "retried": 1, "dead_lettered": 0}
    assert uploader.counts()["pending"] == 1
    second = uploader.flush()
    assert second == {"sent": 1, "retried": 0, "dead_lettered": 0}
    assert uploader.counts() == {"pending": 0, "dead_letters": 0}

    with sqlite3.connect(tmp_path / "outbox.sqlite3") as connection:
        assert connection.execute("SELECT count(*) FROM pending_events").fetchone()[0] == 0


def test_concurrent_flushes_send_each_pending_event_only_once(tmp_path):
    transport = BlockingTransport()
    uploader = DurableUploader(tmp_path / "outbox.sqlite3", transport)
    uploader.enqueue(
        event_from_request(request_with_segments(), source_type="normalized_trace")
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(uploader.flush)
        assert transport.first_send_started.wait(timeout=1)
        second = executor.submit(uploader.flush)
        try:
            assert not transport.second_send_started.wait(timeout=0.1)
        finally:
            transport.release.set()
        results = [first.result(timeout=2), second.result(timeout=2)]

    assert transport.batches and len(transport.batches) == 1
    assert sorted(result["sent"] for result in results) == [0, 1]
    assert uploader.counts() == {"pending": 0, "dead_letters": 0}


def test_outbox_splits_batches_by_count_before_delivery(tmp_path):
    transport = AcceptingTransport()
    uploader = DurableUploader(
        tmp_path / "outbox.sqlite3",
        transport,
        batch_size=1,
    )
    uploader.enqueue(event_from_request(request_with_segments(), source_type="normalized_trace"))
    second_request = request_with_segments()
    second_request.request_id = "request-2"
    uploader.enqueue(event_from_request(second_request, source_type="normalized_trace"))

    assert uploader.flush()["sent"] == 1
    assert uploader.counts()["pending"] == 1
    assert uploader.flush()["sent"] == 1
    assert [len(batch) for batch in transport.batches] == [1, 1]


def test_outbox_splits_batches_by_encoded_size_before_delivery(tmp_path):
    first = event_from_request(request_with_segments(), source_type="normalized_trace")
    second_request = request_with_segments()
    second_request.request_id = "request-2"
    second = event_from_request(second_request, source_type="normalized_trace")
    event_bytes = len(json.dumps(first, separators=(",", ":")).encode("utf-8"))
    transport = AcceptingTransport()
    uploader = DurableUploader(
        tmp_path / "outbox.sqlite3",
        transport,
        batch_size=2,
        max_event_bytes=event_bytes,
        max_batch_bytes=len(b'{"events":[]}') + event_bytes,
    )
    uploader.enqueue(first)
    uploader.enqueue(second)

    assert uploader.flush()["sent"] == 1
    assert uploader.counts()["pending"] == 1
    assert uploader.flush()["sent"] == 1
    assert [len(batch) for batch in transport.batches] == [1, 1]


class RecordingUploader:
    def __init__(self, *, fail=False):
        self.events = []
        self.fail = fail

    def submit(self, event):
        if self.fail:
            raise RuntimeError("do not leak this raw failure")
        self.events.append(event)
        return {"sent": 1, "retried": 0, "dead_lettered": 0}


def test_litellm_callback_never_exports_messages_and_fails_open():
    private_text = "customer password is swordfish"
    uploader = RecordingUploader()
    handler = live_litellm_handler(
        CachePlugin(key=b"k" * 32, warmup=1),
        uploader,
        base=object,
        mutate=False,
        target_id="anthropic/direct",
    )
    data = {
        "litellm_call_id": "call-1",
        "model": "claude-opus-5",
        "messages": [{"role": "user", "content": private_text}],
    }
    returned = asyncio.run(handler.async_pre_call_hook({}, None, data, "completion"))
    assert returned == data
    asyncio.run(
        handler.async_log_success_event(
            data,
            {
                "id": "response-1",
                "usage": {
                    "input_tokens": 100,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "output_tokens": 5,
                },
            },
            datetime.now(timezone.utc),
            datetime.now(timezone.utc),
        )
    )
    assert len(uploader.events) == 1
    assert private_text not in json.dumps(uploader.events[0])
    assert uploader.events[0]["collector"]["source_type"] == "litellm"

    failing = live_litellm_handler(
        CachePlugin(key=b"z" * 32, warmup=1),
        RecordingUploader(fail=True),
        base=object,
        mutate=False,
        target_id="anthropic/direct",
    )
    failing_data = {**data, "litellm_call_id": "call-2"}
    asyncio.run(failing.async_pre_call_hook({}, None, failing_data, "completion"))
    asyncio.run(
        failing.async_log_success_event(
            failing_data,
            {"id": "response-2", "usage": {"input_tokens": 100}},
            datetime.now(timezone.utc),
            datetime.now(timezone.utc),
        )
    )
    assert failing.delivery_failures == 1


def test_litellm_callback_fails_open_when_event_validation_rejects_telemetry(
    caplog,
):
    handler = live_litellm_handler(
        CachePlugin(key=b"v" * 32, warmup=1),
        RecordingUploader(),
        base=object,
        mutate=False,
        target_id="anthropic/direct",
    )
    data = {
        "litellm_call_id": "invalid-telemetry",
        "model": "claude-opus-5",
        "messages": [{"role": "user", "content": "private request"}],
    }
    asyncio.run(handler.async_pre_call_hook({}, None, data, "completion"))

    private_error = "private telemetry detail"

    def rejected_event(*_args, **_kwargs):
        raise ValueError(private_error)

    handler._event = rejected_event
    asyncio.run(
        handler.async_log_success_event(
            data,
            {"id": "response", "usage": {"input_tokens": 100}},
            datetime.now(timezone.utc),
            datetime.now(timezone.utc),
        )
    )

    assert handler.delivery_failures == 1
    assert private_error not in caplog.text
