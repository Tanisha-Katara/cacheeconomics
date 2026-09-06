"""A durable, bounded sender for prompt-free collector events."""

from __future__ import annotations

import json
import ipaddress
import os
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

from . import __version__
from .events import validate_event


class _RejectRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Never copy a collector credential onto a redirected request."""

    def http_error_302(self, req, fp, code, msg, headers):
        if fp is not None:
            fp.close()
        raise urllib.error.HTTPError(
            req.full_url,
            code,
            "collector redirects are disabled",
            headers,
            None,
        )

    http_error_301 = http_error_302
    http_error_303 = http_error_302
    http_error_307 = http_error_302
    http_error_308 = http_error_302


class Transport(Protocol):
    def send(self, events: list[dict[str, Any]]) -> dict[str, Any]: ...


class HttpTransport:
    def __init__(self, endpoint: str, token: str, *, timeout_seconds: float = 10.0):
        parsed = urlparse(endpoint)
        loopback = parsed.hostname == "localhost"
        if parsed.hostname:
            try:
                loopback = loopback or ipaddress.ip_address(parsed.hostname).is_loopback
            except ValueError:
                pass
        if parsed.scheme != "https" and not (parsed.scheme == "http" and loopback):
            raise ValueError(
                "collector endpoint must use HTTPS (HTTP is allowed only on loopback)"
            )
        if (
            not parsed.netloc
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "collector endpoint must be a plain service URL without credentials"
            )
        if not token or "\n" in token or "\r" in token:
            raise ValueError("collector token is required and must fit in one header")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.url = endpoint.rstrip("/") + "/v1/ingest/events"
        self.token = token
        self.timeout_seconds = timeout_seconds
        # urllib's default redirect handler can forward Authorization to a new
        # origin. Collector credentials are source-scoped secrets, so a
        # redirect is an error and the operator must configure the final URL.
        self._opener = urllib.request.build_opener(_RejectRedirectHandler())

    def send(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        body = json.dumps({"events": events}, separators=(",", ":")).encode()
        request = urllib.request.Request(
            self.url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "User-Agent": f"cacheeconomics-collector/{__version__}",
            },
        )
        try:
            with self._opener.open(request, timeout=self.timeout_seconds) as response:
                encoded = response.read(1_048_577)
                if len(encoded) > 1_048_576:
                    raise DeliveryError("response_too_large")
                return json.loads(encoded)
        except urllib.error.HTTPError as error:
            raise DeliveryError(f"http_{error.code}") from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise DeliveryError("network_error") from None
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise DeliveryError("invalid_response") from None


class DeliveryError(RuntimeError):
    """Contains only a normalized category, never a token or response body."""


class DurableUploader:
    def __init__(
        self,
        database_path: str | Path,
        transport: Transport,
        *,
        batch_size: int = 100,
        max_batch_bytes: int = 900_000,
        max_event_bytes: int = 131_072,
        max_attempts: int = 8,
        retry_base_seconds: int = 5,
        retry_max_seconds: int = 300,
    ):
        if batch_size < 1 or max_attempts < 1 or max_batch_bytes < 1 or max_event_bytes < 1:
            raise ValueError("batch, event, and attempt limits must be positive")
        if max_event_bytes + len(b'{"events":[]}') > max_batch_bytes:
            raise ValueError("max_event_bytes plus its envelope must fit max_batch_bytes")
        self.database_path = str(database_path)
        if self.database_path == ":memory:":
            raise ValueError("the durable outbox requires a filesystem path")
        self.transport = transport
        self.batch_size = batch_size
        self.max_batch_bytes = max_batch_bytes
        self.max_event_bytes = max_event_bytes
        self.max_attempts = max_attempts
        self.retry_base_seconds = retry_base_seconds
        self.retry_max_seconds = retry_max_seconds
        # LiteLLM may run completion callbacks concurrently. Keep selection,
        # network delivery, and final row mutation in one in-process critical
        # section so the same uploader cannot send or dead-letter a row twice.
        self._flush_lock = threading.Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS pending_events ("
                "event_id TEXT PRIMARY KEY, payload TEXT NOT NULL, attempts INTEGER NOT NULL "
                "DEFAULT 0, next_attempt_at REAL NOT NULL DEFAULT 0, "
                "last_error_type TEXT)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS dead_letters ("
                "event_id TEXT PRIMARY KEY, payload TEXT NOT NULL, reason TEXT NOT NULL, "
                "moved_at REAL NOT NULL)"
            )
        os.chmod(self.database_path, 0o600)

    def enqueue(self, event: dict[str, Any]) -> bool:
        validate_event(event)
        event_id = event.get("event_id")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("event_id is required")
        payload = json.dumps(event, separators=(",", ":"), allow_nan=False)
        if len(payload.encode("utf-8")) > self.max_event_bytes:
            raise ValueError("event exceeds the configured outbox size limit")
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO pending_events (event_id, payload) VALUES (?, ?)",
                (event_id, payload),
            )
            return cursor.rowcount == 1

    def submit(self, event: dict[str, Any]) -> dict[str, int]:
        self.enqueue(event)
        return self.flush()

    def flush(self) -> dict[str, int]:
        with self._flush_lock:
            return self._flush_once()

    def _flush_once(self) -> dict[str, int]:
        now = time.time()
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT event_id, payload, attempts FROM pending_events "
                "WHERE next_attempt_at <= ? ORDER BY rowid LIMIT ?",
                (now, self.batch_size),
            ).fetchall()
        selected = []
        encoded_size = len(b'{"events":[]}')
        for row in rows:
            addition = len(row["payload"].encode("utf-8")) + (1 if selected else 0)
            if encoded_size + addition > self.max_batch_bytes:
                break
            selected.append(row)
            encoded_size += addition
        rows = selected
        if not rows:
            return {"sent": 0, "retried": 0, "dead_lettered": 0}

        events = [json.loads(row["payload"]) for row in rows]
        try:
            response = self.transport.send(events)
        except DeliveryError as error:
            return self._retry(rows, str(error))
        except Exception:
            # A custom transport is allowed for private networks. Its raw error
            # still must not be persisted because it can contain request data.
            return self._retry(rows, "transport_error")

        items = response.get("items") if isinstance(response, dict) else None
        if not isinstance(items, list):
            return self._retry(rows, "invalid_response")
        outcomes = {
            item.get("event_id"): item
            for item in items
            if isinstance(item, dict) and isinstance(item.get("event_id"), str)
        }
        sent = dead = 0
        missing = []
        with self._connect() as connection:
            for row in rows:
                item = outcomes.get(row["event_id"])
                status = item.get("status") if item else None
                if status in {"accepted", "duplicate"}:
                    connection.execute(
                        "DELETE FROM pending_events WHERE event_id = ?", (row["event_id"],)
                    )
                    sent += 1
                elif status == "rejected":
                    reason = str(item.get("reason") or "server_rejected")[:256]
                    self._move_to_dead_letter(connection, row, reason, now)
                    dead += 1
                else:
                    missing.append(row)
        retried = self._retry(missing, "missing_item")["retried"] if missing else 0
        return {"sent": sent, "retried": retried, "dead_lettered": dead}

    def _retry(self, rows, error_type: str) -> dict[str, int]:
        now = time.time()
        retried = dead = 0
        with self._connect() as connection:
            for row in rows:
                attempts = int(row["attempts"]) + 1
                if attempts >= self.max_attempts:
                    self._move_to_dead_letter(
                        connection,
                        row,
                        f"delivery_exhausted:{error_type}"[:256],
                        now,
                    )
                    dead += 1
                    continue
                delay = min(
                    self.retry_max_seconds,
                    self.retry_base_seconds * (2 ** max(0, attempts - 1)),
                )
                connection.execute(
                    "UPDATE pending_events SET attempts = ?, next_attempt_at = ?, "
                    "last_error_type = ? WHERE event_id = ?",
                    (attempts, now + delay, error_type[:128], row["event_id"]),
                )
                retried += 1
        return {"sent": 0, "retried": retried, "dead_lettered": dead}

    @staticmethod
    def _move_to_dead_letter(connection, row, reason: str, now: float) -> None:
        connection.execute(
            "INSERT OR REPLACE INTO dead_letters (event_id, payload, reason, moved_at) "
            "VALUES (?, ?, ?, ?)",
            (row["event_id"], row["payload"], reason, now),
        )
        connection.execute("DELETE FROM pending_events WHERE event_id = ?", (row["event_id"],))

    def counts(self) -> dict[str, int]:
        with self._connect() as connection:
            pending = connection.execute("SELECT count(*) FROM pending_events").fetchone()[0]
            dead = connection.execute("SELECT count(*) FROM dead_letters").fetchone()[0]
        return {"pending": pending, "dead_letters": dead}
