from __future__ import annotations

from datetime import datetime, timezone

from cacheeconomics.trace import Request
from cacheeconomics_collector.client import DurableUploader
from cacheeconomics_collector.events import event_from_request
from cacheeconomics_control_plane.db import create_session_factory
from cacheeconomics_control_plane.worker import run_once

from test_ingestion import provision_collector


class ApiTransport:
    def __init__(self, client, token):
        self.client = client
        self.token = token
        self.sent_bodies = []

    def send(self, events):
        body = {"events": events}
        self.sent_bodies.append(body)
        response = self.client.post(
            "/v1/ingest/events",
            headers={"Authorization": f"Bearer {self.token}"},
            json=body,
        )
        assert response.status_code == 200, response.text
        return response.json()


def test_collector_to_api_to_worker_to_analysis(client, engine, tmp_path):
    _, source, token, human_headers = provision_collector(client)
    transport = ApiTransport(client, token)
    uploader = DurableUploader(tmp_path / "outbox.sqlite3", transport)
    request = Request(
        request_id="end-to-end-request",
        sent_at=datetime.now(timezone.utc),
        model="claude-opus-5",
        target_id="anthropic/direct",
        usage={
            "input_tokens": 1_000,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "output_tokens": 20,
        },
    )
    event = event_from_request(request, source_type="litellm")

    assert uploader.submit(event)["sent"] == 1
    serialized_body = str(transport.sent_bodies[0]).lower()
    assert "messages" not in serialized_body
    assert "request_body" not in serialized_body
    assert uploader.counts()["pending"] == 0

    assert run_once(
        create_session_factory(engine),
        client.app.state.settings,
        worker_id="end-to-end-worker",
    )
    response = client.get(
        f"/v1/sources/{source['id']}/analyses/latest",
        headers=human_headers,
    )
    assert response.status_code == 200, response.text
    assert response.json()["event_count"] == 1
    assert response.json()["result"]["schema"] == "cacheeconomics.analysis-result"
