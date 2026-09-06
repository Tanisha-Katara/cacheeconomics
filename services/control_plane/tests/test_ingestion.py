from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from cacheeconomics_control_plane.models import IngestEvent, Job

from conftest import create_organization, user_headers


def provision_collector(client, *, slug="alpha-team"):
    organization = create_organization(client, "alice-token", slug)
    headers = user_headers("alice-token", organization["id"])
    source_response = client.post(
        "/v1/sources",
        headers=headers,
        json={"name": "Gateway", "kind": "litellm"},
    )
    assert source_response.status_code == 201, source_response.text
    source = source_response.json()
    credential_response = client.post(
        f"/v1/sources/{source['id']}/credentials",
        headers=headers,
        json={"name": "Test collector"},
    )
    assert credential_response.status_code == 201, credential_response.text
    return organization, source, credential_response.json()["token"], headers


def ingest_event(event_id="event-1", **overrides):
    event = {
        "schema": "cacheeconomics.ingest-event",
        "schema_version": 1,
        "event_id": event_id,
        "request_id": f"request-{event_id}",
        "sent_at": datetime.now(timezone.utc).isoformat(),
        "first_token_at": None,
        "completed_at": None,
        "model": "claude-opus-5",
        "target_id": "anthropic/direct",
        "agent": "test-agent",
        "session": "session-1",
        "workload_tenant": "tenant-a",
        "ttl_requested": "5m",
        "tokens_counted": True,
        "status": {"outcome": "success", "code": 200, "error_type": None},
        "usage": {
            "input_tokens": 1_000,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "output_tokens": 50,
        },
        "segments": [
            {
                "id": "hmac:" + "a" * 64,
                "role": "system",
                "label": "policy",
                "tokens": 1_000,
                "cache_marked": True,
                "index": 0,
                "ttl": "5m",
            }
        ],
        "collector": {
            "name": "cacheeconomics-litellm",
            "version": "0.1.0",
            "source_type": "litellm",
        },
    }
    event.update(overrides)
    return event


def test_ingest_is_prompt_free_idempotent_and_queues_analysis(client, engine):
    _, source, token, human_headers = provision_collector(client)
    good = ingest_event()
    invalid = ingest_event("event-secret", prompt="this must never be stored")

    response = client.post(
        "/v1/ingest/events",
        headers={"Authorization": f"Bearer {token}"},
        json={"events": [good, good, invalid]},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["accepted"], body["duplicates"], body["rejected"]) == (1, 1, 1)
    assert body["job_id"] is not None
    assert "this must never be stored" not in response.text

    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(IngestEvent)) == 1
        assert session.scalar(select(func.count()).select_from(Job)) == 1
        stored = session.scalar(select(IngestEvent))
        assert "prompt" not in stored.payload

    health = client.get(f"/v1/sources/{source['id']}/health", headers=human_headers)
    assert health.status_code == 200
    assert health.json()["accepted_events"] == 1
    assert health.json()["duplicate_events"] == 1
    assert health.json()["rejected_events"] == 1


def test_event_cannot_choose_its_organization_or_source(client):
    _, _, token, _ = provision_collector(client)
    event = ingest_event(organization_id="00000000-0000-0000-0000-000000000000")
    response = client.post(
        "/v1/ingest/events",
        headers={"Authorization": f"Bearer {token}"},
        json={"events": [event]},
    )
    assert response.status_code == 200
    assert response.json()["rejected"] == 1
    assert response.json()["accepted"] == 0


def test_api_enforces_fields_required_by_the_public_schema(client):
    _, _, token, _ = provision_collector(client)
    event = ingest_event()
    del event["status"]["error_type"]
    del event["segments"][0]["ttl"]
    response = client.post(
        "/v1/ingest/events",
        headers={"Authorization": f"Bearer {token}"},
        json={"events": [event]},
    )
    assert response.status_code == 200
    assert response.json()["rejected"] == 1


def test_raw_error_messages_are_not_valid_error_categories(client):
    _, _, token, _ = provision_collector(client)
    event = ingest_event(
        status={
            "outcome": "error",
            "code": 500,
            "error_type": "customer secret appeared in an exception",
        }
    )
    response = client.post(
        "/v1/ingest/events",
        headers={"Authorization": f"Bearer {token}"},
        json={"events": [event]},
    )
    assert response.status_code == 200
    assert response.json()["rejected"] == 1
    assert "customer secret" not in response.text


def test_cancelled_and_unknown_events_cannot_claim_a_success_status(client):
    _, _, token, _ = provision_collector(client)
    events = [
        ingest_event(
            outcome,
            status={"outcome": outcome, "code": 200, "error_type": None},
        )
        for outcome in ("cancelled", "unknown")
    ]
    response = client.post(
        "/v1/ingest/events",
        headers={"Authorization": f"Bearer {token}"},
        json={"events": events},
    )

    assert response.status_code == 200
    assert response.json()["accepted"] == 0
    assert response.json()["rejected"] == 2


def test_batch_limit_is_enforced_before_work(client):
    _, _, token, _ = provision_collector(client)
    client.app.state.settings.ingest_max_batch_events = 1
    response = client.post(
        "/v1/ingest/events",
        headers={"Authorization": f"Bearer {token}"},
        json={"events": [ingest_event("one"), ingest_event("two")]},
    )
    assert response.status_code == 422
    assert "cannot exceed 1" in response.json()["detail"]


def test_ingest_body_size_is_bounded_before_authentication(client):
    oversized = b'{"events":["' + (b"x" * 1_100_000) + b'"]}'
    response = client.post(
        "/v1/ingest/events",
        content=oversized,
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 413


def test_json_integer_over_the_decoder_limit_is_a_client_error(client):
    oversized_integer = b"9" * 5_000
    response = client.post(
        "/v1/ingest/events",
        content=b'{"events":[' + oversized_integer + b"]}",
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "body must be valid JSON"}


def test_individual_event_size_is_bounded_with_a_safe_rejection(client):
    _, _, token, _ = provision_collector(client)
    client.app.state.settings.ingest_max_event_bytes = 1_024
    event = ingest_event(agent="x" * 2_000)
    response = client.post(
        "/v1/ingest/events",
        headers={"Authorization": f"Bearer {token}"},
        json={"events": [event]},
    )
    assert response.status_code == 200
    assert response.json()["rejected"] == 1
    assert response.json()["items"][0]["reason"].endswith("event_too_large_or_nonfinite")


def test_job_management_respects_roles(client):
    _, source, _, owner_headers = provision_collector(client)
    bob = client.get("/v1/me", headers=user_headers("bob-token")).json()["user"]
    created = client.post(
        "/v1/memberships",
        headers=owner_headers,
        json={"user_id": bob["id"], "role": "viewer"},
    )
    assert created.status_code == 201
    organization_id = owner_headers["X-Organization-ID"]
    viewer_headers = user_headers("bob-token", organization_id)

    denied = client.post(
        f"/v1/sources/{source['id']}/jobs",
        headers=viewer_headers,
        json={},
    )
    assert denied.status_code == 403
    queued = client.post(
        f"/v1/sources/{source['id']}/jobs",
        headers=owner_headers,
        json={},
    )
    assert queued.status_code == 201
    assert queued.json()["state"] == "queued"
    cancelled = client.post(
        f"/v1/jobs/{queued.json()['id']}/cancel",
        headers=owner_headers,
    )
    assert cancelled.status_code == 200
    assert cancelled.json()["state"] == "cancelled"
