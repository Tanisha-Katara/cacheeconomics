from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from test_ingestion import ingest_event, provision_collector


def _timed_event(event_id, sent_at, *, outcome="success", error_type=None):
    code = 200 if outcome == "success" else 500
    return ingest_event(
        event_id,
        sent_at=sent_at.isoformat(),
        first_token_at=(sent_at + timedelta(milliseconds=100)).isoformat(),
        completed_at=(
            sent_at + timedelta(milliseconds=500 if outcome == "success" else 1_000)
        ).isoformat(),
        status={"outcome": outcome, "code": code, "error_type": error_type},
    )


def test_dashboard_config_is_public_but_contains_no_secret(client):
    response = client.get("/v1/dashboard/config")

    assert response.status_code == 200
    assert response.json() == {
        "oidc_enabled": False,
        "authorization_endpoint": None,
        "token_endpoint": None,
        "client_id": None,
        "audience": None,
        "scopes": "openid profile email",
        "redirect_uri": None,
        "allow_development_token": False,
    }


def test_dashboard_config_exposes_the_api_audience_needed_for_oidc(client):
    settings = client.app.state.settings
    settings.dashboard_oidc_authorization_endpoint = "https://identity.example.test/authorize"
    settings.dashboard_oidc_token_endpoint = "https://identity.example.test/oauth/token"
    settings.dashboard_oidc_client_id = "public-dashboard"
    settings.dashboard_redirect_uri = "https://dashboard.example.test/"
    settings.oidc_audience = "https://cacheeconomics.example.test/api"

    response = client.get("/v1/dashboard/config")

    assert response.status_code == 200
    assert response.json()["oidc_enabled"] is True
    assert response.json()["audience"] == "https://cacheeconomics.example.test/api"


def test_source_operations_reports_bounded_prompt_free_aggregates(client):
    _, source, token, headers = provision_collector(client)
    now = datetime.now(timezone.utc) - timedelta(minutes=1)
    events = [
        _timed_event("successful", now),
        _timed_event(
            "failed",
            now + timedelta(seconds=1),
            outcome="error",
            error_type="timeout",
        ),
    ]
    ingested = client.post(
        "/v1/ingest/events",
        headers={"Authorization": f"Bearer {token}"},
        json={"events": events},
    )
    assert ingested.status_code == 200
    assert ingested.json()["accepted"] == 2

    response = client.get(
        f"/v1/sources/{source['id']}/operations?window_hours=24",
        headers=headers,
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["window_hours"] == 24
    assert body["bucket_minutes"] == 60
    assert body["events_examined"] == 2
    assert body["valid_events"] == 2
    assert body["invalid_events"] == 0
    assert body["truncated"] is False
    assert body["status"] == {
        "success": 1,
        "error": 1,
        "cancelled": 0,
        "unknown": 0,
    }
    assert body["latency"]["count"] == 2
    assert body["latency"]["p50_ms"] == pytest.approx(750)
    assert body["time_to_first_token"]["p95_ms"] == pytest.approx(100)
    assert body["errors"] == [{"error_type": "timeout", "count": 1}]
    assert sum(bucket["requests"] for bucket in body["buckets"]) == 2
    assert "messages" not in response.text
    assert "prompt" not in response.text


def test_source_operations_is_tenant_scoped_and_window_is_bounded(client):
    _, source_a, _, headers_a = provision_collector(client, slug="operations-a")
    _, source_b, _, _ = provision_collector(client, slug="operations-b")

    hidden = client.get(
        f"/v1/sources/{source_b['id']}/operations",
        headers=headers_a,
    )
    invalid_window = client.get(
        f"/v1/sources/{source_a['id']}/operations?window_hours=721",
        headers=headers_a,
    )

    assert hidden.status_code == 404
    assert invalid_window.status_code == 422


def test_source_operations_marks_a_truncated_sample(client):
    _, source, token, headers = provision_collector(client)
    client.app.state.settings.dashboard_max_events = 1
    response = client.post(
        "/v1/ingest/events",
        headers={"Authorization": f"Bearer {token}"},
        json={"events": [ingest_event("one"), ingest_event("two")]},
    )
    assert response.status_code == 200

    operations = client.get(
        f"/v1/sources/{source['id']}/operations",
        headers=headers,
    ).json()

    assert operations["truncated"] is True
    assert operations["events_examined"] == 1
