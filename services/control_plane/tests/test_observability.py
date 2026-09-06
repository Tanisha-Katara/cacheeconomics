from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import SecretStr

from cacheeconomics_control_plane.observability import JsonEventFormatter, MetricsRegistry

from test_ingestion import ingest_event, provision_collector


def test_metrics_use_route_templates_and_never_tenant_identifiers(client):
    _, source, _, headers = provision_collector(client)
    client.get(f"/v1/sources/{source['id']}/health", headers=headers)

    response = client.get("/metrics")

    assert response.status_code == 200
    assert "cacheeconomics_http_requests_total" in response.text
    assert 'route="/v1/sources/{source_id}/health"' in response.text
    assert source["id"] not in response.text


def test_metrics_endpoint_supports_a_constant_time_bearer_check(client):
    client.app.state.settings.metrics_bearer_token = SecretStr("metrics-secret")

    denied = client.get("/metrics")
    allowed = client.get(
        "/metrics",
        headers={"Authorization": "Bearer metrics-secret"},
    )

    assert denied.status_code == 401
    assert denied.headers["WWW-Authenticate"] == "Bearer"
    assert allowed.status_code == 200
    assert "metrics-secret" not in allowed.text


def test_ingest_metrics_record_only_normalized_outcomes(client):
    _, _, token, _ = provision_collector(client)
    response = client.post(
        "/v1/ingest/events",
        headers={"Authorization": f"Bearer {token}"},
        json={"events": [ingest_event(), ingest_event(), {"prompt": "private"}]},
    )
    assert response.status_code == 200

    metrics = client.get("/metrics").text

    assert 'cacheeconomics_ingest_events_total{outcome="accepted"} 1' in metrics
    assert 'cacheeconomics_ingest_events_total{outcome="duplicate"} 1' in metrics
    assert 'cacheeconomics_ingest_events_total{outcome="rejected"} 1' in metrics
    assert "private" not in metrics


def test_json_formatter_does_not_interpolate_raw_log_arguments():
    formatter = JsonEventFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg="worker heartbeat failed (%s)",
        args=("private database error",),
        exc_info=None,
    )
    record.event_fields = {
        "status_code": 500,
        "unsafe_object": {"prompt": "private"},
    }

    payload = json.loads(formatter.format(record))

    assert payload["event"] == "worker heartbeat failed (%s)"
    assert payload["status_code"] == 500
    assert "unsafe_object" not in payload
    assert "private" not in json.dumps(payload)


def test_prometheus_histogram_is_cumulative():
    registry = MetricsRegistry()
    registry.request_started()
    registry.request_finished(
        method="GET",
        route="/healthz",
        status_code=200,
        duration_seconds=0.02,
    )

    rendered = registry.render()

    assert 'le="0.01"} 0' in rendered
    assert 'le="0.05"} 1' in rendered
    assert 'le="+Inf"} 1' in rendered


def test_alert_rules_use_only_normalized_service_metrics():
    root = Path(__file__).resolve().parents[3]
    rules = (root / "deploy" / "observability" / "alerts.yml").read_text(
        encoding="utf-8"
    )

    assert "cacheeconomics_http_requests_total" in rules
    assert "cacheeconomics_ingest_events_total" in rules
    assert "cacheeconomics_http_request_duration_seconds_bucket" in rules
    assert "organization_id" not in rules
    assert "source_id" not in rules
    assert "authorization" not in rules.lower()
    assert "fingerprint" not in rules.lower()


def test_manual_trace_spans_do_not_record_exception_text():
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "cacheeconomics_control_plane"
        / "observability.py"
    ).read_text(encoding="utf-8")

    assert "record_exception=False" in source
    assert "set_status_on_exception=False" in source
