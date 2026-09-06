#!/usr/bin/env python3
"""Build the deterministic, prompt-free packet used by the portfolio demo.

The builder drives the real collector normalization, API, SQLite test database,
worker, and dashboard endpoints. Random identifiers and database-clock fields
are normalized afterwards so a checked-in packet can be reviewed and compared
byte for byte. It never starts a network listener.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from cacheeconomics.trace import load_jsonl
from cacheeconomics_collector.events import event_from_request
from cacheeconomics_control_plane import api as api_module
from cacheeconomics_control_plane import worker as worker_module
from cacheeconomics_control_plane.app import create_app
from cacheeconomics_control_plane.db import create_session_factory
from cacheeconomics_control_plane.models import Base, IngestEvent
from cacheeconomics_control_plane.security import AuthenticationError, IdentityClaims
from cacheeconomics_control_plane.settings import Settings


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "demo/fixtures/portfolio-demo-v1.json"
TRACE_PATH = ROOT / "harness/fixtures/demo-traces.jsonl"
GOLDEN_PATH = ROOT / "evals/fixtures/golden-analysis-v1.json"
DEMO_NOW = datetime(2026, 9, 5, 12, 0, tzinfo=timezone.utc)
DEMO_TOKEN = "synthetic-demo-only"

DEMO_USER_ID = "00000000-0000-0000-0000-000000000501"
DEMO_ORGANIZATION_ID = "00000000-0000-0000-0000-000000000502"
DEMO_SOURCE_ID = "00000000-0000-0000-0000-000000000503"
DEMO_JOB_ID = "00000000-0000-0000-0000-000000000504"
DEMO_ANALYSIS_ID = "00000000-0000-0000-0000-000000000505"


class _SyntheticVerifier:
    def verify(self, token: str) -> IdentityClaims:
        if token != DEMO_TOKEN:
            raise AuthenticationError("unknown synthetic demo token")
        return IdentityClaims(
            issuer="https://synthetic-demo.invalid/",
            subject="portfolio-demo-user",
            email="demo@example.invalid",
            display_name="Synthetic Demo User",
        )


class _FrozenDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        value = DEMO_NOW
        return value if tz is not None else value.replace(tzinfo=None)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _replace_identifiers(value: Any, identifiers: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {
            key: _replace_identifiers(item, identifiers)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_replace_identifiers(item, identifiers) for item in value]
    if isinstance(value, str):
        return identifiers.get(value, value)
    return value


def _headers(organization_id: str | None = None) -> dict[str, str]:
    result = {"Authorization": f"Bearer {DEMO_TOKEN}"}
    if organization_id is not None:
        result["X-Organization-ID"] = organization_id
    return result


def _assert_response(response, expected: int = 200) -> dict[str, Any]:
    if response.status_code != expected:
        raise RuntimeError(
            f"demo flow returned HTTP {response.status_code}: {response.text}"
        )
    return response.json()


def _assert_withheld(result: dict[str, Any]) -> None:
    analysis = result["analysis"]
    figures = list(analysis["spend"].values())
    figures.append(analysis["total_avoidable_usd_month"])
    for finding in analysis["findings"]:
        figures.extend(
            figure
            for figure in (
                finding["avoidable_usd_window"],
                finding["avoidable_usd_month"],
            )
            if figure is not None
        )
    monetary = [
        figure
        for figure in figures
        if isinstance(figure, dict) and "released" in figure
    ]
    if not monetary:
        raise RuntimeError("the demo analysis returned no monetary figures")
    if any(figure["released"] or figure["amount_usd"] is not None for figure in monetary):
        raise RuntimeError("an unreconciled demo figure crossed the release gate")


def build_packet() -> dict[str, Any]:
    trace = load_jsonl(str(TRACE_PATH))
    latest_sent = max(request.sent_at for request in trace.requests)
    shift = (DEMO_NOW - timedelta(hours=1)) - latest_sent
    events = []
    for index, request in enumerate(trace.requests):
        request.sent_at += shift
        request.first_token_at = request.sent_at + timedelta(
            milliseconds=180 + ((index % 9) * 20)
        )
        completed_at = request.sent_at + timedelta(
            milliseconds=900 + ((index % 13) * 80)
        )
        events.append(
            event_from_request(
                request,
                source_type="normalized_trace",
                tokens_counted=True,
                completed_at=completed_at,
            )
        )

    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    settings = Settings(
        _env_file=None,
        environment="test",
        database_url="sqlite+pysqlite://",
        oidc_issuer="https://synthetic-demo.invalid/",
        oidc_audience="cacheeconomics-demo",
        oidc_jwks_url="https://synthetic-demo.invalid/jwks",
        allowed_hosts=["testserver"],
        ingest_max_body_bytes=4_194_304,
        ingest_max_batch_events=500,
        dashboard_allow_development_token=True,
    )
    application = create_app(
        settings,
        engine=engine,
        token_verifier=_SyntheticVerifier(),
    )

    original_api_datetime = api_module.datetime
    original_worker_now = worker_module._utc_now
    previous_log_disable = logging.root.manager.disable
    api_module.datetime = _FrozenDateTime
    worker_module._utc_now = lambda: DEMO_NOW
    logging.disable(logging.CRITICAL)
    try:
        with TestClient(application) as client:
            organization = _assert_response(
                client.post(
                    "/v1/organizations",
                    headers=_headers(),
                    json={
                        "name": "Synthetic Demo Organization",
                        "slug": "synthetic-demo-organization",
                    },
                ),
                201,
            )
            organization_headers = _headers(organization["id"])
            source = _assert_response(
                client.post(
                    "/v1/sources",
                    headers=organization_headers,
                    json={
                        "name": "Synthetic Normalized Trace",
                        "kind": "normalized_trace",
                    },
                ),
                201,
            )
            credential = _assert_response(
                client.post(
                    f"/v1/sources/{source['id']}/credentials",
                    headers=organization_headers,
                    json={"name": "Synthetic one-run credential"},
                ),
                201,
            )
            collector_headers = {
                "Authorization": f"Bearer {credential['token']}"
            }
            ingest = _assert_response(
                client.post(
                    "/v1/ingest/events",
                    headers=collector_headers,
                    json={"events": events},
                )
            )
            if ingest["accepted"] != len(events) or ingest["rejected"]:
                raise RuntimeError("the synthetic event batch was not fully accepted")

            invalid = dict(events[0])
            invalid["event_id"] = "synthetic-invalid-event"
            invalid["prompt"] = "this value must never be persisted"
            replay = _assert_response(
                client.post(
                    "/v1/ingest/events",
                    headers=collector_headers,
                    json={"events": [events[0], invalid]},
                )
            )
            if (replay["duplicates"], replay["rejected"], replay["accepted"]) != (
                1,
                1,
                0,
            ):
                raise RuntimeError("duplicate/rejection evidence changed")

            # Distribute API receipt times across the trace window so the real
            # operations endpoint produces a useful chart. These are synthetic
            # scenario inputs, not measured service performance.
            with Session(engine) as session:
                for row in session.scalars(select(IngestEvent)):
                    sent_at = datetime.fromisoformat(
                        row.payload["sent_at"].replace("Z", "+00:00")
                    )
                    row.received_at = sent_at + timedelta(seconds=2)
                session.commit()

            session_factory = create_session_factory(engine)
            if not worker_module.run_once(
                session_factory,
                settings,
                worker_id="synthetic-portfolio-worker",
            ):
                raise RuntimeError("the synthetic analysis job was not leased")
            if worker_module.run_once(
                session_factory,
                settings,
                worker_id="synthetic-portfolio-worker",
            ):
                raise RuntimeError("the demo created an unexpected second job")

            me = _assert_response(client.get("/v1/me", headers=_headers()))
            sources = _assert_response(
                client.get("/v1/sources", headers=organization_headers)
            )
            jobs = _assert_response(client.get("/v1/jobs", headers=organization_headers))
            health = _assert_response(
                client.get(
                    f"/v1/sources/{source['id']}/health",
                    headers=organization_headers,
                )
            )
            analysis = _assert_response(
                client.get(
                    f"/v1/sources/{source['id']}/analyses/latest",
                    headers=organization_headers,
                )
            )
            operations = {
                str(hours): _assert_response(
                    client.get(
                        f"/v1/sources/{source['id']}/operations?window_hours={hours}",
                        headers=organization_headers,
                    )
                )
                for hours in (24, 168, 720)
            }
    finally:
        api_module.datetime = original_api_datetime
        worker_module._utc_now = original_worker_now
        Base.metadata.drop_all(engine)
        engine.dispose()
        logging.disable(previous_log_disable)

    _assert_withheld(analysis["result"])
    golden = json.loads(GOLDEN_PATH.read_text())
    expected = golden["expected"]
    actual_body = analysis["result"]["analysis"]
    actual_codes = [finding["code"] for finding in actual_body["findings"]]
    expected_codes = [finding["code"] for finding in expected["findings"]]
    if actual_codes != expected_codes:
        raise RuntimeError("the portfolio finding order drifted from the golden result")
    if actual_body["ratios"] != expected["ratios"]:
        raise RuntimeError("the portfolio cache ratios drifted from the golden result")

    identifiers = {
        me["user"]["id"]: DEMO_USER_ID,
        organization["id"]: DEMO_ORGANIZATION_ID,
        source["id"]: DEMO_SOURCE_ID,
        ingest["job_id"]: DEMO_JOB_ID,
        analysis["id"]: DEMO_ANALYSIS_ID,
    }
    responses = _replace_identifiers(
        {
            "config": {
                "oidc_enabled": False,
                "authorization_endpoint": None,
                "token_endpoint": None,
                "client_id": None,
                "audience": None,
                "scopes": "openid profile email",
                "redirect_uri": None,
                "allow_development_token": True,
            },
            "me": me,
            "sources": sources,
            "jobs": jobs,
            "health": health,
            "analysis": analysis,
            "operations": operations,
        },
        identifiers,
    )

    responses["sources"][0]["created_at"] = "2026-09-05T10:00:00Z"
    responses["health"].update(
        last_received_at="2026-09-05T11:00:02Z",
        last_event_sent_at="2026-09-05T11:00:00Z",
        last_ingest_lag_seconds=2.0,
        last_success_at="2026-09-05T12:00:00Z",
        last_job_duration_ms=None,
    )
    responses["analysis"]["created_at"] = "2026-09-05T12:00:00Z"
    job = responses["jobs"][0]
    job.update(
        created_at="2026-09-05T11:59:55Z",
        scheduled_for="2026-09-05T12:00:00Z",
        started_at="2026-09-05T12:00:01Z",
        finished_at="2026-09-05T12:00:02Z",
    )

    return {
        "schema": "cacheeconomics.portfolio-demo-packet",
        "schema_version": 1,
        "generated_at": "2026-09-05T12:00:00Z",
        "claim_scope": (
            "Synthetic local demonstration only; not production traffic, "
            "capacity, availability, or realized savings."
        ),
        "provenance": {
            "builder": "demo/build_portfolio_packet.py",
            "trace": {
                "path": str(TRACE_PATH.relative_to(ROOT)),
                "sha256": _sha256(TRACE_PATH),
                "rows": len(trace.requests),
            },
            "golden_result": {
                "path": str(GOLDEN_PATH.relative_to(ROOT)),
                "sha256": _sha256(GOLDEN_PATH),
            },
            "flow": [
                "collector normalization",
                "prompt-free API validation",
                "idempotent database ingestion",
                "leased analysis worker",
                "versioned analysis result",
                "bounded dashboard aggregates",
            ],
            "normalization": (
                "Random IDs, database timestamps, and job runtime are replaced "
                "with stable demo values after the real in-process flow runs."
            ),
        },
        "demo_access": {
            "token": DEMO_TOKEN,
            "organization_id": DEMO_ORGANIZATION_ID,
            "source_id": DEMO_SOURCE_ID,
            "read_only": True,
        },
        "checks": {
            "accepted_events": ingest["accepted"],
            "duplicate_events": replay["duplicates"],
            "rejected_events": replay["rejected"],
            "stored_prompt_fields": 0,
            "analysis_event_count": analysis["event_count"],
            "finding_codes": actual_codes,
            "all_monetary_figures_withheld": True,
        },
        "responses": responses,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build or verify the deterministic synthetic portfolio packet"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    encoded = json.dumps(build_packet(), indent=2, sort_keys=True) + "\n"
    if args.check:
        if not args.output.exists() or args.output.read_text() != encoded:
            raise SystemExit(
                f"{args.output} is stale; rebuild it with this script"
            )
        print(f"verified {args.output.relative_to(ROOT)}")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
