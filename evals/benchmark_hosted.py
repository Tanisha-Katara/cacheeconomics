#!/usr/bin/env python3
"""Measure repeatable local regression signals for the hosted application.

This is not a load test and its results are not production capacity claims. It
uses an in-memory SQLite database, synthetic prompt-free events, and local
dashboard assets so a slower release can be detected before deployment.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import platform
import statistics
import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from cacheeconomics.analyzer import analyze
from cacheeconomics.trace import load_jsonl
from cacheeconomics_control_plane.app import create_app
from cacheeconomics_control_plane.models import Base
from cacheeconomics_control_plane.security import AuthenticationError, IdentityClaims
from cacheeconomics_control_plane.settings import Settings


ROOT = Path(__file__).resolve().parents[1]


class _SyntheticVerifier:
    def verify(self, token: str) -> IdentityClaims:
        if token != "synthetic-human-token":
            raise AuthenticationError("unknown synthetic token")
        return IdentityClaims(
            issuer="https://synthetic.invalid/",
            subject="benchmark-user",
            email="benchmark@example.invalid",
            display_name="Synthetic benchmark user",
        )


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile)))
    return ordered[index]


def _event(template: dict, number: int) -> dict:
    event = copy.deepcopy(template)
    event["event_id"] = f"synthetic-benchmark-{number}"
    event["request_id"] = f"synthetic-request-{number}"
    return event


def _ingest_measurement(template: dict, *, samples: int, batch_size: int) -> dict:
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
        oidc_issuer="https://synthetic.invalid/",
        oidc_audience="cacheeconomics-benchmark",
        oidc_jwks_url="https://synthetic.invalid/jwks",
        allowed_hosts=["testserver"],
        ingest_max_batch_events=max(100, batch_size),
    )
    app = create_app(settings, engine=engine, token_verifier=_SyntheticVerifier())
    human = {"Authorization": "Bearer synthetic-human-token"}
    elapsed: list[float] = []
    accepted = 0
    sequence = 0
    try:
        with TestClient(app) as client:
            organization = client.post(
                "/v1/organizations",
                headers=human,
                json={"name": "Synthetic Benchmark", "slug": "synthetic-benchmark"},
            )
            organization.raise_for_status()
            org_headers = human | {"X-Organization-ID": organization.json()["id"]}
            source = client.post(
                "/v1/sources",
                headers=org_headers,
                json={"name": "Synthetic source", "kind": "normalized_trace"},
            )
            source.raise_for_status()
            credential = client.post(
                f"/v1/sources/{source.json()['id']}/credentials",
                headers=org_headers,
                json={"name": "Synthetic benchmark credential"},
            )
            credential.raise_for_status()
            ingest_headers = {
                "Authorization": f"Bearer {credential.json()['token']}"
            }

            # Warm the routing, model validation, and database code without
            # including it in the recorded samples.
            warmup = [_event(template, sequence + index) for index in range(10)]
            sequence += len(warmup)
            response = client.post(
                "/v1/ingest/events",
                headers=ingest_headers,
                json={"events": warmup},
            )
            response.raise_for_status()

            for _ in range(samples):
                batch = [
                    _event(template, sequence + index) for index in range(batch_size)
                ]
                sequence += len(batch)
                started = time.perf_counter()
                response = client.post(
                    "/v1/ingest/events",
                    headers=ingest_headers,
                    json={"events": batch},
                )
                duration = time.perf_counter() - started
                response.raise_for_status()
                body = response.json()
                if body["accepted"] != batch_size or body["rejected"]:
                    raise RuntimeError("synthetic ingestion did not accept the full batch")
                elapsed.append(duration)
                accepted += body["accepted"]
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()

    rates = [batch_size / duration for duration in elapsed]
    return {
        "database": "in-memory SQLite (not production PostgreSQL)",
        "samples": samples,
        "batch_size": batch_size,
        "accepted_events": accepted,
        "median_events_per_second": statistics.median(rates),
        "p95_batch_latency_ms": _percentile(elapsed, 0.95) * 1000,
    }


def _analysis_measurement(*, samples: int) -> dict:
    fixture = ROOT / "harness/fixtures/demo-traces.jsonl"
    trace = load_jsonl(str(fixture))
    elapsed: list[float] = []
    analyze(trace, invoice_usd=17.45)
    for _ in range(samples):
        started = time.perf_counter()
        result = analyze(trace, invoice_usd=17.45)
        elapsed.append(time.perf_counter() - started)
        if len(result.findings) != 4:
            raise RuntimeError("golden analyzer workload changed")
    return {
        "fixture": "harness/fixtures/demo-traces.jsonl",
        "request_rows": len(trace.requests),
        "samples": samples,
        "median_latency_ms": statistics.median(elapsed) * 1000,
        "p95_latency_ms": _percentile(elapsed, 0.95) * 1000,
    }


def _dashboard_measurement() -> dict:
    paths = [
        ROOT / "apps/dashboard/index.html",
        ROOT / "apps/dashboard/styles.css",
        ROOT / "apps/dashboard/app.js",
    ]
    return {
        "assets": [str(path.relative_to(ROOT)) for path in paths],
        "raw_bytes": sum(path.stat().st_size for path in paths),
        "gzip_bytes": sum(len(gzip.compress(path.read_bytes(), mtime=0)) for path in paths),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args(argv)
    if args.samples < 3 or args.batch_size < 1:
        parser.error("use at least three samples and a positive batch size")

    template = json.loads(
        (ROOT / "evals/fixtures/ingest-event-v1.valid.json").read_text()
    )
    result = {
        "schema": "cacheeconomics.synthetic-local-benchmark",
        "schema_version": 1,
        "claim_scope": "local regression signal only; not production capacity",
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "ingestion": _ingest_measurement(
            template,
            samples=args.samples,
            batch_size=args.batch_size,
        ),
        "analysis": _analysis_measurement(samples=args.samples),
        "dashboard": _dashboard_measurement(),
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded)
    else:
        sys.stdout.write(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
