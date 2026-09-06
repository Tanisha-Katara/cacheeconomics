from __future__ import annotations

import uuid
import time
from datetime import datetime, timedelta, timezone

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from cacheeconomics_control_plane.db import create_session_factory
from cacheeconomics_control_plane.models import (
    AnalysisResult,
    Base,
    Job,
    Organization,
    Source,
    SourceHealth,
)
from cacheeconomics_control_plane.settings import Settings
import cacheeconomics_control_plane.worker as worker_module
from cacheeconomics_control_plane.worker import _renew_lease, lease_job, run_once

from test_ingestion import ingest_event, provision_collector


def queue_event(client, event_id="event-1"):
    _, _, token, _ = provision_collector(client)
    response = client.post(
        "/v1/ingest/events",
        headers={"Authorization": f"Bearer {token}"},
        json={"events": [ingest_event(event_id)]},
    )
    assert response.status_code == 200, response.text
    return uuid.UUID(response.json()["job_id"])


def test_worker_runs_the_local_engine_and_records_provenance(client, engine):
    job_id = queue_event(client)
    worked = run_once(
        create_session_factory(engine),
        client.app.state.settings,
        worker_id="worker-test",
    )
    assert worked is True

    with Session(engine) as session:
        job = session.get(Job, job_id)
        result = session.scalar(select(AnalysisResult))
        health = session.scalar(select(SourceHealth))
        assert job.state == "succeeded"
        assert job.attempt == 1
        assert result.job_id == job.id
        assert result.event_count == 1
        assert result.result["schema"] == "cacheeconomics.analysis-result"
        assert "this must never be stored" not in str(result.result).lower()
        assert set(result.result) == {
            "schema",
            "schema_version",
            "engine",
            "registry",
            "analysis",
        }
        assert health.last_success_at is not None
        assert health.consecutive_failures == 0


def test_worker_retries_safely_then_dead_letters(client, engine):
    job_id = queue_event(client)
    with Session(engine) as session:
        job = session.get(Job, job_id)
        job.max_attempts = 1
        session.commit()

    def broken_analyzer(*_args, **_kwargs):
        raise RuntimeError("secret provider message that must not be stored")

    assert run_once(
        create_session_factory(engine),
        client.app.state.settings,
        worker_id="worker-test",
        analyzer=broken_analyzer,
    )
    with Session(engine) as session:
        job = session.get(Job, job_id)
        health = session.scalar(select(SourceHealth))
        assert job.state == "dead_letter"
        assert job.error_type == "analysis_runtime_error"
        assert "secret" not in job.error_type
        assert health.last_error_type == "analysis_runtime_error"
        assert health.consecutive_failures == 1
        assert session.scalar(select(func.count()).select_from(AnalysisResult)) == 0


def test_expired_lease_is_recovered(client, engine):
    job_id = queue_event(client)
    with Session(engine) as session:
        job = session.get(Job, job_id)
        job.state = "running"
        job.attempt = 1
        job.leased_by = "gone-worker"
        job.lease_expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        session.commit()

    assert run_once(
        create_session_factory(engine),
        client.app.state.settings,
        worker_id="replacement-worker",
    )
    with Session(engine) as session:
        job = session.get(Job, job_id)
        assert job.state == "succeeded"
        assert job.attempt == 2


def test_worker_can_renew_only_its_live_lease(client, engine):
    job_id = queue_event(client)
    session_factory = create_session_factory(engine)
    lease = lease_job(
        session_factory,
        worker_id="renewing-worker",
        lease_seconds=10,
    )
    assert lease is not None
    with Session(engine) as session:
        original_expiry = session.get(Job, job_id).lease_expires_at

    assert _renew_lease(
        session_factory,
        lease,
        worker_id="renewing-worker",
        lease_seconds=10,
    )
    assert not _renew_lease(
        session_factory,
        lease,
        worker_id="different-worker",
        lease_seconds=10,
    )
    with Session(engine) as session:
        assert session.get(Job, job_id).lease_expires_at >= original_expiry


def test_long_analysis_renews_its_lease_until_completion(
    client, engine, monkeypatch
):
    job_id = queue_event(client)
    renewals = []

    def record_renewal(*_args, **_kwargs):
        renewals.append(time.monotonic())
        return True

    monkeypatch.setattr(worker_module, "_renew_lease", record_renewal)
    client.app.state.settings.job_lease_seconds = 0.15

    def slow_analyzer(_payloads, *, source, **_kwargs):
        deadline = time.monotonic() + 1
        while len(renewals) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        return {"schema": "cacheeconomics.analysis-result", "source": source}

    assert run_once(
        create_session_factory(engine),
        client.app.state.settings,
        worker_id="long-running-worker",
        analyzer=slow_analyzer,
    )
    assert len(renewals) >= 2
    with Session(engine) as session:
        assert session.get(Job, job_id).state == "succeeded"


def test_active_long_job_cannot_be_reclaimed_by_a_competing_worker(tmp_path):
    database_url = f"sqlite+pysqlite:///{tmp_path / 'leases.sqlite3'}"
    database = create_engine(
        database_url,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(database)
    session_factory = create_session_factory(database)
    organization_id = uuid.uuid4()
    source_id = uuid.uuid4()
    job_id = uuid.uuid4()
    with Session(database) as session:
        session.add(
            Organization(id=organization_id, name="Lease test", slug="lease-test")
        )
        session.add(
            Source(
                id=source_id,
                organization_id=organization_id,
                name="Long analysis",
                kind="test",
            )
        )
        session.add(
            Job(
                id=job_id,
                organization_id=organization_id,
                source_id=source_id,
                kind="analyze_source",
                state="queued",
                attempt=0,
                max_attempts=3,
                payload={},
                scheduled_for=datetime.now(timezone.utc),
            )
        )
        session.commit()

    settings = Settings(_env_file=None, environment="test", database_url=database_url)
    settings.job_lease_seconds = 0.3
    competing_leases = []

    def slow_analyzer(_payloads, *, source, **_kwargs):
        time.sleep(0.45)
        competing_leases.append(
            lease_job(
                session_factory,
                worker_id="competing-worker",
                lease_seconds=0.3,
            )
        )
        return {"schema": "cacheeconomics.analysis-result", "source": source}

    try:
        assert run_once(
            session_factory,
            settings,
            worker_id="active-worker",
            analyzer=slow_analyzer,
        )
        assert competing_leases == [None]
        with Session(database) as session:
            job = session.get(Job, job_id)
            assert job.state == "succeeded"
            assert job.attempt == 1
    finally:
        database.dispose()


def test_cancellation_wins_over_late_worker_result(client, engine):
    job_id = queue_event(client)

    def cancel_during_analysis(_payloads, *, source, **_kwargs):
        with Session(engine) as session:
            job = session.get(Job, job_id)
            job.state = "cancelled"
            job.finished_at = datetime.now(timezone.utc)
            job.leased_by = None
            job.lease_expires_at = None
            session.commit()
        return {"schema": "cacheeconomics.analysis-result", "source": source}

    assert run_once(
        create_session_factory(engine),
        client.app.state.settings,
        worker_id="worker-test",
        analyzer=cancel_during_analysis,
    )
    with Session(engine) as session:
        assert session.get(Job, job_id).state == "cancelled"
        assert session.scalar(select(func.count()).select_from(AnalysisResult)) == 0


def test_worker_bounds_history_and_says_when_it_was_truncated(client, engine):
    _, _, token, _ = provision_collector(client)
    response = client.post(
        "/v1/ingest/events",
        headers={"Authorization": f"Bearer {token}"},
        json={"events": [ingest_event("older"), ingest_event("newer")]},
    )
    assert response.status_code == 200
    client.app.state.settings.analysis_max_events = 1

    assert run_once(
        create_session_factory(engine),
        client.app.state.settings,
        worker_id="bounded-worker",
    )
    with Session(engine) as session:
        result = session.scalar(select(AnalysisResult))
        assert result.event_count == 1
        assert any(
            "outside this worker's configured analysis window" in note
            for note in result.result["analysis"]["notes"]
        )
