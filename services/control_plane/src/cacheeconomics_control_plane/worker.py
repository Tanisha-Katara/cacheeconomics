"""PostgreSQL-backed analysis worker with bounded retries and durable leases."""

from __future__ import annotations

import argparse
import logging
import re
import socket
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

from sqlalchemy import and_, or_, select, text, update
from sqlalchemy.orm import Session, sessionmaker

from .analysis_engine import analyze_event_payloads
from .db import create_database_engine, create_session_factory, set_identity_context
from .models import AnalysisResult, IngestEvent, Job, Source, SourceHealth
from .observability import event_logger, log_event, trace_span
from .settings import Settings


_log = logging.getLogger("cacheeconomics.control_plane.worker")


@dataclass(frozen=True)
class Lease:
    job_id: uuid.UUID
    organization_id: uuid.UUID
    source_id: uuid.UUID
    attempt: int


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _health(session: Session, lease: Lease) -> SourceHealth:
    health = session.scalar(
        select(SourceHealth)
        .where(
            SourceHealth.source_id == lease.source_id,
            SourceHealth.organization_id == lease.organization_id,
        )
        .with_for_update()
    )
    if health is None:
        health = SourceHealth(
            source_id=lease.source_id,
            organization_id=lease.organization_id,
            accepted_events=0,
            duplicate_events=0,
            rejected_events=0,
            consecutive_failures=0,
        )
        session.add(health)
    return health


def _lease_postgres(
    session: Session,
    *,
    worker_id: str,
    lease_seconds: int,
) -> Optional[Lease]:
    row = session.execute(
        text(
            "SELECT job_id, organization_id, source_id, attempt "
            "FROM lease_analysis_job(:worker_id, :lease_seconds)"
        ),
        {"worker_id": worker_id, "lease_seconds": lease_seconds},
    ).mappings().one_or_none()
    if row is None:
        return None
    return Lease(
        job_id=row["job_id"],
        organization_id=row["organization_id"],
        source_id=row["source_id"],
        attempt=row["attempt"],
    )


def _lease_sqlite(
    session: Session,
    *,
    worker_id: str,
    lease_seconds: int,
) -> Optional[Lease]:
    """Single-process fallback used for local development and unit tests."""

    now = _utc_now()
    expired = list(
        session.scalars(
            select(Job).where(
                Job.state == "running",
                Job.lease_expires_at < now,
                Job.attempt >= Job.max_attempts,
            )
        )
    )
    for job in expired:
        job.state = "dead_letter"
        job.finished_at = now
        job.error_type = "lease_expired"
        job.lease_expires_at = None
        job.leased_by = None
        health = _health(
            session,
            Lease(job.id, job.organization_id, job.source_id, job.attempt),
        )
        health.last_error_at = now
        health.last_error_type = "lease_expired"
        health.consecutive_failures += 1

    job = session.scalar(
        select(Job)
        .where(
            Job.kind == "analyze_source",
            Job.attempt < Job.max_attempts,
            or_(
                and_(Job.state == "queued", Job.scheduled_for <= now),
                and_(Job.state == "running", Job.lease_expires_at < now),
            ),
        )
        .order_by(Job.scheduled_for, Job.id)
        .limit(1)
    )
    if job is None:
        return None
    job.state = "running"
    job.attempt += 1
    job.started_at = job.started_at or now
    job.finished_at = None
    job.error_type = None
    job.leased_by = worker_id
    job.lease_expires_at = now + timedelta(seconds=lease_seconds)
    session.flush()
    return Lease(job.id, job.organization_id, job.source_id, job.attempt)


def lease_job(
    session_factory: sessionmaker[Session],
    *,
    worker_id: str,
    lease_seconds: int,
) -> Optional[Lease]:
    if not worker_id or len(worker_id) > 128:
        raise ValueError("worker_id must contain 1 to 128 characters")
    with session_factory() as session:
        if session.bind is not None and session.bind.dialect.name == "postgresql":
            lease = _lease_postgres(
                session,
                worker_id=worker_id,
                lease_seconds=lease_seconds,
            )
        else:
            lease = _lease_sqlite(
                session,
                worker_id=worker_id,
                lease_seconds=lease_seconds,
            )
        session.commit()
        return lease


def _renew_lease(
    session_factory: sessionmaker[Session],
    lease: Lease,
    *,
    worker_id: str,
    lease_seconds: int,
) -> bool:
    """Extend only a job that is still owned by this worker."""

    with session_factory() as session:
        set_identity_context(session, organization_id=lease.organization_id)
        if session.bind is not None and session.bind.dialect.name == "postgresql":
            # Match the leasing function's database clock and duration clamp;
            # an application clock skew must not shorten or expire the lease.
            result = session.execute(
                text(
                    "UPDATE jobs SET lease_expires_at = CURRENT_TIMESTAMP + "
                    "make_interval(secs => GREATEST(1, LEAST(:lease_seconds, 3600))) "
                    "WHERE id = :job_id AND organization_id = :organization_id "
                    "AND state = 'running' AND leased_by = :worker_id"
                ),
                {
                    "lease_seconds": lease_seconds,
                    "job_id": lease.job_id,
                    "organization_id": lease.organization_id,
                    "worker_id": worker_id,
                },
            )
        else:
            result = session.execute(
                update(Job)
                .where(
                    Job.id == lease.job_id,
                    Job.organization_id == lease.organization_id,
                    Job.state == "running",
                    Job.leased_by == worker_id,
                )
                .values(
                    lease_expires_at=_utc_now()
                    + timedelta(seconds=lease_seconds)
                )
            )
        session.commit()
        return result.rowcount == 1


class _LeaseHeartbeat:
    """Renew a lease while the synchronous analyzer is still running."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        lease: Lease,
        *,
        worker_id: str,
        lease_seconds: int,
    ):
        self.session_factory = session_factory
        self.lease = lease
        self.worker_id = worker_id
        self.lease_seconds = lease_seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name=f"lease-heartbeat-{lease.job_id}",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()

    def _run(self) -> None:
        interval = max(0.05, self.lease_seconds / 3)
        while not self._stop.wait(interval):
            try:
                if not _renew_lease(
                    self.session_factory,
                    self.lease,
                    worker_id=self.worker_id,
                    lease_seconds=self.lease_seconds,
                ):
                    return
            except Exception as error:
                # Never persist or log the database error text. A later beat
                # may recover; if it does not, ordinary lease ownership checks
                # ensure this worker cannot store a stale result.
                _log.warning(
                    "analysis lease heartbeat failed (%s)",
                    type(error).__name__,
                )


def _error_type(error: Exception) -> str:
    name = re.sub(r"(?<!^)(?=[A-Z])", "_", type(error).__name__).lower()
    name = re.sub(r"[^a-z0-9_]+", "_", name).strip("_") or "unknown"
    return f"analysis_{name}"[:128]


def _load_payloads(
    session_factory: sessionmaker[Session],
    lease: Lease,
    *,
    worker_id: str,
    max_events: int,
) -> tuple[list[dict[str, Any]], str, bool]:
    with session_factory() as session:
        set_identity_context(session, organization_id=lease.organization_id)
        job = session.scalar(
            select(Job).where(
                Job.id == lease.job_id,
                Job.organization_id == lease.organization_id,
                Job.state == "running",
                Job.leased_by == worker_id,
            )
        )
        if job is None:
            return [], "", False
        source = session.scalar(
            select(Source).where(
                Source.id == lease.source_id,
                Source.organization_id == lease.organization_id,
            )
        )
        if source is None:
            raise LookupError("source is unavailable")
        event_query = select(IngestEvent.payload).where(
            IngestEvent.organization_id == lease.organization_id,
            IngestEvent.source_id == lease.source_id,
        )
        cutoff_value = (
            job.payload.get("through_received_at")
            if isinstance(job.payload, dict)
            else None
        )
        if cutoff_value:
            cutoff = datetime.fromisoformat(str(cutoff_value).replace("Z", "+00:00"))
            if cutoff.tzinfo is None:
                cutoff = cutoff.replace(tzinfo=timezone.utc)
            event_query = event_query.where(IngestEvent.received_at <= cutoff)
        newest_first = list(
            session.scalars(
                event_query.order_by(
                    IngestEvent.received_at.desc(),
                    IngestEvent.id.desc(),
                ).limit(max_events + 1)
            )
        )
        history_truncated = len(newest_first) > max_events
        payloads = list(reversed(newest_first[:max_events]))
        return payloads, source.name, history_truncated


def _complete(
    session_factory: sessionmaker[Session],
    lease: Lease,
    *,
    worker_id: str,
    result: dict[str, Any],
    event_count: int,
    duration_ms: int,
) -> bool:
    with session_factory() as session:
        set_identity_context(session, organization_id=lease.organization_id)
        job = session.scalar(
            select(Job)
            .where(
                Job.id == lease.job_id,
                Job.organization_id == lease.organization_id,
            )
            .with_for_update()
        )
        if job is None or job.state != "running" or job.leased_by != worker_id:
            return False
        now = _utc_now()
        session.add(
            AnalysisResult(
                id=uuid.uuid4(),
                organization_id=lease.organization_id,
                source_id=lease.source_id,
                job_id=lease.job_id,
                schema_version=1,
                event_count=event_count,
                result=result,
            )
        )
        job.state = "succeeded"
        job.finished_at = now
        job.lease_expires_at = None
        job.leased_by = None
        job.error_type = None
        health = _health(session, lease)
        health.last_success_at = now
        health.last_job_duration_ms = duration_ms
        health.consecutive_failures = 0
        session.commit()
        return True


def _fail(
    session_factory: sessionmaker[Session],
    lease: Lease,
    *,
    worker_id: str,
    error_type: str,
    duration_ms: int,
    settings: Settings,
) -> bool:
    with session_factory() as session:
        set_identity_context(session, organization_id=lease.organization_id)
        job = session.scalar(
            select(Job)
            .where(
                Job.id == lease.job_id,
                Job.organization_id == lease.organization_id,
            )
            .with_for_update()
        )
        if job is None or job.state != "running" or job.leased_by != worker_id:
            return False
        now = _utc_now()
        retryable = job.attempt < job.max_attempts
        job.state = "queued" if retryable else "dead_letter"
        job.scheduled_for = now + timedelta(
            seconds=min(
                settings.job_retry_max_seconds,
                settings.job_retry_base_seconds * (2 ** max(0, job.attempt - 1)),
            )
        )
        job.finished_at = None if retryable else now
        job.lease_expires_at = None
        job.leased_by = None
        job.error_type = error_type
        health = _health(session, lease)
        health.last_error_at = now
        health.last_error_type = error_type
        health.last_job_duration_ms = duration_ms
        health.consecutive_failures += 1
        session.commit()
        return True


def run_once(
    session_factory: sessionmaker[Session],
    settings: Settings,
    *,
    worker_id: str,
    analyzer: Callable[..., dict[str, Any]] = analyze_event_payloads,
) -> bool:
    """Lease and process at most one job; return whether a job was leased."""

    logger = event_logger(_log.name, json_logs=settings.json_logs)
    lease = lease_job(
        session_factory,
        worker_id=worker_id,
        lease_seconds=settings.job_lease_seconds,
    )
    if lease is None:
        return False
    log_event(
        logger,
        "analysis_job.leased",
        job_id=str(lease.job_id),
        source_id=str(lease.source_id),
        attempt=lease.attempt,
    )

    started = time.monotonic()
    heartbeat = _LeaseHeartbeat(
        session_factory,
        lease,
        worker_id=worker_id,
        lease_seconds=settings.job_lease_seconds,
    )
    heartbeat.start()
    try:
        payloads, source_name, history_truncated = _load_payloads(
            session_factory,
            lease,
            worker_id=worker_id,
            max_events=settings.analysis_max_events,
        )
        if not source_name:
            return True
        with trace_span(
            "cacheeconomics.analysis_job",
            job_id=str(lease.job_id),
            source_id=str(lease.source_id),
            attempt=lease.attempt,
        ):
            result = analyzer(
                payloads,
                source=source_name,
                history_truncated=history_truncated,
            )
        duration_ms = max(0, round((time.monotonic() - started) * 1000))
        completed = _complete(
            session_factory,
            lease,
            worker_id=worker_id,
            result=result,
            event_count=len(payloads),
            duration_ms=duration_ms,
        )
        log_event(
            logger,
            "analysis_job.succeeded" if completed else "analysis_job.lease_lost",
            job_id=str(lease.job_id),
            source_id=str(lease.source_id),
            attempt=lease.attempt,
            duration_ms=duration_ms,
        )
    except Exception as error:
        duration_ms = max(0, round((time.monotonic() - started) * 1000))
        error_type = _error_type(error)
        failed = _fail(
            session_factory,
            lease,
            worker_id=worker_id,
            error_type=error_type,
            duration_ms=duration_ms,
            settings=settings,
        )
        log_event(
            logger,
            "analysis_job.failed" if failed else "analysis_job.lease_lost",
            job_id=str(lease.job_id),
            source_id=str(lease.source_id),
            attempt=lease.attempt,
            duration_ms=duration_ms,
            error_type=error_type,
        )
    finally:
        heartbeat.stop()
    return True


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the cacheeconomics analysis worker")
    parser.add_argument("--once", action="store_true", help="process at most one job")
    parser.add_argument("--worker-id", default=f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}")
    args = parser.parse_args(argv)

    settings = Settings()
    engine = create_database_engine(settings)
    session_factory = create_session_factory(engine)
    try:
        if args.once:
            run_once(session_factory, settings, worker_id=args.worker_id)
            return 0
        while True:
            worked = run_once(session_factory, settings, worker_id=args.worker_id)
            if not worked:
                time.sleep(settings.worker_poll_seconds)
    except KeyboardInterrupt:
        return 0
    finally:
        engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
