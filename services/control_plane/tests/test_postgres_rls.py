from __future__ import annotations

import hashlib
import os
import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import DBAPIError

from cacheeconomics_control_plane.db import create_session_factory
from cacheeconomics_control_plane.worker import Lease, _renew_lease


APP_URL = os.environ.get("TEST_POSTGRES_APP_URL")
WORKER_URL = os.environ.get("TEST_POSTGRES_WORKER_URL")
pytestmark = pytest.mark.postgres


@pytest.mark.skipif(not APP_URL, reason="TEST_POSTGRES_APP_URL is not configured")
def test_postgres_itself_blocks_cross_organization_rows():
    engine = create_engine(APP_URL)
    organization_a = uuid.uuid4()
    organization_b = uuid.uuid4()
    source_a = uuid.uuid4()
    source_b = uuid.uuid4()
    user_id = uuid.uuid4()
    credential_id = uuid.uuid4()
    credential_prefix = f"test{credential_id.hex[:12]}"
    credential_digest = hashlib.sha256(b"test-collector-secret").hexdigest()

    def set_org(connection, organization_id):
        connection.execute(
            text("SELECT set_config('app.current_organization_id', :value, true)"),
            {"value": str(organization_id)},
        )

    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO users (id, oidc_issuer, oidc_subject) "
                "VALUES (:id, 'https://test.invalid/', :subject)"
            ),
            {"id": user_id, "subject": f"postgres-rls-{user_id.hex}"},
        )
        set_org(connection, organization_a)
        connection.execute(
            text("INSERT INTO organizations (id, name, slug) VALUES (:id, 'A', :slug)"),
            {"id": organization_a, "slug": f"a-{organization_a.hex}"},
        )
        connection.execute(
            text(
                "INSERT INTO sources (id, organization_id, name, kind) "
                "VALUES (:id, :organization_id, 'A source', 'test')"
            ),
            {"id": source_a, "organization_id": organization_a},
        )
        connection.execute(
            text(
                "INSERT INTO source_credentials "
                "(id, organization_id, source_id, name, credential_prefix, "
                "credential_digest, created_by_user_id) "
                "VALUES (:id, :organization_id, :source_id, 'Test credential', "
                ":prefix, :digest, :user_id)"
            ),
            {
                "id": credential_id,
                "organization_id": organization_a,
                "source_id": source_a,
                "prefix": credential_prefix,
                "digest": credential_digest,
                "user_id": user_id,
            },
        )

    with engine.begin() as connection:
        set_org(connection, organization_b)
        connection.execute(
            text("INSERT INTO organizations (id, name, slug) VALUES (:id, 'B', :slug)"),
            {"id": organization_b, "slug": f"b-{organization_b.hex}"},
        )
        connection.execute(
            text(
                "INSERT INTO sources (id, organization_id, name, kind) "
                "VALUES (:id, :organization_id, 'B source', 'test')"
            ),
            {"id": source_b, "organization_id": organization_b},
        )

    with engine.begin() as connection:
        set_org(connection, organization_a)
        visible = connection.execute(text("SELECT id FROM sources")).scalars().all()
        assert visible == [source_a]

    # The runtime role cannot enumerate credentials without tenant context.
    # It can only ask the narrow authentication function about an exact
    # prefix/digest pair; that function returns the tenant scope to apply.
    with engine.begin() as connection:
        assert connection.execute(text("SELECT id FROM source_credentials")).all() == []
        match = connection.execute(
            text(
                "SELECT credential_id, organization_id, source_id "
                "FROM authenticate_source_credential(:prefix, :digest)"
            ),
            {"prefix": credential_prefix, "digest": credential_digest},
        ).one()
        assert match == (credential_id, organization_a, source_a)
        no_match = connection.execute(
            text(
                "SELECT credential_id "
                "FROM authenticate_source_credential(:prefix, :digest)"
            ),
            {"prefix": credential_prefix, "digest": "0" * 64},
        ).one_or_none()
        assert no_match is None

    with pytest.raises(DBAPIError):
        with engine.begin() as connection:
            set_org(connection, organization_a)
            connection.execute(
                text(
                    "INSERT INTO sources (id, organization_id, name, kind) "
                    "VALUES (:id, :organization_id, 'Cross tenant', 'test')"
                ),
                {"id": uuid.uuid4(), "organization_id": organization_b},
            )

    # Even if a buggy app query knows another source UUID, the composite
    # source/organization foreign key prevents attaching this tenant's row to it.
    with pytest.raises(DBAPIError):
        with engine.begin() as connection:
            set_org(connection, organization_a)
            connection.execute(
                text(
                    "INSERT INTO ingest_events "
                    "(id, organization_id, source_id, event_id, schema_version, payload) "
                    "VALUES (:id, :organization_id, :source_id, 'cross-source', 1, '{}')"
                ),
                {
                    "id": uuid.uuid4(),
                    "organization_id": organization_a,
                    "source_id": source_b,
                },
            )

    engine.dispose()


@pytest.mark.skipif(
    not APP_URL or not WORKER_URL,
    reason="PostgreSQL app and worker URLs are not configured",
)
def test_worker_can_atomically_lease_but_cannot_bypass_tenant_rows():
    app_engine = create_engine(APP_URL)
    worker_engine = create_engine(WORKER_URL)
    organization_id = uuid.uuid4()
    source_id = uuid.uuid4()
    job_id = uuid.uuid4()

    with app_engine.begin() as connection:
        connection.execute(
            text("SELECT set_config('app.current_organization_id', :value, true)"),
            {"value": str(organization_id)},
        )
        connection.execute(
            text("INSERT INTO organizations (id, name, slug) VALUES (:id, 'Worker test', :slug)"),
            {"id": organization_id, "slug": f"worker-{organization_id.hex}"},
        )
        connection.execute(
            text(
                "INSERT INTO sources (id, organization_id, name, kind) "
                "VALUES (:id, :organization_id, 'Worker source', 'litellm')"
            ),
            {"id": source_id, "organization_id": organization_id},
        )
        connection.execute(
            text(
                "INSERT INTO jobs (id, organization_id, source_id, kind, state, "
                "scheduled_for) VALUES (:id, :organization_id, :source_id, "
                "'analyze_source', 'queued', :scheduled_for)"
            ),
            {
                "id": job_id,
                "organization_id": organization_id,
                "source_id": source_id,
                "scheduled_for": datetime.now(timezone.utc),
            },
        )

    with worker_engine.begin() as connection:
        assert connection.execute(text("SELECT id FROM jobs")).all() == []
        leased = connection.execute(
            text(
                "SELECT job_id, organization_id, source_id, attempt "
                "FROM lease_analysis_job('postgres-test-worker', 60)"
            )
        ).one()
        assert leased == (job_id, organization_id, source_id, 1)
        assert connection.execute(text("SELECT id FROM jobs")).all() == []
        connection.execute(
            text("SELECT set_config('app.current_organization_id', :value, true)"),
            {"value": str(organization_id)},
        )
        assert connection.execute(text("SELECT id FROM jobs")).scalar_one() == job_id

    lease = Lease(
        job_id=job_id,
        organization_id=organization_id,
        source_id=source_id,
        attempt=1,
    )
    worker_sessions = create_session_factory(worker_engine)
    assert _renew_lease(
        worker_sessions,
        lease,
        worker_id="postgres-test-worker",
        lease_seconds=120,
    )
    assert not _renew_lease(
        worker_sessions,
        lease,
        worker_id="other-worker",
        lease_seconds=120,
    )

    app_engine.dispose()
    worker_engine.dispose()
