"""Add reliable ingestion health, job leasing, and analysis provenance.

Revision ID: 20260905_0002
Revises: 20260905_0001
Create Date: 2026-09-05
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "20260905_0002"
down_revision: Union[str, Sequence[str], None] = "20260905_0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


UUID = sa.Uuid()
JSONB = postgresql.JSONB(astext_type=sa.Text())
NOW = sa.text("CURRENT_TIMESTAMP")


def upgrade() -> None:
    op.create_unique_constraint(
        "uq_sources_id_organization_id",
        "sources",
        ["id", "organization_id"],
    )
    for table in ("source_credentials", "ingest_events", "analyses", "jobs"):
        op.drop_constraint(
            f"fk_{table}_source_id_sources",
            table,
            type_="foreignkey",
        )
        op.create_foreign_key(
            f"fk_{table}_source_organization",
            table,
            "sources",
            ["source_id", "organization_id"],
            ["id", "organization_id"],
            ondelete="CASCADE",
        )

    op.add_column(
        "jobs",
        sa.Column("max_attempts", sa.Integer(), nullable=False, server_default="3"),
    )
    op.add_column(
        "jobs",
        sa.Column("payload", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
    )
    op.add_column("jobs", sa.Column("lease_expires_at", sa.DateTime(timezone=True)))
    op.add_column("jobs", sa.Column("leased_by", sa.String(128)))
    op.drop_constraint(op.f("ck_jobs_job_state"), "jobs", type_="check")
    op.create_check_constraint(
        "job_state",
        "jobs",
        "state IN ('queued', 'running', 'succeeded', 'failed', "
        "'dead_letter', 'cancelled')",
    )
    op.create_index(
        "ix_jobs_state_scheduled_lease",
        "jobs",
        ["state", "scheduled_for", "lease_expires_at"],
    )

    op.add_column("analyses", sa.Column("job_id", UUID))
    op.add_column(
        "analyses",
        sa.Column("event_count", sa.Integer(), nullable=False, server_default="0"),
    )
    op.create_foreign_key(
        "fk_analyses_job_id_jobs",
        "analyses",
        "jobs",
        ["job_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_unique_constraint("uq_analyses_job_id", "analyses", ["job_id"])

    op.create_table(
        "source_health",
        sa.Column(
            "source_id",
            UUID,
            primary_key=True,
        ),
        sa.Column(
            "organization_id",
            UUID,
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("accepted_events", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("duplicate_events", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("rejected_events", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("last_received_at", sa.DateTime(timezone=True)),
        sa.Column("last_event_sent_at", sa.DateTime(timezone=True)),
        sa.Column("last_ingest_lag_seconds", sa.Float()),
        sa.Column("last_success_at", sa.DateTime(timezone=True)),
        sa.Column("last_error_at", sa.DateTime(timezone=True)),
        sa.Column("last_error_type", sa.String(128)),
        sa.Column("last_job_duration_ms", sa.BigInteger()),
        sa.Column("consecutive_failures", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=NOW),
        sa.ForeignKeyConstraint(
            ["source_id", "organization_id"],
            ["sources.id", "sources.organization_id"],
            name="fk_source_health_source_organization",
            ondelete="CASCADE",
        ),
    )
    op.create_index(
        "ix_source_health_organization_id",
        "source_health",
        ["organization_id"],
    )
    op.execute(
        "INSERT INTO source_health (source_id, organization_id) "
        "SELECT id, organization_id FROM sources"
    )
    op.execute('ALTER TABLE "source_health" ENABLE ROW LEVEL SECURITY')
    current_org = "NULLIF(current_setting('app.current_organization_id', true), '')::uuid"
    op.execute(
        f"""
        CREATE POLICY source_health_tenant_isolation ON source_health FOR ALL
        USING (organization_id = {current_org})
        WITH CHECK (organization_id = {current_org})
        """
    )

    _create_job_lease_function()
    _grant_phase_two_permissions()


def _create_job_lease_function() -> None:
    # The worker cannot discover a tenant before it has leased a job. This
    # function does exactly one cross-tenant operation: atomically claim the
    # next due analysis job. All event and result access happens afterwards
    # under the returned organization context and ordinary RLS policies.
    op.execute(
        """
        CREATE OR REPLACE FUNCTION lease_analysis_job(
            p_worker_id text,
            p_lease_seconds integer
        )
        RETURNS TABLE (
            job_id uuid,
            organization_id uuid,
            source_id uuid,
            attempt integer
        )
        LANGUAGE plpgsql
        SECURITY DEFINER
        SET search_path = pg_catalog, public
        AS $$
        BEGIN
            WITH abandoned AS (
                UPDATE public.jobs AS expired_job
                   SET state = 'dead_letter',
                       finished_at = CURRENT_TIMESTAMP,
                       error_type = 'lease_expired',
                       lease_expires_at = NULL,
                       leased_by = NULL
                 WHERE expired_job.state = 'running'
                   AND expired_job.lease_expires_at < CURRENT_TIMESTAMP
                   AND expired_job.attempt >= expired_job.max_attempts
                RETURNING expired_job.source_id, expired_job.organization_id
            )
            UPDATE public.source_health AS health
               SET last_error_at = CURRENT_TIMESTAMP,
                   last_error_type = 'lease_expired',
                   consecutive_failures = health.consecutive_failures + 1,
                   updated_at = CURRENT_TIMESTAMP
              FROM abandoned
             WHERE abandoned.source_id = health.source_id
               AND abandoned.organization_id = health.organization_id;

            RETURN QUERY
            WITH candidate AS (
                SELECT candidate_job.id
                  FROM public.jobs AS candidate_job
                 WHERE candidate_job.kind = 'analyze_source'
                   AND candidate_job.attempt < candidate_job.max_attempts
                   AND (
                       (candidate_job.state = 'queued'
                        AND candidate_job.scheduled_for <= CURRENT_TIMESTAMP)
                       OR
                       (candidate_job.state = 'running'
                        AND candidate_job.lease_expires_at < CURRENT_TIMESTAMP)
                   )
                 ORDER BY candidate_job.scheduled_for, candidate_job.id
                 FOR UPDATE SKIP LOCKED
                 LIMIT 1
            )
            UPDATE public.jobs AS leased
               SET state = 'running',
                   attempt = leased.attempt + 1,
                   started_at = COALESCE(leased.started_at, CURRENT_TIMESTAMP),
                   finished_at = NULL,
                   error_type = NULL,
                   leased_by = p_worker_id,
                   lease_expires_at = CURRENT_TIMESTAMP
                       + make_interval(secs => GREATEST(1, LEAST(p_lease_seconds, 3600)))
              FROM candidate
             WHERE leased.id = candidate.id
            RETURNING leased.id, leased.organization_id, leased.source_id, leased.attempt;
        END;
        $$
        """
    )
    op.execute("REVOKE ALL ON FUNCTION lease_analysis_job(text, integer) FROM PUBLIC")
    op.execute(
        "GRANT EXECUTE ON FUNCTION lease_analysis_job(text, integer) "
        "TO cacheeconomics_worker"
    )


def _grant_phase_two_permissions() -> None:
    op.execute(
        "GRANT SELECT, INSERT, UPDATE ON source_health TO cacheeconomics_app"
    )
    op.execute(
        "GRANT SELECT ON sources, ingest_events, analyses, jobs, source_health "
        "TO cacheeconomics_worker"
    )
    op.execute("GRANT INSERT ON analyses, source_health TO cacheeconomics_worker")
    op.execute("GRANT UPDATE ON jobs, source_health TO cacheeconomics_worker")


def downgrade() -> None:
    op.execute("DROP FUNCTION IF EXISTS lease_analysis_job(text, integer)")
    op.drop_index("ix_source_health_organization_id", table_name="source_health")
    op.drop_table("source_health")

    op.drop_constraint("uq_analyses_job_id", "analyses", type_="unique")
    op.drop_constraint("fk_analyses_job_id_jobs", "analyses", type_="foreignkey")
    op.drop_column("analyses", "event_count")
    op.drop_column("analyses", "job_id")

    op.drop_index("ix_jobs_state_scheduled_lease", table_name="jobs")
    op.drop_constraint(op.f("ck_jobs_job_state"), "jobs", type_="check")
    op.create_check_constraint(
        "job_state",
        "jobs",
        "state IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')",
    )
    op.drop_column("jobs", "leased_by")
    op.drop_column("jobs", "lease_expires_at")
    op.drop_column("jobs", "payload")
    op.drop_column("jobs", "max_attempts")

    for table in ("source_credentials", "ingest_events", "analyses", "jobs"):
        op.drop_constraint(
            f"fk_{table}_source_organization",
            table,
            type_="foreignkey",
        )
        op.create_foreign_key(
            f"fk_{table}_source_id_sources",
            table,
            "sources",
            ["source_id"],
            ["id"],
            ondelete="CASCADE",
        )
    op.drop_constraint(
        "uq_sources_id_organization_id",
        "sources",
        type_="unique",
    )
