"""Create the multi-tenant control-plane schema and row-security policies.

Revision ID: 20260905_0001
Revises: None
Create Date: 2026-09-05
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "20260905_0001"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


UUID = sa.Uuid()
JSONB = postgresql.JSONB(astext_type=sa.Text())
NOW = sa.text("CURRENT_TIMESTAMP")


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("oidc_issuer", sa.String(512), nullable=False),
        sa.Column("oidc_subject", sa.String(512), nullable=False),
        sa.Column("email", sa.String(320)),
        sa.Column("display_name", sa.String(256)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=NOW),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=NOW),
        sa.UniqueConstraint("oidc_issuer", "oidc_subject", name="uq_users_oidc_identity"),
    )
    op.create_table(
        "organizations",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("name", sa.String(160), nullable=False),
        sa.Column("slug", sa.String(80), nullable=False, unique=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=NOW),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=NOW),
    )
    op.create_table(
        "memberships",
        sa.Column(
            "organization_id",
            UUID,
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "user_id",
            UUID,
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("role", sa.String(16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=NOW),
        sa.CheckConstraint(
            "role IN ('viewer', 'analyst', 'admin', 'owner')",
            name="membership_role",
        ),
    )
    op.create_table(
        "sources",
        sa.Column("id", UUID, primary_key=True),
        sa.Column(
            "organization_id",
            UUID,
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(160), nullable=False),
        sa.Column("kind", sa.String(64), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=NOW),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=NOW),
        sa.UniqueConstraint("organization_id", "name", name="uq_sources_org_name"),
    )
    op.create_index(
        "ix_sources_organization_id_created_at",
        "sources",
        ["organization_id", "created_at"],
    )
    op.create_table(
        "source_credentials",
        sa.Column("id", UUID, primary_key=True),
        sa.Column(
            "organization_id",
            UUID,
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "source_id",
            UUID,
            sa.ForeignKey("sources.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(160), nullable=False),
        sa.Column("credential_prefix", sa.String(32), nullable=False, unique=True),
        sa.Column("credential_digest", sa.String(64), nullable=False),
        sa.Column(
            "created_by_user_id",
            UUID,
            sa.ForeignKey("users.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=NOW),
        sa.Column("expires_at", sa.DateTime(timezone=True)),
        sa.Column("last_used_at", sa.DateTime(timezone=True)),
        sa.Column("revoked_at", sa.DateTime(timezone=True)),
    )
    op.create_index(
        "ix_source_credentials_org_source",
        "source_credentials",
        ["organization_id", "source_id"],
    )
    op.create_table(
        "ingest_events",
        sa.Column("id", UUID, primary_key=True),
        sa.Column(
            "organization_id",
            UUID,
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "source_id",
            UUID,
            sa.ForeignKey("sources.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("event_id", sa.String(256), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column("payload", JSONB, nullable=False),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=False, server_default=NOW),
        sa.UniqueConstraint("source_id", "event_id", name="uq_ingest_events_source_event"),
    )
    op.create_index(
        "ix_ingest_events_org_received",
        "ingest_events",
        ["organization_id", "received_at"],
    )
    op.create_table(
        "analyses",
        sa.Column("id", UUID, primary_key=True),
        sa.Column(
            "organization_id",
            UUID,
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "source_id",
            UUID,
            sa.ForeignKey("sources.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column("result", JSONB, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=NOW),
    )
    op.create_index(
        "ix_analyses_org_created", "analyses", ["organization_id", "created_at"]
    )
    op.create_table(
        "jobs",
        sa.Column("id", UUID, primary_key=True),
        sa.Column(
            "organization_id",
            UUID,
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "source_id",
            UUID,
            sa.ForeignKey("sources.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("kind", sa.String(64), nullable=False),
        sa.Column("state", sa.String(16), nullable=False, server_default="queued"),
        sa.Column("attempt", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("scheduled_for", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.Column("error_type", sa.String(128)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=NOW),
        sa.CheckConstraint(
            "state IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')",
            name="job_state",
        ),
    )
    op.create_index("ix_jobs_org_scheduled", "jobs", ["organization_id", "scheduled_for"])
    op.create_table(
        "audit_events",
        sa.Column("id", UUID, primary_key=True),
        sa.Column(
            "organization_id",
            UUID,
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "actor_user_id",
            UUID,
            sa.ForeignKey("users.id", ondelete="SET NULL"),
        ),
        sa.Column(
            "actor_credential_id",
            UUID,
            sa.ForeignKey("source_credentials.id", ondelete="SET NULL"),
        ),
        sa.Column("action", sa.String(128), nullable=False),
        sa.Column("resource_type", sa.String(64), nullable=False),
        sa.Column("resource_id", UUID),
        sa.Column("metadata", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False, server_default=NOW),
    )
    op.create_index(
        "ix_audit_events_org_occurred",
        "audit_events",
        ["organization_id", "occurred_at"],
    )

    _enable_row_security()
    _create_collector_auth_function()
    _grant_application_permissions()


def _setting(name: str) -> str:
    return f"NULLIF(current_setting('{name}', true), '')::uuid"


def _enable_row_security() -> None:
    current_org = _setting("app.current_organization_id")
    current_user = _setting("app.current_user_id")

    for table in (
        "organizations",
        "memberships",
        "sources",
        "source_credentials",
        "ingest_events",
        "analyses",
        "jobs",
        "audit_events",
    ):
        op.execute(f'ALTER TABLE "{table}" ENABLE ROW LEVEL SECURITY')

    op.execute(
        f"""
        CREATE POLICY organizations_select ON organizations FOR SELECT
        USING (
            id = {current_org}
            OR EXISTS (
                SELECT 1 FROM memberships m
                WHERE m.organization_id = organizations.id
                  AND m.user_id = {current_user}
            )
        )
        """
    )
    op.execute(
        f"""
        CREATE POLICY organizations_insert ON organizations FOR INSERT
        WITH CHECK (id = {current_org})
        """
    )
    op.execute(
        f"""
        CREATE POLICY organizations_update ON organizations FOR UPDATE
        USING (id = {current_org}) WITH CHECK (id = {current_org})
        """
    )
    op.execute(
        f"CREATE POLICY organizations_delete ON organizations FOR DELETE "
        f"USING (id = {current_org})"
    )

    op.execute(
        f"""
        CREATE POLICY memberships_select ON memberships FOR SELECT
        USING (organization_id = {current_org} OR user_id = {current_user})
        """
    )
    op.execute(
        f"""
        CREATE POLICY memberships_insert ON memberships FOR INSERT
        WITH CHECK (organization_id = {current_org})
        """
    )
    op.execute(
        f"""
        CREATE POLICY memberships_update ON memberships FOR UPDATE
        USING (organization_id = {current_org})
        WITH CHECK (organization_id = {current_org})
        """
    )
    op.execute(
        f"CREATE POLICY memberships_delete ON memberships FOR DELETE "
        f"USING (organization_id = {current_org})"
    )

    for table in (
        "sources",
        "source_credentials",
        "ingest_events",
        "analyses",
        "jobs",
    ):
        op.execute(
            f"""
            CREATE POLICY {table}_tenant_isolation ON {table} FOR ALL
            USING (organization_id = {current_org})
            WITH CHECK (organization_id = {current_org})
            """
        )

    op.execute(
        f"""
        CREATE POLICY audit_events_select ON audit_events FOR SELECT
        USING (organization_id = {current_org})
        """
    )
    op.execute(
        f"""
        CREATE POLICY audit_events_insert ON audit_events FOR INSERT
        WITH CHECK (organization_id = {current_org})
        """
    )


def _create_collector_auth_function() -> None:
    # The app cannot know an organization before authenticating a collector, so
    # a narrowly scoped SECURITY DEFINER function performs only that lookup.
    # Its owner is the migration role (the table owner), never the API role.
    op.execute(
        """
        CREATE OR REPLACE FUNCTION authenticate_source_credential(
            p_prefix text,
            p_digest text
        )
        RETURNS TABLE (credential_id uuid, organization_id uuid, source_id uuid)
        LANGUAGE plpgsql
        SECURITY DEFINER
        SET search_path = pg_catalog, public
        AS $$
        BEGIN
            RETURN QUERY
            UPDATE public.source_credentials AS credential
               SET last_used_at = CURRENT_TIMESTAMP
              FROM public.sources AS source
             WHERE credential.credential_prefix = p_prefix
               AND credential.credential_digest = p_digest
               AND credential.revoked_at IS NULL
               AND (credential.expires_at IS NULL OR credential.expires_at > CURRENT_TIMESTAMP)
               AND source.id = credential.source_id
               AND source.organization_id = credential.organization_id
               AND source.enabled = true
            RETURNING credential.id, credential.organization_id, credential.source_id;
        END;
        $$
        """
    )
    op.execute(
        "REVOKE ALL ON FUNCTION authenticate_source_credential(text, text) FROM PUBLIC"
    )
    op.execute(
        "GRANT EXECUTE ON FUNCTION authenticate_source_credential(text, text) "
        "TO cacheeconomics_app"
    )


def _grant_application_permissions() -> None:
    op.execute("GRANT SELECT, INSERT, UPDATE ON users TO cacheeconomics_app")
    op.execute(
        "GRANT SELECT, INSERT, UPDATE, DELETE ON organizations, memberships, sources "
        "TO cacheeconomics_app"
    )
    op.execute("GRANT SELECT, INSERT, UPDATE ON source_credentials TO cacheeconomics_app")
    op.execute("GRANT SELECT, INSERT ON ingest_events, analyses TO cacheeconomics_app")
    op.execute("GRANT SELECT, INSERT, UPDATE ON jobs TO cacheeconomics_app")
    op.execute("GRANT SELECT, INSERT ON audit_events TO cacheeconomics_app")


def downgrade() -> None:
    op.execute("DROP FUNCTION IF EXISTS authenticate_source_credential(text, text)")
    for table in (
        "audit_events",
        "jobs",
        "analyses",
        "ingest_events",
        "source_credentials",
        "sources",
        "memberships",
        "organizations",
        "users",
    ):
        op.drop_table(table)
