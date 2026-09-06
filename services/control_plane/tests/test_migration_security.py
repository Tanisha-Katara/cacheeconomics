from __future__ import annotations

import ast
import importlib.util
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[3]
MIGRATION = ROOT / "services/control_plane/migrations/versions/20260905_0001_control_plane.py"
INGESTION_MIGRATION = ROOT / "services/control_plane/migrations/versions/20260905_0002_ingestion_jobs.py"
ROLES = ROOT / "deploy/postgres/init/001_roles.sql"


ORG_TABLES = {
    "organizations",
    "memberships",
    "sources",
    "source_credentials",
    "ingest_events",
    "analyses",
    "jobs",
    "audit_events",
}


def load_migration():
    spec = importlib.util.spec_from_file_location("control_plane_migration", MIGRATION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_ingestion_migration():
    spec = importlib.util.spec_from_file_location(
        "ingestion_migration", INGESTION_MIGRATION
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_organization_owned_table_enables_row_security(monkeypatch):
    migration = load_migration()
    statements = []
    monkeypatch.setattr(migration.op, "execute", lambda statement: statements.append(str(statement)))
    migration._enable_row_security()
    rendered = "\n".join(statements)

    for table in ORG_TABLES:
        assert f'ALTER TABLE "{table}" ENABLE ROW LEVEL SECURITY' in rendered
    for table in {"sources", "source_credentials", "ingest_events", "analyses", "jobs"}:
        assert f"CREATE POLICY {table}_tenant_isolation ON {table}" in rendered
    assert "CREATE POLICY organizations_select" in rendered
    assert "CREATE POLICY memberships_select" in rendered
    assert "CREATE POLICY audit_events_select" in rendered
    assert "app.current_organization_id" in rendered
    assert "WITH CHECK" in rendered


def test_append_only_and_ingest_tables_have_narrow_runtime_grants(monkeypatch):
    migration = load_migration()
    statements = []
    monkeypatch.setattr(migration.op, "execute", lambda statement: statements.append(str(statement)))
    migration._grant_application_permissions()
    rendered = "\n".join(statements)

    assert "GRANT SELECT, INSERT ON ingest_events, analyses" in rendered
    assert "GRANT SELECT, INSERT ON audit_events" in rendered
    audit_grant = next(line for line in statements if "audit_events" in line)
    assert "UPDATE" not in audit_grant
    assert "DELETE" not in audit_grant


def test_collector_bootstrap_lookup_is_narrow_and_not_public():
    source = MIGRATION.read_text()
    assert "SECURITY DEFINER" in source
    assert "SET search_path = pg_catalog, public" in source
    assert "REVOKE ALL ON FUNCTION authenticate_source_credential" in source
    assert "GRANT EXECUTE ON FUNCTION authenticate_source_credential" in source
    assert "credential.revoked_at IS NULL" in source
    assert "source.enabled = true" in source


def test_downgrade_removes_cross_table_policy_before_memberships(monkeypatch):
    migration = load_migration()
    actions = []
    monkeypatch.setattr(
        migration.op,
        "execute",
        lambda statement: actions.append(("execute", str(statement))),
    )
    monkeypatch.setattr(
        migration.op,
        "drop_table",
        lambda table: actions.append(("drop_table", table)),
    )

    migration.downgrade()

    policy = (
        "execute",
        "DROP POLICY IF EXISTS organizations_select ON organizations",
    )
    assert actions.index(policy) < actions.index(("drop_table", "memberships"))


def test_runtime_database_role_cannot_bypass_row_security():
    roles = ROLES.read_text()
    assert "cacheeconomics_app NOSUPERUSER" in roles
    assert "NOBYPASSRLS" in roles
    assert "REVOKE CREATE ON SCHEMA public FROM PUBLIC" in roles


def test_worker_role_and_job_lease_have_narrow_privileges(monkeypatch):
    roles = ROLES.read_text()
    assert "cacheeconomics_worker NOSUPERUSER" in roles
    worker_line = next(line for line in roles.splitlines() if line.startswith("ALTER ROLE cacheeconomics_worker"))
    assert "NOBYPASSRLS" in worker_line

    migration = load_ingestion_migration()
    statements = []
    monkeypatch.setattr(migration.op, "execute", lambda statement: statements.append(str(statement)))
    migration._create_job_lease_function()
    migration._grant_phase_two_permissions()
    rendered = "\n".join(statements)
    assert "FOR UPDATE SKIP LOCKED" in rendered
    assert "expired_job.attempt >= expired_job.max_attempts" in rendered
    assert "RETURNING expired_job.source_id, expired_job.organization_id" in rendered
    assert "SECURITY DEFINER" in rendered
    assert "REVOKE ALL ON FUNCTION lease_analysis_job" in rendered
    assert "GRANT EXECUTE ON FUNCTION lease_analysis_job" in rendered
    assert "TO cacheeconomics_worker" in rendered
    assert "GRANT SELECT ON sources, ingest_events, analyses, jobs, source_health" in rendered
    assert "GRANT UPDATE ON jobs, source_health" in rendered


def test_source_rows_cannot_reference_a_different_organization():
    source = INGESTION_MIGRATION.read_text()
    assert "uq_sources_id_organization_id" in source
    assert 'f"fk_{table}_source_organization"' in source
    assert "fk_source_health_source_organization" in source
    for table in ("source_credentials", "ingest_events", "analyses", "jobs"):
        assert table in source


def test_offline_package_never_imports_the_hosted_service():
    package = ROOT / "harness/cacheeconomics"
    offenders = []
    for path in package.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            else:
                names = []
            if any(name.startswith("cacheeconomics_control_plane") for name in names):
                offenders.append(str(path))
    assert offenders == []
