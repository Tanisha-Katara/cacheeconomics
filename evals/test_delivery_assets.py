from __future__ import annotations

import os
import subprocess
from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory
from cacheeconomics import __version__ as core_version


ROOT = Path(__file__).resolve().parents[1]


def test_hosted_python_packages_pin_the_current_contract_bearing_core_release():
    expected = f"cacheeconomics=={core_version}"
    control_plane = (ROOT / "services/control_plane/pyproject.toml").read_text()
    collector = (ROOT / "collectors/pyproject.toml").read_text()
    collector_image = (ROOT / "collectors/Dockerfile").read_text()

    assert expected in control_plane
    assert expected in collector
    assert expected in collector_image
    assert (ROOT / "harness/cacheeconomics/contracts.py").is_file()


def test_default_pytest_discovery_keeps_hosted_dependencies_out_of_core_checks():
    root_config = (ROOT / "pyproject.toml").read_text()
    assert 'testpaths = ["harness/tests", "contrib/litellm"]' in root_config


def test_every_migration_has_a_downgrade_path_and_one_linear_head():
    config = Config(str(ROOT / "services/control_plane/alembic.ini"))
    config.set_main_option(
        "script_location", str(ROOT / "services/control_plane/migrations")
    )
    scripts = ScriptDirectory.from_config(config)
    assert len(scripts.get_heads()) == 1
    revisions = list(scripts.walk_revisions())
    assert len(revisions) >= 2
    assert all(revision.down_revision is not None for revision in revisions[:-1])
    assert revisions[-1].down_revision is None


def test_backup_restore_script_fails_closed_without_explicit_consent():
    script = ROOT / "deploy/scripts/backup_restore_smoke.sh"
    environment = os.environ.copy()
    environment.update(
        SOURCE_DATABASE_URL="postgresql://unused/source",
        RESTORE_DATABASE_URL="postgresql://unused/unsafe",
    )
    environment.pop("CACHEECONOMICS_ALLOW_DESTRUCTIVE_RESTORE_TEST", None)
    result = subprocess.run(
        ["sh", str(script)],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "CACHEECONOMICS_ALLOW_DESTRUCTIVE_RESTORE_TEST" in result.stderr


def test_backup_restore_script_has_narrow_target_guards():
    script = (ROOT / "deploy/scripts/backup_restore_smoke.sh").read_text()
    assert "*_restore_test" in script
    assert 'if [ "$source_name" = "$restore_name" ]' in script
    assert 'DROP SCHEMA public CASCADE' in script
    assert "pg_dump" in script
    assert "pg_restore" in script
    assert "--no-privileges" not in script
    assert "--no-owner" not in script
    assert "RESTORE_APP_DATABASE_URL" in script
    assert "RESTORE_WORKER_DATABASE_URL" in script
    assert "has_table_privilege" in script
    assert "has_function_privilege" in script


def test_four_runtime_container_roles_are_explicit():
    control_plane = (ROOT / "services/control_plane/Dockerfile").read_text()
    collector = (ROOT / "collectors/Dockerfile").read_text()
    dashboard = (ROOT / "apps/dashboard/Dockerfile").read_text()
    assert "FROM runtime AS api" in control_plane
    assert "FROM runtime AS worker" in control_plane
    assert "FROM runtime AS migrate" in control_plane
    assert 'ENTRYPOINT ["cacheeconomics-collector"]' in collector
    assert "USER collector" in collector
    assert "USER nginx" in dashboard
    assert "libuuid=2.42.3-r1" in dashboard
    for dockerfile in (control_plane, collector, dashboard):
        assert "FROM " in dockerfile
        assert "@sha256:" in dockerfile


def test_docker_context_is_default_deny_for_local_trace_safety():
    dockerignore = (ROOT / ".dockerignore").read_text().splitlines()
    rules = [line for line in dockerignore if line and not line.startswith("#")]
    assert rules[0] == "**"
    assert "!harness/cacheeconomics/**" in rules
    assert "!services/control_plane/src/**" in rules
    assert "!collectors/src/**" in rules
    assert "!apps/dashboard/app.js" in rules
    assert not any("fixtures" in rule for rule in rules)


def test_production_compose_uses_immutable_image_inputs_and_hardening():
    compose = (ROOT / "deploy/compose.production.yml").read_text()
    assert "CACHEECONOMICS_API_IMAGE:?" in compose
    assert "CACHEECONOMICS_WORKER_IMAGE:?" in compose
    assert "CACHEECONOMICS_DASHBOARD_IMAGE:?" in compose
    assert compose.count('cap_drop: ["ALL"]') == 4
    assert compose.count("read_only: true") == 4
    assert "CACHEECONOMICS_DASHBOARD_ALLOW_DEVELOPMENT_TOKEN: \"false\"" in compose


def test_production_image_validator_rejects_mutable_tags():
    script = ROOT / "deploy/scripts/validate_production_images.sh"
    environment = os.environ.copy()
    environment.update(
        CACHEECONOMICS_API_IMAGE="example.invalid/cacheeconomics-api:production",
        CACHEECONOMICS_WORKER_IMAGE="example.invalid/cacheeconomics-worker:production",
        CACHEECONOMICS_DASHBOARD_IMAGE="example.invalid/cacheeconomics-dashboard:production",
    )
    result = subprocess.run(
        ["sh", str(script)],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "must be an image reference ending in @sha256" in result.stderr


def test_production_image_validator_accepts_digest_references():
    script = ROOT / "deploy/scripts/validate_production_images.sh"
    digest = "a" * 64
    environment = os.environ.copy()
    environment.update(
        CACHEECONOMICS_API_IMAGE=f"ghcr.io/example/api@sha256:{digest}",
        CACHEECONOMICS_WORKER_IMAGE=f"ghcr.io/example/worker@sha256:{digest}",
        CACHEECONOMICS_DASHBOARD_IMAGE=f"ghcr.io/example/dashboard@sha256:{digest}",
    )
    result = subprocess.run(
        ["sh", str(script)],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "digest-qualified" in result.stdout


def test_release_requires_approval_and_promotes_scanned_attested_images():
    release = (ROOT / ".github/workflows/release.yml").read_text()
    assert "environment: staging" in release
    assert "environment: production" in release
    assert 'GITHUB_REF" != "refs/heads/main"' in release
    assert 'CONFIRMATION" != "PROMOTE"' in release
    assert "aquasecurity/trivy-action@ed142fd0673e97e23eac54620cfb913e5ce36c25" in release
    assert "actions/attest@1e69f48acb82d1966a394da916b4c1698aa569d6" in release
    assert "gh attestation verify" in release
    assert "imagetools create" in release


def test_workflow_actions_are_pinned_to_full_commit_hashes():
    import re

    for path in sorted((ROOT / ".github/workflows").glob("*.yml")):
        action_refs = re.findall(r"uses:\s+[^\s@]+@([^\s#]+)", path.read_text())
        assert action_refs, path
        assert all(re.fullmatch(r"[0-9a-f]{40}", ref) for ref in action_refs), path
