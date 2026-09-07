from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_google_bootstrap_has_narrow_workload_identity_and_no_secret_values():
    terraform = (ROOT / "deploy/gcp/terraform/main.tf").read_text()
    versions = (ROOT / "deploy/gcp/terraform/versions.tf").read_text()

    assert 'version = "8.1.0"' in versions
    assert "assertion.repository == '${var.github_repository}'" in terraform
    assert "assertion.ref == 'refs/heads/main'" in terraform
    assert "roles/iam.workloadIdentityUser" in terraform
    assert "roles/iam.serviceAccountUser" in terraform
    assert "roles/owner" not in terraform
    assert "roles/editor" not in terraform
    assert terraform.count('role      = "roles/secretmanager.secretAccessor"') == 1
    assert "google_secret_manager_secret_version" not in terraform
    assert "private_key" not in terraform


def test_terraform_uses_an_external_backend_without_committing_its_values():
    versions = (ROOT / "deploy/gcp/terraform/versions.tf").read_text()
    backend_example = (
        ROOT / "deploy/gcp/terraform/backend.gcs.tfbackend.example"
    ).read_text()
    gitignore = (ROOT / ".gitignore").read_text().splitlines()

    assert 'backend "gcs" {}' in versions
    assert 'bucket = "replace-with-private-versioned-state-bucket"' in backend_example
    assert 'prefix = "staging/terraform"' in backend_example
    assert "*.tfbackend" in gitignore
    assert "!*.tfbackend.example" in gitignore


def test_google_bootstrap_separates_runtime_identities_and_secrets():
    terraform = (ROOT / "deploy/gcp/terraform/main.tf").read_text()

    for account in ("api", "dashboard", "migrate", "scheduler", "worker"):
        assert re.search(rf"\b{account}\s+=", terraform)
    for secret in (
        "app_database_url",
        "migration_database_url",
        "worker_database_url",
        "metrics_bearer_token",
    ):
        assert re.search(rf"\b{secret}\s+=", terraform)

    assert 'account = "api"\n      secret  = "app_database_url"' in terraform
    assert 'account = "migrate"\n      secret  = "migration_database_url"' in terraform
    assert 'account = "worker"\n      secret  = "worker_database_url"' in terraform


def test_github_staging_deploy_uses_short_lived_identity_and_immutable_images():
    workflow = (ROOT / ".github/workflows/deploy-gcp-staging.yml").read_text()

    assert "environment: staging" in workflow
    assert "github.ref == 'refs/heads/main'" in workflow
    assert "inputs.confirmation == 'DEPLOY STAGING'" in workflow
    assert "id-token: write" in workflow
    assert "google-github-actions/auth@" in workflow
    assert "workload_identity_provider:" in workflow
    assert "service_account:" in workflow
    assert "credentials_json" not in workflow
    assert "service_account_key" not in workflow
    assert "vars.CACHEECONOMICS_SERVICE_PREFIX || 'cacheeconomics'" in workflow
    assert workflow.count("docker/build-push-action@") == 3
    assert workflow.count("aquasecurity/trivy-action@") == 3
    assert workflow.count("actions/attest@") == 3
    assert workflow.count("@${{ steps.") >= 6
    assert "sh deploy/gcp/deploy_staging.sh" in workflow


def test_staging_deploy_is_bounded_migrates_first_and_keeps_trace_export_off():
    script = (ROOT / "deploy/gcp/deploy_staging.sh").read_text()

    assert script.index('jobs execute "$migrate_job"') < script.index(
        'run deploy "$api_service"'
    )
    assert script.count("--max=1") == 2
    assert script.count("--min=0") == 2
    assert "--once" in script
    assert "--max-retries=0" in script
    assert "--schedule='0 * * * *'" in script
    assert "--oauth-service-account-email" in script
    assert "@sha256:[0-9a-f]{64}$" in script
    assert "OTEL_TRACES_EXPORTER=none" in script
    assert "OTEL_METRICS_EXPORTER=none" in script
    assert "OTEL_LOGS_EXPORTER=none" in script
    assert "CACHEECONOMICS_DASHBOARD_ALLOW_DEVELOPMENT_TOKEN=false" in script
    assert "CACHEECONOMICS_SERVICE_PREFIX must be 3-20" in script
    assert "CACHEECONOMICS_DASHBOARD_REDIRECT_URI=${dashboard_url}/" in script
    assert '"${api_url}/readyz"' in script
    assert '"${dashboard_url}/api/v1/dashboard/config"' in script


def test_staging_deploy_rejects_mutable_images_before_cloud_access():
    script = ROOT / "deploy/gcp/deploy_staging.sh"
    environment = os.environ.copy()
    environment.update(
        GCP_PROJECT_ID="portfolio-staging",
        GCP_REGION="europe-west1",
        AUTH0_DOMAIN="tenant.example.test",
        AUTH0_AUDIENCE="cacheeconomics-api",
        AUTH0_CLIENT_ID="public-client",
        API_IMAGE="example.test/api:latest",
        WORKER_IMAGE="example.test/worker:latest",
        DASHBOARD_IMAGE="example.test/dashboard:latest",
    )

    result = subprocess.run(
        ["sh", str(script)],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "immutable @sha256 digest" in result.stderr


def test_staging_shell_and_neon_bootstrap_fail_closed():
    deploy_script = ROOT / "deploy/gcp/deploy_staging.sh"
    syntax = subprocess.run(
        ["sh", "-n", str(deploy_script)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr

    neon = (ROOT / "deploy/gcp/bootstrap_neon_roles.sql").read_text()
    assert "current_database() = 'cacheeconomics'" in neon
    assert neon.count("\\password cacheeconomics_") == 3
    assert neon.count("\\getenv cacheeconomics_") == 3
    assert "cacheeconomics_passwords_from_environment" in neon
    assert "REVOKE neon_superuser" not in neon
    assert "managed owner cannot revoke" in neon
    assert "runtime database roles are over-privileged" in neon
    assert "runtime roles must not inherit neon_superuser" in neon
    assert "rolsuper OR rolcreatedb OR rolcreaterole OR rolinherit OR rolbypassrls" in neon
    assert "ALTER ROLE cacheeconomics_migrator NOSUPERUSER" not in neon
    assert "dev-only" not in neon
    assert "PASSWORD '" not in neon

    assert ".neon" in (ROOT / ".gitignore").read_text().splitlines()


def test_provider_runbook_does_not_claim_a_live_or_free_guaranteed_system():
    runbook = (ROOT / "deploy/gcp/README.md").read_text()

    assert "not a zero-cost guarantee" in runbook
    assert "not a production SLA" in runbook
    decision = (ROOT / "docs/deployment-decision-record.md").read_text()
    assert "await the first deployment" in " ".join(decision.split())
    assert "terraform init -migrate-state -force-copy" in runbook
    assert "OpenTelemetry export disabled" in runbook
    assert "Do not ingest real customer traces" in runbook
