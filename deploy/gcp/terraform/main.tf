locals {
  labels = merge(
    {
      application = "cacheeconomics"
      environment = "staging"
      managed_by  = "terraform"
    },
    var.labels,
  )

  services = toset([
    "artifactregistry.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "cloudscheduler.googleapis.com",
    "iam.googleapis.com",
    "iamcredentials.googleapis.com",
    "logging.googleapis.com",
    "monitoring.googleapis.com",
    "run.googleapis.com",
    "secretmanager.googleapis.com",
    "sts.googleapis.com",
  ])

  runtime_accounts = {
    api       = "Hosted API runtime"
    dashboard = "Dashboard runtime"
    migrate   = "One-shot database migration"
    scheduler = "Cloud Scheduler job trigger"
    worker    = "One-shot analysis worker"
  }

  secrets = {
    app_database_url       = "Restricted application PostgreSQL URL"
    metrics_bearer_token   = "Bearer token for the protected metrics endpoint"
    migration_database_url = "Migration-owner PostgreSQL URL"
    worker_database_url    = "Restricted worker PostgreSQL URL"
  }

  secret_access = {
    api_database = {
      account = "api"
      secret  = "app_database_url"
    }
    api_metrics = {
      account = "api"
      secret  = "metrics_bearer_token"
    }
    migrate_database = {
      account = "migrate"
      secret  = "migration_database_url"
    }
    worker_database = {
      account = "worker"
      secret  = "worker_database_url"
    }
    worker_metrics = {
      account = "worker"
      secret  = "metrics_bearer_token"
    }
  }

  deployer_project_roles = toset([
    "roles/artifactregistry.writer",
    "roles/cloudscheduler.admin",
    "roles/run.admin",
    "roles/serviceusage.serviceUsageConsumer",
  ])
}

resource "google_project_service" "required" {
  for_each = local.services

  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}

resource "google_artifact_registry_repository" "containers" {
  location      = var.region
  repository_id = var.service_prefix
  description   = "Immutable cacheeconomics staging containers"
  format        = "DOCKER"
  labels        = local.labels

  depends_on = [google_project_service.required]
}

resource "google_service_account" "runtime" {
  for_each = local.runtime_accounts

  account_id   = "${var.service_prefix}-${each.key}"
  display_name = each.value
}

resource "google_service_account" "deployer" {
  account_id   = "${var.service_prefix}-deployer"
  display_name = "GitHub Actions staging deployer"
}

resource "google_secret_manager_secret" "runtime" {
  for_each = local.secrets

  secret_id = "${var.service_prefix}-${replace(each.key, "_", "-")}"
  labels    = local.labels

  replication {
    auto {}
  }

  depends_on = [google_project_service.required]
}

resource "google_secret_manager_secret_iam_member" "runtime" {
  for_each = local.secret_access

  project   = var.project_id
  secret_id = google_secret_manager_secret.runtime[each.value.secret].secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.runtime[each.value.account].email}"
}

resource "google_project_iam_member" "deployer" {
  for_each = local.deployer_project_roles

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.deployer.email}"
}

resource "google_service_account_iam_member" "deployer_can_use_runtime" {
  for_each = google_service_account.runtime

  service_account_id = each.value.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.deployer.email}"
}

resource "google_iam_workload_identity_pool" "github" {
  workload_identity_pool_id = "${var.service_prefix}-github"
  display_name              = "cacheeconomics GitHub Actions"

  depends_on = [google_project_service.required]
}

resource "google_iam_workload_identity_pool_provider" "github" {
  workload_identity_pool_id          = google_iam_workload_identity_pool.github.workload_identity_pool_id
  workload_identity_pool_provider_id = "main"
  display_name                       = "Trusted main-branch workflow"
  attribute_condition                = "assertion.repository == '${var.github_repository}' && assertion.ref == 'refs/heads/main'"
  attribute_mapping = {
    "google.subject"       = "assertion.sub"
    "attribute.repository" = "assertion.repository"
    "attribute.ref"        = "assertion.ref"
  }

  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }
}

resource "google_service_account_iam_member" "github_deployer" {
  service_account_id = google_service_account.deployer.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github.name}/attribute.repository/${var.github_repository}"
}
