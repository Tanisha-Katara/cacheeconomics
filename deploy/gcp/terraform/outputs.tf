output "artifact_registry" {
  description = "Container registry prefix used by the staging workflow."
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.containers.repository_id}"
}

output "deployer_service_account" {
  description = "Set this as the GCP_DEPLOYER_SERVICE_ACCOUNT GitHub environment variable."
  value       = google_service_account.deployer.email
}

output "secret_names" {
  description = "Secret containers that need operator-added versions before deployment."
  value       = { for key, secret in google_secret_manager_secret.runtime : key => secret.secret_id }
}

output "workload_identity_provider" {
  description = "Set this as the GCP_WORKLOAD_IDENTITY_PROVIDER GitHub environment variable."
  value       = google_iam_workload_identity_pool_provider.github.name
}
