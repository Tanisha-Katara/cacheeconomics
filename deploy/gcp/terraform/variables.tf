variable "project_id" {
  description = "Existing Google Cloud project with billing enabled."
  type        = string
}

variable "region" {
  description = "Cloud Run and Artifact Registry region. Keep the services together."
  type        = string
  default     = "europe-west1"
}

variable "github_repository" {
  description = "Exact owner/repository allowed to deploy from its main branch."
  type        = string

  validation {
    condition     = can(regex("^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", var.github_repository))
    error_message = "github_repository must look like owner/repository."
  }
}

variable "service_prefix" {
  description = "Prefix for staging resources."
  type        = string
  default     = "cacheeconomics"

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{1,18}[a-z0-9]$", var.service_prefix))
    error_message = "service_prefix must be 3-20 lowercase letters, numbers, or internal hyphens."
  }
}

variable "labels" {
  description = "Extra labels applied to supported resources."
  type        = map(string)
  default     = {}
}
