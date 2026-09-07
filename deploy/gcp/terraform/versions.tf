terraform {
  required_version = ">= 1.8.0, < 2.0.0"

  # Backend values are supplied from the ignored backend.gcs.tfbackend file.
  # Keeping state in a private, versioned bucket avoids making one laptop the
  # source of truth and keeps deployment metadata out of Git.
  backend "gcs" {}

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "8.1.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}
