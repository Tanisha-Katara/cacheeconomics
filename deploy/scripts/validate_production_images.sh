#!/bin/sh
# Refuse mutable tags in the production Compose input.
set -eu

: "${CACHEECONOMICS_API_IMAGE:?set CACHEECONOMICS_API_IMAGE}"
: "${CACHEECONOMICS_WORKER_IMAGE:?set CACHEECONOMICS_WORKER_IMAGE}"
: "${CACHEECONOMICS_DASHBOARD_IMAGE:?set CACHEECONOMICS_DASHBOARD_IMAGE}"

validate_image() {
  label=$1
  value=$2
  if ! printf '%s' "$value" | grep -Eq '^[^[:space:]]+@sha256:[0-9a-f]{64}$'; then
    echo "$label must be an image reference ending in @sha256:<64 lowercase hex characters>" >&2
    exit 2
  fi
}

validate_image CACHEECONOMICS_API_IMAGE "$CACHEECONOMICS_API_IMAGE"
validate_image CACHEECONOMICS_WORKER_IMAGE "$CACHEECONOMICS_WORKER_IMAGE"
validate_image CACHEECONOMICS_DASHBOARD_IMAGE "$CACHEECONOMICS_DASHBOARD_IMAGE"
echo "production image references are digest-qualified"
