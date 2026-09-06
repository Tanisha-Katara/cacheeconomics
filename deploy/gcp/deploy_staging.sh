#!/bin/sh
set -eu

# Deploy only already-built, digest-qualified images. Database credentials and
# the metrics token are read by Cloud Run from Secret Manager, never from this
# process environment.

require() {
    value=$1
    name=$2
    if [ -z "$value" ]; then
        echo "$name is required" >&2
        exit 2
    fi
}

require "${GCP_PROJECT_ID:-}" GCP_PROJECT_ID
require "${GCP_REGION:-}" GCP_REGION
require "${AUTH0_DOMAIN:-}" AUTH0_DOMAIN
require "${AUTH0_AUDIENCE:-}" AUTH0_AUDIENCE
require "${AUTH0_CLIENT_ID:-}" AUTH0_CLIENT_ID
require "${API_IMAGE:-}" API_IMAGE
require "${WORKER_IMAGE:-}" WORKER_IMAGE
require "${DASHBOARD_IMAGE:-}" DASHBOARD_IMAGE

for image in "$API_IMAGE" "$WORKER_IMAGE" "$DASHBOARD_IMAGE"
do
    if ! printf '%s' "$image" | grep -Eq '@sha256:[0-9a-f]{64}$'; then
        echo "every image must end in an immutable @sha256 digest" >&2
        exit 2
    fi
done

if ! printf '%s' "$AUTH0_DOMAIN" | grep -Eq '^[A-Za-z0-9][A-Za-z0-9.-]*[A-Za-z0-9]$'; then
    echo "AUTH0_DOMAIN must be a hostname without a scheme or path" >&2
    exit 2
fi

prefix=${CACHEECONOMICS_SERVICE_PREFIX:-cacheeconomics}
if ! printf '%s' "$prefix" | grep -Eq '^[a-z][a-z0-9-]{1,18}[a-z0-9]$'; then
    echo "CACHEECONOMICS_SERVICE_PREFIX must be 3-20 lowercase letters, numbers, or internal hyphens" >&2
    exit 2
fi

if ! command -v gcloud >/dev/null 2>&1; then
    echo "gcloud is required" >&2
    exit 2
fi

api_service="${prefix}-api"
dashboard_service="${prefix}-dashboard"
migrate_job="${prefix}-migrate"
worker_job="${prefix}-worker"
api_account="${prefix}-api@${GCP_PROJECT_ID}.iam.gserviceaccount.com"
dashboard_account="${prefix}-dashboard@${GCP_PROJECT_ID}.iam.gserviceaccount.com"
migrate_account="${prefix}-migrate@${GCP_PROJECT_ID}.iam.gserviceaccount.com"
worker_account="${prefix}-worker@${GCP_PROJECT_ID}.iam.gserviceaccount.com"
scheduler_account="${prefix}-scheduler@${GCP_PROJECT_ID}.iam.gserviceaccount.com"

app_database_secret="${prefix}-app-database-url"
migration_database_secret="${prefix}-migration-database-url"
worker_database_secret="${prefix}-worker-database-url"
metrics_secret="${prefix}-metrics-bearer-token"

issuer="https://${AUTH0_DOMAIN}/"
jwks_url="https://${AUTH0_DOMAIN}/.well-known/jwks.json"
authorization_endpoint="https://${AUTH0_DOMAIN}/authorize"
token_endpoint="https://${AUTH0_DOMAIN}/oauth/token"
allowed_hosts='["*.run.app"]'

common_env="^@^CACHEECONOMICS_ENVIRONMENT=production@CACHEECONOMICS_OIDC_ISSUER=${issuer}@CACHEECONOMICS_OIDC_AUDIENCE=${AUTH0_AUDIENCE}@CACHEECONOMICS_OIDC_JWKS_URL=${jwks_url}@CACHEECONOMICS_ALLOWED_HOSTS=${allowed_hosts}@CACHEECONOMICS_DASHBOARD_ALLOW_DEVELOPMENT_TOKEN=false@OTEL_TRACES_EXPORTER=none@OTEL_METRICS_EXPORTER=none@OTEL_LOGS_EXPORTER=none"

echo "Deploying and executing the database migration job"
gcloud run jobs deploy "$migrate_job" \
    --project="$GCP_PROJECT_ID" \
    --region="$GCP_REGION" \
    --image="$API_IMAGE" \
    --service-account="$migrate_account" \
    --command=alembic \
    --args=-c,/app/alembic.ini,upgrade,head \
    --set-secrets="CACHEECONOMICS_DATABASE_URL=${migration_database_secret}:latest" \
    --tasks=1 \
    --max-retries=0 \
    --task-timeout=600s \
    --quiet
gcloud run jobs execute "$migrate_job" \
    --project="$GCP_PROJECT_ID" \
    --region="$GCP_REGION" \
    --wait

echo "Deploying the authenticated API with one-instance staging limits"
gcloud run deploy "$api_service" \
    --project="$GCP_PROJECT_ID" \
    --region="$GCP_REGION" \
    --image="$API_IMAGE" \
    --service-account="$api_account" \
    --port=8000 \
    --cpu=1 \
    --memory=512Mi \
    --concurrency=40 \
    --timeout=60s \
    --min=0 \
    --max=1 \
    --ingress=all \
    --allow-unauthenticated \
    --set-env-vars="$common_env" \
    --set-secrets="CACHEECONOMICS_DATABASE_URL=${app_database_secret}:latest,CACHEECONOMICS_METRICS_BEARER_TOKEN=${metrics_secret}:latest" \
    --quiet

api_url=$(gcloud run services describe "$api_service" \
    --project="$GCP_PROJECT_ID" \
    --region="$GCP_REGION" \
    --format='value(status.url)')

echo "Deploying the same-origin dashboard proxy"
gcloud run deploy "$dashboard_service" \
    --project="$GCP_PROJECT_ID" \
    --region="$GCP_REGION" \
    --image="$DASHBOARD_IMAGE" \
    --service-account="$dashboard_account" \
    --port=8080 \
    --cpu=1 \
    --memory=256Mi \
    --concurrency=80 \
    --timeout=30s \
    --min=0 \
    --max=1 \
    --ingress=all \
    --allow-unauthenticated \
    --set-env-vars="^@^DASHBOARD_API_ORIGIN=${api_url}@DASHBOARD_CONNECT_SRC='self' https://${AUTH0_DOMAIN}" \
    --quiet

dashboard_url=$(gcloud run services describe "$dashboard_service" \
    --project="$GCP_PROJECT_ID" \
    --region="$GCP_REGION" \
    --format='value(status.url)')

echo "Completing the dashboard's Authorization Code with PKCE configuration"
gcloud run services update "$api_service" \
    --project="$GCP_PROJECT_ID" \
    --region="$GCP_REGION" \
    --update-env-vars="^@^CACHEECONOMICS_DASHBOARD_OIDC_AUTHORIZATION_ENDPOINT=${authorization_endpoint}@CACHEECONOMICS_DASHBOARD_OIDC_TOKEN_ENDPOINT=${token_endpoint}@CACHEECONOMICS_DASHBOARD_OIDC_CLIENT_ID=${AUTH0_CLIENT_ID}@CACHEECONOMICS_DASHBOARD_REDIRECT_URI=${dashboard_url}/" \
    --quiet

echo "Deploying the bounded one-job worker"
gcloud run jobs deploy "$worker_job" \
    --project="$GCP_PROJECT_ID" \
    --region="$GCP_REGION" \
    --image="$WORKER_IMAGE" \
    --service-account="$worker_account" \
    --command=opentelemetry-instrument \
    --args=python,-m,cacheeconomics_control_plane.worker,--once \
    --set-env-vars="$common_env" \
    --set-secrets="CACHEECONOMICS_DATABASE_URL=${worker_database_secret}:latest,CACHEECONOMICS_METRICS_BEARER_TOKEN=${metrics_secret}:latest" \
    --tasks=1 \
    --max-retries=0 \
    --task-timeout=300s \
    --cpu=1 \
    --memory=512Mi \
    --quiet

gcloud run jobs add-iam-policy-binding "$worker_job" \
    --project="$GCP_PROJECT_ID" \
    --region="$GCP_REGION" \
    --member="serviceAccount:${scheduler_account}" \
    --role=roles/run.invoker \
    --quiet

worker_uri="https://${GCP_REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${GCP_PROJECT_ID}/jobs/${worker_job}:run"
if gcloud scheduler jobs describe "$worker_job" \
    --project="$GCP_PROJECT_ID" \
    --location="$GCP_REGION" >/dev/null 2>&1
then
    scheduler_action=update
else
    scheduler_action=create
fi

echo "Configuring the hourly scheduled worker trigger"
gcloud scheduler jobs "$scheduler_action" http "$worker_job" \
    --project="$GCP_PROJECT_ID" \
    --location="$GCP_REGION" \
    --schedule='0 * * * *' \
    --time-zone=Etc/UTC \
    --uri="$worker_uri" \
    --http-method=POST \
    --message-body='{}' \
    --oauth-service-account-email="$scheduler_account" \
    --oauth-token-scope=https://www.googleapis.com/auth/cloud-platform \
    --attempt-deadline=60s \
    --quiet

echo "Running unauthenticated health/configuration smoke checks"
curl --fail --silent --show-error --retry 6 --retry-all-errors \
    --retry-delay 5 "${api_url}/readyz" >/dev/null
dashboard_config=$(curl --fail --silent --show-error --retry 6 \
    --retry-all-errors --retry-delay 5 \
    "${dashboard_url}/api/v1/dashboard/config")
printf '%s' "$dashboard_config" | grep -F '"oidc_enabled":true' >/dev/null
printf '%s' "$dashboard_config" | grep -F "${dashboard_url}/" >/dev/null

echo "Staging services are deployed. Interactive Auth0 login still requires"
echo "the exact callback URL to be registered before sign-in testing."
echo "STAGING_DASHBOARD_URL=${dashboard_url}"
echo "STAGING_API_URL=${api_url}"
