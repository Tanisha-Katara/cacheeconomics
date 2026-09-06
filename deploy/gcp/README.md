# Free-friendly GitHub staging deployment

This is the selected **staging** design:

- GitHub Actions builds, scans, attests, and deploys the images.
- Google Cloud Run runs the API, dashboard, migration, and scheduled worker.
- Google Artifact Registry stores the three deployed images.
- Google Secret Manager supplies database URLs and the metrics token.
- Neon supplies PostgreSQL.
- Auth0 supplies browser login and API access tokens. A GitHub social
  connection is optional.

GitHub Pages is not used because it cannot run the Python API. Nothing here
changes the installed `cacheeconomics` package: it still opens no sockets. The
networked API, collector, dashboard, and deployment remain separate artifacts.

This configuration is intended for a low-traffic portfolio staging system. It
is not a zero-cost guarantee and not a production SLA. Google Cloud requires a
billing account and can charge after its free allowances. Neon Free has limited
storage/recovery, and Auth0 Free has plan limits. Set billing budgets and alerts
before deploying, and upgrade or redesign these choices before serving real
customer data.

## What the repository will and will not do

Terraform creates only Google APIs, an Artifact Registry repository, empty
secret containers, separate service accounts, least-privilege bindings, and a
GitHub Workload Identity Federation trust restricted to this repository's
`main` branch. It deliberately creates no secret values.

The manual GitHub workflow builds and deploys only after the exact confirmation
`DEPLOY STAGING`. It uses short-lived Google credentials, not a downloaded
service-account key. It migrates the database before changing the API, limits
each service to one instance, schedules a bounded hourly `--once` worker, and
deploys digest-qualified images. Analysis can also be run manually for a demo;
the hourly default avoids paying a one-minute job minimum every five minutes.

The workflow does not create the Neon or Auth0 accounts, accept terms, add a
billing method, choose DNS, or register the Auth0 callback for you.

## 1. Create the free accounts

1. Create a Google Cloud project, attach billing, and create a small budget
   alert. A budget warns you; it is not a hard spending cap.
2. Create a Neon Free project and a database named `cacheeconomics`. Keep it in
   the same broad geography as Cloud Run when possible.
3. Create an Auth0 tenant, one API, and one **Single Page Application** using
   Authorization Code with PKCE and RS256.
   Use the API identifier as `AUTH0_AUDIENCE`. Do not create a client secret for
   the dashboard because it is a public browser client.
4. Optionally enable Auth0's GitHub social connection. That affects how people
   sign in; GitHub Actions still uses the separate Google workload identity.

Do not ingest real customer traces during setup. Use the repository's synthetic
fixture until the access and telemetry review is complete.

## 2. Bootstrap restricted Neon roles

Use Neon's **direct, non-pooler** owner URL with `psql`:

```bash
psql "OWNER_DIRECT_URL_FOR_CACHEECONOMICS" \
  --file deploy/gcp/bootstrap_neon_roles.sql
```

The script asks for three different generated passwords without putting them
in the file or shell history. Build three SQLAlchemy URLs, all with TLS enabled:

```text
postgresql+psycopg://cacheeconomics_migrator:PASSWORD@HOST/cacheeconomics?sslmode=require
postgresql+psycopg://cacheeconomics_app:PASSWORD@HOST/cacheeconomics?sslmode=require
postgresql+psycopg://cacheeconomics_worker:PASSWORD@HOST/cacheeconomics?sslmode=require
```

Keep the role passwords in a password manager only long enough to add their
URLs to Secret Manager. The migrations own database objects; the API and worker
roles are restricted and cannot bypass row-level security.

## 3. Bootstrap Google Cloud once

Install Terraform 1.8+ and the Google Cloud CLI, then authenticate locally.
Copy the example variables file without committing the copy:

```bash
cd deploy/gcp/terraform
cp terraform.tfvars.example terraform.tfvars
gcloud auth application-default login
terraform init
terraform plan
terraform apply
terraform output
```

Review the plan before applying it. The default Terraform state is local and is
gitignored. For a team or production environment, move it to an access-logged,
versioned remote backend before applying; do not commit state because it can
contain infrastructure metadata.

Add the secret versions from a trusted terminal. The command reads from stdin,
so the values do not appear as command arguments:

```bash
printf '%s' 'MIGRATOR_SQLALCHEMY_URL' | gcloud secrets versions add \
  cacheeconomics-migration-database-url --data-file=-
printf '%s' 'APP_SQLALCHEMY_URL' | gcloud secrets versions add \
  cacheeconomics-app-database-url --data-file=-
printf '%s' 'WORKER_SQLALCHEMY_URL' | gcloud secrets versions add \
  cacheeconomics-worker-database-url --data-file=-
openssl rand -hex 32 | gcloud secrets versions add \
  cacheeconomics-metrics-bearer-token --data-file=-
```

## 4. Configure GitHub

Create a GitHub environment named `staging`. Add a required reviewer if your
repository plan supports it. Add these **environment variables**, using the
Terraform outputs and Auth0 values:

| Variable | Value |
| --- | --- |
| `GCP_PROJECT_ID` | Google Cloud project ID |
| `GCP_REGION` | Terraform region, default `europe-west1` |
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | Terraform output of the same name |
| `GCP_DEPLOYER_SERVICE_ACCOUNT` | Terraform output of the same name |
| `AUTH0_DOMAIN` | Tenant hostname only, with no `https://` or path |
| `AUTH0_AUDIENCE` | Exact Auth0 API identifier |
| `AUTH0_CLIENT_ID` | Public dashboard application's client ID |
| `CACHEECONOMICS_SERVICE_PREFIX` | Optional; leave unset for `cacheeconomics`, or match the Terraform value exactly |

No long-lived Google credential belongs in GitHub Secrets.

Protect `main` and require the existing `ci` and `security` checks before
allowing deployment.

## 5. First deployment and Auth0 callback

From GitHub Actions, run **deploy-gcp-staging** on `main` and enter
`DEPLOY STAGING`. The job prints stable Cloud Run dashboard and API URLs.

In Auth0, set these values to the exact printed dashboard URL (including its
`https://` scheme and no wildcard):

- Allowed Callback URLs: `DASHBOARD_URL/`
- Allowed Web Origins: `DASHBOARD_URL`
- Allowed Logout URLs: `DASHBOARD_URL/`

Use Auth0's documented browser CORS setting if the selected tenant requires it
for `/oauth/token`. Then open the dashboard and complete sign-in. The first
authenticated person can create the demonstration organization through the
API flow; do not seed a fabricated customer identity.

## 6. Verify the staging system

The workflow already checks `/readyz` through both public routes. Complete and
record these checks before calling the environment deployed:

1. Sign in, create two organizations, and repeat the tenant-isolation test with
   the real Neon database.
2. Create a synthetic source and one-time credential. Run the separate
   collector against the synthetic LiteLLM fixture only.
3. Confirm accepted, duplicate, and rejected counts; run the scheduled worker;
   and confirm a completed analysis and withheld monetary figures.
4. Review Cloud Run request/error/latency charts and the allow-listed JSON logs.
   Inspect logs for identifiers and prompt-like content.
5. Keep OpenTelemetry export disabled. It is intentionally set to `none` until
   a staging trace redaction review proves that URLs and span attributes do not
   expose organization, source, job, or event identifiers.
6. Create an isolated `*_restore_test` Neon database/branch, pre-provision the
   three roles, and run `deploy/scripts/backup_restore_smoke.sh` exactly as the
   main runbook describes.
7. Record a rollback drill to the previous three image digests.

Cloud Run automatically supplies platform request counts, latency, errors,
container logs, and job execution state. The dashboard supplies exact
ingestion-health and analysis-job state. The application's protected
Prometheus endpoint is not automatically scraped by this staging definition;
connecting it to a backend and choosing evidence-based alert thresholds remain
explicit production work.

## Rollback and removal

Rollback uses the previous digest-qualified API, worker, and dashboard images.
Run the deployment script with those exact image values after reviewing schema
compatibility. Do not downgrade the database merely to match an old image.

To stop compute charges without deleting evidence, set the scheduler job to
paused and leave both services at minimum zero instances. Removing Google
resources with Terraform is destructive and does not remove Neon/Auth0 data;
review the exact plan and export any required evidence first.

## Known staging limitations

- Cloud Run's default domain is used; custom DNS is deferred.
- Service maximums are deliberately one instance, so this is not a capacity or
  high-availability claim.
- Neon Free recovery and resource limits are not a production backup policy.
- No source rate limiter exists yet; use synthetic/private staging traffic.
- Application Prometheus metrics and OpenTelemetry traces are not exported.
- No real uptime, savings, provider coverage, or capacity number is claimed
  until it is measured and recorded.
