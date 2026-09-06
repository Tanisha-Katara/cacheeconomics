# Deployment runbook

This runbook covers the API, PostgreSQL schema, analysis worker, hosted
dashboard image, operational telemetry, evaluation gates, image delivery, and
backup/restore verification currently in the repository. GitHub Actions,
Google Cloud Run, Neon, and Auth0 are now the selected portfolio-staging stack.
The exact provider setup is in [`deploy/gcp/README.md`](../deploy/gcp/README.md).
These are deployment definitions, not a claim that a public environment exists.

## Before deployment

1. Create three different PostgreSQL logins: migration owner, API runtime, and
   worker runtime. Neither runtime role may own tables or bypass row security.
2. Put database passwords and OIDC configuration in the deployment secret
   manager. Do not reuse the development values in `deploy/postgres/init`.
3. Configure an HTTPS OIDC issuer, audience, JWKS URL, and the exact public API
   hosts. Production settings reject placeholders, SQLite, wildcard hosts, and
   development database credentials.
4. Back up the database and record the current application image and Alembic
   revision.
5. Run the offline package suite separately from the service suite so hosted
   dependencies cannot hide an accidental dependency in the local package.
6. Register the dashboard as a public OIDC client using Authorization Code with
   PKCE. Configure its exact HTTPS redirect URI. The identity provider must
   allow browser token exchange from the dashboard origin.
7. Set all four `CACHEECONOMICS_DASHBOARD_OIDC_*` values. Set
   `DASHBOARD_CONNECT_SRC` to `'self'` plus only the identity provider's exact
   origin. Production refuses partial configuration and development token
   entry.
8. Store the metrics bearer token in the secrets manager. If OTLP is enabled,
   choose its endpoint and authentication through the deployment platform; do
   not put telemetry secrets in the image or repository.
9. In GitHub, protect `main`, require the `ci` and `security` workflows, and
   create `staging` and `production` environments. Give `production` required
   reviewers. The `PROMOTE` text check is an additional guard, not a substitute
   for an independent reviewer.
10. For the selected staging target, complete the Google/Neon/Auth0 checklist in
    `deploy/gcp/README.md`. Record the actual PostgreSQL version, region,
    encryption/recovery settings, secret versions, Auth0 settings, service
    URLs, health checks, and rollback digests. The free-plan labels are not
    evidence of those runtime facts.

## Build and image promotion

The manual `publish-images` workflow has two operations:

1. `publish-staging` builds four role images—API, worker, collector, and
   dashboard—from the selected `main` commit. Each image receives an immutable
   `sha-<commit>` tag.
2. The workflow scans that digest for high/critical fixed vulnerabilities,
   secrets, and misconfiguration. It creates signed GitHub/Sigstore build
   provenance and only then moves the mutable `staging` tag.
3. `promote-production` requires a full 40-character staging commit SHA, the
   exact confirmation `PROMOTE`, and approval from the protected `production`
   environment. It verifies the attestation and points `production` at the same
   manifest digest. It does not rebuild.

Tags are for operator convenience. Put digest-qualified image references in
`CACHEECONOMICS_API_IMAGE`, `CACHEECONOMICS_WORKER_IMAGE`, and
`CACHEECONOMICS_DASHBOARD_IMAGE` when rendering
`deploy/compose.production.yml`. Run
`deploy/scripts/validate_production_images.sh` first; it refuses mutable tags.

## Release order

1. When publishing the standalone Python distributions, publish the exact
   `cacheeconomics` core version pinned by the control plane and collector
   before publishing either hosted distribution. CI builds all three wheels
   together and imports them outside the checkout to prove that the contracts
   module is present. The container images build that same local release set.
2. Resolve the approved API, worker, and dashboard image digests. Retain the
   previous digests as the rollback set.
3. Run `alembic -c /app/alembic.ini upgrade head` once with the migration role.
4. Start the API with the restricted API database URL.
5. Check `/healthz`, then `/readyz` through the same host/routing path clients
   use. Liveness only proves the process is running; readiness checks the DB.
6. Start the dedicated worker image with the restricted worker database URL.
   It has no HTTP health check; lease renewal and source/job health are its
   operational signals.
7. Start the dashboard behind HTTPS. Confirm its `/api/` route reaches the API
   with `Host: api` (or another explicitly allowed internal host). Confirm the
   content-security policy contains only the chosen identity origin.
8. Sign in through OIDC and verify the user sees only organizations to which
   they belong. A page refresh should require sign-in again.
9. Create one source and source credential, send a synthetic prompt-free event,
   and confirm its job moves from `queued` to `succeeded` and appears at the
   latest-analysis endpoint and dashboard.

The collector image is run near the opted-in data source, not inside the hosted
control plane. Mount its trace, outbox directory, token file, and HMAC key file
read-only wherever possible; only the outbox needs write access.

## Backup and restore exercise

Before production and at the chosen recovery-test interval:

1. Create an isolated database whose actual name ends in `_restore_test`.
2. Pre-provision `cacheeconomics_migrator`, `cacheeconomics_app`, and
   `cacheeconomics_worker` on the target cluster with the production role
   flags, and grant their database-level `CONNECT` access. Archive restores
   cannot create missing managed-service roles or restore database-level ACLs.
3. Give the backup process read access to the source and ownership-restoring
   administrative access only to the isolated target. Also provide target URLs
   for the restricted app and worker roles. Keep all URLs in the deployment
   secret store.
4. Run `deploy/scripts/backup_restore_smoke.sh` with
   `CACHEECONOMICS_ALLOW_DESTRUCTIVE_RESTORE_TEST=yes`,
   `RESTORE_APP_DATABASE_URL`, and `RESTORE_WORKER_DATABASE_URL`.
5. The script checks every database name with PostgreSQL itself, preserves
   object owners and ACLs, recreates only the target's `public` schema, and
   compares the migration revision plus key table counts. It then verifies the
   least-privilege table/function grants, function ownership, runtime role
   flags, and real app/worker logins.
6. Record the workflow result and duration, then destroy the isolated restore
   database using the platform's normal database lifecycle control.

Never point the restore URL at a shared, staging, or production database. The
suffix and explicit-consent checks reduce operator error; they do not make an
incorrect credential safe.

## Health checks

- Rising rejected-event counts mean a collector/schema mismatch.
- Rising duplicate counts with stable accepted counts usually mean replay; the
  database remains idempotent, but the collector outbox should be checked.
- Ingest lag is the difference between collector `sent_at` and API receipt. It
  is clamped at zero for future-skewed clocks.
- Consecutive failures, last error category, dead-letter jobs, and job duration
  are the worker signals available in the dashboard. The UI calls an analysis
  stale only when a newer event was received after the result was created; it
  does not invent an age threshold.
- `CACHEECONOMICS_ANALYSIS_MAX_EVENTS` bounds a job's in-memory history (1,000
  by default). If it is reached, the analysis result explicitly says older
  events were omitted. Raise it only after measuring worker memory and duration.

## Metrics, logs, and traces

- Scrape `/metrics` with `Authorization: Bearer <metrics token>`. Production
  refuses to start with metrics enabled and no token. Sum counters and
  histograms across API replicas; the in-process registry resets on restart.
- Load `deploy/observability/alerts.yml` into a Prometheus-compatible rule
  evaluator. The two alerts mean exactly what they say: at least one API 5xx or
  rejected ingest event was observed in the rolling window.
- Route stdout JSON logs to the selected log backend. Search by request ID and
  normalized event name. Do not change the formatter to interpolate arbitrary
  exceptions or request objects.
- In the selected Cloud Run staging design, stdout reaches Cloud Logging and
  Cloud Run exposes platform request/error/latency and job signals to Cloud
  Monitoring. The protected application `/metrics` endpoint is not scraped by
  that definition; do not claim those Prometheus rules are live alerts.
- The image starts through `opentelemetry-instrument`, but traces, metrics, and
  logs exporters default to `none`. To enable traces, explicitly set
  `OTEL_TRACES_EXPORTER=otlp`, `OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf`, the
  approved `OTEL_EXPORTER_OTLP_ENDPOINT`, and backend authentication. Review a
  staging trace for unwanted attributes before enabling production export.
- The repository intentionally has no latency alert value. Establish it from a
  measured staging baseline and an approved service objective.

## Incident actions

- Revoke a suspected collector credential immediately; it is source-scoped and
  a replacement can be created without changing other sources.
- Disable a source to stop its collector authentication.
- Cancel queued/running jobs through the API. A late worker result is discarded
  after cancellation.
- Inspect normalized error categories only. Do not copy raw provider errors,
  request bodies, authorization headers, or segment fingerprints into logs or
  tickets.
- Fix the cause of a dead-letter job, then use the retry endpoint. Retry resets
  its attempt counter but does not delete the event history.

## Rollback

Prefer rolling back the application by restoring the previous API, worker, and
dashboard digests while leaving a forward-compatible schema in place. Run an
Alembic downgrade only after checking that no rows would be discarded: the
Phase 2 downgrade removes source health, job payload/lease fields, and analysis
job provenance. Restore from a backup that passed the isolated restore exercise
if schema rollback loses data. Rehearse this on staging before production.

The dashboard is stateless. Roll it back independently to the previous static
image as long as that image understands the deployed API contract. Rolling back
the dashboard does not change stored events or analysis results.
