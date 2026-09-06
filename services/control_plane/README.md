# cacheeconomics control plane

This is the networked application layer. It is deliberately separate from the
installed `cacheeconomics` analysis package.

## What Phases 1–3 include

- External OIDC bearer-token verification using a configured issuer, audience,
  JWKS URL, and signature-algorithm allow-list
- Automatic local user record creation after a valid first sign-in
- Organizations and memberships
- Viewer, Analyst, Admin, and Owner permissions
- Organization-scoped sources
- One-time collector credentials stored as SHA-256 verifiers, never plaintext
- Revocation, optional expiry, and disabled-source checks
- Append-only application audit events
- PostgreSQL row-level security using transaction-local user/organization IDs
- Separate migration and runtime database roles
- Docker Compose development startup
- Prompt-free, bounded, idempotent event batches with per-item results
- PostgreSQL-backed scheduled analysis jobs, renewable leases, bounded retry,
  cancellation, recovery, and dead-letter state
- A separately restricted worker that runs the unchanged local analyzer
- Versioned analysis results and per-source ingestion health
- Bounded latency, time-to-first-token, volume, and normalized-error aggregates
- Prometheus-format metrics without tenant labels and allow-listed JSON logs
- Optional pinned OpenTelemetry instrumentation with exporters off by default
- A separate PKCE dashboard container in `apps/dashboard/`

There is no password database. The API trusts only tokens cryptographically
verified against the configured identity provider. A person must sign in once
before an owner can add their user ID to an organization.

The service still never accepts prompt or completion bodies. The network client
lives in `collectors/`, not in the installed `cacheeconomics` package.

Main hosted routes:

- `POST /v1/ingest/events` — collector-authenticated batch ingest
- `GET /v1/sources/{source_id}/health` — source counters and worker health
- `POST /v1/sources/{source_id}/jobs` — manual or scheduled analysis
- `GET /v1/jobs` and `POST /v1/jobs/{job_id}/cancel|retry`
- `GET /v1/sources/{source_id}/analyses/latest`
- `GET /v1/sources/{source_id}/operations` — bounded prompt-free aggregates
- `GET /v1/dashboard/config` — public non-secret browser configuration
- `GET /metrics` — operational scrape endpoint; bearer-protected in production

## Run the tests

Use Python 3.10 or newer. From the repository root:

```bash
python3.12 -m venv .venv-control-plane
.venv-control-plane/bin/pip install -r services/control_plane/requirements-test.lock
PYTHONPATH=services/control_plane/src:harness:collectors/src \
  .venv-control-plane/bin/pytest services/control_plane/tests -m "not postgres"
```

`requirements-runtime.lock` contains the API's required dependencies.
`requirements-test.lock` includes it and adds the test client and test runner;
`requirements-observability.lock` layers pinned optional instrumentation over
runtime and is what the production image installs.

The PostgreSQL test is intentionally skipped without
`TEST_POSTGRES_APP_URL`. The worker lease test also needs
`TEST_POSTGRES_WORKER_URL`. CI starts PostgreSQL, applies the real migration,
logs in as both restricted roles, and verifies tenant isolation plus atomic job
leasing.

## Start the local stack

Docker Compose requires real OIDC settings for authenticated routes. The
`.invalid` defaults let the health endpoint start but cannot authenticate a
person.

```bash
cp .env.example .env
# Replace the three identity settings in .env.
docker compose --env-file .env -f deploy/docker-compose.yml up --build
```

Then check:

```bash
curl http://127.0.0.1:8000/healthz
```

The hosted dashboard is at `http://127.0.0.1:8080`. The development stack
enables in-memory token entry because the repository cannot choose an identity
provider for you. Configure the PKCE values described in
[`apps/dashboard/README.md`](../../apps/dashboard/README.md) before production;
production settings reject development token entry.

Interactive API documentation is at `http://127.0.0.1:8000/docs` in development
only. Production disables it and refuses placeholder identity URLs, the sample
database credential, SQLite, and an unconfigured host allow-list.

## Authentication model

Human routes use:

```text
Authorization: Bearer <OIDC token>
X-Organization-ID: <selected organization UUID>
```

`/v1/me` and `/v1/organizations` do not need an organization header. All other
human routes derive their organization context from a membership check.

Collector routes use a `cec_...` bearer credential. The token is displayed
once. The database stores only its random prefix and SHA-256 digest. The
PostgreSQL authentication function can return only credential, organization,
and source IDs; it is not a general row-security bypass.

## Database safety

The migration role owns schema objects. The API and worker each connect as
different roles
with no superuser, role-creation, database-creation, inheritance, or RLS-bypass
capability. Organization-owned tables have row-level security enabled. A narrow
security-definer function can lease exactly one due job before its organization
is known. Every event read and result write happens only after the worker sets
that organization as its transaction context.

The passwords in `deploy/postgres/init/001_roles.sql` are explicitly for local
development and CI. Production must create the roles with secrets supplied by
its deployment platform.
