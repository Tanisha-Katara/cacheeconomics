# Production implementation plan

The work is split into small phases so each step can be demonstrated and tested
before the next one depends on it. “Done” below means implemented and verified,
not merely drawn in a diagram.

## Phase 0 — Protect the boundary

**Status: complete in the current worktree**

Deliverables:

- Architecture decision keeping the installed package offline
- Prompt-free ingest event schema, version 1
- Complete analysis result schema, version 1
- Safe serializer that cannot expose a withheld dollar amount
- Initial architecture and security model
- Regression tests for schemas, HMAC identity, and figure release state

Exit check: existing offline behavior passes, no CLI format changes, and no
network imports enter the installed package.

## Phase 1 — Small secure control plane

**Status: implemented in the worktree; local API tests pass. The real PostgreSQL
isolation test is configured in CI and awaits a CI run.**

Build one end-to-end service slice:

- Separate FastAPI application
- PostgreSQL migrations for users, organizations, memberships, sources, ingest
  events, analyses, jobs, and audit events
- OIDC login boundary for people
- Source-scoped collector credentials stored as one-way hashes
- Viewer, Analyst, Admin, and Owner permissions enforced by the API
- PostgreSQL row-level security for a second tenant-isolation layer
- Docker Compose development environment

Exit check: two-organization tests prove that neither a person nor a collector
can read or write the other organization's data. The core package's socket-ban
test still passes.

## Phase 2 — Reliable ingestion and jobs

**Status: implemented in the worktree; 79 service/collector tests pass locally.
The real PostgreSQL row-isolation and atomic-lease tests are configured in CI
and await a CI run.**

- Implement the v1 ingest endpoint with schema validation, size limits,
  idempotency, batching, and useful rejection reasons
- Build the first live collector around the repository's existing LiteLLM
  callback; do not claim other live integrations yet
- Add explicit local upload for existing LiteLLM JSONL and normalized trace
  formats. Request-body and Claude Code hosted upload remain deferred; their
  existing local analysis commands are unchanged.
- Add scheduling, retries with limits, cancellation, and dead-letter handling
- Convert accepted events to the existing `TraceSet`/`Request` model and run the
  analyzer in a separate worker
- Record last success, last error category, lag, accepted/rejected event counts,
  and job duration for ingestion health

Exit check: recorded fixtures exercise collector → API → database → worker →
analysis result, including replay, partial failure, and recovery. No test sends
raw prompt text to the service.

Implemented details:

- The API accepts a configured-size batch and validates each event separately.
  Replaying the same source/event ID is a successful duplicate, not a second
  row or a second cost claim.
- PostgreSQL itself leases one due job with `FOR UPDATE SKIP LOCKED`. The worker
  then sets the leased organization as its RLS context before reading events or
  writing a result.
- Failed jobs retry with bounded exponential delay and move to a dead-letter
  state when attempts are exhausted. Running jobs have expiring leases so a
  replacement worker can recover abandoned work. Active workers renew their
  leases while a long synchronous analysis is still running.
- The separate collector keeps a local SQLite outbox. Only the collector has an
  HTTP client; the installed core package is unchanged and remains socket-free.
  Events pass a complete allow-list before entering the outbox, and authenticated
  HTTP requests never follow redirects.
- Each job has a fixed event cutoff and a configurable recent-history cap. A
  truncated analysis says so in its notes instead of implying full history.
- Source health records accepted, duplicate, and rejected event totals; ingest
  lag; last successful/error analysis; consecutive failures; and job duration.

## Phase 3 — Useful dashboard and observability

**Status: implemented**

- `apps/dashboard/` is a separate, dependency-free browser application with
  organization and source selection. It uses a same-origin Nginx API proxy and
  OIDC Authorization Code with PKCE. Access tokens stay in memory; development
  token entry is disabled by production configuration.
- The overview renders only the analysis contract's display values and release
  state. It does not reconstruct a withheld amount. It shows cache ratios,
  request coverage, exact source-health counters, and timestamps.
- Recommendations preserve the analyzer's order, mechanism, evidence class,
  action, quality risk, and withheld/draft/released state.
- A bounded aggregate endpoint supplies request volume, p50/p95 latency and
  time-to-first-token, normalized outcomes, and error categories. It returns no
  event payloads and labels truncated or invalid-event views as partial.
- Source and job screens connect deterministic health states to the relevant
  view. Run, cancel, and retry controls appear only for eligible roles, while
  the API remains the authorization boundary.
- The API and worker emit allow-listed JSON events. Prometheus-format API
  metrics use route templates and no tenant/user labels. Optional pinned
  OpenTelemetry instrumentation is present in the service image; exporters are
  off until a deployment explicitly configures them. Alert rules fire on any
  observed API 5xx or rejected ingest event. No latency SLO is guessed.

Exit check: a user can move from a health alert to the affected source and then
to an evidence-backed action without seeing data from another organization.
Empty, loading, partial, stale, error, and withheld states are demonstrated.

## Phase 4 — Evaluation, containers, and delivery

**Status: implemented in the worktree, except for deployment to a real staging
runtime. Local deterministic evaluations and the measured regression budget
pass. Provider-specific staging definitions now exist, while container builds,
PostgreSQL migration/restore exercises, registry scans, GitHub environment
approvals, and the first Cloud Run deployment still require execution.**

- Golden analyzer and recommendation fixtures
- Parser contract, malformed input, and schema compatibility tests
- RBAC matrix and tenant-isolation tests
- Migration upgrade/downgrade and backup/restore tests
- Ingestion throughput and dashboard performance budgets based on measured test
  results—not invented production numbers
- Separate production containers for API, worker, collector, and dashboard
- CI for tests, dependency review, secret scanning, software bill of materials,
  image scanning, and migration checks
- Staging deployment and an explicitly approved production deployment job

Implemented details:

- `evals/` locks the reviewed demo-trace result, monetary release behavior,
  versioned schemas, malformed prompt-free events, the complete RBAC matrix,
  the migration chain, container roles, and release safety controls.
- A synthetic local benchmark measures the full API ingestion route against an
  in-memory SQLite database, the 286-row analyzer fixture, and dashboard
  transfer size. The checked-in baseline is explicitly not a production load
  claim or SLO.
- The API, worker, collector, and dashboard have distinct non-root production
  image roles. Compose drops Linux capabilities, prevents privilege
  escalation, and uses read-only filesystems with narrow temporary mounts.
- CI exercises migration downgrade/re-upgrade and restores a PostgreSQL dump
  into a database whose server-reported name must end in `_restore_test`.
- Security CI runs dependency review, repository secret/configuration/
  vulnerability scanning, image scanning, and SBOM generation. Every external
  action is pinned to a full commit hash and Dependabot proposes updates.
- The manual release workflow publishes immutable commit-tagged staging images,
  scans them, creates signed build provenance, and only then moves the staging
  pointer. Production requires the protected GitHub environment, the exact
  `PROMOTE` confirmation, and a full staging commit SHA; it verifies provenance
  and promotes the same digest without rebuilding.

The selected low-traffic staging target is GitHub Actions, Google Cloud Run,
Artifact Registry, Secret Manager, Neon PostgreSQL, Auth0, Cloud Logging, and
Cloud Monitoring. Terraform bootstraps Google identities/resources, and a
manual approval-gated workflow deploys by digest. Cloud Trace is a future
candidate, not an implemented integration: trace export remains disabled until
a staging privacy review.

Exit check: a clean environment can build, migrate, seed a synthetic demo,
pass smoke tests, roll back the application, and restore a tested backup. The
repository automates the clean build/migration/service smoke, synthetic
end-to-end path, and isolated restore as separate checks; the phase remains
open until the combined procedure passes in CI and on the selected staging
platform.

## Phase 5 — Portfolio evidence

**Status: partially implemented. The provider-neutral synthetic demo, evidence
ledger, product tour, case study, selected staging design, provider runbook,
Terraform bootstrap, and approval-gated deployment workflow are in the
worktree. Browser screenshots, an actual recording, deployment execution, and
runtime evidence remain pending.**

- Final deployment-specific runbook and exercised incident/rollback procedure
- Deployed architecture and data-flow diagrams with the chosen services
- Deployment-specific threat-model review and security limitations
- Repeatable synthetic demo script and short demo recording
- Case study using clearly labelled recorded or synthetic data
- Screenshots and a concise README product tour

Exit check: every screenshot, metric, savings statement, provider claim, and
integration label can be traced to a fixture, test result, registry source, or
clearly marked synthetic example.

Implemented details:

- `demo/build_portfolio_packet.py` drives 286 synthetic rows through collector
  normalization, the real API, in-memory database ingestion, the leased worker,
  analysis, and dashboard aggregates. CI rebuilds the checked packet byte for
  byte.
- The flow proves accepted, duplicate, and rejected outcomes and asserts that
  no prompt-bearing field is stored. It supplies no invoice, so all monetary
  figures remain withheld.
- `demo/serve_portfolio_demo.py` binds only to loopback and serves the real
  dashboard against the read-only packet using a fixed non-secret demo token.
- The product tour, synthetic case study, evidence ledger, deployment decision
  record, provider runbook, and recording script state their evidence scope and
  remaining gaps.
- `deploy/gcp/terraform/` creates empty secret containers, separate runtime
  identities, Artifact Registry, and a repository/branch-restricted GitHub
  workload identity. It creates no secret values or third-party accounts.
- `.github/workflows/deploy-gcp-staging.yml` requires an exact manual
  confirmation, builds/scans/attests three images, migrates first, and deploys
  Cloud Run services/jobs by immutable digest with one-instance staging limits.

Remaining before Phase 5 is complete:

- Capture and hash real dashboard screenshots when browser capture is
  available, then record the scripted walkthrough.
- Create the selected cloud/identity/database accounts, apply the definitions,
  and replace design status with redacted runtime evidence.
- Run and record a staging incident/rollback drill and deployment-specific
  threat-model review. Exercise the Neon restore and Cloud Logging privacy
  review. Repository procedures alone are not exercise evidence.

## Decisions deliberately deferred

Custom DNS, a production database/service tier, a trace exporter, application
metrics scraping, rate limiting, and measured alert/SLO thresholds remain
deferred. Phase 2 deliberately uses PostgreSQL job rows and leases instead of
adding a second queue service. The synthetic Phase 4 benchmark is only a
regression signal; staging load and worker-saturation testing must determine
when that choice needs revisiting. None of these hosted choices changes the
offline package boundary.
