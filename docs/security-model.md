# Security model

This document is the security baseline for the hosted application. The current
worktree implements identity verification, organizations, RBAC, source credentials,
audit events, database row isolation, prompt-free ingestion, a restricted
analysis worker, a PKCE dashboard, and privacy-safe operational telemetry.
GitHub, Google Cloud Run, Neon, and Auth0 are selected for portfolio staging;
those definitions have not yet been applied. This is still not a claim of an
independent security audit or compliance certification.

## Assets to protect

- Organization membership, roles, and source credentials
- Usage counts, timing, model names, costs, and recommendations
- Keyed segment fingerprints, which are still sensitive metadata
- Database backups, audit records, and operational logs

Raw prompts, completions, messages, tool arguments, and raw provider error text
are deliberately outside the hosted data model.

## Identities and roles

There are two identity types:

- **People** sign in through an OpenID Connect provider. A person may belong to
  more than one organization.
- **Collectors** use a revocable, source-scoped credential. They can ingest into
  one source in one organization and cannot use dashboard APIs.

The Phase 1 RBAC model uses four organization roles:

| Role | Read organization/sources | Manage sources/credentials | Manage members | Reserved later-phase action |
| --- | --- | --- | --- | --- |
| Viewer | Yes | No | No | None |
| Analyst | Yes | No | No | Run/cancel/retry jobs |
| Admin | Yes | Yes | Yes | Run/cancel/retry jobs |
| Owner | Yes | Yes | Yes | Run/cancel/retry jobs; delete organization |

The API, not the browser, enforces every implemented permission. Organization
deletion remains reserved in the RBAC model but no such endpoint exists yet.

## Tenant isolation

An authenticated session or collector credential supplies the organization
context. Ingest JSON cannot supply or override it. Every organization-owned row
includes `organization_id`, and PostgreSQL row-level security provides a second
check beneath application authorization.

The existing trace field named `tenant` describes an analyzed workload. It is
not a security boundary and must be stored as `workload_tenant` in the hosted
application to avoid confusion.

## Privacy boundary

The version 1 ingest schema is allow-listed and rejects unknown fields. It can
carry:

- token counts and cache counters;
- request, first-token, and completion times;
- normalized success/error status;
- model, target, agent, and session labels; and
- locally generated keyed HMAC segment IDs.

It cannot carry prompt or completion bodies. Collectors must use a secret HMAC
key that is unique per organization and is not uploaded. Plain hashes are not
sufficient because common prompt text can be guessed. Workload-tenant identity
is also included in each fingerprint so two isolated workloads do not become
joinable merely because they share text. HMAC key rotation starts a new
comparison namespace rather than pretending old and new fingerprints are equal.

## Main threats and control status

| Threat | Implemented now | Still required |
| --- | --- | --- |
| Cross-organization read or write | Server-derived organization context, scoped API queries, PostgreSQL row-level security, API tests, complete RBAC evaluation, restricted worker role and atomic lease function | Require the real PostgreSQL isolation job in branch protection and repeat it on Neon staging |
| Stolen collector secret | One-way verifier, one-time display, source scope, optional expiry, create/revoke rotation, HTTPS enforcement, and disabled redirects | Per-source rate limits before public exposure |
| Prompt leakage | Prompt-free schema, collector/API compatibility evaluation, complete collector-side field allow-list before outbox persistence, local HMAC design, body-size limit, end-to-end rejection tests, allow-listed JSON logs, repository secret scanning, and manual spans that suppress exception text | Deployment-specific collector and telemetry review before public exposure |
| Forged or repeated events | Source-scoped credentials and idempotent `(source_id, event_id)` writes | Per-source rate limits before public exposure |
| Lost or duplicated background work | Atomic PostgreSQL leases with active renewal, unique analysis-per-job, bounded retries, dead-letter state, cancellation check, and deterministic job-health UI | Measure worker saturation on the selected staging platform before setting a saturation threshold |
| Inflated savings claims | Existing reconciliation gates, withheld figures, evidence class, versioned result schema, golden analyzer evaluation, and a dashboard that renders contract display/release fields without reading hidden amounts | Browser screenshot regression coverage in Phase 5 |
| Sensitive operational logs | Request IDs, no credential values in audit details, allow-listed scalar JSON fields, normalized error types, no tenant labels in metrics, sanitized HTTP-header instrumentation, and exception-free manual spans | Review Cloud Logging output and trace attributes before enabling trace export |
| Compromised dependency or image | Separate pinned service dependencies; non-root, read-only role containers; full-SHA CI action pins; dependency and image scans; retained SBOMs; signed provenance; digest-preserving promotion | Configure required checks and protected GitHub environments; review and merge Dependabot updates; add registry retention policy |
| Destructive admin action | Role/source/credential audit events and last-owner protection | Step-up authentication and recoverable organization deletion before that endpoint is added |

## Encryption and secrets

- Use TLS for every network connection.
- Use the deployment platform's encryption for databases, backups, and object
  storage.
- Keep hosted OIDC secrets, database credentials, credential-verification keys,
  and signing keys in a secrets manager—not source control or container images.
- Keep the segment HMAC key in the customer's collector environment. Do not
  upload it to the hosted service.
- Rotate credentials and document emergency revocation before production use.

## Selected staging platform threats

| Platform risk | Staging control | Remaining limitation |
| --- | --- | --- |
| A stolen GitHub deploy credential changes the service | GitHub exchanges an OIDC token for short-lived Google credentials; trust is restricted to the exact repository and `main`; a protected `staging` environment and literal confirmation gate deployment | GitHub branch/environment protection must be configured and verified outside this repository |
| One compromised runtime reads every secret | API, worker, migration, dashboard, and scheduler use separate Google service accounts; secret access is granted per secret and runtime | Google project administrators can still change IAM and must be tightly controlled and audited |
| Public Cloud Run URL bypasses Auth0 | Cloud Run accepts the connection, but the API still validates OIDC or source credentials and derives organization context server-side | Per-source rate limiting and edge abuse controls are still required before untrusted public traffic |
| Dashboard proxy trusts the wrong API | The API origin is injected at container start, TLS server-name verification is enabled, and the upstream Host is the actual Cloud Run host | Compromise of deployment variables or project IAM can redirect the proxy |
| Neon owner privilege leaks into runtime roles | Bootstrap removes managed administrative membership and applies `NOSUPERUSER`, `NOBYPASSRLS`, and separate migration/app/worker roles | Must be verified against the real Neon project and after every restore |
| Free-tier database recovery is treated as a backup | Destructive restore script checks owners, ACLs, functions, role flags, and restricted logins in an isolated target | Neon Free recovery limits are not a production backup policy; an exercised external backup is required |
| Automatic telemetry exposes sensitive metadata | Application logs are allow-listed; authorization headers are sanitized; all OpenTelemetry exporters remain off | Cloud Run request logs still need a staging privacy review, and route/query behavior must be checked before Cloud Trace is enabled |
| Free allowance creates unexpected charges | Services scale to zero, staging maximum is one instance, worker runs once per schedule, and the runbook requires a budget alert | A budget is only an alert, not a hard cap; the operator owns billing review |

## Logging and monitoring

Operational JSON events contain fixed event names plus explicitly supplied
scalar fields such as request IDs, source IDs, route templates, status
categories, durations, attempts, and counts. They do not interpolate logging
arguments or attach nested objects. Manual analysis spans mark failure without
recording exception messages or stacks. Automatic HTTP instrumentation is
configured to sanitize authorization, cookie, and API-key headers.

Prometheus metrics use HTTP method, route template, status, and normalized
ingest outcome. They deliberately omit user, organization, source, event, and
request IDs. The endpoint requires a bearer token in production and is excluded
from its own request metrics. Counters live in each API process and reset on
restart; a backend must sum replicas.

OpenTelemetry exporters default to `none`. Enabling OTLP is an explicit
deployment operation. The selected staging deployment keeps export disabled.
Cloud Run provides platform request count, latency, error, log, and job-state
signals without enabling application spans. The supplied alerts report any observed API 5xx or
rejected ingest event. There is no latency or saturation alert until staging
measurements support an actual service objective.

Security-relevant actions—login changes, role changes, credential creation and
revocation, exports, and deletion—need append-only audit events.

## Browser authentication

The dashboard uses OIDC Authorization Code with PKCE `S256`. OAuth state and the
PKCE verifier are held briefly in session storage and removed before token
exchange. The access token is kept only in memory, never in web storage,
cookies, or the URL, so refresh signs the user out. Token exchange refuses HTTP
redirects. A strict content-security policy permits connections only to the
same origin and the one identity-provider origin selected by the operator.

The deployment must register the dashboard as a public browser client, require
PKCE, allow the exact redirect URI, and permit browser token exchange. The
development token form is explicitly rejected in production settings.

## Claims we do not make

This design is not a compliance certification, penetration test, or guarantee
against every side channel. The included restore automation has not yet run
against Neon, and no Cloud Run or Auth0 environment has been created from these
definitions. Production readiness still requires an external review, an
exercised staging restore and rollback, incident drills, rate limiting, a
Cloud Logging/redaction review, safe distributed-trace export, and stronger
database recovery/capacity guarantees than the selected free staging plans.
