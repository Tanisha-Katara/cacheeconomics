# ADR 0001: Keep hosted features outside the offline package

- **Status:** Accepted
- **Date:** 2026-09-04

## Context

The installed `cacheeconomics` package promises local analysis without opening
sockets. The portfolio target also needs authentication, organizations, live or
scheduled ingestion, shared storage, monitoring, and a hosted dashboard. Putting
those capabilities into the package would break a clear privacy guarantee and
force network dependencies onto local users.

## Decision

Keep the existing package as the offline analysis engine. Build networked
features as separate collectors and hosted application services.

Communication across that boundary uses versioned schemas:

- Ingest v1 accepts prompt-free telemetry and keyed local fingerprints.
- Analysis result v1 exposes complete findings while preserving monetary release
  gates.

The hosted application may depend on the local engine. The local engine may not
depend on the hosted application, authentication libraries, databases, queues,
or HTTP clients.

## Consequences

Benefits:

- Existing local users keep the socket-free behavior and small dependency set.
- Hosted risk is isolated and can be deployed, updated, or removed separately.
- Privacy rules become testable at a narrow data boundary.
- The same analyzer remains the source of truth for CLI and hosted results.

Costs:

- Collectors and the service need explicit version compatibility.
- The service must translate stored events into the local `Request` model.
- Deployment has more than one component.

## Rejected alternatives

**Add HTTP upload and login directly to the package.** Rejected because importing
or using the local package would no longer have a simple offline trust boundary.

**Reimplement analysis inside the dashboard.** Rejected because two analyzers
would drift and could produce conflicting recommendations or cost figures.

**Upload raw prompts and redact them on the server.** Rejected because data that
never leaves the customer's environment has a materially smaller exposure.
