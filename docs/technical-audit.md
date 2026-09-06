# Technical audit

Audited on 2026-09-04 from the canonical checkout at
`/Users/tanishakatara/cacheeconomics`, branch `main`, commit `e3d83f4`. The
worktree was clean before implementation began. The older
`/Users/tanishakatara/cache-economics` checkout was not used or changed.

This is intentionally the pre-implementation baseline. Phase status and the
current repository shape are tracked in the [implementation plan](implementation-plan.md)
and [architecture](architecture.md), so the original audit findings remain
reviewable rather than being rewritten after each phase.

## Plain-language summary

The repository already contains a careful local analysis engine. It can read
several trace formats, calculate cache economics, explain likely problems, and
produce reports. Its tests strongly defend against guessed provider facts,
misleading savings claims, and accidental prompt disclosure.

It is not yet an enterprise web product. There is no account system, shared
database, organization isolation, hosted API, scheduled ingestion, production
dashboard, deployment image, or deployment runbook. The current `web/` page is
a static browser demo, not a signed-in dashboard.

## What is already strong

- The mandatory Python dependency list is empty.
- The package imports no network library, and a regression test scans every
  package module for network imports.
- Monetary figures are withheld until the existing evidence/reconciliation
  rules release them.
- Unknown and contested provider facts fail closed instead of becoming guesses.
- Request-body ingestion requires a keyed HMAC for structural identities.
- Tenant context participates in the HMAC input, preventing equality joins
  across workload tenants.
- CLI, text, HTML, and browser paths have extensive regression coverage.
- The LiteLLM adapter and callback provide the most direct starting point for a
  first live collector, but they do not prove support for every LiteLLM backend.

## Existing architecture

| Component | Current purpose |
| --- | --- |
| `harness/cacheeconomics/` | Registry, cost model, trace loaders, analyzer, reports, CLI, runtime monitor, optional LiteLLM callback |
| `harness/tests/` | Unit, invariant, privacy, adapter, CLI, report, browser-bundle, and experiment-script tests |
| `web/` | Static explanatory site and in-browser Python worker |
| `tier-a/` | Static research evidence |
| `tier-b/` | Networked experiment and measurement scripts, outside the installed package |
| `case-studies/` | Existing narrative evidence |
| `.github/workflows/ci.yml` | Tests, browser bundle check, wheel build/install check |

## Existing ingest paths

- Normalized JSON Lines traces
- Usage-only LiteLLM log rows
- Request-body exports, locally segmented with an HMAC key
- Claude Code usage files
- An optional LiteLLM callback for live observation and marker placement

These are parsers and local integrations. They are not hosted connectors, and
the audit found no basis for claiming live support for additional providers or
observability platforms.

## Gaps to close

### Product and tenancy

- No users, organizations, memberships, roles, or invitations
- No service credentials or source ownership
- No server-enforced tenant isolation or audit log

### Ingestion and operations

- No HTTPS ingestion endpoint, idempotency store, queue, scheduler, retry/dead
  letter handling, or ingestion-health view
- The core `Request` model has a sent time and optional first-token time, but no
  completion time or normalized error category
- No hosted tracing, latency distributions, error rates, or service health

### Recommendations and UI

- The analyzer already creates findings and actions, but the old CLI JSON omits
  ratios and finding detail/action/evidence fields needed by a dashboard
- The static demo is polished editorial content, not an application shell with
  navigation, filters, organization context, source health, or access control
- The browser calculator has its own small model/rate logic, so claims must not
  be expanded without evidence and cross-path tests

### Delivery

- No Dockerfiles, local multi-service environment, migrations, deployment
  manifests, release environments, backups, restore procedure, or rollback plan
- CI tests and packages the library but does not scan service dependencies or
  container images and does not deploy
- No end-to-end tenant-isolation, ingestion replay, migration, load, or restore
  evaluation suite

## Test baseline

Before Phase 0, the offline suite passed 1,583 tests. The full suite in this
restricted development sandbox reported 1,704 passed, 1 skipped, and 59 failed.
All 59 failures were loopback-socket `PermissionError` failures in
`test_tier_b_scripts.py`; the sandbox prohibits the local test servers those
tests create. They were environmental failures, not evidence that the product
paths passed or failed.

After Phase 0, the offline suite passes 1,595 tests. This includes 12 new tests
for the application boundary. Socket-dependent tests remain an ordinary CI
responsibility in an environment that permits loopback networking.

## Phase 0 result

Phase 0 adds documentation, versioned exchange schemas, and a safe complete
analysis serializer. It does not add a server or change existing CLI formats.
See [the implementation plan](implementation-plan.md) for what comes next.
