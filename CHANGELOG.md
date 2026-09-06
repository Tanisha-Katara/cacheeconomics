# Changelog

All notable user-facing changes are recorded here.

## Unreleased

- Selected a free-friendly portfolio staging architecture using GitHub
  Actions, Google Cloud Run, Neon PostgreSQL, and Auth0 without changing the
  socket-free installed package.
- Added Google infrastructure bootstrap, short-lived GitHub workload identity,
  secret-scoped runtime accounts, a digest-only manual deployment workflow,
  scheduled worker execution, and a provider-specific operator runbook.
- Made the dashboard's upstream API configurable for a separate HTTPS Cloud Run
  service while preserving the local Compose default.
- Kept OpenTelemetry export disabled until a staging trace privacy review is
  recorded; no deployed environment or cloud metrics are claimed yet.
- Added the configured API audience to dashboard OIDC authorization requests,
  serialized concurrent collector outbox flushes, and classified JSON decoder
  safety-limit failures as invalid client input.

## 0.3.0 - 2026-09-05

- Added the dependency-free analysis contracts required by the separately
  installed hosted control plane, and aligned the collector on the same core
  release.
- Kept authentication, ingestion, and every network dependency outside the
  local `cacheeconomics` package.
- Scoped default pytest discovery to the dependency-free core suite; hosted
  application suites remain explicit CI jobs.

## 0.2.1 - 2026-08-05

- Prepared the repository for a broader public announcement.
- Added public project URLs to package metadata so PyPI can link back to the
  source, issue tracker, and changelog.
- Reframed `PENDING.md` as a public known-limitations document instead of an
  internal merge journal.
- Added minimal contribution, security, and GitHub issue-template docs.
- Clarified README positioning around user personas, support boundaries,
  similar projects, and provider-specific prompt-cache economics.

## 0.2.0 - 2026-08-05

- Initial public beta package.
- Provides local prompt-cache cost analysis, provider-surface registry checks,
  invoice-gated reporting, trace analysis, policy bake-offs, and conservative
  live marker placement support.
