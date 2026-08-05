# Security Policy

## Supported Versions

Security reports are accepted for the latest `0.2.x` release.

## Reporting A Vulnerability

Please do not open a public issue for vulnerabilities, private trace exposure,
credential handling bugs, or accidental egress paths.

Use GitHub's private vulnerability reporting for this repository. If that is
not available, email the maintainer listed in `pyproject.toml` with:

- affected version or commit
- reproduction steps
- expected impact
- whether any private prompt, trace, token, or credential material is involved

## Project Security Boundaries

The installed package is intended to run without opening network sockets.
Measurement scripts under `tier-b/` and optional proxy/plugin flows can perform
network work when explicitly invoked. A bug that sends prompt or trace content
without an explicit user action should be treated as security-sensitive.
