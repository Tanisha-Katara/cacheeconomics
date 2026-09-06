# Portfolio evidence ledger

This ledger separates what the repository proves from what still needs a real
deployment. “Implemented” means a repeatable artifact or automated check exists;
it does not mean a cloud service has been deployed.

| Claim | Evidence | Status |
| --- | --- | --- |
| Core package remains offline | Socket-import regression test and empty mandatory dependency list | Implemented |
| Human identity, organizations, and RBAC | OIDC verifier tests, organization tests, complete RBAC matrix; Auth0 selected | Implemented locally; Auth0 tenant exercise pending |
| Prompt-free ingestion | Public schema, collector allow-list, API compatibility tests, synthetic rejected-event proof | Implemented |
| Reliable jobs | Retry/cancel/dead-letter tests, lease heartbeat tests, PostgreSQL lease function | Implemented; real PostgreSQL CI/staging run pending |
| Actionable recommendations | Golden result plus deterministic portfolio packet | Implemented on synthetic data |
| Tracing, errors, latency, and health | Bounded aggregate endpoint, metrics/log tests, dashboard views; Cloud Run platform signals selected | Errors/latency/health implemented; Cloud Logging review pending; trace export deliberately off |
| Dashboard | Production container, configurable HTTPS proxy, and real UI replaying a generated packet | Implemented locally; Cloud Run deployment pending |
| Deployment safety | Hardened containers, CI/security workflows, restore/promotion scripts, Google Terraform bootstrap, GitHub-to-Google WIF deploy | Implemented definitions; staging exercise pending |
| Case study | `case-studies/synthetic-hosted-control-plane.md` | Implemented and explicitly synthetic |
| Product tour | `docs/product-tour.md` | Implemented |
| Screenshots | Must be captured from the checked packet and hash-recorded | Pending browser capture |
| Demo recording | Script below; record only after screenshots and staging labels are correct | Pending |
| Deployment-specific architecture and threat model | Selected-service diagram, threat table, and `deploy/gcp/README.md` | Implemented as design; runtime review pending |
| Incident and rollback exercise | Procedures exist in deployment runbook | Pending staging exercise and evidence |

## Source chain for the synthetic demo

- Input: `harness/fixtures/demo-traces.jsonl`
- Golden lock: `evals/fixtures/golden-analysis-v1.json`
- Builder: `demo/build_portfolio_packet.py`
- Checked packet: `demo/fixtures/portfolio-demo-v1.json`
- Replay server: `demo/serve_portfolio_demo.py`
- Regression tests: `demo/tests/test_portfolio_demo.py`

The packet carries the input and golden-file hashes, the exact flow exercised,
the normalization disclosure, and the synthetic claim scope. CI rebuilds it.

## Screenshot evidence format

When browser capture is available, record each image in a JSON manifest with:

- relative image path;
- SHA-256 of the exact demo packet;
- viewport width and height;
- dashboard view name;
- capture date; and
- the label `synthetic local demonstration`.

Do not crop away the synthetic organization/source identity, and do not replace
the dashboard with a design mockup.
