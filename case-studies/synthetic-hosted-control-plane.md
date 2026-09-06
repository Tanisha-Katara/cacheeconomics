# Synthetic case study: from prompt-free events to an action list

> **Evidence label: synthetic local demonstration.** This is not customer
> traffic, a production benchmark, a deployment claim, or realized savings.

The goal of this case study is to prove a product path, not a business result:
can a separate hosted application accept prompt-free cache telemetry, preserve
tenant context, run the existing analyzer, and show useful actions without
leaking an unreconciled dollar amount?

## Input

The input is the repository's 286-row `harness/fixtures/demo-traces.jsonl`.
It contains timestamps, usage counters, model/surface labels, and keyed segment
identifiers. It contains no prompt or completion bodies. The packet records the
fixture's SHA-256 digest so a reviewer can confirm the exact input.

The builder shifts the trace as one block to a fixed demo date, preserving every
inter-request gap. It assigns deterministic first-token and completion times to
exercise the operations charts. Those assigned times are scenario data, not
measurements of cacheeconomics service performance.

## Path exercised

```text
synthetic trace
  → collector normalization
  → prompt-free API validation
  → organization/source-scoped database rows
  → leased analysis worker
  → versioned analysis result
  → bounded dashboard aggregates
  → read-only browser replay
```

The builder uses an in-memory SQLite database so it can run anywhere. PostgreSQL
row-level security, atomic leasing, and backup/restore remain separate CI and
staging checks; this case study does not claim to exercise them.

## Verified result

| Check | Synthetic result | Evidence |
| --- | ---: | --- |
| Accepted events | 286 | `checks.accepted_events` in the demo packet |
| Idempotent duplicates | 1 | `checks.duplicate_events` |
| Rejected prompt-bearing events | 1 | `checks.rejected_events` |
| Stored prompt fields | 0 | `checks.stored_prompt_fields` plus packet privacy test |
| Events analyzed | 286 | `checks.analysis_event_count` |
| Input served from cache | 16.7% | versioned analysis ratio, rounded for display |
| Prefix efficiency | 17.2% | versioned analysis ratio, rounded for display |
| Findings | 4 | `TTL-1`, `EFF-1`, `CAC-1`, `MIN-1` |

The highest-ranked action is workload-specific: the synthetic research-agent
cadence falls inside the one-hour reuse window, so the result proposes testing
a one-hour TTL on that static prefix. The same result also warns that cache
writes are often not read, that caching costs more than uncached input in this
fixture, and that some markers sit below the model-specific minimum.

Those statements describe this checked-in synthetic trace only. They are not
claims about a provider account or another workload.

## The most important non-result

No invoice is supplied to the hosted flow. Therefore every monetary figure is
withheld, every numeric `amount_usd` is `null`, and this case study makes no
savings claim. The demo still gives concrete engineering actions without
turning an unreconciled estimate into a portfolio headline.

## Reproduce it

```sh
PYTHONPATH=harness:services/control_plane/src:collectors/src \
  python demo/build_portfolio_packet.py --check

python demo/serve_portfolio_demo.py
```

The first command reruns the actual in-process flow and compares it byte for
byte with `demo/fixtures/portfolio-demo-v1.json`. The second serves that packet
only on loopback through the real dashboard assets.

## Limitations

- SQLite is used for portability; it is not evidence of PostgreSQL behavior.
- Assigned request timings demonstrate UI states; they are not a load test.
- The packet is read-only and uses a fixed non-secret demo token.
- No public cloud, managed identity provider, telemetry backend, DNS name, TLS
  certificate, or availability result is claimed.
- Production proof still requires the staging checklist in the deployment
  runbook and a deployment-specific threat review.
