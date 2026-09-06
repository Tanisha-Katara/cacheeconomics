# Synthetic portfolio demo

This demo uses the real hosted dashboard with a checked-in, read-only response
packet. Every row comes from the repository's synthetic 286-request trace. The
packet is built by driving the collector normalizer, API validation, database
ingestion, worker, analyzer, and dashboard aggregate endpoints in process.

It is not a production deployment, load test, availability result, or realized
savings claim. Stable IDs and timestamps make screenshots reproducible; timing
fields are assigned synthetic scenario inputs solely to exercise the UI.

## View it

From the repository root:

```sh
python demo/serve_portfolio_demo.py
```

Open `http://127.0.0.1:8765`, paste the printed token, and use the four views.
The server binds only to loopback and accepts only the fixed, non-secret demo
token. It does not connect to any external service.

## Rebuild and verify the packet

Install the control-plane test dependencies, then run:

```sh
PYTHONPATH=harness:services/control_plane/src:collectors/src \
  python demo/build_portfolio_packet.py

PYTHONPATH=harness:services/control_plane/src:collectors/src \
  python demo/build_portfolio_packet.py --check
```

The `--check` form rebuilds the flow in memory and fails if the checked-in JSON
differs. CI runs that form. Raw prompts, completions, request bodies, and raw
provider errors are absent from both the input fixture and replay packet.
