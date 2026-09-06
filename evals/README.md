# Evaluation suite

This directory is the release-level safety net for the hosted application. It
does not replace the unit tests. It checks the promises a deployment makes:

- the reference trace still produces the reviewed recommendation set;
- withheld monetary values never cross the analysis contract;
- the collector and API accept and reject the same prompt-free event shape;
- every role has exactly the documented permissions; and
- local synthetic performance stays inside a measured regression budget.

The fixtures contain synthetic metadata and token counts only. They contain no
prompt, completion, request body, response body, or provider error message.

Run the deterministic checks from the repository root after installing the
control-plane and collector test dependencies:

```sh
PYTHONPATH=harness:services/control_plane/src:collectors/src \
  python -m pytest -q evals
```

The performance check is deliberately separate because elapsed time depends on
the machine running it:

```sh
PYTHONPATH=harness:services/control_plane/src:collectors/src \
  python evals/benchmark_hosted.py --output /tmp/cacheeconomics-benchmark.json
python evals/check_performance_budget.py \
  --result /tmp/cacheeconomics-benchmark.json
```

Those timings describe a synthetic local test only. They are not production
capacity, throughput, savings, or an availability promise.

The portfolio demo is a separate functional proof. Rebuild its checked response
packet through the actual in-process collector/API/worker path with:

```sh
PYTHONPATH=harness:services/control_plane/src:collectors/src \
  python demo/build_portfolio_packet.py --check
```

The packet and its case study are explicitly synthetic. Assigned request timing
values exist to exercise the dashboard, not to extend the performance budget or
create a production claim.
