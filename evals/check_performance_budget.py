#!/usr/bin/env python3
"""Compare a synthetic benchmark result with the checked-in regression budget."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument(
        "--budget",
        type=Path,
        default=ROOT / "evals/performance-budget.json",
    )
    args = parser.parse_args(argv)
    result = json.loads(args.result.read_text())
    budget = json.loads(args.budget.read_text())

    workload = budget["baseline"]["workload"]
    if result.get("schema") != "cacheeconomics.synthetic-local-benchmark":
        raise SystemExit("benchmark result has an unexpected schema")
    if result["ingestion"]["batch_size"] != workload["ingestion_batch_size"]:
        raise SystemExit("benchmark batch size does not match the reviewed workload")
    if result["ingestion"]["samples"] < workload["ingestion_samples"]:
        raise SystemExit("benchmark used fewer samples than the reviewed workload")
    if result["ingestion"]["accepted_events"] != (
        result["ingestion"]["samples"] * result["ingestion"]["batch_size"]
    ):
        raise SystemExit("benchmark did not accept every measured synthetic event")
    if result["analysis"]["request_rows"] != workload["analysis_request_rows"]:
        raise SystemExit("analysis benchmark fixture no longer matches the reviewed workload")
    if result["dashboard"]["assets"] != workload["dashboard_assets"]:
        raise SystemExit("dashboard benchmark asset list changed without budget review")

    checks = {
        "ingestion median events/second": (
            result["ingestion"]["median_events_per_second"]
            >= budget["limits"]["ingestion_median_events_per_second_min"]
        ),
        "ingestion p95 batch latency": (
            result["ingestion"]["p95_batch_latency_ms"]
            <= budget["limits"]["ingestion_p95_batch_latency_ms_max"]
        ),
        "analysis p95 latency": (
            result["analysis"]["p95_latency_ms"]
            <= budget["limits"]["analysis_p95_latency_ms_max"]
        ),
        "dashboard raw bytes": (
            result["dashboard"]["raw_bytes"]
            <= budget["limits"]["dashboard_raw_bytes_max"]
        ),
        "dashboard gzip bytes": (
            result["dashboard"]["gzip_bytes"]
            <= budget["limits"]["dashboard_gzip_bytes_max"]
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}: {name}")
    if failed:
        print("Budget failure: " + ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
