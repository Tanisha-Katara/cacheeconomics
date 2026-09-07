from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKET_PATH = ROOT / "demo/fixtures/portfolio-demo-v1.json"


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text()


def test_portfolio_packet_provenance_matches_the_reviewed_inputs():
    packet = json.loads(PACKET_PATH.read_text())

    assert packet["claim_scope"] == (
        "Synthetic local demonstration only; not production traffic, "
        "capacity, availability, or realized savings."
    )
    for evidence_name in ("trace", "golden_result"):
        evidence = packet["provenance"][evidence_name]
        actual = hashlib.sha256((ROOT / evidence["path"]).read_bytes()).hexdigest()
        assert actual == evidence["sha256"]


def test_case_study_numbers_are_derived_from_the_packet():
    packet = json.loads(PACKET_PATH.read_text())
    case_study = _read("case-studies/synthetic-hosted-control-plane.md")
    checks = packet["checks"]
    ratios = packet["responses"]["analysis"]["result"]["analysis"]["ratios"]

    assert f"| Accepted events | {checks['accepted_events']} |" in case_study
    assert f"| Idempotent duplicates | {checks['duplicate_events']} |" in case_study
    assert f"| Rejected prompt-bearing events | {checks['rejected_events']} |" in case_study
    assert f"| Stored prompt fields | {checks['stored_prompt_fields']} |" in case_study
    assert f"| Events analyzed | {checks['analysis_event_count']} |" in case_study
    assert f"| Input served from cache | {ratios['input_from_cache']:.1%} |" in case_study
    assert f"| Prefix efficiency | {ratios['prefix_efficiency']:.1%} |" in case_study
    assert ", ".join(f"`{code}`" for code in checks["finding_codes"]) in case_study


def test_portfolio_docs_do_not_claim_missing_deployment_evidence():
    ledger = _read("docs/portfolio-evidence.md")
    decisions = _read("docs/deployment-decision-record.md")
    tour = _read("docs/product-tour.md")
    readme = _read("README.md")

    assert "Pending browser capture" in ledger
    assert "Pending staging exercise" in ledger
    normalized_decisions = " ".join(decisions.split())
    assert "Cloud Run workloads await the first deployment" in normalized_decisions
    assert (
        "Evidence still required before the hosted application is called “deployed”"
        in decisions
    )
    assert "not checked in yet" in tour
    assert "python demo/serve_portfolio_demo.py" in readme
    assert "It is not presented as a deployed environment." in readme
