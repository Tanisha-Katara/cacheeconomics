from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKET_PATH = ROOT / "demo/fixtures/portfolio-demo-v1.json"
MEDIA_MANIFEST_PATH = ROOT / "apps/dashboard/media/manifest.json"


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


def test_published_dashboard_media_matches_its_manifest():
    manifest = json.loads(MEDIA_MANIFEST_PATH.read_text())

    assert manifest["schema"] == "cacheeconomics.dashboard-media"
    assert re.fullmatch(r"[0-9a-f]{40}", manifest["source_revision"])
    assert manifest["label"] == "synthetic local demonstration"
    assert manifest["viewport"] == {"width": 1280, "height": 720}
    packet = ROOT / manifest["packet"]["path"]
    assert hashlib.sha256(packet.read_bytes()).hexdigest() == manifest["packet"][
        "sha256"
    ]

    for artifact in manifest["artifacts"]:
        path = ROOT / artifact["path"]
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
        assert artifact["view"] in {"recommendations", "operations"}
        if artifact["kind"] == "video":
            assert 0 < artifact["duration_seconds"] < 45

    social = manifest["social_preview"]
    social_path = ROOT / social["path"]
    assert social_path.is_file()
    assert hashlib.sha256(social_path.read_bytes()).hexdigest() == social["sha256"]
    assert (social["width"], social["height"]) == (1200, 630)


def test_portfolio_docs_separate_completed_and_pending_deployment_evidence():
    ledger = _read("docs/portfolio-evidence.md")
    decisions = _read("docs/deployment-decision-record.md")
    tour = _read("docs/product-tour.md")
    readme = _read("README.md")

    assert "Implemented with synthetic labels" in ledger
    assert "restore and rollback drills pending" in ledger
    normalized_decisions = " ".join(decisions.split())
    assert "deployed to staging" in normalized_decisions
    assert "Still required before customer production use" in decisions
    assert "Recommendations and Operations posters" in tour
    assert "python demo/serve_portfolio_demo.py" in readme
    assert "It is not presented as a deployed environment." in readme
    assert "public Cloud Run staging deployment" in readme
