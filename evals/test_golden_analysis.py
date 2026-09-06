from __future__ import annotations

import hashlib
import json
from pathlib import Path

from cacheeconomics.analyzer import analyze
from cacheeconomics.contracts import analysis_payload
from cacheeconomics.trace import load_jsonl


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "evals/fixtures/golden-analysis-v1.json"


def _figure_tuple(figure: dict) -> list:
    return [
        figure["display"],
        figure["amount_usd"],
        figure["basis"],
        figure["release_state"],
    ]


def test_reviewed_analysis_output_does_not_drift():
    golden = json.loads(MANIFEST.read_text())
    fixture = ROOT / golden["fixture"]
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == golden["fixture_sha256"]

    payload = analysis_payload(
        analyze(load_jsonl(str(fixture)), invoice_usd=golden["invoice_usd"]),
        source=golden["source"],
    )
    analysis = payload["analysis"]
    actual = {
        "schema": payload["schema"],
        "schema_version": payload["schema_version"],
        "registry_sha256": payload["registry"]["sha256"],
        "tier": analysis["tier"],
        "coverage": analysis["coverage"],
        "ratios": analysis["ratios"],
        "figures": {
            key: _figure_tuple(analysis["spend"][key])
            for key in (
                "input_usd",
                "if_uncached_usd",
                "caching_saved_usd",
                "monthly_input_usd",
            )
        }
        | {
            "total_avoidable_usd_month": _figure_tuple(
                analysis["total_avoidable_usd_month"]
            )
        },
        "findings": [
            {
                key: finding[key]
                for key in (
                    "code",
                    "title",
                    "severity",
                    "confidence",
                    "evidence_class",
                    "quality_risk",
                    "affected_requests",
                    "structural",
                    "fix",
                )
            }
            for finding in analysis["findings"]
        ],
        "tokens_counted": analysis["tokens_counted"],
        "draft": analysis["draft"],
        "has_withheld_figures": analysis["has_withheld_figures"],
    }
    assert actual == golden["expected"]


def test_unreconciled_invoice_never_exposes_a_hidden_amount():
    golden = json.loads(MANIFEST.read_text())
    fixture = ROOT / golden["fixture"]
    payload = analysis_payload(
        analyze(load_jsonl(str(fixture)), invoice_usd=999.0),
        source="golden:unreconciled-invoice",
    )

    figures = list(payload["analysis"]["spend"].values())
    figures.append(payload["analysis"]["total_avoidable_usd_month"])
    for finding in payload["analysis"]["findings"]:
        figures.extend(
            value
            for value in (
                finding["avoidable_usd_window"],
                finding["avoidable_usd_month"],
            )
            if value is not None
        )

    monetary = [item for item in figures if isinstance(item, dict) and "released" in item]
    assert monetary
    assert all(item["released"] is False for item in monetary)
    assert all(item["amount_usd"] is None for item in monetary)
