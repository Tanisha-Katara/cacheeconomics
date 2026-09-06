"""Regression tests for the offline-to-hosted data boundary."""

import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cacheeconomics.analyzer import analyze  # noqa: E402
from cacheeconomics.contracts import (  # noqa: E402
    ANALYSIS_SCHEMA,
    ANALYSIS_SCHEMA_VERSION,
    analysis_payload,
    analysis_payload_json,
    figure_payload,
    registry_snapshot,
)
from cacheeconomics.money import WithheldFigure  # noqa: E402
from cacheeconomics.trace import load_jsonl  # noqa: E402


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FIXTURE = os.path.join(HERE, "..", "fixtures", "demo-traces.jsonl")
SCHEMAS = os.path.join(ROOT, "schemas")


def _schema(name):
    with open(os.path.join(SCHEMAS, name)) as handle:
        return json.load(handle)


def _figure_dicts(value):
    """Walk a payload and yield values shaped like contract figures."""

    if isinstance(value, dict):
        if {"amount_usd", "release_state", "released"} <= set(value):
            yield value
        for child in value.values():
            yield from _figure_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _figure_dicts(child)


def _property_names(schema):
    """Collect declared JSON property names, not words in descriptions."""

    names = set()
    if isinstance(schema, dict):
        names.update((schema.get("properties") or {}).keys())
        for value in schema.values():
            names.update(_property_names(value))
    elif isinstance(schema, list):
        for value in schema:
            names.update(_property_names(value))
    return names


class TestAnalysisResultContract(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.traces = load_jsonl(os.path.normpath(FIXTURE))
        cls.reconciled = analyze(cls.traces, invoice_usd=17.45)
        cls.withheld = analyze(cls.traces, invoice_usd=999.0)
        cls.draft = analyze(cls.traces, allow_unreconciled=True)

    def test_envelope_identifies_engine_registry_and_version(self):
        payload = analysis_payload(self.reconciled, source=self.traces.source)
        self.assertEqual(ANALYSIS_SCHEMA, payload["schema"])
        self.assertEqual(ANALYSIS_SCHEMA_VERSION, payload["schema_version"])
        self.assertEqual("cacheeconomics", payload["engine"]["name"])
        self.assertEqual("instrumented", payload["analysis"]["tier"])
        self.assertRegex(payload["registry"]["sha256"], r"^[0-9a-f]{64}$")

    def test_registry_snapshot_is_repeatable_and_dated(self):
        one = registry_snapshot()
        two = registry_snapshot()
        self.assertEqual(one, two)
        self.assertRegex(one["sha256"], r"^[0-9a-f]{64}$")
        for key in ("providers_generated", "pricing_generated"):
            self.assertRegex(one[key], r"^\d{4}-\d{2}-\d{2}$")

    def test_payload_includes_the_actionable_parts_the_old_cli_omits(self):
        result = analysis_payload(self.reconciled)["analysis"]
        self.assertEqual(self.reconciled.ratios, result["ratios"])
        self.assertEqual(
            figure_payload(self.reconciled.total_avoidable_month),
            result["total_avoidable_usd_month"],
        )
        self.assertEqual(self.reconciled.tokens_counted, result["tokens_counted"])
        self.assertEqual(len(self.reconciled.findings), len(result["findings"]))
        for original, encoded in zip(self.reconciled.findings, result["findings"]):
            self.assertEqual(original.detail, encoded["detail"])
            self.assertEqual(original.fix, encoded["fix"])
            self.assertEqual(original.evidence_class, encoded["evidence_class"])
            self.assertEqual(original.quality_risk, encoded["quality_risk"])

    def test_reconciled_figures_have_numbers_and_provenance(self):
        payload = analysis_payload(self.reconciled)
        figures = list(_figure_dicts(payload))
        self.assertTrue(figures)
        released = [item for item in figures if item["released"]]
        self.assertTrue(released)
        for item in released:
            self.assertIsInstance(item["amount_usd"], (int, float))
            self.assertIn(item["release_state"], ("reconciled", "draft"))
            self.assertIsNone(item["withheld_because"])

    def test_withheld_figures_never_expose_the_hidden_number(self):
        figure = self.withheld.spend["input_usd"]
        with self.assertRaises(WithheldFigure):
            _ = figure.amount

        encoded = figure_payload(figure)
        self.assertFalse(encoded["released"])
        self.assertEqual("withheld", encoded["release_state"])
        self.assertIsNone(encoded["amount_usd"])
        self.assertTrue(encoded["withheld_because"])

        payload = analysis_payload(self.withheld)
        withheld = [item for item in _figure_dicts(payload) if not item["released"]]
        self.assertTrue(withheld)
        self.assertTrue(all(item["amount_usd"] is None for item in withheld))

    def test_draft_state_survives_serialization(self):
        payload = analysis_payload(self.draft)["analysis"]
        self.assertTrue(payload["draft"])
        states = {
            item["release_state"]
            for item in _figure_dicts(payload)
            if item["released"]
        }
        self.assertEqual({"draft"}, states)

    def test_json_is_strict_and_round_trips(self):
        encoded = analysis_payload_json(
            self.reconciled,
            source="fixture",
        )
        self.assertNotIn("NaN", encoded)
        self.assertNotIn("Infinity", encoded)
        decoded = json.loads(encoded)
        self.assertEqual("fixture", decoded["analysis"]["source"])

    def test_emitted_sections_match_the_schema_property_names(self):
        schema = _schema("analysis-result-v1.schema.json")
        payload = analysis_payload(self.reconciled)
        self.assertEqual(set(schema["properties"]), set(payload))
        self.assertEqual(
            set(schema["$defs"]["analysis"]["properties"]),
            set(payload["analysis"]),
        )
        self.assertEqual(
            set(schema["$defs"]["spend"]["properties"]),
            set(payload["analysis"]["spend"]),
        )
        for finding in payload["analysis"]["findings"]:
            self.assertEqual(
                set(schema["$defs"]["finding"]["properties"]), set(finding)
            )


class TestIngestContract(unittest.TestCase):

    def test_batch_schema_uses_the_prompt_free_event_contract(self):
        schema = _schema("ingest-batch-v1.schema.json")
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(["events"], schema["required"])
        self.assertEqual(
            "ingest-event-v1.schema.json",
            schema["properties"]["events"]["items"]["$ref"],
        )

    def test_schema_is_valid_json_and_fail_closed_at_every_object(self):
        schema = _schema("ingest-event-v1.schema.json")
        self.assertEqual("cacheeconomics.ingest-event", schema["properties"]["schema"]["const"])
        self.assertFalse(schema["additionalProperties"])
        self.assertFalse(schema["properties"]["status"]["additionalProperties"])
        self.assertFalse(schema["$defs"]["usage"]["additionalProperties"])
        self.assertFalse(schema["$defs"]["segment"]["additionalProperties"])

    def test_no_content_or_organization_selector_can_enter_the_body(self):
        schema = _schema("ingest-event-v1.schema.json")
        names = _property_names(schema)
        forbidden = {
            "organization_id",
            "org_id",
            "prompt",
            "prompts",
            "completion",
            "completions",
            "messages",
            "body",
            "request_body",
            "response_body",
            "error_message",
            "tool_arguments",
        }
        self.assertFalse(names & forbidden, names & forbidden)

    def test_structural_identity_requires_a_keyed_hmac(self):
        schema = _schema("ingest-event-v1.schema.json")
        pattern = schema["$defs"]["segment"]["properties"]["id"]["pattern"]
        self.assertIsNotNone(re.fullmatch(pattern, "hmac:" + "a" * 64))
        self.assertIsNone(re.fullmatch(pattern, "sha256:" + "a" * 64))
        self.assertIsNone(re.fullmatch(pattern, "plain-segment-name"))

    def test_schema_requires_fields_needed_by_the_existing_analyzer(self):
        schema = _schema("ingest-event-v1.schema.json")
        segment_required = set(schema["$defs"]["segment"]["required"])
        self.assertTrue(
            {"id", "role", "tokens", "cache_marked", "index", "ttl"}
            <= segment_required
        )
        usage_required = set(schema["$defs"]["usage"]["required"])
        self.assertTrue(
            {
                "input_tokens",
                "cache_read_input_tokens",
                "cache_creation_input_tokens",
                "output_tokens",
            }
            <= usage_required
        )


if __name__ == "__main__":
    unittest.main()
