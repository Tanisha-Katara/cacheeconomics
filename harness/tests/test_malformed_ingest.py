"""A malformed row is excluded and counted. It never crashes the report.

Round 12 found that a `usage` field written as a JSON string reached the cost
model and raised `AttributeError` on `.get`, taking the whole analysis down with
it. One bad row in an export of half a million aborted the run.

That is a class, not an incident. Exporters write scalars where objects belong,
strings where numbers belong, and nulls anywhere at all. The loaders already
tolerate a missing timestamp, a string status and an absent tenant, each added
after a specific row broke a specific run.

So rather than wait for the next field, this walks every field of a well-formed
row and replaces it with each of a set of hostile values, asserting the same thing every
time: nothing crashes.

A deliberate refusal is not a crash. `load_jsonl` raises `ValueError` when an
export carries segments it cannot identify safely and no HMAC key was supplied,
and that is the documented contract -- refusing to hash content into a bare
digest is the behaviour, not a bug in it. So `ValueError` is allowed and
`TypeError`, `AttributeError`, `KeyError` and `IndexError` are not: the first
says "I will not do this and here is why", the rest say "I fell over".

Whether a row survives into `analysable` is the loader's judgement; whether it
takes the process with it is not.
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cacheeconomics.analyzer import analyze  # noqa: E402
from cacheeconomics.trace import Tier, load_jsonl  # noqa: E402

GOOD = {
    "request_id": "r1",
    "sent_at": "2026-07-29T09:00:00Z",
    "model": "claude-opus-5",
    "agent": "main",
    "session": "s1",
    "tenant": "acme",
    "target_id": "anthropic/direct",
    "status": 200,
    "ttl_requested": "5m",
    "first_token_at": "2026-07-29T09:00:01Z",
    "usage": {"input_tokens": 10, "cache_read_input_tokens": 9_000,
              "cache_creation_input_tokens": 0},
    "segments": [{"id": "hmac:" + "a" * 64, "role": "system", "tokens": 9_000,
                  "index": 0, "label": "sys", "cache_marked": True, "ttl": "5m"}],
}

HOSTILE = [
    None, 0, -1, 3.5, True, "", "   ", "not-a-number", [], {}, [1, 2, 3],
    {"unexpected": "shape"}, "null", '{"nested": "json"}',
    "9" * 400, {"input_tokens": "ten"},
]


def _write(rows):
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    with open(path, "w") as f:
        f.write("\n".join(json.dumps(r, default=str) for r in rows))
    return path


class TestNoFieldCanCrashTheIngest(unittest.TestCase):

    def test_every_field_survives_every_hostile_value(self):
        failures = []
        for field in GOOD:
            for bad in HOSTILE:
                row = dict(GOOD)
                row[field] = bad
                path = _write([row])
                try:
                    analyze(load_jsonl(path), allow_unreconciled=True)
                except ValueError:
                    pass          # a stated refusal, which is the contract
                except Exception as e:      # noqa: BLE001 - this is the assertion
                    failures.append(f"{field}={bad!r}: {type(e).__name__}: {e}")
                finally:
                    os.unlink(path)
        self.assertEqual(failures, [], "\n".join(failures[:12]))

    def test_a_good_row_alongside_a_bad_one_still_counts(self):
        """The point of excluding rather than crashing: the rest of the export
        is still worth analysing."""
        bad = dict(GOOD, request_id="bad", usage="not a mapping")
        path = _write([GOOD, bad])
        try:
            ts = load_jsonl(path)
            self.assertEqual({r.request_id for r in ts.analysable}, {"r1"})
            analyze(ts, allow_unreconciled=True)
        finally:
            os.unlink(path)

    def test_a_segment_list_of_the_wrong_shape_does_not_crash(self):
        for segs in ("string", 42, [None], [[]], {"a": 1}):
            row = dict(GOOD, segments=segs)
            path = _write([row])
            try:
                analyze(load_jsonl(path), allow_unreconciled=True)
            finally:
                os.unlink(path)

    def test_an_unidentifiable_segment_without_content_downgrades(self):
        """A segment with an untrusted id and no content cannot be identified by
        anyone: there is nothing to hash, so a key changes nothing.

        This used to raise "pass an HMAC key", and the remedy was a no-op --
        measured against the committed code, supplying a key produced exactly
        the USAGE_ONLY downgrade the default path now reaches directly. The
        contract that matters is unchanged and still asserted here: the id is
        not trusted, the structure is discarded, and the trace says so.
        """
        row = dict(GOOD, segments=[{"id": 1, "role": "system", "tokens": 10}])
        path = _write([row])
        try:
            ts = load_jsonl(path)
            self.assertIs(ts.tier, Tier.USAGE_ONLY)
            self.assertFalse(any(r.segments for r in ts.requests))
            self.assertTrue([n for n in ts.notes if "identity is unknowable" in n])
        finally:
            os.unlink(path)

    def test_an_unidentifiable_segment_with_content_still_needs_a_key(self):
        """The other half. Here a key genuinely is the remedy -- there is content
        to hash, and writing a bare digest of it is what the refusal prevents."""
        row = dict(GOOD, segments=[{"id": 1, "role": "system", "tokens": 10,
                                    "content": "a short policy line"}])
        path = _write([row])
        try:
            with self.assertRaises(ValueError) as e:
                load_jsonl(path)
            self.assertIn("HMAC key", str(e.exception))
            self.assertIsNotNone(load_jsonl(path, b"k" * 32))
        finally:
            os.unlink(path)

    def test_a_truncated_line_does_not_crash(self):
        """Live sessions are appended to while being read."""
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        try:
            with open(path, "w") as f:
                f.write(json.dumps(GOOD) + "\n")
                f.write(json.dumps(GOOD)[:40])      # partial write
            ts = load_jsonl(path)
            self.assertEqual(len(ts.analysable), 1)
            analyze(ts, allow_unreconciled=True)
        finally:
            os.unlink(path)

    def test_a_valid_json_line_that_is_not_an_object(self):
        """`42` and `[]` parse fine and are not rows. The sweep above varies
        field *values* and never questioned the row itself, so this whole shape
        was outside an audit written to close exactly this class."""
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        try:
            with open(path, "w") as f:
                f.write(json.dumps(GOOD) + "\n42\n[]\n\"a string\"\nnull\ntrue\n")
            ts = load_jsonl(path)
            self.assertEqual(len(ts.analysable), 1)
            analyze(ts, allow_unreconciled=True)
        finally:
            os.unlink(path)

    def test_the_body_loader_survives_them_too(self):
        from cacheeconomics.adapters.bodies import load_bodies
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        try:
            with open(path, "w") as f:
                f.write(json.dumps(TestTheBodyAdapterIsJustAsTolerant.ROW)
                        + "\n42\n[]\nnull\n")
            analyze(load_bodies(path, b"k" * 32, target_id="anthropic/direct"), allow_unreconciled=True)
        finally:
            os.unlink(path)

    def test_an_empty_file_does_not_crash(self):
        path = _write([])
        try:
            analyze(load_jsonl(path), allow_unreconciled=True)
        finally:
            os.unlink(path)


class TestTheBodyAdapterIsJustAsTolerant(unittest.TestCase):
    """It parses logged request bodies, so it meets worse input than the
    normalised loader ever does."""

    ROW = {"sent_at": "2026-07-29T09:00:00Z",
           "request": {"model": "claude-opus-5",
                       "system": [{"type": "text", "text": "p" * 40_000}],
                       "messages": [{"role": "user", "content": "hi"}]},
           "response": {"usage": {"input_tokens": 5,
                                  "cache_read_input_tokens": 0,
                                  "cache_creation_input_tokens": 11_000,
                                  "cache_creation": {
                                      "ephemeral_5m_input_tokens": 11_000,
                                      "ephemeral_1h_input_tokens": 0}}}}

    def test_every_field_survives_every_hostile_value(self):
        from cacheeconomics.adapters.bodies import load_bodies
        failures = []
        for field in self.ROW:
            for bad in HOSTILE:
                row = dict(self.ROW)
                row[field] = bad
                path = _write([row])
                try:
                    analyze(load_bodies(path, b"k" * 32, target_id="anthropic/direct"), allow_unreconciled=True)
                except ValueError:
                    pass
                except Exception as e:      # noqa: BLE001
                    failures.append(f"{field}={bad!r}: {type(e).__name__}: {e}")
                finally:
                    os.unlink(path)
        self.assertEqual(failures, [], "\n".join(failures[:12]))




class TestSegmentSizesMustAgreeWithWhatWasBilled(unittest.TestCase):
    """Two sources of truth for one request, and the gate only checked one.

    Spend is computed from `usage`; every structural figure is computed from
    segment `tokens`. Nothing compared them, so an exporter writing bytes,
    characters or a rescaled estimate reconciled perfectly against the invoice --
    reconciliation never looks at segments -- and then published a structural
    dollar figure from numbers no gate had examined. Measured: a trace billing
    twelve cents released a VOL-1 figure of $78,660,000 a month with the gate
    reporting a pass.

    Structure is kept and money is withheld. Identity is a separate question
    from size, and drift, volatility and relocation all read ids rather than
    counts -- discarding structure would throw away working findings to fix a
    pricing leak.
    """

    HMAC = "hmac:"

    def _rows(self, seg_tokens, n=20):
        return [{"request_id": f"r{i}", "sent_at": f"2026-07-29T09:{i:02d}:00Z",
                 "model": "claude-opus-5", "agent": "main", "session": "s1",
                 "usage": {"input_tokens": 0, "cache_read_input_tokens": 0,
                           "cache_creation_input_tokens": 1_000,
                           "cache_creation": {"ephemeral_5m_input_tokens": 1_000,
                                              "ephemeral_1h_input_tokens": 0}},
                 "segments": [
                     {"id": self.HMAC + ("a" * 63) + str(i % 10), "role": "system",
                      "tokens": seg_tokens, "index": 0, "label": "session_ctx",
                      "cache_marked": False, "ttl": None},
                     {"id": self.HMAC + "b" * 64, "role": "system",
                      "tokens": seg_tokens, "index": 1, "label": "instructions",
                      "cache_marked": True, "ttl": "5m"}],
                 # Authored sizes, so they are the ground truth here rather than
                 # a byte-share guess. This suite is about sizes agreeing with
                 # the billed total, not about where they came from.
                 "tokens_counted": True}
                for i in range(n)]

    def _analyse(self, seg_tokens):
        path = _write(self._rows(seg_tokens))
        try:
            ts = load_jsonl(path)
            draft = analyze(ts, allow_unreconciled=True)
            invoice = draft.spend["input_usd"].raw()
            return ts, analyze(ts, invoice_usd=invoice)
        finally:
            os.unlink(path)

    def test_agreeing_sizes_still_publish(self):
        """The control. Withholding everything would also pass a bad test."""
        _, a = self._analyse(500)
        vol = next(f for f in a.findings if f.code == "VOL-1")
        self.assertTrue(vol.avoidable_usd_month.released)

    def test_sizes_in_the_wrong_units_withhold_the_figure(self):
        _, a = self._analyse(1_000_000_000)
        vol = next(f for f in a.findings if f.code == "VOL-1")
        self.assertFalse(vol.avoidable_usd_month.released)

    def test_the_reason_names_the_disagreement(self):
        _, a = self._analyse(1_000_000_000)
        vol = next(f for f in a.findings if f.code == "VOL-1")
        self.assertIn("billed", vol.avoidable_usd_month.withheld_because)

    def test_the_invoice_still_reconciles_because_usage_is_untouched(self):
        """The point of the leak: the gate passes on the usage half. Passing is
        correct; publishing a figure from the other half is not."""
        _, a = self._analyse(1_000_000_000)
        self.assertTrue(a.reconciliation["within_ship_gate"])

    def test_structure_is_kept_rather_than_discarded(self):
        ts, _ = self._analyse(1_000_000_000)
        self.assertIs(ts.tier, Tier.INSTRUMENTED)
        self.assertTrue(all(r.segments for r in ts.requests))

    def test_the_trace_says_so(self):
        ts, _ = self._analyse(1_000_000_000)
        self.assertFalse(ts.token_sums_reconciled)
        self.assertTrue(any("do not sum" in n for n in ts.notes))

    def test_a_boolean_is_not_a_token_count(self):
        """`isinstance(True, int)` is True, so a flag passed as a count of one."""
        from cacheeconomics.trace import _is_token_count
        for bad in (True, False, -1, float("inf"), float("nan"), "500", None):
            self.assertFalse(_is_token_count(bad), repr(bad))
        for good in (0, 1, 500, 1.5):
            self.assertTrue(_is_token_count(good), repr(good))


if __name__ == "__main__":
    unittest.main()
