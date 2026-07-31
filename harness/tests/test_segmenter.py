"""Post-hoc segmentation of logged bodies, and whether the alignment score is honest.

The interesting tests here are the ones where the gateway did not log exactly
what went on the wire. That is the normal case -- exporters flatten, strip and
truncate -- and it is the only case where an alignment score earns its keep.
"""

import copy
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cacheeconomics.adapters.bodies import (load_bodies, score_alignment,  # noqa: E402
                                            with_alignment)
from cacheeconomics.recorder import Recorder  # noqa: E402
from cacheeconomics.trace import (Request, Segment, Tier,  # noqa: E402
                                  TraceSet, load_jsonl)

KEY = b"segmenter-test-key"
T0 = datetime(2026, 7, 29, 9, tzinfo=timezone.utc)


def body(i):
    return {
        "model": "claude-opus-5",
        "tools": [{"name": f"t{k}", "description": "d" * 200} for k in range(3)],
        "system": [{"type": "text", "text": f"session {i}"},
                   {"type": "text", "text": "S" * 6000,
                    "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": [{"type": "text", "text": f"step {i}"}]}],
    }


def response(i):
    return {"id": f"msg_{i}", "usage": {
        "input_tokens": 150, "output_tokens": 30,
        "cache_read_input_tokens": 8000, "cache_creation_input_tokens": 0,
        "cache_creation": {}}}


def _paired(mangle=lambda b: b, n=8):
    """Record instrumented, and write what a gateway would have logged."""
    inst = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False).name
    gw = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False).name
    rec, rows = Recorder(inst, key=KEY), []
    for i in range(n):
        b = body(i)
        cap = rec.capture(b, agent="coder", session="s1")
        cap.sent_at = T0 + timedelta(seconds=60 * i)
        cap.done(response(i))
        rows.append({"request_id": f"msg_{i}", "sent_at": cap.sent_at.isoformat(),
                     "agent": "coder", "session": "s1",
                     "request": mangle(copy.deepcopy(b)),
                     "response": response(i), "status": 200})
    with open(gw, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return inst, gw


class TestBodyIngest(unittest.TestCase):

    def test_it_loads_as_an_inferred_trace(self):
        inst, gw = _paired()
        ts = load_bodies(gw, key=KEY)
        self.assertIs(ts.tier, Tier.INFERRED)
        self.assertEqual(len(ts.requests), 8)
        os.unlink(inst); os.unlink(gw)

    def test_it_refuses_to_run_without_a_key(self):
        with self.assertRaises(ValueError):
            load_bodies("/tmp/nothing.jsonl", key=b"")

    def test_it_finds_the_body_under_several_exporter_names(self):
        for field in ("request", "request_body", "input", "body", "kwargs"):
            path = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False).name
            with open(path, "w") as f:
                f.write(json.dumps({"request_id": "a", "sent_at": T0.isoformat(),
                                    field: body(0), "response": response(0)}) + "\n")
            self.assertEqual(len(load_bodies(path, key=KEY).requests), 1, field)
            os.unlink(path)

    def test_a_row_with_no_recognisable_body_is_counted_not_dropped_silently(self):
        path = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False).name
        with open(path, "w") as f:
            f.write(json.dumps({"request_id": "a", "note": "no body here"}) + "\n")
        ts = load_bodies(path, key=KEY)
        self.assertEqual(len(ts.requests), 0)
        self.assertTrue(any("no recognisable request body" in n for n in ts.notes))
        os.unlink(path)

    def test_segmentation_matches_the_recorder_on_identical_input(self):
        """Both go through one function on purpose. Two implementations would
        drift, and the drift would surface as phantom volatility."""
        inst, gw = _paired()
        truth, inferred = load_jsonl(inst), load_bodies(gw, key=KEY)
        self.assertEqual([s.id for s in truth.requests[0].segments],
                         [s.id for s in inferred.requests[0].segments])
        os.unlink(inst); os.unlink(gw)


class TestAlignmentIsHonest(unittest.TestCase):
    """A score that only ever returns 1.0 measures nothing."""

    def _score(self, mangle):
        inst, gw = _paired(mangle)
        s = score_alignment(load_jsonl(inst), load_bodies(gw, key=KEY))
        os.unlink(inst); os.unlink(gw)
        return s

    def test_an_identical_body_scores_one(self):
        s = self._score(lambda b: b)
        self.assertEqual(s["mean_alignment"], 1.0)
        self.assertEqual(s["exact_matches"], 8)

    def test_flattening_content_to_a_string_is_detected(self):
        """A very common exporter normalisation."""
        def flatten(b):
            for m in b["messages"]:
                if isinstance(m["content"], list) and len(m["content"]) == 1:
                    m["content"] = m["content"][0]["text"]
            return b
        s = self._score(flatten)
        self.assertLess(s["segment_alignment"], 1.0)
        self.assertEqual(s["exact_matches"], 0)

    def test_truncated_content_is_detected(self):
        def truncate(b):
            for blk in b["system"]:
                blk["text"] = blk["text"][:200]
            return b
        self.assertLess(self._score(truncate)["segment_alignment"], 1.0)

    def test_stripped_markers_are_detected_even_though_ids_still_match(self):
        """The case that exposed a hole in the first version of this score.

        Segment identity deliberately excludes cache_control, so a gateway that
        strips it produces bodies whose ids match perfectly while the entire
        caching configuration is gone. Scoring identity alone reported 1.00.
        """
        def strip(b):
            for blk in b["system"]:
                blk.pop("cache_control", None)
            return b
        s = self._score(strip)
        self.assertEqual(s["segment_alignment"], 1.0, "ids are unaffected by design")
        self.assertEqual(s["marker_alignment"], 0.0, "but every marker was lost")
        self.assertEqual(s["mean_alignment"], 0.0, "the reported score takes the worse")

    def test_the_reported_score_is_never_better_than_its_weakest_dimension(self):
        for mangle in (lambda b: b,
                       lambda b: [blk.pop("cache_control", None) for blk in b["system"]] and b):
            s = self._score(mangle)
            self.assertEqual(s["mean_alignment"],
                             min(s["segment_alignment"], s["marker_alignment"]))

    def test_attaching_a_score_warns_when_markers_were_lost(self):
        inst, gw = _paired(lambda b: [blk.pop("cache_control", None)
                                      for blk in b["system"]] and b)
        inferred = load_bodies(gw, key=KEY)
        with_alignment(inferred, score_alignment(load_jsonl(inst), inferred))
        self.assertEqual(inferred.alignment, 0.0)
        self.assertTrue(any("as-shipped arm" in n for n in inferred.notes))
        os.unlink(inst); os.unlink(gw)



class TestBodyIngestIsLossless(unittest.TestCase):
    """A row with usage and no body is a real request that cost real money.

    Dropping it shrank the denominator, so a mixed export could report 100%
    coverage over the subset that happened to carry bodies -- understating
    ratios and spend while a note took the blame for it.
    """

    def _mixed(self, n_bodies=6, n_usage_only=3):
        path = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False).name
        rows = []
        for i in range(n_bodies):
            rows.append({"request_id": f"b{i}", "sent_at": T0.isoformat(),
                         "request": body(i), "response": response(i), "status": 200})
        for i in range(n_usage_only):
            rows.append({"request_id": f"u{i}", "sent_at": T0.isoformat(),
                         "model": "claude-opus-5",
                         "response": {"usage": {"input_tokens": 900, "output_tokens": 10,
                                                "cache_read_input_tokens": 0,
                                                "cache_creation_input_tokens": 0,
                                                "cache_creation": {}}},
                         "status": 200})
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        return path

    def test_usage_only_rows_survive(self):
        p = self._mixed()
        ts = load_bodies(p, key=KEY)
        self.assertEqual(len(ts.requests), 9)
        self.assertEqual(ts.coverage["total"], 9)
        os.unlink(p)

    def test_structural_coverage_reflects_what_was_segmented(self):
        p = self._mixed()
        ts = load_bodies(p, key=KEY)
        self.assertAlmostEqual(ts.structural_coverage, 6 / 9)
        os.unlink(p)

    def test_the_shortfall_is_stated_not_hidden(self):
        p = self._mixed()
        ts = load_bodies(p, key=KEY)
        self.assertTrue(any("belong in the denominator" in n for n in ts.notes))
        os.unlink(p)

    def test_a_row_with_neither_body_nor_usage_is_still_dropped(self):
        path = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False).name
        with open(path, "w") as f:
            f.write(json.dumps({"request_id": "x", "note": "nothing here"}) + "\n")
        self.assertEqual(len(load_bodies(path, key=KEY).requests), 0)
        os.unlink(path)


class TestLoaderSurvivesPartialRows(unittest.TestCase):
    """A malformed row should be something the coverage line reports on, not a
    KeyError that takes the whole analysis down. A dropped file is a worse
    answer than a stated gap."""

    def test_a_row_missing_model_does_not_raise(self):
        path = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False).name
        with open(path, "w") as f:
            f.write(json.dumps({"request_id": "a", "sent_at": T0.isoformat(),
                                "usage": {"input_tokens": 5}}) + "\n")
        ts = load_jsonl(path)
        self.assertEqual(len(ts.requests), 1)
        self.assertEqual(ts.requests[0].model, "unknown")
        os.unlink(path)

    def test_a_row_missing_everything_optional_still_loads(self):
        path = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False).name
        with open(path, "w") as f:
            f.write(json.dumps({"usage": {"input_tokens": 5}}) + "\n")
        self.assertEqual(len(load_jsonl(path).requests), 1)
        os.unlink(path)



class TestMalformedLinesAreCounted(unittest.TestCase):
    """A truncated line in a gateway export used to vanish from both the
    analysis and the denominator, so the rows that did parse were presented as
    though they were the whole file."""

    def _with_junk(self):
        path = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False).name
        with open(path, "w") as f:
            for i in range(4):
                f.write(json.dumps({"request_id": f"b{i}", "sent_at": T0.isoformat(),
                                    "request": body(i), "response": response(i)}) + "\n")
            f.write('{"request_id": "truncated"\n')
        return path

    def test_the_unparseable_line_is_reported(self):
        p = self._with_junk()
        ts = load_bodies(p, key=KEY)
        self.assertEqual(len(ts.requests), 4)
        self.assertTrue(any("could not be parsed as JSON" in n for n in ts.notes))
        os.unlink(p)

    def test_a_clean_file_carries_no_such_note(self):
        inst, gw = _paired()
        self.assertFalse(any("could not be parsed" in n
                             for n in load_bodies(gw, key=KEY).notes))
        os.unlink(inst); os.unlink(gw)


class TestDefaultLifetimeSurvivesBodyIngest(unittest.TestCase):
    """`cache_control: {"type": "ephemeral"}` with no ttl is the provider's
    five-minute default, and the body proves it. Losing that made every write in
    the main post-hoc path unprovable, withholding spend for traffic whose
    lifetime was sitting in the request all along."""

    def _export(self, cache_control):
        path = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False).name
        with open(path, "w") as f:
            for i in range(4):
                b = {"model": "claude-opus-5",
                     "system": [{"type": "text", "text": "S" * 6000,
                                 "cache_control": cache_control}],
                     "messages": [{"role": "user",
                                   "content": [{"type": "text", "text": f"s{i}"}]}]}
                f.write(json.dumps({
                    "request_id": f"m{i}", "sent_at": T0.isoformat(), "request": b,
                    "response": {"usage": {"input_tokens": 100,
                                           "cache_creation_input_tokens": 6000,
                                           "cache_read_input_tokens": 0,
                                           "cache_creation": {}, "output_tokens": 10}},
                    "status": 200}) + "\n")
        return path

    def test_a_default_marker_becomes_five_minutes(self):
        p = self._export({"type": "ephemeral"})
        self.assertEqual(load_bodies(p, key=KEY).requests[0].ttl_requested, "5m")
        os.unlink(p)

    def test_an_explicit_lifetime_is_preserved(self):
        p = self._export({"type": "ephemeral", "ttl": "1h"})
        self.assertEqual(load_bodies(p, key=KEY).requests[0].ttl_requested, "1h")
        os.unlink(p)

    def test_the_writes_are_now_priceable(self):
        from cacheeconomics.analyzer import _usages
        p = self._export({"type": "ephemeral"})
        priced, unprovable = _usages(load_bodies(p, key=KEY).requests)
        self.assertEqual(unprovable, [])
        self.assertEqual(priced[0].cache_write_5m, 6000)
        os.unlink(p)

    def test_a_body_with_no_marker_declares_no_lifetime(self):
        path = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False).name
        with open(path, "w") as f:
            f.write(json.dumps({"request_id": "a", "sent_at": T0.isoformat(),
                                "request": {"model": "claude-opus-5",
                                            "messages": [{"role": "user", "content": "x"}]},
                                "response": response(0)}) + "\n")
        self.assertIsNone(load_bodies(path, key=KEY).requests[0].ttl_requested)
        os.unlink(path)


class TestAlignmentCannotCertifyAPartialOrLossyExport(unittest.TestCase):
    """A score that returns 1.0 for a one-row sample of a fifty-row capture, or
    for a segmentation that collapses duplicates, is not measuring anything --
    and it is the number that decides whether structural findings carry money."""

    def _ts(self, rows, tier):
        return TraceSet(requests=rows, tier=tier)

    def _req(self, rid, segs):
        return Request(request_id=rid, sent_at=T0, model="claude-opus-5",
                       usage={}, segments=segs)

    def _seg(self, i, sid, marked=False, ttl=None):
        return Segment(id=sid, role="system", tokens=10, index=i,
                       cache_marked=marked, ttl=ttl)

    def test_a_one_row_sample_of_a_large_capture_scores_near_zero(self):
        truth = self._ts([self._req(f"r{i}", [self._seg(0, "a"), self._seg(1, "b")])
                          for i in range(50)], Tier.INSTRUMENTED)
        inferred = self._ts([self._req("r0", [self._seg(0, "a"), self._seg(1, "b")])],
                            Tier.INFERRED)
        s = score_alignment(truth, inferred)
        self.assertAlmostEqual(s["request_coverage"], 1 / 50)
        self.assertLess(s["mean_alignment"], 0.9, "must not clear the release floor")

    def test_collapsing_repeated_segments_is_detected(self):
        """Bare id sets discarded multiplicity, so a segmenter that merged two
        identical adjacent spans scored perfectly."""
        truth = self._ts([self._req("r0", [self._seg(0, "x"), self._seg(1, "x"),
                                           self._seg(2, "y")])], Tier.INSTRUMENTED)
        inferred = self._ts([self._req("r0", [self._seg(0, "x"), self._seg(1, "y")])],
                            Tier.INFERRED)
        self.assertLess(score_alignment(truth, inferred)["segment_alignment"], 1.0)

    def test_reordered_segments_are_detected(self):
        """Position is part of identity: the same spans in a different order are
        a different prompt and a different cache prefix."""
        truth = self._ts([self._req("r0", [self._seg(0, "x"), self._seg(1, "y")])],
                         Tier.INSTRUMENTED)
        inferred = self._ts([self._req("r0", [self._seg(0, "y"), self._seg(1, "x")])],
                            Tier.INFERRED)
        self.assertLess(score_alignment(truth, inferred)["segment_alignment"], 1.0)

    def test_a_request_missing_from_either_side_counts_as_zero(self):
        truth = self._ts([self._req("a", [self._seg(0, "x")]),
                          self._req("b", [self._seg(0, "x")])], Tier.INSTRUMENTED)
        inferred = self._ts([self._req("a", [self._seg(0, "x")]),
                             self._req("c", [self._seg(0, "x")])], Tier.INFERRED)
        s = score_alignment(truth, inferred)
        self.assertEqual(s["unmatched_requests"], 2)
        self.assertLess(s["mean_alignment"], 0.9)

    def test_a_complete_faithful_export_still_scores_one(self):
        rows = [[self._seg(0, "x", marked=True, ttl="5m"), self._seg(1, "y")]
                for _ in range(4)]
        truth = self._ts([self._req(f"r{i}", s) for i, s in enumerate(rows)],
                         Tier.INSTRUMENTED)
        inferred = self._ts([self._req(f"r{i}", s) for i, s in enumerate(rows)],
                            Tier.INFERRED)
        self.assertEqual(score_alignment(truth, inferred)["mean_alignment"], 1.0)

    def test_the_shortfall_in_coverage_is_stated(self):
        truth = self._ts([self._req(f"r{i}", [self._seg(0, "x")]) for i in range(4)],
                         Tier.INSTRUMENTED)
        inferred = self._ts([self._req("r0", [self._seg(0, "x")])], Tier.INFERRED)
        with_alignment(inferred, score_alignment(truth, inferred))
        self.assertTrue(any("only one side" in n for n in inferred.notes))


class TestBodyIngestPreservesTheSurface(unittest.TestCase):
    """Hard-coding the argument analysed a mixed gateway export as one cache
    pool under one provider's minimums, TTL rules and multipliers, which
    fabricates cache sharing between surfaces that cannot share anything."""

    def _mixed(self):
        path = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False).name
        with open(path, "w") as f:
            for i in range(4):
                tgt = "anthropic/direct" if i % 2 else "amazon-bedrock/converse"
                f.write(json.dumps({"request_id": f"r{i}", "sent_at": T0.isoformat(),
                                    "target_id": tgt, "request": body(i),
                                    "response": response(i)}) + "\n")
        return path

    def test_each_row_keeps_its_own_surface(self):
        p = self._mixed()
        ts = load_bodies(p, key=KEY)
        self.assertEqual(sorted({r.target_id for r in ts.requests}),
                         ["amazon-bedrock/converse", "anthropic/direct"])
        os.unlink(p)

    def test_the_mix_is_stated(self):
        p = self._mixed()
        self.assertTrue(any("spans 2 API surfaces" in n
                            for n in load_bodies(p, key=KEY).notes))
        os.unlink(p)

    def test_the_argument_is_still_the_default(self):
        path = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False).name
        with open(path, "w") as f:
            f.write(json.dumps({"request_id": "a", "sent_at": T0.isoformat(),
                                "request": body(0), "response": response(0)}) + "\n")
        ts = load_bodies(path, key=KEY, target_id="google-cloud/vertex")
        self.assertEqual(ts.requests[0].target_id, "google-cloud/vertex")
        os.unlink(path)


class TestExplicitZeroStatusSurvivesBodyIngest(unittest.TestCase):
    """`or` chaining discarded an explicit numeric 0 and promoted a failed
    gateway row to success, so degraded traffic carrying usage counters landed
    in `analysable` and coverage read 100%. load_jsonl already preserved it."""

    def _file(self, row_extra):
        path = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False).name
        row = {"request_id": "a", "sent_at": T0.isoformat(),
               "request": {"model": "claude-opus-5",
                           "messages": [{"role": "user", "content": "x"}]},
               "response": {"usage": {"input_tokens": 100, "output_tokens": 1,
                                      "cache_read_input_tokens": 0,
                                      "cache_creation_input_tokens": 0,
                                      "cache_creation": {}}}}
        row.update(row_extra)
        with open(path, "w") as f:
            f.write(json.dumps(row) + "\n")
        return path

    def test_an_explicit_zero_is_kept(self):
        p = self._file({"status": 0})
        ts = load_bodies(p, key=KEY)
        self.assertEqual(ts.requests[0].status, 0)
        self.assertEqual(len(ts.analysable), 0)
        os.unlink(p)

    def test_a_missing_status_still_defaults_to_success(self):
        p = self._file({})
        self.assertEqual(load_bodies(p, key=KEY).requests[0].status, 200)
        os.unlink(p)

    def test_it_agrees_with_the_normalised_loader(self):
        p = self._file({"status": 0})
        self.assertEqual(len(load_bodies(p, key=KEY).analysable), 0)
        os.unlink(p)


class TestUsageIsFoundWhereverTheExportPutIt(unittest.TestCase):
    """Exports disagree about where usage lives. Taking the first object shaped
    like a response meant a row with `response: {"id": ...}` and a top-level
    `usage` dropped eleven thousand billed write tokens: the request became
    unanalysable and shrank spend, cache ratios and the coverage denominator
    while the ingest looked clean."""

    USAGE = {"input_tokens": 5, "cache_read_input_tokens": 0,
             "cache_creation_input_tokens": 11_000,
             "cache_creation": {"ephemeral_5m_input_tokens": 11_000,
                                "ephemeral_1h_input_tokens": 0}}
    REQUEST = {"model": "claude-opus-5",
               "system": [{"type": "text", "text": "p" * 40_000}],
               "messages": [{"role": "user", "content": "hi"}]}

    def _load(self, **extra):
        import json
        import tempfile
        from cacheeconomics.adapters.bodies import load_bodies
        row = {"sent_at": "2026-07-29T09:00:00Z", "request": self.REQUEST, **extra}
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        try:
            with open(path, "w") as f:
                f.write(json.dumps(row))
            return load_bodies(path, b"k" * 32)
        finally:
            os.unlink(path)

    def test_a_metadata_only_response_does_not_hide_top_level_usage(self):
        ts = self._load(response={"id": "m1", "stop_reason": "end_turn"},
                        usage=dict(self.USAGE))
        self.assertEqual(ts.requests[0].usage["cache_creation_input_tokens"], 11_000)
        self.assertEqual(len(ts.analysable), 1)

    def test_usage_inside_the_response_still_wins(self):
        ts = self._load(response={"id": "m1", "usage": dict(self.USAGE)})
        self.assertEqual(ts.requests[0].usage["cache_creation_input_tokens"], 11_000)

    def test_the_response_is_preferred_when_both_carry_accounting(self):
        ts = self._load(response={"id": "m1", "usage": dict(self.USAGE)},
                        usage={"input_tokens": 999})
        self.assertEqual(ts.requests[0].usage["cache_creation_input_tokens"], 11_000)

    def test_a_row_with_no_accounting_anywhere_is_still_excluded(self):
        """The fix must not turn "no evidence" into "zero"."""
        ts = self._load(response={"id": "m1"})
        self.assertEqual(len(ts.analysable), 0)


class TestAMessageLevelMarkerReachesSegmentation(unittest.TestCase):
    """Anthropic counts `cache_control` on the message object. `walk` yields
    content blocks, by design -- those are the segments whose text is hashed and
    priced -- so nothing tied the two together.

    Measured before the fix: one message-level marker gave `marker_count == 1`,
    zero segments with `cache_marked`, and `_requested_ttl == None`. The recorder
    and the body adapter therefore stored a request the provider billed as cached
    as though it carried no marker, and the as-shipped bake-off arm replayed it
    uncached -- crediting the allocator with a saving over a baseline that was
    never really uncached.
    """

    KEY = b"k" * 32

    def _body(self, ttl=None, blocks=2):
        cc = {"type": "ephemeral"}
        if ttl:
            cc["ttl"] = ttl
        return {"model": "claude-opus-5",
                "messages": [
                    {"role": "system", "cache_control": cc,
                     "content": [{"type": "text", "text": f"policy {i} " * 200}
                                 for i in range(blocks)]},
                    {"role": "user",
                     "content": [{"type": "text", "text": "question"}]}]}

    def test_the_marker_lands_on_the_messages_last_block(self):
        """A message-level marker caches up to and including that message, so
        the boundary is its final content block, not its first."""
        from cacheeconomics.segment import segments_from_request
        segs = segments_from_request(self._body(blocks=3), self.KEY)
        marked = [i for i, s in enumerate(segs) if s["cache_marked"]]
        self.assertEqual(len(marked), 1, segs)
        self.assertEqual(marked[0], 2, "should mark the message's last block")

    def test_counting_and_segmentation_agree(self):
        from cacheeconomics.segment import marker_count, segments_from_request
        body = self._body()
        segs = segments_from_request(body, self.KEY)
        self.assertEqual(marker_count(body),
                         sum(1 for s in segs if s["cache_marked"]))

    def test_the_requested_lifetime_is_visible(self):
        """An unprovable write lifetime is excluded from every dollar figure, so
        losing this silently shrinks the priced set."""
        from cacheeconomics.segment import _requested_ttl
        self.assertEqual(_requested_ttl(self._body()), "5m")
        self.assertEqual(_requested_ttl(self._body(ttl="1h")), "1h")

    def test_a_block_level_lifetime_still_wins(self):
        """It is the more specific statement about that exact boundary."""
        from cacheeconomics.segment import segments_from_request
        body = self._body(ttl="1h", blocks=1)
        body["messages"][0]["content"][0]["cache_control"] = {
            "type": "ephemeral", "ttl": "5m"}
        segs = segments_from_request(body, self.KEY)
        self.assertEqual(segs[0]["ttl"], "5m")

    def test_a_marker_on_an_empty_message_does_not_crash(self):
        from cacheeconomics.segment import segments_from_request
        body = {"model": "claude-opus-5",
                "messages": [{"role": "user", "content": [],
                              "cache_control": {"type": "ephemeral"}},
                             {"role": "user",
                              "content": [{"type": "text", "text": "hi"}]}]}
        segs = segments_from_request(body, self.KEY)
        self.assertFalse(any(s["cache_marked"] for s in segs))


class TestAnEmptyUsageCannotShadowARealOne(unittest.TestCase):
    """`_find_response` takes the first candidate whose usage is truthy, and
    `usage_from_response` treated the mere presence of a `cache_creation` key as
    accounting.

    Measured: a nested `{"usage": {"cache_creation": {}}}` beat a top-level
    `usage` carrying 11,000 real input tokens. The row stayed analysable at zero
    input, so billed spend vanished from every ratio while coverage still read
    100% -- a degraded export presenting as a complete one, which is the exact
    failure the surrounding guard was written to stop.
    """

    KEY = b"k" * 32

    def test_an_empty_cache_creation_is_not_accounting(self):
        from cacheeconomics.segment import usage_from_response
        self.assertEqual(usage_from_response({"usage": {"cache_creation": {}}}), {})

    def test_output_tokens_alone_is_not_input_accounting(self):
        """This package prices input. A response carrying only an output count
        is not accounting it can use, and treating it as such shadows accounting
        that is."""
        from cacheeconomics.segment import usage_from_response
        self.assertEqual(usage_from_response({"usage": {"output_tokens": 5}}), {})

    def test_a_real_per_lifetime_split_is_accounting(self):
        from cacheeconomics.segment import usage_from_response
        u = usage_from_response({"usage": {"cache_creation": {
            "ephemeral_5m_input_tokens": 900, "ephemeral_1h_input_tokens": 0}}})
        self.assertTrue(u)
        self.assertEqual(u["cache_creation"]["ephemeral_5m_input_tokens"], 900)

    def test_an_all_zero_usage_is_not_accounting_either(self):
        """The follow-on defect. Fixing "key missing" and leaving "key present
        but zero" is the same hole one level over: a response that billed
        nothing is a placeholder, and it shadowed real accounting exactly as the
        empty dict had."""
        from cacheeconomics.segment import usage_from_response
        self.assertEqual(usage_from_response({"usage": {
            "input_tokens": 0, "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0}}), {})

    def test_a_single_positive_counter_is_enough(self):
        """A pure cache-read request bills zero uncached input, and it is still
        a real request. The gate has to let it through."""
        from cacheeconomics.segment import usage_from_response
        self.assertTrue(usage_from_response({"usage": {
            "input_tokens": 0, "cache_read_input_tokens": 5_000,
            "cache_creation_input_tokens": 0}}))

    def test_nonsense_counters_are_not_accounting(self):
        from cacheeconomics.segment import usage_from_response
        for bad in (True, float("nan"), float("inf"), -5, "1000"):
            with self.subTest(value=bad):
                self.assertEqual(
                    usage_from_response({"usage": {"input_tokens": bad}}), {})

    def test_a_zeroed_nested_usage_does_not_shadow_a_real_one(self):
        row = {"sent_at": "2026-07-29T09:00:00Z",
               "request": {"model": "claude-opus-5",
                           "system": [{"type": "text", "text": "p" * 40_000}],
                           "messages": [{"role": "user", "content": "hi"}]},
               "usage": {"input_tokens": 11_000, "cache_read_input_tokens": 0,
                         "cache_creation_input_tokens": 0},
               "response": {"usage": {"input_tokens": 0,
                                      "cache_read_input_tokens": 0,
                                      "cache_creation_input_tokens": 0}}}
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        try:
            with open(path, "w") as f:
                f.write("\n".join(json.dumps(row) for _ in range(12)))
            ts = load_bodies(path, self.KEY)
            from cacheeconomics.trace import _billed_input
            self.assertEqual(_billed_input(ts.requests[0].usage), 11_000,
                             "a zeroed response object shadowed real usage")
        finally:
            os.unlink(path)

    def test_top_level_usage_survives_an_empty_nested_one(self):
        row = {"sent_at": "2026-07-29T09:00:00Z",
               "request": {"model": "claude-opus-5",
                           "system": [{"type": "text", "text": "p" * 40_000}],
                           "messages": [{"role": "user", "content": "hi"}]},
               "usage": {"input_tokens": 11_000, "cache_read_input_tokens": 0,
                         "cache_creation_input_tokens": 0},
               "response": {"usage": {"cache_creation": {}}}}
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        try:
            with open(path, "w") as f:
                f.write("\n".join(json.dumps(row) for _ in range(12)))
            ts = load_bodies(path, self.KEY)
            billed = sum(ts.requests[0].usage.get(k, 0) for k in
                         ("input_tokens", "cache_read_input_tokens",
                          "cache_creation_input_tokens"))
            self.assertEqual(billed, 11_000,
                             "an empty nested usage shadowed real accounting")
        finally:
            os.unlink(path)


class TestSplitOnlyCacheWritesAreBilledInput(unittest.TestCase):
    """Anthropic can report the per-lifetime write split without the aggregate.
    `cost.Usage` prices writes off that split; `_billed_input` summed only the
    scalar counters. Two functions that must agree about "what was billed" had
    drifted, and only one of them guarded money.

    Measured: a split-only usage was priced for 9,000 write tokens while
    `_billed_input` reported 0. `_scale_to_measured` then had no measured total
    and fell back to byte estimates, `segment_sum_ratio` returned None for "no
    opinion", and `sums_publishable(None)` passes -- so the gate that stops
    structural dollars being published from unmeasured sizes passed by default,
    on precisely the shape where the sizes were guaranteed to be guesses.
    """

    SPLIT_ONLY = {"input_tokens": 0, "cache_read_input_tokens": 0,
                  "cache_creation": {"ephemeral_5m_input_tokens": 9_000,
                                     "ephemeral_1h_input_tokens": 0}}

    def test_billed_input_counts_the_split(self):
        from cacheeconomics.trace import _billed_input
        self.assertEqual(_billed_input(self.SPLIT_ONLY), 9_000)

    def test_it_agrees_with_what_the_cost_model_prices(self):
        """The invariant behind the finding, asserted directly."""
        from cacheeconomics import cost
        from cacheeconomics.trace import _billed_input
        u = cost.Usage.from_anthropic(self.SPLIT_ONLY)
        priced = u.uncached_input + u.cache_read + u.cache_write_5m + u.cache_write_1h
        self.assertEqual(_billed_input(self.SPLIT_ONLY), priced)

    def test_the_size_gate_now_has_an_opinion(self):
        from cacheeconomics.trace import segment_sum_ratio, sums_publishable
        ratio = segment_sum_ratio([500, 500], self.SPLIT_ONLY)
        self.assertIsNotNone(ratio, "no opinion means the gate passes by default")
        self.assertFalse(sums_publishable(ratio))

    def test_the_aggregate_still_wins_when_present(self):
        """It is what the provider states. A split disagreeing with it is a data
        quality problem, not an arithmetic one to paper over here -- and adding
        both would double-count the normal case, where the split sums to it."""
        from cacheeconomics.trace import _billed_input
        both = {"input_tokens": 0, "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 9_000,
                "cache_creation": {"ephemeral_5m_input_tokens": 9_000,
                                   "ephemeral_1h_input_tokens": 0}}
        self.assertEqual(_billed_input(both), 9_000)

    def test_a_nonsense_split_does_not_inflate_billed_input(self):
        from cacheeconomics.trace import _billed_input
        self.assertEqual(_billed_input({
            "input_tokens": 100, "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": "lots",
                               "ephemeral_1h_input_tokens": True}}), 100)


class TestMalformedCountersCannotBePriced(unittest.TestCase):
    """Pricing multiplies a counter by a rate and sums across a trace, so a
    negative one *subtracts* from real spend instead of failing.

    Measured: a single row with `input_tokens: -1,000,000` cancelled a real $10
    request down to exactly $5.00, matched a $5.00 invoice at 0.0%, passed the
    publication gate and released the figure. NaN produced `$nan`, `True` priced
    as one token, and a string in the nested split raised TypeError out of the
    multiplication where no caller could catch it.

    ValueError rather than a silent zero, because the analyzer already routes
    that to `unprovable`: excluded from every dollar figure, and said so.
    """

    def test_each_bad_shape_is_refused(self):
        from cacheeconomics import cost
        for label, usage in (
                ("negative input", {"input_tokens": -1}),
                ("negative read", {"cache_read_input_tokens": -1}),
                ("negative write", {"cache_creation_input_tokens": -1}),
                ("nan", {"input_tokens": float("nan")}),
                ("inf", {"input_tokens": float("inf")}),
                ("bool", {"input_tokens": True}),
                ("string", {"input_tokens": "1000"}),
                ("split not an object", {"cache_creation": 5}),
                ("negative in split", {"cache_creation": {
                    "ephemeral_5m_input_tokens": -1,
                    "ephemeral_1h_input_tokens": 0}}),
                ("string in split", {"cache_creation": {
                    "ephemeral_5m_input_tokens": "9000",
                    "ephemeral_1h_input_tokens": 0}})):
            with self.subTest(shape=label):
                with self.assertRaises(ValueError):
                    cost.Usage.from_anthropic(usage)

    def test_a_valid_row_still_prices(self):
        """The guard has to let real accounting through, or it is an off switch."""
        from cacheeconomics import cost
        u = cost.Usage.from_anthropic({
            "input_tokens": 1_000, "cache_read_input_tokens": 5_000,
            "cache_creation_input_tokens": 900,
            "cache_creation": {"ephemeral_5m_input_tokens": 900,
                               "ephemeral_1h_input_tokens": 0}})
        self.assertEqual(u.uncached_input, 1_000)
        self.assertEqual(u.cache_write_5m, 900)

    def test_a_negative_row_no_longer_reconciles(self):
        from datetime import datetime, timedelta, timezone
        from cacheeconomics.analyzer import analyze
        from cacheeconomics.trace import Request, Tier, TraceSet
        t0 = datetime(2026, 7, 29, 9, tzinfo=timezone.utc)
        reqs = [Request(request_id="good", sent_at=t0, model="claude-opus-5",
                        agent="a", session="s", ttl_requested="5m",
                        usage={"input_tokens": 2_000_000,
                               "cache_read_input_tokens": 0,
                               "cache_creation_input_tokens": 0}, segments=[]),
                Request(request_id="evil", sent_at=t0 + timedelta(seconds=60),
                        model="claude-opus-5", agent="a", session="s",
                        ttl_requested="5m",
                        usage={"input_tokens": -1_000_000,
                               "cache_read_input_tokens": 0,
                               "cache_creation_input_tokens": 0}, segments=[])]
        a = analyze(TraceSet(requests=reqs, tier=Tier.USAGE_ONLY, source="x"),
                    invoice_usd=5.0)
        self.assertFalse(a.reconciliation["within_ship_gate"])
        self.assertFalse(a.spend["input_usd"].released)
        self.assertEqual(a.spend["input_usd"].raw(), 10.0,
                         "the real request must still be priced in full")


class TestSplitOnlyWritesReachEveryReader(unittest.TestCase):
    """The cost model reads the per-lifetime split; nine other sites read only
    the aggregate. On a split-only export each of them saw zero while the price
    kept counting -- four review rounds of the same defect one place over."""

    SPLIT_ONLY = {"input_tokens": 0, "cache_read_input_tokens": 0,
                  "cache_creation": {"ephemeral_5m_input_tokens": 200_000,
                                     "ephemeral_1h_input_tokens": 0}}

    def test_a_split_only_request_is_analysable_at_all(self):
        """Found by the audit, not by the review. `has_usage` checked the three
        scalars, so such a request never reached `.analysable` and coverage
        called it "no usage fields" while it carried 200,000 billable tokens."""
        from cacheeconomics.trace import Request, Tier, TraceSet
        r = Request(request_id="r", sent_at=None, model="claude-opus-5",
                    usage=dict(self.SPLIT_ONLY), segments=[])
        self.assertTrue(r.has_usage)
        ts = TraceSet(requests=[r], tier=Tier.USAGE_ONLY, source="x")
        self.assertEqual(len(ts.analysable), 1)

    def test_the_rebuild_rule_sees_it(self):
        from datetime import datetime, timedelta, timezone
        from cacheeconomics.analyzer import analyze
        from cacheeconomics.trace import Request, Tier, TraceSet
        t0 = datetime(2026, 7, 29, 9, tzinfo=timezone.utc)

        def trace(split_only):
            usage = dict(self.SPLIT_ONLY)
            if not split_only:
                usage["cache_creation_input_tokens"] = 200_000
            return TraceSet(
                requests=[Request(request_id=f"r{i}",
                                  sent_at=t0 + timedelta(seconds=120 * i),
                                  model="claude-opus-5", agent="main",
                                  session="s", ttl_requested="5m",
                                  usage=dict(usage), segments=[])
                          for i in range(14)],
                tier=Tier.USAGE_ONLY, source="x")

        with_agg = [f.code for f in analyze(trace(False),
                                            allow_unreconciled=True).findings]
        without = [f.code for f in analyze(trace(True),
                                           allow_unreconciled=True).findings]
        self.assertIn("REB-1", with_agg)
        self.assertEqual(with_agg, without,
                         "the same billed traffic produced different findings "
                         "depending only on which shape the provider reported")


class TestCorruptTranscriptLinesAreCounted(unittest.TestCase):
    """`load_sessions` treated every JSONDecodeError as a live partial write and
    moved on. Only a *trailing* unparseable line is plausibly that; one in the
    middle is a corrupt or truncated record, and if it was a billed assistant
    turn its spend vanished from usage, from coverage and from the reconciliation
    denominator -- letting a parsed subset inside the 5% window release figures
    over an unknown total.

    These transcripts are the one input this tool reads while the writer may
    still be appending, which is why the trailing case keeps the benefit of the
    doubt and no other line does.
    """

    GOOD = {"type": "assistant", "sessionId": "s1",
            "timestamp": "2026-07-29T09:00:00Z",
            "message": {"usage": {"input_tokens": 1_000_000,
                                  "cache_read_input_tokens": 0,
                                  "cache_creation_input_tokens": 0}}}

    def _root(self, body):
        import tempfile
        root = tempfile.mkdtemp()
        proj = os.path.join(root, "proj")
        os.makedirs(proj)
        with open(os.path.join(proj, "a.jsonl"), "w") as f:
            f.write(body)
        return root

    def test_a_multi_line_record_is_read_not_counted_as_loss(self):
        """Transcript records carry raw newlines inside string values, so a
        record is not always a line. Counting each continuation fragment as lost
        data was an over-block I shipped: on this machine's own transcripts it
        flagged hundreds of lines, and the recoverable ones were `user` turns
        with no usage. Joining them also *recovers* records a line-at-a-time
        reader dropped entirely -- the real trace gained 24."""
        from cacheeconomics.adapters.claude_code import load_sessions
        multi = dict(self.GOOD)
        body = json.dumps(multi).replace('"s1"', '"s1\\n\\nwith a newline"')
        # Write it with the escape turned into a real newline, as the writer does.
        body = body.replace("\\n", "\n")
        ts = load_sessions(root=self._root(body + "\n"))
        self.assertEqual(ts.skipped_rows, 0, "a joinable record is not loss")
        self.assertEqual(len(ts.requests), 1)

    def test_a_mid_file_corrupt_line_is_counted(self):
        from cacheeconomics.adapters.claude_code import load_sessions
        # `{corrupt` opens a record that never closes; the next line opens
        # another, which is what proves the first was unrecoverable.
        body = (json.dumps(self.GOOD) + "\n{corrupt\n" + json.dumps(self.GOOD) + "\n")
        ts = load_sessions(root=self._root(body))
        self.assertEqual(len(ts.requests), 2)
        self.assertEqual(ts.skipped_rows, 1)

    def test_a_trailing_partial_write_is_not(self):
        """A session being written right now ends mid-line. Counting that as
        data loss would put every live transcript permanently over the gate."""
        from cacheeconomics.adapters.claude_code import load_sessions
        body = json.dumps(self.GOOD) + "\n{partial write in progress"
        ts = load_sessions(root=self._root(body))
        self.assertEqual(len(ts.requests), 1)
        self.assertEqual(ts.skipped_rows, 0)

    def test_a_corrupt_line_blocks_reconciliation(self):
        from cacheeconomics.adapters.claude_code import load_sessions
        body = (json.dumps(self.GOOD) + "\n{corrupt\n" + json.dumps(self.GOOD) + "\n")
        from cacheeconomics.analyzer import analyze
        ts = load_sessions(root=self._root(body))
        a = analyze(ts, invoice_usd=10.0)
        self.assertFalse(a.reconciliation["within_ship_gate"])
        self.assertEqual(a.reconciliation["blockers"]["skipped_rows"], 1)


if __name__ == "__main__":
    unittest.main()
