"""Tests for allocator-lite, relocation-lite, the cache simulator, and the
three publication guards an adversarial review found holes in.

Stdlib unittest, no pytest dependency. Run: python3 -m unittest discover tests
"""

import json
import os
import sys
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cacheeconomics import money, relocate, simulate  # noqa: E402
from cacheeconomics.allocate import (allocator_lite, litellm_auto,  # noqa: E402
                                     observed_volatility)
from cacheeconomics.analyzer import _usages, analyze  # noqa: E402
from cacheeconomics.report import render_text  # noqa: E402
from cacheeconomics.trace import (Request, Segment, Tier,  # noqa: E402
                                  TraceSet, is_trusted_id, load_jsonl)

T0 = datetime(2026, 7, 1, 9, 0, 0, tzinfo=timezone.utc)


def hid(name: str) -> str:
    """An id shaped like one segment_id() would produce.

    Fixtures used to carry short strings like "hmac:abc", which the loader
    accepted while the recorder could never emit them. Tightening the check
    caught the fixtures rather than the code.
    """
    import hashlib
    return "hmac:" + hashlib.sha256(name.encode()).hexdigest()


def seg(i, role, tokens, sid, label="", marked=False, ttl=None):
    return Segment(id=sid, role=role, tokens=tokens, index=i, label=label,
                   cache_marked=marked, ttl=ttl)


def req(n, segments, gap=60, model="claude-opus-5", usage=None):
    return Request(request_id=f"r{n}", sent_at=T0 + timedelta(seconds=gap * n),
                   model=model, usage=usage or {}, segments=segments,
                   agent="a", target_id="anthropic/direct")


def instrumented(reqs, **kw):
    """The trace a list of hand-built requests came from, stated.

    `bake_off` will not answer the gate over segment boundaries whose
    provenance nobody has claimed -- alignment, coverage and whether the sizes
    were counted live on the `TraceSet` and on nothing a `Request` carries, so
    a bare list means "not stated" and not "fine". Tests that assert a verdict
    say what they are asserting it over; tests that assert an *indeterminate*
    verdict do not need this, because the condition they are testing blocks
    first either way.

    Rows with no `usage` get the minimum that makes them internally consistent:
    the segment total as uncached input. Many simulator fixtures were written
    structure-only, with `usage={}` standing for "this test is not about
    counters" -- but a row with no counters is not in `analysable`, so a
    `TraceSet` built from it declares every row a billed row no arm could
    model, which is both true of the declaration and false of the intent.

    Synthetic, and safe to synthesise here because it was measured rather than
    assumed: over `volatile_head(20)`, `volatile_head(30)` and two other
    structure-only fixtures, adding this usage left every arm's reads, writes
    and raw spend byte-identical. The arms price from segments and the modelled
    cache and never read `usage`. It deliberately claims no cache behaviour --
    zero reads, zero writes -- because the fixture never recorded any; it claims
    only that the prompt these segments describe was billed once at its own
    size, which is what makes `segment_sum_ratio` 1.0 and the row analysable.
    """
    fixed = [r if r.has_usage
             else replace(r, usage={"input_tokens": sum(s.tokens for s in r.segments),
                                    "cache_read_input_tokens": 0,
                                    "cache_creation_input_tokens": 0})
             for r in reqs]
    return TraceSet(requests=fixed, tier=Tier.INSTRUMENTED, **kw)


def volatile_head(n=6, model="claude-opus-5"):
    """A volatile 90-token system header above 20k of stable system content.

    The shape that makes relocation matter: the marker has nowhere useful to go
    until the header moves.
    """
    return [req(i, [seg(0, "system", 90, f"vary{i}", "session_ctx"),
                    seg(1, "tools", 6000, "tools", "tool_definitions"),
                    seg(2, "system", 14000, "sys", "system_instructions"),
                    seg(3, "user", 200, f"turn{i}", "user_turn")], model=model)
            for i in range(n)]


class TestAllocatorLite(unittest.TestCase):

    def test_marker_stops_before_the_first_volatile_segment(self):
        reqs = volatile_head()
        v = observed_volatility(reqs)
        plan = allocator_lite(reqs[0], volatility=v)
        # Segment 0 varies, so nothing can be cached at all.
        self.assertEqual(plan.marker_indices, [])
        self.assertTrue(any("changes between requests" in n for n in plan.notes))

    def test_marker_lands_on_the_deepest_stable_segment(self):
        reqs = [req(i, [seg(0, "system", 6000, "sys"),
                        seg(1, "tools", 4000, "tools"),
                        seg(2, "user", 100, f"turn{i}")]) for i in range(5)]
        plan = allocator_lite(reqs[0], volatility=observed_volatility(reqs))
        self.assertEqual(plan.marker_indices, [1])

    def test_refuses_to_mark_a_prefix_below_the_model_minimum(self):
        """Below the minimum the provider caches nothing and reports nothing.

        A marker there is pure write premium, so it must not be placed.
        """
        reqs = [req(i, [seg(0, "system", 100, "sys"),
                        seg(1, "user", 50, f"turn{i}")],
                    model="claude-haiku-4-5") for i in range(5)]
        plan = allocator_lite(reqs[0], volatility=observed_volatility(reqs))
        self.assertEqual(plan.marker_indices, [])
        self.assertTrue(any("4,096 minimum" in n for n in plan.notes))

    def test_ttl_follows_the_band_not_a_preference(self):
        """1h only wins between 5 minutes and an hour. Outside it, 5m is cheaper."""
        stable = [seg(0, "system", 6000, "sys"), seg(1, "user", 50, "t")]
        inside = allocator_lite(req(0, stable), volatility={0: 1, 1: 1},
                                cadence_seconds=900)
        outside_low = allocator_lite(req(0, stable), volatility={0: 1, 1: 1},
                                     cadence_seconds=30)
        outside_high = allocator_lite(req(0, stable), volatility={0: 1, 1: 1},
                                      cadence_seconds=7200)
        self.assertEqual(inside.ttls[1], "1h")
        self.assertEqual(outside_low.ttls[1], "5m")
        self.assertEqual(outside_high.ttls[1], "5m")


class TestMultiBreakpointCache(unittest.TestCase):
    """The simulator must model each marker as its own cache entry.

    Treating only the outermost marker as the cache span made a policy that
    pairs a stable system breakpoint with an advancing trailing-turn breakpoint
    appear never to hit, which understated the baseline sixfold and would have
    handed the tool a fabricated win.
    """

    def test_stable_breakpoint_hits_while_trailing_one_advances(self):
        """Markers on the request, not invented by the policy.

        This drove `litellm_auto` for its two markers, back when that policy
        placed them from a role heuristic. LiteLLM does no such thing -- it
        forwards the caller's markers and adds none -- so the request now
        carries them, which is what the simulator is being tested on anyway.
        """
        reqs = [req(i, [seg(0, "system", 6000, "sys", marked=True, ttl="5m"),
                        seg(1, "user", 200, f"turn{i}", marked=True, ttl="5m")])
                for i in range(6)]
        plan = litellm_auto(reqs[0])
        self.assertEqual(len(plan.marker_indices), 2)      # system + trailing
        res = simulate.simulate(reqs, "litellm-auto")
        self.assertGreater(res.reads, 0,
                           "the system prefix is identical every request and must hit")

    def test_a_prefix_below_the_minimum_is_not_a_cache_entry(self):
        reqs = [req(i, [seg(0, "system", 100, "sys"),
                        seg(1, "user", 50, f"t{i}")],
                    model="claude-haiku-4-5") for i in range(4)]
        res = simulate.simulate(reqs, "litellm-auto")
        self.assertEqual(res.reads, 0)
        self.assertEqual(res.writes, 0)
        self.assertEqual(res.cold, 4)

    def test_hit_refreshes_the_ttl_at_no_write_cost(self):
        """Measured on 2026-07-28. Requests 290s apart never expire a 5m entry."""
        segs = lambda i: [seg(0, "system", 6000, "sys"), seg(1, "user", 50, f"t{i}")]
        reqs = [req(i, segs(i), gap=290) for i in range(8)]
        res = simulate.simulate(reqs, "allocator-lite")
        self.assertEqual(res.writes, 1, "one write, then hits refreshing indefinitely")
        self.assertEqual(res.reads, 7)


class TestRelocationLite(unittest.TestCase):

    def test_prefers_moving_within_the_authority_block(self):
        """Moving to the end of the system block preserves authority.

        Only if that recovers nothing should a cross-authority move be
        considered, because that one demotes system content to user authority.
        """
        reqs = volatile_head()
        moves, order = relocate.propose(reqs, observed_volatility(reqs),
                                        "claude-opus-5")
        self.assertEqual(len(moves), 1)
        m = moves[0]
        self.assertEqual(m.segment_index, 0)
        self.assertEqual(m.scope, "within-container")
        self.assertEqual(m.risk, "medium")
        self.assertEqual(m.tokens_unlocked, 20000)
        self.assertLess(order.index(1), order.index(0), "header now sits after tools")

    def test_within_authority_move_is_not_blocked_by_an_unlisted_model(self):
        """sonnet-4-6 has no recorded cross-authority mechanism.

        That must not block a move that never crosses authority in the first
        place — conflating the two blocks the safe case for the unsafe case's
        reasons.
        """
        reqs = volatile_head(model="claude-sonnet-4-6")
        moves, _ = relocate.propose(reqs, observed_volatility(reqs),
                                    "claude-sonnet-4-6")
        self.assertEqual(moves[0].risk, "medium")
        self.assertEqual(moves[0].scope, "within-container")

    def test_cross_authority_move_blocks_without_a_recorded_mechanism(self):
        """All the stable content is behind the header, so nothing else works."""
        reqs = [req(i, [seg(0, "system", 90, f"vary{i}", "session_ctx"),
                        seg(1, "user", 14000, "hist", "history")],
                    model="claude-sonnet-4-6") for i in range(5)]
        moves, _ = relocate.propose(reqs, observed_volatility(reqs),
                                    "claude-sonnet-4-6")
        self.assertEqual(moves[0].risk, "blocked")
        self.assertEqual(moves[0].scope, "cross-authority")
        self.assertIn("no recorded", moves[0].blocked_by)

    def test_blocked_moves_are_never_applied(self):
        reqs = [req(i, [seg(0, "system", 90, f"vary{i}"),
                        seg(1, "user", 14000, "hist")],
                    model="claude-sonnet-4-6") for i in range(5)]
        moves, order = relocate.propose(reqs, observed_volatility(reqs),
                                        "claude-sonnet-4-6")
        self.assertFalse(moves[0].applicable)
        self.assertEqual(order, [0, 1], "order is unchanged when the move is blocked")

    def test_conversation_turns_are_never_reordered_at_a_safe_risk_class(self):
        """The conversation block is a transcript, not a bag of segments.

        Moving a turn within it changes recency and the apparent order of
        events, so it is never offered as a within-container move the way
        system content is.
        """
        reqs = [req(i, [seg(0, "user", 500, f"vary{i}", "scratch"),
                        seg(1, "user", 14000, "hist", "history")])
                for i in range(5)]
        moves, order = relocate.propose(reqs, observed_volatility(reqs),
                                        "claude-opus-5")
        self.assertEqual(moves[0].risk, "high")
        self.assertEqual(moves[0].scope, "history-reorder")
        self.assertEqual(order, [0, 1], "high risk is reported, not applied by default")

    def test_no_move_proposed_when_nothing_is_recovered(self):
        """A volatile segment with nothing stable behind it is already correct."""
        reqs = [req(i, [seg(0, "system", 6000, "sys"),
                        seg(1, "user", 100, f"t{i}")]) for i in range(5)]
        moves, _ = relocate.propose(reqs, observed_volatility(reqs), "claude-opus-5")
        self.assertEqual(moves, [])

    def test_every_applicable_move_carries_a_rollback_and_eval_decision(self):
        reqs = volatile_head()
        moves, _ = relocate.propose(reqs, observed_volatility(reqs), "claude-opus-5")
        for m in moves:
            if m.applicable:
                self.assertTrue(m.rollback)
                self.assertIsInstance(m.eval_required, bool)

    def test_relocation_arm_flags_that_a_behavioural_eval_is_required(self):
        reqs = volatile_head()
        res = simulate.simulate(reqs, "relocation-lite")
        self.assertTrue(any("EVAL REQUIRED" in n for n in res.notes))


class TestBakeOff(unittest.TestCase):

    def test_relocation_beats_placement_when_the_prefix_is_blocked(self):
        reqs = volatile_head(n=20)
        ts = instrumented(reqs)
        b = simulate.bake_off(ts.analysable, trace=ts)
        self.assertGreater(b.delta_pct_relocation, b.delta_pct)

    def test_verdict_says_linter_below_the_gate(self):
        self.assertIn("linter", simulate._verdict("allocator-lite", 1.6))
        self.assertIn("beats", simulate._verdict("allocator-lite", 25.0))
        self.assertIn("WORSE", simulate._verdict("allocator-lite", -4.0))
        self.assertIn("ties", simulate._verdict("allocator-lite", 0.0))

    def test_relocation_verdict_is_eval_gated(self):
        v = simulate._verdict("relocation-lite", 40.0, eval_gated=True)
        self.assertIn("eval-gated", v)


class TestTTLGuardNotBypassed(unittest.TestCase):
    """`Usage.from_anthropic` requires an explicit TTL on purpose.

    Anthropic reports cache_creation_input_tokens as one number with no
    lifetime split, and a 1h write costs 2x where a 5m write costs 1.25x. The
    analyzer once supplied "5m" whenever the trace was silent, defeating that
    guard from one layer above it and understating 1h writes by 38%.
    """

    def _r(self, ttl, created=5000):
        return Request(request_id="x", sent_at=T0, model="claude-opus-5",
                       usage={"input_tokens": 100, "cache_creation_input_tokens": created},
                       segments=[], agent="a", ttl_requested=ttl, target_id="anthropic/direct")

    def test_missing_ttl_on_a_write_is_excluded_not_guessed(self):
        priced, unprovable = _usages([self._r(None)])
        self.assertEqual(priced, [])
        self.assertEqual(len(unprovable), 1)

    def test_invalid_ttl_spelling_is_excluded_not_coerced(self):
        priced, unprovable = _usages([self._r("300s")])
        self.assertEqual(priced, [])
        self.assertEqual(len(unprovable), 1)

    def test_one_hour_writes_are_priced_at_2x_not_1_25x(self):
        priced, unprovable = _usages([self._r("1h")])
        self.assertEqual(unprovable, [])
        self.assertEqual(priced[0].cache_write_1h, 5000)
        self.assertEqual(priced[0].cache_write_5m, 0)

    def test_a_request_with_no_writes_needs_no_ttl(self):
        """The multiplier lands on zero either way, so silence is harmless here."""
        r = Request(request_id="x", sent_at=T0, model="claude-opus-5",
                    usage={"input_tokens": 100, "cache_read_input_tokens": 900},
                    segments=[], agent="a", target_id="anthropic/direct")
        priced, unprovable = _usages([r])
        self.assertEqual(unprovable, [])
        self.assertEqual(priced[0].cache_read, 900)

    def test_unprovable_writes_fail_the_reconciliation_gate(self):
        ts = TraceSet(requests=[self._r(None), self._r("5m")], tier=Tier.USAGE_ONLY)
        a = analyze(ts, invoice_usd=0.03)
        self.assertEqual(a.reconciliation["unpriced_requests"], 1)
        self.assertFalse(a.reconciliation["within_ship_gate"],
                         "a known-incomplete total must not pass a gate")


class TestTextReportHonoursTheGate(unittest.TestCase):
    """HTML withheld dollars on a failed reconciliation; text printed them.

    The text report is the one that gets pasted into an email, so a figure
    escaping there defeats the gate entirely.
    """

    def _trace(self):
        r = Request(request_id="x", sent_at=T0, model="claude-opus-5",
                    usage={"input_tokens": 400000, "cache_creation_input_tokens": 100000},
                    segments=[], agent="a", ttl_requested="5m", target_id="anthropic/direct")
        return TraceSet(requests=[r] * 5, tier=Tier.USAGE_ONLY)

    def _analysis(self, invoice):
        return analyze(self._trace(), invoice_usd=invoice)

    def test_failed_reconciliation_publishes_no_dollar_figures(self):
        a = self._analysis(invoice=1.0)          # wildly wrong on purpose
        self.assertFalse(a.reconciliation["within_ship_gate"])
        out = render_text(a)
        self.assertIn("FIGURES WITHHELD", out)
        # Normalised: the report wraps to a fixed width, so a phrase that spans
        # the wrap point is present and still fails a raw substring check.
        self.assertIn("outside the ±5% gate", " ".join(out.split()),
                      "the banner names the real reason")
        self.assertNotIn("$", out, "no dollar figure may appear anywhere")
        # Visible per finding, not only in the banner. The literal moved from
        # "[figure withheld]" into the report's money column; the claim it
        # guards is that a reader can see *which* rows are missing a number,
        # which a banner on its own does not tell them.
        self.assertIn("withheld", out, "withholding is visible, not silent")
        self.assertEqual(
            sum(1 for f in a.findings
                if f.avoidable_usd_month and not f.avoidable_usd_month.released),
            out.count("withheld  "),
            "every unreleased finding says so in its own row")
        self.assertNotIn("total avoidable", out)

    def test_a_correct_invoice_publishes(self):
        """Written when a missing invoice counted as a pass; it no longer does,
        so this now exercises the path it was always meant to."""
        priced = analyze(self._trace(), invoice_usd=None, allow_unreconciled=True)
        a = self._analysis(invoice=priced.spend["input_usd"].raw())
        self.assertTrue(a.reconciliation["within_ship_gate"])
        self.assertNotIn("FIGURES WITHHELD", render_text(a))
        self.assertIn("$", render_text(a))

    def test_a_missing_invoice_names_that_as_the_reason(self):
        out = render_text(self._analysis(invoice=None))
        self.assertIn("FIGURES WITHHELD", out)
        self.assertIn("no invoice", out)


class TestTraceIdentityIsRealOrAbsent(unittest.TestCase):
    """A synthesised identity is worse than no identity.

    Redacted content makes every segment hash to the digest of the empty
    string, so they collapse to one id and read downstream as a perfectly
    stable prefix that does not exist.
    """

    def _write(self, rows):
        f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.close()
        return f.name

    def _row(self, i, segments):
        return {"request_id": f"r{i}", "sent_at": T0.isoformat(),
                "model": "claude-opus-5", "usage": {"input_tokens": 10},
                "segments": segments}

    def test_hashing_content_without_a_key_is_refused(self):
        path = self._write([self._row(i, [{"index": 0, "role": "system",
                                           "tokens": 10, "content": "hello"}])
                            for i in range(2)])
        with self.assertRaises(ValueError) as ctx:
            load_jsonl(path)
        self.assertIn("HMAC key", str(ctx.exception))
        os.unlink(path)

    def test_redacted_content_downgrades_instead_of_collapsing(self):
        path = self._write([self._row(i, [{"index": 0, "role": "system", "tokens": 10},
                                          {"index": 1, "role": "user", "tokens": 5}])
                            for i in range(3)])
        ts = load_jsonl(path, key=b"k")
        self.assertIs(ts.tier, Tier.USAGE_ONLY)
        self.assertTrue(all(not r.segments for r in ts.requests))
        self.assertTrue(any("unknowable" in n for n in ts.notes))
        os.unlink(path)

    def test_inferred_traces_claim_no_alignment_score(self):
        """Alignment can only be scored against instrumented ground truth."""
        path = self._write([self._row(i, [{"index": 0, "role": "system",
                                           "tokens": 10, "content": f"a{i}"}])
                            for i in range(2)])
        ts = load_jsonl(path, key=b"k")
        self.assertIs(ts.tier, Tier.INFERRED)
        self.assertIsNone(ts.alignment)
        os.unlink(path)

    def test_ids_present_means_instrumented(self):
        path = self._write([self._row(i, [{"index": 0, "role": "system",
                                           "tokens": 10, "id": hid("abc")}])
                            for i in range(2)])
        ts = load_jsonl(path)
        self.assertIs(ts.tier, Tier.INSTRUMENTED)
        os.unlink(path)



class TestCacheIsolationAndVisibility(unittest.TestCase):
    """Both scenarios below appear in the real 29-day trace this was built on.

    Four sessions switched model mid-conversation, and subagents run
    concurrently with the main loop. A cache keyed on the prefix alone, made
    live at send time, fabricates reads in exactly those two cases and can flip
    a bake-off verdict.
    """

    def test_a_prefix_is_not_read_across_a_model_switch(self):
        segs = lambda i: [seg(0, "system", 6000, "sys"), seg(1, "user", 50, f"t{i}")]
        alternating = []
        for i in range(6):
            r = req(i, segs(i), model="claude-opus-5" if i % 2 else "claude-opus-4-8")
            alternating.append(r)
        res = simulate.simulate(alternating, "allocator-lite")
        self.assertEqual(res.reads, 4, "each model warms its own cache separately")
        self.assertEqual(res.writes, 2, "one cold write per model, not one overall")

    def test_a_prefix_is_not_read_across_api_surfaces(self):
        segs = [seg(0, "system", 6000, "sys"), seg(1, "user", 50, "t")]
        a = req(0, segs)
        b = Request(request_id="b", sent_at=a.sent_at + timedelta(seconds=30),
                    model=a.model, usage={}, segments=segs, agent="a",
                    target_id="google-cloud/vertex")
        res = simulate.simulate([a, b], "allocator-lite")
        self.assertEqual(res.reads, 0, "a Vertex request cannot read a direct-API entry")

    def test_a_concurrent_request_cannot_read_an_unfinished_write(self):
        """Two subagent calls dispatched together must both pay the write."""
        segs = [seg(0, "system", 6000, "sys"), seg(1, "user", 50, "t")]
        first = Request(request_id="a", sent_at=T0, model="claude-opus-5", usage={},
                        segments=segs, agent="a", target_id="anthropic/direct",
                        first_token_at=T0 + timedelta(seconds=8))
        second = Request(request_id="b", sent_at=T0 + timedelta(seconds=2),
                         model="claude-opus-5", usage={}, segments=segs, agent="a",
                         target_id="anthropic/direct")
        res = simulate.simulate([first, second], "allocator-lite")
        self.assertEqual(res.reads, 0, "the second was in flight before the first returned")
        self.assertEqual(res.writes, 2)

    def test_a_request_after_the_response_does_read(self):
        segs = [seg(0, "system", 6000, "sys"), seg(1, "user", 50, "t")]
        first = Request(request_id="a", sent_at=T0, model="claude-opus-5", usage={},
                        segments=segs, agent="a", target_id="anthropic/direct",
                        first_token_at=T0 + timedelta(seconds=8))
        second = Request(request_id="b", sent_at=T0 + timedelta(seconds=20),
                         model="claude-opus-5", usage={}, segments=segs, agent="a",
                         target_id="anthropic/direct")
        res = simulate.simulate([first, second], "allocator-lite")
        self.assertEqual(res.reads, 1)


class TestFindingsNeverOutrunPricing(unittest.TestCase):
    """A note saying requests were excluded from every dollar figure is a lie
    if a finding then publishes a saving computed from those same requests."""

    def _req(self, ttl, created=200000):
        return Request(request_id="x", sent_at=T0, model="claude-opus-5",
                       usage={"input_tokens": 1000,
                              "cache_creation_input_tokens": created},
                       segments=[], agent="a", ttl_requested=ttl, target_id="anthropic/direct")

    def test_unpriceable_requests_produce_no_dollar_findings(self):
        ts = TraceSet(requests=[self._req(None) for _ in range(40)],
                      tier=Tier.USAGE_ONLY)
        a = analyze(ts)
        self.assertEqual(a.total_avoidable_month.raw(), 0,
                         "nothing priceable means nothing to claim")
        self.assertFalse(a.total_avoidable_month.released)

    def test_mixed_breakpoint_ttls_are_unprovable_without_a_breakdown(self):
        """A single aggregate write cannot be split across two lifetimes."""
        for order in (["5m", "1h"], ["1h", "5m"]):
            r = Request(request_id="x", sent_at=T0, model="claude-opus-5",
                        usage={"input_tokens": 10,
                               "cache_creation_input_tokens": 50000},
                        segments=[seg(i, "system", 6000, f"s{i}", marked=True, ttl=t)
                                  for i, t in enumerate(order)], agent="a", target_id="anthropic/direct")
            priced, unprovable = _usages([r])
            self.assertEqual(priced, [], f"{order} must not price as {order[0]}")
            self.assertEqual(len(unprovable), 1)

    def test_a_single_repeated_ttl_is_still_provable(self):
        r = Request(request_id="x", sent_at=T0, model="claude-opus-5",
                    usage={"input_tokens": 10, "cache_creation_input_tokens": 50000},
                    segments=[seg(i, "system", 6000, f"s{i}", marked=True, ttl="1h")
                              for i in range(2)], agent="a", target_id="anthropic/direct")
        priced, unprovable = _usages([r])
        self.assertEqual(unprovable, [])
        self.assertEqual(priced[0].cache_write_1h, 50000)

    def test_a_provider_breakdown_settles_mixed_ttls(self):
        """When the response states the split, mixed markers are fine."""
        r = Request(request_id="x", sent_at=T0, model="claude-opus-5",
                    usage={"input_tokens": 10, "cache_creation_input_tokens": 3000,
                           "cache_creation": {"ephemeral_5m_input_tokens": 1000,
                                              "ephemeral_1h_input_tokens": 2000}},
                    segments=[seg(i, "system", 6000, f"s{i}", marked=True, ttl=t)
                              for i, t in enumerate(["5m", "1h"])], agent="a", target_id="anthropic/direct")
        priced, unprovable = _usages([r])
        self.assertEqual(unprovable, [])
        self.assertEqual((priced[0].cache_write_5m, priced[0].cache_write_1h),
                         (1000, 2000))


class TestNoFabricatedReads(unittest.TestCase):
    """The invariant that catches the whole class, not one instance of it.

    Both simulator defects found so far — a missing isolation scope, and writes
    visible before the response returned — were cases of a read with no valid
    write behind it. Rather than testing each known case, this replays
    randomised traces and re-derives every read independently, asserting that a
    matching entry was genuinely written, visible, and unexpired.
    """

    def _mixed_traces(self, seed):
        """Deterministic pseudo-random traces. No RNG: reproducibility matters
        more than statistical purity when a failure has to be re-run."""
        models = ["claude-opus-5", "claude-opus-4-8"]
        targets = ["anthropic/direct", "google-cloud/vertex"]
        reqs, t = [], T0
        for i in range(60):
            step = ((seed * 7 + i * 13) % 900)          # 0..899s: straddles the 5m TTL
            t = t + timedelta(seconds=step)
            vary = (seed + i) % 3 == 0
            segs = [seg(0, "system", 7000, "sys"),
                    seg(1, "tools", 3000, f"tools{i}" if vary else "tools"),
                    seg(2, "user", 300, f"turn{i}")]
            reqs.append(Request(
                request_id=f"r{i}", sent_at=t, model=models[(seed + i) % 2],
                usage={}, segments=segs, agent="a",
                target_id=targets[(seed + i // 7) % 2],
                first_token_at=t + timedelta(seconds=2 + (i % 5))))
        return reqs

    def _replay_and_audit(self, reqs, policy, assume):
        """Independently re-derive what each read was allowed to be."""
        from cacheeconomics.allocate import observed_cadence, observed_volatility
        ordered = sorted(reqs, key=lambda r: r.sent_at)
        # The audit must mirror the simulator's inputs exactly. Passing a
        # different cadence changes the TTL allocator-lite picks, which changes
        # expiry, which makes the audit disagree for a reason that has nothing
        # to do with the invariant being tested.
        v, cad = observed_volatility(ordered), observed_cadence(ordered)
        res = simulate.simulate(ordered, policy, volatility=v, cadence=cad,
                                assume=assume)
        written = {}          # (scope, key) -> (visible_from, expires_at)
        for r, u in zip(res.reqs, res.usages):
            now = r.sent_at.timestamp()
            scope = (r.target_id, r.model)
            if u.cache_read:
                ok = any(e[0] <= now < e[1] for k, e in written.items()
                         if k[0] == scope)
                self.assertTrue(ok, f"{policy}/{assume.label}: {r.request_id} read "
                                    f"{u.cache_read} tokens with no live entry in its "
                                    f"own isolation scope")
            plan = simulate.POLICIES[policy](r, volatility=v, cadence_seconds=cad,
                                             moves=None, order=None)
            vis = r.first_token_at.timestamp() + assume.write_latency_s
            for key, _, ttl in plan.prefixes(r.segments):
                written[(scope, key)] = (
                    vis, now + simulate.TTL_SECONDS[ttl] * (1 - assume.eviction_haircut))
        return res

    def test_no_read_without_a_live_write_in_the_same_scope(self):
        for seed in range(6):
            for policy in ("as-shipped", "litellm-auto", "allocator-lite"):
                for assume in (simulate.NEUTRAL, simulate.PESSIMISTIC):
                    self._replay_and_audit(self._mixed_traces(seed), policy, assume)

    def test_pessimistic_never_reports_more_hits_than_neutral(self):
        """Pessimism must actually cost something, or it is decoration."""
        for seed in range(4):
            reqs = self._mixed_traces(seed)
            n = simulate.simulate(reqs, "allocator-lite", assume=simulate.NEUTRAL)
            p = simulate.simulate(reqs, "allocator-lite", assume=simulate.PESSIMISTIC)
            self.assertLessEqual(p.reads, n.reads)

    def test_the_headline_is_the_pessimistic_end_of_the_range(self):
        reqs = volatile_head(n=30)
        ts = instrumented(reqs)
        b = simulate.bake_off(ts.analysable, trace=ts)
        self.assertLessEqual(b.delta_pct, b.delta_pct_optimistic,
                             "the reported verdict must not be the flattering end")
        # A tie states itself in words rather than as "0.0%". That case became
        # reachable once litellm-auto stopped inventing markers: on a request
        # nobody marked, the automatic baseline and as-shipped are the same
        # request, so the arms genuinely tie.
        self.assertTrue(f"{b.delta_pct:.1f}" in b.verdict or "ties" in b.verdict,
                        b.verdict)

    def test_identical_policies_produce_identical_spend(self):
        """A differential check: nothing about arm identity may change cost."""
        reqs = self._mixed_traces(1)
        a = simulate.simulate(reqs, "allocator-lite", assume=simulate.PESSIMISTIC)
        b = simulate.simulate(reqs, "allocator-lite", assume=simulate.PESSIMISTIC)
        self.assertEqual(a.spend("2026-07-29"), b.spend("2026-07-29"))

    def test_simulation_is_reproducible_across_runs(self):
        """Determinism is a requirement, not a nicety: a bake-off verdict that
        moves between runs cannot be put in front of a client."""
        one = simulate.bake_off(self._mixed_traces(3))
        two = simulate.bake_off(self._mixed_traces(3))
        self.assertEqual(one.delta_pct, two.delta_pct)
        self.assertEqual(one.delta_pct_relocation, two.delta_pct_relocation)


class TestFiguresCannotLeak(unittest.TestCase):
    """Class A, closed at the value rather than at each output site."""

    def test_an_unreleased_figure_refuses_to_become_a_number(self):
        f = money.Figure(1234.5, money.MODELED)
        with self.assertRaises(money.WithheldFigure):
            float(f)
        with self.assertRaises(money.WithheldFigure):
            f.amount

    def test_an_unreleased_figure_renders_as_withheld_not_as_a_number(self):
        f = money.Figure(1234.5, money.MODELED, withheld_because="not reconciled")
        self.assertIn("withheld", str(f))
        self.assertIn("withheld", f"{f}")
        self.assertIn("withheld", f"{f:,.2f}", "a format spec must not bypass the guard")
        self.assertNotIn("1234", f"{f:,.2f}")

    def test_raw_is_the_only_way_past_the_guard(self):
        """Deliberately ugly and greppable: one place to audit."""
        self.assertEqual(money.Figure(10.0, money.MODELED).raw(), 10.0)

    def test_released_figures_render_normally(self):
        f = money.Figure(1234.5, money.MEASURED).release(True)
        self.assertIn("1,234", str(f))

    def test_a_total_is_withheld_if_any_part_is(self):
        from cacheeconomics.analyzer import Analysis, Finding
        released = Finding("A", "t", "low", "modeled", "d", 1,
                           money.Figure(10.0, money.MODELED).release(True))
        withheld = Finding("B", "t", "low", "modeled", "d", 1,
                           money.Figure(90.0, money.MODELED))
        a = Analysis(ratios={}, coverage={}, tier=Tier.USAGE_ONLY,
                     findings=[released, withheld])
        self.assertFalse(a.total_avoidable_month.released)
        self.assertIn("withheld", str(a.total_avoidable_month))

    def test_an_invalid_basis_is_refused(self):
        with self.assertRaises(ValueError):
            money.Figure(1.0, "guessed")

    def test_abs_preserves_release_state_and_magnitude(self):
        """The sign often lives in the surrounding words: "COST $0.83", not
        "COST $-0.83". A refactor once dropped abs() and produced the latter."""
        neg = money.Figure(-0.83, money.MEASURED).release(True)
        self.assertEqual(abs(neg).raw(), 0.83)
        self.assertTrue(abs(neg).released)
        self.assertNotIn("-", str(abs(neg)))
        self.assertFalse(abs(money.Figure(-1.0, money.MEASURED)).released,
                         "abs must not launder an unreleased figure")


class TestIdentityMustBeRealNotPresent(unittest.TestCase):
    """A key being present is not proof of identity.

    `{"id": ""}` and `{"id": null}` both satisfy `"id" in s`, which classified
    a trace as instrumented, skipped the privacy guards, and then hashed
    content anyway -- unkeyed, or crashing when there was no content.
    """

    def _write(self, rows):
        f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.close()
        return f.name

    def _rows(self, seg_extra, n=3):
        return [{"request_id": f"r{i}", "sent_at": T0.isoformat(),
                 "model": "claude-opus-5", "usage": {"input_tokens": 10},
                 "segments": [dict({"index": 0, "role": "system", "tokens": 10}, **seg_extra)]}
                for i in range(n)]

    def test_empty_string_id_does_not_count_as_instrumented(self):
        path = self._write(self._rows({"id": "", "content": "abc"}))
        ts = load_jsonl(path, key=b"k")
        self.assertIsNot(ts.tier, Tier.INSTRUMENTED)
        os.unlink(path)

    def test_null_id_does_not_count_as_instrumented(self):
        path = self._write(self._rows({"id": None, "content": "abc"}))
        ts = load_jsonl(path, key=b"k")
        self.assertIsNot(ts.tier, Tier.INSTRUMENTED)
        os.unlink(path)

    def test_empty_id_still_requires_an_hmac_key(self):
        """The whole point of the guard: it must not be reachable around."""
        path = self._write(self._rows({"id": "", "content": "abc"}))
        with self.assertRaises(ValueError):
            load_jsonl(path)
        os.unlink(path)

    def test_empty_id_with_no_content_downgrades_rather_than_crashing(self):
        path = self._write(self._rows({"id": ""}))
        ts = load_jsonl(path, key=b"k")
        self.assertIs(ts.tier, Tier.USAGE_ONLY)
        os.unlink(path)

    def test_whitespace_only_id_is_not_identity(self):
        path = self._write(self._rows({"id": "   ", "content": "abc"}))
        ts = load_jsonl(path, key=b"k")
        self.assertIsNot(ts.tier, Tier.INSTRUMENTED)
        os.unlink(path)


class TestRelocationSafetySpansEveryScope(unittest.TestCase):
    """Groups are not homogeneous. The real trace had four sessions switch
    model mid-conversation, so clearing a move on whichever request sorted
    first can apply it to a model the registry excludes."""

    def _mixed(self, models, target="anthropic/direct"):
        out = []
        for i in range(6):
            out.append(Request(
                request_id=f"r{i}", sent_at=T0 + timedelta(seconds=60 * i),
                model=models[i % len(models)], usage={},
                segments=[seg(0, "system", 90, f"vary{i}", "session_ctx"),
                          seg(1, "user", 14000, "hist", "history")],
                agent="a", target_id=target))
        return out

    def test_a_move_blocked_for_one_model_is_blocked_for_the_group(self):
        # Opus 5 permits authority-preserving relocation; Sonnet 5 is excluded.
        reqs = self._mixed(["claude-opus-5", "claude-sonnet-5"])
        moves, order = relocate.propose(reqs, observed_volatility(reqs))
        self.assertEqual(moves[0].risk, "blocked")
        self.assertIn("claude-sonnet-5", moves[0].blocked_by)
        self.assertIn("combinations", moves[0].blocked_by)
        self.assertEqual(order, [0, 1], "a blocked move is never applied")

    def test_a_uniform_permitted_group_still_works(self):
        reqs = self._mixed(["claude-opus-5"])
        moves, _ = relocate.propose(reqs, observed_volatility(reqs))
        self.assertEqual(moves[0].risk, "medium")

    def test_mixed_api_surfaces_are_also_a_scope_split(self):
        reqs = self._mixed(["claude-opus-5"])
        for r in reqs[::2]:
            object.__setattr__(r, "target_id", "amazon-bedrock/converse")
        moves, _ = relocate.propose(reqs, observed_volatility(reqs))
        self.assertEqual(moves[0].risk, "blocked",
                         "bedrock records no authority-preserving mechanism")

    def test_scopes_of_reports_every_combination(self):
        reqs = self._mixed(["claude-opus-5", "claude-sonnet-5"])
        self.assertEqual(relocate.scopes_of(reqs),
                         [("anthropic/direct", "claude-opus-5"),
                          ("anthropic/direct", "claude-sonnet-5")])


class TestTTLRuleChecksTheLifetimeInUse(unittest.TestCase):
    """The rule decided from cadence alone and assumed every write was 5m, so a
    workload already on 1h was told to "set a one-hour TTL" and shown a saving
    for a no-op. The real trace wrote 70M of 98M cache tokens at 1h."""

    def _agent(self, ttl, n=12, reads=5000):
        # The marker and the row agree. They used to disagree -- every request
        # claimed ttl_requested="1h" while its marked segment said "5m" -- which
        # is now treated as unprovable rather than silently trusting the row, so
        # the fixture was quietly describing a workload it did not mean.
        out = []
        for i in range(n):
            out.append(Request(
                request_id=f"r{i}", sent_at=T0 + timedelta(seconds=900 * i),
                model="claude-opus-5",
                usage={"input_tokens": 100, "cache_creation_input_tokens": 200000,
                       "cache_read_input_tokens": reads},
                segments=[seg(0, "system", 7000, "stable-prefix", "sys", marked=True, ttl=ttl),
                              seg(1, "user", 100, f"turn{i}")],
                agent="a", session="s", ttl_requested=ttl, target_id="anthropic/direct"))
        return out

    def test_no_ttl_finding_when_the_workload_already_uses_one_hour(self):
        a = analyze(TraceSet(requests=self._agent("1h"), tier=Tier.USAGE_ONLY))
        self.assertNotIn("TTL-1", [f.code for f in a.findings])

    def test_the_finding_still_fires_on_genuine_five_minute_writes(self):
        a = analyze(TraceSet(requests=self._agent("5m"), tier=Tier.USAGE_ONLY))
        self.assertIn("TTL-1", [f.code for f in a.findings])

    def test_no_ttl_finding_when_the_prefix_itself_changes(self):
        """A prefix that changes every request cannot be fixed with a lifetime.

        The fixture has to actually change it. This test previously used
        `_agent(reads=0)`, whose segment id is the constant "stable-prefix" --
        so it asserted "no TTL finding" on a workload with a perfectly stable
        prefix and merely zero reads, which is the canonical in-band case where
        a one-hour lifetime is exactly the fix. The docstring described one
        workload and the fixture built another.
        """
        out = []
        for i in range(12):
            out.append(Request(
                request_id=f"r{i}", sent_at=T0 + timedelta(seconds=900 * i),
                model="claude-opus-5",
                usage={"input_tokens": 100, "cache_creation_input_tokens": 200000,
                       "cache_read_input_tokens": 0},
                segments=[seg(0, "system", 7000, f"drifts-{i}", "sys",
                              marked=True, ttl="5m"),
                          seg(1, "user", 100, f"turn{i}")],
                agent="a", session="s", ttl_requested="5m", target_id="anthropic/direct"))
        a = analyze(TraceSet(requests=out, tier=Tier.USAGE_ONLY))
        self.assertNotIn("TTL-1", [f.code for f in a.findings])

    def test_a_stable_prefix_with_no_reads_is_exactly_the_ttl_case(self):
        """Zero reads is what an in-band cadence under a five-minute lifetime
        looks like: the entry expires before the next request, so every request
        rewrites and none reads. Requiring reads suppressed the fix and left
        EFF-1 and REB-1 pointing the operator at prefix drift that was not
        happening -- the tool's own headline thesis, misdiagnosed."""
        a = analyze(TraceSet(requests=self._agent("5m", reads=0), tier=Tier.USAGE_ONLY))
        self.assertIn("TTL-1", [f.code for f in a.findings])

    def test_and_it_is_not_also_called_a_rebuild(self):
        """An entry that expired was not rebuilt. Reporting both sends the
        reader to hunt for compaction while the real fix sits beside it."""
        a = analyze(TraceSet(requests=self._agent("5m", reads=0), tier=Tier.USAGE_ONLY))
        self.assertNotIn("REB-1", [f.code for f in a.findings])

    def test_a_mixed_workload_prices_only_the_five_minute_share(self):
        """Control holds the window and request count fixed, since the monthly
        figure is extrapolated by window and would otherwise differ for a
        reason unrelated to what is being tested."""
        def build(second_half_ttl, second_half_writes):
            out = self._agent("5m", n=6) + self._agent(second_half_ttl, n=6)
            for i, r in enumerate(out):
                object.__setattr__(r, "sent_at", T0 + timedelta(seconds=900 * i))
                if i >= 6:
                    r.usage["cache_creation_input_tokens"] = second_half_writes
            return analyze(TraceSet(requests=out, tier=Tier.USAGE_ONLY))

        mixed = build("1h", 200000)          # second half writes 1h entries
        control = build("1h", 0)             # second half writes nothing at all
        m = next(f for f in mixed.findings if f.code == "TTL-1")
        c = next(f for f in control.findings if f.code == "TTL-1")
        self.assertIn("already written at the one-hour lifetime", m.detail)
        self.assertAlmostEqual(m.avoidable_usd_month.raw(),
                               c.avoidable_usd_month.raw(), places=4,
                               msg="1h writes must contribute exactly as much as no writes")


class TestNoInvoiceIsNotAPassedGate(unittest.TestCase):
    """Omitting invoice_usd used to release every figure.

    `recon is None or within_ship_gate` read absence of evidence as evidence.
    The public claim is that a dollar figure is not publishable until it ties
    to money that actually left the account, so the default has to withhold.
    """

    def _ts(self):
        r = Request(request_id="x", sent_at=T0, model="claude-opus-5",
                    usage={"input_tokens": 400000, "cache_creation_input_tokens": 100000},
                    segments=[], agent="a", ttl_requested="5m", target_id="anthropic/direct")
        return TraceSet(requests=[r] * 5, tier=Tier.USAGE_ONLY)

    def test_absent_invoice_withholds_every_figure(self):
        a = analyze(self._ts())
        self.assertFalse(a.spend["input_usd"].released)
        self.assertFalse(a.spend["caching_saved_usd"].released)
        self.assertFalse(a.total_avoidable_month.released)
        self.assertIn("no invoice", a.spend["input_usd"].withheld_because)

    def test_absent_invoice_publishes_no_dollars_in_the_text_report(self):
        self.assertNotIn("$", render_text(analyze(self._ts())))

    def test_the_draft_override_is_explicit_and_says_so(self):
        a = analyze(self._ts(), allow_unreconciled=True)
        self.assertTrue(a.spend["input_usd"].released)
        self.assertTrue(any("DRAFT" in n for n in a.notes))

    def test_the_override_cannot_rescue_a_failed_reconciliation(self):
        """It only covers the no-invoice case. A wrong invoice still fails."""
        a = analyze(self._ts(), invoice_usd=1.0, allow_unreconciled=True)
        self.assertFalse(a.spend["input_usd"].released)


class TestTenantIsolationInTheSimulator(unittest.TestCase):
    """A shared gateway export is the normal case. Without tenant in the cache
    key, one tenant's write pays for another tenant's read."""

    def _pair(self, tenants):
        segs = [seg(0, "system", 6000, "sys"), seg(1, "user", 50, "t")]
        return [Request(request_id=f"r{i}", sent_at=T0 + timedelta(seconds=60 * i),
                        model="claude-opus-5", usage={}, segments=segs, agent="a",
                        tenant=t, target_id="anthropic/direct")
                for i, t in enumerate(tenants)]

    def test_identical_prefixes_do_not_cross_tenants(self):
        res = simulate.simulate(self._pair(["acme", "globex", "acme", "globex"]),
                                "allocator-lite")
        self.assertEqual(res.writes, 2, "each tenant warms its own cache")
        self.assertEqual(res.reads, 2)

    def test_one_tenant_alone_still_shares_its_own_cache(self):
        res = simulate.simulate(self._pair(["acme"] * 4), "allocator-lite")
        self.assertEqual(res.writes, 1)
        self.assertEqual(res.reads, 3)


class TestPartialStructureIsNotFullCoverage(unittest.TestCase):
    """`any(rows have segments)` classified a half-structured file as
    instrumented, and the simulator then priced the structureless half at zero
    tokens -- free -- understating spend while claiming counterfactual support."""

    def _write(self, rows):
        f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.close()
        return f.name

    def _mixed_file(self):
        rows = []
        for i in range(6):
            row = {"request_id": f"r{i}", "sent_at": (T0 + timedelta(seconds=60 * i)).isoformat(),
                   "model": "claude-opus-5", "usage": {"input_tokens": 10}}
            if i % 2 == 0:
                row["segments"] = [{"index": 0, "role": "system", "tokens": 6000,
                                    "id": hid("sys")},
                                   {"index": 1, "role": "user", "tokens": 50,
                                    "id": hid(f"t{i}")}]
            rows.append(row)
        return self._write(rows)

    def test_structural_coverage_is_recorded_and_stated(self):
        path = self._mixed_file()
        ts = load_jsonl(path)
        self.assertAlmostEqual(ts.structural_coverage, 0.5)
        self.assertTrue(any("carry prompt structure" in n for n in ts.notes))
        os.unlink(path)

    def test_structureless_requests_are_excluded_not_priced_as_free(self):
        path = self._mixed_file()
        ts = load_jsonl(path)
        res = simulate.simulate(ts.analysable, "allocator-lite")
        self.assertEqual(res.unstructured, 3)
        self.assertEqual(len(res.usages), 3)
        self.assertEqual(len(res.priced), 3, "usages must stay paired with requests")
        os.unlink(path)

    def test_the_bake_off_reports_what_it_skipped(self):
        path = self._mixed_file()
        b = simulate.bake_off(load_jsonl(path).analysable)
        self.assertEqual(b.unstructured, 3)
        self.assertIn("carry no prompt structure", str(b))
        os.unlink(path)

    def test_a_fully_structured_file_reports_full_coverage(self):
        path = self._write([
            {"request_id": f"r{i}", "sent_at": T0.isoformat(), "model": "claude-opus-5",
             "usage": {"input_tokens": 10},
             "segments": [{"index": 0, "role": "system", "tokens": 10, "id": hid("a")}]}
            for i in range(3)])
        ts = load_jsonl(path)
        self.assertEqual(ts.structural_coverage, 1.0)
        self.assertIs(ts.tier, Tier.INSTRUMENTED)
        os.unlink(path)


class TestTTLSavingsChargeTheColdWrite(unittest.TestCase):
    """Switching to one hour does not turn the first write into a read -- it
    makes that cold write cost 2.0x instead of 1.25x. Charging every write the
    full write-to-read delta overstated short sessions by nearly 6x."""

    def _sessions(self, n_sessions, per_session, gap=900):
        reqs, t = [], T0
        for s in range(n_sessions):
            for i in range(per_session):
                reqs.append(Request(
                    request_id=f"{s}-{i}", sent_at=t, model="claude-opus-5",
                    usage={"input_tokens": 100, "cache_creation_input_tokens": 1_000_000,
                           "cache_read_input_tokens": 5000},
                    segments=[seg(0, "system", 7000, "stable-prefix", "sys", marked=True, ttl="5m"),
                              seg(1, "user", 100, f"turn{i}")],
                    agent="a", session=f"sess{s}", ttl_requested="5m", target_id="anthropic/direct"))
                t += timedelta(seconds=gap)
        return reqs

    def _figure(self, reqs):
        a = analyze(TraceSet(requests=reqs, tier=Tier.USAGE_ONLY), allow_unreconciled=True)
        f = [x for x in a.findings if x.code == "TTL-1"]
        return f[0].avoidable_usd_month.raw() if f else 0.0

    def test_short_sessions_are_worth_far_less_than_long_ones(self):
        """Same request count, same window, same tokens -- only the session
        boundaries differ, and each boundary costs a cold write."""
        short = self._figure(self._sessions(6, 2))
        long_ = self._figure(self._sessions(1, 12))
        self.assertLess(short, long_)
        self.assertLess(short, long_ / 3,
                        "six cold writes instead of one is a large difference")

    def test_a_single_write_per_session_is_never_a_saving(self):
        """Nothing follows it inside the window, so there is no read to gain
        and the one-hour premium is pure cost."""
        self.assertEqual(self._figure(self._sessions(6, 1)), 0.0)

    def test_sessions_are_isolated_from_each_other(self):
        """Identical traffic, identical window, split into two sessions instead
        of one. The split costs one extra cold write, so it must be worth less.
        If sessions shared a cache scope the two would be equal."""
        one = self._figure(self._sessions(1, 6))
        two = self._figure(self._sessions(2, 3))
        self.assertGreater(one, 0)
        self.assertGreater(two, 0)
        self.assertLess(two, one, "the second session pays its own cold write")


class TestRelocationShortcutCannotSkipTheMechanismCheck(unittest.TestCase):
    """The observed-movement shortcut returned before _mechanism_for_all, so a
    cross-authority move could be classified low risk with no eval -- including
    on models the registry excludes. It is now within-container only."""

    def test_position_variance_from_an_optional_segment_is_not_movability(self):
        """Emission order is derived by sorting on index, so an ordinal only
        shifts when an earlier segment is absent. That is 'this block is
        optional', not 'this block moves', and it must not waive an eval."""
        reqs = []
        for i in range(6):
            # The optional block sits ABOVE the volatile one, so the volatile
            # segment's own ordinal shifts between requests -- which is what the
            # old shortcut read as "this segment moves".
            segs = [seg(1, "system", 90, f"vary{i}", "session_ctx"),
                    seg(2, "user", 14000, "hist", "history")]
            if i % 2 == 0:
                segs.insert(0, seg(0, "tools", 500, "tools", "tool_defs"))
            reqs.append(Request(request_id=f"r{i}", sent_at=T0 + timedelta(seconds=60 * i),
                                model="claude-sonnet-5", usage={}, segments=segs,
                                agent="a", target_id="anthropic/direct"))
        # Volatility now counts absence, so the optional block at position 0 is
        # itself the first blocker and nothing below it can be unlocked by
        # moving it -- position 1 varies too. No move is the right answer, and
        # the property under test still holds: an ordinal that shifts because
        # something above it vanished is not evidence the block moves.
        moves, order = relocate.propose(reqs, observed_volatility(reqs))
        self.assertEqual(moves, [])
        self.assertEqual(order, sorted(order), "nothing was reordered")

    def test_an_excluded_model_still_blocks_a_real_cross_authority_move(self):
        """The other half of what the fixture above used to cover, on a shape
        that does produce a move: a stable prefix, one volatile block, stable
        content behind it."""
        reqs = [req(i, [seg(0, "tools", 9000, "tools", "tool_defs"),
                        seg(1, "system", 90, f"vary{i}", "session_ctx"),
                        seg(2, "user", 14000, "hist", "history")],
                    model="claude-sonnet-5")
                for i in range(6)]
        moves, order = relocate.propose(reqs, observed_volatility(reqs))
        self.assertTrue(moves, "a volatile block above stable content is movable")
        self.assertEqual(moves[0].risk, "blocked",
                         "sonnet-5 is excluded from authority-preserving relocation")
        self.assertEqual(order, sorted(order), "a blocked move is never applied")

    def test_reordering_evidence_is_relative_not_ordinal(self):
        reqs = [req(i, [seg(0, "system", 90, f"v{i}"), seg(1, "user", 9000, "h")])
                for i in range(4)]
        self.assertEqual(relocate.observed_reordering(reqs), set(),
                         "index-derived order can never show relative movement")


class TestVolatilityIsPerCachePool(unittest.TestCase):
    """The simulator keys cache entries by (tenant, target, model). Deriving
    stability across the whole export marked per-tenant content volatile even
    though it never invalidated any real cache."""

    def test_per_tenant_content_is_stable_inside_its_own_pool(self):
        reqs = []
        for i in range(8):
            tenant = "acme" if i % 2 else "globex"
            reqs.append(Request(
                request_id=f"r{i}", sent_at=T0 + timedelta(seconds=60 * i),
                model="claude-opus-5", usage={},
                segments=[seg(0, "system", 7000, f"hdr-{tenant}", "tenant_header"),
                          seg(1, "user", 200, f"t{i}")],
                agent="a", tenant=tenant, target_id="anthropic/direct"))
        v = observed_volatility(reqs)
        self.assertEqual(v[0], 1, "one value per tenant pool, so stable")

    def test_genuinely_changing_content_is_still_volatile(self):
        reqs = []
        for i in range(8):
            reqs.append(Request(
                request_id=f"r{i}", sent_at=T0 + timedelta(seconds=60 * i),
                model="claude-opus-5", usage={},
                segments=[seg(0, "system", 7000, f"changes{i}", "ts")],
                agent="a", tenant="acme", target_id="anthropic/direct"))
        self.assertGreater(observed_volatility(reqs)[0], 1)


class TestDraftOverrideStillRespectsUnprovableWrites(unittest.TestCase):
    """allow_unreconciled covers a missing invoice and nothing else. Requests
    with unprovable write lifetimes are excluded from the totals, so releasing
    anyway publishes a number the notes call incomplete."""

    def _ts(self, ttl):
        r = Request(request_id="x", sent_at=T0, model="claude-opus-5",
                    usage={"input_tokens": 1000, "cache_creation_input_tokens": 50000},
                    segments=[], agent="a", ttl_requested=ttl, target_id="anthropic/direct")
        return TraceSet(requests=[r] * 4, tier=Tier.USAGE_ONLY)

    def test_the_override_does_not_release_when_writes_are_unprovable(self):
        a = analyze(self._ts(None), allow_unreconciled=True)
        self.assertFalse(a.spend["input_usd"].released)

    def test_the_override_works_when_every_write_is_provable(self):
        a = analyze(self._ts("1h"), allow_unreconciled=True)
        self.assertTrue(a.spend["input_usd"].released)

    def test_the_text_renderer_surfaces_coverage_notes(self):
        """HTML printed notes and text did not, so the same analysis disclosed
        different things depending on which file someone forwarded.

        Still true after the notes block was folded behind --detail: a note
        that caveats a *published figure* is not provenance and does not fold.
        It now prints beside the figures instead of four sections below them.
        Normalised on whitespace because the report wraps.
        """
        out = " ".join(render_text(
            analyze(self._ts(None), allow_unreconciled=True)).split())
        self.assertIn("without a provable 5m/1h lifetime", out)
        self.assertIn("excluded from every dollar figure", out)

    def test_a_spend_caveat_shows_without_detail_and_provenance_does_not(self):
        """The line the fold is drawn on. Provenance can wait for --detail; a
        sentence saying what a published number leaves out cannot."""
        from cacheeconomics.analyzer import spend_caveats
        a = analyze(self._ts(None), allow_unreconciled=True)
        brief = " ".join(render_text(a).split())
        full = " ".join(render_text(a, detail=True).split())
        caveats = spend_caveats(a.notes)
        self.assertTrue(caveats, "fixture has no spend caveat, so this is vacuous")
        for n in caveats:
            self.assertIn(" ".join(n.split())[:60], brief)
        others = [n for n in a.notes if n not in caveats]
        self.assertTrue(others, "fixture has no provenance note")
        for n in others:
            self.assertNotIn(" ".join(n.split())[:60], brief)
            self.assertIn(" ".join(n.split())[:60], full)

    def test_the_marker_is_one_string_the_analyzer_owns(self):
        """`spend_caveats` selects on a phrase. If a note is reworded without
        the constant, it silently demotes itself from money-caveat to
        provenance and stops printing beside the figure it qualifies."""
        import inspect

        from cacheeconomics import analyzer
        src = inspect.getsource(analyzer)
        literal = src.count('"' + analyzer.QUALIFIES_SPEND + '"')
        self.assertLessEqual(
            literal, 1,
            "a note spells the marker out instead of interpolating "
            "QUALIFIES_SPEND, so rewording it will not fail anything")


class TestPrefixEfficiencyPricesTheRealLifetime(unittest.TestCase):
    """An unread 5m write wastes 0.25x; an unread 1h write wastes 1.0x. The
    rule hard-coded 0.25x, underreporting an hour-lifetime workload fourfold."""

    def _figure(self, ttl):
        reqs = [Request(request_id=f"r{i}", sent_at=T0 + timedelta(seconds=60 * i),
                        model="claude-opus-5",
                        usage={"input_tokens": 100, "cache_creation_input_tokens": 1_000_000,
                               "cache_read_input_tokens": 0},
                        segments=[], agent="a", ttl_requested=ttl, target_id="anthropic/direct") for i in range(5)]
        a = analyze(TraceSet(requests=reqs, tier=Tier.USAGE_ONLY), allow_unreconciled=True)
        f = [x for x in a.findings if x.code == "EFF-1"]
        return f[0].avoidable_usd_month.raw() if f else 0.0

    def test_one_hour_waste_is_four_times_five_minute_waste(self):
        self.assertAlmostEqual(self._figure("1h") / self._figure("5m"), 4.0, places=6)

    def test_both_are_non_zero(self):
        self.assertGreater(self._figure("5m"), 0)
        self.assertGreater(self._figure("1h"), 0)


class TestWhitespaceIdsDoNotSurvive(unittest.TestCase):
    """Fixing the tier classification without fixing the construction left the
    bug. `_identified` rejected whitespace, but the parse loop used truthiness,
    so `""` fell through to hashing (correct) while `"   "` was kept as the
    identifier -- collapsing every such segment to one id that reads downstream
    as a perfectly stable prefix."""

    def _write(self, rows):
        f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.close()
        return f.name

    def _rows(self, sid, contents):
        return [{"request_id": f"r{i}", "sent_at": T0.isoformat(),
                 "model": "claude-opus-5", "usage": {"input_tokens": 10},
                 "segments": [{"index": 0, "role": "system", "tokens": 6000,
                               "id": sid, "content": c}]}
                for i, c in enumerate(contents)]

    def test_whitespace_id_is_replaced_by_a_keyed_hash(self):
        path = self._write(self._rows("   ", ["a", "b", "c"]))
        ts = load_jsonl(path, key=b"k")
        ids = [r.segments[0].id for r in ts.requests]
        self.assertTrue(all(i.startswith("hmac:") for i in ids))
        self.assertEqual(len(set(ids)), 3, "changing content must yield changing ids")
        os.unlink(path)

    def test_changing_content_behind_a_whitespace_id_reads_as_volatile(self):
        """The consequence that matters: it must not look like a stable prefix."""
        path = self._write(self._rows("   ", ["a", "b", "c"]))
        ts = load_jsonl(path, key=b"k")
        self.assertGreater(observed_volatility(ts.analysable)[0], 1)
        os.unlink(path)

    def test_a_real_id_is_kept(self):
        path = self._write(self._rows(hid("abc"), ["a", "b"]))
        ts = load_jsonl(path)
        self.assertEqual({r.segments[0].id for r in ts.requests}, {hid("abc")})
        self.assertIs(ts.tier, Tier.INSTRUMENTED)
        os.unlink(path)

    def test_a_padded_real_id_is_stripped_not_duplicated(self):
        path = self._write(self._rows("  " + hid("abc") + "  ", ["a", "b"]))
        ts = load_jsonl(path)
        self.assertEqual({r.segments[0].id for r in ts.requests}, {hid("abc")})
        os.unlink(path)


class TestVolatilityIsDecidedPerPoolNotGlobally(unittest.TestCase):
    """Taking the best pool was wrong: the plan is applied to every request
    regardless of pool, so a segment stable for tenant A and changing on every
    request for tenant B got a marker that rewrote a prefix nothing reads."""

    def _mixed(self):
        reqs = []
        for i in range(10):
            tenant = "steady" if i % 2 else "churny"
            sid = "fixed" if tenant == "steady" else f"changes{i}"
            reqs.append(Request(
                request_id=f"r{i}", sent_at=T0 + timedelta(seconds=60 * i),
                model="claude-opus-5", usage={},
                segments=[seg(0, "system", 7000, sid, "hdr"),
                          seg(1, "user", 200, f"t{i}")],
                agent="a", tenant=tenant, target_id="anthropic/direct"))
        return reqs

    def test_the_global_reduction_fails_closed(self):
        """One answer covering all traffic must take the worst pool."""
        self.assertGreater(observed_volatility(self._mixed())[0], 1)

    def test_per_pool_keeps_the_two_apart(self):
        from cacheeconomics.allocate import observed_volatility_by_pool
        by_pool = observed_volatility_by_pool(self._mixed())
        steady = by_pool[(("steady"), "anthropic/direct", "claude-opus-5")]
        churny = by_pool[(("churny"), "anthropic/direct", "claude-opus-5")]
        self.assertEqual(steady[0], 1)
        self.assertGreater(churny[0], 1)

    def test_the_simulator_does_not_mark_the_churning_tenant(self):
        """The steady tenant should still cache; the churning one should not be
        made to pay a write premium for a prefix it will never read."""
        res = simulate.simulate(self._mixed(), "allocator-lite")
        self.assertGreater(res.reads, 0, "the steady tenant still benefits")
        self.assertLessEqual(res.writes, 2,
                             "the churning tenant must not rewrite every request")


class TestTTLSavingsExcludeSubFiveMinuteRewrites(unittest.TestCase):
    """A rewrite 60 seconds after the last one happened while a five-minute
    entry should still have been alive. That is prefix drift or fan-out, not
    TTL expiry, and a longer lifetime would not have prevented it."""

    def _agent(self, gap):
        return [Request(request_id=f"r{i}", sent_at=T0 + timedelta(seconds=gap * i),
                        model="claude-opus-5",
                        usage={"input_tokens": 100,
                               "cache_creation_input_tokens": 1_000_000,
                               "cache_read_input_tokens": 5000},
                        segments=[seg(0, "system", 7000, "stable-prefix", "sys", marked=True, ttl="5m"),
                              seg(1, "user", 100, f"turn{i}")],
                        agent="a", session="s", ttl_requested="5m", target_id="anthropic/direct")
                for i in range(12)]

    def _finding(self, reqs):
        a = analyze(TraceSet(requests=reqs, tier=Tier.USAGE_ONLY), allow_unreconciled=True)
        f = [x for x in a.findings if x.code == "TTL-1"]
        return f[0] if f else None

    def test_rewrites_inside_five_minutes_are_not_ttl_recoverable(self):
        """Cadence still puts some gaps in band via the median, but the dollar
        loop must not count a 60-second rewrite as a one-hour win."""
        mixed = self._agent(900)[:6]
        # bolt on a burst of sub-5-minute rewrites in the same session
        t = mixed[-1].sent_at
        for i in range(6):
            t = t + timedelta(seconds=60)
            mixed.append(Request(request_id=f"b{i}", sent_at=t, model="claude-opus-5",
                                 usage={"input_tokens": 100,
                                        "cache_creation_input_tokens": 1_000_000,
                                        "cache_read_input_tokens": 5000},
                                 segments=[seg(0, "system", 7000, "stable-prefix",
                                               "sys", marked=True, ttl="5m"),
                                           seg(1, "user", 100, f"burst{i}")],
                                 agent="a", session="s", ttl_requested="5m", target_id="anthropic/direct"))
        f = self._finding(mixed)
        if f is not None:
            self.assertIn("within five minutes", f.detail)

    def test_an_all_burst_workload_produces_no_ttl_finding(self):
        """Every gap under five minutes: nothing here a longer lifetime fixes."""
        self.assertIsNone(self._finding(self._agent(60)))

    def test_a_genuinely_in_band_workload_still_fires(self):
        f = self._finding(self._agent(900))
        self.assertIsNotNone(f)
        self.assertNotIn("within five minutes", f.detail)


class TestTTLMoneyRequiresProvenPrefixReuse(unittest.TestCase):
    """Two writes fifteen minutes apart in one session prove a one-hour TTL
    would have helped only if they wrote the same prefix. Usage fields cannot
    distinguish that from a drifted prefix, so the observation may be reported
    but the dollar figure may not."""

    def _reqs(self, segments_for):
        out = []
        for i in range(10):
            out.append(Request(
                request_id=f"r{i}", sent_at=T0 + timedelta(seconds=900 * i),
                model="claude-opus-5",
                usage={"input_tokens": 100, "cache_creation_input_tokens": 500_000,
                       "cache_read_input_tokens": 5000},
                segments=segments_for(i), agent="a", session="s",
                target_id="anthropic/direct", ttl_requested="5m"))
        return out

    def _finding(self, reqs, tier=Tier.USAGE_ONLY):
        a = analyze(TraceSet(requests=reqs, tier=tier), allow_unreconciled=True)
        return next((f for f in a.findings if f.code == "TTL-1"), None)

    def test_usage_only_reports_the_hypothesis_without_a_figure(self):
        f = self._finding(self._reqs(lambda i: []))
        self.assertIsNotNone(f, "the cadence observation still stands")
        self.assertIsNone(f.avoidable_usd_month, "but it may not be priced")
        self.assertIn("does not carry segment identity", f.detail)
        self.assertEqual(f.severity, "medium")

    def _stable(self):
        return self._reqs(
            lambda i: [seg(0, "system", 7000, "same", "sys", marked=True, ttl="5m"),
                       seg(1, "user", 100, f"t{i}")])

    def test_a_stable_marked_prefix_earns_the_figure(self):
        """Instrumented, because the figure is now costed from those segments.

        This asked for a priced TTL-1 off a `USAGE_ONLY` trace that carried
        segments anyway -- a shape a loader does not produce, and the one
        combination in which TTL-1's figure was computed from segment spans
        while the gate that judges segment quality never looked at it. The claim
        the test makes is about proven prefix reuse, not about escaping that
        gate, so the trace now says what it is.
        """
        f = self._finding(self._stable(), tier=Tier.INSTRUMENTED)
        self.assertIsNotNone(f.avoidable_usd_month)
        self.assertGreater(f.avoidable_usd_month.raw(), 0)
        self.assertEqual(f.severity, "high")

    def test_the_priced_form_is_gated_by_the_structural_trust_gate(self):
        """The same trace, unaligned, must not publish the same figure.

        TTL-1 decides which spans matched, and now what share of a write each
        span accounts for, from `marked_spans` -- so its figure is costed from
        segment boundaries and sizes exactly like VOL-1's. It carried no
        `structural` flag, so on a trace whose segmentation nothing had scored
        it published dollars while VOL-1 beside it withheld them for that reason.
        """
        f = self._finding(self._stable(), tier=Tier.USAGE_ONLY)
        self.assertTrue(f.structural, "the priced form is derived from spans")
        self.assertFalse(f.avoidable_usd_month.released)
        self.assertFalse(f.avoidable_usd_window.released)
        self.assertEqual(f.confidence, "low")

    def test_the_unpriced_form_is_not_marked_structural(self):
        """The other arm has no segments to be judged on, and says so. Marking
        it structural would print the report's "these rest on segment boundaries
        inferred from logged bodies" note above a finding whose own text says
        the trace carries no segment identity."""
        f = self._finding(self._reqs(lambda i: []))
        self.assertFalse(f.structural)

    def test_a_drifting_prefix_earns_nothing(self):
        """Each request writes a different span, so no later write is a rewrite
        of an earlier one and a longer lifetime recovers nothing."""
        f = self._finding(self._reqs(
            lambda i: [seg(0, "system", 7000, f"drifts{i}", "sys", marked=True, ttl="5m"),
                       seg(1, "user", 100, f"t{i}")]))
        if f is not None:
            self.assertIsNone(f.avoidable_usd_month)


class TestTTLRecoveryIsProportionalToTheSpanThatMatched(unittest.TestCase):
    """A request with nested markers writes one billed total, and TTL-1 credited
    all of it to whichever span matched.

    `earlier.append((sent_at, span, tokens))` stored the whole request's
    `cache_write_5m` against every one of its spans, so when the lookback window
    rejected the outer marker and an inner one matched instead, `min(match[1],
    tokens)` handed the inner span the outer span's write. That is not a corner
    case: it is what a tool-calling agent looks like, because appending turns
    pushes the previous outermost marker out of the provider's lookback while
    leaving the stable inner prefix reachable.

    SYNTHETIC fixture. Three traces identical in every respect -- same billed
    writes, same block counts, same cadence, same segment total -- differing
    only in how much of the cached prefix sits inside the inner marker. The
    defect makes all three produce the same figure, because the credit never
    looked at the span. The fix makes them ordered.

    Written as a comparison rather than as an expected dollar amount on purpose:
    an assertion computed from the multipliers is a restatement of the code
    under test, and would go on passing if the same mistake were made twice.
    """

    N = 10
    STRIDE = 22          # blocks appended per turn, past the recorded lookback of 20
    PREFIX = 100_000     # tokens in the cached prefix, and the billed write

    def _reqs(self, inner_share, segment_scale=1):
        """`inner_share` of the prefix sits inside the inner marker.

        Turn blocks carry no tokens, so the outermost span is exactly the
        segment total in every request and the share the fix computes is exactly
        `inner_share`. Block *counts* are identical across the three traces, so
        the lookback rejects the outer marker in all of them alike.

        `segment_scale` multiplies the segment sizes and leaves the billed write
        alone, which is how the two get told apart.
        """
        sized = self.PREFIX * segment_scale
        inner = int(sized * inner_share)
        out = []
        for i in range(self.N):
            segs = [seg(0, "system", inner, "inner", "sys", marked=True, ttl="5m"),
                    seg(1, "system", sized - inner, "mid", "sys"),
                    # The outer marker exists only on the first request; later
                    # ones mark the end of their appended history instead, which
                    # is what makes the outer span unreachable by lookback.
                    seg(2, "user", 0, "tail", "hist",
                        marked=(i == 0), ttl="5m" if i == 0 else None)]
            idx = 3
            for turn in range(i):
                for k in range(self.STRIDE):
                    last = (turn == i - 1 and k == self.STRIDE - 1)
                    segs.append(seg(idx, "user", 0, f"t{turn}_{k}", "hist",
                                    marked=last, ttl="5m" if last else None))
                    idx += 1
            out.append(Request(
                request_id=f"r{i}", sent_at=T0 + timedelta(seconds=600 * i),
                model="claude-opus-5", target_id="anthropic/direct",
                tenant="t", session="s", agent="a", ttl_requested="5m",
                usage={"input_tokens": 0, "cache_read_input_tokens": 0,
                       "cache_creation_input_tokens": self.PREFIX},
                segments=segs))
        return out

    def _recovered(self, inner_share, segment_scale=1):
        a = analyze(TraceSet(requests=self._reqs(inner_share, segment_scale),
                             tier=Tier.INSTRUMENTED), allow_unreconciled=True)
        f = next((x for x in a.findings if x.code == "TTL-1"), None)
        self.assertIsNotNone(f, f"TTL-1 did not fire at share {inner_share}")
        self.assertIsNotNone(f.avoidable_usd_window,
                             f"TTL-1 published no figure at share {inner_share}")
        return f.avoidable_usd_window.raw()

    def test_the_fixture_makes_an_inner_span_the_only_match(self):
        """Guard the guard. If the outer marker were still reachable, the outer
        span would match, the share would be 1.0, and every assertion below
        would pass while testing nothing at all."""
        from cacheeconomics import registry
        from cacheeconomics.trace import marked_spans, span_is_reusable_by
        reqs = self._reqs(0.5)
        lookback = registry.capability("anthropic/direct", "lookback_blocks")
        prev, cur = marked_spans(reqs[0].segments), marked_spans(reqs[1].segments)
        inner, outer = prev[0], prev[-1]
        self.assertNotEqual(inner, outer, "fixture has only one marker")
        self.assertFalse(span_is_reusable_by(outer, cur, lookback, 512),
                         "the outer span is still reachable, so nothing forces "
                         "the inner match this class is about")
        self.assertTrue(span_is_reusable_by(inner, cur, lookback, 512))

    def test_a_smaller_matched_span_recovers_strictly_less(self):
        """The whole finding. Under the defect these three are equal, because
        the credit was the writing request's total however little of it the
        matched span covered."""
        quarter, half, whole = (self._recovered(0.25), self._recovered(0.5),
                                self._recovered(1.0))
        self.assertLess(quarter, half,
                        f"a quarter-sized match recovered as much as a "
                        f"half-sized one ({quarter} vs {half}): the credit is "
                        f"not reading the span that matched")
        self.assertLess(half, whole,
                        f"a half-sized match recovered as much as a whole-prefix "
                        f"one ({half} vs {whole})")

    def test_the_share_is_a_ratio_rather_than_the_estimate_itself(self):
        """The control, and the reason the fix is a ratio and not a substitution.

        Bounding recovery by the raw segment estimate is the obvious repair and
        it is wrong: the estimate is a split of the billed total, not a second
        opinion on it, and using it cut a fixture writing 200,000 tokens down to
        its 7,000-token segment sum and silenced the rule. A ratio of two
        estimates cancels, so multiplying every segment by ten -- while the
        provider still reports the same billed write -- must not move the figure
        at all, at any nesting.
        """
        for share in (0.25, 0.5, 1.0):
            with self.subTest(inner_share=share):
                self.assertAlmostEqual(
                    self._recovered(share), self._recovered(share, segment_scale=10),
                    places=9,
                    msg="the figure tracks the segment estimate rather than the "
                        "billed write it is a split of")


class TestTTLRecoveryReadsTheEntrySizeNotTheWriteSize(unittest.TestCase):
    """A read refreshes the entry it read, so a small write can leave a large
    live entry.

    `earlier` stored the size of the *write* that touched a span. A request that
    reads a 100k prefix back and writes a 1k appended suffix leaves a live 101k
    entry while writing 1,000 tokens, so the later TTL miss on that entry was
    credited with recovering 1,000 tokens instead of 101,000. The cold-write
    premium is charged against the real write, so the netting came out negative
    and TTL-1 went silent altogether -- the recommendation, which is correct and
    worth roughly $3 over this window, never reached the client at all.

    SYNTHETIC. Refresh-then-miss cycles: a refresh two minutes after the last
    write (inside the five-minute lifetime, so the entry is alive and gets
    extended), then a miss ten minutes later (outside it, so a five-minute entry
    is gone and a one-hour one would not have been).

    Not a regression from the proportional-share fix in the class above:
    measured with that share reverted, this fixture was equally silent. It is
    invisible in the shipped fixtures because none of their cache-writing rows
    carries a read -- 0 of 136 in demo-traces.jsonl -- which is both why it
    survived and why closing it moves no published figure.
    """

    PREFIX = 100_000
    SUFFIX = 1_000
    CYCLES = 6

    def _req(self, rid, when, read, write, marks):
        segs = [seg(0, "system", self.PREFIX, "sys", "sys", marked=True, ttl="5m")]
        if marks > 1:
            segs.append(seg(1, "user", self.SUFFIX, "suffix", "hist",
                            marked=True, ttl="5m"))
        return Request(request_id=rid, sent_at=when, model="claude-opus-5",
                       target_id="anthropic/direct", tenant="t", session="s",
                       agent="a", ttl_requested="5m",
                       usage={"input_tokens": 0, "cache_read_input_tokens": read,
                              "cache_creation_input_tokens": write},
                       segments=segs)

    def _reqs(self, refresh_read):
        """`refresh_read` is how much the refreshing request reads back.

        Zero turns each refresh into an ordinary small cold write, which is the
        control: with no read the entry really is only as big as the write, and
        the figure must not move.
        """
        out, t = [self._req("r0", T0, 0, self.PREFIX, 1)], T0
        for c in range(self.CYCLES):
            t += timedelta(seconds=120)
            out.append(self._req(f"refresh{c}", t, refresh_read, self.SUFFIX, 2))
            t += timedelta(seconds=600)
            out.append(self._req(f"miss{c}", t, 0, self.PREFIX + self.SUFFIX, 2))
        return out

    def _ttl(self, refresh_read):
        a = analyze(TraceSet(requests=self._reqs(refresh_read),
                             tier=Tier.INSTRUMENTED), allow_unreconciled=True)
        return next((f for f in a.findings if f.code == "TTL-1"), None)

    def test_a_refreshed_entry_is_recovered_at_its_own_size(self):
        f = self._ttl(self.PREFIX)
        self.assertIsNotNone(
            f, "TTL-1 stayed silent on a workload a one-hour lifetime would "
               "genuinely help: the recommendation never reaches the client")
        self.assertIsNotNone(f.avoidable_usd_window, "fired without a figure")
        self.assertGreater(f.avoidable_usd_window.raw(), 0)

    def test_the_recovery_scales_with_what_was_refreshed(self):
        """The claim, stated as a comparison rather than as an expected dollar
        amount. Reading a hundred times more back must recover more, and under
        the defect both read the write size and come out identical."""
        small = self._ttl(self.SUFFIX)
        large = self._ttl(self.PREFIX)
        self.assertIsNotNone(large.avoidable_usd_window)
        small_usd = (small.avoidable_usd_window.raw()
                     if small is not None and small.avoidable_usd_window else 0.0)
        self.assertLess(
            small_usd, large.avoidable_usd_window.raw(),
            "the credit is reading the write size, not the entry size")

    def test_a_write_with_no_read_is_unchanged(self):
        """The control, and the property that keeps this from moving published
        figures: with no read the entry is exactly the write, so every trace in
        which nothing reads back -- which is every cache-writing row in the
        shipped fixtures -- prices identically."""
        self.assertIsNone(
            self._ttl(0),
            "a trace whose refreshes read nothing back has no reuse for a "
            "longer lifetime to recover, and TTL-1 should stay silent on it")

    def test_recovery_never_exceeds_what_the_later_request_wrote(self):
        """The bound that keeps entry size from becoming an over-credit. A
        longer lifetime can only save what was actually paid to write again,
        however large the entry behind it was."""
        from cacheeconomics import registry
        f = self._ttl(self.PREFIX)
        m = registry.multipliers("anthropic/direct")
        rate = registry.base_rate("claude-opus-5", "2026-01-01", "anthropic/direct")
        ceiling = (self.CYCLES * (self.PREFIX + self.SUFFIX)
                   * (rate / 1e6) * (m["write_5m"] - m["read"]))
        self.assertLessEqual(f.avoidable_usd_window.raw(), ceiling)


class TestVolatileFindingRespectsCachePools(unittest.TestCase):
    """VOL-1 bucketed segment ids across the whole export while the simulator
    isolates by (tenant, target, model). A per-tenant header stable in every
    real pool was reported volatile, with a relocation recommendation and
    dollars attached."""

    def _reqs(self, header_for):
        out = []
        for i in range(10):
            tenant = "acme" if i % 2 else "globex"
            out.append(Request(
                request_id=f"r{i}", sent_at=T0 + timedelta(seconds=60 * i),
                model="claude-opus-5",
                usage={"input_tokens": 100, "cache_creation_input_tokens": 50_000,
                       "cache_read_input_tokens": 5000},
                segments=[seg(0, "system", 200, header_for(tenant, i), "tenant_hdr"),
                          seg(1, "system", 14000, "body", "sys", marked=True, ttl="5m")],
                agent="a", tenant=tenant, target_id="anthropic/direct",
                ttl_requested="5m"))
        return out

    def _vol(self, reqs):
        a = analyze(TraceSet(requests=reqs, tier=Tier.INSTRUMENTED), allow_unreconciled=True)
        return next((f for f in a.findings if f.code == "VOL-1"), None)

    def test_a_per_tenant_header_is_not_reported_volatile(self):
        f = self._vol(self._reqs(lambda tenant, i: f"hdr-{tenant}"))
        self.assertIsNone(f, "stable inside every pool that exists")

    def test_a_header_that_changes_within_a_pool_is_still_caught(self):
        f = self._vol(self._reqs(lambda tenant, i: f"hdr-{tenant}-{i}"))
        self.assertIsNotNone(f)


class TestUnknownPricingDoesNotAbortTheReport(unittest.TestCase):
    """Pricing rows are date-effective and model ids drift. A new model should
    cost that model's requests, not the whole analysis."""

    def _mixed(self, unknown="claude-not-yet-released-9"):
        out = []
        for i in range(8):
            out.append(Request(
                request_id=f"r{i}", sent_at=T0 + timedelta(seconds=60 * i),
                model=unknown if i % 4 == 0 else "claude-opus-5",
                usage={"input_tokens": 5000, "cache_creation_input_tokens": 100_000,
                       "cache_read_input_tokens": 1000},
                segments=[], agent="a", ttl_requested="5m", target_id="anthropic/direct"))
        return out

    def test_the_analysis_completes(self):
        a = analyze(TraceSet(requests=self._mixed(), tier=Tier.USAGE_ONLY))
        self.assertIsNotNone(a)

    def test_the_unpriceable_model_is_named_in_the_notes(self):
        a = analyze(TraceSet(requests=self._mixed(), tier=Tier.USAGE_ONLY))
        self.assertTrue(any("claude-not-yet-released-9" in n for n in a.notes))

    def test_it_fails_the_publication_gate(self):
        a = analyze(TraceSet(requests=self._mixed(), tier=Tier.USAGE_ONLY),
                    allow_unreconciled=True)
        self.assertFalse(a.spend["input_usd"].released,
                         "a known-incomplete total must not publish")

    def test_a_fully_priceable_trace_is_unaffected(self):
        reqs = [r for r in self._mixed() if r.model == "claude-opus-5"]
        a = analyze(TraceSet(requests=reqs, tier=Tier.USAGE_ONLY), allow_unreconciled=True)
        self.assertTrue(a.spend["input_usd"].released)
        self.assertFalse(any("no pricing recorded" in n.lower() for n in a.notes))


class TestFanOutRespectsPrefixAndScope(unittest.TestCase):
    """FAN-1 keyed on the marked segment ids alone. That is not the cached
    prefix -- unmarked content before the marker is part of it too -- and it
    ignored tenant, surface and model, so requests that could never share a
    cache entry were priced as duplicate writes of one another."""

    def _burst(self, header_for, tenant_for):
        out = []
        for i in range(8):
            out.append(Request(
                request_id=f"r{i}", sent_at=T0 + timedelta(seconds=i),
                model="claude-opus-5",
                usage={"input_tokens": 100, "cache_creation_input_tokens": 40_000},
                segments=[seg(0, "system", 300, header_for(i), "hdr"),
                          seg(1, "system", 20000, "body", "sys", marked=True, ttl="5m")],
                agent="a", tenant=tenant_for(i), target_id="anthropic/direct",
                ttl_requested="5m"))
        return out

    def _fan(self, reqs):
        a = analyze(TraceSet(requests=reqs, tier=Tier.INSTRUMENTED), allow_unreconciled=True)
        return next((f for f in a.findings if f.code == "FAN-1"), None)

    def test_genuinely_identical_concurrent_prefixes_are_caught(self):
        self.assertIsNotNone(self._fan(self._burst(lambda i: "same", lambda i: "acme")))

    def test_different_leading_content_is_not_the_same_prefix(self):
        """Unmarked content above the marker still forms part of the prefix."""
        self.assertIsNone(self._fan(self._burst(lambda i: f"hdr{i}", lambda i: "acme")))

    def test_different_tenants_cannot_waste_each_others_writes(self):
        self.assertIsNone(self._fan(
            self._burst(lambda i: "same", lambda i: f"tenant{i}")))


class TestReconciliationVerdictMatchesWhatWasPublished(unittest.TestCase):
    """render_html shows within_ship_gate as the visible verdict, so a partial
    flag let a report read 'inside the gate, so figures follow' directly above
    a column of withheld figures."""

    def _ts(self):
        reqs = [Request(request_id=f"r{i}", sent_at=T0 + timedelta(seconds=60 * i),
                        model="claude-not-a-real-model" if i == 0 else "claude-opus-5",
                        usage={"input_tokens": 5000,
                               "cache_creation_input_tokens": 100_000},
                        segments=[], agent="a", ttl_requested="5m", target_id="anthropic/direct")
                for i in range(6)]
        return TraceSet(requests=reqs, tier=Tier.USAGE_ONLY)

    def test_an_unpriceable_model_fails_the_stated_gate(self):
        a = analyze(self._ts(), invoice_usd=1.0)
        self.assertFalse(a.reconciliation["within_ship_gate"])

    def test_the_excluded_requests_are_counted_in_unpriced(self):
        a = analyze(self._ts(), invoice_usd=1.0)
        self.assertGreaterEqual(a.reconciliation["unpriced_requests"], 1)

    def test_the_verdict_never_disagrees_with_the_figures(self):
        a = analyze(self._ts(), invoice_usd=1.0)
        self.assertEqual(a.reconciliation["within_ship_gate"],
                         a.spend["input_usd"].released)


class TestSimulatorSplitsWritesByBreakpointLifetime(unittest.TestCase):
    """Collapsing every written token into the outermost marker's TTL is the
    same 5m/1h pricing error the analyzer refuses to make, one layer down."""

    def _req(self, i, inner_ttl, outer_ttl):
        return Request(
            request_id=f"r{i}", sent_at=T0 + timedelta(seconds=30 * i),
            model="claude-opus-5", usage={},
            segments=[seg(0, "system", 8000, "a", "inner", marked=True, ttl=inner_ttl),
                      seg(1, "system", 8000, "b", "outer", marked=True, ttl=outer_ttl),
                      seg(2, "user", 100, f"t{i}")],
            agent="a", target_id="anthropic/direct")

    def test_each_breakpoint_is_written_at_its_own_lifetime(self):
        res = simulate.simulate([self._req(0, "1h", "5m")], "as-shipped")
        u = res.usages[0]
        self.assertEqual(u.cache_write_1h, 8000, "inner span keeps its 1h lifetime")
        self.assertEqual(u.cache_write_5m, 8000, "outer span keeps its 5m lifetime")

    def test_the_reverse_ordering_is_also_split(self):
        res = simulate.simulate([self._req(0, "5m", "1h")], "as-shipped")
        u = res.usages[0]
        self.assertEqual(u.cache_write_5m, 8000)
        self.assertEqual(u.cache_write_1h, 8000)

    def test_a_uniform_request_is_unchanged(self):
        res = simulate.simulate([self._req(0, "5m", "5m")], "as-shipped")
        self.assertEqual(res.usages[0].cache_write_5m, 16000)
        self.assertEqual(res.usages[0].cache_write_1h, 0)

    def test_mixed_lifetimes_cost_more_than_all_five_minute(self):
        """The whole point: collapsing them underpriced the 1h half."""
        mixed = simulate.simulate([self._req(0, "1h", "5m")], "as-shipped")
        flat = simulate.simulate([self._req(0, "5m", "5m")], "as-shipped")
        self.assertGreater(mixed.spend("2026-07-29").raw(),
                           flat.spend("2026-07-29").raw())


class TestPricingDateIsNotPinned(unittest.TestCase):
    """bake_off defaulted on_date to a literal '2026-07-29'. The registry is
    date-effective and already carries a claude-sonnet-5 change on 2026-09-01,
    so a pinned default silently underprices every run made after it."""

    def _reqs(self, when):
        segs = [seg(0, "system", 6000, "sys", marked=True, ttl="5m"),
                seg(1, "user", 100, "t")]
        return [Request(request_id=f"r{i}", sent_at=when + timedelta(seconds=60 * i),
                        model="claude-sonnet-5", usage={}, segments=segs, agent="a",
                        target_id="anthropic/direct") for i in range(4)]

    def test_a_trace_is_priced_on_the_day_it_was_sent(self):
        before = simulate.simulate(self._reqs(datetime(2026, 8, 1, tzinfo=timezone.utc)),
                                   "as-shipped").spend().raw()
        after = simulate.simulate(self._reqs(datetime(2026, 10, 1, tzinfo=timezone.utc)),
                                  "as-shipped").spend().raw()
        self.assertGreater(after, before,
                           "sonnet-5 goes from $2.00 to $3.00 on 2026-09-01")

    def test_an_explicit_date_still_overrides(self):
        reqs = self._reqs(datetime(2026, 10, 1, tzinfo=timezone.utc))
        pinned = simulate.simulate(reqs, "as-shipped").spend("2026-08-01").raw()
        derived = simulate.simulate(reqs, "as-shipped").spend().raw()
        self.assertLess(pinned, derived)

    def test_bake_off_no_longer_carries_a_literal_default(self):
        import inspect
        for fn in (simulate.bake_off, simulate.bake_off_by_agent):
            default = inspect.signature(fn).parameters["on_date"].default
            self.assertIsNone(default, f"{fn.__name__} must not pin a date")


class TestSameFixAppliedEverywhere(unittest.TestCase):
    """Round 15 was four instances of one habit: fixing a pattern in one
    function and leaving its twin. These assert the twins."""

    def _sonnet(self, when):
        return [Request(request_id=f"r{i}", sent_at=when + timedelta(seconds=60 * i),
                        model="claude-sonnet-5",
                        usage={"input_tokens": 1_000_000,
                               "cache_creation_input_tokens": 0},
                        segments=[], agent="a", ttl_requested="5m", target_id="anthropic/direct") for i in range(4)]

    def test_the_analyzer_prices_on_the_day_the_request_was_sent(self):
        """spend() was fixed for this and analyze() was not."""
        before = analyze(TraceSet(requests=self._sonnet(datetime(2026, 8, 1, tzinfo=timezone.utc)),
                                  tier=Tier.USAGE_ONLY), allow_unreconciled=True)
        after = analyze(TraceSet(requests=self._sonnet(datetime(2026, 10, 1, tzinfo=timezone.utc)),
                                 tier=Tier.USAGE_ONLY), allow_unreconciled=True)
        self.assertGreater(after.spend["input_usd"].raw(), before.spend["input_usd"].raw(),
                           "sonnet-5 goes $2.00 -> $3.00 on 2026-09-01")

    def test_a_reprice_date_still_overrides(self):
        a = analyze(TraceSet(requests=self._sonnet(datetime(2026, 10, 1, tzinfo=timezone.utc)),
                             tier=Tier.USAGE_ONLY), on_date="2026-08-01",
                    allow_unreconciled=True)
        b = analyze(TraceSet(requests=self._sonnet(datetime(2026, 10, 1, tzinfo=timezone.utc)),
                             tier=Tier.USAGE_ONLY), allow_unreconciled=True)
        self.assertLess(a.spend["input_usd"].raw(), b.spend["input_usd"].raw())

    def test_volatility_charged_only_to_the_pools_that_saw_it(self):
        """Detected per pool, then charged to everyone, was half a fix."""
        reqs = []
        for i in range(20):
            churny = i % 2 == 0
            reqs.append(Request(
                request_id=f"r{i}", sent_at=T0 + timedelta(seconds=60 * i),
                model="claude-opus-5",
                usage={"input_tokens": 100, "cache_creation_input_tokens": 40_000},
                segments=[seg(0, "system", 200, f"hdr{i}" if churny else "hdr", "hdr"),
                          seg(1, "system", 20000, "body", "sys", marked=True, ttl="5m")],
                agent="a", tenant="churny" if churny else "steady",
                target_id="anthropic/direct", ttl_requested="5m"))
        a = analyze(TraceSet(requests=reqs, tier=Tier.INSTRUMENTED), allow_unreconciled=True)
        vol = next(f for f in a.findings if f.code == "VOL-1")
        self.assertEqual(vol.affected_requests, 10, "only the churning tenant")
        self.assertIn("need no change", vol.detail)

    def test_cadence_is_derived_per_pool_not_globally(self):
        """A slow pool must not inherit a fast pool's one-hour TTL and pay the
        2x premium with no chance of a read."""
        segs = lambda i: [seg(0, "system", 8000, "sys"), seg(1, "user", 50, f"t{i}")]
        reqs = []
        for i in range(12):        # fast pool: 30s apart
            reqs.append(Request(request_id=f"f{i}", sent_at=T0 + timedelta(seconds=30 * i),
                                model="claude-opus-5", usage={}, segments=segs(i),
                                agent="a", tenant="fast", target_id="anthropic/direct"))
        for i in range(12):        # slow pool: 3 hours apart, far outside the band
            reqs.append(Request(request_id=f"s{i}", sent_at=T0 + timedelta(seconds=10800 * i),
                                model="claude-opus-5", usage={}, segments=segs(i),
                                agent="a", tenant="slow", target_id="anthropic/direct"))
        res = simulate.simulate(reqs, "allocator-lite")
        slow = [u for u, r in zip(res.usages, res.priced) if r.tenant == "slow"]
        self.assertEqual(sum(u.cache_write_1h for u in slow), 0,
                         "the slow pool must not be put on a one-hour lifetime")


class TestFindingDollarsUseTraceDatePricing(unittest.TestCase):
    """Spend priced per request date while the finding rules used the run date,
    so one report could reconcile at $2/Mtok and publish avoidable dollars at
    $3 -- the same number computed two ways."""

    def _trace(self, when):
        reqs = [Request(request_id=f"r{i}", sent_at=when + timedelta(seconds=60 * i),
                        model="claude-sonnet-5",
                        usage={"input_tokens": 1000,
                               "cache_creation_input_tokens": 400_000,
                               "cache_read_input_tokens": 0},
                        segments=[], agent="a", ttl_requested="5m", target_id="anthropic/direct") for i in range(6)]
        return TraceSet(requests=reqs, tier=Tier.USAGE_ONLY)

    def _eff(self, when):
        a = analyze(self._trace(when), allow_unreconciled=True)
        f = next((x for x in a.findings if x.code == "EFF-1"), None)
        return f.avoidable_usd_month.raw() if f and f.avoidable_usd_month else 0.0

    def test_findings_move_with_the_trace_date_not_the_run_date(self):
        before = self._eff(datetime(2026, 8, 1, tzinfo=timezone.utc))
        after = self._eff(datetime(2026, 10, 1, tzinfo=timezone.utc))
        self.assertGreater(before, 0)
        self.assertAlmostEqual(after / before, 1.5, places=3,
                               msg="sonnet-5 goes $2.00 -> $3.00 on 2026-09-01")

    def test_spend_and_findings_agree_on_the_rate(self):
        when = datetime(2026, 8, 1, tzinfo=timezone.utc)
        a = analyze(self._trace(when), allow_unreconciled=True)
        # Both sides priced at the August rate: the ratio of the EFF-1 premium
        # to total write spend is fixed by the multipliers, not by the date.
        self.assertGreater(a.spend["input_usd"].raw(), 0)
        eff = next(x for x in a.findings if x.code == "EFF-1")
        self.assertGreater(eff.avoidable_usd_month.raw(), 0)


class TestOnlyKeyedIdsCountAsInstrumented(unittest.TestCase):
    """A bare digest is what this tool refuses to generate without a key,
    because short segments are dictionary-guessable and stable bare digests
    reveal cross-tenant equality. Accepting one from an exporter and calling it
    instrumented claimed a guarantee that was never met."""

    def _write(self, sid, n=3):
        f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        for i in range(n):
            row = {"request_id": f"r{i}", "sent_at": T0.isoformat(),
                   "model": "claude-opus-5", "usage": {"input_tokens": 10},
                   "segments": [{"index": 0, "role": "system", "tokens": 10,
                                 "content": f"c{i}"}]}
            if sid is not None:
                row["segments"][0]["id"] = sid.format(i=i)
            f.write(json.dumps(row) + "\n")
        f.close()
        return f.name

    def test_a_keyed_id_is_instrumented(self):
        p = self._write(hid("abc") + "{i}"[:0] or hid("abc"))
        self.assertIs(load_jsonl(p).tier, Tier.INSTRUMENTED)
        os.unlink(p)

    def test_a_bare_digest_is_not_instrumented(self):
        p = self._write("sha256:" + "a" * 64)
        ts = load_jsonl(p, key=b"k")
        self.assertIsNot(ts.tier, Tier.INSTRUMENTED)
        self.assertTrue(any("does not trust" in n for n in ts.notes))
        os.unlink(p)

    def test_an_arbitrary_exporter_id_is_not_instrumented(self):
        p = self._write("seg-{i}")
        self.assertIsNot(load_jsonl(p, key=b"k").tier, Tier.INSTRUMENTED)
        os.unlink(p)

    def test_an_untrusted_id_still_requires_a_key_to_re_derive(self):
        """It falls through to synthesis, which fails closed without a key."""
        p = self._write("sha256:" + "a" * 64)
        with self.assertRaises(ValueError):
            load_jsonl(p)
        os.unlink(p)


class TestRelocationRespectsTheWireContainer(unittest.TestCase):
    """`tools` and `system` share an authority class but are separate top-level
    fields, and no ordering of a request interleaves them. Treating the class as
    one movable block priced a saving for a prompt nobody can send."""

    def _reqs(self):
        return [Request(
            request_id=f"r{i}", sent_at=T0 + timedelta(seconds=60 * i),
            model="claude-opus-5", usage={},
            segments=[seg(0, "tools", 300, f"vary{i}", "tool:search"),
                      seg(1, "tools", 4000, "stable_tool", "tool:read"),
                      seg(2, "system", 14000, "sys", "instructions"),
                      seg(3, "user", 100, f"t{i}")],
            agent="a", target_id="anthropic/direct") for i in range(6)]

    def test_a_tool_is_not_reordered_past_system_content(self):
        moves, order = relocate.propose(self._reqs(), observed_volatility(self._reqs()))
        if moves and moves[0].applicable and moves[0].scope == "within-container":
            by_i = {0: "tools", 1: "tools", 2: "system", 3: "user"}
            containers = [by_i[i] for i in order]
            first_system = containers.index("system")
            self.assertNotIn("tools", containers[first_system:],
                             "tools cannot appear after system on the wire")

    def test_a_within_tools_move_is_still_offered(self):
        moves, _ = relocate.propose(self._reqs(), observed_volatility(self._reqs()))
        self.assertTrue(moves)
        if moves[0].applicable:
            self.assertEqual(moves[0].scope, "within-container")
            self.assertIn("tools field", moves[0].mechanism)


class TestVolatileWasteIsPerRequest(unittest.TestCase):
    """Charging every request the longest suffix anyone had bills early short
    turns as though they carried a later long conversation."""

    def _reqs(self, lengths):
        out = []
        for i, n in enumerate(lengths):
            out.append(Request(
                request_id=f"r{i}", sent_at=T0 + timedelta(seconds=60 * i),
                model="claude-opus-5",
                usage={"input_tokens": 100, "cache_creation_input_tokens": 40_000},
                segments=[seg(0, "system", 200, f"hdr{i}", "hdr"),
                          seg(1, "system", n, "body", "sys", marked=True, ttl="5m")],
                agent="a", tenant="t", target_id="anthropic/direct",
                ttl_requested="5m"))
        return out

    def _fig(self, lengths):
        a = analyze(TraceSet(requests=self._reqs(lengths), tier=Tier.INSTRUMENTED),
                    allow_unreconciled=True)
        f = next(x for x in a.findings if x.code == "VOL-1")
        return f

    def test_short_requests_are_not_charged_the_longest_suffix(self):
        mixed = self._fig([1000, 1000, 1000, 40000])
        allmax = self._fig([40000] * 4)
        self.assertLess(mixed.avoidable_usd_month.raw(),
                        allmax.avoidable_usd_month.raw() / 2,
                        "three short requests must not be billed as long ones")

    def test_it_reports_typical_and_longest(self):
        f = self._fig([1000, 1000, 1000, 40000])
        self.assertIn("on a typical request", f.detail)
        self.assertIn("at the longest", f.detail)



class TestUntimedRequestsDoNotCrashTheSimulator(unittest.TestCase):
    """Hardening the loader against malformed rows meant accepting a missing
    timestamp rather than dropping the row, which made the simulator sort on
    None and raise. Timing is the substrate here -- expiry, visibility and
    cadence all read sent_at -- so an untimed request is excluded and counted,
    the same answer the structureless case already gets."""

    def _segs(self):
        return [seg(0, "system", 6000, "sys", marked=True, ttl="5m"),
                seg(1, "user", 100, "t")]

    def _mix(self, timed=3, untimed=1):
        out = [Request(request_id=f"t{i}", sent_at=T0 + timedelta(seconds=60 * i),
                       model="claude-opus-5", usage={}, segments=self._segs(), agent="a", target_id="anthropic/direct")
               for i in range(timed)]
        out += [Request(request_id=f"u{i}", sent_at=None, model="claude-opus-5",
                        usage={}, segments=self._segs(), agent="a", target_id="anthropic/direct")
                for i in range(untimed)]
        return out

    def test_a_mixed_trace_simulates_and_reports_the_gap(self):
        res = simulate.simulate(self._mix(), "allocator-lite")
        self.assertEqual(res.untimed, 1)
        self.assertEqual(len(res.priced), 3)

    def test_an_entirely_untimed_trace_returns_rather_than_raising(self):
        res = simulate.simulate(self._mix(timed=0, untimed=3), "allocator-lite")
        self.assertEqual(res.untimed, 3)
        self.assertEqual(res.priced, [])
        self.assertEqual(res.spend().raw(), 0.0)

    def test_relocation_survives_an_entirely_untimed_trace(self):
        """It indexes reqs[0], which an all-untimed trace does not have."""
        self.assertEqual(
            simulate.simulate(self._mix(timed=0, untimed=2), "relocation-lite").untimed, 2)

    def test_the_bake_off_states_what_it_skipped(self):
        b = simulate.bake_off(self._mix(timed=8, untimed=2))
        self.assertEqual(b.untimed, 2)
        self.assertIn("carry no timestamp", str(b))


class TestStructuralClaimsNeedMeasuredSegmentation(unittest.TestCase):
    """A note saying structural findings are unvalidated, sitting above a
    released dollar figure at confidence high, is the contradiction this gate
    exists to prevent. VOL-1 is the finding that tells someone to reorder
    prompt authority."""

    def _reqs(self):
        out = []
        for i in range(12):
            out.append(Request(
                request_id=f"r{i}", sent_at=T0 + timedelta(seconds=60 * i),
                model="claude-opus-5",
                usage={"input_tokens": 200, "cache_creation_input_tokens": 40_000,
                       "cache_read_input_tokens": 0},
                segments=[seg(0, "system", 300, f"hdr{i}", "hdr"),
                          seg(1, "system", 30000, "body", "sys", marked=True, ttl="5m")],
                agent="a", tenant="t", target_id="anthropic/direct", ttl_requested="5m"))
        return out

    def _analyse(self, tier, alignment):
        ts = TraceSet(requests=self._reqs(), tier=tier, alignment=alignment)
        return analyze(ts, allow_unreconciled=True)

    def _vol(self, a):
        return next(f for f in a.findings if f.code == "VOL-1")

    def test_inferred_with_no_alignment_withholds_structural_money(self):
        a = self._analyse(Tier.INFERRED, None)
        v = self._vol(a)
        self.assertFalse(v.avoidable_usd_month.released)
        self.assertIn("unmeasured", v.avoidable_usd_month.withheld_because)
        self.assertEqual(v.confidence, "low")
        self.assertEqual(v.severity, "medium")

    def test_inferred_below_the_floor_still_withholds(self):
        v = self._vol(self._analyse(Tier.INFERRED, 0.72))
        self.assertFalse(v.avoidable_usd_month.released)
        self.assertIn("72%", v.avoidable_usd_month.withheld_because)

    # `avoidable_usd_window`, not `_month`, in every positive control below.
    # This fixture is twelve requests over eleven minutes, which is far under
    # the one-day projection floor, so its monthly figures are withheld for a
    # reason that has nothing to do with segmentation and asserting them here
    # would test the wrong gate. The window figure is the one this class is
    # about: it is what the structural gate releases or withholds, and it is
    # not an extrapolation. The negative controls still read `_month`, because
    # withholding is withholding whichever figure carries the reason.
    def test_inferred_at_or_above_the_floor_releases(self):
        v = self._vol(self._analyse(Tier.INFERRED, 0.95))
        self.assertTrue(v.avoidable_usd_window.released)
        self.assertEqual(v.severity, "high")

    def test_instrumented_releases_without_an_alignment_score(self):
        """There is nothing to align against when the ids came from source."""
        self.assertTrue(self._vol(self._analyse(Tier.INSTRUMENTED, None))
                        .avoidable_usd_window.released)

    def test_usage_derived_findings_are_unaffected(self):
        """EFF-1 reads usage counters, not structure, so unmeasured
        segmentation says nothing about it."""
        a = self._analyse(Tier.INFERRED, None)
        eff = next(f for f in a.findings if f.code == "EFF-1")
        self.assertFalse(eff.structural)
        self.assertTrue(eff.avoidable_usd_window.released)

    def test_the_structural_gate_reaches_the_window_figure_too(self):
        """The migration above is only honest if this gate reaches both figures.

        Were the window figure left released while the monthly one was withheld
        for segmentation, the positive controls above would be reporting a pass
        from a figure the structural gate had never touched -- and the report
        would print the window amount for a finding it had just refused to
        cost. That is the fix-one-site-leave-the-rest shape this whole suite
        exists to catch, occurring inside the fix for it.
        """
        for alignment in (None, 0.72):
            with self.subTest(alignment=alignment):
                v = self._vol(self._analyse(Tier.INFERRED, alignment))
                self.assertFalse(v.avoidable_usd_window.released,
                                 "structural money escaped through the window "
                                 "figure")
                self.assertEqual(v.avoidable_usd_month.withheld_because,
                                 v.avoidable_usd_window.withheld_because,
                                 "the two figures give different reasons for "
                                 "one refusal")

    def test_the_reason_is_stated_in_the_notes(self):
        a = self._analyse(Tier.INFERRED, None)
        self.assertTrue(any("without dollar figures" in n for n in a.notes))


class TestVolatileWasteIsTheRecoverableDelta(unittest.TestCase):
    """Charging every affected request its full write cost was wrong three ways:
    the first request still writes after the move, later ones become reads at
    0.1x rather than free, and a 1h write bills 2.0x not 1.25x."""

    def _reqs(self, n, gap, ttl="5m"):
        return [Request(
            request_id=f"r{i}", sent_at=T0 + timedelta(seconds=gap * i),
            model="claude-opus-5",
            usage={"input_tokens": 100, "cache_creation_input_tokens": 30000},
            segments=[seg(0, "system", 200, f"hdr{i}", "hdr"),
                      seg(1, "system", 30000, "body", "sys", marked=True, ttl=ttl)],
            agent="a", tenant="t", session="s", target_id="anthropic/direct",
            ttl_requested=ttl) for i in range(n)]

    def _fig(self, n, gap, ttl="5m"):
        a = analyze(TraceSet(requests=self._reqs(n, gap, ttl), tier=Tier.INSTRUMENTED),
                    allow_unreconciled=True)
        f = next((x for x in a.findings if x.code == "VOL-1"), None)
        return f.avoidable_usd_month.raw() if f and f.avoidable_usd_month else 0.0

    def test_the_first_write_is_not_recoverable(self):
        """Two requests, one transition. Charging both doubled the figure."""
        two, three = self._fig(2, 60), self._fig(3, 60)
        self.assertAlmostEqual(three / two, 2.0, places=6,
                               msg="n requests yield n-1 recoverable transitions")

    def test_one_hour_writes_recover_more_than_five_minute_writes(self):
        ratio = self._fig(12, 60, "1h") / self._fig(12, 60, "5m")
        self.assertAlmostEqual(ratio, (2.0 - 0.10) / (1.25 - 0.10), places=6)

    def test_requests_further_apart_than_the_lifetime_recover_nothing(self):
        """Cold before the move and cold after it, so relocation cannot help."""
        self.assertEqual(self._fig(12, 7200, "5m"), 0.0)

    def test_a_one_hour_lifetime_keeps_wider_gaps_recoverable(self):
        """The same 30-minute cadence is hopeless at 5m and recoverable at 1h."""
        self.assertEqual(self._fig(12, 1800, "5m"), 0.0)
        self.assertGreater(self._fig(12, 1800, "1h"), 0.0)


class TestRelocationRefusesMixedPromptShapes(unittest.TestCase):
    """One order is emitted for a whole group, so every request has to agree on
    what an index means. Last-writer-wins over heterogeneous prompts let index 0
    be `system` where a system prompt existed and `user` where it did not, then
    applied one order to both -- reordering conversation history under the label
    of a safe system move."""

    def _mixed(self):
        return [
            Request(request_id="a", sent_at=T0, model="claude-opus-5", usage={}, agent="a",
                    segments=[seg(0, "user", 300, "user0"),
                              seg(1, "assistant", 9000, "assistant1")], target_id="anthropic/direct"),
            Request(request_id="b", sent_at=T0 + timedelta(seconds=60), model="claude-opus-5",
                    usage={}, agent="a",
                    segments=[seg(0, "system", 300, "sysvary"),
                              seg(1, "system", 9000, "stable")], target_id="anthropic/direct"),
        ]

    def test_a_mixed_group_proposes_nothing_applicable(self):
        moves, order = relocate.propose(self._mixed(), observed_volatility(self._mixed()))
        self.assertTrue(moves)
        self.assertFalse(any(m.applicable for m in moves))
        self.assertEqual(order, sorted(order), "the natural order is left alone")

    def test_the_conflicting_index_is_named(self):
        moves, _ = relocate.propose(self._mixed(), observed_volatility(self._mixed()))
        self.assertIn("segment index 0", moves[0].blocked_by)
        self.assertIn("system", moves[0].blocked_by)
        self.assertIn("user", moves[0].blocked_by)

    def test_a_homogeneous_group_still_gets_its_move(self):
        reqs = volatile_head(n=6)
        moves, order = relocate.propose(reqs, observed_volatility(reqs))
        self.assertTrue(moves[0].applicable)
        self.assertEqual(moves[0].scope, "within-container")


class TestStructuralCoverageGatesStructuralMoney(unittest.TestCase):
    """A mixed file whose present ids are all trusted still classifies as
    instrumented, but the requests carrying no segments cannot support a
    VOL/FAN/MIN claim -- and they may be exactly the ones that would change its
    cause or scope. structural_coverage was measured by the loader and never
    wired to the gate that needs it."""

    def _ts(self, coverage, structured=5, total=10):
        reqs = []
        for i in range(total):
            segs = ([seg(0, "system", 200, f"h{i}", "hdr"),
                     seg(1, "system", 30000, "body", "sys", marked=True, ttl="5m")]
                    if i < structured else [])
            reqs.append(Request(
                request_id=f"r{i}", sent_at=T0 + timedelta(seconds=60 * i),
                model="claude-opus-5",
                usage={"input_tokens": 100, "cache_creation_input_tokens": 30000},
                segments=segs, agent="a", tenant="t", target_id="anthropic/direct",
                ttl_requested="5m"))
        return TraceSet(requests=reqs, tier=Tier.INSTRUMENTED,
                        structural_coverage=coverage)

    def _vol(self, a):
        return next((f for f in a.findings if f.code == "VOL-1"), None)

    def test_half_covered_withholds_structural_money(self):
        a = analyze(self._ts(0.5), allow_unreconciled=True)
        self.assertFalse(self._vol(a).avoidable_usd_month.released)
        self.assertIn("carry prompt structure", self._vol(a).avoidable_usd_month.withheld_because)

    # `avoidable_usd_window` in the positive controls: this fixture is ten
    # requests over nine minutes, so its monthly figures are withheld by the
    # projection floor regardless of coverage, and asserting them would test a
    # gate this class is not about. The window figure is what coverage gates.
    def test_fully_covered_releases(self):
        a = analyze(self._ts(1.0, structured=10), allow_unreconciled=True)
        self.assertTrue(self._vol(a).avoidable_usd_window.released)

    def test_usage_derived_findings_are_unaffected_by_coverage(self):
        a = analyze(self._ts(0.5), allow_unreconciled=True)
        eff = next(f for f in a.findings if f.code == "EFF-1")
        self.assertTrue(eff.avoidable_usd_window.released)

    def test_partial_coverage_withholds_the_window_figure_too(self):
        """The other half of the pair above. A gate that reached the monthly
        figure and not the window one would put the withheld amount straight
        back on the page, because the report falls back to the window figure
        exactly when the monthly one is missing."""
        v = self._vol(analyze(self._ts(0.5), allow_unreconciled=True))
        self.assertFalse(v.avoidable_usd_window.released)
        self.assertIn("carry prompt structure", v.avoidable_usd_window.withheld_because)

    def test_affected_requests_counts_only_rows_that_contributed(self):
        a = analyze(self._ts(1.0, structured=10), allow_unreconciled=True)
        self.assertEqual(self._vol(a).affected_requests, 10)
        b = analyze(self._ts(0.5), allow_unreconciled=True)
        self.assertLessEqual(self._vol(b).affected_requests, 5,
                             "rows with no segments cannot be affected by a prefix finding")


class TestNormalisedIngestToleratesATruncatedLine(unittest.TestCase):
    """Append-only trace files get truncated by crashes, and a partly written
    final line is the normal shape of that. The body path already counted them
    as gaps; the normalised loader aborted the whole file."""

    def _file(self, trailing):
        path = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False).name
        with open(path, "w") as f:
            for i in range(4):
                f.write(json.dumps({"request_id": f"r{i}", "sent_at": T0.isoformat(),
                                    "model": "claude-opus-5",
                                    "usage": {"input_tokens": 10}}) + "\n")
            f.write(trailing)
        return path

    def test_the_valid_rows_survive(self):
        p = self._file('{"request_id": "trunc"')
        ts = load_jsonl(p)
        self.assertEqual(len(ts.requests), 4)
        os.unlink(p)

    def test_the_bad_line_is_counted_not_hidden(self):
        p = self._file('{"request_id": "trunc"')
        self.assertTrue(any("could not be parsed" in n for n in load_jsonl(p).notes))
        os.unlink(p)

    def test_a_clean_file_carries_no_such_note(self):
        p = self._file("")
        self.assertFalse(any("could not be parsed" in n for n in load_jsonl(p).notes))
        os.unlink(p)


class TestVolatilityIsKeyedOnPositionNotLabel(unittest.TestCase):
    """Including the label meant a segment whose label changed at the same wire
    position landed in a different bucket each time, so every bucket held one
    value and the drift disappeared -- while the cached prefix bytes had changed
    exactly as much as if the text had."""

    def _reqs(self, changing_label):
        out = []
        for i in range(6):
            out.append(Request(
                request_id=f"r{i}", sent_at=T0 + timedelta(seconds=60 * i),
                model="claude-opus-5",
                usage={"input_tokens": 100, "cache_creation_input_tokens": 30000},
                segments=[seg(0, "tools", 300, f"v{i}",
                              f"tool:name{i}" if changing_label else "tool:fixed"),
                          seg(1, "system", 30000, "body", "sys", marked=True, ttl="5m")],
                agent="a", tenant="t", target_id="anthropic/direct", ttl_requested="5m"))
        return out

    def _codes(self, changing_label):
        a = analyze(TraceSet(requests=self._reqs(changing_label), tier=Tier.INSTRUMENTED),
                    allow_unreconciled=True)
        return [f.code for f in a.findings], a

    def test_a_changing_label_is_still_detected_as_drift(self):
        codes, _ = self._codes(True)
        self.assertIn("VOL-1", codes)

    def test_a_stable_label_is_detected_the_same_way(self):
        codes, _ = self._codes(False)
        self.assertIn("VOL-1", codes)

    def test_the_finding_names_the_labels_it_saw(self):
        _, a = self._codes(True)
        vol = next(f for f in a.findings if f.code == "VOL-1")
        self.assertIn("tool:name0", vol.detail)


class TestToolsAreNotRelocatableAsSystemText(unittest.TestCase):
    """tools and system share an authority class, which is why round 17 split
    container from authority for within-block moves. The cross-authority branch
    was still keyed on authority, so a volatile tool definition could be moved
    into messages[] via the role:system mechanism -- and a tool in a message is
    not a tool."""

    def _reqs(self):
        return [Request(
            request_id=f"r{i}", sent_at=T0 + timedelta(seconds=60 * i),
            model="claude-opus-5", usage={}, agent="a", target_id="anthropic/direct",
            segments=[seg(0, "tools", 400, f"vary{i}", "tool:search"),
                      seg(1, "system", 14000, "stable", "instructions"),
                      seg(2, "user", 100, f"t{i}")]) for i in range(6)]

    def test_the_move_is_blocked(self):
        moves, order = relocate.propose(self._reqs(), observed_volatility(self._reqs()))
        self.assertEqual(moves[0].risk, "blocked")
        self.assertEqual(moves[0].scope, "cross-container")
        self.assertEqual(order, sorted(order), "the order is left untouched")

    def test_the_reason_names_the_container(self):
        moves, _ = relocate.propose(self._reqs(), observed_volatility(self._reqs()))
        self.assertIn("no longer a tool", moves[0].blocked_by)

    def test_a_within_tools_move_is_still_allowed(self):
        reqs = []
        for i in range(6):
            reqs.append(Request(
                request_id=f"r{i}", sent_at=T0 + timedelta(seconds=60 * i),
                model="claude-opus-5", usage={}, agent="a", target_id="anthropic/direct",
                segments=[seg(0, "tools", 400, f"vary{i}", "tool:search"),
                          seg(1, "tools", 9000, "stable_tool", "tool:read"),
                          seg(2, "user", 100, f"t{i}")]))
        moves, _ = relocate.propose(reqs, observed_volatility(reqs))
        self.assertTrue(moves[0].applicable)
        self.assertEqual(moves[0].scope, "within-container")


class TestTrustedIdsMustBeTheRightShape(unittest.TestCase):
    """`hmac:` and `hmac:redacted` both start with a trusted scheme. Accepting
    them classified a trace as instrumented while preserving ids that identify
    nothing, so unrelated content collapsed into one apparently stable prefix
    and carried structural findings at high confidence."""

    def test_a_full_digest_is_trusted(self):
        self.assertTrue(is_trusted_id(hid("x")))

    def test_a_bare_scheme_is_not(self):
        self.assertFalse(is_trusted_id("hmac:"))

    def test_a_placeholder_is_not(self):
        self.assertFalse(is_trusted_id("hmac:redacted"))

    def test_a_bare_digest_scheme_is_not(self):
        self.assertFalse(is_trusted_id("sha256:" + "a" * 64))

    def test_surrounding_whitespace_is_tolerated(self):
        self.assertTrue(is_trusted_id("  " + hid("x") + "  "))

    def test_a_placeholder_id_does_not_make_a_trace_instrumented(self):
        path = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False).name
        with open(path, "w") as f:
            for i in range(3):
                f.write(json.dumps({
                    "request_id": f"r{i}", "sent_at": T0.isoformat(),
                    "model": "claude-opus-5", "usage": {"input_tokens": 10},
                    "segments": [{"index": 0, "role": "system", "tokens": 10,
                                  "id": "hmac:redacted", "content": f"c{i}"}]}) + "\n")
        ts = load_jsonl(path, key=b"k")
        self.assertIsNot(ts.tier, Tier.INSTRUMENTED)
        self.assertEqual(len({r.segments[0].id for r in ts.requests}), 3,
                         "re-derived from content, so changing text changes the id")
        os.unlink(path)


class TestIngestGuardsExistOnBothLoaders(unittest.TestCase):
    """Three times now a tolerance has been written into one loader and not its
    twin. These assert the pair rather than the instance."""

    def _jsonl(self, **row):
        path = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False).name
        base = {"request_id": "a", "sent_at": T0.isoformat(),
                "model": "claude-opus-5", "usage": {"input_tokens": 10}}
        base.update(row)
        with open(path, "w") as f:
            f.write(json.dumps(base) + "\n")
        return path

    def test_a_malformed_timestamp_is_a_gap_not_a_crash(self):
        p = self._jsonl(sent_at="not-a-date")
        ts = load_jsonl(p)
        self.assertEqual(len(ts.requests), 1)
        self.assertIsNone(ts.requests[0].sent_at)
        self.assertTrue(any("no usable timestamp" in n for n in ts.notes))
        os.unlink(p)

    def test_the_normalised_loader_strips_a_date_suffix(self):
        p = self._jsonl(model="claude-haiku-4-5-20251001")
        ts = load_jsonl(p)
        self.assertEqual(ts.requests[0].model, "claude-haiku-4-5")
        self.assertTrue(any("normalised" in n for n in ts.notes))
        os.unlink(p)

    def test_the_body_loader_strips_it_too(self):
        from cacheeconomics.adapters.bodies import load_bodies
        path = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False).name
        with open(path, "w") as f:
            f.write(json.dumps({
                "request_id": "a", "sent_at": T0.isoformat(),
                "request": {"model": "claude-haiku-4-5-20251001",
                            "messages": [{"role": "user", "content": "x"}]},
                "response": {"usage": {"input_tokens": 10, "output_tokens": 1,
                                       "cache_read_input_tokens": 0,
                                       "cache_creation_input_tokens": 0,
                                       "cache_creation": {}}}}) + "\n")
        ts = load_bodies(path, key=b"k")
        self.assertEqual(ts.requests[0].model, "claude-haiku-4-5")
        os.unlink(path)

    def test_an_unpriceable_model_is_left_alone_by_normalisation(self):
        p = self._jsonl(model="claude-not-real-20251001")
        self.assertEqual(load_jsonl(p).requests[0].model, "claude-not-real-20251001")
        os.unlink(p)


class TestAsShippedHonoursRowLevelTtl(unittest.TestCase):
    """The analyzer accepts ttl_requested as proof of lifetime, so defaulting
    to 5m here priced a 1h trace as 1h in spend and replayed it as 5m in the
    bake-off -- fabricating expiries and reporting savings against a baseline
    nobody shipped."""

    def _reqs(self, ttl_requested, gap=600):
        return [Request(
            request_id=f"r{i}", sent_at=T0 + timedelta(seconds=gap * i),
            model="claude-opus-5", usage={}, agent="a", target_id="anthropic/direct",
            ttl_requested=ttl_requested,
            segments=[seg(0, "system", 8000, "sys", marked=True),
                      seg(1, "user", 100, f"t{i}")]) for i in range(4)]

    def test_a_one_hour_row_survives_ten_minute_gaps(self):
        res = simulate.simulate(self._reqs("1h"), "as-shipped")
        self.assertEqual(res.writes, 1)
        self.assertEqual(res.reads, 3)

    def test_the_same_trace_at_five_minutes_rewrites(self):
        res = simulate.simulate(self._reqs("5m"), "as-shipped")
        self.assertEqual(res.reads, 0)

    def test_a_segment_ttl_still_wins_over_the_row(self):
        reqs = self._reqs("1h")
        for r in reqs:
            r.segments[0] = seg(0, "system", 8000, "sys", marked=True, ttl="5m")
        self.assertEqual(simulate.simulate(reqs, "as-shipped").reads, 0)

    def test_an_implicit_marker_beside_an_explicit_one_stays_five_minutes(self):
        """Row metadata cannot upgrade a silent sibling marker. The body on the
        wire asked for the provider default there."""
        reqs = self._reqs("1h")
        for r in reqs:
            r.segments = [
                seg(0, "system", 8000, "sys", marked=True, ttl="1h"),
                seg(1, "user", 100, "stable-turn", marked=True),
            ]
        res = simulate.simulate(reqs, "as-shipped")
        self.assertEqual(res.reads, 3)
        self.assertEqual(res.writes, 4)


class TestBakeOffSurvivesUnpriceableSurfaces(unittest.TestCase):
    """cost.price refuses surfaces without Anthropic-shaped multipliers, and
    the analyzer already excludes those requests. The bake-off had no such
    path, so one mixed-surface export aborted the whole comparison."""

    def _mixed(self):
        out = []
        for i in range(8):
            openai = i % 4 == 0
            out.append(Request(
                request_id=f"m{i}", sent_at=T0 + timedelta(seconds=60 * i),
                model="gpt-5.6" if openai else "claude-opus-5", usage={}, agent="a",
                target_id="openai/direct" if openai else "anthropic/direct",
                segments=[seg(0, "system", 8000, "sys", marked=True, ttl="5m"),
                          seg(1, "user", 100, f"t{i}")]))
        return out

    def test_the_comparison_completes_without_raising(self):
        """It must not crash. It must also not produce a percentage: this
        assertion originally required delta_pct, which enshrined the bug of
        reporting a whole-workload verdict over a priced subset."""
        b = simulate.bake_off(self._mixed())
        self.assertIsNone(b.delta_pct)
        self.assertIn("indeterminate", b.verdict)

    def test_the_unpriceable_rows_are_counted(self):
        self.assertEqual(simulate.bake_off(self._mixed()).unpriceable, 2)

    def test_the_shortfall_is_stated(self):
        self.assertIn("cannot price", str(simulate.bake_off(self._mixed())))


class TestUntimedRowsAreNotPricedAtTheRunDate(unittest.TestCase):
    """cost.price falls back to today when handed no date, so once the loaders
    tolerated malformed timestamps those rows silently repriced historical
    traffic at whatever rate applies on the day the report runs."""

    def _ts(self):
        return TraceSet(requests=[Request(
            request_id=f"u{i}", sent_at=None, model="claude-sonnet-5",
            usage={"input_tokens": 100000, "cache_creation_input_tokens": 0},
            segments=[], agent="a", target_id="anthropic/direct") for i in range(3)], tier=Tier.USAGE_ONLY)

    def test_they_are_excluded_and_the_gate_fails(self):
        a = analyze(self._ts(), allow_unreconciled=True)
        self.assertFalse(a.spend["input_usd"].released)
        self.assertTrue(any("no usable timestamp" in n for n in a.notes))

    def test_an_explicit_date_makes_them_priceable(self):
        a = analyze(self._ts(), on_date="2026-08-01", allow_unreconciled=True)
        self.assertTrue(a.spend["input_usd"].released)

    def test_an_effective_rate_also_settles_it(self):
        a = analyze(self._ts(), effective_rate=2.0, allow_unreconciled=True)
        self.assertTrue(a.spend["input_usd"].released)


class TestVolatileChainsMatchTheReportedPosition(unittest.TestCase):
    """affected_chains was read from the unsorted list while idx was chosen
    after sorting, so a trace whose segments arrive out of index order could
    report one position and price another position's reuse chains."""

    def _reqs(self):
        def mk(i, tenant):
            return Request(
                request_id=f"r{i}", sent_at=T0 + timedelta(seconds=60 * i),
                model="claude-opus-5",
                usage={"input_tokens": 100, "cache_creation_input_tokens": 30000},
                agent="a", tenant=tenant, target_id="anthropic/direct",
                ttl_requested="5m",
                segments=[seg(2, "system", 20000, "stable2", "body", marked=True, ttl="5m"),
                          seg(1, "system", 300, f"v1-{i}" if tenant == "A" else "fixed1", "p1"),
                          seg(0, "system", 300, f"v0-{i}" if tenant == "B" else "fixed0", "p0")])
        return [mk(i, "A") for i in range(6)] + [mk(i + 100, "B") for i in range(6)]

    def test_the_lowest_volatile_position_is_reported(self):
        a = analyze(TraceSet(requests=self._reqs(), tier=Tier.INSTRUMENTED),
                    allow_unreconciled=True)
        vol = next(f for f in a.findings if f.code == "VOL-1")
        self.assertIn("position 0", vol.title)

    def test_only_that_positions_chain_is_charged(self):
        a = analyze(TraceSet(requests=self._reqs(), tier=Tier.INSTRUMENTED),
                    allow_unreconciled=True)
        vol = next(f for f in a.findings if f.code == "VOL-1")
        self.assertEqual(vol.affected_requests, 6,
                         "position 0 is volatile only in tenant B")


class TestUnmodelledLifetimesDoNotCrashTheReplay(unittest.TestCase):
    """This replay models the two Anthropic lifetimes. The registry advertises
    openai/direct with a 30m TTL, and indexing the map by a marker's lifetime
    raised KeyError before spend() could reach its fail-closed path."""

    def _reqs(self, ttl):
        return [Request(
            request_id=f"r{i}", sent_at=T0 + timedelta(seconds=60 * i),
            model="gpt-5.6", usage={}, agent="a", target_id="openai/direct",
            segments=[seg(0, "system", 8000, "sys", marked=True, ttl=ttl),
                      seg(1, "user", 100, f"t{i}")]) for i in range(3)]

    def test_a_thirty_minute_marker_is_counted_not_raised(self):
        res = simulate.simulate(self._reqs("30m"), "as-shipped")
        self.assertEqual(res.unmodelled_ttl, 3)

    def test_those_requests_are_treated_as_uncached(self):
        res = simulate.simulate(self._reqs("30m"), "as-shipped")
        self.assertEqual(res.reads, 0)
        self.assertEqual(res.writes, 0)

    def test_the_bake_off_completes_and_says_so(self):
        b = simulate.bake_off(self._reqs("30m"))
        self.assertEqual(b.unmodelled_ttl, 3)
        self.assertIn("does not model", str(b))

    def test_a_modelled_lifetime_is_unaffected(self):
        res = simulate.simulate(self._reqs("5m"), "as-shipped")
        self.assertEqual(res.unmodelled_ttl, 0)


class TestUndatedRowsCannotCertifyReconciliation(unittest.TestCase):
    """render_html shows within_ship_gate as the visible verdict. Excluding
    undated rows from spend without counting them here let an invoice match a
    partial subtotal and certify it while the figures were withheld."""

    def _ts(self):
        return TraceSet(requests=[Request(
            request_id=f"u{i}", sent_at=None, model="claude-opus-5",
            usage={"input_tokens": 1000, "cache_creation_input_tokens": 0},
            segments=[], agent="a", target_id="anthropic/direct") for i in range(3)], tier=Tier.USAGE_ONLY)

    def test_the_gate_fails_with_undated_rows(self):
        a = analyze(self._ts(), invoice_usd=5.0)
        self.assertFalse(a.reconciliation["within_ship_gate"])

    def test_they_are_counted_as_unpriced(self):
        a = analyze(self._ts(), invoice_usd=5.0)
        self.assertEqual(a.reconciliation["unpriced_requests"], 3)

    def test_a_zero_invoice_does_not_crash(self):
        """A client can hand over a zero invoice, usually meaning the export and
        the bill do not describe the same period."""
        a = analyze(self._ts(), invoice_usd=0.0)
        self.assertFalse(a.reconciliation["within_ship_gate"])
        self.assertIn("zero", a.spend["input_usd"].withheld_because)

    def test_the_verdict_matches_the_figures(self):
        a = analyze(self._ts(), invoice_usd=5.0)
        self.assertEqual(a.reconciliation["within_ship_gate"],
                         a.spend["input_usd"].released)

    def test_a_negative_invoice_cannot_release_figures(self):
        """`delta` is an absolute value; the divisor was signed. So a negative
        invoice made every mismatch a negative percentage, and the gate asks
        `pct <= 5.0`.

        Measured before the fix: -$60 against $60 of computed spend gave -200.0%,
        passed the ship gate, and published the figures. -$1bn did the same. This
        walked straight through the one invariant the project sells."""
        for invoice in (-60.0, -1e9, -0.01):
            with self.subTest(invoice=invoice):
                a = analyze(self._ts(), invoice_usd=invoice)
                r = a.reconciliation
                self.assertFalse(r["within_ship_gate"])
                self.assertIsNone(r["delta_pct"])
                self.assertFalse(a.spend["input_usd"].released)
                # And it must name the real blocker, not the ±5% one.
                self.assertEqual(r["invalid_invoice"], "negative")
                self.assertIn("negative", a.spend["input_usd"].withheld_because)

    def test_a_negative_invoice_reports_no_delta(self):
        """"$60 computed, -$60 invoiced, delta $120" reads like a finding rather
        than a rejected input."""
        self.assertIsNone(
            analyze(self._ts(), invoice_usd=-60.0).reconciliation["delta_usd"])

    def test_a_nonfinite_invoice_cannot_release_figures(self):
        for invoice in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(invoice=invoice):
                a = analyze(self._ts(), invoice_usd=invoice)
                self.assertFalse(a.reconciliation["within_ship_gate"])
                self.assertFalse(a.spend["input_usd"].released)
                self.assertEqual(a.reconciliation["invalid_invoice"], "not-finite")

    def test_zero_keeps_its_own_diagnosis(self):
        """Zero and negative are different problems -- a mismatched period versus
        a credit passed as a spend total -- so collapsing both into "not
        positive" would have thrown away the more useful sentence."""
        a = analyze(self._ts(), invoice_usd=0.0)
        self.assertEqual(a.reconciliation["invalid_invoice"], "zero")
        self.assertIn("zero", a.spend["input_usd"].withheld_because)
        self.assertNotIn("negative", a.spend["input_usd"].withheld_because)

    def test_a_valid_invoice_still_reconciles(self):
        """The gate has to let a real invoice through, or it is just an off
        switch."""
        a = analyze(self._ts(), invoice_usd=5.0)
        self.assertEqual(a.reconciliation["invalid_invoice"], "")
        self.assertIsNotNone(a.reconciliation["delta_pct"])


class TestRelocationRequiresSectionAgreement(unittest.TestCase):
    """A volatile session header and a stable safety policy are both
    system/system, so index 0 passed the shape check and segs was built
    last-writer-wins -- letting a move be proposed for content that was not the
    blocker in half the group."""

    def _mixed(self):
        out = []
        for i in range(6):
            lbl = "session_header" if i % 2 else "safety_policy"
            out.append(Request(
                request_id=f"m{i}", sent_at=T0 + timedelta(seconds=60 * i),
                model="claude-opus-5", usage={}, agent="a", target_id="anthropic/direct",
                segments=[seg(0, "system", 300, f"v{i}", lbl),
                          seg(1, "system", 14000, "stable", "instructions",
                              marked=True, ttl="5m")]))
        return out

    def test_differing_sections_at_one_index_block_the_move(self):
        moves, order = relocate.propose(self._mixed(), observed_volatility(self._mixed()))
        self.assertEqual(moves[0].risk, "blocked")
        self.assertEqual(order, sorted(order))

    def test_the_reason_names_both_sections(self):
        moves, _ = relocate.propose(self._mixed(), observed_volatility(self._mixed()))
        self.assertIn("safety_policy", moves[0].blocked_by)
        self.assertIn("session_header", moves[0].blocked_by)

    def test_a_consistent_schema_still_proposes(self):
        reqs = volatile_head(n=6)
        moves, _ = relocate.propose(reqs, observed_volatility(reqs))
        self.assertTrue(moves[0].applicable)


class TestPartialBakeOffsAreIndeterminate(unittest.TestCase):
    """Gate 1 asks whether placement beats automatic injection on a workload.
    If some of that workload contributed nothing, the comparison describes
    whatever was left, and the omitted traffic could dominate spend. Printing a
    percentage anyway is how a subset gets read as a whole-workload result."""

    def _clean(self, n=8):
        return [Request(
            request_id=f"c{i}", sent_at=T0 + timedelta(seconds=60 * i),
            model="claude-opus-5", usage={}, agent="a", target_id="anthropic/direct",
            segments=[seg(0, "system", 8000, "sys", marked=True, ttl="5m"),
                      seg(1, "user", 100, f"t{i}")]) for i in range(n)]

    def test_a_clean_trace_still_produces_a_verdict(self):
        reqs = self._clean()
        ts = instrumented(reqs)
        b = simulate.bake_off(ts.analysable, trace=ts)
        self.assertIsNotNone(b.delta_pct)
        self.assertNotIn("indeterminate", b.verdict)

    def test_one_structureless_request_makes_it_indeterminate(self):
        reqs = self._clean()
        reqs.append(Request(request_id="x", sent_at=T0 + timedelta(seconds=900),
                            model="claude-opus-5", usage={}, segments=[], agent="a", target_id="anthropic/direct"))
        b = simulate.bake_off(reqs)
        self.assertIsNone(b.delta_pct)
        self.assertIn("indeterminate", b.verdict)
        self.assertIn("without structure", b.verdict)

    def test_an_untimed_request_makes_it_indeterminate(self):
        reqs = self._clean()
        reqs.append(Request(request_id="x", sent_at=None, model="claude-opus-5",
                            usage={}, agent="a",
                            segments=[seg(0, "system", 8000, "sys", marked=True, ttl="5m")], target_id="anthropic/direct"))
        self.assertIn("without timestamps", simulate.bake_off(reqs).verdict)

    def test_the_relocation_verdict_is_indeterminate_too(self):
        reqs = self._clean()
        reqs.append(Request(request_id="x", sent_at=T0, model="claude-opus-5",
                            usage={}, segments=[], agent="a", target_id="anthropic/direct"))
        b = simulate.bake_off(reqs)
        self.assertIsNone(b.delta_pct_relocation)
        self.assertIn("indeterminate", b.verdict_relocation)

    def test_the_range_reads_indeterminate_rather_than_a_dash(self):
        reqs = self._clean()
        reqs.append(Request(request_id="x", sent_at=T0, model="claude-opus-5",
                            usage={}, segments=[], agent="a", target_id="anthropic/direct"))
        self.assertIn("indeterminate", str(simulate.bake_off(reqs)))


class TestEffOnePricesNetExcessNotGrossPremium(unittest.TestCase):
    """A low efficiency ratio does not mean caching is losing money. The reads
    it earned are credited at 0.1x and can more than pay for the writes.
    Charging every write premium told a customer their cache was wasting money
    on a workload that was already saving it."""

    def _reqs(self, writes, reads, n=10):
        return [Request(
            request_id=f"r{i}", sent_at=T0 + timedelta(seconds=60 * i),
            model="claude-opus-5",
            usage={"input_tokens": 0, "cache_creation_input_tokens": writes,
                   "cache_read_input_tokens": reads},
            segments=[], agent="a", ttl_requested="5m", target_id="anthropic/direct") for i in range(n)]

    def _eff(self, writes, reads):
        a = analyze(TraceSet(requests=self._reqs(writes, reads), tier=Tier.USAGE_ONLY),
                    allow_unreconciled=True)
        return next((f for f in a.findings if f.code == "EFF-1"), None)

    def test_a_net_beneficial_cache_is_not_accused(self):
        """29% efficiency, and still cheaper than sending everything uncached."""
        self.assertIsNone(self._eff(100_000, 40_000))

    def test_a_genuinely_wasteful_cache_is_still_reported(self):
        f = self._eff(100_000, 0)
        self.assertIsNotNone(f)
        self.assertGreater(f.avoidable_usd_month.raw(), 0)

    def test_the_figure_is_the_excess_over_uncached(self):
        """1.25x on writes with no reads means 0.25x of the write volume."""
        f = self._eff(100_000, 0)
        expected_per_request = 100_000 * (5.0 / 1e6) * 0.25
        window_days = (9 * 60) / 86400
        self.assertAlmostEqual(
            f.avoidable_usd_month.raw(),
            expected_per_request * 10 * (30.0 / max(window_days, 1 / 24)),
            delta=1.0)


class TestVolatileWasteRequiresAProvenWrite(unittest.TestCase):
    """Charging every later request inside the lifetime assumed the suffix was
    re-processed, but a request reporting cache reads and no creation read it
    instead -- so a header stable within a session and differing across sessions
    was billed as prefix drift it never caused."""

    def _sessions(self, second_request_writes, header_changes=False):
        out = []
        for sess in range(2):
            for i in range(2):
                wrote = (i == 0) or second_request_writes
                header = f"hdr-s{sess}-t{i}" if header_changes else f"hdr-s{sess}"
                out.append(Request(
                    request_id=f"s{sess}-{i}",
                    sent_at=T0 + timedelta(seconds=600 * sess + 60 * i),
                    model="claude-opus-5",
                    usage={"input_tokens": 100,
                           "cache_creation_input_tokens": 30000 if wrote else 0,
                           "cache_read_input_tokens": 0 if wrote else 30000},
                    segments=[seg(0, "system", 300, header, "hdr"),
                              seg(1, "system", 30000, "body", "sys",
                                  marked=True, ttl="5m")],
                    agent="a", tenant="t", session=f"s{sess}",
                    target_id="anthropic/direct", ttl_requested="5m"))
        return out

    def _vol(self, second_writes, header_changes=False):
        a = analyze(TraceSet(requests=self._sessions(second_writes, header_changes),
                             tier=Tier.INSTRUMENTED), allow_unreconciled=True)
        return next((f for f in a.findings if f.code == "VOL-1"), None)

    def test_working_within_session_reuse_is_not_charged(self):
        self.assertIsNone(self._vol(False))

    def test_a_per_session_stable_header_is_not_volatility_even_if_it_rewrites(self):
        self.assertIsNone(self._vol(True))

    def test_a_genuine_within_session_change_is_still_charged(self):
        f = self._vol(True, header_changes=True)
        self.assertIsNotNone(f)
        self.assertGreater(f.avoidable_usd_month.raw(), 0)


class TestLoadersUseTargetAwareNormalisation(unittest.TestCase):
    """normalize_model became target-aware and neither loader passed a target,
    so Bedrock rows kept their `anthropic.` prefix and the minimum guard was
    skipped on invoice-priced reports."""

    def _file(self, model, target):
        path = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False).name
        with open(path, "w") as f:
            f.write(json.dumps({"request_id": "a", "sent_at": T0.isoformat(),
                                "model": model, "target_id": target,
                                "usage": {"input_tokens": 10}}) + "\n")
        return path

    def test_the_bedrock_prefix_is_stripped_on_ingest(self):
        p = self._file("anthropic.claude-haiku-4-5", "amazon-bedrock/converse")
        ts = load_jsonl(p)
        self.assertEqual(ts.requests[0].model, "claude-haiku-4-5")
        os.unlink(p)

    def test_the_normalised_model_resolves_a_minimum(self):
        from cacheeconomics import registry
        p = self._file("anthropic.claude-haiku-4-5", "amazon-bedrock/converse")
        m = load_jsonl(p).requests[0].model
        self.assertEqual(registry.min_cacheable_tokens("amazon-bedrock/converse", m), 4096)
        os.unlink(p)

    def test_a_direct_row_is_not_stripped_by_the_bedrock_rule(self):
        p = self._file("claude-haiku-4-5", "anthropic/direct")
        self.assertEqual(load_jsonl(p).requests[0].model, "claude-haiku-4-5")
        os.unlink(p)


class TestEffOneCreditsPaybackOnLaterRequests(unittest.TestCase):
    """Cache payback lands on the later read-only requests. Filtering the sum
    to rows with cache_creation excluded exactly the savings the comparison
    exists to weigh, leaving excess as a fixed fraction of write volume no
    matter how much was read back."""

    def _reqs(self, writes, reads):
        return [
            Request(request_id="w", sent_at=T0, model="claude-opus-5",
                    usage={"input_tokens": 0, "cache_creation_input_tokens": writes,
                           "cache_read_input_tokens": 0},
                    segments=[], agent="a", ttl_requested="5m", target_id="anthropic/direct"),
            # A small real input on the tail request. It used to be zero on all
            # three counters, which is a request that billed nothing at all --
            # not a thing that happens, and now excluded as a placeholder rather
            # than analysed as a free request. A turn that reads nothing still
            # sends a prompt. The figure under test is unchanged at $90.00.
            Request(request_id="r", sent_at=T0 + timedelta(seconds=60),
                    model="claude-opus-5",
                    usage={"input_tokens": 100, "cache_creation_input_tokens": 0,
                           "cache_read_input_tokens": reads},
                    segments=[], agent="a", ttl_requested="5m", target_id="anthropic/direct")]

    def _eff(self, writes, reads):
        a = analyze(TraceSet(requests=self._reqs(writes, reads), tier=Tier.USAGE_ONLY),
                    allow_unreconciled=True)
        return next((f for f in a.findings if f.code == "EFF-1"), None)

    def test_payback_on_a_separate_request_is_credited(self):
        """29% efficiency, net cheaper than uncached, and the payback arrives on
        a row that wrote nothing."""
        self.assertIsNone(self._eff(100_000, 40_000))

    def test_no_payback_still_reports(self):
        f = self._eff(100_000, 0)
        self.assertIsNotNone(f)
        self.assertGreater(f.avoidable_usd_month.raw(), 0)

    def test_the_two_cases_no_longer_report_the_same_figure(self):
        """Both returned $90 before, because reads were excluded from the sum."""
        beneficial = self._eff(100_000, 40_000)
        wasteful = self._eff(100_000, 0)
        self.assertIsNone(beneficial)
        self.assertIsNotNone(wasteful)


class TestFindingsUseTheInvoiceRate(unittest.TestCase):
    """A report that ties out against an invoice must not publish findings at
    list price. Spend reconciled at $0.25 while EFF-1 claimed $180 -- five
    times what the invoice supports."""

    def _reqs(self):
        return [Request(
            request_id=f"r{i}", sent_at=T0 + timedelta(seconds=60 * i),
            model="claude-opus-5",
            usage={"input_tokens": 0, "cache_creation_input_tokens": 100_000,
                   "cache_read_input_tokens": 0},
            segments=[], agent="a", ttl_requested="5m", target_id="anthropic/direct") for i in range(2)]

    def _eff(self, rate):
        a = analyze(TraceSet(requests=self._reqs(), tier=Tier.USAGE_ONLY),
                    effective_rate=rate, allow_unreconciled=True)
        return next(f for f in a.findings if f.code == "EFF-1").avoidable_usd_month.raw()

    def test_the_finding_scales_with_the_effective_rate(self):
        """claude-opus-5 lists at $5.00; an invoice rate of $1.00 is one fifth."""
        self.assertAlmostEqual(self._eff(None) / self._eff(1.0), 5.0, places=6)

    def test_spend_and_findings_agree_on_the_rate(self):
        a = analyze(TraceSet(requests=self._reqs(), tier=Tier.USAGE_ONLY),
                    effective_rate=1.0, allow_unreconciled=True)
        eff = next(f for f in a.findings if f.code == "EFF-1")
        # 200k written at 1.25x on a $1.00 rate = $0.25; the excess is 0.25x.
        self.assertAlmostEqual(a.spend["input_usd"].raw(), 0.25, places=6)
        self.assertGreater(eff.avoidable_usd_month.raw(), 0)


class TestIngestNormalisesTimestampsAndStatuses(unittest.TestCase):
    """Two shapes a real export produces, both of which used to remove data
    silently or take the report down."""

    def _file(self, rows):
        path = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False).name
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        return path

    def _row(self, rid, sent_at, status):
        return {"request_id": rid, "sent_at": sent_at, "model": "claude-opus-5",
                "usage": {"input_tokens": 10}, "status": status}

    def test_mixed_aware_and_naive_timestamps_do_not_crash(self):
        """An ISO string without an offset parses naive, and sorting the two
        forms together raised rather than producing a report."""
        p = self._file([self._row("a", "2026-07-29T09:00:00Z", 200),
                        self._row("b", "2026-07-29T10:00:00", 200)])
        ts = load_jsonl(p)
        self.assertEqual(len(ts.analysable), 2)
        self.assertTrue(all(r.sent_at.tzinfo is not None for r in ts.requests))
        os.unlink(p)

    def test_a_string_status_still_counts_as_success(self):
        """`analysable` compares against 200, so "200" made every successful
        request vanish from spend and findings."""
        p = self._file([self._row("a", T0.isoformat(), "200")])
        self.assertEqual(len(load_jsonl(p).analysable), 1)
        os.unlink(p)

    def test_an_unparseable_status_stays_excluded(self):
        p = self._file([self._row("a", T0.isoformat(), "nonsense")])
        ts = load_jsonl(p)
        self.assertEqual(len(ts.analysable), 0)
        self.assertEqual(ts.coverage["total"], 1)
        os.unlink(p)


class TestSessionSplitIsScopedToItsTenant(unittest.TestCase):
    """Session ids are not globally unique in a shared gateway export, so two
    tenants under the same string were reported as one session switching model
    -- a user-visible finding, and a remediation, for traffic that could never
    share a cache context."""

    def _reqs(self, same_tenant):
        rows = [("A", "claude-opus-5"), ("A", "claude-opus-5"),
                ("B", "claude-sonnet-5"), ("B", "claude-sonnet-5")]
        return [Request(
            request_id=f"r{i}", sent_at=T0 + timedelta(seconds=60 * i), model=model,
            usage={"input_tokens": 10}, segments=[], agent="a",
            tenant="A" if same_tenant else tenant, session="shared-id", target_id="anthropic/direct")
            for i, (tenant, model) in enumerate(rows)]

    def test_two_tenants_sharing_a_session_id_is_not_a_split(self):
        a = analyze(TraceSet(requests=self._reqs(False), tier=Tier.USAGE_ONLY),
                    allow_unreconciled=True)
        self.assertNotIn("SPL-1", [f.code for f in a.findings])

    def test_a_real_switch_within_one_tenant_is_still_caught(self):
        a = analyze(TraceSet(requests=self._reqs(True), tier=Tier.USAGE_ONLY),
                    allow_unreconciled=True)
        self.assertIn("SPL-1", [f.code for f in a.findings])


class TestUnknownMinimumsAreNotAssumedCacheable(unittest.TestCase):
    """Below the minimum a provider caches nothing and returns no error, and
    the registry refuses to guess because minimums are non-monotonic. Defaulting
    a missing one to zero let an unregistered model produce a confident verdict
    built on hits that may not exist -- and with an invoice rate supplied,
    pricing no longer failed either, so nothing else caught it."""

    def _reqs(self, model):
        return [Request(
            request_id=f"r{i}", sent_at=T0 + timedelta(seconds=60 * i), model=model,
            usage={}, agent="a", target_id="anthropic/direct",
            segments=[seg(0, "system", 8000, "sys", marked=True, ttl="5m"),
                      seg(1, "user", 100, f"t{i}")]) for i in range(6)]

    def test_an_unregistered_model_makes_the_bake_off_indeterminate(self):
        b = simulate.bake_off(self._reqs("claude-unregistered-9"), effective_rate=1.0)
        self.assertIsNone(b.delta_pct)
        self.assertIn("indeterminate", b.verdict)

    def test_those_requests_are_counted_not_silently_cached(self):
        res = simulate.simulate(self._reqs("claude-unregistered-9"), "allocator-lite")
        self.assertEqual(res.reads, 0)
        self.assertEqual(res.unmodelled_ttl, 6)

    def test_a_registered_model_still_produces_a_verdict(self):
        reqs = self._reqs("claude-opus-5")
        ts = instrumented(reqs)
        b = simulate.bake_off(ts.analysable, trace=ts)
        self.assertIsNotNone(b.delta_pct)


class TestTtlIsNotRecommendedWhenItLosesMoney(unittest.TestCase):
    """The rule subtracts the one-hour cold-write premium, then reported a
    candidate anyway whenever any in-band rewrite existed -- so the fix told the
    reader to set a one-hour TTL on a workload this same rule had just priced as
    a net loss."""

    def _reqs(self, offsets):
        return [Request(
            request_id=f"r{i}", sent_at=T0 + timedelta(seconds=o), model="claude-opus-5",
            usage={"input_tokens": 100, "cache_creation_input_tokens": 1_000_000,
                   "cache_read_input_tokens": 5000},
            segments=[seg(0, "system", 7000, "stable", marked=True, ttl="5m"),
                      seg(1, "user", 100, f"t{i}")],
            agent="a", session="s", tenant="t", target_id="anthropic/direct",
            ttl_requested="5m") for i, o in enumerate(offsets)]

    def _ttl(self, offsets):
        a = analyze(TraceSet(requests=self._reqs(offsets), tier=Tier.INSTRUMENTED),
                    allow_unreconciled=True)
        return next((f for f in a.findings if f.code == "TTL-1"), None)

    def test_a_net_negative_workload_produces_no_recommendation(self):
        """One in-band rewrite against four cold writes: 1.15x recovered
        against 4 x 0.75x paid."""
        self.assertIsNone(self._ttl([0, 900, 7200, 14400, 21600]))

    def test_a_profitable_workload_still_recommends(self):
        f = self._ttl([0, 900, 1800, 2700, 3400])
        self.assertIsNotNone(f)
        self.assertGreater(f.avoidable_usd_month.raw(), 0)

    def test_the_identity_caveat_only_appears_when_identity_is_missing(self):
        f = self._ttl([0, 900, 1800, 2700, 3400])
        self.assertNotIn("does not carry segment identity", f.detail)


class TestBothLoadersShareOneConstructor(unittest.TestCase):
    """The two loaders diverged on a guard four separate times -- tolerant
    timestamps, model normalisation, trusted-id shape, explicit status zero --
    each written for one and not the other, each silently removing data or
    crashing a run. They now build Requests through one function, and these
    assert the guards hold identically on both paths."""

    def _jsonl(self, **over):
        row = {"request_id": "a", "sent_at": T0.isoformat(), "model": "claude-opus-5",
               "usage": {"input_tokens": 10}}
        row.update(over)
        path = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False).name
        with open(path, "w") as f:
            f.write(json.dumps(row) + "\n")
        return path

    def _bodies(self, **over):
        row = {"request_id": "a", "sent_at": T0.isoformat(),
               "request": {"model": "claude-opus-5",
                           "messages": [{"role": "user", "content": "x"}]},
               "response": {"usage": {"input_tokens": 10, "output_tokens": 1,
                                      "cache_read_input_tokens": 0,
                                      "cache_creation_input_tokens": 0,
                                      "cache_creation": {}}}}
        row.update(over)
        path = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False).name
        with open(path, "w") as f:
            f.write(json.dumps(row) + "\n")
        return path

    def _both(self, **over):
        from cacheeconomics.adapters.bodies import load_bodies
        a = load_jsonl(self._jsonl(**over)).requests[0]
        b = load_bodies(self._bodies(**over), key=b"k").requests[0]
        return a, b

    def test_an_explicit_zero_status_is_a_failure_on_both(self):
        a, b = self._both(status=0)
        self.assertEqual((a.status, b.status), (0, 0))

    def test_a_malformed_timestamp_is_untimed_on_both(self):
        a, b = self._both(sent_at="not-a-date")
        self.assertIsNone(a.sent_at)
        self.assertIsNone(b.sent_at)

    def test_timestamps_are_utc_aware_on_both(self):
        a, b = self._both(sent_at="2026-07-29T09:00:00")
        self.assertIsNotNone(a.sent_at.tzinfo)
        self.assertIsNotNone(b.sent_at.tzinfo)

    def test_the_bedrock_prefix_is_stripped_on_both(self):
        from cacheeconomics.adapters.bodies import load_bodies
        a = load_jsonl(self._jsonl(model="anthropic.claude-haiku-4-5",
                                   target_id="amazon-bedrock/converse")).requests[0]
        p = self._bodies(target_id="amazon-bedrock/converse")
        with open(p) as f:
            row = json.loads(f.read())
        row["request"]["model"] = "anthropic.claude-haiku-4-5"
        with open(p, "w") as f:
            f.write(json.dumps(row) + "\n")
        b = load_bodies(p, key=b"k").requests[0]
        self.assertEqual(a.model, "claude-haiku-4-5")
        self.assertEqual(b.model, "claude-haiku-4-5")
        os.unlink(p)

    def test_the_row_target_wins_on_both(self):
        a, b = self._both(target_id="google-cloud/vertex")
        self.assertEqual((a.target_id, b.target_id),
                         ("google-cloud/vertex", "google-cloud/vertex"))

    def test_the_body_loader_still_prefers_the_body_model(self):
        """Consolidation must not lose what each caller knows that the other
        does not: the body is more authoritative than the exporter's row."""
        from cacheeconomics.adapters.bodies import load_bodies
        p = self._bodies(model="claude-sonnet-5")
        self.assertEqual(load_bodies(p, key=b"k").requests[0].model, "claude-opus-5")
        os.unlink(p)


class TestARefreshDoesNotHideAWarmEntry(unittest.TestCase):
    """A hit refreshes the lifetime. Overwriting the entry's visibility with the
    new response's time hid an already-readable prefix while the refresh was in
    flight, so a request arriving a second after a hit missed and wrote again --
    fabricating cache writes in the arm that feeds Gate 1's headline."""

    def _reqs(self, times):
        segs = [seg(0, "system", 50_000, "sys", "sys", marked=True, ttl="5m"),
                seg(1, "user", 100, "turn")]
        return [Request(request_id=f"r{i}", sent_at=T0 + timedelta(seconds=s),
                        model="claude-opus-5", usage={"input_tokens": 10},
                        segments=segs, agent="a", session="s", ttl_requested="5m", target_id="anthropic/direct")
                for i, s in enumerate(times)]

    def _reads(self, times, assume):
        return [u.cache_read for u in simulate.simulate(
            self._reqs(times), "as-shipped", assume=assume).usages]

    def test_a_request_just_after_a_hit_still_reads(self):
        self.assertEqual(self._reads([0, 10, 11], simulate.PESSIMISTIC), [0, 50_000, 50_000])

    def test_the_same_holds_under_neutral_assumptions(self):
        self.assertEqual(self._reads([0, 10, 11], simulate.NEUTRAL), [0, 50_000, 50_000])

    def test_a_genuinely_cold_entry_is_still_invisible_to_a_concurrent_request(self):
        """The write-visibility latency this replays under is the whole reason
        cold fan-out costs money; the fix must not erase it."""
        self.assertEqual(self._reads([0, 1], simulate.PESSIMISTIC), [0, 0])

    def test_an_expired_entry_still_misses(self):
        self.assertEqual(self._reads([0, 400], simulate.PESSIMISTIC), [0, 0])

    def test_a_long_warm_run_reads_every_time_after_the_first(self):
        reads = self._reads([0] + [i * 30 for i in range(1, 8)], simulate.PESSIMISTIC)
        self.assertEqual(reads[0], 0)
        self.assertTrue(all(r == 50_000 for r in reads[1:]), reads)


class TestEveryUsageIsPairedWithItsRequest(unittest.TestCase):
    """`spend()` zips `usages` against `priced`, so a branch that records one
    without the other shifts every pair after it. Each arm's total then prices
    real tokens against some other request's model, and the last request's usage
    falls off the end.

    The unmodelled-target branch exists to survive schema drift in a single row.
    Corrupting the comparison while surviving it is not surviving it."""

    def _mixed(self):
        def r(i, target, model, tokens):
            return Request(request_id=f"r{i}",
                           sent_at=datetime(2026, 7, 29, 9, tzinfo=timezone.utc)
                           + timedelta(seconds=60 * i),
                           model=model, usage={"input_tokens": 10},
                           target_id=target, session="s",
                           segments=[Segment(id=f"s{i}", role="system", tokens=tokens,
                                             index=0, cache_marked=True, ttl="5m")])
        return [r(0, "anthropic/direct", "claude-opus-5", 9_000),
                r(1, "mystery/surface", "claude-opus-5", 250_000),
                r(2, "anthropic/direct", "claude-haiku-4-5", 9_000),
                r(3, "anthropic/direct", "claude-opus-5", 9_000)]

    def test_an_unmodellable_row_still_records_both_sides(self):
        s = simulate.simulate(self._mixed(), "litellm-auto")
        self.assertEqual(s.unmodelled_target, 1)
        self.assertEqual(len(s.usages), len(s.priced))

    def test_each_usage_is_priced_against_its_own_request(self):
        s = simulate.simulate(self._mixed(), "litellm-auto")
        pairs = {q.request_id: u.total for u, q in zip(s.usages, s.priced)}
        self.assertEqual(pairs["r1"], 250_000)
        self.assertEqual(pairs["r2"], 9_000)

    def test_no_request_falls_off_the_end(self):
        s = simulate.simulate(self._mixed(), "litellm-auto")
        self.assertIn("r3", {q.request_id for q in s.priced})

    def test_every_arm_keeps_the_pairing(self):
        for policy in ("as-shipped", "litellm-auto", "allocator-lite", "relocation-lite"):
            s = simulate.simulate(self._mixed(), policy)
            self.assertEqual(len(s.usages), len(s.priced), f"{policy} lost the pairing")


class TestPessimismDoesNotPunishEarlyMarkers(unittest.TestCase):
    """The lookback was measured from the end of the request, so a marker's
    survival depended on how much conversation followed it. A 40,000-token
    system prefix marked at the top replayed as zero cache reads once forty
    blocks sat behind it -- on exactly the long sessions caching exists for,
    feeding exactly the arm Gate 1 reads."""

    def _reqs(self, tail, n=10):
        def one(i):
            segs = [Segment(id="sys", role="system", tokens=40_000, index=0,
                            cache_marked=True, ttl="5m")]
            segs += [Segment(id=f"m{j}", role="user", tokens=200, index=1 + j)
                     for j in range(tail)]
            return Request(request_id=f"r{i}",
                           sent_at=datetime(2026, 7, 29, 9, tzinfo=timezone.utc)
                           + timedelta(seconds=60 * i),
                           model="claude-opus-5", usage={"input_tokens": 10},
                           segments=segs, session="s", target_id="anthropic/direct")
        return [one(i) for i in range(n)]

    def _reads(self, reqs, assume):
        return sum(u.cache_read for u in
                   simulate.simulate(reqs, "as-shipped", assume=assume).usages)

    def test_a_long_tail_does_not_invalidate_an_early_marker(self):
        reqs = self._reqs(tail=40)
        self.assertEqual(self._reads(reqs, simulate.PESSIMISTIC),
                         self._reads(reqs, simulate.NEUTRAL))

    def test_nor_does_a_very_long_one(self):
        reqs = self._reqs(tail=200)
        self.assertGreater(self._reads(reqs, simulate.PESSIMISTIC), 0)

    def test_the_window_still_constrains_widely_spaced_markers(self):
        """The fix must not quietly delete the assumption it corrects."""
        def one(i, gap):
            segs = [Segment(id="a", role="system", tokens=9_000, index=0,
                            cache_marked=True, ttl="5m")]
            segs += [Segment(id=f"f{j}", role="system", tokens=100, index=1 + j)
                     for j in range(gap)]
            segs += [Segment(id="b", role="system", tokens=9_000, index=1 + gap,
                             cache_marked=True, ttl="5m"),
                     Segment(id=f"t{i}", role="user", tokens=50, index=2 + gap)]
            return Request(request_id=f"r{i}",
                           sent_at=datetime(2026, 7, 29, 9, tzinfo=timezone.utc)
                           + timedelta(seconds=60 * i),
                           model="claude-opus-5", usage={"input_tokens": 10},
                           segments=segs, session="s", target_id="anthropic/direct")
        far = [one(i, 40) for i in range(10)]
        self.assertLess(self._reads(far, simulate.PESSIMISTIC),
                        self._reads(far, simulate.NEUTRAL))


class TestTheOmittedDenominatorIsPossible(unittest.TestCase):
    """The sentence explaining why a gate cannot be answered is the last place
    an impossible number belongs. Per-arm counts were summed, so a request that
    was unpriceable under one arm and unmodelled under another was counted
    twice: "6 of 3 requests contributed nothing"."""

    def _reqs(self, n=3):
        segs = [Segment(id="a", role="system", tokens=9_000, index=0,
                        cache_marked=True, ttl="30m"),
                Segment(id="b", role="user", tokens=400, index=1)]
        return [Request(request_id=f"r{i}",
                        sent_at=datetime(2026, 7, 29, 9, tzinfo=timezone.utc)
                        + timedelta(seconds=60 * i),
                        model="gpt-5.6", usage={"input_tokens": 10},
                        segments=segs, target_id="openai/direct", session="s")
                for i in range(n)]

    def test_it_never_omits_more_requests_than_exist(self):
        import re
        reqs = self._reqs()
        v = simulate.bake_off(reqs).verdict
        omitted, total = map(int, re.search(r"(\d+) of (\d+) requests", v).groups())
        self.assertLessEqual(omitted, total)
        self.assertEqual(total, len(reqs))

    def test_a_request_omitted_twice_is_still_one_request(self):
        v = simulate.bake_off(self._reqs()).verdict
        self.assertIn("3 of 3 requests", v)

    def test_the_per_reason_counts_are_still_reported(self):
        v = simulate.bake_off(self._reqs()).verdict
        self.assertIn("with an unmodelled lifetime", v)
        self.assertIn("without pricing", v)

    def test_the_overlap_is_stated_rather_than_left_to_arithmetic(self):
        """Reporting 3 and 3 under a denominator of 3 looks like an error unless
        the report says the reasons overlap."""
        v = simulate.bake_off(self._reqs()).verdict
        self.assertIn("do not sum", v)

    def test_the_verdict_is_still_indeterminate(self):
        self.assertTrue(simulate.bake_off(self._reqs()).verdict.startswith("indeterminate"))




class TestTheBakeOffRefusesUnvalidatedSizes(unittest.TestCase):
    """The one dollar-producing subsystem here that never imports `money.Figure`.

    The analyzer routes every figure through it so a number carries its own
    release state and the safety property does not depend on remembering the
    gate. This module reports arm spend as a plain float, computed from segment
    sizes nothing had checked against the tokens the provider billed.

    Measured before fixing: a 1,000x size error turned $0.125 of arm spend into
    $125.00. It also left `delta_pct` at exactly 20.0% both times -- every arm
    scales together, so the *percentage* Gate 1 reads is genuinely robust and
    guarding it would have been theatre. Only the absolutes needed a gate.
    """

    HMAC = "hmac:"

    def _trace(self, seg_tokens, n=20):
        import json
        import tempfile
        from cacheeconomics.trace import load_jsonl
        rows = [{"request_id": f"r{i}", "sent_at": f"2026-07-29T09:{i:02d}:00Z",
                 "model": "claude-opus-5", "agent": "main", "session": "s1",
                 # Stated. These tests are about size agreement and the
                 # publication gate; the surface is a precondition they
                 # used to get from the loader defaulting an unnamed row
                 # to first-party.
                 "target_id": "anthropic/direct",
                 # Stated alongside the surface, and for the same reason: these
                 # tests are about size *agreement*, so counted sizes are a
                 # precondition rather than the thing under test. A row that does
                 # not say counts as estimated, and an estimated split now blocks
                 # the comparison as well as the dollars.
                 "tokens_counted": True,
                 "usage": {"input_tokens": 0, "cache_read_input_tokens": 0,
                           "cache_creation_input_tokens": 1_000,
                           "cache_creation": {"ephemeral_5m_input_tokens": 1_000,
                                              "ephemeral_1h_input_tokens": 0}},
                 "segments": [
                     {"id": self.HMAC + ("a" * 63) + str(i % 10), "role": "system",
                      "tokens": seg_tokens, "index": 0, "cache_marked": False,
                      "ttl": None},
                     {"id": self.HMAC + "b" * 64, "role": "system",
                      "tokens": seg_tokens, "index": 1, "cache_marked": True,
                      "ttl": "5m"}]}
                for i in range(n)]
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        try:
            with open(path, "w") as f:
                f.write("\n".join(json.dumps(r) for r in rows))
            return load_jsonl(path)
        finally:
            os.unlink(path)

    def _bake(self, seg_tokens, n=20):
        ts = self._trace(seg_tokens, n=n)
        return simulate.bake_off(ts.analysable, trace=ts)

    def test_agreeing_sizes_produce_a_verdict(self):
        """The control. Refusing everything would also pass a bad test."""
        b = self._bake(500)
        self.assertIsNotNone(b.delta_pct)

    def test_mis_scaled_sizes_produce_no_percentage(self):
        b = self._bake(500_000)
        self.assertIsNone(b.delta_pct)
        self.assertTrue(b.verdict.startswith("indeterminate"))

    def test_the_verdict_says_the_percentage_survives_but_the_absolutes_do_not(self):
        b = self._bake(500_000)
        self.assertIn("differ from", b.verdict)
        self.assertIn("does not cancel out of the absolutes", b.verdict)
        # The observed magnitude, not just that something was wrong. An operator
        # reading "worst: 1000.00x" knows to look at units; "sizes disagree"
        # sends them nowhere.
        self.assertRegex(b.verdict, r"worst: \d+\.\d\dx")

    def test_it_does_not_mask_the_omission_verdict(self):
        """A short-circuit on sizes hid the omission reason on a trace where
        both were true. One guard concealing another is its own defect."""
        segs = [Segment(id="a", role="system", tokens=9_000, index=0,
                        cache_marked=True, ttl="30m"),
                Segment(id="b", role="user", tokens=400, index=1)]
        reqs = [Request(request_id=f"r{i}",
                        sent_at=datetime(2026, 7, 29, 9, tzinfo=timezone.utc)
                        + timedelta(seconds=60 * i),
                        model="gpt-5.6", usage={"input_tokens": 10},
                        segments=segs, target_id="openai/direct", session="s")
                for i in range(3)]
        v = simulate.bake_off(reqs).verdict
        self.assertIn("contributed nothing", v)
        self.assertIn("do not sum", v)

    def test_the_reconciliation_is_the_loader_s_own(self):
        """Two copies of one question drift, and this one did: the bake-off's
        copy kept the coarse factor-of-two while the question it was answering
        had become "may this dollar amount be printed". It now calls the loader's
        helpers, so there is nothing left to drift.
        """
        import inspect

        from cacheeconomics import simulate as s
        from cacheeconomics import trace

        self.assertIs(s.segment_sum_ratio, trace.segment_sum_ratio)
        self.assertIs(s.sums_publishable, trace.sums_publishable)
        self.assertIs(s.PUBLISH_TOLERANCE, trace.PUBLISH_TOLERANCE)
        # And it must not have quietly reintroduced the loose one.
        src = inspect.getsource(s.bake_off)
        self.assertNotIn("TOKEN_SUM_FACTOR", src,
                         "the release gate is back on the coarse factor, which "
                         "admits a doubling")

    def test_the_two_thresholds_are_not_the_same_question(self):
        """A ratio of 2.0 is plausible structure and an unpublishable price.

        Both gates existing is the point: collapsing them is what let a 100%
        size error clear the check that guards money.
        """
        from cacheeconomics.trace import sums_publishable, sums_within_factor
        self.assertTrue(sums_within_factor(2.0))
        self.assertFalse(sums_publishable(2.0))
        self.assertTrue(sums_within_factor(0.51))
        self.assertFalse(sums_publishable(0.51))
        # Zero is a disagreement, not an absence of one. Both must reject it.
        self.assertFalse(sums_within_factor(0.0))
        self.assertFalse(sums_publishable(0.0))
        # None is genuinely no opinion, and must not be read as failure.
        self.assertTrue(sums_within_factor(None))
        self.assertTrue(sums_publishable(None))


class TestEverySkipCounterRecordsItsRequest(unittest.TestCase):
    """`omitted == 0` is read as "nothing was left out", and the size-only
    verdict relies on it to know the five skip counters are all zero.

    That holds only because each of the five increment sites also records the
    request id in `res.omitted` -- an invariant spread across five branches in
    two methods, not a property of any one of them. A sixth branch that counts a
    skip without recording the id would make a partial comparison report a full
    denominator, which is the failure this whole verdict exists to prevent.

    Audited structurally rather than by example, because the defect only appears
    with the branch that does not exist yet.
    """

    COUNTERS = ("unstructured", "untimed", "unpriceable",
                "unmodelled_ttl", "unmodelled_target")

    def test_each_increment_is_paired_with_an_omitted_entry(self):
        import ast
        import pathlib

        src = pathlib.Path(simulate.__file__).read_text()
        tree = ast.parse(src)

        def counter_name(node):
            """`res.unstructured += 1` / `self.unpriceable += 1` -> the field."""
            if not isinstance(node, ast.AugAssign):
                return None
            t = node.target
            if isinstance(t, ast.Attribute) and t.attr in self.COUNTERS:
                return t.attr
            return None

        # Every statement list that contains an increment must also touch
        # `omitted` somewhere in the same list -- the branch body, not the file.
        found = 0
        for parent in ast.walk(tree):
            for field in ("body", "orelse", "finalbody"):
                block = getattr(parent, field, None)
                if not isinstance(block, list):
                    continue
                names = [name for stmt in block for node in ast.walk(stmt)
                         if (name := counter_name(node))]
                if not names:
                    continue
                found += len(names)
                touches_omitted = any(
                    isinstance(n, ast.Attribute) and n.attr == "omitted"
                    for stmt in block for n in ast.walk(stmt))
                self.assertTrue(
                    touches_omitted,
                    f"{names} incremented in a block that never records the "
                    f"request in `omitted`; `omitted == 0` would then claim "
                    f"nothing was left out. Line "
                    f"{getattr(block[0], 'lineno', '?')} of simulate.py.")

        # The audit is worthless if it matched nothing.
        self.assertGreaterEqual(
            found, len(self.COUNTERS),
            f"audit found only {found} increments across {len(self.COUNTERS)} "
            f"counters; the AST shape it looks for has drifted")


class TestArmSpendIsATypedFigure(unittest.TestCase):
    """The size gate first shipped as a verdict string while arm spend stayed a
    bare float, so the guard depended on a reader noticing a sentence. These
    assert the structural version: the number carries its own release state, the
    way every figure the analyzer produces already did.

    Each of these fails against a float-returning `spend()`, which is the point
    -- a test that passes either way pins nothing.
    """

    HMAC = "hmac:"

    # Every case here now passes `allow_unreconciled=True`. Releasing arm spend
    # gained a second condition -- an invoice, or an explicit draft flag -- after
    # a review pointed out that `cacheeconomics bakeoff` printed $17.14 against
    # $6.18 with no invoice anywhere in the command, which is the exact thing the
    # README's "No invoice, no dollars" forbids. These tests are about the *size*
    # half of the gate, so they opt into the draft to isolate it.
    def _trace(self, seg_tokens, n=20, unstructured=(), **kw):
        rows = []
        for i in range(n):
            segs = [] if i in unstructured else [
                {"id": self.HMAC + ("a" * 63) + str(i % 10), "role": "system",
                 "tokens": seg_tokens, "index": 0, "cache_marked": False,
                 "ttl": None},
                {"id": self.HMAC + "b" * 64, "role": "system",
                 "tokens": seg_tokens, "index": 1, "cache_marked": True,
                 "ttl": "5m"}]
            rows.append({"request_id": f"r{i}",
                         "sent_at": f"2026-07-29T09:{i:02d}:00Z",
                         "model": "claude-opus-5", "agent": "main",
                         "session": "s1",
                         # Stated, because these tests are about the size gate
                         # and need a priceable surface as a precondition. It
                         # used to arrive by way of the loader defaulting an
                         # unnamed row to first-party, which is the thing that
                         # default now refuses to do.
                         "target_id": "anthropic/direct",
                         # Stated for the same reason `target_id` is. Releasing
                         # arm spend now also asks whether the segment sizes it
                         # is priced from were counted rather than split by byte
                         # share, and a row that does not say counts as
                         # estimated. These tests are about the *size* gate, so
                         # they state the precondition instead of inheriting it.
                         "tokens_counted": True,
                         "usage": {"input_tokens": 0,
                                   "cache_read_input_tokens": 0,
                                   "cache_creation_input_tokens": 1_000,
                                   "cache_creation": {
                                       "ephemeral_5m_input_tokens": 1_000,
                                       "ephemeral_1h_input_tokens": 0}},
                         "segments": segs})
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        try:
            with open(path, "w") as f:
                f.write("\n".join(json.dumps(r) for r in rows))
            kw.setdefault("allow_unreconciled", True)
            ts = load_jsonl(path)
            # The trace, not just the rows the arms replay. Alignment, coverage
            # and counted-versus-estimated sizes live here and nowhere on a
            # `Request`, so a bake-off handed only `analysable` cannot ask them.
            return simulate.bake_off(ts.analysable, trace=ts, **kw)
        finally:
            os.unlink(path)

    def _every_figure(self, b):
        """Every arm at both assumption ends. `run()` is called twice."""
        return [(end, p, a["spend"])
                for end, d in (("pessimistic", b.arms), ("optimistic", b.optimistic))
                for p, a in d.items()]

    def test_a_bare_simresult_withholds_its_own_spend(self):
        """Fail-closed by default. A caller holding a SimResult has established
        nothing about whether its segment sizes reconcile, so the number is not
        publishable until `bake_off` says so."""
        res = simulate.simulate([], "as-shipped")
        fig = res.spend()
        self.assertIsInstance(fig, money.Figure)
        self.assertFalse(fig.released, "spend() must start withheld")
        self.assertIn("reconcile", fig.withheld_because)

    def test_a_withheld_arm_cannot_be_used_as_a_number(self):
        """The safety property itself. Before this, a renderer that forgot the
        gate got a plausible float; now it gets a loud error."""
        b = self._trace(500_000)
        with self.assertRaises(money.WithheldFigure):
            float(b.arms["litellm-auto"]["spend"])

    def test_every_arm_is_modeled_not_measured(self):
        """Including as-shipped, which replays the lifetimes the trace really
        carried -- but still against a cache whose hits this module decides."""
        for end, p, fig in self._every_figure(self._trace(500)):
            self.assertEqual(fig.basis, money.MODELED, f"{end}/{p}")

    def test_a_clean_trace_releases_every_arm_at_both_ends(self):
        """With the draft flag: this asserts the size half of the gate."""
        for end, p, fig in self._every_figure(self._trace(500)):
            self.assertTrue(fig.released, f"{end}/{p} should be released")
            self.assertEqual(fig.withheld_because, "")

    def test_misscaled_sizes_withhold_every_arm_at_both_ends(self):
        """The twin-path assertion. Release happens in one place precisely so
        that eight figures in two dicts cannot disagree; this is what would
        catch a future edit that released only `pess`."""
        for end, p, fig in self._every_figure(self._trace(500_000)):
            self.assertFalse(fig.released, f"{end}/{p} should be withheld")
            self.assertIn("unknown scale", fig.withheld_because)

    def test_omitted_requests_withhold_for_the_subtotal_reason(self):
        """Two conditions, two reasons, not conflated. A subtotal is wrong for a
        different cause than an unknown scale, and telling a client the sizes
        disagree when the real problem is a dropped request reads as a defect in
        the tool."""
        for end, p, fig in self._every_figure(self._trace(500, unstructured=(3,))):
            self.assertFalse(fig.released, f"{end}/{p}")
            self.assertIn("subtotal", fig.withheld_because)
            self.assertNotIn("unknown scale", fig.withheld_because)

    def test_an_indeterminate_verdict_always_withholds(self):
        """One direction, not both, and the direction matters.

        These used to be asserted as equal, which was true while size agreement
        was the only release condition. It is not any more: a run with no invoice
        has perfectly good ratios and withheld dollars, so `released` can be False
        while `delta_pct` is a real number. What must still hold is the
        implication the render depends on -- an indeterminate verdict never
        accompanies a released figure, so a bare "[withheld]" always has its
        reason printed beside it."""
        for tokens, unstructured in ((500, ()), (500_000, ()), (500, (3,))):
            b = self._trace(tokens, unstructured=unstructured)
            if b.delta_pct is None:
                self.assertFalse(b.arms["litellm-auto"]["spend"].released)
                self.assertIn("indeterminate", b.verdict)

    def test_dollars_and_percentages_are_gated_separately(self):
        """The conflation the invoice rule exposed. Sizes agree and nothing was
        omitted, so the comparison is sound; no invoice, so the money is not
        publishable. Both facts are true at once and the report says so."""
        b = self._trace(500, allow_unreconciled=False)
        self.assertFalse(b.arms["litellm-auto"]["spend"].released)
        self.assertIsNotNone(b.delta_pct)
        self.assertIn("vs litellm-auto", str(b), "the percentage still prints")
        self.assertIn("no invoice was supplied", str(b))

    def test_the_render_shows_no_digits_for_a_withheld_arm(self):
        b = self._trace(500_000)
        arm_lines = [l for l in str(b).splitlines()
                     if any(l.strip().startswith(p) for p in simulate.ARMS)]
        self.assertEqual(len(arm_lines), len(simulate.ARMS))
        for line in arm_lines:
            self.assertIn("[withheld]", line)
            self.assertNotIn("$", line, f"a dollar amount survived: {line}")

    def test_the_reason_is_stated_once_beside_the_withheld_arms(self):
        text = str(self._trace(500_000))
        self.assertEqual(text.count("per-arm spend withheld:"), 1)
        self.assertIn("unknown scale", text)

    def test_the_inline_percentage_is_gated_with_the_absolutes(self):
        """`delta_pct` is None on both indeterminate paths, but this column was
        computed from raw floats and printed anyway -- so a block could read
        "+20.0% vs litellm-auto" three lines above "placement: indeterminate".

        A uniform size error does cancel, which is why the verdict says so.
        `misscaled` does not establish uniformity, and nothing survives a subset.
        """
        for tokens, unstructured in ((500_000, ()), (500, (3,))):
            text = str(self._trace(tokens, unstructured=unstructured))
            self.assertNotIn("vs litellm-auto", text,
                             "a subset/unknown-scale percentage was published")
        self.assertIn("vs litellm-auto", str(self._trace(500)),
                      "the released case must still show the comparison")


class TestSizesInsideTheOldFactorStillWithhold(unittest.TestCase):
    """An adversarial review reproduced this: the release gate used the loader's
    coarse factor of two, so segment sums anywhere in [0.5x, 2.0x] of the billed
    total released arm spend and printed a confident verdict beside it.

    Each case here published a wrong dollar amount before the fix. The ratios are
    the boundary ones, because a gate is only ever wrong at its edges.
    """

    HMAC = "hmac:"
    BILLED = 1_000

    def _bake(self, seg_total, n=20, **kw):
        rows = []
        for i in range(n):
            half = seg_total // 2
            rows.append({
                "request_id": f"r{i}", "sent_at": f"2026-07-29T09:{i:02d}:00Z",
                "model": "claude-opus-5", "agent": "main", "session": "s1",
                # Stated. This class is about the *size* release gate and needs
                # a priceable surface to have anything to release; it used to
                # get one from the loader answering an unnamed row with
                # first-party.
                "target_id": "anthropic/direct",
                # As above: this class isolates the size gate, so the
                # counted-sizes precondition is stated rather than assumed.
                "tokens_counted": True,
                "usage": {"input_tokens": 0, "cache_read_input_tokens": 0,
                          "cache_creation_input_tokens": self.BILLED,
                          "cache_creation": {
                              "ephemeral_5m_input_tokens": self.BILLED,
                              "ephemeral_1h_input_tokens": 0}},
                "segments": [
                    {"id": self.HMAC + ("a" * 63) + str(i % 10), "role": "system",
                     "tokens": half, "index": 0, "cache_marked": False,
                     "ttl": None},
                    {"id": self.HMAC + "b" * 64, "role": "system",
                     "tokens": seg_total - half, "index": 1,
                     "cache_marked": True, "ttl": "5m"}]})
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        try:
            with open(path, "w") as f:
                f.write("\n".join(json.dumps(r) for r in rows))
            ts = load_jsonl(path)
            return ts, simulate.bake_off(ts.analysable, trace=ts, **kw)
        finally:
            os.unlink(path)

    def test_half_the_billed_total_is_not_publishable(self):
        """0.51x cleared the old gate and printed spend ~49% too low."""
        _ts, b = self._bake(510)
        self.assertFalse(b.arms["litellm-auto"]["spend"].released)
        self.assertIsNone(b.delta_pct)

    def test_exactly_double_is_not_publishable(self):
        """The old check was `ratio > factor`, so exactly 2.0 passed it."""
        _ts, b = self._bake(2 * self.BILLED)
        self.assertFalse(b.arms["litellm-auto"]["spend"].released)

    def test_a_zero_segment_total_against_billed_tokens_withholds(self):
        """Fail-open on missing input. `if not billed or not total: continue`
        skipped this row entirely, so it counted as neither disagreement nor
        omission and the arm printed $0.0000 with the gate reporting a pass.

        The same line existed in the loader, so `token_sums_reconciled` said
        True as well -- one defect written twice, which is why the check now
        lives in one place."""
        ts, b = self._bake(0)
        self.assertFalse(b.arms["litellm-auto"]["spend"].released)
        self.assertFalse(ts.token_sums_reconciled)
        self.assertFalse(ts.token_sums_publishable)

    def test_exact_agreement_still_publishes(self):
        """The gate has to let honest data through, or it is just an off switch.
        The recorder scales segment sizes to the measured total, so this is what
        a well-formed instrumented export looks like.

        `allow_unreconciled` because releasing also needs an invoice now; this
        test is about the size condition."""
        ts, b = self._bake(self.BILLED, allow_unreconciled=True)
        self.assertTrue(b.arms["litellm-auto"]["spend"].released)
        self.assertTrue(ts.token_sums_publishable)
        self.assertIsNotNone(b.delta_pct)

    def test_the_analyzer_uses_the_same_gate_for_structural_money(self):
        """The twin. Structural figures are computed from segment sizes too, so
        fixing only the bake-off would have left `avoidable_usd_month` publishing
        on sizes off by up to 2x -- the same defect one module along."""
        ts, _b = self._bake(1_500)          # 1.5x: clears the factor, not the gate
        self.assertTrue(ts.token_sums_reconciled)
        self.assertFalse(ts.token_sums_publishable)
        a = analyze(ts, allow_unreconciled=True)
        for f in a.findings:
            if f.structural and f.avoidable_usd_month is not None:
                self.assertFalse(
                    f.avoidable_usd_month.released,
                    f"{f.code} published structural money on 1.5x sizes")


class TestEveryRawCallSiteIsJustified(unittest.TestCase):
    """`raw()` is the documented escape hatch -- "deliberately ugly, deliberately
    greppable ... the one place a reviewer needs to look to audit whether a guard
    was bypassed".

    That only holds while the call sites stay few and enumerated. `Figure`
    deliberately defines no comparison operators for the same reason: every
    pre-release numeric use has to spell `raw()` and show up here.
    """

    # Function -> why reaching past the gate is correct there.
    JUSTIFIED = {
        "delta": "percentages are scale-invariant; arithmetic before release",
        "__str__": "renders the percentage, which is releasable when the "
                   "absolutes are not; the absolutes go through __format__",
        "bake_off_by_agent": "ranks groups by size, which must work before "
                             "anyone decides the sizes may be printed",
        "bake_off": "compares the as-shipped arm against the invoice, which is "
                    "the arithmetic that *decides* release and therefore has to "
                    "happen before it",
    }

    def test_no_unjustified_function_reaches_past_the_gate(self):
        import ast
        import pathlib

        src = pathlib.Path(simulate.__file__).read_text()
        tree = ast.parse(src)

        # Attributed to the *innermost* enclosing function. `delta` is nested
        # inside `bake_off`, and walking from the outer one blamed both -- which
        # would have let a genuine new call site in `bake_off` hide behind
        # `delta`'s justification.
        # Lambdas are transparent: a `key=lambda ...` is not a unit anyone
        # reviews, so its calls are attributed to the named function around it.
        found = {}
        FN = (ast.FunctionDef, ast.AsyncFunctionDef)

        def visit(node, enclosing):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "raw"):
                found.setdefault(enclosing, []).append(node.lineno)
            for child in ast.iter_child_nodes(node):
                inner = child.name if isinstance(child, FN) else enclosing
                visit(child, inner)

        visit(tree, "<module>")

        unjustified = {k: v for k, v in found.items() if k not in self.JUSTIFIED}
        self.assertFalse(
            unjustified,
            f"raw() reached past the release gate in {unjustified}. Either use "
            f"the Figure directly, or add the function to JUSTIFIED with the "
            f"reason a reviewer needs.")
        # And the allowlist must not outlive its entries.
        self.assertEqual(set(found), set(self.JUSTIFIED),
                         "JUSTIFIED names a function that no longer calls raw()")


class TestBakeOffNeedsAnInvoiceToo(unittest.TestCase):
    """The README's first promise is "No invoice, no dollars". The analyzer
    enforced it and the bake-off did not: `cacheeconomics bakeoff` printed
    $17.14 against $6.18 with no invoice anywhere in the command, and the command
    did not even accept one. Modelled list-price figures wearing the authority of
    reconciled ones.

    The as-shipped arm is what an invoice can settle -- it replays the lifetimes
    the trace actually carried. The counterfactual arms come from the same trace
    at the same rates, so they inherit its credibility once it has earned it.
    """

    def _ts(self):
        from cacheeconomics.trace import load_jsonl
        return load_jsonl(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "fixtures", "demo-traces.jsonl"))

    def _bake(self, **kw):
        """The whole trace, not only the rows the arms replay.

        This class isolates the *invoice* half of the release rule, and the
        structural half now has to be satisfied for the invoice to be the thing
        under test. The demo fixture is instrumented with counted segment sizes
        and full coverage, so passing it establishes the precondition rather
        than waiving it.
        """
        ts = self._ts()
        return simulate.bake_off(ts.analysable, trace=ts, **kw)

    def test_no_invoice_withholds_the_absolutes(self):
        b = self._bake()
        self.assertFalse(b.arms["as-shipped"]["spend"].released)
        self.assertIn("no invoice was supplied",
                      b.arms["as-shipped"]["spend"].withheld_because)

    def test_the_percentage_survives_that(self):
        """It is scale-invariant and it is what Gate 1 reads. Withholding it
        because the dollars are unreconciled would be theatre."""
        b = self._bake()
        self.assertIsNotNone(b.delta_pct)
        self.assertIn("vs litellm-auto", str(b))

    def test_a_reconciling_invoice_releases(self):
        b = self._bake(invoice_usd=17.45)
        self.assertTrue(b.arms["as-shipped"]["spend"].released)

    def test_an_invoice_that_does_not_reconcile_withholds(self):
        b = self._bake(invoice_usd=99.00)
        self.assertFalse(b.arms["as-shipped"]["spend"].released)
        self.assertIn("does not reconcile",
                      b.arms["as-shipped"]["spend"].withheld_because)

    def test_the_draft_flag_is_the_deliberate_override(self):
        b = self._bake(allow_unreconciled=True)
        self.assertTrue(b.arms["as-shipped"]["spend"].released)

    def test_a_nonsense_invoice_does_not_release(self):
        for bad in (0.0, -17.45):
            with self.subTest(invoice=bad):
                b = self._bake(invoice_usd=bad)
                self.assertFalse(b.arms["as-shipped"]["spend"].released)

    def test_the_same_trace_without_the_trace_argument_withholds(self):
        """The reconciling invoice from two tests up, minus the trace.

        Nothing about the workload changed -- only whether the caller stated the
        evidence. An unstated gate is not a passed one, so the figures that
        released above are withheld here.
        """
        ts = self._ts()
        b = simulate.bake_off(ts.analysable, invoice_usd=17.45)
        self.assertFalse(b.arms["as-shipped"]["spend"].released)
        self.assertIn("no trace was supplied",
                      b.arms["as-shipped"]["spend"].withheld_because)

    def _two_agents(self, n=6):
        """Two agents with identical, clean traffic, so the whole workload
        prices at almost exactly twice either group."""
        reqs = []
        for agent in ("alpha", "beta"):
            for i in range(n):
                reqs.append(Request(
                    request_id=f"{agent}{i}",
                    sent_at=T0 + timedelta(seconds=60 * i),
                    model="claude-opus-5", agent=agent, tenant="t",
                    session=f"s{agent}", target_id="anthropic/direct",
                    ttl_requested="5m",
                    usage={"input_tokens": 300,
                           "cache_creation_input_tokens": 30_000,
                           "cache_read_input_tokens": 0},
                    segments=[seg(0, "system", 300, f"h{agent}{i}"),
                              seg(1, "system", 30_000, "body", marked=True,
                                  ttl="5m")]))
        return TraceSet(requests=reqs, tier=Tier.INSTRUMENTED)

    def test_the_per_agent_run_does_not_reconcile_against_the_whole_invoice(self):
        """The bill covers the workload, not one agent's slice of it.

        Behavioural, because the source check this replaced asserted that
        `invoice_usd=invoice_usd` does not appear in `bake_off_by_agent` -- which
        pins an implementation rather than the property, and went red on a
        correct one. The invoice is passed there now, to a single whole-workload
        run whose reconciliation every group inherits. What must never happen is
        a *group* releasing because its own slice matched the full bill.

        So the invoice here is one group's spend. Each group's share matches it
        within 5%; the workload is twice it and does not. Nothing may release.
        """
        ts = self._two_agents()
        alpha = [r for r in ts.analysable if r.agent == "alpha"]
        scoped = replace(ts, requests=alpha)
        one_group = simulate.bake_off(
            scoped.analysable, trace=scoped,
            allow_unreconciled=True).arms["as-shipped"]["spend"].raw()
        whole = simulate.bake_off(
            ts.analysable, trace=ts,
            allow_unreconciled=True).arms["as-shipped"]["spend"].raw()
        self.assertAlmostEqual(whole / one_group, 2.0, places=2,
                               msg="the fixture must make the slice and the "
                                   "workload disagree, or this proves nothing")
        out = simulate.bake_off_by_agent(ts.analysable, trace=ts,
                                         invoice_usd=one_group)
        self.assertTrue(out, "no groups came back, so nothing was checked")
        for b in out:
            self.assertFalse(
                b.arms["as-shipped"]["spend"].released,
                f"{b.group} released because its own slice matched the whole "
                f"workload's invoice")
            self.assertIsNone(b.delta_pct, b.group)

    def test_a_reconciling_invoice_frees_the_comparison_but_not_the_dollars(self):
        """The other direction, and the asymmetry is deliberate.

        A *failed* reconciliation is evidence about the capture and taints every
        slice cut from it, so it blocks the groups outright. A *passed* one is
        evidence about the total, and a total that matches does not establish
        that any particular agent's share does -- per-group errors can cancel in
        the sum. So the comparison is freed and the dollars are not.

        What must not survive is the caption. A group whose workload reconciled
        against a real bill used to be told "no invoice was supplied", which is
        simply false.
        """
        ts = self._two_agents()
        whole = simulate.bake_off(
            ts.analysable, trace=ts,
            allow_unreconciled=True).arms["as-shipped"]["spend"].raw()
        out = simulate.bake_off_by_agent(ts.analysable, trace=ts,
                                         invoice_usd=whole)
        self.assertTrue(out, "no groups came back, so nothing was checked")
        for b in out:
            fig = b.arms["as-shipped"]["spend"]
            self.assertFalse(fig.released, b.group)
            self.assertNotIn("no invoice was supplied", fig.withheld_because,
                             f"{b.group} was told no invoice was supplied when "
                             f"one was supplied and reconciled")
            self.assertIn("reconciles against the whole workload",
                          fig.withheld_because)
            self.assertIsNotNone(b.delta_pct, b.group)
            self.assertNotIn("indeterminate", b.verdict)

    def test_the_draft_flag_still_reaches_the_per_agent_figures(self):
        """The escape hatch has to keep working, or the by-agent path can never
        show a dollar figure at all and the flag is decoration there."""
        ts = self._two_agents()
        whole = simulate.bake_off(
            ts.analysable, trace=ts,
            allow_unreconciled=True).arms["as-shipped"]["spend"].raw()
        out = simulate.bake_off_by_agent(ts.analysable, trace=ts,
                                         invoice_usd=whole,
                                         allow_unreconciled=True)
        self.assertTrue(out)
        for b in out:
            self.assertTrue(b.arms["as-shipped"]["spend"].released, b.group)

    def test_the_draft_flag_does_not_reach_them_when_the_invoice_failed(self):
        """And it must not become a way round a bill the trace contradicts.
        Measured on the real command before this: `--by-agent --invoice-usd
        999999 --allow-unreconciled` printed $16.2077 per agent with a 68.9%
        Gate 1 pass beside it."""
        ts = self._two_agents()
        out = simulate.bake_off_by_agent(ts.analysable, trace=ts,
                                         invoice_usd=999_999.0,
                                         allow_unreconciled=True)
        self.assertTrue(out)
        for b in out:
            self.assertFalse(b.arms["as-shipped"]["spend"].released, b.group)
            self.assertIsNone(b.delta_pct, b.group)
            self.assertNotIn("beats the automatic baseline", b.verdict)

    def test_the_cli_exposes_both_flags(self):
        """It accepted neither, which is why the default printed dollars."""
        from cacheeconomics import cli
        parser = cli.build_parser()
        sub = {}
        for action in parser._actions:
            if isinstance(getattr(action, "choices", None), dict):
                sub.update(action.choices)
        flags = {o for a in sub["bakeoff"]._actions for o in a.option_strings}
        self.assertIn("--invoice-usd", flags)
        self.assertIn("--allow-unreconciled", flags)


if __name__ == "__main__":
    unittest.main()
