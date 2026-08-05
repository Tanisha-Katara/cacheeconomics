"""Where the same fact is computed twice, the two must agree.

Nine rounds of adversarial review on this branch produced findings that cluster,
and the largest cluster by some margin is a guard or a quantity that exists in
one code path and not in its twin:

  - the two trace loaders diverged on four separate guards before one
    `request_from_row` replaced both
  - the batch loader counted a missing segment as a change and the runtime did
    not, so an optional block read as perfectly stable and the live allocator
    marked a prefix that vanished
  - the plugin refused to place markers on an unknown minimum and
    `tiers.allocate` placed them anyway, reporting an 86% saving on a threshold
    nobody knows
  - the batch rebuild rule required a session id and the runtime substituted a
    constant, so unrelated one-shot calls were reported as one conversation
    rebuilding every turn
  - `cost.ttl_crossover` learned to check applicability before pricing, and
    `tiers._surface` had to learn it again

Each was fixed where it was found. None of those fixes stops the next one,
because the defect is not in any single function -- it is that two functions
were free to disagree and nothing was watching.

This file watches. Every test here takes one quantity and asserts the two (or
three, or four) implementations produce the same answer on the same input,
rather than asserting each works in isolation. A test that checks one side
cannot catch a divergence; that is precisely how these shipped.

Where a constant appears in several modules the honest fix is one definition and
imports. These tests are the cheaper half of that: they do not prevent the
duplication, they make drift fail loudly the moment it happens.
"""

import dataclasses
import os
import tempfile
import json
import sys
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone

def _vouched(row, body_key="request", target_id=None):
    """Stamp a counted fixture with the record the loader now requires.

    `load_bodies` only treats `segment_tokens` as exact when a
    `segment_tokens_provenance` record shows the counts were taken from this
    body and these prefix cuts, by a counter version it knows -- shape alone
    used to be the whole test, so a stale or hostile enrichment loaded as exact.
    These fixtures are hand-built stand-ins for a real counted export, so they
    carry the same record one would.
    """
    from cacheeconomics.tokenizer import (COUNTS_PROVENANCE_KEY,
                                          FIRST_PARTY_COUNT_ENDPOINT,
                                          counts_provenance)
    # The first-party endpoint, so the loader's serveability check actually runs
    # against these fixtures rather than being waved through: a counted export
    # is only believed when the id it names can be shown to be served.
    row[COUNTS_PROVENANCE_KEY] = counts_provenance(
        row[body_key], row, target_id, FIRST_PARTY_COUNT_ENDPOINT,
        "test-tokenizer")
    return row


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cacheeconomics import (analyzer, checks, cost, money, monitor,  # noqa: E402
                            trace, plugin, registry, segment, simulate, tiers)
from cacheeconomics.allocate import (observed_change_rates_by_chain,  # noqa: E402
                                     reuse_chain_of)
from cacheeconomics.analyzer import analyze  # noqa: E402
from cacheeconomics.trace import Request, Segment, Tier, TraceSet  # noqa: E402

T0 = datetime(2026, 7, 29, 9, tzinfo=timezone.utc)
KEY = b"k" * 32


class TestOneFactOneValue(unittest.TestCase):
    """Constants that describe the same thing, spelled out in several modules.

    None of these is wrong today. All of them are free to drift tomorrow, and a
    drift here is not a crash -- it is two components quietly disagreeing about
    how long a cache entry lives while both keep producing numbers.
    """

    def test_cache_lifetimes_agree_across_every_module_that_names_them(self):
        self.assertEqual(monitor.TTL_SECONDS, analyzer._TTL_SECONDS)
        self.assertEqual(monitor.TTL_SECONDS, simulate.TTL_SECONDS)

    def test_the_generic_parser_agrees_with_the_hard_coded_tables(self):
        """`tiers.ttl_seconds` parses any lifetime; the tables enumerate two.
        A table that disagreed with the parser would price one arm's entries
        against a lifetime another arm never modelled."""
        for name, seconds in monitor.TTL_SECONDS.items():
            self.assertEqual(tiers.ttl_seconds(name), seconds)

    def test_the_published_crossover_window_is_those_same_two_numbers(self):
        """The article's whole thesis is this band. If `ttl_crossover` and the
        simulator disagreed about where it sits, the page and the report would
        recommend different things."""
        self.assertEqual(cost.ttl_crossover("anthropic/direct")["window_seconds"],
                         (monitor.TTL_SECONDS["5m"], monitor.TTL_SECONDS["1h"]))

    def test_a_rebuild_means_the_same_share_in_both_implementations(self):
        self.assertEqual(analyzer.REBUILD_FRACTION, monitor.REBUILD_FRACTION)

    def test_both_token_estimators_use_one_ratio(self):
        """The plugin estimates prompt size to decide placement; the segmenter
        estimates it to divide a measured total. Different ratios would make the
        audit unable to reproduce the plugin's own decisions."""
        self.assertEqual(plugin.BYTES_PER_TOKEN, segment._BYTES_PER_TOKEN)

    def test_the_unattributed_surface_is_one_string(self):
        """`trace` spells it and `registry` spells it, because `trace` imports
        nothing from the package -- that is what lets every loader and both rule
        paths share it -- while `registry` reaches back into `trace` lazily, so
        a module-level edge either way would be a real cycle.

        Two spellings is the price of that, and drift between them would not
        crash: it would mean a loader marking rows with a surface the registry's
        rate scope does not recognise, which prices them at first-party rates.
        Exactly the failure the constant exists to prevent."""
        self.assertEqual(trace.UNATTRIBUTED, registry.UNATTRIBUTED)
        self.assertIn(trace.UNATTRIBUTED,
                      registry.rate_scope().get("unpriced_surfaces", {}),
                      "the loader's default is not registered as unpriceable")

    def test_write_visibility_is_one_latency_under_three_names(self):
        """`FANOUT_SECONDS`, `WRITE_LATENCY_SECONDS` and the simulator's
        pessimistic `write_latency_s` are the same physical fact: how long
        before a written entry can be read. Three names, one number."""
        self.assertEqual(tiers.WRITE_LATENCY_SECONDS,
                         simulate.PESSIMISTIC.write_latency_s)
        self.assertEqual(monitor.FANOUT_SECONDS, tiers.WRITE_LATENCY_SECONDS)

    def test_the_flat_latency_is_only_a_fallback(self):
        """Pinning the three names equal locks the number, and a number is the
        weaker half of this. The stronger half is that nothing uses it when the
        trace says what actually happened: a request whose first token lands at
        20s has a sibling at 8s that could not have read its entry, and a flat
        five-second window calls them sequential."""
        sent = T0
        observed = Request(request_id="o", sent_at=sent,
                           first_token_at=sent + timedelta(seconds=20),
                           model="claude-opus-5", usage={}, segments=[], target_id="anthropic/direct")
        assumed = Request(request_id="a", sent_at=sent,
                          model="claude-opus-5", usage={}, segments=[], target_id="anthropic/direct")
        at, was_observed = trace.write_visible_at(observed)
        self.assertEqual(at, sent + timedelta(seconds=20))
        self.assertTrue(was_observed)
        at, was_observed = trace.write_visible_at(assumed)
        self.assertEqual(at, sent + timedelta(seconds=monitor.FANOUT_SECONDS))
        self.assertFalse(was_observed)


class TestVolatilityIsMeasuredTheSameWay(unittest.TestCase):
    """The runtime keeps a rolling window and the batch loader sees the whole
    trace, but on the same requests they must report the same rates. They did
    not: absence of a segment was a change to one and silence to the other."""

    def _reqs(self, n=40):
        def segs(i):
            out = [Segment(id="tools", role="tools", tokens=9_000, index=0),
                   Segment(id="policy", role="system", tokens=6_000, index=1)]
            if i % 2:
                out.append(Segment(id="optional", role="system",
                                   tokens=4_000, index=2))
            if i % 3:
                out.append(Segment(id=f"ctx{i // 5}", role="system",
                                   tokens=1_000, index=3))
            out.append(Segment(id=f"turn{i}", role="user", tokens=200, index=9))
            return out
        return [Request(request_id=f"r{i}", sent_at=T0 + timedelta(seconds=240 * i),
                        model="claude-opus-5", usage={"input_tokens": 10},
                        segments=segs(i), session="s", target_id="anthropic/direct") for i in range(n)]

    def test_they_agree_exactly_when_every_position_appears_from_the_start(self):
        """The strict case, and the one that matters: no late arrivals, so both
        have seen the same history and any difference is drift."""
        def segs(i):
            return [Segment(id="tools", role="tools", tokens=9_000, index=0),
                    Segment(id=f"ctx{i // 5}", role="system", tokens=1_000, index=1),
                    Segment(id="A" if i % 2 else "B", role="system",
                            tokens=2_000, index=2),
                    Segment(id=f"turn{i}", role="user", tokens=200, index=3)]
        reqs = [Request(request_id=f"r{i}", sent_at=T0 + timedelta(seconds=240 * i),
                        model="claude-opus-5", usage={"input_tokens": 10},
                        segments=segs(i), session="s", target_id="anthropic/direct") for i in range(40)]
        m = monitor.Monitor()
        for r in reqs:
            m.observe_shape(r)
        scope = reuse_chain_of(reqs[0])
        runtime = m.change_rates(scope)
        batch = observed_change_rates_by_chain(reqs)[scope]
        self.assertEqual({k: round(v, 9) for k, v in runtime.items()},
                         {k: round(v, 9) for k, v in batch.items()})

    def test_a_late_arriving_position_differs_by_at_most_one_observation(self):
        """Not drift, and worth stating precisely rather than loosening the
        test until it passes.

        The batch loader reads the whole trace first, so it knows position 2
        exists and records it as *absent* on request 0. A streaming estimator
        cannot know a position exists before it first sees one, so its history
        for that position starts one request later. Both are right about what
        they can see, and the gap is bounded by a single observation.

        If the two ever differ by more than that, something is wrong that this
        explanation does not cover.
        """
        reqs = self._reqs()
        m = monitor.Monitor()
        for r in reqs:
            m.observe_shape(r)
        scope = reuse_chain_of(reqs[0])
        runtime = m.change_rates(scope)
        batch = observed_change_rates_by_chain(reqs)[scope]
        self.assertEqual(set(runtime), set(batch))
        bound = 1.0 / (len(reqs) - 1)
        for i, rate in runtime.items():
            self.assertLessEqual(
                abs(rate - batch[i]), bound,
                f"position {i}: runtime {rate:.4f} vs batch {batch[i]:.4f} is "
                f"further apart than one observation can explain")

    def test_they_agree_on_a_perfectly_stable_prompt_too(self):
        reqs = [Request(request_id=f"r{i}", sent_at=T0 + timedelta(seconds=60 * i),
                        model="claude-opus-5", usage={},
                        segments=[Segment(id="a", role="system", tokens=9_000, index=0)],
                        session="s", target_id="anthropic/direct") for i in range(20)]
        m = monitor.Monitor()
        for r in reqs:
            m.observe_shape(r)
        scope = reuse_chain_of(reqs[0])
        self.assertEqual(m.change_rates(scope),
                         observed_change_rates_by_chain(reqs)[scope])


class TestTheMinimumIsEnforcedEverywhereOrNowhere(unittest.TestCase):
    """Four components decide whether a prefix is long enough to cache. A prefix
    one of them accepts and another refuses is a marker somebody pays for and
    nothing reads."""

    SHORT = 200          # far below every registered minimum
    MODEL = "claude-haiku-4-5"     # 4,096

    def test_the_static_check_refuses_a_short_prefix(self):
        # The surface is named, as it is on the three siblings below. It used to
        # be omitted here and the check defaulted to `anthropic/direct`, so this
        # member of a four-way twin-path invariant was the one asking a
        # different question from the other three.
        r = checks.check_minimum(self.SHORT, self.MODEL, "anthropic/direct")
        self.assertIs(r.status, checks.Status.FAIL)

    def test_the_allocator_refuses_the_same_prefix(self):
        a = tiers.allocate([Segment(id="a", role="system", tokens=self.SHORT, index=0)],
                           {0: 0.0}, target_id="anthropic/direct",
                           model=self.MODEL, gaps=[120.0] * 20)
        self.assertEqual(a.tiers, [])

    def test_the_live_plugin_refuses_it(self):
        p = plugin.CachePlugin(key=KEY, warmup=4)
        body = {"system": [{"type": "text", "text": "s" * 400}],
                "messages": [{"role": "user", "content": "t"}]}
        last = None
        for i in range(20):
            # `apply=True` and a named surface, so this still tests the minimum.
            # With the observe-only default and no surface it would refuse for
            # two other reasons and the assertion below would pass without ever
            # reaching the guard it is named after.
            _, last = p.on_request(body, model=self.MODEL,
                                   target_id="anthropic/direct", apply=True,
                                   at=T0 + timedelta(seconds=90 * i))
        self.assertFalse(last.applied)

    def test_allocator_lite_refuses_the_same_prefix(self):
        """The fourth implementation of this guard. The twin-path suite covered
        `tiers.allocate`, `checks.check_minimum` and the plugin, and missed this
        one -- which is the same failure the suite exists to prevent, committed
        while writing the suite."""
        from cacheeconomics.allocate import allocator_lite
        r = Request(request_id="r", sent_at=T0, model=self.MODEL, usage={},
                    segments=[Segment(id="a", role="system",
                                      tokens=self.SHORT, index=0)], target_id="anthropic/direct")
        self.assertEqual(allocator_lite(r, volatility={0: 1},
                                        cadence_seconds=120).marker_indices, [])

    def test_an_unknown_model_is_refused_by_all_of_them(self):
        """Not knowing the threshold is when a marker is most likely to be paid
        for and cache nothing. `tiers.allocate` used to place them anyway."""
        with self.assertRaises(tiers.Unsupported):
            tiers.allocate([Segment(id="a", role="system", tokens=9_000, index=0)],
                           {0: 0.0}, target_id="anthropic/direct",
                           model="nobody-registered-this", gaps=[120.0] * 20)
        self.assertIs(checks.check_minimum(9_000, "nobody-registered-this",
                                           "anthropic/direct").status,
                      checks.Status.ABSTAIN)
        from cacheeconomics.allocate import allocator_lite
        r = Request(request_id="r", sent_at=T0, model="nobody-registered-this",
                    usage={}, segments=[Segment(id="a", role="system",
                                                tokens=9_000, index=0)], target_id="anthropic/direct")
        self.assertEqual(allocator_lite(r, volatility={0: 1},
                                        cadence_seconds=120).marker_indices, [])


class TestRebuildMeansTheSameThingLiveAndInReport(unittest.TestCase):

    def _session(self, turns=60, rebuild_every=None, sessioned=True):
        out, p = [], 100_000
        for t in range(turns):
            cold = t == 0 or (rebuild_every and t % rebuild_every == 0)
            w = p if cold else 2_000
            out.append(Request(
                request_id=f"r{t}", sent_at=T0 + timedelta(seconds=60 * t),
                model="claude-opus-5", session="s1" if sessioned else None,
                usage={"input_tokens": 0,
                       "cache_read_input_tokens": 0 if cold else p,
                       "cache_creation_input_tokens": w,
                       "cache_creation": {"ephemeral_5m_input_tokens": w,
                                          "ephemeral_1h_input_tokens": 0}}, target_id="anthropic/direct"))
            p += 2_000
        return out

    def _both(self, reqs):
        m, fired = monitor.Monitor(), []
        for r in reqs:
            fired += m.observe_usage(r)
        live = {a.code for a in fired}
        ts = TraceSet(requests=reqs, tier=Tier.USAGE_ONLY, source="twin")
        batch = {f.code for f in analyzer.analyze(ts, allow_unreconciled=True).findings}
        return live, batch

    def test_both_see_a_rebuilding_session(self):
        live, batch = self._both(self._session(rebuild_every=6))
        self.assertIn("RT-REBUILD", live)
        self.assertIn("REB-1", batch)

    def test_both_stay_quiet_on_a_session_that_only_extends(self):
        live, batch = self._both(self._session())
        self.assertNotIn("RT-REBUILD", live)
        self.assertNotIn("REB-1", batch)

    def test_both_refuse_to_answer_without_a_session_id(self):
        """Substituting a constant compared unrelated one-shot calls against
        each other and reported them as one conversation rebuilding every turn."""
        live, batch = self._both(self._session(rebuild_every=6, sessioned=False))
        self.assertNotIn("RT-REBUILD", live)
        self.assertNotIn("REB-1", batch)

    def test_and_both_say_so_rather_than_going_silent(self):
        live, batch = self._both(self._session(rebuild_every=6, sessioned=False))
        self.assertIn("RT-NOSESSION", live)
        self.assertIn("REB-0", batch)


class TestSegmentIdentityIsOneImplementation(unittest.TestCase):
    """A body segmented post-hoc and a request recorded live must produce
    identical ids for identical content, or a trace assembled from both reads as
    volatility that never happened."""

    BODY = {"tools": [{"name": "read", "description": "x" * 3000}],
            "system": [{"type": "text", "text": "policy " * 500}],
            "messages": [{"role": "user", "content": "hello"}]}

    def test_wire_order_is_the_same_walk_everywhere(self):
        from cacheeconomics.segment import walk
        self.assertEqual([lbl for _, lbl, _, _ in walk(self.BODY)],
                         [s["label"] for s in
                          segment.segments_from_request(self.BODY, KEY)])

    def test_marking_a_block_does_not_change_its_identity(self):
        """`cache_control` is an instruction to the cache, not content. Hashing
        it turned a caching change into apparent prompt drift, so the tool would
        react to its own recommendations."""
        before = [s["id"] for s in segment.segments_from_request(self.BODY, KEY)]
        marked = segment.apply_markers(self.BODY, {1: "5m"})
        after = [s["id"] for s in segment.segments_from_request(marked, KEY)]
        self.assertEqual(before, after)

    def test_the_same_text_in_two_containers_is_two_segments(self):
        """Identity covers the container. Hashing content alone let text move
        between roles and read downstream as a cache hit."""
        a = segment.segments_from_request(
            {"system": [{"type": "text", "text": "shared"}], "messages": []}, KEY)
        b = segment.segments_from_request(
            {"messages": [{"role": "user",
                           "content": [{"type": "text", "text": "shared"}]}]}, KEY)
        self.assertNotEqual(a[0]["id"], b[0]["id"])


class TestTheLifetimeBandIsOneBand(unittest.TestCase):
    """Every component that chooses between a five-minute and a one-hour write
    is answering the same question, and the answer is the same window."""

    def test_the_allocator_only_values_one_hour_inside_the_band(self):
        below = tiers.survival([60.0] * 50, "1h")
        inside = tiers.survival([900.0] * 50, "1h")
        above = tiers.survival([7200.0] * 50, "1h")
        self.assertEqual(inside, 1.0)
        self.assertEqual(above, 0.0)
        self.assertEqual(below, 1.0)      # alive, but 5m is alive there too

    def test_five_minutes_dies_where_the_band_opens(self):
        lo, _ = cost.ttl_crossover("anthropic/direct")["window_seconds"]
        self.assertEqual(tiers.survival([lo + 1.0] * 50, "5m"), 0.0)
        self.assertEqual(tiers.survival([lo - 1.0] * 50, "5m"), 1.0)

    def test_one_hour_dies_where_the_band_closes(self):
        _, hi = cost.ttl_crossover("anthropic/direct")["window_seconds"]
        self.assertEqual(tiers.survival([hi + 1.0] * 50, "1h"), 0.0)
        self.assertEqual(tiers.survival([hi - 1.0] * 50, "1h"), 1.0)

    def test_the_runtime_recommends_the_upgrade_only_inside_the_band(self):
        lo, hi = cost.ttl_crossover("anthropic/direct")["window_seconds"]

        def ttl_alerts(gap):
            m, fired = monitor.Monitor(), []
            for i in range(20):
                fired += m.observe(Request(
                    request_id=f"r{i}", sent_at=T0 + timedelta(seconds=gap * i),
                    model="claude-opus-5", usage={"input_tokens": 10},
                    ttl_requested="5m", session="s",
                    segments=[Segment(id="a", role="system", tokens=9_000, index=0,
                                      cache_marked=True, ttl="5m"),
                              Segment(id=f"t{i}", role="user", tokens=100, index=1)], target_id="anthropic/direct"))
            return [a for a in fired if a.code == "RT-TTL"]

        self.assertTrue(ttl_alerts((lo + hi) / 2))
        self.assertFalse(ttl_alerts(lo / 4))
        self.assertFalse(ttl_alerts(hi * 2))




class TestMixedLifetimesAreUnprovable(unittest.TestCase):
    """A durable prefix under an advancing turn is a deliberate pattern, and it
    is the one shape a single row-level `ttl_requested` cannot express.

    Both paths reached for the row field and got it wrong in their own way: the
    monitor told an operator to set a one-hour TTL on a prefix that already had
    one, and the analyzer priced a possible 2x write at 1.25x -- the same 38%
    understatement, in the same flattering direction, that the rest of the cost
    path exists to prevent."""

    def _req(self, i, prefix_ttl, turn_ttl, row_ttl, gap=900):
        return Request(request_id=f"r{i}", sent_at=T0 + timedelta(seconds=gap * i),
                       model="claude-opus-5", usage={"input_tokens": 10},
                       ttl_requested=row_ttl, session="s",
                       segments=[Segment(id="sys", role="system", tokens=40_000,
                                         index=0, cache_marked=True, ttl=prefix_ttl),
                                 Segment(id=f"t{i}", role="user", tokens=200,
                                         index=1, cache_marked=True, ttl=turn_ttl)], target_id="anthropic/direct")

    def _ttl_alert(self, prefix_ttl, turn_ttl, row_ttl, gap=900):
        m, fired = monitor.Monitor(), []
        for i in range(20):
            fired += m.observe(self._req(i, prefix_ttl, turn_ttl, row_ttl, gap))
        return next((a for a in fired if a.code == "RT-TTL"), None)

    def test_the_runtime_abstains_on_mixed_lifetimes(self):
        self.assertIsNone(self._ttl_alert("1h", "5m", "5m"))

    def test_the_analyzer_abstains_on_the_same_request(self):
        self.assertIsNone(analyzer._declared_ttl(self._req(0, "1h", "5m", "5m")))

    def test_an_implicit_marker_counts_as_the_default_when_mixed(self):
        """A silent marker beside an explicit marker is a real 5m marker, not
        row metadata waiting to inherit the explicit value."""
        r = self._req(0, "1h", None, "1h")
        self.assertEqual(r.marker_lifetimes, {"1h", "5m"})
        self.assertIsNone(analyzer._declared_ttl(r))
        self.assertIsNone(self._ttl_alert("1h", None, "1h", gap=4000))

    def test_a_uniform_lifetime_still_produces_advice(self):
        """Abstaining must not become silence on the case the check is for.

        One marked span, marked again each time. The turn is no longer marked:
        with an advancing marker the cached span differs on every request, and
        a longer lifetime cannot turn a write of *this* span into a read of
        *that* one, which is why the analyzer refuses that shape too. The
        companion test below pins that they refuse it together.
        """
        m, fired = monitor.Monitor(), []
        for i in range(20):
            fired += m.observe(Request(
                request_id=f"r{i}", sent_at=T0 + timedelta(seconds=900 * i),
                model="claude-opus-5", usage={"input_tokens": 10},
                ttl_requested="5m", session="s",
                segments=[Segment(id="sys", role="system", tokens=40_000,
                                  index=0, cache_marked=True, ttl="5m"),
                          Segment(id=f"t{i}", role="user", tokens=200, index=1)], target_id="anthropic/direct"))
        self.assertIsNotNone(next((a for a in fired if a.code == "RT-TTL"), None))

    def test_an_advancing_marker_is_reported_by_both(self):
        """A rolling conversation marks a stable prefix *and* the turn, so the
        span at the outermost marker is different every request.

        This test used to assert both sides refused, and that was the wrong
        answer twice over. RT-TTL fired here on the median request gap alone
        while TTL-1 found no span written *twice* and refused; making them agree
        settled it at silence, which pinned the gap instead of the fix. But the
        stable prefix underneath is rewritten on every one of these requests,
        and a one-hour lifetime turns those rewrites into reads -- which is the
        whole of what this rule is for.

        The equality test was wrong, not the finding. `('sys',)` is a prefix of
        `('sys','turn0')`, and `simulate.py` has always credited an advancing
        breakpoint reading the shorter entry a previous turn wrote:
        `seq[:len(key)] == key` is "the provider's own condition for a read".
        Both sides use containment now, and both speak.
        """
        reqs = [self._req(i, "5m", "5m", "5m") for i in range(20)]
        for r in reqs:
            r.usage.update({"input_tokens": 40_200,
                            "cache_creation_input_tokens": 40_000})
        m, fired = monitor.Monitor(), []
        for r in reqs:
            fired += m.observe(r)
        runtime = [a for a in fired if a.code == "RT-TTL"]
        ts = TraceSet(requests=reqs, tier=Tier.INSTRUMENTED, source="twin")
        batch = [f for f in analyze(ts).findings if f.code == "TTL-1"]
        self.assertEqual(bool(runtime), bool(batch))
        self.assertTrue(runtime, "the canonical agent shape, and neither speaks")

    def test_and_the_analyzer_still_prices_it(self):
        self.assertEqual(analyzer._declared_ttl(self._req(0, "5m", "5m", "5m")), "5m")

    def test_the_marker_wins_over_the_row(self):
        """The marker is on the block that was sent; the row is metadata about
        it. When they disagree, one is stale."""
        self.assertIsNone(analyzer._declared_ttl(self._req(0, "1h", "1h", "5m")))




class TestSessionIdentityIsFoundInOnePlace(unittest.TestCase):
    """The runtime LiteLLM adapter treats `metadata.trace_id` as the
    conversation id -- the only field LiteLLM documents as spanning calls --
    and ingest read only top-level keys. So a real LiteLLM export loaded with
    `session=None`, switching REB-1 off and making the report say rebuilds
    could not be measured on a file that carried the key all along."""

    def _load(self, row):
        import json
        import tempfile
        from cacheeconomics.trace import load_jsonl
        base = {"request_id": "r1", "sent_at": "2026-07-29T09:00:00Z",
                "model": "claude-opus-5",
                "usage": {"input_tokens": 10, "cache_read_input_tokens": 100,
                          "cache_creation_input_tokens": 0}}
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        try:
            with open(path, "w") as f:
                f.write(json.dumps({**base, **row}))
            return load_jsonl(path).requests[0]
        finally:
            os.unlink(path)

    def test_ingest_reads_the_field_the_adapter_writes(self):
        """Was `metadata.trace_id`. That field is LiteLLM's retry correlation,
        not a conversation, and using it produced single-request groups that
        silenced rebuild detection instead of measuring it. The twin-path
        property under test is unchanged: whatever the live path calls a
        session, the loader must read back as one."""
        from cacheeconomics.plugin import default_session_from
        data = {"metadata": {"session_id": "conv-42"}}
        self.assertEqual(default_session_from(data), "conv-42")
        self.assertEqual(self._load(data).session, "conv-42")

    def test_neither_side_treats_a_retry_id_as_a_conversation(self):
        from cacheeconomics.plugin import default_session_from
        data = {"metadata": {"trace_id": "retry-group"}}
        self.assertIsNone(default_session_from(data))
        self.assertIsNone(self._load(data).session)

    def test_an_explicit_session_id_still_wins_on_both_sides(self):
        from cacheeconomics.plugin import default_session_from
        data = {"metadata": {"session_id": "s", "trace_id": "t"}}
        self.assertEqual(default_session_from(data), "s")
        self.assertEqual(self._load(data).session, "s")

    def test_a_top_level_session_still_wins_over_metadata(self):
        self.assertEqual(
            self._load({"session": "top", "metadata": {"trace_id": "nested"}}).session,
            "top")


class TestWriteTokensHasOneReader(unittest.TestCase):
    """Anthropic reports cache writes as an aggregate, as a per-lifetime split,
    or both. Nine sites each read the aggregate directly and called it the
    answer, so every one of them saw zero on a split-only export while the cost
    model -- which does read the split -- kept billing it.

    That produced four review rounds of the same defect one place over: the size
    gate, then `_billed_input`, then the rebuild rule. This audit is the answer
    to the class rather than the instance. A tenth site has to either use the
    helper or say why.
    """

    # Sites that legitimately name the raw field: the helper itself, the cost
    # model's split-vs-aggregate reconciliation, and the loader's own evidence
    # test. Everything else asks `write_tokens`.
    JUSTIFIED = {
        ("trace.py", "write_tokens"): "is the helper",
        ("trace.py", "usage_from_row"): "normalises the raw field on ingest",
        ("cost.py", "from_anthropic"): "reconciles the aggregate against the "
                                       "split and refuses to guess between them",
        ("cost.py", "write_lifetime"): "decides which lifetime billed, which is "
                                       "a different question from how many",
        ("cost.py", "expiry_seconds"): "same, for expiry",
        ("segment.py", "usage_from_response"): "decides whether the counters are "
                                               "evidence at all",
        ("claude_code.py", "_usage_only"): "copies the accounting fields "
                                           "verbatim, split included",
        ("trace.py", "has_usage"): "asks whether there is any accounting, not "
                                   "how much was written",
        ("cost.py", "<module>"): "_TOP_COUNTS, the validator's field list",
        ("litellm.py", "usage_from_payload"): "reads the provider's own field "
                                              "names to build the canonical "
                                              "dict, split included",
    }

    def test_no_new_site_reads_the_aggregate_directly(self):
        import ast
        import pathlib

        import cacheeconomics
        pkg = pathlib.Path(cacheeconomics.__file__).parent
        found = {}

        def visit(node, where, fname):
            # `usage.get("cache_creation_input_tokens")` and
            # `usage["cache_creation_input_tokens"]` are reads. A key in a dict
            # *literal* is the opposite -- code building the canonical shape --
            # and flagging those made a normaliser look like a violation. An
            # audit with false positives teaches people to silence it, so the
            # keys of an ast.Dict are skipped rather than justified away.
            if isinstance(node, ast.Dict):
                for value in node.values:
                    visit(value, where, fname)
                return
            if isinstance(node, ast.Constant) and \
                    node.value == "cache_creation_input_tokens":
                found.setdefault((where, fname), []).append(node.lineno)
            for child in ast.iter_child_nodes(node):
                inner = (child.name
                         if isinstance(child, (ast.FunctionDef,
                                               ast.AsyncFunctionDef))
                         else fname)
                visit(child, where, inner)

        for path in sorted(pkg.rglob("*.py")):
            tree = ast.parse(path.read_text())
            # Docstrings and message text name the field constantly and that is
            # not a read. Only expression context counts, so strip the bodies of
            # string-only statements first.
            for node in ast.walk(tree):
                if isinstance(node, ast.Expr) and \
                        isinstance(node.value, ast.Constant) and \
                        isinstance(node.value.value, str):
                    node.value = ast.Constant(value="")
            visit(tree, path.name, "<module>")

        unjustified = {k: v for k, v in found.items() if k not in self.JUSTIFIED}
        self.assertFalse(
            unjustified,
            f"these read the aggregate write counter directly: {unjustified}. "
            f"A split-only export reports it as absent while still being billed, "
            f"so each of these silently sees zero. Use `trace.write_tokens`, or "
            f"add the site to JUSTIFIED with the reason.")

    def test_the_helper_reads_both_shapes(self):
        from cacheeconomics.trace import write_tokens
        self.assertEqual(write_tokens({"cache_creation_input_tokens": 900}), 900)
        self.assertEqual(write_tokens({"cache_creation": {
            "ephemeral_5m_input_tokens": 700,
            "ephemeral_1h_input_tokens": 200}}), 900)
        # The aggregate wins; adding both would double-count the normal case.
        self.assertEqual(write_tokens({
            "cache_creation_input_tokens": 900,
            "cache_creation": {"ephemeral_5m_input_tokens": 900,
                               "ephemeral_1h_input_tokens": 0}}), 900)
        self.assertEqual(write_tokens({}), 0)


class TestNoSurfaceIsHardCodedOutsideTheRegistry(unittest.TestCase):
    """The audit below was scoped to `_load`, and that was the mistake.

    "The surface is not threaded through" has now been found three times:
    `cost.price` took a model and no target, `_load` accepted `--target-id` and
    forwarded it to one of three loaders, and the live LiteLLM hook called
    `on_request` with no target at all while re-checking the breakpoint budget
    against a hard-coded `anthropic/direct`. Each fix was scoped to the function
    where it was found, so the next instance was free to happen somewhere else.

    This one is keyed to the invariant instead: no execution path may name a
    provider surface as a literal when asking the registry a question about it.
    Defaults declared in a signature are fine -- that is what a default is --
    and so is the registry's own data. What is not fine is a *call* that decides
    behaviour from a surface nobody resolved.
    """

    LOOKUPS = {"capability", "multipliers", "min_cacheable_tokens",
               "supported_ttls", "base_rate", "require_priceable"}
    # `cost.price` and the loaders declare `anthropic/direct` as a parameter
    # default, which is the documented fallback, not a hard-coded surface.
    ALLOWED_FILES = {"registry.py"}

    def test_no_registry_lookup_is_given_a_literal_surface(self):
        import ast
        import os

        import cacheeconomics
        root = os.path.dirname(os.path.abspath(cacheeconomics.__file__))
        offenders = []
        for dirpath, _, names in os.walk(root):
            for name in sorted(names):
                if not name.endswith(".py") or name in self.ALLOWED_FILES:
                    continue
                path = os.path.join(dirpath, name)
                with open(path) as fh:
                    tree = ast.parse(fh.read(), filename=name)
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Call):
                        continue
                    fn = node.func
                    attr = fn.attr if isinstance(fn, ast.Attribute) else (
                        fn.id if isinstance(fn, ast.Name) else None)
                    if attr not in self.LOOKUPS or not node.args:
                        continue
                    first = node.args[0]
                    if isinstance(first, ast.Constant) and isinstance(first.value, str) \
                            and "/" in first.value:
                        offenders.append(f"{name}:{node.lineno} "
                                         f"{attr}({first.value!r})")
        self.assertEqual(
            offenders, [],
            "a registry lookup names a surface as a literal, so it answers for "
            "the wrong provider on every other one:\n  " + "\n  ".join(offenders))


class TestTheDraftOverrideMeansTheSameThingEverywhere(unittest.TestCase):
    """`allow_unreconciled` covers a missing invoice and nothing else.

    The analyzer enforced that with `recon is None`. `bake_off` wrote
    `allow_unreconciled or reconciled is True`, so an invoice that was supplied
    and *failed* still released: measured, a $999,999 invoice against $0.27 of
    computed spend published $0.27, as did a negative invoice and a NaN. The
    analyzer's copy was corrected in the same session that left this one alone
    -- twin-path divergence in the pair this file is named after.

    So the rule is one function now, and this asserts both modules ask it rather
    than restating it. Behaviour first, then the structure that keeps it.
    """

    def _reqs(self):
        return [Request(request_id=f"r{i}", sent_at=T0 + timedelta(seconds=120 * i),
                        model="claude-opus-5", agent="a", session="s",
                        ttl_requested="5m",
                        usage={"input_tokens": 200,
                               "cache_read_input_tokens": 9000 if i else 0,
                               "cache_creation_input_tokens": 0 if i else 9000},
                        segments=[Segment(id="sys", role="system", tokens=9000,
                                          index=0, cache_marked=True, ttl="5m"),
                                  Segment(id=f"t{i}", role="user", tokens=200,
                                          index=1)], target_id="anthropic/direct")
                for i in range(30)]

    def _ts(self):
        """Instrumented, fully covered, sizes counted and agreeing with the bill.

        Stated so this class keeps testing the *invoice* rule. Arm spend now also
        asks whether the segment sizes it is priced from can carry money at all,
        and a bake-off handed a bare list of requests has been told nothing about
        that -- so without this every case below would withhold for the wrong
        reason and `test_no_invoice_plus_the_override_still_releases` would pass
        by accident in the one direction it must not.
        """
        return TraceSet(requests=self._reqs(), tier=Tier.INSTRUMENTED, source="x")

    def test_a_failed_invoice_blocks_the_bake_off_despite_the_override(self):
        for invoice in (999_999.0, -5.0, 0.0, float("nan")):
            with self.subTest(invoice=invoice):
                ts = self._ts()
                b = simulate.bake_off(ts.analysable, group="g", trace=ts,
                                      invoice_usd=invoice,
                                      allow_unreconciled=True)
                self.assertFalse(b.arms["as-shipped"]["spend"].released)

    def test_no_invoice_plus_the_override_still_releases(self):
        """The override has to keep working, or internal drafts are impossible
        and the flag is decoration."""
        ts = self._ts()
        b = simulate.bake_off(ts.analysable, group="g", trace=ts,
                              allow_unreconciled=True)
        self.assertTrue(b.arms["as-shipped"]["spend"].released)

    def test_both_modules_ask_the_shared_rule(self):
        """Structural. The behavioural test above covers `bake_off` and the
        analyzer separately; this is what stops a third release site restating
        the condition a fourth way."""
        import ast
        import inspect

        from cacheeconomics import analyzer, simulate as sim

        def offenders(mod):
            tree = ast.parse(inspect.getsource(mod))
            for parent in ast.walk(tree):
                for child in ast.iter_child_nodes(parent):
                    child._parent = parent

            def enclosing_call(node):
                while node is not None:
                    if isinstance(node, ast.Call):
                        return node
                    node = getattr(node, "_parent", None)
                return None

            bad = []
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Name)
                        and node.id == "allow_unreconciled"
                        and isinstance(node.ctx, ast.Load)):
                    continue
                call = enclosing_call(node)
                if call is None:
                    # Read outside any call: that is a bare condition, which is
                    # how `bake_off` came to mean something the analyzer did not.
                    bad.append(node.lineno)
                    continue
                fn = call.func
                name = fn.attr if isinstance(fn, ast.Attribute) else getattr(
                    fn, "id", None)
                # Passing it onward to another entry point is fine; deciding
                # with it is not.
                if name != "draft_override_applies" and not any(
                        kw.value is node for kw in call.keywords):
                    bad.append(node.lineno)
            return bad

        for mod in (analyzer, sim):
            self.assertEqual(
                offenders(mod), [],
                f"{mod.__name__} decides on allow_unreconciled outside "
                f"money.draft_override_applies at line(s) {offenders(mod)}")
            self.assertIn("draft_override_applies", inspect.getsource(mod),
                          f"{mod.__name__} does not use the shared rule")


class TestTheBakeOffRefusesOnTheSameEvidenceAsTheReport(unittest.TestCase):
    """Two release gates over one trace. They must not disagree.

    `cmd_bakeoff` passed `ts.analysable` to the simulator and nothing else, so
    every row the loader had excluded vanished before the spend gate ran.
    Measured: a trace with one failed-but-billed 5,000,000-token request and one
    unreadable line released $0.2685 from the bake-off while `analyze` over the
    identical file withheld. `analysable` is the right input for the *arms* -- a
    failed call populated no cache entry -- and the wrong denominator for a
    dollar figure.

    Asserted as behaviour across every blocker the analyzer knows rather than
    for the one that was reported, because the reported one was the third
    instance of a report/simulator gate splitting apart on this branch.
    """

    def _base(self, n=30):
        return [Request(
            request_id=f"r{i}", sent_at=T0 + timedelta(seconds=120 * i),
            model="claude-opus-5", agent="a", session="s", ttl_requested="5m",
            usage={"input_tokens": 200,
                   "cache_read_input_tokens": 9000 if i else 0,
                   "cache_creation_input_tokens": 0 if i else 9000},
            segments=[Segment(id="sys", role="system", tokens=9000, index=0,
                              cache_marked=True, ttl="5m"),
                      Segment(id=f"t{i}", role="user", tokens=200, index=1)], target_id="anthropic/direct")
            for i in range(n)]

    def _both_withhold(self, ts, label):
        a = analyze(ts, allow_unreconciled=True)
        b = simulate.bake_off(ts.analysable, group="g", allow_unreconciled=True,
                              excluded_billed=ts.excluded_billed, trace=ts)
        self.assertFalse(a.spend["input_usd"].released, f"{label}: analyzer released")
        self.assertFalse(b.arms["as-shipped"]["spend"].released,
                         f"{label}: bake-off released while the report withheld")

    def test_a_failed_but_billed_request(self):
        reqs = self._base()
        reqs.append(Request(
            request_id="failed", sent_at=T0 + timedelta(seconds=9999),
            model="claude-opus-5", agent="a", session="s", ttl_requested="5m",
            status=500,
            usage={"input_tokens": 5_000_000, "cache_read_input_tokens": 0,
                   "cache_creation_input_tokens": 0}, segments=[], target_id="anthropic/direct"))
        self._both_withhold(
            TraceSet(requests=reqs, tier=Tier.INSTRUMENTED, source="x"),
            "failed but billed")

    def test_a_row_carrying_no_usage_at_all(self):
        reqs = self._base()
        reqs.append(Request(
            request_id="blind", sent_at=T0 + timedelta(seconds=9999),
            model="claude-opus-5", agent="a", session="s", ttl_requested="5m",
            usage={}, segments=[], target_id="anthropic/direct"))
        self._both_withhold(
            TraceSet(requests=reqs, tier=Tier.INSTRUMENTED, source="x"),
            "no usage fields")

    def test_lines_the_loader_could_not_read(self):
        """Not expressible as a Request at all -- it is a line that never became
        one. The bake-off could not see it even in principle until the count was
        handed over."""
        self._both_withhold(
            TraceSet(requests=self._base(), tier=Tier.INSTRUMENTED, source="x",
                     skipped_rows=1),
            "unreadable rows")

    def test_a_partner_surface_the_rates_do_not_cover(self):
        reqs = [replace(r, target_id="amazon-bedrock/converse") for r in self._base()]
        self._both_withhold(
            TraceSet(requests=reqs, tier=Tier.INSTRUMENTED, source="x"),
            "unpriceable surface")

    def test_a_clean_trace_still_releases_on_both(self):
        """The gate has to stay usable. If every trace withheld, the agreement
        above would be vacuous."""
        ts = TraceSet(requests=self._base(), tier=Tier.INSTRUMENTED, source="x")
        self.assertEqual(ts.excluded_billed, {})
        a = analyze(ts, allow_unreconciled=True)
        b = simulate.bake_off(ts.analysable, group="g", allow_unreconciled=True,
                              excluded_billed=ts.excluded_billed, trace=ts)
        self.assertTrue(a.spend["input_usd"].released)
        self.assertTrue(b.arms["as-shipped"]["spend"].released)

    def test_the_verdict_goes_indeterminate_not_just_the_dollars(self):
        """Withholding the money and keeping the claim it supports is the worse
        half of both options.

        `excluded_billed` was wired to `spend_ok` alone, so the arms rendered
        `[withheld]` while the headline still read "allocator-lite beats the
        automatic baseline by 20.0% (gate: >=10%)" over a trace missing a
        5,000,000-token billed request. The module's own comment already says a
        verdict over a partial denominator is not a verdict; it just was not
        applied to rows excluded one layer earlier than `omitted`.
        """
        reqs = self._base()
        reqs.append(Request(
            request_id="failed", sent_at=T0 + timedelta(seconds=99999),
            model="claude-opus-5", agent="a", session="s", ttl_requested="5m",
            status=500,
            usage={"input_tokens": 5_000_000, "cache_read_input_tokens": 0,
                   "cache_creation_input_tokens": 0}, segments=[], target_id="anthropic/direct"))
        ts = TraceSet(requests=reqs, tier=Tier.INSTRUMENTED, source="x")
        b = simulate.bake_off(ts.analysable, group="g", allow_unreconciled=True,
                              excluded_billed=ts.excluded_billed)
        self.assertIsNone(b.delta_pct)
        self.assertIsNone(b.delta_pct_optimistic)
        self.assertIn("indeterminate", b.verdict)
        self.assertIn("indeterminate", b.verdict_relocation)

    def test_a_clean_trace_still_reaches_a_verdict(self):
        ts = TraceSet(requests=self._base(), tier=Tier.INSTRUMENTED, source="x")
        b = simulate.bake_off(ts.analysable, group="g", allow_unreconciled=True,
                              excluded_billed=ts.excluded_billed, trace=ts)
        self.assertIsNotNone(b.delta_pct)
        self.assertNotIn("indeterminate", b.verdict)

    def test_the_cli_hands_the_exclusions_over(self):
        """Behavioural now, because the route changed and the requirement did not.

        This used to assert that `cmd_bakeoff` passes `excluded_billed=`. The
        simulator derives that from the trace now, and the CLI passing both was
        the exact call shape that hid the original defect -- every test of the
        parity supplied the argument *and* the trace, so the derivation was
        never the thing under test. Asserting the argument is still there would
        pin the shape that concealed the bug.

        So this asserts the requirement instead: rows the loader dropped reach
        the spend gate through the real command. An implementation that stops
        deriving them, or a CLI that stops handing the trace over, fails here
        whichever way the plumbing is arranged.
        """
        import hashlib
        import io
        import json
        import os
        import tempfile
        from contextlib import redirect_stdout

        from cacheeconomics import cli
        from cacheeconomics.trace import load_jsonl

        # Shaped like `segment_id()` output: the loader rejects short ids, and
        # an unrecognised id would make this trace INFERRED and block for a
        # different reason than the one under test.
        def hid(name):
            return "hmac:" + hashlib.sha256(name.encode()).hexdigest()

        HMAC_A, HMAC_B = hid("head"), hid("body")
        rows = []
        for i in range(8):
            rows.append({
                "request_id": f"r{i}", "sent_at": f"2026-07-29T09:{i:02d}:00Z",
                "model": "claude-opus-5", "agent": "main", "session": "s1",
                "target_id": "anthropic/direct", "tokens_counted": True,
                "usage": {"input_tokens": 300, "cache_read_input_tokens": 0,
                          "cache_creation_input_tokens": 30_000,
                          "cache_creation": {
                              "ephemeral_5m_input_tokens": 30_000,
                              "ephemeral_1h_input_tokens": 0}},
                "segments": [
                    {"id": HMAC_A + str(i % 10), "role": "system",
                     "tokens": 300, "index": 0, "cache_marked": False,
                     "ttl": None},
                    {"id": HMAC_B, "role": "system", "tokens": 30_000,
                     "index": 1, "cache_marked": True, "ttl": "5m"}]})
        # One row that failed and billed anyway: not analysable, so no arm can
        # model it, and every figure computed without it describes a subset.
        rows.append({
            "request_id": "failed", "sent_at": "2026-07-29T09:30:00Z",
            "model": "claude-opus-5", "agent": "main", "session": "s1",
            "target_id": "anthropic/direct", "status": 500,
            "tokens_counted": True,
            "usage": {"input_tokens": 5_000_000, "cache_read_input_tokens": 0,
                      "cache_creation_input_tokens": 0},
            "segments": [{"id": HMAC_B, "role": "system", "tokens": 5_000_000,
                          "index": 0, "cache_marked": False, "ttl": None}]})
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        try:
            with open(path, "w") as f:
                f.write("\n".join(json.dumps(r) for r in rows))
            ts = load_jsonl(path)
            self.assertEqual(ts.excluded_billed, {"failed but billed": 1},
                             "the fixture stopped reproducing the condition")
            args = cli.build_parser().parse_args(
                ["bakeoff", path, "--allow-unreconciled"])
            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(args.func(args), 0)
            text = out.getvalue()
        finally:
            os.unlink(path)
        self.assertIn("[withheld]", text,
                      "the command published arm spend over a trace carrying a "
                      "billed row no arm could model")
        self.assertIn("failed but billed", text)
        self.assertNotIn("beats the automatic baseline", text)

    def test_the_cli_hands_the_trace_over(self):
        """The structural half of the same parity, and the same failure mode.

        `excluded_billed` had to be threaded through before the simulator could
        refuse on rows the loader dropped. Tier, alignment, structural coverage
        and counted-versus-estimated sizes are the rest of that class: they live
        on the `TraceSet` and on nothing the arms receive, so `cmd_bakeoff`
        handing over `ts.analysable` alone leaves them unreachable rather than
        merely unsupplied.

        This was an expected failure for two rounds on the grounds that cli.py
        belonged to another track and the wiring would land at merge. That was
        wrong: with the simulator's gate in place and the CLI on the no-trace
        path, the shipped command withheld for every input including a clean
        instrumented capture with a reconciling invoice. An xfail describes a
        defect; it does not ship a fix, and the product was broken in the
        meantime.
        """
        import ast
        import inspect

        from cacheeconomics import cli
        tree = ast.parse(inspect.getsource(cli.cmd_bakeoff))
        calls = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute)
                 and n.func.attr.startswith("bake_off")]
        self.assertTrue(calls, "no bake-off call found in cmd_bakeoff")
        for c in calls:
            self.assertIn(
                "trace", {k.arg for k in c.keywords},
                f"{c.func.attr} is called without trace, so the facts that "
                f"decide whether a figure priced from segment boundaries may be "
                f"published cannot reach the spend gate")

    # Kept separate from the structural check above because it fails for a
    # different reason and would otherwise hide behind it: that one asserts the
    # argument is passed, this one asserts the command still works once it is.
    def _run_bakeoff(self, *extra):
        import io
        import os
        from contextlib import redirect_stdout

        from cacheeconomics import cli

        fixture = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "fixtures", "demo-traces.jsonl")
        args = cli.build_parser().parse_args(["bakeoff", fixture, *extra])
        out = io.StringIO()
        with redirect_stdout(out):
            rc = args.func(args)
        return rc, out.getvalue()

    def test_the_cli_still_prints_dollars_when_the_evidence_is_there(self):
        """The regression the argument-passing check cannot catch.

        A fail-closed gate that nothing opens is indistinguishable from a broken
        command. For two rounds `cmd_bakeoff` was on the no-trace path
        permanently and the shipped `cacheeconomics bakeoff` could not release a
        dollar figure for any input at all. Asserting only that `trace=` appears
        in the source would also go green on wiring that passed the wrong
        object, so this drives the real command over the real fixture.

        $17.1443 against a $17.45 invoice is what it printed before this branch
        touched it.
        """
        rc, text = self._run_bakeoff("--invoice-usd", "17.45")
        self.assertEqual(rc, 0)
        self.assertIn("$17.1443", text,
                      "the shipped bakeoff cannot print a dollar figure for a "
                      "clean instrumented trace with a reconciling invoice")
        self.assertNotIn("[withheld]", text)
        self.assertNotIn("no trace was supplied", text)
        self.assertIn("beats the automatic baseline", text)

    def test_the_cli_by_agent_path_is_wired_the_same_way(self):
        """Two call sites, and only one of them was ever exercised by a test.

        `--by-agent` is the path where the per-agent exclusion scoping lives, so
        wiring one call and not the other would leave the scoped derivation
        unreachable from the product while every unit test of it passed.
        """
        rc, text = self._run_bakeoff("--by-agent", "--allow-unreconciled")
        self.assertEqual(rc, 0)
        self.assertNotIn("no trace was supplied", text)
        self.assertIn("$", text, "no group released a figure")

    def test_the_cli_by_agent_path_honours_a_failed_invoice(self):
        """The second call site, which the first round of CLI wiring left half
        done.

        `bake_off_by_agent` accepts `invoice_usd` and never handed it to a group,
        so with no invoice in scope every group read `reconciled is None`,
        `draft_override_applies` took that for "no invoice was supplied", and the
        command printed per-agent Gate 1 results over a bill it contradicted by a
        factor of 58,000. With `--allow-unreconciled` it printed $16.2077 of
        draft dollars per agent beside a 68.9% pass.

        Both flags, because the override was the worse of the two.
        """
        for extra in (["--invoice-usd", "999999"],
                      ["--invoice-usd", "999999", "--allow-unreconciled"]):
            with self.subTest(" ".join(extra)):
                rc, text = self._run_bakeoff("--by-agent", *extra)
                self.assertEqual(rc, 0)
                self.assertIn("[withheld]", text)
                self.assertNotIn("beats the automatic baseline", text)
                self.assertNotIn("vs litellm-auto", text)
                self.assertNotIn("no invoice was supplied", text,
                                 "an invoice was supplied and it failed")
                self.assertIn("does not reconcile", text)

    def test_the_cli_by_agent_path_still_works_without_an_invoice(self):
        """So the guard above cannot pass by `--by-agent` having stopped
        producing anything."""
        rc, text = self._run_bakeoff("--by-agent", "--allow-unreconciled")
        self.assertEqual(rc, 0)
        self.assertIn("$", text)
        self.assertIn("beats the automatic baseline", text)

    def test_the_cli_withholds_when_the_evidence_is_not_there(self):
        """The other direction, so the test above cannot pass by the gate having
        been removed. Same command, same fixture, an invoice that does not
        reconcile: dollars withheld and no Gate 1 pass printed."""
        rc, text = self._run_bakeoff("--invoice-usd", "999999")
        self.assertEqual(rc, 0)
        self.assertNotIn("$17.1443", text)
        self.assertIn("[withheld]", text)
        self.assertNotIn("beats the automatic baseline", text)


@dataclasses.dataclass
class _TraceSetWithAssumedInputs(TraceSet):
    """Stand-in for `TraceSet.assumed_inputs` while it lands on another track.

    Declared as a real dataclass field rather than set as an instance attribute,
    and that is load-bearing rather than tidiness: `_agent_trace` narrows the
    trace with `dataclasses.replace`, which copies *fields* and not stray
    attributes. An attribute-only stand-in would silently vanish on the by-agent
    path, and the test asserting the narrowed trace still carries the assumption
    would pass for the wrong reason -- or fail for one.

    Redeclaring the field once the real one exists is harmless, and
    `assumed_for` below prefers the real one as soon as it is there.
    """
    assumed_inputs: tuple = ()


def assumed_for(ts, inputs=("provider surface",)):
    """`ts` with its pricing inputs marked assumed, on either schema."""
    names = {f.name for f in dataclasses.fields(ts)}
    if "assumed_inputs" in names:
        return replace(ts, assumed_inputs=tuple(inputs))
    return _TraceSetWithAssumedInputs(
        assumed_inputs=tuple(inputs),
        **{f.name: getattr(ts, f.name) for f in dataclasses.fields(ts)})


class TestAnAssumedPricingInputCapsTheLabel(unittest.TestCase):
    """A figure is RECONCILED when an invoice checked it. An invoice cannot
    check a rate that was guessed.

    The assumption sits upstream of the number the invoice was compared
    against, so agreement proves the arithmetic and not the input. Today the one
    assumed input is the provider surface -- `load_sessions(...,
    surface_assumed=True)` on the claude_code adapter, which already raised a
    blocking note and had a structured field beside it that nothing read.

    This caps the *label* and nothing else. `spend_as` reaches only
    `Figure.release(..., as_=)`, which sets `released_as`; whether a figure is
    released at all is `spend_ok`, which this does not touch. An assumed surface
    with a reconciling invoice still publishes -- as DRAFT. Both halves are
    asserted, because a cap that quietly became a withhold would be the same
    class of defect in the other direction.
    """

    def _reqs(self, per_agent=6):
        out = []
        for agent in ("alpha", "beta"):
            for i in range(per_agent):
                out.append(Request(
                    request_id=f"{agent}{i}",
                    sent_at=T0 + timedelta(seconds=60 * i),
                    model="claude-opus-5", agent=agent, tenant="t",
                    session=f"s{agent}", target_id="anthropic/direct",
                    ttl_requested="5m",
                    usage={"input_tokens": 300,
                           "cache_creation_input_tokens": 30_000,
                           "cache_read_input_tokens": 0},
                    segments=[Segment(id=f"h{agent}{i}", role="system",
                                      tokens=300, index=0),
                              Segment(id="body", role="system", tokens=30_000,
                                      index=1, cache_marked=True, ttl="5m")]))
        return out

    def _clean(self):
        return TraceSet(requests=self._reqs(), tier=Tier.INSTRUMENTED,
                        source="x")

    def _invoice(self, ts):
        return simulate.bake_off(
            ts.analysable, trace=ts,
            allow_unreconciled=True).arms["as-shipped"]["spend"].raw()

    def test_a_trace_that_says_nothing_still_reconciles(self):
        """The control, and the guard against the cap being unconditional."""
        ts = self._clean()
        b = simulate.bake_off(ts.analysable, trace=ts,
                              invoice_usd=self._invoice(ts))
        for _end, p, fig in [(e, p, a["spend"])
                             for e, d in (("pess", b.arms), ("opt", b.optimistic))
                             for p, a in d.items()]:
            self.assertTrue(fig.released, p)
            self.assertEqual(fig.released_as, money.RECONCILED, p)

    def test_an_assumed_input_caps_every_arm_at_draft(self):
        ts = assumed_for(self._clean())
        b = simulate.bake_off(ts.analysable, trace=ts,
                              invoice_usd=self._invoice(ts))
        for end, d in (("pessimistic", b.arms), ("optimistic", b.optimistic)):
            for p, a in d.items():
                self.assertEqual(a["spend"].released_as, money.DRAFT,
                                 f"{end}/{p} published as reconciled over an "
                                 f"assumed pricing input")

    def test_the_cap_does_not_withhold_anything(self):
        """The half that is easy to get wrong in the other direction. Same
        trace, same invoice, same release decision -- only the label moves."""
        clean = self._clean()
        invoice = self._invoice(clean)
        plain = simulate.bake_off(clean.analysable, trace=clean,
                                  invoice_usd=invoice)
        ts = assumed_for(clean)
        capped = simulate.bake_off(ts.analysable, trace=ts,
                                   invoice_usd=invoice)
        for p in simulate.ARMS:
            self.assertTrue(capped.arms[p]["spend"].released,
                            f"{p} was withheld by a label cap")
            self.assertEqual(capped.arms[p]["spend"].raw(),
                             plain.arms[p]["spend"].raw(), p)
        self.assertEqual(capped.delta_pct, plain.delta_pct)
        self.assertEqual(capped.verdict, plain.verdict)

    def test_the_cap_does_not_lift_a_withhold(self):
        """An assumed input is not an excuse to release. A trace that would be
        withheld stays withheld, and does not acquire a DRAFT label as a way
        through."""
        ts = assumed_for(self._clean())
        b = simulate.bake_off(ts.analysable, trace=ts, invoice_usd=999_999.0)
        for p in simulate.ARMS:
            self.assertFalse(b.arms[p]["spend"].released, p)

    def test_the_by_agent_path_inherits_the_assumption(self):
        """Structural, and deliberately not asserted through the output.

        Per-agent figures are DRAFT already, for a reason that has nothing to do
        with this: `bake_off_by_agent` never hands a group the invoice, so no
        group can ever be RECONCILED. Asserting "the groups came back DRAFT"
        would therefore pass with this cap deleted, which is a test of nothing.

        What can be asserted is that the assumption survives the narrowing, so
        the cap is in place if the by-agent path ever does release a reconciled
        figure. That it lands once and travels is the whole point of reading the
        decision off `bake_off` rather than restating it here.
        """
        ts = assumed_for(self._clean())
        narrowed = simulate._agent_trace(ts, "alpha")
        self.assertEqual(getattr(narrowed, "assumed_inputs", ()),
                         ("provider surface",),
                         "the narrowed trace lost the assumption, so a group "
                         "could publish provenance the file cannot support")
        self.assertTrue(all(r.agent == "alpha" for r in narrowed.requests),
                        "the narrowing itself stopped working")

    def test_the_by_agent_path_still_releases_over_an_assumed_input(self):
        """And the cap does not turn into a withhold there either."""
        ts = assumed_for(self._clean())
        out = simulate.bake_off_by_agent(ts.analysable, trace=ts,
                                         allow_unreconciled=True)
        self.assertTrue(out, "no groups came back, so nothing was checked")
        for b in out:
            self.assertTrue(b.arms["as-shipped"]["spend"].released, b.group)
            self.assertEqual(b.arms["as-shipped"]["spend"].released_as,
                             money.DRAFT, b.group)

    def _arm_line(self, b, arm="as-shipped"):
        return next(l for l in str(b).splitlines()
                    if l.strip().startswith(arm))

    def test_the_render_marks_an_assumed_input_run_as_draft(self):
        """Against the string, because the object was already right.

        `Figure.released_as` exists because "a figure released by
        `--allow-unreconciled` was byte-identical to one an invoice had checked
        -- same value, same basis, same string -- and no renderer *could* mark
        one and not the other". `BakeOff.__str__` then did not read it, so the
        state was correct and the output was unchanged: measured, an
        assumed-surface run and a reconciled one produced byte-identical arm
        lines and the word "draft" appeared nowhere in the block.

        That is the shape this repo has shipped before -- a DRAFT banner
        inserted into `<head>`, rendering nowhere, with a test asserting its
        offset and passing. So this asserts what a user reads.
        """
        clean = self._clean()
        invoice = self._invoice(clean)
        plain = simulate.bake_off(clean.analysable, trace=clean,
                                  invoice_usd=invoice)
        ts = assumed_for(clean)
        capped = simulate.bake_off(ts.analysable, trace=ts,
                                   invoice_usd=invoice)
        self.assertNotEqual(self._arm_line(plain), self._arm_line(capped),
                            "an assumed-input run renders identically to a "
                            "reconciled one")
        self.assertIn("DRAFT", self._arm_line(capped),
                      "the dollar cell carries no provenance, so a line copied "
                      "out of this block reads as reconciled")
        self.assertNotIn("DRAFT", self._arm_line(plain))
        text = str(capped)
        self.assertIn("per-arm spend is DRAFT", text)
        self.assertIn("provider surface", text,
                      "the reader is told it is draft but not why")
        self.assertIn("Not for external use", text)

    def test_the_render_marks_an_unreconciled_run_as_draft_too(self):
        """The other route to DRAFT, so the marker is not specific to one
        cause. No invoice plus the override releases figures nothing checked."""
        ts = self._clean()
        b = simulate.bake_off(ts.analysable, trace=ts, allow_unreconciled=True)
        self.assertIn("DRAFT", self._arm_line(b))
        self.assertIn("not been tied to a provider invoice", str(b))

    def test_a_reconciled_render_carries_no_draft_marker(self):
        """So the marker cannot pass by being unconditional."""
        ts = self._clean()
        b = simulate.bake_off(ts.analysable, trace=ts,
                              invoice_usd=self._invoice(ts))
        self.assertNotIn("DRAFT", str(b))
        self.assertIn("$", self._arm_line(b))

    def test_the_cli_shows_the_draft_marker(self):
        """End to end, because the object field was right for a whole round
        while the command printed no sign of it."""
        import io
        import os
        from contextlib import redirect_stdout

        from cacheeconomics import cli

        fixture = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "fixtures", "demo-traces.jsonl")
        args = cli.build_parser().parse_args(
            ["bakeoff", fixture, "--allow-unreconciled"])
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(args.func(args), 0)
        text = out.getvalue()
        self.assertIn("$", text)
        self.assertIn("DRAFT", text,
                      "the shipped command prints draft dollars with nothing "
                      "marking them as draft")

    def test_a_traceset_without_the_field_reads_as_no_assumption(self):
        """Forward and backward compatibility, asserted rather than assumed.

        The field is arriving on another track. Until it does, and for any
        object that predates it, the read has to mean "nothing was assumed" --
        not "unknown, therefore withhold", because that would relabel every
        honest figure in the product on a merge-order accident.
        """
        ts = self._clean()
        self.assertEqual(getattr(ts, "assumed_inputs", ()), ())
        b = simulate.bake_off(ts.analysable, trace=ts,
                              invoice_usd=self._invoice(ts))
        self.assertEqual(b.arms["as-shipped"]["spend"].released_as,
                         money.RECONCILED)

    def test_the_simulator_reads_the_field_the_analyzer_gates_on(self):
        """The parity this branch exists to hold, at the level it can be held
        at today.

        The analyzer's own cap is landing on another track, so there is no
        behavioural twin to compare against on this branch and asserting one
        would be asserting my own stub. What is checkable now is that the
        simulator consults the field by the name the schema gives it -- which is
        what the drift alarm elsewhere in this file does for the structural
        gates, and is what makes the two agree once both halves are in.
        """
        import ast
        import inspect
        import textwrap

        tree = ast.parse(textwrap.dedent(inspect.getsource(simulate.bake_off)))
        names = set()
        for n in ast.walk(tree):
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                    and n.func.id == "getattr" and len(n.args) >= 2
                    and isinstance(n.args[0], ast.Name)
                    and n.args[0].id == "trace"
                    and isinstance(n.args[1], ast.Constant)):
                names.add(n.args[1].value)
        self.assertIn("assumed_inputs", names,
                      "bake_off does not read assumed_inputs off the trace, so "
                      "an assumed pricing input can publish as reconciled")


class TestTheBakeOffIsNeverMorePermissiveThanTheReport(unittest.TestCase):
    """One trace, two dollar-publishing paths, and the simulator may not be the
    lenient one.

    Both paths price money from the same segment sizes. `analyze` has refused to
    attach money to those sizes since it learned to -- on tier, on alignment
    score, on structural coverage by rows and by billed tokens, on whether the
    sums agree with the bill, and on whether they were counted rather than split
    by byte share. `bake_off` could ask none of it: its parameters were `reqs,
    group, on_date, effective_rate, invoice_usd, allow_unreconciled,
    excluded_billed`, and not one of those facts lives on a `Request`, so the
    gates were unreachable rather than merely unsupplied.

    Measured before the fix, on the fixture below -- twelve requests whose
    segment sizes reconcile with the bill exactly, so the size gate already in
    the simulator had nothing to say:

        condition                       analyze VOL-1     bake_off arms
        tokens_counted = 0.0            [withheld]        $2.27 / $1.82
        INFERRED, alignment unmeasured  [withheld]        $2.27 / $1.82
        INFERRED, alignment 0.72        [withheld]        $2.27 / $1.82
        structural_coverage = 0.50      [withheld]        $2.27 / $1.82
        token_sums_publishable = False  [withheld]        $2.27 / $1.82
        tier = USAGE_ONLY               [withheld]        $2.27 / $1.82

    In every row `analyze` withheld $1,366/month of VOL-1 and the bake-off over
    the identical requests printed four per-arm totals with a verdict beside
    them.

    So this asserts the implication rather than any one condition: on the same
    trace, the simulator never releases arm spend where the report withheld
    structural money. The conditions are enumerated because they are what exists
    today; the drift alarm at the bottom is what covers the next one.

    The comparison goes with the dollars, which is the correction to a call I
    first made the other way and recorded as deliberate. Gating only the
    absolutes reproduced the defect this file is named for one layer up: every
    arm rendered `[withheld]` and the block still printed "allocator-lite beats
    the automatic baseline by 50.0% (gate: >=10%)" with a "+0.0% vs
    litellm-auto" column beside it. A reader takes a percentage headline more
    seriously than a dollar figure, not less.

    The measurements that settle it, since "scale-invariant" was the argument
    for keeping the percentage:

      an unprovable lifetime is a guess between 1.25x and 2.0x -- the same
      twelve requests read 20.0% with a provable 5m lifetime and 50.0% when the
      row and the marker disagreed;

      an estimated size is a byte-share split that moves the boundary between
      segments at a fixed billed total -- holding the total at 30,300 tokens and
      moving only the split, the relocation headline ran 83.7% -> 71.6% as the
      volatile head grew from 300 to 6,000 tokens, while the placement headline
      held at exactly 20.0%. Not every arm moves; one Gate 1 verdict moving
      12.1 points is enough, and both are blocked together.
    """

    # Volatile 300-token head in front of a 30,000-token marked body: enough to
    # raise VOL-1, and 300 + 30,000 sums to exactly what the provider billed, so
    # the simulator's own size gate is satisfied and cannot be the thing doing
    # the withholding. Synthetic fixture, not a capture.
    def _reqs(self, n=12):
        return [Request(
            request_id=f"r{i}", sent_at=T0 + timedelta(seconds=60 * i),
            model="claude-opus-5", agent="a", tenant="t", session="s",
            target_id="anthropic/direct", ttl_requested="5m",
            usage={"input_tokens": 300, "cache_creation_input_tokens": 30_000,
                   "cache_read_input_tokens": 0},
            segments=[Segment(id=f"hdr{i}", role="system", tokens=300, index=0),
                      Segment(id="body", role="system", tokens=30_000, index=1,
                              cache_marked=True, ttl="5m")])
            for i in range(n)]

    def _clean(self, **kw):
        return TraceSet(requests=self._reqs(), tier=Tier.INSTRUMENTED,
                        source="x", **kw)

    # The conditions the analyzer's structural gate knows about, each expressed
    # as the trace it would be measured from. Named, because the failure message
    # has to say which one split the two paths apart.
    def _untrusted_traces(self):
        return {
            "sizes estimated rather than counted":
                self._clean(tokens_counted=0.0),
            "inferred structure, alignment never measured":
                TraceSet(requests=self._reqs(), tier=Tier.INFERRED, source="x"),
            "inferred structure, alignment below the floor":
                TraceSet(requests=self._reqs(), tier=Tier.INFERRED,
                         alignment=0.72, source="x"),
            "half the requests carry no structure":
                self._clean(structural_coverage=0.50),
            "segment sums do not agree with the bill":
                self._clean(token_sums_publishable=False),
            "usage-only ingest":
                TraceSet(requests=self._reqs(), tier=Tier.USAGE_ONLY, source="x"),
        }

    def _structural_money(self, a):
        """The MEASURED figure, not the monthly projection.

        The arms price the trace's own requests, so their spend is
        window-scoped. Comparing it against `avoidable_usd_month` was comparing
        a measured amount to an extrapolated one, and only worked while the
        projection floor was global -- once Track A made the floor per-finding,
        this fixture's one-hour window correctly withheld VOL-1's month while
        releasing its window figure, and the parity check failed on the
        mismatch rather than on any permissiveness.

        `avoidable_usd_window` is what the arms are comparable to, and it is
        what the parity claim was always about.
        """
        return [(f.code, f.avoidable_usd_window) for f in a.findings
                if f.structural and f.avoidable_usd_window is not None]

    def _arms(self, b):
        return [(end, p, arm["spend"])
                for end, d in (("pessimistic", b.arms), ("optimistic", b.optimistic))
                for p, arm in d.items()]

    def test_the_fixture_gives_both_paths_a_figure_to_publish(self):
        """Guard the guard. If the trace raised no structural finding, or the
        arms priced nothing, the implication below would hold vacuously -- which
        is how the first version of this file's sibling passed while checking
        nothing."""
        ts = self._clean()
        a = analyze(ts, allow_unreconciled=True)
        figs = self._structural_money(a)
        self.assertTrue(figs, "no structural finding carries money on this "
                              "fixture, so the parity below checks nothing")
        for code, fig in figs:
            self.assertTrue(fig.released, f"{code} withheld on the clean trace")
        b = simulate.bake_off(ts.analysable, group="g", allow_unreconciled=True,
                              excluded_billed=ts.excluded_billed, trace=ts)
        for end, p, fig in self._arms(b):
            self.assertTrue(fig.released, f"{end}/{p} withheld on the clean trace")
            self.assertTrue(fig.raw() > 0, f"{end}/{p} priced nothing")

    def test_the_conditions_actually_make_the_report_withhold(self):
        """And that each named trace reproduces the condition it is named for.
        A trace that quietly stopped triggering the analyzer's gate would turn
        its row below into an assertion about nothing."""
        for label, ts in self._untrusted_traces().items():
            with self.subTest(label):
                figs = self._structural_money(
                    analyze(ts, allow_unreconciled=True))
                self.assertTrue(figs, f"{label}: no structural money to withhold")
                for code, fig in figs:
                    self.assertFalse(
                        fig.released,
                        f"{label}: {code} released, so this row is testing "
                        f"something other than what it claims")

    def test_the_simulator_withholds_wherever_the_report_does(self):
        """The invariant. Same trace, both paths, every arm at both assumption
        ends -- and the simulator is never the permissive one."""
        for label, ts in self._untrusted_traces().items():
            with self.subTest(label):
                b = simulate.bake_off(ts.analysable, group="g",
                                      allow_unreconciled=True,
                                      excluded_billed=ts.excluded_billed,
                                      trace=ts)
                for end, p, fig in self._arms(b):
                    self.assertFalse(
                        fig.released,
                        f"{label}: {end}/{p} published arm spend that `analyze` "
                        f"withheld on the same trace")

    def test_writes_of_unprovable_lifetime_withhold_on_both(self):
        """The other half of the class, found by probing rather than reported.

        `_residual_blocker` names four conditions. Three of them already stopped
        the bake-off one layer earlier, because a request with no timestamp or no
        price contributes to no arm and lands in `omitted`. The fourth does not:
        a marked block whose lifetime the trace never states still replays
        perfectly well, so the simulator priced it and published.

        Measured on twelve requests, sizes reconciling exactly with the bill:

          marked block, no lifetime anywhere -- `analyze` withheld, bake-off
          printed $2.2725, which is the 5m guess; a 1h write is 2.0x against a
          5m write's 1.25x, so the guess runs in the flattering direction.

          row says 5m and the marker says 1h, which `_declared_ttl` calls
          unprovable because one of the two is stale -- `analyze` withheld,
          bake-off printed $3.6360 and a 50.0% headline where the same trace
          with a provable lifetime reads 20.0%.
        """
        for label, row_ttl, seg_ttl in (
                ("no lifetime anywhere", None, None),
                ("the row and the marker disagree", "5m", "1h")):
            with self.subTest(label):
                reqs = [replace(
                    r, ttl_requested=row_ttl,
                    segments=[replace(s, ttl=seg_ttl) if s.cache_marked else s
                              for s in r.segments])
                    for r in self._reqs()]
                ts = TraceSet(requests=reqs, tier=Tier.INSTRUMENTED, source="x")
                a = analyze(ts, allow_unreconciled=True)
                self.assertFalse(a.spend["input_usd"].released,
                                 f"{label}: the report released, so this case is "
                                 f"no longer testing what it claims")
                b = simulate.bake_off(ts.analysable, group="g",
                                      allow_unreconciled=True,
                                      excluded_billed=ts.excluded_billed,
                                      trace=ts)
                for end, p, fig in self._arms(b):
                    self.assertFalse(
                        fig.released,
                        f"{label}: {end}/{p} priced a write whose lifetime the "
                        f"trace never established and published it")
                self.assertIn("lifetime the trace never established",
                              b.arms["as-shipped"]["spend"].withheld_because)

    def test_the_billed_coverage_floor_is_asked_even_though_it_is_shadowed(self):
        """Nine small structured requests beside one enormous unstructured one:
        the shape that published $3,175 a month from 8.3% of the bill.

        Today the bake-off would refuse this anyway, one layer earlier -- the
        unstructured request contributes to no arm and lands in `omitted` -- so
        the assertion is on the gate rather than on the arms. Stated that way on
        purpose: a test asserting the arms would pass with the billed-coverage
        check deleted, which is a test of the wrong thing.
        """
        reqs = self._reqs(n=11)
        reqs.append(Request(
            request_id="huge", sent_at=T0 + timedelta(seconds=99999),
            model="claude-opus-5", agent="a", tenant="t", session="s",
            target_id="anthropic/direct", ttl_requested="5m",
            usage={"input_tokens": 5_000_000, "cache_creation_input_tokens": 0,
                   "cache_read_input_tokens": 0}, segments=[]))
        ts = TraceSet(requests=reqs, tier=Tier.INSTRUMENTED, source="x")
        self.assertEqual(ts.structural_coverage, 1.0,
                         "the row-count figure has to look healthy, or this is "
                         "not the shape the billed figure exists to catch")
        self.assertLess(ts.structural_coverage_billed, 0.10)
        ok, why = simulate.structural_evidence(ts)
        self.assertFalse(ok)
        self.assertIn("billed input tokens", why)

    def test_no_gate_one_verdict_survives_withheld_evidence(self):
        """The headline goes with the dollars when the *evidence* is what failed.

        Stated precisely, because "any withheld arm means no percentage" is a
        stronger claim than the code makes and than it should make. A run with
        no invoice withholds every arm and keeps its percentage on purpose:
        nothing about the trace is in doubt there, only the tie to money that
        left an account, and the comparison is scale-invariant to that.
        `test_dollars_and_percentages_are_gated_separately` in test_bakeoff.py
        is that case and it must keep passing.

        The rule is narrower: when what is missing is evidence the *comparison*
        rests on -- a lifetime the arms had to guess, boundaries nobody scored,
        sizes nobody counted, a denominator with holes in it -- the percentage
        goes too. Every case below is run with `allow_unreconciled=True` so the
        invoice cannot be the blocker and the condition under test is the only
        one left.

        The first version of this fix gated `spend_ok` alone and left
        `delta_pct` and `_verdict(...)` on the normal return, so four
        `[withheld]` arms sat under "beats the automatic baseline by 50.0%".
        """
        cases = dict(self._untrusted_traces())
        cases["writes of unprovable lifetime"] = TraceSet(
            requests=[replace(r, ttl_requested=None,
                              segments=[replace(s, ttl=None) if s.cache_marked
                                        else s for s in r.segments])
                      for r in self._reqs()],
            tier=Tier.INSTRUMENTED, source="x")
        for label, ts in cases.items():
            with self.subTest(label):
                b = simulate.bake_off(ts.analysable, group="g",
                                      allow_unreconciled=True, trace=ts)
                held = [p for p, a in b.arms.items() if not a["spend"].released]
                self.assertTrue(held, f"{label}: nothing was withheld, so this "
                                      f"row asserts nothing")
                self.assertIsNone(b.delta_pct, f"{label}: placement percentage")
                self.assertIsNone(b.delta_pct_relocation,
                                  f"{label}: relocation percentage")
                self.assertIsNone(b.delta_pct_optimistic, f"{label}: optimistic")
                self.assertIsNone(b.delta_pct_relocation_optimistic, label)
                for v in (b.verdict, b.verdict_relocation):
                    self.assertIn("indeterminate", v, f"{label}: {v[:60]}")
                    self.assertNotIn("beats the automatic baseline", v, label)
                text = str(b)
                self.assertNotIn("vs litellm-auto", text,
                                 f"{label}: the inline percentage column "
                                 f"printed beside withheld arms")

    def test_a_missing_invoice_is_the_deliberate_exception(self):
        """The boundary of the rule above, pinned so it cannot drift into it.

        No invoice withholds the dollars and keeps the comparison: the trace is
        not in doubt, only the tie to money. If this ever starts refusing, the
        gate above has grown past what its evidence supports and the percentage
        Gate 1 reads has been withheld for a reason that does not touch it.
        """
        ts = self._clean()
        b = simulate.bake_off(ts.analysable, group="g", trace=ts)
        self.assertFalse(b.arms["as-shipped"]["spend"].released)
        self.assertIn("no invoice was supplied",
                      b.arms["as-shipped"]["spend"].withheld_because)
        self.assertIsNotNone(b.delta_pct)
        self.assertNotIn("indeterminate", b.verdict)

    def test_the_invoice_is_not_an_input_to_what_the_arms_compute(self):
        """The measurement, separated from the conclusion I wrongly drew from it.

        I used this to defend keeping the percentage for *any* invoice state,
        including a failed reconciliation. The measurement was right and the
        inference was not, and the two are worth keeping apart:

        What is true, and what this asserts -- the invoice is not an input. It
        is compared against the as-shipped arm after every arm has been priced
        and never enters a spend calculation, so the arms' raw numbers are
        bit-identical whether it is absent, exact, a thousand times too large, a
        cent, or negative.

        What does not follow -- that a failed reconciliation is therefore
        harmless to the comparison. A failed reconciliation is not the invoice
        acting as an input; it is evidence that the export and the bill may not
        describe the same workload, which means rows are missing from one of
        them, and missing rows are precisely what this module already treats as
        moving the arms differentially. That case is asserted below.
        """
        ts = self._clean()
        exact = simulate.bake_off(ts.analysable, trace=ts,
                                  allow_unreconciled=True
                                  ).arms["as-shipped"]["spend"].raw()
        seen = set()
        for label, kw in (("no invoice, override", {"allow_unreconciled": True}),
                          ("no invoice, no flag", {}),
                          ("exact", {"invoice_usd": exact}),
                          ("wildly high", {"invoice_usd": 999_999.0}),
                          ("a cent", {"invoice_usd": 0.01}),
                          ("negative", {"invoice_usd": -5.0})):
            with self.subTest(label):
                b = simulate.bake_off(ts.analysable, trace=ts, **kw)
                # `raw()` reads through a withheld figure, which is what makes
                # this measurable at all: the question is what the arms computed,
                # not what they were allowed to print.
                seen.add(tuple(round(a["spend"].raw(), 10)
                               for _, a in sorted(b.arms.items())))
        self.assertEqual(
            len(seen), 1,
            f"the invoice moved what the arms computed across {len(seen)} "
            f"distinct outcomes, so it is an input after all: {sorted(seen)}")

    def test_only_a_missing_invoice_keeps_the_percentage(self):
        """The carve-out, narrowed to what the evidence actually supports.

        Absence of a bill says nothing about the trace, so the comparison
        stands. A bill that was supplied and disagrees says the export and the
        bill may not describe the same workload -- rows missing from one of
        them, which could behave any way at all. That is the same fact
        `excluded_billed` and `omitted` block the percentage for, arriving
        without a list of which rows.
        """
        ts = self._clean()
        keeps = simulate.bake_off(ts.analysable, group="g", trace=ts)
        self.assertIsNotNone(keeps.delta_pct)
        self.assertFalse(keeps.arms["as-shipped"]["spend"].released)
        for label, inv in (("wildly wrong", 999_999.0), ("zero", 0.0),
                           ("negative", -5.0), ("not a number", float("nan"))):
            with self.subTest(label):
                b = simulate.bake_off(ts.analysable, group="g", trace=ts,
                                      invoice_usd=inv)
                self.assertIsNone(b.delta_pct, label)
                self.assertIsNone(b.delta_pct_relocation, label)
                self.assertIn("indeterminate", b.verdict)
                self.assertNotIn("beats the automatic baseline", b.verdict)

    def test_the_override_does_not_resurrect_a_failed_reconciliation(self):
        """`allow_unreconciled` covers a missing invoice and nothing else, and
        that has to hold for the percentage now that it holds for the dollars.
        Otherwise the flag becomes a way to print a Gate 1 pass over a bill the
        trace contradicts."""
        ts = self._clean()
        b = simulate.bake_off(ts.analysable, group="g", trace=ts,
                              invoice_usd=999_999.0, allow_unreconciled=True)
        self.assertIsNone(b.delta_pct)
        self.assertIn("indeterminate", b.verdict)

    def test_the_verdict_still_names_the_specific_blocker(self):
        """Refusing is not enough if the refusal sends the reader nowhere. Each
        condition has to say which one it was, or "indeterminate" becomes the
        tool shrugging."""
        expect = {
            "sizes estimated rather than counted": "estimated rather than counted",
            "inferred structure, alignment never measured": "unmeasured",
            "inferred structure, alignment below the floor": "72%",
            "half the requests carry no structure": "50% of requests",
            "segment sums do not agree with the bill": "do not sum to",
        }
        for label, ts in self._untrusted_traces().items():
            if label not in expect:
                continue
            with self.subTest(label):
                b = simulate.bake_off(ts.analysable, group="g",
                                      allow_unreconciled=True, trace=ts)
                self.assertIn(expect[label], b.verdict)

    # --- the trace is the argument, not a hint beside it --------------------

    def _only_trace(self, ts):
        """The call a caller actually writes once `trace=` exists.

        No `excluded_billed=`. That argument being independent of `trace` is
        what made the first version of this parity claim false: every test that
        was supposed to prove it passed both, so the one shape that could fail
        was the one nobody exercised.
        """
        return simulate.bake_off(ts.analysable, group="g",
                                 allow_unreconciled=True, trace=ts)

    def test_a_failed_but_billed_row_reaches_the_gate_through_the_trace_alone(self):
        reqs = self._reqs()
        reqs.append(Request(
            request_id="failed", sent_at=T0 + timedelta(seconds=99999),
            model="claude-opus-5", agent="a", tenant="t", session="s",
            target_id="anthropic/direct", ttl_requested="5m", status=500,
            usage={"input_tokens": 5_000_000, "cache_creation_input_tokens": 0,
                   "cache_read_input_tokens": 0},
            segments=[Segment(id="hdr", role="system", tokens=5_000_000,
                              index=0)]))
        ts = TraceSet(requests=reqs, tier=Tier.INSTRUMENTED, source="x")
        self.assertEqual(ts.excluded_billed, {"failed but billed": 1})
        b = self._only_trace(ts)
        for end, p, fig in self._arms(b):
            self.assertFalse(fig.released, f"{end}/{p}")
        self.assertIn("failed but billed",
                      b.arms["as-shipped"]["spend"].withheld_because)

    def test_a_row_with_no_usage_reaches_the_gate_through_the_trace_alone(self):
        reqs = self._reqs()
        reqs.append(Request(
            request_id="blind", sent_at=T0 + timedelta(seconds=99999),
            model="claude-opus-5", agent="a", tenant="t", session="s",
            target_id="anthropic/direct", ttl_requested="5m",
            usage={}, segments=[]))
        ts = TraceSet(requests=reqs, tier=Tier.INSTRUMENTED, source="x")
        self.assertEqual(ts.excluded_billed, {"carrying no usage fields": 1})
        for end, p, fig in self._arms(self._only_trace(ts)):
            self.assertFalse(fig.released, f"{end}/{p}")

    def test_a_skipped_row_reaches_the_gate_through_the_trace_alone(self):
        ts = TraceSet(requests=self._reqs(), tier=Tier.INSTRUMENTED, source="x",
                      skipped_rows=1)
        self.assertEqual(ts.excluded_billed, {"unreadable in the export": 1})
        for end, p, fig in self._arms(self._only_trace(ts)):
            self.assertFalse(fig.released, f"{end}/{p}")

    def test_an_invoice_that_reconciles_does_not_rescue_an_excluded_row(self):
        """The shape the reviewer reproduced. `analyze` calls this "a subset
        that happens to agree rather than a reconciliation"; the bake-off used
        to call it $2.27."""
        reqs = self._reqs()
        reqs.append(Request(
            request_id="failed", sent_at=T0 + timedelta(seconds=99999),
            model="claude-opus-5", agent="a", tenant="t", session="s",
            target_id="anthropic/direct", ttl_requested="5m", status=500,
            usage={"input_tokens": 5_000_000, "cache_creation_input_tokens": 0,
                   "cache_read_input_tokens": 0},
            segments=[Segment(id="hdr", role="system", tokens=5_000_000,
                              index=0)]))
        ts = TraceSet(requests=reqs, tier=Tier.INSTRUMENTED, source="x")
        priced = simulate.bake_off(ts.analysable, trace=ts,
                                   allow_unreconciled=True)
        invoice = priced.arms["as-shipped"]["spend"].raw()
        a = analyze(ts, invoice_usd=invoice)
        b = simulate.bake_off(ts.analysable, group="g", trace=ts,
                              invoice_usd=invoice)
        self.assertFalse(a.spend["input_usd"].released)
        self.assertFalse(b.arms["as-shipped"]["spend"].released,
                         "the bake-off released against an invoice the report "
                         "called a coincidence over a partial denominator")

    def test_handing_over_an_unmodellable_row_does_not_launder_it(self):
        """The trap the previous version of this test walked into.

        It asserted that rows the caller passed stop counting as excluded, on
        the reasoning that "what blocks a figure is a billed row the arms never
        saw". That reasoning treats presence in `reqs` as evidence a row is safe
        to model, and it is not: a failed call populated no cache entry and a
        no-usage row has no counters to reconcile, so handing them over means
        every arm has now modelled a cache entry that never existed. Worse, not
        better.

        Measured on the version that shipped it, twelve good requests plus one
        structured status-500 billing 5,000,000 input tokens:
        `bake_off(ts.analysable, trace=ts)` withheld, and `bake_off(ts.requests,
        trace=ts)` released spend and printed a Gate 1 verdict. With a no-usage
        row instead of the failed one it printed "beats the automatic baseline
        by 20.0%".
        """
        for label, stray in (
                ("failed but billed", Request(
                    request_id="failed", sent_at=T0 + timedelta(seconds=99999),
                    model="claude-opus-5", agent="a", tenant="t", session="s",
                    target_id="anthropic/direct", ttl_requested="5m", status=500,
                    usage={"input_tokens": 5_000_000,
                           "cache_creation_input_tokens": 0,
                           "cache_read_input_tokens": 0},
                    segments=[Segment(id="hdrF", role="system",
                                      tokens=5_000_000, index=0)])),
                ("carrying no usage fields", Request(
                    request_id="blind", sent_at=T0 + timedelta(seconds=99998),
                    model="claude-opus-5", agent="a", tenant="t", session="s",
                    target_id="anthropic/direct", ttl_requested="5m",
                    usage={}, segments=[Segment(id="hdrB", role="system",
                                                tokens=100, index=0)]))):
            with self.subTest(label):
                reqs = self._reqs() + [stray]
                ts = TraceSet(requests=reqs, tier=Tier.INSTRUMENTED, source="x")
                self.assertTrue(ts.excluded_billed,
                                f"{label}: the fixture stopped reproducing")
                b = simulate.bake_off(ts.requests, group="g",
                                      allow_unreconciled=True, trace=ts)
                for end, p, fig in self._arms(b):
                    self.assertFalse(fig.released, f"{label}: {end}/{p}")
                self.assertIsNone(b.delta_pct, label)
                self.assertIn("indeterminate", b.verdict)

    def test_the_subset_invariant_is_about_requests_not_their_names(self):
        """It was checked by `request_id`, which is a statement about what a
        request is called rather than what it is.

        Three ways past it, all measured on the same clean twelve-row trace that
        releases at 20.0%. The first two matter most: they carry usage that
        still agrees with their segments, so `misscaled` has nothing to say and
        the invariant is the only thing that could object.
        """
        ts = self._clean()
        clean = list(ts.analysable)
        cases = {
            # A failed call populated no cache entry, so modelling it as if it
            # had is the whole reason the invariant exists.
            "status-500 wearing an allowed id":
                [replace(clean[0], status=500)] + clean[1:],
            "no-usage row wearing an allowed id":
                [replace(clean[0], usage={})] + clean[1:],
            # Cardinality, not membership: every id is allowed, one is used
            # three times where the trace contains it once.
            "one row presented three times":
                clean + [clean[0], clean[0]],
        }
        for label, reqs in cases.items():
            with self.subTest(label):
                b = simulate.bake_off(reqs, group="g", allow_unreconciled=True,
                                      trace=ts)
                for end, p, fig in self._arms(b):
                    self.assertFalse(fig.released, f"{label}: {end}/{p}")
                self.assertIsNone(b.delta_pct, label)
                self.assertIn("not in the trace's analysable set", b.verdict)

    def test_the_invariant_accepts_equal_rows_from_a_second_load(self):
        """And it must not be identity-only, or re-reading the same file makes
        every row a stray. Equal-but-distinct objects are the same requests."""
        ts = self._clean()
        copies = [replace(r) for r in ts.analysable]
        self.assertFalse(any(a is b for a, b in zip(copies, ts.analysable)))
        b = simulate.bake_off(copies, group="g", allow_unreconciled=True,
                              trace=ts)
        self.assertIsNotNone(b.delta_pct)
        self.assertTrue(b.arms["as-shipped"]["spend"].released)

    def test_an_exclusion_owned_by_an_agent_with_no_group_still_blocks(self):
        """Narrowing used the set I was handed rather than the set that exists.

        Groups are built from `reqs` and only survive at three rows, so a failed
        billed row belonging to an agent with no analysable traffic was filtered
        out of every other agent's narrowed trace and given no group of its own.
        Measured: alpha and beta clean with six requests each, one failed billed
        row belonging to gamma, `trace.excluded_billed` non-empty -- and both
        groups released at 20.0% with no gamma result anywhere.
        """
        reqs = []
        for agent in ("alpha", "beta"):
            for i in range(6):
                reqs.append(Request(
                    request_id=f"{agent}{i}",
                    sent_at=T0 + timedelta(seconds=60 * i),
                    model="claude-opus-5", agent=agent, tenant="t",
                    session=f"s{agent}", target_id="anthropic/direct",
                    ttl_requested="5m",
                    usage={"input_tokens": 300,
                           "cache_creation_input_tokens": 30_000,
                           "cache_read_input_tokens": 0},
                    segments=[Segment(id=f"hdr{agent}{i}", role="system",
                                      tokens=300, index=0),
                              Segment(id="body", role="system", tokens=30_000,
                                      index=1, cache_marked=True, ttl="5m")]))
        reqs.append(Request(
            request_id="gamma-failed", sent_at=T0 + timedelta(seconds=99999),
            model="claude-opus-5", agent="gamma", tenant="t", session="sg",
            target_id="anthropic/direct", ttl_requested="5m", status=500,
            usage={"input_tokens": 5_000_000, "cache_creation_input_tokens": 0,
                   "cache_read_input_tokens": 0},
            segments=[Segment(id="hg", role="system", tokens=5_000_000,
                              index=0)]))
        ts = TraceSet(requests=reqs, tier=Tier.INSTRUMENTED, source="x")
        self.assertEqual(ts.excluded_billed, {"failed but billed": 1})
        out = {b.group: b for b in simulate.bake_off_by_agent(
            ts.analysable, allow_unreconciled=True, trace=ts)}
        self.assertEqual(set(out), {"alpha", "beta"},
                         "gamma has no analysable rows, so it gets no group -- "
                         "which is exactly why its excluded row has to land "
                         "somewhere else")
        for g, b in out.items():
            self.assertFalse(b.arms["as-shipped"]["spend"].released,
                             f"{g} released over a billed row nobody reports")
            self.assertIsNone(b.delta_pct, g)

    def test_the_two_entry_points_mean_different_things_by_a_narrower_reqs(self):
        """The contract, resolved rather than patched.

        `bake_off` permits a narrower `reqs`: it is one comparison over whatever
        you hand it, and its exclusions come from the trace you hand it. A group
        produced internally is exactly such a call, with a narrowed trace.

        `bake_off_by_agent` does not, because it partitions, and a partition of a
        workload requires the workload. Handing it alpha's rows with the whole
        file's trace does not say whether you mean "alpha out of this workload"
        or "alpha's workload", and those disagree about whose excluded rows
        count: measured, alpha's six clean requests came back indeterminate with
        the full trace and released at 20.0% with the trace narrowed first, on
        rows that were identical.

        The contract is `reqs == trace.analysable`, not "the same agents". That
        distinction is load-bearing and the next test is why.
        """
        reqs = []
        for agent in ("alpha", "beta"):
            for i in range(6):
                reqs.append(Request(
                    request_id=f"{agent}{i}",
                    sent_at=T0 + timedelta(seconds=60 * i),
                    model="claude-opus-5", agent=agent, tenant="t",
                    session=f"s{agent}", target_id="anthropic/direct",
                    ttl_requested="5m",
                    usage={"input_tokens": 300,
                           "cache_creation_input_tokens": 30_000,
                           "cache_read_input_tokens": 0},
                    segments=[Segment(id=f"h{agent}{i}", role="system",
                                      tokens=300, index=0),
                              Segment(id="body", role="system", tokens=30_000,
                                      index=1, cache_marked=True, ttl="5m")]))
        ts = TraceSet(requests=reqs, tier=Tier.INSTRUMENTED, source="x")
        alpha = [r for r in ts.analysable if r.agent == "alpha"]

        # Narrower reqs, whole trace: a breach, and it says what to do.
        out = simulate.bake_off_by_agent(alpha, allow_unreconciled=True,
                                         trace=ts)
        self.assertTrue(out)
        for b in out:
            self.assertIsNone(b.delta_pct)
            self.assertIn("not handed over", b.verdict)
            self.assertIn("Narrow the trace", b.verdict)

        # The same rows with the trace narrowed to match: fine.
        scoped = replace(ts, requests=alpha)
        out = simulate.bake_off_by_agent(scoped.analysable,
                                         allow_unreconciled=True, trace=scoped)
        self.assertTrue(out)
        for b in out:
            self.assertIsNotNone(b.delta_pct)
            self.assertTrue(b.arms["as-shipped"]["spend"].released)

        # And `bake_off` itself still accepts the narrower slice, because that
        # is a different question with a different answer.
        one = simulate.bake_off(alpha, group="alpha", allow_unreconciled=True,
                                trace=ts)
        self.assertIsNotNone(one.delta_pct)

    def test_the_cli_shape_is_not_a_breach(self):
        """`ts.analysable` with the full `ts` is what `cmd_bakeoff` passes, and
        it must stay legal even when an agent's traffic all failed.

        `analysable` legitimately drops such an agent, so "the same agent
        universe" would call the CLI's own shape a breach and send the reader to
        fix their call when the finding is in their export. The row still has to
        block -- it just has to block as an unreported exclusion, with the
        reason naming the export.
        """
        reqs = []
        for agent in ("alpha", "beta"):
            for i in range(6):
                reqs.append(Request(
                    request_id=f"{agent}{i}",
                    sent_at=T0 + timedelta(seconds=60 * i),
                    model="claude-opus-5", agent=agent, tenant="t",
                    session=f"s{agent}", target_id="anthropic/direct",
                    ttl_requested="5m",
                    usage={"input_tokens": 300,
                           "cache_creation_input_tokens": 30_000,
                           "cache_read_input_tokens": 0},
                    segments=[Segment(id=f"h{agent}{i}", role="system",
                                      tokens=300, index=0),
                              Segment(id="body", role="system", tokens=30_000,
                                      index=1, cache_marked=True, ttl="5m")]))
        reqs.append(Request(
            request_id="gamma-failed", sent_at=T0 + timedelta(seconds=99999),
            model="claude-opus-5", agent="gamma", tenant="t", session="sg",
            target_id="anthropic/direct", ttl_requested="5m", status=500,
            usage={"input_tokens": 5_000_000, "cache_creation_input_tokens": 0,
                   "cache_read_input_tokens": 0},
            segments=[Segment(id="hg", role="system", tokens=5_000_000,
                              index=0)]))
        ts = TraceSet(requests=reqs, tier=Tier.INSTRUMENTED, source="x")
        out = simulate.bake_off_by_agent(ts.analysable, allow_unreconciled=True,
                                         trace=ts)
        self.assertTrue(out)
        for b in out:
            self.assertNotIn("not handed over", b.verdict,
                             "the CLI's own shape was reported as a breach")
            self.assertIn("never reached the arms", b.verdict)
            self.assertIsNone(b.delta_pct)

    def _agents(self, spec):
        """`spec` maps agent name -> request count, all clean."""
        out = []
        for agent, n in spec.items():
            for i in range(n):
                out.append(Request(
                    request_id=f"{agent}{i}",
                    sent_at=T0 + timedelta(seconds=60 * i),
                    model="claude-opus-5", agent=agent, tenant="t",
                    session=f"s{agent}", target_id="anthropic/direct",
                    ttl_requested="5m",
                    usage={"input_tokens": 300,
                           "cache_creation_input_tokens": 30_000,
                           "cache_read_input_tokens": 0},
                    segments=[Segment(id=f"h{agent}{i}", role="system",
                                      tokens=300, index=0),
                              Segment(id="body", role="system", tokens=30_000,
                                      index=1, cache_marked=True, ttl="5m")]))
        return out

    def test_a_sub_threshold_agents_rows_are_not_dropped_silently(self):
        """The by-agent path read one field of the whole result.

        `_unreported_exclusions` covers billed rows that are not analysable. It
        does not cover *analysable* rows nobody reports, and an agent below the
        three-request minimum is dropped without a word: its requests contribute
        to no group, appear in no output, and qualify nothing.

        This is the CLI's own shape -- `reqs == trace.analysable`, no breach.
        Measured before: the whole-workload run returned "indeterminate: 1 of 7
        requests contributed nothing" and the by-agent run printed alpha at
        20.0% with draft dollars beside it.

        The rows are now *reported* rather than turned into a blocker on
        everyone else, which is the stronger fix and the one that survives a
        determinate-but-bad dropped row. Alpha's answer about alpha stands --
        blanking it would be the round-2 finding again -- and the dropped row
        appears as its own result with its own verdict and its own spend.
        """
        reqs = self._agents({"alpha": 6})
        reqs.append(Request(
            request_id="solo0", sent_at=T0 + timedelta(seconds=999),
            model="claude-opus-5", agent="solo", tenant="t", session="ssolo",
            target_id="anthropic/direct", ttl_requested="5m",
            usage={"input_tokens": 40_000, "cache_creation_input_tokens": 0,
                   "cache_read_input_tokens": 0},
            segments=[]))
        ts = TraceSet(requests=reqs, tier=Tier.INSTRUMENTED, source="x")
        self.assertEqual(len(ts.analysable), len(reqs),
                         "the fixture must be the CLI's own shape, or this "
                         "tests a breach instead of the hole")
        self.assertEqual(ts.excluded_billed, {},
                         "the row must be analysable, or the older derivation "
                         "catches it and this proves nothing")
        whole = simulate.bake_off(ts.analysable, trace=ts,
                                  allow_unreconciled=True)
        self.assertIsNone(whole.delta_pct,
                          "the whole workload must be indeterminate here")
        out = {b.group: b for b in simulate.bake_off_by_agent(
            ts.analysable, trace=ts, allow_unreconciled=True)}
        dropped = [g for g in out if g.startswith("(unreported")]
        self.assertEqual(len(dropped), 1,
                         f"the dropped row is not reported anywhere: {list(out)}")
        label = dropped[0]
        self.assertIn("solo", label, "the label does not name the agent")
        self.assertIn("1 request(s)", label)
        # Its own verdict, not a summary of it: this row contributed to no arm
        # and the result says so where a reader will see it.
        self.assertIn("contributed nothing", out[label].verdict)
        # And the clean group is not blanked by it.
        alpha = out["alpha"]
        self.assertIsNotNone(alpha.delta_pct)
        self.assertTrue(alpha.arms["as-shipped"]["spend"].released)

    def test_a_dropped_row_that_is_clean_and_bad_is_still_reported(self):
        """The case a blocker could never have caught.

        The previous fix inherited `dropped.blocked_because`, which is empty for
        a *determinate* result -- so a sub-threshold agent that was perfectly
        clean and simply expensive vanished. Measured: six clean alpha requests
        plus one clean single-request agent sending an unmarked million-token
        prompt. The whole workload comes to -16.7%, that row on its own to
        -25.0%, and the by-agent output printed alpha at +20.0% and nothing
        else. The sign of the answer flips and the row moving it by 37 points is
        not mentioned.

        Reading a lossy summary of a comparison is what failed twice here. The
        row is reported now, so there is nothing to summarise.
        """
        reqs = self._agents({"alpha": 6})
        reqs.append(Request(
            request_id="solo0", sent_at=T0 + timedelta(seconds=999),
            model="claude-opus-5", agent="solo", tenant="t", session="ssolo",
            target_id="anthropic/direct", ttl_requested="5m",
            usage={"input_tokens": 1_000_000,
                   "cache_creation_input_tokens": 0,
                   "cache_read_input_tokens": 0},
            segments=[Segment(id="hsolo", role="system", tokens=1_000_000,
                              index=0)]))
        ts = TraceSet(requests=reqs, tier=Tier.INSTRUMENTED, source="x")
        whole = simulate.bake_off(ts.analysable, trace=ts,
                                  allow_unreconciled=True)
        self.assertEqual(whole.blocked_because, "",
                         "the whole run must be DETERMINATE, or a blocker "
                         "would have caught this and the test proves nothing")
        self.assertLess(whole.delta_pct, 0,
                        "the workload must come out negative, or the sign does "
                        "not flip and there is nothing to hide")
        out = {b.group: b for b in simulate.bake_off_by_agent(
            ts.analysable, trace=ts, allow_unreconciled=True)}
        dropped = [g for g in out if g.startswith("(unreported")]
        self.assertEqual(len(dropped), 1,
                         f"a clean but expensive dropped row vanished: "
                         f"{list(out)}")
        label = dropped[0]
        self.assertIn("solo", label)
        self.assertLess(out[label].delta_pct, 0,
                        "the dropped row's own verdict is not reported")
        self.assertTrue(out[label].arms["as-shipped"]["spend"].released,
                        "the dropped row's spend is not reported either")
        # And it reaches the render, not just the object.
        self.assertIn("solo", str(out[label]))

    def test_the_invoice_note_is_not_stated_over_an_indeterminate_whole(self):
        """The worst part of that finding: a sentence asserting something the
        object it came from declined to assert.

        `reconciled` is one field of the whole result. A run can reconcile
        against an invoice that matched a subtotal computed over rows
        contributing nothing, and reading that field alone put "the invoice
        reconciles against the whole workload" directly beneath a verdict of
        "indeterminate: 1 of 7 requests contributed nothing".

        The fixture is beta-is-misscaled rather than the dropped-agent one, and
        that matters. With a dropped agent the unreported-rows blocker fires
        first and this assertion passes with the guard deleted -- checked, and
        it did. Here every row belongs to a reported group, so nothing else can
        be doing the work: beta diagnoses its own sizes, alpha is clean, and the
        whole workload is indeterminate with no unreported rows anywhere.
        """
        reqs = self._agents({"alpha": 6})
        for i in range(6):
            reqs.append(Request(
                request_id=f"beta{i}", sent_at=T0 + timedelta(seconds=60 * i),
                model="claude-opus-5", agent="beta", tenant="t", session="sb",
                target_id="anthropic/direct", ttl_requested="5m",
                usage={"input_tokens": 300,
                       "cache_creation_input_tokens": 30_000,
                       "cache_read_input_tokens": 0},
                segments=[Segment(id=f"hb{i}", role="system", tokens=300,
                                  index=0),
                          Segment(id="body", role="system", tokens=3_000_000,
                                  index=1, cache_marked=True, ttl="5m")]))
        ts = TraceSet(requests=reqs, tier=Tier.INSTRUMENTED, source="x")
        subtotal = simulate.bake_off(
            ts.analysable, trace=ts,
            allow_unreconciled=True).arms["as-shipped"]["spend"].raw()
        checked = simulate.bake_off(ts.analysable, trace=ts,
                                    invoice_usd=subtotal)
        self.assertTrue(checked.reconciled,
                        "the invoice must match the subtotal, or the "
                        "contradiction this guards cannot arise")
        self.assertTrue(checked.blocked_because,
                        "and the whole result must still be indeterminate")
        out = simulate.bake_off_by_agent(ts.analysable, trace=ts,
                                         invoice_usd=subtotal)
        self.assertEqual({b.group for b in out}, {"alpha", "beta"},
                         "every row must belong to a reported group, or the "
                         "unreported-rows blocker does this test's work")
        for b in out:
            why = b.arms["as-shipped"]["spend"].withheld_because
            self.assertNotIn("reconciles against the whole workload", why,
                             f"{b.group} asserted a reconciliation the whole "
                             f"result refused")
            self.assertNotIn("no invoice was supplied", why,
                             f"{b.group} was told no invoice was supplied when "
                             f"one was")

    def test_extra_rows_on_a_sub_threshold_agent_are_still_a_breach(self):
        """The equality check computed both halves of the match and blocked on
        one. A caller passing every analysable row *plus* extra rows attributed
        to an agent that never reaches the minimum had those rows filtered out
        by the group threshold, so no per-group call ever saw them and the
        remaining groups released over a `reqs` that was not the analysable set
        at all. `bake_off` catches this; by-agent discarded its verdict."""
        clean = self._agents({"alpha": 6, "beta": 6})
        ts = TraceSet(requests=clean, tier=Tier.INSTRUMENTED, source="x")
        ghost = Request(
            request_id="ghost0", sent_at=T0 + timedelta(seconds=5000),
            model="claude-opus-5", agent="ghost", tenant="t", session="sg",
            target_id="anthropic/direct", ttl_requested="5m",
            usage={"input_tokens": 300, "cache_creation_input_tokens": 30_000,
                   "cache_read_input_tokens": 0},
            segments=[Segment(id="hg", role="system", tokens=300, index=0),
                      Segment(id="body", role="system", tokens=30_000, index=1,
                              cache_marked=True, ttl="5m")])
        passed = list(ts.analysable) + [ghost]
        self.assertIsNone(
            simulate.bake_off(passed, trace=ts,
                              allow_unreconciled=True).delta_pct,
            "the single-arm path must already refuse this")
        out = simulate.bake_off_by_agent(passed, trace=ts,
                                         allow_unreconciled=True)
        self.assertTrue(out)
        for b in out:
            self.assertIsNone(b.delta_pct, b.group)
            self.assertFalse(b.arms["as-shipped"]["spend"].released, b.group)
            # The kept groups carry the inherited breach and the dropped-rows
            # group carries its own, in slightly different words. Both name the
            # analysable set, which is the fact that matters.
            self.assertIn("analysable set", b.verdict, b.group)

    def test_a_defect_inside_one_kept_group_does_not_blank_another(self):
        """The scoping this function spent three rounds learning, guarded
        against the fix above.

        `whole` is indeterminate here too -- beta's sizes disagree with its bill
        -- but beta is reported and diagnoses that itself. Inheriting the whole
        verdict wholesale would blank alpha for a defect in beta, which is the
        round-2 finding arriving from the opposite direction. What a group
        cannot see is a row it does not have; that is the only thing inherited.
        """
        reqs = self._agents({"alpha": 6})
        for i in range(6):
            reqs.append(Request(
                request_id=f"beta{i}", sent_at=T0 + timedelta(seconds=60 * i),
                model="claude-opus-5", agent="beta", tenant="t", session="sb",
                target_id="anthropic/direct", ttl_requested="5m",
                usage={"input_tokens": 300,
                       "cache_creation_input_tokens": 30_000,
                       "cache_read_input_tokens": 0},
                # Segments summing to 100x what the provider billed.
                segments=[Segment(id=f"hb{i}", role="system", tokens=300,
                                  index=0),
                          Segment(id="body", role="system", tokens=3_000_000,
                                  index=1, cache_marked=True, ttl="5m")]))
        ts = TraceSet(requests=reqs, tier=Tier.INSTRUMENTED, source="x")
        self.assertIsNone(
            simulate.bake_off(ts.analysable, trace=ts,
                              allow_unreconciled=True).delta_pct,
            "the whole workload must be indeterminate, or this proves nothing")
        out = {b.group: b for b in simulate.bake_off_by_agent(
            ts.analysable, trace=ts, allow_unreconciled=True)}
        self.assertEqual(set(out), {"alpha", "beta"})
        self.assertIsNotNone(out["alpha"].delta_pct,
                             "alpha was blanked by a defect in beta")
        self.assertTrue(out["alpha"].arms["as-shipped"]["spend"].released)
        self.assertIsNone(out["beta"].delta_pct)

    def test_the_subset_invariant_is_what_says_so(self):
        """And it is an invariant, not a condition: every request handed to the
        arms is in `trace.analysable`. Stated as one checkable sentence so there
        is no case analysis to get subtly wrong the way the last two versions
        did."""
        ts = self._clean()
        self.assertEqual([r.request_id for r in ts.analysable],
                         [r.request_id for r in ts.requests],
                         "the clean fixture must be wholly analysable, or the "
                         "breach below is indistinguishable from the baseline")
        ok = simulate.bake_off(ts.analysable, group="g",
                               allow_unreconciled=True, trace=ts)
        self.assertIsNotNone(ok.delta_pct)
        stray = Request(
            request_id="stray", sent_at=T0 + timedelta(seconds=99999),
            model="claude-opus-5", agent="a", tenant="t", session="s",
            target_id="anthropic/direct", ttl_requested="5m", usage={},
            segments=[Segment(id="s", role="system", tokens=10, index=0)])
        b = simulate.bake_off(list(ts.analysable) + [stray], group="g",
                              allow_unreconciled=True, trace=ts)
        self.assertIsNone(b.delta_pct)
        self.assertIn("not in the trace's analysable set", b.verdict)
        self.assertIn("'stray'", b.verdict, "the breach must name a row")

    # --- the exclusions are scoped to whoever they belong to ----------------

    def _two_agents(self, stray_agent):
        """Six clean requests each for alpha and beta, plus one failed billed
        row belonging to `stray_agent`."""
        reqs = []
        for agent in ("alpha", "beta"):
            for i in range(6):
                reqs.append(Request(
                    request_id=f"{agent}{i}",
                    sent_at=T0 + timedelta(seconds=60 * i),
                    model="claude-opus-5", agent=agent, tenant="t",
                    session=f"s{agent}", target_id="anthropic/direct",
                    ttl_requested="5m",
                    usage={"input_tokens": 300,
                           "cache_creation_input_tokens": 30_000,
                           "cache_read_input_tokens": 0},
                    segments=[Segment(id=f"hdr{agent}{i}", role="system",
                                      tokens=300, index=0),
                              Segment(id="body", role="system", tokens=30_000,
                                      index=1, cache_marked=True, ttl="5m")]))
        reqs.append(Request(
            request_id="failed", sent_at=T0 + timedelta(seconds=99999),
            model="claude-opus-5", agent=stray_agent, tenant="t", session="sx",
            target_id="anthropic/direct", ttl_requested="5m", status=500,
            usage={"input_tokens": 5_000_000, "cache_creation_input_tokens": 0,
                   "cache_read_input_tokens": 0},
            segments=[Segment(id="hdrF", role="system", tokens=5_000_000,
                              index=0)]))
        return TraceSet(requests=reqs, tier=Tier.INSTRUMENTED, source="x")

    def _by_agent(self, ts):
        return {b.group: b for b in simulate.bake_off_by_agent(
            ts.analysable, allow_unreconciled=True, trace=ts)}

    def test_one_agents_excluded_row_does_not_blank_another_agents_group(self):
        """The derivation was global where the question is group-scoped.

        `bake_off_by_agent` handed every group the whole file's
        `excluded_billed` and the whole trace, on the reasoning that an excluded
        row cannot be attributed to an agent. That is true of some of them and
        false of most -- a failed request usually carries the same `agent` field
        every other row does -- so one failed row belonging to beta made alpha's
        six clean requests indeterminate.
        """
        ts = self._two_agents("beta")
        self.assertEqual(ts.excluded_billed, {"failed but billed": 1})
        out = self._by_agent(ts)
        self.assertEqual(set(out), {"alpha", "beta"})
        self.assertTrue(out["alpha"].arms["as-shipped"]["spend"].released,
                        "alpha's clean group was blanked by beta's failed row")
        self.assertIsNotNone(out["alpha"].delta_pct)
        self.assertFalse(out["beta"].arms["as-shipped"]["spend"].released,
                         "beta owns the failed row and must carry it")
        self.assertIsNone(out["beta"].delta_pct)

    def test_a_row_belonging_to_nobody_still_blocks_every_group(self):
        """The other half, and the half that makes the scoping safe. A row the
        loader could not attribute could belong to any group, so it qualifies
        all of them -- scoping must not become a way to lose a blocker."""
        ts = self._two_agents("unknown")
        out = self._by_agent(ts)
        self.assertEqual(set(out), {"alpha", "beta"})
        for g, b in out.items():
            self.assertFalse(b.arms["as-shipped"]["spend"].released,
                             f"{g} released over an unattributable billed row")
            self.assertIsNone(b.delta_pct, g)

    def test_an_unreadable_line_stays_global(self):
        """`skipped_rows` names no agent and cannot, so it blocks everyone."""
        ts = self._two_agents("beta")
        ts = replace(ts, skipped_rows=1)
        for g, b in self._by_agent(ts).items():
            self.assertFalse(b.arms["as-shipped"]["spend"].released,
                             f"{g} released over a line the loader dropped")

    def test_a_caller_that_says_nothing_gets_nothing(self):
        """Fail closed, which is the half a keyword argument cannot enforce on
        its own. The trace here is the clean one that releases above; the only
        difference is that the caller did not state it."""
        ts = self._clean()
        b = simulate.bake_off(ts.analysable, group="g", allow_unreconciled=True,
                              excluded_billed=ts.excluded_billed)
        for end, p, fig in self._arms(b):
            self.assertFalse(fig.released,
                             f"{end}/{p} released with no trace supplied")
        self.assertIn("no trace was supplied",
                      b.arms["as-shipped"]["spend"].withheld_because)

    def test_the_reason_reaches_the_render(self):
        """A withheld figure whose reason never prints is the same as a missing
        caveat, which this package treats as the defect and not the fix."""
        ts = self._clean(tokens_counted=0.0)
        text = str(simulate.bake_off(ts.analysable, group="g",
                                     allow_unreconciled=True,
                                     excluded_billed=ts.excluded_billed,
                                     trace=ts))
        self.assertIn("per-arm spend withheld:", text)
        self.assertIn("estimated rather than counted", text)
        for line in text.splitlines():
            if any(line.strip().startswith(p) for p in simulate.ARMS):
                self.assertIn("[withheld]", line)
                self.assertNotIn("$", line, f"a dollar amount survived: {line}")

    def test_the_per_agent_run_gates_on_the_same_evidence(self):
        """`bake_off_by_agent` is a second entry point to the same figures, and
        a gate applied at one entry point and not the other is this file's
        subject."""
        ts = self._clean(tokens_counted=0.0)
        for b in simulate.bake_off_by_agent(ts.analysable,
                                            allow_unreconciled=True,
                                            excluded_billed=ts.excluded_billed,
                                            trace=ts):
            for end, p, fig in self._arms(b):
                self.assertFalse(fig.released,
                                 f"by-agent {b.group}: {end}/{p} released")
        released = simulate.bake_off_by_agent(
            self._clean().analysable, allow_unreconciled=True,
            excluded_billed=self._clean().excluded_billed, trace=self._clean())
        self.assertTrue(released, "no group survived the 3-request minimum, so "
                                  "the assertion above checked nothing")
        for b in released:
            self.assertTrue(b.arms["as-shipped"]["spend"].released,
                            f"by-agent {b.group} withheld on the clean trace")

    # --- the drift alarm ---------------------------------------------------

    @staticmethod
    def _attrs_read_off(node, var):
        """Every `<var>.x` and `getattr(<var>, "x")` under an AST node.

        Read off the parsed code, never off the source text. A first version of
        this grepped `inspect.getsource`, and deleting the gate while leaving the
        attribute named in the docstring above it passed -- a drift alarm that
        prose could satisfy.
        """
        import ast

        out = set()
        for n in ast.walk(node):
            if (isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                    and n.value.id == var):
                out.add(n.attr)
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                    and n.func.id == "getattr" and len(n.args) >= 2
                    and isinstance(n.args[0], ast.Name) and n.args[0].id == var
                    and isinstance(n.args[1], ast.Constant)):
                out.add(n.args[1].value)
        return out

    @classmethod
    def _analyzer_gate_inputs(cls):
        """The `TraceSet` attributes `analyze` feeds into `structure_trusted`.

        Discovered, not listed. The enumerated conditions above are today's
        members; this finds tomorrow's. Walks from the `structure_trusted`
        assignment back through the local names it reads, collecting every
        `ts.<attr>` and `getattr(ts, "<attr>")` those names are built from.
        """
        import ast
        import inspect
        import textwrap

        from cacheeconomics import analyzer

        tree = ast.parse(textwrap.dedent(inspect.getsource(analyzer.analyze)))
        assigns = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        assigns.setdefault(t.id, []).append(node.value)

        found, seen, queue = set(), set(), ["structure_trusted"]
        while queue:
            name = queue.pop()
            if name in seen:
                continue
            seen.add(name)
            for value in assigns.get(name, []):
                found |= cls._attrs_read_off(value, "ts")
                for n in ast.walk(value):
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                        queue.append(n.id)
        return found

    @classmethod
    def _simulator_gate_inputs(cls):
        """Every `TraceSet` attribute the bake-off's gates actually read.

        Every helper the gate reads the trace through, not just the structural
        one. `_trace_exclusions` is where the excluded-row half lives and
        `_agent_trace` is where the per-group narrowing does; a walk that looked
        only at `structural_evidence` would report `excluded_billed` as
        unconsulted at the exact moment it started being consulted.
        """
        import ast
        import inspect
        import textwrap

        found = set()
        # `bake_off` itself is in the list because the assumed-input label cap
        # lives there rather than in a helper: it decides the release
        # PROVENANCE of a figure, which is a gate on what may be published even
        # though it is not a gate on whether anything is. Track B added
        # `assumed_inputs`, Track E read it in `bake_off`, and this walk
        # reported it unconsulted at the exact moment it started being
        # consulted -- the same shape this alarm was written for.
        for fn in (simulate.structural_evidence, simulate._trace_exclusions,
                   simulate._agent_trace, simulate.bake_off):
            tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
            found |= cls._attrs_read_off(tree, "trace")
        return found

    def test_the_walker_finds_the_gate_it_is_pointed_at(self):
        """Guard the guard, again. A walker that returns the empty set makes the
        drift alarm below pass forever."""
        found = self._analyzer_gate_inputs()
        self.assertGreaterEqual(
            len(found), 5,
            f"the discovery walk found only {sorted(found)} -- `structure_trusted` "
            f"was restructured and this alarm is now checking nothing")
        self.assertTrue(
            self._simulator_gate_inputs(),
            "the simulator-side walk found nothing, so the comparison below "
            "would fail for the wrong reason or pass for none")

    def test_the_simulator_consults_every_input_the_report_does(self):
        """The alarm itself. A sixth fact added to the analyzer's structural
        gate fails here until the simulator asks it too -- which is the whole
        reason this finding existed: the fifth was added and the simulator was
        never told."""
        missing = sorted(self._analyzer_gate_inputs()
                         - self._simulator_gate_inputs())
        self.assertEqual(
            missing, [],
            f"`analyze` gates structural money on {missing} and "
            f"`simulate.structural_evidence` never reads it, so a trace the "
            f"report refuses to cost can still publish per-arm dollars")

    # Every `TraceSet` member the simulator does NOT consult, with the reason it
    # does not have to. The point of writing them down is that the list is
    # checked against the dataclass at runtime: a field added to `TraceSet`
    # later is neither consulted nor excused, so it fails here until somebody
    # decides which it is. `excluded_billed` was exactly this defect -- a member
    # that reached the analyzer's gate and the simulator's optional side-channel
    # argument, and nothing said the two had to agree.
    NOT_A_GATE = {
        "source": "provenance for the reader, not a condition on a figure",
        "notes": "prose the report renders; the simulator has no note surface",
        "blocking_notes": "same, and already classified by the analyzer",
        "coverage": "reaches the gate through `excluded_billed`, which is read",
        "skipped_rows": "reaches the gate through `excluded_billed`, which is read",
        "tokens_counted": (
            "the raw share; `tokens_are_counted` is the threshold over it and is "
            "read. Consulting both would be two thresholds for one question"),
        "token_sums_reconciled": (
            "the coarse factor-of-two check. `token_sums_publishable` is the "
            "strict one and is what guards money -- reading the coarse flag here "
            "was the original defect, one module along"),
    }

    def test_every_traceset_member_is_either_consulted_or_excused(self):
        """Discovery over the dataclass, not over a sentence about it."""
        import dataclasses

        members = {f.name for f in dataclasses.fields(TraceSet)}
        members |= {n for n in dir(TraceSet)
                    if not n.startswith("_")
                    and isinstance(getattr(TraceSet, n, None), property)}
        self.assertGreaterEqual(len(members), 12,
                                f"only found {sorted(members)} -- the member "
                                f"walk is broken and this checks nothing")
        consulted = self._simulator_gate_inputs()
        unclassified = sorted(members - consulted - set(self.NOT_A_GATE))
        self.assertEqual(
            unclassified, [],
            f"`TraceSet` carries {unclassified}, which the bake-off neither "
            f"reads nor excuses. Either it gates a figure and belongs in the "
            f"simulator's gates, or it does not and belongs in NOT_A_GATE with "
            f"the reason -- an unclassified member is how `excluded_billed` "
            f"stayed an optional side-channel while the analyzer gated on it")

    def test_the_excuse_list_does_not_rot(self):
        """A name in NOT_A_GATE that `TraceSet` no longer has is an excuse for
        a member that stopped existing, which quietly shrinks the check."""
        import dataclasses

        members = {f.name for f in dataclasses.fields(TraceSet)} | {
            n for n in dir(TraceSet) if not n.startswith("_")}
        stale = sorted(set(self.NOT_A_GATE) - members)
        self.assertEqual(stale, [],
                         f"NOT_A_GATE excuses {stale}, which `TraceSet` no "
                         f"longer carries")
        # An excuse for something the gate does read is worse than stale: it
        # would keep the member classified if the read were ever removed, which
        # is the silent-partial-closure shape this file exists to catch.
        both = sorted(set(self.NOT_A_GATE) & self._simulator_gate_inputs())
        self.assertEqual(both, [],
                         f"NOT_A_GATE excuses {both}, which the gate actually "
                         f"reads -- drop the excuse or the read")

    def test_the_floor_is_one_number_and_not_two(self):
        """The threshold is imported from the analyzer rather than restated.
        `PUBLISH_TOLERANCE` and the coarse factor had already drifted between
        these two modules once."""
        from cacheeconomics import analyzer

        self.assertIs(simulate.ALIGNMENT_FLOOR, analyzer.ALIGNMENT_FLOOR)


class TestTheLiveResponseParserReadsBothLiteLLMShapes(unittest.TestCase):
    """`usage_from_response` grew a LiteLLM fallback that searched the wrong
    object.

    `u = _get(response, "usage") or {}` runs first, so when the counters are at
    the *top level* -- which is the shape the fallback was added for -- `u` is
    `{}` and the fallback searched an empty dict. Measured through the live
    handler over ten requests: the nested shape raised a rebuild alert and the
    identical counters at top level raised none, so the plugin kept placing
    markers while its diagnostics stayed silent.
    """

    TOP = {"prompt_tokens": 12000, "completion_tokens": 50,
           "prompt_tokens_details": {"cached_tokens": 9000,
                                     "cache_creation_tokens": 3000}}

    def test_both_shapes_yield_the_same_counters(self):
        from cacheeconomics.segment import usage_from_response
        self.assertEqual(usage_from_response(dict(self.TOP)),
                         usage_from_response({"usage": dict(self.TOP)}))

    def test_the_counters_are_actually_right(self):
        from cacheeconomics.segment import usage_from_response
        got = usage_from_response(dict(self.TOP))
        self.assertEqual(got["cache_read_input_tokens"], 9000)
        self.assertEqual(got["cache_creation_input_tokens"], 3000)

    def test_presence_is_still_not_evidence(self):
        """The rule the rest of this function is built on. A tolerant fallback
        must not start accepting husks."""
        from cacheeconomics.segment import usage_from_response
        for empty in ({}, {"usage": {"model": "x"}},
                      {"prompt_tokens": 0, "prompt_tokens_details": {"cached_tokens": 0}}):
            self.assertEqual(usage_from_response(empty), {}, empty)

    def test_the_live_handler_sees_both(self):
        import asyncio

        def alerts_for(resp):
            p = plugin.CachePlugin(key=KEY, warmup=2)
            h = plugin.litellm_handler(p, mutate=True,
                                       target_id="anthropic/direct")

            async def go():
                for i in range(10):
                    data = {"model": "claude-opus-5", "litellm_call_id": f"c{i}",
                            "messages": [{"role": "user",
                                          "content": f"policy {i} " + "x " * 3000}]}
                    await h.async_pre_call_hook(None, None, data, "completion")
                    await h.async_log_success_event(
                        data, resp, T0 + timedelta(seconds=60 * i),
                        T0 + timedelta(seconds=60 * i + 1))
            asyncio.run(go())
            return len(p.alerts)

        self.assertEqual(alerts_for(dict(self.TOP)),
                         alerts_for({"usage": dict(self.TOP)}))
        self.assertGreater(alerts_for(dict(self.TOP)), 0,
                           "neither shape produced diagnostics; the test proves nothing")


class TestACaveatPrecedesTheFigureItQualifies(unittest.TestCase):
    """In both renderers, checked by offset rather than by presence.

    A caveat is not a checkbox. The text report printed the spend caveats below
    the findings table and the TOTAL, and the HTML one carried them in the
    Standing block at section 04 -- after the Input spend KPI in 02 and every
    finding in 03. So a reader met every number in the document and the sentence
    saying what those numbers rest on afterwards. Measured on the fixture below:
    the first "$" in the text report at character 2,213 against the caveat at
    3,013.

    This is the same defect, and the same test shape, as the DRAFT banner that
    was inserted at index 1 -- inside `<head>` -- and passed a substring check
    while rendering nowhere. Presence was never the claim; order is.
    """

    # Two, and the second one is load-bearing. The DRAFT banner quotes the
    # *first* blocking note verbatim and prints at the top of both reports, so a
    # check that searched for that text found the banner and passed wherever the
    # caveat block itself sat -- measured by reverting the fix and watching this
    # class stay green. A test whose subject is placement, satisfied by a
    # different element that happens to contain the same words, is the same
    # failure as asserting the DRAFT banner's presence while it rendered inside
    # `<head>`. The second note appears only in the caveat block, so it is the
    # one that actually pins the ordering.
    NOTES = [("The surface was assumed to be anthropic/direct; the export names "
              "none. Rates and cache multipliers here are an assumption."),
             ("Sampling: this export carries one row in every four, so the "
              "totals below describe a quarter of the traffic.")]

    def _analysis(self):
        reqs = [Request(request_id=f"r{i}",
                        sent_at=T0 + timedelta(hours=6 * i),
                        model="claude-opus-5", target_id="anthropic/direct",
                        tenant="t", session="s", agent="a", ttl_requested="5m",
                        usage={"input_tokens": 0, "cache_read_input_tokens": 0,
                               "cache_creation_input_tokens": 100_000},
                        segments=[])
                for i in range(12)]
        ts = TraceSet(requests=reqs, tier=Tier.USAGE_ONLY, source="x",
                      notes=list(self.NOTES),
                      blocking_notes=list(self.NOTES))
        invoice = analyze(ts, allow_unreconciled=True).spend["input_usd"].raw()
        return analyze(ts, invoice_usd=invoice)

    def _renderers(self):
        from cacheeconomics import report
        return [(n, getattr(report, n)) for n in dir(report)
                if n.startswith("render_") and callable(getattr(report, n))]

    def test_the_fixture_publishes_a_figure_to_be_qualified(self):
        """Guard the guard: with nothing published there is no ordering to
        check and every assertion below is vacuous."""
        from cacheeconomics.analyzer import spend_caveats
        a = self._analysis()
        self.assertEqual(2, len(spend_caveats(a)), "both caveats must survive")
        self.assertTrue(a.spend["input_usd"].released, "nothing published")

    def test_the_second_caveat_is_not_carried_by_the_draft_banner(self):
        """Guard the guard again, and this one is the reason the class has two
        notes. The banner quotes the first blocking note, so only a caveat it
        does not quote can prove where the caveat block itself sits."""
        a = self._analysis()
        banner = next((n for n in a.notes if n.startswith("DRAFT")), "")
        self.assertNotIn(self._key(self.NOTES[1]), banner)

    @staticmethod
    def _key(note):
        return note.split(";")[0][:40]

    def test_the_first_dollar_is_a_computed_figure_not_the_invoice_echo(self):
        """The check below measures against the first "$". That is the right
        thing to measure only while the first "$" is a number this tool
        *computed*.

        The HTML report also echoes the client's own invoice back at them, and
        an echo is an input rather than a claim -- a caveat about what the
        analysis rests on has nothing to say about a number the reader typed in,
        so placing a caveat relative to it would be answering a question nobody
        asked. It happens not to be first in either renderer: the text report
        states reconciliation as a percentage and never prints the invoice at
        all, and the HTML hero carries "Caching saved $X" well above the
        reconciliation card.

        Asserted rather than assumed. If either of those ever changes, the probe
        below starts silently measuring the wrong thing, and this is what says so.
        """
        a = self._analysis()
        rendered = dict(self._renderers())
        html = rendered["render_html"](a)
        echo = html.find("against invoice")
        self.assertNotEqual(-1, echo, "fixture no longer echoes the invoice")
        self.assertLess(html.find("$"), echo,
                        "the HTML report's first dollar is now the invoice echo")
        text = rendered["render_text"](a)
        head = text[:text.find("what it is costing you")]
        self.assertNotIn("$", head,
                         "the text report now prints a dollar before its "
                         "findings table; check whether it is the invoice echo")

    def test_every_renderer_puts_every_caveat_before_the_first_figure(self):
        """Over every renderer found at runtime, so a third one cannot phrase
        the ordering a third way -- which is exactly how these two diverged."""
        from cacheeconomics.analyzer import spend_caveats
        a = self._analysis()
        for name, fn in self._renderers():
            text = fn(a)
            at_money = text.find("$")
            self.assertNotEqual(-1, at_money, f"{name} printed no figure")
            for note in spend_caveats(a):
                with self.subTest(renderer=name, caveat=note[:30]):
                    at_caveat = text.find(self._key(note))
                    self.assertNotEqual(-1, at_caveat,
                                        f"{name} dropped a caveat entirely")
                    self.assertLess(
                        at_caveat, at_money,
                        f"{name} prints a computed figure at {at_money} and the "
                        f"caveat that qualifies it at {at_caveat}")


class TestCoverageThatGatesMoneyIsWeightedByMoney(unittest.TestCase):
    """Rows are the wrong denominator for a dollar figure.

    `structural_coverage` counted rows-with-segments over rows, and the gate on
    structural money read it. Nine small structured requests beside one enormous
    usage-only request is 90% of rows and 8% of the bill, so VOL-1 published
    $3,175 a month costed from traffic the structural rules had never seen --
    with the invoice reconciling, because reconciliation proves the *total* is
    right and says nothing about whether the structured subset is where the
    spend came from.

    `score_alignment` had the same defect independently: a plain mean over
    requests, so nine perfect tiny segmentations beside one large one segmented
    entirely wrong scored exactly 0.90 and cleared the floor. Found by sweeping
    the class rather than reported, which is the only reason it is fixed in the
    same commit.
    """

    def _trace(self, unstructured_tokens):
        reqs = []
        for i in range(27):
            reqs.append(Request(
                request_id=f"s{i}", sent_at=T0 + timedelta(seconds=120 * i),
                model="claude-opus-5", agent="a", session="s", ttl_requested="5m",
                usage={"input_tokens": 200, "cache_read_input_tokens": 0,
                       "cache_creation_input_tokens": 30000},
                segments=[Segment(id=f"vol{i}", role="system", tokens=500, index=0),
                          Segment(id="tools", role="tools", tokens=29500, index=1,
                                  cache_marked=True, ttl="5m"),
                          Segment(id=f"t{i}", role="user", tokens=200, index=2)], target_id="anthropic/direct"))
        for i in range(3):
            reqs.append(Request(
                request_id=f"h{i}", sent_at=T0 + timedelta(seconds=120 * (27 + i)),
                model="claude-opus-5", agent="a", session="s", ttl_requested="5m",
                usage={"input_tokens": unstructured_tokens,
                       "cache_read_input_tokens": 0,
                       "cache_creation_input_tokens": 0},
                segments=[], target_id="anthropic/direct"))
        return TraceSet(requests=reqs, tier=Tier.INSTRUMENTED, source="x",
                        structural_coverage=27 / 30)

    def _vol(self, a):
        return next((f for f in a.findings if f.code == "VOL-1"), None)

    def test_a_dominant_unstructured_row_withholds_structural_money(self):
        ts = self._trace(3_000_000)
        self.assertGreaterEqual(ts.structural_coverage, 0.90)      # rows look fine
        self.assertLess(ts.structural_coverage_billed, 0.20)       # the money does not
        a = analyze(ts, allow_unreconciled=True)
        self.assertFalse(self._vol(a).avoidable_usd_month.released)

    def test_the_reason_says_it_is_the_tokens_not_the_rows(self):
        """The reader is holding a healthy-looking row coverage and a reconciled
        invoice. Saying "90% of requests carry structure" would confirm exactly
        the wrong thing."""
        a = analyze(self._trace(3_000_000), allow_unreconciled=True)
        why = self._vol(a).avoidable_usd_month.withheld_because
        self.assertIn("billed input tokens", why)

    def test_small_unstructured_rows_do_not_over_block(self):
        """Immaterial dark rows must still release. Refusing on any unstructured
        row at all would make the gate useless on real exports.

        `avoidable_usd_window`: these thirty requests span an hour, so the
        monthly figure is withheld by the projection floor whatever the billed
        coverage is. The window figure is what this gate decides.
        """
        ts = self._trace(100)
        self.assertGreaterEqual(ts.structural_coverage_billed, 0.90)
        a = analyze(ts, allow_unreconciled=True)
        self.assertTrue(self._vol(a).avoidable_usd_window.released)

    def test_a_dominant_unstructured_row_withholds_the_window_figure_too(self):
        """The pair to the test above. The report falls back to the window
        figure exactly where the monthly one is missing, so a gate that reached
        only the monthly figure would put the withheld amount back on the page
        under a different label."""
        a = analyze(self._trace(3_000_000), allow_unreconciled=True)
        self.assertFalse(self._vol(a).avoidable_usd_window.released)
        self.assertIn("billed input tokens",
                      self._vol(a).avoidable_usd_window.withheld_because)

    def test_alignment_is_weighted_by_billed_tokens_too(self):
        from cacheeconomics.adapters.bodies import score_alignment

        def side(rid, ids, billed):
            return Request(request_id=rid, sent_at=T0, model="claude-opus-5",
                           usage={"input_tokens": billed,
                                  "cache_read_input_tokens": 0,
                                  "cache_creation_input_tokens": 0},
                           segments=[Segment(id=i, role="system", tokens=100, index=n)
                                     for n, i in enumerate(ids)], target_id="anthropic/direct")
        truth = [side(f"t{i}", [f"a{i}", f"b{i}"], 200) for i in range(9)]
        inferred = [side(f"t{i}", [f"a{i}", f"b{i}"], 200) for i in range(9)]
        truth.append(side("huge", ["x", "y", "z"], 900_000))
        inferred.append(side("huge", ["W1", "W2", "W3"], 900_000))
        sc = score_alignment(
            TraceSet(requests=truth, tier=Tier.INSTRUMENTED, source="t"),
            TraceSet(requests=inferred, tier=Tier.INFERRED, source="i"))
        # Per row it is exactly at the floor; by money it is essentially nothing
        # -- the nine correct requests are 1,800 tokens of a 901,800-token
        # workload, so they cannot vouch for the segmentation that priced it.
        self.assertAlmostEqual(sc["segment_alignment"], 0.90, places=2)
        self.assertLess(sc["segment_alignment_billed"], 0.01)
        self.assertLess(sc["mean_alignment"], 0.90)

    def test_no_money_gate_reads_a_row_ratio(self):
        """Structural, because two independent instances of this shipped.

        Every coverage-style quantity that gates a dollar figure must be
        weighted by what was billed. The row figures stay for the notes; they
        may not be the thing money is decided on.
        """
        import ast
        import inspect

        from cacheeconomics import analyzer as an
        src = inspect.getsource(an._release_gate if hasattr(an, "_release_gate")
                                else an.analyze)
        tree = ast.parse(src.lstrip())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            targets = {t.id for t in node.targets if isinstance(t, ast.Name)}
            if "structure_trusted" not in targets:
                continue
            names = {n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)}
            self.assertIn(
                "covered_billed", names,
                "structure_trusted decides structural money without consulting "
                "billed-token coverage; a row ratio cannot see which requests "
                "carried the spend")


class TestCadenceIsMeasuredInsideACacheScope(unittest.TestCase):
    """A cache lives in `(tenant, target, model, session)`.

    TTL-1 grouped timestamps by agent alone and applied its in-band threshold
    before splitting into scopes, so on a shared gateway one agent serving many
    tenants had its gaps compressed by interleaving and the finding was skipped.
    The failure curve is the wrong way round: identical per-tenant work, TTL-1
    fired at one tenant and vanished at four and at twenty.
    """

    def _trace(self, tenants):
        reqs = []
        for k in range(40):
            for t in range(tenants):
                at = T0 + timedelta(seconds=600 * k + t * (600 // tenants))
                reqs.append(Request(
                    request_id=f"r{t}-{k}", sent_at=at, model="claude-opus-5",
                    agent="shared", tenant=f"tenant-{t}", session=f"s{t}",
                    ttl_requested="5m",
                    usage={"input_tokens": 200, "cache_read_input_tokens": 0,
                           "cache_creation_input_tokens": 30000},
                    segments=[Segment(id=f"sys{t}", role="system", tokens=30000,
                                      index=0, cache_marked=True, ttl="5m"),
                              Segment(id=f"u{t}{k}", role="user", tokens=200,
                                      index=1)], target_id="anthropic/direct"))
        reqs.sort(key=lambda r: r.sent_at)
        return TraceSet(requests=reqs, tier=Tier.INSTRUMENTED, source="t")

    def test_the_finding_survives_tenants_sharing_an_agent(self):
        for n in (1, 4, 20):
            with self.subTest(tenants=n):
                a = analyze(self._trace(n), allow_unreconciled=True)
                self.assertIn("TTL-1", [f.code for f in a.findings],
                              f"TTL-1 suppressed by pooling {n} tenants")

    def test_the_reported_percentage_describes_one_cache(self):
        """The number in the evidence line is what a reader acts on, so it has
        to describe gaps a cache could actually see, not an interleaved
        stream."""
        a = analyze(self._trace(8), allow_unreconciled=True)
        ttl = next(f for f in a.findings if f.code == "TTL-1")
        self.assertIn("100%", ttl.detail)


class TestEveryIngestBranchForwardsTheSurface(unittest.TestCase):
    """Read once is not honoured everywhere.

    `TestEveryCliFlagIsRead` below passed the whole time `--target-id` was being
    dropped, because it asks whether a flag is read *anywhere* and the flag was
    read -- in the one branch of `_load` that forwarded it. The other two
    ingested at `anthropic/direct` regardless, so an operator who selected
    Bedrock got partner traffic priced at Anthropic first-party rates.

    So this asks the stricter question of the one function where surface is
    decided: every loader call in `_load` must be handed a surface. Structural
    rather than behavioural, so a fourth ingest mode fails here on the day it is
    added rather than in a client report.
    """

    SURFACE_KWARGS = {"default_target", "target_id"}

    def test_every_loader_call_in_load_is_given_a_surface(self):
        """Structural on purpose: a fourth ingest mode fails here the day it is
        added, which behaviour cannot do for code nobody has written yet.

        Now also rejects a constant. The earlier version collected only keyword
        *names*, so `target_id="anthropic/direct"` hard-coded passed it -- and
        that mutation reopens partner traffic priced at first-party rates.
        Verified by applying it before this was rewritten.
        """
        import ast
        import inspect

        from cacheeconomics import cli

        tree = ast.parse(inspect.getsource(cli._load))
        # Attribute callees too. `adapters.load_bodies(...)` was invisible to
        # the old ast.Name-only filter, so a mode called that way skipped both
        # the count check and the surface check.
        def _name(fn):
            return getattr(fn, "id", None) or getattr(fn, "attr", None) or ""

        calls = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and _name(n.func).startswith("load_")]
        self.assertGreaterEqual(len(calls), 3, "expected one call per ingest mode")
        for c in calls:
            passed = {k.arg: k.value for k in c.keywords}
            named = self.SURFACE_KWARGS & set(passed)
            self.assertTrue(
                named,
                f"{_name(c.func)} is called without a surface, so rows that "
                f"carry none silently ingest as anthropic/direct whatever "
                f"--target-id said. Pass one of {sorted(self.SURFACE_KWARGS)}.")
            for kw in named:
                self.assertNotIsInstance(
                    passed[kw], ast.Constant,
                    f"{_name(c.func)} is handed a hard-coded {kw}, which "
                    f"ignores --target-id entirely")

    def test_target_id_actually_reaches_the_trace_for_every_mode(self):
        """The behavioural half. The structural check above cannot tell whether
        the value forwarded is the operator's or something invented on the way."""
        import json
        import os
        import tempfile

        from cacheeconomics import cli

        want = "amazon-bedrock/converse"
        with tempfile.TemporaryDirectory() as tmp:
            trace = os.path.join(tmp, "t.jsonl")
            with open(trace, "w") as f:
                f.write(json.dumps({
                    "request_id": "r", "sent_at": "2026-07-29T09:00:00Z",
                    "model": "claude-opus-5", "session": "s",
                    "usage": {"input_tokens": 100,
                              "cache_read_input_tokens": 0,
                              "cache_creation_input_tokens": 0}}) + "\n")
            bodies = os.path.join(tmp, "b.jsonl")
            with open(bodies, "w") as f:
                f.write(json.dumps({
                    "sent_at": "2026-07-29T09:00:00Z",
                    "body": {"model": "claude-opus-5",
                             "messages": [{"role": "user", "content": "hi"}]},
                    "usage": {"input_tokens": 100}}) + "\n")
            lite = os.path.join(tmp, "l.jsonl")
            with open(lite, "w") as f:
                f.write(json.dumps({
                    "id": "r", "startTime": 1_780_000_000, "model": "claude-opus-5",
                    "response": {"usage": {
                        "prompt_tokens": 100, "completion_tokens": 1,
                        "prompt_tokens_details": {"cached_tokens": 0},
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 0}}}) + "\n")

            key = os.environ.get("CACHEECONOMICS_HMAC_KEY")
            os.environ["CACHEECONOMICS_HMAC_KEY"] = "k" * 32
            try:
                for source, path in (("trace", trace), ("bodies", bodies),
                                     ("litellm", lite)):
                    with self.subTest(source=source):
                        args = cli.build_parser().parse_args(
                            ["analyze", path, "--from", source,
                             "--target-id", want])
                        ts = cli._load(args)
                        self.assertTrue(ts.requests, f"{source} loaded nothing")
                        self.assertEqual({r.target_id for r in ts.requests}, {want},
                                         f"--target-id did not reach the {source} trace")
            finally:
                if key is None:
                    os.environ.pop("CACHEECONOMICS_HMAC_KEY", None)
                else:
                    os.environ["CACHEECONOMICS_HMAC_KEY"] = key


class TestEveryCliFlagIsRead(unittest.TestCase):
    """`claude-code --on-date` was parsed and dropped, so a user asking for
    date-effective pricing silently got something else.

    A flag that is accepted and ignored is worse than one that does not exist:
    the first produces a wrong answer the user believes they asked for. Audited
    across every subcommand rather than fixed on the one that was reported.
    """

    def test_no_subcommand_accepts_an_argument_it_never_reads(self):
        import ast
        import inspect

        from cacheeconomics import cli

        parser = cli.build_parser()
        subparsers = {}
        for action in parser._actions:
            if isinstance(getattr(action, "choices", None), dict):
                subparsers.update(action.choices)

        ignored = {}
        for name, sub in subparsers.items():
            handler = sub.get_default("func")
            self.assertIsNotNone(handler, f"{name} has no handler")

            # Follow one level of delegation. `cmd_analyze` reads `args.path`
            # and friends through `_load(args)`, and an audit that only looked
            # at the handler body called five correctly-used arguments ignored.
            # A false positive here would train someone to silence this test.
            def reads(fn, seen=frozenset()):
                if fn.__name__ in seen:
                    return set()
                tree = ast.parse(inspect.getsource(fn))
                out = {n.attr for n in ast.walk(tree)
                       if isinstance(n, ast.Attribute)}
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                        callee = getattr(cli, node.func.id, None)
                        if callable(callee) and getattr(callee, "__module__",
                                                        "") == cli.__name__:
                            out |= reads(callee, seen | {fn.__name__})
                return out

            read = reads(handler)
            for action in sub._actions:
                dest = action.dest
                if dest in ("help", "==SUPPRESS==", "func"):
                    continue
                if dest not in read:
                    ignored.setdefault(name, []).append(dest)

        self.assertFalse(
            ignored,
            f"these subcommands accept arguments their handler never reads: "
            f"{ignored}. An accepted-and-ignored flag gives the user a wrong "
            f"answer they believe they asked for.")


class TestOneSessionExtractor(unittest.TestCase):
    """The live plugin and the batch adapter answer the same question, and they
    disagreed: the adapter read the top-level `trace_id`, the plugin read only
    `metadata`. So on a StandardLoggingPayload carrying exactly the field its own
    docstring names, `default_session_from` returned None and switched off
    in-session rebuild detection -- the highest-value finding a usage-only trace
    can produce, silenced on logs that contained the key to find it.

    Both now call `trace.session_of`. This asserts they keep doing so, on the
    shapes where they used to differ."""

    ROWS = [
        {"trace_id": "top"},
        {"metadata": {"session_id": "sess"}, "trace_id": "top"},
        {"metadata": {"conversation_id": "conv"}},
        {"metadata": {"trace_id": "meta"}, "trace_id": "top"},
        {"litellm_trace_id": "llt"},
        {"metadata": {}, "litellm_session_id": "lls"},
        {},
        {"metadata": {"session_id": {"unhashable": 1}}, "trace_id": "top"},
        {"metadata": {"session_id": 12345}},
    ]

    def test_the_plugin_and_the_adapter_never_disagree(self):
        from cacheeconomics.adapters.litellm import request_from_payload
        from cacheeconomics.plugin import default_session_from
        from cacheeconomics.trace import session_of
        for row in self.ROWS:
            with self.subTest(row=row):
                shared = session_of(row)
                self.assertEqual(default_session_from(row), shared)
                full = dict(row)
                full.setdefault("model", "claude-opus-5")
                self.assertEqual(request_from_payload(full).session, shared)

    def test_an_explicit_session_outranks_a_retry_correlation_id(self):
        """LiteLLM's schema: `trace_id` spans "fallbacks/retries" of one overall
        request. That is not a conversation."""
        from cacheeconomics.trace import session_of
        self.assertEqual(
            session_of({"metadata": {"session_id": "sess"}, "trace_id": "top"}),
            "sess")

    def test_nothing_means_none_rather_than_a_placeholder(self):
        """Inventing a conversation out of unrelated calls reports every request
        as a rebuild of the last one."""
        from cacheeconomics.trace import session_of
        self.assertIsNone(session_of({}))
        self.assertIsNone(session_of({"metadata": {}}))
        self.assertIsNone(session_of("not a dict"))




class TestGatewayAndSurfacePrefixesCompose(unittest.TestCase):
    """`bedrock/anthropic.claude-haiku-4-5` is the id LiteLLM actually emits.

    Two prefixes, and each pass could only see past its own. The surface pass
    could not match because the id starts with `bedrock/`; the routing pass
    rejected `anthropic.claude-haiku-4-5` because that is not a registry model.
    Both halves worked alone, and the combination -- the routing prefix for the
    very surface whose `model_id_prefix` the other half strips -- fell through
    unnormalised, so Bedrock traffic lost its minimum check, its bake-off
    modelling and its live marker placement.
    """

    T = "amazon-bedrock/converse"

    def test_the_composed_id_normalises(self):
        self.assertEqual(
            registry.normalize_model("bedrock/anthropic.claude-haiku-4-5", self.T),
            ("claude-haiku-4-5", None))

    def test_a_date_survives_the_composition(self):
        self.assertEqual(
            registry.normalize_model(
                "bedrock/anthropic.claude-haiku-4-5-20251001", self.T),
            ("claude-haiku-4-5", "20251001"))

    def test_the_minimum_check_reaches_it(self):
        """The consequence that matters: a minimum the registry knows was
        refused for a model it knows, on the surface it knows it for."""
        bare, _ = registry.normalize_model(
            "bedrock/anthropic.claude-haiku-4-5", self.T)
        self.assertIsInstance(registry.min_cacheable_tokens(self.T, bare), int)

    def test_it_still_invents_nothing(self):
        """Every branch requires what remains to be an id the registry already
        knows. A tolerant composition must not become a guessing one."""
        for unknown in ("bedrock/anthropic.not-a-real-model",
                        "anthropic/not-a-real-model-20250101",
                        "us.anthropic.claude-haiku-4-5"):
            self.assertEqual(registry.normalize_model(unknown, self.T)[0], unknown,
                             f"{unknown} was rewritten")

    def test_the_batch_adapter_and_the_live_hook_agree(self):
        """Both feed this path, and a divergence here means a Bedrock trace and
        the live proxy disagree about which model they are looking at."""
        from cacheeconomics.adapters.litellm import request_from_payload
        row = {"id": "c1", "model": "bedrock/anthropic.claude-haiku-4-5",
               "custom_llm_provider": "bedrock", "status": "success",
               "startTime": 1785000000, "prompt_tokens": 1000,
               "completion_tokens": 10}
        batch = request_from_payload(row)
        self.assertEqual(batch.model, "claude-haiku-4-5")
        self.assertEqual(batch.target_id, self.T)
        live, _ = registry.normalize_model(row["model"], batch.target_id)
        self.assertEqual(live, batch.model)




class TestMalformedInputStaysCountable(unittest.TestCase):
    """Three ways a degraded export stopped being describable.

    None of these crashes. Each quietly makes the damage look smaller than it
    is, on the failure path where the count is the thing a reader needs.
    """

    def test_a_zero_only_split_is_not_accounting(self):
        """`or got.get("cache_creation")` accepted a non-empty dict of zeros as
        evidence, so a husk carrying `{5m: 0, 1h: 0}` returned a full set of
        zeroed counters and `_find_response` -- which ranks candidates by
        whether they carry usage -- picked it over a top-level `usage` holding
        11,000 real input tokens."""
        from cacheeconomics.adapters.bodies import _find_response
        from cacheeconomics.segment import usage_from_response
        husk = {"usage": {"prompt_tokens": 0, "completion_tokens": 0,
                          "prompt_tokens_details": {
                              "cached_tokens": 0,
                              "cache_creation_token_details": {
                                  "ephemeral_5m_input_tokens": 0,
                                  "ephemeral_1h_input_tokens": 0}}}}
        self.assertEqual(usage_from_response(husk), {})
        row = {"response": husk,
               "usage": {"input_tokens": 11000, "cache_read_input_tokens": 0,
                         "cache_creation_input_tokens": 0}}
        self.assertEqual(
            usage_from_response(_find_response(row))["input_tokens"], 11000)

    def test_a_split_only_export_is_still_accounting(self):
        """The other half. Writes are counted through `write_tokens`, so an
        export reporting only the per-lifetime split -- zero in the aggregate,
        billed anyway -- must not be refused as empty."""
        from cacheeconomics.segment import usage_from_response
        got = usage_from_response({"usage": {
            "prompt_tokens": 12000, "completion_tokens": 5,
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "cache_creation_token_details": {
                    "ephemeral_5m_input_tokens": 3000,
                    "ephemeral_1h_input_tokens": 0}}}})
        self.assertEqual(got["cache_creation_input_tokens"], 3000)

    def test_id_less_rows_are_counted_separately(self):
        """Omission accounting dedupes on request id, so a shared "" collapsed
        three dropped rows into one: "1 of 3 requests contributed nothing" when
        all three had."""
        import json
        import tempfile

        from cacheeconomics.adapters.litellm import load_litellm
        rows = [{"model": "claude-opus-5", "status": "success",
                 "startTime": 1785000000 + i, "prompt_tokens": 100,
                 "completion_tokens": 1,
                 # A real split, so these rows are dropped for having no
                 # structure rather than for an unproved token class.
                 "prompt_tokens_details": {"cached_tokens": 0}}
                for i in range(3)]
        fh = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        for r in rows:
            fh.write(json.dumps(r) + "\n")
        fh.close()
        self.addCleanup(os.unlink, fh.name)
        ts = load_litellm(fh.name)
        self.assertEqual(len({r.request_id for r in ts.requests}), 3)
        b = simulate.bake_off(ts.analysable, group="g", allow_unreconciled=True)
        self.assertIn("3 of 3 requests contributed nothing", b.verdict)

    def test_json_output_parses_strictly_on_every_invoice_shape(self):
        """`--format json` is the machine-readable surface, and Python emits a
        bare `NaN` token no strict parser accepts -- on exactly the failure path
        automation needs to read.

        Driven through `cmd_analyze` rather than the sanitiser. The first draft
        of this test called `_json_safe` directly and passed with the sanitiser
        unwired from the renderer, which is the same gap as testing a loader
        instead of the CLI that forgets to call it.
        """
        import argparse
        import contextlib
        import io
        import json
        import tempfile

        from cacheeconomics import cli

        rows = [{"request_id": f"r{i}",
                 "sent_at": (T0 + timedelta(seconds=60 * i)).isoformat(),
                 "model": "claude-opus-5", "session": "s",
                 "usage": {"input_tokens": 1000,
                           "cache_read_input_tokens": 0,
                           "cache_creation_input_tokens": 0}} for i in range(5)]
        fh = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        for r in rows:
            fh.write(json.dumps(r) + "\n")
        fh.close()
        self.addCleanup(os.unlink, fh.name)

        parser = cli.build_parser()
        for invoice in ("nan", "inf", "-5", "0"):
            with self.subTest(invoice=invoice):
                args = parser.parse_args(
                    ["analyze", fh.name, "--format", "json",
                     "--invoice-usd", invoice])
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    cli.cmd_analyze(args)
                json.loads(buf.getvalue(),
                           parse_constant=lambda c: self.fail(
                               f"--invoice-usd {invoice} emitted the "
                               f"non-standard JSON token {c}"))




class TestAbsenceIsNeverProofOfZero(unittest.TestCase):
    """Two ways a missing or malformed number became a confident zero."""

    def test_a_malformed_counter_cannot_hide_a_billed_failure(self):
        """`_billed_input` summed raw counters, so one NaN poisoned the total
        and `nan > 0` is False. A failed row carrying 1,000,000 real input
        tokens beside a NaN read counter looked unbilled: it left coverage,
        `excluded_billed` came back empty, reconciliation passed at 0.0% and
        $5.00 published over the single surviving request."""
        from cacheeconomics.trace import _billed_input
        poisoned = {"input_tokens": 1_000_000,
                    "cache_read_input_tokens": float("nan"),
                    "cache_creation_input_tokens": 0}
        self.assertEqual(_billed_input(poisoned), 1_000_000)

        ok = Request(request_id="ok", sent_at=T0, model="claude-opus-5",
                     agent="a", session="s", ttl_requested="5m",
                     usage={"input_tokens": 1_000_000,
                            "cache_read_input_tokens": 0,
                            "cache_creation_input_tokens": 0}, segments=[], target_id="anthropic/direct")
        bad = Request(request_id="failed", sent_at=T0 + timedelta(seconds=60),
                      model="claude-opus-5", agent="a", session="s",
                      ttl_requested="5m", status=500, usage=poisoned, segments=[], target_id="anthropic/direct")
        ts = TraceSet(requests=[ok, bad], tier=Tier.USAGE_ONLY, source="x")
        self.assertEqual(ts.excluded_billed, {"failed but billed": 1})
        a = analyze(ts, invoice_usd=5.00)
        self.assertFalse(a.reconciliation["within_ship_gate"])
        self.assertFalse(a.spend["input_usd"].released)

    def test_a_row_whose_counters_are_all_malformed_is_still_blocked(self):
        """Summing only the valid counters must not make a wholly malformed row
        look like a clean zero. `has_usage` catches those as blind rows."""
        r = Request(request_id="x", sent_at=T0, model="claude-opus-5",
                    usage={"input_tokens": float("nan")}, segments=[], target_id="anthropic/direct")
        self.assertFalse(r.has_usage)

    def test_a_missing_litellm_split_is_not_proof_of_no_caching(self):
        """`prompt_tokens` is inclusive, so with no split there is no evidence
        which class those tokens fell into. Reconstructing them as uncached
        prices 0.1x reads at 1x and reports a working cache as absent."""
        from cacheeconomics.adapters.litellm import usage_from_payload
        self.assertEqual(
            usage_from_payload({"model": "claude-opus-5", "prompt_tokens": 10000,
                                "completion_tokens": 5}), {})
        # With the split, the same row is read normally.
        got = usage_from_payload({"model": "claude-opus-5", "prompt_tokens": 10000,
                                  "completion_tokens": 5,
                                  "prompt_tokens_details": {"cached_tokens": 9000}})
        self.assertEqual(got["cache_read_input_tokens"], 9000)
        self.assertEqual(got["input_tokens"], 1000)




class TestAFixMustRecoverWhatItClaims(unittest.TestCase):
    """VOL-1 priced a suffix without proving the move would free it.

    `idx` is the *lowest* volatile position and the suffix ran to the outermost
    marker, so a second volatile block sitting between them was ignored. Moving
    the first one alone leaves the prefix invalidated at the second and recovers
    nothing past it. Measured: identical $3,751 whether there was one blocker or
    two, attached to a recommendation naming only the first.
    """

    def _trace(self, blockers):
        reqs = []
        for i in range(40):
            segs = [Segment(id=f"volA{i}", role="system", tokens=200, index=0),
                    Segment(id=(f"volB{i}" if blockers == 2 else "stableB"),
                            role="system", tokens=200, index=1),
                    Segment(id="tools", role="tools", tokens=30000, index=2,
                            cache_marked=True, ttl="5m"),
                    Segment(id=f"turn{i}", role="user", tokens=200, index=3)]
            reqs.append(Request(
                request_id=f"r{i}", sent_at=T0 + timedelta(seconds=120 * i),
                model="claude-opus-5", agent="a", session="s", ttl_requested="5m",
                usage={"input_tokens": 200, "cache_read_input_tokens": 0,
                       "cache_creation_input_tokens": 30400}, segments=segs, target_id="anthropic/direct"))
        return TraceSet(requests=reqs, tier=Tier.INSTRUMENTED, source="x")

    def _vol(self, blockers):
        a = analyze(self._trace(blockers), allow_unreconciled=True)
        return next((f for f in a.findings if f.code == "VOL-1"), None)

    def test_one_blocker_still_carries_a_figure(self):
        self.assertIsNotNone(self._vol(1).avoidable_usd_month)

    def test_a_second_blocker_removes_the_figure(self):
        self.assertIsNone(self._vol(2).avoidable_usd_month)

    def test_but_the_finding_is_still_reported(self):
        """Dropping it silently would be worse than the overstatement: the
        volatility is real and costing money, and the client never hears it."""
        v = self._vol(2)
        self.assertIsNotNone(v)
        self.assertIn("volatile too", v.detail)
        self.assertIn("move together", v.detail)

    def test_the_fix_text_names_the_full_move_set(self):
        self.assertIn("other volatile position", self._vol(2).fix)


class TestAGuardIsUselessBehindACrash(unittest.TestCase):
    """Work added *ahead* of a guard is outside it.

    The `no_split` pre-count dereferenced `metadata.get(...)` before the
    try/except that turns a malformed foreign payload into a counted skip, so
    one row whose metadata was a string took down the whole ingest instead of
    showing up as skipped_rows. Introduced by the note that counter feeds.
    """

    def test_a_non_dict_metadata_or_response_does_not_abort_the_load(self):
        import json
        import tempfile

        from cacheeconomics.adapters.litellm import load_litellm
        rows = [{"model": "claude-opus-5", "status": "success",
                 "startTime": 1785000000, "prompt_tokens": 100,
                 "completion_tokens": 1,
                 "prompt_tokens_details": {"cached_tokens": 0}},
                {"model": "claude-opus-5", "prompt_tokens": 500,
                 "metadata": "a string"},
                {"model": "claude-opus-5", "prompt_tokens": 500,
                 "response": ["a", "list"]},
                {"model": "claude-opus-5", "prompt_tokens": 500,
                 "metadata": 42, "response": "x"}]
        fh = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        for r in rows:
            fh.write(json.dumps(r) + "\n")
        fh.close()
        self.addCleanup(os.unlink, fh.name)
        ts = load_litellm(fh.name)          # must not raise
        self.assertEqual(len(ts.requests), 4)


class TestAKeyIsOnlyRequiredWhereSomethingIsHashed(unittest.TestCase):
    """Refusing without a remedy is not a guard.

    Segments with no trusted id *and no content* cannot be identified by
    anyone -- there is nothing to hash. The loader demanded a key before
    reaching the branch that discards them, so a deliberately
    privacy-preserving export (structure skeleton, no ids, no content) was
    refused on the default path. Measured against the committed code, passing
    the key it asked for produced exactly the USAGE_ONLY downgrade the refusal
    was standing in front of.
    """

    def _write(self, with_content):
        import json
        import tempfile
        rows = [{"request_id": f"r{i}",
                 "sent_at": (T0 + timedelta(seconds=60 * i)).isoformat(),
                 "model": "claude-opus-5",
                 "usage": {"input_tokens": 1000, "cache_read_input_tokens": 0,
                           "cache_creation_input_tokens": 0},
                 "segments": [{"role": "system", "tokens": 900, "index": 0,
                               **({"content": "policy"} if with_content else {})}]}
                for i in range(5)]
        fh = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        for r in rows:
            fh.write(json.dumps(r) + "\n")
        fh.close()
        self.addCleanup(os.unlink, fh.name)
        return fh.name

    def test_a_redacted_export_loads_without_a_key(self):
        from cacheeconomics.trace import load_jsonl
        ts = load_jsonl(self._write(False), None)
        self.assertIs(ts.tier, Tier.USAGE_ONLY)
        self.assertEqual(len(ts.requests), 5)

    def test_content_without_a_key_is_still_refused(self):
        """The protection that is real: a bare digest of a short segment is
        guessable, so hashing content unkeyed stays refused."""
        from cacheeconomics.trace import load_jsonl
        with self.assertRaises(ValueError):
            load_jsonl(self._write(True), None)


class TestStructuralMoneyNeedsCountedTokens(unittest.TestCase):
    """The tool held its own spend to 5% and costed recommendations from 19%.

    `_scale_to_measured` divides the billed input total between segments by byte
    share. Measured against the provider's own tokenizer: 19.2% median error per
    segment, 181% worst. Every structural finding is costed from that split,
    while `PUBLISH_TOLERANCE` refuses to publish spend reconciling worse than
    5%. Two standards, and the looser one was on the number nobody could check.

    Not an INFERRED-only problem: the recorder runs the same estimator over its
    own captures, and had been stamping `tokens_are_estimated: True` on every
    row it wrote since it was written. Nothing read it.
    """

    def _trace(self, **kw):
        segs = [Segment(id=f"vol{{i}}", role="system", tokens=400, index=0),
                Segment(id="tools", role="tools", tokens=30000, index=1,
                        cache_marked=True, ttl="5m"),
                Segment(id="turn", role="user", tokens=200, index=2)]
        reqs = [Request(request_id=f"r{i}", sent_at=T0 + timedelta(seconds=120 * i),
                        model="claude-opus-5", agent="a", session="s",
                        ttl_requested="5m",
                        usage={"input_tokens": 200, "cache_read_input_tokens": 0,
                               "cache_creation_input_tokens": 30400},
                        segments=[replace(s, id=f"{s.id}{i}" if s.index == 0 else s.id)
                                  for s in segs], target_id="anthropic/direct")
                for i in range(40)]
        return TraceSet(requests=reqs, tier=Tier.INSTRUMENTED, source="x", **kw)

    def _vol(self, ts):
        a = analyze(ts, allow_unreconciled=True)
        return next((f for f in a.findings if f.code == "VOL-1"), None)

    def test_estimated_tokens_carry_no_dollar_figure(self):
        v = self._vol(self._trace(tokens_counted=0.0))
        for name in ("avoidable_usd_month", "avoidable_usd_window"):
            with self.subTest(figure=name):
                # Both, not just the monthly one. The report falls back to the
                # window figure wherever the monthly one is missing, so a gate
                # reaching one of them would print the amount it refused.
                fig = getattr(v, name)
                self.assertFalse(fig.released)
                self.assertIn("19.2%", fig.withheld_because)

    def test_counted_tokens_do(self):
        """`avoidable_usd_window`: this fixture is forty requests two minutes
        apart, so its monthly figure is withheld by the projection floor however
        the tokens were sized. The window figure is what counting gates."""
        self.assertTrue(self._vol(self._trace(tokens_counted=1.0))
                        .avoidable_usd_window.released)

    def test_the_finding_still_reports_without_the_figure(self):
        """Withholding the number must not withhold the diagnosis. The volatile
        block is real whether or not its size was counted, and a client who is
        told nothing cannot act on it."""
        v = self._vol(self._trace(tokens_counted=0.0))
        self.assertIsNotNone(v)
        self.assertTrue(v.detail)
        self.assertIn("count_tokens", v.avoidable_usd_month.withheld_because)

    def test_a_partly_counted_trace_does_not_qualify(self):
        """Part measurement and part 19% error, with nothing saying which
        finding drew from which."""
        self.assertFalse(self._vol(self._trace(tokens_counted=0.5))
                         .avoidable_usd_month.released)

    def test_the_recorders_own_declaration_is_believed(self):
        """It stamps `tokens_are_estimated: True` on every row and always has.
        A loader that ignored it was overriding the only component that knows."""
        import json
        import tempfile

        from cacheeconomics.trace import load_jsonl
        row = {"request_id": "r", "sent_at": T0.isoformat(), "model": "claude-opus-5",
               "usage": {"input_tokens": 100, "cache_read_input_tokens": 0,
                         "cache_creation_input_tokens": 9000},
               "segments": [{"id": "hmac:" + "a" * 64, "role": "system",
                             "tokens": 9000, "index": 0, "cache_marked": True,
                             "ttl": "5m"}]}
        for flags, expected in (({"tokens_are_estimated": True}, False),
                                ({"tokens_counted": True}, True),
                                ({}, False),
                                # An explicit denial beats an affirmation.
                                ({"tokens_are_estimated": True,
                                  "tokens_counted": True}, False)):
            with self.subTest(flags=flags):
                fh = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
                fh.write(json.dumps(dict(row, **flags)) + "\n")
                fh.close()
                self.addCleanup(os.unlink, fh.name)
                self.assertIs(load_jsonl(fh.name).tokens_are_counted, expected)


class TestBothRenderersSayWhatToDo(unittest.TestCase):
    """The text report printed the diagnosis and withheld the remedy.

    Every finding carries a computed `fix`. The HTML renderer printed it under
    "Action"; `Finding.describe` did not print it at all. So the report that
    actually gets pasted into an email told the reader what was wrong and never
    what to do, for every finding, and the two renderers disagreed about the
    only part anybody acts on.

    This file already asserted that both renderers agree about *money*. Nothing
    compared what they tell the user to do, which is why it survived.
    """

    def _analysis(self):
        segs = [Segment(id="vol", role="system", tokens=400, index=0),
                Segment(id="tools", role="tools", tokens=30000, index=1,
                        cache_marked=True, ttl="5m"),
                Segment(id="turn", role="user", tokens=200, index=2)]
        reqs = [Request(request_id=f"r{i}", sent_at=T0 + timedelta(seconds=120 * i),
                        model="claude-opus-5", agent="a", session="s",
                        ttl_requested="5m",
                        usage={"input_tokens": 200, "cache_read_input_tokens": 0,
                               "cache_creation_input_tokens": 30400},
                        segments=[replace(s, id=f"{s.id}{i}") if s.index == 0 else s
                                  for s in segs], target_id="anthropic/direct")
                for i in range(40)]
        return analyze(TraceSet(requests=reqs, tier=Tier.INSTRUMENTED, source="x"),
                       allow_unreconciled=True)

    def test_every_fix_reaches_both_renderers(self):
        from cacheeconomics.report import render_html, render_text
        a = self._analysis()
        html, text = render_html(a), render_text(a)
        # Normalised: both renderers wrap, so a fix that survives intact still
        # fails a raw substring check the moment it crosses a wrap point.
        flat_text, flat_html = " ".join(text.split()), " ".join(html.split())
        missing = [f.code for f in a.findings
                   if f.fix and " ".join(f.fix.split())[:40] not in flat_text]
        self.assertEqual(missing, [], "findings whose remedy the text report drops")
        missing_html = [f.code for f in a.findings
                        if f.fix and " ".join(f.fix.split())[:40] not in flat_html]
        self.assertEqual(missing_html, [], "findings whose remedy the HTML report drops")

    def test_a_first_run_says_what_to_run_next(self):
        """It used to end on five notes explaining what it would not answer and
        stop there. A refusal with no next step reads as a dead end."""
        from cacheeconomics.report import _next_steps, render_text
        a = self._analysis()
        steps = _next_steps(a)
        self.assertTrue(steps, "no next steps offered")
        self.assertIn("what to do next", render_text(a))

    def test_the_steps_name_real_flags(self):
        """A step telling someone to pass a flag that does not exist is worse
        than no step."""
        import argparse

        from cacheeconomics import cli, report
        a = self._analysis()
        p = argparse.ArgumentParser()
        cli._ingest_args(p)
        cli._pricing_args(p)
        cli._detail_arg(p)
        p.add_argument("--invoice-usd", type=float)
        p.add_argument("--allow-unreconciled", action="store_true")
        known = {s for a_ in p._actions for s in a_.option_strings}
        for step in report._next_steps(a):
            for word in step.split():
                flag = word.strip(".,'\"")
                if flag.startswith("--"):
                    self.assertIn(flag, known, f"step names unknown flag {flag}")


class TestTheShortVersionLosesNothing(unittest.TestCase):
    """The findings table is the short version of each finding.

    That is a real reduction: on a live run the default output dropped from
    five screens of prose to one table. The risk it introduces is that the
    reasoning becomes unreachable rather than merely folded away, which would
    turn a readability fix into a disclosure regression.
    """

    def _analysis(self):
        segs = [Segment(id="vol", role="system", tokens=400, index=0),
                Segment(id="tools", role="tools", tokens=30000, index=1,
                        cache_marked=True, ttl="5m"),
                Segment(id="turn", role="user", tokens=200, index=2)]
        reqs = [Request(request_id=f"r{i}", sent_at=T0 + timedelta(seconds=120 * i),
                        model="claude-opus-5", agent="a", session="s",
                        ttl_requested="5m",
                        usage={"input_tokens": 200, "cache_read_input_tokens": 0,
                               "cache_creation_input_tokens": 30400},
                        segments=[replace(s, id=f"{s.id}{i}") if s.index == 0 else s
                                  for s in segs], target_id="anthropic/direct")
                for i in range(40)]
        return analyze(TraceSet(requests=reqs, tier=Tier.INSTRUMENTED, source="x"),
                       allow_unreconciled=True)

    def test_detail_carries_every_finding_s_reasoning(self):
        from cacheeconomics.report import render_text
        a = self._analysis()
        flat = " ".join(render_text(a, detail=True).split())
        missing = [f.code for f in a.findings
                   if " ".join(f.detail.split())[:60] not in flat]
        self.assertEqual(missing, [], "--detail drops a finding's reasoning")

    def test_the_default_says_how_to_reach_it(self):
        """Folded away is fine. Folded away silently is not: a reader who cannot
        see the reasoning and is not told it exists has no way to audit a
        number, which is the whole premise of this report."""
        from cacheeconomics.report import render_text
        out = render_text(self._analysis())
        self.assertIn("--detail", out)

    def test_the_default_is_actually_shorter(self):
        from cacheeconomics.report import render_text
        a = self._analysis()
        brief, full = render_text(a), render_text(a, detail=True)
        self.assertLess(len(brief), len(full),
                        "--detail added nothing, so the default hid nothing")

    # Any amount the money column can print, as one pattern. Written against
    # the column rather than against `/mo` alone: this fixture runs 78 minutes,
    # so the projection floor withholds every monthly figure on it and a check
    # spelled `~\$[\d,]+/mo` went vacuous the moment that floor reached the
    # per-finding figures -- it stopped finding an amount, and its own guard is
    # the only reason that was visible rather than silent.
    ANY_AMOUNT = r"~\$[\d,]+(?:\.\d+)?(?: \[DRAFT\])?/(?:mo|window)"

    def test_the_money_column_never_shows_a_number_the_gate_withheld(self):
        """The table is a new surface for a figure to escape through. It reads
        each Figure's own release state, so this checks the wiring rather than
        restating the gate: same trace, no `allow_unreconciled`, no amounts."""
        import re

        from cacheeconomics.report import render_text
        released = self._analysis()
        self.assertRegex(render_text(released), self.ANY_AMOUNT,
                         "fixture produces no amounts, so the check is vacuous")

        gated = self._gated()
        self.assertFalse(any(f.avoidable_usd_month and f.avoidable_usd_month.released
                             for f in gated.findings))
        self.assertFalse(any(f.avoidable_usd_window and f.avoidable_usd_window.released
                             for f in gated.findings))
        self.assertIsNone(re.search(self.ANY_AMOUNT, render_text(gated)))

    def test_the_column_prints_the_window_amount_when_the_month_is_withheld(self):
        """What the released run above is actually showing, named rather than
        left to a regex alternation.

        Without this the migration of that pattern reads as loosening it. The
        claim is specific: this fixture reconciles, its monthly figures are
        withheld because 78 minutes cannot be scaled to a month, and the amount
        the window *did* support is printed instead of nothing at all.
        """
        from cacheeconomics.report import render_text
        a = self._analysis()
        eff = next(f for f in a.findings if f.code == "EFF-1")
        self.assertFalse(eff.avoidable_usd_month.released,
                         "fixture no longer sits below the projection floor")
        self.assertTrue(eff.avoidable_usd_window.released)
        out = render_text(a)
        self.assertIn(f"~{eff.avoidable_usd_window}/window", out)
        self.assertNotIn("/mo", out)

    def _gated(self):
        """The same analysis with the gate left on."""
        segs = [Segment(id="vol", role="system", tokens=400, index=0),
                Segment(id="tools", role="tools", tokens=30000, index=1,
                        cache_marked=True, ttl="5m"),
                Segment(id="turn", role="user", tokens=200, index=2)]
        reqs = [Request(request_id=f"r{i}", sent_at=T0 + timedelta(seconds=120 * i),
                        model="claude-opus-5", agent="a", session="s",
                        ttl_requested="5m",
                        usage={"input_tokens": 200, "cache_read_input_tokens": 0,
                               "cache_creation_input_tokens": 30400},
                        segments=[replace(s, id=f"{s.id}{i}") if s.index == 0 else s
                                  for s in segs], target_id="anthropic/direct")
                for i in range(40)]
        return analyze(TraceSet(requests=reqs, tier=Tier.INSTRUMENTED, source="x"))


class TestTheDraftStampSurvives(unittest.TestCase):
    """`--allow-unreconciled` releases dollar figures that no invoice has
    checked, and stamps the report DRAFT so nobody forwards it as fact.

    That stamp used to be the first of five notes at the bottom of the report.
    Every test on it asserted against `Analysis.notes` -- the object, not the
    rendered page -- so nothing would have failed if a renderer stopped
    printing it. Removing the notes section from the default text view nearly
    did exactly that.
    """

    def _analysis(self, **kw):
        segs = [Segment(id="vol", role="system", tokens=400, index=0),
                Segment(id="tools", role="tools", tokens=30000, index=1,
                        cache_marked=True, ttl="5m")]
        reqs = [Request(request_id=f"r{i}", sent_at=T0 + timedelta(seconds=120 * i),
                        model="claude-opus-5", agent="a", session="s",
                        ttl_requested="5m",
                        usage={"input_tokens": 200, "cache_read_input_tokens": 0,
                               "cache_creation_input_tokens": 30400},
                        segments=[replace(s, id=f"{s.id}{i}") if s.index == 0 else s
                                  for s in segs], target_id="anthropic/direct")
                for i in range(40)]
        return analyze(TraceSet(requests=reqs, tier=Tier.INSTRUMENTED,
                                source="x"), **kw)

    def test_both_renderers_stamp_a_released_report(self):
        from cacheeconomics.report import render_html, render_text
        a = self._analysis(allow_unreconciled=True)
        self.assertTrue(any(n.startswith("DRAFT") for n in a.notes),
                        "fixture is not a draft, so the check is vacuous")
        for name, page in (("text", render_text(a)),
                           ("text --detail", render_text(a, detail=True)),
                           ("html", render_html(a))):
            with self.subTest(renderer=name):
                # Normalised: every renderer wraps, and the stamp is long
                # enough to cross a wrap point in all three.
                self.assertIn("not for external use", " ".join(page.split()).lower())

    def test_it_is_near_the_top_where_somebody_forwarding_will_see_it(self):
        """At the bottom of a five-section report it warns the one reader who
        already finished reading."""
        from cacheeconomics.report import render_text
        page = render_text(self._analysis(allow_unreconciled=True))
        self.assertLess(page.index("DRAFT"), len(page) // 4,
                        "the stamp is buried below the first quarter")

    def test_an_unreleased_report_is_not_stamped(self):
        """Matched on the stamp, not on the word. `DRAFT` also appears in the
        next-step that *offers* --allow-unreconciled, so asserting on the bare
        word would have failed on a report that was never stamped at all."""
        from cacheeconomics.report import render_text
        page = " ".join(render_text(self._analysis()).split())
        self.assertNotIn("DRAFT — figures released", page)
        self.assertIn("--allow-unreconciled", page, "the offer should still be made")

    def test_the_notes_are_folded_away_but_not_gone(self):
        from cacheeconomics.report import render_text
        a = self._analysis()
        brief, full = render_text(a), render_text(a, detail=True)
        for note in a.notes:
            with self.subTest(note=note[:40]):
                self.assertIn(" ".join(note.split())[:50], " ".join(full.split()))
        self.assertIn("--detail", brief, "notes vanished with no way to reach them")


class TestNoteKindIsRecordedNotInferred(unittest.TestCase):
    """A note's kind is decided where it is raised, not where it is rendered.

    Both renderers used to search note prose for one phrase to decide whether a
    note qualified a published figure. That means a rewording silently demotes
    a release blocker to provenance, and provenance is folded behind --detail --
    so the sentence saying what a dollar figure excludes stops being printed
    beside the dollar figure, with nothing failing.
    """

    def _analysis(self, **kw):
        """No segments and no declared TTL, so the write lifetime is unprovable
        and those requests are excluded from the dollar figures. That exclusion
        is the blocker this class is about."""
        reqs = [Request(request_id=f"r{i}", sent_at=T0 + timedelta(seconds=120 * i),
                        model="claude-opus-5", agent="a", session="s",
                        ttl_requested=None,
                        usage={"input_tokens": 200,
                               "cache_creation_input_tokens": 30000},
                        segments=[], target_id="anthropic/direct") for i in range(20)]
        return analyze(TraceSet(requests=reqs, tier=Tier.USAGE_ONLY, source="x"),
                       **kw)

    def test_the_analysis_carries_the_classification(self):
        from cacheeconomics.analyzer import spend_caveats
        a = self._analysis(allow_unreconciled=True)
        self.assertTrue(a.blocking_notes, "fixture raises no blocker; vacuous")
        self.assertEqual(spend_caveats(a), a.blocking_notes)
        for n in a.blocking_notes:
            self.assertIn(n, a.notes, "a blocker that is not among the notes")

    def test_renderers_read_the_field_not_the_prose(self):
        """Rewriting a blocker's wording must not change whether it is shown."""
        from dataclasses import replace

        from cacheeconomics.analyzer import spend_caveats
        from cacheeconomics.report import render_text
        a = self._analysis(allow_unreconciled=True)
        original = a.blocking_notes[0]
        reworded = "Some requests were left out of the totals. " + original[-40:]
        b = replace(a, notes=[reworded] + a.notes[1:],
                    blocking_notes=[reworded])
        self.assertEqual(spend_caveats(b), [reworded])
        self.assertIn(" ".join(reworded.split())[:50],
                      " ".join(render_text(b).split()),
                      "a reworded blocker stopped being printed by default")

    def test_a_bare_list_still_works_for_callers_without_an_analysis(self):
        """Adapters hold notes before an Analysis exists."""
        from cacheeconomics.analyzer import spend_caveats
        from cacheeconomics.trace import QUALIFIES_SPEND
        notes = [f"9 rows are {QUALIFIES_SPEND}.", "ids were normalised"]
        self.assertEqual(len(spend_caveats(notes)), 1)

    def test_an_ingest_blocker_survives_into_the_analysis(self):
        """The adapter knows things the analyzer cannot see -- that a row stated
        no surface, for one -- so its classification has to travel."""
        import json
        import os
        import tempfile

        from cacheeconomics.adapters.litellm import load_litellm
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "x.jsonl")
            with open(p, "w") as f:
                for i in range(30):
                    f.write(json.dumps({
                        "id": f"r{i}", "startTime": 1_780_000_000 + i * 60,
                        "model": "claude-opus-5",
                        "response": {"usage": {
                            "prompt_tokens": 1000, "completion_tokens": 10,
                            "prompt_tokens_details": {"cached_tokens": 0},
                            "cache_creation_input_tokens": 0,
                            "cache_read_input_tokens": 0}}}) + "\n")
            ts = load_litellm(p)
            self.assertTrue(ts.blocking_notes, "adapter classified nothing")
            a = analyze(ts, allow_unreconciled=True)
            self.assertTrue(any("surface is unknown" in n for n in a.blocking_notes),
                            "the ingest blocker did not survive into the analysis")


class TestReconciliationDollarsGoThroughTheGate(unittest.TestCase):
    """The reconciliation block published the number the gate had just refused.

    On a failed reconciliation the HTML report printed "$16.97" and
    `--format json` emitted `computed_usd: 16.97244499999996` as a bare float,
    while every other figure in the same document read "[withheld: ...]". The
    text report printed neither, so this was also a twin-path divergence: two
    renderers of one analysis disagreeing about whether a number was publishable.

    `invoice_usd` is deliberately still a plain number. It is the reader's own
    input, not a claim the tool is making. `delta_pct` is deliberately still a
    plain number too: a ratio is not a spend total, and it is the entire
    diagnostic -- the reason for a refusal has to survive the refusal.
    """

    FIXTURE = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "fixtures", "demo-traces.jsonl")

    def _analysis(self, invoice):
        from cacheeconomics.trace import load_jsonl
        return analyze(load_jsonl(self.FIXTURE), invoice_usd=invoice)

    def test_a_failed_gate_withholds_the_computed_total(self):
        a = self._analysis(1.00)
        self.assertFalse(a.reconciliation["within_ship_gate"])
        for key in ("computed_usd", "delta_usd"):
            with self.subTest(key=key):
                self.assertFalse(a.reconciliation[key].released)

    def test_no_renderer_prints_it(self):
        import re

        from cacheeconomics.report import render_html, render_text
        a = self._analysis(1.00)
        for name, page in (("text", render_text(a)), ("html", render_html(a))):
            with self.subTest(renderer=name):
                amounts = set(re.findall(r"\$[0-9][0-9,]*\.?[0-9]*", page))
                # The invoice is the reader's own number and may appear.
                self.assertEqual(amounts - {"$1.00"}, set(),
                                 f"{name} printed a figure the gate withheld")

    def test_json_does_not_serialise_it_as_a_number(self):
        """A dashboard consuming this would read the float as authoritative."""
        import json
        import subprocess
        import sys
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out = subprocess.run(
            [sys.executable, "-m", "cacheeconomics.cli", "analyze", self.FIXTURE,
             "--invoice-usd", "1.00", "--format", "json"],
            capture_output=True, text=True,
            env=dict(os.environ, PYTHONPATH=root)).stdout
        recon = json.loads(out)["reconciliation"]
        for key in ("computed_usd", "delta_usd"):
            with self.subTest(key=key):
                self.assertIsInstance(recon[key], str)
                self.assertIn("withheld", recon[key])

    def test_a_passing_gate_still_publishes_both(self):
        """The refusal has to be escapable, or reconciliation stops being
        readable on the path where it worked."""
        a = self._analysis(17.45)
        self.assertTrue(a.reconciliation["within_ship_gate"])
        self.assertTrue(a.reconciliation["computed_usd"].released)
        self.assertIn("$", str(a.reconciliation["computed_usd"]))

    def test_the_reason_survives_the_refusal(self):
        """delta_pct is not money and must keep rendering, or the reader cannot
        tell a 3% miss from a 1,600% one."""
        a = self._analysis(1.00)
        self.assertIsInstance(a.reconciliation["delta_pct"], float)
        self.assertGreater(a.reconciliation["delta_pct"], 100)
        self.assertEqual(a.reconciliation["invoice_usd"], 1.00)


class TestTheRuntimeAndTheReportAgreePerFindingCode(unittest.TestCase):
    """Three checks exist in both paths and each drifted, because nothing here
    had ever compared them by code.

    Every other test in this file takes one *quantity* and pins the two
    implementations to the same answer. These take one *fixture* and pin the two
    to the same verdict, which is the thing an operator actually experiences: a
    dashboard that says fix this and a report that says there is nothing to fix
    are not two opinions, they are a tool that cannot be trusted on either.

    What each of them was doing before:

      RT-MIN summed to the outermost marker, so an inner marker below the
      threshold read as a 30k prefix and never surfaced, while MIN-1 walked
      every marker and reported it.

      RT-FANOUT used a flat five seconds, so on a target whose first token lands
      at 14s an 8-second sibling was called sequential, while FAN-1 used the
      observed boundary and called it concurrent.

      RT-TTL fired on the scope's median request gap, so a workload marking a
      fresh prefix every request was told to switch to a one-hour lifetime,
      while TTL-1 walked a per-prefix timeline, found nothing written twice, and
      refused. The runtime was the one with no evidence, and it was the one that
      spoke.

    Both directions are pinned. A rule that never fires agrees with a rule that
    never fires, and that is not the property being asked for.
    """

    PAIRS = (("MIN-1", "RT-MIN"), ("FAN-1", "RT-FANOUT"), ("TTL-1", "RT-TTL"))

    def _both(self, reqs, batch_code, runtime_code):
        ts = TraceSet(requests=reqs, tier=Tier.INSTRUMENTED, source="twin")
        batch = [f for f in analyze(ts).findings if f.code == batch_code]
        m, fired = monitor.Monitor(), []
        for r in reqs:
            fired += m.observe(r)
        runtime = [a for a in fired if a.code == runtime_code]
        return bool(batch), bool(runtime)

    def _reqs(self, segs, n=12, gap=600, **kw):
        base = dict(model="claude-opus-5", target_id="anthropic/direct",
                    ttl_requested="5m", session="s",
                    usage={"input_tokens": 30_200,
                           "cache_creation_input_tokens": 30_000})
        base.update(kw)
        out = []
        for i in range(n):
            at = T0 + timedelta(seconds=gap * i)
            out.append(Request(request_id=f"r{i}", sent_at=at,
                               segments=segs(i),
                               **{**base, "usage": dict(base["usage"])}))
        return out

    # -- an inner marker that cannot cache ---------------------------------

    def _short_inner(self, i):
        return [Segment(id="sys", role="system", tokens=200, index=0, cache_marked=True),
                Segment(id="tools", role="tools", tokens=20_000, index=1),
                Segment(id="policy", role="system", tokens=10_000, index=2),
                Segment(id="ctx", role="system", tokens=1_000, index=3, cache_marked=True),
                Segment(id=f"turn{i}", role="user", tokens=200, index=9)]

    def test_a_short_inner_marker_is_seen_by_both(self):
        batch, runtime = self._both(self._reqs(self._short_inner, n=6, gap=60),
                                    "MIN-1", "RT-MIN")
        self.assertTrue(batch)
        self.assertEqual(batch, runtime)

    def test_a_prefix_over_the_minimum_is_seen_by_neither(self):
        def segs(i):
            return [Segment(id="tools", role="tools", tokens=30_000, index=0,
                            cache_marked=True),
                    Segment(id=f"turn{i}", role="user", tokens=200, index=1)]
        batch, runtime = self._both(self._reqs(segs, n=6, gap=60), "MIN-1", "RT-MIN")
        self.assertFalse(batch)
        self.assertEqual(batch, runtime)

    # -- concurrency measured against the observed boundary ----------------

    def _fanned(self, spacing, latency):
        segs = [Segment(id="tools", role="tools", tokens=30_000, index=0,
                        cache_marked=True),
                Segment(id="turn", role="user", tokens=200, index=1)]
        out = []
        for i in range(4):
            for j in range(2):
                at = T0 + timedelta(seconds=600 * i + spacing * j)
                out.append(Request(
                    request_id=f"r{i}-{j}", sent_at=at,
                    first_token_at=(at + timedelta(seconds=latency)) if latency else None,
                    model="claude-opus-5", target_id="anthropic/direct",
                    ttl_requested="5m", session="s",
                    usage={"input_tokens": 30_200,
                           "cache_creation_input_tokens": 30_000},
                    segments=list(segs)))
        return out

    def test_a_slow_first_token_makes_both_call_it_concurrent(self):
        """Eight seconds apart, first token at fourteen. Outside the flat
        window and inside the observed one."""
        batch, runtime = self._both(self._fanned(8, 14), "FAN-1", "RT-FANOUT")
        self.assertTrue(batch)
        self.assertEqual(batch, runtime)

    def test_without_first_token_timing_both_fall_back_together(self):
        batch, runtime = self._both(self._fanned(8, None), "FAN-1", "RT-FANOUT")
        self.assertFalse(batch)
        self.assertEqual(batch, runtime)

    # -- lifetime advice needs a span written twice ------------------------

    def test_a_stable_prefix_rewritten_in_band_is_seen_by_both(self):
        def segs(i):
            return [Segment(id="tools", role="tools", tokens=30_000, index=0,
                            cache_marked=True),
                    Segment(id=f"turn{i}", role="user", tokens=200, index=1)]
        batch, runtime = self._both(self._reqs(segs), "TTL-1", "RT-TTL")
        self.assertTrue(batch)
        self.assertEqual(batch, runtime)

    def test_a_prefix_that_is_never_rewritten_is_refused_by_both(self):
        """A different span every request. The cadence looks identical to the
        case above and there is nothing a longer lifetime could recover."""
        def segs(i):
            return [Segment(id=f"tools{i}", role="tools", tokens=30_000, index=0,
                            cache_marked=True),
                    Segment(id=f"turn{i}", role="user", tokens=200, index=1)]
        batch, runtime = self._both(self._reqs(segs), "TTL-1", "RT-TTL")
        self.assertFalse(batch)
        self.assertEqual(batch, runtime)

    # -- the rolling conversation, and everything it must not swallow --------

    def _rolling(self, n=12, gap=600, root=lambda i: "sys", ttl="5m"):
        """A stable prefix marked at index 0 and an advancing marker at the end
        of history. The most common agent shape there is, and the one both
        rules refused because the span at the outermost marker is different on
        every request."""
        out = []
        for i in range(n):
            segs = [Segment(id=root(i), role="system", tokens=30_000, index=0,
                            cache_marked=True, ttl=ttl)]
            for t in range(i + 1):
                segs.append(Segment(id=f"turn{t}", role="user", tokens=500,
                                    index=t + 1, cache_marked=(t == i),
                                    ttl=ttl if t == i else None))
            out.append(Request(
                request_id=f"r{i}", sent_at=T0 + timedelta(seconds=gap * i),
                model="claude-opus-5", target_id="anthropic/direct",
                ttl_requested=ttl, session="s",
                usage={"input_tokens": 30_000 + 500 * (i + 1),
                       "cache_creation_input_tokens": 30_000 + 500 * (i + 1)},
                segments=segs))
        return out

    def test_a_rolling_conversation_in_band_is_seen_by_both(self):
        batch, runtime = self._both(self._rolling(), "TTL-1", "RT-TTL")
        self.assertTrue(batch, "the canonical shape this rule exists for")
        self.assertEqual(batch, runtime)

    def test_rolling_under_five_minutes_is_refused_by_both(self):
        """The 5m entry was still alive, so whatever caused the miss, a longer
        lifetime is not the fix. EFF-1 and VOL-1 name those causes."""
        batch, runtime = self._both(self._rolling(gap=60), "TTL-1", "RT-TTL")
        self.assertFalse(batch)
        self.assertEqual(batch, runtime)

    def test_rolling_beyond_an_hour_is_refused_by_both(self):
        batch, runtime = self._both(self._rolling(gap=7200), "TTL-1", "RT-TTL")
        self.assertFalse(batch)
        self.assertEqual(batch, runtime)

    def test_a_workload_already_on_one_hour_is_not_told_to_switch_to_one_hour(self):
        batch, runtime = self._both(self._rolling(ttl="1h"), "TTL-1", "RT-TTL")
        self.assertFalse(batch)
        self.assertEqual(batch, runtime)

    def test_a_drifting_root_is_refused_by_both(self):
        """Containment must not degrade into "anything matches". These ids
        differ at position 0, so no span is a prefix of any later one and there
        is nothing a lifetime could recover."""
        batch, runtime = self._both(
            self._rolling(root=lambda i: f"sys{i}"), "TTL-1", "RT-TTL")
        self.assertFalse(batch)
        self.assertEqual(batch, runtime)

    def test_the_two_ttl_rules_never_both_speak(self):
        """TTL-1 and TTL-2 are mirrors -- "move to 1h" and "5m would do". This
        is the first change that could make them collide on one trace, and a
        report carrying both tells the reader to do two opposite things."""
        for label, reqs in (("rolling 5m", self._rolling()),
                            ("rolling 1h", self._rolling(ttl="1h")),
                            ("rolling fast", self._rolling(gap=60))):
            with self.subTest(trace=label):
                ts = TraceSet(requests=reqs, tier=Tier.INSTRUMENTED, source="twin")
                codes = [f.code for f in analyze(ts).findings]
                self.assertFalse("TTL-1" in codes and "TTL-2" in codes,
                                 f"both fired on {label}: {codes}")

    def test_the_credited_figure_cannot_exceed_what_was_written(self):
        """Under containment the request writes the prefix *plus* a new turn,
        and only the prefix becomes a read. Crediting the whole write bills the
        turn as recovered too, which overstates a client's saving."""
        reqs = self._rolling()
        ts = TraceSet(requests=reqs, tier=Tier.INSTRUMENTED, source="twin")
        a = analyze(ts, invoice_usd=500.0)
        found = [f for f in a.findings if f.code == "TTL-1"]
        self.assertTrue(found)
        written = sum(r.usage["cache_creation_input_tokens"] for r in reqs)
        # Every written token recovered at the full write-to-read delta is the
        # ceiling no honest figure can pass. Scaled to a month the same way the
        # finding is, or the comparison is between two different units.
        rate = registry.base_rate("claude-opus-5", "2026-07-29", "anthropic/direct")
        mult = registry.multipliers("anthropic/direct")
        ceiling = (written * (rate / 1e6) * (mult["write_5m"] - mult["read"])
                   * (30.0 / a.window_days))
        self.assertLess(found[0].avoidable_usd_month.raw(), ceiling)

    def _tool_loop(self, stride, n=12, gap=600):
        """A tool-heavy agent: `stride` messages appended per LLM call, which
        is what a tool call/result loop looks like on the wire. Content is
        perfectly stable, so containment always holds -- the only question is
        whether the provider could still reach the entry."""
        out = []
        for i in range(n):
            segs = [Segment(id="sys", role="system", tokens=20_000, index=0)]
            total = stride * (i + 1)
            for t in range(total):
                segs.append(Segment(id=f"m{t}", role="user", tokens=100,
                                    index=t + 1, cache_marked=(t == total - 1),
                                    ttl="5m" if t == total - 1 else None))
            wrote = 20_000 + 100 * total
            out.append(Request(
                request_id=f"r{i}", sent_at=T0 + timedelta(seconds=gap * i),
                model="claude-opus-5", target_id="anthropic/direct",
                ttl_requested="5m", session="s",
                usage={"input_tokens": wrote, "cache_creation_input_tokens": wrote},
                segments=segs))
        return out

    def test_a_read_past_the_lookback_window_is_credited_by_neither(self):
        """The provider searches back a bounded number of blocks from a
        breakpoint. `lookback_blocks` is 20 on this surface and is recorded.

        A tool loop appending 25 messages a call puts every marker 25 blocks
        past the previous one, so no earlier entry is reachable -- and both
        rules credited them anyway, publishing $836/month, rising to $1,058 at
        a stride of 40. The figure grew as it got more wrong.
        """
        window = registry.capability("anthropic/direct", "lookback_blocks")
        self.assertEqual(window, 20, "fixture is calibrated to this number")
        batch, runtime = self._both(self._tool_loop(window + 5), "TTL-1", "RT-TTL")
        self.assertFalse(batch, "credited a read the provider would not give")
        self.assertEqual(batch, runtime)

    def test_a_read_inside_the_window_is_still_credited_by_both(self):
        """The other direction, and the one that makes the test above mean
        something: bounding must not silence the rule wholesale."""
        window = registry.capability("anthropic/direct", "lookback_blocks")
        batch, runtime = self._both(self._tool_loop(window - 5), "TTL-1", "RT-TTL")
        self.assertTrue(batch)
        self.assertEqual(batch, runtime)

    def test_an_unrecorded_window_falls_back_to_reachability_alone(self):
        """`lookback=None` means the surface records no window. Unbounded is the
        honest reading of a number nobody wrote down; inventing 20 there would
        be the same fabrication this package refuses everywhere else."""
        near = (5, 400, ("a", "b", "c", "d", "e"))
        far = [(500, 40_000, tuple("abcde") + tuple(f"x{i}" for i in range(495)))]
        self.assertTrue(trace.span_is_reusable_by(near, far, None))
        self.assertFalse(trace.span_is_reusable_by(near, far, 20))

    def test_the_turn_delta_is_not_billed_as_recovered(self):
        """Sharper than the ceiling above. Grow the per-turn delta and the
        matched prefix does not change, so a figure that scales with the delta
        is crediting content that is new under either lifetime."""
        def rolling_with_turn(turn_tokens):
            out = []
            for i in range(12):
                segs = [Segment(id="sys", role="system", tokens=30_000, index=0,
                                cache_marked=True, ttl="5m")]
                for t in range(i + 1):
                    segs.append(Segment(id=f"turn{t}", role="user",
                                        tokens=turn_tokens, index=t + 1,
                                        cache_marked=(t == i),
                                        ttl="5m" if t == i else None))
                wrote = 30_000 + turn_tokens * (i + 1)
                out.append(Request(
                    request_id=f"r{i}", sent_at=T0 + timedelta(seconds=600 * i),
                    model="claude-opus-5", target_id="anthropic/direct",
                    ttl_requested="5m", session="s",
                    usage={"input_tokens": wrote,
                           "cache_creation_input_tokens": wrote},
                    segments=segs))
            return out

        def figure(turn_tokens):
            ts = TraceSet(requests=rolling_with_turn(turn_tokens),
                          tier=Tier.INSTRUMENTED, source="twin")
            f = [x for x in analyze(ts, invoice_usd=500.0).findings
                 if x.code == "TTL-1"]
            self.assertTrue(f, f"no finding at turn={turn_tokens}")
            return f[0].avoidable_usd_month.raw()

        small, large = figure(500), figure(5_000)
        # Ten times the delta. The recovered prefix is bounded by the previous
        # write, which grows only by the deltas already counted, so the figure
        # must not grow anywhere near tenfold.
        self.assertLess(large, small * 4,
                        f"figure tracks the delta: {small:.2f} -> {large:.2f}")

    def test_every_pair_has_a_case_where_it_fires(self):
        """Guards the guard. Each pair above needs at least one fixture on which
        both sides fire, or `assertEqual(batch, runtime)` is satisfied by two
        rules that have been switched off."""
        fired = {
            "MIN-1": self._both(self._reqs(self._short_inner, n=6, gap=60),
                                "MIN-1", "RT-MIN"),
            "FAN-1": self._both(self._fanned(8, 14), "FAN-1", "RT-FANOUT"),
            "TTL-1": self._both(self._reqs(
                lambda i: [Segment(id="tools", role="tools", tokens=30_000,
                                   index=0, cache_marked=True),
                           Segment(id=f"turn{i}", role="user", tokens=200, index=1)]),
                "TTL-1", "RT-TTL"),
        }
        for code, (batch, runtime) in fired.items():
            with self.subTest(code=code):
                self.assertTrue(batch, f"{code} fires on no fixture here")
                self.assertTrue(runtime, f"{code}'s runtime twin fires on no fixture")


if __name__ == "__main__":
    unittest.main()


class TestBothLoadersMeasureCoverageInMoney(unittest.TestCase):
    """`tokens_counted` decides whether structural dollars are published, and
    the two loaders computed it differently.

    `load_jsonl` weights by billed input tokens and carries the reasoning:
    ninety-nine tiny counted rows beside one huge uncounted one is 99% of rows
    and can be 0.02% of the spend. `load_bodies` divided counted rows by
    segmented rows, so a bodies export whose dominant cost was never counted
    reported near-perfect coverage and released money resting on a byte-share
    estimate -- which measures 19.2% off at the median.

    The lesson was learned in one loader and never reached the other. Both call
    `trace.counted_share` now; these pin that they answer alike.
    """

    KEY = b"k" * 32

    @staticmethod
    def _body(n):
        return {"model": "claude-opus-5",
                "system": [{"type": "text", "text": "x" * n}],
                "messages": [{"role": "user", "content": "hi"}]}

    def _write(self, rows):
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        with open(path, "w") as f:
            f.write("\n".join(json.dumps(r) for r in rows))
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        return path

    def _bodies(self, big_counted):
        rows = [{"request": self._body(40),
                 "response": {"usage": {"input_tokens": 10}},
                 "segment_tokens": [5, 5]} for _ in range(99)]
        rows = [_vouched(r, target_id="anthropic/direct")
                for r in rows]
        big = {"request": self._body(400_000),
               "response": {"usage": {"input_tokens": 1_000_000}}}
        if big_counted:
            big["segment_tokens"] = [500_000, 500_000]
            _vouched(big, target_id="anthropic/direct")
        rows.append(big)
        from cacheeconomics.adapters.bodies import load_bodies
        return load_bodies(self._write(rows), self.KEY,
                           target_id="anthropic/direct").tokens_counted

    def _jsonl(self, big_counted):
        def row(i, tokens, counted):
            return {"request_id": f"r{i}", "sent_at": f"2026-07-29T09:{i % 60:02d}:00Z",
                    "model": "claude-opus-5", "session": "s",
                    "target_id": "anthropic/direct",
                    "usage": {"input_tokens": tokens},
                    "tokens_counted": counted,
                    "segments": [{"id": "hmac:" + "a" * 64, "role": "system",
                                  "tokens": tokens, "index": 0,
                                  "cache_marked": False, "ttl": None}]}
        rows = [row(i, 10, True) for i in range(99)]
        rows.append(row(99, 1_000_000, big_counted))
        from cacheeconomics.trace import load_jsonl
        return load_jsonl(self._write(rows), self.KEY).tokens_counted

    def test_the_expensive_uncounted_row_dominates_in_both(self):
        b, j = self._bodies(False), self._jsonl(False)
        self.assertLess(b, 0.05, f"bodies reported {b:.2%} on 0.001 of the money")
        self.assertLess(j, 0.05)
        self.assertAlmostEqual(b, j, places=3)

    def test_counting_the_expensive_row_clears_both(self):
        """The other direction, so the fix cannot be 'always report near zero'."""
        b, j = self._bodies(True), self._jsonl(True)
        self.assertGreater(b, 0.95)
        self.assertGreater(j, 0.95)
        self.assertAlmostEqual(b, j, places=3)

    def test_a_zero_usage_trace_falls_back_to_rows_in_both(self):
        """With no billed tokens to weight by, dividing by zero would read as
        fully counted. Both fall back to counting rows."""
        self.assertEqual(trace.counted_share([], lambda x: True, lambda x: 0), 0.0)
        items = [("a", True), ("b", False)]
        self.assertEqual(
            trace.counted_share(items, lambda p: p[1], lambda p: 0), 0.5)


class TestTheCountedClaimIsStatedInMoney(unittest.TestCase):
    """`tokens_counted` scopes its ratio to rows that carry structure, in both
    loaders, and that is deliberate -- it answers "are the sizes we have exact",
    while `structural_coverage_billed` answers "is there structure at all".

    But the two compose. A capture whose no-body rows carry 9% of the billed
    tokens reports `tokens_counted == 1.0` *and* clears the 90% structural
    floor, so both gates pass while a tenth of the spend was never counted --
    and the note beside it said "exact for all 99 request(s)", which a reader
    takes as "the figures rest on exact counts".

    The denominator is left alone on purpose: making the two gates agree by
    conflating them would be a worse answer than having each say what it covers.
    What changed is that the claim is stated in money.
    """

    KEY = b"k" * 32

    def _notes(self, nobody_tokens):
        from cacheeconomics.adapters.bodies import load_bodies
        rows = [{"request": {"model": "claude-opus-5",
                             "system": [{"type": "text", "text": "x" * 4000}],
                             "messages": [{"role": "user", "content": "hi"}]},
                 "response": {"usage": {"input_tokens": 10_000}},
                 "segment_tokens": [500, 500]} for _ in range(99)]
        rows = [_vouched(r, target_id="anthropic/direct")
                for r in rows]
        if nobody_tokens:
            rows.append({"response": {"usage": {"input_tokens": nobody_tokens}}})
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        with open(path, "w") as f:
            f.write("\n".join(json.dumps(r) for r in rows))
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        ts = load_bodies(path, self.KEY, target_id="anthropic/direct")
        return [n for n in ts.notes if "exact for" in n], ts

    def test_the_claim_names_the_share_of_spend_it_covers(self):
        notes, _ = self._notes(100_000)
        self.assertTrue(notes)
        self.assertIn("90.8% of the billed input tokens", notes[0])

    def test_and_says_what_the_remainder_is(self):
        notes, _ = self._notes(100_000)
        self.assertIn("9.2%", notes[0])
        self.assertIn("no recognisable body", notes[0])

    def test_a_fully_bodied_capture_says_one_hundred_percent(self):
        """The other direction: a real 100% must not carry a caveat about a
        remainder that does not exist."""
        notes, _ = self._notes(0)
        self.assertIn("100.0% of the billed input tokens", notes[0])
        self.assertNotIn("no recognisable body", notes[0])

    def test_the_two_gates_still_measure_different_things(self):
        """Pins the deliberate part. If someone later makes `tokens_counted`
        unconditional, these stop differing and this test says so."""
        _, ts = self._notes(100_000)
        self.assertEqual(ts.tokens_counted, 1.0)
        self.assertLess(ts.structural_coverage_billed, 0.99)


class TestASpanTooShortToCacheIsNotReused(unittest.TestCase):
    """`simulate()` drops markers below the provider's minimum before they can
    be read: below it no entry is written and the provider returns no error.

    `span_is_reusable_by` implemented three of the simulator's four read
    conditions -- containment, reachability, lookback -- and not that one. So a
    span too short to have been written could still be matched and priced as a
    recovered read, and MIN-1 and TTL-1 could appear in one report, one saying
    the marker cannot cache and the other pricing the recovery from it.
    """

    def _reqs(self, marker_tokens):
        out = []
        for i in range(12):
            segs = [Segment(id="sys", role="system", tokens=marker_tokens,
                            index=0, cache_marked=True, ttl="5m"),
                    Segment(id=f"turn{i}", role="user", tokens=100, index=1)]
            w = marker_tokens + 100
            out.append(Request(
                request_id=f"r{i}", sent_at=T0 + timedelta(seconds=600 * i),
                model="claude-opus-5", target_id="anthropic/direct",
                ttl_requested="5m", session="s",
                usage={"input_tokens": w, "cache_creation_input_tokens": w},
                segments=segs))
        return out

    def _codes(self, marker_tokens):
        ts = TraceSet(requests=self._reqs(marker_tokens),
                      tier=Tier.INSTRUMENTED, source="twin")
        return {f.code for f in analyze(ts, invoice_usd=500.0).findings}

    def test_a_sub_minimum_marker_earns_no_ttl_recovery(self):
        minimum = registry.min_cacheable_tokens("anthropic/direct", "claude-opus-5")
        codes = self._codes(minimum // 2)
        self.assertIn("MIN-1", codes, "fixture no longer trips the minimum rule")
        self.assertNotIn("TTL-1", codes)

    def test_a_marker_above_the_minimum_still_does(self):
        """The other direction, and the one that makes the test above mean
        something: adding the condition must not silence the rule."""
        minimum = registry.min_cacheable_tokens("anthropic/direct", "claude-opus-5")
        codes = self._codes(minimum * 60)
        self.assertNotIn("MIN-1", codes)
        self.assertIn("TTL-1", codes)

    def test_the_runtime_twin_refuses_it_too(self):
        """Fixing only the batch side made RT-TTL fire where TTL-1 refuses --
        a divergence created by the fix, which is this branch's most-repeated
        shape and the reason the parity class exists."""
        m, fired = monitor.Monitor(), []
        minimum = registry.min_cacheable_tokens("anthropic/direct", "claude-opus-5")
        for r in self._reqs(minimum // 2):
            fired += m.observe(r)
        codes = {a.code for a in fired}
        self.assertIn("RT-MIN", codes)
        self.assertNotIn("RT-TTL", codes)

    def test_and_the_runtime_twin_still_fires_above_it(self):
        m, fired = monitor.Monitor(), []
        minimum = registry.min_cacheable_tokens("anthropic/direct", "claude-opus-5")
        for r in self._reqs(minimum * 60):
            fired += m.observe(r)
        self.assertIn("RT-TTL", {a.code for a in fired})

    def test_the_helper_implements_the_condition_directly(self):
        span = (1, 200, ("sys",))
        later = [(1, 200, ("sys",)), (2, 3200, ("sys", "t"))]
        self.assertTrue(trace.span_is_reusable_by(span, later))
        self.assertFalse(trace.span_is_reusable_by(span, later, None, 512))
        self.assertTrue(trace.span_is_reusable_by(span, later, None, 100))


class TestAZeroSumCountIsNotAnExactCount(unittest.TestCase):
    """Both loaders took a "counted" claim at its word.

    `load_bodies` accepted any `segment_tokens` array of the right length whose
    entries are non-negative numbers -- an all-zero array satisfies every one of
    those and was classified as exact. `load_jsonl` read the row's own
    `tokens_counted: true` flag without checking it against the row's own
    segment sizes. Either way `tokens_counted` reported 1.0, which is the gate
    that releases structural money, on counts saying the prompt was empty.

    The dry-run guard stops *this* tool producing such a file. It does not stop
    one arriving, from a stale enrichment or a hostile one.
    """

    KEY = b"k" * 32
    BODY = {"model": "claude-opus-5",
            "system": [{"type": "text", "text": "x" * 4000}],
            "messages": [{"role": "user", "content": "hi"}]}

    def _write(self, rows):
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        with open(path, "w") as f:
            f.write("\n".join(json.dumps(r) for r in rows))
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        return path

    def _bodies(self, tok):
        from cacheeconomics.adapters.bodies import load_bodies
        rows = [{"request": self.BODY,
                 "response": {"usage": {"input_tokens": 50_000}},
                 "segment_tokens": [tok, tok]} for _ in range(20)]
        rows = [_vouched(r, target_id="anthropic/direct")
                for r in rows]
        return load_bodies(self._write(rows), self.KEY,
                           target_id="anthropic/direct").tokens_counted

    def _jsonl(self, tok):
        from cacheeconomics.trace import load_jsonl
        rows = [{"request_id": f"r{i}", "sent_at": f"2026-07-29T09:{i:02d}:00Z",
                 "model": "claude-opus-5", "target_id": "anthropic/direct",
                 "session": "s", "usage": {"input_tokens": 50_000},
                 "tokens_counted": True,
                 "segments": [{"id": "hmac:" + "a" * 64, "role": "system",
                               "tokens": tok, "index": 0,
                               "cache_marked": False, "ttl": None}]}
                for i in range(20)]
        return load_jsonl(self._write(rows), self.KEY).tokens_counted

    def test_all_zero_sizes_are_not_counted_in_either_loader(self):
        self.assertEqual(self._bodies(0), 0.0)
        self.assertEqual(self._jsonl(0), 0.0)

    def test_real_counts_still_are(self):
        """The other direction, so the fix cannot be 'trust nothing'."""
        self.assertEqual(self._bodies(500), 1.0)
        self.assertEqual(self._jsonl(500), 1.0)

    def test_the_two_loaders_agree(self):
        for tok in (0, 500):
            with self.subTest(tokens=tok):
                self.assertEqual(self._bodies(tok), self._jsonl(tok))
