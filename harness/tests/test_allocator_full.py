"""The full allocator: does it partition better than one marker, and does it
refuse where it cannot see?

The optimiser is the first thing in this project that searches rather than
applies a rule, which makes it the first thing that can be confidently wrong.
Two independent checks matter more than any single-case assertion: the dynamic
program's objective must equal an exact evaluator that shares none of its
reasoning, and a plan must never beat sending the prompt uncached unless it
genuinely does.
"""

import os
import sys
import unittest
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cacheeconomics import registry, tiers                                # noqa: E402
from cacheeconomics.allocate import (allocator_lite,                      # noqa: E402
                                     observed_change_rates,
                                     observed_change_rates_by_pool,
                                     observed_gaps_by_pool,
                                     observed_volatility_by_pool, pool_of)
from cacheeconomics.allocator import (allocator_full,                     # noqa: E402
                                      allocator_full_no_relocation)
from cacheeconomics.trace import Request, Segment                         # noqa: E402

T0 = datetime(2026, 7, 29, 9, tzinfo=timezone.utc)


def sg(i, role, tokens, sid, label=""):
    return Segment(id=sid, role=role, tokens=tokens, index=i, label=label)


def shape(i):
    """A coding agent's prompt: stable tools, slow context, fast turn."""
    return [sg(0, "tools", 6000, "tools", "tool_defs"),
            sg(1, "system", 3000, "policy", "instructions"),
            sg(2, "system", 2000, f"ctx{i // 10}", "project_ctx"),
            sg(3, "system", 1500, f"mem{i // 3}", "memory"),
            sg(4, "assistant", 800, f"hist{i // 2}", "history"),
            sg(5, "user", 400, f"turn{i}", "user_turn")]


def trace(n=60, gap=120, model="claude-opus-5", target="anthropic/direct"):
    return [Request(request_id=f"r{i}", sent_at=T0 + timedelta(seconds=gap * i),
                    model=model, usage={"input_tokens": 100},
                    segments=shape(i), target_id=target)
            for i in range(n)]


def inputs(reqs):
    p = pool_of(reqs[0])
    return (observed_change_rates_by_pool(reqs)[p], observed_gaps_by_pool(reqs)[p])


class TestTheSearchAgreesWithAnIndependentEvaluator(unittest.TestCase):
    """`_search` is a dynamic program that assumes blocks are independent.
    `expected_cost` conditions on each observed gap and assumes nothing. Where
    they overlap they must produce the same number, or one of them is wrong."""

    def _fixture(self):
        segs = [sg(i, "system", t, f"s{i}") for i, t in
                enumerate([6000, 3000, 2000, 1500, 800, 400, 300, 200])]
        rates = {0: 0.0, 1: 0.0, 2: 0.02, 3: 0.05, 4: 0.3, 5: 0.5, 6: 0.9, 7: 1.0}
        gaps = [30.0] * 40 + [900.0] * 40 + [7200.0] * 20
        cum, survive, running, total = [], [], 1.0, 0
        for s in segs:
            total += s.tokens
            cum.append(total)
            running *= 1 - rates[s.index]
            survive.append(running)
        return segs, rates, gaps, cum, survive, float(total)

    def test_the_dynamic_program_matches_the_exact_evaluator(self):
        segs, rates, gaps, cum, survive, total = self._fixture()
        _, write_rates, read_rate = tiers._surface("anthropic/direct")
        for ttl in ("5m", "1h"):
            live = tiers.survival(gaps, ttl)
            a = tiers._search(segs, cum, survive, 512, 4, ttl, live,
                              read_rate, write_rates, total, [])
            blocks = [(t.tokens, survive[t.marker_position]) for t in a.tiers]
            exact = tiers.expected_cost(
                blocks, [ttl] * len(blocks), read_rate, write_rates, gaps,
                [live] * len(blocks)) + (cum[-1] - a.tiers[-1].prefix_tokens)
            self.assertAlmostEqual(a.expected_cost, exact, places=6,
                                   msg=f"the two cost models disagree on {ttl}")

    def test_it_never_proposes_a_plan_worse_than_no_caching(self):
        segs, rates, gaps, *_ = self._fixture()
        a = tiers.allocate(segs, rates, target_id="anthropic/direct",
                           model="claude-opus-5", gaps=gaps)
        self.assertLessEqual(a.expected_cost, a.uncached_cost)

    def test_a_gap_of_zero_is_a_miss_not_a_certain_hit(self):
        """Two requests dispatched simultaneously each pay to write the same
        prefix and neither can read the other's. Counting the interval as
        `gap < ttl` rated a burst of concurrent requests as a perfect cache."""
        self.assertEqual(tiers.survival([0.0] * 20, "5m"), 0.0)

    def test_a_gap_inside_the_lifetime_still_counts(self):
        self.assertEqual(tiers.survival([60.0] * 20, "5m"), 1.0)

    def test_a_gap_past_the_lifetime_does_not(self):
        self.assertEqual(tiers.survival([7200.0] * 20, "1h"), 0.0)

    def test_concurrent_traffic_gets_no_markers(self):
        segs = [sg(0, "system", 9000, "a"), sg(1, "user", 400, "b")]
        a = tiers.allocate(segs, {0: 0.0, 1: 1.0}, target_id="anthropic/direct",
                           model="claude-opus-5", gaps=[0.0] * 30)
        self.assertEqual(a.tiers, [])

    def test_a_worse_lifetime_is_scored_and_rejected_not_ignored(self):
        segs, rates, gaps, *_ = self._fixture()
        a = tiers.allocate(segs, rates, target_id="anthropic/direct",
                           model="claude-opus-5", gaps=gaps)
        labels = [l for l, _ in a.searched]
        self.assertIn("uniform 5m", labels)
        self.assertIn("uniform 1h", labels)


class TestItSpendsTheBudget(unittest.TestCase):

    def test_it_places_more_markers_than_allocator_lite(self):
        reqs = trace()
        rates, gaps = inputs(reqs)
        vol = observed_volatility_by_pool(reqs)[pool_of(reqs[0])]
        lite = allocator_lite(reqs[-1], volatility=vol, cadence_seconds=120)
        full = allocator_full(reqs[-1], rates=rates, gaps=gaps)
        self.assertEqual(len(lite.marker_indices), 1)
        self.assertGreater(len(full.marker_indices), 1)

    def test_it_stays_inside_the_surface_budget(self):
        reqs = trace()
        rates, gaps = inputs(reqs)
        budget = registry.capability("anthropic/direct", "max_breakpoints")
        self.assertLessEqual(len(allocator_full(reqs[-1], rates=rates,
                                                gaps=gaps).marker_indices), budget)

    def test_it_leaves_the_segment_that_changes_every_request_uncached(self):
        reqs = trace()
        rates, gaps = inputs(reqs)
        self.assertNotIn(5, allocator_full(reqs[-1], rates=rates, gaps=gaps).marker_indices)

    def test_a_tier_is_only_bought_when_it_beats_plain_input(self):
        """A 5m marker pays for itself above about a 22% hit rate: below that,
        the write premium on the misses costs more than sending the tokens."""
        reqs = trace()
        rates, gaps = inputs(reqs)
        full = allocator_full(reqs[-1], rates=rates, gaps=gaps)
        _, write_rates, read_rate = tiers._surface("anthropic/direct")
        for t in full.allocation.tiers:
            q = t.hit_probability
            self.assertLess(q * read_rate + (1 - q) * write_rates[t.ttl], 1.0,
                            f"tier at {t.marker_position} costs more than not caching it")


class TestRatesAreNotCounts(unittest.TestCase):
    """The same blind spot the runtime monitor had: a field alternating between
    two states changes the prefix every request and shows two distinct values."""

    def test_an_alternating_segment_reads_as_fully_volatile(self):
        reqs = [Request(request_id=f"r{i}", sent_at=T0 + timedelta(seconds=60 * i),
                        model="claude-opus-5", usage={},
                        segments=[sg(0, "system", 5000, "A" if i % 2 else "B"),
                                  sg(1, "system", 5000, "stable")], target_id="anthropic/direct")
                for i in range(20)]
        rates = observed_change_rates(reqs)
        self.assertEqual(rates[0], 1.0)
        self.assertEqual(rates[1], 0.0)
        counts = observed_volatility_by_pool(reqs)[pool_of(reqs[0])]
        self.assertEqual(counts[0], 2, "the count cannot tell these apart; the rate can")

    def test_a_segment_that_vanishes_counts_as_a_change(self):
        reqs = []
        for i in range(20):
            segs = [sg(0, "system", 5000, "stable")]
            if i % 2:
                segs.append(sg(1, "system", 100, "sometimes"))
            reqs.append(Request(request_id=f"r{i}", sent_at=T0 + timedelta(seconds=60 * i),
                                model="claude-opus-5", usage={}, segments=segs, target_id="anthropic/direct"))
        self.assertGreater(observed_change_rates(reqs)[1], 0.5)

    def test_the_fail_closed_reduction_takes_the_worst_pool(self):
        reqs = []
        for i in range(20):
            for tenant, sid in (("calm", "fixed"), ("busy", f"v{i}")):
                reqs.append(Request(request_id=f"{tenant}{i}",
                                    sent_at=T0 + timedelta(seconds=60 * i),
                                    model="claude-opus-5", usage={}, tenant=tenant,
                                    segments=[sg(0, "system", 5000, sid)], target_id="anthropic/direct"))
        self.assertEqual(observed_change_rates(reqs)[0], 1.0)

    def test_per_session_headers_are_not_pool_level_volatility(self):
        reqs = []
        for i in range(20):
            for sess in ("a", "b"):
                reqs.append(Request(
                    request_id=f"{sess}{i}", sent_at=T0 + timedelta(seconds=60 * i),
                    model="claude-opus-5", usage={}, tenant="tenant", session=sess,
                    segments=[sg(0, "system", 5000, f"session-{sess}"),
                              sg(1, "system", 5000, "body")], target_id="anthropic/direct"))
        self.assertEqual(observed_change_rates(reqs)[0], 0.0)

    def test_a_changing_reuse_chain_still_fails_closed(self):
        reqs = []
        for i in range(20):
            reqs.append(Request(request_id=f"stable{i}",
                                sent_at=T0 + timedelta(seconds=60 * i),
                                model="claude-opus-5", usage={}, tenant="tenant",
                                session="stable",
                                segments=[sg(0, "system", 5000, "fixed")], target_id="anthropic/direct"))
            reqs.append(Request(request_id=f"changing{i}",
                                sent_at=T0 + timedelta(seconds=60 * i),
                                model="claude-opus-5", usage={}, tenant="tenant",
                                session="changing",
                                segments=[sg(0, "system", 5000, f"v{i}")], target_id="anthropic/direct"))
        self.assertEqual(observed_change_rates(reqs)[0], 1.0)


class TestItRefusesRatherThanGuesses(unittest.TestCase):

    def _one(self, **kw):
        segs = [sg(0, "system", 6000, "a"), sg(1, "system", 3000, "b"),
                sg(2, "user", 400, "c")]
        kw.setdefault("target_id", "anthropic/direct")
        r = Request(request_id="r", sent_at=T0, model="claude-opus-5",
                    usage={}, segments=segs, **kw)
        return r

    def test_an_implicit_prefix_surface_says_there_is_nothing_to_place(self):
        p = allocator_full(self._one(target_id="deepseek/direct"),
                           rates={0: 0.0}, gaps=[60.0] * 20)
        self.assertEqual(p.marker_indices, [])
        self.assertTrue(any("takes no markers" in n for n in p.notes))

    def test_it_names_applicability_before_pricing(self):
        """`cost.ttl_crossover` carries this note because the same mistake was
        made there first: an implicit surface refused with 'no multipliers'."""
        p = allocator_full(self._one(target_id="deepseek/direct"),
                           rates={0: 0.0}, gaps=[60.0] * 20)
        self.assertFalse(any("multipliers" in n for n in p.notes))

    def test_a_surface_with_no_marker_budget_refuses(self):
        p = allocator_full(self._one(target_id="openai/direct"),
                           rates={0: 0.0}, gaps=[60.0] * 20)
        self.assertEqual(p.marker_indices, [])
        self.assertTrue(any("max_breakpoints" in n for n in p.notes))

    def test_a_different_control_model_is_disclosed_not_hidden(self):
        p = allocator_full(self._one(target_id="amazon-bedrock/converse"),
                           rates={0: 0.0, 1: 0.05, 2: 1.0}, gaps=[120.0] * 20)
        self.assertTrue(any("checkpoint_backward_search" in n for n in p.notes))

    def test_no_observed_gaps_means_no_markers(self):
        p = allocator_full(self._one(), rates={0: 0.0, 1: 0.0, 2: 1.0}, gaps=[])
        self.assertEqual(p.marker_indices, [])
        self.assertTrue(any("no inter-request gaps" in n for n in p.notes))

    def test_no_observed_rates_means_no_markers(self):
        p = allocator_full(self._one(), rates={}, gaps=[120.0] * 20)
        self.assertEqual(p.marker_indices, [])
        self.assertTrue(any("changing on every request" in n for n in p.notes))

    def test_a_prompt_below_the_minimum_says_so(self):
        r = Request(request_id="r", sent_at=T0, model="claude-haiku-4-5", usage={},
                    segments=[sg(0, "system", 200, "a"), sg(1, "user", 50, "b")], target_id="anthropic/direct")
        p = allocator_full(r, rates={0: 0.0, 1: 1.0}, gaps=[120.0] * 20)
        self.assertEqual(p.marker_indices, [])
        self.assertTrue(any("4,096-token minimum" in n for n in p.notes))

    def test_an_infeasible_candidate_is_not_reported_as_costing_nothing(self):
        r = Request(request_id="r", sent_at=T0, model="claude-haiku-4-5", usage={},
                    segments=[sg(0, "system", 200, "a"), sg(1, "user", 50, "b")], target_id="anthropic/direct")
        scored = next(n for n in allocator_full(r, rates={0: 0.0}, gaps=[120.0] * 20).notes
                      if n.startswith("candidates scored"))
        self.assertIn("not feasible", scored)
        self.assertNotIn("5m 0", scored)


def buried(i):
    """A timestamp at position 1, above 12,000 stable tokens. The shape
    relocation exists for: nothing can be cached until it moves."""
    return [sg(0, "system", 800, "hdr", "preamble"),
            sg(1, "system", 50, f"stamp{i}", "timestamp"),
            sg(2, "tools", 8000, "tools", "tool_defs"),
            sg(3, "system", 4000, "policy", "instructions"),
            sg(4, "user", 400, f"turn{i}", "user_turn")]


class TestCountsAndRatesAreNotInterchangeable(unittest.TestCase):
    """They invert each other. A rate of 1.0 means the segment changes on every
    request; a count of 1 means it never changes. Passing one where the other
    is wanted proposes no moves on the worst prompt in the trace."""

    def test_propose_refuses_a_rate_map(self):
        from cacheeconomics.relocate import propose
        reqs = [Request(request_id=f"r{i}", sent_at=T0 + timedelta(seconds=60 * i),
                        model="claude-opus-5", usage={}, segments=buried(i), target_id="anthropic/direct")
                for i in range(20)]
        with self.assertRaises(TypeError) as e:
            propose(reqs, observed_change_rates(reqs))
        self.assertIn("change rate", str(e.exception))

    def test_it_catches_the_all_or_nothing_rate_map_that_passes_for_counts(self):
        from cacheeconomics.relocate import propose
        reqs = [Request(request_id="r", sent_at=T0, model="claude-opus-5",
                        usage={}, segments=buried(0), target_id="anthropic/direct")]
        with self.assertRaises(TypeError):
            propose(reqs, {0: 0.0, 1: 1.0, 2: 0.0})

    def test_counts_are_still_accepted(self):
        from cacheeconomics.relocate import propose
        reqs = [Request(request_id=f"r{i}", sent_at=T0 + timedelta(seconds=60 * i),
                        model="claude-opus-5", usage={}, segments=buried(i), target_id="anthropic/direct")
                for i in range(20)]
        moves, _ = propose(reqs, observed_volatility_by_pool(reqs)[pool_of(reqs[0])])
        self.assertTrue(moves, "the buried timestamp is exactly what propose is for")


class TestRelocationComposes(unittest.TestCase):

    def _buried(self, n=40):
        return [Request(request_id=f"r{i}", sent_at=T0 + timedelta(seconds=120 * i),
                        model="claude-opus-5", usage={"input_tokens": 100},
                        segments=buried(i), target_id="anthropic/direct") for i in range(n)]

    def test_moving_the_buried_segment_unlocks_the_prefix_behind_it(self):
        from cacheeconomics.relocate import propose
        reqs = self._buried()
        rates, gaps = inputs(reqs)
        moves, order = propose(reqs, observed_volatility_by_pool(reqs)[pool_of(reqs[0])])
        without = allocator_full_no_relocation(reqs[-1], rates=rates, gaps=gaps,
                                               moves=moves, order=order)
        with_ = allocator_full(reqs[-1], rates=rates, gaps=gaps,
                               moves=moves, order=order)
        self.assertLess(with_.allocation.expected_cost,
                        without.allocation.expected_cost,
                        "relocation has to change the answer or it is decorative")
        self.assertTrue(any(n.startswith("moved segment 1") for n in with_.notes))

    def test_the_no_relocation_variant_keeps_the_authored_order(self):
        from cacheeconomics.relocate import propose
        reqs = self._buried()
        rates, gaps = inputs(reqs)
        moves, order = propose(reqs, observed_volatility_by_pool(reqs)[pool_of(reqs[0])])
        p = allocator_full_no_relocation(reqs[-1], rates=rates, gaps=gaps,
                                         moves=moves, order=order)
        self.assertIsNone(p.order)
        self.assertFalse(any(n.startswith("moved segment") for n in p.notes))

    def test_a_plan_that_moves_content_demands_an_eval(self):
        from cacheeconomics.relocate import Move
        reqs = trace()
        rates, gaps = inputs(reqs)
        move = Move(segment_index=3, label="memory", tokens_moved=1500,
                    tokens_unlocked=800, risk="low", scope="within-container",
                    reason="changes between requests", eval_required=True,
                    new_order=(0, 1, 2, 4, 5, 3))
        p = allocator_full(reqs[-1], rates=rates, gaps=gaps, moves=[move])
        self.assertTrue(any("EVAL REQUIRED" in n for n in p.notes))


class TestItIsNotABakeOffArm(unittest.TestCase):
    """Gate 1 decides whether the full allocator should exist. It cannot be one
    of the things Gate 1 measures."""

    def test_the_bake_off_policies_do_not_include_it(self):
        from cacheeconomics.simulate import POLICIES
        self.assertNotIn("allocator-full", POLICIES)
        self.assertEqual(set(POLICIES), {"as-shipped", "litellm-auto",
                                         "allocator-lite", "relocation-lite"})

    def test_it_is_reachable_by_name(self):
        from cacheeconomics.allocator import GATED_POLICIES
        self.assertIn("allocator-full", GATED_POLICIES)


class TestTheAllocatorEmitsNoMoney(unittest.TestCase):
    """Its objective is a ratio between plans, so the price cancels. An
    optimiser that emitted dollars would route around the reconciliation gate."""

    def test_no_note_carries_a_dollar_figure(self):
        reqs = trace()
        rates, gaps = inputs(reqs)
        for n in allocator_full(reqs[-1], rates=rates, gaps=gaps).notes:
            self.assertNotIn("$", n)


if __name__ == "__main__":
    unittest.main()
