"""TTL-2: the one-hour premium, and whether it was earned.

TTL-1 answers "your cadence sits in the band and you are not using the long
lifetime". Nothing answered the mirror, so a workload whose agent defaults to
1h -- which Claude Code does -- got no finding at all. Silence reads as
approval.

The reason this rule is worth more than a subtraction is that the obvious
version of it is wrong. Counting 1h write tokens and multiplying by the
2.0-to-1.25 premium says "switch and save" on a workload where switching loses
money, because the rare gaps that land in the five-minute-to-one-hour band each
sit on a prefix that has been accumulating all session. On the trace this rule
was built against, 1.4% of gaps carried more cost than the other 98.6%.

Every test here is about that netting. A rule that got the sign right by
accident on one fixture would be worse than no rule.
"""

import os
import sys
import unittest
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cacheeconomics import registry
from cacheeconomics.analyzer import analyze                       # noqa: E402
from cacheeconomics.trace import Request, Segment, Tier, TraceSet          # noqa: E402

T0 = datetime(2026, 7, 29, 9, tzinfo=timezone.utc)


def req(offset_s, *, write_1h=0, write_5m=0, read=0, model="claude-opus-5",
        session="s", target="anthropic/direct"):
    created = write_1h + write_5m
    return Request(
        request_id=f"r{offset_s}", sent_at=T0 + timedelta(seconds=offset_s),
        model=model, agent="a", session=session, target_id=target,
        usage={"input_tokens": 50, "cache_read_input_tokens": read,
               "cache_creation_input_tokens": created,
               "cache_creation": {"ephemeral_5m_input_tokens": write_5m,
                                  "ephemeral_1h_input_tokens": write_1h}})


def findings(reqs, **kw):
    a = analyze(TraceSet(requests=reqs, tier=Tier.USAGE_ONLY, source="t"), **kw)
    return {f.code: f for f in a.findings}


class TestItAnswersTheQuestionAtAll(unittest.TestCase):

    def test_a_fast_workload_on_1h_is_told_to_drop_to_5m(self):
        """Every gap under five minutes, so the long lifetime buys nothing and
        the premium is pure waste."""
        reqs = [req(i * 60, write_1h=40_000, read=200_000) for i in range(40)]
        f = findings(reqs, allow_unreconciled=True)["TTL-2"]
        self.assertIn("premium where five minutes would do", f.title)
        self.assertGreater(f.avoidable_usd_month.raw(), 0)

    def test_no_1h_writes_means_no_finding(self):
        """This rule is about a premium that was paid. Nothing paid it here."""
        reqs = [req(i * 60, write_5m=40_000, read=200_000) for i in range(40)]
        self.assertNotIn("TTL-2", findings(reqs, allow_unreconciled=True))

    def test_it_needs_enough_gaps_before_it_speaks(self):
        reqs = [req(i * 60, write_1h=40_000, read=200_000) for i in range(5)]
        self.assertNotIn("TTL-2", findings(reqs, allow_unreconciled=True))


class TestTheNettingIsWhatDecidesIt(unittest.TestCase):
    """The whole rule. A version that skipped this would recommend a change
    that costs money on the most common real workload it will meet."""

    def _mostly_fast_with_rare_expensive_gaps(self, prefix):
        """98% of gaps under five minutes. Two land in the band, each sitting
        on `prefix` tokens that a five-minute lifetime would force to be
        rewritten. Same shape as a long Claude Code session."""
        reqs, t = [], 0
        for i in range(100):
            # every 50th turn, a gap that only the 1h entry survives
            t += 900 if i and i % 50 == 0 else 30
            reqs.append(req(t, write_1h=2_000, read=prefix))
        return reqs

    def test_a_small_prefix_makes_the_switch_worth_it(self):
        f = findings(self._mostly_fast_with_rare_expensive_gaps(5_000),
                     allow_unreconciled=True)["TTL-2"]
        self.assertIn("premium where five minutes would do", f.title)

    def test_a_large_prefix_flips_the_answer(self):
        """Same cadence, same 1h writes, same number of band gaps. Only the
        size of what would be rewritten changed, and the recommendation
        reverses. A rule that counted gaps by frequency could not do this."""
        f = findings(self._mostly_fast_with_rare_expensive_gaps(900_000),
                     allow_unreconciled=True)["TTL-2"]
        self.assertIn("earning its premium", f.title)
        self.assertIsNone(f.avoidable_usd_month,
                          "there is nothing to recover, so nothing to publish")

    def test_the_flip_is_not_an_artifact_of_the_gap_count(self):
        """Pin the band gaps and sweep only the prefix. The verdict has to
        cross exactly once, and in the direction that says bigger prefixes
        favour the longer lifetime."""
        verdicts = []
        for prefix in (1_000, 10_000, 100_000, 1_000_000):
            f = findings(self._mostly_fast_with_rare_expensive_gaps(prefix),
                         allow_unreconciled=True)["TTL-2"]
            verdicts.append("switch" if "would do" in f.title else "keep")
        self.assertEqual(verdicts.count("switch") + verdicts.count("keep"), 4)
        self.assertLess(verdicts.index("keep"), 4, "never flips at any size")
        self.assertNotIn("switch", verdicts[verdicts.index("keep"):],
                         "flips back and forth, so it is not monotone in prefix")

    def test_the_naive_answer_would_have_been_wrong(self):
        """Guarding the actual defect. Multiplying 1h writes by the premium and
        stopping there is what this rule must not do."""
        reqs = self._mostly_fast_with_rare_expensive_gaps(900_000)
        f = findings(reqs, allow_unreconciled=True)["TTL-2"]
        naive = sum(r.usage["cache_creation"]["ephemeral_1h_input_tokens"]
                    for r in reqs) * (2.00 - 1.25) * 5 / 1e6
        self.assertGreater(naive, 0, "fixture writes nothing, so this is vacuous")
        self.assertIn("earning its premium", f.title,
                      "took the naive answer and recommended a losing change")


class TestItStaysInsideTheRulesTheRestOfThisObeys(unittest.TestCase):

    def test_multipliers_come_from_the_registry_not_the_arithmetic(self):
        """A surface whose lifetimes are not first-party priced has no premium
        this rule may reason about. Writing 1.25 and 2.0 into the code is how a
        Bedrock trace once produced a total no AWS bill would match."""
        reqs = [req(i * 60, write_1h=40_000, read=200_000,
                    target="amazon-bedrock/converse") for i in range(40)]
        self.assertNotIn("TTL-2", findings(reqs, allow_unreconciled=True))

    def test_the_figure_is_withheld_without_an_invoice(self):
        reqs = [req(i * 60, write_1h=40_000, read=200_000) for i in range(40)]
        f = findings(reqs)["TTL-2"]
        self.assertFalse(f.avoidable_usd_month.released)
        self.assertIn("withheld", str(f.avoidable_usd_month))

    def test_it_is_modeled_and_says_so(self):
        """The five-minute arm was never run. Calling this measured would put a
        projection in the same column as an observation."""
        reqs = [req(i * 60, write_1h=40_000, read=200_000) for i in range(40)]
        self.assertEqual(findings(reqs)["TTL-2"].evidence_class, "modeled")

    def test_cadence_is_measured_per_isolation_scope(self):
        """Twenty sessions interleaved in wall-clock time. Each one's real
        cadence is fifteen minutes, so every gap is in the band and the long
        lifetime is exactly what it needs. Pooled, the gaps read as 45 seconds
        and this rule would recommend shortening the lifetime on a workload
        that depends on it.

        The mirror of the failure TTL-1 was bitten by, and worse: pooling
        always compresses gaps, so it pushes TTL-2 toward speaking rather than
        toward silence."""
        reqs = []
        for i in range(20):
            for k in range(20):
                reqs.append(req(i * 900 + k * 45, write_1h=2_000,
                                read=900_000, session=f"s{k}"))
        got = findings(reqs, allow_unreconciled=True)
        self.assertNotIn("TTL-2", got,
                         "gaps were pooled across sessions, so a band-cadence "
                         "workload was told to shorten its lifetime")

    def test_a_model_switch_does_not_pool_either(self):
        """Caches are per model. Two models alternating in one session are two
        pools, and each one's own gaps decide its lifetime."""
        reqs = []
        for i in range(40):
            reqs.append(req(i * 900, write_1h=2_000, read=900_000,
                            model="claude-opus-5"))
            reqs.append(req(i * 900 + 5, write_1h=2_000, read=900_000,
                            model="claude-sonnet-5"))
        self.assertNotIn("TTL-2", findings(reqs, allow_unreconciled=True))


class TestItDoesNotContradictTTL1(unittest.TestCase):
    """Both rules can look at the same trace. They must not tell the reader to
    move the lifetime in both directions at once."""

    def test_the_two_never_both_recommend_a_change(self):
        for prefix in (1_000, 50_000, 900_000):
            for gap in (30, 600, 4000):
                reqs = [req(i * gap, write_1h=20_000, write_5m=20_000,
                            read=prefix) for i in range(40)]
                got = findings(reqs, allow_unreconciled=True)
                t1, t2 = got.get("TTL-1"), got.get("TTL-2")
                if t1 and t2 and "would do" in t2.title:
                    self.fail(f"prefix={prefix} gap={gap}: TTL-1 says lengthen "
                              f"and TTL-2 says shorten")


if __name__ == "__main__":
    unittest.main()


class TestInBandGapsThatReadNothing(unittest.TestCase):
    """An in-band gap whose request read nothing did not have an entry to use.

    `band_gaps` did double duty: the count of gaps in the band, and the count of
    gaps whose rebuild cost could be priced. It only incremented inside
    `if prefix`, so an in-band gap with no read vanished from the rarity
    denominator entirely.

    Twenty fast gaps beside twenty in-band ones then reported "100.0% of gaps
    are under five minutes" -- a false statement of fact in the detail text --
    and recommended shortening the lifetime. That is the worst case to get
    backwards: no read at an in-band gap means the prefix is not surviving, and
    a shorter lifetime makes that harder to see rather than better.
    """

    def _mix(self, fast, band_read, band_unread, prefix=200_000):
        reqs, t = [], 0
        for _ in range(fast):
            t += 30
            reqs.append(req(t, write_1h=5_000, read=prefix))
        for _ in range(band_read):
            t += 900
            reqs.append(req(t, write_1h=5_000, read=prefix))
        for _ in range(band_unread):
            t += 900
            reqs.append(req(t, write_1h=40_000, read=0))
        return reqs

    def test_unread_in_band_gaps_count_toward_the_rarity_premise(self):
        """Half the gaps are in the band. The rule's premise does not hold and
        it must stay quiet, whatever the arithmetic says."""
        got = findings(self._mix(20, 0, 20), allow_unreconciled=True)
        self.assertNotIn("TTL-2", got,
                         "in-band gaps with no read were dropped from the "
                         "denominator, so the band looked empty")

    def test_the_percentage_it_prints_is_true(self):
        """The detail said 100% of gaps were under five minutes when half were
        not. A reader acts on that sentence."""
        got = findings(self._mix(96, 4, 0), allow_unreconciled=True)
        f = got.get("TTL-2")
        self.assertIsNotNone(f)
        import re
        m = re.search(r"([\d.]+)% of gaps between requests", f.detail)
        self.assertIsNotNone(m)
        self.assertAlmostEqual(float(m.group(1)), 96.0, delta=1.0,
                               msg="the share it prints is not the real share")

    def test_no_saving_is_published_when_band_gaps_read_nothing(self):
        """Their rebuild cost cannot be priced, so `net` counts them as free and
        is biased toward recommending the switch.

        Small prefix on purpose: that is the shape where the bias actually
        changes the answer. With a large prefix the net is already negative and
        "keep the hour" is correct regardless, so the bias is harmless there and
        this branch is not the one that should fire.
        """
        f = findings(self._mix(96, 2, 2, prefix=3_000),
                     allow_unreconciled=True)["TTL-2"]
        self.assertIn("not being read back", f.title)
        self.assertIsNone(f.avoidable_usd_month)
        self.assertIn("EFF-1", f.fix, "does not point at the real cause")

    def test_the_priceable_case_still_prices(self):
        """The guard must not swallow the finding it was built for."""
        f = findings(self._mix(96, 4, 0, prefix=3_000),
                     allow_unreconciled=True)["TTL-2"]
        self.assertIn("premium where five minutes would do", f.title)
        self.assertGreater(f.avoidable_usd_month.raw(), 0)

    def test_a_large_prefix_still_flips_to_keeping_the_hour(self):
        f = findings(self._mix(96, 4, 0, prefix=900_000),
                     allow_unreconciled=True)["TTL-2"]
        self.assertIn("earning its premium", f.title)


class TestThresholdsAreEconomicNotFrequency(unittest.TestCase):
    """Two rules vetoed on how often something happened before pricing it.

    EFF-1 returned early unless prefix efficiency was under 50%. TTL-1 returned
    early unless more than 40% of gaps fell in the band. Both sat in front of an
    economic test that was already correct, and both could only ever drop real
    findings, never add false ones.

    Neither constant was derived. Caching breaks even at `(W-1)/(W-R)`, which is
    21.7% efficiency for a 5m write and 52.6% for a 1h write, so 0.5 was above
    one and below the other. The band it got wrong was a 1h workload between 50%
    and 52.6%: losing money on every request and silently dropped.
    """

    def _usage(self, w, r, ttl):
        return {"input_tokens": 10, "cache_read_input_tokens": r,
                "cache_creation_input_tokens": w,
                "cache_creation": {"ephemeral_5m_input_tokens": w if ttl == "5m" else 0,
                                   "ephemeral_1h_input_tokens": w if ttl == "1h" else 0}}

    def _trace(self, w, r, ttl, n=20):
        return TraceSet(requests=[
            Request(request_id=f"r{i}", sent_at=T0 + timedelta(seconds=60 * i),
                    model="claude-opus-5", agent="a", session="s",
                    ttl_requested=ttl, usage=self._usage(w, r, ttl))
            for i in range(n)], tier=Tier.USAGE_ONLY, source="t")

    def _codes(self, ts):
        return {f.code: f for f in analyze(ts, allow_unreconciled=True).findings}

    def test_the_break_even_is_not_one_half(self):
        """Derived, so a multiplier change moves it and nothing has to remember."""
        m = registry.multipliers("anthropic/direct")
        for key, expect in (("write_5m", 0.217), ("write_1h", 0.526)):
            W, R = m[key], m["read"]
            self.assertAlmostEqual((W - 1) / (W - R), expect, places=2)

    def test_a_1h_workload_just_above_the_old_cutoff_still_loses_money(self):
        """Efficiency 51.2%: over the old 0.5 veto, under the 1h break-even."""
        f = self._codes(self._trace(100_000, 105_000, "1h")).get("EFF-1")
        self.assertIsNotNone(f, "the frequency veto is still suppressing it")
        self.assertGreater(f.avoidable_usd_month.raw(), 0)

    def test_a_5m_workload_below_the_old_cutoff_is_still_left_alone(self):
        """Efficiency 30%: under the old veto, but over the 5m break-even of
        21.7%, so caching is winning and EFF-1 must stay quiet. Removing a veto
        must not turn it into a false positive."""
        self.assertNotIn("EFF-1", self._codes(self._trace(70_000, 30_000, "5m")))

    def test_ttl1_prices_a_band_share_below_the_old_forty_percent(self):
        """A stable million-token prefix rewritten across 35 ten-minute gaps,
        with 64 one-minute reads between them, is 34% in-band. The rewrites are
        worth more than most findings here and were dropped unpriced."""
        seg = [Segment(id="p", role="system", tokens=1_000_000, index=0,
                       cache_marked=True, ttl="5m")]
        reqs, t, i = [], 0, 0
        for _ in range(35):
            t += 600
            reqs.append(Request(request_id=f"r{i}", sent_at=T0 + timedelta(seconds=t),
                                model="claude-opus-5", agent="a", session="s",
                                ttl_requested="5m", segments=seg,
                                usage=self._usage(1_000_000, 0, "5m")))
            i += 1
            for _ in range(2):
                t += 60
                reqs.append(Request(request_id=f"r{i}", sent_at=T0 + timedelta(seconds=t),
                                    model="claude-opus-5", agent="a", session="s",
                                    ttl_requested="5m", segments=seg,
                                    usage=self._usage(0, 1_000_000, "5m")))
                i += 1
        f = self._codes(TraceSet(requests=reqs, tier=Tier.INSTRUMENTED,
                                 source="t")).get("TTL-1")
        self.assertIsNotNone(f, "34% in-band was vetoed before pricing")
        self.assertGreater(f.avoidable_usd_month.raw(), 0)

    def test_ttl1_will_not_claim_an_in_band_cadence_with_no_in_band_gaps(self):
        """The premise, which the 40% veto had been enforcing by accident. Every
        gap 30s and no segment identity: TTL-1 fired anyway and announced a
        cadence inside the one-hour window at 0% in-band, contradicting TTL-2 on
        the same trace."""
        ts = TraceSet(requests=[
            Request(request_id=f"r{i}", sent_at=T0 + timedelta(seconds=30 * i),
                    model="claude-opus-5", agent="a", session="s",
                    usage={"input_tokens": 50, "cache_read_input_tokens": 1000,
                           "cache_creation_input_tokens": 40000,
                           "cache_creation": {"ephemeral_5m_input_tokens": 20000,
                                              "ephemeral_1h_input_tokens": 20000}})
            for i in range(40)], tier=Tier.USAGE_ONLY, source="t")
        got = self._codes(ts)
        self.assertNotIn("TTL-1", got)
        self.assertIn("TTL-2", got, "TTL-2 owns this trace and should still speak")
