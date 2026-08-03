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


class TestTheOtherRulesStoppedGuessing(unittest.TestCase):
    """Four rules that each excluded, hard-coded or double-counted something."""

    def _seg(self, **kw):
        return Segment(**kw)

    def _usage(self, w, r=0):
        return {"input_tokens": 10, "cache_read_input_tokens": r,
                "cache_creation_input_tokens": w,
                "cache_creation": {"ephemeral_5m_input_tokens": w,
                                   "ephemeral_1h_input_tokens": 0}}

    def _codes(self, reqs, tier=Tier.INSTRUMENTED):
        return {f.code: f for f in analyze(
            TraceSet(requests=reqs, tier=tier, source="t"),
            allow_unreconciled=True).findings}

    def test_an_intermittent_prefix_tail_is_reported_by_the_rules_that_can_help(self):
        """A review flagged that VOL-1 ignores absence. It does not, wherever
        its fix applies: an optional block in front of other content renumbers
        everything behind it, so the position holds two ids and the existing
        buckets catch it.

        What slips through is an optional block at the *end* of the prefix. That
        forces a rewrite, but VOL-1's remedy is to move the volatile block
        behind the stable span and there is no stable span behind it. Reporting
        it there would attach a fix that recovers nothing. These three cover it
        instead, and their fixes are the ones that apply.
        """
        def mk(i, present):
            segs = [Segment(id="tools", role="tools", tokens=30_000, index=0)]
            if present:
                segs.append(Segment(id="opt", role="system", tokens=200, index=1,
                                    cache_marked=True, ttl="5m"))
            else:
                segs[0] = Segment(id="tools", role="tools", tokens=30_000,
                                  index=0, cache_marked=True, ttl="5m")
            return Request(request_id=f"r{i}", sent_at=T0 + timedelta(seconds=60 * i),
                           model="claude-opus-5", agent="a", session="s",
                           ttl_requested="5m", segments=segs,
                           usage=self._usage(30_000))
        got = self._codes([mk(i, i % 2 == 0) for i in range(30)])
        for code in ("EFF-1", "REB-1", "CAC-1"):
            self.assertIn(code, got)

    def test_min1_sees_an_inner_marker_below_the_minimum(self):
        """`cached_prefix_tokens` is the prefix at the *last* marker and the
        counters are request-wide, so a 200-token marker in front of a 30k one
        was invisible: the outer marker wrote, the inner did nothing, and
        nothing anywhere said so."""
        segs = [Segment(id="tiny", role="system", tokens=200, index=0,
                        cache_marked=True, ttl="5m"),
                Segment(id="big", role="tools", tokens=30_000, index=1,
                        cache_marked=True, ttl="5m")]
        reqs = [Request(request_id=f"r{i}", sent_at=T0 + timedelta(seconds=60 * i),
                        model="claude-opus-5", agent="a", session="s",
                        ttl_requested="5m", segments=segs,
                        usage=self._usage(30_200, r=0)) for i in range(20)]
        self.assertIn("MIN-1", self._codes(reqs))

    def test_reb0_reports_sessionless_writers_in_a_mixed_export(self):
        """`if not sessions` meant one sessioned request anywhere silenced this
        for every sessionless one -- and those are exactly the rows REB-1 cannot
        reason about."""
        u = self._usage(50_000)
        reqs = [Request(request_id=f"s{i}", sent_at=T0 + timedelta(seconds=60 * i),
                        model="claude-opus-5", session="s",
                        usage=self._usage(100)) for i in range(2)]
        reqs += [Request(request_id=f"n{i}", sent_at=T0 + timedelta(seconds=60 * i),
                         model="claude-opus-5", usage=u) for i in range(50)]
        f = self._codes(reqs, tier=Tier.USAGE_ONLY).get("REB-0")
        self.assertIsNotNone(f, "the sessionless writers vanished")
        self.assertIn("50 of 52", f.detail)
        self.assertEqual(f.affected_requests, 50)

    def test_fan1_uses_observed_first_token_time_over_a_flat_window(self):
        """A sibling sent at t=8s cannot have read an entry whose first token
        arrives at t=20s. The flat five-second window skipped it."""
        seg = [Segment(id="p", role="system", tokens=30_000, index=0,
                       cache_marked=True, ttl="5m")]
        reqs = []
        for k in range(12):
            base = k * 600
            for tag, off in (("a", 0), ("b", 8)):
                reqs.append(Request(
                    request_id=f"{tag}{k}", sent_at=T0 + timedelta(seconds=base + off),
                    first_token_at=T0 + timedelta(seconds=base + off + 20),
                    model="claude-opus-5", agent="a", session=f"{tag}{k}",
                    ttl_requested="5m", segments=seg, usage=self._usage(30_000)))
        f = self._codes(reqs).get("FAN-1")
        self.assertIsNotNone(f, "the flat window skipped an 8-second sibling")
        self.assertIn("observed first-token time", f.detail)

    def test_fan1_says_so_when_it_had_to_fall_back(self):
        """A stand-in for provider latency must not read as an observation."""
        seg = [Segment(id="p", role="system", tokens=30_000, index=0,
                       cache_marked=True, ttl="5m")]
        reqs = []
        for k in range(12):
            base = k * 600
            for tag, off in (("a", 0), ("b", 2)):
                reqs.append(Request(
                    request_id=f"{tag}{k}", sent_at=T0 + timedelta(seconds=base + off),
                    model="claude-opus-5", agent="a", session=f"{tag}{k}",
                    ttl_requested="5m", segments=seg, usage=self._usage(30_000)))
        f = self._codes(reqs).get("FAN-1")
        self.assertIsNotNone(f)
        self.assertIn("five-second window", f.detail)
        self.assertIn("stand-in", f.detail)


class TestTheSimulatorSearchesBackwardLikeTheProvider(unittest.TestCase):
    """The read path matched only prefixes this request happens to mark.

    Anthropic searches backward from each breakpoint for an earlier cached
    prefix, so a breakpoint that advances with the conversation reads the
    shorter entry a previous turn wrote even though nothing marks that position
    now. Matching only the current request's own markers modelled that policy as
    never hitting anything.

    It is not an academic gap. On a three-turn conversation the bake-off
    verdict reverses: a moving trailing breakpoint scored $0.3375 against a
    static system marker's $0.2545 and lost, when it actually costs $0.1765 and
    wins. The tool would have told a client to abandon the cheaper placement.

    `Plan.prefixes` was fixed for this same shape one layer up, with a comment
    calling it a defamatory way to model a baseline. The read path had it too.
    """

    LAYERS = [("system", 2000, "sys"), ("user", 8000, "u1"),
              ("assistant", 8000, "a1"), ("assistant", 8000, "a2")]

    def _reqs(self):
        return [Request(
            request_id=f"r{n}", sent_at=T0 + timedelta(seconds=60 * n),
            model="claude-opus-5", agent="a", session="s",
            target_id="anthropic/direct", usage={},
            segments=[Segment(id=sid, role=role, tokens=tok, index=i)
                      for i, (role, tok, sid) in enumerate(self.LAYERS[:n])])
            for n in (2, 3, 4)]

    @staticmethod
    def _static(r, **kw):
        from cacheeconomics.allocate import Plan
        return Plan(policy="static", marker_indices=[0], ttls={0: "5m"})

    @staticmethod
    def _moving(r, **kw):
        from cacheeconomics.allocate import Plan
        i = len(r.segments) - 1
        return Plan(policy="moving", marker_indices=[i], ttls={i: "5m"})

    def _cost(self, policy, assume=None):
        from cacheeconomics import simulate
        res = simulate.simulate(self._reqs(), policy,
                                assume=assume or simulate.NEUTRAL)
        rd = sum(u.cache_read for u in res.usages)
        w = sum(u.cache_write_5m + u.cache_write_1h for u in res.usages)
        un = sum(u.uncached_input for u in res.usages)
        return (rd * 0.1 + w * 1.25 + un) * 5 / 1e6, rd

    def test_an_advancing_breakpoint_reads_what_an_earlier_turn_wrote(self):
        _cost, reads = self._cost(self._moving)
        self.assertGreater(reads, 0,
                           "a moving breakpoint still reads nothing, so the "
                           "backward search is not being modelled")

    def test_and_that_reverses_the_bakeoff_verdict(self):
        moving, _ = self._cost(self._moving)
        static, _ = self._cost(self._static)
        self.assertLess(moving, static,
                        "the cheaper placement is still scored as the worse one")

    def test_it_cannot_invent_a_hit_on_content_that_was_never_sent(self):
        """The key is the tuple of segment ids, so a read requires the cached
        span to be a literal prefix of this request. Change the first block and
        nothing behind it may be read."""
        # A distinct first block on *every* turn. An earlier version of this
        # test changed it on all but the first, which broke one link in the
        # chain and left the others intact -- the simulator then correctly read
        # 18,000 tokens and the test read that as a fabrication.
        reqs = self._reqs()
        for n, r in enumerate(reqs):
            r.segments[0] = Segment(id=f"different-{n}", role="system",
                                    tokens=2000, index=0)
        from cacheeconomics import simulate
        res = simulate.simulate(reqs, self._moving, assume=simulate.NEUTRAL)
        self.assertEqual(sum(u.cache_read for u in res.usages), 0,
                         "read a prefix that was never sent")

    def test_the_pessimistic_arm_still_bounds_the_search(self):
        """The window is a real provider constraint and the pessimistic arm is
        where this project enforces it. Removing the exact-match requirement
        must not quietly remove that too.

        The fixture has to put a cached prefix OUTSIDE the window or the test
        proves nothing. An earlier version reused `_reqs()`, whose three turns
        are all within any plausible lookback, and asserted `cost_p >= cost_n`
        -- which holds trivially when the two arms are identical. Deleting the
        window enforcement passed it.
        """
        from cacheeconomics import simulate
        from cacheeconomics.allocate import Plan
        from cacheeconomics.simulate import registry_lookback
        window = registry_lookback("anthropic/direct")
        self.assertTrue(window, "no lookback recorded, so this cannot be tested")
        self.assertTrue(simulate.PESSIMISTIC.enforce_lookback)

        # A conversation far longer than the window, so the entry the first
        # turn wrote sits well behind the advancing breakpoint.
        n = window + 6
        layers = [("system", 4000, "sys")] + [
            ("user", 4000, f"m{i}") for i in range(n)]

        def reqs():
            out = []
            for k in range(2, len(layers) + 1):
                out.append(Request(
                    request_id=f"r{k}", sent_at=T0 + timedelta(seconds=30 * k),
                    model="claude-opus-5", agent="a", session="s",
                    target_id="anthropic/direct", usage={},
                    segments=[Segment(id=sid, role=role, tokens=tok, index=i)
                              for i, (role, tok, sid) in enumerate(layers[:k])]))
            return out

        def moving(r, **kw):
            i = len(r.segments) - 1
            return Plan(policy="moving", marker_indices=[i], ttls={i: "5m"})

        def reads(assume):
            res = simulate.simulate(reqs(), moving, assume=assume)
            return sum(u.cache_read for u in res.usages)

        neutral, pessimistic = reads(simulate.NEUTRAL), reads(simulate.PESSIMISTIC)
        self.assertGreater(neutral, 0, "nothing read even unbounded; fixture is wrong")
        self.assertLess(pessimistic, neutral,
                        "the pessimistic arm read as much as the unbounded one, "
                        "so the window is not being enforced")


class TestTodaysFixesDidNotBreakSomethingElse(unittest.TestCase):
    """Three regressions introduced by fixes made earlier the same day.

    Each fix was verified only against the case that motivated it, which is
    exactly how a fix introduces a second defect.
    """

    def _u(self, w, r=0):
        return {"input_tokens": 0, "cache_read_input_tokens": r,
                "cache_creation_input_tokens": w,
                "cache_creation": {"ephemeral_5m_input_tokens": w,
                                   "ephemeral_1h_input_tokens": 0}}

    def test_reb0_does_not_suppress_reb1(self):
        """REB-0 and REB-1 shared a function and REB-0 returned first, so one
        sessionless cache write anywhere silenced the rebuild analysis of every
        grouped session -- while REB-0's detail told the reader REB-1 covered
        the rest. It never ran."""
        reqs = [Request(request_id="lone", sent_at=T0, model="claude-opus-5",
                        usage=self._u(50_000))]
        p = 100_000
        for t in range(24):
            cold = t % 3 == 0
            reqs.append(Request(
                request_id=f"s{t}", sent_at=T0 + timedelta(seconds=60 * (t + 1)),
                model="claude-opus-5", session="s1",
                usage=self._u(p if cold else 2_000, 0 if cold else p)))
            p += 2_000
        got = {f.code for f in analyze(
            TraceSet(requests=reqs, tier=Tier.USAGE_ONLY, source="t"),
            allow_unreconciled=True).findings}
        self.assertIn("REB-0", got)
        self.assertIn("REB-1", got, "the unmeasurable slice hid the measurable one")

    def test_the_simulator_cannot_read_past_every_current_marker(self):
        """The backward search added today skipped the reachability test in the
        neutral arm, because it sat inside `if window is not None`. The provider
        searches back FROM a breakpoint, so an entry past every marker this
        request places cannot be found. Reachability is not a pessimistic
        assumption; only the window is."""
        from cacheeconomics import simulate
        from cacheeconomics.allocate import Plan
        segs = lambda n: [Segment(id=c, role="system", tokens=2000, index=i)
                          for i, c in enumerate("abcd"[:n])]
        reqs = [Request(request_id="r1", sent_at=T0, model="claude-opus-5",
                        agent="a", session="s", target_id="anthropic/direct",
                        usage={}, segments=segs(3)),
                Request(request_id="r2", sent_at=T0 + timedelta(seconds=30),
                        model="claude-opus-5", agent="a", session="s",
                        target_id="anthropic/direct", usage={}, segments=segs(4))]

        def plan(r, **kw):
            # r1 marks all three; r2 marks only the first.
            i = 2 if r.request_id == "r1" else 0
            return Plan(policy="p", marker_indices=[i], ttls={i: "5m"})

        res = simulate.simulate(reqs, plan, assume=simulate.NEUTRAL)
        self.assertEqual(res.usages[1].cache_read, 0,
                         "read an entry no current breakpoint can reach")

    def test_token_weighting_normalises_usage_first(self):
        """`_weight` returned 0 for any non-dict usage, but this loader accepts
        usage as a JSON string. One uncounted million-token row in that shape
        weighed nothing, so 99 tiny counted rows reported fully counted -- the
        row-counted gate reopening under schema drift."""
        import json
        import os
        import tempfile

        from cacheeconomics.trace import load_jsonl

        def row(i, tok, counted, as_string):
            u = {"input_tokens": tok, "cache_read_input_tokens": 0,
                 "cache_creation_input_tokens": 0}
            return {"request_id": f"r{i}", "sent_at": f"2026-07-29T09:{i % 60:02d}:00Z",
                    "model": "claude-opus-5", "target_id": "anthropic/direct",
                    "session": "s", "ttl_requested": "5m", "tokens_counted": counted,
                    "usage": json.dumps(u) if as_string else u,
                    "segments": [{"id": f"a{i}", "role": "system", "tokens": tok // 2,
                                  "index": 0, "cache_marked": True, "ttl": "5m"},
                                 {"id": f"b{i}", "role": "user", "tokens": tok // 2,
                                  "index": 1}]}
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "x.jsonl")
            with open(p, "w") as f:
                for i in range(99):
                    f.write(json.dumps(row(i, 100, True, False)) + "\n")
                f.write(json.dumps(row(99, 1_000_000, False, True)) + "\n")
            ts = load_jsonl(p, b"k" * 32)
        self.assertLess(ts.tokens_counted, 0.01,
                        "a JSON-string usage row weighed nothing")
        self.assertFalse(ts.tokens_are_counted)


class TestTheAllocatorSearchesPlansItCanCost(unittest.TestCase):
    """The exact evaluator scored a plan the search could never reach.

    `_mixed_variants` re-labels the positions the uniform DP already chose, and
    the DP optimises each lifetime alone. When neither uniform plan beats
    sending the prompt uncached it returns no positions, so the mixed pattern
    has nothing to vary and a plan cheaper than both is never scored.

    Two implementations of one question disagreeing is the strongest signal
    available in this module, and it was sitting there.
    """

    def _segs(self, n=2, tokens=600):
        return [Segment(id=chr(97 + i), role="system", tokens=tokens, index=i)
                for i in range(n)]

    def _stable(self, n=2):
        return {i: 0.0 for i in range(n)}

    def test_it_finds_the_plan_the_exact_evaluator_prices(self):
        from cacheeconomics import tiers
        gaps = [600.0] * 10 + [7200.0] * 10
        a = tiers.allocate(self._segs(), self._stable(),
                           target_id="anthropic/direct", model="claude-opus-5",
                           gaps=gaps)
        sb, wr, rr = tiers._surface("anthropic/direct", "claude-opus-5")
        exact = tiers.expected_cost([(600, 1.0), (600, 1.0)], ["5m", "1h"],
                                    rr, wr, gaps, [tiers.survival(gaps, "5m"),
                                                   tiers.survival(gaps, "1h")])
        self.assertAlmostEqual(a.expected_cost, exact, places=1,
                               msg="the search still cannot reach the plan its "
                                   "own evaluator prices cheapest")
        self.assertEqual([(t.marker_position, t.ttl) for t in a.tiers],
                         [(0, "5m"), (1, "1h")])

    def test_and_it_beats_doing_nothing(self):
        """1035 against 1200 uncached. The old answer was 'no markers'."""
        from cacheeconomics import tiers
        a = tiers.allocate(self._segs(), self._stable(),
                           target_id="anthropic/direct", model="claude-opus-5",
                           gaps=[600.0] * 10 + [7200.0] * 10)
        self.assertTrue(a.tiers)
        self.assertLess(a.expected_cost, a.uncached_cost)

    def test_the_canonical_pattern_was_the_wrong_way_round_here(self):
        """`_mixed_variants` only ever tries long-at-the-bottom, on the theory
        that a stable prefix outlives an advancing turn. The winning plan here
        is the opposite, which is why re-labelling could not have found it even
        with positions."""
        from cacheeconomics import tiers
        a = tiers.allocate(self._segs(), self._stable(),
                           target_id="anthropic/direct", model="claude-opus-5",
                           gaps=[600.0] * 10 + [7200.0] * 10)
        self.assertEqual([t.ttl for t in a.tiers], ["5m", "1h"])

    def test_a_zero_budget_means_zero(self):
        """`budget or surf_budget` made 0 indistinguishable from None, so a
        caller with no breakpoints left got one emitted over their cap. The
        underlying DP also raised IndexError on it rather than returning a
        plan."""
        from cacheeconomics import tiers
        a = tiers.allocate(self._segs(), self._stable(),
                           target_id="anthropic/direct", model="claude-opus-5",
                           gaps=[60.0] * 20, budget=0)
        self.assertEqual(a.tiers, [])
        self.assertTrue(any("budget is zero" in n for n in a.notes))

    def test_a_negative_budget_is_refused(self):
        from cacheeconomics import tiers
        with self.assertRaises(tiers.Unsupported):
            tiers.allocate(self._segs(), self._stable(),
                           target_id="anthropic/direct", model="claude-opus-5",
                           gaps=[60.0] * 20, budget=-1)

    def test_a_nonzero_budget_still_places(self):
        """The guard must not swallow the ordinary case."""
        from cacheeconomics import tiers
        a = tiers.allocate(self._segs(), self._stable(),
                           target_id="anthropic/direct", model="claude-opus-5",
                           gaps=[60.0] * 20, budget=1)
        self.assertTrue(a.tiers)


class TestAPlacementIsShippedNotPublished(unittest.TestCase):
    """Allocator output goes into somebody's production prompt.

    There is no reconciliation gate in front of it, so a wrong placement is
    applied rather than merely printed. These two were both silent: one
    recommends a lifetime the provider will not honour, the other prices one
    breakpoint and returns another.
    """

    def _req(self, segs, **kw):
        base = dict(request_id="r", sent_at=T0, model="claude-opus-5",
                    target_id="anthropic/direct", segments=segs, usage={})
        base.update(kw)
        return Request(**base)

    def test_lite_will_not_recommend_a_lifetime_the_model_lacks(self):
        """Bedrock narrows claude-opus-4-1 to 5m. Cadence chose the lifetime on
        its own, so a ten-minute median produced a 1h marker and a note saying
        the longer lifetime pays."""
        from cacheeconomics import registry
        from cacheeconomics.allocate import allocator_lite
        self.assertEqual(
            list(registry.supported_ttls("amazon-bedrock/converse",
                                         "claude-opus-4-1")), ["5m"],
            "registry changed; this fixture no longer tests what it says")
        segs = [Segment(id="s", role="system", tokens=2000, index=0),
                Segment(id="u", role="user", tokens=100, index=1)]
        p = allocator_lite(self._req(segs, model="claude-opus-4-1",
                                     target_id="amazon-bedrock/converse"),
                           volatility={0: 1, 1: 2}, cadence_seconds=600)
        self.assertNotIn("1h", p.ttls.values())
        self.assertTrue(any("not available here" in n for n in p.notes),
                        "silently downgraded without saying why")

    def test_lite_still_takes_the_long_lifetime_where_it_exists(self):
        """The guard must not suppress the recommendation it was built for."""
        from cacheeconomics.allocate import allocator_lite
        segs = [Segment(id="s", role="system", tokens=2000, index=0),
                Segment(id="u", role="user", tokens=100, index=1)]
        p = allocator_lite(self._req(segs), volatility={0: 1, 1: 2},
                           cadence_seconds=600)
        self.assertEqual(set(p.ttls.values()), {"1h"})

    def test_full_keeps_the_order_it_priced(self):
        """`wire_order if applied else None` dropped an explicit order when no
        Move justified it, so the allocation was costed over the reordered
        emission and the plan reverted to authored order. `Plan.prefixes` then
        marks segments nobody priced."""
        from cacheeconomics.allocator import allocator_full
        segs = [Segment(id=c, role="system", tokens=6000, index=i)
                for i, c in enumerate("abc")]
        p = allocator_full(self._req(segs), volatility={0: 0, 1: 0, 2: 0},
                           cadence_seconds=None, gaps=[60.0] * 20,
                           order=[2, 1, 0])
        self.assertEqual(p.order, [2, 1, 0])
        self.assertEqual([s.index for s in p.emission(segs)], [2, 1, 0],
                         "the plan emits a different order than it was costed on")

    def test_full_leaves_the_authored_order_alone(self):
        """An order equal to the authored one carries no information and should
        not be stamped onto the plan."""
        from cacheeconomics.allocator import allocator_full
        segs = [Segment(id=c, role="system", tokens=6000, index=i)
                for i, c in enumerate("abc")]
        p = allocator_full(self._req(segs), volatility={0: 0, 1: 0, 2: 0},
                           cadence_seconds=None, gaps=[60.0] * 20,
                           order=[0, 1, 2])
        self.assertIsNone(p.order)
