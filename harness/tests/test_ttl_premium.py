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

from cacheeconomics.analyzer import analyze                       # noqa: E402
from cacheeconomics.trace import Request, Tier, TraceSet          # noqa: E402

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
