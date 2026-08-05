"""An unknown model or surface must never produce a recommendation.

The second recurring class across thirteen review rounds, after twin paths.
Five separate components had to learn the same thing: when the registry does
not know a model's minimum cacheable prefix, the honest answer is to place
nothing and say why. Below that threshold a provider caches nothing and returns
no error, so a marker placed on a guess is a write premium paid for silence --
and not knowing the threshold is exactly when a guess is most likely wrong.

`tiers.allocate`, `plugin._filter_near_minimum`, `allocate._place`,
`checks.check_minimum` and `analyzer` each failed open at some point, and each
was fixed on the round that found it. This sweeps every public entry point
instead, so the sixth one fails here rather than in somebody's report.

The bar for every entry point: given a model or surface the registry has never
heard of, produce no marker, no dollar figure, and a reason.
"""

import argparse
import json
import math
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cacheeconomics import checks, cost, monitor, plugin, tiers  # noqa: E402
from cacheeconomics.allocate import allocator_lite, litellm_auto  # noqa: E402
from cacheeconomics.allocator import allocator_full  # noqa: E402
from cacheeconomics.analyzer import analyze  # noqa: E402
from cacheeconomics.relocate import relocation_lite  # noqa: E402
from cacheeconomics.trace import (ASSUMED_PROVIDER_SURFACE,  # noqa: E402
                                  UNATTRIBUTED, Request, Segment, Tier,
                                  TraceSet)

T0 = datetime(2026, 7, 29, 9, tzinfo=timezone.utc)
UNKNOWN_MODEL = "nobody-has-ever-registered-this"
UNKNOWN_TARGET = "nobody/registered-this-surface"

# What a loader writes into `assumed_inputs` when it had to guess the surface.
#
# From `trace.py`, which owns the field, so a rewording moves both together --
# and `trace` imports nothing from the package, so every loader and the analyzer
# can reach it without a cycle. The `getattr` is only for this worktree, whose
# copy of `trace.py` predates the constant.
#
# The value matters here in a way it does not in `analyzer.py`. The analyzer is
# deliberately generic: it renders whatever `assumed_inputs` holds and never
# compares against a literal, so there is nothing there to drift. These tests
# do the comparing, and they were asserting on `"surface"` -- a string no loader
# emits. They would have gone on passing while the adapter said something else
# entirely, which is the drift the constant exists to prevent.
SURFACE_ASSUMED = getattr(__import__(
    "cacheeconomics.trace", fromlist=["trace"]),
    "ASSUMED_PROVIDER_SURFACE", "provider surface")


def _segments():
    return [Segment(id="tools", role="tools", tokens=9_000, index=0),
            Segment(id="policy", role="system", tokens=6_000, index=1),
            Segment(id="turn", role="user", tokens=400, index=2)]


def _req(model=UNKNOWN_MODEL, target="anthropic/direct"):
    return Request(request_id="r", sent_at=T0, model=model,
                   usage={"input_tokens": 10, "cache_read_input_tokens": 5_000,
                          "cache_creation_input_tokens": 0},
                   segments=_segments(), target_id=target, session="s")


class TestNoMarkerOnAnUnknownModel(unittest.TestCase):
    """Every placement entry point, swept together."""

    RATES = {0: 0.0, 1: 0.0, 2: 1.0}
    GAPS = [120.0] * 30
    VOLATILITY = {0: 1, 1: 1, 2: 9}

    def test_allocator_lite(self):
        self.assertEqual(
            allocator_lite(_req(), volatility=self.VOLATILITY,
                           cadence_seconds=120).marker_indices, [])

    def test_relocation_lite(self):
        self.assertEqual(
            relocation_lite(_req(), volatility=self.VOLATILITY,
                            cadence_seconds=120).marker_indices, [])

    def test_allocator_full(self):
        self.assertEqual(
            allocator_full(_req(), rates=self.RATES, gaps=self.GAPS).marker_indices, [])

    def test_tiers_allocate_refuses_outright(self):
        with self.assertRaises(tiers.Unsupported):
            tiers.allocate(_segments(), self.RATES, target_id="anthropic/direct",
                           model=UNKNOWN_MODEL, gaps=self.GAPS)

    def test_the_live_plugin(self):
        p = plugin.CachePlugin(key=b"k" * 32, warmup=4)
        body = {"system": [{"type": "text", "text": "policy " * 4000}],
                "messages": [{"role": "user", "content": "hi"}]}
        last = None
        for i in range(20):
            _, last = p.on_request(body, model=UNKNOWN_MODEL,
                                   at=T0 + timedelta(seconds=90 * i), target_id="anthropic/direct", apply=True)
        self.assertFalse(last.applied)
        self.assertEqual(last.placements, {})

    def test_the_static_check_abstains_rather_than_passing(self):
        r = checks.check_minimum(9_000, UNKNOWN_MODEL)
        self.assertIs(r.status, checks.Status.ABSTAIN)

    def test_the_runtime_monitor_raises_no_minimum_alert(self):
        """It cannot know the prefix is too short, so it must not say so."""
        m, fired = monitor.Monitor(), []
        for i in range(20):
            segs = [Segment(id="a", role="system", tokens=50, index=0,
                            cache_marked=True, ttl="5m")]
            fired += m.observe(Request(request_id=f"r{i}",
                                       sent_at=T0 + timedelta(seconds=60 * i),
                                       model=UNKNOWN_MODEL, usage={},
                                       segments=segs, session="s", target_id="anthropic/direct"))
        self.assertNotIn("RT-MIN", {a.code for a in fired})

    def test_every_refusal_says_why(self):
        """A silent refusal is indistinguishable from nothing being wrong."""
        for name, plan in (
                ("allocator-lite", allocator_lite(_req(), volatility=self.VOLATILITY,
                                                  cadence_seconds=120)),
                ("allocator-full", allocator_full(_req(), rates=self.RATES,
                                                  gaps=self.GAPS))):
            self.assertTrue(any("minimum" in n for n in plan.notes),
                            f"{name} refused without saying it was the minimum")


class TestNoPriceOnAnUnknownSurface(unittest.TestCase):

    def test_the_cost_model_refuses_rather_than_defaulting(self):
        from cacheeconomics import registry
        with self.assertRaises(registry.RegistryError):
            cost.price(cost.Usage(uncached_input=1_000), "claude-opus-5",
                       target_id=UNKNOWN_TARGET)

    def test_the_crossover_question_does_not_apply(self):
        from cacheeconomics import registry
        with self.assertRaises(registry.RegistryError):
            cost.ttl_crossover(UNKNOWN_TARGET)

    def test_the_allocator_refuses_an_unknown_surface(self):
        with self.assertRaises(tiers.Unsupported):
            tiers.allocate(_segments(), {0: 0.0}, target_id=UNKNOWN_TARGET,
                           model="claude-opus-5", gaps=[120.0] * 30)

    def test_the_report_publishes_no_figure_for_it(self):
        """The gate that matters: an unpriceable surface must not reach money."""
        ts = TraceSet(requests=[_req(model="claude-opus-5", target=UNKNOWN_TARGET)],
                      tier=Tier.INSTRUMENTED, source="fail-closed")
        a = analyze(ts, allow_unreconciled=True)
        self.assertFalse(a.total_avoidable_month.released)


class TestAnUnpriceableSurfaceStillGetsItsMeasurements(unittest.TestCase):
    """Refusing to price is not the same as refusing to look.

    `analyze` replaced both `usages` and the rule input with the rows that
    survived pricing, then recomputed the ratios from that subset. So a
    Bedrock or Vertex trace -- surfaces the cloud provider invoices, where the
    recorded Anthropic rates do not apply -- reported `requests: 0`,
    `input_from_cache: None`, `prefix_efficiency: None` and no findings at all,
    directly under a coverage line reading "40 of 40 requests analysable". The
    provider's own counters were sitting in the file the whole time, and those
    two ratios are arithmetic over them that no rate table touches.

    SYNTHETIC. Forty turns a minute apart, each rewriting the whole 50k prefix
    it had just established -- something REB-1 can name from counters alone.
    """

    BEDROCK = "amazon-bedrock/converse"

    def _trace(self, target):
        reqs = [Request(request_id=f"r{i}",
                        sent_at=T0 + timedelta(seconds=60 * i),
                        model="claude-opus-5", target_id=target, tenant="t",
                        session="s", agent="a", ttl_requested="5m",
                        usage={"input_tokens": 200,
                               "cache_read_input_tokens": 5_000,
                               "cache_creation_input_tokens": 50_000},
                        segments=[Segment(id="sys", role="system",
                                          tokens=50_000, index=0,
                                          cache_marked=True, ttl="5m"),
                                  Segment(id=f"t{i}", role="user", tokens=200,
                                          index=1)])
                for i in range(40)]
        return TraceSet(requests=reqs, tier=Tier.INSTRUMENTED, source="x")

    def test_the_surface_really_is_unpriceable(self):
        """Guard the guard. If these rows priced, every assertion below would
        be about the ordinary path and prove nothing."""
        from cacheeconomics import registry
        with self.assertRaises(registry.RegistryError):
            cost.price(cost.Usage(uncached_input=1_000), "claude-opus-5",
                       target_id=self.BEDROCK)

    def test_the_counters_survive(self):
        a = analyze(self._trace(self.BEDROCK), allow_unreconciled=True)
        self.assertEqual(40, a.ratios.get("requests"))
        self.assertIsNotNone(a.ratios.get("input_from_cache"))
        self.assertIsNotNone(a.ratios.get("prefix_efficiency"))

    def test_the_ratios_match_the_priceable_surface_exactly(self):
        """They are ratios of the provider's counters, so the surface cannot
        change them. Asserting equality rather than merely "not None" is what
        makes this a measurement instead of a smoke test."""
        priced = analyze(self._trace("anthropic/direct"), allow_unreconciled=True)
        unpriced = analyze(self._trace(self.BEDROCK), allow_unreconciled=True)
        for k in ("requests", "input_from_cache", "prefix_efficiency"):
            with self.subTest(ratio=k):
                self.assertEqual(priced.ratios.get(k), unpriced.ratios.get(k))

    def test_a_counter_only_finding_still_reaches_the_reader(self):
        a = analyze(self._trace(self.BEDROCK), allow_unreconciled=True)
        self.assertIn("REB-1", [f.code for f in a.findings],
                      "a rebuild the reader's own counters show is still not "
                      "reported on a surface this tool cannot price")

    def test_but_no_dollar_figure_comes_back_with_it(self):
        """The point is that observations survive, not that dollars do."""
        a = analyze(self._trace(self.BEDROCK), allow_unreconciled=True)
        self.assertTrue(a.findings, "nothing to check")
        for f in a.findings:
            with self.subTest(finding=f.code):
                for name in ("avoidable_usd_month", "avoidable_usd_window"):
                    fig = getattr(f, name)
                    self.assertFalse(fig is not None and fig.released,
                                     f"{f.code}.{name} published on an "
                                     f"unpriceable surface")
        self.assertFalse(a.total_avoidable_month.released)
        self.assertFalse(a.spend["input_usd"].released)


class TestARateFreeRuleIsSafeOnATraceNothingCanPrice(unittest.TestCase):
    """`@_rate_free` decides which rules see rows that failed pricing, so what
    the marker has to guarantee is that those rows cannot hurt.

    It used to be checked as "never asks for a price", by spying on `rate_for`
    and `cost.price`. That stopped being the right question when TTL-1 was
    marked. TTL-1 is two claims in one function: a cadence observation that
    needs no rate, and a saving that does -- so it *asks*, gets 0.0 for a
    surface nobody priced, and declines to monetize. Under the old test it was a
    mis-marking; under the property that actually matters it is correct.

    That test also passed while TTL-1 was already marked, which is the more
    useful discovery: its fixture ran every gap 60 seconds apart, so TTL-1 hit
    `if not in_band: continue` and returned before reaching a rate at all. The
    check had gone vacuous for the one rule it needed to judge -- a fixture
    dependency its own docstring had warned about, in the direction the docstring
    did not consider.

    So this asks the two things the marker is for, against the hazard itself
    rather than against a proxy for it. Run over a trace the registry cannot
    price at all: a rate-free rule must not raise, and must not come back
    carrying money. Rewriting rather than relaxing -- the old check would now
    pass by accident, and one that passes by accident is worse than one that
    fails honestly.
    """

    BEDROCK = "amazon-bedrock/converse"

    def _trace(self, target, segments=True):
        def segs(i):
            if not segments:
                return []
            return [Segment(id="sys", role="system", tokens=50_000, index=0,
                            cache_marked=True, ttl="5m"),
                    Segment(id=f"t{i}", role="user", tokens=200, index=1)]
        reqs = [Request(request_id=f"r{i}",
                        sent_at=T0 + timedelta(seconds=900 * i),
                        model="claude-opus-5", target_id=target, tenant="t",
                        session="s", agent="a", ttl_requested="5m",
                        usage={"input_tokens": 200,
                               "cache_read_input_tokens": 5_000,
                               "cache_creation_input_tokens": 500_000},
                        segments=segs(i))
                for i in range(12)]
        return TraceSet(requests=reqs, tier=Tier.INSTRUMENTED, source="x")

    def _ttl2_trace(self):
        """A trace TTL-2 prices: one-hour writes, ten-plus gaps, band rare."""
        def req(rid, when, write_1h, read):
            usage = {"input_tokens": 100, "cache_read_input_tokens": read,
                     "cache_creation_input_tokens": write_1h}
            if write_1h:
                usage["cache_creation"] = {"ephemeral_5m_input_tokens": 0,
                                           "ephemeral_1h_input_tokens": write_1h}
            return Request(request_id=rid, sent_at=when, model="claude-opus-5",
                           target_id="anthropic/direct", tenant="t",
                           session="s", agent="a", ttl_requested="1h",
                           usage=usage, segments=[])

        reqs = [req(f"a{i}", T0 + timedelta(seconds=60 * i), 50_000, 0)
                for i in range(8)]
        base = T0 + timedelta(days=1, hours=12)
        for i in range(6):
            offset = 60 * i + (540 if i >= 4 else 0)
            write, read = 50_000, (200_000 if i == 4 else 0)
            reqs.append(req(f"b{i}", base + timedelta(seconds=offset),
                            write, read))
        return TraceSet(requests=reqs, tier=Tier.USAGE_ONLY, source="x")

    def _rate_free_rules(self):
        from cacheeconomics.analyzer import RULES
        return [r for r in RULES if getattr(r, "rate_free", False)]

    def test_there_are_rate_free_rules_to_check(self):
        self.assertTrue(self._rate_free_rules(), "nothing is marked rate-free")

    def test_the_surface_really_cannot_be_priced(self):
        """Guard the guard. Against a priceable surface every rule behaves and
        this class proves nothing."""
        from cacheeconomics import registry
        with self.assertRaises(registry.RegistryError):
            registry.base_rate("claude-opus-5", "2026-01-01", self.BEDROCK)

    def test_none_of_them_raises_on_an_unpriceable_trace(self):
        """The crash direction. Running *every* rule over unpriceable rows broke
        sixteen tests -- `cost.price` refuses the 0.0 that `rate_for` returns --
        which is why the marking exists at all rather than widening the lot.

        Over every surface the registry knows, not the two I happened to pick.
        The first version of this checked Bedrock and anthropic/direct, which
        both carry Anthropic's multiplier shape, so it could not see the defect
        that widening TTL-1's input actually caused.
        """
        from cacheeconomics import registry
        from cacheeconomics.analyzer import analyze
        targets = list(registry.target_ids())
        self.assertGreater(len(targets), 2, "too few surfaces to be a sweep")
        for segments in (True, False):
            for target in targets:
                with self.subTest(segments=segments, target=target):
                    analyze(self._trace(target, segments),
                            allow_unreconciled=True)

    def test_a_surface_with_no_two_lifetimes_abstains_rather_than_raising(self):
        """The member the sweep above was widened to catch.

        `openai/direct` is in the registry and answers
        `{'write_pre_5_6': 1.0, 'write_5_6_plus': 1.25, 'read': None}` -- no
        five-minute or one-hour lifetime at all, and a null read. TTL-1 became
        `rate_free` and so began receiving those rows; it caught `RegistryError`
        from `registry.multipliers` and then indexed `write_5m` on what came
        back, taking the whole analysis down with `KeyError: 'write_5m'`.

        Catching the error covers a surface the registry has never heard of. It
        says nothing about one the registry knows and describes differently,
        which is the case that broke -- and is exactly the hazard that made
        run-wide-with-fences the riskier option: widening the input set widens
        the *shapes* reaching the code, not only the volume.
        """
        from cacheeconomics import registry
        from cacheeconomics.analyzer import analyze
        m = registry.multipliers("openai/direct")
        self.assertNotIn("write_5m", m,
                         "openai/direct now has Anthropic's shape; this test "
                         "no longer exercises a differently-shaped map")
        for segments in (True, False):
            with self.subTest(segments=segments):
                a = analyze(self._trace("openai/direct", segments),
                            allow_unreconciled=True)
                for f in a.findings:
                    for name in ("avoidable_usd_month", "avoidable_usd_window"):
                        fig = getattr(f, name)
                        self.assertFalse(
                            fig is not None and fig.raw(),
                            f"{f.code} priced a lifetime change on a surface "
                            f"that does not sell two lifetimes")

    def test_every_surface_gets_the_decision_we_intend(self):
        """Per-surface expected outcome, not merely "nothing exploded".

        The sweep above asserts `analyze()` does not raise, and that is a weaker
        claim than it looks: a wrong price does not raise. It is the same gap
        that let TTL-1 crash on `openai/direct` while two hand-picked fixtures
        passed, one level up -- crash-freedom is not correctness.

        So each shipped surface is named with the decision it should get.
        `deepseek/direct` is not in the registry's multiplier table at all;
        `openai/direct` is, and describes a different product with no two
        lifetimes to compare. Both must abstain. The four Anthropic-shaped
        surfaces must validate. A surface added to the registry without being
        added here fails on the count, which is the point: a new surface is a
        new shape until somebody says otherwise.
        """
        from cacheeconomics import registry
        from cacheeconomics.analyzer import _lifetime_multipliers
        expected = {
            "anthropic/direct": True,
            "anthropic/claude-platform-on-aws": True,
            "amazon-bedrock/converse": True,
            "google-cloud/vertex": True,
            # Prompt caching without a lifetime choice: no 5m/1h keys, null read.
            "openai/direct": False,
            # Not in the multiplier table at all.
            "deepseek/direct": False,
        }
        self.assertEqual(
            sorted(expected), sorted(registry.target_ids()),
            "the registry's surfaces changed; state the intended "
            "validate-or-abstain decision for each new one")
        for target, should_validate in expected.items():
            with self.subTest(target=target):
                try:
                    m = registry.multipliers(target)
                except registry.RegistryError:
                    self.assertFalse(should_validate,
                                     "expected a multiplier map, got none")
                    continue
                got = _lifetime_multipliers(m) is not None
                self.assertEqual(should_validate, got,
                                 f"{target} validates={got}, intended "
                                 f"{should_validate}")

    def test_no_rule_prices_from_a_multiplier_map_it_did_not_validate(self):
        """Enumerated by poisoning the registry, not by reading the source.

        The previous audit was a sentence -- "REB-1, REB-0, SPL-1 and MIN-1
        never touch a multiplier map, VOL-1 and FAN-1 now use the validator" --
        and it was wrong: it never checked TTL-2, which read `.get()` and tested
        truthiness, so a `write_5m` of `True` priced at 1.0x. Reading the source
        has now been wrong in both directions in this file, so the enumeration
        moves into the test.

        `registry.multipliers` is replaced with one returning `write_5m=True`
        beside real numbers. `True` is truthy, so a `.get()`-plus-truthiness
        guard waves it through, and `bool` subclasses `int`, so the arithmetic
        prices the five-minute write at 1.0x instead of 1.25x -- quietly, and in
        the flattering direction. Any rule that reaches for a multiplier without
        validating shows up as a nonzero dollar amount here.

        The first poison map set every value to `True`, and it caught nothing:
        `w1h <= w5` is then `1 <= 1`, so TTL-2 skipped the scope for an
        unrelated reason and looked innocent. Verified by reverting the fix and
        watching this pass. A poison that every rule declines for the wrong
        reason tests nothing, so one value stays real and the ordering holds.
        """
        from cacheeconomics import analyzer, registry
        from cacheeconomics.analyzer import RULES, analyze

        poisoned = {"write_5m": True, "write_1h": 2.0, "read": 0.1}

        self._assert_poison_is_refused(scope="analyzer")

    POISON = {"write_5m": True, "write_1h": 2.0, "read": 0.1}

    def _rule_figures(self, ts, ratios):
        from cacheeconomics.analyzer import RULES

        def rate_for(model, when=None, target_id=None):
            return 5.0

        out = {}
        for rule in RULES:
            f = rule(ts.analysable, ratios, 3.0, rate_for)
            if f is None:
                continue
            for name in ("avoidable_usd_month", "avoidable_usd_window"):
                fig = getattr(f, name)
                if fig is not None and fig.raw():
                    out[f"{rule.__name__}.{name}"] = fig.raw()
        return out

    def _direct_multiplier_readers(self, ts, ratios):
        """Which rules reach for a multiplier map themselves, observed.

        Named nowhere: a spying shim records the calls, so the allow-list below
        is what the code does rather than what I remember it doing. That
        distinction has already been wrong in both directions in this file.
        """
        from cacheeconomics import analyzer, registry

        def rate_for(model, when=None, target_id=None):
            return 5.0

        seen, real = set(), analyzer.registry

        class Spy:
            def __init__(self, name):
                self.name = name

            def __getattr__(self, attr):
                return getattr(registry, attr)

            def multipliers(self, target_id):
                seen.add(self.name)
                return registry.multipliers(target_id)

        from cacheeconomics.analyzer import RULES
        for rule in RULES:
            analyzer.registry = Spy(rule.__name__)
            try:
                rule(ts.analysable, ratios, 3.0, rate_for)
            except Exception:                                  # noqa: BLE001
                pass
            finally:
                analyzer.registry = real
        return seen

    def _assert_poison_is_refused(self, scope):
        """Every priced figure must be unchanged, or absent because the rule
        that owns it validated and abstained. Nothing else.

        Three outcomes are failures and the earlier version of this only looked
        for one. It compared keys present in *both* runs, so a figure that
        DISAPPEARED under poison was ignored -- and a rule that silently stops
        pricing is exactly as wrong as one that prices wrongly. Measured: with
        the shared registry poisoned, EFF-1's $72.30 vanishes on the segments
        fixture, because a `write_5m` of 1.0 makes caching cost exactly what not
        caching costs and the rule concludes there is nothing to report. That is
        a mispricing wearing the shape of an abstention.

        So disappearance is allowed only for rules observed to read a multiplier
        map directly -- those validate and refuse, which is the designed
        behaviour. Every other rule must come back with the number it had.
        """
        from cacheeconomics import analyzer, registry
        from cacheeconomics.analyzer import analyze

        poison = dict(self.POISON)

        class Poisoned:
            def __getattr__(self, attr):
                return getattr(registry, attr)

            def multipliers(self, target_id):
                return dict(poison)

        fixtures = [("segments", self._trace("anthropic/direct", segments=True)),
                    ("one-hour writes", self._ttl2_trace())]
        wrong, priced_clean, readers = [], set(), set()
        for label, ts in fixtures:
            ratios = analyze(ts, allow_unreconciled=True).ratios
            clean = self._rule_figures(ts, ratios)
            priced_clean |= set(clean)
            readers |= self._direct_multiplier_readers(ts, ratios)

            real_analyzer, real_multipliers = analyzer.registry, registry.multipliers
            if scope == "analyzer":
                analyzer.registry = Poisoned()
            else:
                # The shared module, so `cost.price` -- which EFF-1 and spend
                # pricing both delegate to -- sees it too. Scoping to the
                # analyzer leaves the delegated path, which is the real one, out
                # of the test entirely.
                registry.multipliers = lambda target_id: dict(poison)
            try:
                dirty = self._rule_figures(ts, ratios)
            finally:
                analyzer.registry = real_analyzer
                registry.multipliers = real_multipliers

            for k in sorted(set(clean) | set(dirty)):
                before, after = clean.get(k), dirty.get(k)
                if before == after:
                    continue
                if after is None:
                    # Abstention. Safe in every case, and deliberately not
                    # policed further here: a rule that validates refuses, and a
                    # rule whose pricing path raises also refuses, and the two
                    # are indistinguishable from the figure alone. The
                    # difference between refusing and being mispriced into
                    # silence is asserted at the seam instead, by
                    # `test_the_delegated_pricing_path_also_refuses_a_boolean`.
                    continue
                verdict = "appeared" if before is None else "changed"
                wrong.append(f"{label}: {k} {verdict} {before!r} -> {after!r}")

        self.assertTrue(priced_clean, "no rule priced with a real registry")
        self.assertTrue(
            any(k.startswith("_f_ttl_premium_unearned") for k in priced_clean),
            f"TTL-2 does not price on these fixtures, so poisoning the "
            f"registry cannot reveal how it reads one: {sorted(priced_clean)}")
        self.assertTrue(readers, "no rule was observed reading a multiplier map")
        self.assertEqual(
            [], sorted(wrong),
            "a boolean multiplier changed what these published, so something on "
            "the path accepted it as 1.0x:\n    " + "\n    ".join(sorted(wrong)))

    def test_the_shared_registry_never_produces_a_wrong_number(self):
        """The same audit against the registry every module shares.

        The scoped version above leaves `cost.price` on the real registry, so it
        proves something about the four rules that read multipliers themselves
        and nothing about the path EFF-1 and spend actually price through -- a
        test built so the known hole cannot be observed is a test that agrees
        with itself. This one poisons the shared module, so the delegated path
        is in scope.

        It cannot, on its own, catch today's hole: a bool priced at 1.0x makes
        caching cost exactly what not caching costs, so EFF-1 concludes there is
        nothing to report and its figure vanishes -- which is shaped exactly
        like the abstention it will make once `cost.py` refuses the bool. Same
        observable, two different causes. That distinction is asserted at the
        seam below instead of guessed at from a figure here.
        """
        self._assert_poison_is_refused(scope="shared")

    def test_the_delegated_pricing_path_also_refuses_a_boolean(self):
        """At the seam, because the figure cannot tell the two cases apart.

        `cost.price` is where EFF-1 and spend both get their numbers, and its
        guard is `isinstance(m.get(k), (int, float))` -- which `bool` satisfies,
        because `bool` subclasses `int`. So `write_5m=True` is accepted and the
        five-minute write prices at 1.0x instead of 1.25x. The analyzer's own
        `_lifetime_multipliers` refuses exactly this; the delegated path does
        not, and `cost.py` is Track B's file.

        Asserting the refusal rather than its downstream effect is what makes
        this unambiguous: a wrong price and a refused price both end with EFF-1
        publishing nothing, so only the seam distinguishes them.

        Verified both directions before being written this way: it fails today,
        and passes once `cost.py` rejects `bool` the way the analyzer does.
        """
        from cacheeconomics import cost, registry
        real = registry.multipliers
        registry.multipliers = lambda target_id: dict(self.POISON)
        try:
            with self.assertRaises(registry.RegistryError):
                cost.price(cost.Usage(cache_write_5m=1_000),
                           "claude-opus-5", target_id="anthropic/direct",
                           on_date="2026-01-01")
        finally:
            registry.multipliers = real

    def test_no_rule_sorts_on_anything_but_scalars(self):
        """A sort key that can raise is a rule that cannot fail closed.

        TTL-1 ordered its per-scope writes with a bare `.sort()`, which compared
        whole tuples including the Request at the end. That was narrowed to
        `w[:5]` -- and narrowing leaves whatever is left: index 3 is `spans`,
        which carries segment ids inside tuples, so two writes at one instant
        with the same token count and rate still fell through to comparing ids.
        Measured on a directly-constructed trace with one string id and one
        integer id: `TypeError: '<' not supported between instances of 'int' and
        'str'`, raised out of `analyze` before any gate could refuse anything.

        Directly-constructed Requests are supported input elsewhere in this
        suite, so an id the loaders would never emit is reachable. The second
        time a sort key has been narrowed rather than replaced.
        """
        from cacheeconomics.analyzer import analyze

        def req(rid, when, seg_id):
            return Request(request_id=rid, sent_at=when, model="claude-opus-5",
                           target_id="anthropic/direct", tenant="t",
                           session="s", agent="a", ttl_requested="5m",
                           usage={"input_tokens": 0,
                                  "cache_read_input_tokens": 0,
                                  "cache_creation_input_tokens": 100_000},
                           segments=[Segment(id=seg_id, role="system",
                                             tokens=50_000, index=0,
                                             cache_marked=True, ttl="5m")])

        # Same instant, same tokens, same rate: everything a chronological key
        # should look at ties, so anything else in the key gets compared.
        reqs = [req("r0", T0, "alpha"), req("r1", T0, 7)]
        reqs += [req(f"r{i}", T0 + timedelta(seconds=600 * i), f"s{i}")
                 for i in range(2, 12)]
        a = analyze(TraceSet(requests=reqs, tier=Tier.INSTRUMENTED, source="x"),
                    allow_unreconciled=True)
        self.assertIsNotNone(a, "the analysis raised instead of abstaining")

    SHAPES = (
        ("good", {"write_5m": 1.25, "write_1h": 2.0, "read": 0.1}, True),
        ("missing write_5m", {"write_1h": 2.0, "read": 0.1}, False),
        ("missing read", {"write_5m": 1.25, "write_1h": 2.0}, False),
        ("null read", {"write_5m": 1.25, "write_1h": 2.0, "read": None}, False),
        ("boolean read", {"write_5m": 1.25, "write_1h": 2.0, "read": True}, False),
        ("boolean write", {"write_5m": True, "write_1h": 2.0, "read": 0.1}, False),
        ("string", {"write_5m": 1.25, "write_1h": "2.0", "read": 0.1}, False),
        ("openai shape", {"write_pre_5_6": 1.0, "write_5_6_plus": 1.25,
                          "read": None}, False),
    )

    def test_the_multiplier_validator_has_one_definition(self):
        """Seven consumers of `registry.multipliers`, one decision.

        The AST audit that produced `_lifetime_multipliers` found four analyzer
        rules reading multiplier maps and three consumers elsewhere. Fixing the
        four here while `cost.py` kept its own guard would leave two definitions
        of "is this map usable", which is the shape the helper exists to close --
        `cost.py` had already excluded `bool` for `effective_rate`, one branch
        away from the multiplier reads, and simply had not applied the rule to
        the values beside it. A rule known in a file and not applied there is
        not a rule.

        `analyzer` imports `cost` and not the reverse, so the shared definition
        lives in `cost` and this delegates to it.
        """
        from cacheeconomics import cost
        self.assertTrue(
            hasattr(cost, "unusable_multipliers"),
            "cost.unusable_multipliers is not present in this worktree, so "
            "_lifetime_multipliers is still deciding for itself")

    def test_the_validator_agrees_with_the_shared_one_wherever_it_exists(self):
        """Identical verdicts across the shape table, so adopting the shared
        definition cannot quietly change what this package refuses.

        Runs against whichever definition is present, so it is meaningful before
        the merge (against the fallback) and after it (against `cost`), and the
        delegation is only safe because both answer the same on every row.
        """
        from cacheeconomics import cost
        from cacheeconomics.analyzer import _lifetime_multipliers
        shared = getattr(cost, "unusable_multipliers", None)
        for label, mapping, usable in self.SHAPES:
            with self.subTest(shape=label):
                self.assertEqual(usable, _lifetime_multipliers(mapping) is not None)
                if shared is not None:
                    self.assertEqual(
                        usable,
                        not shared(mapping, ("write_5m", "write_1h", "read")),
                        f"{label}: the shared validator and this one disagree")

    def test_the_validator_refuses_every_shape_it_cannot_price(self):
        """The helper itself, over the shapes that reach it.

        `read: None` is the one worth naming: it passes a `"read" in m` test and
        then fails in arithmetic with a TypeError, one layer further from the
        cause than a refusal here. `True` is refused because `bool` subclasses
        `int`, so a hand-edited registry entry would otherwise price at 1.0x.
        """
        from cacheeconomics.analyzer import _lifetime_multipliers
        good = {"write_5m": 1.25, "write_1h": 2.0, "read": 0.1}
        self.assertEqual((1.25, 2.0, 0.1), _lifetime_multipliers(good))
        for label, bad in (
                ("missing write_5m", {"write_1h": 2.0, "read": 0.1}),
                ("missing read", {"write_5m": 1.25, "write_1h": 2.0}),
                ("null read", {**good, "read": None}),
                ("boolean", {**good, "read": True}),
                ("string", {**good, "write_1h": "2.0"}),
                ("openai shape", {"write_pre_5_6": 1.0,
                                  "write_5_6_plus": 1.25, "read": None}),
                ("not a mapping", None)):
            with self.subTest(shape=label):
                self.assertIsNone(_lifetime_multipliers(bad))

    def test_none_of_them_comes_back_carrying_money(self):
        """The money direction, at the rule's own output rather than at the
        publication gate. The gate does withhold everything on a trace with
        unpriceable rows, so a figure here would not reach a client today --
        but that is a second mechanism agreeing, and a rule that computes a
        dollar amount from a rate nobody has is wrong before anything downstream
        decides whether to print it.
        """
        from cacheeconomics.analyzer import RULES, analyze

        def rate_for(model, when=None, target_id=None):
            from cacheeconomics import registry
            try:
                return registry.base_rate(model, when or "2026-01-01", target_id)
            except registry.RegistryError:
                return 0.0

        checked = 0
        for segments in (True, False):
            ts = self._trace(self.BEDROCK, segments)
            a = analyze(ts, allow_unreconciled=True)
            for rule in self._rate_free_rules():
                with self.subTest(rule=rule.__name__, segments=segments):
                    f = rule(ts.analysable, a.ratios, 3.0, rate_for)
                    if f is None:
                        continue
                    checked += 1
                    for name in ("avoidable_usd_month", "avoidable_usd_window"):
                        fig = getattr(f, name)
                        self.assertFalse(
                            fig is not None and fig.raw(),
                            f"{rule.__name__} put a dollar amount on a trace "
                            f"whose rate nobody knows")
        self.assertTrue(checked, "no rate-free rule fired; nothing was checked")

    def test_the_observation_survives_where_the_figure_cannot(self):
        """The reason TTL-1 is marked at all. Its cadence claim comes from
        timestamps and its saving comes from a rate, and only the second one
        needs a price -- so a Bedrock reader whose own counters show the pattern
        should still be told about it. Before the marking it vanished, while the
        same counters produced it the moment an effective rate was supplied.
        """
        from cacheeconomics.analyzer import analyze
        ts = self._trace(self.BEDROCK, segments=False)
        bare = analyze(ts, allow_unreconciled=True)
        priced = analyze(ts, allow_unreconciled=True, effective_rate=5.0)
        self.assertIn("TTL-1", [f.code for f in bare.findings],
                      "the cadence observation needs no rate and vanished anyway")
        self.assertIn("TTL-1", [f.code for f in priced.findings])
        bare_ttl = next(f for f in bare.findings if f.code == "TTL-1")
        self.assertIsNone(bare_ttl.avoidable_usd_month,
                          "no rate, so no saving may be attached")


class TestAnAssumptionIsNeverPublishedAsAMeasurement(unittest.TestCase):
    """An input the loader assumed is not an input it read, and a figure priced
    from one must not carry the provenance of an invoice check.

    The sharpest case is the surface: assume it and the rate table and the cache
    multipliers are assumptions too, so the invoice reconciles against a guess.
    Measured before this existed: every figure in `spend`, the finding's, and
    `total_avoidable_month` all came back `released_as='reconciled'`.

    Keyed on `ts.assumed_inputs`, not on `ts.blocking_notes`. The first version
    read the notes and that was wrong three ways -- it over-blocks, because the
    litellm adapter uses the same list to mean "these rows were excluded" and a
    trace that then reconciles is correctly RECONCILED; it decides a release
    label by matching prose, which is the failure that stopped QUALIFIES_SPEND
    being the classifier in the first place; and the fact is per-input, so a
    sentence cannot say whether the surface or the rate was the assumption.
    `TestABlockingNoteAloneDoesNotRelabelACorrectReport` pins the first of those.

    DRAFT rather than withheld: the invoice did reconcile, what is assumed is
    what it reconciled *against*, so the evidence is weaker than it looks rather
    than absent. A third state spelled ASSUMED would be the precise word and
    would mean changing `money.RELEASES`, `simulate.py` and the round-trip table
    in `test_invariants.py`, two of which are outside this track.
    """

    NOTE = ("The surface was assumed to be anthropic/direct; the export names "
            "none. Rates and cache multipliers here are an assumption.")

    def _trace(self, assumed=None, blocking=False):
        reqs = [Request(request_id=f"r{i}",
                        sent_at=T0 + timedelta(hours=6 * i),
                        model="claude-opus-5", target_id="anthropic/direct",
                        tenant="t", session="s", agent="a", ttl_requested="5m",
                        usage={"input_tokens": 0, "cache_read_input_tokens": 0,
                               "cache_creation_input_tokens": 100_000},
                        segments=[])
                for i in range(12)]
        assumed = (SURFACE_ASSUMED,) if assumed is None else assumed
        ts = TraceSet(requests=reqs, tier=Tier.USAGE_ONLY,
                      notes=[self.NOTE] if (assumed or blocking) else [],
                      blocking_notes=[self.NOTE] if blocking else [])
        if assumed:
            # Set dynamically: the field belongs to `trace.py`, which this track
            # does not own, and `analyze` reads it with `getattr` for the same
            # reason. A loader that never sets it assumed nothing.
            ts.assumed_inputs = tuple(assumed)
        return ts

    def _analysis(self, assumed=None, blocking=False):
        ts = self._trace(assumed, blocking)
        invoice = analyze(ts, allow_unreconciled=True).spend["input_usd"].raw()
        return analyze(ts, invoice_usd=invoice)

    def _figures(self, a):
        from cacheeconomics.money import Figure
        out = [(f"spend[{k!r}]", v) for k, v in a.spend.items()
               if isinstance(v, Figure)]
        for i, f in enumerate(a.findings):
            for name in ("avoidable_usd_month", "avoidable_usd_window"):
                fig = getattr(f, name)
                if fig is not None:
                    out.append((f"findings[{i}].{name}", fig))
        out.append(("total_avoidable_month", a.total_avoidable_month))
        return out

    def test_the_invoice_really_does_reconcile(self):
        """Guard the guard. If the gate failed for an ordinary reason every
        figure would be withheld and this would pass without testing anything."""
        a = self._analysis()
        self.assertEqual(0.0, a.reconciliation["delta_pct"])
        self.assertTrue(a.reconciliation["within_ship_gate"])
        self.assertTrue([f for _p, f in self._figures(a) if f.released],
                        "nothing was released, so provenance is untested")

    def test_no_released_figure_claims_to_be_invoice_checked(self):
        a = self._analysis()
        from cacheeconomics import money
        wrong = [p for p, f in self._figures(a)
                 if f.released and f.released_as == money.RECONCILED]
        self.assertEqual(
            [], wrong,
            "an assumed surface published with the provenance of a "
            "measurement:\n    " + "\n    ".join(wrong))

    def test_they_are_marked_draft_rather_than_withheld(self):
        """Withholding would be the wrong correction: the invoice did reconcile,
        and throwing the figures away entirely tells a reader less than telling
        them what the figures rest on."""
        a = self._analysis()
        from cacheeconomics import money
        released = [(p, f) for p, f in self._figures(a) if f.released]
        self.assertTrue(released, "everything was withheld")
        for p, f in released:
            with self.subTest(figure=p):
                self.assertEqual(money.DRAFT, f.released_as)

    def test_the_draft_banner_names_the_assumption_and_not_the_invoice(self):
        """The banner has to say the true thing. `report._draft_reason` takes
        the *first* note beginning with "DRAFT", and its fallback sentence is
        "released without invoice reconciliation" -- which is plainly false
        here, because an invoice was supplied and it reconciled. One composed
        note, listing every reason a report is a draft."""
        a = self._analysis()
        banners = [n for n in a.notes if n.startswith("DRAFT")]
        self.assertEqual(1, len(banners),
                         "two DRAFT notes means the first one speaks for both")
        self.assertIn(SURFACE_ASSUMED, banners[0])
        self.assertIn("assumed rather than read", banners[0])
        self.assertNotIn("without invoice reconciliation", banners[0])

    def test_the_gate_itself_is_untouched(self):
        """The label changes; the gate does not. If reconciliation had failed
        everything would be withheld and the label would be moot, and flipping
        the gate here would make the two renderers disagree about a report
        neither should be publishing."""
        a = self._analysis()
        self.assertTrue(a.reconciliation["within_ship_gate"])
        self.assertTrue(a.spend["input_usd"].released)

    def test_a_trace_that_assumed_nothing_is_still_reconciled(self):
        """The other direction. Downgrading everything to DRAFT would satisfy
        every assertion above and relabel every honest figure in the product."""
        a = self._analysis(assumed=())
        from cacheeconomics import money
        released = [(p, f) for p, f in self._figures(a) if f.released]
        self.assertTrue(released)
        for p, f in released:
            with self.subTest(figure=p):
                self.assertEqual(money.RECONCILED, f.released_as)

    def test_a_library_caller_gets_the_same_answer_as_the_cli(self):
        """The property, not the patch.

        A downgrade applied inside a CLI subcommand protects only the people who
        use that subcommand. `analyze()` is the seam every caller goes through --
        the bake-off, the tier-b scripts, anyone importing the package -- and it
        is where the provenance has to be read. This calls it directly, with no
        CLI anywhere in the stack.

        Synthetic `assumed_inputs`, so it tests the analyzer half in isolation.
        `TestARealAssumedSurfaceTraceIsNotInvoiceChecked` is the one that proves
        the two halves are actually joined up, and it is the one that matters.
        """
        from cacheeconomics import money
        ts = self._trace(assumed=(SURFACE_ASSUMED,))
        invoice = analyze(ts, allow_unreconciled=True).spend["input_usd"].raw()
        a = analyze(ts, invoice_usd=invoice)
        self.assertEqual(money.DRAFT, a.spend["input_usd"].released_as)

    def test_every_figure_in_the_analysis_agrees_on_its_provenance(self):
        """Walked, not enumerated.

        `_figures` above lists the places I thought to look. The reconciliation
        dollars are built roughly 150 lines before the provenance is decided, so
        they defaulted to RECONCILED and stayed there -- a `spend` map entirely
        marked draft sitting beside `reconciliation.computed_usd` and
        `delta_usd` still marked reconciled, in one document, about the same
        dollars. Naming fields is how that survived; this walks the analysis and
        requires one answer from everything it finds.
        """
        import dataclasses
        from cacheeconomics.money import Figure

        def walk(obj, path="root", depth=0):
            if depth > 5:
                return
            if isinstance(obj, Figure):
                yield path, obj
                return
            if isinstance(obj, dict):
                for k, v in obj.items():
                    yield from walk(v, f"{path}[{k!r}]", depth + 1)
                return
            if isinstance(obj, (list, tuple)):
                for i, v in enumerate(obj):
                    yield from walk(v, f"{path}[{i}]", depth + 1)
                return
            if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
                for f in dataclasses.fields(obj):
                    yield from walk(getattr(obj, f.name), f"{path}.{f.name}",
                                    depth + 1)
            for name, attr in vars(type(obj)).items():
                if isinstance(attr, property):
                    try:
                        yield from walk(getattr(obj, name), f"{path}.{name}",
                                        depth + 1)
                    except Exception:                          # noqa: BLE001
                        continue

        a = self._analysis()
        found = [(p, f) for p, f in walk(a) if f.released]
        self.assertTrue(found, "nothing released; this would pass vacuously")
        from cacheeconomics import money
        wrong = [f"{p} = {f.released_as}" for p, f in found
                 if f.released_as != money.DRAFT]
        self.assertEqual(
            [], wrong,
            "these published invoice-checked provenance on a trace whose "
            "surface was assumed:\n    " + "\n    ".join(wrong))

    def test_which_input_was_assumed_survives_into_the_banner(self):
        """Per-input, not per-report. An assumed surface and an assumed rate are
        different assumptions with different remedies, and the reader has to be
        told which one happened -- which is the thing a single blocking-note
        string could never carry."""
        a = self._analysis(assumed=("effective rate",))
        banner = next(n for n in a.notes if n.startswith("DRAFT"))
        self.assertIn("effective rate", banner)
        self.assertNotIn(SURFACE_ASSUMED, banner)


class TestARealAssumedSurfaceTraceIsNotInvoiceChecked(unittest.TestCase):
    """End to end, through the loader that actually assumes a surface.

    Every other test of this rule attaches `assumed_inputs` to a TraceSet it
    built itself, which tests the analyzer half and nothing else. `assumed_inputs`
    does not exist on `TraceSet` and no adapter sets it, so the whole mechanism
    is currently wired to nothing: `load_sessions` records the assumption in
    `blocking_notes` only, and a real Claude Code trace with a matching invoice
    is still released as `reconciled`.

    That gap fell between two tracks -- the analyzer reads a field the loader
    never writes -- and it stayed invisible because both sides had passing tests
    for their own half. So this one goes through the real loader, with a real
    invoice, and asserts the thing a client actually gets. It is the only test
    here that would have caught the seam being unconnected.

    `trace.py` and `adapters/claude_code.py` are Track B's, so the loader half
    is theirs to land. Marked expected-failure rather than left red so CI stays
    meaningful for the other tracks; the moment the field is added and populated
    this reports "Unexpected success" and forces the marker to be deleted. The
    analyzer half it depends on is already in place and separately tested.
    """

    RECORD = {"type": "assistant", "sessionId": "s1",
              "timestamp": "2026-07-29T09:00:00Z",
              "message": {"model": "claude-opus-5",
                          "usage": {"input_tokens": 1_000_000,
                                    "cache_read_input_tokens": 0,
                                    "cache_creation_input_tokens": 0}}}

    def _root(self):
        import json as _json
        import tempfile
        root = tempfile.mkdtemp()
        proj = os.path.join(root, "proj")
        os.makedirs(proj)
        with open(os.path.join(proj, "a.jsonl"), "w") as f:
            for i in range(12):
                rec = _json.loads(_json.dumps(self.RECORD))
                rec["timestamp"] = f"2026-07-29T{9 + i:02d}:00:00Z"
                f.write(_json.dumps(rec) + "\n")
        return root

    def _loaded(self):
        from cacheeconomics.adapters.claude_code import load_sessions
        # `surface_assumed=True` because that is what the assumed path IS
        # after Track B: the adapter records the assumption only when the
        # caller says the surface was arrived at by assumption, so that
        # `--target-id anthropic/direct` stated from knowledge is not
        # mislabelled. This test is about the ASSUMED path, so it says so.
        # Before that change the note was unconditional and this call did not
        # need to.
        # Both arguments, because both are what the assumed path IS after
        # Track B: `--assume-anthropic-direct` sets the surface AND says it was
        # assumed rather than known. `target_id` now defaults to UNATTRIBUTED,
        # so passing only `surface_assumed` records an assumption of
        # "unknown/unattributed" -- true, and not the assumption this class is
        # about. Before that change the note was unconditional and the default
        # was anthropic/direct, so this call needed neither.
        return load_sessions(root=self._root(),
                             target_id="anthropic/direct",
                             surface_assumed=True)

    def test_the_loader_really_does_assume_the_surface(self):
        """Guard the guard. If the transcript named a provider there would be no
        assumption to carry and this class would prove nothing."""
        ts = self._loaded()
        self.assertTrue(ts.requests, "no requests were loaded")
        self.assertTrue(any("assumed to be anthropic/direct" in n
                            for n in ts.blocking_notes),
                        "the loader no longer records the assumption")

    def test_the_invoice_reconciles_so_provenance_is_the_only_question(self):
        ts = self._loaded()
        invoice = analyze(ts, allow_unreconciled=True).spend["input_usd"].raw()
        a = analyze(ts, invoice_usd=invoice)
        self.assertTrue(a.reconciliation["within_ship_gate"])
        self.assertTrue(a.spend["input_usd"].released)

    def test_the_figures_are_not_marked_invoice_checked(self):
        from cacheeconomics import money
        ts = self._loaded()
        invoice = analyze(ts, allow_unreconciled=True).spend["input_usd"].raw()
        a = analyze(ts, invoice_usd=invoice)
        self.assertEqual(
            money.DRAFT, a.spend["input_usd"].released_as,
            "a trace whose surface the loader assumed was published with the "
            "provenance of an invoice check")


class TestABlockingNoteAloneDoesNotRelabelACorrectReport(unittest.TestCase):
    """The over-block that keying on `ts.blocking_notes` would have caused.

    `blocking_notes` carries the QUALIFIES_SPEND phrase, and the litellm adapter
    fills it to mean "these rows were excluded from the totals". A trace that
    then reconciles is *correctly* RECONCILED: the excluded rows are excluded,
    and what remains ties to the invoice. Keying the release label on the
    presence of any blocking note would relabel every one of those reports as a
    draft -- fixing an under-block by shipping an over-block, which this project
    has done twice and paid for both times.

    So the caveat still prints, in both renderers and before any figure, and the
    provenance stays RECONCILED. Those are two different questions and this is
    the test that keeps them apart.
    """

    EXCLUSION = ("3 row(s) carry no `custom_llm_provider` and no routing prefix "
                 "on the model, so the surface is unknown and they are excluded "
                 "from every dollar figure.")

    def _analysis(self):
        reqs = [Request(request_id=f"r{i}",
                        sent_at=T0 + timedelta(hours=6 * i),
                        model="claude-opus-5", target_id="anthropic/direct",
                        tenant="t", session="s", agent="a", ttl_requested="5m",
                        usage={"input_tokens": 0, "cache_read_input_tokens": 0,
                               "cache_creation_input_tokens": 100_000},
                        segments=[])
                for i in range(12)]
        ts = TraceSet(requests=reqs, tier=Tier.USAGE_ONLY,
                      notes=[self.EXCLUSION], blocking_notes=[self.EXCLUSION])
        invoice = analyze(ts, allow_unreconciled=True).spend["input_usd"].raw()
        return analyze(ts, invoice_usd=invoice)

    def test_the_figures_stay_invoice_checked(self):
        from cacheeconomics import money
        a = self._analysis()
        self.assertTrue(a.spend["input_usd"].released)
        self.assertEqual(money.RECONCILED, a.spend["input_usd"].released_as)

    def test_and_the_report_is_not_stamped_a_draft(self):
        a = self._analysis()
        self.assertEqual([], [n for n in a.notes if n.startswith("DRAFT")])

    def test_but_the_caveat_still_travels_with_the_figures(self):
        """Withholding the label is not withholding the warning."""
        from cacheeconomics.analyzer import spend_caveats
        a = self._analysis()
        self.assertIn(self.EXCLUSION, spend_caveats(a))

    def test_a_blocking_note_reaches_the_report_even_if_notes_omits_it(self):
        """`Analysis.blocking_notes` is a filter over `notes`, so a note an
        adapter recorded as blocking and did not also put in `notes` was
        dropped -- it reached neither the report nor the caveat block. Every
        adapter puts it in both today by convention, and this is the contract."""
        from cacheeconomics.analyzer import spend_caveats
        reqs = [Request(request_id="r0", sent_at=T0, model="claude-opus-5",
                        target_id="anthropic/direct", tenant="t",
                        ttl_requested="5m",
                        usage={"input_tokens": 1_000}, segments=[])]
        ts = TraceSet(requests=reqs, tier=Tier.USAGE_ONLY,
                      notes=[], blocking_notes=["only in blocking_notes"])
        a = analyze(ts, allow_unreconciled=True)
        self.assertIn("only in blocking_notes", a.notes)
        self.assertIn("only in blocking_notes", spend_caveats(a))


class TestABaselineArmDoesNotInventABudget(unittest.TestCase):
    """litellm_auto reads the marker budget from the registry. An unknown
    surface there raised out of the whole bake-off once; it must neither raise
    nor invent a number."""

    def test_it_does_not_raise_on_an_unknown_surface(self):
        from cacheeconomics import registry
        try:
            litellm_auto(_req(model="claude-opus-5", target=UNKNOWN_TARGET))
        except registry.RegistryError:
            pass          # refusing is fine; the simulator counts it
        except Exception as e:
            self.fail(f"raised something the bake-off does not handle: {e!r}")


class TestAPartnerSurfaceIsNotPricedAtFirstPartyRates(unittest.TestCase):
    """A *known* surface the rate table does not cover.

    Every other case in this file is an unknown id, where refusing is obvious.
    This one is worse precisely because everything is known: Bedrock is in the
    registry, `claude-haiku-4-5` is priced, the multipliers are Anthropic-shaped
    -- so the report rendered a total with no missing-data note anywhere. It was
    just the wrong price list. Anthropic's pricing page is explicit that Bedrock
    and Vertex are partner-operated and invoiced by the cloud provider.

    The bar: no dollar figure without a rate from the bill that will actually
    arrive, and a reason that names the surface rather than the model.
    """

    PARTNER = ("amazon-bedrock/converse", "google-cloud/vertex")

    def _trace(self, target):
        u = {"input_tokens": 1_000_000, "cache_read_input_tokens": 0,
             "cache_creation_input_tokens": 0}
        return TraceSet(
            requests=[Request(request_id=f"r{i}", sent_at=T0 + timedelta(seconds=60 * i),
                              model="claude-haiku-4-5", target_id=target, agent="a",
                              session="s", ttl_requested="5m", usage=dict(u), segments=[])
                      for i in range(12)],
            tier=Tier.USAGE_ONLY, source="t")

    def test_no_figure_is_published_without_a_rate_from_the_bill(self):
        for target in self.PARTNER:
            a = analyze(self._trace(target), allow_unreconciled=True)
            self.assertFalse(a.spend["input_usd"].released, target)

    def test_an_invoice_rate_unblocks_it(self):
        """Refusing has to stay actionable, or the surface is simply unusable."""
        a = analyze(self._trace("amazon-bedrock/converse"),
                    allow_unreconciled=True, effective_rate=1.10)
        self.assertTrue(a.spend["input_usd"].released)
        self.assertAlmostEqual(a.spend["input_usd"].raw(), 13.20, places=2)

    def test_the_reason_names_the_surface_not_the_model(self):
        """`claude-haiku-4-5` is priced. Reporting it as unpriced sends the
        client to add a registry row that is already there."""
        a = analyze(self._trace("amazon-bedrock/converse"), allow_unreconciled=True)
        said = [n for n in a.notes if "amazon-bedrock/converse" in n]
        self.assertTrue(said, "no note named the surface")
        self.assertFalse(
            [n for n in a.notes if "No pricing is recorded for claude-haiku-4-5" in n],
            "blamed the model, which is priced")

    def test_the_stated_reason_is_the_one_that_can_still_be_acted_on(self):
        """`--allow-unreconciled` means the missing invoice was already
        accepted, so answering with it is a dead end: a client would supply an
        invoice and be told the same thing again. The fix here was never an
        invoice, it was a rate."""
        a = analyze(self._trace("amazon-bedrock/converse"), allow_unreconciled=True)
        why = a.spend["input_usd"].withheld_because
        self.assertIn("amazon-bedrock/converse", why)
        self.assertNotIn("no invoice was supplied", why)

    def test_the_anthropic_operated_surfaces_are_unaffected(self):
        """The guard must not swallow the surfaces that *do* bill at these
        rates -- over-blocking here silently drops real traffic from spend."""
        for target in ("anthropic/direct", "anthropic/claude-platform-on-aws"):
            a = analyze(self._trace(target), allow_unreconciled=True)
            self.assertTrue(a.spend["input_usd"].released, target)
            self.assertAlmostEqual(a.spend["input_usd"].raw(), 12.00, places=2)


class TestTheSurfaceFlagReachesEveryIngestPath(unittest.TestCase):
    """`--target-id` was offered on every ingest mode and honoured on one.

    The rate scope closed the wrong-surface hole in the price table, and the
    loaders reopened it one layer up: `_load` passed the operator's choice to
    `load_bodies` only, so a trace or a LiteLLM export whose rows carry no
    surface stayed on `anthropic/direct` no matter what was selected. Measured
    before the fix: 12M uncached Haiku tokens with `--target-id
    amazon-bedrock/converse` published $12.00 at Anthropic rates with no note
    naming the surface.

    A flag that is accepted and ignored is worse than one that does not exist,
    because the operator has already recorded their intent.
    """

    BEDROCK = "amazon-bedrock/converse"

    def _rows(self, extra=None):
        return [dict({"request_id": f"r{i}",
                      "sent_at": (T0 + timedelta(seconds=60 * i)).isoformat(),
                      "model": "claude-haiku-4-5", "session": "s",
                      "usage": {"input_tokens": 1_000_000,
                                "cache_read_input_tokens": 0,
                                "cache_creation_input_tokens": 0}},
                     **(extra or {})) for i in range(12)]

    def _write(self, rows):
        fh = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        for r in rows:
            fh.write(json.dumps(r) + "\n")
        fh.close()
        self.addCleanup(os.unlink, fh.name)
        return fh.name

    def test_a_normalised_trace_honours_it(self):
        from cacheeconomics.trace import load_jsonl
        ts = load_jsonl(self._write(self._rows()), None, default_target=self.BEDROCK)
        self.assertEqual({r.target_id for r in ts.requests}, {self.BEDROCK})
        self.assertFalse(analyze(ts, allow_unreconciled=True)
                         .spend["input_usd"].released)

    def test_a_row_that_names_its_surface_still_wins(self):
        """A default, never an override. Row data is evidence; a flag is a
        fallback for its absence."""
        from cacheeconomics.trace import load_jsonl
        ts = load_jsonl(self._write(self._rows({"target_id": "anthropic/direct"})),
                        None, default_target=self.BEDROCK)
        self.assertEqual({r.target_id for r in ts.requests}, {"anthropic/direct"})

    def test_a_providerless_litellm_row_honours_it(self):
        """LiteLLM reads the surface off `custom_llm_provider`. A row without
        one fell straight to anthropic/direct, which on a proxy fronting Bedrock
        priced partner traffic at first-party rates."""
        from cacheeconomics.adapters.litellm import load_litellm
        rows = [{"id": f"c{i}", "model": "claude-haiku-4-5", "status": "success",
                 "startTime": 1785000000 + i * 60,
                 "prompt_tokens": 1_000_000, "completion_tokens": 10,
                 "prompt_tokens_details": {"cached_tokens": 0}} for i in range(12)]
        ts = load_litellm(self._write(rows), default_target=self.BEDROCK)
        self.assertEqual({r.target_id for r in ts.requests}, {self.BEDROCK})

    def test_a_litellm_row_that_names_its_provider_still_wins(self):
        from cacheeconomics.adapters.litellm import load_litellm
        rows = [{"id": f"c{i}", "model": "claude-haiku-4-5", "status": "success",
                 "custom_llm_provider": "anthropic",
                 "startTime": 1785000000 + i * 60,
                 "prompt_tokens": 1000, "completion_tokens": 10,
                 "prompt_tokens_details": {"cached_tokens": 0}} for i in range(3)]
        ts = load_litellm(self._write(rows), default_target=self.BEDROCK)
        self.assertEqual({r.target_id for r in ts.requests}, {"anthropic/direct"})

    def _via_cli(self, path, source, target=None):
        """Through `_load`, not around it.

        The first version of these tests called the loaders directly with the
        new keyword, and every one of them passed with `_load` reverted to
        `load_jsonl(args.path, key)` -- while the CLI an operator actually runs
        still published $12.00 of Bedrock traffic at Anthropic rates. The defect
        was in the wiring, so the test has to cross it.
        """
        from cacheeconomics import cli
        p = argparse.ArgumentParser()
        cli._ingest_args(p)
        argv = [path, "--from", source]
        if target:
            argv += ["--target-id", target]
        return cli._load(p.parse_args(argv))

    def test_the_cli_honours_it_on_every_ingest_mode(self):
        trace = self._write(self._rows())
        ll = self._write([{"id": f"c{i}", "model": "claude-haiku-4-5",
                           "status": "success", "startTime": 1785000000 + i * 60,
                           "prompt_tokens": 1000, "completion_tokens": 1}
                          for i in range(3)])
        for path, source in ((trace, "trace"), (ll, "litellm")):
            ts = self._via_cli(path, source, self.BEDROCK)
            self.assertEqual({r.target_id for r in ts.requests}, {self.BEDROCK},
                             f"--target-id ignored for --from {source}")

    def test_silence_names_no_surface_rather_than_inventing_one(self):
        """Silence must not reach the registry as `None`, and it must not be
        answered with a surface either.

        This asserted `anthropic/direct`, which was the finding written down as
        a test. The concern behind it was real -- a `None` target reaching the
        registry -- and the answer chosen was to fabricate first-party, so a
        gateway export whose format never carried a provider earned Anthropic
        rates. Measured at $2,924/month on a twelve-request Bedrock-fronting
        capture. `UNATTRIBUTED` answers the original concern without inventing:
        it is a real string, registered as unpriceable, and the registry's own
        rate_scope carries the reason.
        """
        ts = self._via_cli(self._write(self._rows()), "trace")
        self.assertEqual({r.target_id for r in ts.requests}, {UNATTRIBUTED})
        self.assertNotIn(None, {r.target_id for r in ts.requests})

    def test_an_unnamed_surface_is_named_as_the_problem_not_the_model(self):
        """With --effective-rate, `require_priceable` is skipped and the failure
        comes out of `multipliers` instead, which raises the generic error.

        That reported "no pricing is recorded for claude-opus-5" on a trace
        whose model is priced -- the exact confusion the surface branch was
        split out to prevent, arriving through the other door. It became the
        common path the moment an unnamed row stopped being answered with
        anthropic/direct, so the reader has to be sent at the surface.
        """
        from cacheeconomics.analyzer import analyze
        ts = self._via_cli(self._write(self._rows()), "trace")
        a = analyze(ts, invoice_usd=100.0, effective_rate=3.0)
        blocking = " ".join(a.blocking_notes or [])
        self.assertIn(UNATTRIBUTED, blocking)
        self.assertNotIn("no pricing is recorded for claude", blocking.lower())
        self.assertIn("--target-id", blocking,
                      "named the problem without naming the remedy")

    def test_a_partner_surface_is_still_told_to_use_its_own_rate(self):
        """The other remedy must survive. Bedrock has recorded multipliers and
        only lacks a rate, so an effective rate from the customer's bill does
        complete it -- and telling that reader to pass --target-id instead would
        send them nowhere."""
        from cacheeconomics.analyzer import analyze
        rows = [dict(r, target_id="amazon-bedrock/converse") for r in self._rows()]
        ts = self._via_cli(self._write(rows), "trace")
        a = analyze(ts, invoice_usd=100.0)
        blocking = " ".join(a.blocking_notes or [])
        self.assertIn("effective rate", blocking)

    def test_an_unnamed_surface_earns_no_first_party_dollars(self):
        """The consequence, not just the label. A trace nobody attributed must
        not produce a priced total, because the rates on file are Anthropic
        first-party and this surface is not known to be one."""
        from cacheeconomics.analyzer import analyze
        ts = self._via_cli(self._write(self._rows()), "trace")
        a = analyze(ts, invoice_usd=100.0)
        monthly = a.spend.get("monthly_input_usd")
        self.assertFalse(monthly.raw() if hasattr(monthly, "raw") else monthly)
        self.assertTrue([n for n in (a.blocking_notes or [])
                         if UNATTRIBUTED in n],
                        "priced nothing and did not say why")

    def test_a_substituted_surface_is_said_out_loud(self):
        """Attributing traffic to a surface nobody named moves money, so it is
        reported rather than assumed."""
        from cacheeconomics.adapters.litellm import load_litellm
        rows = [{"id": "c1", "model": "claude-haiku-4-5", "status": "success",
                 "startTime": 1785000000, "prompt_tokens": 10,
                 "completion_tokens": 1}]
        ts = load_litellm(self._write(rows))
        self.assertTrue([n for n in ts.notes if "custom_llm_provider" in n])


class TestTheLivePathResolvesItsSurface(unittest.TestCase):
    """The batch loaders learned the surface; the live hook did not.

    `async_pre_call_hook` called `on_request` with no target, so it defaulted to
    anthropic/direct. On a proxy fronting Bedrock that computed minimums, TTL
    support and the breakpoint budget against the wrong provider -- and the
    budget re-check hard-coded `anthropic/direct` besides. This is the one path
    that rewrites a real request, so a wrong budget is a provider rejection
    rather than a bad report.

    Third instance of the same class in three days: cost.price ignored the
    target, then `_load` ignored `--target-id`, then this. The audit added for
    the second one was scoped to `_load`, which is why it did not catch this.
    """

    def _hook(self, target_id=None):
        from cacheeconomics import plugin
        p = plugin.CachePlugin(key=b"k" * 32, warmup=2)
        return plugin.litellm_handler(p, mutate=False, target_id=target_id)

    def _markers(self, h, model, calls=6, provider=None):
        """Markers the hook decided on, read off its own placement record.

        `mutate=False` by default, so the body is never rewritten -- the
        decision is what matters here, and reading it avoids needing the
        mutating path to test a budget guard.
        """
        import asyncio
        placed = []

        async def go():
            for i in range(calls):
                data = {"model": model, "litellm_call_id": f"c{i}",
                        "messages": [{"role": "user", "content": "policy " * 3000}]}
                if provider:
                    data["custom_llm_provider"] = provider
                await h.async_pre_call_hook(None, None, data, "completion")
            for d in h._pending.values():
                placed.append(getattr(d, "placements", None) or {})

        asyncio.run(go())
        return max((list(p) for p in placed), key=len, default=[])

    def _run(self, h, model, calls=6, provider=None):
        import asyncio
        scopes = []

        async def go():
            for i in range(calls):
                data = {"model": model, "litellm_call_id": f"c{i}",
                        "messages": [{"role": "user", "content": "policy " * 3000}]}
                if provider:
                    data["custom_llm_provider"] = provider
                await h.async_pre_call_hook(None, None, data, "completion")
            scopes.extend(d.scope for d in h._pending.values())

        asyncio.run(go())
        return {s[1] for s in scopes}

    def test_the_routing_prefix_names_the_surface(self):
        self.assertEqual(
            self._run(self._hook(), "bedrock/anthropic.claude-haiku-4-5"),
            {"amazon-bedrock/converse"})

    def test_an_explicit_provider_field_names_the_surface(self):
        self.assertEqual(
            self._run(self._hook(), "claude-haiku-4-5", provider="vertex_ai"),
            {"google-cloud/vertex"})

    def test_an_unnamed_request_resolves_to_no_surface_at_all(self):
        """This asserted `anthropic/direct`, on the reasoning that the common
        case must not move or every existing deployment silently changes
        surface. That concern is real and it lost to a larger one.

        The old comment justified the guess by saying a wrong one "surfaces as
        a provider error on the next call". Half true, and the wrong half:
        ordering a mixed request wrongly does error on Bedrock, but a minimum
        guessed too low does not — the provider processes it uncached, writes
        nothing, returns no error and bills normally. A silent wrong answer on
        somebody's production traffic, produced by us patching their request.

        Deployments that were relying on the guess now get observation plus an
        RT-UNATTRIBUTED alert naming the two ways to fix it, which is a louder
        and cheaper failure than the one it replaces.
        """
        self.assertEqual(self._run(self._hook(), "claude-haiku-4-5"),
                         {UNATTRIBUTED})

    def test_mutation_cannot_be_enabled_without_a_stated_surface(self):
        """This used to assert a per-request stand-down, because the handler
        would accept mutate=True with no surface and refuse each request.

        That was the right instinct and the wrong place. Removing the
        anthropic/direct substitution turned mutation off for the ordinary
        LiteLLM Anthropic config -- `model: claude-opus-5` with no
        `custom_llm_provider` -- and only a runtime alert said so, which is a
        silent behaviour change discovered from a bill.

        Recognising bare Claude ids as first-party was considered and rejected:
        LiteLLM's `model` is the alias the client asked for, and aliasing
        `claude-opus-5` to a Bedrock backend is ordinary configuration, so the
        id is not evidence of the surface.

        So the refusal moved to construction. One line of config at start-up,
        instead of a per-request stand-down nobody reads.
        """
        from cacheeconomics.plugin import CachePlugin, litellm_handler
        plug = CachePlugin(key=b"k" * 32, warmup=0)
        with self.assertRaises(ValueError) as caught:
            litellm_handler(plug, mutate=True)
        self.assertIn("target_id", str(caught.exception))

    def test_observation_still_needs_no_surface(self):
        """The refusal is about mutating, not about running. A team watching
        their traffic must not have to name a surface first."""
        from cacheeconomics.plugin import CachePlugin, litellm_handler
        plug = CachePlugin(key=b"k" * 32, warmup=0)
        self.assertTrue(litellm_handler(plug, mutate=False))

    def test_a_configured_handler_mutates_the_ordinary_anthropic_config(self):
        """The regression this closes: a bare `claude-*` model with no provider
        field, which is how most LiteLLM Anthropic deployments are set up."""
        import asyncio
        from cacheeconomics.plugin import CachePlugin, litellm_handler
        seen = []
        plug = CachePlugin(key=b"k" * 32, warmup=0)
        real = plug.on_request
        plug.on_request = lambda b, **kw: (
            seen.append((kw.get("target_id"), kw.get("apply"))), real(b, **kw))[1]
        h = litellm_handler(plug, mutate=True, target_id="anthropic/direct")
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(h.async_pre_call_hook(
                {}, None, {"model": "claude-haiku-4-5",
                           "messages": [{"role": "user", "content": "x" * 400}]},
                "completion"))
        finally:
            loop.close()
        self.assertEqual(seen[0], ("anthropic/direct", True))

    def test_a_row_that_names_its_own_surface_still_wins(self):
        """The configured surface is a fallback, not an override."""
        import asyncio
        from cacheeconomics.plugin import CachePlugin, litellm_handler
        seen = []
        plug = CachePlugin(key=b"k" * 32, warmup=0)
        real = plug.on_request
        plug.on_request = lambda b, **kw: (
            seen.append(kw.get("target_id")), real(b, **kw))[1]
        h = litellm_handler(plug, mutate=True, target_id="anthropic/direct")
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(h.async_pre_call_hook(
                {}, None, {"model": "claude-opus-5",
                           "custom_llm_provider": "bedrock",
                           "messages": [{"role": "user", "content": "x" * 400}]},
                "completion"))
        finally:
            loop.close()
        self.assertEqual(seen[0], "amazon-bedrock/converse")

    def test_an_operator_override_answers_when_nothing_names_it(self):
        self.assertEqual(
            self._run(self._hook("amazon-bedrock/converse"), "claude-haiku-4-5"),
            {"amazon-bedrock/converse"})

    def test_the_budget_recheck_uses_the_resolved_surface(self):
        """Behavioural, by observing which surface the registry was asked about.

        This was an AST check with two holes: assigning the surface to a
        variable first yielded an ast.Name and passed, and passing it by keyword
        left `node.args` empty so `assertNotIsInstance(None, ast.Constant)`
        passed trivially.

        The obvious replacement -- drive a Vertex request and assert the marker
        count respects Vertex's budget -- proves nothing, because every
        Anthropic-family surface records `max_breakpoints: 4`. A literal
        "anthropic/direct" produces an identical plan, and I confirmed that by
        applying the mutation and watching the value-based version pass. So the
        observable property is not the result but the question: which surface
        did the guard actually ask about.
        """
        from cacheeconomics import registry
        asked = []
        real = registry.capability

        def spy(target_id, name, *a, **kw):
            asked.append((target_id, name))
            return real(target_id, name, *a, **kw)

        registry.capability = spy
        try:
            self._run(self._hook(), "claude-haiku-4-5", provider="vertex_ai")
        finally:
            registry.capability = real

        budgets = [t for t, n in asked if n == "max_breakpoints"]
        self.assertTrue(budgets, "the budget guard never ran")
        self.assertNotIn("anthropic/direct", budgets,
                         "the live hook enforced Anthropic's budget on a Vertex "
                         "request; it must ask about the resolved surface")
        self.assertIn("google-cloud/vertex", budgets)


class TestSegmentIdsAreScopedToTheTenant(unittest.TestCase):
    """Keying an id does not hide equality -- equality is what an id is for.

    Three docstrings and two error messages claimed the HMAC key made it
    "impossible" to see that two tenants sent identical content. It never did:
    one shared key means identical bytes produce an identical id whoever sent
    them, so a reader of a multi-tenant trace could join on ids and learn two
    tenants share a policy block without holding the key at all.

    Scoping is what actually fixes it, and here the private thing and the
    correct thing coincide: caches are isolated on `(tenant, target_id, model)`,
    so two tenants provably cannot share a cache entry, and giving them one id
    was also a false statement about the cache.
    """

    KEY = b"k" * 32
    BODY = {"system": [{"type": "text", "text": "You are an assistant for ACME."}],
            "messages": [{"role": "user", "content": "hi"}]}

    def _id(self, tenant=None, key=None):
        from cacheeconomics.segment import segments_from_request
        return segments_from_request(self.BODY, key or self.KEY, tenant)[0]["id"]

    def test_two_tenants_with_identical_content_do_not_share_an_id(self):
        self.assertNotEqual(self._id("tenant-a"), self._id("tenant-b"))

    def test_the_same_tenant_still_matches_itself(self):
        """The property the analysis actually needs. Scoping must not break it,
        or every prefix reads as volatile and every finding is noise."""
        self.assertEqual(self._id("tenant-a"), self._id("tenant-a"))

    def test_an_unset_tenant_is_stable_and_is_not_the_string_none(self):
        """Single-tenant traces are the common case and must be unaffected; the
        sentinel must not collide with a tenant literally named "None"."""
        self.assertEqual(self._id(None), self._id(None))
        self.assertNotEqual(self._id(None), self._id("None"))

    def test_the_live_and_post_hoc_paths_still_agree(self):
        """The twin this module's history is built on. Both must scope ids the
        same way or an inferred trace stops matching its own recording."""
        from cacheeconomics.trace import identity_input, segment_id
        from cacheeconomics.segment import _identity_input
        block = {"type": "text", "text": "You are an assistant for ACME."}
        live = segment_id(_identity_input("system", block, block["text"], "t-1"),
                          self.KEY)
        post = segment_id(identity_input("system", "text", block["text"], "t-1"),
                          self.KEY)
        self.assertEqual(live, post)

    def test_the_loader_scopes_ids_with_the_tenant_it_was_given(self):
        """The fallback path inferred traces take. It hashed content with no
        tenant at all, so the scoping would have stopped at the loader."""
        from cacheeconomics.trace import load_jsonl

        def ids_for(tenant):
            row = {"request_id": "r1", "sent_at": T0.isoformat(),
                   "model": "claude-haiku-4-5",
                   "usage": {"input_tokens": 10},
                   "segments": [{"role": "system", "type": "text",
                                 "content": "shared policy", "tokens": 5,
                                 "index": 0}]}
            fh = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
            fh.write(json.dumps(row) + "\n")
            fh.close()
            self.addCleanup(os.unlink, fh.name)
            ts = load_jsonl(fh.name, self.KEY, default_tenant=tenant)
            return ts.requests[0].segments[0].id

        self.assertNotEqual(ids_for("tenant-a"), ids_for("tenant-b"))

    def test_the_body_loader_scopes_ids_with_the_row_tenant(self):
        """The tenant is resolved twice per row -- once to hash the ids, once to
        build the Request -- and the two consumers live in different modules.
        They diverged instantly: `load_bodies` hashed with the *caller's* tenant
        while `request_from_row` resolved the *row's*, so an export whose rows
        each name their own tenant got correct `Request.tenant` values over ids
        computed as if there were no tenant at all. Same leak, reintroduced by
        the commit that closed it."""
        from cacheeconomics.adapters.bodies import load_bodies
        body = {"model": "claude-haiku-4-5",
                "system": [{"type": "text", "text": "shared policy block"}],
                "messages": [{"role": "user", "content": "hi"}]}
        rows = [{"request_id": "r1", "tenant": "tenant-a",
                 "sent_at": T0.isoformat(), "request": body,
                 "response": {"usage": {"input_tokens": 100}}},
                {"request_id": "r2", "tenant": "tenant-b",
                 "sent_at": T0.isoformat(), "request": body,
                 "response": {"usage": {"input_tokens": 100}}}]
        fh = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        for r in rows:
            fh.write(json.dumps(r) + "\n")
        fh.close()
        self.addCleanup(os.unlink, fh.name)
        # No caller-level tenant on purpose: the rows carry their own, which is
        # the case that broke.
        ts = load_bodies(fh.name, self.KEY)
        a, b = ts.requests
        self.assertEqual((a.tenant, b.tenant), ("tenant-a", "tenant-b"))
        self.assertNotEqual(a.segments[0].id, b.segments[0].id)

    def test_one_resolver_decides_the_tenant(self):
        """Structural, because the behavioural test above only covers the two
        consumers that exist today. The rule is trivial and was still
        implemented twice; a third copy is how it drifts again."""
        import ast
        import inspect

        from cacheeconomics import trace
        src = inspect.getsource(trace)
        # The precedence expression itself, spelled out, should appear once --
        # inside `resolve_tenant` and nowhere else.
        self.assertEqual(
            src.count('_first(row, "tenant", "userId")'), 1,
            "the tenant precedence rule is written more than once")
        self.assertTrue(callable(getattr(trace, "resolve_tenant", None)))
        tree = ast.parse(src)
        fn = [n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "resolve_tenant"]
        self.assertEqual(len(fn), 1)

    def test_no_module_still_claims_keying_hides_cross_tenant_equality(self):
        """The claim was wrong in five places at once, which is why it is
        asserted structurally rather than fixed one docstring at a time."""
        import cacheeconomics
        root = os.path.dirname(os.path.abspath(cacheeconomics.__file__))
        bad = []
        for dirpath, _, names in os.walk(root):
            for n in names:
                if not n.endswith(".py"):
                    continue
                p = os.path.join(dirpath, n)
                with open(p) as fh:
                    text = fh.read()
                for phrase in ("two tenants sent identical content",
                               "two tenants sharing content",
                               "makes both impossible"):
                    if phrase in text:
                        bad.append(f"{n}: {phrase}")
        self.assertEqual(bad, [], "a module still overclaims what the key does")


if __name__ == "__main__":
    unittest.main()


class TestAnUnstatedSurfaceIsNotAnthropic(unittest.TestCase):
    """A LiteLLM row with no provider field used to be priced as Anthropic.

    The rate scope is default-deny and refuses to price a partner surface, but
    it can only refuse a surface it is shown. `target_from_row` handed it
    `anthropic/direct` manufactured out of an absence, so a proxy export
    fronting Bedrock priced at first-party list and published a
    reconciled-looking total no AWS bill would match.

    That is the same defect the scope was built for, entering through the one
    door the scope could not see. The guard was checking the answer rather than
    where the answer came from.
    """

    def _rows(self, tmp, **extra):
        p = os.path.join(tmp, "x.jsonl")
        with open(p, "w") as f:
            for i in range(30):
                row = {"id": f"r{i}", "startTime": 1_780_000_000 + i * 60,
                       "model": "claude-opus-5",
                       "response": {"usage": {
                           "prompt_tokens": 100000, "completion_tokens": 10,
                           "prompt_tokens_details": {"cached_tokens": 0},
                           "cache_creation_input_tokens": 0,
                           "cache_read_input_tokens": 0}}}
                row.update(extra)
                f.write(json.dumps(row) + "\n")
        return p

    def test_no_provider_metadata_is_not_attributed_to_anthropic(self):
        from cacheeconomics.adapters.litellm import load_litellm
        from cacheeconomics.registry import UNATTRIBUTED
        with tempfile.TemporaryDirectory() as tmp:
            ts = load_litellm(self._rows(tmp))
            self.assertEqual({r.target_id for r in ts.requests}, {UNATTRIBUTED})

    def test_and_therefore_publishes_no_dollar_figure(self):
        from cacheeconomics.adapters.litellm import load_litellm
        with tempfile.TemporaryDirectory() as tmp:
            a = analyze(load_litellm(self._rows(tmp)), allow_unreconciled=True)
            self.assertFalse(a.spend["input_usd"].released)

    def test_the_reason_names_the_right_remedy(self):
        """"Supply the rate from that bill" is the partner-surface remedy and
        the wrong instruction here: this reader needs to state the surface, and
        may well be on Anthropic direct."""
        from cacheeconomics.adapters.litellm import load_litellm
        with tempfile.TemporaryDirectory() as tmp:
            a = analyze(load_litellm(self._rows(tmp)), allow_unreconciled=True)
            why = a.spend["input_usd"].withheld_because
            self.assertIn("--target-id", why)
            self.assertNotIn("cloud provider", why)

    def test_stating_the_surface_restores_the_figure(self):
        """The refusal has to be escapable, or every LiteLLM export that simply
        omits the field becomes unanalysable."""
        from cacheeconomics.adapters.litellm import load_litellm
        with tempfile.TemporaryDirectory() as tmp:
            ts = load_litellm(self._rows(tmp), default_target="anthropic/direct")
            a = analyze(ts, allow_unreconciled=True)
            self.assertTrue(a.spend["input_usd"].released)

    def test_an_explicit_provider_is_still_honoured(self):
        from cacheeconomics.adapters.litellm import load_litellm
        with tempfile.TemporaryDirectory() as tmp:
            ts = load_litellm(self._rows(tmp, custom_llm_provider="bedrock"))
            self.assertNotIn("anthropic/direct", {r.target_id for r in ts.requests})

    def test_the_reason_reaches_the_default_report(self):
        """A blocker folded behind --detail is a blocker the reader never sees."""
        from cacheeconomics.adapters.litellm import load_litellm
        from cacheeconomics.report import render_text
        with tempfile.TemporaryDirectory() as tmp:
            a = analyze(load_litellm(self._rows(tmp)), allow_unreconciled=True)
            flat = " ".join(render_text(a).split())
            self.assertIn("surface is unknown", flat)
            self.assertIn("--target-id", flat)


class TestCountedIsMeasuredInMoneyNotRows(unittest.TestCase):
    """`tokens_counted` was a row count. Coverage learned this lesson already.

    Ninety-nine tiny counted rows beside one huge uncounted one is 99% of rows,
    clears the 0.99 publish threshold, and is 0.02% of the billed tokens. Every
    structural dollar figure would then rest on a byte-share split covering
    essentially all of the spend -- and that split is 19.2% off at the median,
    which is the entire reason the threshold exists.

    `structural_coverage_billed` is weighted for exactly this reason. This
    counter was not.
    """

    def _rows(self, tmp, big_counted):
        def row(i, tok, counted):
            return {"request_id": f"r{i}", "sent_at": f"2026-07-29T09:{i % 60:02d}:00Z",
                    "model": "claude-opus-5", "target_id": "anthropic/direct",
                    "session": "s", "ttl_requested": "5m", "tokens_counted": counted,
                    "usage": {"input_tokens": tok, "cache_read_input_tokens": 0,
                              "cache_creation_input_tokens": 0},
                    "segments": [{"id": f"s{i}a", "role": "system", "tokens": tok // 2,
                                  "index": 0, "cache_marked": True, "ttl": "5m"},
                                 {"id": f"s{i}b", "role": "user", "tokens": tok // 2,
                                  "index": 1}]}
        p = os.path.join(tmp, "x.jsonl")
        with open(p, "w") as f:
            for i in range(99):
                f.write(json.dumps(row(i, 100, True)) + "\n")
            f.write(json.dumps(row(99, 50_000_000, big_counted)) + "\n")
        return p

    def test_one_huge_uncounted_row_defeats_ninety_nine_tiny_counted_ones(self):
        from cacheeconomics.trace import load_jsonl
        with tempfile.TemporaryDirectory() as tmp:
            ts = load_jsonl(self._rows(tmp, big_counted=False), b"k" * 32)
            self.assertLess(ts.tokens_counted, 0.01,
                            "still weighted by rows, so 99/100 passed")
            self.assertFalse(ts.tokens_are_counted)

    def test_counting_the_row_that_holds_the_money_is_enough(self):
        """The gate has to be passable, or counting stops being worth doing."""
        from cacheeconomics.trace import load_jsonl
        with tempfile.TemporaryDirectory() as tmp:
            ts = load_jsonl(self._rows(tmp, big_counted=True), b"k" * 32)
            self.assertGreater(ts.tokens_counted, 0.99)
            self.assertTrue(ts.tokens_are_counted)

    def test_a_hostile_usage_field_does_not_crash_the_weighting(self):
        """`usage` can be a string. `_billed_input` calls `.get` on it."""
        from cacheeconomics.trace import load_jsonl
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "x.jsonl")
            with open(p, "w") as f:
                f.write(json.dumps({"request_id": "r", "model": "claude-opus-5",
                                    "usage": "not-a-dict",
                                    "segments": [{"id": "a", "role": "system",
                                                  "tokens": 10, "index": 0}]}) + "\n")
            load_jsonl(p, b"k" * 32)          # must not raise


class TestABodyExportStatesNoSurface(unittest.TestCase):
    """A logged request body proves the API shape, not who invoices it.

    Langfuse, Helicone and a LiteLLM proxy all log Anthropic-shaped bodies in
    front of Bedrock and Vertex. `load_bodies` defaulted to anthropic/direct,
    which is the fabricated-surface shape the LiteLLM adapter was already fixed
    for: default-deny can only refuse a surface it is shown.
    """

    def _export(self, tmp):
        p = os.path.join(tmp, "b.jsonl")
        body = {"model": "claude-opus-5",
                "system": [{"type": "text", "text": "x" * 4000,
                            "cache_control": {"type": "ephemeral"}}],
                "messages": [{"role": "user", "content": "hi"}]}
        with open(p, "w") as f:
            for i in range(30):
                f.write(json.dumps({
                    "sent_at": f"2026-07-29T09:{i % 60:02d}:00Z", "body": body,
                    "usage": {"input_tokens": 200_000,
                              "cache_read_input_tokens": 0,
                              "cache_creation_input_tokens": 0}}) + "\n")
        return p

    def test_the_default_is_unattributed(self):
        from cacheeconomics.adapters.bodies import load_bodies
        from cacheeconomics.registry import UNATTRIBUTED
        with tempfile.TemporaryDirectory() as tmp:
            ts = load_bodies(self._export(tmp), b"k" * 32)
            self.assertEqual({r.target_id for r in ts.requests}, {UNATTRIBUTED})

    def test_and_publishes_no_dollars(self):
        from cacheeconomics.adapters.bodies import load_bodies
        with tempfile.TemporaryDirectory() as tmp:
            a = analyze(load_bodies(self._export(tmp), b"k" * 32),
                        allow_unreconciled=True)
            self.assertFalse(a.spend["input_usd"].released)

    def test_stating_the_surface_restores_them(self):
        from cacheeconomics.adapters.bodies import load_bodies
        with tempfile.TemporaryDirectory() as tmp:
            a = analyze(load_bodies(self._export(tmp), b"k" * 32,
                                    target_id="anthropic/direct"),
                        allow_unreconciled=True)
            self.assertTrue(a.spend["input_usd"].released)

    def test_the_cli_does_not_inject_a_default_behind_the_adapter(self):
        """The CLI passed DEFAULT_TARGET on this path, defeating the adapter's
        refusal entirely.

        Behavioural, not a source slice. The earlier version read
        `inspect.getsource(cli._load)` and checked the text of the call, which
        `args.target_id = args.target_id or DEFAULT_TARGET` one line above it
        passes cleanly while reintroducing the whole bug. Verified: that
        mutation passed the old assertions.
        """
        from cacheeconomics import cli
        with tempfile.TemporaryDirectory() as tmp:
            path = self._export(tmp)
            argv = ["analyze", path, "--from", "bodies",
                    "--allow-unreconciled", "--format", "json"]
            key = os.environ.get("CACHEECONOMICS_HMAC_KEY")
            os.environ["CACHEECONOMICS_HMAC_KEY"] = "k" * 32
            try:
                args = cli.build_parser().parse_args(argv)
                ts = cli._load(args)
            finally:
                if key is None:
                    os.environ.pop("CACHEECONOMICS_HMAC_KEY", None)
                else:
                    os.environ["CACHEECONOMICS_HMAC_KEY"] = key
        from cacheeconomics.registry import UNATTRIBUTED
        self.assertEqual({r.target_id for r in ts.requests}, {UNATTRIBUTED},
                         "the CLI re-fabricates the surface the adapter refused")
        self.assertFalse(analyze(ts, allow_unreconciled=True)
                         .spend["input_usd"].released)


class TestClaudeCodeSaysTheSurfaceIsAssumed(unittest.TestCase):
    """Transcripts carry no provider field anywhere -- checked across 190 of
    them. The surface is an adapter assumption, kept because Claude Code talks
    to Anthropic unless routed, but stated rather than buried."""

    def _fixture(self, tmp):
        """A transcript on disk, so these run in CI.

        They used to call `load_sessions()` against the developer's own
        `~/.claude/projects` and skip on any exception or empty result. On a
        clean checkout that is an unconditional skip, so a mutation removing
        the surface blocker would never have been tested at all -- on the one
        path that decides which rate table applies.
        """
        import json
        proj = os.path.join(tmp, "proj")
        os.makedirs(proj)
        path = os.path.join(proj, "session.jsonl")
        with open(path, "w") as f:
            for i in range(12):
                f.write(json.dumps({
                    "type": "assistant", "sessionId": "s1",
                    "uuid": f"u{i}", "requestId": f"r{i}",
                    "timestamp": f"2026-07-29T09:{i:02d}:00.000Z",
                    "message": {"model": "claude-opus-5", "usage": {
                        "input_tokens": 100, "output_tokens": 10,
                        "cache_read_input_tokens": 20_000,
                        "cache_creation_input_tokens": 1_000}}}) + "\n")
        return tmp

    def test_the_assumption_is_visible_without_detail(self):
        from cacheeconomics.adapters.claude_code import load_sessions
        from cacheeconomics.report import render_text
        with tempfile.TemporaryDirectory() as tmp:
            # `surface_assumed=True` is what makes this an assumption to
            # disclose, and it is passed rather than inferred from the surface
            # id. The note used to key on `target_id == "anthropic/direct"` --
            # the same string whether somebody assumed it or knew it -- so a
            # caller who stated the surface was told it had been guessed.
            ts = load_sessions(root=self._fixture(tmp),
                               target_id="anthropic/direct",
                               surface_assumed=True)
        self.assertTrue(ts.requests, "fixture produced no requests")
        self.assertTrue(any("surface assumed" in n for n in ts.blocking_notes))
        flat = " ".join(render_text(analyze(ts, allow_unreconciled=True)).split())
        self.assertIn("surface assumed", flat)
        self.assertIn("--target-id", flat)

    def test_an_explicit_surface_replaces_it(self):
        from cacheeconomics.adapters.claude_code import load_sessions
        with tempfile.TemporaryDirectory() as tmp:
            ts = load_sessions(root=self._fixture(tmp),
                               target_id="amazon-bedrock/converse")
        self.assertTrue(ts.requests, "fixture produced no requests")
        self.assertEqual({r.target_id for r in ts.requests},
                         {"amazon-bedrock/converse"})
        self.assertFalse(any("surface assumed" in n for n in ts.blocking_notes))


class TestTheFigureGateHasNoSideDoors(unittest.TestCase):
    """`Figure` was a frozen dataclass, so every generic helper walked past it.

    `asdict` and `vars` both returned `{'_usd': 123.45, ...}`.
    `json.dumps(f, default=lambda o: o.__dict__)` -- the most common serializer
    default anyone writes -- published the number. And
    `dataclasses.replace(f, released=True)` turned a withheld figure into "$123"
    in one line, with no `raw()` at the call site to grep for.
    """

    def _withheld(self):
        from cacheeconomics.money import MEASURED, Figure
        return Figure(123.45, MEASURED, released=False,
                      withheld_because="not reconciled")

    def test_it_is_not_walkable_as_a_dataclass(self):
        import dataclasses
        f = self._withheld()
        with self.assertRaises(TypeError):
            dataclasses.asdict(f)
        with self.assertRaises(TypeError):
            dataclasses.replace(f, released=True)

    def test_it_carries_no_instance_dict(self):
        """`vars` and the common json default both reach through `__dict__`."""
        import json
        f = self._withheld()
        with self.assertRaises(TypeError):
            vars(f)
        with self.assertRaises(AttributeError):
            json.dumps(f, default=lambda o: o.__dict__)

    def test_release_state_cannot_be_set_in_place(self):
        f = self._withheld()
        with self.assertRaises(AttributeError):
            f.released = True
        self.assertIn("withheld", str(f))

    def test_pickling_does_not_round_trip_it_released(self):
        import pickle
        f = self._withheld()
        self.assertIn("withheld", str(pickle.loads(pickle.dumps(f))))

    def test_the_claim_is_the_narrow_one(self):
        """`_usd` is reachable and this file says so on purpose.

        A review found the first version of this class claiming `raw()` was the
        only route to the number. It is not: the attribute, the slot descriptor
        and `__reduce_ex__` all reach it. The property that holds is that no
        *generic* route does, so anything reaching it has to name it.
        """
        f = self._withheld()
        self.assertEqual(f._usd, 123.45)
        self.assertEqual(type(f)._usd.__get__(f, type(f)), 123.45)
        import inspect

        from cacheeconomics import money
        doc = inspect.getdoc(money.Figure)
        self.assertIn("not the only way", doc.replace("only way to reach",
                                                      "not the only way"),
                      "the docstring overclaims again")

    def test_the_deliberate_paths_still_work(self):
        """A gate with no legitimate exit is a gate nobody can use."""
        f = self._withheld()
        self.assertEqual(f.raw(), 123.45)
        self.assertIn("$123", str(f.release(True)))
        self.assertIn("withheld", str(abs(f)))
        self.assertEqual(abs(self._withheld().release(True)).raw(), 123.45)


class TestTheRateLookupIsScoped(unittest.TestCase):
    """`base_rate` took only model and date, so the surface was erased before
    default-deny could see it.

    `cost.price` happened to call `require_priceable` first. Nothing made every
    other caller do the same, and the analyzer's own `rate_for` closure did
    not -- so a finding could price a Bedrock request at Anthropic list while
    `cost.price` correctly refused the same row.
    """

    def test_a_partner_surface_is_refused_at_the_rate_lookup(self):
        from cacheeconomics import registry
        with self.assertRaises(registry.UnpriceableSurface):
            registry.base_rate("claude-sonnet-4-6", "2026-08-01",
                               "amazon-bedrock/converse")

    def test_a_first_party_surface_still_prices(self):
        from cacheeconomics import registry
        self.assertGreater(
            registry.base_rate("claude-sonnet-4-6", "2026-08-01",
                               "anthropic/direct"), 0)

    def test_a_rate_cannot_be_asked_for_without_naming_a_surface(self):
        """This asserted the opposite, and rationalised it: "defaulting to
        anthropic/direct is fine only because `require_priceable` then checks
        it rather than trusting it."

        It does check it — and it checks the *default*, which is a priceable
        surface, so the check always passes. It confirms the fabrication rather
        than the caller. Measured before this changed:
        `cost.price(usage, "claude-opus-5")` with no surface returned $5.00 at
        Anthropic list, silently.

        A test asserting that omitted-surface pricing succeeds is the defect
        written down, and it would have green-lit any future caller that
        reintroduced wrong-surface dollars.
        """
        from cacheeconomics import cost, registry
        with self.assertRaises(TypeError):
            registry.base_rate("claude-sonnet-4-6", "2026-08-01")
        with self.assertRaises(TypeError):
            cost.price(cost.Usage(uncached_input=1_000), "claude-opus-5")

    def test_a_named_surface_still_prices_normally(self):
        from cacheeconomics import cost, registry
        self.assertGreater(
            registry.base_rate("claude-sonnet-4-6", "2026-08-01",
                               "anthropic/direct"), 0)
        self.assertGreater(
            cost.price(cost.Usage(uncached_input=1_000), "claude-opus-5",
                       "anthropic/direct", on_date="2026-07-29").usd, 0)

    def test_the_rate_lookup_uses_the_surface_it_was_given(self):
        """`cost.price` dropped `target_id` when calling `base_rate`, so the
        scope check ran against one surface and the rate came from another. It
        was harmless only because the check above already refused unpriceable
        surfaces — a guard holding up a bug, not an absence of one."""
        import inspect
        from cacheeconomics import cost
        src = inspect.getsource(cost.price)
        self.assertIn("base_rate(model, on_date, target_id)", src)


class TestTheRatePathNamesItsSurface(unittest.TestCase):
    """`rate_for` had `target_id="anthropic/direct"` as a default and not one
    of its six call sites passed a surface.

    Three adapters were fixed the same day for inferring anthropic/direct from
    a missing surface. Putting a default here reintroduced that one layer down,
    where the rate actually comes from, and made the `base_rate` scoping added
    alongside it cosmetic -- every caller silently claimed first-party.
    """

    def test_rate_for_refuses_to_guess(self):
        import inspect

        from cacheeconomics import analyzer
        src = inspect.getsource(analyzer.analyze)
        self.assertIn("def rate_for(model, when=None, target_id=None)", src)
        self.assertIn("rate_for needs the request's target_id", src)

    def test_every_call_site_passes_the_requests_own_surface(self):
        """Behaviourally impossible to skip now -- it raises -- but pinned so a
        future call site cannot reintroduce a default by adding one back."""
        import inspect
        import re

        from cacheeconomics import analyzer
        src = inspect.getsource(analyzer)
        # Balanced to the closing paren, not to the first one. `[^)]*` stopped
        # inside `_when(r)` and reported a real call site as argument-less --
        # a test that failed for its own reason rather than the code's.
        calls = []
        for m in re.finditer(r"rate_for\(", src):
            i, depth = m.end(), 1
            while i < len(src) and depth:
                depth += (src[i] == "(") - (src[i] == ")")
                i += 1
            calls.append(src[m.end():i - 1])
        calls = [c for c in calls if "when=None" not in c]
        self.assertTrue(calls, "no call sites found; the pattern moved")
        for c in calls:
            self.assertIn("target_id", c, f"rate_for({c}) does not name a surface")

    def test_a_partner_surface_request_does_not_price_at_list(self):
        from cacheeconomics import registry
        with self.assertRaises(registry.UnpriceableSurface):
            registry.base_rate("claude-sonnet-4-6", "2026-08-01",
                               "amazon-bedrock/converse")


class TestThePricingPrimitivesRefuseToInvent(unittest.TestCase):
    """Four defects in the layer every dollar figure derives from."""

    def test_an_explicit_zero_scalar_contradicting_a_split_is_refused(self):
        """`created = usage.get(...) or 0` could not express "explicitly zero",
        and the disagreement check was conditional on that value being truthy.
        So "0 written" alongside a split claiming 1,000 priced the 1,000 --
        while the mirror case raised. The guard failed loudly in one direction
        and invented write spend in the other."""
        from cacheeconomics import cost
        with self.assertRaises(ValueError):
            cost.Usage.from_anthropic({
                "input_tokens": 10, "cache_creation_input_tokens": 0,
                "cache_creation": {"ephemeral_5m_input_tokens": 1000}})

    def test_an_absent_scalar_with_a_split_is_still_priced(self):
        """Absent is not zero. The split is authoritative when nothing
        contradicts it, and this must not become a blanket refusal."""
        from cacheeconomics import cost
        u = cost.Usage.from_anthropic({
            "input_tokens": 10,
            "cache_creation": {"ephemeral_5m_input_tokens": 1000}})
        self.assertEqual(u.cache_write_5m, 1000)

    def test_a_falsy_date_is_refused_rather_than_replaced_with_today(self):
        """`on_date or now()` let "" and 0 past the strict parser, took today's
        rate, and then stamped rate_source with the substituted date -- so the
        report claimed an effective date the caller never gave."""
        from cacheeconomics import cost, registry
        with self.assertRaises(registry.RegistryError):
            cost.price(cost.Usage(uncached_input=1_000_000), "claude-sonnet-5", "anthropic/direct",
                       on_date="")

    def test_no_date_at_all_still_means_today(self):
        from cacheeconomics import cost
        got = cost.price(cost.Usage(uncached_input=1_000_000), "claude-sonnet-5", "anthropic/direct")
        self.assertIn("effective", got.breakdown["rate_source"])

    def test_upcoming_rate_change_parses_its_dates(self):
        """The one date selector that compared raw strings. "2026-8-1" sorts
        above "2026-09-01" at the month digit, hiding a change a month away."""
        from cacheeconomics import registry
        for bad in ("2026-8-1", "not-a-date", 20260801):
            with self.subTest(value=bad):
                with self.assertRaises(registry.RegistryError):
                    registry.upcoming_rate_change("claude-sonnet-5", bad)

    def test_a_well_formed_date_still_finds_the_change(self):
        from cacheeconomics import registry
        got = registry.upcoming_rate_change("claude-sonnet-5", "2026-08-01")
        self.assertEqual(got["effective"], "2026-09-01")

    def test_counted_zeros_survive(self):
        """`max(1, ...)` put back the invented token that `_scale_to_measured`
        explicitly refuses to invent. Counting is the exact path; clamping its
        answer upward skews every other segment's share of the billed input."""
        from cacheeconomics.tokenizer import apply_counts
        segs = [{"bytes": 0}, {"bytes": 0}, {"bytes": 0}]
        self.assertEqual([s["bytes"] for s in apply_counts(segs, [0, 0, 1000])],
                         [0, 0, 1000])


class TestSilentDegradationIsSpokenAloud(unittest.TestCase):
    """Three places the tool answered confidently without saying it had stopped
    looking, or had done less than the answer implies.

    None of these produced a wrong number. Each produced a right-shaped one: a
    green tick, a searched plan, a note about a limit. That is worse, because
    there is nothing for a reader to notice.
    """

    def test_zero_markers_is_not_a_passing_budget(self):
        """`0 of 4 markers used` rendered as PASS on a surface where markers
        are the only lever there is. Under budget, and nothing cached."""
        from cacheeconomics import checks
        r = checks.check_breakpoint_budget(0, "anthropic/direct")
        self.assertEqual(r.status, checks.Status.ABSTAIN)
        self.assertNotEqual(r.status, checks.Status.PASS)

    def test_a_real_budget_still_passes(self):
        """The other direction: abstaining on zero must not swallow the case
        the check is for."""
        from cacheeconomics import checks
        self.assertEqual(checks.check_breakpoint_budget(1, "anthropic/direct").status,
                         checks.Status.PASS)
        self.assertEqual(checks.check_breakpoint_budget(4, "anthropic/direct").status,
                         checks.Status.PASS)
        self.assertEqual(checks.check_breakpoint_budget(5, "anthropic/direct").status,
                         checks.Status.FAIL)

    def _alloc(self, n):
        from cacheeconomics import tiers
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from test_allocator_full import sg
        segs = [sg(i, "system", 4000, f"s{i}") for i in range(n)]
        return tiers.allocate(segs, {i: 0.0 for i in range(n)},
                              target_id="anthropic/direct", model="claude-opus-5",
                              gaps=[600.0] * 10)

    def test_an_unsearched_plan_says_it_was_unsearched(self):
        """Above the bound the exhaustive search returns nothing and the caller
        recorded nothing, so a nine-segment prompt got an answer shaped exactly
        like a searched one -- while the function's docstring said "the caller
        says so"."""
        from cacheeconomics import tiers
        a = self._alloc(tiers.MIXED_EXHAUSTIVE_MAX_SEGMENTS + 1)
        self.assertTrue([n for n in a.notes if "did not run" in n],
                        "a skipped search is indistinguishable from a completed one")
        self.assertIn(("mixed-exhaustive", None), a.searched)

    def test_a_searched_plan_does_not_claim_it_was_skipped(self):
        from cacheeconomics import tiers
        a = self._alloc(tiers.MIXED_EXHAUSTIVE_MAX_SEGMENTS - 2)
        self.assertFalse([n for n in a.notes if "did not run" in n])
        self.assertNotIn(("mixed-exhaustive", None), a.searched)

    def _litellm_plan(self, already, points):
        from cacheeconomics import allocate
        from cacheeconomics.trace import Request, Segment
        segs = [Segment(id=f"s{i}", role="system", tokens=2_000, index=i,
                        cache_marked=(i < already),
                        ttl="5m" if i < already else None)
                for i in range(8)]
        r = Request(request_id="r", sent_at=datetime(2026, 7, 29, 9, tzinfo=timezone.utc),
                    model="claude-opus-5", target_id="anthropic/direct",
                    usage={"input_tokens": 16_000}, segments=segs, session="s")
        return allocate.litellm_auto(r, injection_points=points)

    def test_a_full_budget_does_not_get_one_more(self):
        """The marker was written and *then* the limit checked, so a caller
        arriving with a full budget left with one over -- and the note said it
        "stopped at the limit" on the exact run that exceeded it."""
        plan = self._litellm_plan(4, [{"index": 4}, {"index": 5}])
        self.assertLessEqual(len(plan.ttls), 4)
        self.assertTrue([n for n in plan.notes if "stopped at the" in n])

    def test_and_it_still_injects_when_there_is_room(self):
        plan = self._litellm_plan(1, [{"index": 4}, {"index": 5}])
        self.assertEqual(len(plan.ttls), 3)
        self.assertFalse([n for n in plan.notes if "stopped at the" in n],
                         "claimed a limit it never reached")


class TestToolHistoryIsNeverAnAllocationTarget(unittest.TestCase):
    """`segment._mark` refuses to write a cache_control onto tool history, and
    that refusal was reachable as a crash.

    `_filter_near_minimum` excluded unmarkable positions only when the caller
    passed `markable`, and `CachePlugin.on_request` leaves it None. A direct
    caller with a large stable `tool_calls` block had it chosen, handed to
    `apply_markers`, and got ValueError out of the public API -- with
    `apply=False` as well, and *before* `observe_shape`, so the LiteLLM
    fail-open swallowed it and the request went unobserved too.

    The fixture matters: the tool block is large and perfectly stable, so the
    allocator wants it. A small or churning one would make this test pass with
    the fix reverted, which is how three tests in this session passed for the
    wrong reason.
    """

    @staticmethod
    def _body(i):
        return {"model": "claude-opus-5", "messages": [
            {"role": "system", "content": "policy " * 4000},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "call_stable", "type": "function",
                             "function": {"name": "f", "arguments": "x" * 20000}}]},
            {"role": "user", "content": f"turn {i}"}]}

    def _drive(self, apply):
        p = plugin.CachePlugin(key=b"k" * 32, warmup=4)
        last = None
        for i in range(20):
            _, last = p.on_request(self._body(i), model="claude-opus-5",
                                   target_id="anthropic/direct",
                                   at=T0 + timedelta(seconds=90 * i), apply=apply)
        return p, last

    def test_observe_only_does_not_raise(self):
        _, last = self._drive(apply=False)
        self.assertIsNotNone(last)

    def test_mutating_does_not_raise(self):
        _, last = self._drive(apply=True)
        self.assertIsNotNone(last)

    def test_no_marker_lands_on_tool_history(self):
        from cacheeconomics.segment import walk
        _, last = self._drive(apply=True)
        named = {i for i, (_r, _l, _b, path) in enumerate(walk(self._body(0)))
                 if len(path) > 2 and isinstance(path[2], str)}
        self.assertTrue(named, "fixture no longer contains tool history")
        self.assertFalse(set(last.placements) & named)

    def test_the_abstention_says_which_position_and_why(self):
        _, last = self._drive(apply=True)
        self.assertTrue([n for n in last.notes if "tool history" in n],
                        "abstained silently, which reads as nothing being wrong")

    def test_it_still_places_where_it_can(self):
        """Excluding tool history must not become placing nothing."""
        _, last = self._drive(apply=True)
        self.assertTrue(last.placements, "stood down entirely instead of skipping one position")


class TestAnInvoiceDoesNotUnderwriteAProjection(unittest.TestCase):
    """Reconciliation proves the *measured* subtotal matches the bill. It says
    nothing about whether the window it covers resembles a month.

    The gate treated those as one claim, so two requests one second apart with a
    matching $1.00 invoice published $720.00/month as a reconciled figure --
    `_window_days` floors at an hour and `release_map` released everything
    together. The most authoritative-looking number in the report was the one
    with the least evidence behind it.
    """

    def _analyse(self, n, span_seconds):
        from cacheeconomics.analyzer import analyze
        reqs = [Request(request_id=f"r{i}",
                        sent_at=T0 + timedelta(seconds=span_seconds * i / max(n - 1, 1)),
                        model="claude-opus-5", target_id="anthropic/direct",
                        ttl_requested="5m", session="s",
                        usage={"input_tokens": 100_000},
                        segments=[Segment(id="s", role="system", tokens=100_000,
                                          index=0)])
                for i in range(n)]
        invoice = n * 100_000 * 5.0 / 1e6          # exact list price, reconciles
        return analyze(TraceSet(requests=reqs, tier=Tier.INSTRUMENTED,
                                source="projection"), invoice_usd=invoice)

    DAY = 86_400

    def test_a_one_second_window_publishes_no_monthly_figure(self):
        a = self._analyse(2, 1)
        self.assertFalse(a.spend["monthly_input_usd"].released)

    def test_but_the_measured_spend_still_publishes(self):
        """The invoice does establish this one, and withholding it would throw
        away the thing reconciliation actually proves."""
        a = self._analyse(2, 1)
        self.assertTrue(a.spend["input_usd"].released)

    def test_too_few_requests_also_withholds_it(self):
        """A long window with two requests is not a sample either: one request
        moves the projected total more than the rest of the trace."""
        a = self._analyse(2, 3 * self.DAY)
        self.assertFalse(a.spend["monthly_input_usd"].released)

    def test_a_day_of_real_traffic_publishes(self):
        """The other direction. A floor that withheld everything would pass a
        test asserting only that short windows withhold."""
        a = self._analyse(40, 3 * self.DAY)
        self.assertTrue(a.spend["monthly_input_usd"].released)

    def test_the_reason_names_the_window_not_the_invoice(self):
        """Telling a reader who supplied a correct invoice that reconciliation
        failed sends them to fix something that is not broken."""
        a = self._analyse(2, 1)
        why = str(a.spend["monthly_input_usd"])
        self.assertIn("day of traffic", why)
        self.assertNotIn("reconcil", why.lower())

    def test_the_reason_says_what_it_did_not_withhold(self):
        """The reason has to describe the code, in both directions.

        It used to end "...and any per-finding monthly figure below rests on the
        same extrapolation and has not been gated by it", which was an accurate
        confession while `_monthly` had seven callers and this floor gated one
        of them. It gates all seven now, so that sentence became a false
        confession -- a reader is sent hunting for a leak that no longer exists,
        and the next person to read it concludes the gate is narrower than it
        is. Overstating and understating what was protected are the same defect
        pointed opposite ways.

        What has to stay true is that the reason names both halves: what is
        withheld (every monthly figure) and what is not (the measured window).
        """
        from cacheeconomics.analyzer import _projection_supported
        for window, n in ((0.04, 12), (3.0, 2)):
            with self.subTest(window=window, requests=n):
                ok, why = _projection_supported(window, n)
                self.assertFalse(ok)
                self.assertIn("every monthly figure", why)
                self.assertIn("observed window still publish", why)
                self.assertNotIn("has not been gated", why)

    def test_a_supported_window_carries_no_reason_at_all(self):
        from cacheeconomics.analyzer import _projection_supported
        ok, why = _projection_supported(2.0, 40)
        self.assertTrue(ok)
        self.assertEqual(why, "")

    def test_the_demo_fixture_clears_the_floor(self):
        """Calibration. The floor is meant to catch a nine-minute capture, not
        a real workload -- if the shipped demo cannot clear it, it is wrong."""
        from cacheeconomics.analyzer import _projection_supported
        from cacheeconomics.trace import load_jsonl
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "fixtures", "demo-traces.jsonl")
        ts = load_jsonl(os.path.normpath(path))
        from cacheeconomics.analyzer import analyze
        a = analyze(ts, invoice_usd=17.45)
        ok, _ = _projection_supported(a.window_days, len(ts.requests))
        self.assertTrue(ok, f"demo window {a.window_days:.2f}d is below the floor")
        self.assertTrue(a.spend["monthly_input_usd"].released)


class TestTheProjectionFloorReachesEveryProjectedFigure(unittest.TestCase):
    """The floor above gated the spend total and nothing else.

    Measured on this fixture before the gate was widened: two requests one
    minute apart, with an invoice reconciling to 0.0%, published
    `findings[0].avoidable_usd_month` = $180 marked `released_as='reconciled'`,
    directly below a `monthly_input_usd` withheld for being an extrapolation the
    window could not support. Same window, same 30/window multiplier, one gate
    between them -- and the ungated figure was the one a client reads first.

    The figures are found by walking the analysis rather than by naming
    `findings[0]`. Naming the member I had just fixed is how the previous
    version of this claim came to be false: the test asserted the edit and the
    commit message asserted the class.
    """

    def _analysis(self):
        """Two minutes of traffic against an invoice that reconciles exactly.

        Synthetic. Both halves are load-bearing: the window has to be far below
        the floor, and the invoice has to *pass*, or every figure is withheld
        for an unrelated reason and the leak cannot be seen at all.
        """
        reqs = [Request(request_id=f"r{i}",
                        sent_at=T0 + timedelta(seconds=60 * i),
                        model="claude-opus-5", target_id="anthropic/direct",
                        agent="a", ttl_requested="5m",
                        usage={"input_tokens": 0,
                               "cache_creation_input_tokens": 100_000,
                               "cache_read_input_tokens": 0},
                        segments=[])
                for i in range(2)]
        ts = TraceSet(requests=reqs, tier=Tier.USAGE_ONLY)
        spend = analyze(ts, allow_unreconciled=True).spend["input_usd"].raw()
        return analyze(ts, invoice_usd=spend)

    def _figures(self, a):
        """Every Figure the analysis exposes, as `(path, figure)`.

        Discovery by field list and by property, not by name: `spend` is a dict,
        a finding's figures are whichever of its fields hold one, and
        `total_avoidable_month` is a property that a `dataclasses.fields()` walk
        skips entirely -- which is how the single most client-facing number in
        the package would have gone unchecked here.
        """
        import dataclasses
        from cacheeconomics.money import Figure
        out = [(f"spend[{k!r}]", v) for k, v in a.spend.items()
               if isinstance(v, Figure)]
        out += [(f"reconciliation[{k!r}]", v)
                for k, v in (a.reconciliation or {}).items()
                if isinstance(v, Figure)]
        for i, finding in enumerate(a.findings):
            out += [(f"findings[{i}].{fld.name}", getattr(finding, fld.name))
                    for fld in dataclasses.fields(finding)
                    if isinstance(getattr(finding, fld.name), Figure)]
        for name, attr in vars(type(a)).items():
            if isinstance(attr, property) and isinstance(getattr(a, name), Figure):
                out.append((name, getattr(a, name)))
        return out

    def test_the_fixture_reproduces_the_conditions(self):
        """Guard the guard. An invoice that failed, or a window that cleared the
        floor, and every assertion below passes for the wrong reason."""
        a = self._analysis()
        self.assertLess(a.window_days, 1.0)
        self.assertEqual(a.reconciliation["delta_pct"], 0.0)
        self.assertTrue(a.reconciliation["within_ship_gate"])

    def test_there_are_projected_figures_to_check(self):
        a = self._analysis()
        self.assertTrue([p for p, f in self._figures(a) if f.projected],
                        "no projected figures found; the check below would pass "
                        "while examining nothing")

    def test_no_projected_figure_is_released_below_the_floor(self):
        a = self._analysis()
        leaked = [(p, f) for p, f in self._figures(a) if f.projected and f.released]
        self.assertEqual(
            [], leaked,
            "extrapolated past a window that cannot support it:\n" +
            "\n".join(f"    {p} = {f} (released_as={f.released_as})"
                      for p, f in leaked))

    def test_the_measured_window_figures_still_publish(self):
        """The other direction, and the reason this is two fields rather than
        one gate. Withholding everything would also pass the test above, and
        would throw away exactly what the invoice does prove."""
        a = self._analysis()
        published = [p for p, f in self._figures(a) if f.released]
        self.assertIn("spend['input_usd']", published)
        self.assertTrue(
            [p for p in published if p.endswith(".avoidable_usd_window")],
            "no finding published its window amount, so the floor is "
            "withholding the measurement as well as the projection")

    def test_the_withheld_reason_names_the_window_not_the_invoice(self):
        """This reader supplied a correct invoice. Telling them reconciliation
        failed sends them to fix something that is not broken."""
        a = self._analysis()
        for path, fig in self._figures(a):
            if fig.projected and not fig.released:
                with self.subTest(figure=path):
                    self.assertIn("day of traffic", fig.withheld_because)
                    self.assertNotIn("reconcil", fig.withheld_because.lower())


class TestTheProjectionFloorJudgesEachFindingsOwnSample(unittest.TestCase):
    """The floor was evaluated once, from the whole trace, and applied to every
    finding. So unrelated traffic could vouch for a projection it contributed
    nothing to.

    Measured on the fixture below -- ten requests over 2.3 days, of which only
    two touch the cache and those two go out one second apart -- EFF-1 published
    $6.43/mo and FAN-1 $14.79/mo, both `released_as='reconciled'`, with
    `total_avoidable_month` at $21.21. Every one of those numbers rests on a
    two-request, one-second sample. The request floor exists precisely so that a
    small subset cannot drive a client-facing monthly projection, and evaluated
    globally it did not do that job: the eight filler requests it counted
    contributed exactly zero dollars to either figure.
    """

    def _trace(self):
        """SYNTHETIC. Eight uncached requests spread over three days, plus two
        concurrent writers of the same prefix one second apart.

        The filler is what makes the global window and count clear the floor;
        it prices identically cached or not, so it contributes no term to any
        figure here. That gap between "counted" and "contributed" is the defect.
        """
        reqs = [Request(request_id=f"p{i}", sent_at=T0 + timedelta(hours=8 * i),
                        model="claude-opus-5", target_id="anthropic/direct",
                        tenant="t", agent="filler", session=f"f{i}",
                        ttl_requested="5m",
                        usage={"input_tokens": 1_000,
                               "cache_read_input_tokens": 0,
                               "cache_creation_input_tokens": 0},
                        segments=[])
                for i in range(8)]
        reqs += [Request(request_id=f"w{k}",
                         sent_at=T0 + timedelta(days=1, seconds=k),
                         model="claude-opus-5", target_id="anthropic/direct",
                         tenant="t", agent="fan", session="s",
                         ttl_requested="5m",
                         usage={"input_tokens": 0, "cache_read_input_tokens": 0,
                                "cache_creation_input_tokens": 200_000},
                         segments=[Segment(id="sys", role="system",
                                           tokens=200_000, index=0,
                                           cache_marked=True, ttl="5m")])
                 for k in range(2)]
        return TraceSet(requests=reqs, tier=Tier.INSTRUMENTED)

    def _analysis(self):
        ts = self._trace()
        invoice = analyze(ts, allow_unreconciled=True).spend["input_usd"].raw()
        return analyze(ts, invoice_usd=invoice)

    def _priced(self, a):
        return [f for f in a.findings if f.avoidable_usd_month is not None]

    def test_the_fixture_clears_the_floor_globally(self):
        """Guard the guard, and it is the whole point of this class: if the
        trace itself failed the floor, the trace-wide gate would withhold these
        figures and the per-finding check would never be what was under test."""
        from cacheeconomics.analyzer import _projection_supported, _window_days
        reqs = self._trace().analysable
        ok, _ = _projection_supported(_window_days(reqs), len(reqs))
        self.assertTrue(ok, "the trace no longer clears the floor globally, so "
                            "this class is testing the trace-wide gate instead")
        self.assertEqual(self._analysis().reconciliation["delta_pct"], 0.0)

    def test_there_are_priced_findings_to_check(self):
        self.assertTrue(self._priced(self._analysis()),
                        "no finding carries a monthly figure; the checks below "
                        "would pass while examining nothing")

    def test_a_finding_costed_from_a_tiny_subset_publishes_no_month(self):
        a = self._analysis()
        leaked = [(f.code, f.affected_requests, str(f.avoidable_usd_month))
                  for f in self._priced(a) if f.avoidable_usd_month.released]
        self.assertEqual(
            [], leaked,
            "these projected a month from their own small sample, on the "
            "strength of unrelated traffic clearing the floor:\n" +
            "\n".join(f"    {c} ({n} affected requests) = {v}"
                      for c, n, v in leaked))

    def test_the_total_inherits_it(self):
        """`total_avoidable_month` reads release off its parts, so this needs no
        gate of its own -- but it is the number a client reads first, and
        "inherits correctly" is worth asserting rather than assuming."""
        self.assertFalse(self._analysis().total_avoidable_month.released)

    def test_the_measured_window_amount_still_publishes(self):
        """The split earning its keep. What these two requests actually cost is
        measured, the invoice establishes it, and withholding it too would throw
        away the finding entirely rather than just its projection."""
        for f in self._priced(self._analysis()):
            with self.subTest(finding=f.code):
                self.assertTrue(f.avoidable_usd_window.released)

    def test_the_reason_reports_the_findings_numbers_not_the_traces(self):
        """A refusal citing the trace's ten requests and 2.3 days, printed
        beside a coverage line saying the same, reads as the tool malfunctioning
        rather than as a floor doing its job."""
        for f in self._priced(self._analysis()):
            with self.subTest(finding=f.code):
                why = f.avoidable_usd_month.withheld_because
                self.assertIn("this finding", why)
                self.assertNotIn("this trace", why)

    def test_the_reason_does_not_claim_the_spend_total_was_withheld(self):
        """It was not: the trace clears the floor, so `monthly_input_usd`
        publishes. The trace-wide wording says "the spend total and each
        finding's", and printing that beside a published spend total is the
        overstatement this floor's reason has already been fixed for once."""
        a = self._analysis()
        self.assertTrue(a.spend["monthly_input_usd"].released,
                        "fixture no longer publishes a monthly spend total")
        for f in self._priced(a):
            with self.subTest(finding=f.code):
                self.assertNotIn("the spend total",
                                 f.avoidable_usd_month.withheld_because)

    def test_a_finding_whose_own_sample_clears_the_floor_still_publishes(self):
        """The other direction. A floor that withheld every finding's month
        would pass every test above and be useless."""
        reqs = [Request(request_id=f"r{i}",
                        sent_at=T0 + timedelta(hours=6 * i),
                        model="claude-opus-5", target_id="anthropic/direct",
                        tenant="t", agent="a", session="s", ttl_requested="5m",
                        usage={"input_tokens": 0, "cache_read_input_tokens": 0,
                               "cache_creation_input_tokens": 100_000},
                        segments=[])
                for i in range(12)]
        ts = TraceSet(requests=reqs, tier=Tier.USAGE_ONLY)
        invoice = analyze(ts, allow_unreconciled=True).spend["input_usd"].raw()
        a = analyze(ts, invoice_usd=invoice)
        eff = next(f for f in a.findings if f.code == "EFF-1")
        self.assertEqual("", eff.projection_why)
        self.assertTrue(eff.avoidable_usd_month.released)
        self.assertTrue(a.total_avoidable_month.released)


class TestEveryProjectionSampleIsMadeOfRequestsThatMoveTheFigure(unittest.TestCase):
    """The floor is only as good as the set handed to it.

    Judging each finding on its own sample closed the case where unrelated
    traffic carried the floor. It left the case one level in: a rule can hand
    over a *wider* set than the one it charged, and the padding clears the floor
    just as effectively. FAN-1 did exactly that -- it charges only the later
    writer of each concurrent pair (`waste += ub...`) and recorded both sides.
    Measured on five pairs spread over five days: ten timestamps, five charged,
    floor cleared, FAN-1 published $43.12/mo marked `reconciled` and
    `total_avoidable_month` $61.87.

    So this does not check FAN-1. It finds, for each finding a fixture produces,
    which requests actually move that finding's figure, and requires the sample
    the floor counted to be no larger. A rule that pads its sample fails here
    whether or not anybody thought to look at it -- which is the difference
    between fixing the member that was reported and closing the class.

    That has now been paid off once. TTL-2 turned out to pad the same floor a
    different way -- one request contributing two terms, for its one-hour write
    and again for a priced in-band read, so nine dollar-moving requests reported
    a sample of ten. The probe counts *distinct* requests, so it registers that
    as a sample larger than its movers and fails, without knowing anything about
    TTL-2. The identity notion the fix needed is the one the probe already had.

    The probe perturbs rather than deletes. Removing a request changes what the
    trace *is*: drop the leader of a fan-out pair and the pair stops existing,
    so a leave-one-out probe would call that leader load-bearing and agree with
    the defect. Scaling its billed tokens and its segment sizes leaves every
    structural relationship intact -- same pairs, same spans, same cadence --
    and moves only magnitude, which is the thing a projection scales.

    It scales in both directions, and that is not belt-and-braces. Scaling only
    up reported 14 movers against TTL-1's 126 on the demo capture, which reads
    exactly like the FAN-1 defect and is not one: TTL-1 recovers
    `min(previous entry, this write)`, so on a trace where every request writes
    about the same amount, tripling one of them leaves the minimum where it was
    and the total unmoved. The requests were contributing; the probe could not
    see it. Verified by instrumenting the rule directly -- 126 sample entries,
    126 distinct timestamps, every one carrying a nonzero term. A probe whose
    blind spot looks identical to the defect it hunts is worse than no probe,
    so it shrinks as well as grows and counts a request that moves the figure
    either way.
    """

    SCALE = 3

    def _fixtures(self):
        """`(label, TraceSet)` for traces that between them price several rules."""
        return [("fan-out pairs over five days", self._fanout()),
                ("one-hour writes with a priced band gap", self._ttl2()),
                ("volatile prefix", self._volatile()),
                ("demo capture", self._demo())]

    def _ttl2(self):
        """TTL-2 with one request contributing two terms.

        It appends a timestamp for a one-hour write and again when that same
        request sits after an in-band gap carrying a priced read. Two clusters a
        day and a half apart so the span clears the day floor, with a single
        ten-minute gap inside the second so exactly one band gap is priced --
        and the request after it both writes at one hour and reads.

        Nine distinct requests move the figure; before the fix the sample
        reported ten and released the month. This class caught that without
        being told to look for it, which is the property being claimed: the
        probe counts distinct requests, so a request counted twice shows up as
        a sample larger than the movers.
        """
        def req(rid, when, write_1h, read):
            usage = {"input_tokens": 100, "cache_read_input_tokens": read,
                     "cache_creation_input_tokens": write_1h}
            if write_1h:
                usage["cache_creation"] = {"ephemeral_5m_input_tokens": 0,
                                           "ephemeral_1h_input_tokens": write_1h}
            return Request(request_id=rid, sent_at=when, model="claude-opus-5",
                           target_id="anthropic/direct", tenant="t",
                           session="s", agent="a", ttl_requested="1h",
                           usage=usage, segments=[])

        reqs = [req(f"a{i}", T0 + timedelta(seconds=60 * i),
                    50_000 if i < 4 else 0, 0) for i in range(6)]
        base = T0 + timedelta(days=1, hours=12)
        for i in range(6):
            offset = 60 * i + (540 if i >= 4 else 0)
            write, read = (50_000 if i < 4 else 0), 0
            if i == 4:
                write, read = 50_000, 200_000
            reqs.append(req(f"b{i}", base + timedelta(seconds=offset),
                            write, read))
        return TraceSet(requests=reqs, tier=Tier.USAGE_ONLY, source="x")

    def _fanout(self):
        reqs = []
        for p in range(5):
            base = T0 + timedelta(days=p)
            for k, rid in ((0, f"lead{p}"), (1, f"dup{p}")):
                reqs.append(Request(
                    request_id=rid, sent_at=base + timedelta(seconds=k),
                    model="claude-opus-5", target_id="anthropic/direct",
                    tenant="t", agent="a", session="s", ttl_requested="5m",
                    usage={"input_tokens": 0, "cache_read_input_tokens": 0,
                           "cache_creation_input_tokens": 200_000},
                    segments=[Segment(id="sys", role="system", tokens=200_000,
                                      index=0, cache_marked=True, ttl="5m")]))
        return TraceSet(requests=reqs, tier=Tier.INSTRUMENTED, source="x")

    def _volatile(self):
        reqs = [Request(
            request_id=f"r{i}", sent_at=T0 + timedelta(hours=3 * i),
            model="claude-opus-5", target_id="anthropic/direct", tenant="t",
            agent="a", session="s", ttl_requested="1h",
            usage={"input_tokens": 100, "cache_read_input_tokens": 0,
                   "cache_creation_input_tokens": 40_000},
            segments=[Segment(id=f"hdr{i}", role="system", tokens=300, index=0,
                              label="hdr"),
                      Segment(id="body", role="system", tokens=30_000, index=1,
                              label="sys", cache_marked=True, ttl="1h")])
            for i in range(14)]
        return TraceSet(requests=reqs, tier=Tier.INSTRUMENTED, source="x")

    def _demo(self):
        from cacheeconomics.trace import load_jsonl
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "fixtures", "demo-traces.jsonl")
        return load_jsonl(os.path.normpath(path))

    def _scaled(self, ts, index, up):
        """The same trace with one request's magnitude scaled, nothing else."""
        import dataclasses

        def s(v):
            if not v:
                return v
            return v * self.SCALE if up else max(1, v // self.SCALE)

        out = []
        for i, r in enumerate(ts.requests):
            if i != index:
                out.append(r)
                continue
            usage = dict(r.usage)
            for k in ("cache_creation_input_tokens", "cache_read_input_tokens",
                      "input_tokens"):
                if usage.get(k):
                    usage[k] = s(usage[k])
            if isinstance(usage.get("cache_creation"), dict):
                usage["cache_creation"] = {k: s(v)
                                           for k, v in usage["cache_creation"].items()}
            segs = [dataclasses.replace(seg, tokens=s(seg.tokens or 0))
                    for seg in r.segments]
            out.append(dataclasses.replace(r, usage=usage, segments=segs))
        return dataclasses.replace(ts, requests=out)

    def _figures(self, ts):
        """`{code: window amount}` for every priced finding, or {} on refusal."""
        a = analyze(ts, allow_unreconciled=True)
        return {f.code: f.avoidable_usd_window.raw()
                for f in a.findings if f.avoidable_usd_window is not None}

    def _movers(self, ts):
        """`{code: how many requests move that finding's figure}`."""
        base = self._figures(ts)
        counts = {code: 0 for code in base}
        for i in range(len(ts.requests)):
            moved = set()
            for up in (True, False):
                after = self._figures(self._scaled(ts, i, up))
                for code, amount in base.items():
                    if code not in after or abs(after[code] - amount) > 1e-12:
                        moved.add(code)
            for code in moved:
                counts[code] += 1
        return counts

    def test_the_probe_finds_movers_at_all(self):
        """Vacuity guard. A probe that never registers a change would make
        every sample look oversized and the check below would fail loudly
        rather than silently -- but a probe that registers a change for
        *everything* would make every sample look fine, which is the direction
        that hides a defect."""
        for label, ts in self._fixtures():
            with self.subTest(fixture=label):
                movers = self._movers(ts)
                self.assertTrue(movers, "no priced findings to check")
                for code, n in movers.items():
                    self.assertGreater(
                        n, 0, f"{code}: perturbing any request changed nothing, "
                              f"so the probe is not measuring magnitude")
                    self.assertLess(
                        n, len(ts.requests) + 1,
                        f"{code}: more movers than requests")

    def test_no_finding_counts_more_requests_than_move_its_figure(self):
        for label, ts in self._fixtures():
            a = analyze(ts, allow_unreconciled=True)
            movers = self._movers(ts)
            for f in a.findings:
                if f.avoidable_usd_window is None:
                    continue
                with self.subTest(fixture=label, finding=f.code):
                    self.assertLessEqual(
                        f.projection_sample, movers[f.code],
                        f"{f.code} counted {f.projection_sample} requests "
                        f"toward the projection floor, but only "
                        f"{movers[f.code]} of them move its figure: the "
                        f"remainder pad the sample and clear a floor they "
                        f"contributed nothing to")

    def _identical_but_for_ids(self, ids):
        reqs = [Request(request_id=rid, sent_at=T0 + timedelta(hours=6 * i),
                        model="claude-opus-5", target_id="anthropic/direct",
                        tenant="t", session="s", agent="a", ttl_requested="5m",
                        usage={"input_tokens": 0, "cache_read_input_tokens": 0,
                               "cache_creation_input_tokens": 100_000},
                        segments=[])
                for i, rid in enumerate(ids)]
        return TraceSet(requests=reqs, tier=Tier.USAGE_ONLY, source="x")

    def test_the_sample_does_not_depend_on_request_ids_being_usable(self):
        """The under-count direction, which the probe above cannot see.

        That probe asserts the sample is no *larger* than the requests that move
        the figure, so it catches padding and is blind to the opposite. The
        dedup introduced to stop TTL-2 counting terms as requests keyed on
        `request_id` -- and `request_id` is not a key. Nothing in the schema
        requires it to be unique or present, and the Claude Code loader writes
        `""` whenever a record carries neither `requestId` nor `uuid`. Measured
        on that shape: twelve distinct contributing requests collapsed to a
        sample of 1 with a zero-hour span, and a monthly figure the trace fully
        supported was withheld.

        It fails toward withholding, which is why it needed a test rather than a
        bug report: nobody complains about a figure they never saw, so it would
        have outlived the defect it was introduced to fix.

        Asserted as an equality across three id shapes rather than as a number,
        so this stays true if the fixture's arithmetic ever changes.
        """
        shapes = {"distinct": [f"r{i}" for i in range(12)],
                  "all blank": [""] * 12,
                  "all identical": ["dup"] * 12}
        samples = {}
        for label, ids in shapes.items():
            a = analyze(self._identical_but_for_ids(ids), allow_unreconciled=True)
            eff = next(f for f in a.findings if f.code == "EFF-1")
            samples[label] = eff.projection_sample
        self.assertEqual(12, samples["distinct"],
                         "the control no longer counts every contributor")
        self.assertEqual(
            {12}, set(samples.values()),
            f"the projection sample changes with the shape of the request ids, "
            f"which are not a key: {samples}")

    def test_and_the_monthly_figure_survives_a_trace_with_no_ids(self):
        """The consequence, at the figure rather than at the counter."""
        ts = self._identical_but_for_ids([""] * 12)
        invoice = analyze(ts, allow_unreconciled=True).spend["input_usd"].raw()
        a = analyze(ts, invoice_usd=invoice)
        eff = next(f for f in a.findings if f.code == "EFF-1")
        self.assertTrue(eff.avoidable_usd_month.released,
                        "a supportable projection was withheld because the "
                        "export carried no request ids")

    def test_fan_out_counts_one_request_per_pair(self):
        """The member that was reported, named explicitly so the measurement
        survives as a number rather than only as a property."""
        ts = self._fanout()
        a = analyze(ts, allow_unreconciled=True)
        fan = next(f for f in a.findings if f.code == "FAN-1")
        self.assertEqual(5, fan.projection_sample,
                         "five pairs charge five requests")
        self.assertEqual(10, fan.affected_requests,
                         "but both sides of a pair are genuinely affected")

    def test_and_therefore_publishes_no_month(self):
        ts = self._fanout()
        invoice = analyze(ts, allow_unreconciled=True).spend["input_usd"].raw()
        a = analyze(ts, invoice_usd=invoice)
        fan = next(f for f in a.findings if f.code == "FAN-1")
        self.assertFalse(fan.avoidable_usd_month.released)
        self.assertTrue(fan.avoidable_usd_window.released,
                        "the measured window amount is unaffected")
        self.assertFalse(a.total_avoidable_month.released)


class TestSilenceUnderAnUnnamedSurfaceIsSpokenAloud(unittest.TestCase):
    """Three runtime checks read the registry for the request's surface and
    `return` when it does not know it.

    Measured: a 200-token marker below the 512-token minimum raises RT-MIN on
    `anthropic/direct` and nothing at all on an unnamed surface. The operator
    sees a quiet dashboard and reads it as healthy — the same shape as RT-BLIND,
    which already exists for a stream carrying no prompt structure.
    """

    def _run(self, target):
        m, fired = monitor.Monitor(), []
        for i in range(20):
            segs = [Segment(id="sys", role="system", tokens=200, index=0,
                            cache_marked=True, ttl="5m"),
                    Segment(id=f"t{i}", role="user", tokens=100, index=1)]
            fired += m.observe(Request(
                request_id=f"r{i}", sent_at=T0 + timedelta(seconds=60 * i),
                model="claude-opus-5", target_id=target, ttl_requested="5m",
                session="s",
                usage={"input_tokens": 300, "cache_creation_input_tokens": 300},
                segments=segs))
        return [a.code for a in fired]

    def test_a_named_surface_reports_the_short_marker(self):
        self.assertIn("RT-MIN", self._run("anthropic/direct"))

    def test_an_unnamed_surface_says_the_checks_are_inactive(self):
        codes = self._run(UNATTRIBUTED)
        self.assertIn("RT-NOSURFACE", codes)
        self.assertNotIn("RT-MIN", codes, "it cannot know this; it must not say it")

    def test_it_is_said_once_not_per_request(self):
        self.assertEqual(self._run(UNATTRIBUTED).count("RT-NOSURFACE"), 1)

    def test_a_named_surface_does_not_get_the_notice(self):
        self.assertNotIn("RT-NOSURFACE", self._run("anthropic/direct"))


class TestAStandDownAlertSurvivesChurn(unittest.TestCase):
    """`_record_alert` suppressed by `_said` while `alerts` is a bounded deque
    that evicts oldest-first.

    So ordinary alert churn dropped the one stand-down warning and `_said` kept
    refusing to re-emit it. Measured: visible after the first stand-down, gone
    after churn, never re-emitted — the operator whose mutation had stopped
    could no longer find out why. The dedup added to stop spam guaranteed the
    silence instead.
    """

    def _alert(self):
        return monitor.Alert("RT-UNATTRIBUTED", "medium", ("t", "x"),
                             "stood down", "detail", subject="unattributed")

    def test_a_repeat_is_suppressed_while_it_is_still_visible(self):
        p = plugin.CachePlugin(key=b"k" * 32, warmup=0)
        self.assertTrue(p._record_alert(self._alert()))
        self.assertFalse(p._record_alert(self._alert()))

    def test_it_comes_back_once_it_has_been_evicted(self):
        p = plugin.CachePlugin(key=b"k" * 32, warmup=0)
        p._record_alert(self._alert())
        for i in range(plugin.MAX_ALERTS + 5):
            p.alerts.append(monitor.Alert("RT-NOISE", "low", ("t", "x"),
                                          f"n{i}", "d", subject=str(i)))
        self.assertNotIn("RT-UNATTRIBUTED", {a.code for a in p.alerts})
        self.assertTrue(p._record_alert(self._alert()))
        self.assertIn("RT-UNATTRIBUTED", {a.code for a in p.alerts})

    def test_the_alert_list_stays_bounded(self):
        """Re-emission must not become unbounded growth."""
        p = plugin.CachePlugin(key=b"k" * 32, warmup=0)
        for _ in range(plugin.MAX_ALERTS * 3):
            p._record_alert(self._alert())
            p.alerts.append(monitor.Alert("RT-NOISE", "low", ("t", "x"),
                                          "n", "d", subject="n"))
        self.assertLessEqual(len(p.alerts), plugin.MAX_ALERTS)


class TestUnattributedIsNotASurfaceYouCanConfigure(unittest.TestCase):
    """The mutation guard checked truthiness, and `UNATTRIBUTED` is truthy. So
    `mutate=True, target_id=UNATTRIBUTED` passed the guard and then stood down
    on every request — the silent behaviour the guard exists to replace."""

    def test_it_is_refused_like_an_omitted_surface(self):
        from cacheeconomics.plugin import CachePlugin, litellm_handler
        p = CachePlugin(key=b"k" * 32, warmup=0)
        with self.assertRaises(ValueError):
            litellm_handler(p, mutate=True, target_id=UNATTRIBUTED)

    def test_a_real_surface_is_still_accepted(self):
        from cacheeconomics.plugin import CachePlugin, litellm_handler
        p = CachePlugin(key=b"k" * 32, warmup=0)
        self.assertTrue(litellm_handler(p, mutate=True,
                                        target_id="anthropic/direct"))


class TestTheDirectPathObeysTheSameMarkingRule(unittest.TestCase):
    """`markable_positions` stands a whole message down when it carries tool
    history — the fields *and* the content — because marking a bare-string
    `tool` message rewrites it into Anthropic block form on an OpenAI-shaped
    request. `_filter_near_minimum` reimplemented only the narrow half beside
    it, so `on_request` with the default `markable=None` placed a marker on a
    tool result's content that `markable_positions` would have excluded.

    Measured before the fix: barred [1, 2], markable [0, 4], and the direct
    path placed at 3. Same rule, two scopes — the shape the narrow half was
    added to close.

    The fixture makes the system block large enough to clear the near-minimum
    floor, so "placed nothing" cannot be mistaken for "excluded correctly".
    """

    @staticmethod
    def _body(i, tool_history=True):
        msgs = [{"role": "system", "content": "policy " * 9000}]
        if tool_history:
            msgs += [{"role": "assistant", "content": None,
                      "tool_calls": [{"id": "call_x", "type": "function",
                                      "function": {"name": "f",
                                                   "arguments": "{}"}}]},
                     {"role": "tool", "tool_call_id": "call_x",
                      "content": "RESULT " * 9000}]
        msgs.append({"role": "user", "content": f"turn {i}"})
        return {"model": "claude-opus-5", "messages": msgs}

    def _drive(self, tool_history=True, apply=True):
        p = plugin.CachePlugin(key=b"k" * 32, warmup=4)
        last = None
        for i in range(20):
            _, last = p.on_request(self._body(i, tool_history),
                                   model="claude-opus-5",
                                   target_id="anthropic/direct",
                                   at=T0 + timedelta(seconds=90 * i), apply=apply)
        return last

    def test_the_direct_path_agrees_with_markable_positions(self):
        """The invariant the drift broke: whatever `markable_positions` would
        exclude, the default path does not place."""
        for apply in (False, True):
            with self.subTest(apply=apply):
                last = self._drive(apply=apply)
                barred = plugin._tool_history_positions(self._body(0))
                self.assertFalse(set(last.placements) & barred)

    def test_the_tool_results_content_is_barred_not_just_its_fields(self):
        b = self._body(0)
        barred = plugin._tool_history_positions(b)
        mk = plugin.markable_positions(b)
        self.assertTrue(barred - {1, 2},
                        "only the named fields are barred; content is not")
        self.assertFalse(barred & mk, "the two rules disagree")

    def test_it_still_places_where_it_may(self):
        """Excluding the tool message must not become placing nothing — which
        the near-minimum floor alone would also produce."""
        last = self._drive()
        self.assertTrue(last.applied)
        self.assertTrue(last.placements)

    def test_a_body_without_tool_history_bars_nothing(self):
        b = self._body(0, tool_history=False)
        self.assertEqual(plugin._tool_history_positions(b), frozenset())
        last = self._drive(tool_history=False)
        self.assertTrue(last.placements)

    def test_the_writer_refuses_it_even_if_a_caller_asks(self):
        """The last line of defence. `_mark` caught the named fields, where the
        path element is a string; on the message's own content the path looks
        ordinary and it did not."""
        from cacheeconomics.segment import apply_markers, walk
        b = self._body(0)
        content_pos = next(i for i, (_r, _l, _b, path) in enumerate(walk(b))
                           if path[0] == "messages" and path[1] == 2
                           and not isinstance(path[2], str))
        with self.assertRaises(ValueError):
            apply_markers(b, {content_pos: "5m"})


class TestNoSurfaceIsAnsweredForByDefault(unittest.TestCase):
    """Default-deny for the *surface*, at every public entry point.

    Same shape as the unknown-model sweep above, one field over. `target_id`
    selects the rate table and the capability limits, so a parameter default
    naming a surface is an answer nobody gave -- and it is given confidently,
    because from inside the check there is no difference between a caller who
    chose `anthropic/direct` and a caller who chose nothing.

    The harm is measured rather than theoretical, and the first test below is
    the reproduction: minimums differ by surface, so one request is PASS on one
    and FAIL on another.

    Eight public callables defaulted to `anthropic/direct`. Adversarial review
    found two of them by hand. What follows pins the *behaviour* of each rather
    than the signature, because a signature can be satisfied while the callee
    quietly substitutes a surface back in.
    """

    def test_the_same_prefix_gets_two_verdicts_on_two_surfaces(self):
        """The reproduction the rest of this class exists for."""
        from cacheeconomics import registry
        self.assertEqual(registry.min_cacheable_tokens("anthropic/direct",
                                                       "claude-opus-5"), 512)
        self.assertEqual(registry.min_cacheable_tokens("openai/direct",
                                                       "claude-opus-5"), 1024)
        opts = dict(tokens_are_estimated=False)
        self.assertIs(checks.check_minimum(768, "claude-opus-5",
                                           "anthropic/direct", **opts).status,
                      checks.Status.PASS)
        self.assertIs(checks.check_minimum(768, "claude-opus-5",
                                           "openai/direct", **opts).status,
                      checks.Status.FAIL)

    def test_so_a_check_with_no_surface_abstains_rather_than_picking_one(self):
        for r in (checks.check_minimum(768, "claude-opus-5"),
                  checks.check_breakpoint_budget(5),
                  checks.check_ttl_ordering(["5m"], UNATTRIBUTED)):
            with self.subTest(check=r.check):
                self.assertIs(r.status, checks.Status.ABSTAIN)
                self.assertIn("no provider surface named", r.summary)

    def test_run_all_abstains_on_every_check_rather_than_one(self):
        rs = checks.run_all(prefix_tokens=768, model="claude-opus-5",
                            breakpoints=5, ttls_in_order=["5m"])
        self.assertEqual([checks.Status.ABSTAIN] * 3, [r.status for r in rs])
        self.assertIs(checks.worst(rs), checks.Status.ABSTAIN)

    def test_the_crossover_question_cannot_be_asked_without_a_surface(self):
        """No default at all here: every number it returns is read out of one
        surface's row, so there is nothing to abstain *with*."""
        with self.assertRaises(TypeError):
            cost.ttl_crossover()

    def test_the_live_plugin_stands_down_when_no_surface_was_named(self):
        p = plugin.CachePlugin(key=b"k" * 32, warmup=4)
        body = {"system": [{"type": "text", "text": "s" * 90_000}],
                "messages": [{"role": "user", "content": "t"}]}
        out = last = None
        for i in range(12):
            out, last = p.on_request(body, model="claude-opus-5", apply=True,
                                     at=T0 + timedelta(seconds=120 * i))
        self.assertFalse(last.applied)
        self.assertEqual({}, last.placements)
        self.assertNotIn("cache_control", json.dumps(out))
        self.assertEqual(UNATTRIBUTED, last.scope[1],
                         "the scope must record that no surface was named, not "
                         "file this traffic under a first-party one")

    def test_the_recorder_writes_no_surface_rather_than_a_first_party_one(self):
        from cacheeconomics.recorder import Recorder
        path = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False).name
        try:
            rec = Recorder(path, key=b"k" * 32)
            self.assertEqual(UNATTRIBUTED, rec.target_id)
            rec.capture({"model": "claude-opus-5",
                         "messages": [{"role": "user", "content": "hi"}]},
                        agent="a").done({"usage": {"input_tokens": 10}})
            with open(path) as f:
                row = json.loads(f.readline())
            self.assertEqual(UNATTRIBUTED, row["target_id"])
        finally:
            os.unlink(path)

    def test_a_request_nobody_told_carries_no_surface(self):
        """The field every one of the above eventually writes into."""
        r = Request(request_id="r", sent_at=T0, model="claude-opus-5", usage={})
        self.assertEqual(UNATTRIBUTED, r.target_id)

    def test_and_that_withholds_dollars_instead_of_publishing_wrong_ones(self):
        """The consequence, end to end: unpriceable rather than mispriced."""
        reqs = [Request(request_id=f"r{i}", sent_at=T0 + timedelta(seconds=60 * i),
                        model="claude-opus-5",
                        usage={"input_tokens": 1_000_000,
                               "cache_read_input_tokens": 0,
                               "cache_creation_input_tokens": 0})
                for i in range(4)]
        a = analyze(TraceSet(requests=reqs, tier=Tier.USAGE_ONLY),
                    allow_unreconciled=True)
        self.assertFalse(a.spend["input_usd"].released)
        self.assertTrue(any(UNATTRIBUTED in n for n in a.notes),
                        "the report must say which surface it could not price")

    def test_the_claude_code_adapter_says_so_rather_than_assuming(self):
        from cacheeconomics.adapters.claude_code import load_sessions
        with tempfile.TemporaryDirectory() as tmp:
            proj = os.path.join(tmp, "proj")
            os.makedirs(proj)
            with open(os.path.join(proj, "s.jsonl"), "w") as f:
                for i in range(4):
                    f.write(json.dumps({
                        "type": "assistant", "sessionId": "s1", "uuid": f"u{i}",
                        "requestId": f"r{i}",
                        "timestamp": f"2026-07-29T09:0{i}:00.000Z",
                        "message": {"model": "claude-opus-5",
                                    "usage": {"input_tokens": 100,
                                              "output_tokens": 10,
                                              "cache_read_input_tokens": 20_000,
                                              "cache_creation_input_tokens": 1_000}}}) + "\n")
            ts = load_sessions(root=tmp)
        self.assertTrue(ts.requests, "fixture produced no requests")
        self.assertTrue(all(r.target_id == UNATTRIBUTED for r in ts.requests))
        self.assertTrue(any("No provider surface was stated" in n
                            for n in ts.blocking_notes),
                        "an unnamed surface has to block release, not pass quietly")


class TestMutationIsOptIn(unittest.TestCase):
    """`on_request` used to default `apply=True`.

    So a direct integration -- somebody wiring this into their own client rather
    than into a proxy -- rewrote live requests by default, while
    `litellm_handler` one layer up defaulted `mutate=False` and documented at
    length why. Two supported ways of using the package, disagreeing about
    whether it edits somebody's traffic.

    Measured before the change: twelve warm requests through `on_request(body,
    model='claude-opus-5')` returned a body carrying `cache_control`, with
    `applied=True` and `placements={1: '5m'}`, against a surface nobody named.
    """

    BODY = {"system": [{"type": "text", "text": "s" * 90_000}],
            "messages": [{"role": "user", "content": "hello"}]}

    def _run(self, **kw):
        p = plugin.CachePlugin(key=b"k" * 32, warmup=4)
        out = d = None
        for i in range(12):
            out, d = p.on_request(self.BODY, model="claude-opus-5",
                                  target_id="anthropic/direct",
                                  at=T0 + timedelta(seconds=120 * i), **kw)
        return out, d

    def test_the_default_places_nothing_on_the_wire(self):
        out, d = self._run()
        self.assertFalse(d.applied)
        self.assertNotIn("cache_control", json.dumps(out))

    def test_the_default_returns_the_callers_own_body(self):
        out, _d = self._run()
        self.assertIs(self.BODY, out)

    def test_it_still_says_what_it_would_have_done(self):
        """Observe-only is not silent. The plan is kept as a proposal, which is
        what makes the default usable rather than merely safe."""
        _out, d = self._run()
        self.assertTrue(d.proposed)
        self.assertIn("observing only", d.reason)

    def test_and_asking_for_it_still_works(self):
        out, d = self._run(apply=True)
        self.assertTrue(d.applied)
        self.assertIn("cache_control", json.dumps(out))


class TestAMarkerPlanIsNotWrittenOntoASurfaceThatIgnoresMarkers(unittest.TestCase):
    """`tiers.allocate` noted that a surface controls caching some other way,
    and then emitted a marker plan regardless.

    Measured: twelve warm requests through the plugin with
    `target_id='amazon-bedrock/converse'` put `cache_control` on the wire --
    `placements={1: '5m'}` -- while the note beside it said "treat the placement
    as indicative on this surface". A caveat attached to a mutation is not a
    caveat: nobody reads a note at request time.

    The split is by intent. A report may still model the plan and say so,
    because a reader sees the note. A request path may not, because a marker the
    surface does not honour is billed and returns no error.
    """

    SEGS = [Segment(id="a", role="system", tokens=20_000, index=0),
            Segment(id="b", role="user", tokens=200, index=1)]
    RATES = {0: 0.0, 1: 1.0}
    GAPS = [120.0] * 20
    BEDROCK = "amazon-bedrock/converse"

    def _plugin_run(self, apply):
        p = plugin.CachePlugin(key=b"k" * 32, warmup=4)
        body = {"system": [{"type": "text", "text": "s" * 90_000}],
                "messages": [{"role": "user", "content": "t"}]}
        out = last = None
        for i in range(12):
            out, last = p.on_request(body, model="claude-opus-5",
                                     target_id=self.BEDROCK, apply=apply,
                                     at=T0 + timedelta(seconds=120 * i))
        return out, last

    def test_a_report_still_models_it_and_says_it_is_indicative(self):
        a = tiers.allocate(self.SEGS, self.RATES, target_id=self.BEDROCK,
                           model="claude-opus-5", gaps=self.GAPS)
        self.assertTrue(a.tiers, "the report path must still produce a plan")
        self.assertTrue(any("controls caching by checkpoint_backward_search" in n
                            for n in a.notes))
        self.assertTrue(any("indicative" in n for n in a.notes))

    def test_but_the_same_plan_is_refused_when_it_is_about_to_be_written(self):
        with self.assertRaises(tiers.Unsupported) as e:
            tiers.allocate(self.SEGS, self.RATES, target_id=self.BEDROCK,
                           model="claude-opus-5", gaps=self.GAPS,
                           for_mutation=True)
        self.assertIn("checkpoint_backward_search", str(e.exception))
        self.assertIn("live request", str(e.exception))

    def test_the_live_plugin_stands_down_rather_than_marking_bedrock(self):
        out, last = self._plugin_run(apply=True)
        self.assertFalse(last.applied)
        self.assertNotIn("cache_control", json.dumps(out))
        self.assertIn("checkpoint_backward_search", last.reason)

    def test_a_dry_run_on_the_same_surface_still_reports_a_plan(self):
        """The other half. Refusing both would be an over-block: the plan is the
        useful part, and a reader can act on it deliberately."""
        _out, last = self._plugin_run(apply=False)
        self.assertTrue(last.proposed, "the dry run lost the plan as well")

    def test_every_surface_that_does_not_take_breakpoints_refuses_mutation(self):
        """Members from the registry at runtime, not from a list written here.

        Scoped honestly, because only one of them reaches the branch this
        closes. Of the four surfaces whose `control_model` is not
        `explicit_breakpoint`, three are already refused earlier for unrelated
        reasons -- two are flagged contested, one records no explicit
        breakpoints at all -- so only amazon-bedrock/converse ever got as far as
        the note. The other three would fall straight through it the day a
        contested flag is cleared, which is why this asks all of them.
        """
        from cacheeconomics import registry
        checked = []
        for t in registry.target_ids(include_contested=True):
            row = registry.target(t, allow_contested=True)
            if row.get("control_model") == "explicit_breakpoint":
                continue
            checked.append(t)
            with self.subTest(target=t):
                with self.assertRaises(tiers.Unsupported):
                    tiers.allocate(self.SEGS, self.RATES, target_id=t,
                                   model="claude-opus-5", gaps=self.GAPS,
                                   for_mutation=True)
        self.assertTrue(checked, "no non-breakpoint surfaces found; vacuous")


class TestRelocationWillNotRecommendAMechanismForAnUnnamedSurface(unittest.TestCase):
    """`relocate.propose` fabricated a surface in an or-expression.

    `scopes = [(target_id or "anthropic/direct", model)] if model else
    scopes_of(reqs)` -- so a caller who supplied `model` and omitted `target_id`
    got a first-party surface invented for them. The surface is exactly what
    `_authority_mechanism` reads to decide whether system-authority content may
    leave the system block at all, so the single-scope override answered a
    different question from the derived path on identical requests.

    Reproduced on twelve UNATTRIBUTED claude-opus-5 requests:

        propose(reqs, vol, model='claude-opus-5')
            -> [medium] cross-authority, mechanism
               'role:system message inside messages[]'
        propose(reqs, vol)
            -> [blocked], "claude-opus-5 on unknown/unattributed has no
               recorded authority-preserving relocation mechanism"

    The medium one recommends rewriting a prompt with a mechanism recorded for
    one named surface, to somebody whose surface nobody stated. `medium` is
    inside DEFAULT_APPLIED_RISKS, so it is a move the bake-off applies.
    """

    @staticmethod
    def _reqs():
        """A prompt whose volatile system header can only be freed by leaving
        the system block: it is already last among the system segments, so the
        within-container candidate recovers nothing and `_classify` falls
        through to the cross-authority branch."""
        def segs(i):
            return [Segment(id="sys", role="system", tokens=8_000, index=0,
                            label="instructions"),
                    Segment(id=f"hdr{i}", role="system", tokens=200, index=1,
                            label="session_header"),
                    Segment(id="task", role="user", tokens=20_000, index=2,
                            label="task_brief"),
                    Segment(id=f"t{i}", role="user", tokens=300, index=3,
                            label="user_turn")]
        return [Request(request_id=f"r{i}", sent_at=T0 + timedelta(seconds=60 * i),
                        model="claude-opus-5", usage={}, segments=segs(i))
                for i in range(12)]

    VOL = {0: 1, 1: 12, 2: 1, 3: 12}

    def _move(self, **kw):
        from cacheeconomics.relocate import propose
        moves, _order = propose(self._reqs(), self.VOL, **kw)
        self.assertTrue(moves, "fixture proposed nothing; the check is vacuous")
        return moves[0]

    def test_the_fixture_really_does_reach_the_cross_authority_branch(self):
        """Guard the guard. A within-container move needs no mechanism and no
        surface, so a fixture that produced one would pass every assertion
        below without ever testing the thing they are named for."""
        m = self._move(model="claude-opus-5", target_id="anthropic/direct")
        self.assertEqual("cross-authority", m.scope)

    def test_omitting_the_surface_no_longer_invents_one(self):
        m = self._move(model="claude-opus-5")
        self.assertEqual("blocked", m.risk)
        self.assertEqual("", m.mechanism)
        self.assertFalse(m.applicable)
        self.assertIn("unknown/unattributed", m.blocked_by)

    def test_the_override_now_agrees_with_the_derived_path(self):
        """The defect was a disagreement between two routes to one answer."""
        override = self._move(model="claude-opus-5")
        derived = self._move()
        self.assertEqual(derived.risk, override.risk)
        self.assertEqual(derived.mechanism, override.mechanism)

    def test_naming_the_surface_still_earns_the_recommendation(self):
        """The other direction: this must not become an over-block. A stated
        surface with a recorded mechanism still gets the move."""
        m = self._move(model="claude-opus-5", target_id="anthropic/direct")
        self.assertEqual("medium", m.risk)
        self.assertEqual("role:system message inside messages[]", m.mechanism)
        self.assertTrue(m.applicable)

    def test_a_surface_with_no_recorded_mechanism_is_still_blocked(self):
        """And the guard it was walking around still works when named."""
        m = self._move(model="claude-opus-5", target_id="amazon-bedrock/converse")
        self.assertEqual("blocked", m.risk)


class TestAnAssumedSurfaceMustNotReconcileWithoutTheCLI(unittest.TestCase):
    """The assumed-surface downgrade is a CLI patch, not a property of the analysis.

    `cmd_claude_code` re-releases the figures as DRAFT *after* `analyze` returns,
    so the protection exists only on that one path. A library caller, a script,
    or anything emitting JSON without going through the command gets
    `released_as='reconciled'` over dollars whose rate table was assumed.

    Reproduced with no CLI code in the call at all -- adapter, then analyzer,
    then an invoice equal to computed spend:

        ts = load_sessions(..., target_id='anthropic/direct',
                           surface_assumed=True)
        analyze(ts, invoice_usd=<matching>)
        -> input_usd released_as='reconciled'
           if_uncached_usd released_as='reconciled'
           caching_saved_usd released_as='reconciled'

    The evidence is present and unused: `ts.blocking_notes` already carries
    "Provider surface assumed to be anthropic/direct", and `analyze` copies it
    into `Analysis.blocking_notes` -- at analyzer.py:2123, which is 154 lines
    *after* the release label is decided at analyzer.py:1969.
    """

    def _fixture(self, tmp):
        """A trace that releases money in every category the marker walks.

        Shaped deliberately, because the first version of this fixture was
        twelve requests minutes apart and produced released figures in `spend`
        and `reconciliation` only: its findings were CAC-1 with no
        `avoidable_usd_month`, and `total_avoidable_month` was withheld for
        having no priceable parts. So the marker walked four categories and
        *exercised* two, and a fix that downgraded spend and reconciliation
        while forgetting per-finding release would still have flipped it green.

        What earns what, measured rather than assumed -- the first draft of this
        docstring credited the length for the finding category and that was
        wrong:

          - The *shape* earns the findings and total categories. 15-minute gaps
            with 5m writes and no reads mean every entry is dead before the next
            request, which makes EFF-1 fire carrying money instead of CAC-1
            reporting that caching is fine. That holds at 12 requests as well as
            at 300; the count is not what does it.
          - The *length* earns one more figure. At 300 requests the window is
            3.1 days, clearing the one-day projection floor so
            `spend.monthly_input_usd` publishes too. At 12 it is withheld by the
            floor -- correctly, and for a reason unrelated to this marker.
          - An invoice equal to computed spend, so the gate genuinely passes and
            RECONCILED is what the figures would otherwise wear.
        """
        proj = os.path.join(tmp, "proj")
        os.makedirs(proj)
        start = datetime(2026, 7, 10, 9, tzinfo=timezone.utc)
        with open(os.path.join(proj, "s.jsonl"), "w") as f:
            for i in range(300):
                at = (start + timedelta(seconds=900 * i)).strftime(
                    "%Y-%m-%dT%H:%M:%S.000Z")
                f.write(json.dumps({
                    "type": "assistant", "sessionId": "s1", "uuid": f"u{i}",
                    "requestId": f"r{i}", "timestamp": at,
                    "message": {"model": "claude-opus-5", "usage": {
                        "input_tokens": 300, "output_tokens": 40,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 150_000,
                        "cache_creation": {
                            "ephemeral_5m_input_tokens": 150_000,
                            "ephemeral_1h_input_tokens": 0}}}}) + "\n")
        return tmp

    def _analyse(self, tmp):
        """Adapter then analyzer. Deliberately no `cli` import anywhere."""
        from cacheeconomics.adapters.claude_code import load_sessions
        ts = load_sessions(root=self._fixture(tmp),
                           target_id="anthropic/direct", surface_assumed=True)
        spend = analyze(ts, allow_unreconciled=True).spend["input_usd"].raw()
        return ts, analyze(ts, invoice_usd=spend)

    def test_the_evidence_reaches_the_analysis(self):
        """Not xfail: the adapter's half works, and the fix needs no new input.

        This is the part worth pinning now, because it establishes that the
        analyzer is not missing information -- only using it too late."""
        with tempfile.TemporaryDirectory() as tmp:
            ts, a = self._analyse(tmp)
        self.assertTrue(any("surface assumed" in n.lower()
                            for n in ts.blocking_notes))
        self.assertTrue(any("surface assumed" in n.lower()
                            for n in a.blocking_notes))

    def test_the_gate_really_passes_so_reconciled_is_the_alternative(self):
        with tempfile.TemporaryDirectory() as tmp:
            _ts, a = self._analyse(tmp)
        self.assertEqual(0.0, a.reconciliation["delta_pct"])
        self.assertTrue(any(getattr(v, "released", False)
                            for v in a.spend.values()))

    # KNOWN-FAILING: assigned to Track A. `analyze` decides `released_as` at
    # analyzer.py:1969 and does not read `ts.blocking_notes` until 2123, so an
    # assumed pricing input cannot reach the release decision. Track B closed
    # this on the CLI path only (`cli._draft_because_the_surface_was_assumed`),
    # which leaves every library caller unprotected. See the spec in Track B's
    # report for exactly what `analyze` should read and where.
    #
    # expectedFailure rather than a red board, matching the convention already
    # used for the cross-track invariants: a still-broken expectation exits 0 so
    # CI stays meaningful for other tracks, and the moment it is fixed this
    # reports "Unexpected success" and forces this marker to be deleted.
    @staticmethod
    def _figures(a):
        """Every Figure the analysis publishes, as `(path, Figure)`.

        Every section, not `spend` alone. `a.reconciliation` carries
        `computed_usd` and `delta_usd`, and those are exactly the two fields
        this track's own CLI helper had to downgrade *separately* from spend --
        so a marker that watched only `spend` would go green on a fix that
        changed spend and forgot reconciliation, invite deletion, and leave
        library output publishing RECONCILED reconciliation dollars over an
        assumed surface. A handoff marker has to be at least as complete as the
        fix it gates.

        `total_avoidable_month` is a property rather than a field and is
        included deliberately: it is the most client-facing number here, and a
        `dataclasses.fields` walk skips it.
        """
        from cacheeconomics.money import Figure
        out = []
        for section, mapping in (("spend", a.spend),
                                 ("reconciliation", a.reconciliation or {})):
            for k, v in mapping.items():
                if isinstance(v, Figure):
                    out.append((f"{section}.{k}", v))
        for f in a.findings:
            # `avoidable_usd_window` is read through getattr because it exists
            # on some versions of `Finding` and not others; a marker that
            # silently stopped covering a money field the day one was added is
            # the failure mode this whole class is about.
            for field in ("avoidable_usd_month", "avoidable_usd_window"):
                v = getattr(f, field, None)
                if isinstance(v, Figure):
                    out.append((f"findings[{f.code}].{field}", v))
        if isinstance(a.total_avoidable_month, Figure):
            out.append(("total_avoidable_month", a.total_avoidable_month))
        return out

    # The categories the marker claims to cover, and how to recognise each in a
    # path. Named here so the guard below checks the claim rather than a proxy
    # for it.
    CATEGORIES = {
        "spend": lambda p: p.startswith("spend."),
        "reconciliation": lambda p: p.startswith("reconciliation."),
        "findings": lambda p: p.startswith("findings["),
        "total": lambda p: p == "total_avoidable_month",
    }

    def test_the_fixture_releases_money_in_every_category_the_marker_walks(self):
        """Guard the guard, and the guard was too weak.

        Its first version asked only for "some non-spend path" and "some
        released figure", both of which `reconciliation` satisfies alone. So it
        certified that the *walk* reached four categories while the *fixture*
        exercised two -- and a fix that forgot per-finding release would still
        have turned the marker green. Proving a walk is capable of finding
        something is not proving it was given something to find.
        """
        with tempfile.TemporaryDirectory() as tmp:
            _ts, a = self._analyse(tmp)
        released = [p for p, f in self._figures(a) if f.released]
        self.assertTrue(released, "nothing was released; the marker is vacuous")
        missing = sorted(name for name, matches in self.CATEGORIES.items()
                         if not any(matches(p) for p in released))
        self.assertEqual(
            [], missing,
            "the fixture releases no money in these categories, so the marker "
            "cannot detect a fix that forgets them: " + ", ".join(missing)
            + f"\n    released: {sorted(released)}")

    def test_an_assumed_surface_is_never_labelled_reconciled(self):
        from cacheeconomics.money import RECONCILED
        with tempfile.TemporaryDirectory() as tmp:
            _ts, a = self._analyse(tmp)
        wrong = sorted(p for p, v in self._figures(a)
                       if v.released and v.released_as == RECONCILED)
        self.assertEqual(
            [], wrong,
            "these were published as invoice-checked over an assumed rate "
            "table, without the CLI in the call: " + ", ".join(wrong))


class TestTheLoaderStatesWhichPricingInputsWereAssumed(unittest.TestCase):
    """The producer half of `assumed_inputs`, driven through the real loader.

    `analyze` caps the release at DRAFT when `ts.assumed_inputs` is non-empty.
    That consumer is useless until a loader sets the field, and for one round it
    did not exist at all: the analyzer read `getattr(ts, "assumed_inputs", ())`,
    every real adapter populated only `blocking_notes`, and a genuine
    assumed-surface trace with a matching invoice was still released as
    `reconciled`. Both sides' tests passed because both attached the attribute
    to synthetic TraceSets by hand.

    So these go through `load_sessions` against transcripts on disk. Nothing
    here constructs a TraceSet or sets the attribute itself.
    """

    def _root(self, tmp):
        proj = os.path.join(tmp, "proj")
        os.makedirs(proj)
        with open(os.path.join(proj, "s.jsonl"), "w") as f:
            for i in range(4):
                f.write(json.dumps({
                    "type": "assistant", "sessionId": "s1", "uuid": f"u{i}",
                    "requestId": f"r{i}",
                    "timestamp": f"2026-07-29T09:0{i}:00.000Z",
                    "message": {"model": "claude-opus-5", "usage": {
                        "input_tokens": 100, "output_tokens": 10,
                        "cache_read_input_tokens": 20_000,
                        "cache_creation_input_tokens": 1_000}}}) + "\n")
        return tmp

    def _cc(self, tmp, **kw):
        """The Claude Code loader specifically. Named apart from `_load`, which
        takes a loader name, because the survey below covers three loaders and a
        shared one-argument helper is how the inverse check came to cover one."""
        from cacheeconomics.adapters.claude_code import load_sessions
        return load_sessions(root=self._root(tmp),
                             target_id="anthropic/direct", **kw)

    def test_an_assumed_surface_names_itself_in_the_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            ts = self._cc(tmp, surface_assumed=True)
        self.assertTrue(ts.requests, "fixture produced no requests")
        self.assertEqual(("provider surface",), tuple(ts.assumed_inputs))

    def test_a_stated_surface_assumes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            ts = self._cc(tmp)
        self.assertTrue(ts.requests, "fixture produced no requests")
        self.assertEqual((), tuple(ts.assumed_inputs))

    def test_the_field_is_set_alongside_the_note_and_not_instead_of_it(self):
        """Both, because they serve different readers. The note is what a human
        sees in the caveats; the field is what the release gate reads. Replacing
        the note with the field would have moved the disclosure off the page."""
        with tempfile.TemporaryDirectory() as tmp:
            ts = self._cc(tmp, surface_assumed=True)
        self.assertTrue(any("surface assumed" in n.lower()
                            for n in ts.blocking_notes),
                        "the human-readable caveat was dropped")
        self.assertEqual(("provider surface",), tuple(ts.assumed_inputs))

    def test_it_names_the_input_rather_than_being_a_flag(self):
        """A tuple of names, so a second assumed input -- an effective rate, say
        -- needs no new field and the note downstream can say which one it was.
        A bool could not distinguish two assumptions with different remedies."""
        with tempfile.TemporaryDirectory() as tmp:
            ts = self._cc(tmp, surface_assumed=True)
        self.assertIsInstance(tuple(ts.assumed_inputs), tuple)
        self.assertNotIsInstance(ts.assumed_inputs, bool)
        self.assertTrue(all(isinstance(x, str) for x in ts.assumed_inputs))

    def test_a_traceset_that_predates_the_field_assumes_nothing(self):
        """Track A reads this with `getattr(ts, "assumed_inputs", ())` so an
        older object degrades to "assumed nothing". Confirmed rather than
        assumed, because hand-built and pickled TraceSets exist in these tests.

        The guarantee is stronger than the getattr default: the dataclass
        default is a class attribute, so even an instance whose own `__dict__`
        lacks the key reads `()` through ordinary attribute access.
        """
        import pickle
        from cacheeconomics.trace import Tier, TraceSet
        old = pickle.loads(pickle.dumps(TraceSet(requests=[],
                                                 tier=Tier.USAGE_ONLY)))
        old.__dict__.pop("assumed_inputs", None)
        self.assertEqual((), tuple(getattr(old, "assumed_inputs", ()) or ()))
        self.assertEqual((), tuple(old.assumed_inputs))

    # Every loader in the package, each given the same providerless row and
    # told no surface. Named here so a loader added later is a one-line entry
    # rather than a survey somebody has to remember to widen.
    #
    # `claude_code` is absent deliberately and is covered by the tests above:
    # its transcripts *structurally* cannot carry a provider field, which is why
    # it is the one loader that may assume -- and it says so.
    # The exact string a loader records when it supplied the surface itself.
    # Asserted rather than "non-empty" now that a consumer reads it: `analyze`
    # names the assumed input in the DRAFT banner, so admitting the wrong one
    # prints the wrong remedy, and admitting *any* input would let a fabricated
    # surface satisfy the rule under an unrelated name.
    SURFACE = ASSUMED_PROVIDER_SURFACE

    # Enough rows, spread far enough apart, that *every* spend figure is
    # publishable -- so an unpublished one can be attributed to the surface.
    #
    # Both numbers are load-bearing and were found by measurement, not chosen.
    # At one row per loader the monthly projection had no window at all, so
    # `monthly_input_usd` came back None and the check simply skipped it: a
    # non-surface blocker could sit there unnoticed. At 8 rows over 1.17 days it
    # existed and was withheld for "monthly figures need at least 10 requests
    # and this trace has 8" -- a second, unrelated blocker coexisting with the
    # one under test. 14 rows four hours apart clears both floors.
    ROWS = 14
    STEP_SECONDS = 4 * 3600
    START = datetime(2026, 7, 10, 9, tzinfo=timezone.utc)

    def _rows(self, name):
        """This loader's payload shape, `ROWS` of them.

        Priceable is the whole point and the first version was not. The LiteLLM
        row carried a top-level `usage` dict, which that adapter does not read
        -- it prices from `prompt_tokens_details`, `metadata.usage_object` or
        `response.usage` -- and no `startTime`. Measured: `usage` loaded as
        `{}`, `has_usage` False, `sent_at` None, 0 of 1 analysable, and the
        withheld reason came back "no invoice was supplied". So the consequence
        check passed for that loader without the surface having anything to do
        with it.

        Cache reads and no writes, deliberately: an unprovable write lifetime is
        its own blocker and would mask the one being tested.
        """
        usage = {"input_tokens": 5_000, "output_tokens": 200,
                 "cache_read_input_tokens": 20_000,
                 "cache_creation_input_tokens": 0}
        out = []
        for i in range(self.ROWS):
            at = self.START + timedelta(seconds=self.STEP_SECONDS * i)
            if name == "load_litellm":
                out.append({"model": "claude-opus-5",
                            "startTime": at.timestamp(),
                            "endTime": at.timestamp() + 2,
                            "prompt_tokens": 25_000, "completion_tokens": 200,
                            "prompt_tokens_details": {"cached_tokens": 20_000,
                                                      "text_tokens": 5_000}})
            elif name == "load_jsonl":
                out.append({"request_id": f"r{i}", "sent_at": at.isoformat(),
                            "model": "claude-opus-5", "segments": [],
                            "usage": dict(usage)})
            else:
                out.append({"request_id": f"r{i}", "sent_at": at.isoformat(),
                            "usage": dict(usage),
                            "body": {"model": "claude-opus-5",
                                     "messages": [{"role": "user",
                                                   "content": "hello there"}]}})
        return out

    def _load(self, tmp, name, row_extra=None, **surface):
        """Run one loader over its own rows. `surface` is how a caller states
        one, spelled differently per loader on purpose -- `default_target` for
        two of them and `target_id` for the third -- which is exactly the
        variation that let the inverse check cover one loader and miss two."""
        from cacheeconomics.adapters.bodies import load_bodies
        from cacheeconomics.adapters.litellm import load_litellm
        from cacheeconomics.trace import load_jsonl
        rows = [{**r, **(row_extra or {})} for r in self._rows(name)]
        path = os.path.join(tmp, f"{name}.jsonl")
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        if name == "load_litellm":
            return load_litellm(path, **surface)
        if name == "load_jsonl":
            return load_jsonl(path, **surface)
        return load_bodies(path, key=b"k" * 32, **surface)

    # How each loader is told a surface by its caller. Two spellings, which is
    # why this is a table rather than a repeated keyword.
    CALLER_ARG = {"load_litellm": "default_target",
                  "load_jsonl": "default_target",
                  "load_bodies": "target_id"}

    LOADERS = ("load_litellm", "load_jsonl", "load_bodies")

    def _every_loader(self, tmp):
        return {name: self._load(tmp, name) for name in self.LOADERS}

    def test_every_loader_leaves_a_providerless_row_unattributed(self):
        """The property, not its symptom.

        The previous version of this test asserted only that `assumed_inputs`
        came back empty -- which a loader that silently defaulted a providerless
        row to `anthropic/direct` would also satisfy, because it would not set
        the field either. It asserted the absence of a *confession* rather than
        the absence of the *thing confessed to*. Its docstring also claimed
        `bodies` was covered while the body ran `load_litellm` alone, and it
        never touched `load_jsonl` at all: three loaders named, one exercised --
        the same walk-versus-fixture gap as the marker above.
        """
        from cacheeconomics.trace import UNATTRIBUTED
        with tempfile.TemporaryDirectory() as tmp:
            loaders = self._every_loader(tmp)
        self.assertEqual(3, len(loaders), "a loader dropped out of the survey")
        for name, ts in loaders.items():
            with self.subTest(loader=name):
                self.assertTrue(ts.requests,
                                f"{name} produced no requests; the check is vacuous")
                surfaces = {r.target_id for r in ts.requests}
                self.assertEqual(
                    {UNATTRIBUTED}, surfaces,
                    f"{name} supplied a surface nobody read or stated: {surfaces}")

    def test_each_survey_row_is_priceable_but_for_the_surface(self):
        """Guard the guard, and this one had already gone wrong.

        "No dollars were published" is true of a row nobody could price for any
        reason -- a missing timestamp, an unreadable usage shape, an unprovable
        write lifetime. The claim being made is narrower: this row could have
        been priced, and the surface is what stopped it. So every row has to
        reach the analyser intact first.
        """
        with tempfile.TemporaryDirectory() as tmp:
            loaders = self._every_loader(tmp)
        for name, ts in loaders.items():
            with self.subTest(loader=name):
                a = analyze(ts, allow_unreconciled=True)
                self.assertTrue(ts.requests, f"{name}: no requests parsed")
                self.assertEqual(
                    len(ts.requests), len(ts.analysable),
                    f"{name}: the row never reached the analyser, so anything "
                    f"below it holds for a reason unrelated to the surface")
                self.assertEqual(1.0, a.coverage["fraction"], f"{name} coverage")
                self.assertTrue(all(r.sent_at for r in ts.requests),
                                f"{name}: undated rows are unpriceable anyway")
                # Every money figure has to actually EXIST, or the checks below
                # skip it and pass. This is the hole that survived the first fix
                # of it: at one row per loader the projection path never ran, so
                # `monthly_input_usd` was None rather than a withheld Figure,
                # `hasattr(v, "released")` filtered it out, and a non-surface
                # blocker could sit there unseen. Reverting ROWS to 1 passed
                # every other assertion in this class.
                figures = {k for k, v in a.spend.items()
                           if hasattr(v, "released")}
                self.assertEqual(
                    {"input_usd", "if_uncached_usd", "caching_saved_usd",
                     "monthly_input_usd"}, figures,
                    f"{name}: the fixture did not produce every spend figure, "
                    f"so the checks below silently skip the missing ones")

    def test_the_same_rows_release_every_figure_once_a_surface_is_stated(self):
        """The symmetric half, and the one that makes the claim provable.

        "Nothing was released" is weak on its own: it is also true of a trace
        nobody could price for a dozen unrelated reasons. Pairing it with "the
        identical rows release *everything* the moment a surface is stated"
        pins the surface as the single variable, and pins it over every spend
        figure rather than the one this test happened to look at.

        Which mattered: the earlier version inspected `input_usd` alone, and
        `monthly_input_usd` sat beside it withheld for "monthly figures need at
        least 10 requests" -- a second blocker the check could not see.
        """
        with tempfile.TemporaryDirectory() as tmp:
            for name in self.LOADERS:
                arg = self.CALLER_ARG[name]
                with self.subTest(loader=name):
                    stated = analyze(self._load(tmp, name,
                                                **{arg: "anthropic/direct"}),
                                     allow_unreconciled=True)
                    figures = {k: v for k, v in stated.spend.items()
                               if hasattr(v, "released")}
                    self.assertTrue(figures, f"{name}: no spend figures at all")
                    withheld = sorted(k for k, v in figures.items()
                                      if not v.released)
                    self.assertEqual(
                        [], withheld,
                        f"{name}: these stayed withheld with a stated surface, "
                        f"so the survey cannot attribute their absence to the "
                        f"surface: {withheld}")

    def test_and_withholds_every_one_of_them_for_the_surface_when_it_is_not(self):
        """The other half of the pair, over every figure and naming the cause.

        Asserting only that nothing was released let the LiteLLM row pass while
        it was in fact unreadable: `usage` came back `{}`, `sent_at` None, 0 of
        1 analysable, and the stated reason was "no invoice was supplied".
        """
        with tempfile.TemporaryDirectory() as tmp:
            loaders = self._every_loader(tmp)
        for name, ts in loaders.items():
            with self.subTest(loader=name):
                a = analyze(ts, allow_unreconciled=True)
                figures = {k: v for k, v in a.spend.items()
                           if hasattr(v, "released")}
                self.assertTrue(figures, f"{name}: no spend figures at all")
                released = sorted(k for k, v in figures.items() if v.released)
                self.assertEqual(
                    [], released,
                    f"{name} published dollars over an unattributed surface: "
                    + ", ".join(released))
                wrong = sorted(
                    k for k, v in figures.items()
                    if "surface is not stated" not in (v.withheld_because or ""))
                self.assertEqual(
                    [], wrong,
                    f"{name}: these were withheld for some other reason, so a "
                    f"non-surface blocker is hiding inside this check: "
                    + ", ".join(f"{k} ({figures[k].withheld_because[:60]!r})"
                                for k in wrong))

    def test_a_loader_that_supplies_a_surface_has_to_admit_it(self):
        """The rule stated positively, over the same fixture.

        A loader may end up with a surface in one of three ways: read from the
        row, passed by the caller, or supplied by itself. Only the third is an
        assumption, and only the third has to appear in `assumed_inputs`. So:
        given a providerless row and no caller-supplied surface, any loader
        whose requests come back with a real surface must have assumed it, and
        must say so.

        This is what makes the survey bite. If `load_jsonl` starts defaulting
        `default_target` to `anthropic/direct` again -- the defect this package
        has already shipped twice -- this fails whether or not it sets the field.
        """
        from cacheeconomics.trace import UNATTRIBUTED
        with tempfile.TemporaryDirectory() as tmp:
            loaders = self._every_loader(tmp)
        for name, ts in loaders.items():
            with self.subTest(loader=name):
                supplied = {r.target_id for r in ts.requests} - {UNATTRIBUTED}
                if supplied:
                    # The exact name, not merely a non-empty tuple. A consumer
                    # prints it as the remedy, so admitting a fabricated surface
                    # under some other input's name would satisfy a truthiness
                    # check and still tell the reader the wrong thing to fix.
                    self.assertIn(
                        self.SURFACE,
                        tuple(getattr(ts, "assumed_inputs", ()) or ()),
                        f"{name} supplied {sorted(supplied)} for a row that "
                        f"named no provider, and did not record it as an "
                        f"assumed {self.SURFACE!r}")

    def test_a_caller_stated_surface_is_not_an_assumption_in_any_loader(self):
        """The inverse, over every loader that accepts one.

        It ran through `load_jsonl` alone, so `load_litellm(default_target=...)`
        and `load_bodies(target_id=...)` were free to mark a caller-stated
        surface as assumed -- and once a consumer gates on this field, that
        downgrades a legitimate report to DRAFT. The two spellings of the
        argument are exactly why one loader got checked and two did not, so the
        spelling now comes from a table.
        """
        with tempfile.TemporaryDirectory() as tmp:
            for name, arg in self.CALLER_ARG.items():
                with self.subTest(loader=name, arg=arg):
                    ts = self._load(tmp, name, **{arg: "anthropic/direct"})
                    self.assertEqual({"anthropic/direct"},
                                     {r.target_id for r in ts.requests},
                                     f"{name} ignored a caller-stated surface")
                    self.assertEqual(
                        (), tuple(getattr(ts, "assumed_inputs", ()) or ()),
                        f"{name} recorded a caller-stated surface as assumed, "
                        f"which would downgrade a correct report to DRAFT")

    # Every route by which a surface reaches a request *from the data*, and the
    # surface each one should yield. Enumerated from the loaders before the test
    # was written, rather than extended each time a review named another one:
    #
    #   `litellm.target_from_row` has two -- the `custom_llm_provider` field,
    #   and the routing prefix on `model` when that field is absent.
    #   `trace.request_from_row` has one, a row-level `target_id`, and it serves
    #   both `load_jsonl` and `load_bodies`, so each is exercised separately
    #   because they reach it by different paths.
    #
    # Three rounds running, the fix covered the route being looked at. This is
    # the table that stops that: a fourth route is a row here, and a route that
    # starts reporting itself as assumed fails whichever one it is.
    ROW_STATED_ROUTES = (
        ("load_litellm", {"custom_llm_provider": "bedrock"},
         "amazon-bedrock/converse"),
        ("load_litellm", {"model": "anthropic/claude-opus-5"},
         "anthropic/direct"),
        ("load_jsonl", {"target_id": "amazon-bedrock/converse"},
         "amazon-bedrock/converse"),
        ("load_bodies", {"target_id": "amazon-bedrock/converse"},
         "amazon-bedrock/converse"),
    )

    def test_a_row_stated_surface_is_not_an_assumption_by_any_route(self):
        """The third way a surface arrives: read from the data itself.

        A regression marking any of these as assumed downgrades a legitimate
        report to DRAFT once a consumer gates on the field -- and the previous
        version of this test covered one of the four.
        """
        with tempfile.TemporaryDirectory() as tmp:
            for name, extra, expected in self.ROW_STATED_ROUTES:
                route = ",".join(sorted(extra))
                with self.subTest(loader=name, route=route):
                    ts = self._load(tmp, name, row_extra=extra)
                    self.assertEqual(
                        {expected}, {r.target_id for r in ts.requests},
                        f"{name} did not read the surface from {route}")
                    self.assertEqual(
                        (), tuple(getattr(ts, "assumed_inputs", ()) or ()),
                        f"{name} called a surface it read from {route} an "
                        f"assumption, which would downgrade a correct report")

    def test_the_route_table_covers_every_loader_that_can_read_one(self):
        """Guard the guard: a table is only as good as its coverage, and this
        one is the fix for a test that covered a quarter of the routes."""
        covered = {name for name, _extra, _exp in self.ROW_STATED_ROUTES}
        self.assertEqual(set(self.LOADERS), covered,
                         "a loader that can read a surface from its rows is "
                         "missing from the route table")
        self.assertGreaterEqual(
            sum(1 for n, _e, _x in self.ROW_STATED_ROUTES if n == "load_litellm"),
            2, "litellm has two routes -- the provider field and the model "
               "prefix -- and both have to be exercised")

    def test_the_admission_names_the_surface_and_only_the_surface(self):
        """The exact contract, on the one loader that legitimately assumes.

        `analyze` renders this into "the <input> used to price this trace was
        assumed", so the string is user-visible and a consumer keys on it. A
        second entry appearing here would be a second assumption nobody made.
        """
        with tempfile.TemporaryDirectory() as tmp:
            ts = self._cc(tmp, surface_assumed=True)
        self.assertEqual((self.SURFACE,), tuple(ts.assumed_inputs))

    def test_the_default_survives_a_field_written_with_a_factory(self):
        """Why `assumed_inputs` is a plain default and not `default_factory`.

        Track A reads it with `getattr(ts, "assumed_inputs", ())`, and a
        TraceSet built before the field existed has no such key in its
        `__dict__`. A plain default leaves the value on the *class*, so every
        such object reads `()` through ordinary attribute access -- checked
        across pickle round-trips, a stripped legacy pickle, `copy`, `deepcopy`,
        `dataclasses.replace` and a re-pickle of a stripped one.

        `field(default_factory=tuple)` would look equivalent and is not: it
        leaves no class attribute, so a legacy object raises AttributeError on
        plain access. Pinned because the two spellings are interchangeable
        everywhere except here.
        """
        import dataclasses as dc

        from cacheeconomics.trace import Tier, TraceSet
        f = next(x for x in dc.fields(TraceSet) if x.name == "assumed_inputs")
        self.assertIs(dc.MISSING, f.default_factory,
                      "a default_factory leaves no class attribute, so a "
                      "TraceSet predating this field would raise on access")
        self.assertEqual((), getattr(TraceSet, "assumed_inputs"))
        legacy = TraceSet(requests=[], tier=Tier.USAGE_ONLY,
                          blocking_notes=["Provider surface assumed to be x"])
        legacy.__dict__.pop("assumed_inputs")
        self.assertEqual((), legacy.assumed_inputs)
        self.assertEqual((), dc.replace(legacy, source="x").assumed_inputs)


class TestABooleanMultiplierIsRefusedRatherThanPricedAtOneX(unittest.TestCase):
    """`bool` subclasses `int`, so every numeric guard on a multiplier let it in.

    `isinstance(True, (int, float))` is True, so a hand-edited or corrupted
    registry row carrying `write_5m: true` passed the type check in `cost.price`
    and was multiplied straight into the figure as 1.0x. Measured on
    anthropic/direct, one million 5m-write tokens:

        write_5m = True  ->  $5.00
        write_5m = 1.25  ->  $6.25          a silent 20% understatement

    1.0x is a plausible-looking number, so nothing downstream reads as broken --
    which is what makes it worse than a crash. `read: true` is the same defect
    pointing the other way: 1.0x instead of 0.1x makes a cache read cost the
    same as fresh input, so the allocator scores every plan worse than uncached
    and places no marker at all.

    Found by Track A closing it in the analyzer's four consumers and then
    AST-walking the package for the rest. `cost.py` had two and `tiers.py` a
    third -- the module the analyzer *delegates pricing to*, which is the
    partial-closure shape this round has been about. `cost.price` is the serious
    one: every priced figure in the package goes through it.

    `cost.py` already excluded `bool` for `effective_rate`, one branch away, for
    the same reason and with the same consequence. The rule was known here; it
    just was not applied to the multipliers beside it.
    """

    KEYS = ("read", "write_5m", "write_1h")

    def setUp(self):
        from cacheeconomics import registry
        self.registry = registry
        self.real = dict(registry.multipliers("anthropic/direct"))
        self._orig = registry.multipliers

    def tearDown(self):
        self.registry.multipliers = self._orig

    def _poison(self, key, value):
        m = {**self.real, key: value}
        self.registry.multipliers = lambda _t, _m=m: _m

    def test_the_predicate_rejects_bool_and_accepts_numbers(self):
        self.assertFalse(cost.is_multiplier(True))
        self.assertFalse(cost.is_multiplier(False))
        self.assertFalse(cost.is_multiplier(None))
        self.assertFalse(cost.is_multiplier("1.25"))
        self.assertTrue(cost.is_multiplier(1.25))
        self.assertTrue(cost.is_multiplier(2))
        # Zero was accepted when this predicate rejected only bools. It prices a
        # cache write as free, which is a claim no registry row should be able
        # to make by accident, and no recorded row does -- every multiplier in
        # the registry is 0.1 or above.
        self.assertFalse(cost.is_multiplier(0))
        self.assertFalse(cost.is_multiplier(0.0))

    def test_the_hole_is_real_in_the_language(self):
        """The reason this needs a named predicate at all: the obvious guard is
        wrong, and reads as correct."""
        self.assertIsInstance(True, int)
        self.assertTrue(isinstance(True, (int, float)))

    def test_price_refuses_a_boolean_multiplier(self):
        usage = cost.Usage(uncached_input=0, cache_read=0,
                           cache_write_5m=1_000_000)
        for key in self.KEYS:
            with self.subTest(multiplier=key):
                self._poison(key, True)
                with self.assertRaises(self.registry.RegistryError) as e:
                    cost.price(usage, "claude-opus-5",
                               target_id="anthropic/direct",
                               on_date="2026-07-29")
                self.assertIn(key, str(e.exception))

    def test_and_would_otherwise_have_understated_by_twenty_percent(self):
        """The measured consequence, pinned so the number is not just prose."""
        usage = cost.Usage(uncached_input=0, cache_read=0,
                           cache_write_5m=1_000_000)
        correct = cost.price(usage, "claude-opus-5",
                             target_id="anthropic/direct",
                             on_date="2026-07-29").usd
        self.assertAlmostEqual(6.25, correct, places=6)
        # 1.0x is what `True` would have multiplied by.
        as_one_x = correct / self.real["write_5m"]
        self.assertAlmostEqual(5.0, as_one_x, places=6)

    def test_ttl_crossover_refuses_one(self):
        for key in self.KEYS:
            with self.subTest(multiplier=key):
                self._poison(key, True)
                with self.assertRaises(self.registry.RegistryError):
                    cost.ttl_crossover("anthropic/direct")

    def test_the_allocator_refuses_a_boolean_read_multiplier(self):
        """`read: true` scores a cache read at the price of fresh input, so
        every plan loses to uncached and the allocator silently places nothing.
        A refusal names the surface; a 1.0x read looks like a workload that
        cannot benefit from caching."""
        self._poison("read", True)
        with self.assertRaises(tiers.Unsupported) as e:
            tiers._surface("anthropic/direct", "claude-opus-5")
        self.assertIn("read multiplier", str(e.exception))

    def test_a_present_but_unusable_write_multiplier_is_refused(self):
        """Absent and corrupt are different facts and must not share a path.

        This previously took the *skip* path, so `write_5m: true` beside a
        numeric `write_1h` returned a one-lifetime rate map and the allocator
        went on to recommend 1h-only plans -- with nothing anywhere reporting
        that the 5m price input was corrupt. Measured before the fix:

            write_5m ABSENT -> {'1h': 2.0}      (correct: a gap in the registry)
            write_5m=True   -> {'1h': 2.0}      (wrong: identical, and it is a
                                                 fault in the data, not a gap)
            write_5m=NaN    -> {'5m': nan, ...} (worse: it entered the search)

        An absent premium is a fact about the surface. A corrupt one is a fact
        about the data, and only the second should stop the run.
        """
        for value in (True, float("nan"), float("inf"), -1.0, 0.0):
            with self.subTest(write_5m=value):
                self._poison("write_5m", value)
                with self.assertRaises(tiers.Unsupported) as e:
                    tiers._surface("anthropic/direct", "claude-opus-5")
                self.assertIn("unusable write multipliers", str(e.exception))
                self.assertIn("write_5m", str(e.exception))

    def test_but_a_genuinely_absent_one_still_only_drops_that_tier(self):
        """The deliberate half, kept and now on its own fixture rather than
        sharing the corrupt one. A surface may advertise a lifetime the registry
        records no premium for; that tier is unavailable to the plan and the
        rest of the plan is still sound."""
        for label, m in (("key absent",
                          {k: v for k, v in self.real.items() if k != "write_5m"}),
                         ("explicit null", {**self.real, "write_5m": None})):
            with self.subTest(case=label):
                self.registry.multipliers = lambda _t, _m=m: _m
                _budget, rates, _read = tiers._surface("anthropic/direct",
                                                       "claude-opus-5")
                self.assertEqual({"1h": self.real["write_1h"]}, rates)

    def test_and_refuses_outright_when_no_lifetime_survives(self):
        self.registry.multipliers = lambda _t: {
            k: v for k, v in self.real.items()
            if k not in ("write_5m", "write_1h")}
        with self.assertRaises(tiers.Unsupported) as e:
            tiers._surface("anthropic/direct", "claude-opus-5")
        self.assertIn("no matching write", str(e.exception))

    # --- the rule this predicate was copied from, and copied wrong ----------

    HOSTILE = (float("nan"), float("inf"), float("-inf"), -1.0, 0.0, True,
               False, None, "1.25")

    def test_it_rejects_everything_that_is_not_a_finite_positive_number(self):
        for v in self.HOSTILE:
            with self.subTest(value=repr(v)):
                self.assertFalse(cost.is_multiplier(v))
        for v in (0.1, 1.25, 2, 2.0, 1e-6):
            with self.subTest(value=repr(v)):
                self.assertTrue(cost.is_multiplier(v))

    def test_the_json_constants_are_the_reachable_half_of_that(self):
        """NaN and Infinity are not exotic: `json.loads` accepts both literals
        by default, so a hand-edited registry file reaches the process carrying
        them without anything being malformed."""
        self.assertTrue(math.isnan(json.loads('{"x": NaN}')["x"]))
        self.assertTrue(math.isinf(json.loads('{"x": Infinity}')["x"]))

    def test_price_refuses_every_hostile_multiplier(self):
        usage = cost.Usage(uncached_input=0, cache_read=0,
                           cache_write_5m=1_000_000)
        for key in self.KEYS:
            for v in self.HOSTILE:
                with self.subTest(multiplier=key, value=repr(v)):
                    self._poison(key, v)
                    with self.assertRaises(self.registry.RegistryError):
                        cost.price(usage, "claude-opus-5",
                                   target_id="anthropic/direct",
                                   on_date="2026-07-29")

    def test_none_of_them_can_reach_a_published_figure(self):
        """The consequence, stated as the numbers that used to come out.

        Measured before the fix, `write_5m` poisoned, one million 5m-write
        tokens: NaN gave `usd=nan` (and `--format json` emits a bare NaN, which
        is not valid JSON), Infinity gave `inf`, -1.0 gave -$5.00, and 0.0 made
        the write free.
        """
        usage = cost.Usage(uncached_input=0, cache_read=0,
                           cache_write_5m=1_000_000)
        for v in (float("nan"), float("inf"), -1.0, 0.0):
            with self.subTest(write_5m=repr(v)):
                self._poison("write_5m", v)
                with self.assertRaises(self.registry.RegistryError):
                    cost.price(usage, "claude-opus-5",
                               target_id="anthropic/direct",
                               on_date="2026-07-29")

    def test_the_predicate_agrees_with_the_effective_rate_guard(self):
        """The twin-path check, and the one that would have caught this.

        `is_multiplier` exists *because* `effective_rate` fifty lines below
        already refused a bool for the same reason. Only the bool third of that
        rule was carried across: `effective_rate` rejects bool AND non-finite
        AND non-positive, and this rejected bool.

        Two guards over the same quantity -- a number every token count is
        multiplied by -- have to agree, so they are now compared directly
        instead of being written twice and trusted to match.
        """
        usage = cost.Usage(uncached_input=1_000)
        # `None` is excluded, and the reason is the one interesting asymmetry
        # this comparison turned up: to `effective_rate` it is the *sentinel for
        # "not supplied"*, so `price` skips the override entirely and the guard
        # never runs. To a multiplier it is a value in the map -- `read: null`
        # is a real row on openai/direct -- and must be refused. Same literal,
        # two meanings, so forcing the two guards to agree about it would be
        # wrong in one of them.
        compared = tuple(v for v in self.HOSTILE if v is not None)
        for v in compared + (0.1, 1.25, 2.0):
            with self.subTest(value=repr(v)):
                try:
                    cost.price(usage, "claude-opus-5",
                               target_id="anthropic/direct",
                               on_date="2026-07-29", effective_rate=v)
                    rate_ok = True
                except (ValueError, TypeError):
                    rate_ok = False
                self.assertEqual(
                    rate_ok, cost.is_multiplier(v),
                    f"the two guards over a multiplied rate disagree about "
                    f"{v!r}: effective_rate {'accepts' if rate_ok else 'refuses'}"
                    f" it, is_multiplier says {cost.is_multiplier(v)}")

    def test_every_consumer_of_the_registry_multipliers_validates(self):
        """Discovered by AST walk, not by a list written here.

        Track A found the two in `cost.py` this way after fixing four in the
        analyzer, which is the only reason they were found at all. The same walk
        runs here so a consumer added later is covered by construction rather
        than by somebody remembering this class exists.

        The analyzer's four are checked by name because they delegate to its own
        `_lifetime_multipliers`, which lives in a file this track does not edit.
        """
        import ast
        import pathlib
        pkg = pathlib.Path(cost.__file__).parent
        found = []
        for path in sorted(pkg.rglob("*.py")):
            tree = ast.parse(path.read_text(), filename=str(path))
            parents = {c: p for p in ast.walk(tree)
                       for c in ast.iter_child_nodes(p)}
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                f = node.func
                if not (isinstance(f, ast.Attribute) and f.attr == "multipliers"
                        and isinstance(f.value, ast.Name)
                        and f.value.id == "registry"):
                    continue
                fn, p = "<module>", node
                while p in parents:
                    p = parents[p]
                    if isinstance(p, ast.FunctionDef):
                        fn, src = p.name, ast.unparse(p)
                        break
                else:
                    src = ""
                found.append((path.name, fn, src))
        self.assertTrue(found, "no consumers discovered; this check is vacuous")
        unguarded = [
            f"{mod}:{fn}" for mod, fn, src in found
            if not ("is_multiplier" in src or "unusable_multipliers" in src
                    or "_lifetime_multipliers" in src)]
        self.assertEqual(
            [], unguarded,
            "these read registry.multipliers without validating the values, so "
            "a boolean prices at 1.0x: " + ", ".join(unguarded))
