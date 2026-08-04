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
from cacheeconomics.trace import UNATTRIBUTED, Request, Segment, Tier, TraceSet  # noqa: E402

T0 = datetime(2026, 7, 29, 9, tzinfo=timezone.utc)
UNKNOWN_MODEL = "nobody-has-ever-registered-this"
UNKNOWN_TARGET = "nobody/registered-this-surface"


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
                                   at=T0 + timedelta(seconds=90 * i))
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
                                       segments=segs, session="s"))
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
        which is why the marking exists at all rather than widening the lot."""
        from cacheeconomics.analyzer import analyze
        for segments in (True, False):
            for target in (self.BEDROCK, "anthropic/direct"):
                with self.subTest(segments=segments, target=target):
                    analyze(self._trace(target, segments),
                            allow_unreconciled=True)

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

    def _trace(self, assumed=("surface",), blocking=False):
        reqs = [Request(request_id=f"r{i}",
                        sent_at=T0 + timedelta(hours=6 * i),
                        model="claude-opus-5", target_id="anthropic/direct",
                        tenant="t", session="s", agent="a", ttl_requested="5m",
                        usage={"input_tokens": 0, "cache_read_input_tokens": 0,
                               "cache_creation_input_tokens": 100_000},
                        segments=[])
                for i in range(12)]
        ts = TraceSet(requests=reqs, tier=Tier.USAGE_ONLY,
                      notes=[self.NOTE] if (assumed or blocking) else [],
                      blocking_notes=[self.NOTE] if blocking else [])
        if assumed:
            # Set dynamically: the field belongs to `trace.py`, which this track
            # does not own, and `analyze` reads it with `getattr` for the same
            # reason. A loader that never sets it assumed nothing.
            ts.assumed_inputs = tuple(assumed)
        return ts

    def _analysis(self, assumed=("surface",), blocking=False):
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
        self.assertIn("surface", banners[0])
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
        """
        from cacheeconomics import money
        ts = self._trace(assumed=("surface",))
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
        self.assertNotIn("surface", banner)


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
            ts = load_sessions(root=self._fixture(tmp))
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
                ("volatile prefix", self._volatile()),
                ("demo capture", self._demo())]

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
