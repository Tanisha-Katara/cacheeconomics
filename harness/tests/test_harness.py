"""Tests for the registry, cost model and checks.

Stdlib unittest, no pytest dependency. Run: python3 -m unittest discover tests
"""

import os
import sys
import unittest
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cacheeconomics import checks, cost, registry  # noqa: E402
from cacheeconomics.checks import Status  # noqa: E402


class TestRegistry(unittest.TestCase):

    def test_contested_row_refuses_by_default(self):
        """The whole point of the contested flag is that it blocks."""
        with self.assertRaises(registry.ContestedRow):
            registry.target("openai/bedrock")
        row = registry.target("openai/bedrock", allow_contested=True)
        self.assertTrue(row["provenance"]["contested"])

    def test_contested_rows_hidden_from_default_listing(self):
        self.assertNotIn("openai/bedrock", registry.target_ids())
        self.assertIn("openai/bedrock", registry.target_ids(include_contested=True))

    def test_minimums_are_non_monotonic(self):
        """Newer is not always lower. This is why they cannot be inferred."""
        t = "anthropic/direct"
        self.assertEqual(registry.min_cacheable_tokens(t, "claude-opus-5"), 512)
        self.assertEqual(registry.min_cacheable_tokens(t, "claude-opus-4-8"), 1024)
        self.assertEqual(registry.min_cacheable_tokens(t, "claude-opus-4-7"), 2048)
        self.assertEqual(registry.min_cacheable_tokens(t, "claude-opus-4-6"), 4096)
        self.assertEqual(registry.min_cacheable_tokens(t, "claude-haiku-4-5"), 4096)

    def test_unknown_minimum_raises_rather_than_guessing(self):
        with self.assertRaises(registry.RegistryError):
            registry.min_cacheable_tokens("anthropic/direct", "claude-not-a-model")

    def test_pricing_is_date_effective(self):
        self.assertEqual(registry.base_rate("claude-sonnet-5", "2026-07-29", "anthropic/direct"), 2.00)
        self.assertEqual(registry.base_rate("claude-sonnet-5", "2026-08-31", "anthropic/direct"), 2.00)
        self.assertEqual(registry.base_rate("claude-sonnet-5", "2026-09-01", "anthropic/direct"), 3.00)
        self.assertEqual(registry.base_rate("claude-sonnet-5", "2027-06-01", "anthropic/direct"), 3.00)

    def test_upcoming_rate_change_is_surfaced(self):
        up = registry.upcoming_rate_change("claude-sonnet-5", "2026-07-29")
        self.assertEqual(up, {"effective": "2026-09-01", "rate": 3.00})
        self.assertIsNone(registry.upcoming_rate_change("claude-sonnet-5", "2026-09-02"))

    def test_bedrock_lacks_automatic_caching(self):
        self.assertFalse(registry.capability("amazon-bedrock/converse", "automatic_prefix_cache"))
        self.assertTrue(registry.capability("google-cloud/vertex", "automatic_prefix_cache"))

    def test_every_published_row_carries_provenance(self):
        for tid in registry.target_ids():
            p = registry.target(tid)["provenance"]
            self.assertIn("checked_on", p, f"{tid} has no checked_on date")
            self.assertIn("confidence", p, f"{tid} has no confidence")


class TestCost(unittest.TestCase):

    def test_caching_can_cost_more_than_not_caching(self):
        """A write nothing reads is strictly worse than no cache."""
        u = cost.Usage(cache_write_5m=20000)
        s = cost.price(u, "claude-sonnet-4-6", "anthropic/direct", on_date="2026-07-29")
        self.assertLess(s.saving_vs_uncached, 0)
        self.assertAlmostEqual(s.usd, 20000 * 3.0 / 1e6 * 1.25, places=9)

    def test_reads_are_cheap(self):
        u = cost.Usage(cache_read=20000)
        s = cost.price(u, "claude-sonnet-4-6", "anthropic/direct", on_date="2026-07-29")
        self.assertGreater(s.saving_pct, 89)

    def test_effective_rate_overrides_list(self):
        u = cost.Usage(cache_read=1_000_000)
        listed = cost.price(u, "claude-sonnet-4-6", "anthropic/direct", on_date="2026-07-29")
        negotiated = cost.price(u, "claude-sonnet-4-6", "anthropic/direct", on_date="2026-07-29", effective_rate=1.50)
        self.assertAlmostEqual(negotiated.usd, listed.usd / 2, places=9)
        self.assertIn("invoice", negotiated.breakdown["rate_source"])

    def test_from_anthropic_requires_explicit_ttl(self):
        raw = {"input_tokens": 6, "cache_read_input_tokens": 0,
               "cache_creation_input_tokens": 16429}
        u5 = cost.Usage.from_anthropic(raw, ttl="5m")
        u1 = cost.Usage.from_anthropic(raw, ttl="1h")
        self.assertEqual(u5.cache_write_5m, 16429)
        self.assertEqual(u1.cache_write_1h, 16429)
        with self.assertRaises(ValueError):
            cost.Usage.from_anthropic(raw, ttl="1 hour")

    def test_omitting_ttl_is_an_error_not_a_default(self):
        """A default would underprice 1h writes by 60%, silently and flatteringly."""
        raw = {"cache_creation_input_tokens": 16429}
        with self.assertRaises(ValueError):
            cost.Usage.from_anthropic(raw)
        with self.assertRaises(TypeError):
            cost.Usage.from_anthropic(raw, "1h")   # positional is refused too

    def test_no_ttl_needed_when_there_are_no_writes(self):
        """The multiplier lands on zero, so silence costs nothing here."""
        u = cost.Usage.from_anthropic({"input_tokens": 40, "cache_read_input_tokens": 900})
        self.assertEqual(u.cache_read, 900)

    def test_cache_creation_breakdown_is_preferred_over_the_ttl_argument(self):
        """Real responses state the split outright:

            "cache_creation": {"ephemeral_5m_input_tokens": 0,
                               "ephemeral_1h_input_tokens": 14979}

        When the provider says which lifetime it wrote, nothing is inferred.
        """
        raw = {"cache_creation_input_tokens": 14979,
               "cache_creation": {"ephemeral_5m_input_tokens": 0,
                                  "ephemeral_1h_input_tokens": 14979}}
        u = cost.Usage.from_anthropic(raw)          # no ttl supplied at all
        self.assertEqual(u.cache_write_1h, 14979)
        self.assertEqual(u.cache_write_5m, 0)

    def test_mixed_lifetime_writes_in_one_request_are_represented_exactly(self):
        """A scalar plus a single `ttl` cannot express this at all.

        One request can write both lifetimes. Collapsing it to one TTL prices
        part of it wrong no matter which one is chosen.
        """
        raw = {"cache_creation_input_tokens": 3000,
               "cache_creation": {"ephemeral_5m_input_tokens": 1000,
                                  "ephemeral_1h_input_tokens": 2000}}
        u = cost.Usage.from_anthropic(raw)
        self.assertEqual((u.cache_write_5m, u.cache_write_1h), (1000, 2000))

    def test_breakdown_disagreeing_with_the_scalar_is_refused(self):
        """Both come from the same response; disagreement means one is misread."""
        raw = {"cache_creation_input_tokens": 5000,
               "cache_creation": {"ephemeral_5m_input_tokens": 1000,
                                  "ephemeral_1h_input_tokens": 2000}}
        with self.assertRaises(ValueError):
            cost.Usage.from_anthropic(raw)


class TestModelIdNormalization(unittest.TestCase):
    """Real traces carry date-suffixed ids next to bare ones."""

    def test_date_suffix_is_stripped_for_a_known_model(self):
        self.assertEqual(registry.normalize_model("claude-haiku-4-5-20251001"),
                         ("claude-haiku-4-5", "20251001"))

    def test_bare_ids_are_untouched(self):
        self.assertEqual(registry.normalize_model("claude-opus-5"),
                         ("claude-opus-5", None))

    def test_an_unknown_model_is_not_rewritten_into_a_known_one(self):
        """Stripping must not invent a match the registry never recorded."""
        self.assertEqual(registry.normalize_model("claude-nonexistent-9-20251001"),
                         ("claude-nonexistent-9-20251001", None))

    def test_base_rate_still_refuses_unnormalised_ids(self):
        """Normalisation is a deliberate call, not a silent fallback in lookup.

        A lookup that rewrites its own argument cannot distinguish a known model
        with a date suffix from a genuinely unknown one.
        """
        with self.assertRaises(registry.RegistryError):
            registry.base_rate("claude-haiku-4-5-20251001", "2026-07-29", "anthropic/direct")

    def test_one_hour_creation_prices_at_2x_not_1_25x(self):
        raw = {"cache_creation_input_tokens": 1_000_000}
        u = cost.Usage.from_anthropic(raw, ttl="1h")
        s = cost.price(u, "claude-sonnet-4-6", "anthropic/direct", on_date="2026-07-29")
        self.assertEqual(u.cache_write_1h, 1_000_000)
        self.assertEqual(u.cache_write_5m, 0)
        self.assertAlmostEqual(s.usd, 3.0 * 2.0, places=9)
        mispriced = cost.price(cost.Usage(cache_write_5m=1_000_000),
                               "claude-sonnet-4-6", "anthropic/direct", on_date="2026-07-29")
        self.assertAlmostEqual(s.usd / mispriced.usd, 1.6, places=9)

    def test_reproduces_the_measured_run(self):
        """The published 40.2% must fall out of this model, not be asserted.

        Figures from run 20260728T194028Z: 16,429-token prefix, 6 uncached
        tokens per call, four probes per arm.
        """
        P, U = 16429, 6
        arm5 = [cost.Usage(uncached_input=U, cache_write_5m=P),
                cost.Usage(uncached_input=U, cache_write_5m=P),
                cost.Usage(uncached_input=U, cache_read=P),
                cost.Usage(uncached_input=U, cache_write_5m=P)]
        arm1 = [cost.Usage(uncached_input=U, cache_write_1h=P),
                cost.Usage(uncached_input=U, cache_read=P),
                cost.Usage(uncached_input=U, cache_read=P),
                cost.Usage(uncached_input=U, cache_read=P)]
        t5 = sum(cost.price(u, "claude-sonnet-4-6", "anthropic/direct", on_date="2026-07-28").usd for u in arm5)
        t1 = sum(cost.price(u, "claude-sonnet-4-6", "anthropic/direct", on_date="2026-07-28").usd for u in arm1)
        self.assertAlmostEqual(t5, 0.18983, places=5)
        self.assertAlmostEqual(t1, 0.11343, places=5)
        self.assertAlmostEqual(100 * (t5 - t1) / t5, 40.2, places=1)

    def test_ratios_separate_the_two_questions(self):
        # Plenty of reads, but far more writes: looks busy, still losing.
        us = [cost.Usage(cache_read=1000, cache_write_5m=9000) for _ in range(10)]
        r = cost.ratios(us)
        self.assertAlmostEqual(r["input_from_cache"], 0.10, places=6)
        self.assertAlmostEqual(r["prefix_efficiency"], 0.10, places=6)

    def test_crossover_window_derived_not_asserted(self):
        w = cost.ttl_crossover("anthropic/direct")
        self.assertTrue(w["applicable"])
        self.assertEqual(w["window_seconds"], (300, 3600))
        self.assertEqual(w["inside_window"]["winner"], "1h")
        self.assertEqual(w["below_window"]["winner"], "5m")
        self.assertEqual(w["above_window"]["winner"], "5m")

    def test_crossover_not_applicable_without_1h(self):
        self.assertFalse(cost.ttl_crossover("deepseek/direct")["applicable"])


class TestChecks(unittest.TestCase):

    def test_below_minimum_fails(self):
        r = checks.check_minimum(3000, "claude-haiku-4-5", "anthropic/direct",
                                 tokens_are_estimated=False)
        self.assertIs(r.status, Status.FAIL)
        self.assertIn("no error is returned", r.detail)

    def test_same_prefix_passes_on_a_lower_minimum_model(self):
        """3,000 tokens fails on Haiku 4.5 and passes on Opus 5. Same prompt."""
        self.assertIs(checks.check_minimum(3000, "claude-haiku-4-5", "anthropic/direct",
                                           tokens_are_estimated=False).status, Status.FAIL)
        self.assertIs(checks.check_minimum(3000, "claude-opus-5", "anthropic/direct",
                                           tokens_are_estimated=False).status, Status.PASS)

    def test_estimate_near_threshold_abstains(self):
        r = checks.check_minimum(4000, "claude-haiku-4-5", "anthropic/direct",
                                 tokens_are_estimated=True)
        self.assertIs(r.status, Status.ABSTAIN)
        self.assertIn("token counter", r.resolve)

    def test_exact_count_near_threshold_decides(self):
        r = checks.check_minimum(4000, "claude-haiku-4-5", "anthropic/direct",
                                 tokens_are_estimated=False)
        self.assertIs(r.status, Status.FAIL)

    def test_breakpoint_overrun_fails(self):
        r = checks.check_breakpoint_budget(5, "anthropic/direct")
        self.assertIs(r.status, Status.FAIL)
        self.assertIn("limit of 4", r.summary)

    def test_rolling_marker_reserves_two(self):
        self.assertIs(checks.check_breakpoint_budget(
            3, "anthropic/direct", rolling_marker=True).status, Status.ABSTAIN)
        self.assertIs(checks.check_breakpoint_budget(
            2, "anthropic/direct", rolling_marker=True).status, Status.PASS)

    def test_no_breakpoints_surface_abstains(self):
        r = checks.check_breakpoint_budget(1, target_id="deepseek/direct")
        self.assertIs(r.status, Status.ABSTAIN)

    def test_ttl_order_is_surface_specific(self):
        """Identical prompt: fine on Anthropic direct, wrong on Bedrock."""
        order = ["5m", "1h"]
        self.assertIs(checks.check_ttl_ordering(order, "anthropic/direct").status, Status.PASS)
        self.assertIs(checks.check_ttl_ordering(order, "amazon-bedrock/converse").status, Status.FAIL)
        self.assertIs(checks.check_ttl_ordering(["1h", "5m"], "amazon-bedrock/converse").status, Status.PASS)

    def test_unsupported_ttl_fails_even_without_ordering_rule(self):
        """An unconstrained target must still reject a TTL it cannot honour."""
        r = checks.check_ttl_ordering(["bogus"], "anthropic/direct")
        self.assertIs(r.status, Status.FAIL)
        r = checks.check_ttl_ordering(["1h"], "openai/direct")
        self.assertIs(r.status, Status.FAIL)
        self.assertIn("30m", r.detail)

    def test_unsupported_ttl_cannot_evade_ordering_check(self):
        r = checks.check_ttl_ordering(["bogus", "1h"], "amazon-bedrock/converse")
        self.assertIs(r.status, Status.FAIL)

    def test_ttls_on_a_surface_with_none_fail(self):
        r = checks.check_ttl_ordering(["5m"], "deepseek/direct")
        self.assertIs(r.status, Status.FAIL)
        self.assertIn("no TTL values", r.summary)

    def test_run_all_cannot_pass_with_a_bogus_ttl(self):
        rs = checks.run_all(prefix_tokens=20000, model="claude-sonnet-4-6",
                            breakpoints=2, ttls_in_order=["bogus"],
                            target_id="anthropic/direct",
                            tokens_are_estimated=False)
        self.assertIs(checks.worst(rs), Status.FAIL)

    def test_worst_reduces_correctly(self):
        rs = checks.run_all(prefix_tokens=3000, model="claude-haiku-4-5",
                            breakpoints=2, target_id="anthropic/direct",
                            tokens_are_estimated=False)
        self.assertIs(checks.worst(rs), Status.FAIL)
        rs = checks.run_all(prefix_tokens=20000, model="claude-sonnet-4-6",
                            breakpoints=2, target_id="anthropic/direct",
                            tokens_are_estimated=False)
        self.assertIs(checks.worst(rs), Status.PASS)





class TestUnpriceableSurfacesAbstain(unittest.TestCase):
    """The registry publishes surfaces whose pricing does not follow the
    Anthropic shape. Multiplying by a null multiplier raised a TypeError that
    aborted the whole report instead of excluding those requests."""

    def test_a_null_read_multiplier_raises_registry_error(self):
        with self.assertRaises(registry.RegistryError) as ctx:
            cost.price(cost.Usage(uncached_input=1000), "gpt-5.6",
                       target_id="openai/direct", on_date="2026-07-29",
                       effective_rate=1.0)
        self.assertIn("cannot price", str(ctx.exception))

    def test_it_names_which_multipliers_are_missing(self):
        with self.assertRaises(registry.RegistryError) as ctx:
            cost.price(cost.Usage(uncached_input=1), "gpt-5.6",
                       target_id="openai/direct", on_date="2026-07-29",
                       effective_rate=1.0)
        self.assertIn("read", str(ctx.exception))

    def test_anthropic_shaped_surfaces_still_price(self):
        """The multipliers are Anthropic-shaped on the partner surfaces too.

        Given a rate, the cache arithmetic works identically. This used to be
        asserted against *list* pricing, which is what let the rate-scope bug
        through: the surfaces do share the multiplier shape, and the test read
        as if they shared the rates as well.
        """
        for tid in ("anthropic/direct", "amazon-bedrock/converse", "google-cloud/vertex"):
            s = cost.price(cost.Usage(cache_read=1000), "claude-opus-5",
                           target_id=tid, on_date="2026-07-29", effective_rate=5.0)
            self.assertGreater(s.usd, 0, tid)

    def test_partner_surfaces_refuse_first_party_list_pricing(self):
        """Bedrock and Vertex are invoiced by the cloud provider.

        Before the rate table declared its scope, `base_rate` answered for every
        surface: a Bedrock trace of 12M uncached Haiku tokens reported $12.00 at
        Anthropic rates, with no note saying the rate came from the wrong price
        list. A wrong figure that looks right is the failure this whole tool is
        supposed to prevent, and it was arriving through the pricing path.
        """
        for tid in ("amazon-bedrock/converse", "google-cloud/vertex"):
            with self.assertRaises(registry.UnpriceableSurface, msg=tid) as ctx:
                cost.price(cost.Usage(uncached_input=1_000_000), "claude-haiku-4-5",
                           target_id=tid, on_date="2026-07-29")
            self.assertIn("--effective-rate", str(ctx.exception), tid)

    def test_the_scope_is_default_deny(self):
        """A surface earns first-party rates by being named, not by being
        absent from an exclusion list. Silence is how the next surface added
        would otherwise inherit Anthropic pricing."""
        self.assertFalse(registry.rates_apply_to("some-cloud/not-yet-added"))
        self.assertTrue(registry.rates_apply_to("anthropic/direct"))

    def test_the_scope_claim_itself_ages(self):
        """Which surfaces bill at first-party rates is a commercial arrangement,
        not a number on a page -- it can change without any rate changing. It
        decides whether a client's traffic is priced at all, so it comes up for
        review like every other dated claim here."""
        ids = [r["id"] for r in registry.staleness_report()]
        self.assertIn("pricing/rate-scope", ids)

    def test_the_refusal_is_distinguishable_from_a_missing_model(self):
        """Different remedies, so they cannot share a type. Reporting a missing
        model at a Bedrock client sends them to add a registry row that is
        already there."""
        self.assertTrue(issubclass(registry.UnpriceableSurface, registry.RegistryError))
        with self.assertRaises(registry.RegistryError) as ctx:
            cost.price(cost.Usage(uncached_input=1), "claude-nonexistent-9",
                       target_id="anthropic/direct", on_date="2026-07-29")
        self.assertNotIsInstance(ctx.exception, registry.UnpriceableSurface)


class TestDateParsingDoesNotDependOnThePythonVersion(unittest.TestCase):
    """The same trace must price the same way on every interpreter.

    `_as_date` ran `date.fromisoformat(str(value))`. Python 3.11 taught
    `fromisoformat` the compact ISO basic form, so the integer 20260801 was
    refused on 3.9 and parsed as 2026-08-01 on 3.13 -- two different prices for
    one trace, decided by which Python happened to run it, inside the function
    whose whole job is date-effective pricing. CI on 3.13 caught it the first
    time this was pushed somewhere that runs both; 1011 local tests on 3.9 did
    not.

    So the accepted form is pinned by regex rather than delegated to whatever
    the standard library accepts this release.
    """

    def test_only_the_explicit_hyphenated_form_is_a_date(self):
        self.assertEqual(registry._as_date("2026-08-01"), date(2026, 8, 1))

    def test_the_compact_form_is_refused_on_every_version(self):
        """Valid ISO 8601, and not what this registry accepts. Accepting it on
        3.11+ only is worse than refusing it everywhere."""
        for compact in ("20260801", 20260801, b"20260801"):
            with self.subTest(value=compact):
                with self.assertRaises(registry.RegistryError):
                    registry._as_date(compact)

    def test_a_bool_is_not_a_date(self):
        """`str(True)` is 'True', which never parsed -- but the type reaching a
        parse attempt at all is the bug this closes."""
        with self.assertRaises(registry.RegistryError):
            registry._as_date(True)

    def test_the_gate_is_a_regex_and_not_the_standard_library(self):
        """The one check here that can fail on any interpreter.

        The behavioural tests above cannot: on 3.9 `fromisoformat` refuses the
        compact form anyway, so reverting the fix leaves them green and only CI
        on 3.13 goes red. A test that cannot fail on the machine you are writing
        it on is not much of a test, so this asserts the mechanism instead --
        `_as_date` must decide with its own pattern before delegating, and must
        not stringify arbitrary values into a parser.
        """
        import ast
        import inspect

        src = inspect.getsource(registry._as_date)
        tree = ast.parse(src.lstrip())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]

        gated = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Attribute) and n.attr == "match"]
        self.assertTrue(gated, "_as_date no longer gates on an explicit pattern, "
                               "so what counts as a date is whatever this "
                               "Python's fromisoformat accepts")

        for c in calls:
            fn = c.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if name == "fromisoformat":
                arg = c.args[0] if c.args else None
                self.assertNotIsInstance(
                    arg, ast.Call,
                    "fromisoformat is being handed a converted value (str(...)), "
                    "which turns any type into a parse attempt")


class TestTargetAwareModelNormalization(unittest.TestCase):
    """Bedrock records `model_id_prefix: "anthropic."`. A trace carrying
    `anthropic.claude-...` priced fine under an invoice rate while
    min_cacheable_tokens rejected it, so the minimum guard was skipped on
    exactly the reports that publish dollar figures."""

    T = "amazon-bedrock/converse"

    def test_the_surface_prefix_is_stripped(self):
        self.assertEqual(registry.normalize_model("anthropic.claude-opus-5", self.T),
                         ("claude-opus-5", None))

    def test_a_prefix_and_a_date_suffix_together(self):
        self.assertEqual(
            registry.normalize_model("anthropic.claude-haiku-4-5-20251001", self.T),
            ("claude-haiku-4-5", "20251001"))

    def test_an_unprefixed_id_is_untouched(self):
        self.assertEqual(registry.normalize_model("claude-opus-5", self.T),
                         ("claude-opus-5", None))

    def test_without_a_target_the_prefix_is_left_alone(self):
        """Stripping needs to know the surface; guessing would invent a match."""
        self.assertEqual(registry.normalize_model("anthropic.claude-opus-5"),
                         ("anthropic.claude-opus-5", None))

    def test_the_normalised_id_resolves_a_minimum(self):
        base, _ = registry.normalize_model("anthropic.claude-haiku-4-5", self.T)
        self.assertEqual(registry.min_cacheable_tokens(self.T, base), 4096)


def _cell(text, label):
    """Pull one table row out of the report, wrap and all.

    The text report wraps its cells to a fixed width, so a row is a run of
    lines rather than one line. `[l for l in ... if l.startswith(label)][0]`
    read the first fragment only, which means an assertion about the rest of
    the cell passed whatever the rest of the cell said -- and after the report
    gained an indent it stopped matching at all.
    """
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith(label):
            cell = [line.strip()]
            for nxt in lines[i + 1:]:
                if not nxt.strip() or not nxt.startswith(" " * 25):
                    break
                cell.append(nxt.strip())
            return " ".join(cell)
    raise AssertionError(f"no {label!r} row in the report")


class TestZeroInvoiceRenders(unittest.TestCase):
    """analyze() sets delta_pct to None for a zero invoice deliberately. Three
    copies of abs(delta_pct) lived in the renderers, and fixing two left the
    edge case still unreportable."""

    def _analysis(self, invoice):
        import os as _os, sys as _sys
        _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
        from datetime import datetime, timezone
        from cacheeconomics.analyzer import analyze
        from cacheeconomics.trace import Request, TraceSet, Tier
        t0 = datetime(2026, 7, 29, 9, tzinfo=timezone.utc)
        ts = TraceSet(requests=[Request(request_id="a", sent_at=t0,
                                        model="claude-opus-5",
                                        usage={"input_tokens": 1000},
                                        segments=[], agent="a", target_id="anthropic/direct")],
                      tier=Tier.USAGE_ONLY)
        return analyze(ts, invoice_usd=invoice)

    # "supplied is zero" rather than "invoice is zero": the wording gained a word
    # when negative and non-finite invoices got their own sentences. What the
    # test is for is unchanged -- the zero case must be named as itself and not
    # folded into the generic ±5% failure.
    def test_html_renders_for_a_zero_invoice(self):
        from cacheeconomics.report import render_html
        self.assertIn("supplied is zero", render_html(self._analysis(0.0)))

    def test_text_renders_for_a_zero_invoice(self):
        from cacheeconomics.report import render_text
        self.assertIn("supplied is zero", render_text(self._analysis(0.0)))

    def test_the_text_renderer_names_the_real_blocker(self):
        """The HTML renderer learned `invalid_invoice`; the text one did not, and
        the text one is what gets pasted into an email. A negative invoice
        printed "invoice is zero, (OUTSIDE the 5% gate)" -- naming a blocker that
        is not the real one, and naming the gate as the reason when the invoice
        never reached it."""
        from cacheeconomics.report import render_text
        for invoice, expect, forbid in ((-60.0, "negative", "zero"),
                                        (0.0, "zero", "negative"),
                                        (float("inf"), "not a finite number", "zero"),
                                        ("credit", "not a number", "zero")):
            with self.subTest(invoice=invoice):
                line = _cell(render_text(self._analysis(invoice)), "reconciliation")
                self.assertIn(expect, line)
                self.assertIn("not attempted", line)
                self.assertNotIn(forbid, line)
                self.assertNotIn("the 5% gate", line)

    def test_a_valid_invoice_still_reports_the_percentage(self):
        from cacheeconomics.report import render_text
        line = _cell(render_text(self._analysis(5.0)), "reconciliation")
        self.assertIn("%", line)
        self.assertIn("the 5% gate", line)

    def test_both_renderers_survive_an_invalid_invoice(self):
        """A rejected input must not take down the deliverable. `_usd()` raised
        ValueError on a non-numeric invoice, so `render_html` crashed on exactly
        the input the analyzer had just learned to refuse."""
        from cacheeconomics.report import render_html, render_text
        for invoice, expected in ((-60.0, "negative"),
                                  (float("inf"), "not a finite number"),
                                  ("credit", "not a number")):
            with self.subTest(invoice=invoice):
                a = self._analysis(invoice)
                for render in (render_text, render_html):
                    out = render(a)
                    self.assertIn(expected, out)
                    # And it must not claim the blocker was the ±5% gate.
                    self.assertNotIn("outside the ±5% gate", out)

    def test_a_normal_invoice_still_shows_a_percentage(self):
        from cacheeconomics.report import render_text
        self.assertIn("%", render_text(self._analysis(5.0)))


class TestMalformedUsageIsExcludedNotFatal(unittest.TestCase):
    """A `usage` field written as a JSON string satisfied `has_usage`'s
    membership test, survived into `analysable`, and reached the cost model,
    which called `.get` on it. One malformed row aborted the entire report with
    an AttributeError. A stated gap beats a dropped file; a crash is worse than
    either."""

    ROWS = [
        {"request_id": "good", "sent_at": "2026-07-29T09:00:00Z",
         "model": "claude-opus-5",
         "usage": {"input_tokens": 10, "cache_read_input_tokens": 100,
                   "cache_creation_input_tokens": 0}},
        {"request_id": "str-json", "sent_at": "2026-07-29T09:01:00Z",
         "model": "claude-opus-5",
         "usage": '{"input_tokens": 5, "cache_read_input_tokens": 50, '
                  '"cache_creation_input_tokens": 0}'},
        {"request_id": "scalar", "sent_at": "2026-07-29T09:02:00Z",
         "model": "claude-opus-5", "usage": 42},
        {"request_id": "garbage", "sent_at": "2026-07-29T09:03:00Z",
         "model": "claude-opus-5", "usage": "nope"},
    ]

    def _load(self):
        import json
        import tempfile
        from cacheeconomics.trace import load_jsonl
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        try:
            with open(path, "w") as f:
                f.write("\n".join(json.dumps(r) for r in self.ROWS))
            return load_jsonl(path)
        finally:
            os.unlink(path)

    def test_the_report_does_not_crash(self):
        from cacheeconomics.analyzer import analyze
        analyze(self._load(), allow_unreconciled=True)

    def test_a_usage_written_as_json_is_recovered(self):
        """Worth parsing: exporters do this often enough."""
        self.assertIn("str-json", {r.request_id for r in self._load().analysable})

    def test_a_scalar_or_unparseable_usage_is_excluded(self):
        analysable = {r.request_id for r in self._load().analysable}
        self.assertNotIn("scalar", analysable)
        self.assertNotIn("garbage", analysable)

    def test_a_request_built_directly_is_guarded_too(self):
        from cacheeconomics.trace import Request
        self.assertFalse(Request(request_id="x", sent_at=None, model="m",
                                 usage="input_tokens", target_id="anthropic/direct").has_usage)


class TestEveryNamedModelHasARate(unittest.TestCase):
    """A model the registry names but cannot price is worse than one it has
    never heard of.

    `min_cacheable_tokens` naming a model is this project asserting it knows how
    that model caches. If no rate accompanies it, `cost.price` refuses the
    request, and the analyzer excludes it from spend, from every ratio and from
    invoice reconciliation -- so a client still on Sonnet 4 received a confident
    report about the subset of their traffic that happened to be on a newer
    model. Four models sat in exactly that state.

    Scoped to surfaces this cost model can actually price. A surface with no
    Anthropic-shaped multipliers is refused earlier and for a different reason,
    which is a deliberate gap rather than an oversight.
    """

    def _registry(self, name):
        import json
        import os
        from cacheeconomics import registry
        with open(os.path.join(registry.REGISTRY_DIR, name)) as f:
            return json.load(f)

    def test_no_named_model_is_left_unpriceable(self):
        from cacheeconomics import cost, registry
        providers = self._registry("providers.json")
        gaps = {}
        for target in providers.get("targets", []):
            mult = target.get("multipliers") or {}
            # Only surfaces the cost model can price at all.
            if not all(k in mult and mult[k] is not None
                       for k in ("write_5m", "write_1h", "read")):
                continue
            for model in target.get("min_cacheable_tokens", {}):
                try:
                    cost.price(cost.Usage(uncached_input=1_000),
                               model, target_id=target["id"],
                               on_date="2026-07-30")
                except registry.RegistryError:
                    gaps.setdefault(target["id"], []).append(model)
        self.assertFalse(
            gaps,
            f"these models are named with a cache minimum but cannot be "
            f"priced: {gaps}. Every request on them is dropped from spend, "
            f"from the ratios and from invoice reconciliation, so the report "
            f"silently describes a subset.")

    def test_the_recorded_multipliers_match_the_published_columns(self):
        """The four rates added on 2026-07-30 were cross-checked against the
        multipliers rather than copied on their own: a base rate and a write
        column that disagree mean one of them is misread, and both feed the
        same figure."""
        from cacheeconomics import cost
        for model, base in (("claude-opus-4-5", 5.00), ("claude-opus-4-1", 15.00),
                            ("claude-sonnet-4", 3.00), ("claude-haiku-3-5", 0.80)):
            with self.subTest(model=model):
                def usd(**kw):
                    return cost.price(cost.Usage(**kw), model, "anthropic/direct",
                                      on_date="2026-07-30").usd
                self.assertAlmostEqual(usd(uncached_input=1_000_000), base)
                self.assertAlmostEqual(usd(cache_write_5m=1_000_000), base * 1.25)
                self.assertAlmostEqual(usd(cache_write_1h=1_000_000), base * 2.0)
                self.assertAlmostEqual(usd(cache_read=1_000_000), base * 0.1)


class TestAnUnknownCacheLifetimeIsNotFree(unittest.TestCase):
    """`trace.write_tokens` sums every positive value under `cache_creation`,
    because for detection the provider saying it wrote is enough. Pricing needs
    the lifetime, and `from_anthropic` read only the two Anthropic keys.

    So a response carrying `ephemeral_30m_input_tokens: 10000` and no aggregate
    was analysable, counted as having writes, and priced at $0.00. The two
    functions answer different questions and are allowed to; they are not
    allowed to disagree silently in the direction of free.
    """

    def test_a_positive_unknown_lifetime_is_refused(self):
        with self.assertRaises(ValueError) as e:
            cost.Usage.from_anthropic(
                {"cache_creation": {"ephemeral_30m_input_tokens": 10_000}})
        self.assertIn("ephemeral_30m_input_tokens", str(e.exception))

    def test_the_known_lifetimes_still_price(self):
        u = cost.Usage.from_anthropic(
            {"cache_creation": {"ephemeral_5m_input_tokens": 900,
                                "ephemeral_1h_input_tokens": 100}})
        self.assertEqual((u.cache_write_5m, u.cache_write_1h), (900, 100))

    def test_a_zero_unknown_key_is_not_a_problem(self):
        """An exporter emitting a field it never populates is noise, not drift."""
        u = cost.Usage.from_anthropic(
            {"cache_creation": {"ephemeral_5m_input_tokens": 900,
                                "ephemeral_30m_input_tokens": 0}})
        self.assertEqual(u.cache_write_5m, 900)

    def test_the_detector_and_the_pricer_do_not_disagree_into_free(self):
        """The invariant behind the finding: if writes are detected, either they
        price or the row is refused. Never priced at zero."""
        from cacheeconomics.trace import write_tokens
        usage = {"cache_creation": {"ephemeral_30m_input_tokens": 10_000}}
        self.assertGreater(write_tokens(usage), 0)
        with self.assertRaises(ValueError):
            cost.Usage.from_anthropic(usage)


class TestTheReviewBatch(unittest.TestCase):
    """One test per finding from the 2026-07-31 review batch."""

    def test_repr_does_not_leak_a_withheld_figure(self):
        """`__str__` and `__format__` honour the gate; the dataclass auto-repr
        did not -- and repr is what a traceback, a pytest diff and
        `print(list_of_figures)` all reach for, with no `raw()` to grep for."""
        from cacheeconomics import money
        f = money.modeled(1234.56)
        self.assertNotIn("1234.56", repr(f))
        self.assertIn("withheld", repr(f))
        self.assertIn("1234.56", repr(f.release(True)))

    def test_a_warmup_above_the_window_is_refused(self):
        """`Monitor.samples()` caps at WINDOW, so a higher warmup is never
        reached: measured, warmup=100 sat at "still learning (65/100)" through
        300 requests and placed nothing."""
        from cacheeconomics.monitor import WINDOW
        from cacheeconomics.plugin import CachePlugin
        CachePlugin(key=b"k" * 32, warmup=WINDOW)
        with self.assertRaises(ValueError):
            CachePlugin(key=b"k" * 32, warmup=WINDOW + 1)

    def test_a_null_breakpoint_budget_is_not_a_missing_capability(self):
        """openai/direct records explicit_breakpoints true with max_breakpoints
        null. Reporting that it "does not expose developer-placed breakpoints"
        published a false capability claim into a client report."""
        r = checks.check_breakpoint_budget(breakpoints=2, target_id="openai/direct")
        self.assertIs(r.status, Status.ABSTAIN)
        self.assertIn("no breakpoint budget is recorded", r.summary)
        self.assertNotIn("does not expose", r.summary)

    def test_normalize_model_does_not_mutilate_an_unknown_id(self):
        """It promises never to invent a model; half-rewriting an unknown one is
        a smaller version of the same broken promise, and it made the error
        quote an id the caller never sent."""
        self.assertEqual(registry.normalize_model("anthropic/not-a-real-model-20250101"),
                         ("anthropic/not-a-real-model-20250101", None))
        self.assertEqual(registry.normalize_model("anthropic/claude-opus-5-20250101"),
                         ("claude-opus-5", "20250101"))

    def test_inherits_minimums_from_refuses_a_cycle(self):
        """One level was assumed. A row inheriting from itself recursed until
        the error had nothing to do with the registry."""
        import copy
        original = registry._PROVIDERS
        try:
            d = copy.deepcopy(registry.providers())
            for t in d["targets"]:
                if t["id"] == "amazon-bedrock/converse":
                    t["inherits_minimums_from"] = "amazon-bedrock/converse"
            registry._PROVIDERS = d
            with self.assertRaises(registry.RegistryError) as e:
                registry.min_cacheable_tokens("amazon-bedrock/converse", "claude-opus-5")
            self.assertIn("cycle", str(e.exception))
        finally:
            registry._PROVIDERS = original

    def test_bedrock_1h_is_not_blessed_on_every_model(self):
        """The row's own provenance limits 1h to three models while the
        capability stayed surface-wide, so the linter blessed a marker the
        provider would reject."""
        self.assertEqual(
            registry.supported_ttls("amazon-bedrock/converse", "claude-opus-4-1"),
            ["5m"])
        self.assertEqual(
            sorted(registry.supported_ttls("amazon-bedrock/converse", "claude-haiku-4-5")),
            ["1h", "5m"])
        r = checks.check_ttl_ordering(["1h"], "amazon-bedrock/converse",
                                      "claude-opus-4-1")
        self.assertIs(r.status, Status.FAIL)

    def test_anthropic_direct_is_unaffected_by_the_per_model_map(self):
        """A row without the map keeps the surface-wide answer."""
        self.assertEqual(sorted(registry.supported_ttls("anthropic/direct",
                                                        "claude-opus-4-1")),
                         ["1h", "5m"])

    def test_ttl_one_uses_the_registry_multipliers(self):
        """VOL-1 and FAN-1 look them up; TTL-1 carried 1.25/2.00/0.10 as
        literals -- the duplication this repo's twin-path tests exist to catch,
        in the one copy nothing watched."""
        import inspect

        from cacheeconomics import analyzer
        src = inspect.getsource(analyzer._f_ttl_vs_cadence)
        self.assertIn("registry.multipliers", src)
        for literal in ("1.25 - 0.10", "2.00 - 1.25"):
            self.assertNotIn(literal, src)

    def test_the_dead_now_helper_is_gone(self):
        """It referenced datetime/timezone that segment.py never imports, so it
        raised NameError if anyone called it."""
        from cacheeconomics import segment
        self.assertFalse(hasattr(segment, "_now"))


class TestRoundTwelve(unittest.TestCase):
    """One test per finding."""

    def test_a_malformed_pricing_date_is_refused(self):
        """Rates were selected by comparing raw strings. `2026-8-1` is August
        and sorts *after* `2026-09-01`, so claude-sonnet-5 priced at the
        post-September $3.00 instead of $2.00 -- a 50% overstatement on a report
        whose entire premise is date-effective pricing. `not-a-date` did the
        same without failing."""
        self.assertEqual(registry.base_rate("claude-sonnet-5", "2026-08-01", "anthropic/direct"), 2.00)
        self.assertEqual(registry.base_rate("claude-sonnet-5", "2026-09-01", "anthropic/direct"), 3.00)
        for bad in ("2026-8-1", "not-a-date", "08-01-2026", "2026/08/01", None,
                    20260801, "20260801", 20260801.0, True, b"2026-08-01"):
            with self.subTest(on_date=bad):
                with self.assertRaises(registry.RegistryError):
                    registry.base_rate("claude-sonnet-5", bad, "anthropic/direct")

    def test_a_real_date_object_still_works(self):
        from datetime import date as _date
        self.assertEqual(registry.base_rate("claude-sonnet-5", _date(2026, 8, 1), "anthropic/direct"), 2.00)

    def test_unread_transcripts_block_release(self):
        """`load_sessions` counted them and put the count only in `notes`, which
        the reconciliation gate cannot read -- so a readable subset matching the
        invoice released figures while the notes said a transcript was missing."""
        import inspect

        from cacheeconomics.adapters import claude_code
        src = inspect.getsource(claude_code.load_sessions)
        self.assertIn("skipped_rows=skipped", src)

    def test_the_live_path_understands_a_litellm_response(self):
        """`litellm_handler`'s success event hands `response_obj` to
        `on_response`, which used the Anthropic-only parser. A proxy response
        carries `prompt_tokens_details` and none of the Anthropic keys, so it
        read as absent usage and the named LiteLLM integration never learned
        whether prefixes were rebuilding or its markers were being read."""
        from cacheeconomics.segment import usage_from_response
        u = usage_from_response({"usage": {
            "prompt_tokens": 200_300, "completion_tokens": 120,
            "prompt_tokens_details": {"cached_tokens": 200_000,
                                      "cache_write_tokens": 0,
                                      "text_tokens": 300}}})
        self.assertEqual(u["input_tokens"], 300)
        self.assertEqual(u["cache_read_input_tokens"], 200_000)

    def test_the_anthropic_shape_is_unaffected(self):
        from cacheeconomics.segment import usage_from_response
        u = usage_from_response({"usage": {"input_tokens": 42,
                                           "cache_read_input_tokens": 7,
                                           "cache_creation_input_tokens": 0}})
        self.assertEqual(u["input_tokens"], 42)

    def test_a_response_with_neither_shape_is_still_absent(self):
        from cacheeconomics.segment import usage_from_response
        self.assertEqual(usage_from_response({"usage": {}}), {})
        self.assertEqual(usage_from_response({"usage": {"completion_tokens": 5}}), {})

    def test_both_adapters_share_one_details_parser(self):
        """The batch path parsed this shape correctly while the live path could
        not, which is the divergence itself."""
        from cacheeconomics.adapters import litellm as ad
        from cacheeconomics.segment import usage_from_details
        self.assertIs(ad._from_details, usage_from_details)


class TestRoundFourteen(unittest.TestCase):

    def test_the_allocator_respects_per_model_ttl_support(self):
        """`checks` learned the per-model map last round and `tiers` did not, so
        the allocator planned a 1h tier for claude-opus-4-1 on Bedrock -- a
        lifetime the registry says that model does not have. The bake-off would
        have reported a saving the provider cannot deliver, and the live plugin
        would have put the marker on the wire."""
        from cacheeconomics import tiers
        from cacheeconomics.trace import Segment
        segs = [Segment(id="a", role="system", tokens=9_000, index=0),
                Segment(id="b", role="user", tokens=200, index=1)]

        def ttls(model):
            alloc = tiers.allocate(segs, {0: 0.0, 1: 1.0},
                                   target_id="amazon-bedrock/converse",
                                   model=model, gaps=[900] * 10)
            return {t.ttl for t in alloc.tiers}

        self.assertNotIn("1h", ttls("claude-opus-4-1"),
                         "registry says this model is 5m only on Bedrock")
        self.assertIn("1h", ttls("claude-haiku-4-5"),
                      "and 1h where the registry does record it")

    def test_anthropic_direct_is_unnarrowed(self):
        """A surface without a per-model map keeps the surface-wide answer."""
        from cacheeconomics import tiers
        from cacheeconomics.trace import Segment
        segs = [Segment(id="a", role="system", tokens=9_000, index=0),
                Segment(id="b", role="user", tokens=200, index=1)]
        alloc = tiers.allocate(segs, {0: 0.0, 1: 1.0},
                               target_id="anthropic/direct",
                               model="claude-opus-4-1", gaps=[900] * 10)
        self.assertIn("1h", {t.ttl for t in alloc.tiers})

    def _mixed(self, fail_usage):
        from datetime import datetime, timedelta, timezone
        from cacheeconomics.trace import Request, Tier, TraceSet
        t0 = datetime(2026, 7, 29, 9, tzinfo=timezone.utc)
        u = {"input_tokens": 1_000_000, "cache_read_input_tokens": 0,
             "cache_creation_input_tokens": 0}
        return TraceSet(requests=[
            Request(request_id="ok", sent_at=t0, model="claude-opus-5", agent="a",
                    session="s", ttl_requested="5m", usage=dict(u), segments=[],
                    status=200, target_id="anthropic/direct"),
            Request(request_id="fail", sent_at=t0 + timedelta(seconds=60),
                    model="claude-opus-5", agent="a", session="s",
                    ttl_requested="5m", usage=fail_usage, segments=[],
                    status=500, target_id="anthropic/direct")], tier=Tier.USAGE_ONLY, source="x")

    def test_a_failed_call_that_billed_blocks_a_draft(self):
        """`analyze` starts from `ts.analysable`, which drops non-200 rows, so
        their cost left with them: one 200 and one 500 both billing a million
        tokens released $5.00 over 50% coverage. Partial-failure cost is exactly
        where spend hides."""
        from cacheeconomics.analyzer import analyze
        a = analyze(self._mixed({"input_tokens": 1_000_000,
                                 "cache_read_input_tokens": 0,
                                 "cache_creation_input_tokens": 0}),
                    allow_unreconciled=True)
        self.assertFalse(a.spend["input_usd"].released)

    def test_a_failed_call_that_billed_nothing_does_not(self):
        """Blocking on every transport error would withhold every report
        containing one -- the over-block this project has shipped twice."""
        from cacheeconomics.analyzer import analyze
        a = analyze(self._mixed({}), allow_unreconciled=True)
        self.assertTrue(a.spend["input_usd"].released)

    def test_it_blocks_the_invoice_gate_too(self):
        from cacheeconomics.analyzer import analyze
        a = analyze(self._mixed({"input_tokens": 1_000_000,
                                 "cache_read_input_tokens": 0,
                                 "cache_creation_input_tokens": 0}),
                    invoice_usd=5.0)
        self.assertFalse(a.reconciliation["within_ship_gate"])
        self.assertEqual(a.reconciliation["blockers"]["failed_but_billed"], 1)


if __name__ == "__main__":
    unittest.main()


class TestTheReportDoesNotSpeakForStepsItDidNotRun(unittest.TestCase):
    """The HTML footer asserted "No prompt content left the environment" on
    every report, unconditionally.

    That is false on the workflow this project recommends. `run_diagnostic.py`
    calls `count_tokens.py` unless `--estimate-only` is passed, and counting
    sends prompt prefixes to a tokenizer -- that is the point of it, and it is
    what takes the segment split from 19.2% median error to 0.2%. A client
    reading only the report was told nothing left, in a report produced by a run
    that had just sent their prompts to a provider.

    The analysis half stays true and is still said: this package imports no
    network library and `test_cli` asserts it. What changed is that the report
    stopped speaking for a step it did not perform.
    """

    def _analysis(self, counted):
        from dataclasses import replace
        from cacheeconomics.analyzer import analyze
        from cacheeconomics.trace import load_jsonl
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "fixtures", "demo-traces.jsonl")
        a = analyze(load_jsonl(os.path.normpath(path)), invoice_usd=17.45)
        return replace(a, tokens_counted=counted)

    def _html(self, counted):
        from cacheeconomics.report import render_html
        return render_html(self._analysis(counted))

    NOTHING_LEFT = "No prompt content left the environment"

    def test_a_counted_trace_does_not_claim_nothing_left(self):
        self.assertNotIn(self.NOTHING_LEFT, self._html(True))

    def test_a_counted_trace_says_where_the_prompts_went(self):
        self.assertIn("counting is a separate step", self._html(True))

    def test_an_estimated_trace_still_makes_the_claim(self):
        """The other direction. Nothing was sent on this path, and saying so is
        the honest answer -- a fix that removed the claim everywhere would
        understate a real property of the tool."""
        html = self._html(False)
        self.assertIn(self.NOTHING_LEFT, html)
        self.assertNotIn("counting is a separate step", html)

    def test_the_text_report_makes_the_same_claim_conditionally(self):
        """The footer was made conditional and `render_text` was not: it said
        "Nothing was sent anywhere", the same false claim in different words.

        Different words is why it survived — the sweep that fixed the footer
        searched for the footer's sentence, matching by string rather than by
        claim. Both renderers read one helper now.
        """
        from cacheeconomics.report import render_text
        counted = render_text(self._analysis(True))
        self.assertNotIn("Nothing was sent anywhere", counted)
        self.assertIn("counting sent prompt prefixes", counted)

    def test_the_text_report_still_says_it_when_nothing_was_sent(self):
        from cacheeconomics.report import render_text
        estimated = render_text(self._analysis(False))
        self.assertIn("Nothing was sent anywhere", estimated)
        self.assertNotIn("counting sent prompt prefixes", estimated)

    def test_no_renderer_asserts_transmission_the_others_deny(self):
        """The class, not the two call sites. Any renderer claiming nothing was
        transmitted must agree with `tokens_counted`, whatever words it uses."""
        from cacheeconomics.report import render_html, render_text
        DENIALS = ("Nothing was sent anywhere",
                   "No prompt content left the environment")
        for counted in (True, False):
            a = self._analysis(counted)
            for name, out in (("text", render_text(a)), ("html", render_html(a))):
                with self.subTest(renderer=name, counted=counted):
                    denies = any(d in out for d in DENIALS)
                    self.assertEqual(denies, not counted,
                                     f"{name} denies transmission on a "
                                     f"tokens_counted={counted} run")

    def test_the_counting_flag_is_actually_set_by_the_analyzer(self):
        """`report._next_steps` read `_tokens_counted` off the Analysis, which
        nothing ever set, so `getattr(..., True)` always won and the branch
        offering counting as a next step could never fire. A flag no producer
        writes is not a flag."""
        from cacheeconomics.analyzer import analyze
        from cacheeconomics.trace import load_jsonl
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "fixtures", "demo-traces.jsonl")
        a = analyze(load_jsonl(os.path.normpath(path)), invoice_usd=17.45)
        self.assertTrue(hasattr(a, "tokens_counted"))
        self.assertTrue(a.tokens_counted, "the demo fixture states counted rows")


class TestADraftFigureIsNotADollarFigureThatWasChecked(unittest.TestCase):
    """`Figure.release(True)` recorded only that a figure was released, so one
    released by `--allow-unreconciled` was byte-identical to one an invoice had
    checked: same value, same basis, same string.

    Measured on the demo trace, both rendered "$229" with nothing to tell them
    apart, so no renderer *could* mark one and not the other. And HTML put the
    first "$" at character 6,554 with the first "DRAFT" at 14,510 -- a forwarded
    report read as client-ready for eight thousand characters. The text renderer
    already led with the stamp; only HTML did not.
    """

    def _a(self, **kw):
        from cacheeconomics.analyzer import analyze
        from cacheeconomics.trace import load_jsonl
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "fixtures", "demo-traces.jsonl")
        return analyze(load_jsonl(os.path.normpath(path)), **kw)

    def test_the_two_release_states_are_distinguishable(self):
        from cacheeconomics.money import DRAFT, RECONCILED
        d = self._a(allow_unreconciled=True).spend["input_usd"]
        r = self._a(invoice_usd=17.45).spend["input_usd"]
        self.assertEqual(d.released_as, DRAFT)
        self.assertEqual(r.released_as, RECONCILED)
        self.assertTrue(d.released and r.released)
        self.assertNotEqual(d, r)

    def test_every_figure_in_a_draft_run_agrees(self):
        """Releasing the spend map as draft while findings said reconciled
        would put both states in one report -- the fix-one-site-leave-the-rest
        shape this distinction exists to expose."""
        from cacheeconomics.money import DRAFT
        a = self._a(allow_unreconciled=True)
        states = {v.released_as for v in a.spend.values()
                  if hasattr(v, "released_as") and v.released}
        states |= {f.avoidable_usd_month.released_as for f in a.findings
                   if f.avoidable_usd_month and f.avoidable_usd_month.released}
        states.add(a.total_avoidable_month.released_as)
        self.assertEqual(states, {DRAFT})

    def test_html_stamps_the_draft_before_any_dollar(self):
        """Inside <body>, not merely earlier in the file.

        The first version of this test asserted only that DRAFT preceded the
        first "$", and passed while the banner sat inside <head> -- present in
        the source, rendered nowhere. A stamp that satisfies a substring test
        and is invisible on the page is worse than no stamp.
        """
        from cacheeconomics.report import render_html
        h = render_html(self._a(allow_unreconciled=True))
        stamp, dollar, body = h.find("DRAFT"), h.find("$"), h.find("<body")
        self.assertGreater(stamp, body, "the banner is not inside <body>")
        self.assertLess(stamp, dollar)
        self.assertIn("not for external use", h.lower())

    def test_a_reconciled_report_carries_no_draft_banner(self):
        """The other direction: stamping everything would make the stamp
        meaningless."""
        from cacheeconomics.report import render_html
        self.assertNotIn("DRAFT", render_html(self._a(invoice_usd=17.45)))

    def test_the_json_output_carries_the_state_for_machines(self):
        """A script reading --format json saw only strings and could not tell a
        draft figure from an invoice-checked one."""
        from cacheeconomics.money import DRAFT
        a = self._a(allow_unreconciled=True)
        states = {k: v.released_as for k, v in a.spend.items()
                  if hasattr(v, "released_as")}
        self.assertIn(DRAFT, states.values())


class TestTheEstimateBandIsTheMeasuredErrorNotATidyTenPercent(unittest.TestCase):
    """`check_minimum` treated a byte-ratio estimate as accurate to +-10%.

    The overestimate side is measured, and it is not 10%.
    `segment.ESTIMATOR_WORST_OVERESTIMATE` records 2.81x -- the worst cumulative
    overestimate of that estimator against the provider's own tokenizer, over 26
    prefixes from six bodies. The plugin has used it as its placement margin for
    as long as it has existed. The static check used a symmetric band beside it,
    and the two disagreed about the same question on the same input.

    Reproduced before the change: `check_minimum(600, 'claude-opus-5')` returned
    PASS -- "600 tokens clears the 512 minimum" -- while at the measured worst
    those 600 estimated tokens are 213 real ones, far below 512. The +-10% band
    spans 540-660, never straddles 512, so it did not even abstain.

    What the 2.81 does *not* support is a claim about the estimator's true
    bound: 26 prefixes over six bodies is a floor on the error, not a proof. It
    is used to decide when the check may not answer, never to produce a figure.
    """

    MODEL = "claude-opus-5"          # 512 on anthropic/direct
    SURFACE = "anthropic/direct"

    def _est(self, n):
        return checks.check_minimum(n, self.MODEL, self.SURFACE,
                                    tokens_are_estimated=True)

    def test_the_reproduction_now_abstains(self):
        r = self._est(600)
        self.assertIs(r.status, Status.ABSTAIN)
        self.assertNotIn("clears", r.summary)

    def test_and_says_how_low_the_real_count_could_be(self):
        """An abstention nobody can act on is only a quieter guess."""
        r = self._est(600)
        self.assertIn("213", r.detail)          # 600 / 2.81
        self.assertIn("2.81", r.detail)

    def test_it_does_not_overstate_what_the_measurement_proves(self):
        """26 prefixes over six bodies is a floor on the error, and the text
        that leans on it has to say so where a client reads it."""
        r = self._est(600)
        self.assertIn("26 prefixes", r.detail)
        self.assertIn("not a proof", r.detail)

    def test_an_estimate_clear_of_its_own_error_still_passes(self):
        """The other direction. A check that abstains on everything is a check
        somebody switches off, and then it catches nothing."""
        r = self._est(512 * 3)
        self.assertIs(r.status, Status.PASS)

    def test_an_exact_count_is_still_decided_at_the_threshold(self):
        """The band applies to estimates. A counted 600 is above 512, and
        widening the estimate's band must not make a measured number vague."""
        r = checks.check_minimum(600, self.MODEL, self.SURFACE,
                                 tokens_are_estimated=False)
        self.assertIs(r.status, Status.PASS)

    def test_the_check_and_the_plugin_use_one_margin(self):
        """The twin-path half. These are two implementations of "is this prefix
        long enough to be worth a marker", and they disagreed: the plugin
        refused at 2.81x while the check blessed at 1.1x, so a linter in CI
        passed a prompt the runtime would then decline to mark.
        """
        from cacheeconomics.plugin import CachePlugin
        from cacheeconomics.segment import ESTIMATOR_WORST_OVERESTIMATE
        p = CachePlugin(key=b"k" * 32)
        self.assertAlmostEqual(ESTIMATOR_WORST_OVERESTIMATE - 1,
                               p.minimum_margin)
        minimum = registry.min_cacheable_tokens(self.SURFACE, self.MODEL)
        floor = minimum * (1 + p.minimum_margin)
        self.assertIs(self._est(int(floor) + 1).status, Status.PASS)
        self.assertIs(self._est(int(floor) - 1).status, Status.ABSTAIN)
