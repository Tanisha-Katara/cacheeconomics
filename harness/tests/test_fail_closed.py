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
from cacheeconomics.trace import Request, Segment, Tier, TraceSet  # noqa: E402

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

    def test_the_cli_default_is_unchanged_when_nothing_is_said(self):
        """The flag now defaults to None so silence is distinguishable. That
        must not turn into a surface of None reaching the registry."""
        ts = self._via_cli(self._write(self._rows()), "trace")
        self.assertEqual({r.target_id for r in ts.requests}, {"anthropic/direct"})

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

    def test_an_unmarked_request_is_still_anthropic_direct(self):
        """The common case must not move, or every existing deployment
        silently changes surface."""
        self.assertEqual(self._run(self._hook(), "claude-haiku-4-5"),
                         {"anthropic/direct"})

    def test_an_operator_override_answers_when_nothing_names_it(self):
        self.assertEqual(
            self._run(self._hook("amazon-bedrock/converse"), "claude-haiku-4-5"),
            {"amazon-bedrock/converse"})

    def test_the_budget_recheck_uses_the_resolved_surface(self):
        """Structural: the guard between a miscount and a rejected request read
        a literal, so it enforced Anthropic's budget on every provider."""
        import ast
        import inspect

        from cacheeconomics import plugin
        src = inspect.getsource(plugin.litellm_handler)
        for node in ast.walk(ast.parse(src.lstrip())):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "capability"):
                first = node.args[0] if node.args else None
                self.assertNotIsInstance(
                    first, ast.Constant,
                    "registry.capability is called with a literal surface in the "
                    "live hook; it must use the resolved target")


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
        """The CLI passed DEFAULT_TARGET for this path, which defeated the
        adapter's refusal entirely. The litellm path already knew better."""
        import inspect

        from cacheeconomics import cli
        src = inspect.getsource(cli._load)
        bodies = src.split("if args.source == \"bodies\"")[1]
        # The call, not any mention of it: the comment above the call names
        # DEFAULT_TARGET to explain why it is not used, and matching prose
        # rather than code is how a test ends up asserting nothing.
        call = bodies[bodies.index("load_bodies("):]
        call = call[:call.index(")")]
        self.assertIn("target_id=args.target_id", call)
        self.assertNotIn("DEFAULT_TARGET", call,
                         "the CLI re-fabricates the surface the adapter refused")


class TestClaudeCodeSaysTheSurfaceIsAssumed(unittest.TestCase):
    """Transcripts carry no provider field anywhere -- checked across 190 of
    them. The surface is an adapter assumption, kept because Claude Code talks
    to Anthropic unless routed, but stated rather than buried."""

    def test_the_assumption_is_visible_without_detail(self):
        from cacheeconomics.adapters.claude_code import load_sessions
        from cacheeconomics.report import render_text
        try:
            ts = load_sessions()
        except Exception:
            self.skipTest("no local transcripts")
        if not ts.requests:
            self.skipTest("no local transcripts")
        self.assertTrue(any("surface assumed" in n for n in ts.blocking_notes))
        flat = " ".join(render_text(analyze(ts, allow_unreconciled=True)).split())
        self.assertIn("surface assumed", flat)
        self.assertIn("--target-id", flat)

    def test_an_explicit_surface_replaces_it(self):
        from cacheeconomics.adapters.claude_code import load_sessions
        try:
            ts = load_sessions(target_id="amazon-bedrock/converse")
        except Exception:
            self.skipTest("no local transcripts")
        if not ts.requests:
            self.skipTest("no local transcripts")
        self.assertEqual({r.target_id for r in ts.requests},
                         {"amazon-bedrock/converse"})
        self.assertFalse(any("surface assumed" in n for n in ts.blocking_notes))
