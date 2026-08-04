"""Invariants that discover their own members, instead of trusting a sentence.

Every defect this file exists to catch was, at the time it shipped, covered by
a passing test and a commit message asserting the class was closed. The tests
were true and the sentences were false, because the tests asserted *the edit*
and the sentence claimed *the class*:

  - "every monthly figure is gated" -- the headline was; six per-finding
    figures were not, and a one-hour trace published $180/mo as `reconciled`.
  - "no renderer prints a withheld figure" -- the HTML one didn't; the text
    one said the same thing in different words.
  - "the DRAFT banner is shown" -- it was inserted at index 1, inside `<head>`,
    and the test asserted its offset was lower than the first `$`, which it was.

The shape of the mistake never changed: a claim of the form *"all X have
property P"* verified by checking one X that I had just edited. So the rule
this file enforces is that such a claim has to be a test which **finds X at
runtime** -- by walking the object graph, reading `inspect.signature`, probing
which registry keys a check actually touches -- and then checks P on whatever
it found. A member added later is covered by construction rather than by
somebody remembering this file exists.

What this cannot do, stated plainly because the last version of this claim was
overstated: an invariant discovers members *by a mechanism*, and a member
reachable by a route the mechanism does not model stays invisible. INV-1 finds
projected figures because `_monthly` marks them; a second extrapolation site
that forgets to mark is not found. This lowers silent partial closure from
"anything I did not think of" to "requires bypassing a named mechanism". It is
an improvement, not a proof.
"""

import dataclasses
import inspect
import json
import os
import sys
import unittest
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cacheeconomics import money, monitor, plugin, registry, report  # noqa: E402
from cacheeconomics.analyzer import Analysis, analyze  # noqa: E402
from cacheeconomics.money import Figure  # noqa: E402
from cacheeconomics.trace import Request, Segment, Tier, TraceSet  # noqa: E402

T0 = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)


# --- discovery -------------------------------------------------------------

def walk_figures(obj, path="root", _depth=0):
    """Yield `(path, Figure)` for every Figure reachable from `obj`.

    Walks dataclass fields, dicts, sequences **and properties**. Properties are
    included deliberately and are the reason this is not four lines: the single
    most client-facing number in the package, `Analysis.total_avoidable_month`,
    is a property and not a field. A `dataclasses.fields()` walk -- the obvious
    implementation, and the one I wrote first -- silently skips it, so an
    invariant built on it would have passed while the headline figure leaked.
    That is this file's own failure mode, one level up.
    """
    if _depth > 6:
        return
    if isinstance(obj, Figure):
        yield path, obj
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk_figures(v, f"{path}[{k!r}]", _depth + 1)
        return
    if isinstance(obj, (list, tuple, set, frozenset)):
        for i, v in enumerate(obj):
            yield from walk_figures(v, f"{path}[{i}]", _depth + 1)
        return
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for f in dataclasses.fields(obj):
            yield from walk_figures(getattr(obj, f.name), f"{path}.{f.name}",
                                    _depth + 1)
    for name, attr in vars(type(obj)).items():
        if isinstance(attr, property):
            try:
                value = getattr(obj, name)
            except Exception:
                continue
            yield from walk_figures(value, f"{path}.{name}", _depth + 1)


def _figure_returning_methods():
    """Nullary Figure->Figure methods, found by calling them.

    Named-list versions of this are what let `__abs__` drop `released_as` for
    as long as it did: every test listed the operations it already knew about,
    so the one nobody thought of was the one that broke.
    """
    probe = Figure(-1.0, money.MEASURED)
    out = []
    for name in dir(Figure):
        if not name.startswith("__") or name in ("__class__", "__init__"):
            continue
        attr = getattr(Figure, name, None)
        if not callable(attr):
            continue
        try:
            if isinstance(attr(probe), Figure):
                out.append(name)
        except Exception:
            continue
    return out


def renderers():
    """Every renderer the report module exposes, found by name at runtime.

    A `render_pdf` added tomorrow is covered without editing this file, which
    is the entire point: the text/HTML divergence happened because a test named
    the two renderers it knew about.
    """
    return [(n, getattr(report, n)) for n in dir(report)
            if n.startswith("render_") and callable(getattr(report, n))]


# --- fixtures --------------------------------------------------------------

def short_window_analysis():
    """A trace far below the projection floor, with an invoice that reconciles.

    Both halves matter. The window is one hour, so no monthly figure is
    supportable; the invoice reconciles to 0.0%, so the reconciliation gate
    releases everything it is asked to release. An earlier attempt at this
    fixture used a non-reconciling invoice, which withheld every figure for an
    unrelated reason and let me record the defect as unreproducible.
    """
    reqs = [Request(request_id=f"r{i}", sent_at=T0 + timedelta(seconds=60 * i),
                    model="claude-opus-5",
                    usage={"input_tokens": 0,
                           "cache_creation_input_tokens": 100_000,
                           "cache_read_input_tokens": 0},
                    segments=[], agent="a", ttl_requested="5m")
            for i in range(2)]
    ts = TraceSet(requests=reqs, tier=Tier.USAGE_ONLY)
    spend = analyze(ts, allow_unreconciled=True).spend["input_usd"].raw()
    return analyze(ts, invoice_usd=spend)


# --- INV-1 -----------------------------------------------------------------

class TestEveryProjectedFigureRespectsTheProjectionFloor(unittest.TestCase):
    """INV-1. Members discovered by walking the analysis, not by naming fields.

    `_monthly` is the one site that multiplies by a window ratio, so it marks
    what it builds and this test finds every one of them.
    """

    def test_the_fixture_is_actually_below_the_floor(self):
        """Guard the guard: an invariant on a fixture that does not reproduce
        the condition passes for the wrong reason, which is this file's topic."""
        a = short_window_analysis()
        self.assertLess(a.window_days, 1.0)
        self.assertEqual(a.reconciliation["delta_pct"], 0.0)

    def test_the_fixture_produces_projected_figures_to_check(self):
        """And that there is something to find. A walker that silently returns
        nothing makes every assertion below vacuously true."""
        a = short_window_analysis()
        found = [p for p, f in walk_figures(a) if f.projected]
        self.assertTrue(found, "no projected figures discovered — the walker or "
                               "the `projected` tag is broken, and INV-1 would "
                               "pass while checking nothing")

    def test_no_projected_figure_is_released_below_the_floor(self):
        a = short_window_analysis()
        leaked = [(p, f) for p, f in walk_figures(a) if f.projected and f.released]
        self.assertEqual(
            [], leaked,
            "these projected figures published from a window that cannot "
            "support a projection:\n" +
            "\n".join(f"    {p} = {f} (released_as={f.released_as})"
                      for p, f in leaked))


# --- INV-2 -----------------------------------------------------------------

class TestEveryFigureCarriesItsReleaseProvenance(unittest.TestCase):
    """INV-2. A released figure must say *how* it earned release, everywhere a
    figure is published -- including the JSON a script consumes."""

    def test_release_state_and_provenance_never_disagree(self):
        a = short_window_analysis()
        wrong = [(p, f) for p, f in walk_figures(a)
                 if bool(f.released) != bool(f.released_as)]
        self.assertEqual([], wrong,
                         "released/released_as disagree at: " +
                         ", ".join(p for p, _ in wrong))

    def test_provenance_survives_every_figure_operation(self):
        """Discovered by calling each Figure->Figure method, not by naming them.

        `__abs__` dropped `released_as`, laundering a draft figure into an
        invoice-checked one. It was found by reading, not by a test, because
        every test named the operations it knew about.

        Not asserted here: that `release(True)` on an already-released draft
        keeps it a draft. It does not -- `as_` defaults to RECONCILED, so a
        re-release silently upgrades provenance. That default is deliberate and
        documented, and all three call sites operate on a withheld figure
        straight from `_monthly`, so nothing reaches it. Asserting a property
        no caller exercises, against a documented choice, is how a test comes
        to encode an opinion rather than a requirement. Recorded in PENDING.md
        as a latent sharp edge instead.
        """
        draft = Figure(-12.34, money.MEASURED).release(True, as_=money.DRAFT)
        object.__setattr__(draft, "projected", True)
        for name in _figure_returning_methods():
            with self.subTest(op=name):
                out = getattr(draft, name)()
                self.assertEqual(money.DRAFT, out.released_as,
                                 f"{name} laundered DRAFT into {out.released_as!r}")
                self.assertTrue(out.projected, f"{name} dropped `projected`")

    def test_the_json_output_carries_release_state_for_every_figure(self):
        """Every dollar field in `--format json`, not only `spend`.

        A consumer cannot tell a withheld figure from a published one if the
        state is attached to one section and not the others.
        """
        a = short_window_analysis()
        payload = json.loads(_analysis_json(a))
        missing = []
        for section, entries in _json_money_sections(payload).items():
            for key in entries:
                if key not in payload.get("release_state", {}) and \
                        not _has_inline_state(payload, section, key):
                    missing.append(f"{section}.{key}")
        self.assertEqual([], missing,
                         "dollar fields in the JSON with no release state: " +
                         ", ".join(missing))


def _analysis_json(a: Analysis) -> str:
    """The CLI's own JSON, reached without re-implementing it here.

    Re-implementing is how the two renderers diverged. This drives the real
    code path so the invariant checks what a user actually receives.
    """
    from cacheeconomics import cli
    import argparse
    import io
    import contextlib
    ns = argparse.Namespace(format="json", client="", window_label="")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cli._emit_analysis_json(a, ns) if hasattr(cli, "_emit_analysis_json") \
            else print(_fallback_json(a))
    return buf.getvalue()


def _fallback_json(a: Analysis) -> str:
    """Mirrors `cmd_analyze`'s json branch for the fields this test reads.

    Marked clearly as a mirror: if `cmd_analyze` grows a section, this does not
    know, and the test above will under-report. That is a real limit of this
    invariant and the reason the CLI branch should be factored out.
    """
    return json.dumps({
        "spend": {k: str(v) for k, v in a.spend.items()},
        "release_state": {k: getattr(v, "released_as", "")
                          for k, v in a.spend.items()
                          if hasattr(v, "released_as")},
        "findings": [{"code": f.code,
                      "avoidable_usd_month": (str(f.avoidable_usd_month)
                                              if f.avoidable_usd_month else None)}
                     for f in a.findings],
    }, indent=2, default=str, allow_nan=False)


def _json_money_sections(payload: dict) -> dict:
    """Only the entries that actually carry money.

    `a.spend` also holds `window_days`, a plain float. Reading every key of the
    section flagged it as an ungoverned dollar field, which would have been a
    permanently-failing invariant reporting a defect that does not exist -- as
    corrosive as one that passes while a real defect ships, because the fix for
    a noisy invariant is always to stop reading it.
    """
    money_like = [k for k, v in payload.get("spend", {}).items()
                  if isinstance(v, str) and ("$" in v or v.startswith("[withheld"))]
    return {"spend": money_like,
            "findings": [f["code"] for f in payload.get("findings", [])
                         if f.get("avoidable_usd_month") is not None]}


def _has_inline_state(payload: dict, section: str, key: str) -> bool:
    if section != "findings":
        return False
    for f in payload.get("findings", []):
        if f.get("code") == key:
            return "release_state" in f or "avoidable_usd_month_state" in f
    return False


# --- INV-3 -----------------------------------------------------------------

class TestNoRendererPrintsAWithheldFigure(unittest.TestCase):
    """INV-3. Asserted over every renderer found at runtime, in both directions:
    withheld figures never show a number, and the renderer list is non-empty."""

    def test_there_are_renderers_to_check(self):
        self.assertTrue(renderers(), "no renderers discovered; INV-3 is vacuous")

    def test_no_withheld_figures_value_appears_in_any_rendering(self):
        """Each withheld figure gets a distinctive value; none may appear.

        The first version asserted that no `$`-amount appeared *anywhere*, and
        it failed against correct code: the HTML report echoes back the
        client's own invoice ("against invoice $1.25"), which is an input, not
        a withheld computation. Sentinel values ask the precise question --
        *did the number behind this withheld figure reach the page* -- and are
        immune to a figure that happens to equal the invoice, which in this
        fixture it does.
        """
        a = short_window_analysis()
        withheld, sentinels = _withhold_everything(a)
        self.assertTrue(sentinels, "nothing was withheld; the check is vacuous")
        for name, fn in renderers():
            with self.subTest(renderer=name):
                text = fn(withheld)
                leaked = sorted(
                    {f"{s:,.2f}" for s in sentinels
                     if f"{s:,.2f}" in text or f"{s:,.0f}" in text})
                self.assertEqual(
                    [], leaked,
                    f"{name} printed the value behind a withheld figure: {leaked}")


def _withhold_everything(a: Analysis):
    """Withhold every Figure, each stamped with a unique traceable value.

    Returns `(analysis, sentinels)`. The values are improbable and distinct so
    that finding one in a rendering identifies which figure leaked, rather than
    merely that something did.
    """
    counter = iter(range(1, 10_000))
    sentinels = []

    def hide(v):
        if not isinstance(v, Figure):
            return v
        s = 900_000.0 + next(counter) * 111.11
        sentinels.append(s)
        return Figure(s, v.basis, released=False,
                      withheld_because="test: withheld", projected=v.projected)

    spend = {k: hide(v) for k, v in a.spend.items()}
    recon = {k: hide(v) for k, v in a.reconciliation.items()}
    findings = [dataclasses.replace(f, avoidable_usd_month=hide(f.avoidable_usd_month))
                for f in a.findings]
    return dataclasses.replace(a, spend=spend, findings=findings,
                               reconciliation=recon), sentinels


# --- INV-4 -----------------------------------------------------------------

class TestNoMutatingEntryPointDefaultsToASurface(unittest.TestCase):
    """INV-4. Members found with `inspect.signature`, over everything public in
    the plugin module -- so a new entry point is covered without edits here.

    A surface default is not cosmetic: `target_id` selects the rate table and
    the capability limits. Defaulting it means a call that never named a
    surface is billed and bounded as though it had, and `on_request` did
    exactly that while also defaulting `apply=True`.
    """

    MUTATION_SWITCHES = ("apply", "mutate")

    def _public_callables(self):
        for name, obj in vars(plugin).items():
            if name.startswith("_"):
                continue
            if inspect.isfunction(obj):
                yield f"plugin.{name}", obj
            elif inspect.isclass(obj) and obj.__module__ == plugin.__name__:
                for mname, m in vars(obj).items():
                    if not mname.startswith("_") and inspect.isfunction(m):
                        yield f"plugin.{name}.{mname}", m

    def test_there_are_entry_points_to_check(self):
        found = [n for n, _ in self._public_callables()]
        self.assertTrue(found, "no public plugin callables discovered")

    def test_no_mutating_entry_point_has_a_real_surface_default(self):
        offenders = []
        for name, fn in self._public_callables():
            sig = inspect.signature(fn)
            if "target_id" not in sig.parameters:
                continue
            can_mutate = any(s in sig.parameters for s in self.MUTATION_SWITCHES)
            if not can_mutate:
                continue
            default = sig.parameters["target_id"].default
            if default is inspect.Parameter.empty:
                continue
            if default != registry.UNATTRIBUTED:
                offenders.append(f"{name}(target_id={default!r})")
        self.assertEqual(
            [], offenders,
            "these can mutate a request and default to a named surface, so a "
            "caller that never chose one is priced and bounded as though it "
            "had: " + ", ".join(offenders))

    def test_mutation_defaults_to_off(self):
        """The other half of the same door. A surface guard does not help if
        mutation happens by default before anyone considered the surface."""
        offenders = []
        for name, fn in self._public_callables():
            sig = inspect.signature(fn)
            for switch in self.MUTATION_SWITCHES:
                p = sig.parameters.get(switch)
                if p is not None and p.default is True:
                    offenders.append(f"{name}({switch}=True)")
        self.assertEqual([], offenders,
                         "mutation is on by default at: " + ", ".join(offenders))


# --- INV-5 -----------------------------------------------------------------

class TestEveryRegistryDependencyThatDisablesACheckIsAnnounced(unittest.TestCase):
    """INV-5. Dependencies discovered by *watching which registry keys the
    checks actually read*, not by grepping the source for them.

    The failure being prevented is a quiet dashboard. Three shape checks return
    early when the registry cannot answer for a surface, and an operator reads
    the resulting silence as a clean bill of health. RT-NOSURFACE exists to
    break that silence -- but it probes only `min_cacheable_tokens`, so a
    surface that records a minimum and no breakpoint budget disables the budget
    check with nothing said.
    """

    def _observed_dependencies(self):
        """Run the monitor with a recording registry and collect what it asked
        for. This is the discovery step: it reports what the code does, not
        what a comment says it does."""
        asked = set()
        real_capability = registry.capability
        real_min = registry.min_cacheable_tokens

        def spy_capability(target_id, name, allow_contested=False):
            asked.add(("capability", name))
            return real_capability(target_id, name, allow_contested)

        def spy_min(target_id, model):
            asked.add(("min_cacheable_tokens", None))
            return real_min(target_id, model)

        registry.capability = spy_capability
        registry.min_cacheable_tokens = spy_min
        try:
            m = monitor.Monitor()
            for i in range(30):
                m.observe(_shape_request(i))
        finally:
            registry.capability = real_capability
            registry.min_cacheable_tokens = real_min
        return asked

    def test_the_checks_read_something_from_the_registry(self):
        self.assertTrue(self._observed_dependencies(),
                        "no registry reads observed; INV-5 is vacuous")

    def test_each_registry_dependency_announces_itself_when_unavailable(self):
        """For every key the checks read, a surface that cannot answer it must
        produce an alert rather than a silent early return."""
        silent = []
        for kind, name in sorted(self._observed_dependencies(),
                                 key=lambda d: (d[0], d[1] or "")):
            with self.subTest(dependency=f"{kind}:{name}"):
                alerts = _run_with_failing(kind, name)
                if not alerts:
                    silent.append(f"{kind}({name})" if name else kind)
        self.assertEqual(
            [], silent,
            "these registry dependencies disable a check with no alert, so the "
            "operator sees silence and reads it as healthy: " + ", ".join(silent))


def _shape_request(i):
    segs = [Segment(id="sys", role="system", tokens=8000, index=0,
                    label="instructions", cache_marked=True, ttl="5m"),
            Segment(id=f"t{i}", role="user", tokens=100, index=1,
                    label="user_turn")]
    return Request(request_id=f"r{i}", sent_at=T0 + timedelta(seconds=60 * i),
                   model="claude-opus-5",
                   usage={"input_tokens": 100, "cache_creation_input_tokens": 0,
                          "cache_read_input_tokens": 0},
                   segments=segs, agent="a", target_id="anthropic/direct",
                   ttl_requested="5m")


def _run_with_failing(kind, name):
    """Observe a stream where exactly one registry dependency is unavailable."""
    real_capability = registry.capability
    real_min = registry.min_cacheable_tokens

    def cap(target_id, cname, allow_contested=False):
        if kind == "capability" and cname == name:
            raise registry.RegistryError(f"test: {cname} unavailable")
        return real_capability(target_id, cname, allow_contested)

    def mn(target_id, model):
        if kind == "min_cacheable_tokens":
            raise registry.RegistryError("test: minimum unavailable")
        return real_min(target_id, model)

    registry.capability = cap
    registry.min_cacheable_tokens = mn
    try:
        m, fired = monitor.Monitor(), []
        for i in range(30):
            fired += m.observe(_shape_request(i))
    finally:
        registry.capability = real_capability
        registry.min_cacheable_tokens = real_min
    return fired


if __name__ == "__main__":
    unittest.main()
