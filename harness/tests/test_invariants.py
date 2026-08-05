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


def figure_round_trips():
    """Every Figure -> Figure path, as `(label, callable)`.

    Was a `dir()` probe that kept only dunder names callable with no arguments.
    That found exactly one operation -- `__abs__` -- so an invariant advertising
    "survives every Figure operation" checked one, and `release()`, pickling,
    copying, equality and hashing all went unexamined while a new `projected`
    field was threaded through every one of them.

    Discovery is still real, in the direction that matters: the *table* is
    explicit, and `test_the_round_trip_table_covers_every_state_carrier` fails
    if `Figure.__slots__` grows a field this table does not exercise. Naming the
    operations while checking the *fields* automatically is the way round that
    holds -- a new slot is the thing likely to be forgotten, not a new dunder.
    """
    import copy
    import pickle
    return [
        ("abs()", lambda f: abs(f)),
        # The literal default call, with no `as_`. The first version of this
        # entry passed `as_=f.released_as`, so the one path whose label it bore
        # -- plain `release(True)` on an already-released draft -- was the exact
        # path it never took. A lambda that steps around the case its label
        # names is worse than no case at all: it reports coverage it does not
        # have. Found by external review, not by me.
        ("release(True)", lambda f: f.release(True)),
        ("release(True, as_=DRAFT)", lambda f: f.release(True, as_=money.DRAFT)),
        ("release(False)", lambda f: f.release(False, "test")),
        ("pickle", lambda f: pickle.loads(pickle.dumps(f))),
        ("copy", lambda f: copy.copy(f)),
        ("deepcopy", lambda f: copy.deepcopy(f)),
    ]


# Every slot on `Figure` that carries state rather than the amount itself.
# `withheld_because` is here because the guard below caught its absence: I
# listed the slots I had just been editing and missed the one I had not, which
# is the failure this whole file is about, occurring inside it.
STATE_SLOTS = ("basis", "released", "released_as", "projected",
               "withheld_because")


def _unwrap_callable(raw):
    """The underlying function behind a class attribute, or None.

    `inspect.isfunction` is False for `classmethod` and `staticmethod` entries
    in `vars(cls)` -- they are descriptors, not functions -- so a signature walk
    built on it skips every factory. Measured: `cost.Usage.from_anthropic` is a
    classmethod and was absent from the 164 callables INV-4 claimed to inspect.

    Also unwraps `functools.wraps` chains and `functools.partial`, both of which
    hide the real signature INV-4 reads.
    """
    import functools
    for _ in range(8):
        if isinstance(raw, (classmethod, staticmethod)):
            raw = raw.__func__
            continue
        if isinstance(raw, functools.partial):
            raw = raw.func
            continue
        wrapped = getattr(raw, "__wrapped__", None)
        if wrapped is not None and wrapped is not raw:
            raw = wrapped
            continue
        break
    return raw if inspect.isfunction(raw) else None


def renderers():
    """Every renderer the report module exposes, found by name at runtime.

    A `render_pdf` added tomorrow is covered without editing this file, which
    is the entire point: the text/HTML divergence happened because a test named
    the two renderers it knew about.
    """
    return [(n, getattr(report, n)) for n in dir(report)
            if n.startswith("render_") and callable(getattr(report, n))]




def _fallbacks_in(tree, label):
    """The AST shapes meaning "a surface supplied when nobody named one".

    Split out so the detector can be pointed at a synthetic source. The guard
    proving this mechanism works used to assert the package HAD offenders,
    which stopped being a fair question the moment the class was fully closed.
    """
    import ast
    ids = {t for t in registry.target_ids() if t != registry.UNATTRIBUTED}

    def surface(n):
        return (isinstance(n, ast.Constant) and isinstance(n.value, str)
                and n.value in ids)

    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.arguments):
            for d in list(node.defaults) + [k for k in node.kw_defaults if k]:
                if surface(d):
                    out.append(f"{label} parameter default")
        elif isinstance(node, (ast.AnnAssign, ast.Assign)):
            v = node.value
            if v is not None and surface(v):
                out.append(f"{label} field default")
        elif isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
            for v in node.values[1:]:
                if surface(v):
                    out.append(f"{label} `or` fallback")
        elif isinstance(node, ast.Call):
            fn = node.func
            if (isinstance(fn, ast.Attribute) and fn.attr == "get"
                    and len(node.args) == 2 and surface(node.args[1])):
                out.append(f"{label} dict.get fallback")
            for kw in node.keywords:
                if kw.arg in ("default", "target_id") and surface(kw.value):
                    out.append(f"{label} {kw.arg}= keyword")
    return out


def surface_fallback_sites():
    """Every place package source supplies a first-party surface as a FALLBACK.

    INV-4 reads `inspect.signature`, so it sees parameter defaults and nothing
    else. Measured on main after the signature class was closed: four sites
    still fabricated `anthropic/direct` and INV-4 passed green -- an argparse
    `default=`, and three `or` expressions in the function body. The shipped
    `cacheeconomics checks` command answered with Anthropic's 512-token minimum
    for a caller who never named a surface, and the same prefix FAILS on
    openai/direct at 1024.

    So this walks the AST instead, and looks for the *shape* rather than the
    string: a surface id standing where a value goes when nobody supplied one.
    Four shapes, plus class-level field defaults, which the first version of
    this function missed -- `trace.py`'s `target_id: str = "anthropic/direct"`
    is an `AnnAssign`, not `ast.arguments`, so a detector written from the
    other four found eleven sites and quietly skipped the twelfth. Caught by
    cross-checking against a broader scan, which is the only reason this
    docstring is not making the same overconfident claim as the last one.

    Deliberately NOT flagged: a surface id used as a VALUE, in a mapping that
    translates something observed (`adapters/litellm.py`'s provider table) or
    in a comparison. Those name a surface because the data named it. Flagging
    them would make this noisy, and a noisy invariant gets switched off.
    """
    import ast
    import pathlib
    root = pathlib.Path(registry.HERE)
    out = []
    for f in sorted(root.rglob("*.py")):
        if f.name == "registry.py":
            continue
        out.extend(_fallbacks_in(ast.parse(f.read_text()),
                                 str(f.relative_to(root.parent))))
    # Counted, not de-duplicated, and keyed on file+shape rather than
    # file:line. Line numbers churn on every unrelated edit in the same file --
    # the first version went red because a help string three lines above moved
    # two entries -- and an invariant that cries wolf on unrelated edits gets
    # its expected-set pasted over without reading, which is worse than not
    # having it.
    from collections import Counter
    return sorted(f"{k} x{n}" for k, n in Counter(out).items())



def help_text_surface_claims():
    """argparse help strings that name a surface the default does not supply.

    Found by Track B, invisible to `surface_fallback_sites()` because it is not
    a fallback shape -- the CODE is right and the SENTENCE is wrong. `cli.py`'s
    `--target-id` correctly uses `default=None`, with a comment explaining that
    "the operator chose anthropic/direct" and "the operator said nothing" must
    stay distinguishable, and then tells the operator in its help text
    "(default: anthropic/direct)".

    That is this whole project's failure mode in one line: prose asserting a
    behaviour nothing checked, sitting directly above the code that contradicts
    it. An operator reading `--help` to decide whether they need the flag is
    told they do not.
    """
    import ast
    import pathlib
    ids = {t for t in registry.target_ids() if t != registry.UNATTRIBUTED}
    out = []
    root = pathlib.Path(registry.HERE)
    for f in sorted(root.rglob("*.py")):
        for node in ast.walk(ast.parse(f.read_text())):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_argument"):
                continue
            kw = {k.arg: k.value for k in node.keywords}
            h = kw.get("help")
            if not (isinstance(h, ast.Constant) and isinstance(h.value, str)):
                continue
            # Only a DEFAULT CLAIM counts, not a mention. Track B's new help
            # text names surfaces as examples -- "e.g. anthropic/direct,
            # openai/direct, amazon-bedrock/converse" -- which is correct and
            # useful, and the first version of this check flagged all three.
            # An invariant that fires on good help text gets the help text
            # changed to appease it, which is worse than not having the check.
            #
            # The claim being hunted is "this flag defaults to <surface>" when
            # it does not. That needs the word near the name, not the name.
            import re as _re
            claim = _re.compile(
                r"default[s]?\b[^.;]{0,40}?(" + "|".join(
                    _re.escape(i) for i in ids) + r")", _re.I)
            named = {m.group(1) for m in claim.finditer(h.value)}
            if not named:
                continue
            d = kw.get("default")
            actual = d.value if isinstance(d, ast.Constant) else None
            if actual not in named:
                out.append(f"{f.relative_to(root.parent)} "
                           f"help says {sorted(named)} but default is {actual!r}")
    return sorted(out)


# --- known defects, as exact sets -----------------------------------------
#
# These replace `@unittest.expectedFailure`, which round 3 of external review
# killed and was right to. `unittest.expectedFailure` treats an ERROR as an
# expected failure too, so a discovery helper that regressed to returning
# nothing, or raised before reaching its assertion, still reported OK. The
# marker was introduced to stop a red board hiding regressions and would have
# hidden them itself. Measured: a test that raises under @expectedFailure exits 0.
#
# An exact set is strictly better and needs no marker. Verified in all three
# directions before being relied on:
#
#   harness breaks (discovery empty)  -> guard tests fail          (3 failed)
#   defect fixed, set not updated     -> set shrinks, fails        (1 failed)
#   defect widens or a new one lands  -> set grows, fails          (1 failed)
#   nothing changes                   -> green, and honestly so
#
# A track closing one of these edits its set to `()`. Leaving it stale fails.

# INV-1, Track A. A 1-hour window publishes these as released projections.
KNOWN_PROJECTION_LEAKS = ()   # closed by Track A

# INV-2, Track A. Money in --format json with no state scoped to that field.
KNOWN_JSON_UNPROVENANCED = ()   # closed by Track A

# INV-4, Track B. Public callables defaulting target_id to a named surface.
KNOWN_SURFACE_DEFAULTS = ()   # closed by Track B

# INV-4, Track B. Entry points that mutate unless told not to.
KNOWN_MUTATE_BY_DEFAULT = ()   # closed by Track B

# INV-6. Every place the package supplies a first-party surface when nobody
# named one. Found by AST shape, not by name, so it covers routes
# `inspect.signature` cannot see. Line numbers move -- re-run
# `surface_fallback_sites()` and paste, do not hand-edit.
KNOWN_SURFACE_FALLBACKS = ()   # closed by Track B

# INV-6b. argparse help text naming a surface its default does not supply.
# The shared ingest flag is fixed. The remaining entry belongs to
# `cmd_claude_code`, whose flags Track B is rewriting to add an explicit
# `--assume-anthropic-direct` opt-in -- its help text changes with that work, so
# correcting the sentence here would collide with the fix for the behaviour it
# describes.
KNOWN_HELP_TEXT_CLAIMS = ()   # closed by Track B

# INV-5, Track C. Registry dependencies that disable a check in silence.
KNOWN_SILENT_ABSTENTIONS = (
    "capability(lookback_blocks)",
    "capability(max_breakpoints)",
)

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
                    segments=[], agent="a", ttl_requested="5m",
                    # Named explicitly, not inherited from a default. Once
                    # `Request.target_id` stopped defaulting to a real surface,
                    # these requests became unpriceable and the fixture's spend
                    # fell to $0.00 with `delta_pct` None -- so every figure was
                    # withheld for the wrong reason and INV-1 reported
                    # "unexpected success" while checking nothing.
                    #
                    # `test_the_fixture_is_actually_below_the_floor` caught it
                    # by failing on `None != 0.0`, which is the entire reason
                    # that guard exists: it fails loudly instead of letting the
                    # invariant above it go quietly green.
                    target_id="anthropic/direct")
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
            sorted(KNOWN_PROJECTION_LEAKS), sorted(q for q, _ in leaked),
            "the set of projected figures published from a window that cannot "
            "support one has CHANGED. If you closed some, update "
            "KNOWN_PROJECTION_LEAKS (or set it to `()`):\n" +
            "\n".join(f"    {p} = {f} (released_as={f.released_as})"
                      for p, f in leaked))

    def test_every_monthly_named_figure_is_actually_tagged(self):
        """The bypass check: `projected` is a tag, and a tag can be skipped.

        The test above finds figures *marked* projected. A monthly figure built
        without going through `_monthly` would be semantically projected,
        untagged, and therefore invisible to it -- the invariant would go green
        while the defect it was written for shipped. Acknowledging that in a
        docstring, which the first version did, does not close it.

        So this asks the question from the other side, using a signal the code
        cannot fake: the *name of the field the figure is stored in*. Anything
        reachable at a path saying "month" must carry the tag. A new monthly
        figure that skips `_monthly` fails here, because it still has to be
        called something.

        Not a proof either. A projected figure stored under a name with no
        "month" in it evades both checks. Together they require a bypass to be
        both untagged and unnamed, which is a good deal narrower than either
        alone.
        """
        a = short_window_analysis()
        untagged = [p for p, f in walk_figures(a)
                    if "month" in p.lower() and not f.projected]
        self.assertEqual(
            [], untagged,
            "figures stored under a monthly name that never went through "
            "`_monthly`, so the projection floor cannot see them: " +
            ", ".join(untagged))


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
        draft = Figure(-12.34, money.MEASURED, released=True,
                       released_as=money.DRAFT, projected=True)
        for label, op in figure_round_trips():
            with self.subTest(op=label):
                out = op(draft)
                self.assertTrue(out.projected,
                                f"{label} dropped `projected`")
                if out.released:
                    self.assertTrue(out.released_as,
                                    f"{label} released a figure with no provenance")
                if label != "release(False)":
                    # Includes the bare `release(True)`: re-releasing a draft
                    # must not silently upgrade it to invoice-checked.
                    self.assertEqual(
                        money.DRAFT, out.released_as,
                        f"{label} laundered DRAFT into {out.released_as!r}")

    def test_the_round_trip_table_covers_every_state_carrying_slot(self):
        """If `Figure` grows a slot, this fails until the table exercises it.

        The point of failure this guards is precise and has already happened
        once: a field was added to `__slots__` and threaded through some of the
        paths that rebuild a Figure. Naming operations by hand is fine; letting
        the *fields* go unchecked is not.
        """
        carriers = [s for s in Figure.__slots__ if s not in ("_usd",)]
        self.assertEqual(
            sorted(STATE_SLOTS), sorted(carriers),
            "Figure.__slots__ changed. Add the new field to STATE_SLOTS and "
            "make sure every entry in figure_round_trips() preserves it.")

    def test_state_carrying_slots_survive_every_round_trip(self):
        """The generic form: no round trip may alter any state slot.

        Written against the slot list rather than against named attributes, so
        a field added tomorrow is checked on every path from the moment
        `STATE_SLOTS` learns about it.
        """
        original = Figure(42.0, money.MODELED, released=True,
                          released_as=money.DRAFT, projected=True)
        for label, op in figure_round_trips():
            if label.startswith("release("):
                continue        # release deliberately changes release state
            out = op(original)
            for slot in STATE_SLOTS:
                with self.subTest(op=label, slot=slot):
                    self.assertEqual(
                        getattr(original, slot), getattr(out, slot),
                        f"{label} changed {slot}")

    def test_equality_and_hash_see_every_state_slot(self):
        """Two figures differing only in a state slot must not compare equal.

        `__eq__` and `__hash__` were both updated for `projected`; nothing
        checked that they were. A field invisible to equality lets a withheld
        and a released figure test as interchangeable in any set or dict.
        """
        base = dict(released=True, released_as=money.RECONCILED, projected=False)
        a = Figure(10.0, money.MODELED, **base)
        for slot, other in (("projected", {**base, "projected": True}),
                            ("released_as", {**base, "released_as": money.DRAFT}),
                            ("released", {**base, "released": False,
                                          "released_as": ""})):
            with self.subTest(slot=slot):
                b = Figure(10.0, money.MODELED, **other)
                self.assertNotEqual(a, b, f"__eq__ ignores {slot}")
                self.assertNotEqual(hash(a), hash(b), f"__hash__ ignores {slot}")

    def test_the_json_output_carries_release_state_for_every_figure(self):
        """Every dollar field in `--format json`, found by scanning the payload.

        Was a hand-listed pair of sections -- `spend` and `findings` -- which
        is how `reconciliation.computed_usd` and `reconciliation.delta_usd`
        shipped as bare dollar strings with no provenance while a test named
        "every figure" passed. Named sections are the same mistake as named
        modules and named methods, which this file has now made three times.

        So: walk the whole decoded payload for anything that looks like money,
        and require state for each. A section added to `analysis_json`
        tomorrow is covered without editing this test.
        """
        a = short_window_analysis()
        payload = json.loads(_analysis_json(a))
        found = sorted(_money_paths(payload))
        self.assertTrue(found, "no money-like fields found in the JSON; "
                               "this invariant would pass vacuously")
        missing = [p for p in found if not _has_release_state(payload, p)]
        self.assertEqual(
            sorted(KNOWN_JSON_UNPROVENANCED), sorted(missing),
            "dollar fields in --format json with no release state, so a script "
            "consuming this cannot tell a published figure from a withheld or "
            "draft one:\n    " + "\n    ".join(missing))


def _analysis_json(a: Analysis) -> str:
    """The CLI's own JSON, via the CLI's own function. No fallback.

    The previous version called `cli._emit_analysis_json` *if it existed* and
    otherwise used a local mirror. It did not exist. So this invariant spent
    its whole life checking a copy of the serializer that lived in the test
    file, which of course agreed with itself, while the real `--format json`
    shipped finding figures carrying no release state.

    `getattr` with a fallback is how a test quietly stops testing the thing it
    names. If the seam disappears, this raises.
    """
    from cacheeconomics import cli
    return cli.analysis_json(a, tier_name="USAGE_ONLY", coverage=1.0)


def _money_paths(node, path="") -> list:
    """Every path in the decoded JSON whose value looks like a money figure.

    A `Figure` serialises through `str`, so it is either "$..." or
    "[withheld: ...]". Both are money: a withheld one still needs to be
    identifiable as withheld rather than absent.

    Plain floats are deliberately not money-like. `window_days` lives in the
    same dict as the figures and flagging it produced a permanently-failing
    invariant reporting a defect that does not exist -- which gets switched off,
    and then it protects nothing.
    """
    out = []
    leaf = path.rsplit(".", 1)[-1].split("[")[0]
    if isinstance(node, str):
        if node.startswith("[withheld") or _looks_like_usd(node):
            out.append(path)
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        # A raw number under a money-shaped name. Only rendered strings were
        # matched before, so a future field emitting a bare float -- the most
        # likely way this JSON grows -- was invisible to a scan claiming to
        # cover every dollar field.
        if leaf.endswith("_usd") and leaf not in _INPUT_ONLY_MONEY:
            out.append(path)
    elif isinstance(node, dict):
        for k, v in node.items():
            if k == "release_state":
                continue
            out.extend(_money_paths(v, f"{path}.{k}" if path else k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            out.extend(_money_paths(v, f"{path}[{i}]"))
    return out


# Money the client supplied, not money this tool computed. It carries no
# release state because there is nothing to release: echoing an invoice back is
# not a claim. Exempting it explicitly beats letting the scan quietly skip
# whatever it fails to match.
_INPUT_ONLY_MONEY = frozenset({"invoice_usd"})


def _looks_like_usd(s: str) -> bool:
    import re
    return bool(re.fullmatch(r"\$-?[\d,]+(?:\.\d+)?", s.strip()))


def _has_release_state(payload: dict, path: str) -> bool:
    """Is there provenance for the money at `path`, specifically?

    Three ways the first version said yes when the answer was no, all the same
    mistake -- accepting evidence about *something else* as evidence about this
    field:

      - `leaf in payload['release_state']` matched by leaf name at any depth, so
        a top-level entry vouched for a same-named field nested anywhere else;
      - `'release_state' in node` was true if the container had a state map at
        all, whatever it covered: `{'avoidable_usd_month': '$12',
        'release_state': {'other': 'draft'}}` passed;
      - presence alone counted, so state recorded as `""` read as provenance.

    Now the state must be scoped to the container the money sits in, keyed by
    that exact leaf, and *consistent with what was rendered* -- `""` for a
    withheld figure, a RELEASES member for a published one. Requiring merely
    non-empty flagged `monthly_input_usd`, which is correctly withheld, and an
    invariant that fails on correct code gets switched off.
    """
    parts = path.split(".")
    leaf = parts[-1]
    node = payload
    for p in parts[:-1]:
        if "[" in p:
            name, idx = p.split("[")
            node = node.get(name, [])
            i = int(idx.rstrip("]"))
            if not isinstance(node, list) or i >= len(node):
                return False
            node = node[i]
        else:
            node = node.get(p, {})
        if not isinstance(node, (dict, list)):
            return False
    if not isinstance(node, dict):
        return False
    if len(parts) == 2 and parts[0] == "spend":
        state = payload.get("release_state", {})
    else:
        state = node.get("release_state", {})
    if not isinstance(state, dict):
        return False
    sentinel = object()
    value = state.get(leaf, node.get(f"{leaf}_state", sentinel))
    if value is sentinel:
        return False
    rendered = _rendered_at(payload, path)
    if isinstance(rendered, str) and rendered.startswith("[withheld"):
        return value == ""
    return value in money.RELEASES


def _rendered_at(payload: dict, path: str):
    node = payload
    for p in path.split("."):
        if "[" in p:
            name, idx = p.split("[")
            node = node.get(name, [])
            i = int(idx.rstrip("]"))
            if not isinstance(node, list) or i >= len(node):
                return None
            node = node[i]
        elif isinstance(node, dict):
            node = node.get(p)
        else:
            return None
    return node


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
        """Every public callable in the package, not just the plugin module.

        The first version walked `plugin` alone, and that scoping *was* the
        defect one level up: `Recorder.__init__` defaults
        `target_id='anthropic/direct'` and lives in `cacheeconomics.recorder`,
        so an invariant named after one module would have gone green while an
        identical door stood open in the module next to it. An invariant that
        picks its own scope narrowly proves only that the author looked where
        they already knew to look.

        `__init__` is included deliberately: a constructor that takes a surface
        and a mutation switch is an entry point, whatever it is named.
        """
        import importlib
        import pkgutil
        import cacheeconomics

        # `walk_packages`, not `iter_modules`: the latter stops at the top
        # level, and `cacheeconomics.adapters.claude_code.load_sessions`
        # defaults `target_id='anthropic/direct'` one package down. So the
        # invariant reported seven offenders with confidence while an eighth
        # sat in a subpackage it never opened. Twice now this test has picked
        # its own scope too small and gone green on the strength of it.
        seen = set()
        for mod_info in pkgutil.walk_packages(cacheeconomics.__path__,
                                              prefix="cacheeconomics."):
            short = mod_info.name[len("cacheeconomics."):]
            if any(p.startswith("_") for p in short.split(".")):
                continue
            try:
                mod = importlib.import_module(mod_info.name)
            except Exception as e:                                # noqa: BLE001
                # Loud, not silent. A module that fails to import is a module
                # this invariant cannot vouch for, and swallowing that turns an
                # unopened door into a clean bill of health.
                raise AssertionError(
                    f"cannot import {mod_info.name}, so INV-4 cannot claim to "
                    f"have inspected it: {e!r}") from e
            for name, obj in vars(mod).items():
                if name.startswith("_"):
                    continue
                if inspect.isfunction(obj) and obj.__module__ == mod.__name__:
                    key = f"{short}.{name}"
                    if key not in seen:
                        seen.add(key)
                        yield key, obj
                elif inspect.isclass(obj) and obj.__module__ == mod.__name__:
                    for mname, raw in vars(obj).items():
                        m = _unwrap_callable(raw)
                        if m is None:
                            continue
                        if mname.startswith("_") and mname != "__init__":
                            continue
                        key = f"{short}.{name}.{mname}"
                        if key not in seen:
                            seen.add(key)
                            yield key, m

    def test_there_are_entry_points_to_check(self):
        found = [n for n, _ in self._public_callables()]
        self.assertTrue(found, "no public plugin callables discovered")

    def test_no_entry_point_defaults_to_a_named_surface(self):
        """Not "no *mutating* entry point" -- no entry point at all.

        The narrower version was the second scoping mistake in this one test.
        `target_id` selects the rate table and the capability limits on every
        path that reads it, not only the ones that rewrite a request: a static
        check that silently answers for `anthropic/direct` while the caller runs
        on Bedrock returns a confident PASS from the wrong surface's minimum.
        The package's stated rule is default-deny -- a surface earns first-party
        rates by being *named* -- and a parameter default is exactly how a
        surface gets named without anyone choosing it.

        Requiring a mutation switch alongside narrowed this from 7 members to
        2, and the 2 it kept were the 2 the review had already found by hand.

        `None` is not flagged: it means "unspecified", and what a callee does
        with it is that callee's contract. A named surface is different -- it
        is an answer nobody gave.
        """
        offenders = []
        for name, fn in self._public_callables():
            p = inspect.signature(fn).parameters.get("target_id")
            if p is None or p.default is inspect.Parameter.empty:
                continue
            if p.default is None or p.default == registry.UNATTRIBUTED:
                continue
            offenders.append(f"{name}(target_id={p.default!r})")
        self.assertEqual(
            sorted(KNOWN_SURFACE_DEFAULTS), sorted(offenders),
            "the set of entry points defaulting to a named surface has CHANGED. "
            "If you closed some, update KNOWN_SURFACE_DEFAULTS (or set it to "
            "`()`):\n    " +
            "\n    ".join(sorted(offenders)))

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
        self.assertEqual(sorted(KNOWN_MUTATE_BY_DEFAULT), sorted(offenders),
                         "the set of entry points mutating by default has "
                         "CHANGED; update KNOWN_MUTATE_BY_DEFAULT (or set it "
                         "to `()`): " + ", ".join(offenders))


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
            sorted(KNOWN_SILENT_ABSTENTIONS), sorted(silent),
            "the set of registry dependencies that disable a check with no "
            "alert has CHANGED; update KNOWN_SILENT_ABSTENTIONS (or set it to "
            "`()`): " + ", ".join(silent))


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



class TestNoSurfaceIsFabricatedAnywhereInTheSource(unittest.TestCase):
    """INV-6. The class INV-4 structurally cannot see.

    `inspect.signature` shows parameter defaults. It does not show an argparse
    `default=`, an `or` fallback in a function body, a `dict.get` second
    argument, or a dataclass field default -- and a surface fabricated by any
    of those is priced and bounded exactly like one fabricated by a parameter
    default. INV-4 went green on a package with four such sites.

    Whether a caller "named a surface" is a property of the source, so the
    source is what this reads.
    """

    def test_the_scanner_can_still_find_a_fallback(self):
        """Prove the MECHANISM works, not that offenders exist.

        The first version asserted `surface_fallback_sites()` was non-empty, so
        the invariant could not pass on a broken scan. That was right while the
        class had members and became WRONG the moment Track B closed the last
        one: "no offenders" and "the scanner is broken" are the same
        observation, and the guard read the good news as the bad.

        A guard that cannot tell success from failure is the defect this file
        exists to catch. It now scans a synthetic source carrying a known
        fallback, which answers "does the detector detect" without depending on
        whether the package currently offends.
        """
        import ast
        found = _fallbacks_in(
            ast.parse('def go(target_id="anthropic/direct"):\n    pass\n'),
            "probe.py")
        self.assertEqual(["probe.py parameter default"], found,
                         "the AST fallback detector no longer detects a "
                         "parameter default it is looking straight at")

    def test_the_scan_ignores_legitimate_surface_names(self):
        """Guard against over-reach in the other direction.

        `adapters/litellm.py` maps observed provider names onto surface ids.
        That is a translation table, not a fabrication -- the surface is named
        because the data named it. An invariant that flags it is noise, and
        noise is how an invariant gets deleted.
        """
        found = " ".join(surface_fallback_sites())
        self.assertNotIn("adapters/litellm.py", found,
                         "the provider translation table is being reported as "
                         "a fabricated surface")


    def test_no_help_text_promises_a_surface_the_default_does_not_supply(self):
        """The documentation half of the same class.

        INV-6 above reads what the code DOES. This reads what the code SAYS it
        does, because an operator deciding whether they need `--target-id`
        reads the help string, not the source.
        """
        self.assertEqual(
            sorted(KNOWN_HELP_TEXT_CLAIMS), help_text_surface_claims(),
            "argparse help text names a surface its default does not actually "
            "supply, so `--help` tells the operator something the code does "
            "not do:\n    " + "\n    ".join(help_text_surface_claims()))

    def test_no_new_site_fabricates_a_surface(self):
        self.assertEqual(
            sorted(KNOWN_SURFACE_FALLBACKS), surface_fallback_sites(),
            "the set of places the package supplies a first-party surface "
            "when nobody named one has CHANGED. If you closed some, update "
            "KNOWN_SURFACE_FALLBACKS (or set it to `()`).")


if __name__ == "__main__":
    unittest.main()
