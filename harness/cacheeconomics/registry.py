"""Load and query the provider registry.

The registry is the point of the whole exercise: machine-readable, dated,
sourced facts about what each provider surface can actually do. Instrumentation
and prefix linting already exist elsewhere. A capability registry with
provenance and a contested-row rule does not.

Two rules are enforced here rather than left to callers:

1. A contested row cannot be read without asking for it. `target()` refuses to
   return a row flagged contested unless `allow_contested=True`, because the
   failure mode this guards against is one wrong row quietly discrediting the
   other twenty.

2. Pricing is date-effective. `base_rate()` requires a date. There is no
   "current price" API, because a static answer to that question goes stale
   silently and every downstream dollar figure inherits the error.
"""

from __future__ import annotations

import json
import re
import os
from datetime import date, datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
# Inside the package, not beside it. This used to be `../registry`, which works
# from a checkout and breaks the moment the package is installed: `..` from
# site-packages/cacheeconomics is site-packages, so a wheel that did not also
# ship a top-level `registry/` raised "registry file missing" on first use. It is
# `data/` rather than `registry/` because `registry.py` already owns that name in
# this package and a directory beside it would shadow the module.
REGISTRY_DIR = os.path.join(HERE, "data")

STALE_AFTER_DAYS = 90


class RegistryError(Exception):
    pass


class ContestedRow(RegistryError):
    """Raised when a caller reads a disputed fact without acknowledging it."""


class UnpriceableSurface(RegistryError):
    """Raised when the rate table does not cover the surface being priced.

    Distinct from a missing model row because the remedy is different and the
    caller reports it to a client. "No pricing recorded for claude-haiku-4-5"
    sends them to add a registry row that is already there; the real answer is
    that their surface is invoiced by AWS or Google and the rate has to come
    from that bill.
    """


def _load(name):
    path = os.path.join(REGISTRY_DIR, name)
    if not os.path.exists(path):
        raise RegistryError(f"registry file missing: {path}")

    def reject_non_finite(value):
        raise RegistryError(
            f"{name} contains non-finite JSON literal {value}. Registry "
            f"numbers must be finite, because they feed dollar figures and "
            f"cache limits.")

    with open(path) as f:
        return json.load(f, parse_constant=reject_non_finite)


_PROVIDERS = None
_PRICING = None


def providers():
    global _PROVIDERS
    if _PROVIDERS is None:
        _PROVIDERS = _load("providers.json")
    return _PROVIDERS


def pricing():
    global _PRICING
    if _PRICING is None:
        _PRICING = _load("pricing.json")
    return _PRICING


def target_ids(include_contested=False):
    return [t["id"] for t in providers()["targets"]
            if include_contested or not t["provenance"].get("contested")]


def target(target_id, allow_contested=False):
    """Return one target row.

    Refuses contested rows by default. A disputed capability published as fact
    is worse than an absent one: it discredits every row beside it.
    """
    for t in providers()["targets"]:
        if t["id"] != target_id:
            continue
        prov = t["provenance"]
        if prov.get("contested") and not allow_contested:
            raise ContestedRow(
                f"{target_id} is flagged contested and must not be treated as "
                f"fact.\n  reason: {prov.get('contested_reason', 'unspecified')}\n"
                f"  pass allow_contested=True only to inspect it, never to "
                f"publish from it."
            )
        return t
    raise RegistryError(f"unknown target: {target_id!r}. "
                        f"known: {', '.join(target_ids())}")


def capability(target_id, name, allow_contested=False):
    t = target(target_id, allow_contested)
    caps = t.get("capabilities", {})
    if name not in caps:
        raise RegistryError(f"{target_id} records no capability {name!r}")
    return caps[name]


def multipliers(target_id):
    t = target(target_id)
    m = t.get("multipliers")
    if not m:
        raise RegistryError(f"{target_id} records no multipliers")
    return m


def min_cacheable_tokens(target_id, model, allow_contested=False):
    """Minimum prefix that will actually cache, for this target and model.

    Deliberately has no default. Guessing a minimum is how the silent failure
    this function exists to prevent gets reintroduced: below the threshold the
    provider processes the request uncached and returns no error at all.

    `allow_contested` matches `capability` and means the same thing: inspect,
    never publish. Without it a caller could not tell a contested row that
    records a minimum from one that records none, because `ContestedRow` was
    raised before the row was ever read -- so the live monitor reported
    `openai/bedrock`'s missing minimum as merely "disputed", and an operator
    who settled the contest would have found the same checks still off for a
    second reason nobody had mentioned. Defaults to False, so every existing
    call keeps refusing contested rows exactly as before.
    """
    t = target(target_id, allow_contested)
    # Followed to the end, with a cycle guard. One level was assumed, so a chain
    # silently stopped short and a row inheriting from itself recursed until the
    # error a caller saw had nothing to do with the registry.
    src, seen = t, {target_id}
    while "inherits_minimums_from" in src:
        nxt = src["inherits_minimums_from"]
        if nxt in seen:
            raise RegistryError(
                f"inherits_minimums_from forms a cycle: "
                f"{' -> '.join(list(seen) + [nxt])}. A minimum has to come from "
                f"somewhere; a loop means no row actually records one.")
        seen.add(nxt)
        # The inherited row is inspected on the same terms as the one that
        # pointed at it. A chain that ends on a contested row must not become
        # fact by being reached indirectly.
        src = target(nxt, allow_contested)
    mins = src.get("min_cacheable_tokens", {})
    if model in mins:
        return _finite_number(mins[model],
                              f"the cache minimum for {model!r} on {target_id}")
    if "_default" in mins:
        return _finite_number(mins["_default"],
                              f"the default cache minimum on {target_id}")
    raise RegistryError(
        f"no minimum recorded for {model!r} on {target_id}. The registry does "
        f"not guess: minimums are non-monotonic across model generations "
        f"(512 on Opus 5, 4096 on Opus 4.6 and Haiku 4.5), so an inferred "
        f"value would be wrong in exactly the cases that matter."
    )


def supported_ttls(target_id: str, model: str | None = None,
                   allow_contested=False) -> list:
    """Lifetimes this surface accepts, narrowed to the model when it matters.

    `supported_ttls` is the surface's maximum. On Bedrock the 1h lifetime went
    GA for three models only, which the row's provenance note recorded in prose
    while the capability stayed surface-wide -- so the linter blessed a 1h
    marker on a model that would reject it. A per-model map is consulted when
    the row carries one.

    `allow_contested` is the same inspect-never-publish affordance `capability`
    carries, threaded through so a caller can find out whether a contested row
    records lifetimes at all. Defaults to False; existing calls are unchanged.
    """
    surface = capability(target_id, "supported_ttls", allow_contested) or []
    if model is None:
        return list(surface)
    # An optional refinement, read as one. This used to call `capability` and
    # swallow the `RegistryError` it raises for an absent key, which is the
    # wrong tool twice over: `capability`'s contract is that a missing key is
    # an error, and this caller's contract -- stated in the docstring above --
    # is that a missing map means no per-model narrowing is recorded. A
    # swallowed exception standing in for "this field is optional" also made
    # the map look like a dependency to anything watching which capabilities
    # get read, when its absence disables nothing and changes no answer.
    #
    # `target` rather than a bare dict walk, so a contested row is still
    # refused on exactly the same terms as every other read here.
    by_model = target(target_id, allow_contested).get(
        "capabilities", {}).get("supported_ttls_by_model")
    if not isinstance(by_model, dict):
        return list(surface)
    bare, _date = normalize_model(model, target_id)
    allowed = by_model.get(bare, by_model.get(model, by_model.get("_default")))
    if not isinstance(allowed, list):
        return list(surface)
    # Never wider than the surface itself.
    return [t for t in surface if t in allowed]


_DATE_SUFFIX = re.compile(r"^(?P<base>.+?)-(?P<date>\d{8})$")

# Exactly yyyy-mm-dd. Not a sorting heuristic, and not whatever the running
# interpreter's `fromisoformat` happens to accept this release.
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def normalize_model(model: str, target_id: str | None = None) -> tuple[str, str | None]:
    """Strip a trailing date snapshot from a model id.

    Real traces carry ids like `claude-haiku-4-5-20251001` alongside bare ones
    like `claude-haiku-4-5`, sometimes in the same file. The registry is keyed
    on the bare id, so an unnormalised trace prices as an unknown model and the
    request drops out of the total — silently understating spend.

    Deliberately not folded into `base_rate`. A lookup that quietly rewrites its
    own argument cannot tell a known model with a date suffix from a genuinely
    unknown one, and the second case has to keep failing. Callers normalise on
    purpose and report that they did.

    Returns (bare id, stripped date or None).
    """
    model = model or ""
    # Strip the surface's own id prefix. Bedrock records
    # `model_id_prefix: "anthropic."`, and a trace carrying `anthropic.claude-…`
    # priced fine under an invoice rate while min_cacheable_tokens rejected it
    # -- so the minimum guard was skipped on exactly the reports that publish
    # dollar figures.
    prefix = None
    if target_id:
        try:
            prefix = target(target_id).get("model_id_prefix")
        except RegistryError:
            prefix = None

    def _strip_surface(value):
        if prefix and value.startswith(prefix):
            stripped = value[len(prefix):]
            if stripped in pricing()["models"] or _DATE_SUFFIX.match(stripped):
                return stripped
        return value

    model = _strip_surface(model)

    # Then a gateway routing prefix. LiteLLM and friends address models as
    # `anthropic/claude-opus-5`, which is a routing decision rather than a
    # provider model id, so no target's `model_id_prefix` covers it. Left
    # unstripped it reached min_cacheable_tokens as an unknown model, and the
    # live plugin -- whose ids come from a gateway by definition -- then had no
    # minimum to check a marker against.
    #
    # Only accepted when what follows the slash is a model the registry already
    # knows, or a known model wearing a date. This recognises a known model
    # behind a prefix; it never invents one.
    #
    # `_DATE_SUFFIX.match(candidate)` alone used to be enough, which stripped
    # unknown ids that merely looked date-stamped:
    # `anthropic/not-a-real-model-20250101` came back as
    # `not-a-real-model-20250101`, so the RegistryError downstream quoted an id
    # the caller never sent. The comment above promises this never invents a
    # model, and half-rewriting an unknown one is a smaller version of the same
    # broken promise.
    # The two prefixes compose, and each pass alone could not see past the
    # other. LiteLLM addresses Bedrock as `bedrock/anthropic.claude-haiku-4-5`:
    # the surface pass could not match because the id starts with `bedrock/`,
    # and this pass rejected `anthropic.claude-haiku-4-5` because that is not a
    # registry model. Each half worked in isolation and the combination -- the
    # shape LiteLLM actually emits for the surface whose prefix this is -- fell
    # through unnormalised, so Bedrock traffic lost its minimum check, its
    # bake-off modelling and its live marker placement.
    #
    # So the candidate is tested with the surface prefix removed as well. Still
    # never invents a model: every branch requires what remains to be an id the
    # registry already knows, or one wearing a date.
    if "/" in model and model not in pricing()["models"]:
        candidate = model.rsplit("/", 1)[-1]
        for form in (candidate, _strip_surface(candidate)):
            dated = _DATE_SUFFIX.match(form)
            bare = dated.group("base") if dated else form
            if form in pricing()["models"] or bare in pricing()["models"]:
                model = form
                break

    m = _DATE_SUFFIX.match(model)
    if not m:
        return model, None
    base = m.group("base")
    return (base, m.group("date")) if base in pricing()["models"] else (model, None)


def _as_date(value, what="date"):
    """Parse strictly, or refuse.

    Rates were selected by comparing raw strings, so anything sorting after an
    effective date silently selected the later tier. `2026-8-1` is August, and
    it sorts after `2026-09-01`: claude-sonnet-5 priced at the post-September
    $3.00 instead of $2.00, a 50% overstatement on a report whose whole point is
    date-effective pricing. `not-a-date` did the same, without failing.
    """
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    # Only the explicit `yyyy-mm-dd` string, and only a string.
    #
    # `date.fromisoformat(str(value))` made this answer differ by interpreter.
    # Python 3.11 taught `fromisoformat` the compact ISO basic form, so the
    # integer 20260801 raised on 3.9 and parsed as 2026-08-01 on 3.13: the same
    # trace priced two different ways depending on which Python ran it, in the
    # one function whose entire job is date-effective pricing. CI caught it on
    # 3.13 the first time this was pushed to a repo that runs both.
    #
    # `str(value)` was the other half of it -- it happily stringifies anything,
    # so a type that has no business being a date got a parse attempt instead of
    # a refusal.
    if isinstance(value, str) and _ISO_DATE.match(value):
        try:
            return date.fromisoformat(value)
        except ValueError:
            pass
    raise RegistryError(
            f"{what} must be an ISO yyyy-mm-dd date, got {value!r}. Rates are "
            f"selected by comparing dates, and a string that merely sorts is "
            f"not a date -- '2026-8-1' sorts after '2026-09-01' and would have "
            f"picked the wrong rate tier without failing.") from None


def _finite_number(value, what):
    """A registry number, or a refusal naming what was wrong with it.

    `_load` refuses the JSON literals `NaN`, `Infinity` and `-Infinity`, which
    closes the route somebody hand-editing a row would take. It does not close
    the arithmetic one: `1e999` is **valid JSON**, `parse_constant` never sees
    it, and it overflows to `inf` on the way in. Measured after the literal
    route was closed -- a `1e999` rate produced `base_rate -> inf`, then
    `price().usd -> nan`, rendered as `$nan`.

    So the rule "registry numbers are finite" is enforced where the numbers are
    READ, not only where the file is parsed. The door and the rooms, because a
    door only stops what walks through it.

    Deliberately not `cost.is_multiplier`, which additionally requires `> 0`:
    that is right for a multiplier and wrong here, since `0` is a legitimate
    rate for a free tier and a legitimate cache minimum. Same three exclusions
    -- bool, non-finite, negative -- and the positivity rule stays where it
    belongs. `bool` is excluded for the reason `is_multiplier` records: it
    subclasses `int` and slips past every numeric check.
    """
    import math
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RegistryError(
            f"{what} is {value!r}, which is not a number. Registry numbers are "
            f"multiplied into published dollar figures, and a non-number here "
            f"becomes a wrong figure rather than an error.")
    if not math.isfinite(value):
        raise RegistryError(
            f"{what} is {value!r}. Registry numbers must be finite: a "
            f"non-finite one propagates through pricing as `nan` and renders "
            f"as `$nan` in "
            f"a client-facing report. Note `1e999` is valid JSON and overflows "
            f"to infinity, so this is not caught by refusing the NaN literal.")
    if value < 0:
        raise RegistryError(
            f"{what} is {value!r}. A negative registry number produces a "
            f"negative bill, which is never the honest reading of a trace.")
    return value


def _base_rate_unscoped(model, on_date):
    """Base input USD/Mtok effective on `on_date` (ISO yyyy-mm-dd or date)."""
    when = _as_date(on_date, "on_date")
    models = pricing()["models"]
    if model not in models:
        raise RegistryError(f"no pricing recorded for {model!r}")
    applicable = [r for start, r in sorted(models[model]["rates"])
                  if _as_date(start, "rate effective date") <= when]
    if not applicable:
        raise RegistryError(f"no rate for {model!r} effective on {when}")
    return _finite_number(applicable[-1], f"the rate for {model!r} on {when}")


# The surface an ingest reports when a row states no provider at all. Registered
# in `unpriced_surfaces`, so default-deny refuses to price it. Named here rather
# than in the adapter because the registry owns what a surface is, and because
# more than one adapter will eventually need it.
UNATTRIBUTED = "unknown/unattributed"


def rate_scope():
    """Which surfaces the base rate table is valid for."""
    return pricing().get("rate_scope") or {}


def rates_apply_to(target_id):
    """True if `target_id` is billed at the rates in the pricing table.

    Default-deny. The table holds Anthropic first-party list prices, and a
    surface earns them by being named, not by being absent from an exclusion
    list -- otherwise every surface added later inherits Anthropic pricing by
    silence, which is how this got wrong in the first place.
    """
    return target_id in set(rate_scope().get("applies_to") or ())


def require_priceable(target_id):
    """Raise unless the base rate table may be used for `target_id`.

    Refusing is the whole point. A partner-operated surface priced at
    first-party rates does not fail loudly -- it publishes a dollar figure that
    looks right and does not match the invoice the client is holding.
    """
    if rates_apply_to(target_id):
        return
    scope = rate_scope()
    detail = (scope.get("unpriced_surfaces") or {}).get(target_id) or {}
    official = detail.get("official_pricing")
    where = f" Its published rates are at {official}." if official else ""
    raise UnpriceableSurface(
        f"{target_id} is not covered by the recorded rate table, which holds "
        f"Anthropic first-party list prices ({', '.join(scope.get('applies_to') or ['none'])})."
        f"{where} Pricing this surface at first-party rates would produce a "
        f"figure that does not match the bill the customer actually receives, "
        f"so it is refused rather than estimated. Supply the rate from their "
        f"invoice with --effective-rate to analyse this trace.")


def upcoming_rate_change(model, on_date):
    """The next scheduled rate change after `on_date`, or None."""
    # Through `_as_date`, like every other date selector here. This was the one
    # that compared raw strings, so "2026-8-1" sorted above "2026-09-01" at the
    # month digit and hid a rate change a month away, "not-a-date" silently
    # returned None, and an int raised a bare TypeError that callers catching
    # RegistryError do not catch.
    try:
        rows = pricing()["models"][model]["rates"]
    except KeyError as e:
        raise RegistryError(f"no recorded rates for {model}") from e
    when = _as_date(on_date)
    for start, rate in sorted(rows):
        if _as_date(start) > when:
            return {"effective": start, "rate": rate}
    return None


def staleness_report(as_of=None):
    """Rows whose provenance is older than the staleness window.

    A hand-verified capability table decays like any other fact. This project
    has already been caught by that twice: a landscape check missed two
    competitors published before the check date, and a registry row recorded
    the opposite of what the vendor docs said.
    """
    as_of = as_of or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    as_of_d = datetime.strptime(as_of, "%Y-%m-%d").date()
    rows = []
    # `pricing/rate-scope` ages separately from the rates themselves. It records
    # which surfaces those rates are *valid for*, and that is a commercial
    # arrangement between Anthropic and two cloud providers, not a number on a
    # page -- it can change without any rate changing. Left out of this report it
    # would be the one dated claim in the registry that never came up for review,
    # while silently deciding whether a client's traffic gets priced at all.
    scope_rows = ([{"id": "pricing/rate-scope", "provenance": rate_scope()["provenance"]}]
                  if rate_scope().get("provenance") else [])
    for t in (providers()["targets"]
              + [{"id": "pricing", "provenance": pricing()["provenance"]}]
              + scope_rows):
        p = t["provenance"]
        checked = p.get("checked_on")
        if not checked:
            rows.append({"id": t["id"], "age_days": None, "stale": True,
                         "reason": "no checked_on date"})
            continue
        age = (as_of_d - datetime.strptime(checked, "%Y-%m-%d").date()).days
        rows.append({
            "id": t["id"], "checked_on": checked, "age_days": age,
            "stale": age > STALE_AFTER_DAYS,
            "confidence": p.get("confidence"),
            "contested": bool(p.get("contested")),
            "live_canary": p.get("verified_by", {}).get("live_canary", False),
        })
    return rows


def base_rate(model: str, on_date, target_id: str) -> float:
    """The dated first-party input rate, for a surface those rates apply to.

    Scoped on purpose. This took only model and date, so it handed back
    Anthropic list prices for any model regardless of who invoices it -- the
    surface was erased before default-deny could see it. `cost.price` happened
    to call `require_priceable` first; nothing made every other caller do the
    same, and the analyzer's own `rate_for` closure did not.

    The default is anthropic/direct because that is the surface these rates are
    recorded for, and `require_priceable` then confirms it rather than assuming
    it -- but a caller that knows its surface must say so. `analyzer.rate_for`
    refuses to be called without one for exactly that reason: three adapters
    were fixed today for inferring anthropic/direct from a missing surface, and
    a silent default here would put the same bug one layer down.
    """
    require_priceable(target_id)
    return _base_rate_unscoped(model, on_date)
