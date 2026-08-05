"""Three stateless checks that need no traffic, no keys and no logs.

These run against a request you are about to send, or a fixture in CI. They
catch failures the provider does not report: below the minimum, the request is
processed uncached and returns no error at all.

Every check is tri-state. A linter that fails confidently on uncertain input
gets switched off within a week, and then catches nothing. ABSTAIN is a
first-class result: it says "I cannot tell from here, and here is what would
settle it".

Stateless means exactly that. Nothing here inspects conversation history,
usage fields or anything across requests. Drift detection and hit-rate analysis
need traces and belong in the analyzer, not in a linter that claims to need
nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from . import registry
# One constant, imported rather than copied. The live plugin already refuses to
# place a marker unless the estimate clears the minimum by this factor
# (`CachePlugin.minimum_margin` is `ESTIMATOR_WORST_OVERESTIMATE - 1`), and a
# linter that blesses a prefix the plugin would refuse is the two-standards
# failure this package exists to find in other people's tools.
from .segment import ESTIMATOR_WORST_OVERESTIMATE


class Status(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    ABSTAIN = "ABSTAIN"

    def __str__(self):
        return self.value


@dataclass(frozen=True)
class Result:
    check: str
    status: Status
    summary: str
    detail: str = ""
    resolve: str = ""

    @property
    def ok(self) -> bool:
        return self.status is not Status.FAIL

    def __str__(self):
        s = f"[{self.status}] {self.check}: {self.summary}"
        if self.detail:
            s += f"\n    {self.detail}"
        if self.resolve:
            s += f"\n    -> {self.resolve}"
        return s


# The *underestimate* side of a byte-ratio token estimate: how far below the
# truth it can land. Not measured. It is kept at 10% because being wrong in this
# direction is cheap and visible -- the check refuses a prefix that would in fact
# have cached, and somebody sees a marker they were told to skip.
#
# It used to be the whole story, applied symmetrically, and that put the
# unmeasured number on the side where being wrong costs money. See
# `ESTIMATOR_WORST_OVERESTIMATE` below.
ESTIMATE_BAND = 0.10

# The overestimate side is measured, and it is 2.81x, not 10%.
#
# `segment.ESTIMATOR_WORST_OVERESTIMATE` is the worst *cumulative* overestimate
# of a byte-ratio estimate against the provider's own tokenizer: median 1.002,
# p90 1.071, max 2.812, on prefixes rather than single segments because a prefix
# is what a marker sits on. So an estimate of N tokens is consistent with a true
# count as low as N / 2.81.
#
# Stated with its limits, because this is the number the check now leans on:
# 2.81 rests on 26 prefixes over six bodies. It is a floor on the estimator's
# error, not a proof of its bound, and it is used here only to decide when this
# check may not answer at all -- never to produce a figure.
#
# Reproduced before it was changed: `check_minimum(600, 'claude-opus-5')`
# returned PASS, "600 tokens clears the 512 minimum", while at the measured
# worst those 600 estimated tokens are 213 real ones. The +-10% band spans
# 540-660, never straddles 512, and so did not even abstain.


def _no_surface_named(check: str, target_id) -> Result | None:
    """ABSTAIN when the caller named no provider surface.

    `target_id` selects the rate table *and* the capability limits, so a check
    that answers without one answers for whichever surface a parameter default
    happened to name. Measured: `check_minimum(768, 'claude-opus-5')` returned
    PASS against anthropic/direct's 512 minimum and FAIL against openai/direct's
    1,024 -- the same request, two verdicts, and the caller had chosen neither.

    Default-deny is the package's stated rule: a surface earns first-party rates
    and first-party limits by being *named*. `UNATTRIBUTED` is the absence of a
    surface rather than a surface, so it must not satisfy a requirement to name
    one, and it is checked by value rather than truthiness for exactly the reason
    `litellm_handler` learned to: that string is truthy.
    """
    if target_id and target_id != registry.UNATTRIBUTED:
        return None
    return Result(
        check, Status.ABSTAIN,
        "no provider surface named, so there is nothing to check against",
        "Minimums, breakpoint budgets and supported lifetimes are all "
        "per-surface: the 512-token minimum on anthropic/direct is 1,024 on "
        "openai/direct, so answering without a surface answers for whichever "
        "one a default happened to pick.",
        "pass target_id, for example 'anthropic/direct', 'openai/direct' or "
        "'amazon-bedrock/converse'")


def _contested_result(check: str, target_id: str, err: Exception,
                      inspect=None) -> Result:
    detail = str(err)
    try:
        if inspect is not None:
            inspect()
    except registry.RegistryError as e:
        detail += f" After inspecting the contested row: {e}"
    return Result(
        check, Status.ABSTAIN,
        f"{target_id} is flagged contested, so this check cannot treat its "
        f"registry row as fact",
        detail,
        "settle the contested row against a dated source before relying on "
        "this check; if the inspected row also lacks the needed value, record "
        "that value separately")


def check_minimum(prefix_tokens: int, model: str,
                  target_id: str = registry.UNATTRIBUTED,
                  tokens_are_estimated: bool = True) -> Result:
    """Is the cacheable prefix long enough to cache at all?

    Minimums are non-monotonic across generations: 512 on Opus 5, 1024 on
    Opus 4.8, 2048 on Opus 4.7, 4096 on Opus 4.6 and Haiku 4.5. Newer is not
    always lower, so this cannot be reasoned about and has to be looked up.

    They are also non-monotonic *across surfaces*, which is why `target_id` no
    longer defaults to a named one.
    """
    name = "minimum-cacheable-tokens"
    unnamed = _no_surface_named(name, target_id)
    if unnamed is not None:
        return unnamed
    try:
        minimum = registry.min_cacheable_tokens(target_id, model)
    except registry.ContestedRow as e:
        return _contested_result(
            name, target_id, e,
            lambda: registry.min_cacheable_tokens(
                target_id, model, allow_contested=True))
    except registry.RegistryError as e:
        return Result(name, Status.ABSTAIN,
                      f"no minimum recorded for {model} on {target_id}",
                      str(e),
                      "add the model to the registry with a dated source before relying on this")

    if tokens_are_estimated:
        # Asymmetric, because the two directions of the estimator's error cost
        # different things. Below the true count the check merely refuses a
        # prefix that would have cached, and somebody sees the marker it told
        # them to skip. Above it, a marker is placed on a prefix the provider
        # silently declines to cache: no error, no cache write, and the write
        # premium billed as usual.
        lo = prefix_tokens / ESTIMATOR_WORST_OVERESTIMATE
        hi = prefix_tokens * (1 + ESTIMATE_BAND)
        if lo < minimum <= hi:
            return Result(
                name, Status.ABSTAIN,
                f"estimated {prefix_tokens:,} tokens cannot be shown to clear "
                f"the {minimum:,} minimum for {model}",
                f"the count is a bytes-per-token estimate, measured as much as "
                f"{ESTIMATOR_WORST_OVERESTIMATE:.2f}x above the real count on a "
                f"prefix of dense content, so {prefix_tokens:,} estimated tokens "
                f"span {int(lo):,}-{int(hi):,} real ones and that straddles the "
                f"threshold. (That factor comes from 26 prefixes over six "
                f"bodies: a floor on the error, not a proof of its bound.)",
                "count exactly with the provider's token counter before trusting either answer")

    if prefix_tokens < minimum:
        return Result(
            name, Status.FAIL,
            f"{prefix_tokens:,} tokens is below the {minimum:,} minimum for "
            f"{model} on {target_id}",
            "The request will be processed without caching and no error is returned. "
            "cache_creation_input_tokens comes back as 0 and nothing else signals it.",
            f"lengthen the cached prefix past {minimum:,} tokens, or stop paying for a marker that does nothing")

    return Result(name, Status.PASS,
                  f"{prefix_tokens:,} tokens clears the {minimum:,} minimum for "
                  f"{model} on {target_id}")


def check_breakpoint_budget(breakpoints: int,
                            target_id: str = registry.UNATTRIBUTED,
                            rolling_marker: bool = False) -> Result:
    """Are there more cache markers than the provider accepts?

    Exceeding the budget is a hard API error, not a silent degradation, and it
    surfaces in production on long tool-calling turns rather than in testing.

    The budget is a per-surface fact -- four on anthropic/direct, unpublished on
    openai/direct -- so this abstains rather than defaulting to a surface.
    """
    name = "breakpoint-budget"
    unnamed = _no_surface_named(name, target_id)
    if unnamed is not None:
        return unnamed
    try:
        maximum = registry.capability(target_id, "max_breakpoints")
    except registry.ContestedRow as e:
        return _contested_result(
            name, target_id, e,
            lambda: registry.capability(
                target_id, "max_breakpoints", allow_contested=True))
    except registry.RegistryError as e:
        return Result(name, Status.ABSTAIN, f"no breakpoint budget recorded for {target_id}", str(e))

    # `null` and `0` are different facts and were collapsed into one sentence.
    # openai/direct records `explicit_breakpoints: true` with `max_breakpoints:
    # null` -- it has developer-placed breakpoints and no published budget -- and
    # this reported that it "does not expose developer-placed breakpoints", a
    # false capability claim published into a client report.
    if maximum == 0 or not registry.capability(target_id, "explicit_breakpoints"):
        return Result(name, Status.ABSTAIN,
                      f"{target_id} does not expose developer-placed breakpoints",
                      "ordering is the only lever on this surface; a budget check does not apply")
    if maximum is None:
        return Result(name, Status.ABSTAIN,
                      f"no breakpoint budget is recorded for {target_id}",
                      f"{target_id} accepts developer-placed breakpoints, but the "
                      f"registry carries no maximum for it, so {breakpoints} "
                      f"cannot be checked against one. Add the limit with a dated "
                      f"source before relying on this check.")

    if breakpoints > maximum:
        return Result(
            name, Status.FAIL,
            f"{breakpoints} markers exceeds the limit of {maximum} on {target_id}",
            "This is a request-level error, not a silent miss. It typically appears once a "
            "multi-turn tool session has grown, so it reaches production rather than CI.",
            f"merge sections that change at similar rates until {maximum} or fewer markers remain")

    if rolling_marker and breakpoints > maximum - 2:
        return Result(
            name, Status.ABSTAIN,
            f"{breakpoints} of {maximum} markers used, and a rolling marker needs two",
            "A rolling conversation marker consumes two of the budget: one holding the previous "
            "end-of-history so it hits, one at the new end so only the delta is written.",
            f"leave two free for the rolling pair, so at most {maximum - 2} static markers")

    if breakpoints == 0:
        # Not a pass. On a surface that exposes developer-placed breakpoints,
        # markers are the only lever there is, so "0 of 4 markers used" rendered
        # as a green tick is the check reporting success on a prompt that caches
        # nothing at all -- the same silent-uncached shape `check_minimum` fails
        # rather than blesses. Under budget is true and beside the point.
        return Result(
            name, Status.ABSTAIN,
            f"no cache markers on {target_id}, so there is no budget to check",
            f"{target_id} accepts developer-placed breakpoints and this request "
            f"carries none, so nothing is cached and the {maximum}-marker budget "
            f"is not the constraint. The provider returns no error for this.",
            "place a marker at the deepest stable boundary, or state that this "
            "prompt is deliberately uncached")

    return Result(name, Status.PASS, f"{breakpoints} of {maximum} markers used on {target_id}")


def check_ttl_ordering(ttls_in_order: list[str], target_id: str,
                       model: str | None = None) -> Result:
    """Are mixed TTLs ordered the way this surface requires?

    Bedrock requires longer-lived checkpoints before shorter-lived ones in a
    single request. Anthropic direct has no such rule, so the same prompt is
    valid on one surface and wrong on the other. That is precisely the sort of
    per-surface constraint a compiler should enforce rather than a human
    remember.
    """
    name = "ttl-ordering"
    unnamed = _no_surface_named(name, target_id)
    if unnamed is not None:
        return unnamed
    try:
        supported = registry.supported_ttls(target_id, model)
    except registry.ContestedRow as e:
        return _contested_result(
            name, target_id, e,
            lambda: registry.supported_ttls(
                target_id, model, allow_contested=True))
    except registry.RegistryError as e:
        return Result(name, Status.ABSTAIN,
                      f"no TTL support recorded for {target_id}", str(e),
                      "add supported_ttls to the registry row before linting against it")

    # Validate support before ordering. Checking the order of values the surface
    # cannot accept is answering the wrong question: an earlier version returned
    # PASS for a bogus TTL on an unconstrained target, and silently dropped
    # unknown values out of the ordering comparison on a constrained one. Either
    # way a linter blessed a request the provider would reject.
    unsupported = [t for t in ttls_in_order if t not in supported]
    if unsupported and not supported:
        return Result(
            name, Status.FAIL,
            f"{target_id} accepts no TTL values, but {len(ttls_in_order)} were supplied",
            f"supplied: {', '.join(ttls_in_order)}. This surface offers no developer-set "
            f"cache lifetime, so these have no effect and signal a wrong assumption "
            f"about the target.",
            "remove the TTLs, or compile for a surface that supports them")
    if unsupported:
        return Result(
            name, Status.FAIL,
            f"{', '.join(sorted(set(unsupported)))} not supported on {target_id}",
            f"this surface accepts {', '.join(supported)}. An unsupported value is "
            f"either rejected outright or quietly gives a different lifetime than intended.",
            f"use one of {', '.join(supported)}")

    try:
        constraint = registry.capability(target_id, "ttl_ordering_constraint")
    except registry.RegistryError as e:
        return Result(name, Status.ABSTAIN,
                      f"no ordering rule recorded for {target_id}", str(e),
                      "record ttl_ordering_constraint (null if none) rather than omitting it")

    if constraint is None:
        return Result(name, Status.PASS,
                      f"{len(ttls_in_order)} TTL(s) supported on {target_id}; "
                      f"no ordering constraint applies")

    if constraint != "longest_first":
        return Result(name, Status.ABSTAIN,
                      f"unrecognised ordering constraint {constraint!r} on {target_id}",
                      "the registry records a rule this checker does not implement",
                      "implement the rule or remove it from the registry; do not ignore it")

    rank = {"1h": 0, "5m": 1}
    seen = [t for t in ttls_in_order if t in rank]
    for i in range(1, len(seen)):
        if rank[seen[i]] < rank[seen[i - 1]]:
            return Result(
                name, Status.FAIL,
                f"{seen[i]} checkpoint appears after {seen[i-1]} on {target_id}",
                f"order given: {' then '.join(ttls_in_order)}. This surface requires longer-lived "
                f"checkpoints first when TTLs are mixed in one request.",
                "reorder so every 1h checkpoint precedes every 5m checkpoint")

    return Result(name, Status.PASS,
                  f"TTL order {' then '.join(ttls_in_order) or '(none)'} satisfies longest-first on {target_id}")


def run_all(*, prefix_tokens: int, model: str, breakpoints: int,
            ttls_in_order: list[str] | None = None,
            target_id: str = registry.UNATTRIBUTED,
            tokens_are_estimated: bool = True,
            rolling_marker: bool = False) -> list[Result]:
    """All three checks against one surface.

    No default surface, for the reason each check states individually: every
    threshold below is per-surface, so a caller that named none would otherwise
    receive three confident answers about a provider it never chose. All three
    abstain instead, which is the tri-state contract doing its job.
    """
    return [
        check_minimum(prefix_tokens, model, target_id, tokens_are_estimated),
        check_breakpoint_budget(breakpoints, target_id, rolling_marker),
        check_ttl_ordering(ttls_in_order or [], target_id, model),
    ]


def worst(results: list[Result]) -> Status:
    if any(r.status is Status.FAIL for r in results):
        return Status.FAIL
    if any(r.status is Status.ABSTAIN for r in results):
        return Status.ABSTAIN
    return Status.PASS
