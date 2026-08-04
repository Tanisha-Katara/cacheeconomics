"""Diagnosis: why cache spend is being wasted, from real traces.

Each finding names a cause, quantifies it, and states what would confirm it.
A finding without a mechanism is an observation, and an observation dressed as
a diagnosis is how this kind of report loses a client's trust.

Nothing here rounds toward good news. `avoidable_usd` is what the evidence
supports and no more, and every figure carries its evidence class.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, fields, replace
from datetime import datetime, timezone

from . import cost, money, registry
from .allocate import reuse_chain_of
from .trace import QUALIFIES_SPEND as _QUALIFIES_SPEND
from .trace import note_blocks_spend as _note_blocks_spend
from .trace import (Request, Tier, TraceSet, _billed_input,  # noqa: E402
                    UNATTRIBUTED, marked_prefixes, marked_spans,
                    span_is_reusable_by,
                    write_tokens, write_visible_at)

MEASURED = money.MEASURED
MODELED = money.MODELED

# How well post-hoc segmentation has to match instrumented ground truth before
# a structural finding may carry money. Below this the boundaries are a guess,
# and a relocation recommendation built on guessed boundaries asks someone to
# reorder prompt authority on the strength of nobody having checked.
ALIGNMENT_FLOOR = 0.90

# TTL-2 only speaks where the five-minute-to-one-hour band is nearly empty,
# which is its whole premise: the long lifetime is being paid for and the
# cadence never needed it. Deliberately well below TTL-1's 0.4 trigger so the
# two rules cannot both recommend a change on one trace, which they did.
BAND_IS_RARE = 0.10

# Not every note is the same kind of thing. Most describe provenance -- which
# records were read, which ids were normalised -- and can sit out of the way.
# Some caveat a number that was published, and a caveat the reader never sees
# is the same as no caveat. This phrase is how the analyzer marks the second
# kind, and both renderers ask for the subset rather than pattern-matching
# prose the analyzer happens to emit today.
QUALIFIES_SPEND = _QUALIFIES_SPEND      # re-exported; defined in trace.py


def spend_caveats(analysis_or_notes) -> list:
    """The notes that qualify a published figure, in the order they were made.

    Reads `blocking_notes`, which the analyzer records when each note is raised.
    A bare list is still accepted -- several tests and the ingest adapters hold
    notes without an Analysis around them -- and falls back to the predicate.
    That fallback is the old behaviour and is why the structured field exists:
    deciding a note's kind by searching its prose means a rewording silently
    demotes a blocker to provenance.
    """
    blocking = getattr(analysis_or_notes, "blocking_notes", None)
    if blocking is not None:
        return list(blocking)
    return [n for n in (analysis_or_notes or []) if _note_blocks_spend(n)]

# Measured on 2026-07-28: a five-minute entry is gone somewhere between 300 and
# 420 seconds, and a one-hour entry survived 56 minutes.
_TTL_SECONDS = {"5m": 300, "1h": 3600}


@dataclass
class Finding:
    code: str
    title: str
    severity: str                  # high | medium | low
    evidence_class: str            # measured | modeled
    detail: str
    affected_requests: int
    avoidable_usd_month: money.Figure | None = None
    # The same claim over the window that was actually observed, never
    # extrapolated. Two fields because they are two claims with two different
    # burdens of proof: this one needs the reconciliation gate, the monthly one
    # needs that *and* a window long enough to scale from. With only the monthly
    # field, gating the projection threw away a measurement the trace fully
    # supported, so the projection floor was left off the per-finding figures
    # entirely and a one-hour trace published $180/mo as `reconciled`.
    #
    # Declared after `avoidable_usd_month` rather than beside the amount it
    # belongs with, because the positional order of this dataclass is part of
    # its interface and a caller passing the seventh argument means the monthly
    # figure. Inserting ahead of it would have silently rebound that argument.
    avoidable_usd_window: money.Figure | None = None
    confidence: str = "medium"
    quality_risk: str = "low"
    fix: str = ""
    # True when the finding is derived from prompt structure rather than usage
    # counters. Those claims are only as good as the segmentation behind them,
    # which on an inferred trace is a guess until somebody measures it.
    structural: bool = False
    # Why *this finding's own* sample cannot support a monthly projection, or ""
    # if it can. Set by `_avoidable` from the timestamps of the requests that
    # actually contributed to the amount, and applied by `analyze` after the
    # reconciliation gate has had its say.
    #
    # Per-finding rather than per-trace because the floor was evaluated once
    # from the whole window and request count, so a finding costed from two
    # requests one second apart cleared it on the strength of unrelated traffic.
    # It is a string rather than a bool so the reason carries the finding's own
    # numbers -- "2 requests", not the trace's ten.
    projection_why: str = ""
    # How many contributing requests that verdict counted.
    #
    # Recorded because the verdict is only as good as the set handed to it, and
    # "these are the requests that contributed" is exactly the kind of claim
    # this package keeps getting wrong: FAN-1 passed both sides of every
    # concurrent pair while charging one of them, so the sample read double and
    # cleared a floor that exists to stop a small sample driving a projection.
    # A count on the finding lets a test compare the sample against the requests
    # that actually move the figure instead of taking the rule's word for it.
    projection_sample: int = 0

    def describe(self) -> str:
        """The structural finding always renders; the dollar claim may not.

        Whether the figure appears is decided by the Figure itself, not by a
        flag this method has to be passed correctly. A caller that forgets
        cannot leak a number, because there is no longer a way to ask for one.
        """
        f = self.avoidable_usd_month
        # Cosmetic only. If this check were forgotten the figure would still
        # render as "[withheld: ...]" rather than a number, which is the whole
        # point: the safety property does not depend on remembering it.
        amount = (f"  ~{f}/mo" if f and f.released
                  else ("  [figure withheld]" if f else ""))
        # The recommended action, which this renderer omitted entirely while the
        # HTML one printed it under "Action". So a reader of the text report --
        # the one that gets pasted into an email -- got the diagnosis and not
        # the remedy, for every finding, and the two renderers disagreed about
        # what the user was told to do. The twin-path tests never caught it
        # because none of them compared the *actionable* half.
        action = f"\n    do this: {self.fix}" if self.fix else ""
        return (f"[{self.severity.upper()}] {self.code} {self.title}{amount}\n"
                f"    {self.detail}{action}\n"
                f"    evidence: {self.evidence_class} · confidence: {self.confidence} "
                f"· quality risk: {self.quality_risk}")

    def __str__(self):
        return self.describe()

    def released(self, ok: bool, because: str = "", *, as_: str = "") -> "Finding":
        """`as_` threads release provenance through, so a finding's dollar
        figure is marked the same way the totals are. Releasing the spend map
        as DRAFT and leaving every finding saying RECONCILED would put both
        states in one report -- the same fix-one-site-leave-the-rest shape this
        distinction exists to expose.

        Every Figure on the finding, discovered from the field list rather than
        named. Naming `avoidable_usd_month` here is what would have left
        `avoidable_usd_window` released on a failed reconciliation the day it
        was added -- a second field threaded through some of the paths that
        carry the first, which is this package's most-repeated defect.
        """
        figures = {f.name: getattr(self, f.name).release(ok, because, as_=as_)
                   for f in fields(self)
                   if isinstance(getattr(self, f.name), money.Figure)}
        return replace(self, **figures) if figures else self


@dataclass
class Analysis:
    ratios: dict
    coverage: dict
    tier: Tier
    findings: list[Finding] = field(default_factory=list)
    spend: dict = field(default_factory=dict)
    reconciliation: dict | None = None
    window_days: float | None = None
    notes: list[str] = field(default_factory=list)
    # The subset of `notes` that qualifies a published figure. Recorded when the
    # note is raised, not recovered from its wording afterwards.
    blocking_notes: list[str] = field(default_factory=list)
    # Whether segment sizes came from a tokenizer rather than a byte estimate.
    # Carried because the report makes a claim that depends on it: counting is a
    # separate step that sends prompt prefixes to a provider, and a report that
    # says "no prompt content left the environment" is false on the workflow
    # `run_diagnostic.py` recommends. `report._next_steps` already read
    # `_tokens_counted` off this object, which nothing ever set, so the branch
    # that offers counting as a next step could never fire either.
    tokens_counted: bool = False

    @property
    def total_avoidable_month(self) -> money.Figure:
        """Inherits the release state of its parts.

        A total is publishable only if every figure feeding it is, which is
        automatic here rather than a rule someone has to remember.
        """
        parts = [f.avoidable_usd_month for f in self.findings if f.avoidable_usd_month]
        total = sum(p.raw() for p in parts)
        ok = bool(parts) and all(p.released for p in parts)
        # The first *unreleased* part, not the first part. Taking parts[0] meant
        # a released finding followed by a withheld one produced a total whose
        # stated reason was whatever the released one carried -- which is
        # nothing, so the aggregate fell back to a generic "not released" and
        # lost the sentence telling the reader what to do about it.
        why = "" if ok else next((p.withheld_because for p in parts
                                  if not p.released and p.withheld_because),
                                 "no priceable findings" if not parts else "not released")
        # And its release *provenance*, on the same reasoning: a total built
        # from draft parts is a draft. Constructing this Figure fresh made it
        # default to RECONCILED, so a --allow-unreconciled run put a
        # draft-marked spend map and findings beside a reconciled-looking
        # headline. DRAFT wins if any part is a draft, because a total is only
        # as checked as its weakest input.
        as_ = (money.DRAFT if any(p.released_as == money.DRAFT for p in parts)
               else money.RECONCILED)
        # Projected if any part is, discovered rather than asserted. Every part
        # comes from `_monthly` today so this is always True in practice, but
        # reading it off the parts means a future non-extrapolated contributor
        # does not silently mark the total as a projection, and vice versa.
        return money.Figure(total, money.MODELED, released=ok,
                            withheld_because=why, released_as=as_ if ok else "",
                            projected=any(p.projected for p in parts))


def _when(r: Request) -> str | None:
    """The day a request was sent, for date-effective pricing."""
    return r.sent_at.strftime("%Y-%m-%d") if r.sent_at else None


def _declared_ttl(r: Request) -> str | None:
    """The lifetime this request asked for, when it is unambiguous.

    One request can carry breakpoints at different lifetimes. Against a single
    aggregate cache_creation_input_tokens that split is genuinely unknowable, so
    mixed markers return None rather than the first one -- pricing the whole
    write at whichever lifetime sorted first is a guess wearing a number. Only
    the provider's own per-lifetime breakdown settles it, and from_anthropic
    prefers that when present.
    """
    # A marked block with no lifetime of its own is the provider's five-minute
    # default -- `cache_control: {"type": "ephemeral"}` with no ttl. That only
    # matters when *another* marker names a lifetime explicitly, because then
    # the request wrote two and a single number cannot describe it: an explicit
    # 1h marker beside a default one reported one lifetime and priced a mixed
    # write as pure 1h.
    #
    # When no marker names a lifetime the request is uniform-unknown, and the
    # row field is a legitimate fallback rather than a contradiction. Treating
    # every silent marker as 5m made an export that simply does not record
    # per-marker lifetimes entirely unpriceable, which is a real and common
    # shape.
    distinct = r.marker_lifetimes
    marked = next(iter(distinct)) if len(distinct) == 1 else None
    row = r.ttl_requested
    if len(distinct) > 1:
        # Mixed markers, which the docstring above has always said are
        # unprovable -- and `marked or row` then fell back to the row anyway.
        # The row field cannot express a request that wrote two lifetimes, so
        # taking its word prices a possible 2x write at 1.25x: the same 38%
        # understatement, in the same flattering direction, that the rest of
        # this path exists to prevent.
        return None
    if marked and row and marked != row:
        # The marker is on the block that was actually sent; the row field is
        # exporter metadata about it. When they disagree one of them is stale,
        # and taking the row's word for it priced 1h writes at the 5m rate --
        # the exact 38% understatement, in the flattering direction, that the
        # rest of this path exists to prevent. Unprovable is the honest answer.
        return None
    return marked or row


def _expiry_lifetime_seconds(r: Request) -> int | None:
    """When every known cache entry from this request has definitely expired.

    One line, because the derivation is shared with the runtime: the two had
    already disagreed twice about which lifetime governs expiry.
    """
    return cost.expiry_seconds(r.usage, r.marker_lifetimes, _declared_ttl(r))


def _usages(reqs: list[Request]) -> tuple[list[cost.Usage], list[Request]]:
    """Price what can be proven; hand back what cannot, rather than guessing.

    `Usage.from_anthropic` requires an explicit TTL because Anthropic reports
    `cache_creation_input_tokens` as one number with no lifetime split, and a
    1h write costs 2x against a 5m write's 1.25x. An earlier version of this
    function supplied "5m" whenever the trace was silent and coerced any
    unrecognised spelling to "5m" as well, which defeated that guard from one
    layer above it and understated 1h writes by 38% — silently, and in the
    direction that flatters the tool.

    A request with no write tokens needs no TTL: the multiplier lands on zero
    either way. Only writes require proof.
    """
    priced, unprovable = [], []
    for r in reqs:
        try:
            priced.append(cost.Usage.from_anthropic(r.usage, ttl=_declared_ttl(r)))
        except ValueError:
            # Either no lifetime could be established for real write tokens, or
            # the response contradicted itself. Both are unpriceable, not
            # guessable.
            unprovable.append(r)
    return priced, unprovable


def _span_days(times) -> float | None:
    """How long a set of timestamps spans, floored at an hour.

    Split out from `_window_days` because the projection floor has to ask this
    of a *finding's own* sample as well as of the whole trace, and a rule holds
    timestamps rather than Requests by the time it knows which ones paid for its
    figure.
    """
    ts = sorted(t for t in times if t)
    if len(ts) < 2:
        return None
    return max((ts[-1] - ts[0]).total_seconds() / 86400.0, 1 / 24)


def _window_days(reqs: list[Request]) -> float | None:
    return _span_days(r.sent_at for r in reqs)


# A projection multiplies the observed window by `30/window_days`, so the
# shorter the window the more the answer rests on the sample being typical. An
# invoice cannot establish that: it proves the *measured* subtotal matches the
# bill, and says nothing about whether the hour it covers resembles the month.
#
# One full day, because agent traffic has a daily cycle -- working hours, batch
# jobs, nightly runs -- and a sample shorter than one cycle cannot represent it.
# Measured before this existed: two requests one second apart with a matching
# $1.00 invoice published $720.00/month as a reconciled figure, because
# `_window_days` floors at an hour and the invoice gate released everything.
PROJECTION_MIN_DAYS = 1.0

# And enough requests that no single one dominates. Below this a lone outlier
# moves the monthly total by more than the rest of the trace combined, which is
# a different way for a projection to be confidently wrong.
PROJECTION_MIN_REQUESTS = 10


# What this floor does and does not withhold. Said once per sample it can be
# asked about, because the previous version of this sentence described the code
# as it was *not* -- twice.
#
# First it said any per-finding monthly figure "has not been gated by it", which
# was true until the gate reached them. Then the replacement said the refusal
# withholds "the spend total and each finding's", which is true when the *trace*
# fails the floor and false when the trace clears it and one finding's own
# evidence does not: there the spend total publishes and the sentence beside the
# withheld finding claims otherwise. Overstating and understating what was
# protected are the same defect pointed opposite ways, and this one had already
# been fixed once in the other direction.
#
# The word "reconciled" is deliberately not in either, and a test asserts that:
# this reader supplied a correct invoice, and any mention of reconciliation in
# the refusal sends them to fix something that is not broken.
_SCOPE_TRACE = (
    "This withholds every monthly figure -- the spend total and each finding's. "
    "Nothing measured over the window itself is affected: measured spend and "
    "each finding's avoidable amount over the observed window still publish, "
    "because the invoice does establish those.")

_SCOPE_FINDING = (
    "This withholds this finding's monthly figure and nothing else -- other "
    "findings are judged on their own evidence, and the trace as a whole may "
    "well carry enough of it. The amount avoided over the observed window still "
    "publishes, because the invoice does establish that.")


def _projection_supported(window_days, n_requests, *,
                          sample: str = "trace") -> tuple[bool, str]:
    """May a per-month figure be published from this sample?

    `sample` names what was measured, because this is asked twice about two
    different populations: the whole trace, for the spend total, and a single
    finding's contributing requests, for that finding's figure. Reporting the
    trace's numbers in a refusal about a finding is how a reader concludes the
    tool is broken -- they can see ten requests over two days in the coverage
    line directly above a sentence saying there were two.
    """
    subject, scope = {
        "trace": ("this trace covers", _SCOPE_TRACE),
        "finding": ("the evidence behind this finding covers", _SCOPE_FINDING),
    }[sample]
    if not window_days or window_days < PROJECTION_MIN_DAYS:
        got = f"{(window_days or 0) * 24:.1f} hours"
        return False, (
            f"monthly figures need at least {PROJECTION_MIN_DAYS:.0f} day of "
            f"traffic and {subject} {got}. Agent workloads have a daily "
            f"cycle, so a shorter sample cannot be scaled to a month -- an "
            f"invoice proves the measured subtotal, not that the window is "
            f"typical. " + scope)
    if n_requests < PROJECTION_MIN_REQUESTS:
        # "rests on" rather than "covers": the count is of the requests that
        # contributed, which for a finding is not the same as the requests its
        # window spans.
        rests = ("this trace has" if sample == "trace"
                 else "this finding rests on")
        return False, (
            f"monthly figures need at least {PROJECTION_MIN_REQUESTS} requests "
            f"and {rests} {n_requests}. Below that a single request moves "
            f"the projected total more than the rest of the trace combined. "
            + scope)
    return True, ""


def _monthly(amount: float, window_days: float | None) -> money.Figure | None:
    """Extrapolate to a month, as a Figure that starts withheld.

    Withheld is the default state on purpose. A figure becomes publishable only
    when `analyze` releases it against the reconciliation gate, so a rule that
    forgets to think about publication produces something visibly withheld
    rather than a bare number that looks authoritative.
    """
    if not window_days or window_days <= 0:
        return None
    # `projected=True` is set here and nowhere else. This is the single site in
    # the package that multiplies a measured amount by a window ratio, so
    # tagging it here means every extrapolated figure carries the mark without
    # any caller having to know it should, and a test can find them all.
    return money.Figure(amount * (30.0 / window_days), money.MODELED,
                        released=False, withheld_because="not yet reconciled",
                        projected=True)


def _withhold_projection(fig, why: str):
    """Withhold an extrapolation the observed window cannot support.

    One function because it is applied in two places -- the spend total and
    every per-finding monthly figure -- and a rule enforced twice is a rule that
    has already started to diverge somewhere else in this file.

    It only ever downgrades a figure that is *released*. A figure already
    withheld carries a reason somebody can act on -- segment sizes that
    disagree with the bill, an invoice outside the gate -- and
    `release(False, because)` replaces that reason with this one. Both refusals
    are true and this is the weaker of the two: "this cannot be scaled to a
    month" tells nothing further to a reader who has just been told the
    underlying number does not tie to the bill.
    """
    return fig.release(False, why) if fig is not None and fig.released else fig


def _avoidable(amount: float | None, window_days: float | None,
               sample_times) -> dict:
    """Both halves of a finding's dollar claim, built at one site.

    Returned as kwargs so a rule cannot supply one and forget the other. Every
    rule that priced something used to write `avoidable_usd_month=_monthly(...)`
    by hand, and a per-rule pair of hand-written fields is how the two would
    come to disagree about the same arithmetic.

    `amount is None` means the rule declined to price this finding -- VOL-1 when
    a second blocker makes the move recover nothing, TTL-1 when the trace
    carries no segment identity. Both figures are absent then, which is what
    "not costed" means in the report.

    The window figure is deliberately not `projected`: it is the amount over the
    window that was observed, so there is no extrapolation in it for the
    projection floor to gate. It carries MODELED because it is still a
    counterfactual -- what the workload would not have spent under a different
    configuration -- and no counterfactual was run.

    `sample_times` is the timestamp of every request that contributed a term to
    `amount`, and it is required rather than optional. The floor used to be
    evaluated once, from the whole trace, and applied to every finding -- so a
    finding costed from two requests one second apart published a month because
    eight unrelated requests elsewhere in the file made the *global* window and
    count clear the floor. Measured on a ten-request, 2.3-day trace whose only
    two cache writers went out one second apart: EFF-1 published $6.43/mo and
    FAN-1 $14.79/mo, both `reconciled`, on a two-request sample. The request
    floor exists precisely so a small subset cannot drive a client-facing
    projection, and evaluated globally it did not do that job.

    Required, with no default, because a default is how the next rule silently
    gets the old behaviour back.

    The subset check subsumes the global one for findings and is not merely
    added to it: a finding's sample is drawn from the trace, so its window and
    count can only be smaller. Where the whole trace fails the floor, every
    finding fails it too, and with its own numbers in the reason rather than the
    trace's.
    """
    if amount is None:
        return {"avoidable_usd_window": None, "avoidable_usd_month": None,
                "projection_why": "", "projection_sample": 0}
    sample = list(sample_times)
    supported, why = _projection_supported(_span_days(sample), len(sample),
                                           sample="finding")
    return {
        "avoidable_usd_window": money.Figure(
            amount, money.MODELED, released=False,
            withheld_because="not yet reconciled"),
        "avoidable_usd_month": _monthly(amount, window_days),
        # Recorded rather than applied. Both figures are withheld at this point
        # and `Finding.released` releases whatever it finds, so a refusal
        # applied here would be undone by the reconciliation gate a moment
        # later. `analyze` re-applies this after release, which is the same
        # order the trace-wide floor already used for `monthly_input_usd`.
        "projection_why": "" if supported else why,
        "projection_sample": len(sample),
    }


# --- diagnosis rules -------------------------------------------------------

def _rate_free(fn):
    """Marks a rule that is safe to run over rows nothing can price.

    Those rules run over every analysable request; the rest run over the priced
    subset, because a dollar figure must not be derived from a row that spend
    excluded. Marking the *safe* ones rather than the pricing ones makes the
    unmarked default the narrow, older behaviour, so a rule added later cannot
    widen its own input by omission.

    "Safe" and not "never asks for a price", and the difference is TTL-1. Most
    marked rules -- REB-1, REB-0, SPL-1, MIN-1 -- genuinely never reach for a
    rate. TTL-1 is two claims in one function: a cadence observation that comes
    from timestamps, and a saving that comes from a rate. It asks, gets 0.0 for
    a surface nobody priced, and declines to monetize. Withholding the whole
    finding because half of it needs a rate meant a Bedrock reader was told
    nothing about a pattern their own counters show, while the same trace with
    `--effective-rate` produced it.

    So the obligations are: do not raise on a row the registry cannot answer
    for, and do not come back carrying money computed from a rate nobody has.
    `TestARateFreeRuleIsSafeOnATraceNothingCanPrice` checks both against an
    unpriceable trace rather than against a proxy -- the previous version spied
    on `rate_for` and had gone vacuous for the one rule it needed to judge,
    because its fixture made TTL-1 return before it ever asked.
    """
    fn.rate_free = True
    return fn


def _f_prefix_efficiency(reqs, ratios, window, rate_for) -> Finding | None:
    """Are the writes being read? This is where money actually leaks."""
    eff = ratios.get("prefix_efficiency")
    if eff is None:
        return None
    # No efficiency cutoff. There used to be `eff >= 0.5: return None` here, a
    # frequency veto in front of an economic test that was already correct --
    # `wasted <= 0` below fires only when caching genuinely costs more than not
    # caching, and it needs no help deciding that.
    #
    # 0.5 was also not the break-even for either lifetime. Caching pays for
    # itself once `W*w + R*r <= w + r`, which solves to an efficiency of
    # `(W-1)/(W-R)`: 21.7% for a 5m write at 1.25x and 52.6% for a 1h write at
    # 2.0x. So the constant sat above one and below the other, and the band it
    # got wrong was the 1h workload between 50% and 52.6% -- losing money and
    # silently dropped, because the guard ran before the arithmetic that would
    # have caught it.
    #
    # Deriving the number and keeping the guard would have been the smaller
    # change. Deleting it is the better one: the money test subsumes it, and two
    # thresholds for one question is how they drift apart.
    # The premium depends on the lifetime that was actually written. A 5m write
    # bills 1.25x and a 1h write bills 2.0x, so the wasted part of an unread
    # write is 0.25x or 1.0x of the base rate -- a factor of four apart. This
    # rule used to hard-code 0.25x against the raw creation total, which
    # underreported an hour-lifetime workload fourfold, even though the split is
    # already parsed a few lines earlier.
    # Net excess over sending everything uncached, not the gross write premium.
    #
    # A low efficiency ratio does not mean caching is losing money: the reads it
    # did earn are credited at 0.1x, and they can more than pay for the writes.
    # Charging every write premium and ignoring the reads told a customer their
    # cache was wasting money on a workload that was already saving it -- $900 a
    # month "avoidable" on a trace where caching saved $0.55 against uncached.
    # What is genuinely avoidable is the amount by which caching costs *more*
    # than not caching at all, and nothing else.
    # Every priced request, not just the writers. Cache payback lands on the
    # *later* read-only requests, so filtering to rows with cache_creation
    # excluded exactly the savings the comparison exists to weigh -- leaving
    # excess as a fixed fraction of write volume no matter how much was read
    # back. A write of 100k with a 40k payback and one with no payback at all
    # both reported the same $90.
    # The requests that actually moved `excess`, for the projection floor. A row
    # of plain uncached input prices identically either way, so it contributes a
    # term of exactly zero and is not part of the sample this figure would be
    # scaled from -- counting it would let filler traffic vouch for a projection
    # it contributed nothing to, which is the defect this list exists to close.
    contributed: list = []
    excess, unprovable = 0.0, 0
    for r in reqs:
        try:
            u = cost.Usage.from_anthropic(r.usage, ttl=_declared_ttl(r))
        except ValueError:
            unprovable += 1
            continue
        try:
            # The same rate the reconciliation used. `rate_for` is the closure
            # analyze() builds, so it already returns the invoice rate when one
            # was supplied and the date-effective list rate otherwise. Passing
            # None here meant a report that tied out against an invoice still
            # published EFF-1 at list price: spend reconciled at $0.25 while the
            # finding claimed $180, five times what the invoice supports.
            spend = cost.price(u, r.model, target_id=r.target_id, on_date=_when(r),
                               effective_rate=rate_for(r.model, _when(r), r.target_id))
        except registry.RegistryError:
            unprovable += 1
            continue
        term = spend.usd - spend.hypothetical_uncached_usd
        if term:
            contributed.append(r.sent_at)
        excess += term
    wasted = excess
    if wasted <= 0:
        # Caching is paying for itself on this workload even at a low ratio.
        # The efficiency observation still belongs elsewhere; a dollar claim
        # does not belong here.
        return None
    return Finding(
        code="EFF-1", title="Cache writes are largely not being read",
        severity="high", evidence_class=MEASURED,
        detail=(f"Prefix efficiency is {eff:.0%}: of every 100 tokens written to cache, "
                f"{eff*100:.0f} are later read. A write bills above the standard input "
                f"rate, so the unread remainder is a pure premium paid for nothing."
                + (f" {unprovable} request(s) had writes of unprovable lifetime and are "
                   f"excluded from the figure." if unprovable else "")),
        affected_requests=sum(1 for r in reqs if write_tokens(r.usage)),
        **_avoidable(wasted, window, contributed),
        confidence="high", quality_risk="low",
        fix="Find what invalidates the prefix between requests before changing any TTL. "
            "A longer lifetime on an unstable prefix buys nothing.")


def _f_volatile_prefix(reqs, ratios, window, rate_for) -> Finding | None:
    """A segment that should be stable but changes, sitting above stable content."""
    # Bucketed per reuse chain, not across the whole cache pool. Caches are
    # isolated by tenant, surface and model, but stability is judged along the
    # sequence that can reuse a prefix. Two sessions with different, internally
    # stable headers are two prefixes, not one prefix changing.
    per_chain = defaultdict(lambda: defaultdict(set))
    labels = defaultdict(set)
    # A review asked why absence is not treated as volatility here, and the
    # answer is that it is, wherever this rule's remedy applies. An optional
    # block *in front of* other content renumbers everything behind it, so the
    # position holds two different ids and the buckets below already catch it.
    #
    # The case that slips through is an optional block at the *end* of the
    # cached prefix. That does change the cached bytes and does force a rewrite
    # -- but VOL-1's whole recommendation is to move the volatile block behind
    # the stable span, and there is no stable span behind it. `tokens` for the
    # suffix comes out zero and the finding correctly recovers nothing.
    #
    # Detecting it here and reporting it under a fix that cannot help would be
    # worse than the gap. EFF-1, REB-1 and CAC-1 all fire on that trace, and
    # their remedies are the ones that apply. Tried the change, measured what it
    # produced, took it out.
    for r in reqs:
        if not r.segments:
            continue
        chain = reuse_chain_of(r)
        mk = [s.index for s in r.segments if s.cache_marked]
        top = max(mk) if mk else -1
        for s in r.segments:
            if s.index <= top:
                # Keyed on wire position alone. Including the label meant a
                # segment whose label changed at the same position landed in a
                # different bucket each time, so every bucket held one value and
                # the drift disappeared -- while the cached prefix bytes had
                # changed exactly as much as if the text had. Labels are display,
                # not identity.
                per_chain[s.index][chain].add(s.id)
                labels[s.index].add(s.label or s.role)

    # A position is volatile only where some real reuse chain sees it change.
    # Keep *which* chains were volatile, not just that one was, so a shared export
    # with one changing session does not charge or advise the stable sessions.
    by_pos, volatile_chains = {}, {}
    for key, chains in per_chain.items():
        bad = {chain: ids for chain, ids in chains.items() if len(ids) > 1}
        if bad:
            by_pos[key] = max(bad.values(), key=len)
            volatile_chains[key] = set(bad)

    unstable = [(pos, ids) for pos, ids in by_pos.items() if len(ids) > 1]
    if not unstable:
        return None
    # Sort first, then scope. Reading the pools from the unsorted list meant a
    # trace whose segments arrive out of index order could report volatility at
    # one position while pricing the pools of another -- attaching dollars to
    # tenants whose reported segment never moved.
    unstable.sort(key=lambda kv: kv[0])
    idx, ids = unstable[0]
    affected_chains = volatile_chains.get(idx, set())
    # Every other position that also breaks the prefix, on the chains this one
    # affects. Moving the lowest blocker only stabilises the span up to the next
    # one: anything past it is still invalidated on every request, so charging
    # the whole suffix to a single move claims dollars the recommendation cannot
    # recover. Measured on a prompt with two volatile blocks ahead of a 30k
    # marked prefix: identical $3,751 whether there was one blocker or two,
    # while moving segment 0 alone would have recovered nothing at all.
    later_blockers = sorted(
        pos for pos, _ in unstable
        if pos > idx and volatile_chains.get(pos, set()) & affected_chains)
    next_blocker = later_blockers[0] if later_blockers else None
    seen_labels = sorted(labels.get(idx, set()))
    label = (seen_labels[0] if len(seen_labels) == 1
             else " / ".join(seen_labels[:3]) + ("…" if len(seen_labels) > 3 else ""))

    # Only the chains that actually saw the segment change. Everything else is
    # paying nothing for this and should not be told to reorder its prompt.
    hit = [r for r in reqs if reuse_chain_of(r) in affected_chains]

    # Price the transition relocation would create, not the whole write.
    #
    # Charging every affected request its full write cost was wrong three ways
    # at once. The first request in a pool still has to write the stable prefix
    # after the move, so it recovers nothing. Later requests do not become free,
    # they become reads at 0.1x. And a 1h write bills 2.0x, not the 1.25x this
    # assumed. On a short window that overstated the figure by more than double
    # -- for a finding whose recommendation is to reorder prompt authority.
    #
    # So the timeline is walked per cache pool: a request that would have hit a
    # live entry had the volatile block been moved is the only kind that
    # recovers anything, and it recovers write-minus-read on its own suffix.
    by_chain: dict = defaultdict(list)
    unprovable_lifetime = 0
    for r in hit:
        mk = [s.index for s in r.segments if s.cache_marked]
        if not mk or not r.sent_at:
            continue
        # Bounded by the next blocker, not by the marker. The span past it does
        # not become cacheable until that one moves too.
        ceiling = max(mk) if next_blocker is None else min(max(mk), next_blocker - 1)
        tokens = sum(s.tokens for s in r.segments if idx < s.index <= ceiling)
        if not tokens:
            continue
        # The provider has to have actually written on this request. Charging
        # every later request inside the lifetime assumed the suffix was
        # re-processed, but a request reporting cache reads and no creation read
        # it instead -- so a header stable within a session and differing across
        # sessions was billed as prefix drift it never caused.
        if not write_tokens(r.usage):
            continue
        ttl = _declared_ttl(r)
        if ttl not in ("5m", "1h"):
            unprovable_lifetime += 1
            continue
        by_chain[reuse_chain_of(r)].append((r.sent_at, tokens, r, ttl))

    # Timestamps of the requests that actually contributed to `wasted`, for the
    # projection floor. Only the transitions that were still warm recover
    # anything, so only they are the sample this figure would be scaled from.
    contributed: list = []
    per_request, wasted = [], 0.0
    for entries in by_chain.values():
        entries.sort(key=lambda e: e[0])
        last = None
        for sent_at, tokens, r, ttl in entries:
            per_request.append(tokens)
            alive = last is not None and (sent_at - last).total_seconds() < _TTL_SECONDS[ttl]
            last = sent_at
            if not alive:
                continue            # cold before the move and cold after it
            contributed.append(sent_at)
            try:
                m = registry.multipliers(r.target_id)
            except registry.RegistryError:
                m = {"write_5m": 1.25, "write_1h": 2.0, "read": 0.10}
            delta = m[f"write_{ttl}"] - m.get("read", 0.10)
            wasted += tokens * (rate_for(r.model, _when(r), r.target_id) / 1e6) * delta
    # A later blocker can leave nothing recoverable by moving this one alone.
    # Reporting nothing at all would be worse than the overstatement it
    # replaces: the volatility is real, it is costing money, and the client
    # simply never hears about it. So the finding still fires -- without a
    # figure, and naming every position that has to move.
    stuck = next_blocker is not None and (not per_request or wasted <= 0)
    if not stuck:
        if not per_request:
            return None
        if not hit or wasted <= 0:
            return None
    behind = max(per_request) if per_request else 0
    typical = sorted(per_request)[len(per_request) // 2] if per_request else 0
    scope = ""
    if next_blocker is not None:
        others = ", ".join(str(p) for p in later_blockers)
        scope += (f" Position(s) {others} on the same reuse chain(s) are volatile too, "
                  f"so moving segment {idx} alone leaves the prefix invalidated there "
                  f"and recovers nothing past position {next_blocker}. All of them have "
                  f"to move together.")
    if len(affected_chains) < len({reuse_chain_of(r) for r in reqs}):
        scope = (f" This affects {len(hit):,} of {len(reqs):,} analysed requests, in "
                 f"{len(affected_chains)} reuse chain(s); the rest see this segment as stable "
                 f"and need no change.")
    if unprovable_lifetime:
        scope += (f" {unprovable_lifetime} request(s) had writes of unprovable lifetime and "
                  f"are excluded from the figure.")
    return Finding(
        code="VOL-1", structural=True, title=f"Volatile content at position {idx} spoils the prefix behind it",
        severity="high", evidence_class=MEASURED,
        detail=(f"Segment {idx} ({label}) took {len(ids)} distinct values across the window "
                f"while sitting inside the cached prefix. Caching matches from the start of "
                f"the prompt, so the tokens behind it are re-processed on every request "
                f"regardless of how stable they are: {typical:,} on a typical request, "
                f"{behind:,} at the longest.{scope}"),
        # Only the requests that actually contributed a suffix. Counting every
        # request in the pool included rows with no segments at all, which
        # cannot support this finding.
        affected_requests=len(per_request) or len(hit),
        # No figure when a second blocker means this move recovers nothing on
        # its own. The observation stands; the arithmetic for the full move set
        # is not something this rule models.
        **_avoidable(None if stuck else wasted, window, contributed),
        confidence="medium" if stuck else "high", quality_risk="medium",
        fix=("Move that segment after the last breakpoint. Where it carries operator "
             "authority, a mid-conversation system message preserves that while keeping "
             "it out of the cached prefix."
             + ("" if next_blocker is None else
                f" Move the other volatile position(s) with it -- until they all sit "
                f"behind the breakpoint the prefix keeps being rewritten, which is why "
                f"no figure is attached to moving this one by itself.")))


@_rate_free
def _f_below_minimum(reqs, ratios, window, rate_for) -> Finding | None:
    """Markers on prefixes too short to cache. Silent: no error is returned."""
    hits = []
    for r in reqs:
        if not r.segments or not r.breakpoints:
            continue
        try:
            minimum = registry.min_cacheable_tokens(r.target_id, r.model)
        except registry.RegistryError:
            continue
        created = write_tokens(r.usage)
        read = r.usage.get("cache_read_input_tokens", 0) or 0
        # Every marked breakpoint, not only the outermost. `cached_prefix_tokens`
        # is the prefix at the last marker, and the counters are request-wide, so
        # a 200-token marker followed by a 30k one looked fine: the outer marker
        # wrote, `created` was nonzero, and the inner marker sat below the
        # minimum doing nothing with nothing to show it. That is the silent
        # failure this rule exists to catch, hiding inside a request where
        # caching otherwise worked.
        prefixes = marked_prefixes(r.segments)
        if prefixes:
            if any(p < minimum for p in prefixes):
                hits.append((r, minimum))
        elif r.cached_prefix_tokens < minimum and created == 0 and read == 0:
            # No segment structure to walk. Fall back to the request-wide test,
            # which is all a usage-only trace can support.
            hits.append((r, minimum))
    if not hits:
        return None
    r0, minimum = hits[0]
    return Finding(
        code="MIN-1", structural=True, title="Cache markers on prefixes below the model minimum",
        severity="medium", evidence_class=MEASURED,
        detail=(f"{len(hits)} requests carry a cache marker on a prefix shorter than the "
                f"{minimum:,}-token minimum for {r0.model}, and returned zero cached tokens "
                f"in both directions. The provider processes these uncached and reports no "
                f"error, so nothing surfaces this in normal monitoring."),
        affected_requests=len(hits),
        avoidable_usd_month=None,
        confidence="high", quality_risk="low",
        fix=f"Either lengthen the cached prefix past {minimum:,} tokens or drop the marker. "
            f"Minimums are not monotonic across model generations, so this must be looked "
            f"up per model rather than reasoned about.")


def _prefix_key(r: Request) -> tuple | None:
    """Identity of the span this request asked to cache, or None if unknowable.

    Usage fields alone cannot answer it: two writes in the same session may or
    may not have written the same prefix, and only structure distinguishes them.
    """
    marked = [s.index for s in r.segments if s.cache_marked]
    if not marked:
        return None
    top = max(marked)
    return tuple(s.id for s in sorted(r.segments, key=lambda s: s.index)
                 if s.index <= top)


@_rate_free
def _f_ttl_vs_cadence(reqs, ratios, window, rate_for) -> Finding | None:
    """Is the chosen lifetime matched to how often requests actually arrive?"""
    per_agent = defaultdict(list)
    for r in reqs:
        if r.sent_at:
            per_agent[r.agent].append(r.sent_at)
    # Rank by money at stake, not by how band-shaped the cadence looks. An
    # earlier version ranked on the in-band fraction and so reported an agent
    # that writes nothing at all (100% in band, $0 recoverable) ahead of one
    # rewriting a 34k prefix on every request.
    candidates = []
    for agent, ts in per_agent.items():
        ts.sort()
        if len(ts) < 3:
            continue
        mine = [r for r in reqs if r.agent == agent]
        # Cadence is measured inside a cache isolation scope, never across the
        # agent's whole stream. A cache lives in `(tenant, target, model,
        # session)`; the gap that decides whether a five-minute entry survived
        # is the gap between two requests *that share one*, and pooling them
        # measured a quantity no cache ever sees.
        #
        # It failed in the direction that hides work. On a shared gateway one
        # agent serves many tenants, so interleaving compresses the pooled gaps
        # below the band and the finding was skipped. Measured on an identical
        # per-tenant workload rewriting a 30k prefix every ten minutes: TTL-1
        # fired at one tenant, and vanished at four and at twenty. The finding
        # disappeared precisely as the deployment got big enough to matter,
        # which is the worst possible failure curve for a cost tool.
        by_isolation = defaultdict(list)
        for r in mine:
            if r.sent_at:
                by_isolation[(r.tenant, r.target_id, r.model, r.session)].append(r.sent_at)
        gaps = []
        for stamps in by_isolation.values():
            stamps.sort()
            gaps += [(b - a).total_seconds() for a, b in zip(stamps, stamps[1:])]
        if not gaps:
            continue
        gaps.sort()
        median = gaps[len(gaps) // 2]
        in_band = sum(1 for g in gaps if 300 < g < 3600) / len(gaps)
        # No band-share veto. `if in_band <= 0.4: continue` used to sit here and
        # returned before anything was priced, which made a frequency counter
        # the arbiter of a money question.
        #
        # What it dropped: a stable million-token prefix rewritten across 35
        # ten-minute gaps, alongside 64 one-minute reads, is 35% in-band and
        # suppressed -- while the 35 rewrites it would convert to reads are
        # worth more than most findings this tool publishes. Band share scales
        # with how chatty a workload is between its slow gaps, and that has no
        # bearing on what the slow gaps cost.
        #
        # The timeline walk below already prices this properly, per cache scope,
        # charging the cold write premium and crediting only the rewrites that
        # actually fall in the band. `recoverable <= 0` is the gate. The share
        # stays in the detail text as context for the reader, which is the job
        # it can actually do.
        #
        # What does stay is the rule's own premise. Its title says the cadence
        # sits inside the one-hour window, so at least one gap has to. Removing
        # the 40% veto exposed that the veto had been doing this job by
        # accident: on a trace with no segment identity and every gap 30
        # seconds apart, TTL-1 fired and announced an in-band cadence at 0%
        # in-band, contradicting TTL-2 on the same trace. That is the same
        # defect TTL-2 had -- a headline asserting a number the data denies --
        # and a threshold is not the way to prevent it. Requiring the premise
        # is.
        if not in_band:
            continue
        # Only writes proven to be five-minute can be recovered by moving to
        # one hour. An earlier version multiplied every cache_creation token by
        # the 5m-to-1h delta without ever looking at the lifetime in use, so a
        # workload already writing 1h entries got told to "set a one-hour TTL"
        # and was shown a saving for a no-op. Real traces do this: one 29-day
        # capture wrote 70M of its 98M cache tokens at 1h already.
        # Switching to one hour does not turn the first write into a read. It
        # makes that cold write *more* expensive — 2.0x against 1.25x — and only
        # the rewrites that follow inside the window become reads. Charging
        # every write the full 5m-to-read delta ignored the premium entirely and
        # could more than double the figure on short sessions.
        #
        # So the timeline is walked per cache scope: first write in a scope pays
        # the premium, later writes still inside an hour become reads, and a
        # write after more than an hour is cold again and pays the premium again.
        recoverable, unprovable, already_1h, unkeyed = 0.0, 0, 0, 0
        by_scope = defaultdict(list)
        for r in mine:
            try:
                u = cost.Usage.from_anthropic(r.usage, ttl=_declared_ttl(r))
            except ValueError:
                unprovable += 1
                continue
            already_1h += u.cache_write_1h
            if not (u.cache_write_5m and r.sent_at):
                continue
            # Two writes fifteen minutes apart are only evidence that a longer
            # lifetime would have helped if the second could have *read* what
            # the first wrote. Without segment identity they could be different
            # or drifted prefixes, and a longer lifetime does nothing for those.
            #
            # The span is carried rather than folded into the bucket key,
            # because equality is the wrong test. A rolling conversation marks a
            # stable prefix and an advancing end of history, so the outermost
            # span differs on every request: keyed on it, every bucket held one
            # write, every write looked cold, and each one *subtracted* the 1h
            # premium before the agent was dropped for having no in-band
            # rewrite. On the most common agent shape there is, this rule scored
            # against its own recommendation and then stayed silent -- while the
            # report raised EFF-1 and sent the reader hunting for a rotating id.
            #
            # `span_is_reusable_by` is the simulator's own read condition, which
            # has always credited an advancing breakpoint reading the shorter
            # entry a previous turn wrote.
            spans = marked_spans(r.segments)
            if not spans:
                unkeyed += 1
                continue
            by_scope[(r.tenant, r.target_id, r.model, r.session)].append(
                (r.sent_at, u.cache_write_5m, rate_for(r.model, _when(r), r.target_id),
                 spans, (r.usage.get("cache_read_input_tokens") or 0)))

        # Only a rewrite in the 5m-to-1h band is evidence that the five-minute
        # lifetime is what caused the miss. A rewrite 60 seconds after the last
        # one happened while a five-minute entry should still have been alive,
        # so something else invalidated it: prefix drift, fan-out, a different
        # cache pool. A longer lifetime does not fix any of those, and counting
        # them here published savings for traffic the recommendation cannot
        # help. EFF-1 and VOL-1 are the findings that name those causes.
        # Every write that moved `recoverable`, in either direction, for the
        # projection floor. Both arms count: the in-band rewrites this rule would
        # convert to reads and the cold writes it charges the 1h premium for are
        # equally part of the sample the net figure is scaled from. A rewrite
        # excluded as a non-TTL miss is not -- it contributes no term.
        contributed: list = []
        non_ttl_misses, in_band_rewrites = 0, 0
        for scope, writes in by_scope.items():
            writes.sort()
            # The scope's own surface. This read `r.target_id` -- `r` left over
            # from the loop that *built* by_scope, so on a mixed-surface agent
            # every scope was priced with whichever request happened to be last.
            # Hoisted out of the innermost loop too: it never varied within a
            # scope, and re-reading the registry per write only made the wrong
            # value harder to see.
            # Guarded, because this rule now runs over rows the registry may not
            # be able to answer for at all. It was unguarded while the
            # dispatcher only ever handed it the priced subset, which is the
            # kind of assumption that stops being true one line of dispatch
            # away.
            try:
                _m = registry.multipliers(scope[1])
            except registry.RegistryError:
                continue
            _w5, _w1h, _read = _m["write_5m"], _m["write_1h"], _m["read"]
            # The provider searches back a bounded number of blocks from a
            # breakpoint, and this rule publishes a dollar figure, so the bound
            # is enforced. Without it a tool loop appending 25 messages per call
            # was credited for reads at 25 blocks against a recorded window of
            # 20 -- every one of them a read the provider would not give.
            #
            # None where the surface records no window: unbounded is the honest
            # reading of a number nobody has written down, and the containment
            # and reachability tests still apply.
            try:
                _lookback = registry.capability(scope[1], "lookback_blocks")
            except registry.RegistryError:
                _lookback = None
            # Below the provider's minimum no entry is written, so there is
            # nothing for a later request to read and nothing a longer lifetime
            # could recover. `simulate()` drops these markers before they can be
            # read; this rule was pricing recovery from them.
            try:
                _minimum = registry.min_cacheable_tokens(scope[1], scope[2])
            except (registry.RegistryError, IndexError):
                _minimum = None
            # The most recent earlier write whose span this request could read,
            # not simply the previous write in the scope. Walking backwards
            # stops at the first match, which is the longest still-relevant one
            # because writes are in time order.
            earlier: list = []
            for sent_at, tokens, rate, spans, read in writes:
                per_token = rate / 1e6
                # The size of the cache ENTRY this request leaves behind, which
                # is not the size of the write that touched it.
                #
                # A read refreshes the entry it read, so a request that reads a
                # 100k prefix back and writes a 1k appended suffix leaves a live
                # 100k+1k entry while writing 1,000 tokens. Storing the write
                # meant the later TTL miss on that entry was credited with
                # recovering 1,000 tokens instead of 101,000 -- and since the
                # cold-write premium is charged against the real write, the
                # netting came out negative and the rule went silent. Measured
                # on a synthetic thirteen-request trace of refresh-then-miss
                # cycles: TTL-1 did not fire at all, against $3.11 of genuine
                # recovery over the window. A correct recommendation never
                # reaching the client is a different kind of wrong from an
                # inflated one, but it is still wrong.
                #
                # `read + write` is the cached part of this prompt: the provider
                # bills each input token exactly once as a read, a write, or
                # plain input, so the two are disjoint and adding them
                # double-counts nothing.
                #
                # Pre-existing rather than introduced by the proportional share
                # above -- measured on the same fixture with that share reverted,
                # TTL-1 was equally silent. `read` is zero on every cache-writing
                # row in every shipped fixture (0 of 136 in demo-traces.jsonl),
                # which is why nothing caught it and why no published figure
                # moves: with no read this is exactly `tokens` and the
                # single-marker case is unchanged to the cent.
                #
                # The cap below still bounds recovery by what the later request
                # actually wrote, so this cannot credit a read that never had to
                # be re-paid for.
                established = read + tokens
                match = None
                for prev_at, prev_span, prev_tokens in reversed(earlier):
                    if span_is_reusable_by(prev_span, spans, _lookback,
                                          _minimum):
                        match = (prev_at, prev_tokens)
                        break
                # The share of this request's write that each span accounts for,
                # not the whole write against every one of them.
                #
                # A request with nested markers writes one billed total covering
                # its outermost span, and this list is what a *later* request
                # matches against. Storing `tokens` on every entry meant a match
                # on an inner span was credited with the outer span's write --
                # and that is the common case rather than a corner one, because
                # the lookback window rejects the outer marker of a request that
                # has since appended a few turns, leaving the inner one to match.
                #
                # Measured on a synthetic ten-request trace with a 50k inner
                # marker inside a 100k outer one, ten minutes apart, appending 22
                # blocks a turn past a recorded lookback of 20: $4.80 recoverable
                # over the window against $2.21 once each span carries only its
                # own share. The rule was crediting a one-hour lifetime with
                # converting a write it never made.
                #
                # Proportional to the segment estimate, applied to the billed
                # figure -- not the segment estimate itself. The estimate is a
                # split of the billed total, not a second opinion on it, and
                # substituting it cut a fixture writing 200,000 tokens down to
                # its 7,000-token segment sum and silenced the rule entirely. A
                # ratio of two estimates is scale-invariant, so it survives an
                # estimate that is uniformly wrong, and it is exactly 1.0 for a
                # single marker -- which is why every existing figure is
                # unchanged.
                outermost_tokens = spans[-1][1] or 0
                for span in spans:
                    share = (span[1] / outermost_tokens) if outermost_tokens else 1.0
                    earlier.append((sent_at, span, established * share))
                gap = None if match is None else (sent_at - match[0]).total_seconds()
                if gap is not None and gap <= 300:
                    non_ttl_misses += 1        # a 5m entry was still alive; not a TTL problem
                    continue
                if gap is not None and gap < 3600:
                    in_band_rewrites += 1
                    # What becomes a read is the entry the *earlier* request
                    # wrote, not this one's whole write. On equal spans the two
                    # are the same number and nothing moves; under containment
                    # this write is prefix plus a new turn, and the turn is new
                    # content under either lifetime, so crediting all of it
                    # overstates the saving.
                    #
                    # Both sides are the provider's own billed counters rather
                    # than our segment token estimates. Bounding by the estimate
                    # instead cut a fixture writing 200,000 tokens down to its
                    # 7,000-token segment sum and silenced the rule -- the
                    # estimate is a split of the billed total, not a second
                    # opinion on it.
                    #
                    # `match[1]` is the matched entry's size and `tokens` is what
                    # this request actually wrote. The cap is what keeps this
                    # honest in the other direction: however large the entry was,
                    # a longer lifetime can only save what this request paid to
                    # write again.
                    recovered = min(match[1], tokens)
                    recoverable += recovered * per_token * (_w5 - _read)
                    contributed.append(sent_at)
                else:
                    recoverable -= tokens * per_token * (_w1h - _w5)    # cold write costs more
                    contributed.append(sent_at)

        # A longer lifetime only helps if the prefix is actually reused, and
        # there are two ways to prove reuse. Observed reads are one. The other
        # is the same prefix key rewritten with an in-band gap, which `by_scope`
        # measures directly because it is keyed on the cached span.
        #
        # Requiring reads excluded the canonical case this entire rule exists
        # for: a stable prefix called every ten minutes under a five-minute
        # lifetime rewrites on every request and therefore reads *nothing*. Zero
        # reads was treated as proof the prefix was drifting, so the report
        # raised EFF-1 and REB-1 and sent the operator hunting for compaction or
        # a rotating id, while the actual fix -- a one-hour TTL -- was
        # suppressed. The tool's own headline thesis, misdiagnosed.
        reads = sum(r.usage.get("cache_read_input_tokens", 0) or 0 for r in mine)
        wrote_5m = unkeyed > 0 or any(by_scope.values())
        if not wrote_5m:
            continue
        if not reads and not in_band_rewrites:
            continue
        # Something a longer lifetime could actually fix has to exist. With no
        # in-band rewrite and nothing unkeyed, every miss had another cause.
        if not in_band_rewrites and not unkeyed:
            continue
        # If the lifetime is provable and the arithmetic says the change loses
        # money, say nothing. Reporting the cadence observation anyway left a
        # finding whose fix told the reader to set a one-hour TTL on a workload
        # where this rule had just computed that doing so costs more -- the cold
        # writes at 2.0x outweighing the rewrites recovered at 0.1x.
        if unkeyed == 0 and recoverable <= 0:
            continue
        # Money only when the reuse is proven. Otherwise the cadence
        # observation still stands and is worth reporting, but as a hypothesis
        # rather than a figure.
        #
        # `priced` is an explicit fence rather than a consequence. This rule now
        # runs over rows the registry cannot price, where `rate_for` answers 0.0
        # and every term multiplies out to exactly zero -- so `recoverable > 0`
        # already happens to be False and the arithmetic already happens to
        # protect us. Depending on a float landing on zero to keep a dollar
        # figure off the page is not a gate, it is a coincidence that holds
        # today. Naming the condition means a future rate of 0.001 on a surface
        # nobody priced cannot turn into a published saving.
        priced = any(rate for _s, _t, rate, _sp, _rd in
                     (row for rows in by_scope.values() for row in rows))
        monetizable = unkeyed == 0 and recoverable > 0 and priced
        rank = recoverable if monetizable else 0.0
        candidates.append((rank, agent, median, in_band, len(ts),
                           already_1h, unprovable, non_ttl_misses, monetizable,
                           # Carried per candidate, not read off the winner
                           # afterwards: `contributed` is rebuilt per agent, so
                           # taking it from the enclosing scope after the loop
                           # would hand the winning agent whichever agent
                           # happened to be iterated last.
                           contributed))
    if not candidates:
        return None
    # Nothing written means nothing to recover by changing the lifetime; that
    # agent has a different problem and a different finding will name it.
    (extra, agent, median, in_band, n, already_1h, unprovable, non_ttl,
     monetizable, contributed) = max(candidates, key=lambda c: (c[0], c[3]))
    caveat = ""
    if already_1h:
        caveat = (f" {already_1h:,} tokens for this agent are already written at the "
                  f"one-hour lifetime and are excluded; the figure covers only the "
                  f"five-minute writes.")
    if unprovable:
        caveat += (f" {unprovable} request(s) had writes of unprovable lifetime and are "
                   f"excluded entirely.")
    if non_ttl:
        caveat += (f" {non_ttl} rewrite(s) happened within five minutes of the previous "
                   f"one, while a five-minute entry should still have been alive. Those "
                   f"are a different miss mechanism and are excluded: a longer lifetime "
                   f"would not have prevented them.")
    if not monetizable and unkeyed:
        caveat += (" No dollar figure is given: this trace does not carry segment identity, "
                   "so there is no evidence that the writes fifteen minutes apart were the "
                   "same prefix rather than a drifted one. A longer lifetime only helps if "
                   "the same span is being rewritten. Instrument the workload to settle it.")
    return Finding(
        code="TTL-1",
        # Structural exactly when it carries money, because that is exactly
        # when the claim rests on `marked_spans`. Both figures above are built
        # from the span walk -- which spans matched, and since the nested-marker
        # fix, what share of a write each span accounts for -- so on an
        # unaligned or uncounted trace they are costed from boundaries nobody
        # measured. Without this flag they walked past the structural trust gate
        # entirely, which is the gate that exists for precisely that.
        #
        # `monetizable` is `unkeyed == 0` here: the branch above drops any agent
        # with no unkeyed requests and nothing recoverable, so the two are the
        # same condition. Not a coincidence to rely on quietly -- it is what
        # makes this flag mean "the figure came from spans" rather than
        # "somebody thought this looked structural".
        #
        # Not unconditional. The other arm fires when the trace carries no
        # segment identity, says so in its own detail, and publishes nothing;
        # marking it structural would attach the report's "these rest on
        # segment boundaries inferred from logged bodies" note to a finding
        # whose text says there are no segments to rest on.
        structural=monetizable,
        title=f"'{agent}' request cadence sits inside the one-hour window",
        severity="high" if monetizable else "medium", evidence_class=MODELED,
        detail=(f"{in_band:.0%} of gaps between requests fall between five minutes and one "
                f"hour (median {median/60:.1f} min across {n} requests). In that band a "
                f"five-minute cache expires before the next request and rewrites at 1.25x, "
                f"while a one-hour cache would have read at 0.1x.{caveat}"),
        affected_requests=n,
        **_avoidable(extra if monetizable else None, window, contributed),
        confidence="medium" if monetizable else "low", quality_risk="low",
        fix="Set a one-hour TTL on the static prefix for this agent. Outside the "
            "five-minute-to-one-hour band the five-minute default is cheaper, so this "
            "is not a blanket change.")


def _f_ttl_premium_unearned(reqs, ratios, window, rate_for) -> Finding | None:
    """The other direction: a one-hour lifetime bought where five minutes would do.

    TTL-1 answers "your cadence sits in the band and you are not using the long
    lifetime". This is its mirror, and until now nothing here asked it: the long
    lifetime is in use and the cadence never needed it. A reader whose agent
    defaults to 1h got no finding at all, which reads as approval.

    The naive version of this rule is wrong, and wrong in the expensive
    direction. Counting 1h write tokens and multiplying by the 2.0-to-1.25
    premium says "switch and save", and on the workload that prompted this rule
    that answer is backwards. The gaps that fall in the five-minute-to-one-hour
    band are rare -- 1.4% of them on the trace this was built against -- but each
    one sits on a prefix that has been accumulating all session. Dropping to 5m
    kills those entries, and the next request rewrites the whole prefix.

    So both sides get counted, per isolation scope, per gap:

        under 5 min   the 5m entry survives too, so the only difference is the
                      write premium.                        saving 0.75x
        5 min - 1 hr  the 5m entry is dead. The request writes prefix+delta at
                      1.25x instead of reading prefix at 0.1x and writing delta
                      at 2.0x.                       cost 1.15x on the prefix
        over 1 hour   both are dead. No difference.

    Net = 0.75 * (1h writes) - 1.15 * (prefix rewritten at band gaps), in
    multiplier-units of the input rate. On the trace above that is +$482 against
    -$526, so the long lifetime is right and the rule says so rather than
    staying silent, because "I checked and 1h is correct here" is the answer
    people actually arrive with the question about.

    Multipliers come from the registry rather than being written into the
    arithmetic, so a surface that prices its lifetimes differently is handled by
    the same code and a surface with no 1h at all is skipped.
    """
    by_scope = defaultdict(list)
    unprovable = 0
    for r in reqs:
        if not r.sent_at:
            continue
        try:
            u = cost.Usage.from_anthropic(r.usage, ttl=_declared_ttl(r))
        except ValueError:
            unprovable += 1
            continue
        by_scope[(r.tenant, r.target_id, r.model, r.session)].append((r.sent_at, r, u))

    written_1h = 0
    band_prefix = 0            # tokens that would be rewritten if 1h became 5m
    # `band_gaps` counts every gap in the band. `band_priced` counts the subset
    # carrying a read, which is the only subset whose rebuild cost can be
    # measured. Conflating the two put the rarity premise on the priced subset,
    # so in-band gaps that read nothing vanished from the denominator: 20 fast
    # gaps beside 20 in-band ones with unstable prefixes reported "100% of gaps
    # are under five minutes" and recommended shortening the lifetime. That is
    # the worst case to be wrong on -- no reads at an in-band gap means the
    # prefix is not surviving, and a shorter lifetime makes that harder to see.
    band_gaps = band_priced = under_5m = over_1h = 0
    saving = cost_of_switch = 0.0
    surfaces = set()
    # Requests that moved `net`, for the projection floor: the ones that wrote
    # at the one-hour lifetime (the saving) and the ones at an in-band gap that
    # carried a read (the rebuild it would cost). A gap counted only for the
    # rarity premise contributes no dollars and is not part of the sample.
    contributed: list = []
    for (tenant, target_id, model, session), rows in by_scope.items():
        try:
            m = registry.multipliers(target_id)
        except registry.RegistryError:
            continue
        w5, w1h, read = m.get("write_5m"), m.get("write_1h"), m.get("read")
        # An implicit-prefix surface has no lifetimes to choose between, so
        # there is no premium to have overpaid. Skipping is the honest answer;
        # pricing it against Anthropic's multipliers is how a Bedrock trace once
        # produced a total no bill would match.
        if not (w5 and w1h and read) or w1h <= w5:
            continue
        surfaces.add(target_id)
        rows.sort(key=lambda x: x[0])
        prev = None
        for sent_at, r, u in rows:
            per_token = rate_for(model, _when(r), r.target_id) / 1e6
            if u.cache_write_1h:
                written_1h += u.cache_write_1h
                saving += u.cache_write_1h * per_token * (w1h - w5)
                contributed.append(sent_at)
            if prev is not None:
                gap = (sent_at - prev[0]).total_seconds()
                if gap < 300:
                    under_5m += 1
                elif gap >= 3600:
                    over_1h += 1
                else:
                    # The entry the 1h lifetime kept alive. Under 5m it is gone
                    # and this request rewrites what it would have read.
                    band_gaps += 1
                    prefix = r.usage.get("cache_read_input_tokens") or 0
                    if prefix:
                        band_priced += 1
                        band_prefix += prefix
                        cost_of_switch += prefix * per_token * (w5 - read)
                        contributed.append(sent_at)
            prev = (sent_at, r, u)

    if not written_1h:
        return None
    total_gaps = under_5m + band_gaps + over_1h
    if total_gaps < 10:
        return None

    net = saving - cost_of_switch
    share_short = under_5m / total_gaps if total_gaps else 0.0
    share_band = band_gaps / total_gaps if total_gaps else 0.0
    # The rule's own premise: the cadence never needed the long lifetime. Where
    # a real share of gaps land in the band, it is load-bearing and this rule
    # has nothing to say -- TTL-1 owns lifetime questions there.
    #
    # Without this, both rules fired on one trace and told the reader to move
    # the lifetime in both directions at once. The arithmetic below is not what
    # went wrong: on a churning workload that reads almost nothing, dropping to
    # 5m genuinely does look cheap, because the model prices the rebuild from
    # the reads that are actually happening. It is the recommendation that is
    # wrong. A workload whose requests arrive ten minutes apart wants the longer
    # lifetime and wants its prefix stabilised, and shortening the lifetime
    # first makes the second problem harder to see.
    if share_band > BAND_IS_RARE:
        return None
    # An in-band gap that read nothing is a prefix that did not survive its own
    # lifetime. The rebuild it would cost under 5m cannot be priced -- there is
    # no read to price it from -- so `net` is computed as though those gaps were
    # free, which biases it toward recommending the switch. They are also the
    # signal that the prefix itself is the problem, and shortening the lifetime
    # buries that. Report the observation, publish no saving.
    unread_band = band_gaps - band_priced
    common = (
        f"{written_1h:,} tokens were written at the one-hour lifetime, which "
        f"bills at {w1h}x where the five-minute lifetime bills at {w5}x. "
        f"{share_short:.1%} of gaps between requests in a cache scope are under "
        f"five minutes, so a five-minute entry would have survived them too. "
        f"{band_gaps:,} gap(s) fall in the five-minute-to-one-hour band, where "
        f"only the one-hour entry survives; {band_priced:,} of those carry a "
        f"read and sit on {band_prefix:,} tokens of prefix that a five-minute "
        f"lifetime would have forced to be written again."
        + (f" {unread_band:,} in-band gap(s) read nothing at all, so the entry "
           f"the one-hour lifetime was holding did not survive to be used. "
           f"Their rebuild cost cannot be priced and no saving is published."
           if unread_band else "")
        + (f" {unprovable} request(s) had writes of unprovable lifetime and are "
           f"excluded." if unprovable else ""))

    if net > 0 and unread_band:
        return Finding(
            code="TTL-2",
            title="One-hour writes whose entries are not being read back",
            severity="medium", evidence_class=MODELED,
            detail=common + (
                " Shortening the lifetime is not the indicated change. The "
                "arithmetic favours it only because the unpriced gaps are "
                "counted as free, and a prefix that expires unread would expire "
                "unread faster."),
            affected_requests=sum(len(v) for v in by_scope.values()),
            avoidable_usd_month=None,
            confidence="low", quality_risk="medium",
            fix="Find what invalidates the prefix between requests before "
                "touching the lifetime. EFF-1 and REB-1 name the usual causes. "
                "Once the prefix survives, re-run and this rule can price the "
                "lifetime question properly.")

    if net > 0:
        return Finding(
            code="TTL-2",
            title="Paying the one-hour premium where five minutes would do",
            severity="high" if band_gaps == 0 else "medium",
            evidence_class=MODELED,
            detail=common + (
                f" Netting the two: the cheaper writes are worth more than the "
                f"rewrites they would cause, so dropping to five minutes comes "
                f"out ahead. This is a projection, not an observation -- the "
                f"five-minute arm was never run."),
            affected_requests=sum(len(v) for v in by_scope.values()),
            **_avoidable(abs(net), window, contributed),
            confidence="medium" if band_gaps else "high", quality_risk="medium",
            fix="Drop the static prefix to the five-minute lifetime and measure "
                "again before assuming it held. Where a scope genuinely idles "
                "past five minutes, leave that one at an hour: this is a "
                "per-workload call, not a global switch.")

    # Nothing to recover, and worth saying. Staying silent here is what left a
    # reader with a 1h default no answer at all, and "checked, and the long
    # lifetime is earning its premium" is the answer to the question they came
    # with. Priced at the cost of the change rather than a saving, so the figure
    # is what switching would *lose*.
    return Finding(
        code="TTL-2", title="The one-hour lifetime is earning its premium here",
        severity="low", evidence_class=MODELED,
        detail=common + (
            f" Netting the two: the rewrites a five-minute lifetime would force "
            f"outweigh the cheaper writes it would buy, because those band gaps "
            f"sit on prefixes averaging {band_prefix // max(1, band_priced):,} "
            f"tokens. Switching is modelled to cost money rather than save it. "
            f"The band gaps are rare and expensive, which is why counting them "
            f"by frequency gives the wrong answer."),
        affected_requests=sum(len(v) for v in by_scope.values()),
        avoidable_usd_month=None,
        confidence="medium", quality_risk="medium",
        fix="No change indicated. Leave the one-hour lifetime where it is. If "
            "the cache bill is still too high, the lever is prefix stability "
            "or volume, not lifetime.")


def _f_cold_fanout(reqs, ratios, window, rate_for) -> Finding | None:
    """Concurrent identical prefixes all paying to write the same thing."""
    # Bucketed on the cached prefix inside one isolation scope. The old key was
    # the ids of the *marked* segments only, which is not the cached span: every
    # unmarked segment before the outermost marker is part of the prefix too. So
    # two requests with different leading content collapsed into one bucket and
    # were reported as duplicate writes of the same thing. Worse, the key
    # ignored tenant, surface and model entirely, so requests that could never
    # share a cache entry were priced as if one had wasted the other's write.
    buckets = defaultdict(list)
    for r in reqs:
        if not r.sent_at or not r.segments:
            continue
        pk = _prefix_key(r)
        if pk:
            buckets[((r.tenant, r.target_id, r.model), pk)].append(r)
    waste, groups, unprovable = 0.0, 0, 0
    affected = set()
    # The requests in a counted pair, for the projection floor. Fan-out evidence
    # is concurrent by definition, so a trace with a single pair spans seconds
    # and cannot support a month -- which is exactly the case that published
    # $14.79/mo off two requests one second apart while eight unrelated rows
    # elsewhere in the file carried the global floor.
    contributed: dict = {}
    # Which rule decided each pair was concurrent, so the detail can say so
    # rather than implying every trace was measured the same way.
    boundaries: set = set()
    for _key, group in buckets.items():
        group.sort(key=lambda r: r.sent_at)
        for a, b in zip(group, group[1:]):
            # Fan-out is `b` going out before `a`'s entry could be readable, and
            # the trace often says when that was. A flat five seconds is a guess
            # about provider latency dressed as a measurement: a request whose
            # first token arrives at t=20s has a sibling at t=8s that could not
            # possibly have read its cache, and the flat window skipped it
            # because 8 >= 5.
            #
            # `first_token_at` is the observed boundary and is used wherever the
            # recorder captured it. The five seconds stays only as the fallback
            # for traces that carry no first-token timing, and the detail says
            # which one was used.
            visible, observed_boundary = write_visible_at(a)
            if b.sent_at >= visible:
                continue
            boundaries.add(observed_boundary)
            # Both must have written, and the later one's lifetime must be
            # provable, or the premium being reclaimed is a guess.
            try:
                ua = cost.Usage.from_anthropic(a.usage, ttl=_declared_ttl(a))
                ub = cost.Usage.from_anthropic(b.usage, ttl=_declared_ttl(b))
            except ValueError:
                unprovable += 1
                continue
            if not (ua.cache_write_5m + ua.cache_write_1h):
                continue
            if not (ub.cache_write_5m + ub.cache_write_1h):
                continue
            try:
                m = registry.multipliers(b.target_id)
            except registry.RegistryError:
                m = {"write_5m": 1.25, "write_1h": 2.0}
            per_token = rate_for(b.model, _when(b), b.target_id) / 1e6
            groups += 1
            affected.update((a.request_id, b.request_id))
            # `affected` keeps both sides; the projection sample keeps only `b`.
            #
            # They answer different questions and the difference is the whole
            # defect. "How many requests does this affect" is honestly two per
            # pair -- both went out, both are part of the phenomenon. "How many
            # independent observations is the monthly figure averaging over" is
            # one per pair, because `waste` below charges `ub` and nothing else:
            # the leader's write is the one that had to happen, and it moves the
            # figure by exactly zero.
            #
            # Recording both padded the sample to twice its real size, which is
            # precisely how the request floor gets cleared by requests that
            # contributed nothing. Measured on five pairs spread over five days:
            # ten timestamps, five charged, floor cleared, and FAN-1 published
            # $43.12/mo marked `reconciled` with `total_avoidable_month` at
            # $61.87.
            #
            # Keyed by request id because three requests fanning out form two
            # adjacent pairs, so the middle one is charged once and seen twice.
            contributed[b.request_id] = b.sent_at
            waste += ub.cache_write_5m * per_token * (m["write_5m"] - 0.10)
            waste += ub.cache_write_1h * per_token * (m["write_1h"] - 0.10)
    # One pair is the whole phenomenon. Two concurrent requests writing the
    # same prefix is the minimum case and also the commonest, and the runtime
    # monitor alerts on exactly two -- so requiring two pairs meant the report
    # stayed silent about something the live check had already flagged.
    if groups < 1:
        return None
    return Finding(
        code="FAN-1", structural=True, title="Concurrent requests each writing the same prefix",
        severity="medium", evidence_class=MEASURED,
        detail=(f"{groups} pair(s) of requests with an identical cached prefix went out "
                f"before the first of them could have been read back, and both paid to "
                f"write it. A cache entry only becomes readable once the first response "
                f"has begun streaming, so requests fired in parallel cannot share the "
                f"write."
                + (" Concurrency was measured against the observed first-token time "
                   "on this trace."
                   if boundaries == {True} else
                   " This trace carries no first-token timing, so concurrency was taken "
                   "as a five-second window -- a stand-in for provider latency rather "
                   "than an observation. Capture first_token_at to measure it."
                   if boundaries == {False} else
                   " Some pairs were measured against an observed first-token time and "
                   "the rest against a five-second window, which is a stand-in used "
                   "where that timing is missing.")),
        # Unique requests, not pairs times two. Three requests fanning out form
        # two adjacent pairs and would have been reported as four requests.
        affected_requests=len(affected),
        **_avoidable(waste, window, contributed.values()),
        confidence="medium", quality_risk="low",
        fix="Send one request, wait for its first token, then release the rest of the batch.")


@_rate_free
def _f_model_split(reqs, ratios, window, rate_for) -> Finding | None:
    """One session spread across models means separate cache pools."""
    # Scoped by tenant and surface. Session ids are not globally unique in a
    # shared gateway export, so two tenants under the same string were reported
    # as one session switching model -- a user-visible finding, and a
    # remediation, for traffic that could never share a cache context.
    # Agent is part of the key, for the same reason `_sessions` keys on it: a
    # subagent runs its own context and Claude Code's subagents share the
    # parent's sessionId. Without it, a main loop on one model and a subagent on
    # another read as one conversation switching mid-flight -- a finding, and a
    # remediation, for two contexts that were never one.
    per_session = defaultdict(set)
    for r in reqs:
        if r.session:
            per_session[(r.tenant, r.target_id, r.agent, r.session)].add(r.model)
    split = {s: m for s, m in per_session.items() if len(m) > 1}
    if not split:
        return None
    models = sorted({m for ms in split.values() for m in ms})
    return Finding(
        code="SPL-1", title="Sessions switching model mid-conversation",
        severity="medium", evidence_class=MEASURED,
        detail=(f"{len(split)} sessions used more than one model ({', '.join(models)}). "
                f"Caches are per-model, so the accumulated prefix does not carry across "
                f"the switch and is written again from cold."),
        # Counted against the same scoped key `split` was built from. Testing
        # `r.session in split` compared a bare session id against a set of
        # (tenant, target, session) tuples, so it matched nothing and every
        # SPL-1 reported zero affected requests -- the finding fired, and its
        # blast radius read as none.
        affected_requests=sum(1 for r in reqs
                              if (r.tenant, r.target_id, r.agent, r.session) in split),
        avoidable_usd_month=None,
        confidence="high", quality_risk="medium",
        fix="Keep a session on one model. Where a cheaper model is wanted for a sub-task, "
            "run it as a separate call rather than switching the main loop.")


def _sessions(reqs):
    """Requests grouped by cache pool, agent and session, in time order.

    Agent is part of the key because a subagent runs its own context. Claude
    Code's subagents share the parent's `sessionId`, so grouping on session
    alone compared one context's write against another context's established
    prefix -- and the answer then depended on how the two interleaved rather
    than on what either did. Measured on a fixture where subagents spawn
    repeatedly: each cold start is a genuine repeated write and shows up as
    REB-1 on its own, and pooling it with a main loop that was merely extending
    made the finding disappear entirely.
    """
    out = defaultdict(list)
    for r in reqs:
        if r.session and r.sent_at:
            out[(r.tenant, r.target_id, r.agent, r.session)].append(r)
    for group in out.values():
        group.sort(key=lambda r: r.sent_at)
    return out


# A request that rewrites at least this share of the prefix it had established
# is rebuilding it, not extending it. An extension writes the increment; a
# rebuild writes the lot.
REBUILD_FRACTION = 0.5

# What the read:write cost split looks like on a session that never rebuilds:
# read the whole prefix each turn, write only what was added.
HEALTHY_READ_SHARE = 0.75


@_rate_free
def _f_prefix_rebuild(reqs, ratios, window, rate_for) -> Finding | None:
    """How often the accumulated prefix is thrown away and paid for again.

    The most useful thing this tool can say about a trace with no prompt
    structure at all, which is the trace most people can actually produce. It
    needs three usage counters and a session id.

    Worth stating plainly because the field gets it backwards: a large cache
    share of the bill is not evidence of a cache problem. Reads bill at a tenth
    of the input rate, so they dominate a bill only when the volume behind them
    is enormous -- which is caching working. What costs money is *rebuilding*:
    a write bills 1.25x or 2x to replace reads that billed 0.1x, so every
    rebuild is roughly a twelve-fold swing on the tokens involved.
    """
    sessions = _sessions(reqs)
    # Counted, not all-or-nothing. `if not sessions` meant one sessioned request
    # anywhere in the export silenced this rule for every sessionless one --
    # and the rows without a session are exactly the rows REB-1 cannot reason
    # about. Two small sessioned requests beside fifty sessionless 50k writers
    # reported no rebuilds and no abstention, which reads as "measured, none
    # found" when the question was never asked of the expensive half.
    grouped = {r.request_id for g in sessions.values() for r in g}
    unmeasurable = [r for r in reqs
                    if write_tokens(r.usage) and r.request_id not in grouped]
    # Reported alongside REB-1, not instead of it. Returning here meant one
    # sessionless write anywhere suppressed the rebuild analysis of every
    # grouped session -- and the detail then told the reader that REB-1 covered
    # the rest, which it did not, because it never ran. A small unmeasurable
    # slice hid the expensive pattern in the measurable one.
    #
    # `_rebuild_abstention` is raised first only when there is nothing left to
    # measure. Otherwise it is stashed and the caller emits both.
    # REB-0 belongs to `_f_rebuild_abstention` alone. Splitting it into its own
    # rule left this early return in place, so an all-sessionless export emitted
    # REB-0 twice -- the same population counted once by each path, which is the
    # double-count the split was meant to prevent in the other direction.
    rebuilds, extended, switched, tokens, expired = 0, 0, 0, 0, 0
    unprovable_turns = 0
    for group in sessions.values():
        prev = None
        for r in group:
            created = write_tokens(r.usage)
            read = r.usage.get("cache_read_input_tokens") or 0
            if prev is not None:
                established = ((prev.usage.get("cache_read_input_tokens") or 0)
                               + write_tokens(prev.usage))
                gap = ((r.sent_at - prev.sent_at).total_seconds()
                       if r.sent_at and prev.sent_at else None)
                lifetime = _expiry_lifetime_seconds(prev)
                if established and gap is not None and lifetime is None:
                    # Without the previous entry's lifetime, expiry and rebuild
                    # are indistinguishable, and counting it either way invents
                    # a finding. Excluded from both, and said in the detail.
                    unprovable_turns += 1
                elif (established and gap is not None and lifetime is not None
                        and gap >= lifetime):
                    # The entry had expired before this request arrived, so a
                    # full write is what the provider had to do. Calling that a
                    # rebuild points at compaction or a rotating id when the
                    # cause is the lifetime, and TTL-1 is the finding that fixes
                    # it.
                    expired += 1
                elif established and created >= REBUILD_FRACTION * established:
                    rebuilds += 1
                    tokens += created
                    if r.model != prev.model:
                        switched += 1
                elif read:
                    extended += 1
            prev = r
    # Model-switch rebuilds are excluded in the detail text and were counted in
    # every number: `turns`, `interval`, severity and affected_requests. So a
    # session switching model every ten turns fired REB-1 at high severity,
    # said the rebuilds were expected, and pointed the reader at compaction --
    # while SPL-1 correctly named the model switch on the same trace. Excluded
    # means excluded.
    unexplained = rebuilds - switched
    turns = rebuilds + extended + expired - switched
    if not unexplained or turns < 10:
        # Every rebuild here crossed a model change. That is a separate cache
        # pool behaving as designed, and SPL-1 owns it.
        return None
    interval = turns / unexplained
    if interval >= 40:
        return None
    return Finding(
        code="REB-1", title="The cached prefix is being rebuilt, not extended",
        severity="high" if interval < 15 else "medium",
        evidence_class=MEASURED,
        # Structure: the number, what it costs, then what is causing it. The
        # exclusions used to sit between the count and the conclusion, three
        # subordinate clauses deep, so the sentence that says what is actually
        # wrong arrived last and read as an afterthought. They are real and they
        # stay; they go in a bracketed tail where a reader can skip them.
        detail=(f"{rebuilds:,} of {turns:,} in-session turns rewrote at least half the "
                f"prefix they had already established, about one every "
                f"{interval:.0f} turns, covering {tokens:,} written tokens. "
                f"Extending a prefix writes only what was added and reads the rest "
                f"at 0.1x. A rebuild writes the lot again at 1.25x or 2x."
                + (f" {unexplained:,} of them have no innocent explanation, so "
                   f"something is changing the prefix itself. Context compaction, a "
                   f"rotating identifier and reordered tool definitions are what "
                   f"this usually turns out to be."
                   if unexplained else "")
                + ((" Excluded: "
                    + "; ".join(
                        [f"{switched} rebuild(s) across a model change, which is a "
                         f"separate cache pool and is expected"] * bool(switched)
                        + [f"{expired:,} turn(s) whose entry had already expired, "
                           f"which is a lifetime question and is TTL-1's, not this "
                           f"finding's"] * bool(expired)
                        + [f"{unprovable_turns:,} turn(s) whose previous write has no "
                           f"known lifetime, so expiry and rebuild cannot be told "
                           f"apart"] * bool(unprovable_turns))
                    + ".")
                   if (switched or expired or unprovable_turns) else "")),
        affected_requests=unexplained,
        avoidable_usd_month=None,
        confidence="high", quality_risk="low",
        fix="Find what changes the prefix between turns before touching a TTL. "
            "Compaction in particular trades a smaller prompt for a full rewrite, "
            "and immediately after it every read you were getting at 0.1x is "
            "rebought at 1.25x.")


def _f_cache_verdict(reqs, ratios, window, rate_for) -> Finding | None:
    """Is the cache helping, and by how much? Stated whether or not it is.

    Exists because the common field diagnosis is exactly backwards. "Most of my
    bill is cache reads and writes" is read as a cache problem and reported as
    one; on a working cache it is the signature of high volume being served
    cheaply. The only thing that settles it is the counterfactual, and a report
    that stays silent when nothing is wrong leaves the reader with the number
    that misled them.
    """
    spend = uncached = read_usd = write_usd = 0.0
    priced = 0
    for r in reqs:
        try:
            u = cost.Usage.from_anthropic(r.usage, ttl=_declared_ttl(r))
            s = cost.price(u, r.model, target_id=r.target_id, on_date=_when(r),
                           effective_rate=rate_for(r.model, _when(r), r.target_id))
        except (ValueError, registry.RegistryError):
            continue
        priced += 1
        spend += s.usd
        uncached += s.hypothetical_uncached_usd
        read_usd += s.read_usd
        write_usd += s.write_usd
    cache_usd = read_usd + write_usd
    if not priced or not spend or not cache_usd:
        return None
    share = cache_usd / spend
    saving = (uncached - spend) / uncached if uncached else 0.0
    read_share = read_usd / cache_usd
    helping = spend < uncached
    return Finding(
        code="CAC-1",
        title=("Caching is paying for itself" if helping
               else "Caching costs more here than sending the prompts uncached"),
        severity="low" if helping else "high", evidence_class=MEASURED,
        detail=(f"Cache reads and writes are {share:.0%} of input spend across "
                f"{priced:,} priced requests. That share on its own says nothing about "
                f"whether caching is working -- reads bill at a tenth of the input rate, "
                f"so they dominate a bill only when the volume behind them is large. "
                f"The counterfactual is what settles it: the same traffic sent uncached "
                f"would cost {uncached/spend:.1f}x what it does now"
                + (f", so caching is removing {saving:.0%} of input spend."
                   if helping else
                   f", so caching is adding {-saving:.0%} to input spend.")
                + f" Of the cache spend itself, {read_share:.0%} is reads. A session "
                  f"that extends its prefix rather than rebuilding it tends to run near "
                  f"{HEALTHY_READ_SHARE:.0%}, so a lower share is worth looking into -- "
                  f"but it is a hint and not a finding, because a workload whose turns "
                  f"add a lot relative to the prefix (large tool results) moves toward "
                  f"even without rebuilding anything. REB-1 counts rebuilds directly and "
                  f"is what settles it."),
        affected_requests=priced,
        avoidable_usd_month=None,
        confidence="high", quality_risk="low",
        fix=("No action indicated on this measure. If the bill is still too high, the "
             "driver is input volume rather than cache configuration, and the questions "
             "are prompt size and turn count." if helping else
             "Stop writing caches that nothing reads before tuning anything else."))


@_rate_free
def _f_rebuild_abstention(reqs, ratios, window, rate_for) -> Finding | None:
    """REB-0 as its own rule, so it cannot suppress REB-1.

    These shared a function and REB-0 returned first. One sessionless cache
    write anywhere therefore silenced the rebuild analysis of every grouped
    session -- while REB-0's own detail told the reader that REB-1 covered the
    rest. It did not; it never ran. A small unmeasurable slice hid the
    expensive pattern in the measurable one.
    """
    sessions = _sessions(reqs)
    grouped = {r.request_id for g in sessions.values() for r in g}
    unmeasurable = [r for r in reqs
                    if write_tokens(r.usage) and r.request_id not in grouped]
    if not unmeasurable:
        return None
    return Finding(
        code="REB-0", title="Prefix rebuilds could not be measured",
        severity="low", evidence_class=MEASURED,
        detail=(f"{len(unmeasurable):,} of {len(reqs):,} cache-writing "
                f"request(s) carry no session id and timestamp pair, so there "
                f"is no way to say which request followed which for them. They "
                f"wrote {sum(write_tokens(r.usage) for r in unmeasurable):,} "
                f"tokens to cache between them, and nothing here can say "
                f"whether that was extension or teardown."
                + (" The rest of the export is grouped, and REB-1 reports on it "
                   "separately." if grouped else "")),
        affected_requests=len(unmeasurable), avoidable_usd_month=None,
        confidence="high", quality_risk="low",
        fix="Export a session or conversation id alongside usage for these "
            "requests. Nothing else about the ingest needs to change.")


RULES = [_f_prefix_efficiency, _f_volatile_prefix, _f_below_minimum,
         _f_ttl_vs_cadence, _f_ttl_premium_unearned, _f_cold_fanout,
         _f_model_split, _f_prefix_rebuild, _f_rebuild_abstention,
         _f_cache_verdict]


def analyze(ts: TraceSet, invoice_usd: float | None = None,
            effective_rate: float | None = None, on_date: str | None = None,
            allow_unreconciled: bool = False) -> Analysis:
    """`allow_unreconciled` releases figures with no invoice, for internal drafts.

    Defaults to False so the absence of an invoice withholds rather than
    publishes. It is a named argument because publishing an unreconciled number
    should be a decision someone made, not something that happens by leaving an
    argument out.
    """
    reqs = ts.analysable
    # Not one date for the whole trace. Pricing is date-effective and the
    # registry already carries a claude-sonnet-5 change on 2026-09-01, so
    # pinning the run date reprices historical traffic at today's rate -- which
    # would make a reconciliation fail for a reason that has nothing to do with
    # the ingest. Same rule SimResult.spend() already follows; it was fixed
    # there and not here.
    def _date_for(r):
        if on_date:
            return on_date
        return r.sent_at.strftime("%Y-%m-%d") if r.sent_at else None

    def _priceable_date(r) -> bool:
        """A row with no timestamp has no date-effective rate.

        cost.price falls back to today when handed None, so once the loaders
        started tolerating malformed timestamps those rows silently repriced
        historical traffic at whatever rate applies on the day the report runs.
        An explicit on_date or effective_rate settles it; nothing else does.
        """
        return bool(on_date) or effective_rate is not None or r.sent_at is not None

    def rate_for(model, when=None, target_id=None):
        """`when` is the day the request was sent, and every caller must pass it.

        Spend already prices per request date. The finding rules did not, so
        after the 2026-09-01 sonnet-5 change an August trace reconciled at
        $2/Mtok while its findings published avoidable dollars at $3 -- the same
        number computed two ways in one report.
        """
        if effective_rate is not None:
            return effective_rate
        try:
            # The surface, not just the model. Without it this returned
            # Anthropic list prices for a Bedrock request and every finding
            # priced from them, while `cost.price` -- which does check --
            # correctly refused the same row.
            if target_id is None:
                # No default. Three adapters were fixed today for inferring
                # anthropic/direct from a missing surface, and a default here
                # would put that back one layer down, where the rate actually
                # comes from. Every call site names the request's own surface.
                raise TypeError(
                    "rate_for needs the request's target_id; defaulting a "
                    "surface is how partner traffic gets first-party rates")
            return registry.base_rate(
                model,
                when or on_date or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                target_id)
        except registry.RegistryError:
            return 0.0

    usages, unprovable = _usages(reqs)
    priced = [r for r in reqs if r not in unprovable]
    ratios = cost.ratios(usages)
    window = _window_days(reqs)

    # One unpriceable model must not take the whole report with it. Pricing
    # rows are date-effective and model ids drift -- a real 29-day trace here
    # already carried `claude-haiku-4-5-20251001`, and only the adapter's
    # normalisation kept it out of this path. A new model shipping should cost
    # that model's requests, not the analysis. Those requests are excluded and
    # fail the publication gate, exactly like an unprovable write lifetime.
    spend_total = uncached_total = 0.0
    unpriceable_models: dict[str, int] = defaultdict(int)
    unpriceable_surfaces: dict[str, int] = defaultdict(int)
    priced_usages, priced_reqs = [], []
    # Rows the loader could not turn into a Request at all. Counted with the
    # other exclusions because it is the same kind of fact: the report describes
    # less traffic than the invoice does.
    skipped_rows = int(getattr(ts, "skipped_rows", 0) or 0)
    # And rows that became a Request but carry no usage at all. `analyze` works
    # over `ts.analysable`, which drops them before anything here can see them,
    # so they were invisible to every count below while still being real
    # requests that really cost money. Measured: a two-request trace where one
    # row had no usage reconciled its priced half against the invoice exactly,
    # passed the gate at 0.0%, and released the figure -- coverage said 1/2 on
    # the line above it.
    #
    # Failed calls are deliberately not counted here. A non-200 row is excluded
    # for a different reason: it did not populate a cache entry, and the whole
    # analysis is about entries. Unknown spend on a *successful* request is the
    # thing an invoice would include and this report would not.
    blind_rows = int((ts.coverage.get("excluded") or {}).get("no usage fields", 0))
    # A failed call that still billed input is unaccounted spend.
    #
    # Non-200 rows are excluded from the analysis on purpose -- they populated no
    # cache entry, and every finding here is about entries. But `analyze` starts
    # from `ts.analysable`, so their *cost* left with them: a trace of one 200 and
    # one 500, both billing a million input tokens, released $5.00 as a draft over
    # 50% coverage. Partial-failure cost is exactly where spend hides.
    #
    # Only the ones carrying positive usage count. A 500 that billed nothing cost
    # nothing, and blocking on it would withhold every report containing a single
    # transport error -- the over-block this project has already shipped twice.
    failed_billed = sum(
        1 for r in ts.requests
        if r.status != 200 and isinstance(r.usage, dict) and _billed_input(r.usage) > 0)
    undated = 0
    for u, r in zip(usages, priced):
        if not _priceable_date(r):
            undated += 1
            continue
        try:
            s = cost.price(u, r.model, target_id=r.target_id, on_date=_date_for(r),
                           effective_rate=effective_rate)
        except registry.UnpriceableSurface:
            # Counted apart from a missing model row: same exclusion, different
            # remedy. Folding it in reported "no pricing recorded for
            # claude-haiku-4-5" at a Bedrock client whose model *is* priced.
            unpriceable_surfaces[r.target_id] += 1
            continue
        except registry.RegistryError:
            # A surface the registry has never heard of is a surface problem,
            # and saying otherwise sends the reader after the wrong thing. With
            # `--effective-rate`, `require_priceable` is skipped and the failure
            # instead comes out of `multipliers`, which raises the generic error
            # -- so an unattributed gateway trace was reported as "no pricing is
            # recorded for claude-opus-5", a model that is priced. That is the
            # exact confusion the branch above was split out to prevent, and it
            # became the common path the moment an unnamed row stopped being
            # answered with anthropic/direct.
            if r.target_id not in registry.target_ids():
                unpriceable_surfaces[r.target_id] += 1
            else:
                unpriceable_models[r.model] += 1
            continue
        spend_total += s.usd
        uncached_total += s.hypothetical_uncached_usd
        priced_usages.append(u)
        priced_reqs.append(r)
    # `ratios` stays the observed one, computed above from every row whose usage
    # counters parse. It used to be recomputed here over the priced subset, and
    # that threw away measurements pricing has no bearing on: `input_from_cache`
    # and `prefix_efficiency` are ratios of the provider's own counters and
    # depend on no rate table at all.
    #
    # Measured on a 40-request Bedrock trace with full usage counters -- a
    # surface the cloud provider invoices, so the recorded Anthropic rates do
    # not apply and every row fails pricing: `requests` reported 0,
    # `input_from_cache` and `prefix_efficiency` both None, and no findings,
    # under a coverage line reading "40 of 40 requests analysable". The report
    # said it had read everything and then said nothing about any of it.
    priced = priced_reqs

    # Each rule gets the population its own evidence can support, rather than
    # one list for all of them.
    #
    # A rule marked `@_rate_free` reads counters and identity and never asks for
    # a price, so a surface with no recorded rate takes nothing away from it:
    # REB-1 counting prefix rebuilds, REB-0 saying which rows it could not group,
    # SPL-1 naming a session that switched model, MIN-1 finding a marker under
    # the provider's minimum. Those run over every analysable request. The
    # pricing rules keep the priced subset, which is what stops a dollar figure
    # being derived from a row spend excluded.
    #
    # Running everything over the wide set was tried first and is wrong: on the
    # Bedrock trace `rate_for` returns 0.0 for a surface it cannot price, and
    # `cost.price` refuses a zero rate with a `ValueError` that EFF-1 does not
    # catch, so the whole analysis raised. Sixteen tests said so.
    #
    # The default for an unmarked rule is the narrow set, so a rule added later
    # keeps today's behaviour rather than silently widening. The marking is
    # verified at runtime by `TestTheRateFreeMarkingIsAccurate`, which spies on
    # `rate_for` and `cost.price` per rule -- a comment claiming a function does
    # not price is exactly the kind of claim this suite exists to stop trusting.
    findings = [f for f in
                (rule(reqs if getattr(rule, "rate_free", False) else priced,
                      ratios, window, rate_for) for rule in RULES) if f]
    findings.sort(key=lambda f: (
        {"high": 0, "medium": 1, "low": 2}[f.severity],
        -(f.avoidable_usd_month.raw() if f.avoidable_usd_month else 0)))

    recon = None
    if invoice_usd is not None:
        # `delta` is an absolute value and the divisor was signed, so a negative
        # invoice turned every mismatch into a negative percentage -- and the gate
        # below asks `pct <= 5.0`. Measured: an invoice of -$60 against $60 of
        # computed spend produced -200.0%, passed the ship gate, and released the
        # figures. An invoice of -$1bn did the same. The one invariant this
        # project sells is that dollars stay withheld until they tie to money that
        # actually left the account, and a sign flip walked straight through it.
        #
        # Non-positive rather than just negative: a zero invoice already returned
        # None here, and it keeps doing so, but it belongs in the same sentence.
        # Credits and refunds are real things a client can hand over; they are not
        # a spend total to reconcile against, and modelling them is a separate
        # decision rather than something to infer from a minus sign.
        # Kept distinct rather than collapsed to "not positive", because the
        # diagnoses differ and the report says which one happened. Zero usually
        # means the export and the bill do not describe the same period; negative
        # means someone passed a credit.
        if not isinstance(invoice_usd, (int, float)) or isinstance(invoice_usd, bool):
            invalid_invoice = "not-a-number"
        elif invoice_usd != invoice_usd or invoice_usd in (
                float("inf"), float("-inf")):
            invalid_invoice = "not-finite"
        elif invoice_usd == 0:
            invalid_invoice = "zero"
        elif invoice_usd < 0:
            invalid_invoice = "negative"
        else:
            invalid_invoice = ""
        delta = abs(spend_total - invoice_usd) if not invalid_invoice else None
        pct = (delta / invoice_usd * 100) if not invalid_invoice else None
        # Fail closed on unprovable writes. A computed total that silently omits
        # requests is being compared against an invoice that includes them, so
        # it will read as underspend no matter how good the ingest is. Passing a
        # gate on a known-incomplete number is worse than failing one.
        recon = {
            "computed_usd": spend_total, "invoice_usd": invoice_usd,
            # None rather than a subtraction against a number that is not a spend
            # total: "$60 computed, -$60 invoiced, delta $120" reads like a
            # finding instead of a rejected input.
            "delta_usd": (None if invalid_invoice
                          else spend_total - invoice_usd),
            "delta_pct": pct,
            "invalid_invoice": invalid_invoice,
            # Every reason the total is known-incomplete counts here, not just
            # unprovable write lifetimes. render_html shows this field as the
            # visible reconciliation verdict, so a partial flag meant a report
            # could read "inside the gate, so figures follow" directly above a
            # column of withheld figures.
            # Every reason the total is known-incomplete belongs here, because
            # render_html shows within_ship_gate as the visible verdict. Adding
            # undated rows to the exclusions last round without adding them here
            # let an invoice match a partial subtotal and certify it as passed
            # while the figures beside it were withheld.
            # Rows the loader never turned into a Request count here too. An
            # invoice covers the traffic that ran, not the traffic that parsed,
            # so a lossy export whose surviving subset happens to match the
            # invoice is a coincidence over an unknown denominator. Previously
            # these existed only as free-text notes, which the gate cannot read.
            "unpriced_requests": (len(unprovable) + sum(unpriceable_models.values())
                                  + sum(unpriceable_surfaces.values())
                                  + undated + skipped_rows + blind_rows
                                  + failed_billed),
            "within_ship_gate": (pct is not None and pct <= 5.0
                                 and not unprovable and not unpriceable_models
                                 and not unpriceable_surfaces
                                 and not undated and not skipped_rows
                                 and not blind_rows and not failed_billed),
            # Which of the four conditions failed, so the withheld-reason can
            # say the true one. Reporting "0.0% against the invoice, outside
            # the +/-5% gate" when the invoice matched exactly and the real
            # blocker was an unpriceable model reads as a defect in the tool.
            "blockers": {
                "invalid_invoice": bool(invalid_invoice),
                "skipped_rows": skipped_rows,
                "no_usage": blind_rows,
                "failed_but_billed": failed_billed,
                "delta": pct is None or pct > 5.0,
                "unprovable_lifetime": len(unprovable),
                "unpriceable_model": sum(unpriceable_models.values()),
                "unpriceable_surface": sum(unpriceable_surfaces.values()),
                "undated": undated,
            },
            "gate": "±5% to publish a figure; ±1–2% before savings-share pricing is defensible",
        }
        # The two reconciliation dollars are money and go through the same gate
        # as every other figure. They did not, and it showed: on a failed
        # reconciliation the HTML report printed "$16.97" and `--format json`
        # emitted `computed_usd: 16.97244499999996` as a bare float, while
        # every other figure in the same document read "[withheld: ...]". That
        # is the number the gate had just refused to publish, in the document
        # explaining why it was refused.
        #
        # `invoice_usd` stays a plain number: it is the reader's own input, not
        # a claim this tool is making. `delta_pct` stays too -- a ratio is not a
        # spend total, and it is the whole diagnostic. Without it the reader
        # cannot tell a 3% miss from a 1,600% one, and the reason for the
        # refusal has to survive the refusal.
        _why = ("reconciliation did not pass the gate, so the computed total is "
                "not published; delta_pct states how far off it was")
        recon["computed_usd"] = money.Figure(
            spend_total, money.MEASURED, released=recon["within_ship_gate"],
            withheld_because=_why)
        if recon["delta_usd"] is not None:
            recon["delta_usd"] = money.Figure(
                recon["delta_usd"], money.MEASURED,
                released=recon["within_ship_gate"], withheld_because=_why)

    notes = list(ts.notes)
    # `Analysis.blocking_notes` is built by *filtering* `notes`, so a note an
    # adapter recorded as blocking and did not also put in `notes` was dropped
    # on the floor -- it reached neither the report nor the caveat block that
    # exists to print it before any figure. Every adapter happens to put such a
    # note in both lists today, which is a convention rather than a contract,
    # and the filter is where the convention silently becomes a requirement.
    for n in ts.blocking_notes:
        if n not in notes:
            notes.append(n)
    if unprovable:
        notes.append(
            f"{len(unprovable)} of {len(reqs)} requests recorded cache writes without a "
            f"provable 5m/1h lifetime and are {QUALIFIES_SPEND}. Anthropic "
            f"reports cache_creation_input_tokens as a single number, and a 1h write costs "
            f"2x where a 5m write costs 1.25x, so the lifetime cannot be inferred from usage "
            f"fields. Capture ttl_requested at source to price these.")
    if undated:
        notes.append(
            f"{undated} request(s) carry no usable timestamp and are excluded from every "
            f"dollar figure. Pricing is date-effective, and a row with no date would "
            f"otherwise be repriced at whatever rate applies on the day the report runs. "
            f"Supply an explicit pricing date or an effective rate to include them.")
    if unpriceable_models:
        detail = ", ".join(f"{m} ({n})" for m, n in sorted(unpriceable_models.items()))
        notes.append(
            f"No pricing is recorded for {detail}. Those requests are excluded from every "
            f"dollar figure and the totals below cover the remainder only. Add the model to "
            f"the pricing registry, or pass an effective rate, before publishing a figure.")
    if unpriceable_surfaces:
        detail = ", ".join(f"{t} ({n})" for t, n in sorted(unpriceable_surfaces.items()))
        where = sorted({
            u for t in unpriceable_surfaces
            for u in [((registry.rate_scope().get("unpriced_surfaces") or {})
                       .get(t, {}).get("official_pricing"))] if u})
        # Two different remedies, and offering the wrong one wastes the reader's
        # time. A partner surface is known and merely unpriced here, so an
        # effective rate from their bill completes it. An *unnamed* surface has
        # no recorded cache multipliers either, so a rate alone cannot price it
        # -- reads and writes would still be guesses. What unblocks that reader
        # is naming the surface.
        unnamed = sorted(t for t in unpriceable_surfaces if t == UNATTRIBUTED)
        partner = sorted(t for t in unpriceable_surfaces if t != UNATTRIBUTED)
        remedy = []
        if partner:
            remedy.append(
                "These surfaces are operated and invoiced by the cloud provider, so "
                "pricing them at Anthropic rates would produce a total the customer's "
                "bill contradicts. Pass their effective rate from that bill to include "
                "them" + (f" (published rates: {', '.join(where)})." if where else "."))
        if unnamed:
            remedy.append(
                "Rows carrying no provider metadata name no surface at all, so neither "
                "the rate nor the cache multipliers are known for them. Pass --target-id "
                "to state which surface this traffic went to; an effective rate alone "
                "cannot price them, because the read and write multipliers would still "
                "be guesses.")
        notes.append(
            f"The recorded rates are Anthropic first-party list prices and do not cover "
            f"{detail}. Those requests are {QUALIFIES_SPEND} and the totals "
            f"below cover the remainder only. " + " ".join(remedy))
    if not ts.tier.supports_counterfactual:
        notes.append("Usage-only ingest: findings are limited to what usage fields reveal. "
                     "Structural causes cannot be identified without prompt structure.")
    if ts.tier is Tier.INFERRED:
        notes.append(
            f"Structure was inferred rather than instrumented "
            f"(alignment {ts.alignment:.0%}); structural findings inherit that confidence."
            if ts.alignment is not None else
            "Structure was inferred rather than instrumented and carries no alignment score. "
            "Structural findings are unvalidated: instrument the workload, or treat them as "
            "hypotheses to confirm rather than conclusions.")

    # Release is one decision, made once, applied to everything. Every renderer
    # downstream is then safe by construction: there is no gate for it to
    # forget, because a figure that did not pass carries that fact itself.
    # No invoice is not a passed gate. It used to be: `recon is None or
    # within_ship_gate` meant simply omitting invoice_usd released every figure,
    # which is the opposite of the rule this project states publicly — a dollar
    # figure is not publishable until it ties to money that actually left the
    # account. Absence of evidence was being read as evidence.
    # The conditions that withhold figures for reasons an invoice cannot fix.
    # Named once because they are consulted twice: here, and again after the
    # draft override, where the missing invoice has already been accepted and
    # repeating it back tells the reader nothing they can act on.
    def _residual_blocker():
        if unprovable:
            return (f"{len(unprovable)} request(s) recorded cache writes of "
                    f"unprovable lifetime")
        if unpriceable_models:
            return f"no pricing recorded for {', '.join(sorted(unpriceable_models))}"
        if unpriceable_surfaces:
            # Two different reasons wear the same refusal. A partner surface is
            # unpriceable because somebody else sets the rate; an unattributed
            # one is unpriceable because nobody said what it is. Telling a
            # reader whose export lacks a provider field to "supply the rate
            # from that bill" sends them to the wrong fix -- they need to state
            # the surface, and may well be on Anthropic direct.
            named = sorted(unpriceable_surfaces)
            if named == [registry.UNATTRIBUTED]:
                return ("the surface is not stated anywhere in this export, so "
                        "no rate table can be shown to apply to it; pass "
                        "--target-id to state it")
            return (f"{', '.join(named)} is invoiced by the "
                    f"cloud provider and the recorded rates are Anthropic first-party "
                    f"list prices; supply the effective rate from that bill")
        if undated:
            return (f"{undated} request(s) carry no usable timestamp, so no "
                    f"date-effective rate applies to them")
        return ""

    if recon is None:
        gate_ok, why = False, ("no invoice was supplied, so nothing here has been "
                               "reconciled against money actually spent")
    elif not recon["within_ship_gate"]:
        # delta_pct is None when the invoice is zero, and abs(None) raised here
        # -- a zero invoice is a real thing a client can hand over, usually
        # meaning the export and the bill do not describe the same period.
        pct = recon.get("delta_pct")
        blockers = recon.get("blockers") or {}
        excluded = [(n, label) for n, label in (
            (blockers.get("unprovable_lifetime", 0), "with cache writes of unprovable lifetime"),
            (blockers.get("unpriceable_model", 0), "on a model or surface with no recorded price"),
            (blockers.get("undated", 0), "with no usable timestamp, so no date-effective rate"),
            (blockers.get("skipped_rows", 0), "the loader could not read at all"),
            (blockers.get("no_usage", 0), "carrying no usage fields, so their cost is unknown"),
            (blockers.get("failed_but_billed", 0),
             "that failed but still billed input, so their cost is outside this total"))
            if n]
        if blockers.get("invalid_invoice"):
            # Said before the percentage cases, because `pct` is None for all of
            # these and the fallback below would otherwise report "the invoice
            # supplied is zero" for a negative one -- naming a blocker that is not
            # the real one, which is the complaint this branch already carries
            # about itself a few lines up.
            kind = recon.get("invalid_invoice")
            gate_ok, why = False, {
                "zero": ("the invoice supplied is zero, so computed spend cannot "
                         "be reconciled against it. Usually the export and the "
                         "bill do not describe the same period"),
                "negative": (
                    f"the invoice supplied is negative "
                    f"({recon.get('invoice_usd')}), so computed spend has nothing "
                    f"valid to reconcile against. A credit or refund is a real "
                    f"thing to receive, but it is not a spend total: reading one "
                    f"as the denominator turns any mismatch into a negative "
                    f"percentage, and a negative percentage passes a ≤ 5% gate"),
                "not-finite": ("the invoice supplied is not a finite number, so "
                               "computed spend cannot be reconciled against it"),
            }.get(kind, "the invoice supplied is not a number, so computed spend "
                        "cannot be reconciled against it")
        elif pct is not None and not blockers.get("delta") and excluded:
            # The percentage is fine; the subtotal it matched is incomplete.
            gate_ok, why = False, (
                "the invoice matches to " + f"{abs(pct):.1f}%" + ", but the computed "
                "subtotal excludes " + ", ".join(f"{n} request(s) {label}" for n, label in excluded)
                + ", so it is a subset that happens to agree rather than a reconciliation")
        else:
            gate_ok, why = False, (
                f"reconciliation is {abs(pct):.1f}% against the invoice, "
                f"outside the ±5% gate"
                if pct is not None else
                "the invoice supplied is zero, so computed spend cannot be "
                "reconciled against it")
    elif _residual_blocker():
        gate_ok, why = False, _residual_blocker()
    else:
        gate_ok, why = True, ""

    # The deliberate override, for internal drafts before an invoice arrives.
    # It is a named argument rather than a silent default so that publishing an
    # unreconciled number is always someone's explicit decision, and the report
    # says so where the reconciliation section would otherwise be.
    # The override covers a missing invoice and nothing else. It deliberately
    # does not cover unprovable write lifetimes: those requests are excluded
    # from the totals, so releasing anyway would publish a number the notes
    # simultaneously describe as incomplete — the same contradiction the gate
    # exists to prevent, reintroduced through its own escape hatch.
    #
    # `skipped_rows` and `blind_rows` belong in that list for exactly the same
    # reason, and were missing from it: the previous round added them to the
    # invoice gate and not to its escape hatch, so a draft run released $10.00
    # over a trace that had dropped a row or could not see one's cost. Fixing a
    # gate and not its override is the same defect as fixing a guard and not its
    # twin, which is this branch's most-repeated shape.
    # Reconciled unless the override below says otherwise. A silent default of
    # DRAFT would relabel every invoice-checked figure in the product.
    released_as = money.RECONCILED
    # Collected rather than inserted, because there are two independent reasons
    # a report can be a draft and `report._draft_reason` reads the *first* note
    # beginning with "DRAFT". Inserting a second one meant whichever landed
    # first spoke for both, and its fallback sentence -- "released without
    # invoice reconciliation" -- is plainly false on a run where an invoice was
    # supplied and reconciled. One note, composed, listing every reason.
    draft_reasons: list[str] = []
    if (not gate_ok and money.draft_override_applies(recon is not None, allow_unreconciled)
            and not unprovable and not unpriceable_models
            and not unpriceable_surfaces and not undated
            and not skipped_rows and not blind_rows and not failed_billed):
        gate_ok = True
        released_as = money.DRAFT
        draft_reasons.append(
            "figures released without invoice reconciliation. These numbers "
            "have not been tied to a provider invoice and may not survive one.")
    elif (not gate_ok
          and money.draft_override_applies(recon is not None, allow_unreconciled)
          and _residual_blocker()):
        # They passed --allow-unreconciled, so "no invoice was supplied" is a
        # condition they already accepted and answering with it sends them
        # nowhere. Say what actually defeated the override. A Bedrock client got
        # told to supply an invoice, supplied one, and was then told the same
        # thing again -- the fix was never an invoice, it was a rate.
        why = _residual_blocker()

    # An input the loader had to assume rather than read is not a measurement,
    # and a figure computed from one must not carry the provenance of an invoice
    # check. The sharpest case is the *surface*: assume it and the rate table and
    # the cache multipliers are assumptions too, so the invoice reconciles
    # against a guess. Measured before this existed: every figure in `spend`,
    # every finding's, and `total_avoidable_month` all came back
    # `released_as='reconciled'`, with the assumption surviving only as free
    # text a renderer appended afterwards.
    #
    # Read from `assumed_inputs`, not from `ts.blocking_notes`, and the first
    # version of this did read the notes. Three reasons it was wrong, the first
    # measured by Track B:
    #
    #   - It over-blocks. `blocking_notes` carries the QUALIFIES_SPEND phrase,
    #     which the litellm adapter uses to mean "these rows were excluded from
    #     the totals" -- a trace that then reconciles is *correctly* RECONCILED,
    #     because what remains does tie to the invoice. "Any blocking note means
    #     draft" relabels reports that were right.
    #   - It decides a release label by matching prose, which is the failure
    #     `trace.py` records as the reason QUALIFIES_SPEND stopped being the
    #     classifier in the first place.
    #   - The fact is per-input, not per-report. An assumed surface and an
    #     assumed effective rate are different assumptions with different
    #     remedies, and a sentence cannot say which one happened.
    #
    # `getattr` with a default because the field belongs to `trace.py`, which is
    # not this track's to edit; a loader that does not set it is a loader that
    # assumed nothing, which is the safe reading either way.
    #
    # DRAFT rather than withheld, and the distinction is the point. The invoice
    # did reconcile; what is assumed is what it reconciled *against*, so the
    # evidence is weaker than it looks rather than absent. DRAFT already means
    # exactly that -- released, not for forwarding -- and carries a banner both
    # renderers print before any figure.
    #
    # Not a third release state. `ASSUMED` would be the precise word and would
    # mean changing `money.RELEASES`, `simulate.py` and the round-trip table in
    # `test_invariants.py`, two of which are outside this track. Measured and
    # reported rather than half-built.
    #
    # `gate_ok` is deliberately not touched. If the invoice fails, everything is
    # withheld and the label is moot; flipping it here would make the two
    # renderers disagree about a report neither should be publishing.
    assumed = tuple(getattr(ts, "assumed_inputs", ()) or ())
    if gate_ok and assumed:
        released_as = money.DRAFT
        draft_reasons.append(
            "the " + ", ".join(sorted(assumed)) + " used to price this trace "
            "was assumed rather than read from the export"
            + (". An invoice was supplied and reconciles, but it reconciles "
               "against that assumption, so these figures are not "
               "invoice-checked in the sense the word usually carries."
               if recon is not None else "."))
    if draft_reasons:
        notes.insert(0, "DRAFT — " + " ".join(draft_reasons)
                     + " Not for external use.")

    # The two reconciliation dollars are built roughly 150 lines above this,
    # where the reconciliation dict is assembled and the final provenance is not
    # known yet -- so they defaulted to RECONCILED and stayed there while every
    # other figure was relabelled. Measured on an assumed-surface trace: a
    # `spend` map entirely marked `draft` sitting beside
    # `reconciliation.computed_usd` and `delta_usd` still marked `reconciled`,
    # in one document, describing the same dollars.
    #
    # Re-released rather than moved. Building them down here would work and
    # would separate the two reconciliation figures from the dict that explains
    # them, which is where a reader and a maintainer both look for them.
    #
    # Only figures that are already released are touched: a withheld one carries
    # the reason the gate refused, and `release` would overwrite it -- the same
    # rule `_withhold_projection` follows a few lines down, for the same reason.
    if recon is not None:
        for _k in ("computed_usd", "delta_usd"):
            _fig = recon.get(_k)
            if isinstance(_fig, money.Figure) and _fig.released:
                recon[_k] = _fig.release(True, as_=released_as)

    # Structural claims carry a second gate. The first asks whether the money
    # ties to an invoice; this one asks whether the segmentation the claim rests
    # on was ever checked. An inferred trace with no alignment score published
    # released VOL-1 dollars at confidence "high" directly above a note saying
    # structural findings were unvalidated -- and VOL-1 is the finding that
    # tells someone to reorder prompt authority.
    # Coverage is part of trust, not a separate note. A mixed file whose
    # present ids are all trusted still classifies as instrumented, but the
    # requests carrying no segments cannot support a VOL/FAN/MIN claim -- so a
    # structural recommendation could be costed from a subset while the missing
    # rows were exactly the ones that would have changed its cause or scope.
    # structural_coverage has existed since the loader learned to measure it and
    # was never wired to the gate that needs it.
    covered = getattr(ts, "structural_coverage", 1.0)
    # Rows for the note, billed tokens for the money. Both must clear the floor:
    # the row figure says the observed *cause* is representative, the billed
    # figure says the spend the recommendation is costed against is the spend
    # structure actually saw. A trace can pass either one alone and still put
    # out a figure describing a fraction of the bill.
    covered_billed = getattr(ts, "structural_coverage_billed", 1.0)
    aligned = (ts.tier is Tier.INSTRUMENTED
               or (ts.tier is Tier.INFERRED and ts.alignment is not None
                   and ts.alignment >= ALIGNMENT_FLOOR))
    # Segment sizes have to agree with what the provider billed before any
    # structural claim carries money. Spend comes from `usage` and structural
    # figures come from segment `tokens`, and nothing compared the two -- so an
    # invoice reconciled against one half while a figure was published from the
    # other. Measured: a trace billing twelve cents released $78,660,000 a month
    # with the gate reporting a pass.
    # The publishable signal, not the coarse one. Both exist because they answer
    # different questions, and this line guards *money*: a structural figure is
    # computed from segment sizes, so sizes within a factor of two of billed --
    # which the coarse check accepts -- can still put a figure out at double.
    # Reading the coarse flag here was the same defect as the bake-off's, one
    # module along.
    sums_ok = getattr(ts, "token_sums_publishable", True)
    # Segment sizes have to be counted, not guessed, before they are money.
    # `_scale_to_measured` divides the billed total by byte share, which
    # measured at 19.2% median error per segment and 181% worst against the
    # provider's own tokenizer. This package refuses to publish spend that
    # reconciles worse than 5%; costing a recommendation from a 19% split while
    # holding the invoice to 5% is two standards, not one.
    tokens_counted = getattr(ts, "tokens_are_counted", False)
    structure_trusted = (aligned and covered >= ALIGNMENT_FLOOR
                         and covered_billed >= ALIGNMENT_FLOOR and sums_ok
                         and tokens_counted)
    struct_why = ""
    if not sums_ok:
        struct_why = (
            "segment token counts do not sum to the tokens the provider billed, so "
            "the sizes every structural figure is computed from disagree with the "
            "usage that spend and the invoice are reconciled against")
    elif not aligned:
        struct_why = (
            f"segmentation is {'unmeasured' if ts.alignment is None else f'{ts.alignment:.0%}'}"
            f" against instrumented ground truth, below the {ALIGNMENT_FLOOR:.0%} floor"
            f" required to attach money to a structural claim")
    elif covered < ALIGNMENT_FLOOR:
        struct_why = (
            f"only {covered:.0%} of requests carry prompt structure, below the "
            f"{ALIGNMENT_FLOOR:.0%} floor. The requests without it cannot support a "
            f"structural claim, and they may be the ones that would change it")
    elif not tokens_counted:
        struct_why = (
            "segment token counts are estimated rather than counted: the billed "
            "total is divided between segments in proportion to their bytes, which "
            "measures 19.2% off at the median and 181% at worst against the "
            "provider's own tokenizer, because dense JSON tool schemas run about "
            "2.74 bytes per token where prose runs 5.22. Every figure here would be "
            "costed from that split, and this report will not publish spend that "
            "reconciles worse than 5%. Run tier-b/count_tokens.py over the export "
            "and the same figures become measurements")
    elif not structure_trusted:
        # Rows look fine and the money does not. This is the branch a row count
        # cannot reach, and the one worth spelling out: the reader is holding a
        # reconciled invoice and a coverage figure that both look healthy.
        struct_why = (
            f"requests carrying prompt structure account for only "
            f"{covered_billed:.0%} of the billed input tokens, below the "
            f"{ALIGNMENT_FLOOR:.0%} floor, even though {covered:.0%} of requests carry "
            f"it. The invoice reconciles the total spend, but it cannot show whether "
            f"the requests structure was recorded for are the ones the spend came "
            f"from -- and here they are not")

    # CLASS, closed rather than narrowed: `_monthly` has seven callers, six of
    # them per-finding, and every one of them is now gated here. It used to gate
    # the spend total alone -- measured on a two-request, one-minute trace with
    # a matching invoice, `findings[0].avoidable_usd_month` published $180/mo
    # marked `reconciled` directly under a `monthly_input_usd` withheld for
    # being an unsupportable extrapolation. Two figures, one extrapolation, and
    # the more client-facing of them was the ungated one.
    #
    # The reason it stayed open was real and is now answered rather than
    # deferred: extending the gate withheld money in existing tests whose
    # fixtures run 9 to 18 minutes, and widening those windows was not an option
    # because the request gaps are the quantity several of those rules measure.
    # None of them was about projections -- each needed *some* released figure to
    # exist so it could assert a different gate -- so what they needed was a
    # figure the observed window supports. That is `avoidable_usd_window`, and
    # they assert it now.
    #
    # This call answers for the *spend total* only. Each finding is judged
    # against its own contributing sample, recorded by `_avoidable` as
    # `Finding.projection_why`, because this one is computed from the whole
    # trace: applied to a finding it let unrelated traffic vouch for a
    # projection it contributed nothing to. Measured on a ten-request, 2.3-day
    # trace whose only two cache writers went out one second apart -- EFF-1
    # published $6.43/mo and FAN-1 $14.79/mo, both `reconciled`, on a
    # two-request sample, with `total_avoidable_month` at $21.21.
    projection_ok, projection_why = _projection_supported(window, len(reqs))
    if not projection_ok:
        notes.append(projection_why[0].upper() + projection_why[1:])

    released = []
    for f in findings:
        if f.structural and not structure_trusted:
            released.append(replace(f.released(False, struct_why, as_=released_as),
                                    confidence="low", severity="medium"))
        else:
            released.append(f.released(gate_ok, why, as_=released_as))
    # Only the extrapolation loses its release; the measured window keeps it,
    # exactly as `input_usd` does beside `monthly_input_usd` below. Applied
    # after the release loop rather than inside it, because `Finding.released`
    # releases every Figure it finds and would undo a refusal applied earlier.
    #
    # `f.projection_why` and not the trace-wide `projection_why`: a finding's
    # sample is drawn from the trace, so its window and count can only be
    # smaller, and the per-finding verdict therefore subsumes the trace-wide one
    # rather than merely adding to it. Where the whole trace fails the floor,
    # every finding fails it too -- and reports its own numbers in the reason
    # instead of the trace's, which is the difference between "2 requests" and
    # "10 requests" in a sentence a client reads.
    #
    # `Analysis.total_avoidable_month` needs no line of its own: it reads its
    # release off its parts, so withholding them withholds it. Verified rather
    # than assumed -- INV-1 walks the analysis and would find it either way.
    released = [replace(f, avoidable_usd_month=_withhold_projection(
        f.avoidable_usd_month, f.projection_why)) if f.projection_why else f
        for f in released]
    findings = released
    if not structure_trusted and any(f.structural for f in findings):
        notes.append(
            f"Structural findings below are reported without dollar figures. They rest on "
            f"segment boundaries inferred from logged bodies, and {struct_why}. Score the "
            f"segmenter against an instrumented capture, or instrument the workload, before "
            f"acting on them as costed recommendations.")
    spend = money.release_map({
        "input_usd": money.Figure(spend_total, money.MEASURED),
        "if_uncached_usd": money.Figure(uncached_total, money.MODELED),
        "caching_saved_usd": money.Figure(uncached_total - spend_total, money.MEASURED),
        "monthly_input_usd": _monthly(spend_total, window),
        "window_days": window,
    }, gate_ok, why, as_=released_as)
    # The measured window keeps its release; only the extrapolation loses it.
    # An invoice does establish that `input_usd` matches the bill.
    if not projection_ok and isinstance(spend.get("monthly_input_usd"), money.Figure):
        spend["monthly_input_usd"] = _withhold_projection(
            spend["monthly_input_usd"], projection_why)

    # Classified once, here, where every note has been collected and the code
    # still knows why each was raised. Renderers read the field; none of them
    # searches the prose. `ts.blocking_notes` carries anything an ingest adapter
    # already marked, because the adapter knows things the analyzer cannot see
    # -- that a row stated no surface, for one.
    blocking = [n for n in notes
                if _note_blocks_spend(n) or n in set(ts.blocking_notes)]
    return Analysis(ratios=ratios, coverage=ts.coverage, tier=ts.tier,
                    findings=findings, spend=spend, reconciliation=recon,
                    window_days=window, notes=notes, blocking_notes=blocking,
                    tokens_counted=bool(tokens_counted))
