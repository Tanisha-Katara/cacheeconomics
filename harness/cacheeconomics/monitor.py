"""Live cache diagnostics: what is going wrong now, not what went wrong.

The analyzer answers a question about a window that has already closed. That is
the right shape for a paid assessment and the wrong shape for the thing a team
actually wants afterwards, which is to know the day a prefix starts drifting
rather than the month.

Four differences from the batch rules, each of which constrains the design:

**Bounded memory, including the scope map.** This runs inside a request path,
possibly for weeks. Every structure here is fixed-size *and so is the number of
structures* -- an earlier version of this file bounded each scope's history and
then kept one scope per tenant forever, which on a multi-tenant gateway is the
same leak with extra steps. Scopes evict least-recently-seen at `max_scopes`.
An evicted scope loses its history and its alert state, so the next thing it
does wrong is reported again from scratch. That is the honest trade: a fixed
ceiling costs repeated alerts on a very wide fleet, and nothing here pretends
otherwise.

**No dollar figures.** There is no invoice at runtime, and this project's
standing rule is that a number nobody has reconciled does not get published.
Alerts describe what changed and what it costs *structurally* -- a prefix
invalidated, a marker below the minimum -- and leave money to the report.

**Alerts repeat, so they must not spam.** A drifting prefix drifts on every
request. Each alert fires once per *subject* until cleared, where the subject is
the specific thing that is wrong: this segment, this prefix, this direction. Two
segments drifting are two alerts; the same segment drifting for a week is one.

**It sees one request at a time.** Nothing here can sort, take a median over a
whole trace, or look ahead. Where the batch rules do that, the runtime version
uses a rolling window and says so.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime

from . import cost, registry
from .allocate import pool_of, reuse_chain_of
from .trace import (WRITE_VISIBLE_FALLBACK_SECONDS, marked_prefixes,
                    write_tokens, write_visible_at)

# How much history a single scope keeps. Small enough that a thousand scopes
# stay cheap, large enough that a rolling median over request gaps is not noise.
WINDOW = 64

# How many cache pools to track at once, evicting least-recently-seen.
MAX_SCOPES = 1024

# Distinct live alerts remembered per scope, so the dedup table cannot grow
# without limit on a workload that keeps inventing new prefixes to fan out.
MAX_FIRING = 64

# Fallback concurrency window for fan-out, used only where the trace carries no
# `first_token_at`. One definition, in trace.py, because FAN-1 needs the same
# number and the two spellings drifted apart in what they meant.
FANOUT_SECONDS = WRITE_VISIBLE_FALLBACK_SECONDS

TTL_SECONDS = {"5m": 300, "1h": 3600}

# A turn that rewrites at least this share of the prefix it had established is
# rebuilding it, not extending it. Shared with the batch rule so the runtime and
# the report cannot disagree about what a rebuild is.
REBUILD_FRACTION = 0.5

# Rebuilding more often than this is worth interrupting somebody about.
REBUILD_TURNS = 20

# How many wire positions a scope tracks. A long conversation can carry
# thousands of message blocks, and one bounded deque each is still unbounded in
# the number of deques. Positions past this are not tracked, so the checks that
# need history stay silent about them rather than guessing -- the cacheable
# prefix lives at the front, which is what these positions are for.
MAX_POSITIONS = 512

# A position that was not present on this request. Distinct from every hash, and
# deliberately not "unchanged": a section appearing and disappearing moves
# everything behind it exactly as an edit does. `observed_change_rates_by_pool`
# has counted absence as a change since it was written; the runtime did not, so
# an optional block that appeared every other request read as perfectly stable
# and the live allocator marked a prefix that could not survive to be read.
_ABSENT = None


@dataclass(frozen=True)
class Alert:
    """One thing worth telling somebody about, now."""
    code: str
    severity: str            # high | medium | low
    scope: tuple
    summary: str
    detail: str
    subject: str = ""        # what specifically is wrong, for dedup
    at: datetime | None = None
    fix: str = ""

    def __str__(self):
        # Scopes are not all three-wide. Rebuild tracking deliberately keys on
        # `(tenant, target_id)` and drops the model, because a rebuild is a
        # property of the prefix rather than of the pool -- so unpacking three
        # names raised ValueError on exactly the alerts that report a live cache
        # failure, the moment anything logged or printed one. Rendering must not
        # be able to fail on a shape the emitter is allowed to produce.
        who = " · ".join(str(x) for x in self.scope[:3] if x)
        return (f"[{self.severity.upper()}] {self.code} {self.summary}\n"
                f"    {who}\n    {self.detail}"
                + (f"\n    -> {self.fix}" if self.fix else ""))


@dataclass
class _ScopeState:
    """Rolling state for one cache pool. Every field is bounded."""
    # One digest per request per position, not one per change. Counting
    # *distinct* values missed the worst case there is: a field alternating
    # between two states changes the prefix on every single request and only
    # ever shows two values.
    seg_values: dict = field(default_factory=lambda: defaultdict(
        lambda: deque(maxlen=WINDOW)))
    gaps: deque = field(default_factory=lambda: deque(maxlen=WINDOW))
    last_sent: datetime | None = None
    recent_writes: deque = field(default_factory=lambda: deque(maxlen=WINDOW))
    concurrent_writers: int = 0
    # Whether the fan-out boundary came from an observed `first_token_at` or
    # fell back to the flat guess. The alert says which, rather than implying
    # every trace was measured the same way.
    fanout_observed: bool = False
    prefix_key: int = 0
    seen: int = 0
    # Per session key, not per scope. One timestamp for the whole pool meant
    # interleaved sessions read each other's timing: four sessions whose entries
    # had each expired after fifteen minutes reported "rebuilt about every 1
    # turns", because the previous request in the *scope* was seconds old.
    last_seen: OrderedDict = field(default_factory=OrderedDict)
    # The lifetime the *previous* write used, per session. The expiry check
    # asks whether that entry was still alive, and reading the current
    # request's TTL answered a different question: a 5m write followed ten
    # minutes later by a 1h one was measured against 1h and filed as a rebuild.
    last_lifetime: OrderedDict = field(default_factory=OrderedDict)
    # session -> tokens the prefix had reached, so a rebuild can be told from an
    # extension. Bounded: a long-lived gateway sees unboundedly many sessions.
    established: OrderedDict = field(default_factory=OrderedDict)
    # prefix_key -> when that span was last marked, and the gaps between two
    # markings of the same span. TTL-1 walks a per-prefix timeline and credits
    # only rewrites of a span that already existed; RT-TTL read the scope's
    # median request gap alone, so a workload marking a different prefix every
    # request -- where a longer lifetime recovers nothing, because there is no
    # second write of anything to turn into a read -- got told to switch to one
    # hour while the report refused to make that claim from the same trace.
    #
    # The value is `(when, lifetime)`, and the lifetime half is load-bearing.
    # Gaps are partitioned by the lifetime in force, because RT-TTL's question
    # is "at this cadence, would a different lifetime be cheaper" and the
    # answer is only supportable from rewrites that actually ran under the
    # lifetime being argued about. Pooling them let a run of 30m traffic --
    # which RT-TTL cannot evaluate at all -- fill the window, and then a single
    # 5m request fired the alert off ten gaps of 30m evidence, reporting a
    # median it had no 5m observation of. Gating the *wording* on the lifetime
    # and leaving the *state* pooled is the worse half of that bug: the
    # original was silence, this was a live alert computed from evidence the
    # rule says it cannot read.
    #
    # Partitioned by lifetime, like the gaps it feeds. One shared map meant a
    # lifetime the rule refuses could overwrite a span's timestamp and destroy
    # the evidence for one it can evaluate -- the mirror image of the pooled
    # window that let it invent evidence.
    last_marked_at: dict = field(default_factory=lambda: {
        name: OrderedDict() for name in TTL_SECONDS})
    # One bucket per lifetime RT-TTL can evaluate, created once and never
    # added to -- `_ttl_rt_ttl_can_read` returns a member of TTL_SECONDS or
    # None, so no other key can be written and the map cannot grow.
    rewrite_gaps: dict = field(default_factory=lambda: {
        name: deque(maxlen=WINDOW) for name in TTL_SECONDS})
    rebuilds: deque = field(default_factory=lambda: deque(maxlen=WINDOW))
    firing: OrderedDict = field(default_factory=OrderedDict)


# Why a lookup went unanswered, and what the operator does about it.
#
# These are flags, not alternatives, and that is the whole point. A row can be
# contested *and* not record the key, and the shipped `openai/bedrock` is
# exactly that: flagged contested, with `capabilities: {"_unknown": true}`, so
# `max_breakpoints` and `lookback_blocks` are absent as well.
#
# Both halves have been got wrong here in turn. `ContestedRow` subclasses
# `RegistryError`, so one `except registry.RegistryError` reported a disputed
# row as missing data and sent the reader to add a value already on file --
# against this project's rule that a contested row is never treated as fact.
# Catching `ContestedRow` first then reported an absent key as present-but-
# disputed and emitted only the settle remedy, so the one shipped surface that
# genuinely presents both causes was told about one of them. Settling a row and
# recording a value are different jobs in different places; a surface needing
# both has to be told both.
_CONTESTED = "contested"
_NOT_RECORDED = "absent"
_RECORDED_NULL = "null"
_UNINSPECTED = "uninspected"

# Ordered so a cause reads row-level fact first, then the key-level one.
_CAUSE_TEXT = (
    (_CONTESTED, "the row is flagged contested, so nothing on it may be "
                 "treated as fact"),
    (_NOT_RECORDED, "not recorded for this surface"),
    (_RECORDED_NULL, "recorded as null"),
    # Not a cause so much as the limit of what was checked, said out loud
    # rather than resolved by guessing in either direction.
    (_UNINSPECTED, "whether it is recorded at all was not inspected, because "
                   "this lookup has no contested-inspecting form"),
)

_RECORD_IT = ("record the missing limits in the registry with a dated source, "
              "or name a surface on the stream that has them")
_REMEDIES = {
    _CONTESTED: "settle the contested row against a dated source, or leave it "
                "contested and accept that these checks stay off",
    _NOT_RECORDED: _RECORD_IT,
    _RECORDED_NULL: _RECORD_IT,
}


class _RegistryReads:
    """Every registry lookup one request's checks made, and which of them the
    registry could not answer.

    The seam exists so that the announcement follows the *actual* dependency
    instead of a hardcoded probe. RT-NOSURFACE used to test
    `min_cacheable_tokens` and nothing else, so a surface that recorded a
    minimum and no breakpoint budget disabled the marker-budget check in
    silence -- and `lookback_blocks`, which nobody had thought to list at all,
    did the same to RT-TTL's rewrite timeline. A check that reads a new key
    through `get` is announced from the moment it is written; one that calls
    `registry` directly is not, and INV-5 in the test suite is what catches
    that, by watching which keys the module actually reads.

    Ways a lookup goes unanswered, all reaching the same silent state: the
    surface records no such key, the surface is contested, the key is recorded
    as `None` -- and any combination of those, which is why the cause is a set
    of flags rather than one label. A recorded *zero* is an answer --
    `deepseek/direct` allows no explicit breakpoints, which is a fact about the
    surface and not a gap in the registry -- so `is None` is the test, never
    falsiness.

    Bounded by construction: one entry per literal key named in this file, a
    fixed set of cause flags, and one `needed_for` string per call site. All
    fixed at import time.
    """
    __slots__ = ("unanswered",)

    def __init__(self):
        self.unanswered: dict = {}

    def get(self, key, fn, *, needed_for, inspect=None):
        """Read `key` from the registry, recording it when the answer is not
        there. Returns None in that case, and the caller decides what to do --
        abstain, or carry on unbounded -- because that differs per check.

        `inspect` is the same lookup in its contested-inspecting form, where
        the registry offers one. It is called only when the row turns out to
        be contested, and only to find out whether the key is *also* absent --
        never to obtain a value the checks then use. `allow_contested=True` is
        the registry's own affordance for looking without publishing, and this
        is the looking.

        `needed_for` may be None, meaning this request reads the value but no
        check here can act on its absence. The value is still read and still
        used; it is simply not announced, because an alert naming a cause that
        is not the reason the operator is getting no answer is an alert that
        fires on ordinary traffic -- and one that fires on ordinary traffic
        gets switched off, which ends in the same silence this class exists to
        break.
        """
        try:
            value = fn()
        except registry.ContestedRow:
            # Before the base class, which it subclasses. Reversing these two
            # clauses is the defect, not a style choice.
            causes = {_CONTESTED} | self._also(inspect)
        except registry.RegistryError:
            causes = {_NOT_RECORDED}
        else:
            if value is not None:
                return value
            causes = {_RECORDED_NULL}
        if needed_for:
            entry = self.unanswered.setdefault(key, (set(), set()))
            entry[0].update(causes)
            entry[1].add(needed_for)
        return None

    @staticmethod
    def _also(inspect):
        """What a contested row holds, looked at without publishing from it.

        Returning the empty set means the key is there and the dispute is the
        only problem. Where the registry exposes no contested-inspecting form
        -- `min_cacheable_tokens` takes no `allow_contested` -- this says so
        rather than guessing, because guessing "present" understates a row that
        needs values recorded and guessing "absent" invents a gap. Reproducing
        the registry's own inheritance walk here to answer it anyway would be a
        second copy of that logic, which is how the two drift.
        """
        if inspect is None:
            return {_UNINSPECTED}
        try:
            return set() if inspect() is not None else {_RECORDED_NULL}
        except registry.RegistryError:
            return {_NOT_RECORDED}


class Monitor:
    """Feed it requests as they happen; it hands back alerts.

    Deliberately accepts the same `Request` the recorder and loaders produce, so
    a team can run it over a live stream or replay it against a captured trace
    and get identical answers. A monitor that disagrees with the report is worse
    than no monitor.
    """

    def __init__(self, *, drift_rate: float = 0.2, min_samples: int = 8,
                 max_scopes: int = MAX_SCOPES):
        # Share of requests on which a position inside the prefix changes before
        # it counts as drift. A rate, not a count of distinct values: a boolean
        # that flips every request only ever has two values and is the most
        # expensive thing that can sit in a cached prefix.
        self.drift_rate = drift_rate
        # Drift, cadence and fan-out all need history before they mean anything.
        self.min_samples = min_samples
        self.max_scopes = max_scopes
        self._scopes: OrderedDict[tuple, _ScopeState] = OrderedDict()

    # --- public ----------------------------------------------------------

    def observe(self, request) -> list[Alert]:
        """Record one request, shape and usage together.

        The right entry point when both arrive at once: a replayed trace, or a
        gateway that hands over the finished exchange.
        """
        return self._observe(request, shape=True, usage=True)

    def observe_shape(self, request) -> list[Alert]:
        """Record what was sent, before the response exists.

        Split out for the request path, where the prompt is known and the usage
        counters are not. Pairs with `observe_usage`; calling both is equivalent
        to one `observe`, because the two track disjoint state.
        """
        return self._observe(request, shape=True, usage=False)

    def observe_usage(self, request) -> list[Alert]:
        """Record what the provider reported, once the response is back.

        Needs the segments too -- fan-out is keyed on which prefix was written,
        and reading that from scope state instead would race against every other
        request in flight. The caller kept them; this asks for them back.
        """
        return self._observe(request, shape=False, usage=True)

    def _observe(self, request, *, shape: bool, usage: bool) -> list[Alert]:
        pool_scope = pool_of(request)
        shape_scope = reuse_chain_of(request)
        pool_st = self._scope(pool_scope)
        # Rebuild counting deliberately drops the model from its scope.
        #
        # `pool_of` includes it, because a different model is a different cache
        # pool at the provider -- correct for every other usage check. But the
        # question "did this session pay to write its prefix again" spans models
        # by definition: switching model is one of the ways a prefix gets
        # rebuilt, and the provider charges for it. Keyed by pool, the switch
        # landed in a fresh scope with empty history and the rebuild vanished.
        #
        # Measured: a 60-turn session alternating model every 10 turns produced
        # REB-1 from the analyzer and silence from the monitor -- while
        # RT-REBUILD's own fix text names "a model switch" as a cause it could
        # never surface. The batch rule has always compared consecutive requests
        # in a session regardless of model, and counts the switches separately.
        rebuild_scope = (request.tenant, request.target_id)
        rebuild_st = self._scope(rebuild_scope)
        shape_st = self._scope(shape_scope) if shape_scope != pool_scope else pool_st
        alerts: list[Alert] = []
        # One per request, drained at the bottom. Every registry lookup any
        # check makes goes through it, so what gets announced is whatever was
        # actually asked for rather than a list somebody maintained by hand.
        reads = _RegistryReads()

        if shape:
            self._track_shape(shape_st, request, reads)
        if usage:
            self._track_usage(pool_st, request)
        shape_checks = []
        usage_checks = []
        if shape:
            shape_checks += [self._drift, self._blocked, self._below_minimum,
                             self._marker_budget, self._cadence_vs_ttl]
        rebuild_checks = []
        if usage:
            usage_checks += [self._cold_fanout]
            rebuild_checks += [self._rebuild]
        if shape and not request.segments:
            # Say so once. Every structural check below needs prompt structure,
            # and on a usage-only stream they all abstain — which produces a
            # silence indistinguishable from a clean bill of health. The batch
            # report states its coverage for exactly this reason.
            blind = Alert(
                "RT-BLIND", "low", shape_scope,
                "no prompt structure on this stream, so most checks are inactive",
                "Drift, prefix-minimum, marker-budget and cold-fan-out all need to know "
                "how the prompt was assembled. This source carries usage counters only, "
                "so their silence means unmeasured, not healthy. RT-REBUILD still "
                "applies -- it needs only the usage counters -- so a prefix being "
                "rebuilt will still be reported.",
                subject="blind", at=request.sent_at,
                fix="Capture through the recorder, or export request bodies, to turn "
                    "the structural checks on.")
            self._fire(shape_st, blind, alerts)
        for st, scope, checks in (
            (shape_st, shape_scope, shape_checks),
            (pool_st, pool_scope, usage_checks),
            (rebuild_st, rebuild_scope, rebuild_checks),
        ):
            for check in checks:
                # `reads` is handed to every check, including the four that do
                # not currently touch the registry. A check that grows a lookup
                # later gets the announcing seam without anyone editing this
                # loop, which is the whole point of the change.
                for a in check(st, request, scope, reads) or ():
                    self._fire(st, a, alerts)
        # The surface equivalent of RT-BLIND, and for the same reason: several
        # checks read the registry for this target and abstain when it cannot
        # answer. Measured: a 200-token marker below the 512-token minimum
        # raises RT-MIN on `anthropic/direct` and *nothing at all* on an
        # unnamed surface. The operator sees a quiet dashboard and reads it as
        # healthy.
        #
        # Emitted after the checks, not before, because the checks are what
        # establishes which lookups this request actually needed. Said once per
        # scope through the same `firing` table as every other alert rather
        # than a second dedup mechanism -- keyed on the *set* of unanswered
        # lookups, so fixing one of them and leaving another still gets
        # reported instead of being swallowed as a repeat.
        if reads.unanswered:
            self._fire(shape_st,
                       self._nosurface(request, shape_scope, reads), alerts)
        return alerts

    @staticmethod
    def _nosurface(r, scope, reads: _RegistryReads) -> Alert:
        """One alert naming every registry lookup this request could not get.

        Composed from what was asked, so the text cannot drift out of date the
        way the hardcoded version did: it named three checks and omitted
        RT-TTL, whose rewrite timeline silently loses its lookback bound when
        `lookback_blocks` is missing and over-credits as a result.
        """
        missing = sorted(reads.unanswered)
        parts = []
        for key, (causes, users) in sorted(reads.unanswered.items()):
            # Ordered by `_CAUSE_TEXT`, not by set iteration, so the same pair
            # of causes always reads the same way round.
            why = ", ".join(text for flag, text in _CAUSE_TEXT
                            if flag in causes)
            parts.append(f"{key} ({why}) -- needed by "
                         f"{', '.join(sorted(users))}")
        detail = "; ".join(parts)
        what = (f"the {missing[0]} lookup" if len(missing) == 1
                else f"{len(missing)} of the registry lookups")
        # The illustration is attached to the dependency it illustrates. The
        # hardcoded version carried it unconditionally, so a surface missing
        # only `lookback_blocks` was told about sub-minimum markers, which is
        # the same defect as the hardcoded probe wearing different clothes.
        example = (" A marker below the minimum caches nothing and the "
                   "provider returns no error either."
                   if "min_cacheable_tokens" in reads.unanswered else "")
        # The remedy follows the cause too, for the same reason the detail
        # does. Adding a value and settling a disputed one are different jobs
        # in different files, and a surface can present both at once --
        # `openai/bedrock` is contested *and* records none of these keys, so it
        # gets both remedies rather than whichever one was matched first.
        seen = set()
        for causes, _ in reads.unanswered.values():
            seen |= causes
        fix = "; ".join(sorted({_REMEDIES[c] for c in seen if c in _REMEDIES}))
        subject = ",".join(
            f"{key}:{'|'.join(sorted(users))}"
            for key, (_causes, users) in sorted(reads.unanswered.items()))
        return Alert(
            "RT-NOSURFACE", "low", scope,
            f"{r.target_id!r} cannot answer {what} these checks make, so they "
            f"are inactive or unbounded",
            f"The registry could not answer: {detail}. Their silence means "
            f"unmeasured, not healthy.{example} Only the checks named here are "
            f"affected -- the rest read nothing from the registry.",
            subject="nosurface:" + subject, at=r.sent_at,
            fix=fix[:1].upper() + fix[1:] + ".")

    def _fire(self, st: _ScopeState, a: Alert, into: list) -> None:
        """Emit an alert once per `(code, subject)` for this scope.

        The one suppression table. RT-BLIND and RT-NOSURFACE wrote to `firing`
        by hand and skipped the eviction the check loop does, which was
        harmless while both used a constant subject and is not once a subject
        varies -- and RT-NOSURFACE's now does.
        """
        key = (a.code, a.subject)
        if key in st.firing:
            return
        # Once per subject until it clears, so a drifting prefix does not alert
        # on every request forever -- while a *second* drifting segment is
        # still its own alert.
        st.firing[key] = True
        while len(st.firing) > MAX_FIRING:
            st.firing.popitem(last=False)
        into.append(a)

    def clear(self, scope: tuple, code: str, subject: str | None = None) -> None:
        """Let a code fire again for this scope once it has been dealt with.

        With no subject, clears every live alert carrying that code.
        """
        st = self._scopes.get(scope)
        if not st:
            return
        for k in [k for k in st.firing if k[0] == code
                  and (subject is None or k[1] == subject)]:
            del st.firing[k]

    @property
    def scopes(self) -> int:
        return len(self._scopes)

    # --- the rolling state, as an estimator -------------------------------
    #
    # The runtime plugin needs exactly what these checks already maintain: how
    # often each position changes, and how far apart requests arrive. It reads
    # them from here rather than keeping its own rolling window, because two
    # copies of the same estimator are two things to keep in agreement and this
    # branch has already lost that argument four times.

    def change_rates(self, scope: tuple) -> dict:
        """`{segment index: share of observations on which it changed}`."""
        st = self._scopes.get(scope)
        if not st:
            return {}
        return {i: self._churn(v) for i, v in st.seg_values.items()}

    def gaps(self, scope: tuple) -> list:
        """Observed seconds between consecutive requests, most recent last."""
        st = self._scopes.get(scope)
        return list(st.gaps) if st else []

    def unfamiliar(self, scope: tuple, segments, min_samples: int | None = None) -> set:
        """Positions whose content right now this scope has no evidence about.

        Change rates are keyed by wire position, and a position is not a thing --
        it is wherever a block happens to land. Insert a section and everything
        below it shifts, so a rate learned about the old occupant is applied to
        a new one. A live path acting on that will mark a block it has never
        seen, on the strength of a number about different content.

        Two ways to be unfamiliar: too few observations at that position to
        say anything, or a current id that has not appeared there. Both mean the
        same thing to a caller about to mutate a request -- there is no evidence
        here -- and the caller should treat the position as fully volatile
        rather than inherit somebody else's history.

        Runtime-only by nature. The batch rules see every request before
        deciding anything, so there is no moment at which content is new to
        them; this is the cost of deciding in the request path.
        """
        st = self._scopes.get(scope)
        if st is None:
            return {s.index for s in segments}
        need = self.min_samples if min_samples is None else min_samples
        out = set()
        for s in segments:
            seen = st.seg_values.get(s.index)
            if seen is None or len(seen) < need or hash(s.id) not in seen:
                out.add(s.index)
        return out

    def samples(self, scope: tuple) -> int:
        """Requests observed in this scope, capped by the rolling window."""
        st = self._scopes.get(scope)
        return min(st.seen, WINDOW) if st else 0

    # --- bookkeeping ------------------------------------------------------

    def _scope(self, scope: tuple) -> _ScopeState:
        st = self._scopes.get(scope)
        if st is None:
            st = _ScopeState()
            self._scopes[scope] = st
            while len(self._scopes) > self.max_scopes:
                self._scopes.popitem(last=False)
        else:
            self._scopes.move_to_end(scope)
        return st

    def _track_shape(self, st: _ScopeState, r, reads: _RegistryReads) -> None:
        """State derived from what was sent. Disjoint from `_track_usage`, so
        the two can run at different moments without double-counting.

        Every tracked position gets an observation, present or not. Appending
        only for the segments that turned up meant an optional section's few
        observations were all identical, so it scored zero churn -- and the live
        allocator, reading those rates, would place a marker above a block that
        vanishes on the next request.
        """
        st.seen += 1
        present = {s.index: hash(s.id) for s in r.segments}
        for i in present:
            if i not in st.seg_values and len(st.seg_values) < MAX_POSITIONS:
                st.seg_values[i]                      # start tracking it
        for i, seen in st.seg_values.items():
            seen.append(present.get(i, _ABSENT))
        if not r.sent_at:
            return
        if st.last_sent is not None:
            gap = (r.sent_at - st.last_sent).total_seconds()
            if gap >= 0:
                st.gaps.append(gap)
        # Gaps between two markings of the *same* span. Tracked here rather than
        # with the writes because prefix identity is shape, and the shape and
        # usage checks run against different scope states -- recording it beside
        # `recent_writes` put the number in one state object and the check that
        # reads it in another, so RT-TTL simply stopped firing.
        #
        # Containment, not equality, and the lookup is inverted to get it. The
        # batch rule tests whether an earlier span is a prefix of what is being
        # sent, which needs the id sequence -- and this window deliberately
        # holds hashes so no prompt content sits in a long-lived process. So
        # instead of storing sequences and testing prefixes, hash *every*
        # boundary of this request up to its outermost marker and look each one
        # up. Any earlier marked span that is a prefix of this request has its
        # hash among them, by construction.
        #
        # Bounded exactly as before: `int -> (datetime, str|None)`, capped at
        # MAX_FIRING. Cost is one hash per segment per request.
        #
        # Collected only for a lifetime RT-TTL can actually argue about, and
        # filed under that lifetime. This is the same predicate the check and
        # the RT-NOSURFACE wording read, so the evidence, the rule and the
        # announcement now all agree about which requests count -- and the
        # registry lookups inside `_prefix_hashes` are not made at all on
        # traffic whose lifetime the check will refuse, which is why they are
        # no longer announced there either.
        lifetime, _offered = self._ttl_rt_ttl_can_read(r, reads)
        # Everything below is per evaluable lifetime, timeline included.
        #
        # Partitioning the *gaps* alone left one shared span->timestamp map
        # that every marked request wrote to, so a 30m write on a span
        # overwrote the 5m timestamp sitting there and the next 5m request
        # found nothing to measure from. Measured: a 5m stream that fires
        # RT-TTL with 19 gaps recorded zero gaps and never fired once
        # identical 30m writes were interleaved between the same requests.
        # The first version of this fix let an unevaluable lifetime invent
        # evidence; that one let it destroy evidence instead. A lifetime the
        # rule refuses now touches none of this state at all.
        #
        # Not an early return: `last_sent` below is the scope's request
        # cadence, which every lifetime contributes to and the plugin reads.
        # Only the per-lifetime rewrite timeline is skipped.
        if lifetime is not None:
            marks = st.last_marked_at[lifetime]
            seen_before = None
            for boundary in self._prefix_hashes(r, reads):
                when = marks.get(boundary)
                if when is not None and (seen_before is None
                                         or when > seen_before):
                    seen_before = when
            if seen_before is not None:
                rewrite_gap = (r.sent_at - seen_before).total_seconds()
                if rewrite_gap >= 0:
                    st.rewrite_gaps[lifetime].append(rewrite_gap)
            for span_hash in self._marked_hashes(r):
                marks[span_hash] = r.sent_at
                marks.move_to_end(span_hash)
            # Capped per lifetime, so the whole map is bounded by
            # len(TTL_SECONDS) * MAX_FIRING and neither bucket can evict the
            # other.
            while len(marks) > MAX_FIRING:
                marks.popitem(last=False)
        # Only forward. A negative gap was already dropped, but `last_sent` moved
        # backwards anyway, so the *next* request measured from the regressed
        # clock and recorded a gap inflated by however far time had gone back --
        # a five-minute cadence reading as fifteen, which is the difference
        # between TTL-1 firing and not. The batch path sorts; this sees one
        # request at a time and cannot, so out-of-order input is real here.
        # Replaying a captured trace is an advertised use of this class.
        if st.last_sent is None or r.sent_at >= st.last_sent:
            st.last_sent = r.sent_at

    def _track_usage(self, st: _ScopeState, r) -> None:
        """State derived from what the provider reported."""
        st.prefix_key = self._prefix_key(r)
        st.concurrent_writers = 0
        if r.sent_at and write_tokens(r.usage or {}) and st.prefix_key:
            # Counted before this request joins the window. Counting after made
            # every alert claim one more concurrent writer than there were.
            # Absolute distance, not elapsed time. Fan-out is a property of
            # when requests were *sent*, and usage arrives when responses come
            # *back* -- so on the live path these are observed in completion
            # order, and requiring the earlier write to have been seen first
            # meant an out-of-order completion read its own concurrent partner
            # as being in the future and ignored it. Two genuinely simultaneous
            # requests then produced no alert at all, which is the case the
            # check exists for.
            #
            # The boundary is `first_token_at` where the recorder captured it,
            # which is what FAN-1 uses. A flat five seconds is a guess about
            # provider latency, and on a target whose first token lands at 14s
            # it called an 8-second sibling sequential while the batch rule
            # called it concurrent -- the same trace, two answers.
            #
            # Concurrent means *neither* could see the other, so both halves
            # have to hold. With `or`, the second half is true of essentially
            # every earlier write -- an hour-old one is still before this
            # request's own visibility time -- and every spaced write got
            # reported as fan-out.
            mine, observed = write_visible_at(r, FANOUT_SECONDS)
            prior = sum(1 for t, vis, k in st.recent_writes
                        if k == st.prefix_key
                        and r.sent_at < vis and t < mine)
            st.concurrent_writers = prior + 1
            st.fanout_observed = observed
            st.recent_writes.append((r.sent_at, mine, st.prefix_key))

    @staticmethod
    def _prefix_key(r) -> int:
        """Digest of the cached prefix. An int, so the window holds no prompt."""
        marked = [s.index for s in r.segments if s.cache_marked]
        if not marked:
            return 0
        top = max(marked)
        return hash(tuple(s.id for s in sorted(r.segments, key=lambda s: s.index)
                          if s.index <= top))

    @staticmethod
    def _prefix_hashes(r, reads: _RegistryReads) -> list:
        """One hash per prefix boundary this request could actually read from.

        The read side of the containment test. An entry past every marker this
        request places is unreachable -- the provider searches back *from* a
        breakpoint -- so boundaries beyond the outermost one are not generated,
        which is `span_is_reusable_by`'s reachability condition expressed as
        which hashes get computed at all.

        The provider also searches back only a bounded number of blocks, so
        boundaries further than that behind every marker are not generated
        either. Leaving that out gave RT-TTL the same over-credit as TTL-1 on a
        tool loop appending 25 messages a call, and the parity class exists to
        make a one-sided fix fail rather than pass.
        """
        marked = [s.index for s in r.segments if s.cache_marked]
        if not marked:
            return []
        # Both lookups abstain into "no narrowing" rather than into silence,
        # which over-credits rather than under-reporting -- so RT-NOSURFACE has
        # to say the timeline is unbounded, not merely that a check is off.
        #
        # Reached only for a lifetime RT-TTL can argue about: `_track_shape`
        # applies that predicate before calling, so this builds no timeline and
        # makes no registry lookup on traffic the check will refuse. That is
        # why `openai/direct`, which advertises 30m only, is no longer told its
        # missing lookback unbounds a check that was quiet for an unrelated
        # reason -- the gap is not read here at all rather than read and then
        # not mentioned.
        window = reads.get(
            "lookback_blocks",
            lambda: registry.capability(r.target_id, "lookback_blocks"),
            inspect=lambda: registry.capability(r.target_id, "lookback_blocks",
                                                allow_contested=True),
            needed_for="RT-TTL (its rewrite timeline is then unbounded and "
                       "credits rewrites the provider could not have read back)")
        # Below the provider's minimum no entry is written, so a boundary
        # shorter than it can never be read back. The batch helper applies this
        # and this side did not, so RT-TTL fired on a sub-minimum marker where
        # TTL-1 refuses -- a divergence introduced by fixing one side.
        #
        # A missing minimum still gets announced on a marked request, by
        # RT-MIN's own call site in `_below_minimum`, with RT-MIN named as the
        # consequence. Gating here drops a second attribution, never the alert.
        floor = reads.get(
            "min_cacheable_tokens",
            lambda: registry.min_cacheable_tokens(r.target_id, r.model),
            inspect=lambda: registry.min_cacheable_tokens(
                r.target_id, r.model, allow_contested=True),
            needed_for="RT-TTL (rewrites of a prefix too short to cache are "
                       "then counted as recoverable)")
        top = max(marked)
        ordered = sorted(r.segments, key=lambda s: s.index)
        # Marker positions as lengths in this sequence, which is the unit
        # `lookback_blocks` and `span_is_reusable_by` both count in.
        marks = [i + 1 for i, s in enumerate(ordered) if s.cache_marked]
        out, seq, running = [], [], 0
        for s in ordered:
            if s.index > top:
                break
            seq.append(s.id)
            running += s.tokens or 0
            length = len(seq)
            if window is not None and not any(
                    0 <= m - length <= window for m in marks):
                continue
            if floor is not None and running < floor:
                continue
            out.append(hash(tuple(seq)))
        return out

    @staticmethod
    def _offered_ttls(r, allow_contested=False):
        """Lifetimes this surface accepts for this model, or None if it does
        not say.

        `registry.supported_ttls` collapses a null capability into `[]`, and
        `[]` is also what `deepseek/direct` legitimately records to mean the
        surface offers no lifetime control at all. Those two need opposite
        answers here -- the first is a registry gap to announce, the second is
        a fact RT-TTL should simply respect -- so presence is asked separately.
        The narrowing itself is still the registry's, including the per-model
        map that keeps a 1h recommendation off a Bedrock model where 1h never
        went GA.
        """
        if registry.capability(r.target_id, "supported_ttls",
                               allow_contested) is None:
            return None
        return registry.supported_ttls(r.target_id, r.model, allow_contested)

    @staticmethod
    def _ttl_rt_ttl_can_read(r, reads: _RegistryReads):
        """`(lifetime, offered)` -- what RT-TTL may reason about here.

        `lifetime` is the one in force, or None when this request is not
        something RT-TTL can argue about. `offered` is every lifetime the
        surface accepts for this model, because the *destination* of a switch
        needs checking as much as its source.

        One definition, read by `_cadence_vs_ttl` to decide whether to speak
        and what it may recommend, by `_track_shape` to decide whether to
        collect evidence at all, and by `_prefix_hashes` for its registry
        lookups. Restating the condition in any of them is how the
        announcement, the evidence and the check drift apart -- which is the
        shape of defect this file has already paid for three times.

        Two conditions, and the second was missing. `TTL_SECONDS` is what this
        module can *reason* about; `registry.supported_ttls` is what the
        surface will *accept*. Checking only the first told an operator to
        "set a one-hour TTL" on `openai/direct`, whose row advertises 30m and
        nothing else -- a recommendation the provider would reject, produced
        with full confidence from twenty perfectly good observations. Both
        halves have to hold: a lifetime this module cannot model is not
        actionable, and a lifetime the surface does not offer is not available.

        Returning the set rather than just the verdict is the other half of
        that. Proving the *current* lifetime is offered says nothing about the
        one being recommended, and on `amazon-bedrock/converse` +
        `claude-opus-5`, where only 5m is offered, a 5m stream still advised
        switching to 1h. The caller needs the set to check the destination.
        """
        lifetimes = r.marker_lifetimes
        # Two lifetimes in one request is a deliberate pattern -- a durable
        # prefix under an advancing turn -- and which one a cadence argument
        # is about is genuinely ambiguous. Abstaining beats guessing at the
        # operator's expense.
        if len(lifetimes) > 1:
            return None, ()
        # The lifetime on the prefix this advice is about, not the row's.
        #
        # Preferring `ttl_requested` told an operator to "set a one-hour TTL on
        # the stable prefix" on a request whose stable prefix was already 1h and
        # whose trailing turn was 5m -- the row field cannot express a mixed
        # request, so it reported the wrong one and the recommendation was a
        # no-op the operator would have had to disprove themselves.
        ttl = next(iter(lifetimes), None) or r.ttl_requested
        if ttl not in TTL_SECONDS:
            return None, ()
        # Through the seam, so a surface that cannot answer this is announced
        # rather than quietly treated as permissive. Defaulting to "supported"
        # is how the false recommendation above would come back.
        offered = reads.get(
            "supported_ttls",
            lambda: Monitor._offered_ttls(r),
            inspect=lambda: Monitor._offered_ttls(r, allow_contested=True),
            needed_for="RT-TTL (inactive: without the surface's lifetimes it "
                       "cannot tell a switch the provider would accept from "
                       "one it would reject)")
        if offered is None:
            return None, ()
        offered = tuple(offered)
        return (ttl if ttl in offered else None), offered

    @staticmethod
    def _marked_hashes(r) -> list:
        """One hash per *marked* span. The write side: only a marked position
        leaves an entry an later request could find."""
        marked = {s.index for s in r.segments if s.cache_marked}
        if not marked:
            return []
        out, seq = [], []
        for s in sorted(r.segments, key=lambda s: s.index):
            seq.append(s.id)
            if s.index in marked:
                out.append(hash(tuple(seq)))
        return out

    @staticmethod
    def _churn(seen: deque) -> float:
        """Share of observations on which this position changed."""
        if len(seen) < 2:
            return 0.0
        changes = sum(1 for a, b in zip(seen, list(seen)[1:]) if a != b)
        return changes / (len(seen) - 1)

    # --- checks -----------------------------------------------------------

    # Every check takes `reads` whether or not it uses one, so that adding a
    # registry lookup to any of them is a one-line change that is announced
    # from the moment it is written.

    def _drift(self, st: _ScopeState, r, scope, reads: _RegistryReads):
        """Positions inside the cached prefix that keep changing."""
        marked = [s.index for s in r.segments if s.cache_marked]
        if not marked:
            return
        top = max(marked)
        for s in sorted(r.segments, key=lambda s: s.index):
            if s.index > top:
                break
            seen = st.seg_values.get(s.index)
            if seen is None or len(seen) < self.min_samples:
                continue
            rate = self._churn(seen)
            if rate < self.drift_rate:
                continue
            # Tokens stranded by this position. Everything from the last marker
            # below it through the top of the prefix stops matching, not merely
            # what sits after it -- with no marker below, that is the whole
            # prefix. An earlier version counted only the tail and understated
            # a mid-prefix timestamp by most of its cost.
            below = max((x.index for x in r.segments
                         if x.cache_marked and x.index < s.index), default=-1)
            stranded = sum(x.tokens for x in r.segments
                           if below < x.index <= top and x.index != s.index)
            if not stranded:
                continue
            yield Alert(
                "RT-DRIFT", "high", scope,
                f"segment {s.index} ({s.label or s.role}) is changing inside the cached prefix",
                f"It changed on {rate:.0%} of the last {len(seen)} requests, stranding "
                f"roughly {stranded:,} tokens that would otherwise have been read from "
                f"cache. Caching matches from the start of the prompt, so those tokens "
                f"are re-sent every time this position moves.",
                subject=f"seg{s.index}", at=r.sent_at,
                fix="Move that section below the last cache marker, or stop varying it.")

    def _rebuild(self, st: _ScopeState, r, scope, reads: _RegistryReads):
        """The prefix is being thrown away and paid for again.

        The only check here that needs no prompt structure at all -- three usage
        counters and a session id -- which makes it the one thing this can say
        about the traces most teams can actually produce.

        A turn that extends a prefix writes what was added and reads the rest at
        0.1x. A turn that rebuilds writes the lot at 1.25x or 2x. The swing is
        roughly twelvefold on the tokens involved, and nothing in a usage
        dashboard distinguishes the two: both show up as "cache writes".
        """
        usage = r.usage or {}
        created = write_tokens(usage)
        read = usage.get("cache_read_input_tokens") or 0
        # Agent is part of the key for the same reason the batch rule uses it:
        # a subagent runs its own context and shares the parent's session id, so
        # keying on session alone measures one context's write against another's
        # prefix.
        key = (getattr(r, "agent", None), r.session) if r.session else None
        if not key:
            # Without a session id there is nothing to say a request follows
            # another. Substituting a constant compared every unrelated request
            # in the scope against whichever happened to arrive before it, and
            # a gateway export of independent one-shot calls reported "rebuilt
            # every 1 turns" -- a high-severity alert about a conversation that
            # does not exist. The batch rule has always required a session; this
            # is the runtime agreeing with it.
            #
            # ...and agreeing on the second condition too. REB-0 fires only when
            # something was actually written to cache: telling an operator that
            # rebuild detection is unavailable is useful when they are paying for
            # writes, and pure noise when the workload never caches anything.
            # This fired on every session-less request regardless, which is the
            # noise the batch twin deliberately suppresses.
            if not created:
                return
            yield Alert(
                "RT-NOSESSION", "low", scope,
                "rebuild detection is off: these requests carry no session id",
                "Telling a prefix being extended from one being rebuilt means "
                "knowing which request followed which. Nothing here says these "
                "belong to the same conversation, and assuming they do would "
                "invent the most expensive finding this monitor makes.",
                subject="nosession", at=r.sent_at,
                fix="Pass a stable conversation id through as the session, and "
                    "this check turns on by itself.")
            return
        # Seeded for every sessioned request, before the early return below.
        # Recording it only after that return meant a session's first request
        # left no timestamp, so its second had nothing to compare against and an
        # expired entry was counted as a rebuild -- forty short sessions whose
        # second turn arrived ten minutes later reported "rebuilt about every 1
        # turns", which is the misdiagnosis this exclusion exists to prevent.
        #
        # `last_seen` is bounded like every other map here, so on a gateway with
        # more live sessions than the cap an evicted session loses its timing
        # and its next write can be misclassified once. That is the same trade
        # the scope map makes: a fixed ceiling costs occasional repetition.
        previous = st.last_seen.get(key)
        # Both per-session facts are recorded here, together, before any early
        # return. Seeding one of them later is the same defect twice: the
        # timestamp was recorded after the return and every session's second
        # turn lost its comparison, and the fix for that left the *lifetime*
        # behind the same return, so the second turn then had a timestamp and
        # no lifetime and was dropped instead. One place, one time, both.
        prior = st.last_lifetime.get(key)
        if r.sent_at:
            st.last_seen[key] = r.sent_at
            st.last_seen.move_to_end(key)
            while len(st.last_seen) > MAX_FIRING:
                st.last_seen.popitem(last=False)
        declared = next((s.ttl for s in sorted(r.segments, key=lambda s: s.index)
                         if s.cache_marked and s.ttl), r.ttl_requested)
        # Seconds, not a lifetime name: a request can write both, and what the
        # next request needs to know is when the last of them expires.
        st.last_lifetime[key] = cost.expiry_seconds(
            r.usage, getattr(r, "marker_lifetimes", ()), declared)
        st.last_lifetime.move_to_end(key)
        while len(st.last_lifetime) > MAX_FIRING:
            st.last_lifetime.popitem(last=False)
        before = st.established.get(key, 0)
        st.established[key] = read + created
        st.established.move_to_end(key)
        while len(st.established) > MAX_FIRING:
            st.established.popitem(last=False)
        if not before:
            return
        # An entry that had already expired was not rebuilt, it was re-created
        # because the lifetime ran out. Counting it as a rebuild points the
        # operator at compaction or a rotating id when the answer is a longer
        # TTL -- which is RT-TTL's job, and the batch rule excludes these for
        # the same reason.
        gap = ((r.sent_at - previous).total_seconds()
               if r.sent_at and previous else None)

        # `prior` is the lifetime the *previous* request wrote, captured above.
        # Reading the current request's answered a different question: a
        # five-minute write followed ten minutes later by a one-hour one was
        # measured against the hour and filed as a rebuild, and the reverse
        # suppressed a real one.
        if gap is not None:
            seconds = prior
            if seconds is None:
                # Expiry and rebuild are indistinguishable without it, and
                # guessing produces the alert this exclusion exists to prevent.
                return
            if gap >= seconds:
                return
        st.rebuilds.append(created >= REBUILD_FRACTION * before)
        # Half the window, not the usual minimum. This alert fires once and
        # carries a rate in its text, and firing at the earliest qualifying
        # moment published the least accurate estimate this will ever hold --
        # one rebuild seen in sixteen turns reads as "every 16 turns" on a
        # workload rebuilding every 10.
        if len(st.rebuilds) < max(self.min_samples * 2, WINDOW // 2):
            return
        n = sum(st.rebuilds)
        if not n:
            return
        interval = len(st.rebuilds) / n
        if interval >= REBUILD_TURNS:
            return
        yield Alert(
            "RT-REBUILD", "high", scope,
            f"the cached prefix is rebuilt about every {interval:.0f} turns",
            f"{n} of the last {len(st.rebuilds)} turns rewrote at least half the prefix "
            f"they had already established. Extending a prefix writes only what was "
            f"added; rebuilding writes all of it, at 1.25x or 2x, replacing reads that "
            f"billed at 0.1x. Nothing in a usage dashboard tells the two apart.",
            subject="rebuild", at=r.sent_at,
            fix="Find what changes the prefix between turns. Context compaction, a "
                "rotating identifier and reordered tool definitions are the usual "
                "three, and a model switch is a separate cache pool by design.")

    def _blocked(self, st: _ScopeState, r, scope, reads: _RegistryReads):
        """Nothing is cached here, and this position is the reason.

        `_drift` reports a marked prefix being invalidated, which means it can
        only speak once somebody has already placed a marker. The more valuable
        case is the one where the volatility came first: a session id or a
        timestamp near the front of the prompt, a large stable block behind it,
        and consequently no marker anywhere because there is no stable prefix to
        put one on. That is silent to every check that starts by looking for a
        cache marker -- which was all of them.
        """
        if any(s.cache_marked for s in r.segments):
            return
        segs = sorted(r.segments, key=lambda s: s.index)
        minimum = reads.get(
            "min_cacheable_tokens",
            lambda: registry.min_cacheable_tokens(r.target_id, r.model),
            inspect=lambda: registry.min_cacheable_tokens(
                r.target_id, r.model, allow_contested=True),
            needed_for="RT-BLOCKED (inactive)")
        if minimum is None:
            return
        for pos, s in enumerate(segs):
            seen = st.seg_values.get(s.index)
            if (seen is None or len(seen) < self.min_samples
                    or self._churn(seen) < self.drift_rate):
                continue
            # The contiguous stable run behind it: what a marker could cover if
            # this segment were not sitting in front of it.
            stable = 0
            for nxt in segs[pos + 1:]:
                later = st.seg_values.get(nxt.index)
                if (later is None or len(later) < self.min_samples
                        or self._churn(later) >= self.drift_rate):
                    break
                stable += nxt.tokens
            if stable < minimum:
                continue
            yield Alert(
                "RT-BLOCKED", "high", scope,
                f"nothing is cached, and segment {s.index} "
                f"({s.label or s.role}) is why",
                f"It changed on {self._churn(seen):.0%} of the last {len(seen)} "
                f"requests and sits in front of {stable:,} tokens that did not "
                f"change at all. Caching matches from the start of the prompt, so "
                f"no marker can reach that stable block while this comes first.",
                subject=f"blocked{s.index}", at=r.sent_at,
                fix="Move that section below the stable block. Ordering changes "
                    "instruction priority and authority, so this needs a "
                    "behavioural eval before it ships.")
            return

    def _below_minimum(self, st: _ScopeState, r, scope, reads: _RegistryReads):
        """A marker on a prefix too short to cache, which fails silently."""
        # Every marker, through the same walk MIN-1 uses. This summed to the
        # outermost marker only, so a 200-token marker under a 30k one read as
        # 30k and the runtime stayed silent on exactly the case the batch rule
        # reports -- two answers to one question, from the same trace.
        prefixes = marked_prefixes(r.segments)
        if not prefixes:
            return
        minimum = reads.get(
            "min_cacheable_tokens",
            lambda: registry.min_cacheable_tokens(r.target_id, r.model),
            inspect=lambda: registry.min_cacheable_tokens(
                r.target_id, r.model, allow_contested=True),
            needed_for="RT-MIN (inactive)")
        if minimum is None:
            return
        short = [p for p in prefixes if p < minimum]
        if not short:
            return
        tokens = min(short)
        yield Alert(
            "RT-MIN", "medium", scope,
            f"cache marker on a prefix below the {minimum:,}-token minimum",
            f"The marked prefix is about {tokens:,} tokens. The provider processes this "
            f"uncached and returns no error, so nothing in normal monitoring shows it.",
            subject=str(minimum), at=r.sent_at,
            fix=f"Lengthen the cached prefix past {minimum:,} tokens, or drop the marker.")

    def _marker_budget(self, st: _ScopeState, r, scope, reads: _RegistryReads):
        """Marker count creeping toward the limit, which errors at the limit."""
        count = sum(1 for s in r.segments if s.cache_marked)
        # Before the lookup, not after. Zero markers cannot exhaust any
        # non-negative budget, so this request is answered without the registry
        # and reading first raised RT-NOSURFACE on ordinary unmarked traffic to
        # `openai/direct` -- a coverage warning about a check that was not
        # abstaining, it was concluding. Worse, the subject it burned is the
        # one a later marked request on the same scope would have used, so the
        # genuine abstention was then suppressed as a repeat.
        if not count:
            return
        budget = reads.get(
            "max_breakpoints",
            lambda: registry.capability(r.target_id, "max_breakpoints"),
            inspect=lambda: registry.capability(r.target_id, "max_breakpoints",
                                                allow_contested=True),
            needed_for="RT-BUDGET (inactive)")
        # A recorded zero is an answer, not a gap: `deepseek/direct` allows no
        # explicit breakpoints and there is no budget to run out of. Only the
        # `None` from `reads.get` above is unanswered, and that one has already
        # been recorded for RT-NOSURFACE by the time it gets here.
        if not budget or count < budget:
            return
        if count > budget:
            yield Alert(
                "RT-BUDGET", "high", scope,
                f"{count} cache markers exceeds the limit of {budget} on "
                f"{r.target_id}",
                f"This request already exceeds the provider's marker budget. "
                f"That is a request-level error, not a future headroom warning.",
                subject=f"budget-exceeded:{budget}", at=r.sent_at,
                fix=f"Merge sections that change at similar rates until "
                    f"{budget} or fewer markers remain.")
            return
        yield Alert(
            "RT-BUDGET", "medium", scope,
            f"using all {budget} cache markers this surface allows",
            f"This request carries {count}. A rolling conversation marker needs two of "
            f"them, so there is no headroom left for one and the next section that wants "
            f"a marker cannot have it.",
            subject="budget", at=r.sent_at,
            fix="Merge sections that change at similar rates until markers are free.")

    def _cadence_vs_ttl(self, st: _ScopeState, r, scope, reads: _RegistryReads):
        """Request spacing that the configured lifetime does not suit."""
        # Gaps between rewrites of one prefix, not gaps between requests. TTL-1
        # walks a per-prefix timeline for the same reason: a longer lifetime
        # recovers a rewrite by turning it into a read, so where nothing is
        # rewritten there is nothing to recover no matter how the requests are
        # spaced. Reading the scope's median gap fired on a workload marking a
        # fresh prefix every request, recommending a change worth nothing, while
        # the report refused to make the same claim from the same trace.
        #
        # Which lifetime this advice is about, resolved first, because it also
        # selects the evidence. Shared with `_track_shape`, which files each
        # gap under the lifetime that was in force when it was observed.
        ttl, offered = self._ttl_rt_ttl_can_read(r, reads)
        if ttl is None:
            return
        # This lifetime's own rewrites, never the scope's pooled history. A
        # median taken across lifetimes describes a workload that does not
        # exist, and the recommendation it produces is about traffic the
        # operator would have to disprove themselves.
        observed = st.rewrite_gaps[ttl]
        if len(observed) < self.min_samples:
            return
        gaps = sorted(observed)
        median = gaps[len(gaps) // 2]
        # The destination, not only the source. Proving the lifetime in force
        # is offered says nothing about the one being recommended, and on
        # `amazon-bedrock/converse` + `claude-opus-5` -- offered 5m and
        # nothing else -- a ten-minute 5m rewrite cadence still advised
        # switching to 1h. Every fix line below names a specific lifetime, so
        # every one of them has to clear the same bar the current one did.
        if ttl == "5m" and 300 < median < 3600 and "1h" in offered:
            yield Alert(
                "RT-TTL", "medium", scope,
                "requests arrive after the five-minute cache has expired",
                f"Median gap between rewrites of this prefix, over the last "
                f"{len(observed)} at this lifetime, is {median/60:.1f} "
                f"minutes, inside the window where a one-hour lifetime reads instead of "
                f"rewriting. Outside that window the five-minute default is cheaper, so "
                f"this is not a blanket change.",
                subject="to-1h", at=r.sent_at,
                fix="Set a one-hour TTL on the stable prefix for this workload.")
        elif ttl == "1h" and median >= 3600 and "5m" in offered:
            yield Alert(
                "RT-TTL", "medium", scope,
                "one-hour cache is expiring before the next request anyway",
                f"Median gap is {median/60:.1f} minutes, beyond the one-hour lifetime, so "
                f"every write is paying 2x and expiring unread. The five-minute write at "
                f"1.25x costs less for traffic this sparse.",
                subject="to-5m", at=r.sent_at,
                fix="Drop back to the five-minute default for this workload.")

    def _cold_fanout(self, st: _ScopeState, r, scope, reads: _RegistryReads):
        """Concurrent requests each paying to write the same prefix."""
        if st.concurrent_writers < 2:
            return
        yield Alert(
            "RT-FANOUT", "medium", scope,
            f"{st.concurrent_writers} concurrent requests writing the same prefix",
            f"Each went out before the previous one's entry could be read, so none "
            f"could see the others'. Every one of them paid the write premium for "
            f"the same tokens. "
            + ("The boundary is the observed first token on this traffic."
               if st.fanout_observed else
               f"No first-token timing was captured, so the boundary is an assumed "
               f"{FANOUT_SECONDS:.0f} seconds rather than a measurement."),
            subject=str(st.prefix_key), at=r.sent_at,
            fix="Warm the prefix with one request before fanning out, or stagger the "
                "dispatch enough for the first write to land.")
