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
    last_marked_at: OrderedDict = field(default_factory=OrderedDict)
    rewrite_gaps: deque = field(default_factory=lambda: deque(maxlen=WINDOW))
    rebuilds: deque = field(default_factory=lambda: deque(maxlen=WINDOW))
    firing: OrderedDict = field(default_factory=OrderedDict)


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

        if shape:
            self._track_shape(shape_st, request)
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
            if ("RT-BLIND", "blind") not in shape_st.firing:
                shape_st.firing[("RT-BLIND", "blind")] = True
                alerts.append(blind)
        for st, scope, checks in (
            (shape_st, shape_scope, shape_checks),
            (pool_st, pool_scope, usage_checks),
            (rebuild_st, rebuild_scope, rebuild_checks),
        ):
            for check in checks:
                for a in check(st, request, scope) or ():
                    # Once per subject until it clears, so a drifting prefix does
                    # not alert on every request forever -- while a *second*
                    # drifting segment is still its own alert.
                    key = (a.code, a.subject)
                    if key in st.firing:
                        continue
                    st.firing[key] = True
                    while len(st.firing) > MAX_FIRING:
                        st.firing.popitem(last=False)
                    alerts.append(a)
        return alerts

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

    def _track_shape(self, st: _ScopeState, r) -> None:
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
        pk = self._prefix_key(r)
        if pk:
            before = st.last_marked_at.get(pk)
            if before is not None:
                rewrite_gap = (r.sent_at - before).total_seconds()
                if rewrite_gap >= 0:
                    st.rewrite_gaps.append(rewrite_gap)
            st.last_marked_at[pk] = r.sent_at
            st.last_marked_at.move_to_end(pk)
            while len(st.last_marked_at) > MAX_FIRING:
                st.last_marked_at.popitem(last=False)
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
    def _churn(seen: deque) -> float:
        """Share of observations on which this position changed."""
        if len(seen) < 2:
            return 0.0
        changes = sum(1 for a, b in zip(seen, list(seen)[1:]) if a != b)
        return changes / (len(seen) - 1)

    # --- checks -----------------------------------------------------------

    def _drift(self, st: _ScopeState, r, scope):
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

    def _rebuild(self, st: _ScopeState, r, scope):
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

    def _blocked(self, st: _ScopeState, r, scope):
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
        try:
            minimum = registry.min_cacheable_tokens(r.target_id, r.model)
        except registry.RegistryError:
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

    def _below_minimum(self, st: _ScopeState, r, scope):
        """A marker on a prefix too short to cache, which fails silently."""
        # Every marker, through the same walk MIN-1 uses. This summed to the
        # outermost marker only, so a 200-token marker under a 30k one read as
        # 30k and the runtime stayed silent on exactly the case the batch rule
        # reports -- two answers to one question, from the same trace.
        prefixes = marked_prefixes(r.segments)
        if not prefixes:
            return
        try:
            minimum = registry.min_cacheable_tokens(r.target_id, r.model)
        except registry.RegistryError:
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

    def _marker_budget(self, st: _ScopeState, r, scope):
        """Marker count creeping toward the limit, which errors at the limit."""
        count = sum(1 for s in r.segments if s.cache_marked)
        try:
            budget = registry.capability(r.target_id, "max_breakpoints")
        except registry.RegistryError:
            return
        if not budget or count < budget:
            return
        yield Alert(
            "RT-BUDGET", "medium", scope,
            f"using all {budget} cache markers this surface allows",
            f"This request carries {count}. A rolling conversation marker needs two of "
            f"them, so there is no headroom left for one and the next section that wants "
            f"a marker cannot have it.",
            subject="budget", at=r.sent_at,
            fix="Merge sections that change at similar rates until markers are free.")

    def _cadence_vs_ttl(self, st: _ScopeState, r, scope):
        """Request spacing that the configured lifetime does not suit."""
        # Gaps between rewrites of one prefix, not gaps between requests. TTL-1
        # walks a per-prefix timeline for the same reason: a longer lifetime
        # recovers a rewrite by turning it into a read, so where nothing is
        # rewritten there is nothing to recover no matter how the requests are
        # spaced. Reading the scope's median gap fired on a workload marking a
        # fresh prefix every request, recommending a change worth nothing, while
        # the report refused to make the same claim from the same trace.
        if len(st.rewrite_gaps) < self.min_samples:
            return
        # The lifetime on the prefix this advice is about, not the row's.
        #
        # Preferring `ttl_requested` told an operator to "set a one-hour TTL on
        # the stable prefix" on a request whose stable prefix was already 1h and
        # whose trailing turn was 5m -- the row field cannot express a mixed
        # request, so it reported the wrong one and the recommendation was a
        # no-op the operator would have had to disprove themselves.
        lifetimes = r.marker_lifetimes
        if len(lifetimes) > 1:
            # Two lifetimes in one request is a deliberate pattern -- a durable
            # prefix under an advancing turn -- and which one this cadence
            # argues about is genuinely ambiguous. Abstaining beats guessing at
            # the operator's expense.
            return
        ttl = next(iter(lifetimes), None) or r.ttl_requested
        if ttl not in TTL_SECONDS:
            return
        gaps = sorted(st.rewrite_gaps)
        median = gaps[len(gaps) // 2]
        if ttl == "5m" and 300 < median < 3600:
            yield Alert(
                "RT-TTL", "medium", scope,
                "requests arrive after the five-minute cache has expired",
                f"Median gap between rewrites of this prefix, over the last "
                f"{len(st.rewrite_gaps)}, is {median/60:.1f} "
                f"minutes, inside the window where a one-hour lifetime reads instead of "
                f"rewriting. Outside that window the five-minute default is cheaper, so "
                f"this is not a blanket change.",
                subject="to-1h", at=r.sent_at,
                fix="Set a one-hour TTL on the stable prefix for this workload.")
        elif ttl == "1h" and median >= 3600:
            yield Alert(
                "RT-TTL", "medium", scope,
                "one-hour cache is expiring before the next request anyway",
                f"Median gap is {median/60:.1f} minutes, beyond the one-hour lifetime, so "
                f"every write is paying 2x and expiring unread. The five-minute write at "
                f"1.25x costs less for traffic this sparse.",
                subject="to-5m", at=r.sent_at,
                fix="Drop back to the five-minute default for this workload.")

    def _cold_fanout(self, st: _ScopeState, r, scope):
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
