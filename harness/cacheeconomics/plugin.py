"""Runtime plugin: place cache markers from what this process has actually seen.

Build item 9. The analyzer decides where markers go after the fact and hands a
human a patch. This decides in the request path, from a rolling window of the
traffic it just served.

That is a strictly worse position to decide from, and the design is mostly about
admitting it:

**It learns before it acts.** Placement rests on how often each position
changes, and one request says nothing about that. Until a scope has served
`warmup` requests with observed gaps between them, the plugin returns the
request untouched. A cold plugin that guesses is worse than no plugin, because
it charges the write premium for entries chosen at random.

**It counts tokens by estimate, so it abstains near the threshold.** At request
time nothing has counted the prompt. Below the model's minimum a marker caches
nothing and the provider returns no error, so a marker placed on an estimate
that lands just above a threshold fails silently and for money. Anything within
`minimum_margin` of the minimum is left unplaced with a note, which is the
tri-state ABSTAIN rule the static checks use, applied where it costs the most.

**It never moves content.** Relocation is the highest-value transform available
and it changes instruction priority, recency and authority. It requires a
behavioural eval, and an eval is not something a request path can run. The
plugin places markers on the prompt as authored, and reports the relocation it
would recommend as a finding for a human.

**It stands down on a request that already carries markers.** Same rule LiteLLM
follows, and for the same reason: two systems placing markers on one request
produce a result neither chose. A caller who has decided keeps their decision.

**It shares the analyzer's estimator, not a copy of it.** The rolling change
rates and gap distribution come from `Monitor`, which already maintains exactly
those to raise alerts. Two rolling windows of the same quantity are two things
to keep in agreement, and this branch has lost that argument four times.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone

import logging

from . import registry, tiers

# Named, so an operator can silence or route it. A hook that fails open silently is a hook nobody knows is failing.
_log = logging.getLogger("cacheeconomics.plugin")
from .allocate import reuse_chain
from .monitor import MAX_FIRING, MAX_SCOPES, WINDOW, Alert, Monitor
from .segment import (apply_markers, marker_count, segments_from_request,  # noqa: F401
                      strip_markers, walk)
from .trace import (UNATTRIBUTED, Request, Segment, _text, read_field,
                    resolve_litellm_tenant, session_of, write_tokens)

# Bytes per token, matching the recorder's estimator. Deliberately the same
# constant: a plugin that estimated prompt size differently from the tool that
# audits it would produce placements the audit could not reproduce.
BYTES_PER_TOKEN = 3.6

# Wire containers a caller may be able to patch. Membership is decided by the
# path `walk` reports, not by a segment's role: in the LiteLLM shape a system
# instruction can arrive either as a top-level `system` field or as a message
# with role "system", and role cannot tell those apart. Expressing markability
# as roles let the handler patch a top-level `system` while its own docstring
# said messages only -- crossing the unverified wire boundary that keeps markers
# off `tools`, for exactly the same reason and without noticing.
MESSAGE_CONTAINERS = frozenset({"messages"})


def _wire_ttl(body: dict) -> str | None:
    """The one lifetime this body's markers request, or None if they disagree.

    Reads the body rather than the plugin's own plan, because the caller's
    markers count too. A marker with no `ttl` is the provider's five-minute
    default; that is what the wire means, not an unknown.
    """
    from .segment import marker_paths, _at_path
    ttls = set()
    for path in marker_paths(body):
        blk = _at_path(body, path)
        if isinstance(blk, dict) and isinstance(blk.get("cache_control"), dict):
            ttls.add(blk["cache_control"].get("ttl") or "5m")
    return next(iter(ttls)) if len(ttls) == 1 else None


def markable_positions(body: dict, containers=MESSAGE_CONTAINERS) -> frozenset:
    """Wire positions living in a container this caller can patch.

    The caller knows its own transport; the plugin does not. So markability is
    stated as positions, computed from the same `walk` that produces the
    segments, rather than inferred from a role.

    Two positions are excluded whatever the caller says. Tool history --
    `tool_calls` and `tool_call_id` -- is visible to the segmenter so a rotating
    call id shows up as the drift it is, and is not a marker location: it is not
    content, and the message carrying it belongs to a protocol this package does
    not model. Modelling LiteLLM's translation of it is the trap `litellm_auto`
    already fell into, and the cost of guessing wrong here is a mutated request
    on somebody's live traffic.

    So does the *content* of a message that carries either field. Marking a
    bare-string `tool` message rewrote `"18C"` into Anthropic block form on an
    OpenAI-shaped request -- a body neither provider documents, produced by a
    tool whose entire claim is that it does not guess.
    """
    stood_down = set()
    for i, (_, _, _, path) in enumerate(walk(body)):
        if len(path) > 2 and isinstance(path[2], str):
            stood_down.add(path[1])
    return frozenset(
        i for i, (_, _, _, path) in enumerate(walk(body))
        if path[0] in containers
        and not (len(path) > 2 and isinstance(path[2], str))
        and not (path[0] == "messages" and path[1] in stood_down))

# Alerts a plugin keeps for a human to read. The monitor caps its own state and
# this did not: a long-running proxy accumulated one list entry per alert for
# the life of the process, which is the bounded-memory premise of putting any of
# this in a request path failing in the one component that mutates requests.
MAX_ALERTS = 1024


@dataclass(frozen=True)
class Decision:
    """What the plugin did to one request, and why.

    Carries enough context to complete the observation when the response comes
    back: the scope, the session, when it was sent, and the segments as they
    actually went on the wire. Fan-out is keyed on which prefix was written, so
    recovering that from live scope state instead would race against every other
    request in flight.
    """
    applied: bool
    placements: dict = field(default_factory=dict)     # wire index -> ttl, as sent
    # What a dry run would have placed. Deliberately a separate field: an
    # observe-only run that reported its proposal as `placements` had the
    # effectiveness counters crediting markers that never reached the provider.
    proposed: dict = field(default_factory=dict)
    reason: str = ""
    notes: list = field(default_factory=list)
    estimated_tokens: int = 0
    scope: tuple = ()
    session: str | None = None
    agent: str = "unknown"
    sent_at: datetime | None = None
    segments: tuple = ()

    def __str__(self):
        if self.applied:
            head = "placed " + ", ".join(f"{i}@{t}" for i, t in
                                         sorted(self.placements.items()))
        else:
            head = f"stood down: {self.reason}"
            # Otherwise the proposal this deliberately preserves is invisible to
            # anyone reading a log, and only a programmatic caller ever sees it.
            if self.proposed:
                head += ("\n    would have placed " +
                         ", ".join(f"{i}@{t}" for i, t in
                                   sorted(self.proposed.items())))
        return head + "".join(f"\n    {n}" for n in self.notes)


class CachePlugin:
    """Observe requests, then place markers on them once there is evidence to.

    Usage is two calls per request, in this order:

        body, decision = plugin.on_request(body, model=..., tenant=...)
        ...                                   # send `body`
        plugin.on_response(body, response)    # optional; sharpens fan-out

    `on_request` is the whole plugin. `on_response` only feeds usage back so the
    monitor can see cold fan-out, and skipping it costs one alert code and
    nothing else.
    """

    def __init__(self, *, key: bytes, warmup: int = 32,
                 minimum_margin: float = 0.15,
                 respect_existing: bool = True,
                 max_scopes: int = MAX_SCOPES,
                 monitor: Monitor | None = None):
        if not key:
            raise ValueError(
                "a keyed segment id needs a key. Raw digests of low-entropy "
                "prompt sections are dictionary-guessable: anyone holding a "
                "candidate prompt can confirm it by recomputing the digest. "
                "Cross-tenant equality is a separate problem the key does not "
                "solve -- the tenant is part of segment identity for that.")
        self.key = key
        # `Monitor.samples()` reports at most WINDOW observations, because the
        # window is bounded on purpose. A warmup above it is therefore never
        # reached: measured, CachePlugin(warmup=100) sat at "still learning
        # (65/100 requests)" through 300 well-spaced requests, placed nothing,
        # and said nothing about why. A plugin that silently does nothing for the
        # lifetime of the process is worse than one that refuses to start.
        if warmup > WINDOW:
            raise ValueError(
                f"warmup={warmup} exceeds the monitor's observation window of "
                f"{WINDOW}, so the plugin would never finish learning "
                f"and would never place a marker. Lower the warmup, or raise "
                f"monitor.WINDOW if you genuinely need a longer history.")
        self.warmup = warmup
        self.minimum_margin = minimum_margin
        self.respect_existing = respect_existing
        self.monitor = monitor or Monitor(max_scopes=max_scopes)
        self.max_scopes = max_scopes
        # Both bounded, on the same reasoning as Monitor's own state: this runs
        # inside somebody else's proxy, for weeks. Alerts drop oldest-first;
        # effectiveness evicts least-recently-seen scope, matching the monitor's
        # policy so the two do not disagree about which scopes still exist.
        self.alerts: deque = deque(maxlen=MAX_ALERTS)
        self._effect: OrderedDict = OrderedDict()
        # Which (scope, code, subject) triples have already been reported, so a
        # condition true on every request is one alert rather than one per
        # request. Same contract the monitor holds itself to, and bounded for
        # the same reason: this lives in a request path for weeks.
        self._said: OrderedDict = OrderedDict()

    def _record_alert(self, alert) -> bool:
        """Keep an alert, once per subject per scope. Returns whether it was
        new, so a caller can tell "reported" from "already reported"."""
        key = (alert.scope, alert.code, alert.subject)
        if key in self._said:
            return False
        self._said[key] = True
        while len(self._said) > MAX_FIRING * 8:
            self._said.popitem(last=False)
        self.alerts.append(alert)
        return True

    # --- the request path -------------------------------------------------

    def on_request(self, body: dict, *, model: str,
                   target_id: str = "anthropic/direct",
                   tenant: str | None = None,
                   session: str | None = None,
                   agent: str = "unknown",
                   markable: frozenset | None = None,
                   apply: bool = True,
                   at: datetime | None = None) -> tuple[dict, Decision]:
        """Return the body to send, and what was decided about it.

        `markable` names the segment roles this caller can actually attach a
        marker to. Pass it when the transport cannot carry `cache_control`
        everywhere -- a gateway that rewrites tool definitions, say. The whole
        prompt is still modelled either way, because a marker's prefix contains
        the blocks above it whether or not they are markable, and hiding them
        from the model produces a plan for a prompt nobody sends.
        """
        at = at or datetime.now(timezone.utc)
        # Normalised before anything looks it up. Gateways pass provider-
        # prefixed and date-stamped ids -- LiteLLM sends "anthropic/claude-opus-5"
        # -- and the registry does not recognise those, so every minimum lookup
        # raised and the one check standing between this and a silently
        # uncacheable marker was off on exactly the integration path that
        # mutates live requests.
        model = registry.normalize_model(model, target_id)[0]
        raw = segments_from_request(body, self.key, tenant)
        segs = [Segment(id=s["id"], role=s["role"],
                        tokens=max(1, round(s["bytes"] / BYTES_PER_TOKEN)),
                        index=s["index"], label=s["label"],
                        cache_marked=s["cache_marked"], ttl=s["ttl"])
                for s in raw]
        scope = (tenant, target_id, model)
        shape_scope = reuse_chain(scope, agent, session)
        out, decision = self._decide(body, segs, scope, target_id, model,
                                     markable, apply, shape_scope)

        # A dry run plans and does not send. Reporting the plan as though it had
        # gone out meant the monitor observed a prefix nobody transmitted and
        # the effectiveness counters credited markers the provider never saw --
        # on the mode that exists precisely to be the safe one.
        if not apply and decision.applied:
            decision = replace(
                decision, applied=False, placements={},
                proposed=decision.placements,
                reason="observing only: markers were planned and not sent")
            out = body

        # Observed last, and against the segments as they will actually be sent.
        #
        # Two things fall out of that order. The decision rests only on traffic
        # that came before it, rather than partly on the request it is deciding
        # about. And the monitor sees the markers this plugin just placed, which
        # is what its drift and budget checks are about -- observing the
        # unmarked body first meant every check that needs a cached prefix found
        # none and stayed silent forever.
        # The markers that will actually be on the wire, not the union of what
        # was there and what was chosen. Override strips the caller's markers
        # before applying ours, so OR-ing the two told the monitor a volatile
        # trailing turn was cached on a request where it had been stripped --
        # and drift, budget and fan-out then reasoned about a prefix nobody
        # sent. When nothing was applied the wire is what the caller wrote.
        placed = decision.placements
        on_wire = set(placed) if decision.applied else {
            s.index for s in segs if s.cache_marked}
        as_sent = tuple(Segment(id=s.id, role=s.role, tokens=s.tokens, index=s.index,
                                label=s.label,
                                cache_marked=s.index in on_wire,
                                ttl=placed.get(s.index, s.ttl if s.index in on_wire
                                               else None)) for s in segs)
        # Shape only. The usage counters do not exist yet, and handing the
        # monitor an empty `usage` here made every usage-driven check -- cold
        # fan-out and prefix rebuild, the two most expensive live failures --
        # permanently unreachable from the request path.
        self.alerts.extend(self.monitor.observe_shape(Request(
            request_id="live", sent_at=at, model=model, usage={},
            segments=list(as_sent), tenant=tenant, target_id=target_id,
            session=session, agent=agent,
            # From the markers actually on the wire, not only the plugin's own.
            #
            # This read `placed`, which is empty whenever the plugin stands down
            # -- and standing down for a caller's existing markers is the common
            # case, the whole observe-only path. So the monitor was handed
            # `ttl_requested=None` for a request that plainly carried a 5m
            # marker, and the cadence and expiry checks went quiet. Reproduced:
            # twenty default markers fifteen minutes apart, no RT-TTL.
            #
            # A silent `cache_control` is Anthropic's 5m default, which is a fact
            # about the wire rather than an assumption. Still None when the wire
            # carries a genuine mix, because a request can write both lifetimes
            # and the batch path refuses to price that as one.
            ttl_requested=_wire_ttl(out if apply else body))))
        return out, replace(decision, scope=(tenant, target_id, model),
                            session=session, agent=agent, sent_at=at,
                            segments=as_sent)

    def _decide(self, body, segs, scope, target_id, model, markable=None,
                apply: bool = True, shape_scope=None):
        """Placement from prior evidence only. Returns `(body, Decision)`."""
        estimated = sum(s.tokens for s in segs)
        if not segs:
            return body, Decision(False, reason="no prompt structure",
                                  estimated_tokens=estimated)
        # `marker_count`, not `any(s.cache_marked)`. Segments are content blocks,
        # so a caller who marks the message object instead was invisible here and
        # the plugin placed on top of markers it did not know existed.
        if self.respect_existing and (any(s.cache_marked for s in segs)
                                      or marker_count(body)):
            return body, Decision(
                False, reason="the caller already placed cache markers",
                notes=["Standing down rather than combining two placement "
                       "decisions into one nobody made. Pass "
                       "respect_existing=False to override."],
                estimated_tokens=estimated)

        evidence_scope = shape_scope or scope
        samples = self.monitor.samples(evidence_scope)
        gaps = self.monitor.gaps(evidence_scope)
        if samples < self.warmup or not gaps:
            return body, Decision(
                False, reason=f"still learning ({samples + 1}/{self.warmup} requests)",
                notes=["Placement rests on how often each position changes and "
                       "how far apart requests arrive. Neither is knowable from "
                       "one request, and a marker chosen without them charges "
                       "the write premium for a guess."],
                estimated_tokens=estimated)

        # Evidence about *this* prompt, not about whatever used to occupy these
        # positions. Rates are keyed by wire index, so inserting a section
        # shifts every block below it and hands each one a history belonging to
        # its predecessor. Without this, a block appearing for the first time
        # inherits "perfectly stable" and gets a cache marker on the request
        # that introduces it.
        learned = dict(self.monitor.change_rates(evidence_scope))
        # The caller's warmup is the bar, not the monitor's alerting threshold.
        # Using the latter meant a plugin configured to act early was silently
        # held to a stricter per-position rule than the one it was given.
        unfamiliar = self.monitor.unfamiliar(evidence_scope, segs,
                                             min_samples=self.warmup)
        rates = dict(learned)
        for i in unfamiliar:
            rates[i] = 1.0

        try:
            alloc = tiers.allocate(segs, rates, target_id=target_id,
                                   model=model, gaps=gaps)
        except tiers.Unsupported as e:
            return body, Decision(False, reason=str(e), estimated_tokens=estimated)

        placements, notes = self._filter_near_minimum(alloc, segs, target_id, model,
                                                      markable)
        # Only the positions where inheriting a neighbour's history would have
        # changed the answer. A trailing turn is new content on every request by
        # design; saying so each time would bury the case that matters.
        deepest = max(placements) if placements else -1
        blocking = [i for i in sorted(unfamiliar)
                    if i > deepest and learned.get(i, 1.0) < 1.0]
        if blocking:
            notes.append(
                f"positions {blocking} carry content this scope has not seen before, "
                f"so they are treated as fully volatile rather than inheriting the "
                f"history of whatever previously sat at that index. A prompt that "
                f"just changed shape earns its evidence back within a few requests.")
        notes += list(alloc.notes)
        if not placements:
            # "Nothing was worth placing" and "I could not tell whether placing
            # anything was safe" are different answers, and only one of them
            # means the workload is fine.
            # Any abstention, not only the whole-request one. A near-minimum
            # filter that dropped every candidate is the plugin saying "I could
            # not prove this is safe"; reporting it as "no marker worth placing"
            # is the "your workload is fine" reading, which is the opposite.
            blocked = next((n for n in notes if n.startswith("ABSTAIN")), None)
            return body, Decision(
                False,
                reason=blocked or "no marker worth placing",
                notes=notes, estimated_tokens=estimated)
        # A bare string has nowhere to put cache_control, so apply_markers
        # rewrites it into the block form the API documents as equivalent. That
        # changes the bytes on the wire, which is worth saying out loud rather
        # than leaving for someone to discover by diffing their own logs.
        reshaped = [i for i, (_, _, blk, _) in enumerate(walk(body))
                    if i in placements and isinstance(blk, str)]
        if reshaped:
            notes.append(
                f"positions {reshaped} were bare strings and "
                f"{'are' if apply else 'would be'} sent as text "
                f"blocks, because a string has nowhere to carry a cache marker. "
                f"The prompt is unchanged as the model sees it; the request "
                f"bytes are not, so the first marked request cannot read an "
                f"entry written before the plugin started marking.")
        # Override means replace. With `respect_existing=False` the caller's
        # markers were left in place and ours were added on top, so a request
        # already carrying the surface's four went out with five -- a provider
        # error dressed as an override, on the path that mutates live requests.
        #
        # Gated on `marker_count`, not on `any(s.cache_marked)`. Segments are
        # content blocks, so this gate could not see a message-level marker and
        # therefore did not strip: with `respect_existing=False` the caller's four
        # survived and the plugin's two were added on top, producing exactly the
        # six-against-four the comment above says this exists to prevent. Fixing
        # the stand-down check and leaving this one would have been the same defect
        # twice in the same function.
        base, replaced_note = body, None
        existing = max(sum(1 for s in segs if s.cache_marked), marker_count(body))
        if existing:
            base = strip_markers(body)
            replaced_note = (
                f"{'replaced' if apply else 'would replace'} "
                f"{existing} marker(s) the caller "
                f"had placed. Override replaces rather than combines: adding to "
                f"them can exceed the surface's breakpoint budget, and two plans "
                f"layered together are a plan nobody chose.")
        out = apply_markers(base, placements)
        try:
            budget = registry.capability(target_id, "max_breakpoints")
        except registry.RegistryError:
            budget = None
        if budget:
            # Counted the way the provider counts, over the patched body. This
            # used to count `walk(out)`, which yields content blocks only, so a
            # caller's message-level markers were absent from the total the check
            # compared against the budget -- and they survive onto the wire.
            on_wire = marker_count(out)
            if on_wire > budget:
                # The unmodified body goes back, so the replacement note stays
                # off: nothing was replaced on the request the caller sends.
                return body, Decision(
                    False,
                    reason=f"the patched request would carry {on_wire} markers "
                           f"against a budget of {budget}",
                    notes=notes, estimated_tokens=estimated)
        if replaced_note:
            notes.append(replaced_note)
        return out, Decision(True, placements, notes=notes,
                             estimated_tokens=estimated)

    def on_response(self, decision: Decision, response, *, model: str | None = None,
                    target_id: str | None = None,
                    tenant: str | None = None) -> list:
        """Complete the observation, and record whether our markers were read.

        Two jobs. It finishes what `on_request` started: the monitor's fan-out
        and rebuild checks are driven by the provider's counters, which only
        exist now, and skipping this call leaves both silent. And it records the
        one thing a response can answer that a request cannot -- whether the
        markers this plugin placed were actually read back.

        Returns any alerts the usage revealed. Calling this is optional in the
        sense that placement still works without it, and not optional if you
        want to be told about a prefix being rebuilt.
        """
        from .segment import usage_from_response
        usage = usage_from_response(response) or {}
        scope = decision.scope or (tenant, target_id or "anthropic/direct", model)
        alerts = []
        if usage:
            new = self.monitor.observe_usage(Request(
                request_id="live", sent_at=decision.sent_at,
                model=scope[2] or model or "", usage=usage,
                segments=list(decision.segments), tenant=scope[0],
                target_id=scope[1], session=decision.session,
                # Subagents run their own context and share a conversation id,
                # so the rebuild check keys on both. Omitting it here left every
                # live request under the Request default and pooled exactly the
                # contexts the batch rule had just been taught to separate.
                agent=decision.agent))
            self.alerts.extend(new)
            alerts = new
        if not decision.applied:
            return alerts
        if not usage:
            # "Absent usage is not zero usage" -- the rule the guard above this
            # already applies to the monitor, and this block ran regardless. So a
            # transport that drops the usage object counted `placed += 1` and
            # `read += 0`, and `read_share` came back 0.0 with the full weight of
            # a measured number. `effectiveness()` describes itself as "the
            # provider's own counters", which is exactly what was missing.
            #
            # A response that told us nothing must not move a counter in either
            # direction. The alerts still return: the request happened, and the
            # monitor's own state above has already recorded that.
            return alerts
        st = self._effect.setdefault(scope, {"placed": 0, "read": 0,
                                             "read_tokens": 0, "write_tokens": 0})
        self._effect.move_to_end(scope)
        while len(self._effect) > self.max_scopes:
            self._effect.popitem(last=False)
        st["placed"] += 1
        read = usage.get("cache_read_input_tokens") or 0
        st["read"] += 1 if read else 0
        st["read_tokens"] += read
        st["write_tokens"] += write_tokens(usage)
        return alerts

    def effectiveness(self, scope: tuple) -> dict | None:
        """How the plugin's own placements have performed in this scope.

        Measured, not modelled: these are the provider's own counters for
        requests this plugin marked. It is the only number here that is not an
        estimate, and it is still not money -- pricing it needs an invoice.
        """
        st = self._effect.get(scope)
        if not st or not st["placed"]:
            return None
        return {**st, "read_share": st["read"] / st["placed"]}

    # --- abstention -------------------------------------------------------

    def _filter_near_minimum(self, alloc, segs, target_id, model, markable=None):
        """Drop markers this cannot prove are safe to place.

        Two gates, and both fail closed, because this one mutates a live request
        rather than describing one after the fact.

        The token count is a byte-ratio estimate and below the minimum a
        provider caches nothing while returning no error, so anything inside
        `minimum_margin` of the threshold is left alone. An unknown minimum is
        the same situation with less information, and an earlier version placed
        every marker anyway with a note attached -- which is precisely backwards:
        not knowing the threshold is the case where a marker is most likely to
        be paid for and cache nothing.

        `markable` restricts placement to wire containers the caller can
        actually patch. A gateway that cannot carry `cache_control` on tool
        definitions still wants markers on its messages, and the alternative --
        pretending tools are unmarkable by hiding them from the model entirely --
        is what made the prefix wrong rather than merely smaller.
        """
        by_pos = {s.index: s for s in segs}
        try:
            minimum = registry.min_cacheable_tokens(target_id, model)
        except registry.RegistryError:
            return {}, [
                f"ABSTAIN entirely: no recorded minimum cacheable prefix for "
                f"{model} on {target_id}. Below that threshold a marker caches "
                f"nothing and the provider returns no error, so placing one "
                f"without knowing it risks paying the write premium for nothing."]
        floor = minimum * (1 + self.minimum_margin)
        keep, notes = {}, []
        for t in alloc.tiers:
            seg = by_pos.get(t.marker_position)
            if markable is not None and t.marker_position not in markable:
                where = f"{seg.role!r} block" if seg is not None else "block"
                notes.append(
                    f"ABSTAIN at wire position {t.marker_position}: it is a "
                    f"{where} this integration cannot carry a marker on. It stays "
                    f"inside the prefix of any marker placed below it, so it is "
                    f"still cached -- just not at its own boundary.")
                continue
            if t.prefix_tokens >= floor:
                keep[t.marker_position] = t.ttl
            else:
                notes.append(
                    f"ABSTAIN at wire position {t.marker_position}: estimated "
                    f"prefix {t.prefix_tokens:,} tokens against a {minimum:,} "
                    f"minimum, inside the {self.minimum_margin:.0%} margin this "
                    f"estimate cannot resolve. Below the minimum a marker caches "
                    f"nothing and no error is returned.")
        return keep, notes

    # --- what it would recommend a human do -------------------------------

    def recommendations(self, scope: tuple) -> list:
        """What this scope should change, that the request path cannot do itself.

        These are the monitor's alerts, and the relevant one is RT-DRIFT: it
        names a position that keeps changing inside the cached prefix, and its
        fix is to move that section below the last marker. That move is the
        relocation the plugin will not make on its own -- reordering changes
        instruction priority and authority, which needs a behavioural eval, and
        a request path cannot run one. So it is surfaced here for a human rather
        than performed.
        """
        return [a for a in self.alerts if a.scope == scope]


def _field(obj, *names):
    """Mapping-or-object identity read. Lives in trace.py now, because the
    loader needed the same accessor and reaching for `.get` alone dropped every
    object-shaped tenant on that side."""
    return read_field(obj, *names)


def default_session_from(data: dict):
    """The conversation this request belongs to, if LiteLLM was told.

    Re-read against LiteLLM's schema on 30 Jul 2026, and the earlier reading here
    was wrong. `trace_id` is documented to "trace multiple LLM calls belonging to
    same overall request (e.g. fallbacks/retries)" -- that is one call and its
    retries, so a conversation's turns carry different ones. It is per-request in
    the way that matters, exactly like `litellm_call_id`.

    Using either as the session makes every turn look like the first turn of a new
    conversation, and rebuild detection -- the one thing worth having here --
    never accumulates any evidence. Worse, a non-null session suppresses REB-0,
    the finding that exists to say the measurement could not be made.

    Returns None when nothing stable is present, which is the honest answer.
    Downstream that turns rebuild detection off and says so, rather than
    inventing a conversation out of unrelated calls.
    """
    # Delegated. This used to read `metadata` and then `litellm_trace_id`,
    # skipping the plain top-level `trace_id` that the docstring above names as
    # the documented field -- so a payload carrying exactly that returned None.
    return session_of(data)


def default_agent_from(data: dict) -> str:
    """Which context inside the conversation this request belongs to.

    A subagent runs its own prompt and shares its parent's conversation id, so
    rebuild counting keys on both. LiteLLM has no standard field for it, and
    inventing one would be worse than admitting there is none -- so this reads
    the places a caller would put it and otherwise returns "unknown", which
    pools contexts exactly as before and is at least honest about doing so.
    """
    meta = data.get("metadata") or {}
    return (_field(meta, "agent", "agent_name", "subagent", "context_id")
            or _field(data, "litellm_agent") or "unknown")


# Kept next to the handler rather than imported from the adapter, because the
# adapter lives behind an optional dependency and this path must not acquire
# one. The field list is the thing that has to agree, and a test asserts it
# does.
def _tenant_of(user_api_key_dict, data) -> str | None:
    """Whose traffic this is. Delegates, so the live hook and the loader cannot
    answer differently for one row -- they did, and tenant is part of the cache
    isolation scope."""
    return resolve_litellm_tenant(data, key=user_api_key_dict)


def litellm_handler(plugin: CachePlugin, *, base=None, session_from=None,
                    agent_from=None, mutate: bool = False,
                    target_id: str | None = None):
    """A LiteLLM proxy callback that places markers on outgoing requests.

    Signature checked against LiteLLM's proxy hook documentation on 29 Jul 2026:
    the proxy awaits `async_pre_call_hook(user_api_key_dict, cache, data,
    call_type)` as a method on a `CustomLogger` subclass, and uses the returned
    dict as the request. It is **proxy-only** -- library-mode LiteLLM has no hook
    that can modify a request.

    Returns an *instance*, ready for `callbacks: my_module.handler` in
    config.yaml. An earlier version returned a plain synchronous closure, which
    matched none of that: the proxy would have awaited a dict, or not found the
    method at all. The docstring above it described the correct shape while the
    code beneath it did something else, which is the failure this project keeps
    finding in other people's tools.

    `CustomLogger` is subclassed when LiteLLM is importable and skipped when it
    is not, so the harness keeps its standard-library-only guarantee and the
    object is still the type the proxy expects wherever it actually runs.

    **Observes by default; mutates only when asked.** `mutate=False` runs the
    whole learning path -- segmentation, change rates, gaps, every alert the
    monitor raises -- and hands LiteLLM back the request it was given. That half
    is verified here and cannot break a request, because it changes nothing.

    `mutate=True` puts `cache_control` on the wire. That half was unverified
    for most of this file's life, on the specific worry that LiteLLM normalises
    Anthropic-shaped bodies through an OpenAI-shaped intermediate and might drop
    the field -- which would mean churn, no cache writes, and effectiveness
    counters reporting confidently on placements that never arrived.

    It has now been watched. `tier-b/litellm_marker_survival.py` sends four real
    calls through litellm 1.83.9 and reads the provider's own counters back: an
    unmarked control writes nothing, a marked call writes 15,624 tokens, the
    same body a moment later reads 15,624, and a request that went through this
    handler with `mutate=True` also writes. Evidence in
    `tier-b/evidence/litellm-marker-survival.json`, and the script starts from a
    cold cache on every run so it can be repeated.

    So the marker survives, and the entry it writes is readable. `mutate` still
    defaults to False, because that is a decision about rewriting somebody's
    live traffic rather than a question about whether the mechanism works, and
    the answer to the second one does not settle the first.

    Markers are placed on `messages` only. LiteLLM carries OpenAI-shaped tool
    definitions and translates them, and whether a `cache_control` key survives
    that translation is exactly the kind of thing this project does not assert
    without checking. The cost is that tool definitions -- usually the largest
    stable block there is -- are covered only when a marker sits above them.
    """
    if base is None:
        base = _custom_logger_base()

    class CacheEconomicsHandler(base):
        """Placement on the way out; usage folded back in on the way home."""

        def __init__(self):
            try:
                super().__init__()
            except TypeError:                      # object() takes no arguments
                pass
            self.plugin = plugin
            self.mutate = mutate
            # An operator override for proxies that do not name the provider in
            # either place the resolver looks. None means "resolve per request".
            self.target_id = target_id
            self._failures = 0
            self.session_from = session_from or default_session_from
            self.agent_from = agent_from or default_agent_from
            self._pending = {}

        async def async_pre_call_hook(self, user_api_key_dict, cache, data,
                                      call_type):
            """Fail open, always.

            This sits in front of a live request. Anything it raises is an
            error the caller's own API call never made, and there is no input
            from a proxy's traffic worth breaking a request over -- least of
            all in the observe-only default, where the plugin is not even
            supposed to change anything.

            Measured before this guard existed: `{"model": {"bad": "..."}}`
            raised TypeError out of model normalisation, and a message whose
            content is an integer raised out of segmentation. Both with
            mutate=False. A schema this code has never seen is LiteLLM's
            business, not a reason to take down somebody's completion.

            The narrow exceptions are deliberate: no `except BaseException`, so
            a cancellation or a KeyboardInterrupt still propagates.
            """
            try:
                return await self._pre_call(user_api_key_dict, cache, data,
                                            call_type)
            except (LookupError, TypeError, ValueError, AttributeError,
                    ArithmeticError, registry.RegistryError) as e:
                self._failures += 1
                _log.warning("cacheeconomics: pre-call hook failed open (%s: %s)",
                             type(e).__name__, e)
                return data

        async def _pre_call(self, user_api_key_dict, cache, data, call_type):
            if call_type not in ("completion", "text_completion"):
                return data
            messages = data.get("messages")
            if not messages:
                return data
            # The whole prompt, not just the part that gets patched. Tool
            # definitions precede messages on the wire, so a marker on a
            # message caches a prefix containing them. Passing messages alone
            # meant volatile tools were invisible to the change rates, and the
            # plugin would keep placing markers from a perfectly stable
            # messages-only view of a prefix that was being rewritten every
            # request.
            prompt = {"messages": messages}
            if data.get("tools"):
                prompt["tools"] = data["tools"]
            # Anthropic's top-level `system` precedes everything on the wire.
            # Leaving it out meant a volatile system header sat above every
            # marker this placed while the plugin believed the prefix was
            # stable -- optimising against a prompt that was not the one sent.
            if data.get("system"):
                prompt["system"] = data["system"]
            # The surface this request is actually bound for, read from the
            # same place the batch adapter reads it. Omitting it let
            # `on_request` default to anthropic/direct, so on a proxy fronting
            # Bedrock the minimums, the TTLs and the breakpoint budget were all
            # computed against the wrong provider -- and this is the path that
            # rewrites the request rather than merely reporting on it, where
            # being wrong produces a provider error.
            from .adapters.litellm import target_from_row
            # No guess, on either half. This substituted `anthropic/direct` for
            # an unresolvable surface, on the reasoning that a wrong guess
            # "surfaces as a provider error on the next call" — which is only
            # half true, and the wrong half. Ordering a mixed request wrongly
            # does error on Bedrock. A *minimum* guessed too low does not: the
            # provider processes the request uncached, writes nothing, returns
            # no error, and the bill looks ordinary. That is the same silent
            # shape this project's own LiteLLM disclosure is about, and here it
            # would be arriving on somebody's production traffic because we
            # patched their request.
            #
            # So an unresolvable surface means observe only. The markers this
            # request does not get are worth less than a mutation made against
            # minimums, TTLs and a breakpoint budget belonging to a provider it
            # is not going to.
            target_id = target_from_row(data, self.target_id)
            unattributed = target_id == UNATTRIBUTED
            out, decision = self.plugin.on_request(
                prompt, model=data.get("model", ""),
                target_id=target_id,
                # The same precedence the batch adapter uses, read from both
                # places LiteLLM puts it. This read three fields where the
                # adapter reads six, so `{"user_api_key_team_id": "team-a"}`
                # resolved to None and two teams shared one `(None, target,
                # model)` scope -- for caches the provider keeps apart.
                tenant=_tenant_of(user_api_key_dict, data),
                # The conversation, not the call. These are different ids and
                # only one of them is stable across turns.
                session=self.session_from(data),
                agent=self.agent_from(data),
                # Positions, not roles. Only the message content blocks are a
                # shape this project has reasoned about; a top-level `system`
                # field goes through the same unverified translation as `tools`.
                markable=markable_positions(prompt),
                # Planning without sending has to be the plugin's decision, not
                # a `return data` after it has already recorded the placement.
                # And never applied against a surface nobody named.
                apply=self.mutate and not unattributed)
            # Said once per scope, not per request. A handler that stands down
            # silently is indistinguishable from one that found nothing worth
            # marking, and an operator who turned `mutate` on has every reason
            # to expect markers.
            if unattributed and self.mutate:
                self.plugin._record_alert(Alert(
                    "RT-UNATTRIBUTED", "medium",
                    (_tenant_of(user_api_key_dict, data), UNATTRIBUTED),
                    "mutation is standing down: this request names no provider",
                    "Markers are placed against a surface's minimum cacheable "
                    "prefix, its supported lifetimes and its breakpoint budget, "
                    "and none of those are known here. A minimum guessed too low "
                    "is not an error: the provider processes the request "
                    "uncached, writes nothing and bills normally, so nothing "
                    "would report it. Observation continues.",
                    subject="unattributed",
                    fix="Set `target_id` on the handler, or configure LiteLLM to "
                        "pass `custom_llm_provider` on the request."))
            # The call id correlates this request with its own response, and
            # nothing else.
            key = data.get("litellm_call_id")
            if key is not None:
                self._pending[key] = decision
                # A response that never arrives must not pin a decision here
                # forever. This runs inside somebody's proxy.
                while len(self._pending) > 256:
                    self._pending.pop(next(iter(self._pending)), None)
            if not decision.applied:
                # Observation still happened above, against the request as it
                # actually goes out: the monitor has the shape, the rates and
                # the gaps, so the diagnostics work. What is withheld is the
                # part that rewrites somebody's request over an unverified wire
                # path, and `apply=self.mutate` above means the plugin knows it
                # was withheld rather than being told after the fact.
                return data
            # Every field the plugin patched that can carry a marker, not just
            # the ones it places on.
            #
            # Markers are never *placed* on tools: whether `cache_control`
            # survives LiteLLM's translation of OpenAI-shaped tool definitions
            # is not something this project asserts without checking, and
            # `markable` above makes the allocator work around that. But
            # override *strips*, and returning only messages left the caller's
            # four tool markers on the wire beside the plugin's new one --
            # five against a budget of four, a provider error produced by the
            # one path that rewrites live requests.
            #
            # Removing a key is safe under any translation, which is exactly
            # why a stripped `tools` can be returned where a placed one could
            # not.
            patched = {**data, "messages": out["messages"]}
            for extra in ("system", "tools"):
                if extra in out:
                    patched[extra] = out[extra]
            # Re-checked against the dict actually handed to LiteLLM, because
            # the budget check inside `_decide` saw a different object.
            try:
                # The resolved surface, not a literal. Bedrock and Vertex do not
                # necessarily share Anthropic's breakpoint budget, and this
                # guard is the last thing standing between a miscount and a
                # request the provider rejects.
                budget = registry.capability(target_id, "max_breakpoints")
            except registry.RegistryError:
                budget = None
            if budget:
                on_wire = marker_count(patched)
                if on_wire > budget:
                    # The original body goes out, so nothing may still be
                    # recorded as applied. `_pending` was written above, and
                    # `async_log_success_event` hands whatever is there to
                    # `on_response`, which credits the response's cache reads and
                    # writes to this decision's placements -- markers that in
                    # this branch never left the process. That is a metric
                    # attributing provider behaviour to a request nobody sent,
                    # and it feeds the next placement decision.
                    #
                    # Replaced rather than deleted: a missing key makes
                    # `on_response` return silently, which loses the usage the
                    # monitor's rebuild and fan-out checks need. What has to go
                    # is the claim that markers were applied, not the
                    # observation that a request happened.
                    # Into `proposed`, which exists for this exact distinction:
                    # "what would have been placed" is worth keeping, and the
                    # field's own docstring records that reporting it as
                    # `placements` is what once had the effectiveness counters
                    # crediting markers that never reached the provider.
                    vetoed = replace(
                        decision, applied=False, placements={},
                        proposed=dict(decision.placements or {}),
                        reason=(f"the patched request would carry {on_wire} markers "
                                f"against a budget of {budget}, counted on the body "
                                f"handed to LiteLLM"))
                    if key is not None:
                        self._pending[key] = vetoed
                    return data
            return patched

        async def async_log_success_event(self, kwargs, response_obj,
                                          start_time, end_time):
            """Feed the provider's counters back, so the usage-driven checks
            can fire. Without this the plugin places markers and never learns
            whether a prefix is being rebuilt."""
            decision = self._pending.pop(kwargs.get("litellm_call_id"), None)
            if decision is None:
                return
            self.plugin.on_response(decision, response_obj)

    return CacheEconomicsHandler()


def _marker_blocks(value):
    """Every dict in a request field that could carry `cache_control`."""
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        if not isinstance(item, dict):
            continue
        out.append(item)
        content = item.get("content")
        if isinstance(content, list):
            out.extend(b for b in content if isinstance(b, dict))
    return out


def _custom_logger_base():
    """LiteLLM's `CustomLogger` if it is installed, otherwise a plain object.

    Imported here rather than at module scope so the package keeps working --
    and keeps bundling for the browser -- with no third-party dependency.
    """
    try:
        from litellm.integrations.custom_logger import CustomLogger
        return CustomLogger
    except Exception:
        return object
