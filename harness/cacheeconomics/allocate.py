"""Allocator-lite: where to put cache markers, and which lifetime to ask for.

This is the Gate 1 instrument. Its job is to answer one question honestly:
does deliberate marker placement beat LiteLLM's automatic injection by enough
to be worth anything?

Deliberately NOT here:

  Relocation. Moving a volatile segment out of the prefix is the single
  highest-value transform available, and including it would confound the
  bake-off: a win could come from better markers or from moved content and
  nothing would distinguish them. It is reported as a separate opportunity
  instead, so the comparison measures placement and only placement.

  Tier merging under a constrained budget. That is the full allocator, which
  stays gated behind the result of this measurement.

Three policies, so the comparison has a competent baseline rather than a
strawman. Beating "no caching at all" would prove nothing.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from statistics import median

from . import registry
from .trace import Request, Segment


@dataclass
class Plan:
    """Where markers go for one request, and in what order the segments ship."""
    policy: str
    marker_indices: list[int] = field(default_factory=list)
    ttls: dict = field(default_factory=dict)          # index -> "5m" | "1h"
    notes: list[str] = field(default_factory=list)
    order: list[int] | None = None                    # emission order; None = as authored
    allocation: object | None = None                  # full allocator's tier solution

    def emission(self, segs: list[Segment]) -> list[Segment]:
        """Segments in the order they go on the wire."""
        if self.order is None:
            return sorted(segs, key=lambda s: s.index)
        by_i = {s.index: s for s in segs}
        moved = [by_i[i] for i in self.order if i in by_i]
        rest = [s for s in sorted(segs, key=lambda s: s.index) if s.index not in set(self.order)]
        return moved + rest

    def _cut(self, segs: list[Segment]) -> list[Segment]:
        """Everything up to and including the outermost marker.

        Outermost is by position on the wire, not by segment index. Once
        relocation is in play those differ, and the cache boundary follows the
        wire.
        """
        if not self.marker_indices:
            return []
        em = self.emission(segs)
        marked = {i for i in self.marker_indices}
        pos = [p for p, s in enumerate(em) if s.index in marked]
        return em[:max(pos) + 1] if pos else []

    def prefixes(self, segs: list[Segment]) -> list[tuple]:
        """One candidate cache entry per marker, shortest first.

        Multiple breakpoints are not one cache span. Each marks its own prefix,
        and the provider reads the longest one that is still alive, then writes
        the rest. Modelling only the outermost marker makes a policy that pairs
        a stable system breakpoint with an advancing trailing-turn breakpoint
        look like it never hits at all — which is a defamatory way to model a
        baseline, and would have handed us a fabricated win on this fixture.

        Returns (segment-id tuple, cumulative tokens, ttl) per marker.
        """
        marked = set(self.marker_indices)
        out, ids, toks = [], [], 0
        for s in self.emission(segs):
            ids.append(s.id)
            toks += s.tokens
            if s.index in marked:
                out.append((tuple(ids), toks, self.ttls.get(s.index, "5m")))
        return out

    def cached_prefix(self, segs: list[Segment]) -> tuple:
        """The outermost marker's prefix. Kept for callers that want one span."""
        return tuple(s.id for s in self._cut(segs))

    def cached_tokens(self, segs: list[Segment]) -> int:
        return sum(s.tokens for s in self._cut(segs))

    def effective_ttl(self, segs: list[Segment] | None = None) -> str:
        """The lifetime governing the outermost marker, which is what expires."""
        if not self.marker_indices:
            return "5m"
        if segs is None:
            return self.ttls.get(max(self.marker_indices), "5m")
        cut = self._cut(segs)
        return self.ttls.get(cut[-1].index, "5m") if cut else "5m"


# --- policies --------------------------------------------------------------

def as_shipped(req: Request, **_) -> Plan:
    """Whatever the traced code actually did. Arm A.

    A marked segment with no lifetime of its own falls back to the row's
    `ttl_requested` before the provider default. The analyzer already accepts
    that field as proof of lifetime, so defaulting straight to 5m here priced a
    1h trace as 1h in spend and replayed it as 5m in the bake-off -- fabricating
    expiries and reporting savings against a baseline nobody shipped.
    """
    ttls = req.ttl_by_marker_index()
    return Plan("as-shipped", sorted(ttls), ttls)


def litellm_auto(req: Request, max_markers: int | None = None,
                 injection_points: list | None = None, **_) -> Plan:
    """What LiteLLM actually does to a request's cache markers.

    Which is nothing, unless an operator configured injection points.

    This used to model `enable_anthropic_prompt_caching`: a checkpoint on the
    system prompt plus one on the advancing trailing turn, always 5m. Its own
    docstring said that came from the published description rather than from
    reading the source, and the bake-off arm inherited the caveat. A review
    challenged it, so both were checked.

    litellm 1.83.9, read and captured 2026-08-03:

    - `integrations/anthropic_cache_control_hook.py` is the only injection
      path. `AnthropicCacheControlHook.get_chat_completion_prompt` pops
      `cache_control_injection_points` and returns `model, messages,
      non_default_params` unchanged when the list is empty. No role heuristic,
      no automatic placement.
    - Every "automatic" in the Anthropic transformation
      (`llms/anthropic/chat/transformation.py:374`,
      `llms/anthropic/common_utils.py:399,433`) reads "prompt caching now works
      automatically *when cache_control is used in messages*". That is
      automatic header handling for markers the caller supplied, not automatic
      marker placement.
    - Confirmed on the wire rather than by reading alone: a completion routed
      through `tier-b/capture_proxy.py` with no injection points configured,
      carrying a 25k system prompt and a tool, sent zero `cache_control` keys
      anywhere in the body.

    So the old model invented a baseline that does not exist, and it was wrong
    in both directions depending on the input. On an unmarked request it placed
    two markers where LiteLLM places none, making the baseline look better than
    reality. On a request already carrying a 1h marker it replaced that with two
    5m ones, making it look worse -- and a baseline that looks worse than the
    real thing flatters whatever we compare against it, which is the direction
    that should never be guessed at.

    With injection points supplied this models them: each point contributes a
    marker, the caller's own `control` decides the lifetime, and a segment the
    caller already marked is left alone, because `_safe_insert_cache_control_in_message`
    writes into the message rather than adding a second entry.
    """
    if max_markers is None:
        max_markers = registry.capability(req.target_id, "max_breakpoints") or 4

    shipped = req.ttl_by_marker_index()
    if not injection_points:
        # The default configuration. LiteLLM forwards the caller's markers and
        # adds none of its own, so this arm is the request as it was sent.
        return Plan("litellm-auto", sorted(shipped), dict(shipped),
                    ["LiteLLM adds no cache markers unless "
                     "`cache_control_injection_points` is configured, so this "
                     "arm is the request as shipped. Verified against litellm "
                     "1.83.9 by reading the injection hook and by capturing a "
                     "request on the wire."])

    by_index = {s.index: s for s in req.segments}
    ttls = dict(shipped)
    added = []
    for point in injection_points:
        i = point.get("index") if isinstance(point, dict) else None
        if not isinstance(i, int):
            continue
        if i < 0:
            i += len(req.segments)
        if i not in by_index or i in shipped:
            # Already marked by the caller: LiteLLM writes into the existing
            # message rather than adding a second marker.
            continue
        control = (point.get("control") or {}) if isinstance(point, dict) else {}
        ttls[i] = control.get("ttl") or "5m"
        added.append(i)
        if len(ttls) >= max_markers:
            break

    notes = [f"{len(added)} marker(s) injected at configured points; "
             f"{len(shipped)} already carried one and were left alone"]
    if len(ttls) >= max_markers:
        notes.append(f"stopped at the {max_markers}-marker limit for "
                     f"{req.target_id}, counting the caller's own markers")
    return Plan("litellm-auto", sorted(ttls), ttls, notes)


def allocator_lite(req: Request, volatility: dict | None = None,
                   cadence_seconds: float | None = None,
                   max_markers: int | None = None, **_) -> Plan:
    """Place markers at the deepest genuinely stable boundary, and pick a TTL.

    Two decisions, both cheap and both things automatic injection cannot make:

    1. Put the marker after the last segment observed to be stable, rather than
       after the last system-role segment. A volatile segment sitting among the
       system blocks makes everything behind it uncacheable, and role is not a
       proxy for stability.

    2. Choose the lifetime from the observed gap between requests. The one-hour
       cache only pays between five minutes and an hour; outside that band the
       cheaper write wins, so this is not a blanket upgrade.
    """
    segs = sorted(req.segments, key=lambda s: s.index)
    return _place(req, segs, "allocator-lite", volatility or {}, cadence_seconds)


def _place(req: Request, emission: list[Segment], policy: str,
           volatility: dict, cadence_seconds: float | None,
           order: list[int] | None = None, extra_notes: list[str] | None = None) -> Plan:
    """Deepest stable boundary in an already-ordered segment list, plus a TTL.

    Shared by allocator-lite and relocation-lite. The only difference between
    them is what order they hand in.
    """
    notes = list(extra_notes or [])

    # One unstable segment invalidates everything after it, so this scans
    # forward and stops at the first thing that moved.
    boundary = None
    for s in emission:
        if volatility.get(s.index, 1) > 1:      # >1 distinct value observed
            notes.append(f"segment {s.index} ({s.label or s.role}) changes between "
                         f"requests; the cacheable prefix stops before it")
            break
        boundary = s.index

    if boundary is None:
        return Plan(policy, [], {},
                    notes + ["nothing stable to cache: the first segment already varies"],
                    order)

    # Never mark a prefix too short to cache. Below the minimum the provider
    # processes it uncached and returns no error, so the marker is pure cost.
    cut, tokens = [], 0
    for s in emission:
        cut.append(s)
        tokens += s.tokens
        if s.index == boundary:
            break
    try:
        minimum = registry.min_cacheable_tokens(req.target_id, req.model)
    except registry.RegistryError as e:
        # Fail closed, as `tiers.allocate` and the plugin's filter already do.
        # This is the fourth implementation of the same guard on this branch and
        # the last one still failing open: a direct caller of allocator_lite or
        # relocation_lite got a marker recommendation for a model whose
        # threshold nobody knows. Below it the provider caches nothing and
        # returns no error, so the marker is a write premium paid for silence.
        return Plan(policy, [], {},
                    notes + [f"no marker placed: no recorded minimum cacheable "
                             f"prefix for {req.model} on {req.target_id}, so none "
                             f"can be shown to cache anything ({e})"], order)
    if tokens < minimum:
        return Plan(policy, [], {},
                    notes + [f"stable prefix is {tokens:,} tokens, under the "
                             f"{minimum:,} minimum for {req.model}; marking it would "
                             f"cost without caching"], order)

    # What the surface and model will actually accept. Cadence chose the
    # lifetime on its own, so a Bedrock model the registry narrows to 5m got a
    # 1h marker and a note saying the longer lifetime pays -- a recommendation
    # the provider rejects, or silently ignores, on the one output of this tool
    # that gets applied to somebody's production prompt rather than published.
    try:
        supported = list(registry.supported_ttls(req.target_id, req.model))
    except registry.RegistryError:
        # Nothing recorded is not permission. The short lifetime is the one
        # every explicit-breakpoint surface has, so it is the safe floor.
        supported = ["5m"]
        notes.append(f"no supported lifetimes recorded for {req.model} on "
                     f"{req.target_id}, so only the 5m write is assumed")

    ttl = "5m" if "5m" in supported else (supported[0] if supported else "5m")
    if cadence_seconds is not None:
        in_band = 300 < cadence_seconds < 3600
        if in_band and "1h" in supported:
            ttl = "1h"
            notes.append(f"median gap {cadence_seconds/60:.1f} min falls inside the "
                         f"one-hour window, so the longer lifetime pays")
        elif in_band:
            notes.append(f"median gap {cadence_seconds/60:.1f} min falls inside the "
                         f"one-hour window, but {req.model} on {req.target_id} "
                         f"supports only {', '.join(supported)}, so the longer "
                         f"lifetime is not available here")
        else:
            notes.append(f"median gap {cadence_seconds/60:.1f} min is outside the "
                         f"5min-1hr window, so the cheaper 5m write wins")

    return Plan(policy, [boundary], {boundary: ttl}, notes, order)


POLICIES = {"as-shipped": as_shipped,
            "litellm-auto": litellm_auto,
            "allocator-lite": allocator_lite}


# --- inputs the allocator needs, derived from the trace ---------------------

def observed_volatility(reqs: list[Request]) -> dict:
    """How many distinct values each segment took *within one reuse chain*.

    Observed, not declared. Developers do not hand-write annotations, and what
    a segment actually did across a window is better evidence than what someone
    believed it would do.

    Counted per reuse chain. Cache isolation is tenant/surface/model, but prompt
    stability is a sequence property: two sessions with different, internally
    stable headers are two prefixes, not one prefix changing.

    This function returns the *fail-closed* reduction -- the worst reuse chain --
    for callers that must produce one answer covering all traffic, such as a
    relocation plan applied to a whole group. Anything that can act per request
    should use `observed_volatility_by_chain` and look up the chain it is
    actually serving.
    """
    return {i: max(len(v) for v in chains.values())
            for i, chains in _by_index_and(reqs, reuse_chain_of).items()}


def pool_of(r: Request) -> tuple:
    """The cache isolation scope a request belongs to."""
    return (r.tenant, r.target_id, r.model)


def reuse_chain(pool: tuple, agent: str | None = None,
                session: str | None = None) -> tuple:
    """The prompt reuse chain inside a cache pool.

    Cache entries are content-addressed and isolated by pool, but volatility is a
    question about a sequence of requests that could reuse one another's prefix.
    Two sessions with different, internally stable headers are two stable
    prefixes, not one volatile prefix. When no session is known, the pool is the
    narrowest honest grouping available.
    """
    return (*pool, agent or "unknown", session) if session else pool


def reuse_chain_of(r: Request) -> tuple:
    return reuse_chain(pool_of(r), r.agent, r.session)


def _by_index_and(reqs: list[Request], key_of) -> dict:
    """Distinct values per position per group, counting absence as a value.

    A position that is present on half the requests took one id and a gap, and
    a gap moves everything behind it exactly as an edit does. Recording only
    what was there gave an optional trailing block a count of 1 -- perfectly
    stable -- so `_place` was free to make a vanishing section the cache
    boundary, splitting the entry and putting the write premium on the part
    that disappears.

    `observed_change_rates_by_pool` has counted absence since it was written.
    This is the same fact, counted the other way, and the two disagreed.
    """
    groups = defaultdict(set)
    indices = defaultdict(set)
    for r in reqs:
        group = key_of(r)
        groups[group].add(r.request_id)
        for s in r.segments:
            indices[group].add(s.index)

    seen = defaultdict(lambda: defaultdict(set))
    for r in reqs:
        group = key_of(r)
        present = {s.index: s.id for s in r.segments}
        for i in indices[group]:
            seen[i][group].add(present.get(i))       # None marks absence
    return seen


def observed_volatility_by_pool(reqs: list[Request]) -> dict:
    """`{pool: {segment index: distinct values seen in that pool}}`.

    The cache-isolation view. Use `observed_volatility_by_chain` for prompt
    stability and placement decisions when session ids are available.
    """
    out = defaultdict(dict)
    for i, pools in _by_index_and(reqs, pool_of).items():
        for pool, ids in pools.items():
            out[pool][i] = len(ids)
    return dict(out)


def observed_volatility_by_chain(reqs: list[Request]) -> dict:
    """`{reuse chain: {segment index: distinct values seen in that chain}}`."""
    out = defaultdict(dict)
    for i, chains in _by_index_and(reqs, reuse_chain_of).items():
        for chain, ids in chains.items():
            out[chain][i] = len(ids)
    return dict(out)


def _ordered_groups(reqs: list[Request], key_of) -> dict:
    out = defaultdict(list)
    for r in reqs:
        out[key_of(r)].append(r)
    for rs in out.values():
        rs.sort(key=lambda r: (r.sent_at is None, r.sent_at))
    return out


def observed_change_rates_by_pool(reqs: list[Request]) -> dict:
    """`{pool: {segment index: share of requests on which it changed}}`.

    The cache-isolation view. Use `observed_change_rates_by_chain` for prompt
    stability and placement decisions when session ids are available.

    A rate, not the count of distinct values `observed_volatility` returns.
    Those answer different questions and only one of them is any use for
    deciding where a marker goes: a field alternating between two states
    invalidates the prefix on every single request and shows two values, while a
    field with twenty values that changes once a week shows twenty. Ranking by
    distinct values puts them in the wrong order.

    The binary use in allocator-lite -- "did this ever change" -- is unaffected
    and stays on the count.

    A segment missing from a request counts as a change, because appearing and
    disappearing moves everything behind it just as surely as editing it does.
    """
    return _observed_change_rates_by(reqs, pool_of)


def observed_change_rates_by_chain(reqs: list[Request]) -> dict:
    """`{reuse chain: {segment index: share of requests on which it changed}}`."""
    return _observed_change_rates_by(reqs, reuse_chain_of)


def _observed_change_rates_by(reqs: list[Request], key_of) -> dict:
    out = defaultdict(dict)
    for group, rs in _ordered_groups(reqs, key_of).items():
        # The index set is computed once per group. Rebuilding it inside the
        # request loop made this quadratic, which is invisible on a fixture and
        # is 75 million comparisons on the 8,700-request trace this is meant to
        # run against.
        indices = {s.index for r in rs for s in r.segments}
        seen = defaultdict(list)
        for r in rs:
            by_i = {s.index: s.id for s in r.segments}
            for i in indices:
                seen[i].append(by_i.get(i))
        for i, vals in seen.items():
            if len(vals) < 2:
                out[group][i] = 0.0
                continue
            changes = sum(1 for a, b in zip(vals, vals[1:]) if a != b)
            out[group][i] = changes / (len(vals) - 1)
    return dict(out)


def observed_change_rates(reqs: list[Request]) -> dict:
    """Fail-closed reduction of the above: the worst reuse chain, per segment.

    Same reasoning as `observed_volatility`. A plan applied to every request
    regardless of reuse chain has to survive the chain where the segment churns
    most, or it places a marker that rewrites a prefix nothing will read.
    """
    per_pool = observed_change_rates_by_chain(reqs)
    worst: dict = {}
    for rates in per_pool.values():
        for i, rate in rates.items():
            worst[i] = max(worst.get(i, 0.0), rate)
    return worst


def observed_gaps_by_pool(reqs: list[Request]) -> dict:
    """`{pool: [seconds between consecutive requests]}`.

    The distribution, not a median. Agent traffic is routinely bimodal -- a
    burst of turns inside a session, then hours until the next one -- and a
    median over both modes describes neither. Choosing a lifetime off it buys a
    2x write premium for entries that expire unread.
    """
    return _observed_gaps_by(reqs, pool_of)


def observed_gaps_by_chain(reqs: list[Request]) -> dict:
    """`{reuse chain: [seconds between consecutive requests]}`."""
    return _observed_gaps_by(reqs, reuse_chain_of)


def _observed_gaps_by(reqs: list[Request], key_of) -> dict:
    out = {}
    for group, rs in _ordered_groups(reqs, key_of).items():
        ts = [r.sent_at for r in rs if r.sent_at]
        out[group] = [(b - a).total_seconds() for a, b in zip(ts, ts[1:])
                      if (b - a).total_seconds() >= 0]
    return out


def observed_cadence(reqs: list[Request]) -> float | None:
    """Median seconds between consecutive requests in this group."""
    ts = sorted(r.sent_at for r in reqs if r.sent_at)
    if len(ts) < 3:
        return None
    return median((b - a).total_seconds() for a, b in zip(ts, ts[1:]))
