"""How to cut a prompt into cache tiers when markers are scarce.

Allocator-lite places one marker at the deepest stable boundary and stops. That
is the right instrument for Gate 1 -- it isolates placement -- and it leaves
three of the four markers unused and everything past the first volatile segment
uncacheable. The full allocator answers the harder question: given a fixed
marker budget, where do the cuts go?

The model, stated so it can be argued with
---------------------------------------------

Markers at wire positions p_1 < ... < p_k cut the prompt into blocks. Block i is
the tokens between marker i-1 and marker i. Per request:

    cost = sum_i  B_i * ( a_i * read + (1 - a_i) * write_i )  +  tail

where B_i is the block's tokens, a_i is the probability that block is served
from cache, `write_i` is the multiplier for that entry's lifetime, and `tail` is
everything past the last marker, which is always plain input. `a_i` is the
product of two independent things:

  **Content survival.** The prefix through p_i matches only if nothing in it
  changed. Estimated as the product of per-segment change rates observed in the
  trace. This assumes segments change independently, which is not true -- a
  deploy changes several at once -- and correlated changes make the real hit
  rate *higher* than modelled, so the assumption is conservative in the
  direction of understating the benefit.

  **Aliveness.** The entry survives only if the next request arrives inside the
  lifetime. Estimated as the share of observed inter-request gaps shorter than
  the TTL, taken from the trace rather than from a median. A workload with a
  bimodal cadence -- bursts inside a session, hours between sessions -- has a
  median that describes neither mode, and picking a lifetime off it is how you
  buy a 2x write premium for entries that expire unread.

**Everything here is denominated in tokens, never dollars.** The objective is a
ratio between plans, so the price cancels. Converting to money is the reporting
layer's job, behind the reconciliation gate, and an optimiser that emitted
dollars would route around it.

One deliberate restriction
--------------------------

The search assigns one lifetime to the whole plan. Mixed lifetimes are legal --
Anthropic reports the write split precisely because a request can create both --
but the moment two markers hold different TTLs, which entry serves a read
depends on the gap, the blocks stop being independent, and the search is no
longer a shortest-path problem. So the DP runs once per uniform lifetime, and
the handful of mixed patterns worth having (a long-lived prefix under a
short-lived advancing turn) are scored afterwards by `expected_cost`, which
models the gap exactly and makes no independence claim. The optimum over a
restricted candidate set is a lower bound on what is achievable, not a proof of
optimality, and `Allocation.searched` says which candidates were considered.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import registry

_TTL = re.compile(r"^(\d+)([smh])$")
_UNIT = {"s": 1, "m": 60, "h": 3600}


class Unsupported(Exception):
    """This surface cannot be allocated for, and guessing would be worse."""


def ttl_seconds(ttl: str) -> int:
    m = _TTL.match(ttl or "")
    if not m:
        raise Unsupported(f"unrecognised cache lifetime {ttl!r}")
    return int(m.group(1)) * _UNIT[m.group(2)]


@dataclass(frozen=True)
class Tier:
    """One block of the prompt sharing a cache entry."""
    marker_position: int      # wire position of the marker closing this tier
    first_position: int       # wire position this tier starts at
    tokens: int               # tokens in the block, not in the whole prefix
    prefix_tokens: int        # cumulative tokens through the marker
    ttl: str
    hit_probability: float
    segment_indices: tuple = ()


@dataclass(frozen=True)
class Allocation:
    tiers: list = field(default_factory=list)
    expected_cost: float = 0.0        # token-equivalents per request
    uncached_cost: float = 0.0        # same units, no caching at all
    searched: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    @property
    def marker_positions(self) -> list:
        return [t.marker_position for t in self.tiers]

    @property
    def saving_ratio(self) -> float | None:
        """Share of input cost removed. Negative if the plan costs more."""
        if not self.uncached_cost:
            return None
        return (self.uncached_cost - self.expected_cost) / self.uncached_cost


# A cache write is readable once the response carrying it has returned. Requests
# dispatched closer together than this could not have seen each other's entry --
# that is the cold fan-out case, and counting those gaps as hits is how a model
# fabricates them. Matches the pessimistic assumption the simulator replays under.
WRITE_LATENCY_SECONDS = 5.0


def survival(gaps, ttl: str, write_latency: float = WRITE_LATENCY_SECONDS) -> float:
    """Share of observed gaps on which an entry is both written and still alive.

    Bounded at both ends, and the lower bound is the one that is easy to forget.
    A gap of zero is not a certain hit, it is a certain miss: two requests
    dispatched simultaneously each pay to write the same prefix and neither can
    read the other's. Counting the interval as `g < ttl` rated a burst of
    concurrent requests as a perfect cache.

    With no gaps observed this returns 0.0 rather than a default. A hit rate
    invented for a workload nobody timed is the assumption that flatters every
    caching tool, and the allocator would then propose markers on the strength
    of it.
    """
    if not gaps:
        return 0.0
    limit = ttl_seconds(ttl)
    return sum(1 for g in gaps if write_latency <= g < limit) / len(gaps)


def _surface(target_id: str, model: str | None = None):
    """Budget, lifetimes and multipliers for a target, or a refusal.

    Applicability is checked before pricing, in that order and deliberately.
    Asking for multipliers first makes an implicit-prefix surface refuse with
    "records no multipliers", which is true and is not the reason: the reason is
    that there is nothing to place. `cost.ttl_crossover` carries the same note
    because the same mistake was made there first.
    """
    try:
        row = registry.target(target_id)
    except registry.RegistryError as e:
        raise Unsupported(str(e)) from e
    caps = row.get("capabilities", {})
    if not caps.get("explicit_breakpoints"):
        raise Unsupported(
            f"{target_id} caches implicitly and takes no markers, so there is "
            f"nothing for an allocator to place. Prompt ordering still matters "
            f"and relocation still applies.")
    try:
        mult = registry.multipliers(target_id)
    except registry.RegistryError as e:
        raise Unsupported(str(e)) from e
    budget = caps.get("max_breakpoints")
    if not budget:
        raise Unsupported(
            f"{target_id} records no marker budget "
            f"(max_breakpoints={budget!r}), and inventing one would put a "
            f"guess inside a recommendation.")
    # Narrowed to the model. The surface-wide list is the *maximum*: Bedrock
    # advertises 5m and 1h while its own provenance limits 1h to three named
    # models, which the registry now records per model. Reading the wide list
    # here made the allocator plan a 1h tier for claude-opus-4-1 on Bedrock --
    # a lifetime `registry.supported_ttls` says that model does not have -- so
    # the bake-off would report a saving the provider cannot deliver and the
    # live plugin would put the marker on the wire.
    #
    # `checks` learned this last round and this did not, which is the same twin
    # one module over.
    ttls = list(registry.supported_ttls(target_id, model))
    if not ttls:
        raise Unsupported(
            f"{target_id} records no supported cache lifetimes"
            + (f" for {model}" if model else ""))
    if not isinstance(mult.get("read"), (int, float)):
        raise Unsupported(
            f"{target_id} does not record an Anthropic-shaped read multiplier, "
            f"so this cost model cannot compare plans on it.")
    rates = {}
    for t in ttls:
        key = f"write_{t}"
        if isinstance(mult.get(key), (int, float)):
            rates[t] = mult[key]
    if not rates:
        raise Unsupported(
            f"{target_id} records lifetimes {ttls} but no matching write "
            f"multipliers, so the write premium cannot be priced.")
    return budget, rates, float(mult["read"])


def expected_cost(blocks, ttls, read_rate, write_rates, gaps, survivals) -> float:
    """Exact expected cost of a marker plan, in token-equivalents.

    `blocks` is [(block_tokens, content_survival_of_prefix)] per marker in wire
    order. Makes no independence assumption across markers: for each observed
    gap it works out which entries are still alive, and a block is served from
    cache when the shallowest live marker at or above it still matches. That is
    what lets mixed lifetimes be scored at all -- with two TTLs in play, a block
    can be read out of an entry two markers above it.
    """
    if not blocks:
        return 0.0
    k = len(blocks)
    total = 0.0
    for g in (gaps or [None]):
        for i in range(k):
            # Shallowest marker at or above i whose entry is still alive.
            served = None
            for j in range(i, k):
                alive = survivals[j] if g is None else (
                    WRITE_LATENCY_SECONDS <= g < ttl_seconds(ttls[j]))
                if alive:
                    served = j
                    break
            tokens, _ = blocks[i]
            a = 0.0 if served is None else blocks[served][1]
            if g is None:
                # No gaps observed: fall back to per-marker survival rates,
                # which `survival()` sets to zero, so this is the no-hit case.
                a *= survivals[i]
            total += tokens * (a * read_rate + (1 - a) * write_rates[ttls[i]])
    return total / max(1, len(gaps or [None]))


def allocate(segments, change_rates, *, target_id: str, model: str,
             gaps=None, budget=None) -> Allocation:
    """Best marker placement for one prompt shape under the surface's budget.

    `segments` are in wire order. `change_rates` maps segment index to the share
    of requests on which that segment changed -- a rate, not a count of distinct
    values, because a field alternating between two states changes the prefix on
    every single request and only ever shows two values.
    """
    surf_budget, write_rates, read_rate = _surface(target_id, model)
    # `budget or surf_budget` made 0 indistinguishable from None, so a caller
    # passing "no breakpoints left" got a marker emitted over their own cap.
    if budget is not None and budget < 0:
        raise Unsupported(f"negative marker budget: {budget}")
    budget = surf_budget if budget is None else min(budget, surf_budget)
    gaps = list(gaps or [])
    notes, searched = [], []

    n = len(segments)
    if not n:
        return Allocation(notes=["no segments to place markers in"])

    cum, survive, running = [], [], 1.0
    total = 0
    for s in segments:
        total += s.tokens
        cum.append(total)
        running *= max(0.0, 1.0 - float(change_rates.get(s.index, 1.0)))
        survive.append(running)
    uncached = float(total)

    try:
        minimum = registry.min_cacheable_tokens(target_id, model)
    except registry.RegistryError as e:
        # Fail closed, like every other missing input here.
        #
        # Noting it and searching anyway made every cut point feasible, so an
        # unregistered model got marker positions and an 86% modelled saving on
        # a threshold nobody knows. Below the real minimum the provider ignores
        # the marker and returns no error, so the user pays the write premium
        # for nothing while the report calls it a saving.
        #
        # The plugin's own filter already refused this case; `allocate` did not,
        # and `allocator_full` calls straight through to here. A guard that
        # lives in one of two paths is the defect this branch keeps producing.
        raise Unsupported(
            f"no recorded minimum cacheable prefix for {model} on {target_id}, "
            f"so no marker can be shown to cache anything. Below the minimum a "
            f"provider caches nothing and returns no error. ({e})") from e

    control = registry.target(target_id).get("control_model")
    if control and control != "explicit_breakpoint":
        notes.append(
            f"{target_id} controls caching by {control}, not by an explicit "
            f"breakpoint over a prefix. This plan is modelled on "
            f"explicit-breakpoint semantics, so treat the placement as "
            f"indicative on this surface until it is confirmed against it")

    if not gaps:
        notes.append("no inter-request gaps observed, so no entry can be shown "
                     "to survive to the next request; caching cannot be "
                     "justified from this trace")
    if not change_rates:
        notes.append("no observed change rates supplied, so every segment is "
                     "treated as changing on every request; this refuses to "
                     "cache rather than assuming stability nobody measured")

    best = Allocation(expected_cost=uncached, uncached_cost=uncached,
                      notes=notes + ["no markers: every plan considered cost "
                                     "more than sending the prompt uncached"],
                      searched=searched)

    blocked = []
    if not budget:
        # Zero is a real answer: the caller has no breakpoints left. `_search`
        # indexes `cost[1]` and raised IndexError on it, so this was a crash
        # rather than a plan.
        return Allocation(
            [], uncached, uncached, searched,
            notes + ["marker budget is zero, so no plan can place one"])

    arms = []
    for ttl in write_rates:
        live = survival(gaps, ttl)
        alloc = _search(segments, cum, survive, minimum, budget, ttl,
                        live, read_rate, write_rates, uncached, notes)
        if alloc.tiers:
            arms.append(alloc)
        # A search that placed nothing has no cost, and recording its default
        # 0.0 puts "uniform 5m 0" in front of a reader as though the plan were
        # free. Infeasible is not cheap.
        searched.append((f"uniform {ttl}",
                         round(alloc.expected_cost, 2) if alloc.tiers else None))
        if alloc.tiers and alloc.expected_cost < best.expected_cost:
            best = alloc
        elif not alloc.tiers:
            # Why a lifetime yielded nothing is the finding, not a detail. A
            # prompt too short to cache is the silent failure this whole tool
            # exists to surface, and dropping the note loses it entirely.
            blocked += [n for n in alloc.notes if n not in notes]
    if not best.tiers and blocked:
        best = Allocation(best.tiers, best.expected_cost, best.uncached_cost,
                          best.searched, notes + list(dict.fromkeys(blocked)))

    # The one mixed pattern worth having: a long-lived stable prefix under a
    # short-lived advancing turn. Scored exactly rather than searched.
    #
    # Scored on every arm's positions, not only the winner's. `_mixed_variants`
    # needs at least two tiers to vary, and it was handed `best` -- which is the
    # zero-tier "send it uncached" plan whenever both uniform arms lose to
    # uncached. Measured: two 600-token segments over ten 10-minute gaps then
    # ten 2-hour ones searched uniform 5m at 1350 and uniform 1h at 1230,
    # neither beating 1200 uncached, and returned no markers -- while 5m over 1h
    # on those same positions costs 1035. A 13.75% saving discarded because the
    # only plan allowed to carry positions forward was the one that had none.
    #
    # The registry records `ttl_ordering_constraint` as null for this surface,
    # so short-before-long is legal here; on a surface that constrains it,
    # `_mixed_variants` still has to respect that.
    seen_labels = set()
    for label, alloc in _mixed_exhaustive(segments, cum, survive, minimum,
                                          budget, write_rates, read_rate, gaps,
                                          uncached, notes, target_id):
        seen_labels.add(label)
        searched.append((label, round(alloc.expected_cost, 2)))
        if alloc.expected_cost < best.expected_cost:
            best = alloc
    for arm in arms or [best]:
        for label, alloc in _mixed_variants(arm, segments, cum, survive,
                                            write_rates, read_rate, gaps, notes):
            if label in seen_labels:
                continue
            seen_labels.add(label)
            searched.append((label, round(alloc.expected_cost, 2)))
            if alloc.expected_cost < best.expected_cost:
                best = alloc

    return Allocation(best.tiers, best.expected_cost, uncached,
                      searched, best.notes)


def _search(segments, cum, survive, minimum, budget, ttl, live,
            read_rate, write_rates, uncached, base_notes) -> Allocation:
    """Shortest-path over cut points for one uniform lifetime.

    With a single lifetime every entry lives or dies together, so a block's hit
    probability depends only on its own marker and the blocks become
    independent. That is what makes this a DP rather than a subset search:
    O(positions^2 * budget) instead of one term per marker combination.
    """
    n = len(segments)
    w = write_rates[ttl]
    feasible = [p for p in range(n)
                if minimum is None or cum[p] >= minimum]
    if not feasible:
        return Allocation(notes=base_notes + [
            f"no cut point reaches the {minimum:,}-token minimum for this "
            f"surface; the whole prompt is too short to cache"])

    # cost[j][p] = best cost of blocks 0..p using j markers, deepest at p.
    INF = float("inf")
    cost = [[INF] * n for _ in range(budget + 1)]
    back = [[None] * n for _ in range(budget + 1)]

    def block(prev_cum, p):
        a = survive[p] * live
        return (cum[p] - prev_cum) * (a * read_rate + (1 - a) * w)

    for p in feasible:
        cost[1][p] = block(0, p)
    for j in range(2, budget + 1):
        for p in feasible:
            for q in feasible:
                if q >= p:
                    break
                if cost[j - 1][q] == INF:
                    continue
                c = cost[j - 1][q] + block(cum[q], p)
                if c < cost[j][p]:
                    cost[j][p], back[j][p] = c, q

    best_total, best_at, best_j = INF, None, None
    for j in range(1, budget + 1):
        for p in feasible:
            if cost[j][p] == INF:
                continue
            total = cost[j][p] + (cum[-1] - cum[p])   # tail is plain input
            if total < best_total:
                best_total, best_at, best_j = total, p, j

    if best_at is None:
        return Allocation(notes=base_notes + ["no feasible marker placement"])

    positions, p, j = [], best_at, best_j
    while p is not None:
        positions.append(p)
        p, j = back[j][p], j - 1
    positions.reverse()

    tiers, prev = [], -1
    for p in positions:
        tiers.append(Tier(
            marker_position=p, first_position=prev + 1,
            tokens=cum[p] - (cum[prev] if prev >= 0 else 0),
            prefix_tokens=cum[p], ttl=ttl,
            hit_probability=survive[p] * live,
            segment_indices=tuple(s.index for s in segments[prev + 1:p + 1])))
        prev = p
    return Allocation(tiers, best_total, uncached, [], list(base_notes))


# Above this many segments the exhaustive mixed pass is skipped and the search
# stays on the positions the uniform DP found. Chosen so the worst case stays
# in the low thousands of evaluations: sum(C(n,k) for k<=4) * 2**4.
# 8, not 12: at 12 with a budget of 4 this is ~10k evaluations per call and it
# tripled the test suite's runtime. Real prompts carry few *segments* -- system,
# tools, and grouped messages -- so 8 covers the shape this exists for while
# staying under ~2.6k evaluations.
MIXED_EXHAUSTIVE_MAX_SEGMENTS = 8


def _mixed_exhaustive(segments, cum, survive, minimum, budget, write_rates,
                      read_rate, gaps, uncached, base_notes, target_id):
    """Every marker placement up to `budget`, with every lifetime assignment.

    `_mixed_variants` can only re-label positions the uniform DP already chose,
    and the DP optimises each lifetime alone. When neither uniform plan beats
    sending the prompt uncached, it returns no positions at all -- and the
    mixed pattern then has nothing to vary, so a plan that is cheaper than both
    is never scored. Measured on two 600-token segments: uniform 5m 1350,
    uniform 1h 1230, uncached 1200, and 5m-over-1h 1035.

    Exhaustive rather than clever, and bounded rather than complete: this is a
    per-request placement over a handful of segments with a budget of four, not
    a general optimiser. Above the bound it does not run and the caller says so.
    """
    from itertools import combinations
    n = len(segments)
    if n > MIXED_EXHAUSTIVE_MAX_SEGMENTS or len(write_rates) < 2 or not gaps:
        return []
    ttl_names = sorted(write_rates, key=ttl_seconds)
    surv_by_ttl = {t: survival(gaps, t) for t in ttl_names}
    out, best_c, best_alloc = [], None, None
    for k in range(1, min(budget, n) + 1):
        for positions in combinations(range(n), k):
            if cum[positions[0]] < minimum:
                continue                      # first prefix cannot cache
            blocks, prev = [], 0
            for p in positions:
                blocks.append((cum[p] - prev, survive[p]))
                prev = cum[p]
            tail = cum[-1] - cum[positions[-1]]
            for assignment in _ttl_assignments(ttl_names, k, target_id):
                survivals = [surv_by_ttl[t] for t in assignment]
                c = expected_cost(blocks, list(assignment), read_rate,
                                  write_rates, gaps, survivals) + tail
                if best_c is None or c < best_c:
                    best_c, best_alloc = c, (positions, assignment, blocks)
    if best_alloc is None or best_c >= uncached:
        return []
    positions, assignment, blocks = best_alloc
    tiers, first = [], 0
    for i, p in enumerate(positions):
        tiers.append(Tier(p, first, blocks[i][0], cum[p], assignment[i],
                          survive[p] * surv_by_ttl[assignment[i]],
                          tuple(range(first, p + 1))))
        first = p + 1
    label = "mixed " + "/".join(assignment)
    note = ("placement found by scoring every marker set up to the budget "
            "against every lifetime assignment; the uniform search reached "
            "none of them")
    return [(label, Allocation(tiers, best_c, uncached, [],
                               list(base_notes) + [note]))]


def _ttl_assignments(ttl_names, k, target_id):
    """Lifetime per marker, respecting any recorded ordering constraint.

    This yielded every permutation while its own docstring said constrained
    surfaces were filtered. They were not, and the comment made that harder to
    notice rather than easier. Measured: two 5,000-token segments on
    amazon-bedrock/converse produced `[(0, "5m"), (1, "1h")]`, which
    `checks.check_ttl_ordering` fails on the same surface -- the allocator
    recommending a placement the project's own linter rejects, on the output
    that gets applied to a production prompt rather than published.

    A constraint the registry does not record is not a constraint. A constraint
    it does record is enforced here, once, rather than at each call site.
    """
    from itertools import product
    try:
        constraint = registry.capability(target_id, "ttl_ordering_constraint")
    except registry.RegistryError:
        # Unrecorded is not permission on a surface nobody has described. The
        # uniform assignments are always valid under any ordering rule, so they
        # are what survives.
        constraint = "longest_first"
    for combo in product(ttl_names, repeat=k):
        if constraint == "longest_first":
            secs = [ttl_seconds(t) for t in combo]
            if any(b > a for a, b in zip(secs, secs[1:])):
                continue
        yield combo


def _mixed_variants(best, segments, cum, survive, write_rates, read_rate,
                    gaps, base_notes):
    """Score the canonical two-lifetime pattern on the positions already found.

    Not a search. The uniform DP has already chosen where the cuts go; this asks
    whether holding the lower tiers longer than the top one is cheaper, which is
    the shape a stable system prefix under an advancing conversation turn takes.
    """
    if len(best.tiers) < 2 or len(write_rates) < 2:
        return []
    order = sorted(write_rates, key=ttl_seconds)
    short, long_ = order[0], order[-1]
    if short == long_:
        return []

    blocks = [(t.tokens, survive[t.marker_position]) for t in best.tiers]
    tail = cum[-1] - best.tiers[-1].prefix_tokens
    ttls = [long_] * (len(blocks) - 1) + [short]
    survivals = [survival(gaps, t) for t in ttls]
    c = expected_cost(blocks, ttls, read_rate, write_rates, gaps, survivals) + tail

    tiers = [Tier(t.marker_position, t.first_position, t.tokens,
                  t.prefix_tokens, ttls[i],
                  survive[t.marker_position] * survivals[i], t.segment_indices)
             for i, t in enumerate(best.tiers)]
    note = (f"lower tiers held for {long_} under a {short} top tier: the stable "
            f"prefix outlives the turn that keeps advancing")
    return [(f"mixed {long_}/{short}",
             Allocation(tiers, c, best.uncached_cost, [],
                        list(base_notes) + [note]))]
