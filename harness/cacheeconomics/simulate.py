"""Replay a trace under a marker policy and price the result.

This is the offline dual-policy simulation the bake-off runs on. Live A/B on
non-idempotent agentic workloads is unsafe and expensive, so the primary method
is replaying real request timings against a modelled cache.

Everything this produces is MODELED. It is arithmetic over a cache model whose
boundary was measured, not a measurement in itself. The distinction is not
pedantry: a simulated verdict that reads like a measured one is exactly how a
tool like this stops being trustworthy.

What the model gets right, because it was measured directly on 2026-07-28:
  - a 5-minute entry is gone somewhere between 300 and 420 seconds
  - a hit refreshes the lifetime at no write cost
  - a 1-hour entry survived a 56-minute gap

What it models because a real trace showed it mattered:
  - cache isolation per model and API surface, so a prefix cannot be read across
    a mid-session model switch
  - write visibility from response time rather than send time, so concurrent
    subagent calls cannot hit an entry that did not exist yet

What it does not model, and therefore where it will be wrong:
  - server-side eviction under memory pressure
  - cross-region and cross-workspace isolation, which the trace schema does not
    yet carry
  - the 20-block lookback window is modelled at segment granularity, which is
    coarser than the blocks the provider counts, so it is enforced only in the
    pessimistic arm
  - provider-side routing that may not send two identical prefixes to the same
    machine
  - true response completion time; first_token_at is the tightest bound most
    traces carry, so overlapping requests are modelled slightly optimistically
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import cost, money
from .allocate import (POLICIES as _PLACEMENT_POLICIES, Plan, observed_cadence,
                       observed_change_rates_by_chain, observed_gaps_by_chain,
                       observed_volatility, observed_volatility_by_chain, pool_of,
                       reuse_chain_of)
from .registry import RegistryError, capability, min_cacheable_tokens
from .relocate import propose, relocation_lite
from .trace import (PUBLISH_TOLERANCE, Request, segment_sum_ratio,
                    sums_publishable)

TTL_SECONDS = {"5m": 300, "1h": 3600}


@dataclass(frozen=True)
class Assumptions:
    """The knobs where this model is guessing, and which way the guess leans.

    Every simplification in this simulator was individually defensible and all
    of them leaned the same way: toward more cache hits. Arms that rely more on
    caching — ours — benefit most from that, so the aggregate bias flattered the
    conclusion rather than the baseline. Two reviews found instances of this one
    at a time; collecting them here is the fix for the class.

    The headline verdict runs PESSIMISTIC. NEUTRAL is reported beside it as the
    optimistic end of the range, never alone.

    eviction_haircut     Entries are assumed to survive their full lifetime. Real
                         caches evict under memory pressure, so effective life is
                         shorter. Applied as a fraction of the TTL.
    routing_miss_in_n    Assumes every identical prefix reaches a machine holding
                         it. Provider routing does not guarantee that. Forces
                         every Nth otherwise-eligible hit to miss. Deterministic
                         by request index, so runs stay reproducible.
    write_latency_s      A write is readable once the response returns. Without
                         completion timestamps, send time is the fallback and
                         concurrent requests appear to hit early. Adds a floor.
    enforce_lookback     A breakpoint matches back at most 20 blocks. Segments
                         are a coarse proxy for blocks, so this is off by default
                         and on when being pessimistic.
    """
    eviction_haircut: float = 0.0
    routing_miss_in_n: int = 0
    write_latency_s: float = 0.0
    enforce_lookback: bool = False
    label: str = "neutral"


NEUTRAL = Assumptions()
PESSIMISTIC = Assumptions(eviction_haircut=0.20, routing_miss_in_n=20,
                          write_latency_s=5.0, enforce_lookback=True,
                          label="pessimistic")

POLICIES = {**_PLACEMENT_POLICIES, "relocation-lite": relocation_lite}

# Order matters for reading the output: each arm answers one more question than
# the one before it.
ARMS = ("as-shipped", "litellm-auto", "allocator-lite", "relocation-lite")


def registry_lookback(target_id: str):
    try:
        return capability(target_id, "lookback_blocks")
    except RegistryError:
        return None


@dataclass
class SimResult:
    policy: str
    usages: list = field(default_factory=list)
    reqs: list = field(default_factory=list)
    priced: list = field(default_factory=list)   # reqs that produced a usage
    writes: int = 0      # requests that wrote a cache entry
    reads: int = 0       # requests that read one (can overlap with writes)
    cold: int = 0        # requests with no cache activity at all
    unstructured: int = 0  # skipped: no segments, so not priceable at all
    untimed: int = 0       # skipped: no timestamp, so no expiry or ordering
    unpriceable: int = 0   # simulated, but the surface or model has no pricing
    # reason -> {request_id}. Counts alone cannot be combined across arms: the
    # same request is unpriceable in one arm and unmodelled in another, and
    # summing the per-arm maxima reported more requests omitted than existed --
    # "6 of 3 requests contributed nothing", in the sentence explaining why the
    # gate could not be answered.
    omitted: dict = field(default_factory=dict)
    unmodelled_ttl: int = 0  # a cache lifetime this replay does not model
    unmodelled_target: int = 0  # the surface is not in the registry at all
    notes: list = field(default_factory=list)

    def spend(self, on_date=None, effective_rate=None) -> money.Figure:
        """`on_date=None` prices each request on the day it was sent.

        That is the correct default for a historical trace and it removes a
        hard-coded date that would have gone stale silently: the registry is
        date-effective and already carries a claude-sonnet-5 rate change on
        2026-09-01, so a pinned date underprices every run made after it.

        Returns a `money.Figure` that starts **withheld**, so the number cannot
        reach an output until something decides it may. `bake_off` is what
        decides, once, for every arm at both assumption ends. A caller holding a
        bare SimResult has not established that its segment sizes reconcile with
        the tokens the provider billed, and until that is established this total
        is a number of unknown scale: a 1,000x size error turned $0.125 of arm
        spend into $125.00 while every ratio between arms stayed exact.

        MODELED, not MEASURED, for every arm including as-shipped. As-shipped
        replays the lifetimes the trace actually carried, but it still replays
        them against a simulated cache whose hits this module decides.
        """
        # Zips against `priced`, not `reqs`. Structureless requests are skipped
        # in the loop, so zipping the full request list would pair each usage
        # with the wrong request and price it at the wrong model's rate.
        # Skip what cannot be priced rather than raising. cost.price now
        # refuses surfaces without Anthropic-shaped multipliers, and the
        # analyzer already excludes those requests and fails its gate -- the
        # bake-off had no such path, so one mixed-surface export aborted the
        # entire comparison. The count is carried so the caller can say what was
        # left out instead of quietly comparing a subset.
        total = 0.0
        # Both reset together. `unpriceable` was cleared and the matching
        # `omitted` entries were not, so pricing once without an effective rate
        # and again with one left the gate counting requests the second call
        # had priced successfully -- inflating the denominator in the sentence
        # that explains why the gate could not be answered.
        self.unpriceable = 0
        self.omitted.pop("without pricing", None)
        for u, r in zip(self.usages, self.priced):
            try:
                total += cost.price(
                    u, r.model, target_id=r.target_id,
                    on_date=on_date or (r.sent_at.strftime("%Y-%m-%d")
                                        if r.sent_at else None),
                    effective_rate=effective_rate).usd
            except RegistryError:
                self.unpriceable += 1
                self.omitted.setdefault("without pricing", set()).add(r.request_id)
        return money.Figure(
            total, money.MODELED, released=False,
            withheld_because="arm spend is not released until the trace's segment "
                             "sizes reconcile with the tokens the provider billed "
                             "and every request contributes to every arm")

    @property
    def ratios(self) -> dict:
        return cost.ratios(self.usages)


def simulate(reqs: list[Request], policy, volatility=None, cadence=None,
             moves=None, order=None, assume: Assumptions = NEUTRAL) -> SimResult:
    """Replay requests in time order against a modelled cache.

    `policy` is a name from `POLICIES` or a callable with the same signature.
    The callable form exists so a policy that is deliberately not a bake-off arm
    -- the full allocator, which Gate 1 decides the fate of -- can still be
    replayed against real timing. Its own cost model and this replay share no
    code, and a plan that only ever agrees with the optimiser that produced it
    has not been checked by anything.
    """
    fn = POLICIES[policy] if isinstance(policy, str) else policy
    # Timing is the whole substrate here -- expiry, visibility, cadence all read
    # r.sent_at -- and the loaders can legitimately produce a request without
    # one, because hardening them against malformed rows meant accepting a
    # missing timestamp rather than dropping the row. Sorting on None then
    # raised a TypeError that took the whole bake-off down. An untimed request
    # is excluded and counted, which is the same answer the structureless case
    # already gets: a stated gap beats a dropped file.
    untimed = [r for r in reqs if r.sent_at is None]
    reqs = sorted((r for r in reqs if r.sent_at is not None), key=lambda r: r.sent_at)
    # Per pool, not one map for the whole export. Caches are isolated by
    # tenant, surface and model, so a segment can be stable in one pool and
    # change on every request in another. Deciding once for all of them either
    # suppresses a valid marker or places one that rewrites a prefix nothing
    # will read. An explicit `volatility` argument still overrides, for callers
    # that genuinely have one homogeneous group.
    by_chain = None if volatility is not None else observed_volatility_by_chain(reqs)
    # Cadence per reuse chain, for the same reason volatility is. One median over
    # mixed sessions can hand a slow chain a one-hour TTL it will never read from
    # -- paying the 2x write premium on every request while the notes claim the
    # longer lifetime pays.
    # Change rates and gap distributions, for policies that want the shape of
    # the traffic rather than a single median. Placement arms ignore both.
    rates_by_chain = observed_change_rates_by_chain(reqs)
    gaps_by_chain = observed_gaps_by_chain(reqs)
    cadence_by_chain = None
    if cadence is None:
        buckets = {}
        for r in reqs:
            buckets.setdefault(reuse_chain_of(r), []).append(r)
        cadence_by_chain = {p: observed_cadence(rs) for p, rs in buckets.items()}
    wants_moves = policy in ("relocation-lite",) or not isinstance(policy, str)
    if moves is None and wants_moves and reqs:
        # No model/target argument: safety is derived from every scope in the
        # group, not from whichever request happened to sort first.
        #
        # Relocation emits one ordering for the whole group, so it takes the
        # fail-closed reduction rather than any single pool's view. A move is
        # only safe if it is safe for the worst pool it will be applied to.
        moves, order = propose(
            reqs, volatility if volatility is not None else observed_volatility(reqs))

    # key -> (visible_from, expires_at).
    #
    # The key carries the isolation scope, not just the prefix. Caches do not
    # span tenants, models or API surfaces, so keying on the segment tuple alone
    # let a request read an entry written by a different model — which is not
    # exotic: a real 29-day trace analysed here had four sessions switch model
    # mid-conversation, exactly the case that would fabricate hits.
    #
    # Tenant is in the key because a shared trace is the normal case for a
    # gateway export. Without it, one tenant's write pays for another tenant's
    # read, inventing savings and misattributing them at the same time.
    #
    # `visible_from` exists because a write is not readable until the response
    # that produced it comes back. Agentic workloads fan out concurrent
    # subagent calls, and making writes live at send time let overlapping
    # requests hit an entry that did not exist yet.
    live: dict[tuple, tuple[float, float]] = {}
    res = SimResult(policy=policy, reqs=reqs, untimed=len(untimed))
    if untimed:
        res.omitted['without timestamps'] = {r.request_id for r in untimed}
    if not reqs:
        return res

    for idx, r in enumerate(reqs):
        # A request with no segments is not a cheap request, it is an unknown
        # one. Pricing it from `sum(r.segments) == 0` made it free and
        # understated every arm equally, which reads as agreement between
        # policies rather than as missing data.
        if not r.segments:
            res.unstructured += 1
            res.omitted.setdefault("without structure", set()).add(r.request_id)
            continue

        # Computed before the policy runs, because the failure path below needs
        # it. It used to be assigned after, so an unknown target on the first
        # row raised UnboundLocalError and one anywhere else silently reused the
        # previous request's token count -- a 250,000-token row recorded as
        # 9,000. Verifying that the handler "survived" and not that it produced
        # the right number is how that shipped.
        total = sum(s.tokens for s in r.segments)
        chain = reuse_chain_of(r)
        vol = volatility if volatility is not None else by_chain.get(chain, {})
        cad = cadence if cadence is not None else cadence_by_chain.get(chain)
        try:
            plan: Plan = fn(r, volatility=vol, cadence_seconds=cad,
                            rates=rates_by_chain.get(chain, {}),
                            gaps=gaps_by_chain.get(chain, []),
                            moves=moves, order=order)
        except RegistryError:
            # Policies read the registry while building a plan -- litellm_auto
            # asks for the marker budget -- so one row carrying an unknown or
            # contested target_id raised out of the entire bake-off. Schema
            # drift in a single row could take down the comparison instead of
            # producing the indeterminate verdict this module already knows how
            # to report. Counted and priced uncached, like every other thing
            # here that cannot be modelled.
            res.unmodelled_target += 1
            res.omitted.setdefault("on an unregistered surface", set()).add(r.request_id)
            res.usages.append(cost.Usage(uncached_input=total))
            # Paired, like every other branch that records a usage. `spend()`
            # zips usages against `priced`, so appending one without the other
            # shifted the whole list: this row's 250,000 tokens were priced
            # against the *next* request's model and that request's own usage
            # fell off the end. Corrupting arm spend on precisely the
            # schema-drift path this guard exists to survive.
            res.priced.append(r)
            continue
        now = r.sent_at.timestamp()

        # Below the minimum a prefix does not cache, and the provider returns no
        # error. Such markers are dropped rather than treated as cache entries.
        try:
            minimum = min_cacheable_tokens(r.target_id, r.model)
        except RegistryError:
            # Not zero. Below the minimum a provider caches nothing and returns
            # no error, and the registry refuses to guess because minimums are
            # non-monotonic across generations. Assuming everything cacheable
            # let an unregistered model produce a confident verdict built on
            # hits that may not exist -- and with an invoice rate supplied,
            # pricing no longer failed either, so nothing else caught it.
            res.usages.append(cost.Usage(uncached_input=total))
            res.priced.append(r)
            res.unmodelled_ttl += 1
            res.omitted.setdefault("with an unmodelled lifetime", set()).add(r.request_id)
            continue
        pres = [p for p in plan.prefixes(r.segments) if p[1] >= minimum]

        # This replay models the two Anthropic lifetimes and nothing else. The
        # registry advertises openai/direct with a 30m TTL, and indexing the map
        # by a marker's lifetime raised KeyError before spend() could reach its
        # fail-closed path -- so one mixed-surface export aborted the whole
        # comparison rather than reporting a guarded one.
        unsupported = [ttl for _k, _t, ttl in pres if ttl not in TTL_SECONDS]
        if unsupported:
            res.usages.append(cost.Usage(uncached_input=total))
            res.priced.append(r)
            res.unmodelled_ttl += 1
            res.omitted.setdefault("with an unmodelled lifetime", set()).add(r.request_id)
            continue

        # A breakpoint matches back a bounded number of content blocks. Segments
        # are coarser than blocks, so this is only enforced when being
        # deliberately pessimistic.
        #
        # Measured from each breakpoint, not from the end of the request. The
        # earlier form kept a prefix only when it reached within `window`
        # segments of the request tail, which made a marker's survival depend on
        # how much conversation followed it: a 40,000-token system prefix marked
        # at the top replayed as zero cache reads once forty blocks sat behind
        # it, while the neutral arm read it every time.
        #
        # That is not conservatism, it is bias. It penalised deep prefixes
        # specifically -- exactly the placement the allocator recommends and the
        # workload caching exists for -- so it pressed on the Gate 1 scale in
        # one direction while wearing the label "pessimistic".
        #
        # What the window plausibly constrains is the distance the provider
        # searches back from a breakpoint to find an earlier entry, so it is
        # applied to the span between consecutive markers. The first marker has
        # nothing behind it to search for and is not dropped. The exact
        # semantics are not something this project has measured, and inventing a
        # penalty that bites only our own recommendation would be worse than
        # applying none.
        if assume.enforce_lookback:
            window = registry_lookback(r.target_id)
            if window:
                kept, prev = [], 0
                for p in sorted(pres, key=lambda x: len(x[0])):
                    if not kept or len(p[0]) - prev <= window:
                        kept.append(p)
                        prev = len(p[0])
                pres = kept

        if not pres:
            res.usages.append(cost.Usage(uncached_input=total))
            res.priced.append(r)
            res.cold += 1
            continue

        # Longest alive prefix wins the read; everything from there to the
        # outermost marker is written.
        scope = (r.tenant, r.target_id, r.model)
        read = 0
        # Deterministic, not random: the same trace must produce the same
        # verdict every run, or a bake-off result cannot be reproduced.
        routed_away = (assume.routing_miss_in_n
                       and idx % assume.routing_miss_in_n == assume.routing_miss_in_n - 1)
        if not routed_away:
            # Every live prefix of what is being sent, not only the ones this
            # request happens to mark. The provider searches backward from each
            # breakpoint for an earlier cached prefix, so an advancing trailing
            # breakpoint reads the shorter entry a previous turn wrote even
            # though nothing marks that position now.
            #
            # Matching only `pres` modelled a policy that moves its breakpoint
            # as never hitting anything, which is the same defamatory shape
            # `Plan.prefixes` was already fixed for one layer up -- and here it
            # could reverse a bake-off verdict, telling a client to abandon a
            # placement that is actually cheaper.
            #
            # This cannot fabricate a hit. The key is the tuple of segment ids,
            # so `seq[:len(key)] == key` is the literal statement that the
            # cached span is a prefix of what is being sent, which is the
            # provider's own condition for a read.
            seq, cum, at = [], [], []
            running = 0
            for sg in plan.emission(r.segments):
                seq.append(sg.id)
                running += sg.tokens
                cum.append(running)
                at.append(sg.index)
            marks = [len(k) for k, _t, _ttl in pres]
            window = registry_lookback(r.target_id) if assume.enforce_lookback else None
            for length in range(len(seq), 0, -1):
                key = tuple(seq[:length])
                entry = live.get((scope, key))
                if not (entry and entry[0] <= now < entry[1]):
                    continue
                # Reachable from some breakpoint at or after it. The provider
                # searches back FROM a breakpoint, so an entry sitting past
                # every marker this request places cannot be found at all.
                #
                # This condition used to be inside `if window is not None`, so
                # the neutral arm skipped it entirely and would read a 6k entry
                # for a request whose only marker covers 2k -- a read no
                # breakpoint could reach, inflating every neutral verdict.
                # Reachability is not a pessimistic assumption; the window is.
                if not any(m >= length for m in marks):
                    continue
                if window is not None and not any(
                        0 <= m - length <= window for m in marks):
                    continue
                read = cum[length - 1]
                break
        outermost = pres[-1][1]

        # Split the written span by breakpoint, not by the outermost marker's
        # lifetime. A request with an inner 1h marker and an outer 5m one was
        # modelled as writing everything at 5m, which is the same 5m/1h pricing
        # error the analyzer refuses to make -- reintroduced one layer down,
        # where it silently moves bake-off spend.
        written_by_ttl = {"5m": 0, "1h": 0}
        lower = read
        for _key, toks, ttl in pres:
            if toks <= read:
                continue                      # already covered by the read
            written_by_ttl[ttl] += toks - lower
            lower = toks
        written = sum(written_by_ttl.values())

        fields = {"uncached_input": total - outermost}
        if read:
            fields["cache_read"] = read
            res.reads += 1
        if written:
            for ttl, toks in written_by_ttl.items():
                if toks:
                    fields[f"cache_write_{ttl}"] = toks
            res.writes += 1
        if not read and not written:
            res.cold += 1
        res.usages.append(cost.Usage(**fields))
        res.priced.append(r)

        # A hit refreshes the lifetime at no write cost. Measured, not assumed.
        #
        # Readable only once this request's response has come back. first_token_at
        # is the tightest lower bound a trace normally carries; without it, send
        # time is used and genuinely concurrent requests will be modelled as
        # hitting slightly earlier than they could have.
        visible = (r.first_token_at.timestamp() if r.first_token_at else now)
        visible += assume.write_latency_s
        for key, _, ttl in pres:
            expires = now + TTL_SECONDS[ttl] * (1 - assume.eviction_haircut)
            prev = live.get((scope, key))
            # An entry that is already readable stays readable while a refresh
            # is in flight. Overwriting its visibility with the *new* response's
            # time hid a warm prefix for the duration of every refresh, so a
            # request arriving a second after a hit missed and wrote again --
            # fabricating cache writes in the arm that feeds Gate 1's headline.
            # Only a genuinely new or expired entry starts invisible.
            if prev and prev[0] <= now < prev[1]:
                live[(scope, key)] = (prev[0], expires)
            else:
                live[(scope, key)] = (visible, expires)

    # The policy's own notes, taken from the first request it can actually plan
    # for. `reqs[0]` unconditionally meant an unknown target on the very first
    # row raised here even after the main loop learned to survive it -- the same
    # call, guarded in one place and not the other, which is the twin-call-site
    # failure this branch keeps producing.
    for probe in (r for r in reqs if r.segments):
        try:
            res.notes.extend(fn(
                probe,
                volatility=(volatility if volatility is not None
                            else by_chain.get(reuse_chain_of(probe), {})),
                cadence_seconds=(cadence if cadence is not None
                                 else cadence_by_chain.get(reuse_chain_of(probe))),
                rates=rates_by_chain.get(reuse_chain_of(probe), {}),
                gaps=gaps_by_chain.get(reuse_chain_of(probe), []),
                moves=moves, order=order).notes)
        except RegistryError:
            continue
        break
    return res


@dataclass
class BakeOff:
    group: str
    n_requests: int
    window_days: float | None
    arms: dict = field(default_factory=dict)          # policy -> {spend, ratios, ...}
    optimistic: dict = field(default_factory=dict)    # same arms under NEUTRAL
    moves: list = field(default_factory=list)
    unstructured: int = 0
    untimed: int = 0
    unpriceable: int = 0
    unmodelled_ttl: int = 0
    unmodelled_target: int = 0
    verdict: str = ""
    verdict_relocation: str = ""
    delta_pct: float | None = None                    # pessimistic: the headline
    delta_pct_relocation: float | None = None
    delta_pct_optimistic: float | None = None
    delta_pct_relocation_optimistic: float | None = None

    def _range(self, pess, opt):
        if pess is None or opt is None:
            return "indeterminate"
        lo, hi = sorted((pess, opt))
        return f"{lo:+.1f}% to {hi:+.1f}%"

    def __str__(self):
        lines = [f"{self.group}  ({self.n_requests} requests)  "
                 f"[headline = pessimistic assumptions]"]
        base = self.arms.get("litellm-auto", {}).get("spend")
        for p, a in self.arms.items():
            d = ""
            # Gated on the same release state as the absolute beside it, not on
            # being a percentage. `delta_pct` is None on both indeterminate
            # paths and this column printed anyway, so a block could read
            # "+20.0% vs litellm-auto" three lines above "placement:
            # indeterminate" -- the headline's guard not applied to its twin,
            # which is the defect class this file has produced more than once.
            #
            # A *uniform* size error does cancel, which is why the verdict says
            # so. `misscaled` does not establish uniformity: three inflated rows
            # among twenty move each arm by a different amount. And no
            # percentage survives a subset, since the omitted requests could
            # dominate spend.
            # Gated on whether the *percentage* is meaningful -- which is what
            # `delta_pct` records -- not on whether the absolutes may be
            # published. Those were the same condition until an invoice became
            # part of the release rule: a run with no invoice has perfectly good
            # ratios and withheld dollars, and reading `released` here hid a
            # number that was never in question. `delta_pct` is None exactly when
            # the comparison is over a subset or an unknown scale.
            if (base is not None and base.raw() and p != "litellm-auto"
                    and self.delta_pct is not None):
                d = (f"   {100*(base.raw() - a['spend'].raw())/base.raw():+.1f}%"
                     f" vs litellm-auto")
            # Four decimals matter at bake-off scale, where arms differ by cents,
            # so this asks for the spec rather than taking `str(Figure)`. The
            # withheld form is the short label, with the reason stated once
            # below: `str(Figure)` carries the full sentence, and four arms x two
            # ends repeated it until the block was unreadable.
            cell = (f"${a['spend']:.4f}" if a["spend"].released else "[withheld]")
            lines.append(f"  {p:<16} {cell}   "
                         f"hit {a['reads']:>3} wrote {a['writes']:>3}"
                         f" cold {a['cold']:>3}{d}")
        # Once, in the same voice as the other caveats. Safe to shorten the
        # per-arm label to "[withheld]" only because this always accompanies it.
        held = [a["spend"] for a in self.arms.values() if not a["spend"].released]
        if held:
            lines.append(f"  !! per-arm spend withheld: {held[0].withheld_because}.")
        lines.append(f"  range vs baseline  placement  "
                     f"{self._range(self.delta_pct, self.delta_pct_optimistic)}")
        lines.append(f"                     relocation "
                     f"{self._range(self.delta_pct_relocation, self.delta_pct_relocation_optimistic)}")
        if self.unmodelled_ttl:
            lines.append(f"  !! {self.unmodelled_ttl} request(s) use a cache lifetime this "
                         f"replay does not model and were treated as uncached.")
        if self.unmodelled_target:
            lines.append(f"  !! {self.unmodelled_target} request(s) name a surface the "
                         f"registry does not carry, so no policy could plan for them.")
        if self.unpriceable:
            lines.append(f"  !! {self.unpriceable} request(s) are on a surface or model "
                         f"this cost model cannot price and contribute nothing to the "
                         f"figures above.")
        if self.untimed:
            lines.append(f"  !! {self.untimed} of {self.n_requests} requests carry no "
                         f"timestamp and were excluded. Expiry and ordering are undefined "
                         f"without one.")
        if self.unstructured:
            lines.append(f"  !! {self.unstructured} of {self.n_requests} requests carry no "
                         f"prompt structure and were excluded. Every arm is priced over the "
                         f"remainder only.")
        lines.append(f"  -> placement:  {self.verdict}")
        lines.append(f"  -> relocation: {self.verdict_relocation}")
        if self.moves:
            lines.append("  proposed moves (never applied automatically):")
            lines += ["    " + l for m in self.moves for l in str(m).splitlines()]
        return "\n".join(lines)


GATE_THRESHOLD_PCT = 10.0


def bake_off(reqs: list[Request], group: str = "all", on_date: str | None = None,
             effective_rate: float | None = None, invoice_usd: float | None = None,
             allow_unreconciled: bool = False,
             excluded_billed: dict | None = None) -> BakeOff:
    """Run all four arms over the same requests and compare.

    The comparison that matters is against litellm-auto. Beating as-shipped
    proves only that the traced code was not tuned; beating a competent
    automatic baseline is the question Gate 1 actually asks, which is also why
    the baseline is modelled with multi-breakpoint semantics rather than as a
    single span.
    """
    # Relocation produces one ordering for the whole group, so it uses the
    # fail-closed reduction. The arms let simulate() work per pool.
    volatility = observed_volatility(reqs)
    cadence = None          # each arm derives it per pool inside simulate()
    ts = sorted(r.sent_at for r in reqs if r.sent_at)
    window = (ts[-1] - ts[0]).total_seconds() / 86400.0 if len(ts) > 1 else None
    moves, order = propose(reqs, volatility)

    def run(assume):
        out = {}
        for p in ARMS:
            # No volatility argument: each arm derives it per cache pool
            # inside simulate(), which is the only place that knows which pool
            # a given request will land in.
            s = simulate(reqs, p, cadence=cadence,
                         moves=moves, order=order, assume=assume)
            out[p] = {"spend": s.spend(on_date, effective_rate), "reads": s.reads,
                      "writes": s.writes, "cold": s.cold,
                      "unstructured": s.unstructured, "untimed": s.untimed,
                      "unpriceable": s.unpriceable,
                      "unmodelled_ttl": s.unmodelled_ttl,
                      "unmodelled_target": s.unmodelled_target,
                      # Read after `spend()` above, which is what discovers the
                      # unpriceable rows and records their ids.
                      "omitted": s.omitted,
                      "ratios": s.ratios, "notes": s.notes}
        return out

    def delta(arms_, policy):
        # `raw()` is correct here and this is the audit note for it. Every arm is
        # linear in token counts, so a uniform size error scales all of them
        # together and cancels: measured, a 1,000x error left this at exactly
        # 20.0% both times. The percentage therefore does not depend on the
        # release gate that the absolutes do, and computing it before release is
        # precisely the arithmetic `raw()` exists for.
        #
        # It is not subset-invariant, which is a different gate: when requests
        # were omitted this function's result is discarded and delta_pct is None.
        base_ = arms_["litellm-auto"]["spend"].raw()
        return (100 * (base_ - arms_[policy]["spend"].raw()) / base_) if base_ else None

    # Both ends, every time. A point estimate from a model whose every
    # simplification leans one way is not a result, and the plan calls for a
    # range with the pessimistic end as the headline.
    pess, opt = run(PESSIMISTIC), run(NEUTRAL)
    d_place, d_reloc = delta(pess, "allocator-lite"), delta(pess, "relocation-lite")

    # Across arms, not from one of them. These describe the trace rather than a
    # policy, but they do not all surface in every arm: only as-shipped replays
    # the lifetimes the trace actually carried, so a 30m marker was invisible
    # when this read litellm-auto, which places its own 5m markers.
    def _worst(field):
        return max(a[field] for a in pess.values())

    skipped = _worst("unstructured")
    untimed = _worst("untimed")
    unpriceable = _worst("unpriceable")
    unmodelled = _worst("unmodelled_ttl")
    unknown_target = _worst("unmodelled_target")

    # A verdict over a partial denominator is not a verdict. Gate 1 asks whether
    # deliberate placement beats automatic injection on a workload; if some of
    # that workload contributed nothing -- no structure, no timestamp, no
    # pricing, or a lifetime this replay cannot model -- then the comparison
    # describes whatever was left, and the omitted traffic could dominate spend
    # or behave entirely differently. Printing a percentage anyway is how a
    # subset gets read as a whole-workload result.
    # The denominator is a union of request identities, not a sum of per-arm
    # counts. One request can be unpriceable under one arm and carry an
    # unmodelled lifetime under another, and adding the maxima said more
    # requests were omitted than the trace contained -- "6 of 3 requests
    # contributed nothing", inside the sentence whose entire job is to explain
    # honestly why the gate cannot be answered.
    by_reason: dict = {}
    for arm in pess.values():
        for reason, ids in (arm.get("omitted") or {}).items():
            by_reason.setdefault(reason, set()).update(ids)
    everyone = set().union(*by_reason.values()) if by_reason else set()
    # Sizes the provider never confirmed cannot carry a dollar figure.
    #
    # Every arm's spend is linear in token counts, so scaling all sizes by k
    # scales every arm by k and the *ratio* survives -- measured: a 1,000x size
    # error left delta_pct at exactly 20.0% both times, so Gate 1's percentage
    # threshold is genuinely robust and guarding it would be theatre.
    #
    # The absolutes are not robust. The same error turned $0.125 of arm spend
    # into $125.00. This check first shipped as a verdict string only, while arm
    # spend stayed a bare float -- which made the guard depend on a reader
    # noticing the sentence. It now also sets the release state of the figures
    # themselves, below, so the number carries the fact and a renderer cannot
    # print it without one.
    #
    # Computed from the requests rather than accepted as an argument, because a
    # caller who has to pass a trust flag is a caller who can forget to.
    # The publication tolerance, not the coarse factor, and computed with the
    # loader's own helpers rather than a second copy of them. Both mattered: the
    # copy had drifted to the looser threshold, so sums at 0.51x and at exactly
    # 2.0x of billed released arm spend that was wrong by 49% and 100%; and the
    # copy carried the loader's `or not total: continue`, so a structured request
    # billing real tokens whose segments summed to zero printed $0.0000 with a
    # verdict beside it.
    misscaled = 0
    worst = 1.0
    for r in reqs:
        ratio = segment_sum_ratio([s.tokens for s in r.segments] if r.segments
                                  else None, r.usage or {})
        if ratio is None or sums_publishable(ratio):
            continue
        misscaled += 1
        if abs(ratio - 1.0) > abs(worst - 1.0):
            worst = ratio
    size_note = ""
    if misscaled:
        size_note = (
            f"{misscaled} of {len(reqs)} requests carry segment sizes that differ from "
            f"the tokens the provider billed by more than {PUBLISH_TOLERANCE:.0%} "
            f"(worst: {worst:.2f}x), so no per-arm dollar amount here can be read (a "
            f"uniform size error cancels out of the percentage; it does not cancel out "
            f"of the absolutes)")

    omitted = len(everyone)

    # Release: one decision, made once, applied to every arm at both assumption
    # ends, before any return exists to forget it. This is the structural half of
    # the size gate -- the verdict strings below explain *why* in prose, and the
    # figures themselves now carry the same fact so a renderer cannot print one
    # without it. Three return paths follow; none of them can get this wrong,
    # because none of them is where it happens.
    #
    # Both conditions, because an absolute fails for either reason. Mis-scaled
    # sizes make it a number of unknown magnitude; omitted requests make it a
    # subtotal over whatever was left, which the verdict already refuses to read
    # as a whole-workload result.
    # Sizes are necessary and were treated as sufficient. They are not: these
    # are dollar amounts, and this project's stated rule is that a dollar amount
    # is withheld until it ties to money that actually left an account. The
    # analyzer enforces that and this did not, so `cacheeconomics bakeoff` printed
    # $17.14 against $6.18 with no invoice anywhere in the command -- modelled
    # list-price figures wearing the authority of reconciled ones.
    #
    # The as-shipped arm is what an invoice can settle: it replays the lifetimes
    # the trace actually carried, so it is the arm that should match the bill.
    # The counterfactual arms are modelled from the same trace at the same rates,
    # so they inherit that credibility once as-shipped has earned it.
    #
    # The percentage is unaffected and still prints. It is scale-invariant, it is
    # what Gate 1 reads, and withholding it would be theatre.
    reconciled = None
    if invoice_usd is not None:
        try:
            shipped = pess["as-shipped"]["spend"].raw()
        except (KeyError, AttributeError):
            shipped = None
        if shipped is not None and invoice_usd > 0:
            reconciled = abs(shipped - invoice_usd) / invoice_usd <= 0.05
        else:
            reconciled = False

    # Rows that never reached the arms because they were not analysable. They
    # still cost money, so a figure computed without them describes a subset --
    # the same reason `omitted` blocks, arriving one layer earlier. `TraceSet`
    # derives this; a caller that hands over `ts.analysable` and nothing else is
    # exactly the path that published $0.27 over a missing 5M-token request.
    excluded_billed = excluded_billed or {}
    spend_ok = not misscaled and not omitted and not excluded_billed and (
        reconciled is True
        or money.draft_override_applies(reconciled is not None, allow_unreconciled))
    if excluded_billed:
        spend_why = (
            "the trace carries billed rows that no arm could model ("
            + ", ".join(f"{n} {reason}" for reason, n in sorted(excluded_billed.items()))
            + "), so every amount here describes a subset of the workload rather "
              "than its spend")
    elif not misscaled and not omitted and reconciled is None:
        spend_why = ("no invoice was supplied, so these are modelled list-price "
                     "amounts that have not been tied to money actually spent. "
                     "The percentage below does not depend on them")
    elif not misscaled and not omitted and reconciled is False:
        spend_why = (f"the as-shipped arm does not reconcile against the "
                     f"${invoice_usd:,.2f} invoice within 5%, so the modelled "
                     f"arms built from the same trace cannot be read as money")
    elif misscaled and omitted:
        spend_why = ("segment sizes do not reconcile with the tokens the provider "
                     "billed, and some requests contributed to no arm")
    elif misscaled:
        spend_why = ("segment sizes do not reconcile with the tokens the provider "
                     "billed, so this amount has an unknown scale")
    else:
        spend_why = (f"{omitted} of {len(reqs)} requests contributed nothing, so "
                     f"this is a subtotal and not the workload's spend")
    for arms_ in (pess, opt):
        for _p, _a in arms_.items():
            arms_[_p] = money.release_map(_a, spend_ok, spend_why)

    # Rows excluded before the arms ever ran belong in this branch for exactly
    # the reason stated above it: the comparison describes whatever was left,
    # and the excluded traffic could dominate spend or behave entirely
    # differently. They were wired to `spend_ok` and not to the verdict, so the
    # arms went to `[withheld]` while the headline still read "allocator-lite
    # beats the automatic baseline by 20.0% (gate: >=10%)" over a trace missing
    # a 5,000,000-token billed request. Withholding the dollars and keeping the
    # claim they support is the worse half of both options.
    if omitted or misscaled or excluded_billed:
        # Per-reason counts are reported alongside, never summed: they overlap,
        # and a request omitted for two reasons is still one request.
        reasons = ", ".join(f"{len(ids)} {name}"
                            for name, ids in sorted(by_reason.items()) if ids)
        if excluded_billed:
            excluded_note = (
                "indeterminate: the trace carries billed rows that never reached "
                "the arms ("
                + ", ".join(f"{n} {reason}"
                            for reason, n in sorted(excluded_billed.items()))
                + f"), so this compares {len(reqs)} request(s) and cannot answer "
                  f"the gate for the workload they came from. Those rows cost "
                  f"money and could behave entirely differently."
                + (f" Separately, {size_note}." if size_note else ""))
            return BakeOff(group=group, n_requests=len(reqs), window_days=window,
                           arms=pess, optimistic=opt, moves=moves,
                           unstructured=skipped, untimed=untimed,
                           unpriceable=unpriceable, unmodelled_ttl=unmodelled,
                           unmodelled_target=unknown_target,
                           verdict=excluded_note, verdict_relocation=excluded_note,
                           delta_pct=None, delta_pct_relocation=None,
                           delta_pct_optimistic=None,
                           delta_pct_relocation_optimistic=None)
        if not omitted:
            # Sizes alone. Returned through the same path rather than an earlier
            # one, because a short-circuit made this check mask the omission
            # verdict on a trace where both were true -- one guard hiding
            # another is its own defect.
            only_sizes = f"indeterminate: {size_note}."
            # Carries the same counters as the path below even though reaching
            # here means every one of them is zero: that holds only because all
            # five increment sites also record the request in `omitted`, which is
            # an invariant of five separate branches rather than of this return.
            # Passing them costs nothing and this path stops depending on it.
            return BakeOff(group=group, n_requests=len(reqs), window_days=window,
                           arms=pess, optimistic=opt, moves=moves,
                           unstructured=skipped, untimed=untimed,
                           unpriceable=unpriceable, unmodelled_ttl=unmodelled,
                           unmodelled_target=unknown_target,
                           verdict=only_sizes, verdict_relocation=only_sizes,
                           delta_pct=None, delta_pct_relocation=None,
                           delta_pct_optimistic=None,
                           delta_pct_relocation_optimistic=None)
        indeterminate = (
            f"indeterminate: {omitted} of {len(reqs)} requests contributed nothing "
            f"({reasons}; a request can count under more than one reason, so these "
            f"overlap and do not sum), so this compares a subset and cannot answer "
            f"the gate. Fix the ingest, or state the denominator and read it as a "
            f"subset result."
            + (f" Separately, {size_note}." if size_note else ""))
        return BakeOff(group=group, n_requests=len(reqs), window_days=window,
                       arms=pess, optimistic=opt, moves=moves, unstructured=skipped,
                       untimed=untimed, unpriceable=unpriceable, unmodelled_ttl=unmodelled,
                       unmodelled_target=unknown_target, verdict=indeterminate, verdict_relocation=indeterminate,
                       delta_pct=None, delta_pct_relocation=None,
                       delta_pct_optimistic=None, delta_pct_relocation_optimistic=None)

    return BakeOff(group=group, n_requests=len(reqs), window_days=window,
                   arms=pess, optimistic=opt, moves=moves, unstructured=skipped,
                   untimed=untimed, unpriceable=unpriceable, unmodelled_ttl=unmodelled,
                   unmodelled_target=unknown_target, verdict=_verdict("allocator-lite", d_place),
                   verdict_relocation=_verdict("relocation-lite", d_reloc, eval_gated=True),
                   delta_pct=d_place, delta_pct_relocation=d_reloc,
                   delta_pct_optimistic=delta(opt, "allocator-lite"),
                   delta_pct_relocation_optimistic=delta(opt, "relocation-lite"))


def _verdict(arm: str, delta: float | None, eval_gated: bool = False) -> str:
    if delta is None:
        return "no baseline spend to compare against"
    tail = ""
    if eval_gated and delta > 0:
        tail = (". Modeled and eval-gated: not claimable until a behavioural eval "
                "shows no material regression")
    if delta >= GATE_THRESHOLD_PCT:
        return (f"{arm} beats the automatic baseline by {delta:.1f}% "
                f"(gate: >={GATE_THRESHOLD_PCT:.0f}%){tail}")
    if delta > 0:
        return (f"{arm} is {delta:.1f}% cheaper, under the {GATE_THRESHOLD_PCT:.0f}% "
                f"gate: on this workload the compiler is a linter, not a product{tail}")
    if delta == 0:
        return f"{arm} ties the automatic baseline on this workload"
    return f"{arm} is {-delta:.1f}% WORSE than automatic injection on this workload"


def bake_off_by_agent(reqs: list[Request], on_date: str | None = None,
                      effective_rate: float | None = None,
                      invoice_usd: float | None = None,
                      allow_unreconciled: bool = False,
                      excluded_billed: dict | None = None) -> list[BakeOff]:
    """Per agent, because a single blended number hides the interesting cases."""
    groups: dict[str, list[Request]] = {}
    for r in reqs:
        groups.setdefault(r.agent, []).append(r)
    # The invoice covers the whole workload, not one agent's share of it, so a
    # per-agent run cannot reconcile against it. Passing it through would let
    # each group compare its own slice to the full bill and release on whichever
    # slice happened to land within 5%.
    # Passed to every group rather than apportioned. An excluded billed row
    # cannot be attributed to an agent -- a failed request may carry no agent at
    # all, and an unreadable line certainly does not -- so the honest statement
    # is that no group's figure describes complete spend.
    out = [bake_off(rs, group=g, on_date=on_date, effective_rate=effective_rate,
                    allow_unreconciled=allow_unreconciled,
                    excluded_billed=excluded_billed)
           for g, rs in sorted(groups.items()) if len(rs) >= 3]
    # Ranking, which is the other thing `raw()` is for: ordering groups by size
    # has to work before anyone decides whether the sizes may be printed.
    out.sort(key=lambda b: -(b.arms["litellm-auto"]["spend"].raw()))
    return out
