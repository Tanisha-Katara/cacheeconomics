"""The four token classes, and what they cost.

Every input token lands in exactly one of four buckets, and they bill at
different rates. That is the entire economic content of prompt caching, and
almost no tooling separates them: agents commonly count input tokens
client-side, which makes a cache read and a cache write indistinguishable in
their own telemetry.

Nothing here rounds toward optimism. `Spend.saving_vs_uncached` can be
negative, and it frequently is on a workload that writes caches nothing reads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from . import registry

# The counters a response can carry, top level and inside the split. Named
# rather than "whatever keys are present", so an exporter adding a field cannot
# smuggle an unvalidated number into a price.
_TOP_COUNTS = ("input_tokens", "cache_read_input_tokens",
               "cache_creation_input_tokens")
_SPLIT_COUNTS = ("ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens")


def _positive_count(value) -> bool:
    """A count that is evidence something was actually billed."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return value == value and value not in (float("inf"), float("-inf")) and value > 0


def _check_counts(usage: dict) -> None:
    """Reject anything that is not a finite, non-negative token count.

    Booleans are rejected explicitly: `isinstance(True, int)` is True in Python,
    so a flag an exporter wrote into a counter field priced as one token.
    """
    def bad(name, value):
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return f"{name}={value!r} is not a number"
        if value != value:
            return f"{name} is NaN"
        if value in (float("inf"), float("-inf")):
            return f"{name} is infinite"
        if value < 0:
            return f"{name}={value:,} is negative"
        return None

    problems = [p for p in (bad(k, usage.get(k)) for k in _TOP_COUNTS) if p]
    split = usage.get("cache_creation")
    if split is not None:
        if not isinstance(split, dict):
            problems.append(f"cache_creation={split!r} is not an object")
        else:
            problems += [p for p in (bad(f"cache_creation.{k}", split.get(k))
                                     for k in _SPLIT_COUNTS) if p]
    if problems:
        raise ValueError(
            "this response carries usage counters that cannot be priced: "
            + "; ".join(problems)
            + ". Pricing multiplies these by a rate and sums them across a "
              "trace, so a negative one subtracts from real spend rather than "
              "failing -- measured, a single negative row cancelled a $10 "
              "request to $5.00, matched a $5.00 invoice exactly, and passed "
              "the publication gate.")


@dataclass(frozen=True)
class Usage:
    """One request's input-side usage, split by class."""
    uncached_input: int = 0
    cache_read: int = 0
    cache_write_5m: int = 0
    cache_write_1h: int = 0

    @property
    def total(self) -> int:
        return (self.uncached_input + self.cache_read
                + self.cache_write_5m + self.cache_write_1h)

    @classmethod
    def from_anthropic(cls, usage: dict, *, ttl: str | None = None) -> "Usage":
        """Build from an Anthropic `usage` object.

        Prefers the authoritative source. Responses carry a `cache_creation`
        breakdown:

            "cache_creation": {"ephemeral_5m_input_tokens": 0,
                               "ephemeral_1h_input_tokens": 14979}

        which states the lifetime split outright. When it is present nothing is
        inferred, and a request that wrote both lifetimes is represented exactly
        — the scalar `cache_creation_input_tokens` cannot express that at all,
        so a single `ttl` argument silently prices mixed writes wrong.

        `ttl` remains required when there are writes and no breakdown, because
        the scalar alone does not say which lifetime produced it and a 1h write
        costs 2x against a 5m write's 1.25x. Defaulting it understates by 38%,
        silently, in the direction that flatters the tool. An earlier version of
        this method carried `ttl="5m"` directly beneath a docstring explaining
        why it must not.
        """
        # Validated here because this is the boundary where a number stops being
        # data and starts being money. Every counter was trusted raw, so a row
        # with `input_tokens: -1_000_000` priced at -$5.00 and *subtracted* from
        # the trace total: measured, one such row cancelled a real $10 request
        # down to exactly $5.00, matched a $5.00 invoice at 0.0%, passed the ship
        # gate and published. NaN produced `$nan`, `True` priced as one token,
        # and a string in the nested split raised TypeError out of the
        # multiplication instead of anywhere a caller could catch it.
        #
        # ValueError rather than a silent zero, because the analyzer already
        # routes that to `unprovable`: the row is excluded from every dollar
        # figure and the exclusion is reported. A malformed row must not be
        # priced, and it must not vanish quietly either.
        _check_counts(usage)
        # Presence and value are different questions. `or 0` collapsed
        # missing, None and an explicit 0 into one state, and the disagreement
        # check below was conditional on the value being truthy -- so a
        # provider response saying "0 written" alongside a split claiming 1,000
        # priced the 1,000 and said nothing. The mirror case, a nonzero scalar
        # against a zero split, raised. The guard failed loudly in one
        # direction and invented write spend in the other, which is the
        # direction that costs money.
        _created_stated = usage.get("cache_creation_input_tokens") is not None
        created = usage.get("cache_creation_input_tokens", 0) or 0
        split = usage.get("cache_creation") or {}
        m5, h1 = split.get("ephemeral_5m_input_tokens"), split.get("ephemeral_1h_input_tokens")

        # A lifetime this model does not know is not a lifetime worth zero.
        #
        # `trace.write_tokens` sums every positive value in this dict, because
        # for *detection* the provider saying it wrote is enough. Pricing needs
        # the lifetime, and this method read only the two Anthropic keys -- so a
        # response carrying `ephemeral_30m_input_tokens: 10000` and no aggregate
        # was analysable, counted as having writes, and priced at $0.00. The two
        # functions answer different questions and are allowed to; they are not
        # allowed to disagree silently in the direction of free.
        #
        # Raised rather than zeroed because the analyzer already routes
        # ValueError here to `unprovable`: excluded from every dollar figure, and
        # the exclusion reported. Schema drift should read as schema drift.
        unknown = {k: v for k, v in split.items()
                   if k not in _SPLIT_COUNTS and _positive_count(v)}
        if unknown:
            raise ValueError(
                f"cache_creation carries lifetimes this cost model does not "
                f"price: {', '.join(f'{k}={v:,}' for k, v in sorted(unknown.items()))}. "
                f"Each lifetime has its own write multiplier, so a rate cannot be "
                f"inferred from the count -- and treating the tokens as unwritten "
                f"prices real cache writes at zero. Add the lifetime to the "
                f"registry with a dated source before pricing it.")

        if m5 is not None or h1 is not None:
            m5, h1 = m5 or 0, h1 or 0
            if _created_stated and m5 + h1 != created:
                raise ValueError(
                    f"cache_creation breakdown sums to {m5 + h1:,} but "
                    f"cache_creation_input_tokens is {created:,}. These come from the "
                    f"same response and disagreeing means one is being misread; "
                    f"guessing which would put the error straight into a dollar figure.")
        else:
            if created and ttl not in ("5m", "1h"):
                raise ValueError(
                    f"this response has {created:,} cache write tokens, no cache_creation "
                    f"breakdown, and ttl={ttl!r}. The lifetime cannot be inferred from a "
                    f"usage object; it comes from the cache_control that was sent.")
            m5 = created if ttl == "5m" else 0
            h1 = created if ttl == "1h" else 0

        return cls(
            uncached_input=usage.get("input_tokens", 0) or 0,
            cache_read=usage.get("cache_read_input_tokens", 0) or 0,
            cache_write_5m=m5,
            cache_write_1h=h1,
        )


def write_lifetime(usage: dict, declared: str | None = None) -> str | None:
    """The lifetime this request's cache write used, or None if unprovable.

    Ordered by authority. Anthropic's `cache_creation` breakdown states the
    split outright, so a response carrying it settles the question without
    reference to what was requested -- and a usage-only export often carries
    nothing else. Only when the breakdown is absent does the declared lifetime
    (marker, then row) stand in.

    None means genuinely unknown *or* genuinely mixed, and callers have to treat
    those the same way: a request that wrote both lifetimes has no single answer,
    and inventing one is how a 2x write gets priced at 1.25x.

    Shared because two callers need it and both got it wrong separately. The
    rebuild rules ask "had the previous entry expired before this request
    arrived", which is unanswerable without it -- and each reached for a
    different, weaker source.
    """
    split = (usage or {}).get("cache_creation") or {}
    if isinstance(split, dict):
        m5 = split.get("ephemeral_5m_input_tokens") or 0
        h1 = split.get("ephemeral_1h_input_tokens") or 0
        if m5 and h1:
            return None                      # mixed; no single lifetime
        if m5:
            return "5m"
        if h1:
            return "1h"
    return declared


_LIFETIME_SECONDS = {"5m": 300, "1h": 3600}


def expiry_seconds(usage: dict, marker_lifetimes=(), declared: str | None = None):
    """When every entry this request wrote has definitely expired, or None.

    A different question from `write_lifetime`, and the distinction matters.
    Pricing needs *the* lifetime and has no answer for a request that wrote
    both, so `write_lifetime` returns None there. Expiry needs the longest one:
    a request that wrote a five-minute and a one-hour entry has nothing left
    only after the hour.

    Answering the pricing question and using it for expiry dropped every mixed
    write from rebuild classification entirely -- neither expiry nor rebuild,
    just gone -- so a genuine rebuild a minute after a live one-hour entry
    vanished from both the report and the live alerts.

    Shared so the batch rule and the runtime cannot answer it differently; they
    already have, twice.
    """
    split = (usage or {}).get("cache_creation") or {}
    if isinstance(split, dict):
        written = [ttl for ttl, key in (("5m", "ephemeral_5m_input_tokens"),
                                        ("1h", "ephemeral_1h_input_tokens"))
                   if split.get(key)]
        if written:
            return max(_LIFETIME_SECONDS[t] for t in written)
    if marker_lifetimes:
        seconds = [_LIFETIME_SECONDS.get(t) for t in marker_lifetimes]
        return max(seconds) if all(s is not None for s in seconds) else None
    return _LIFETIME_SECONDS.get(declared or "")


@dataclass(frozen=True)
class Spend:
    usd: float
    uncached_usd: float
    read_usd: float
    write_usd: float
    hypothetical_uncached_usd: float
    breakdown: dict = field(default_factory=dict)

    @property
    def saving_vs_uncached(self) -> float:
        """Positive means caching helped. Negative means it cost you money.

        The negative case is real and common: writes bill above the standard
        input rate, so a cache nothing reads is strictly worse than no cache.
        """
        return self.hypothetical_uncached_usd - self.usd

    @property
    def saving_pct(self) -> float | None:
        h = self.hypothetical_uncached_usd
        return None if h == 0 else 100.0 * self.saving_vs_uncached / h


# The three multipliers this cost model is shaped for. Named once because four
# call sites checked for them independently and two of them checked differently.
ANTHROPIC_SHAPED_MULTIPLIERS = ("read", "write_5m", "write_1h")


def is_multiplier(value) -> bool:
    """Is this a number a token count may be multiplied by?

    `bool` is excluded, and that is the whole reason this exists. `bool`
    subclasses `int`, so `isinstance(True, (int, float))` is True and a
    hand-edited registry row carrying `write_5m: true` sailed through every
    numeric guard in the package and priced at 1.0x. Measured on
    anthropic/direct: one million 5m-write tokens came back $5.00 instead of
    $6.25, a silent 20% understatement of the one figure the whole tool exists
    to get right -- and 1.0x is a plausible-looking number, so nothing
    downstream reads as broken.

    `read: true` is worse in a different way: 1.0x instead of 0.1x makes a cache
    read cost the same as fresh input, so the allocator models caching as
    worthless and declines to place a marker at all.

    This file already applied the rule one branch away -- `effective_rate`
    excludes `bool` explicitly, for the same reason and with the same
    consequence -- which is why it is now a named predicate rather than a
    condition each caller writes out.

    `None` fails here too. `read: null` is a real registry value on
    openai/direct, and multiplying by it raised a TypeError one layer from the
    cause instead of a refusal naming the surface.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def unusable_multipliers(mapping, keys=ANTHROPIC_SHAPED_MULTIPLIERS) -> list:
    """Which of `keys` are absent, null, or not a usable multiplier.

    Returns names rather than raising, so each caller can refuse in its own
    idiom: `cost.price` raises because it is where a figure is produced,
    `tiers._surface` raises `Unsupported` because a plan cannot be scored, and
    `analyzer._lifetime_multipliers` returns None because a rule that cannot
    price simply does not run. One predicate, three refusals.
    """
    m = mapping if isinstance(mapping, dict) else {}
    return [k for k in keys if not is_multiplier(m.get(k))]


def price(usage: Usage, model: str, target_id: str,
          on_date: str | None = None, effective_rate: float | None = None) -> Spend:
    """Cost this usage.

    `effective_rate` overrides list pricing and should be used whenever the
    customer's invoice is available. Enterprises at the volume this tooling
    targets rarely pay list, and reconciling computed spend against an invoice
    at list price guarantees a mismatch that looks like a tool defect.
    """
    # Only when nothing was supplied. `or` let "" and 0 through to today's rate
    # without ever reaching the registry's strict parser, and then stamped
    # `rate_source` with the substituted date -- so the report claimed an
    # effective date the caller never gave.
    if on_date is None:
        on_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if effective_rate is not None:
        # Validated here rather than at each caller. This value is multiplied by
        # every token class and is the one number a *client* supplies, from an
        # invoice they read by hand, through a CLI flag that parses any float.
        # Measured, all four flowed straight through to a published figure: NaN
        # gave `$nan`, infinity gave `$nan`, -5.0 gave -$5.00, and 0.0 made the
        # whole trace free. With --allow-unreconciled the report renders them,
        # and `--format json` emits bare NaN, which is not valid JSON.
        #
        # Zero is refused with the rest. A genuinely free workload is not a rate
        # override, and accepting it means "I could not read the invoice" and
        # "the invoice was zero" produce the same $0.00 report.
        bad = (isinstance(effective_rate, bool)
               or not isinstance(effective_rate, (int, float))
               or effective_rate != effective_rate
               or effective_rate in (float("inf"), float("-inf"))
               or effective_rate <= 0)
        if bad:
            raise ValueError(
                f"effective_rate must be a finite positive USD per million "
                f"tokens, got {effective_rate!r}. It multiplies every token "
                f"class, so a negative or non-finite value does not fail -- it "
                f"produces a published dollar figure that is wrong.")
    if effective_rate is not None:
        rate = effective_rate
    else:
        # The rate table is Anthropic first-party list price, not a global price
        # for the model. Bedrock and Vertex are partner-operated and invoiced by
        # the cloud provider. Without this check `base_rate` answered for every
        # surface, so a Bedrock trace priced at Anthropic rates and published a
        # confident total the client's AWS bill would contradict -- the one
        # failure mode this tool exists to prevent, arriving through the pricing
        # path rather than the measurement one.
        registry.require_priceable(target_id)
        # The caller's surface, not the rate table's default. This dropped it
        # entirely, so `base_rate` fell back to its own first-party default and
        # the scope check `require_priceable` had just performed was decided
        # against one surface while the rate came from another. Harmless only
        # because the check above already refused unpriceable surfaces -- which
        # is a guard holding up a bug, not an absence of one.
        rate = registry.base_rate(model, on_date, target_id)
    m = registry.multipliers(target_id)
    # The registry publishes surfaces whose pricing does not follow the
    # Anthropic shape -- openai/direct records `read: null` and its own write
    # keys -- and multiplying by None raised a TypeError that aborted the whole
    # report. A surface this model cannot price has to say so, so the analyzer
    # can exclude those requests and fail the publication gate the same way it
    # does for an unknown model.
    missing = unusable_multipliers(m)
    if missing:
        raise registry.RegistryError(
            f"{target_id} does not record Anthropic-shaped multipliers "
            f"({', '.join(missing)} absent, null or not a number), so this cost "
            f"model cannot price it. "
            f"Add a target-specific pricing path before analysing this surface.")
    per = rate / 1_000_000

    uncached = usage.uncached_input * per
    read = usage.cache_read * per * m["read"]
    w5 = usage.cache_write_5m * per * m["write_5m"]
    w1 = usage.cache_write_1h * per * m["write_1h"]

    return Spend(
        usd=uncached + read + w5 + w1,
        uncached_usd=uncached, read_usd=read, write_usd=w5 + w1,
        hypothetical_uncached_usd=usage.total * per,
        breakdown={
            "rate_usd_per_mtok": rate,
            "rate_source": "effective (invoice)" if effective_rate is not None else f"list, effective {on_date}",
            "multipliers": dict(m),
            "tokens": {
                "uncached_input": usage.uncached_input,
                "cache_read": usage.cache_read,
                "cache_write_5m": usage.cache_write_5m,
                "cache_write_1h": usage.cache_write_1h,
            },
        },
    )


def ratios(usages: list[Usage]) -> dict:
    """The two ratios worth putting on a dashboard.

    `input_from_cache` answers "how much of what I sent was cheap".
    `prefix_efficiency` answers "is what I chose to cache actually being
    reused" — a workload can look busy on the first and still be losing money
    on the second, which is the case that costs real money.
    """
    reads = sum(u.cache_read for u in usages)
    writes = sum(u.cache_write_5m + u.cache_write_1h for u in usages)
    uncached = sum(u.uncached_input for u in usages)
    total = reads + writes + uncached
    return {
        "requests": len(usages),
        "cache_read_tokens": reads,
        "cache_write_tokens": writes,
        "uncached_tokens": uncached,
        "input_from_cache": (reads / total) if total else None,
        "prefix_efficiency": (reads / (reads + writes)) if (reads + writes) else None,
    }


def ttl_crossover(target_id: str) -> dict:
    """Where the one-hour cache is worth its write premium.

    Derived from the registry rather than asserted, so it stays correct if a
    provider changes a multiplier. Both boundaries belong to the cache, not the
    price list, so this is model-independent: changing the model rescales the
    money without moving the window.

    `target_id` is required rather than defaulted, and required rather than
    accepting `UNATTRIBUTED`, because every number below is read out of one
    surface's row: the lifetimes it offers and its read and write multipliers.
    A default named `anthropic/direct`, so a caller who never chose a surface
    was told which lifetime wins on a surface they may not be using -- and
    deepseek/direct, one row away, offers no 1h lifetime at all. Same reasoning
    as `registry.base_rate`, which took only model and date until the surface it
    had erased turned out to decide the answer.
    """
    # Applicability first. An implicit-prefix surface has neither TTLs nor
    # write multipliers, so asking for multipliers before checking would raise
    # where the honest answer is simply "this question does not apply here".
    ttls = registry.capability(target_id, "supported_ttls")
    if "1h" not in ttls:
        return {"applicable": False,
                "reason": f"{target_id} does not offer a 1h TTL, so there is no window",
                "control_model": registry.target(target_id).get("control_model")}
    m = registry.multipliers(target_id)
    # The registry publishes surfaces whose pricing does not follow the
    # Anthropic shape -- openai/direct records `read: null` and its own write
    # keys -- and multiplying by None raised a TypeError that aborted the whole
    # report. A surface this model cannot price has to say so, so the analyzer
    # can exclude those requests and fail the publication gate the same way it
    # does for an unknown model.
    missing = unusable_multipliers(m)
    if missing:
        raise registry.RegistryError(
            f"{target_id} does not record Anthropic-shaped multipliers "
            f"({', '.join(missing)} absent, null or not a number), so this cost "
            f"model cannot price it. "
            f"Add a target-specific pricing path before analysing this surface.")
    return {
        "applicable": True,
        "window_seconds": (300, 3600),
        "below_window": {"winner": "5m",
                         "why": "both survive, so both requests read; the cheaper write wins the one-off"},
        "inside_window": {"winner": "1h",
                          "why": f"the 5m entry is gone and re-writes at {m['write_5m']}x while 1h reads at {m['read']}x"},
        "above_window": {"winner": "5m",
                         "why": f"both are gone every request, so {m['write_5m']}x beats {m['write_1h']}x"},
        "measured": "300s boundary and hit-refreshes-TTL confirmed by direct measurement 2026-07-28",
    }
