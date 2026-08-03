"""The trace schema, and how much you can conclude from a given one.

Response usage fields alone tell you what happened. They cannot tell you what
*would* have happened, because a counterfactual needs to know which parts of the
prompt were stable, and usage fields carry no structure at all.

So an ingest has a confidence tier, and the tier is not cosmetic: it decides
which questions the analyzer will answer and which it will refuse.

    INSTRUMENTED   segment ids and salted hashes recorded at source.
                   Everything is answerable.
    INFERRED       full request bodies, segmented after the fact. Answerable
                   with an alignment score attached to every structural claim.
    USAGE_ONLY     usage fields, no bodies. Ratios and diagnosis only.
                   No counterfactual, ever.

Prompt content is never required. Hashes, lengths and token counts are enough
for every finding here, which is also what makes local-first processing a real
promise rather than a marketing line.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


# Not every note is the same kind of thing. Most describe provenance -- which
# records were read, which ids were normalised -- and can sit out of the way.
# Some caveat a number that was published, and a caveat the reader never sees is
# the same as no caveat.
#
# The phrase below used to be the only marker, and both renderers decided a
# note's kind by searching for it. That made a rewording silently demote a
# blocker to provenance, which a review called correctly: classification driven
# by prose is not classification. It survives as wording that belongs in those
# sentences anyway, but `blocking_notes` is what the renderers read.
QUALIFIES_SPEND = "excluded from every dollar figure"


def note_blocks_spend(text: str) -> bool:
    """Does this note qualify a number that was published?

    One predicate, used at the point a note is raised rather than at the point
    it is rendered, so the decision is made by the code that knows why.
    """
    return QUALIFIES_SPEND in text


class Tier(Enum):
    INSTRUMENTED = "instrumented"
    INFERRED = "inferred"
    USAGE_ONLY = "usage_only"

    def __str__(self):
        return self.value

    @property
    def supports_counterfactual(self) -> bool:
        return self is not Tier.USAGE_ONLY


# Only a keyed scheme counts as instrumented identity. A bare digest is
# guessable over low-entropy segments, so a trace carrying them is inferred at
# best, whatever the exporter called it. (Equality across tenants is a separate
# concern that keying does not address either way -- `identity_input` scopes
# ids to the tenant for that.)
TRUSTED_ID_SCHEMES = ("hmac:",)

# The full shape, not just the prefix. `hmac:` alone and `hmac:redacted` both
# start with a trusted scheme, and accepting them classified a trace as
# instrumented while preserving ids that identify nothing -- so unrelated
# content collapsed into one apparently stable prefix and carried structural
# findings at high confidence.
_TRUSTED_ID = re.compile(r"^hmac:[0-9a-f]{64}$")


def is_trusted_id(value) -> bool:
    return isinstance(value, str) and bool(_TRUSTED_ID.match(value.strip()))


def segment_id(content: str, key: bytes | None = None) -> str:
    """Stable identifier for a segment's content.

    Keyed by default. A bare SHA-256 of a low-entropy segment (a short policy
    line, a product name, a status label) is guessable by dictionary attack, and
    the key removes that: without it, anyone holding a candidate prompt can
    confirm it by recomputing the digest.

    What the key does NOT do is hide equality. This docstring used to claim it
    made cross-tenant equality "impossible", and that was simply false -- one
    shared key means identical content produces an identical id no matter whose
    request it came from, so a reader of a multi-tenant trace could join on ids
    and learn that two tenants send the same policy text, without ever holding
    the key. Equality is the one property an id is *for*; keying cannot remove
    it. Scoping can, which is why the tenant is part of `identity_input` and
    reaches this function inside `content`.
    """
    b = content.encode("utf-8")
    if key is None:
        return "sha256:" + hashlib.sha256(b).hexdigest()
    return "hmac:" + hmac.new(key, b, hashlib.sha256).hexdigest()


def identity_input(role: str, kind: str, text: str, tenant: str | None) -> str:
    """What a segment id is computed over: tenant, container, kind, content.

    Identity has to cover the container. The same words in `system` and in a
    user turn are different bytes on the wire and will not share a cache entry,
    but hashing content alone gave them one id -- and a cached prefix is a tuple
    of ids, so moving text between roles read downstream as a cache hit. Block
    kind is included for the same reason: a text block and a tool_result
    carrying the same string are not interchangeable.

    The tenant is the outermost container and belongs here for exactly the same
    reason, which this function got wrong for longer than the rest. Caches are
    isolated on `(tenant, target_id, model)`, so two tenants sending identical
    text provably cannot share a cache entry -- yet they were given one id,
    which is both a false statement about the cache and a disclosure: a reader
    of a multi-tenant trace could join on that id and learn the two tenants
    share a policy block, with no key required. Scoping the id fixes both at
    once, because here the private thing and the correct thing are the same
    thing.

    `tenant` is explicit rather than defaulted so a caller cannot omit it by
    accident. None is legitimate and means a single-tenant trace, where there is
    no other tenant to collide with; it is rendered as a distinct sentinel so
    "no tenant" never collides with a tenant literally named "None".

    This lives here rather than in the recorder because both the live path and
    the post-hoc loader must agree. They did not: the recorder was fixed and the
    loader's fallback kept hashing content alone, which is the path inferred
    traces take.
    """
    who = "\x00single-tenant\x00" if tenant is None else f"\x00t:{tenant}\x00"
    return f"{who}{role}\x00{kind}\x00{text}"


@dataclass(frozen=True)
class Segment:
    """One contiguous piece of a prompt."""
    id: str
    role: str
    tokens: int
    index: int
    label: str = ""
    cache_marked: bool = False
    ttl: str | None = None


@dataclass
class Request:
    request_id: str
    sent_at: datetime
    model: str
    usage: dict
    segments: list[Segment] = field(default_factory=list)
    agent: str = "unknown"
    session: str | None = None
    tenant: str | None = None
    target_id: str = "anthropic/direct"
    first_token_at: datetime | None = None
    status: int = 200
    ttl_requested: str | None = None

    @property
    def cached_prefix_tokens(self) -> int:
        """Tokens up to and including the last cache-marked segment."""
        marked = [s.index for s in self.segments if s.cache_marked]
        if not marked:
            return 0
        return sum(s.tokens for s in self.segments if s.index <= max(marked))

    @property
    def breakpoints(self) -> int:
        return sum(1 for s in self.segments if s.cache_marked)

    @property
    def ttls_in_order(self) -> list[str]:
        return [s.ttl for s in sorted(self.segments, key=lambda s: s.index)
                if s.cache_marked and s.ttl]

    @property
    def marker_lifetimes(self) -> set[str]:
        """Per-marker lifetimes proven by the segment data itself.

        A marked block with no ttl is ambiguous when every marker is silent:
        some row exporters omit per-marker TTLs and rely on `ttl_requested`.
        But once any marker names a lifetime explicitly, a silent sibling is no
        longer "unknown"; on Anthropic wire syntax it is the provider's 5m
        default. Keeping that rule here stops analyzer, monitor and replay paths
        from each inventing a slightly different version.
        """
        explicit = set(self.ttls_in_order)
        if explicit and any(s.cache_marked and not s.ttl for s in self.segments):
            explicit.add("5m")
        return explicit

    def ttl_by_marker_index(self) -> dict[int, str]:
        """Actual TTL to replay for each marked segment.

        Row-level `ttl_requested` is a fallback only when the segment data has no
        explicit marker lifetimes at all. If one marker says "1h" and another is
        silent, the silent one is still an actual 5m marker on the wire, not a
        copy of the row metadata.
        """
        explicit = bool(self.ttls_in_order)
        row_ttl = self.ttl_requested if self.ttl_requested in ("5m", "1h") else None
        silent_ttl = "5m" if explicit else (row_ttl or "5m")
        return {s.index: (s.ttl or silent_ttl)
                for s in self.segments if s.cache_marked}

    @property
    def has_usage(self) -> bool:
        # A mapping, then the fields. Membership alone was true for any object
        # that happened to contain the substring, so a `usage` written as a JSON
        # string passed the test and reached the cost model, which called `.get`
        # on it. Loaders normalise this now; the belt-and-braces check is here
        # because a Request can be constructed directly.
        #
        # The per-lifetime split counts as accounting too. Checking only the
        # three scalars meant a split-only export -- aggregate absent, writes
        # reported per lifetime -- answered False here, so the request never
        # reached `.analysable` and the coverage line called it "no usage
        # fields" while it carried 9,000 billable write tokens. Found by the
        # audit in test_twin_paths rather than by anyone reading this line,
        # which is the argument for the audit.
        #
        # A *positive* counter, not a present one. All three scalars set to zero
        # is a placeholder, not a request that cost nothing: measured, a
        # two-row trace with one real $10 request and one all-zero row reported
        # 100% coverage, passed the ship gate against a $10 invoice and released
        # the figure. That is the partial-denominator hole reopening through a
        # different door, one round after it was closed.
        #
        # Same rule `usage_from_response` and the LiteLLM adapter already apply.
        # This is where it belongs, because this is what decides `analysable`.
        if not isinstance(self.usage, dict):
            return False
        if any(_positive(self.usage.get(k)) for k in
               ("input_tokens", "cache_read_input_tokens",
                "cache_creation_input_tokens")):
            return True
        split = self.usage.get("cache_creation")
        return isinstance(split, dict) and any(
            _positive(v) for v in split.values())


@dataclass
class TraceSet:
    requests: list[Request]
    tier: Tier
    alignment: float | None = None      # INFERRED only: 0..1 segmentation confidence
    source: str = ""
    notes: list[str] = field(default_factory=list)
    # The subset of `notes` that qualifies a published figure. Recorded when the
    # note is raised, not recovered from its wording afterwards.
    blocking_notes: list[str] = field(default_factory=list)
    # Fraction of rows carrying prompt structure. 1.0 means every request can
    # take part in a counterfactual; anything less means some cannot, and the
    # difference has to be stated rather than averaged away.
    structural_coverage: float = 1.0
    # Share of structured requests whose segment tokens were *counted* rather
    # than divided up by byte share. See `tokens_are_counted`.
    #
    # Defaults to counted because estimation happens in the *loaders* and
    # nowhere else: a TraceSet built directly carries whatever tokens its caller
    # put in the segments, and inventing doubt about those would be the tool
    # second-guessing its own input. Both loaders set this from evidence --
    # `segment_tokens` on the body path, a per-row flag on the trace path -- and
    # both default it to zero when the evidence is absent.
    tokens_counted: float = 1.0
    # False when segment token sums do not agree with the tokens the provider
    # billed. Structure is still usable for identity; it is not usable for money.
    token_sums_reconciled: bool = True
    # Separate from the above because the thresholds differ: this one gates
    # money, and a size error that clears the coarse check can still be a
    # doubling. Defaults True for the same reason the other does -- a TraceSet
    # built without segments has nothing to disagree about.
    token_sums_publishable: bool = True
    # Rows the loader could not turn into a Request at all: unparseable JSON, no
    # recognisable body, no model name. Structured rather than only mentioned in
    # `notes`, because the reconciliation gate reads structure and a human reads
    # notes. An invoice covers the traffic that ran, not the traffic that
    # parsed, so a lossy export matching the invoice is a coincidence over an
    # unknown denominator rather than a reconciliation.
    skipped_rows: int = 0

    def __len__(self):
        return len(self.requests)

    @property
    def analysable(self) -> list[Request]:
        return [r for r in self.requests if r.has_usage and r.status == 200]

    @property
    def tokens_are_counted(self) -> bool:
        """Whether segment sizes are counted, or divided up by byte share.

        `_scale_to_measured` splits the billed input total between segments in
        proportion to their bytes. Measured against the provider's own
        tokenizer, that split lands at 19.2% median error per segment and 181%
        at worst, because dense JSON tool schemas run about 2.74 bytes per token
        where English prose runs 5.22.

        Every structural finding is costed from those sizes, and this package
        refuses to publish spend that reconciles worse than 5%. Holding the
        invoice to 5% while costing a recommendation from a 19% split is not a
        standard, it is two standards. `checks.ESTIMATE_BAND` already made the
        same call for the static linter: inside the band an estimate cannot
        decide the question, so it abstains.

        Both loaders estimate, so this is not an INFERRED-only concern -- the
        recorder runs `_scale_to_measured` on its own captures too.

        A trace that does not say is treated as estimated, because that is what
        every trace written before counting existed actually is.
        """
        return self.tokens_counted >= PUBLISH_TOLERANCE_COUNTED

    @property
    def excluded_billed(self) -> dict:
        """Rows that cost money and are not in `analysable`, by reason.

        `analysable` is the right input for the *arms* -- a failed call
        populated no cache entry, and every arm models cache entries. It is the
        wrong denominator for a *dollar figure*, and `cmd_bakeoff` handed it
        straight to the simulator, so a 5,000,000-token request the provider
        failed and billed anyway simply left the comparison and the bake-off
        released $0.27 over the remainder.

        The analyzer already refuses on exactly these three, and this exists so
        the simulator can refuse on the same evidence rather than a second
        opinion about it. Deriving it here rather than at each caller is the
        point: the gate and its twin drifted once already this branch.
        """
        out: dict = {}
        failed = sum(1 for r in self.requests
                     if r.status != 200 and isinstance(r.usage, dict)
                     and _billed_input(r.usage) > 0)
        if failed:
            out["failed but billed"] = failed
        blind = int((self.coverage.get("excluded") or {}).get("no usage fields", 0))
        if blind:
            out["carrying no usage fields"] = blind
        if self.skipped_rows:
            out["unreadable in the export"] = int(self.skipped_rows)
        return out

    @property
    def structural_coverage_billed(self) -> float:
        """Share of *billed input tokens* on requests that carry structure.

        `structural_coverage` counts rows, and rows are the wrong denominator
        for money. Nine small structured requests beside one enormous
        usage-only request is 90% of rows and a rounding error of the bill, so
        a structural figure computed from the nine was released as if it
        described the workload. Measured on exactly that shape: 90% row
        coverage, 8.3% of billed tokens, and VOL-1 published $3,175 a month
        from traffic the structural rules had never seen.

        The invoice gate cannot catch it. Reconciliation proves total spend is
        right; it says nothing about whether the subset carrying structure is
        the subset the spend came from.

        Both numbers are true and answer different questions, which is why the
        row figure stays: it is what the coverage note reports, and a workload
        can be well covered by tokens while most of its *requests* are dark.
        This is the same split as `TOKEN_SUM_FACTOR` against
        `PUBLISH_TOLERANCE` -- one number for trust, a stricter one for money.

        A property rather than a loader field on purpose. Two loaders set
        `structural_coverage` independently and were free to diverge; deriving
        this from the requests means no ingest path can forget it and no third
        one can get it subtly different.
        """
        seen = total = 0
        for r in self.requests:
            billed = _billed_input(r.usage) if isinstance(r.usage, dict) else 0
            if billed <= 0:
                continue
            total += billed
            if r.segments:
                seen += billed
        # Nothing billed means no structural figure can be material either.
        return 1.0 if not total else seen / total

    @property
    def coverage(self) -> dict:
        """What fraction of requests can actually be analysed, and why not.

        A report that quietly analyses the 80% it could parse and presents the
        result as the whole picture is worse than one that says 80%.
        """
        total = len(self.requests)
        if not total:
            return {"total": 0, "analysed": 0, "fraction": None, "excluded": {}}
        excluded = {}
        for r in self.requests:
            if r.status != 200:
                excluded[f"status {r.status}"] = excluded.get(f"status {r.status}", 0) + 1
            elif not r.has_usage:
                excluded["no usage fields"] = excluded.get("no usage fields", 0) + 1
        n = len(self.analysable)
        return {"total": total, "analysed": n, "fraction": n / total, "excluded": excluded}


def _parse_ts(v):
    """Tolerant on purpose: an unparseable timestamp is an untimed row.

    The loader already survives bad JSON and missing fields, but this raised
    ValueError straight out of Request construction, so one non-ISO `sent_at`
    killed the whole file. The body ingest path caught it and this one did not
    -- the same tolerance written twice and only applied once.
    """
    if v is None:
        return None
    if isinstance(v, datetime):
        parsed = v
    else:
        try:
            parsed = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None
    # Everything comes back UTC-aware. An ISO string without an offset parses
    # naive, and mixing the two in one export made sorting raise -- so a valid
    # file with two timestamp styles took down the report rather than producing
    # one. A missing offset is read as UTC, which is what every provider emits.
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


def _first(row: dict, *names, default=None):
    """First present, non-empty alias. Exporters name the same field differently."""
    for n in names:
        v = row.get(n)
        if v not in (None, ""):
            return v
    return default


def resolve_tenant(row: dict, default: str | None = None) -> str | None:
    """Whose request this row is, by one rule with one implementation.

    The row wins, then the caller's default. Trivial, and it still has to live
    in exactly one place: segment ids are now scoped to the tenant, so this
    value is consumed twice per row -- once when the ids are hashed and once
    when the Request is built -- and the two consumers are in different
    modules.

    They diverged immediately. `load_bodies` hashed with the *caller's* tenant
    while `request_from_row` resolved the *row's*, so an export whose rows each
    carry their own tenant produced correct `Request.tenant` values over
    segment ids computed as if there were no tenant at all: two tenants sending
    identical prompts got identical ids, which is precisely the leak the
    scoping was added to close. Introduced in the same commit that closed it,
    and in the sibling of the loader that got it right.
    """
    return _text(_first(row, "tenant", "userId")) or _text(default)


def request_from_row(row: dict, segments: list, *, renamed: dict,
                     default_target: str = "anthropic/direct",
                     ttl_fallback: str | None = None,
                     model_override: str | None = None,
                     usage_override: dict | None = None,
                     default_tenant: str | None = None,
                     index: int = 0) -> "Request":
    """Build a Request from an exported row, applying every ingest guard once.

    This exists because the normalised loader and the body loader diverged on a
    guard four separate times: tolerant timestamp parsing, model normalisation,
    the trusted-id shape, and explicit status zero. Each was written for one and
    not the other, and each divergence silently removed data or crashed a run.
    Two functions doing the same job will keep drifting; one cannot.

    Segments differ between the two callers -- one reads them from the row, the
    other derives them from a logged body -- so they stay the caller's job.
    Everything else is here.
    """
    target = _text(_first(row, "target_id"), default_target)
    return Request(
        # `.get` throughout: a partial or malformed row should be something the
        # coverage line reports on, not a KeyError that takes the analysis with
        # it. A dropped file is a worse answer than a stated gap.
        request_id=_text(_first(row, "request_id", "id"), f"row-{index}"),
        sent_at=_parse_ts(_first(row, "sent_at", "startTime", "start_time",
                                 "timestamp", "requestTime", "created_at")),
        # The body loader reads the model from the logged request body, which
        # is more authoritative than whatever the exporter wrote on the row.
        model=_normalised(_text(model_override or _first(row, "model"), "unknown"),
                          renamed, target),
        # The body loader extracts usage from a nested response object rather
        # than a top-level field, so it hands the parsed dict in directly.
        usage=_usage_dict(usage_override if usage_override is not None
                          else row.get("usage")),
        segments=segments,
        agent=_text(_first(row, "agent", "name"), "unknown"),
        session=_text(_session_of(row)),
        # The row wins, then the caller's default. An export with no tenant
        # column produced tenant=None even when the operator had said which
        # tenant the file belonged to, and every tenant-scoped rule downstream
        # -- cache keys, volatility pools, fan-out, session splits -- then
        # merged traffic that can never share a cache entry.
        tenant=resolve_tenant(row, default_tenant),
        target_id=target,
        first_token_at=_parse_ts(_first(row, "first_token_at", "completionStartTime")),
        # Key presence, not truthiness: an explicit numeric 0 is a failed call,
        # and `or` chaining promoted it to success.
        status=_status(row["status"] if "status" in row
                       else row.get("status_code", 200)),
        ttl_requested=_text(_first(row, "ttl_requested"), _text(ttl_fallback)),
    )


# Segment sizes must land within this factor of the billed input total before
# any structural claim is trusted. The recorder scales them to the measured
# total, so a well-formed instrumented export agrees exactly -- both fixtures in
# this repository sit at 1.0000 across all 286 rows. A factor of two therefore
# never fires on anything plausible while catching the errors that matter:
# bytes instead of tokens is about 3.6x, characters worse, and a stale or
# rescaled estimate worse still.
TOKEN_SUM_FACTOR = 2.0

# ...but a factor of two is a plausibility check, not a reconciliation, and it is
# far too loose to stand behind a dollar amount. Measured: segment sums at 0.51x
# and at exactly 2.0x of the billed total both passed it, and the bake-off
# published per-arm spend off by 49% and 100% with a confident verdict beside it.
#
# 0.05 is inherited from the invoice gate this project already publishes against
# ("+/-5% to publish a figure"), not invented for this check, and it is generous
# rather than strict: the recorder scales segment sizes to the measured total, so
# honest instrumented data agrees exactly -- all 572 rows across both fixtures
# sit at 1.000000. A ratio outside this band means the structure and the usage
# counters are describing different prompts, and only one of them can be right.
PUBLISH_TOLERANCE = 0.05

# Share of structured requests that must carry counted tokens before a
# structural figure is money. Effectively all of them: a trace where some
# requests are counted and some are guessed produces a figure that is part
# measurement and part 19% error, with nothing saying which part.
PUBLISH_TOLERANCE_COUNTED = 0.99


def _billed_input(usage: dict) -> int:
    """Every input token the provider charged for, across all four classes.

    Lives here rather than in `segment`, which imports this module and now
    re-exports this. There were three copies of this sum: `segment`'s, the
    loader's, and the bake-off's -- and the two that guard money had each inlined
    the same fail-open beside it, which is the defect this consolidation removes
    rather than merely tidies.
    """
    # The per-lifetime split counts when the aggregate is absent. `cost.Usage`
    # prices writes off `cache_creation` when `cache_creation_input_tokens` is
    # missing, and this function did not -- so a split-only export was priced for
    # 9,000 write tokens while this reported 0 billed. Everything downstream then
    # went quiet in the wrong direction: `_scale_to_measured` had no measured
    # total to scale segment sizes to and fell back to byte estimates,
    # `segment_sum_ratio` returned None for "no opinion", and `sums_publishable`
    # passes None -- so the size gate that exists to stop structural dollars
    # being published from unmeasured sizes passed by default, on the one shape
    # where the sizes were guaranteed to be guesses.
    #
    # Two functions that must agree about "what was billed" had drifted apart,
    # and only one of them guarded money.
    # Each counter validated before it joins the sum. Raw addition let one bad
    # value disable the guard: a failed row carrying `input_tokens: 1,000,000`
    # beside `cache_read_input_tokens: NaN` summed to NaN, `nan > 0` is False,
    # so `failed_but_billed` saw an unbilled row. Measured: the 500 dropped out
    # of coverage, `excluded_billed` came back `{}`, reconciliation passed at
    # 0.0% and $5.00 published over the one surviving request.
    #
    # A malformed counter must never make a billed row look free. Summing the
    # valid ones answers the only question asked here -- did this cost money --
    # and a row whose counters are *all* malformed still fails `has_usage`,
    # which routes it to the blind-row blocker instead.
    return sum(v for v in (usage.get("input_tokens", 0),
                           usage.get("cache_read_input_tokens", 0),
                           write_tokens(usage))
               if _is_token_count(v) and v > 0)


def write_tokens(usage: dict) -> int:
    """How many tokens this response wrote to cache, however it was reported.

    Anthropic reports either the aggregate `cache_creation_input_tokens`, the
    per-lifetime `cache_creation` split, or both. Nine places in this package
    each read the aggregate directly and treated it as the answer, so every one
    of them saw zero on a split-only export -- while the cost model, which does
    read the split, went on billing it. The rebuild rule is the clearest case:
    REB-1 vanished on a trace that was still charged for 200,000 write tokens a
    request, because `created` was zero everywhere except in the price.

    The aggregate wins when present and non-zero. It is what the provider
    states, adding both would double-count the normal case where the split sums
    to it, and a split that disagrees with it is a data-quality problem --
    `cost.Usage.from_anthropic` is the place that refuses to guess between them.
    """
    creation = usage.get("cache_creation_input_tokens", 0) or 0
    if creation:
        return creation
    split = usage.get("cache_creation")
    if not isinstance(split, dict):
        return 0
    return sum(v for v in split.values() if _is_token_count(v))


def segment_sum_ratio(seg_tokens, usage: dict) -> float | None:
    """Segment tokens as a fraction of what the provider billed.

    `None` means the question does not apply: no segments, no usage, or nothing
    billed. A total of *zero* is not None. A structured request carrying billed
    tokens whose segments sum to nothing is the strongest disagreement available,
    and skipping it was a fail-open that let `$0.00` publish with the gate
    reporting a pass -- written once here and once in the bake-off, both spelled
    `if not billed or not total: continue`.
    """
    if not seg_tokens or not usage:
        return None
    billed = _billed_input(usage)
    if not billed:
        return None
    return sum(t for t in seg_tokens if _is_token_count(t)) / billed


def sums_within_factor(ratio: float | None,
                       factor: float = TOKEN_SUM_FACTOR) -> bool:
    """Coarse: are the two accounts even the same order of magnitude.

    Catches bytes instead of tokens (~3.6x) and placeholder usage totals. `None`
    passes, because there is nothing for the sizes to disagree with.
    """
    return True if ratio is None else (1 / factor) <= ratio <= factor


def sums_publishable(ratio: float | None,
                     tolerance: float = PUBLISH_TOLERANCE) -> bool:
    """Tight: may an absolute dollar figure derived from these sizes be printed.

    Symmetric about 1.0, unlike the factor form, which admits [0.5, 2.0] -- a
    band that hides a doubling.
    """
    return True if ratio is None else abs(ratio - 1.0) <= tolerance


# Where a conversation id can live. Explicit conversation identifiers only.
#
# `trace_id` is deliberately absent, and the reasoning went the wrong way once
# before. LiteLLM's schema says what that field is for: "Trace multiple LLM
# calls belonging to same overall request (e.g. fallbacks/retries)". That is the
# retries of one call, so each *turn* of a conversation carries a different one.
#
# Using it as a session looked like a weak-but-usable proxy and is worse than
# nothing. Measured on a realistic 14-request log with per-request trace ids:
# fourteen singleton sessions, REB-1 cannot fire because no session has a second
# request to compare against, and REB-0 -- the finding whose whole job is to say
# "rebuild detection is unavailable, here is what to instrument" -- is suppressed
# because sessions are non-null. The report goes quiet in both directions at
# once. A null session produces REB-0 and tells the operator what to fix.
#
# It also mislabels fallbacks: one trace id spanning a retry onto a different
# model reads as one conversation splitting models mid-flight.
SESSION_META_FIELDS = ("session_id", "conversation_id")
SESSION_TOP_FIELDS = ("litellm_session_id",)


def session_of(row: dict) -> str | None:
    """The conversation a request belongs to, or None if nothing says.

    One definition because there were two and they disagreed. Both now answer
    the same question the same way, which is the only way two paths stay agreed.

    None rather than a placeholder, and None rather than a per-request
    correlation id. Inventing a conversation out of unrelated calls reports every
    request as a rebuild of the last one; promoting a retry id to a session is
    quieter and worse, because it suppresses the finding that would have said the
    measurement was impossible.
    """
    if not isinstance(row, dict):
        return None
    meta = row.get("metadata")
    if isinstance(meta, dict):
        for k in SESSION_META_FIELDS:
            v = _text(meta.get(k))
            if v:
                return v
    for k in SESSION_TOP_FIELDS:
        v = _text(row.get(k))
        if v:
            return v
    return None


def _positive(value) -> bool:
    """A token count that is evidence something was actually billed."""
    return _is_token_count(value) and value > 0


def _is_token_count(value) -> bool:
    """A token count is a non-negative, finite number and not a boolean.

    `isinstance(True, int)` is True in Python, so a segment whose `tokens` field
    an exporter filled with a flag passed as a count of one. Infinities and NaN
    passed too, and a negative count would subtract from a prompt's cost.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return value >= 0 and value not in (float("inf"), float("-inf")) and value == value


def _segment_list(row):
    """The row's segments, if they are a list of anything at all.

    Exporters write this field as a string, a scalar, or a dict. Iterating it
    blind raised `TypeError: not iterable` or handed a string to code expecting
    a mapping -- and both took the entire report down rather than excluding one
    row.
    """
    segs = row.get("segments") if isinstance(row, dict) else None
    if not isinstance(segs, list):
        return []
    # Only the mappings. A list of scalars is not a segmented export, and
    # counting it as one made the loader demand an HMAC key to hash content
    # that does not exist -- refusing a row for the wrong reason, which reads
    # to the operator as a configuration error rather than a malformed file.
    return [s for s in segs if isinstance(s, dict)]


def _session_of(row: dict):
    """The conversation this row belongs to, wherever the exporter put it.

    Two field families, and only the first is this project's own. `session` and
    `session_id` at the top level are the normalised trace schema, where a
    session means a conversation because that is what the schema says. Everything
    else is somebody's gateway export, and is read through `session_of` so all
    three paths -- this loader, the live plugin, the LiteLLM adapter -- agree.

    `trace_id` and `traceId` used to be in here, on the reading that LiteLLM
    documented it as "spanning multiple calls". It spans the retries of *one*
    call, so each turn carries a different one. Using it produced a session per
    request: REB-1 cannot fire without a second request to compare against, and
    REB-0 -- which exists to report that rebuild detection is unavailable -- is
    suppressed because the session is non-null. This was the third of three
    places making that mistake, and it was found by the twin-path test written
    for the other two.
    """
    if not isinstance(row, dict):
        return None
    direct = _first(row, "session", "session_id")
    if direct is not None:
        return direct
    return session_of(row)


def _text(value, default=None):
    """An identity field as a string, or nothing.

    Model ids go into regexes and registry lookups; session, tenant and agent
    go into dictionary keys. An exporter writing any of them as a list or a
    dict raised `TypeError: unhashable type` or `expected string or bytes-like
    object` out of the loader and took the whole report with it -- the same
    class as the `usage`-as-a-string crash, one field over.

    Numbers are stringified because an exporter writing a numeric tenant id
    means it; anything structured is discarded, because there is no honest
    reading of a dictionary as a tenant.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return str(value)
    return default


def _usage_dict(value) -> dict:
    """Usage as a mapping, or nothing.

    Exporters write this field as a JSON string often enough to be worth
    parsing, and as a scalar often enough to be worth refusing. Stored raw, a
    string containing "input_tokens" satisfied `has_usage`'s membership test,
    survived into `analysable`, and reached `Usage.from_anthropic`, which called
    `.get` on it -- so one malformed row aborted the entire report with an
    AttributeError instead of being excluded and counted in coverage. A stated
    gap beats a dropped file; a crash is worse than either.
    """
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return {}
    if not isinstance(value, dict):
        return {}
    # And the values have to be numbers. `{"input_tokens": "ten"}` satisfied
    # every membership check, reached the cost model, and raised on `int + str`
    # -- a token count that is not a number is not a token count.
    out = {}
    for k, v in value.items():
        if isinstance(v, dict):
            out[k] = {ik: iv for ik, iv in v.items()
                      if isinstance(iv, (int, float)) and not isinstance(iv, bool)}
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = v
    return out


def _status(v) -> int:
    """Coerce to an int, because `analysable` compares against 200.

    An exporter writing "200" as a string made every successful request fail
    that comparison and vanish from spend and findings, with nothing in the
    coverage line to say why. An unparseable value stays excluded, which is the
    safe direction, but a numeric string is a success.
    """
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _normalised(model: str, renamed: dict, target_id: str | None = None) -> str:
    """Strip a snapshot date and the surface's id prefix, and say we did.

    base_rate refuses date-suffixed ids deliberately, so a loader that does not
    normalise hands the analyzer a model it will exclude as unpriceable. The
    Claude Code adapter has done this since it met a real trace; the other
    loaders had not.
    """
    from .registry import normalize_model
    base, stamp = normalize_model(model, target_id)
    if base != model:
        renamed[model] = base
    return base


def load_jsonl(path: str, key: bytes | None = None, *,
               default_tenant: str | None = None,
               default_target: str = "anthropic/direct") -> TraceSet:
    """Load a normalised trace file.

    The tier is derived from what the file actually contains rather than
    declared by the caller, so an export cannot claim more confidence than its
    contents support.

    `default_target` is the surface for rows that do not name one. It exists
    because the CLI offers `--target-id` on every ingest mode and this loader
    ignored it, so an operator who explicitly selected Bedrock still had their
    rows priced at Anthropic first-party rates -- the exact wrong-surface
    failure the rate scope was added to prevent, arriving through the loader
    instead of the price table. A row that names its own surface still wins:
    the flag supplies a default, never an override.
    """
    requests, notes = [], []
    rows, unparseable, renamed = [], 0, {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
                if not isinstance(parsed, dict):
                    # `42` and `[]` are valid JSON and are not rows. Appending
                    # them meant the first `.get` raised AttributeError and took
                    # the whole file with it -- a malformed line stopping the
                    # analysis instead of being counted as one, which is the
                    # tolerance every other shape on this path already has.
                    unparseable += 1
                    continue
                rows.append(parsed)
            except json.JSONDecodeError:
                # Append-only trace files get truncated by crashes, and a
                # partially written final line is the normal shape of that. One
                # bad row made an otherwise good capture unanalysable, while the
                # body-ingest path already counted them as gaps -- the same
                # tolerance, applied to one loader and not its twin.
                unparseable += 1

    has_segments = any(_segment_list(r) for r in rows)

    # A key being present is not proof of identity. `{"id": ""}` and
    # `{"id": null}` both satisfy `"id" in s`, which classified the trace as
    # instrumented, skipped the guards below, and then fell through to hashing
    # content anyway — unkeyed, or crashing when there was no content. An id
    # counts only if it is a non-empty string.
    def _identified(s):
        """A real id, not merely a non-empty string.

        Accepting any string let a bare `sha256:` digest -- which this module
        refuses to *generate* without a key, precisely because short segments
        are dictionary-guessable -- arrive from an exporter and receive
        instrumented confidence, while the report went on claiming keyed
        hashes. An id has to declare a scheme we trust.
        """
        # A segment that is not a mapping cannot be identified, and asking it
        # for `.get` is how a `segments` field written as a string took the
        # whole load down. Tier classification meets the raw row before the
        # loader has filtered anything, so the guard belongs here too.
        return isinstance(s, dict) and is_trusted_id(s.get("id"))

    # Structure has to cover every row, not merely appear in one.
    #
    # `any(...)` plus an ids check that skipped structureless rows meant a file
    # where half the requests carried segments was classified INSTRUMENTED. The
    # simulator then priced the other half from `sum(r.segments) == 0` — free —
    # undercounting spend while the report claimed full counterfactual support.
    # Partial structure is a coverage fact, so it is recorded and stated rather
    # than rounded up to complete.
    rows_with_segments = sum(1 for r in rows if _segment_list(r))
    structural_coverage = (rows_with_segments / len(rows)) if rows else 0.0
    fully_structured = bool(rows) and rows_with_segments == len(rows)

    has_ids = has_segments and all(
        all(_identified(s) for s in _segment_list(r))
        for r in rows if _segment_list(r)
    )

    # Identity has to be real or absent; a synthesised one is worse than none.
    #
    # Two ways this used to go wrong, both silent. An export that redacts
    # content but keeps the segment skeleton hashes every segment to the digest
    # of the empty string, so they all collapse to one id — which reads
    # downstream as a perfectly stable prefix and invents a cacheable span that
    # does not exist. And hashing content with no key produces a bare SHA-256,
    # which is dictionary-guessable over short low-entropy segments. (Keying
    # answers that one and not cross-tenant equality, which is scoped by putting
    # the tenant into `identity_input`.)
    # An id we do not trust is not an id. Those rows fall through to the
    # synthesis path, which requires a key, rather than being waved through.
    untrusted = has_segments and not has_ids and any(
        isinstance(seg, dict) and isinstance(seg.get("id"), str) and seg["id"].strip()
        for row in rows for seg in _segment_list(row))
    if untrusted:
        notes.append(
            "Some segments carried ids in a scheme this tool does not trust (only "
            "keyed hmac: ids count as instrumented, because a bare digest of a short "
            "segment is guessable by anyone holding a candidate prompt). They were "
            "re-derived with the supplied key, and this trace is inferred rather than "
            "instrumented.")

    synthesised = has_segments and not has_ids
    # Only segments that will actually be hashed need a key.
    hashable = synthesised and any(
        (s.get("content") if isinstance(s, dict) else None)
        for r in rows for s in _segment_list(r)
        if not _identified(s))
    unkeyed = synthesised and key is None and hashable
    redacted = synthesised and any(
        not (s.get("content") if isinstance(s, dict) else None)
        for r in rows for s in _segment_list(r)
        if not _identified(s))
    # Redaction is decided before the key requirement, because a key cannot fix
    # it. These segments have no content, so nothing would be hashed and the
    # trace downgrades to usage-only either way -- supplying a key changed
    # nothing except whether the load raised. So the default path refused a
    # deliberately privacy-preserving export (structure skeleton, no ids, no
    # content) for no protection at all, while the same file loaded fine with a
    # key it never used.
    if redacted:
        synthesised = False
        has_segments = False
        unkeyed = False
        notes.append(
            "Segments were present but some had neither an id nor content, so their "
            "identity is unknowable. Every such segment would hash identically and read "
            "as a stable prefix that does not exist, so structure is discarded and this "
            "trace is treated as usage-only.")
    if unkeyed:
        raise ValueError(
            "segments carry no ids, so identity would come from hashing content, and no "
            "HMAC key was supplied. A bare digest of a short segment is guessable by "
            "anyone holding a candidate prompt. Pass key=..., or export segment ids at "
            "source.")

    # Structure without sizes is not structure. A segment list with valid ids
    # and no `tokens` produced a prompt of zero tokens that still claimed full
    # structural coverage and an instrumented tier, so a counterfactual ran
    # against a prompt costing nothing and reported a confident, meaningless
    # result. An id says what a segment *is*; only a token count says what it
    # costs, and the bake-off needs both.
    sizeless = sum(1 for r in rows for s in _segment_list(r)
                   if not isinstance(s, dict) or not _is_token_count(s.get("tokens")))
    if has_segments and sizeless:
        # Every flag that grants structural confidence, not just the one that
        # builds segments. Clearing `has_segments` alone still left `has_ids`
        # true from an earlier pass, so the tier stayed INSTRUMENTED and
        # coverage stayed at 100% while the segments themselves were dropped --
        # the report would have claimed the strongest evidence tier for a trace
        # it had just decided not to trust.
        has_segments = False
        has_ids = False
        synthesised = False
        structural_coverage = 0.0
        fully_structured = False
        notes.append(
            f"{sizeless:,} segment(s) carry an identifier but no numeric token count. "
            f"Identity says what a segment is; only a size says what it costs, and a "
            f"zero-token prompt would enter the counterfactual as free. Structure is "
            f"discarded and this trace is treated as usage-only, which keeps measured "
            f"spend intact and withholds the comparison rather than certifying one "
            f"built on absent sizes.")

    # Segment sizes are a second source of truth for the same request, and the
    # invoice gate only ever validated the first.
    #
    # Spend is computed from `usage`; every structural finding is computed from
    # segment `tokens`. Nothing compared them, so an exporter writing bytes,
    # characters or a rescaled estimate reconciled perfectly against the invoice
    # -- because reconciliation never looks at segments -- and then released a
    # structural dollar figure derived from numbers no gate had examined.
    # Measured: a trace billing twelve cents released a VOL-1 figure of
    # $78,660,000 a month with the gate reporting a pass.
    #
    # Structure is *kept*, because identity is a separate question from size and
    # drift, volatility and relocation all read ids rather than counts. What is
    # withheld is money: `token_sums_reconciled` feeds the same gate that already
    # withholds structural figures on low alignment. Discarding structure instead
    # was the first attempt and it threw away working findings to fix a pricing
    # leak -- and tripped every fixture whose `usage` is a placeholder rather
    # than a billed total.
    # Two questions, two thresholds, one ratio. The coarse one decides whether
    # the sizes are trustworthy enough to describe structure at all; the tight
    # one decides whether a dollar amount derived from them may be printed.
    # Collapsing them into a single factor-of-two check meant a 2x size error
    # cleared the gate that guards money.
    misscaled = 0
    unpublishable = 0
    for r in rows:
        segs = _segment_list(r)
        usage = _usage_dict(r.get("usage"))
        ratio = segment_sum_ratio([s.get("tokens") for s in segs] if segs else None,
                                  usage)
        if ratio is None:
            continue
        if not sums_within_factor(ratio):
            misscaled += 1
        if not sums_publishable(ratio):
            unpublishable += 1
    token_sums_reconciled = not (has_segments and misscaled)
    token_sums_publishable = not (has_segments and unpublishable)
    if has_segments and misscaled:
        notes.append(
            f"{misscaled:,} request(s) carry segment sizes that do not sum to within "
            f"a factor of {TOKEN_SUM_FACTOR:g} of the tokens the provider billed. Both "
            f"describe the same prompt, so they cannot both be right, and spend is "
            f"computed from the usage counters while every structural claim is computed "
            f"from the segment sizes -- so an invoice can reconcile against one half "
            f"while a dollar figure is published from the other. Structural findings "
            f"are reported without figures. Bytes rather than tokens is the usual "
            f"cause; a placeholder usage total is the other.")
    elif has_segments and unpublishable:
        # The band between the two thresholds: close enough to describe the
        # prompt's shape, not close enough to price it. Said separately because
        # the advice differs -- the note above is a units bug to fix at the
        # exporter, this one is drift to close before quoting a number.
        notes.append(
            f"{unpublishable:,} request(s) carry segment sizes within a factor of "
            f"{TOKEN_SUM_FACTOR:g} of the billed input total but outside "
            f"{PUBLISH_TOLERANCE:.0%}, so structural findings are reported without "
            f"dollar figures. The shape is usable; the magnitude is not. A "
            f"well-formed instrumented export agrees exactly, because the recorder "
            f"scales segment sizes to the measured total -- a gap here usually means "
            f"part of the prompt is not being segmented, tool definitions most often.")

    for row in rows:
        segs = []
        # Resolved the same way `request_from_row` resolves it, and before the
        # segments are built, because ids are scoped to the tenant. Reading it
        # afterwards from the finished Request would be a second implementation
        # of the same precedence rule, which is how this file's twins drifted
        # four times already.
        row_tenant = resolve_tenant(row, default_tenant)
        for i, s in enumerate(_segment_list(row) if has_segments else []):
            if not isinstance(s, dict):
                # An exporter writing a scalar or a nested list where a segment
                # belongs is malformed input, not a segment with odd fields.
                continue
            # Same predicate that decided the tier. Using truthiness here meant
            # `""` fell through to hashing (correct) while `"   "` was kept as
            # the identifier (not), so every whitespace-id segment collapsed to
            # one id and read downstream as a perfectly stable prefix. Fixing
            # the classification without fixing the construction left the bug.
            sid = (s["id"].strip() if _identified(s)
                   else segment_id(identity_input(s.get("role", "user"),
                                                  str(s.get("type", "raw")),
                                                  s["content"], row_tenant), key))
            segs.append(Segment(
                id=sid, role=s.get("role", "user"), tokens=s.get("tokens", 0),
                index=s.get("index", i), label=s.get("label", ""),
                cache_marked=bool(s.get("cache_marked")), ttl=s.get("ttl"),
            ))
        requests.append(request_from_row(row, segs, renamed=renamed,
                                         default_target=default_target,
                                         default_tenant=default_tenant,
                                         index=len(requests)))

    if has_segments and not fully_structured:
        notes.append(
            f"Only {rows_with_segments:,} of {len(rows):,} requests "
            f"({structural_coverage:.0%}) carry prompt structure. The rest have usage "
            f"fields only, so they cannot take part in a counterfactual: a request with "
            f"no segments prices as zero tokens and would silently understate spend. "
            f"They are excluded from simulation and counted in the coverage line.")

    if has_ids:
        tier, alignment = Tier.INSTRUMENTED, None
    elif has_segments:
        # Not 1.0. Alignment is how well inferred boundaries match the real ones,
        # and that can only be scored against instrumented ground truth, which an
        # inferred trace by definition does not have. Claiming perfect alignment
        # for structure nobody checked is the strongest possible statement made
        # on the weakest possible evidence.
        tier, alignment = Tier.INFERRED, None
        notes.append("Segments were present but unidentified; ids derived by keyed hash of "
                     "content. No alignment score has been computed, so structural findings "
                     "are unvalidated — score the segmenter against an instrumented capture "
                     "before attaching confidence to them.")
    else:
        tier, alignment = Tier.USAGE_ONLY, None
        notes.append("No prompt structure in this export. Ratios and diagnosis only; "
                     "a counterfactual is not derivable from usage fields.")

    if renamed:
        notes.append(
            "Model ids normalised (date snapshots stripped so they price against the "
            "registry): " + ", ".join(f"{k} -> {v}" for k, v in sorted(renamed.items())))
    untimed_rows = sum(1 for r in requests if r.sent_at is None)
    if untimed_rows:
        notes.append(
            f"{untimed_rows} request(s) carry no usable timestamp. They are counted in "
            f"coverage but excluded from anything that depends on ordering or expiry.")
    if unparseable:
        notes.append(
            f"{unparseable} line(s) could not be parsed as JSON and are not represented "
            f"anywhere below. Figures describe the {len(requests):,} rows that did parse.")

    # A normalised trace says so per row, and the recorder has been saying it
    # since it was written: `tokens_are_estimated: True` is stamped on every row
    # it emits, with a comment explaining that nothing downstream should mistake
    # a proportional estimate for a counted quantity. Nothing downstream read
    # it. An explicit denial wins over an explicit affirmation, and absent means
    # estimated -- which is what every trace written before counting existed
    # actually is.
    def _counted(r):
        if r.get("tokens_are_estimated") is True:
            return False
        return r.get("tokens_counted") is True

    # Weighted by billed tokens, not by row count. Ninety-nine tiny counted
    # rows beside one huge uncounted one is 99% of rows and can be 0.02% of the
    # money -- and 99% clears the publish threshold, so every structural dollar
    # figure would rest on a byte-share estimate covering essentially all of the
    # spend. This package already learned that lesson once for structural
    # coverage, which is measured in billed tokens for exactly this reason;
    # `tokens_counted` was still counting rows.
    def _weight(r):
        # Type-checked: a hostile row can carry `usage` as a string, and
        # `_billed_input` calls `.get` on it. The malformed-ingest suite caught
        # this immediately, which is the whole reason it exists -- an ingest
        # that crashes on one bad row loses the entire file.
        u = r.get("usage")
        return (_billed_input(u) or 0) if isinstance(u, dict) else 0

    structured = [r for r in rows if _segment_list(r)]
    counted_rows = sum(_weight(r) for r in structured if _counted(r))
    structured_rows = sum(_weight(r) for r in structured)
    if not structured_rows:
        # No billed tokens to weight by. Fall back to rows so a trace of
        # zero-usage requests is not silently treated as fully counted.
        counted_rows = sum(1 for r in structured if _counted(r))
        structured_rows = len(structured) or 1
    return TraceSet(requests=requests, tier=tier, alignment=alignment,
                    source=path, notes=notes,
                    tokens_counted=counted_rows / structured_rows,
                    structural_coverage=structural_coverage,
                    token_sums_reconciled=token_sums_reconciled,
                    token_sums_publishable=token_sums_publishable,
                    skipped_rows=unparseable)
