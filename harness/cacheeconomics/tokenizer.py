"""Exact per-segment token counts, when you are willing to pay for them.

The INFERRED tier divides a request's billed input total between its segments
in proportion to their *bytes*. Measured against the provider's own tokenizer on
six agent-shaped bodies, that split has a median absolute error of 19.2% per
segment, a 90th percentile of 100%, and a worst case of 181%
(`tier-b/evidence/inferred-token-split.json`).

The cause is that bytes-per-token is not one number. Dense JSON tool schemas run
about 2.74 bytes/token, English prose about 5.22. A prompt mixing them -- every
agent prompt -- gets the prose over-allocated and the tools starved, in opposite
directions, inside the same request.

Three cheaper fixes were measured and none of them works:

    bytes / 3.6 (today)             median 19.2%   worst 181%
    tiktoken cl100k                 median 22.2%   worst  72%
    provider API, segments alone    median 23.3%   worst  83%
    provider API, in context        exact

tiktoken is a *different* vocabulary, and it came out worse on the median than
the constant it would replace, so it buys a dependency and loses accuracy.
Counting a segment on its own is worse again, because a tool schema inside
`tools` is not serialised the way the same JSON is as message content, and
because segments interact at their boundaries.

What is left is counting prefixes of the real body in the real shape and taking
differences. That is exact by construction -- it is how the evidence file was
produced -- and it costs one call per distinct prefix cut.

Which sounds ruinous and is not, because prompt caching only pays when the
prefix is stable, so the prefix is shared across requests and each cut is
counted once. Measured on the demo trace: 286 structured requests, 345 distinct
cuts, 1.2 calls per request. The provider's count-tokens endpoint is free.

**Nothing here opens a socket, and nothing in this package does.** This module
is arithmetic over a `count` callable somebody else supplies. The counter that
talks to the provider deliberately lives *outside* the installed package, in
`tier-b/count_tokens.py`, which enriches a bodies export with exact counts as a
separate and obvious step.

That split is not fussiness. `test_cli.TestNothingHereReachesTheNetwork` asserts
that no module in the package imports a network library at all, and its reason
is that zero egress "is the claim clients are asked to trust when they hand over
a trace". A client can verify that claim by grepping the wheel. Putting an HTTP
call behind a flag would have made the claim conditional on reading the flag's
implementation, which is a much weaker thing to ask someone to trust.
"""

from __future__ import annotations

import hashlib
import json
import os
import typing

from .registry import normalize_model, providers
from .segment import walk
from .trace import _first, _text, request_from_row

# `messages` is a required field, so a prefix consisting only of tools or system
# is not a countable request on its own. Every cut carries this sentinel, whose
# cost is identical in every call and therefore cancels out of the differences.
_SENTINEL = {"role": "user", "content": "."}


def prefix_cuts(body: dict) -> list:
    """Every wire prefix of `body`, in segment order, each a valid request.

    Rebuilt in the order `walk` yields -- tools, then system, then messages --
    because that is the order the segmenter indexes in, and a cut that did not
    match it would attribute one segment's tokens to another.
    """
    cuts, out, seen = [], {}, 0
    for role, _label, block, path in walk(body):
        field = path[0]
        if field == "tools":
            out.setdefault("tools", []).append(block)
        elif field == "system":
            if path[1] is None:
                out["system"] = block
            else:
                out.setdefault("system", []).append(block)
        else:
            _, mi, bi = path
            msgs = out.setdefault("messages", [])
            while len(msgs) <= mi:
                msgs.append(None)
            if msgs[mi] is None:
                msgs[mi] = {"role": role}
            if isinstance(bi, str):
                # Tool history is a field on the message, not a content block.
                # Appending it into `content` -- which is what happened when
                # `walk` learned to yield these and this reader was not
                # updated -- nested the whole `tool_calls` array inside the
                # assistant's content and produced a body no provider accepts.
                msgs[mi][bi] = block
            elif bi is None:
                msgs[mi]["content"] = block
            else:
                if not isinstance(msgs[mi].get("content"), list):
                    msgs[mi]["content"] = []
                msgs[mi]["content"].append(block)
        seen += 1
        snapshot = json.loads(json.dumps(
            {k: v for k, v in out.items()
             if k != "messages" or any(m is not None for m in v)}, default=str))
        if "messages" in snapshot:
            snapshot["messages"] = [m for m in snapshot["messages"] if m is not None]
        cuts.append(snapshot)
    return cuts


def _countable(cut: dict) -> dict:
    body = dict(cut)
    body["messages"] = list(body.get("messages") or []) + [_SENTINEL]
    return body


def _cache_key(cut: dict, counter_id: str = "") -> str:
    """A digest of the prefix, not the prefix.

    The key used to be `json.dumps(cut)`, so `<out>.cache.json` was a verbatim
    plaintext copy of every prompt counted -- and it was not gitignored. That
    directly contradicts the README's own promise that "prompt text is optional;
    hashes, structure and token counts are enough", in the one file this tool
    writes to a client's disk without being asked.

    Unkeyed sha256 rather than the keyed HMAC segment ids use. Segment ids are
    published in reports and joined across tenants, so they need a secret. This
    is a local resume file whose whole purpose is that a re-run finds the same
    key, which a per-run secret would defeat. A digest still lets somebody
    confirm a prompt they already guessed; it does not hand them the prompt,
    which is what this replaced.
    `counter_id` names the tokenizer that produced the count -- the model and
    the endpoint. Without it the key was the prefix alone, so a cache written by
    one tokenizer was reused by another without a single call being made, and
    those counts load as *exact*: `tokens_counted` reaches 1.0 and structural
    money is released on per-segment sizes from a model that was never asked.
    Different Claude generations tokenize differently, and a gateway endpoint
    may not be Anthropic's tokenizer at all.

    Scoped rather than versioned. A header recording the counter and refusing
    to resume on a mismatch is this codebase's usual reflex and it is the wrong
    one here: scoped keys let one cache file hold counts from several models
    safely, which is what an operator comparing two models does, and a mismatch
    self-heals -- the entries miss, the counter runs, the new ones are written
    beside the old. The worst case is API calls, not a published figure, which
    is the opposite trade from the places this project fails closed.
    """
    return hashlib.sha256(
        (counter_id + "\x00"
         + json.dumps(cut, sort_keys=True, default=str)).encode()).hexdigest()


# What `tier-b/count_tokens.py` adds to a row when it enriches it. Excluded from
# every digest below, and that is not tidiness -- it is the fix for a real
# defect. For a flattened export `_find_body` returns the ROW ITSELF, so `body`
# and `row` are the same object: storing `segment_tokens` on it mutated the very
# thing whose digest was about to be taken, and adding the record mutated it
# again. On reload the loader hashed the *enriched* flat row, the digest missed,
# and the row was estimated -- after its prompt prefixes had already been sent.
# Flat exports paid full egress and got nothing back.
#
# Excluding the enrichment keys makes a digest describe the request, not
# whatever the row later accumulates, so both sides compute it over the same
# bytes whether the export is nested or flat.
ENRICHMENT_KEYS = ("segment_tokens", "segment_tokens_provenance")


def request_view(obj):
    """`obj` without the fields this toolchain adds when it enriches a row."""
    if isinstance(obj, dict) and any(k in obj for k in ENRICHMENT_KEYS):
        return {k: v for k, v in obj.items() if k not in ENRICHMENT_KEYS}
    return obj


def _canonical(obj) -> bytes:
    """The one serialisation everything here digests.

    `sort_keys` because a JSON round trip through a different exporter must not
    invalidate every count, and `default=str` because an exporter that put a
    datetime in a body must not crash a freshness check. Both were already the
    convention `_cache_key` used; naming it stops a second one growing beside
    the first.
    """
    return json.dumps(obj, sort_keys=True, default=str).encode()


def body_sha256(body) -> str:
    """A digest of a request body, for saying which body a count came from.

    Lives here rather than in `tier-b/count_tokens.py` because both sides of the
    provenance gate need it and they must agree byte for byte: the counter
    writes it into each counted row, and `adapters.bodies` recomputes it to
    decide whether those counts may be trusted as exact. Two implementations of
    "sha256 of the body" that differ by one flag would make every digest
    mismatch, so every counted row would silently fall back to byte-share
    estimation -- the counting feature gone, and gone in the direction that
    looks fine.

    A digest, never the body. This value travels in a file on a client's disk,
    and the count cache one function up made exactly this mistake once already.
    """
    return hashlib.sha256(_canonical(request_view(body))).hexdigest()


def cuts_sha256(body) -> str:
    """A digest of the ordered prefix cuts a body segments into.

    The counts are differences between consecutive cuts, in cut order, so the
    cuts are what a `segment_tokens` array actually corresponds to -- not the
    body bytes. `load_bodies` accepts an array when its length, value types and
    positive sum match the freshly segmented body, and a body digest matching
    says only that the same bytes were counted at some point.

    That leaves a gap the body digest cannot see: re-segmenting a counted export
    while preserving the segment *count* keeps both the length check and the
    body digest satisfied while the counts no longer line up with the segments
    they are applied to. The row loads with `was_counted=True` carrying stale
    proportions, `tokens_counted` clears the publish gate, and structural
    dollars come out of proportions that describe a different segmentation.

    So this digests `prefix_cuts(body)` itself: the boundaries, their order, and
    their content. Change how a body is cut and every count taken under the old
    cutting stops being vouched for, whatever the segment count.
    """
    return hashlib.sha256(
        _canonical(prefix_cuts(request_view(body)))).hexdigest()


def row_sha256(row) -> str:
    """A digest of a whole export row, before enrichment.

    The body digest says the counted content is unchanged. It is blind to
    everything else the loader reads off the row -- usage, timestamps, status,
    session, and a top-level `model` that resolves which tokenizer should have
    answered. Digesting the whole row rather than an enumerated subset is
    deliberate: naming the analysis-relevant fields is a copy of the loader's
    knowledge, and those copies drift.
    """
    return hashlib.sha256(_canonical(request_view(row))).hexdigest()


def count_segments(body: dict, count, cache: dict | None = None,
                   counter_id: str = "") -> list:
    """Exact token count per segment, in segment order.

    `count` takes a request body and returns its input token count. `cache` is
    keyed on the cut itself, so a prefix shared by ten thousand requests is
    counted once -- which is the only reason this is affordable.
    """
    cache = {} if cache is None else cache
    cuts = prefix_cuts(body)
    if not cuts:
        return []

    def counted(cut):
        key = _cache_key(cut, counter_id)
        if key not in cache:
            cache[key] = count(_countable(cut))
        return cache[key]

    base_key = _cache_key({}, counter_id)
    if base_key not in cache:
        cache[base_key] = count(_countable({}))
    prev = cache[base_key]

    out = []
    for cut in cuts:
        n = counted(cut)
        # Never negative. A boundary can measure slightly under its predecessor
        # when the tokenizer merges across it, and a negative token count is
        # refused at the pricing boundary -- which would drop the whole request
        # rather than the one segment.
        out.append(max(0, n - prev))
        prev = n
    return out


def apply_counts(segments: list, counts: list) -> list:
    """Replace byte estimates with counted tokens, in place.

    Writes to `bytes` rather than `tokens` because `_scale_to_measured` still
    runs afterwards: it divides the *billed* total in proportion, and feeding it
    exact counts makes that division exact instead of merely normalising. The
    billed total remains the authority, so a body that does not match what was
    actually sent still cannot inflate the trace.
    """
    for s, n in zip(segments, counts):
        # Exact zeros preserved. `max(1, ...)` put back the invented token that
        # `_scale_to_measured` refuses to invent -- its own comment says "a
        # request with more segments than billed tokens is a real shape and
        # inventing a token per segment is how the total ran away in the first
        # place". Counting is the exact path; clamping its answer upward skews
        # every other segment's share of the billed input.
        s["bytes"] = max(0, int(n))
    return segments


# The model resolver. It lives here rather than in `tier-b/count_tokens.py`
# because both sides of the provenance gate need the same answer: the counter
# stamps the models it resolved, and the loader recomputes them to decide
# whether the counts may be trusted. A copy in tier-b would be a copy the
# package could not check.
class RowModels(typing.NamedTuple):
    """A row has three model-shaped things, not two, and they are three
    different questions.

    Answering them with fewer values is the mistake this type exists to make
    impossible, and it has now been made three times: the raw id recorded as the
    analysed model, then the normalised id sent to the tokenizer, then the raw
    id sent with a *surface routing prefix* still attached.

    `analysis` is what the report names and prices. Normalised through
    `request_from_row`: date stripped, surface prefix stripped. The registry is
    keyed on it.

    `tokenizer` is the id the count endpoint is asked for. The logged id with
    the surface's routing prefix removed but its snapshot date kept, because
    those two strippings answer different questions and `_normalised` does both:
    the date matters (if the bare alias has moved, only the dated id still means
    what the log meant) and the prefix does not (`anthropic.` is how Bedrock
    addresses a model, not a model id any tokenizer answers to).

    `prefix` is the routing prefix that was removed, or None. It belongs to
    neither of the others and is kept so the record can say what was dropped.

    `tokenizer` is None when the row names nothing a tokenizer could be asked
    for, or names something no endpoint could serve. Such a row is not counted
    and nothing is sent for it -- the point being that a call that cannot return
    a usable count is prompt content leaving the machine for nothing.
    """

    tokenizer: "str | None"
    analysis: str
    prefix: "str | None" = None


def _known_routing_prefixes() -> set:
    """Every surface id-prefix the registry knows about.

    Read from the registry rather than listed here, so a surface added to
    providers.json is covered without this file being edited.
    """
    return {t.get("model_id_prefix") for t in providers()["targets"]
            if t.get("model_id_prefix")}


def _without_routing_prefix(raw: str, target_id: str | None):
    """`(id to ask a tokenizer for, routing prefix removed)`.

    Returns `(None, prefix)` when the id carries routing decoration that cannot
    be resolved without knowing the surface. Refusing is the point: sending
    `anthropic.claude-opus-5` to api.anthropic.com is prompt content leaving the
    machine in a call that cannot come back with a usable count.

    The registry decides everything. `normalize_model` strips the prefix *and*
    the date; the date is re-attached from the second return value, and the
    result is required to be a suffix of the original -- if it is not, something
    other than a leading prefix was rewritten and this refuses rather than
    guessing what to send.
    """
    base, stamp = normalize_model(raw, target_id)
    dated = f"{base}-{stamp}" if stamp else base
    if dated != raw:
        if not raw.endswith(dated):
            return None, None        # not a leading-prefix difference
        raw, prefix = dated, raw[:len(raw) - len(dated)]
    else:
        prefix = None
    # A surface prefix survives when no --target-id was given, because
    # `normalize_model` needs the target to know what the prefix is. We cannot
    # tell routing from model id here, so nothing is sent.
    for known in _known_routing_prefixes():
        if raw.startswith(known):
            return None, known
    return raw, prefix


def row_models(row: dict, body: dict,
               target_id: str | None = None) -> RowModels:
    """The tokenizer to ask, and the model the report will name.

    Four rounds of review found four divergences here, each one level below the
    last, and the last two were the same mistake in opposite directions:

      round 1  the CLI's --model was stamped over every row
      round 2  only the extracted body was read, so a row naming its model at
               the top level resolved to the fallback
      round 3  precedence matched but the order of operations did not -- the
               loader picks the raw value first and coerces once, this coerced
               each candidate before choosing, and six shapes disagreed
      round 4  one value answered both questions. `request_from_row` returns the
               *normalised* id, so a request logged as claude-opus-5-20260101
               was counted as bare claude-opus-5: if that alias has moved, the
               counts came from a tokenizer the log never named, and are marked
               exact.
      round 5  stopping before `_normalised` kept the date (right) and also kept
               the surface's routing prefix (wrong). With
               --target-id amazon-bedrock/converse, a body model of
               anthropic.claude-opus-5 was sent verbatim to
               api.anthropic.com -- prompt content leaving the machine in a call
               that could not return a usable count. Egress for nothing, which
               is worse than a wrong count.

    So the analysis side calls `request_from_row`, because that function *is* the
    definition of `Request.model` and matching its behaviour is what failed
    twice. The tokenizer side takes the raw resolved value and removes only the
    routing prefix, and it asks the registry both what the prefix is and whether
    it applies -- `normalize_model(raw, target)` differing from
    `normalize_model(raw, None)` is the registry's own answer to the second
    question, so no copy of its guard lives here.

    The one line still transcribed from the caller is `model_override`, which is
    `bodies.load_bodies`'s argument rather than the resolver's own logic;
    `TestTheCounterAsksTheLoaderWhichModelThisIs` reads it back out of that
    source so it cannot drift either.

    There is no fallback. A row that names nothing gets `tokenizer=None` and is
    not counted; the analyzer estimates it and says so. A `--model` default here
    counted such a row with haiku while the report called it "unknown".
    """
    if not isinstance(row, dict):
        row = {}
    override = body.get("model") if isinstance(body, dict) else None
    kwargs = {"default_target": target_id} if target_id else {}
    analysis = request_from_row(row, [], renamed={}, model_override=override,
                                **kwargs).model
    # The same expression the loader resolves, stopping before `_normalised`.
    # `_text` still applies: a list or a dict is not an id anyone can send, and
    # the loader discards those too.
    raw = _text(override or _first(row, "model"))
    # Trimmed before deciding whether a model exists at all. Not a coercion --
    # `5` still resolves to "5", which round 2 established -- but whitespace is
    # not part of any id, and `"   "` was passing this test and putting a whole
    # prompt body on the wire under a model name of three spaces.
    raw = raw.strip() if isinstance(raw, str) else raw
    if not raw:
        return RowModels(None, analysis)
    tokenizer, prefix = _without_routing_prefix(raw, target_id)
    return RowModels(tokenizer, analysis, prefix)


# The vouching contract, in one place. The writer stamps it, the loader checks
# it, and the key and version were transcribed into both before this -- three
# copies of a constant whose whole job is that two sides agree on it.
COUNTS_PROVENANCE_KEY = "segment_tokens_provenance"
COUNTS_PROVENANCE_VERSION = 3


def recomputable_provenance(body, row=None, target_id=None) -> dict:
    """The record that vouches for counts taken from `body`.

    The minimum a reader needs to decide whether a `segment_tokens` array may be
    trusted as exact: which counter produced it, the body it was taken from, and
    the cuts it is differences of. `tier-b/count_tokens.py` adds what it knows
    on top -- the tokenizer that answered, the endpoint, the surface -- which
    says whether a *re-run* would agree; this is the narrower question of
    whether these counts describe this body.
    """
    models = row_models(row if row is not None else {}, body, target_id)
    return {"version": COUNTS_PROVENANCE_VERSION,
            "body_sha256": body_sha256(body),
            "cuts_sha256": cuts_sha256(body),
            "row_sha256": row_sha256(row if row is not None else body),
            "tokenizer_model": models.tokenizer,
            "analysis_model": models.analysis,
            "target_id": target_id}


def counts_provenance(body, row=None, target_id=None, endpoint=None,
                      tokenizer_id=None) -> dict:
    """The whole record a counted row carries.

    Everything a reader can recompute, plus the two things it cannot: which host
    answered, and which deployment the operator asserted was behind it. Both
    halves come from one function so that a record which satisfies this is a
    record the loader accepts -- an earlier split had the package emitting a
    "vouching record" that its own gate then rejected, which is the kind of seam
    a fixture discovers and a caller discovers in production.
    """
    return {**recomputable_provenance(body, row, target_id),
            "endpoint": endpoint, "tokenizer_id": tokenizer_id}
