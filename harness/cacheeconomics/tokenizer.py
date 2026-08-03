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

from .segment import walk

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


def _cache_key(cut: dict) -> str:
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
    """
    return hashlib.sha256(
        json.dumps(cut, sort_keys=True, default=str).encode()).hexdigest()


def count_segments(body: dict, count, cache: dict | None = None) -> list:
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
        key = _cache_key(cut)
        if key not in cache:
            cache[key] = count(_countable(cut))
        return cache[key]

    base_key = _cache_key({})
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
