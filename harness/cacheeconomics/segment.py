"""Turning an Anthropic request into segments. No IO, no state.

Split out of the recorder because two callers need it and only one of them can
open a file. The recorder uses it on live requests; the body adapter uses it on
request bodies a gateway logged; and the browser bundle needs it without
dragging in file handles, fsync and locks that mean nothing in a tab.

Sharing one implementation is the point. A body segmented here and a request
recorded live must produce identical ids for identical content, and two
implementations would drift the moment either was touched -- surfacing as
phantom volatility in somebody's report.
"""

from __future__ import annotations

import copy
import json

from .trace import (_billed_input, _is_token_count,  # noqa: F401
                    identity_input, segment_id, write_tokens)


# Rough bytes-per-token for English prose plus JSON punctuation. Only ever used
# to divide a measured total between segments, never to produce an absolute
# count, so the constant's inaccuracy cancels out of the proportions.
_BYTES_PER_TOKEN = 3.6


def _model_visible(block):
    """The block as the model sees it, with transport metadata removed.

    `cache_control` is an instruction to the cache, not content. Hashing it into
    the segment id meant the same prompt text scored as a different segment
    depending on whether it carried a marker, or which lifetime that marker
    asked for. Turning a caching config change into apparent prompt drift is
    precisely backwards: it poisons the volatility signal that decides where
    markers go, so the tool would react to its own recommendations.
    """
    if isinstance(block, dict) and "cache_control" in block:
        return {k: v for k, v in block.items() if k != "cache_control"}
    return block


def _canonical(block) -> str:
    """Stable text for hashing.

    Sorted keys so a dict that serialises in a different order on a later run
    does not read as a changed segment. Prompt-order instability is a real
    finding; serialiser-order instability is an artefact, and conflating them
    would manufacture volatility that costs nobody anything.
    """
    block = _model_visible(block)
    if isinstance(block, str):
        return block
    return json.dumps(block, sort_keys=True, separators=(",", ":"), default=str)


def _block_kind(block) -> str:
    if isinstance(block, dict):
        return str(block.get("type", "dict"))
    return "raw"


def _identity_input(role: str, block, text: str, tenant: str | None) -> str:
    """Shared with the post-hoc loader, so live and inferred ids agree."""
    return identity_input(role, _block_kind(block), text, tenant)


def _cache_marker(block) -> tuple[bool, str | None]:
    """Whether this block carries cache_control, and at what lifetime."""
    if not isinstance(block, dict):
        return False, None
    cc = block.get("cache_control")
    if not isinstance(cc, dict):
        return False, None
    return True, cc.get("ttl")


def walk(request: dict):
    """Yield `(role, label, block, path)` in wire order: tools, system, messages.

    The single traversal of a request body in this package. Reading segments out
    and writing cache markers back in have to agree on what "index 3" means, and
    the reliable way to make two functions agree about an order is to give them
    one implementation of it. This branch has already produced four defects from
    twin functions drifting apart on a guard.

    `path` locates the block for mutation: `("tools", i)`, `("system", None)` for
    a bare string, `("system", i)`, or `("messages", i, j)` with `j` None when
    the message content is a bare string.
    """
    for i, tool in enumerate(request.get("tools") or []):
        # Positional, not the tool's name. A tool name is model-visible prompt
        # content -- `query_internal_billing_ledger` names an internal API, a
        # product, sometimes a customer -- and it was being copied verbatim into
        # the segment label, which the recorder persists and the report renders.
        # Hashing the block and then printing its name beside the hash gives
        # away most of what the hash was protecting.
        yield "tools", f"tool[{i}]", tool, ("tools", i)

    system = request.get("system")
    if isinstance(system, str):
        yield "system", "system", system, ("system", None)
    elif isinstance(system, list):
        for i, block in enumerate(system):
            yield "system", f"system[{i}]", block, ("system", i)

    for mi, m in enumerate(request.get("messages") or []):
        role = m.get("role", "user") if isinstance(m, dict) else "user"
        content = m.get("content") if isinstance(m, dict) else m
        # OpenAI-shaped tool history, which this package had no word for: both
        # names appeared zero times across the whole repo. Neither is `content`,
        # so an assistant message that only calls a tool produced no segment at
        # all, and the call id -- which LiteLLM regenerates per call -- sat in
        # the middle of the prefix invalidating everything behind it while the
        # segmenter reported the prompt as perfectly stable. It is on the wire,
        # so it is part of the prefix, so it gets an identity.
        #
        # The path's last element is a name rather than an index, which is what
        # makes these positions unmarkable: `_mark` indexes into `content` and
        # neither field lives there. Seeing drift is the whole job here;
        # rewriting somebody else's tool protocol is not.
        if isinstance(m, dict):
            for name in ("tool_calls", "tool_call_id"):
                value = m.get(name)
                if value is not None:
                    yield role, f"{role}:{name}", value, ("messages", mi, name)
        if isinstance(content, str):
            yield role, role, content, ("messages", mi, None)
            continue
        for bi, block in enumerate(content or []):
            kind = block.get("type", "text") if isinstance(block, dict) else "text"
            yield role, f"{role}:{kind}", block, ("messages", mi, bi)


def segments_from_request(request: dict, key: bytes,
                          tenant: str | None = None) -> list[dict]:
    """Split an Anthropic request into wire-ordered segments.

    `tenant` scopes segment identity. Caches do not cross tenants, so two
    tenants sending the same text must not be given the same id: it asserts a
    shared cache entry that cannot exist, and it lets a reader of a multi-tenant
    trace join on the id to learn the two share content.

    Order matters and follows the wire: tools, then system, then messages. That
    is the order the cache matches in, so it is the order the index has to
    reflect. Sorting by anything else would make the analyser reason about a
    prompt nobody sent.

    Tools are emitted one segment each rather than as a single block. Tool
    reordering between requests is a real and common cause of cache misses, and
    it is invisible if the whole tool array is one hash.
    """
    out: list[dict] = []

    def add(role: str, label: str, block) -> None:
        text = _canonical(block)
        marked, ttl = _cache_marker(block)
        out.append({
            "index": len(out),
            "role": role,
            "label": label,
            # Identity covers the container, not just the text. The same words
            # in `system` and in a user turn are different bytes on the wire and
            # will not share a cache entry, but hashing content alone gave them
            # one id -- and since a cached prefix is a tuple of ids, moving text
            # between roles read downstream as a cache hit. That is the exact
            # authority confusion the relocation rules exist to prevent, arriving
            # through the back door.
            #
            # The block *kind* is included for the same reason: a text block and
            # a tool_result carrying the same string are not interchangeable.
            # The positional label is deliberately excluded, because identical
            # text at two positions genuinely is the same content, and position
            # is already carried by `index`.
            "id": segment_id(_identity_input(role, block, text, tenant), key),
            "bytes": len(text.encode("utf-8")),
            "cache_marked": marked,
            "ttl": ttl,
        })

    paths = []
    for role, label, block, path in walk(request):
        add(role, label, block)
        paths.append(path)

    # A message-level `cache_control` caches up to and including that message,
    # so it marks the boundary at the message's *last* content block. `walk`
    # never yields the message object -- by design, it yields the blocks whose
    # text is hashed and priced -- so without this the provider bills a request
    # as cached while every segment records `cache_marked: False`. Measured: a
    # request with one message-level marker gave `marker_count == 1`, zero marked
    # segments, and `_requested_ttl == None`, so the recorder and the body
    # adapter stored a cached request as uncached and the as-shipped bake-off arm
    # replayed it uncached -- which credits the allocator with a saving against a
    # baseline that was never really uncached.
    for mi, m in enumerate(request.get("messages") or []):
        if not isinstance(m, dict):
            continue
        marked, ttl = _cache_marker(m)
        if not marked:
            continue
        last = None
        for i, p in enumerate(paths):
            if p[0] == "messages" and p[1] == mi:
                last = i
        if last is None:
            # A marker on a message with no content blocks has no boundary to
            # attach to. `marker_count` still sees it, so the budget check is
            # unaffected; there is simply no segment for it to describe.
            continue
        out[last]["cache_marked"] = True
        # A block-level lifetime wins: it is the more specific statement about
        # this exact boundary, and overwriting it would discard the narrower fact.
        if out[last]["ttl"] is None:
            out[last]["ttl"] = ttl
    return out


def marker_paths(request: dict):
    """Yield a path for every place on the wire a `cache_control` can live.

    A superset of `walk`, and deliberately a different function. `walk` yields
    *segments* -- the content blocks whose text is hashed and priced -- and a
    message-level `cache_control` is a marker location that is not a segment:
    Anthropic accepts it on the message object as well as on a content block.

    Counting and stripping have to cover the wire, not the segmentation. Using
    `walk` for both meant a caller's message-level markers were invisible to
    every guard that mattered: `strip_markers` left them in place while its
    docstring said otherwise, `respect_existing` never saw them and so never
    stood down, and the budget check counted only the plugin's own. Measured: a
    request carrying four message-level markers against a budget of four came
    back from `on_request` with six, an applied decision, and no complaint.
    """
    for i, _tool in enumerate(request.get("tools") or []):
        yield ("tools", i)
    system = request.get("system")
    if isinstance(system, list):
        for i, _block in enumerate(system):
            yield ("system", i)
    for mi, m in enumerate(request.get("messages") or []):
        if not isinstance(m, dict):
            continue
        # The message object itself, then its blocks.
        yield ("messages", mi)
        content = m.get("content")
        if isinstance(content, list):
            for bi, _block in enumerate(content):
                yield ("messages", mi, bi)


def marker_count(request: dict) -> int:
    """How many cache breakpoints this body carries as the provider counts them.

    The number the surface's `max_breakpoints` budget is about.
    """
    n = 0
    for path in marker_paths(request):
        target = _at_path(request, path)
        if isinstance(target, dict) and "cache_control" in target:
            n += 1
    return n


def _at_path(body: dict, path: tuple):
    try:
        if path[0] == "tools":
            return body["tools"][path[1]]
        if path[0] == "system":
            return body["system"][path[1]]
        if len(path) == 2:
            return body["messages"][path[1]]
        return body["messages"][path[1]]["content"][path[2]]
    except (KeyError, IndexError, TypeError):
        return None


def strip_markers(request: dict) -> dict:
    """Return a copy with every `cache_control` removed.

    Override has to mean replace. Leaving the caller's markers in place and
    adding our own combines two plans into one nobody chose, and a request that
    already carried the surface's full budget then goes over it -- which is a
    provider error, not an override.

    Iterates `marker_paths`, not `walk`: this used to miss message-level markers
    entirely, so "every" was false in exactly the case the docstring describes.
    """
    out = copy.deepcopy(request)
    for path in marker_paths(request):
        _unmark(out, path)
    return out


def _unmark(body: dict, path: tuple) -> None:
    def clean(value):
        if isinstance(value, dict) and "cache_control" in value:
            return {k: v for k, v in value.items() if k != "cache_control"}
        return value

    if path[0] == "tools":
        body["tools"][path[1]] = clean(body["tools"][path[1]])
    elif path[0] == "system":
        if path[1] is not None:
            body["system"][path[1]] = clean(body["system"][path[1]])
    elif len(path) == 2:
        # The message object itself. `walk` never produces this path, which is
        # why message-level markers survived every strip.
        body["messages"][path[1]] = clean(body["messages"][path[1]])
    else:
        _, mi, bi = path
        if bi is not None:
            body["messages"][mi]["content"][bi] = clean(
                body["messages"][mi]["content"][bi])


def apply_markers(request: dict, placements: dict) -> dict:
    """Return a copy of the request carrying cache markers at these wire indices.

    `placements` maps a wire index from `walk`/`segments_from_request` to a
    lifetime. Returns a new body; the caller's dict is never mutated, because a
    request object that is still being assembled elsewhere is not a hypothetical
    -- the recorder shipped a bug of exactly that shape, where an agent loop
    appending to `messages` turned a one-message record into three segments.

    A bare string container has no room for `cache_control`, so it is rewritten
    into the block form the API documents as equivalent. That rewrite is
    disclosed in the returned body only by its shape, so callers that must not
    be rewritten should not pass string containers.
    """
    if not placements:
        return request
    out = copy.deepcopy(request)
    for i, (_, _, _, path) in enumerate(walk(request)):
        ttl = placements.get(i)
        if ttl is None:
            continue
        _mark(out, path, ttl)
    return out


def _mark(body: dict, path: tuple, ttl: str) -> None:
    control = {"type": "ephemeral"}
    if ttl and ttl != "5m":
        control["ttl"] = ttl

    def blockify(value):
        return ({"type": "text", "text": value, "cache_control": control}
                if isinstance(value, str)
                else {**value, "cache_control": control})

    if path[0] == "tools":
        body["tools"][path[1]] = blockify(body["tools"][path[1]])
    elif path[0] == "system":
        if path[1] is None:
            body["system"] = [blockify(body["system"])]
        else:
            body["system"][path[1]] = blockify(body["system"][path[1]])
    else:
        _, mi, bi = path
        # A named position is tool history, not content. Nothing may write a
        # `cache_control` there: it does not live under `content`, so the write
        # would either miss or invent a shape the provider never documented.
        if isinstance(bi, str):
            raise ValueError(
                f"refusing to place a cache marker on {bi!r} at messages[{mi}]: "
                f"tool history is visible to the segmenter so drift is caught, "
                f"and is deliberately not a marker location")
        msg = body["messages"][mi]
        if bi is None:
            msg["content"] = [blockify(msg["content"])]
        else:
            msg["content"][bi] = blockify(msg["content"][bi])


def _scale_to_measured(segments: list[dict], input_total: int | None) -> list[dict]:
    """Turn byte estimates into token counts that sum to what was billed.

    Without a measured total this falls back to the bytes-per-token constant and
    the row is still flagged estimated. With one, the constant cancels: each
    segment gets its share of a number the provider actually reported.
    """
    total_bytes = sum(s["bytes"] for s in segments) or 1
    if not input_total:
        for s in segments:
            s["tokens"] = max(1, round(s["bytes"] / _BYTES_PER_TOKEN))
            del s["bytes"]
        return segments

    # Largest remainder, so the parts sum to the measured whole exactly.
    #
    # Rounding each segment independently and clamping to a minimum of one made
    # the total drift from what the provider actually billed. That drift feeds
    # bake-off spend and, worse, the minimum-cacheable check: enough small
    # segments can inflate a request past a provider threshold and flip a
    # cacheability decision on tokens nobody was charged for. Zero-token
    # segments are allowed, because a request with more segments than billed
    # tokens is a real shape and inventing a token per segment is how the total
    # ran away in the first place.
    exact = [input_total * s["bytes"] / total_bytes for s in segments]
    floors = [int(x) for x in exact]
    remainder = input_total - sum(floors)
    order = sorted(range(len(segments)), key=lambda i: exact[i] - floors[i], reverse=True)
    for i in order[:max(0, remainder)]:
        floors[i] += 1
    for s, t in zip(segments, floors):
        s["tokens"] = t
        del s["bytes"]
    return segments


def _get(obj, name, default=None):
    """Read from an SDK object or a plain dict without caring which."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _positive_count(value) -> bool:
    """A token count that is actually evidence something was billed.

    Booleans are excluded because `isinstance(True, int)` is True in Python, and
    infinities and NaN because an exporter writing either has told you nothing.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return value > 0 and value != float("inf")


def _tok(value) -> int:
    return int(value) if _is_token_count(value) else 0

def usage_from_details(prompt_tokens, details, completion_tokens) -> dict:
    """Rebuild the four classes from LiteLLM's inclusive total plus its split."""
    details = details if isinstance(details, dict) else {}
    read = _tok(details.get("cached_tokens"))
    # Two names for one number, kept in sync by LiteLLM itself. Reading only one
    # would see zero on a payload written through the other path.
    written = _tok(details.get("cache_write_tokens")) or \
        _tok(details.get("cache_creation_tokens"))

    # ...and a third place the same number can be, which is the one that broke
    # the invariant this whole adapter rests on. A payload carrying
    # `cache_creation_token_details` without either mirrored aggregate left
    # `written` at zero, so the write tokens stayed inside the uncached
    # remainder *and* were emitted again as the split -- counted twice.
    # Measured: prompt_tokens 1,000 with a 700-token 5m write reconstructed as
    # 1,700 billed, a 70% overstatement, on a shape the fixtures never produced
    # because they always set both spellings at once.
    split_total = 0
    detail = details.get("cache_creation_token_details")
    if isinstance(detail, dict):
        split_total = sum(_tok(v) for v in detail.values())
    if not written:
        written = split_total

    # `text_tokens` is LiteLLM's own `prompt_tokens - read - written`, so it is
    # preferred over recomputing: if the payload is internally inconsistent the
    # subtraction can go negative, and a negative counter is refused at pricing.
    if _is_token_count(details.get("text_tokens")):
        uncached = _tok(details.get("text_tokens"))
    else:
        total = _tok(prompt_tokens)
        uncached = max(0, total - read - written)

    split = details.get("cache_creation_token_details")
    creation = {}
    if isinstance(split, dict):
        for k in ("ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens"):
            if split.get(k) is not None:
                creation[k] = _tok(split.get(k))
    return {
        "input_tokens": uncached,
        "output_tokens": _tok(completion_tokens),
        "cache_read_input_tokens": read,
        "cache_creation_input_tokens": written,
        "cache_creation": creation,
    }


def _positive_split(creation) -> bool:
    """Whether a `cache_creation` split records anything actually written."""
    return isinstance(creation, dict) and any(
        _positive_count(v) for v in creation.values())


def _has_positive_usage(got: dict) -> bool:
    """Whether a parsed usage mapping is evidence of billed input.

    One predicate, because the two places that ask were free to disagree and
    did: the primary path tested positive counters and the LiteLLM fallback
    tested key presence, so a non-empty dict of zeros counted as accounting.

    Writes are counted through `write_tokens` rather than by reading the
    aggregate, which is this package's single reader for "how many tokens were
    written" -- an export carrying only the per-lifetime split reports zero in
    the aggregate while still being billed for it. The structural audit in
    `test_twin_paths` caught the first draft of this function reading it
    directly.
    """
    if not isinstance(got, dict):
        return False
    return (any(_positive_count(got.get(k)) for k in
                ("input_tokens", "cache_read_input_tokens"))
            or write_tokens(got) > 0)


def usage_from_response(response) -> dict:
    """The accounting fields, and only those.

    `cache_creation` is carried through when present: Anthropic reports the
    per-lifetime write split there, which is the only thing that settles whether
    a write billed at 1.25x or 2x. Without it the lifetime has to come from the
    request, and a trace that carries neither cannot be priced at all.
    """
    u = _get(response, "usage") or {}
    # Absent usage is not zero usage. Returning a full set of zeroed counters
    # made a row with response metadata and no accounting fields look like an
    # analysable request that cost nothing -- so a degraded export read as
    # complete coverage, and the zero-spend rows diluted every ratio. No
    # accounting field present means no evidence, and the coverage line should
    # say so rather than the spend line absorbing it.
    # Evidence of input accounting, not merely the presence of a key. Two gaps
    # in the first version of this test, both measured:
    #
    #   `{"usage": {"cache_creation": {}}}` passed, because an empty dict is not
    #   None. `_find_response` returns the first candidate whose usage is
    #   truthy, so that nested husk beat a top-level `usage` carrying 11,000 real
    #   input tokens: the row became analysable at zero input, billed spend
    #   vanished, and coverage still read 100%.
    #
    #   `output_tokens` alone passed too, and this package prices input. A
    #   response object carrying only an output count is not accounting this can
    #   use, and treating it as such shadows accounting that is.
    #   `{"input_tokens": 0, "cache_read_input_tokens": 0,
    #     "cache_creation_input_tokens": 0}` passed as well, because the keys
    #   were present. A response that billed nothing is not a response, it is a
    #   placeholder, and it shadowed a top-level `usage` carrying 11,000 real
    #   tokens exactly as the empty dict had. Fixing "key missing" and leaving
    #   "key present but zero" was the same defect one level over.
    #
    # So the test is a *positive* counter, not a present one. A row that really
    # billed nothing prices to zero either way; the difference is that it is now
    # honestly uncovered rather than covered-at-zero, which is what the coverage
    # line is for.
    creation = _get(u, "cache_creation")
    has_input = any(_positive_count(_get(u, k)) for k in
                    ("input_tokens", "cache_read_input_tokens",
                     "cache_creation_input_tokens"))
    has_split = _positive_split(creation)
    if not (has_input or has_split):
        # One more shape before giving up: LiteLLM's normalised counters.
        #
        # `litellm_handler`'s success event hands `response_obj` straight to
        # `on_response`, which called this -- and a proxy response carries
        # `prompt_tokens` with `prompt_tokens_details`, none of the Anthropic
        # keys. So this returned {} and the named LiteLLM integration never
        # learned whether prefixes were rebuilding or whether its own markers
        # were being read. The adapter parsed that shape correctly on the batch
        # path; the live path had its own parser that could not.
        # Looked for on the response as well as inside `usage`. The fallback was
        # added for the LiteLLM shape and then searched `u`, which is `{}`
        # exactly when the counters are top level -- so the one shape it existed
        # to read was the one it could not. Measured through the live handler: a
        # nested response raised a rebuild alert over ten requests and the same
        # counters at top level raised none, silently.
        carrier = u if _get(u, "prompt_tokens_details") is not None else response
        details = _get(carrier, "prompt_tokens_details")
        if isinstance(details, dict):
            got = usage_from_details(_get(carrier, "prompt_tokens"), details,
                                     _get(carrier, "completion_tokens"))
            # The same positive test as above, not a presence test. This read
            # `or got.get("cache_creation")`, and a *non-empty dict of zeros* is
            # truthy -- so a payload whose only accounting was
            # `cache_creation_token_details: {5m: 0, 1h: 0}` returned a full set
            # of zeroed counters. `_find_response` ranks candidates by whether
            # they carry usage at all, so that husk then beat a top-level
            # `usage` holding 11,000 real input tokens: the row priced at zero
            # and its spend vanished while coverage still read complete. Exactly
            # the defect this function's own docstring spends a paragraph on,
            # reintroduced through the branch added after it.
            if _has_positive_usage(got):
                return got
        return {}
    creation = creation or {}
    if not isinstance(creation, dict):
        creation = {k: _get(creation, k, 0) for k in
                    ("ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens")}
    return {
        "input_tokens": _get(u, "input_tokens", 0) or 0,
        "output_tokens": _get(u, "output_tokens", 0) or 0,
        "cache_read_input_tokens": _get(u, "cache_read_input_tokens", 0) or 0,
        "cache_creation_input_tokens": _get(u, "cache_creation_input_tokens", 0) or 0,
        "cache_creation": dict(creation),
    }


# `_billed_input` now comes from `trace` (imported at the top of this module and
# so still importable from here, which `recorder` and the body adapter rely on).
# This module used to define it, and the loader and the bake-off each inlined
# their own -- which is how the two that guard money ended up carrying the same
# fail-open, written twice.


def _requested_ttl(request: dict) -> str | None:
    """The lifetime asked for, when every marker in the request agrees.

    Mixed lifetimes return None rather than the first one. Against a single
    aggregate write total the split is unknowable, and picking whichever sorted
    first is a guess wearing a number.
    """
    # Driven by `marker_paths`, which is the one definition of where a marker can
    # live on the wire. This used to walk tools, system blocks and message
    # *content* blocks by hand and so never saw a message-level marker: a request
    # whose only marker sat on the message object reported no requested lifetime
    # at all, and an unprovable lifetime is excluded from every dollar figure.
    ttls = []
    for path in marker_paths(request):
        marked, ttl = _cache_marker(_at_path(request, path))
        if marked:
            ttls.append(ttl or "5m")
    distinct = set(ttls)
    return ttls[0] if len(distinct) == 1 else None
