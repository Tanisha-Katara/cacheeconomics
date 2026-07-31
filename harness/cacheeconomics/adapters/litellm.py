"""Read LiteLLM proxy logs as a trace.

LiteLLM sits in front of a large share of the traffic this analysis is about,
and its `StandardLoggingPayload` is already being written to disk, S3 or a
database by people who have never heard of this tool. That makes it the one
ingest path where a client has to do nothing at all to get an answer.

Built against the real schema rather than a guess. Provenance is pinned in
`landscape/SOURCES.md`; the two things that matter and would be easy to get
backwards:

**`prompt_tokens` is inclusive.** LiteLLM's Anthropic transform starts from
Anthropic's `input_tokens` and then adds the cache creation and cache read
counts to it (`anthropic/chat/transformation.py`, `prompt_tokens +=` twice). So
`prompt_tokens` is the *billed input total*, not the uncached portion. Mapping it
onto `input_tokens` would double-count every cached token -- on a workload with
90% cache reads that is a 10x overstatement of uncached spend, in the direction
that makes caching look useless.

**The per-lifetime split survives.** `prompt_tokens_details` carries
`cached_tokens` (reads), `cache_write_tokens`/`cache_creation_tokens` (writes,
mirrored names for the same number), and `cache_creation_token_details` with
`ephemeral_5m_input_tokens` and `ephemeral_1h_input_tokens`. That last one is
what makes a write's price provable: 1.25x against 2x cannot be inferred from an
aggregate, and without it every write in the trace is excluded from the dollar
figures. It is genuinely there, so this adapter reads it.

What this does *not* do is reconstruct prompt structure. `messages` in the
payload is LiteLLM's OpenAI-shaped normalisation, not the Anthropic body that
went on the wire, and `walk()` reasons about the latter. Translating between
them is a guess with a cost model attached, so the tier here is USAGE_ONLY and
the structural rules stay quiet. Everything the usage counters can answer --
efficiency, TTL band, rebuilds, model splits, spend, invoice reconciliation --
works.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from .. import registry
from ..segment import usage_from_details as _from_details
from ..trace import (Request, Tier, TraceSet, _is_token_count, _text,
                     session_of, write_tokens)

# `custom_llm_provider` -> a surface the registry carries. Only mappings the
# registry can actually price are listed; anything else keeps the provider name
# and fails closed at pricing time, which is the honest outcome rather than
# assuming Anthropic multipliers for a surface nobody checked.
PROVIDER_TO_TARGET = {
    "anthropic": "anthropic/direct",
    "bedrock": "amazon-bedrock/converse",
    "bedrock_converse": "amazon-bedrock/converse",
    "vertex_ai": "google-cloud/vertex",
    "openai": "openai/direct",
    "deepseek": "deepseek/direct",
}


def target_from_row(row: dict, default: str | None = None) -> str:
    """Which provider surface a LiteLLM request is bound for.

    One reader, because this is consumed by two paths that had already
    diverged: the batch adapter mapped `custom_llm_provider` through the table
    above, while the live proxy hook passed nothing at all and let
    `CachePlugin.on_request` default to `anthropic/direct`. On a proxy fronting
    Bedrock that meant minimums, TTL support and the breakpoint budget were all
    computed against the wrong provider -- on the one path that rewrites a real
    request, where being wrong produces a provider error rather than a bad
    report.

    LiteLLM names the provider two ways and both are read here: the explicit
    `custom_llm_provider` field, and the routing prefix on the model itself
    (`bedrock/anthropic.claude-...`). An unrecognised provider is returned
    unmapped rather than folded into the default, so the registry refuses it
    instead of quietly pricing it as Anthropic.
    """
    provider = (_text(row.get("custom_llm_provider")) or "").lower()
    if not provider:
        model = _text(row.get("model")) or ""
        if "/" in model:
            head = model.split("/", 1)[0].lower()
            if head in PROVIDER_TO_TARGET:
                provider = head
    if not provider:
        return default or "anthropic/direct"
    return PROVIDER_TO_TARGET.get(provider, provider)


def _epoch(value):
    """`startTime`/`endTime` are epoch floats. Seconds or milliseconds."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    if value != value or value in (float("inf"), float("-inf")):
        return None
    if value <= 0:
        return None
    # A timestamp past ~2286 in seconds is almost certainly milliseconds. Some
    # exporters emit one, some the other, and a 1000x error here reads as a
    # workload spanning millennia -- which silently disables every cadence rule
    # rather than failing.
    if value > 1e11:
        value /= 1000.0
    try:
        return datetime.fromtimestamp(value, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


def _int(value) -> int:
    return int(value) if _is_token_count(value) else 0


def usage_from_payload(row: dict) -> dict:
    """The four token classes, from whichever shape this payload carries.

    Ordered by authority. A raw provider usage object states the counts
    outright; LiteLLM's normalised fields are a faithful but derived view of the
    same thing, and deriving the uncached portion by subtraction is the last
    resort because it is the only step that can go negative if the payload is
    inconsistent.
    """
    # Candidates in order of authority, and each is *tried* rather than
    # accepted: a candidate only wins if it carries real input accounting.
    #
    # Selecting on key presence was the defect. A `metadata.usage_object` of all
    # zeros beat a top-level `prompt_tokens_details` holding 300 fresh and
    # 200,000 cached tokens, so the row priced at zero and its entire cache
    # activity vanished. And a payload with a model but no counters at all
    # produced a full set of zeroed fields, which `has_usage` accepts as
    # accounting -- so a row whose cost is simply unknown was analysed as a $0
    # request and diluted every ratio it touched.
    #
    # Same rule as `segment.usage_from_response`, which learned it two rounds
    # ago: presence is not evidence, a positive counter is.
    meta = row.get("metadata")
    meta = meta if isinstance(meta, dict) else {}
    resp = row.get("response") if isinstance(row.get("response"), dict) else {}
    resp_usage = resp.get("usage") if isinstance(resp.get("usage"), dict) else None

    for candidate in (meta.get("usage_object"), resp_usage):
        if not isinstance(candidate, dict):
            continue
        got = {
            "input_tokens": _int(candidate.get("input_tokens")),
            "output_tokens": _int(candidate.get("output_tokens")),
            "cache_read_input_tokens": _int(candidate.get("cache_read_input_tokens")),
            "cache_creation_input_tokens":
                _int(candidate.get("cache_creation_input_tokens")),
            "cache_creation": {k: _int(v) for k, v in
                               (candidate.get("cache_creation") or {}).items()},
        }
        if _has_input(got):
            return got
        # A response object can also carry LiteLLM's normalised shape.
        details = candidate.get("prompt_tokens_details")
        if isinstance(details, dict):
            got = _from_details(candidate.get("prompt_tokens"), details,
                                candidate.get("completion_tokens"))
            if _has_input(got):
                return got

    # LiteLLM's own normalised counters, which is the common case.
    #
    # The split is *required*, not optional. `prompt_tokens` is inclusive -- the
    # adapter's own docstring opens with that -- so without
    # `prompt_tokens_details` there is no evidence about which class those
    # tokens fell into, and reconstructing them as fully uncached is a claim
    # rather than a reading. On a well-cached Anthropic workload that claim
    # prices 0.1x reads at 1x and reports the cache as absent: the client is
    # told to start caching something they already cache, and the dollar figure
    # is ~10x. Absence of the split is not proof of no caching.
    details = row.get("prompt_tokens_details")
    if isinstance(details, dict) and details:
        got = _from_details(row.get("prompt_tokens"), details,
                            row.get("completion_tokens"))
        if _has_input(got):
            return got

    # Nothing anywhere carried input accounting. `{}` rather than zeros, so the
    # row is honestly uncovered instead of counted as a request that cost
    # nothing.
    return {}


def _has_input(usage: dict) -> bool:
    """Does this carry a positive input-side count, anywhere.

    The write side goes through `write_tokens`, which already knows that a
    provider may report the aggregate, the per-lifetime split, or both. Checking
    the aggregate here would have been a tenth site deciding that question for
    itself, which is what the audit in test_twin_paths exists to stop -- and it
    caught this on the first run after it was written.
    """
    if usage.get("input_tokens", 0) > 0 or usage.get("cache_read_input_tokens", 0) > 0:
        return True
    return write_tokens(usage) > 0




def _agent_of(row: dict) -> str:
    """LiteLLM has no standard field for which context a call belongs to.

    These are the places a caller would put one. Returning "unknown" pools every
    context together, which is what happens today and is at least honest about
    it -- inventing a split would make the per-agent bake-off report differences
    between groups nobody defined.
    """
    meta = row.get("metadata")
    meta = meta if isinstance(meta, dict) else {}
    for holder in (meta.get("requester_metadata"), meta.get("spend_logs_metadata")):
        if isinstance(holder, dict):
            for k in ("agent", "agent_name", "subagent", "context_id"):
                name = _text(holder.get(k))
                if name:
                    return name
    tags = row.get("request_tags")
    if isinstance(tags, list):
        for tag in tags:
            t = _text(tag)
            if t and t.startswith("agent:"):
                return t.split(":", 1)[1] or "unknown"
    return "unknown"


def _tenant_of(row: dict) -> str | None:
    """Cache isolation is per key, not per human.

    Ordered narrowest first. Two teams under one org do not share a cache entry,
    so an org id would merge traffic that can never pool -- which reads as a
    prefix being rebuilt when it is simply a different tenant's.
    """
    # `.get` on a string is an AttributeError, and this is the one ingest path
    # whose input is another system's schema. Measured: `metadata` written as a
    # string took the entire export down instead of costing one row.
    meta = row.get("metadata")
    meta = meta if isinstance(meta, dict) else {}
    for k in ("user_api_key_hash", "user_api_key_alias", "user_api_key_team_id",
              "team_id", "user_api_key_user_id"):
        v = _text(meta.get(k))
        if v:
            return v
    return _text(row.get("end_user"))


def request_from_payload(row: dict, *, default_tenant: str | None = None,
                         default_target: str | None = None,
                         index: int | None = None) -> Request | None:
    """One `StandardLoggingPayload` as a Request, or None if it is not one."""
    if not isinstance(row, dict):
        return None
    model = _text(row.get("model"))
    if not model:
        return None

    # A row that names its provider is authoritative. A row that does not used
    # to fall straight to `anthropic/direct`, which on a proxy fronting Bedrock
    # meant partner traffic was priced at Anthropic first-party rates with
    # nothing in the report to say so. The operator's `--target-id` now answers
    # for those rows, and the loader counts them so the substitution is visible
    # rather than assumed.
    target = target_from_row(row, default_target)

    # The registry is keyed on bare model ids, and a proxy is exactly where
    # non-bare ones come from: LiteLLM routes on `anthropic/claude-opus-5`, and
    # date-stamped ids like `claude-opus-5-20250101` arrive from pinned
    # deployments. Measured: both are refused by `cost.price`, so every request
    # on them drops out of spend, out of the ratios and out of reconciliation --
    # on the ingest path whose entire pitch is that a client changes nothing.
    #
    # `normalize_model` is deliberately not folded into the registry lookup, so
    # callers do it on purpose and say that they did. It handles the provider
    # prefix and the date suffix; a Bedrock-shaped `us.anthropic.` id is left
    # alone and still fails closed, which is the honest outcome rather than a
    # second guess at what it maps to.
    normalised, snapshot = registry.normalize_model(model, target)

    status_text = (_text(row.get("status")) or "success").lower()
    # `status` is "success" or "failure". A failed call still consumed input on
    # some surfaces, but it did not populate a usable cache entry, and counting
    # it as a normal request would understate the hit rate of the ones that ran.
    status = 200 if status_text == "success" else 500

    return Request(
        # Distinct per row when the payload names neither id. Everything
        # downstream that counts *requests* dedupes on this, so a shared ""
        # collapsed them: three id-less rows omitted from the bake-off were
        # reported as "1 of 3 requests contributed nothing", understating the
        # ingest damage on precisely the malformed export where the count is
        # the thing a reader needs.
        request_id=(_text(row.get("id")) or _text(row.get("litellm_call_id"))
                    or (f"row-{index}" if index is not None else "")),
        sent_at=_epoch(row.get("startTime")),
        first_token_at=_epoch(row.get("completionStartTime")),
        model=normalised,
        target_id=target,
        tenant=_tenant_of(row) or _text(default_tenant),
        # `trace_id` is documented as spanning the calls of one overall request,
        # which is the closest thing LiteLLM has to a conversation. The plugin's
        # live path keys on the same field, so a trace captured either way groups
        # the same requests.
        session=session_of(row),
        agent=_agent_of(row),
        usage=usage_from_payload(row),
        segments=[],
        status=status,
    )


def load_litellm(path: str, *, default_tenant: str | None = None,
                 default_target: str | None = None) -> TraceSet:
    """Load a JSONL file of LiteLLM StandardLoggingPayload objects.

    USAGE_ONLY by construction: see the module docstring. No key is needed
    because nothing here hashes prompt content -- the payload's `messages` are
    not read at all, which is also the answer to "what does this send anywhere",
    since the answer is nothing.

    `default_target` answers for rows that carry no `custom_llm_provider`. It is
    a default and not an override: a row that names its provider keeps it.
    """
    requests, notes = [], []
    unparseable = skipped = malformed = providerless = 0
    no_split = 0
    malformed_examples: list = []
    with open(path) as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                unparseable += 1
                continue
            # Some exporters wrap the payload; some write it bare.
            if isinstance(row, dict) and isinstance(row.get("standard_logging_object"),
                                                    dict):
                row = row["standard_logging_object"]
            if isinstance(row, dict) and not _text(row.get("custom_llm_provider")):
                providerless += 1
            # Every nested lookup shape-checked. This counter runs *before* the
            # try/except that turns a malformed foreign payload into a counted
            # skip, so `metadata.get(...)` on a row whose metadata is a string
            # raised AttributeError and took the whole ingest down -- one bad
            # line in somebody's proxy log killing the analysis instead of
            # showing up as a skipped row. Introduced by the note this counter
            # feeds, which is the hazard of adding work ahead of a guard.
            if isinstance(row, dict) and row.get("prompt_tokens"):
                meta = row.get("metadata")
                resp = row.get("response")
                if not (row.get("prompt_tokens_details")
                        or (isinstance(meta, dict) and meta.get("usage_object"))
                        or (isinstance(resp, dict) and resp.get("usage"))):
                    no_split += 1
            try:
                r = request_from_payload(row, default_tenant=default_tenant,
                                         default_target=default_target,
                                         index=line_no)
            except (AttributeError, TypeError, ValueError, KeyError) as e:
                # Narrow on purpose: these are the shapes a foreign schema
                # produces when a field is the wrong type. The loader's contract
                # is that a malformed row costs one row, and this path is the
                # most likely of all of them to meet schema drift -- it reads
                # somebody else's log format, versioned on their release
                # schedule, not ours.
                malformed += 1
                if len(malformed_examples) < 3:
                    malformed_examples.append(f"{type(e).__name__}: {e}")
                continue
            if r is None:
                skipped += 1
                continue
            requests.append(r)

    if no_split:
        notes.append(
            f"{no_split:,} row(s) carry `prompt_tokens` with no "
            f"`prompt_tokens_details` and no raw provider usage object. That "
            f"total is inclusive of cache reads and writes, so without the split "
            f"there is no evidence which class those tokens fell into and they "
            f"are left uncovered rather than reconstructed as uncached -- "
            f"assuming uncached prices a 0.1x read at 1x and reports a working "
            f"cache as absent. Upgrade the LiteLLM version writing these logs, or "
            f"export the provider usage object alongside them.")
    if providerless:
        chosen = default_target or "anthropic/direct"
        notes.append(
            f"{providerless:,} row(s) carry no `custom_llm_provider` and were "
            f"attributed to {chosen}"
            + ("" if default_target else
               " (the default -- pass --target-id if this proxy fronts Bedrock or "
               "Vertex, whose rates are not Anthropic's)")
            + ". Caches and prices are per surface, so a wrong attribution here "
              "moves real money.")
    if unparseable:
        notes.append(f"{unparseable:,} line(s) were not valid JSON and were skipped.")
    if malformed:
        notes.append(
            f"{malformed:,} row(s) could not be converted because a field held "
            f"an unexpected type ({'; '.join(malformed_examples)}). They are "
            f"counted as unread rather than dropped, so the reconciliation gate "
            f"knows the export is incomplete.")
    if skipped:
        notes.append(
            f"{skipped:,} row(s) carried no model name and are not LiteLLM "
            f"logging payloads. They are excluded rather than guessed at.")
    if requests and not any(r.session for r in requests):
        notes.append(
            "No request carries a conversation id, so in-session rebuild "
            "detection cannot run (see REB-0). LiteLLM's `trace_id` is "
            "deliberately not used for this: its schema defines it as spanning "
            "the retries of one call, so every turn carries a different one, and "
            "treating it as a session yields single-request groups that silence "
            "the finding rather than answer it. Set `metadata.session_id` or "
            "`metadata.conversation_id` on the calls of one conversation.")
    notes.append(
        "Ingested from LiteLLM proxy logs. Prompt structure is not reconstructed: "
        "the payload's `messages` are LiteLLM's OpenAI-shaped normalisation "
        "rather than the body that went on the wire, so structural findings are "
        "not reported. Usage-driven findings, spend and invoice reconciliation "
        "are unaffected.")
    return TraceSet(requests=requests, tier=Tier.USAGE_ONLY, source=path,
                    notes=notes,
                    skipped_rows=unparseable + skipped + malformed)
