"""Enrich a bodies export with exact per-segment token counts.

Reads a JSONL export of logged request bodies, asks the provider's own
count-tokens endpoint what each prefix of each body costs, and writes the same
rows back with a `segment_tokens` array added. `cacheeconomics analyze --from
bodies` picks that array up and uses it instead of estimating.

Why this is a separate script and not a flag on the analyzer: the installed
package imports no network library at all, and a test asserts it
(`test_cli.TestNothingHereReachesTheNetwork`). Zero egress is the claim a client
is asked to trust when they hand over a trace, and they can check it by grepping
the wheel. A flag would make that claim conditional on reading the flag's
implementation. So the socket lives here, in a script you run deliberately, and
the analysis stays something you can prove is offline.

What it fixes. Without counts, `segment._scale_to_measured` divides the billed
input total between segments in proportion to their *bytes*. Measured against
the provider's tokenizer, that split has a median absolute error of 19.2% per
segment and a worst case of 181%, because dense JSON tool schemas run about 2.74
bytes per token where English prose runs 5.22. Every structural finding is
costed from that split. Counting removes essentially all of that error, since
the sizes then come from the same tokenizer that bills them -- but there is no
committed artifact measuring the residual the way inferred-token-split.json
measures the 19.2%, so this does not quote a number for it.

What it costs. One call per distinct prefix cut *per model*, cached, and the
endpoint is free. Prompt caching only pays when the prefix is stable, so the
prefix is shared across requests and counted once: measured on the demo trace,
286 structured requests produced 345 distinct cuts, or 1.2 calls per request.
That trace is single-model; an export that mixes models costs a separate set of
cuts for each, because each row is counted by the tokenizer it names and the two
do not agree. `--dry-run` reports the real figure for the export in hand and
names the tokenizers it would ask. The cache is written alongside the output so
a re-run costs nothing.

**This sends prompt content to the provider.** For a workload already running on
that provider it is the same content over the same wire to the same company. For
a workload on Bedrock or Vertex it is a new egress path to a different vendor,
and that is a decision to make deliberately rather than discover.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python3 tier-b/count_tokens.py bodies.jsonl -o bodies-counted.jsonl
    cacheeconomics analyze bodies-counted.jsonl --from bodies --invoice-usd 4820.16
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import typing
import urllib.error
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "harness"))

from cacheeconomics.adapters.bodies import _find_body            # noqa: E402
from cacheeconomics.tokenizer import (COUNTS_PROVENANCE_KEY,      # noqa: E402
                                      COUNTS_PROVENANCE_VERSION,
                                      count_segments, counts_provenance,
                                      row_sha256)
from cacheeconomics.registry import normalize_model, providers   # noqa: E402
from cacheeconomics.trace import (_first, _text,                 # noqa: E402
                                  request_from_row)

# Overridable, because the clients most likely to care about egress are the
# ones who cannot reach this host. An enterprise gateway, a Bedrock or Vertex
# deployment, or a self-hosted proxy all mean the counting call has to go
# somewhere else, and a hard-coded host makes the answer "edit the source".
DEFAULT_ENDPOINT = "https://api.anthropic.com/v1/messages/count_tokens"

# Both owned by `cacheeconomics.tokenizer`, which defines the vouching contract
# the loader checks. Constants here as well as there was three copies of a pair
# whose only job is that the two sides agree.
COUNTER_VERSION = COUNTS_PROVENANCE_VERSION
PROVENANCE_KEY = COUNTS_PROVENANCE_KEY


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


def counted_path(path: str) -> str:
    """`run.jsonl` -> `run-counted.jsonl`, beside it.

    The one implementation. `run_diagnostic.py` and `sweep_report.py` both
    derive this name and both used to do it themselves, so when the first was
    fixed the second kept the bug: `path.replace(".jsonl", "-counted.jsonl")`
    rewrites a *directory* component that happens to contain `.jsonl`, replaces
    every occurrence in a name that contains it twice, and returns the input
    unchanged when there is no extension at all -- which makes `count_tokens.py`
    open the capture it is reading for writing.

    Split on the basename so a directory named `something.jsonl` is left alone,
    and give an extensionless input a real extension rather than a name
    identical to itself.
    """
    head, base = os.path.split(path)
    root, ext = os.path.splitext(base)
    return os.path.join(head, f"{root}-counted{ext or '.jsonl'}")


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


def counter_id(models: RowModels, endpoint: str,
               tokenizer_id: str | None) -> str:
    """The cache scope: everything that decides what a count comes back as.

    The version and the tokenizer identity are in here because provenance is
    only as good as the cache it is stamped over. With a key of
    `model\\0endpoint` alone, a rerun after a gateway changed behaviour made
    zero calls, reused every prefix count, and wrote rows stamped with *current*
    provenance -- which then passed the freshness check downstream. A cache is a
    claim that nothing changed, and the claim has to be in the key.
    """
    return "\x00".join([f"v{COUNTER_VERSION}", models.tokenizer or "",
                        endpoint, tokenizer_id or ""])


def provenance(row, body, models: RowModels, endpoint: str,
               target_id: str | None, tokenizer_id: str | None) -> dict:
    """What produced this row's `segment_tokens`.

    Counted rows used to carry the array and nothing else, and the loader
    accepts any correctly-shaped positive array as exact. So a counted export
    left over from a different endpoint, a different resolved model, an older
    version of this script, or a capture that has since changed was
    indistinguishable from a fresh one -- and `sweep_report.counted` reused it
    on the strength of the filename existing.

    Two layers. `counts_provenance` supplies what the *loader* checks -- the
    counter version, a digest of the body and a digest of the prefix cuts the
    counts are differences of -- which answers "do these counts describe this
    body". Everything added here answers the wider question "would a re-run
    agree": both models, because the tokenizer that answered and the model the
    report names are two different things; the row digest, because the body
    digest is blind to usage, timestamps, status and a top-level `model`; the
    endpoint and surface, because they change what comes back; and
    `tokenizer_id`, the operator's assertion about which deployment answered,
    which is the only thing here that cannot be derived -- see `counter_id`.
    """
    # The package's record, plus what only this script knows. The version and the
    # two digests the loader checks are not restated here: `counts_provenance` is
    # the one place they are produced, so a field added to the contract reaches
    # every counted row without this function being edited.
    return {**counts_provenance(body),
            "tool": "tier-b/count_tokens.py",
            "row_sha256": row_sha256(row),
            "tokenizer_model": models.tokenizer,
            "analysis_model": models.analysis,
            "endpoint": endpoint, "target_id": target_id,
            "tokenizer_id": tokenizer_id}


def counter(model: str, key: str, stats: dict, endpoint: str = DEFAULT_ENDPOINT,
            dry_run: bool = False):
    def count(body: dict) -> int:
        if dry_run:
            # Counts the calls and sends nothing, so an operator can see the
            # exact egress volume before agreeing to any of it.
            stats["calls"] += 1
            return 0
        payload = dict(body)
        payload["model"] = model
        req = urllib.request.Request(
            endpoint, data=json.dumps(payload, default=str).encode(),
            headers={"content-type": "application/json", "x-api-key": key,
                     "anthropic-version": "2023-06-01"})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    stats["calls"] += 1
                    return int(json.load(r)["input_tokens"])
            except urllib.error.HTTPError as e:
                # Backing off rather than failing the run: this is a long loop
                # over somebody's whole trace, and losing it at row 9,000 to a
                # rate limit would mean starting again.
                if e.code in (429, 529) and attempt < 4:
                    time.sleep(2 ** attempt)
                    continue
                raise
        raise RuntimeError("unreachable")
    return count


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("path", help="JSONL export of logged request bodies")
    p.add_argument("-o", "--out", required=True, help="where to write the enriched export")
    # Deliberately no flag naming the tokenizer -- neither an override nor a
    # fallback. Both existed here and both were removed on review.
    #
    # An override produces counts that are wrong *and* load as exact, because
    # nothing downstream marks which tokenizer answered for which row. A
    # fallback is the same defect one step quieter: the analyzer calls a row
    # with no resolvable model "unknown", so counting it with haiku attaches
    # haiku's segment sizes to a row the report names otherwise.
    #
    # So the model is whatever `request_from_row` says it is, and a row it
    # cannot name is not counted. It is written without `segment_tokens`, the
    # analyzer estimates it by byte share, and it is not treated as exact. A
    # stated estimate beats a confident wrong number, which is the premise the
    # whole tool is sold on.
    p.add_argument("--target-id",
                   help="the provider surface this traffic went to, if the rows "
                        "do not say. Two effects: it resolves the analysed model "
                        "the way `analyze --target-id` does, so pass the same "
                        "value to both; and it names the surface's routing "
                        "prefix, which is stripped before a tokenizer is asked "
                        "(`anthropic.claude-opus-5` is how Bedrock addresses a "
                        "model, not an id any tokenizer answers to). Without it, "
                        "rows whose model carries such a prefix are NOT counted "
                        "and nothing is sent for them -- a call that cannot come "
                        "back with a usable count is prompt content leaving the "
                        "machine for nothing")
    p.add_argument("--tokenizer-id",
                   help="an identifier for the tokenizer deployment behind "
                        "--endpoint, asserted by you (a gateway build, a date, "
                        "anything that changes when the tokenizer might have). "
                        "Without it the resume cache is neither read nor "
                        "written: a cache is a claim that nothing changed since "
                        "last time, and nothing observable can back that claim. "
                        "Prefixes are still counted once per run either way")
    p.add_argument("--cache", help="cache file to read and write (default: <out>.cache.json)")
    p.add_argument("--allow-partial", action="store_true",
                   help="write the output and exit 0 even when some rows could "
                        "not be counted. Those rows carry no segment_tokens and "
                        "the analyzer estimates them, so the file is a mix of "
                        "counted and estimated sizes")
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT,
                   help="where to send the counting calls. Point it at your own "
                        "gateway to keep the egress inside your perimeter "
                        f"(default: {DEFAULT_ENDPOINT})")
    p.add_argument("--dry-run", action="store_true",
                   help="report how many calls this would make and what host they "
                        "would go to, and send nothing")
    args = p.parse_args()

    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key and not args.dry_run:
        print("ANTHROPIC_API_KEY is not set. This script is the one part of the "
              "toolchain that talks to the provider; the analyzer itself does not.",
              file=sys.stderr)
        return 2

    cache_path = args.cache or (args.out + ".cache.json")
    cache = {}
    # Resuming across runs is opt-in, because it is the one reuse path nothing
    # can check. The keys are scoped to the counter version, the tokenizer model
    # and the endpoint, and every one of those can be identical while the
    # deployment behind the endpoint has been replaced -- at which point a rerun
    # makes zero calls, reuses every prefix count, and stamps the rows with
    # current provenance that then passes the freshness check downstream. So the
    # cache is only read and written when the operator asserts an identity for
    # what is answering. Within a run prefixes are still counted once, which is
    # where the 1.2-calls-per-request figure comes from; this costs re-runs, and
    # a re-run is exactly when the tokenizer may have moved.
    resume = bool(args.tokenizer_id) and not args.dry_run
    if resume and os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)
        # Naming the counter, because after key scoping a model or endpoint
        # change resumes from 0 against a full file, which otherwise reads as a
        # missing cache rather than a deliberate recount. The model half of that
        # name is per row now, so it is reported at the end, once the rows have
        # said which models they are; only the endpoint is knowable up front.
        print(f"  resumed from {len(cache):,} cached counts "
              f"(via {args.endpoint}, tokenizer {args.tokenizer_id})",
              file=sys.stderr)
    elif not args.tokenizer_id and os.path.exists(cache_path) and not args.dry_run:
        print(f"  not resuming from {cache_path}: no --tokenizer-id, so nothing "
              f"shows the tokenizer that wrote it is the one answering now. "
              f"Prefixes are counted once within this run.", file=sys.stderr)
    elif args.dry_run and os.path.exists(cache_path):
        # A dry run starts from an empty cache on purpose. The question it
        # answers is "what would you send", and the honest answer is what a
        # fresh machine would send. Consulting a warm cache reported "0 calls
        # would go to api.anthropic.com" seconds after a real run had populated
        # it -- true here, false everywhere the client would run it, and exactly
        # the wrong thing to say while asking for permission to send anything.
        print(f"  ({len(json.load(open(cache_path))):,} counts are cached locally; "
              f"a dry run ignores them and reports a cold run)", file=sys.stderr)

    stats = {"calls": 0}
    # One counter per model in the export, built on first sight of that model.
    #
    # `--model` used to name the tokenizer for the whole run: it was stamped
    # over whatever model each row's own body carried, and it was the model half
    # of the single `counter_id` every row was cached under. So an export
    # holding both an opus planner and a haiku worker -- the ordinary shape of a
    # month of agent traffic -- was counted end to end by one tokenizer, and the
    # counts loaded as *exact*, releasing structural money on per-segment sizes
    # the other model was never asked for. Measured on a two-row synthetic
    # mixed export: 5/5 outgoing payloads carried `claude-haiku-4-5`, including
    # the row whose body said `claude-opus-5`.
    #
    # The model comes from the row. Which means the counter and its id do too:
    # folding the model into the cache key is what stops a cache written for one
    # model or endpoint being read back for another, and that property is only
    # worth anything if the model in the key is the one that actually answered.
    counters = {}

    def counter_for(models: RowModels):
        if models.tokenizer not in counters:
            counters[models.tokenizer] = (
                counter_id(models, args.endpoint, args.tokenizer_id),
                counter(models.tokenizer, key, stats, args.endpoint,
                        args.dry_run))
        return counters[models.tokenizer]

    if args.dry_run:
        print(f"  DRY RUN: nothing will be sent. Host that would receive the "
              f"prompt content: {args.endpoint}", file=sys.stderr)

    # Streamed, not buffered. This held every enriched row in a list and then
    # built a second full copy with `"\n".join(rows)`, so peak memory was twice
    # the export -- on the one tool in this repo whose input is a client's
    # entire month of traffic. Rows go to a temp file as they are produced and
    # it is renamed once `partial` is known, because the final name depends on
    # whether anything failed and that is not known until the end.
    counted, skipped, failed = 0, 0, 0
    tmp_path = args.out + ".partial-write"
    sink = None if args.dry_run else open(tmp_path, "w")
    # Closed and cleaned up on every exit, not just the normal one. Without
    # this, any exception between the open and the rename -- a write failure, a
    # cache checkpoint failure, an operator hitting ctrl-c -- left an enriched
    # export on disk containing client prompt bodies, under a name `.gitignore`
    # did not cover. That is the same defect as the plaintext count cache this
    # same change was written to remove, one file over.
    renamed = False
    try:

        def emit(text):
            if sink is not None:
                sink.write(text + "\n")

        with open(args.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    emit(line)                 # passed through untouched
                    skipped += 1
                    continue
                body = _find_body(row) if isinstance(row, dict) else None
                if not body:
                    emit(json.dumps(row))
                    skipped += 1
                    continue
                # The row exactly as the capture holds it, digested before this
                # script adds anything to it. Taken here rather than after
                # enrichment so the digest is of the capture, not of our output.
                src_digest_row = json.loads(line)
                try:
                    # Inside the try. Resolving the model runs the loader over
                    # the row, and a row malformed enough to break that should
                    # cost its own counts and nothing else -- outside, it took
                    # the whole run down at whichever row it met.
                    models = row_models(row, body, args.target_id)
                    if not models.tokenizer:
                        # Nothing to ask. Raised rather than sent: the previous
                        # version sent the literal string "unknown" and let the
                        # endpoint refuse it, which is a request whose only
                        # possible answer is an error.
                        raise ValueError(
                            f"its model id {models.prefix!r} names a provider "
                            f"surface rather than a model, and no --target-id "
                            f"says which; pass one so the prefix can be "
                            f"stripped before a tokenizer is asked"
                            if models.prefix else
                            "no model id on this row or its body, so there is "
                            "no tokenizer to ask; the analyzer will estimate it")
                    cid, count = counter_for(models)
                    row["segment_tokens"] = count_segments(body, count, cache,
                                                      cid)
                    # Written together with the counts, never separately: a row
                    # carrying an array and no record of what produced it is
                    # exactly what a stale export looks like.
                    row[PROVENANCE_KEY] = provenance(
                        src_digest_row, body, models, args.endpoint,
                        args.target_id, args.tokenizer_id)
                    counted += 1
                except Exception as e:                                # noqa: BLE001
                    # The row survives without counts and the analyzer falls back to
                    # estimating it, which is worse but is not nothing. Losing the
                    # row entirely would be.
                    row.pop("segment_tokens", None)
                    row.pop(PROVENANCE_KEY, None)
                    print(f"  row {counted + failed + skipped}: {type(e).__name__}: {e}",
                          file=sys.stderr)
                    failed += 1
                emit(json.dumps(row))
                if counted % 25 == 0 and counted and resume:
                    # Only when the cache can be resumed from. A checkpoint whose
                    # file will never be read back is a copy of the client's
                    # prefix shape on disk for nothing.
                    #
                    # Never during a dry run. The dry-run counter returns 0 for
                    # every prefix, and this checkpoint fired before the guard
                    # below, so a dry run over 25+ rows wrote a cache mapping
                    # real prefix keys to zero counts. A later real run with the
                    # same --out resumed from those zeros and emitted
                    # `segment_tokens` that look counted and never touched a
                    # tokenizer -- while the dry run printed "Nothing was sent
                    # and nothing was written."
                    with open(cache_path, "w") as cf:
                        json.dump(cache, cf)
                    print(f"  {counted:,} rows, {stats['calls']:,} calls, "
                          f"{len(cache):,} cached", file=sys.stderr)

        if sink is not None:
            sink.close()

        if args.dry_run:
            # Deliberately writes nothing. A dry run produced an output file whose
            # `segment_tokens` were all zero, and the loader read that as *counted*
            # -- `tokens_are_counted: True` on a file that had never spoken to a
            # tokenizer, with every segment's size collapsed onto one. A file that
            # looks authoritative and is not is the exact failure this toolchain
            # exists to refuse, so the dry run reports and stops.
            print(f"\n  DRY RUN: {stats['calls']:,} calls would go to {args.endpoint}")
            print(f"  {counted:,} rows would be counted, {skipped:,} skipped.")
            # Named, because the models are what the calls carry and a mixed
            # export costs one set of prefix calls per model. An operator being
            # asked to approve the egress should see both facts before agreeing.
            print(f"  tokenizers asked: {', '.join(sorted(counters)) or 'none'}")
            print("  Nothing was sent and nothing was written.")
            return 0

        # A partial count is not a count. Rows whose tokenizer call failed are
        # written without `segment_tokens` and the analyzer estimates them, which is
        # the right fallback -- but this returned 0 either way, and
        # `run_diagnostic.py` reads only the exit code. So a run where the endpoint
        # failed on the largest row proceeded as though counting had succeeded, and
        # nothing downstream could tell.
        partial = failed > 0
        out_path = args.out if not partial or args.allow_partial else args.out + ".partial"
        os.replace(tmp_path, out_path)
        renamed = True
        if resume:
            with open(cache_path, "w") as cf:
                json.dump(cache, cf)

        print(f"\n  counted   {counted:,} rows")
        print(f"  skipped   {skipped:,} (no recognisable body)")
        if failed:
            print(f"  failed    {failed:,} (left for the analyzer to estimate)")
        print(f"  API calls {stats['calls']:,} for {len(cache):,} distinct prefixes"
              + (f", {stats['calls'] / counted:.1f} per row" if counted else ""))
        # Which tokenizers actually answered. Reported rather than assumed:
        # these counts load as *exact*, so the model behind them belongs in the
        # run's own output where an operator reading a mixed export can see it
        # was counted by more than one -- and where a single unexpected name
        # says the rows were not carrying the model anyone thought they were.
        print(f"  models    {', '.join(sorted(counters)) or 'none'}")
        print(f"  wrote     {out_path}")
        if resume:
            print(f"  cache     {cache_path} (re-runs with the same "
                  f"--tokenizer-id are free)")
        else:
            print("  cache     not written (no --tokenizer-id, so a later run "
                  "could not safely resume from it)")
        if partial and not args.allow_partial:
            print(f"\n  PARTIAL: {failed:,} row(s) could not be counted, so this is "
                  f"not a counted export.\n  Written to {out_path} rather than "
                  f"{args.out}. Re-run, or pass --allow-partial to accept a "
                  f"mixed file.", file=sys.stderr)
            return 1
        return 0
    finally:
        if sink is not None and not sink.closed:
            sink.close()
        if not renamed and os.path.exists(tmp_path):
            os.unlink(tmp_path)


if __name__ == "__main__":
    raise SystemExit(main())
