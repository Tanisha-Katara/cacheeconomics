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
import urllib.error
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "harness"))

from cacheeconomics.adapters.bodies import _find_body            # noqa: E402
from cacheeconomics.tokenizer import count_segments              # noqa: E402
from cacheeconomics.trace import request_from_row                # noqa: E402

# Overridable, because the clients most likely to care about egress are the
# ones who cannot reach this host. An enterprise gateway, a Bedrock or Vertex
# deployment, or a self-hosted proxy all mean the counting call has to go
# somewhere else, and a hard-coded host makes the answer "edit the source".
DEFAULT_ENDPOINT = "https://api.anthropic.com/v1/messages/count_tokens"

# Bumped when anything about how a count is produced changes -- the cut
# construction, the differencing, which model is asked. Recorded in every
# counted row so a reader can tell an export this script produced from one an
# older version did, and refuse to reuse the older one.
COUNTER_VERSION = 1

# Where the record of what produced `segment_tokens` lives on each counted row.
PROVENANCE_KEY = "segment_tokens_provenance"


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


def row_model(row: dict, body: dict, target_id: str | None = None) -> str:
    """Which tokenizer counts this row: whichever model the analyzer will say
    it is.

    This calls the loader rather than reproducing it, and the history is the
    argument. Three rounds of review found three different divergences, each in
    a shape the previous fix had not modelled:

      round 1  the CLI's --model was stamped over every row
      round 2  only the extracted body was read, so a row naming its model at
               the top level resolved to the fallback
      round 3  precedence matched but the order of operations did not -- the
               loader picks the raw value first and coerces once, this coerced
               each candidate before choosing, and six shapes disagreed:

                 body ["bad"], row opus  -> loader unknown,  this opus
                 body 0,       row opus  -> loader opus,     this "0"
                 body {...},   row opus  -> loader unknown,  this opus
                 body True,    row opus  -> loader unknown,  this opus
                 absent in both          -> loader unknown,  this the fallback
                 body opus-20260101      -> loader opus,     this opus-20260101

    Each one writes `segment_tokens` that a different model produced and the
    analyzer then accepts as exact. Matching behaviour is what failed twice, so
    this stops matching it: `request_from_row` is the function that decides what
    `Request.model` is, and its answer is the answer. Divergence is no longer a
    thing that can be got wrong, only a thing that can be renamed.

    The one line still transcribed from the caller is `model_override`, which is
    `bodies.load_bodies`'s argument rather than the resolver's own logic;
    `TestTheCounterAsksTheLoaderWhichModelThisIs` reads it back out of that
    source so it cannot drift either.

    There is no fallback parameter. A row the loader calls "unknown" is counted
    as "unknown", the endpoint refuses it, and the row is left for the analyzer
    to estimate -- which is the honest outcome, because nothing in the export
    says what tokenizer that row used. A `--model` default here counted such a
    row with haiku while the report called it "unknown".
    """
    if not isinstance(row, dict):
        row = {}
    override = body.get("model") if isinstance(body, dict) else None
    kwargs = {"default_target": target_id} if target_id else {}
    return request_from_row(row, [], renamed={}, model_override=override,
                            **kwargs).model


def body_sha256(body) -> str:
    """A digest of the body these counts were taken from.

    Not the body. This lands in a file on a client's disk and the whole point of
    the enrichment is that structure and counts are enough; the count cache made
    exactly this mistake once already and was changed to digests.
    """
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


def provenance(body, model: str, endpoint: str,
               target_id: str | None) -> dict:
    """What produced this row's `segment_tokens`.

    Counted rows used to carry the array and nothing else, and the loader
    accepts any correctly-shaped positive array as exact. So a counted export
    left over from a different endpoint, a different resolved model, an older
    version of this script, or a capture that has since changed was
    indistinguishable from a fresh one -- and `sweep_report.counted` reused it
    on the strength of the filename existing.

    Everything here is an input that changes the counts. The digest covers the
    body, the model and endpoint name the tokenizer that answered, `target_id`
    is an input to the resolved model, and the version invalidates the lot when
    this script's own arithmetic changes.
    """
    return {"version": COUNTER_VERSION, "tool": "tier-b/count_tokens.py",
            "body_sha256": body_sha256(body), "model": model,
            "endpoint": endpoint, "target_id": target_id}


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
                        "do not say. It resolves the model the same way "
                        "`analyze --target-id` does -- a surface's id prefix is "
                        "stripped before the tokenizer is asked -- so pass the "
                        "same value to both or the counted model and the "
                        "analysed model can differ")
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
    if os.path.exists(cache_path) and not args.dry_run:
        with open(cache_path) as f:
            cache = json.load(f)
        # Naming the counter, because after key scoping a model or endpoint
        # change resumes from 0 against a full file, which otherwise reads as a
        # missing cache rather than a deliberate recount. The model half of that
        # name is per row now, so it is reported at the end, once the rows have
        # said which models they are; only the endpoint is knowable up front.
        print(f"  resumed from {len(cache):,} cached counts "
              f"(via {args.endpoint})", file=sys.stderr)
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

    def counter_for(model: str):
        if model not in counters:
            counters[model] = (f"{model}\x00{args.endpoint}",
                               counter(model, key, stats, args.endpoint,
                                       args.dry_run))
        return counters[model]

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
                try:
                    # Inside the try. Resolving the model runs the loader over
                    # the row, and a row malformed enough to break that should
                    # cost its own counts and nothing else -- outside, it took
                    # the whole run down at whichever row it met.
                    model = row_model(row, body, args.target_id)
                    counter_id, count = counter_for(model)
                    row["segment_tokens"] = count_segments(body, count, cache,
                                                      counter_id)
                    # Written together with the counts, never separately: a row
                    # carrying an array and no record of what produced it is
                    # exactly what a stale export looks like.
                    row[PROVENANCE_KEY] = provenance(body, model, args.endpoint,
                                                     args.target_id)
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
                if counted % 25 == 0 and counted and not args.dry_run:
                    # Not during a dry run. The dry-run counter returns 0 for every
                    # prefix, and this checkpoint fires before the guard below, so
                    # a dry run over 25+ rows wrote a cache mapping real prefix keys
                    # to zero counts. A later real run with the same --out resumes
                    # from those zeros and emits `segment_tokens` that look counted
                    # and never touched a tokenizer -- while the dry run printed
                    # "Nothing was sent and nothing was written."
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
        print(f"  cache     {cache_path} (re-runs are free)")
        if partial and not args.allow_partial:
            print(f"\n  PARTIAL: {failed:,} row(s) could not be counted, so this is "
                  f"not a counted export.\n  Written to {out_path} rather than "
                  f"{args.out}. Re-run to pick up the cached prefixes, or pass "
                  f"--allow-partial to accept a mixed file.", file=sys.stderr)
            return 1
        return 0
    finally:
        if sink is not None and not sink.closed:
            sink.close()
        if not renamed and os.path.exists(tmp_path):
            os.unlink(tmp_path)


if __name__ == "__main__":
    raise SystemExit(main())
