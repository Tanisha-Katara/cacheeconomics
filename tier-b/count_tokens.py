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

# Overridable, because the clients most likely to care about egress are the
# ones who cannot reach this host. An enterprise gateway, a Bedrock or Vertex
# deployment, or a self-hosted proxy all mean the counting call has to go
# somewhere else, and a hard-coded host makes the answer "edit the source".
DEFAULT_ENDPOINT = "https://api.anthropic.com/v1/messages/count_tokens"


def row_model(body: dict, fallback: str, force: str | None = None) -> str:
    """Which tokenizer counts this row.

    The row's own model, because an export is not one model. A month of a
    client's traffic routinely mixes an opus planner with a haiku worker, and
    the two do not tokenize the same text into the same number of tokens.

    Bound per row rather than read inside `count`, because `prefix_cuts`
    rebuilds each cut out of tools/system/messages alone -- the cut it hands the
    counter does not carry the row's model, so the caller has to say.

    `force` wins when set, for a gateway that will not accept the model ids the
    export happens to contain. `fallback` is for rows that name no model at all.
    """
    if force:
        return force
    m = body.get("model") if isinstance(body, dict) else None
    return m.strip() if isinstance(m, str) and m.strip() else fallback


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
    p.add_argument("--model", default="claude-haiku-4-5",
                   help="fallback tokenizer, used only for rows whose body names "
                        "no model (default: claude-haiku-4-5). Every other row is "
                        "counted with the model it names, so a mixed export is "
                        "counted by each of its models rather than by one")
    p.add_argument("--force-model",
                   help="count every row with this model's tokenizer, ignoring the "
                        "model each row names. For a gateway that does not accept "
                        "the model ids in the export. The counts are then that "
                        "model's, for every row, and are cached under it")
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
                counter_id, count = counter_for(
                    row_model(body, args.model, args.force_model))
                try:
                    row["segment_tokens"] = count_segments(body, count, cache,
                                                      counter_id)
                    counted += 1
                except Exception as e:                                # noqa: BLE001
                    # The row survives without counts and the analyzer falls back to
                    # estimating it, which is worse but is not nothing. Losing the
                    # row entirely would be.
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
        print(f"  models    {', '.join(sorted(counters)) or 'none'}"
              + (" (forced)" if args.force_model else ""))
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
