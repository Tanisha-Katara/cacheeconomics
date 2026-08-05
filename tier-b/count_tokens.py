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
from cacheeconomics.tokenizer import (COUNTS_PROVENANCE_KEY,      # noqa: E402
                                      COUNTS_PROVENANCE_VERSION,
                                      FIRST_PARTY_COUNT_ENDPOINT, RowModels,
                                      count_segments, countable,
                                      counts_provenance, row_models)

# Overridable, because the clients most likely to care about egress are the
# ones who cannot reach this host. An enterprise gateway, a Bedrock or Vertex
# deployment, or a self-hosted proxy all mean the counting call has to go
# somewhere else, and a hard-coded host makes the answer "edit the source".
#
# The identity itself lives in the package, because deciding whether an id can
# be vouched for is the package's job now and both sides need the same answer.
DEFAULT_ENDPOINT = FIRST_PARTY_COUNT_ENDPOINT

# Both owned by `cacheeconomics.tokenizer`, which defines the vouching contract
# the loader checks. Constants here as well as there was three copies of a pair
# whose only job is that the two sides agree.
COUNTER_VERSION = COUNTS_PROVENANCE_VERSION
PROVENANCE_KEY = COUNTS_PROVENANCE_KEY



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


def load_resume_cache(path: str) -> dict:
    """A resume cache must be a dict of non-negative integer token counts."""
    with open(path) as f:
        cache = json.load(f)
    if not isinstance(cache, dict):
        raise ValueError("expected a JSON object mapping prefix keys to counts")
    for key, value in cache.items():
        bad_key = not isinstance(key, str)
        bad_value = (isinstance(value, bool)
                     or not isinstance(value, int)
                     or value < 0)
        if bad_key or bad_value:
            raise ValueError(
                f"invalid cache entry {key!r}: counts must be non-negative "
                f"integers, got {value!r}")
    return cache


def provenance(row, body, endpoint: str, target_id: str | None,
               tokenizer_id: str | None, assume_serves: bool = False) -> dict:
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
    return {**counts_provenance(body, row, target_id, endpoint, tokenizer_id,
                                assume_serves),
            "tool": "tier-b/count_tokens.py"}


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
    p.add_argument("--assume-endpoint-serves", action="store_true",
                   help="send counting calls for model ids this registry does "
                        "not recognise, asserting that --endpoint serves them. "
                        "For a gateway with its own model names. Without it "
                        "such rows are not counted and nothing is sent for "
                        "them, because an unrecognised id is one nothing here "
                        "can price or vouch for")
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
        try:
            cache = load_resume_cache(cache_path)
        except (OSError, ValueError) as e:
            print(f"  refusing to resume from {cache_path}: {e}. No counting "
                  f"calls were made.", file=sys.stderr)
            return 2
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
        print(f"  (cache file {cache_path} exists; a dry run ignores it and "
              f"reports a cold run)", file=sys.stderr)

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

    # Said here, before the first call goes out, because this is the last moment
    # the choice is still the operator's. Without an asserted tokenizer identity
    # the analyzer will estimate these rows rather than trust them -- correct,
    # since the export would claim nothing about what produced the counts -- but
    # learning that afterwards, from a note, means having already spent the money
    # and the egress on a file that will be treated as though it had neither.
    #
    # Stated, not refused: an operator may want the counts for something other
    # than a report, and this script is not the place to decide that for them.
    if not args.tokenizer_id and not args.dry_run:
        # Not during a dry run, which writes nothing and has its own wording
        # below. The point of this notice is that it is accurate at the moment
        # the choice is still open, and "the counts will be written" is not true
        # of a run that writes nothing.
        print("  NOTE: no --tokenizer-id, so this export will NOT load as exact."
              "\n  The counts will be written and the analyzer will estimate "
              "those rows anyway,\n  because nothing in the file would say which "
              "tokenizer deployment produced\n  them. Pass --tokenizer-id <id> "
              "to have them count.", file=sys.stderr)

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
                src_row = json.loads(line)
                src_body = _find_body(src_row) if isinstance(src_row, dict) else None
                try:
                    # Inside the try. Resolving the model runs the loader over
                    # the row, and a row malformed enough to break that should
                    # cost its own counts and nothing else -- outside, it took
                    # the whole run down at whichever row it met.
                    # Resolved from the pristine parse, never from `row`.
                    # For a flattened export `_find_body` returns the row
                    # itself, so enriching it would otherwise mutate the object
                    # the digest is about to be taken over.
                    models = row_models(src_row, src_body, args.target_id)
                    ok, why = countable(models, args.endpoint,
                                        args.assume_endpoint_serves)
                    if not ok:
                        # Nothing to ask. Raised rather than sent: the previous
                        # version sent the literal string "unknown" and let the
                        # endpoint refuse it, which is a request whose only
                        # possible answer is an error.
                        # Nothing has been sent at this point and nothing
                        # will be: the row is left for the analyzer to estimate.
                        raise ValueError(f"{why}; the analyzer will estimate it")
                    cid, count = counter_for(models)
                    row["segment_tokens"] = count_segments(body, count, cache,
                                                      cid)
                    # Written together with the counts, never separately: a row
                    # carrying an array and no record of what produced it is
                    # exactly what a stale export looks like.
                    row[PROVENANCE_KEY] = provenance(
                        src_row, src_body, args.endpoint,
                        args.target_id, args.tokenizer_id,
                        args.assume_endpoint_serves)
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
            # The dry run exists to answer "what would this do" while the answer
            # can still change the decision, and "you would pay for counts the
            # report then estimates" belongs in that answer as much as the call
            # count does.
            if not args.tokenizer_id:
                print("  WITHOUT --tokenizer-id those calls would buy counts the "
                      "analyzer will estimate\n  anyway: the export would say "
                      "nothing about which tokenizer deployment\n  produced them. "
                      "Add --tokenizer-id <id> before spending the egress.")
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
