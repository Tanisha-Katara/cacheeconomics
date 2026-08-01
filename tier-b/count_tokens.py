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
costed from that split. With counts the same comparison lands at 0.2% median.

What it costs. One call per distinct prefix cut, cached, and the endpoint is
free. Prompt caching only pays when the prefix is stable, so the prefix is
shared across requests and counted once: measured on the demo trace, 286
structured requests produced 345 distinct cuts, or 1.2 calls per request. The
cache is written alongside the output so a re-run costs nothing.

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
                   help="model whose tokenizer to ask (default: claude-haiku-4-5). "
                        "Claude models share a tokenizer, so this rarely matters")
    p.add_argument("--cache", help="cache file to read and write (default: <out>.cache.json)")
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
        print(f"  resumed from {len(cache):,} cached counts", file=sys.stderr)
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
    count = counter(args.model, key, stats, args.endpoint, args.dry_run)
    if args.dry_run:
        print(f"  DRY RUN: nothing will be sent. Host that would receive the "
              f"prompt content: {args.endpoint}", file=sys.stderr)

    rows, counted, skipped, failed = [], 0, 0, 0
    with open(args.path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                rows.append(line)          # passed through untouched
                skipped += 1
                continue
            body = _find_body(row) if isinstance(row, dict) else None
            if not body:
                rows.append(json.dumps(row))
                skipped += 1
                continue
            try:
                row["segment_tokens"] = count_segments(body, count, cache)
                counted += 1
            except Exception as e:                                # noqa: BLE001
                # The row survives without counts and the analyzer falls back to
                # estimating it, which is worse but is not nothing. Losing the
                # row entirely would be.
                print(f"  row {counted + failed + skipped}: {type(e).__name__}: {e}",
                      file=sys.stderr)
                failed += 1
            rows.append(json.dumps(row))
            if counted % 25 == 0 and counted:
                with open(cache_path, "w") as cf:
                    json.dump(cache, cf)
                print(f"  {counted:,} rows, {stats['calls']:,} calls, "
                      f"{len(cache):,} cached", file=sys.stderr)

    if args.dry_run:
        # Deliberately writes nothing. A dry run produced an output file whose
        # `segment_tokens` were all zero, and the loader read that as *counted*
        # -- `tokens_are_counted: True` on a file that had never spoken to a
        # tokenizer, with every segment's size collapsed onto one. A file that
        # looks authoritative and is not is the exact failure this toolchain
        # exists to refuse, so the dry run reports and stops.
        print(f"\n  DRY RUN: {stats['calls']:,} calls would go to {args.endpoint}")
        print(f"  {counted:,} rows would be counted, {skipped:,} skipped.")
        print("  Nothing was sent and nothing was written.")
        return 0

    with open(args.out, "w") as f:
        f.write("\n".join(rows) + "\n")
    with open(cache_path, "w") as cf:
        json.dump(cache, cf)

    print(f"\n  counted   {counted:,} rows")
    print(f"  skipped   {skipped:,} (no recognisable body)")
    if failed:
        print(f"  failed    {failed:,} (left for the analyzer to estimate)")
    print(f"  API calls {stats['calls']:,} for {len(cache):,} distinct prefixes"
          + (f", {stats['calls'] / counted:.1f} per row" if counted else ""))
    print(f"  wrote     {args.out}")
    print(f"  cache     {cache_path} (re-runs are free)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
