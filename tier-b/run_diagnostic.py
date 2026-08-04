"""Count, then analyse. One command, counting on by default.

Counting was a separate step and therefore an optional one, and optional in
practice means skipped. That is the wrong default: without counted sizes the
billed input total is divided between prompt segments in proportion to their
bytes, which measures 19.2% off at the median and 181% at worst, and every
structural finding is costed from that split. The analyzer refuses to attach
dollars to it, so skipping the step does not produce a wrong number -- it
produces a report with the interesting half missing.

So this runs both, counting first, and you have to say `--estimate-only` to opt
out. The opt-out exists because the counting call sends prompt content to a
tokenizer and some clients will not permit that. It is a real choice and it
stays available; it is just no longer the path of least resistance.

What it does not change: the installed package still imports no network library,
and a test asserts it. The socket is here, in a script somebody runs on purpose.
That separation is what makes the permission conversation short -- "the analyzer
cannot phone home, and here is exactly what this one step would send" is a much
easier ask than "trust the flag".

Before asking, show them:

    python3 tier-b/run_diagnostic.py traces.jsonl --dry-run

which reports the call count and the destination host and sends nothing.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python3 tier-b/run_diagnostic.py traces.jsonl --invoice-usd 4820.16 \\
        --format html --out report.html --client "Acme"
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Imported, not re-implemented. This file had its own `_counted_path` and
# `sweep_report.py` had its own `counted`, and the fix that landed here never
# reached the copy one file over -- so a sweep directory named `a.jsonl` still
# had every capture in it written to a directory that does not exist. One
# implementation, in the script that owns the counted export.
sys.path.insert(0, HERE)
from count_tokens import counted_path                            # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        epilog="Any flag this does not recognise is passed straight to "
               "`cacheeconomics analyze`.")
    p.add_argument("path", help="the trace or bodies export")
    p.add_argument("--from", dest="source", default="bodies",
                   choices=["trace", "bodies", "litellm"])
    p.add_argument("--estimate-only", action="store_true",
                   help="skip counting. Structural findings still report; their "
                        "dollar figures do not, because a byte-share split is "
                        "19.2%% off at the median and this report will not "
                        "publish spend that reconciles worse than 5%%")
    p.add_argument("--dry-run", action="store_true",
                   help="report what counting would send, and send nothing")
    p.add_argument("--endpoint", help="send counting calls to your own gateway")
    # No default here. `count_tokens.py` owns it, and this passed its own copy
    # of the same string on every run -- which was fine while both said
    # `claude-haiku-4-5` and silently wrong the moment one of them changed.
    # It is a fallback for rows that name no model; rows that name one are
    # counted with that model's tokenizer.
    p.add_argument("--model",
                   help="fallback tokenizer for rows whose body names no model. "
                        "Rows that name a model are counted with it")
    args, passthrough = p.parse_known_args()

    # A sibling path, derived from the basename only -- see `counted_path`.
    # This is the one command that produces client evidence. Destroying the
    # capture it was handed is the worst thing in its reach, so the collision is
    # still checked here even though the helper cannot produce one.
    out_path = counted_path(args.path)
    if os.path.abspath(out_path) == os.path.abspath(args.path):
        print(f"  refusing to run: the counted output would be written over "
              f"the input at {args.path}. Rename the input, or pass "
              f"--estimate-only to skip counting.", file=sys.stderr)
        return 2

    counted = args.path
    if args.estimate_only:
        print("  counting skipped (--estimate-only). Structural findings will "
              "carry no dollar figures.", file=sys.stderr)
    else:
        cmd = [sys.executable, os.path.join(HERE, "count_tokens.py"), args.path,
               "-o", out_path]
        if args.model:
            cmd += ["--model", args.model]
        if args.endpoint:
            cmd += ["--endpoint", args.endpoint]
        if args.dry_run:
            cmd += ["--dry-run"]
        r = subprocess.run(cmd)
        if args.dry_run:
            print("\n  dry run: nothing was sent and nothing was analysed.",
                  file=sys.stderr)
            return r.returncode
        if r.returncode != 0:
            # Falling through to estimates rather than failing: a report with
            # the structural figures withheld is worth more than no report, and
            # the analyzer says which it produced.
            print("  counting failed; continuing with estimates. Structural "
                  "figures will be withheld.", file=sys.stderr)
        else:
            counted = out_path

    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "harness"))
    return subprocess.run(
        [sys.executable, "-m", "cacheeconomics.cli", "analyze", counted,
         "--from", args.source] + passthrough, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
