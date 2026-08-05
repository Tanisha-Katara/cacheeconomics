"""Turn the interval captures into a curve.

One measured point is an anecdote, and a point chosen from inside the band the
finding is about is a weak anecdote. The relationship is the finding: how much
of a workload's input spend a one-hour TTL recovers, as a function of how often
the workload runs.

Each capture is analysed by the same pipeline an engagement would use -- count
the tokens, then run the analyzer -- so nothing here is a special path. What the
report adds is putting the results beside each other and reporting the *share*
rather than the dollar figure, because share is what transfers between workloads
and the dollar figure is an extrapolation from an hour.

Usage:
    python3 tier-b/sweep_report.py --dir sweep/
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
import subprocess
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "harness"))

HERE = os.path.dirname(os.path.abspath(__file__))

# The same helper `run_diagnostic.py` uses, imported rather than copied. This
# file derived the counted path itself with
# `path.replace(".jsonl", "-counted.jsonl")`, which `run_diagnostic.py` was
# fixed for and this copy was not -- so a sweep directory named `a.jsonl` turned
# `sweep/a.jsonl/interval-10m.jsonl` into
# `sweep/a-counted.jsonl/interval-10m-counted.jsonl`, a directory that does not
# exist. Counting then failed, this fell back to the uncounted capture, and the
# curve was drawn from byte-share estimates instead. Two copies of one
# derivation is what produced that, so there is now one.
sys.path.insert(0, HERE)
from count_tokens import (DEFAULT_ENDPOINT, PROVENANCE_KEY,      # noqa: E402
                          counted_path)
from cacheeconomics.adapters.bodies import _find_body            # noqa: E402
from cacheeconomics.tokenizer import counts_provenance           # noqa: E402


def reusable_counts(src: str, out: str, endpoint: str = DEFAULT_ENDPOINT,
                    target_id: str | None = None,
                    tokenizer_id: str | None = None,
                    assume_serves: bool = False):
    """The capture in hand with `out`'s still-valid counts merged onto it, or
    `(None, reason)`.

    Two things were wrong with the version that answered "is `out` fresh?".

    It returned `out` itself, so the counted file supplied the *whole* row --
    usage, timestamps, status, session -- and only the body digest was checked.
    Now the counted file contributes `segment_tokens` and its provenance and
    nothing else; every other field comes from the capture by construction, so
    there is no set of fields left to enumerate and get wrong.

    And it recorded the model without ever comparing it. For the shape where the
    model sits on the row rather than in the body, changing it left the body
    digest identical, so a file counted by the previous tokenizer passed. Both
    models are compared now, resolved from the *current* row, along with a
    digest of that whole row.
    """
    try:
        with open(src) as f:
            src_rows = [json.loads(l) for l in f if l.strip()]
        with open(out) as f:
            out_rows = [json.loads(l) for l in f if l.strip()]
    except (OSError, ValueError) as e:
        return None, f"could not read it ({type(e).__name__})"

    if len(src_rows) != len(out_rows):
        return None, (f"it has {len(out_rows):,} rows and the capture now has "
                      f"{len(src_rows):,}")

    merged = []
    for i, (s, o) in enumerate(zip(src_rows, out_rows)):
        # The current row, never the stored one.
        row = json.loads(json.dumps(s))
        counts = o.get("segment_tokens") if isinstance(o, dict) else None
        if counts is None:
            merged.append(row)           # never counted; nothing to reuse
            continue
        p = o.get(PROVENANCE_KEY)
        if not isinstance(p, dict):
            return None, (f"row {i} carries counts with no record of what "
                          f"produced them (written before counted exports "
                          f"recorded it)")
        body = _find_body(s) if isinstance(s, dict) else None
        if not body:
            return None, (f"row {i} carries counts but the capture no longer "
                          f"has a recognisable body there")
        # The same rule the prefix cache gets, because a counted export IS a
        # cache -- of whole rows rather than of prefixes -- and it was the one
        # of the two reuse paths the rule had not been applied to.
        # `count_tokens.py` refuses to resume without an asserted identity;
        # reusing a counted export written without one was the same claim
        # ("nothing has changed since") made with nothing behind it, and
        # `tokenizer_id=None` on both sides compared equal and passed.
        if not tokenizer_id or not p.get("tokenizer_id"):
            return None, (
                f"row {i} names no tokenizer deployment "
                f"(stored {p.get('tokenizer_id')!r}, this run "
                f"{tokenizer_id!r}), so nothing shows the tokenizer that "
                f"produced these counts is the one answering now")

        # The package's vouching record -- version and the two digests the
        # loader checks -- plus the settings that decide whether a *re-run*
        # would agree. Built from `counts_provenance` rather than restated, so a
        # field added to the contract is compared here the day it is added.
        # The package's record -- version, digests, both models, the surface --
        # plus the two settings only a re-run cares about. Built rather than
        # restated, so this and the loader compare the same fields.
        want = counts_provenance(body, s, target_id, endpoint,
                                 tokenizer_id, assume_serves)
        for field, expected in want.items():
            if p.get(field) != expected:
                return None, (f"row {i} was counted with "
                              f"{field}={p.get(field)!r}, this run needs "
                              f"{expected!r}")
        row["segment_tokens"] = counts
        row[PROVENANCE_KEY] = p
        merged.append(row)
    return merged, None


def counted(path: str, endpoint: str = DEFAULT_ENDPOINT,
            target_id: str | None = None,
            tokenizer_id: str | None = None,
            assume_serves: bool = False) -> str:
    """Exact token counts, or the structural findings carry no figures."""
    out = counted_path(path)
    if os.path.exists(out):
        merged, why = reusable_counts(path, out, endpoint, target_id,
                                      tokenizer_id, assume_serves)
        if why is None:
            # Rewritten from the capture in hand rather than handed back as
            # found. Every row is the current one; only the counts came from the
            # old file, and only after their provenance matched this row.
            with open(out, "w") as f:
                for row in merged:
                    f.write(json.dumps(row) + "\n")
            return out
        # Refused rather than recounted, and this is the deliberate half of the
        # fix. Recounting on a mismatch would send this capture's prompt
        # prefixes to `endpoint` because a file on disk happened to disagree --
        # new egress, decided by the tool, on a path the operator approved for a
        # different question. The stale file is also not silently believed. So
        # the sweep says exactly what it found and analyses the capture
        # uncounted, which the analyzer reports as estimated.
        print(f"    refusing to reuse {os.path.basename(out)}: {why}.\n"
              f"    analysing {os.path.basename(path)} uncounted; its segment "
              f"sizes are estimated.\n"
              f"    delete that file and re-run to count it again.",
              file=sys.stderr)
        return path
    if not tokenizer_id:
        print(f"    counting skipped for {os.path.basename(path)}: no "
              f"--tokenizer-id was supplied, so any new counts would be "
              f"treated as estimates by the analyzer. Pass --tokenizer-id "
              f"to spend counting egress for this evidence artifact.",
              file=sys.stderr)
        return path
    cmd = [sys.executable, os.path.join(HERE, "count_tokens.py"), path,
           "-o", out]
    if endpoint != DEFAULT_ENDPOINT:
        cmd += ["--endpoint", endpoint]
    if target_id:
        cmd += ["--target-id", target_id]
    if tokenizer_id:
        cmd += ["--tokenizer-id", tokenizer_id]
    if assume_serves:
        cmd += ["--assume-endpoint-serves"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        # Named, not just reported. This returns the *uncounted* capture, so
        # every figure derived from this point is estimated while its
        # neighbours in the same sweep are counted -- and a curve mixing the
        # two says nothing about the difference between them.
        print(f"    counting failed for {path}: {r.stderr.strip()[:120]}\n"
              f"    analysing it uncounted; its segment sizes are estimated",
              file=sys.stderr)
        return path
    return out


def cadence(path: str) -> dict:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    pools: dict = {}
    for r in rows:
        pools.setdefault(r.get("session", "?"), []).append(
            datetime.fromisoformat(r["sent_at"]))
    gaps: list = []
    for v in pools.values():
        v.sort()
        gaps += [(b - a).total_seconds() for a, b in zip(v, v[1:])]
    if not gaps:
        return {"n": 0}
    # Two populations, and reporting one median over both is misleading. Most
    # gaps are *within* a cycle -- the seconds between an agent's own steps --
    # so a ten-minute schedule still shows a median gap of eighteen seconds.
    # The schedule appears in the long tail, which is the part the five-minute
    # lifetime dies in, so they are counted separately.
    intra = [g for g in gaps if g < 300]
    inter = [g for g in gaps if g >= 300]
    return {"n": len(gaps), "median": st.median(gaps),
            "median_intra": st.median(intra) if intra else 0.0,
            "median_inter": st.median(inter) if inter else 0.0,
            "in_band": sum(1 for g in gaps if 300 < g < 3600),
            "band_share": sum(1 for g in gaps if 300 < g < 3600) / len(gaps),
            "requests": len(rows)}


def analyse(path: str, target_id: str | None = None) -> dict:
    env = dict(os.environ, PYTHONPATH=os.path.join(
        os.path.dirname(HERE), "harness"),
        CACHEECONOMICS_HMAC_KEY=os.environ.get(
            "CACHEECONOMICS_HMAC_KEY", "0" * 64))
    # Stated by the operator, not assumed by the script. This hard-coded
    # anthropic/direct, which converted a bodies capture from a proxy fronting
    # Bedrock or Vertex back into first-party traffic and -- with
    # --allow-unreconciled also set -- emitted draft dollars for it. The same
    # absence-to-Anthropic failure the adapters were fixed for, at the script
    # layer, falsifying the surface before the analyzer's own guard could see
    # it. Without one the figures stay withheld, which is the correct answer.
    cmd = [sys.executable, "-m", "cacheeconomics.cli", "analyze",
           path, "--from", "bodies", "--allow-unreconciled", "--format", "json"]
    if target_id:
        cmd += ["--target-id", target_id]
    r = subprocess.run(cmd,
                       capture_output=True, text=True, env=env)
    if r.returncode != 0:
        return {"error": r.stderr.strip()[:200]}
    d = json.loads(r.stdout)

    def _usd(text):
        """A figure, or None when the gate withheld it.

        `float("[withheld: ...]")` raised ValueError and took the whole sweep
        down. A withheld figure is an answer, not a crash, and the artifact has
        to be able to say so.
        """
        if not isinstance(text, str) or text.startswith("["):
            return None
        try:
            return float(text.replace("$", "").replace(",", ""))
        except ValueError:
            return None

    monthly = _usd(d["spend"].get("monthly_input_usd"))
    ttl = next((f for f in d["findings"] if f["code"] == "TTL-1"), None)
    rec = None
    if ttl and ttl.get("avoidable_usd_month"):
        rec = _usd(ttl["avoidable_usd_month"])
    # The gate state travels with the numbers. `--allow-unreconciled` releases
    # figures the analyzer stamps DRAFT and marks not for external use, and this
    # parsed the released strings into a committed artifact that said nothing
    # about it -- so a sweep JSON in tier-b/evidence carried dollar projections
    # the normal gate would have withheld, with the caveat left behind in a
    # report nobody keeps.
    draft = [n for n in d.get("notes") or [] if n.startswith("DRAFT")]
    states = d.get("release_state") or {}
    released = [state for state in states.values() if state]
    unreconciled = "draft" in released

    def _withheld_reason(text):
        if isinstance(text, str) and text.startswith("[withheld: "):
            return text[len("[withheld: "):-1] if text.endswith("]") else text
        return "the analyzer withheld the figure"

    gate = (("figures released with --allow-unreconciled and no invoice; not "
             "reconciled against money that left an account")
            if unreconciled else
            f"figures withheld: {_withheld_reason(d['spend'].get('input_usd'))}")
    return {"unreconciled": unreconciled,
            "gate": gate,
            "draft_notes": draft,
            "monthly_input_usd": monthly, "ttl1_usd_month": rec,
            # None, not zero. A withheld figure is unknown, and `rec or 0.0`
            # rendered it as "0%" -- understating a real TTL opportunity in
            # exactly the case where the gate refused to publish the number.
            "recoverable_share": ((rec / monthly)
                                  if (rec is not None and monthly) else None),
            "ttl1_raised": ttl is not None,
            "window_days": d["window_days"],
            "measured_usd": d["spend"]["input_usd"]}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dir", required=True)
    # Reachable at all, which they were not. `counted()` and `analyse()` both
    # take these and `main` called both with defaults, so a sweep of a Bedrock,
    # Vertex or gateway capture could not strip the surface's model prefix and
    # could not send its counting calls anywhere but the default host -- and a
    # counted file produced correctly with --target-id was then refused as stale
    # by this path, because this path passed None. Removing --model left those
    # exports no route through the sweep at all.
    p.add_argument("--target-id",
                   help="the provider surface these captures went to. Passed to "
                        "counting and to analysis, which must agree on it")
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT,
                   help=f"where to send counting calls (default: "
                        f"{DEFAULT_ENDPOINT})")
    p.add_argument("--assume-endpoint-serves", action="store_true",
                   help="count rows whose model id cannot be vouched for "
                        "locally, asserting that --endpoint serves them")
    p.add_argument("--tokenizer-id",
                   help="identifier for the tokenizer deployment behind "
                        "--endpoint; without it counted captures are not "
                        "resumed from a cache")
    args = p.parse_args()

    files = sorted(f for f in glob.glob(os.path.join(args.dir, "interval-*.jsonl"))
                   if "-counted" not in f)
    if not files:
        print("no captures found", file=sys.stderr)
        return 2

    rows = []
    for f in files:
        # `splitext`, not `.replace(".jsonl", "")`: the same idiom that made the
        # counted path wrong, here only cosmetic (it would eat both extensions
        # of `interval-10m.jsonl.jsonl` and label the point wrongly), but there
        # is no reason to keep the one shape of this that reads as correct.
        label = os.path.splitext(os.path.basename(f))[0].replace("interval-", "")
        print(f"  {label} ...", file=sys.stderr)
        c = cadence(f)
        if not c.get("n"):
            print("    no gaps; skipped", file=sys.stderr)
            continue
        a = analyse(counted(f, args.endpoint, args.target_id,
                            args.tokenizer_id, args.assume_endpoint_serves),
                    args.target_id)
        rows.append({"label": label, **c, **a})

    def schedule_seconds(r):
        lab = r["label"]
        return float(lab[:-1]) if lab.endswith("s") and lab[:-1].isdigit() else 0.0

    rows.sort(key=schedule_seconds)
    print()
    print(f"  {'schedule':<12} {'between runs':>13} {'in band':>9} {'reqs':>5} "
          f"{'recoverable':>12}")
    print("  " + "-" * 56)
    for r in rows:
        band = f"{r['in_band']}/{r['n']}"
        share = r.get("recoverable_share")
        rec = ("withheld" if (r.get("ttl1_raised") and share is None)
               else f"{share*100:.0f}%" if r.get("ttl1_raised") else "none")
        between = (f"{r['median_inter']:.0f}s" if r["median_inter"] else "under 5m")
        print(f"  {r['label']:<12} {between:>13} {band:>9} {r['requests']:>5} "
              f"{rec:>12}")
    print()
    print("  `between runs` is the median gap above five minutes -- the schedule.")
    print("  Gaps below it are an agent's own steps, which happen in seconds")
    print("  whatever the schedule, and a single median over both hides the")
    print("  quantity that decides this: a ten-minute timer still shows a median")
    print("  gap of eighteen seconds.")
    print()
    print("  Recoverable is TTL-1 as a share of input spend. The share transfers")
    print("  between workloads; the dollar figures behind it are extrapolations")
    print("  from under two hours of traffic each and are not quoted here.")

    out = os.path.join(args.dir, "sweep-report.json")
    with open(out, "w") as f:
        # The surface and the counting host travel with the numbers. Both change
        # what the points mean -- the surface decides which rate table applies
        # and how model ids resolve, the endpoint decides which tokenizer sized
        # the segments -- and an artifact that records neither cannot be told
        # apart from one run against different settings.
        json.dump({"artifact": "interval-sweep-report", "artifact_version": 2,
                   "project": "browser-use", "model": "claude-haiku-4-5",
                   "target_id": args.target_id,
                   "count_endpoint": args.endpoint,
                   "tokenizer_id": args.tokenizer_id,
                   "points": rows}, f, indent=2)
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
