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


def counted(path: str) -> str:
    """Exact token counts, or the structural findings carry no figures."""
    out = path.replace(".jsonl", "-counted.jsonl")
    if os.path.exists(out):
        return out
    r = subprocess.run([sys.executable, os.path.join(HERE, "count_tokens.py"),
                        path, "-o", out], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"    counting failed: {r.stderr.strip()[:120]}", file=sys.stderr)
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


def analyse(path: str) -> dict:
    env = dict(os.environ, PYTHONPATH=os.path.join(
        os.path.dirname(HERE), "harness"),
        CACHEECONOMICS_HMAC_KEY=os.environ.get(
            "CACHEECONOMICS_HMAC_KEY", "0" * 64))
    # `--target-id` is required now. A bodies export states the API shape and
    # never the billing surface, so the loader resolves it to UNATTRIBUTED and
    # withholds every figure -- correct, and it broke this script silently until
    # a behavioural test called `analyse()` instead of parsing it. These
    # captures come from capture_proxy pointed at Anthropic, which is a fact
    # this script knows and the export does not carry.
    r = subprocess.run([sys.executable, "-m", "cacheeconomics.cli", "analyze",
                        path, "--from", "bodies", "--allow-unreconciled",
                        "--target-id", "anthropic/direct",
                        "--format", "json"],
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
    rec = 0.0
    if ttl and ttl.get("avoidable_usd_month"):
        rec = _usd(ttl["avoidable_usd_month"]) or 0.0
    # The gate state travels with the numbers. `--allow-unreconciled` releases
    # figures the analyzer stamps DRAFT and marks not for external use, and this
    # parsed the released strings into a committed artifact that said nothing
    # about it -- so a sweep JSON in tier-b/evidence carried dollar projections
    # the normal gate would have withheld, with the caveat left behind in a
    # report nobody keeps.
    draft = [n for n in d.get("notes") or [] if n.startswith("DRAFT")]
    return {"unreconciled": True,
            "gate": ("figures released with --allow-unreconciled and no "
                     "invoice; not reconciled against money that left an "
                     "account"),
            "draft_notes": draft,
            "monthly_input_usd": monthly, "ttl1_usd_month": rec,
            "recoverable_share": (rec / monthly) if monthly else 0.0,
            "ttl1_raised": ttl is not None,
            "window_days": d["window_days"],
            "measured_usd": d["spend"]["input_usd"]}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dir", required=True)
    args = p.parse_args()

    files = sorted(f for f in glob.glob(os.path.join(args.dir, "interval-*.jsonl"))
                   if "-counted" not in f)
    if not files:
        print("no captures found", file=sys.stderr)
        return 2

    rows = []
    for f in files:
        label = os.path.basename(f).replace("interval-", "").replace(".jsonl", "")
        print(f"  {label} ...", file=sys.stderr)
        c = cadence(f)
        if not c.get("n"):
            print("    no gaps; skipped", file=sys.stderr)
            continue
        a = analyse(counted(f))
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
        rec = f"{r['recoverable_share']*100:.0f}%" if r.get("ttl1_raised") else "none"
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
        json.dump({"artifact": "interval-sweep-report", "artifact_version": 1,
                   "project": "browser-use", "model": "claude-haiku-4-5",
                   "points": rows}, f, indent=2)
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
