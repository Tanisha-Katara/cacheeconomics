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
from count_tokens import (COUNTER_VERSION, DEFAULT_ENDPOINT,     # noqa: E402
                          PROVENANCE_KEY, body_sha256, counted_path)
from cacheeconomics.adapters.bodies import _find_body            # noqa: E402


def stale_reason(src: str, out: str, endpoint: str = DEFAULT_ENDPOINT,
                 target_id: str | None = None) -> str | None:
    """Why `out` cannot be treated as the counted form of `src`, or None.

    This existed because the previous rule was "the file is there". Counted rows
    carry `segment_tokens`, and the loader accepts any correctly-shaped positive
    array as exact -- so a counted export left over from a capture that has
    since been re-recorded, or from a different endpoint, or from an older
    version of the counter, was analysed as exact and nothing anywhere could
    tell. Removing the flag that produced some of those files closed the door
    and left the window open.

    Every input that changes a count is compared: the body each row was counted
    from, the tokenizer that answered, the host it answered from, the surface
    used to resolve the model, and the version of the script.
    """
    try:
        with open(src) as f:
            src_rows = [json.loads(l) for l in f if l.strip()]
        with open(out) as f:
            out_rows = [json.loads(l) for l in f if l.strip()]
    except (OSError, ValueError) as e:
        return f"could not read it ({type(e).__name__})"

    if len(src_rows) != len(out_rows):
        return (f"it has {len(out_rows):,} rows and the capture now has "
                f"{len(src_rows):,}")

    for i, (s, o) in enumerate(zip(src_rows, out_rows)):
        if not isinstance(o, dict) or "segment_tokens" not in o:
            continue                     # never counted; nothing to trust
        p = o.get(PROVENANCE_KEY)
        if not isinstance(p, dict):
            return (f"row {i} carries counts with no record of what produced "
                    f"them (written before counted exports recorded it)")
        body = _find_body(s) if isinstance(s, dict) else None
        expected = {"version": COUNTER_VERSION,
                    "body_sha256": body_sha256(body) if body else None,
                    "endpoint": endpoint, "target_id": target_id}
        for field, want in expected.items():
            if p.get(field) != want:
                return (f"row {i} was counted with {field}={p.get(field)!r}, "
                        f"this run needs {want!r}")
    return None


def counted(path: str, endpoint: str = DEFAULT_ENDPOINT,
            target_id: str | None = None) -> str:
    """Exact token counts, or the structural findings carry no figures."""
    out = counted_path(path)
    if os.path.exists(out):
        why = stale_reason(path, out, endpoint, target_id)
        if why is None:
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
    cmd = [sys.executable, os.path.join(HERE, "count_tokens.py"), path,
           "-o", out]
    if endpoint != DEFAULT_ENDPOINT:
        cmd += ["--endpoint", endpoint]
    if target_id:
        cmd += ["--target-id", target_id]
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
    return {"unreconciled": True,
            "gate": ("figures released with --allow-unreconciled and no "
                     "invoice; not reconciled against money that left an "
                     "account"),
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
        json.dump({"artifact": "interval-sweep-report", "artifact_version": 1,
                   "project": "browser-use", "model": "claude-haiku-4-5",
                   "points": rows}, f, indent=2)
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
