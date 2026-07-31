#!/usr/bin/env python3
"""
TTL experiment — does a 1-hour cache beat a 5-minute cache, and at what gap?

Tests the convergent Tier A finding: OpenHands, SWE-agent and browser-use all
emit {"type": "ephemeral"} with no ttl key, i.e. the 5-minute cache, never 1h.

Design
------
CRITICAL: a cache HIT REFRESHES the TTL at no write cost. An earlier version of
this script probed at t=120s (a hit, which silently reset the clock) and then
treated t=420s as an expiry test — 300s after a refresh, i.e. exactly on the
boundary. That made the core comparison a race. Fixed by never probing before
the expiry test, and by scheduling every probe relative to the last observed
refresh for that arm.

Schedule (both arms share one wall clock; 1h arm doubles as the stability control):

    t=0     WRITE both arms                 expect: WRITE  / WRITE
    t=420   probe both  (300 TTL + 120 s margin, no prior refresh)
                                            expect: MISS   / HIT
    t=480   probe both  (60 s after the 5m arm's t=420 rewrite)
                                            expect: HIT    / HIT   <- stability control
    t=900   probe both  (420 s after the 5m arm's t=480 refresh)
                                            expect: MISS   / HIT

The t=480 control matters: without it, a miss at t=420 is ambiguous between
"expired" and "prefix was never byte-stable". A hit there proves the prefix is
stable, so the surrounding misses are genuine expiry.

Every request uses max_tokens=0: the API runs prefill (which is what reads and
writes the cache), returns immediately with content=[], and bills zero output
tokens.

Each arm uses a DIFFERENT prefix so the two caches never collide. Within an arm
the prefix is byte-identical across every request.

Stdlib only. No pip install.

Usage
-----
    export ANTHROPIC_API_KEY=sk-ant-...
    python3 ttl_experiment.py --dry-run     # plan + cost estimate, no calls
    python3 ttl_experiment.py               # real run (~15 min wall clock)
    python3 ttl_experiment.py --report      # analyse the most recent run
    python3 ttl_experiment.py --report --run-id <id>
"""

import argparse
import glob
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
HERE = os.path.dirname(os.path.abspath(__file__))

# Verified 2026-07-28 from platform.claude.com/docs/en/build-with-claude/prompt-caching
MULT_WRITE_5M = 1.25
MULT_WRITE_1H = 2.00
MULT_READ = 0.10
TTL_5M_SECONDS = 300

# Base input $/Mtok as DATE-EFFECTIVE tiers, because at least one model has
# introductory pricing with a hard end date. Rates verified 2026-07-28 against
# platform.claude.com/docs/en/build-with-claude/prompt-caching.
# Each entry: (effective_from ISO date, base $/Mtok). Latest applicable wins.
MODELS = {
    "claude-sonnet-4-6": {"min": 1024, "rates": [("1970-01-01", 3.00)]},
    # Sonnet 5 introductory: $2 through 2026-08-31, $3 from 2026-09-01.
    "claude-sonnet-5": {"min": 1024, "rates": [("1970-01-01", 2.00),
                                               ("2026-09-01", 3.00)]},
    "claude-haiku-4-5": {"min": 4096, "rates": [("1970-01-01", 1.00)]},
    "claude-opus-4-8": {"min": 1024, "rates": [("1970-01-01", 5.00)]},
}


def base_rate_for(model: str, on_date: str) -> float:
    """Resolve the base input rate effective on `on_date` (ISO yyyy-mm-dd).

    Pricing is resolved once at run start and persisted into the run's config
    header, so a report generated after a price change still costs the run at
    the rate that actually applied when it executed.
    """
    tiers = sorted(MODELS[model]["rates"])
    applicable = [rate for start, rate in tiers if start <= on_date]
    if not applicable:
        raise ValueError(f"no rate for {model} on {on_date}")
    return applicable[-1]

# (label, seconds from t=0, expected verdict for 5m arm, for 1h arm)
# Expectations are asserted in the report so an anomaly is visible, not silent.
DEFAULT_SCHEDULE = [
    ("t0-write", 0, "WRITE", "WRITE"),
    ("expiry-1", 420, "MISS", "HIT"),
    ("control", 480, "HIT", "HIT"),
    ("expiry-2", 900, "MISS", "HIT"),
]


def build_prefix(arm: str, target_tokens: int, run_id: str) -> str:
    """Filler that is byte-identical within a run and unique across runs.

    The run_id in the header matters more than it looks. Without it the prefix
    is deterministic across runs, so a second run started inside the 1-hour
    window inherits the first run's cache: the 1h arm opens on a HIT while the
    5m arm (long expired) opens on a WRITE, and the comparison is silently
    contaminated. Observed for real on run 20260728T192418Z, 56 minutes after
    its predecessor. Byte-identical *within* a run is the property the cache
    needs; identical *between* runs is a bug.
    """
    header = (
        f"[STABLE REFERENCE CORPUS · ARM {arm} · RUN {run_id}]\n"
        "Stand-in for the static prefix of an agentic system prompt: tool "
        "definitions, operating instructions, policy text. Byte-identical on "
        "every request within this arm.\n\n"
    )
    body = []
    i = 0
    while len(header) + sum(len(s) for s in body) < target_tokens * 4:
        body.append(
            f"{arm}-{i:06d}: reference clause defining permitted operations, "
            "invariants, and failure handling for the agent under test.\n"
        )
        i += 1
    return header + "".join(body)


def call(api_key, model, prefix, ttl, tag, expected):
    """One prefill-only request. Returns a normalised usage record."""
    cache_control = {"type": "ephemeral"}
    if ttl == "1h":
        cache_control["ttl"] = "1h"

    payload = {
        "model": model,
        "max_tokens": 0,  # prefill only: exercises the cache, bills no output
        "system": [{"type": "text", "text": prefix, "cache_control": cache_control}],
        "messages": [{"role": "user", "content": "."}],
    }
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode(),
        headers={
            "x-api-key": api_key,
            "anthropic-version": API_VERSION,
            "content-type": "application/json",
        },
        method="POST",
    )
    sent = time.time()
    base = {"tag": tag, "ttl": ttl, "expected": expected,
            "ts": datetime.now(timezone.utc).isoformat(), "sent_epoch": sent}
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            body = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        base["error"] = f"HTTP {e.code}: {e.read().decode()[:400]}"
        return base
    except Exception as e:
        base["error"] = f"{type(e).__name__}: {e}"
        return base

    u = body.get("usage", {}) or {}
    base.update({
        "latency_s": round(time.time() - sent, 3),
        "input_tokens": u.get("input_tokens", 0),
        "cache_creation_input_tokens": u.get("cache_creation_input_tokens", 0),
        "cache_read_input_tokens": u.get("cache_read_input_tokens", 0),
        "output_tokens": u.get("output_tokens", 0),
        "stop_reason": body.get("stop_reason"),
    })
    return base


def validate_run(cfg, recs):
    """Single source of truth for whether a run is publishable.

    Both report() and export_evidence() call this. They previously applied
    different rules, which meant a run killed after the first probe could fail
    report() for missing cells and still export an artifact stamped valid.

    Returns (errors, missing, timing, anomalies).
    """
    errors = [r for r in recs if "error" in r]
    expected_cells = {(s[0], ttl) for s in cfg["schedule"] for ttl in ("5m", "1h")}
    present_cells = {(r["tag"], r["ttl"]) for r in recs if "error" not in r}
    missing = expected_cells - present_cells
    timing = validate_timing(cfg, recs)
    anomalies = [r for r in recs
                 if "error" not in r and verdict(r) != r.get("expected")]
    return errors, missing, timing, anomalies


def estimate_cost(schedule, tokens, base_rate):
    """Expected and worst-case input cost, derived from the schedule itself.

    An earlier version priced both arms as one write plus reads, which ignored
    that the 5m arm is *expected* to miss and re-write at every expiry probe.
    That under-quoted the run by roughly a third. This walks the declared
    expectations and applies the same verdict-to-cost rules as report(), so
    the quote and the invoice come from one model.
    """
    per_token = base_rate / 1_000_000
    expected = 0.0
    for label, _offset, exp5, exp1 in schedule:
        for ttl, exp in (("5m", exp5), ("1h", exp1)):
            mult_write = MULT_WRITE_1H if ttl == "1h" else MULT_WRITE_5M
            # WRITE and MISS both pay a write; MISS is a miss that re-writes.
            mult = MULT_READ if exp == "HIT" else mult_write
            expected += tokens * per_token * mult
    # Worst case: nothing ever caches, so every call pays the dearer write.
    worst = len(schedule) * 2 * tokens * per_token * MULT_WRITE_1H
    return expected, worst


def validate_timing(cfg, recs, drift_tolerance=30.0, gap_margin=60.0):
    """Check the run actually happened on the schedule it claims.

    Fixed sleeps do not guarantee timing: the machine can suspend, an API call
    can run long, the process can be paused. Since TTL behaviour is the thing
    under test, a late probe silently changes the experiment's meaning.

    Two independent checks:

    1. Drift — each probe's actual offset vs its planned offset.
    2. Semantic gaps — the load-bearing one. Per arm, walk records in time
       order tracking the last refresh (a write or a hit, since a hit refreshes
       the TTL). A probe expected to MISS must be more than TTL+margin after
       that refresh; a probe expected to HIT must be less than TTL-margin.
       This tests the invariant the result depends on, not just the clock.
    """
    problems = []
    planned = {s[0]: s[1] for s in cfg["schedule"]}
    order = [s[0] for s in cfg["schedule"]]

    t0_candidates = [r["sent_epoch"] for r in recs
                     if r.get("tag") == order[0] and "sent_epoch" in r]
    if not t0_candidates:
        return ["cannot validate timing: no t0 record with sent_epoch"]
    t0 = min(t0_candidates)

    for r in recs:
        if "sent_epoch" not in r or "error" in r:
            continue
        actual = r["sent_epoch"] - t0
        drift = actual - planned[r["tag"]]
        if abs(drift) > drift_tolerance:
            problems.append(
                f"{r['tag']}/{r['ttl']}: fired {drift:+.1f}s off schedule "
                f"(planned t={planned[r['tag']]}s, actual t={actual:.1f}s)")

    for ttl in ("5m", "1h"):
        if ttl != "5m":
            continue  # only the 5m arm's expiry is being asserted
        arm = sorted([r for r in recs if r.get("ttl") == ttl and "error" not in r
                      and "sent_epoch" in r], key=lambda r: r["sent_epoch"])
        last_refresh = None
        for r in arm:
            if last_refresh is not None:
                gap = r["sent_epoch"] - last_refresh
                exp = r.get("expected")
                if exp == "MISS" and gap <= TTL_5M_SECONDS + gap_margin:
                    problems.append(
                        f"{r['tag']}/{ttl}: expected MISS but fired only {gap:.1f}s "
                        f"after the last refresh (needs > {TTL_5M_SECONDS + gap_margin:.0f}s "
                        f"to be an unambiguous expiry test)")
                if exp == "HIT" and gap >= TTL_5M_SECONDS - gap_margin:
                    problems.append(
                        f"{r['tag']}/{ttl}: expected HIT but fired {gap:.1f}s after the "
                        f"last refresh (too close to the {TTL_5M_SECONDS}s TTL to be a "
                        f"clean stability control)")
            # A write or a hit both (re)establish the entry and reset the clock.
            if verdict(r) in ("WRITE", "MISS", "HIT"):
                last_refresh = r["sent_epoch"]
    return problems


def verdict(rec):
    if "error" in rec:
        return "ERROR"
    if rec.get("cache_read_input_tokens", 0) > 0:
        return "HIT"
    if rec.get("cache_creation_input_tokens", 0) > 0:
        # A write at t=0 is the intended WRITE; a write later means the prior
        # entry was gone, i.e. a MISS that had to re-write.
        return "WRITE" if rec["tag"] == "t0-write" else "MISS"
    return "NO-CACHE"


def cost_usd(rec, base_rate):
    """Input-side cost. Raises on error records — they must never score as 0."""
    if "error" in rec:
        raise ValueError(f"cannot cost an error record: {rec['tag']}/{rec['ttl']}")
    mult_write = MULT_WRITE_1H if rec["ttl"] == "1h" else MULT_WRITE_5M
    per_token = base_rate / 1_000_000
    return (
        rec["cache_creation_input_tokens"] * per_token * mult_write
        + rec["cache_read_input_tokens"] * per_token * MULT_READ
        + rec["input_tokens"] * per_token
    )


def run_path(run_id):
    return os.path.join(HERE, f"ttl_run_{run_id}.jsonl")


def latest_run_id():
    files = sorted(glob.glob(os.path.join(HERE, "ttl_run_*.jsonl")), key=os.path.getmtime)
    if not files:
        return None
    return os.path.basename(files[-1])[len("ttl_run_"):-len(".jsonl")]


def report(run_id, allow_anomalies=False):
    """Analyse ONE run.

    Fails closed: emits a cost verdict ONLY for a run that is complete, free of
    API errors, and in which every probe matched its declared expectation.
    """
    path = run_path(run_id)
    if not os.path.exists(path):
        sys.exit(f"no results for run {run_id} at {path}")
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if not rows or rows[0].get("record") != "config":
        sys.exit(f"{path}: missing config header — cannot determine model/rates")

    cfg = rows[0]
    recs = [r for r in rows if r.get("record") != "config"]
    base_rate = cfg["base_rate"]

    print(f"\nrun_id      {cfg['run_id']}")
    print(f"model       {cfg['model']}  (${base_rate}/Mtok in)")
    print(f"prefix      ~{cfg['tokens']} tokens/arm")
    print(f"schedule    {[s[0] for s in cfg['schedule']]}")
    print(f"started     {cfg['started']}")

    errors, missing, timing, anomalies = validate_run(cfg, recs)

    print(f"\n{'tag':<10} {'ttl':<4} {'verdict':<8} {'exp':<6} {'ok':<3} "
          f"{'read':>8} {'write':>8} {'uncached':>9} {'cost_usd':>9}")
    print("-" * 78)
    for r in recs:
        if "error" in r:
            print(f"{r['tag']:<10} {r['ttl']:<4} {'ERROR':<8} {'':<6} {'!':<3} "
                  f"{r['error'][:44]}")
            continue
        v = verdict(r)
        ok = "ok" if v == r.get("expected") else "!!"
        print(f"{r['tag']:<10} {r['ttl']:<4} {v:<8} {str(r.get('expected')):<6} {ok:<3} "
              f"{r['cache_read_input_tokens']:>8} {r['cache_creation_input_tokens']:>8} "
              f"{r['input_tokens']:>9} {cost_usd(r, base_rate):>9.5f}")
    print("-" * 78)

    anomalies = [r for r in recs if "error" not in r and verdict(r) != r.get("expected")]

    # Fail closed. An anomaly means the TTL model, the prefix stability, or the
    # provider's behaviour is not what this experiment assumes — in which case
    # the arithmetic is still computable but the comparison is meaningless.
    # Emitting a tidy dollar verdict from an invalid experiment is precisely
    # how a measurement turns into a false published claim.
    hard_fail = bool(errors or missing or timing)   # never overridable
    soft_fail = bool(anomalies) and not allow_anomalies

    if timing:
        print(f"timing      {len(timing)} violation(s)")
    else:
        print("timing      verified — all probes on schedule")

    if hard_fail or soft_fail:
        print("\nRUN INVALID — no cost verdict will be emitted.")
        for r in errors:
            print(f"  error:   {r['tag']}/{r['ttl']}: {r['error'][:100]}")
        for cell in sorted(missing):
            print(f"  missing: {cell[0]}/{cell[1]}")
        for t in timing:
            print(f"  timing:  {t}")
        for r in anomalies:
            print(f"  anomaly: {r['tag']}/{r['ttl']}: got {verdict(r)}, "
                  f"expected {r['expected']}")
        if timing:
            print("\nA probe that fired off-schedule measures a different experiment than"
                  "\nthe one described. Timing failures are never overridable.")
        if soft_fail:
            print("\nAn unexpected probe result invalidates the comparison: the cache did"
                  "\nnot behave the way this schedule assumes, so the arithmetic is"
                  "\ncomputable but meaningless. Investigate, then re-run."
                  "\nTo inspect the numbers anyway (NOT publishable): --allow-anomalies")
        print("\nRe-run before publishing anything.")
        sys.exit(2)

    if anomalies:  # only reachable with --allow-anomalies
        print("\n*** --allow-anomalies: the run did NOT behave as expected. ***")
        for r in anomalies:
            print(f"    {r['tag']}/{r['ttl']}: got {verdict(r)}, expected {r['expected']}")
        print("*** Figures below are diagnostic only and MUST NOT be published. ***")

    totals = {"5m": 0.0, "1h": 0.0}
    for r in recs:
        totals[r["ttl"]] += cost_usd(r, base_rate)

    print(f"{'TOTAL 5m arm':<24} ${totals['5m']:.5f}")
    print(f"{'TOTAL 1h arm':<24} ${totals['1h']:.5f}")

    diff = totals["5m"] - totals["1h"]
    if abs(diff) < 1e-9:
        print("\nVERDICT: identical cost on this schedule.")
    elif diff > 0:
        print(f"\nVERDICT: 1h CHEAPER by ${diff:.5f} "
              f"({100 * diff / totals['5m']:.1f}%) on this schedule.")
    else:
        print(f"\nVERDICT: 5m CHEAPER by ${-diff:.5f} "
              f"({100 * -diff / totals['1h']:.1f}%) on this schedule.")

    print(f"\nMEASURED for THIS gap schedule and prefix size only, priced at the rate "
          f"in effect on {cfg.get('priced_on', 'unknown date')} (${base_rate}/Mtok).")
    print("Extrapolation to any real workload is MODELED and must be labelled so.")


def export_evidence(run_id, allow_anomalies=False):
    """Emit a committable evidence artifact for a completed run.

    The published claims (dollar figures, hit/miss pattern, drift) are only
    checkable if the per-call rows behind them travel with them. Raw run files
    stay gitignored because a real workload's rows could carry account-specific
    volume; this artifact is the auditable subset, plus a hash of the script
    that produced it so the analysis is reproducible.

    Nothing here is redacted for this experiment because nothing sensitive is
    captured: the harness records token counts and timings only, never prompt
    content. A run against real traffic would need a redaction pass first.
    """
    path = run_path(run_id)
    if not os.path.exists(path):
        sys.exit(f"no results for run {run_id}")
    rows = [json.loads(l) for l in open(path) if l.strip()]
    cfg = rows[0]
    recs = [r for r in rows if r.get("record") != "config"]
    base_rate = cfg["base_rate"]

    # Must come from the run, not from disk now: the harness may have changed
    # between the API calls and this export, in which case hashing the current
    # file would point auditors at code that never produced these rows.
    script_hash = cfg.get("script_sha256")
    provenance = "recorded at run time"
    if not script_hash:
        script_hash = None
        provenance = ("UNVERIFIABLE: this run predates run-time script hashing, "
                      "so the code that produced these rows cannot be identified")

    errors, missing, timing, anomalies = validate_run(cfg, recs)
    # Errors, missing cells and timing violations are never exportable: the run
    # did not happen as described. An expectation anomaly is different — the run
    # happened exactly as recorded, it just did not behave as assumed, and that
    # observation can be worth keeping so long as the artifact says loudly that
    # it is not a valid comparison.
    hard = bool(errors or missing or timing)
    if hard or (anomalies and not allow_anomalies):
        print("REFUSING to export evidence for an invalid run:")
        for r in errors:
            print(f"  error:   {r['tag']}/{r['ttl']}")
        for cell in sorted(missing):
            print(f"  missing: {cell[0]}/{cell[1]}")
        for m in timing:
            print(f"  timing:  {m}")
        for r in anomalies:
            print(f"  anomaly: {r['tag']}/{r['ttl']}: got {verdict(r)}, "
                  f"expected {r['expected']}")
        if anomalies and not hard:
            print("\nThis run is complete and correctly timed but did not behave as "
                  "expected.\nTo keep it as an observation rather than a comparison: "
                  "--export-evidence --allow-anomalies")
        sys.exit("\nAn evidence artifact must not silently bless a partial or "
                 "anomalous run.")

    t0 = min(r["sent_epoch"] for r in recs if r["tag"] == cfg["schedule"][0][0])

    calls, totals = [], {"5m": 0.0, "1h": 0.0}
    for r in recs:
        c = cost_usd(r, base_rate)
        totals[r["ttl"]] += c
        calls.append({
            "tag": r["tag"], "ttl": r["ttl"],
            "expected": r.get("expected"), "verdict": verdict(r),
            "planned_offset_s": r.get("planned_offset"),
            "actual_offset_s": round(r["sent_epoch"] - t0, 3),
            "cache_read_input_tokens": r["cache_read_input_tokens"],
            "cache_creation_input_tokens": r["cache_creation_input_tokens"],
            "input_tokens": r["input_tokens"],
            "output_tokens": r["output_tokens"],
            "latency_s": r.get("latency_s"),
            "cost_usd": round(c, 6),
        })

    art = {
        "artifact": "ttl-experiment-evidence", "artifact_version": 1,
        "run_id": cfg["run_id"], "started": cfg["started"],
        "script": "ttl_experiment.py", "script_sha256": script_hash,
        "script_hash_provenance": provenance,
        "model": cfg["model"], "prefix_tokens_requested": cfg["tokens"],
        "prefix_tokens_cached": recs[0]["cache_creation_input_tokens"],
        "min_cacheable_tokens": cfg["min_cacheable_tokens"],
        "base_rate_usd_per_mtok": base_rate,
        "base_rate_priced_on": cfg.get("priced_on"),
        "base_rate_source": cfg.get("base_rate_source"),
        "multipliers": {"write_5m": cfg["mult_write_5m"],
                        "write_1h": cfg["mult_write_1h"], "read": cfg["mult_read"]},
        "schedule": cfg["schedule"],
        "validation": {
            "timing_violations": timing,
            "expectation_anomalies": [
                {"cell": f"{r['tag']}/{r['ttl']}", "got": verdict(r),
                 "expected": r.get("expected")} for r in anomalies],
            "valid": not timing and not anomalies,
            "usable_as": ("comparison" if not anomalies else
                          "observation only — NOT a valid cost comparison"),
        },
        "calls": calls,
        "totals_usd": {k: round(v, 6) for k, v in totals.items()},
        "delta_usd": round(totals["5m"] - totals["1h"], 6),
        "delta_pct": round(100 * (totals["5m"] - totals["1h"]) / totals["5m"], 2)
                     if totals["5m"] else None,
        "scope": ("Measured for this gap schedule and prefix size only. "
                  "Extrapolation to any real workload is modeled, not measured."),
    }

    out_dir = os.path.join(HERE, "evidence")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{run_id}.json")
    with open(out, "w") as f:
        json.dump(art, f, indent=2)
        f.write("\n")
    print(f"wrote {out}")
    if anomalies:
        art["totals_usd"] = None
        art["delta_usd"] = None
        art["delta_pct"] = None
        art["totals_suppressed_because"] = (
            "The run contains an expectation anomaly, so arm totals are not "
            "comparable and are omitted rather than published alongside a caveat "
            "nobody reads. Per-call rows are retained in full.")
        with open(out, "w") as f:
            json.dump(art, f, indent=2); f.write("\n")
        print(f"  valid: False  (observation only, totals suppressed)")
        for a in art["validation"]["expectation_anomalies"]:
            print(f"    anomaly: {a['cell']}: got {a['got']}, expected {a['expected']}")
    else:
        print(f"  valid: {art['validation']['valid']}   "
              f"5m ${totals['5m']:.5f}  1h ${totals['1h']:.5f}  "
              f"delta {art['delta_pct']}%")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="claude-sonnet-4-6", choices=sorted(MODELS))
    p.add_argument("--tokens", type=int, default=20000)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--report", action="store_true")
    p.add_argument("--run-id", default=None)
    p.add_argument("--export-evidence", action="store_true",
                   help="write a committable evidence artifact for a run")
    p.add_argument("--allow-anomalies", action="store_true",
                   help="report costs even when probes defied expectation "
                        "(diagnostic only — never publishable)")
    p.add_argument("--base-rate", type=float, default=None,
                   help="override base input $/Mtok (e.g. negotiated pricing)")
    args = p.parse_args()

    if args.export_evidence:
        rid = args.run_id or latest_run_id()
        if not rid:
            sys.exit("no runs found")
        export_evidence(rid, allow_anomalies=args.allow_anomalies)
        return

    if args.report:
        rid = args.run_id or latest_run_id()
        if not rid:
            sys.exit("no runs found")
        report(rid, allow_anomalies=args.allow_anomalies)
        return

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    min_tokens = MODELS[args.model]["min"]
    base_rate = args.base_rate if args.base_rate is not None \
        else base_rate_for(args.model, today)
    if args.tokens < min_tokens:
        sys.exit(f"ERROR: {args.tokens} tokens is below {args.model}'s minimum of "
                 f"{min_tokens}. The cache would silently not form and the "
                 f"experiment would measure nothing.")

    sched = DEFAULT_SCHEDULE
    n_calls = 2 * len(sched)
    est, worst = estimate_cost(sched, args.tokens, base_rate)

    src = "CLI override" if args.base_rate is not None else f"rate effective {today}"
    print(f"model         {args.model}  (${base_rate}/Mtok in [{src}], min {min_tokens} tok)")
    print(f"prefix        ~{args.tokens} tokens per arm, byte-identical within arm")
    print(f"arms          5m vs 1h, distinct prefixes so caches cannot collide")
    print(f"schedule      {[(s[0], s[1]) for s in sched]}")
    print(f"              hits refresh the TTL — no probe precedes the expiry test")
    print(f"wall clock    {sched[-1][1] / 60:.0f} min")
    print(f"requests      {n_calls} (max_tokens=0, prefill only, no output billed)")
    print(f"est. cost     ~${est:.3f}   worst case ${worst:.3f}")

    if args.dry_run:
        print("\n--dry-run: no API calls made.")
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("\nANTHROPIC_API_KEY is not set. export it and re-run.")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = run_path(run_id)
    if os.path.exists(path):
        sys.exit(f"refusing to overwrite {path}")

    prefixes = {"5m": build_prefix("A", args.tokens, run_id),
                "1h": build_prefix("B", args.tokens, run_id)}
    out = open(path, "w")
    script_sha256 = hashlib.sha256(
        open(os.path.abspath(__file__), "rb").read()).hexdigest()
    out.write(json.dumps({
        "record": "config", "run_id": run_id, "model": args.model,
        "script_sha256": script_sha256,
        "tokens": args.tokens, "base_rate": base_rate,
        "priced_on": today,
        "base_rate_source": "cli-override" if args.base_rate is not None
                            else "date-effective table",
        "min_cacheable_tokens": min_tokens, "schedule": sched,
        "mult_write_5m": MULT_WRITE_5M, "mult_write_1h": MULT_WRITE_1H,
        "mult_read": MULT_READ,
        "started": datetime.now(timezone.utc).isoformat(),
    }) + "\n")
    out.flush()
    print(f"\nrun_id {run_id}  ->  {path}")

    # Monotonic clock: immune to NTP steps, DST, and manual clock changes, any
    # of which would silently shift a TTL experiment's meaning.
    mono0 = time.monotonic()
    ABORT_LATENESS = 30.0

    for label, offset, exp5, exp1 in sched:
        wait = offset - (time.monotonic() - mono0)
        if wait > 0:
            print(f"\nwaiting {wait / 60:.1f} min -> {label} (t={offset}s)")
            time.sleep(wait)
        actual = time.monotonic() - mono0
        lateness = actual - offset
        # Abort rather than continue: a probe fired minutes late (suspended
        # laptop, long API call) is measuring a different gap than the one
        # described, and the whole comparison would be quietly invalid.
        if lateness > ABORT_LATENESS:
            out.close()
            sys.exit(f"\nABORT: {label} would fire {lateness:.0f}s late "
                     f"(tolerance {ABORT_LATENESS:.0f}s). The machine likely "
                     f"suspended.\nPartial results kept at {path}; re-run rather "
                     f"than trusting a drifted schedule.")
        print(f"\n{label} (t={offset}s, actual t={actual:.1f}s)")
        for ttl, exp in (("5m", exp5), ("1h", exp1)):
            rec = call(api_key, args.model, prefixes[ttl], ttl, label, exp)
            rec["planned_offset"] = offset
            rec["actual_offset"] = round(time.monotonic() - mono0, 3)
            out.write(json.dumps(rec) + "\n")
            out.flush()
            if "error" in rec:
                print(f"  [{ttl}] ERROR {rec['error'][:90]}")
            else:
                v = verdict(rec)
                flag = "" if v == exp else f"  <-- expected {exp}"
                print(f"  [{ttl}] {v:<8} read={rec['cache_read_input_tokens']:>6} "
                      f"write={rec['cache_creation_input_tokens']:>6} "
                      f"uncached={rec['input_tokens']:>5}{flag}")

    out.close()
    report(run_id)


if __name__ == "__main__":
    main()
