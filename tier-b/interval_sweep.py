"""Vary the schedule, measure where the one-hour TTL starts paying.

Two runs of browser-use gave opposite answers. Interactive, median gap 7.3s:
nothing to save, because a five-minute entry never expires between requests and
a one-hour write at 2x would only cost more. On a ten-minute timer: 17% of input
spend recoverable, because every cycle starts cold.

Both are correct and neither is a property of the software. They are properties
of how often it runs, and the ten-minute figure has an obvious weakness -- the
interval was chosen, and chosen from inside the band the finding is about.

A single chosen point is an anecdote. The relationship is the finding. So this
runs the same agent, the same task, at several intervals, and reports
recoverable share against cadence. Nobody has to trust the choice of any one
interval, because the curve shows what every interval does, including the ones
where the answer is zero.

What is expected, from the arithmetic rather than the data: below roughly five
minutes the five-minute entry is still alive when the next run starts, so there
is nothing to recover and a one-hour write would cost more. Past an hour both
lifetimes have expired and both pay full price. In between, a 1h write at 2x
replaces repeated 5m writes at 1.25x each, and the more often the prefix is
rewritten inside the hour the better that trade gets.

If the measurement disagrees with that shape, the measurement wins and the model
is wrong. That is the point of running it.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python3 tier-b/interval_sweep.py --out-dir sweep/
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))

# (interval seconds, cycles). Cycles fall as the interval rises so the whole
# sweep fits in an afternoon; each still yields enough inter-cycle gaps for the
# band fraction to mean something.
PLAN = [(120, 8), (420, 6), (900, 5)]

RUNNER = r'''
import asyncio, os, sys
from browser_use import Agent
from browser_use.llm.anthropic.chat import ChatAnthropic
llm = ChatAnthropic(model="claude-haiku-4-5",
                    api_key=os.environ["ANTHROPIC_API_KEY"],
                    base_url=os.environ["CAPTURE_BASE_URL"])
TASK = ("Go to https://en.wikipedia.org/wiki/Special:RecentChanges and report "
        "the title of the most recent change listed.")
async def main():
    a = Agent(task=TASK, llm=llm, max_actions_per_step=2)
    try:
        await asyncio.wait_for(a.run(max_steps=4), timeout=240)
    except Exception as e:
        print("cycle ended:", type(e).__name__, str(e)[:120], file=sys.stderr)
asyncio.run(main())
'''


def run_one(interval: float, cycles: int, out: str, python: str, port: int) -> dict:
    """One interval: start a capture proxy on its own file, run the cycles."""
    proxy = subprocess.Popen(
        [sys.executable, os.path.join(HERE, "capture_proxy.py"),
         "--out", out, "--port", str(port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)

    runner = os.path.join(HERE, f"_sweep_runner_{port}.py")
    with open(runner, "w") as f:
        f.write(RUNNER)
    env = dict(os.environ, CAPTURE_BASE_URL=f"http://127.0.0.1:{port}")

    started = datetime.now(timezone.utc)
    for i in range(cycles):
        t0 = time.time()
        print(f"    cycle {i + 1}/{cycles} at "
              f"{datetime.now(timezone.utc).strftime('%H:%M:%S')}", file=sys.stderr)
        try:
            subprocess.run([python, runner], env=env, timeout=300,
                           capture_output=True)
        except subprocess.TimeoutExpired:
            print("      timed out", file=sys.stderr)
        if i < cycles - 1:
            time.sleep(max(0.0, interval - (time.time() - t0)))

    proxy.terminate()
    try:
        proxy.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proxy.kill()
    os.unlink(runner)
    return {"interval_s": interval, "cycles": cycles, "capture": out,
            "started": started.isoformat(),
            "elapsed_min": (datetime.now(timezone.utc) - started).total_seconds() / 60}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--python", default=os.environ.get("AGENT_PYTHON", sys.executable))
    p.add_argument("--base-port", type=int, default=8800)
    args = p.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        return 2
    os.makedirs(args.out_dir, exist_ok=True)

    total = sum(i * (c - 1) for i, c in PLAN) / 60
    print(f"  {len(PLAN)} intervals, about {total:.0f} min of waiting plus run time",
          file=sys.stderr)

    runs = []
    for n, (interval, cycles) in enumerate(PLAN):
        out = os.path.join(args.out_dir, f"interval-{int(interval)}s.jsonl")
        print(f"\n  === {interval}s x {cycles} cycles ===", file=sys.stderr)
        runs.append(run_one(interval, cycles, out, args.python, args.base_port + n))
        with open(os.path.join(args.out_dir, "sweep.json"), "w") as f:
            json.dump({"artifact": "interval-sweep", "runs": runs}, f, indent=2)

    print(f"\n  sweep complete: {len(runs)} intervals", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
