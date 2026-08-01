"""Run a real agent on a schedule, and capture what it costs.

The interactive run of browser-use settled one thing and left the interesting
question open. Median gap 7.3 seconds, zero of 26 gaps in the 5m-1h band: at
that cadence the five-minute cache never expires, so the one-hour TTL the
disclosures are about would cost more than it saves. That is a real answer for a
tight automation loop and says nothing about the workload the hypothesis was
actually written for.

A scheduled agent is that workload. Monitoring a page every few minutes, a
nightly scrape, a periodic availability check: the prefix is stable across runs,
the gap between runs is minutes rather than seconds, and the five-minute entry
is dead by the time the next run starts. That is the band where a 1h write at 2x
beats repeated 5m writes at 1.25x each.

**The interval here is chosen, not observed, and that changes what may be
claimed.** Ten minutes is a plausible monitoring cadence and it sits inside the
band by construction. So the honest output of this script is not "scheduled
agents save X%" -- it is "this workload, at this interval, would have saved X%",
plus the break-even gap at which the answer flips. Anyone running the same agent
on a ninety-second timer gets the opposite result, and the evidence file says so.

What is measured rather than assumed: the prefix size, how much of it is stable
across runs, what the provider actually billed, and therefore where break-even
falls. Those are properties of the agent's prompts. Only the interval is a
choice.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python3 tier-b/capture_proxy.py --out scheduled.jsonl --port 8788 &
    python3 tier-b/scheduled_agent_run.py --cycles 7 --interval 600
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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--cycles", type=int, default=7)
    p.add_argument("--interval", type=float, default=600.0,
                   help="seconds between the START of each cycle. Chosen, not "
                        "observed: state it in anything published from this run")
    p.add_argument("--python", default=os.environ.get("AGENT_PYTHON", sys.executable),
                   help="interpreter with browser-use installed")
    p.add_argument("--base-url", default="http://127.0.0.1:8788")
    args = p.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        return 2

    runner = os.path.join(HERE, "_scheduled_runner.py")
    with open(runner, "w") as f:
        f.write(RUNNER)

    env = dict(os.environ, CAPTURE_BASE_URL=args.base_url)
    started = datetime.now(timezone.utc)
    print(f"  {args.cycles} cycles, {args.interval:.0f}s apart "
          f"(~{args.cycles * args.interval / 60:.0f} min total)", file=sys.stderr)
    print(f"  interval is a CHOICE: {args.interval:.0f}s sits inside the "
          f"300-3600s band by construction", file=sys.stderr)

    for i in range(args.cycles):
        t0 = time.time()
        print(f"\n  cycle {i + 1}/{args.cycles} at "
              f"{datetime.now(timezone.utc).strftime('%H:%M:%S')}", file=sys.stderr)
        try:
            subprocess.run([args.python, runner], env=env, timeout=300,
                           capture_output=True)
        except subprocess.TimeoutExpired:
            print("    cycle timed out", file=sys.stderr)
        if i < args.cycles - 1:
            wait = max(0.0, args.interval - (time.time() - t0))
            time.sleep(wait)

    os.unlink(runner)
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    print(f"\n  done in {elapsed / 60:.1f} min", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
