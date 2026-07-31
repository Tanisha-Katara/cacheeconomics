"""Does a cache marker survive LiteLLM's translation to the Anthropic wire?

`plugin.litellm_handler` can sit in a LiteLLM proxy and put `cache_control` on
outgoing requests. That half has always defaulted to off, with this written in
the module docstring:

    `mutate=True` puts `cache_control` on the wire, and that half is not
    verified. Nothing in this repository has watched a real LiteLLM proxy
    forward one of these requests.

The specific worry is that LiteLLM normalises Anthropic-shaped bodies through an
OpenAI-shaped intermediate. If `cache_control` is dropped in that translation,
the plugin places markers that never arrive, the provider caches nothing, and
the plugin's own effectiveness counters report placements as if they landed. The
failure is silent and flattering, which is the worst combination.

The provider settles it. A marker that arrives produces
`cache_creation_input_tokens > 0` on the response; a marker that was stripped
produces zero. No inference required.

Four calls, in order:

    1. control      no marker            expect creation == 0
    2. marked       cache_control        expect creation > 0
    3. reuse        same body, marked    expect read > 0
    4. via plugin   through the handler  expect creation > 0

Step 3 matters as much as step 2. A write nothing can read is worse than no
cache: it bills at 1.25x and returns nothing. Step 4 checks the actual code
path a proxy would run rather than a hand-built request that merely resembles
it.

The prefix is sized past the model's minimum on purpose. Below it the provider
caches nothing, returns no error, and reports zero -- which is exactly what a
stripped marker looks like, so a short prompt cannot tell the two apart.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python3 tier-b/litellm_marker_survival.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "harness"))

MODEL = "claude-haiku-4-5"
MIN_TOKENS = 4096          # registry: claude-haiku-4-5 on anthropic/direct


# Unique per run, so every run starts against a cold cache.
#
# Without this the experiment is not repeatable and reads as a failure on the
# second attempt: a 5m entry written by the previous run is still alive, so the
# *marked* call comes back `creation=0, read=13,424` and the verdict cannot tell
# a stripped marker from a warm hit. Measured exactly that way once.
RUN_NONCE = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")


def prefix_text() -> str:
    """Comfortably past the minimum, stable within a run, cold across runs."""
    para = (f"Operational policy {RUN_NONCE} for the shipment desk. Every reply "
            "states the shipment id, the current status, the last confirmed scan "
            "and the revised estimate where one exists. Never promise a delivery "
            "time the routing system has not confirmed. Never speculate about a "
            "cause that has not been recorded against the shipment. ")
    return para * 220          # ~8k tokens, twice the 4,096 minimum


def usage_of(response) -> dict:
    u = getattr(response, "usage", None) or {}
    get = (lambda k: getattr(u, k, None)) if not isinstance(u, dict) else u.get
    out = {"prompt_tokens": get("prompt_tokens"),
           "completion_tokens": get("completion_tokens")}
    details = get("prompt_tokens_details")
    if details is not None:
        dget = (lambda k: getattr(details, k, None)) if not isinstance(details, dict) \
            else details.get
        out["cached_tokens"] = dget("cached_tokens")
        out["cache_creation_tokens"] = (dget("cache_creation_tokens")
                                        or dget("cache_write_tokens"))
    # Anthropic's own names, if LiteLLM passed the raw object through.
    for k in ("cache_creation_input_tokens", "cache_read_input_tokens"):
        v = get(k)
        if v is not None:
            out[k] = v
    return {k: v for k, v in out.items() if v is not None}


def creation(u: dict) -> int:
    return int(u.get("cache_creation_input_tokens")
               or u.get("cache_creation_tokens") or 0)


def reads(u: dict) -> int:
    return int(u.get("cache_read_input_tokens") or u.get("cached_tokens") or 0)


def body(marked: bool) -> dict:
    block = {"type": "text", "text": prefix_text()}
    if marked:
        block["cache_control"] = {"type": "ephemeral"}
    return {"model": MODEL,
            "messages": [{"role": "user", "content": [
                block, {"type": "text", "text": "Status of shipment 88431?"}]}],
            "max_tokens": 16}


def call(litellm, request: dict, tag: str) -> dict:
    started = datetime.now(timezone.utc)
    r = litellm.completion(**request)
    u = usage_of(r)
    print(f"  {tag:26} creation={creation(u):>6,}  read={reads(u):>6,}  "
          f"prompt={u.get('prompt_tokens', 0):>6,}")
    return {"tag": tag, "at": started.isoformat(), "usage": u,
            "creation": creation(u), "read": reads(u)}


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        return 2
    import litellm
    litellm.drop_params = False

    print(f"  model {MODEL}, prefix sized past the {MIN_TOKENS:,}-token minimum\n")
    rows = []

    rows.append(call(litellm, body(marked=False), "1 control, no marker"))
    rows.append(call(litellm, body(marked=True), "2 marked"))
    time.sleep(3)                       # let the entry become visible
    rows.append(call(litellm, body(marked=True), "3 marked again (reuse)"))

    # 4: the actual plugin path a proxy would run.
    import asyncio

    from cacheeconomics import plugin
    p = plugin.CachePlugin(key=b"k" * 32, warmup=0)
    handler = plugin.litellm_handler(p, mutate=True)

    async def through_plugin():
        # Spaced on purpose. The handler timestamps with `now()`, and a tight
        # loop is a *fan-out*, not a conversation: a written entry takes
        # `tiers.WRITE_LATENCY_SECONDS` to become readable, so requests arriving
        # faster than that all pay the write premium and none of them read.
        # The allocator refuses that correctly, and a first version of this
        # script fired twelve calls instantly and concluded the plugin was
        # broken. It was measuring the wrong workload.
        from cacheeconomics import tiers
        gap = tiers.WRITE_LATENCY_SECONDS * 3
        out = None
        for i in range(4):
            data = body(marked=False)
            data["litellm_call_id"] = f"survival-4-{i}"
            out = await handler.async_pre_call_hook(None, None, data, "completion")
            if i < 3:
                time.sleep(gap)
        placed = json.dumps(out).count('"cache_control"')
        print(f"  {'4 via plugin':26} markers the plugin placed: {placed} "
              f"(after 4 requests {gap:.0f}s apart)")
        if not placed:
            print("     plugin placed nothing; nothing to test on the wire")
            return None
        return call(litellm, out, "4 via plugin, sent")

    got = asyncio.run(through_plugin())
    if got:
        rows.append(got)

    print()
    control, marked = rows[0], rows[1]
    # A marker that arrived either wrote an entry or read one it had written.
    # Only the unmarked control is required to have done neither -- that is what
    # says the write came from the marker rather than from anything else.
    survived = (marked["creation"] > 0 or marked["read"] > 0) and \
        control["creation"] == 0 and control["read"] == 0
    reusable = len(rows) > 2 and rows[2]["read"] > 0
    print(f"  marker survives LiteLLM translation : {'YES' if survived else 'NO'}")
    print(f"  written entry is readable           : {'YES' if reusable else 'NO'}")
    if len(rows) > 3:
        print(f"  plugin path also writes             : "
              f"{'YES' if rows[3]['creation'] > 0 else 'NO'}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "evidence", "litellm-marker-survival.json")
    with open(out, "w") as f:
        json.dump({"artifact": "litellm-marker-survival", "artifact_version": 1,
                   "litellm": _litellm_version(), "model": MODEL,
                   "marker_survives": survived, "entry_readable": reusable,
                   "calls": rows}, f, indent=2)
    print(f"  evidence -> {out}")
    return 0 if survived else 1


def _litellm_version() -> str:
    try:
        from importlib.metadata import version
        return version("litellm")
    except Exception:                                          # noqa: BLE001
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
