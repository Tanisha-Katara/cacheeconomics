"""Does the INFERRED tier's token split survive contact with a real tokenizer?

The INFERRED tier segments a request body a gateway logged, then has to say how
many tokens each segment is worth. It cannot count them: it measures each
segment's *bytes*, and divides the provider's billed input total in proportion.
`segment._scale_to_measured` is the whole of it.

Every structural finding on that tier is costed from those proportions. VOL-1
says "this many tokens sit behind the volatile block"; MIN-1 says "your prefix
is under the minimum"; the bake-off replays segment sizes against a modelled
cache. If the byte-share split is wrong, all of them are wrong by the same
factor, quietly, with no counter anywhere disagreeing.

There is a specific reason to doubt it. Bytes per token is not constant across
content. English prose runs near 4 bytes/token; JSON tool definitions are dense
with punctuation and short keys and run lower; a base64 blob or a long
identifier runs lower still. A prompt whose segments are *different kinds of
content* therefore has a different bytes-to-token ratio in each one, and a split
by byte share hands the dense segments too few tokens and the prose too many.

What is deliberately not tested here: segment *identity*. The recorder and the
body loader call the same `segments_from_request`, so on an unmodified body
their ids match by construction and `score_alignment` returns 1.0. Reporting
that as validation would be measuring a shared function against itself.

Method. Build request bodies whose segments are deliberately heterogeneous.
Ask the provider's own `count_tokens` endpoint for the cumulative token count at
each segment boundary; the differences are the real per-segment token counts.
Then run the same body through the INFERRED estimator with the real total, and
compare. `count_tokens` is free, so this costs nothing to re-run.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python3 tier-b/inferred_validation.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "harness"))

from cacheeconomics.segment import _scale_to_measured, segments_from_request  # noqa: E402

API = "https://api.anthropic.com/v1/messages/count_tokens"
MODEL = "claude-haiku-4-5"
KEY = b"validation-key-not-a-secret-000"


def count_tokens(key: str, body: dict) -> int:
    payload = dict(body)
    payload["model"] = MODEL
    payload.pop("cache_control", None)
    req = urllib.request.Request(
        API, data=json.dumps(payload).encode(),
        headers={"content-type": "application/json", "x-api-key": key,
                 "anthropic-version": "2023-06-01"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)["input_tokens"]
        except urllib.error.HTTPError as e:
            if e.code in (429, 529) and attempt < 3:
                time.sleep(2 ** attempt)
                continue
            raise
    raise RuntimeError("unreachable")


# --- the bodies -------------------------------------------------------------
#
# Each is shaped like real agent traffic and each mixes content *kinds* on
# purpose, because a uniform prompt cannot expose a proportional-split error:
# if every segment has the same bytes-per-token ratio, splitting by bytes is
# exactly right and the test passes for the wrong reason.

def _tool(name: str, n_props: int) -> dict:
    return {"name": name,
            "description": f"Operate on {name} records and return the result.",
            "input_schema": {"type": "object", "properties": {
                f"field_{i}": {"type": "string",
                               "description": f"The {i}th parameter for {name}."}
                for i in range(n_props)}, "required": ["field_0"]}}


PROSE = (
    "You are an operations assistant for a logistics company. You answer "
    "questions about shipments, routes and delivery windows. When a customer "
    "asks about a delay you explain the cause in plain language and give a "
    "revised estimate. You never speculate about causes you have not been told "
    "about, and you never promise a delivery time the routing system has not "
    "confirmed. If a question falls outside shipments you say so and stop. ")


def bodies() -> list:
    return [
        ("prose-heavy system, no tools", {
            "system": [{"type": "text", "text": PROSE * 12}],
            "messages": [{"role": "user", "content": "Where is order 88431?"}]}),
        ("dense JSON tools beside prose", {
            "tools": [_tool("shipment", 12), _tool("route", 12), _tool("invoice", 8)],
            "system": [{"type": "text", "text": PROSE * 8}],
            "messages": [{"role": "user", "content": "Reroute 88431 via Leeds."}]}),
        ("tools dominate", {
            "tools": [_tool(f"svc_{i}", 16) for i in range(6)],
            "system": [{"type": "text", "text": PROSE * 2}],
            "messages": [{"role": "user", "content": "List the tools you have."}]}),
        ("long identifiers, low bytes-per-token", {
            "system": [{"type": "text", "text": PROSE * 4},
                       {"type": "text",
                        "text": "Known ids: " + " ".join(
                            f"SHPMT-{i:06d}-AX{i%97:02d}-QQ" for i in range(220))}],
            "messages": [{"role": "user", "content": "Is SHPMT-000042 delayed?"}]}),
        ("multi-turn with tool results", {
            "tools": [_tool("shipment", 10)],
            "system": [{"type": "text", "text": PROSE * 6}],
            "messages": [
                {"role": "user", "content": "Status of 88431?"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "Checking the shipment record."}]},
                {"role": "user", "content": [
                    {"type": "text",
                     "text": json.dumps({"id": "88431", "status": "in_transit",
                                         "eta": "2026-08-02T14:00:00Z",
                                         "stops": [f"hub-{i}" for i in range(30)]})}]},
                {"role": "user", "content": "So when does it arrive?"}]}),
        ("tiny prompt, near the minimum", {
            "system": [{"type": "text", "text": PROSE}],
            "messages": [{"role": "user", "content": "Hi"}]}),
    ]


# `messages` is a required field, so a prefix consisting only of tools or system
# is not a countable request. Every cut therefore carries this sentinel, whose
# cost is identical in every call and so cancels out of the differences.
_SENTINEL = {"role": "user", "content": "."}


def measure(key: str, label: str, body: dict) -> dict:
    """Real per-segment tokens, by asking for cumulative counts at each boundary."""
    segs = segments_from_request(body, KEY)

    # Cumulative real counts. Truncating at a segment boundary and counting
    # gives the tokens up to that point; consecutive differences are the real
    # per-segment values, with the sentinel and the fixed request overhead
    # subtracting away.
    counts = [count_tokens(key, _with_sentinel(_truncate(body, 0)))]
    for cut in range(1, len(segs) + 1):
        counts.append(count_tokens(key, _with_sentinel(_truncate(body, cut))))
    cumulative = [counts[i + 1] - counts[i] for i in range(len(segs))]

    # The figure the estimator is handed is the real billed input for the body
    # as sent, without the sentinel.
    real_total = count_tokens(key, body)

    est = _scale_to_measured([dict(s) for s in segs], real_total)

    rows = []
    for s, real in zip(est, cumulative):
        rows.append({"index": s["index"], "role": s["role"], "label": s["label"],
                     "estimated_tokens": s["tokens"], "real_tokens": real})
    return {"case": label, "real_total": real_total,
            "estimated_total": sum(r["estimated_tokens"] for r in rows),
            "segments": rows}


def _with_sentinel(body: dict) -> dict:
    out = dict(body)
    out["messages"] = list(out.get("messages") or []) + [_SENTINEL]
    return out


def _truncate(body: dict, keep: int) -> dict:
    """The body containing only its first `keep` wire segments.

    Rebuilt in wire order -- tools, then system, then messages -- because that
    is the order `segments_from_request` indexes in, and a cut that did not
    match it would compare a prefix against the wrong segment.
    """
    out: dict = {}
    seen = 0
    for tool in body.get("tools") or []:
        if seen >= keep:
            return out
        out.setdefault("tools", []).append(tool)
        seen += 1
    system = body.get("system")
    if isinstance(system, list):
        for block in system:
            if seen >= keep:
                return out
            out.setdefault("system", []).append(block)
            seen += 1
    for m in body.get("messages") or []:
        content = m.get("content")
        blocks = content if isinstance(content, list) else [content]
        kept = []
        for b in blocks:
            if seen >= keep:
                break
            kept.append(b)
            seen += 1
        if kept:
            out.setdefault("messages", []).append(
                {"role": m.get("role", "user"),
                 "content": kept if isinstance(content, list) else kept[0]})
        if seen >= keep:
            return out
    return out


def main() -> int:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        print("ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        return 2

    results = [measure(key, label, body) for label, body in bodies()]

    print(f"{'case':34} {'seg':>4} {'real':>7} {'est':>7} {'err':>8}")
    print("-" * 64)
    worst = 0.0
    errors = []
    for r in results:
        for s in r["segments"]:
            real, est = s["real_tokens"], s["estimated_tokens"]
            if real <= 0:
                continue
            err = (est - real) / real
            errors.append(abs(err))
            worst = max(worst, abs(err))
            print(f"{r['case'][:34]:34} {s['index']:>4} {real:>7,} {est:>7,} "
                  f"{err:>+7.1%}")
    print("-" * 64)
    errors.sort()
    print(f"  segments compared : {len(errors)}")
    print(f"  median abs error  : {errors[len(errors)//2]:.1%}")
    print(f"  90th percentile   : {errors[int(len(errors)*0.9)]:.1%}")
    print(f"  worst             : {worst:.1%}")
    print()
    print("  Totals reconcile by construction -- the estimator is handed the real")
    print("  total and divides it. What is measured here is the *split*.")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "evidence", "inferred-token-split.json")
    with open(out, "w") as f:
        json.dump({"artifact": "inferred-token-split", "artifact_version": 1,
                   "model": MODEL, "method": "count_tokens at segment boundaries",
                   "cases": results}, f, indent=2)
    print(f"  evidence -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
