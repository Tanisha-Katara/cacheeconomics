"""Reproduce the CrewAI `total_tokens` undercount from committed evidence.

The claim: on the native Anthropic provider, `total_tokens` is
`input_tokens + output_tokens`, and Anthropic reports `input_tokens` as the
uncached remainder. So a cached request reports a total that omits every token
it actually paid for.

    python3 disclosure/verify_crewai_tokens.py

No key, no network, no crewAI install. Numbers are read from
`tier-b/evidence/prompt-tokens-semantics.json` rather than transcribed, so this
file cannot drift from the evidence it cites.

This existed only as a claim in a draft until a review pointed out that the
repository shipped no artifact and no verifier for the direct-API measurement
behind a public 400x number. That was a fair objection to a disclosure filed
against a named project, and this is the answer to it.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EVIDENCE = os.path.join(os.path.dirname(HERE), "tier-b", "evidence",
                        "prompt-tokens-semantics.json")


# lib/crewai/src/crewai/llms/providers/anthropic/completion.py:1977, transcribed
# so the reader can see exactly what is being checked without a crewAI install.
def crewai_total(input_tokens, output_tokens, cache_read, cache_creation):
    return input_tokens + output_tokens


def actual_total(input_tokens, output_tokens, cache_read, cache_creation):
    """Anthropic bills all three input classes. `input_tokens` is only the
    uncached remainder."""
    return input_tokens + cache_read + cache_creation + output_tokens


def main() -> int:
    if not os.path.exists(EVIDENCE):
        print(f"missing evidence artifact: {EVIDENCE}", file=sys.stderr)
        return 1
    doc = json.load(open(EVIDENCE))
    raw = [c for c in doc["calls"] if c["path"] == "anthropic-sdk"]
    if not raw:
        print("artifact carries no direct Anthropic SDK calls", file=sys.stderr)
        return 1

    print(f"evidence: {os.path.relpath(EVIDENCE, os.path.dirname(HERE))}")
    print(f"recorded: {doc['recorded_at']}  model {doc['model']}  "
          f"anthropic-sdk {doc['anthropic_sdk']}\n")
    print(f"  {'call':<14}{'input':>8}{'read':>8}{'write':>8}{'out':>6}"
          f"{'crewAI':>9}{'actual':>9}{'ratio':>8}")
    print("  " + "-" * 70)

    worst = 1.0
    for c in raw:
        u = c["usage"]
        a = crewai_total(u["input_tokens"], u["output_tokens"],
                         u["cache_read_input_tokens"],
                         u["cache_creation_input_tokens"])
        b = actual_total(u["input_tokens"], u["output_tokens"],
                         u["cache_read_input_tokens"],
                         u["cache_creation_input_tokens"])
        ratio = b / a if a else float("inf")
        worst = max(worst, ratio)
        print(f"  {c['tag']:<14}{u['input_tokens']:>8}"
              f"{u['cache_read_input_tokens']:>8}"
              f"{u['cache_creation_input_tokens']:>8}{u['output_tokens']:>6}"
              f"{a:>9}{b:>9}{ratio:>7.0f}x")

    print()
    print("  The uncached call agrees. On a cached one crewAI reports the")
    print(f"  uncached remainder plus output only, understating by up to {worst:.0f}x.")
    print()
    print("RESULT:", "reproduced" if worst > 2 else "NOT reproduced -- claim fails")
    return 0 if worst > 2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
