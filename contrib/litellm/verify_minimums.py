"""Re-derive the proposed minimums from LiteLLM's own file, and check them.

Run this before sending anything to the issue. The comment quotes three counts
and a dependency between its two problems; all four come from here, so a stale
number in the comment is a credibility problem in someone else's repository.

    python3 contrib/litellm/verify_minimums.py            # fetches main
    python3 contrib/litellm/verify_minimums.py FILE.json  # against a local copy

Exits non-zero if anything the comment asserts no longer holds.
"""

import collections
import json
import os
import re
import sys
import urllib.request

UPSTREAM = ("https://raw.githubusercontent.com/BerriAI/litellm/main/"
            "model_prices_and_context_window.json")
HERE = os.path.dirname(os.path.abspath(__file__))
PROPOSED = os.path.join(HERE, "proposed-minimums.json")

# Anthropic's published minimum for Fable 5, which is what settles Problem 1.
# platform.claude.com/docs/en/build-with-claude/prompt-caching, checked 2026-07-28.
FABLE_5_MINIMUM = 512


def base_model(key: str) -> str:
    """Reduce a provider-prefixed key to a comparable model token.

    `us.anthropic.claude-haiku-4-5-v1:0`, `openrouter/anthropic/claude-haiku-4.5`
    and `claude-haiku-4-5` are the same model, and the whole argument for filling
    these in from within the file is that one key already knows the answer.
    """
    k = key.split("/")[-1]
    k = re.sub(r"^(us|eu|global|apac)\.", "", k)
    k = re.sub(r"^anthropic\.", "", k)
    k = k.split("@")[0]
    k = re.sub(r"-v\d+:\d+$", "", k)
    return k.replace(".", "-").replace("_", "-").lower()


def load(path_or_none):
    if path_or_none:
        with open(path_or_none) as f:
            return json.load(f)
    with urllib.request.urlopen(UPSTREAM, timeout=120) as r:
        return json.loads(r.read().decode())


def main(argv):
    d = load(argv[1] if len(argv) > 1 else None)
    entries = {k: v for k, v in d.items() if isinstance(v, dict)}
    claims = {k: v for k, v in entries.items()
              if v.get("supports_prompt_caching") is True}
    no_min = {k for k, v in claims.items()
              if v.get("prompt_cache_min_tokens") is None}

    print(f"entries claiming supports_prompt_caching : {len(claims):,}")
    print(f"  ...with no prompt_cache_min_tokens     : {len(no_min):,}")

    fable = {k: v.get("prompt_cache_min_tokens") for k, v in entries.items()
             if "fable-5" in k and v.get("supports_prompt_caching")}
    distinct = sorted({v for v in fable.values() if v is not None})
    print(f"\nclaude-fable-5 keys claiming caching      : {len(fable)}")
    print(f"  distinct minimums recorded             : {distinct}")
    for k, v in sorted(fable.items()):
        print(f"    {k:<46} {v}")

    known = collections.defaultdict(set)
    for k, v in entries.items():
        m = v.get("prompt_cache_min_tokens")
        if m is not None:
            known[base_model(k)].add(m)

    with open(PROPOSED) as f:
        proposed = json.load(f)

    unambiguous, contingent, problems = [], [], []
    for key, value in proposed.items():
        cur = entries.get(key)
        if cur is None:
            problems.append(f"{key}: no longer in the file")
            continue
        if cur.get("prompt_cache_min_tokens") is not None:
            problems.append(
                f"{key}: already filled upstream as "
                f"{cur['prompt_cache_min_tokens']}")
            continue
        evidence = known.get(base_model(key), set())
        if evidence == {value}:
            unambiguous.append(key)
        elif value in evidence:
            # The file disagrees with itself, so this key cannot be derived
            # until that is settled. Reporting it as derivable would be the
            # error the comment is warning against.
            contingent.append((key, sorted(evidence)))
        else:
            problems.append(
                f"{key}: proposed {value}, file's other keys say "
                f"{sorted(evidence) or 'nothing'}")

    print(f"\nproposed keys                            : {len(proposed)}")
    print(f"  still unfilled upstream                : "
          f"{len(unambiguous) + len(contingent)}")
    print(f"  derivable with no ambiguity            : {len(unambiguous)}")
    print(f"  contingent on Problem 1 (fable-5)      : {len(contingent)}")
    for key, evidence in contingent:
        print(f"    {key:<46} file says {evidence}")

    ok = True
    if problems:
        ok = False
        print("\nNEEDS ATTENTION before sending:")
        for p in problems:
            print(f"  {p}")
    if distinct and distinct != [FABLE_5_MINIMUM]:
        print(f"\nProblem 1 is still open: fable-5 carries {distinct}, and "
              f"{FABLE_5_MINIMUM} is the published value.")
    else:
        print("\nProblem 1 appears resolved upstream; re-read the comment "
              "before sending it.")
        ok = False

    print("\nOK to send." if ok else "\nDo not send unchanged.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
