"""Reproduce the Aider cost double-count, arithmetic first, network optional.

The claim: `Coder.calculate_and_show_tokens_and_cost` charges every cached
token twice -- once at its cache rate, and again at the full input rate inside
`prompt_tokens` -- because it assumes `usage.prompt_tokens` is the uncached
remainder. It is not. LiteLLM reports it as the total, cache included.

    python3 disclosure/verify_aider_cost.py           # arithmetic only
    python3 disclosure/verify_aider_cost.py --live    # + THREE real API calls

The arithmetic path takes no key, no network and no aider install, so a
maintainer can run it in five seconds. `--live` makes three calls -- uncached,
cache write, cache read -- because one call cannot show that `prompt_tokens`
stays constant across them, which is the whole premise. Under a cent at
claude-haiku-4-5 rates.

Every value in OBSERVED is read from
`tier-b/evidence/prompt-tokens-semantics.json`, recorded 2026-08-03 against
litellm 1.83.9 and anthropic 0.120.2, rather than transcribed by hand. An
earlier version hard-coded numbers from a session that was never committed,
which a review correctly called a reproducibility gap in a public disclosure.
`tier-b/evidence/litellm-marker-survival.json` (2026-07-31) shows the same
behaviour independently at a different prefix size.
"""

import sys

# aider/coders/base_coder.py @ 5dc9490, lines 2089-2098. Transcribed rather
# than imported so this runs without installing aider, and so the reader can
# see exactly what is being checked.
def aider_cost(prompt_tokens, cache_write, cache_read, rate, hit_rate=0.0):
    cost = 0.0
    if hit_rate:                                    # "must be deepseek"
        cost += hit_rate * cache_read
        # Subtracting a price from a token count. Almost certainly meant
        # `prompt_tokens - cache_read`.
        cost += (prompt_tokens - hit_rate) * rate
    else:
        cost += cache_write * rate * 1.25
        cost += cache_read * rate * 0.10
        cost += prompt_tokens * rate                # <-- already contains both
    return cost


def correct_cost(prompt_tokens, cache_write, cache_read, rate, hit_rate=0.0):
    """`prompt_tokens` is the total. The uncached part is what is left."""
    uncached = prompt_tokens - cache_write - cache_read
    if hit_rate:
        return hit_rate * cache_read + uncached * rate
    return cache_write * rate * 1.25 + cache_read * rate * 0.10 + uncached * rate


def _observed():
    """(label, prompt_tokens, cache_creation, cache_read), read from the
    committed artifact so the numbers in this file cannot drift from the
    evidence it cites."""
    import json
    import os
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rows = []
    for name, extra in (("prompt-tokens-semantics.json", ""),
                        ("litellm-marker-survival.json", " (31 Jul run)")):
        path = os.path.join(here, "tier-b", "evidence", name)
        if not os.path.exists(path):
            continue
        for c in json.load(open(path))["calls"]:
            if c.get("path") not in (None, "litellm"):
                continue
            u = c["usage"]
            if "prompt_tokens" not in u:
                continue
            rows.append((c["tag"] + extra, u["prompt_tokens"],
                         u.get("cache_creation_input_tokens", 0) or 0,
                         u.get("cache_read_input_tokens", 0) or 0))
    return rows


OBSERVED = _observed()

RATE = 1e-6            # claude-haiku-4-5 input, from litellm.model_cost


def arithmetic() -> int:
    print("Anthropic branch (input_cost_per_token_cache_hit is None for every")
    print("Claude model in litellm.model_cost, so this is the branch taken)\n")
    print(f"  {'call':<24}{'aider $':>12}{'correct $':>12}{'overstated':>12}")
    print("  " + "-" * 60)
    worst = 1.0
    for label, pt, cw, cr in OBSERVED:
        a = aider_cost(pt, cw, cr, RATE)
        c = correct_cost(pt, cw, cr, RATE)
        ratio = a / c if c else float("inf")
        worst = max(worst, ratio)
        print(f"  {label:<24}{a:>12.7f}{c:>12.7f}{ratio:>11.1f}x")
    print()
    print("  The uncached call agrees, because there is nothing to double-count.")
    print("  The error grows with how well the cache is working.\n")

    print("deepseek branch (input_cost_per_token_cache_hit = 2.8e-08)\n")
    ds_rate, ds_hit = 2.8e-07, 2.8e-08
    pt, cr = 10_000, 9_000
    a = aider_cost(pt, 0, cr, ds_rate, hit_rate=ds_hit)
    c = correct_cost(pt, 0, cr, ds_rate, hit_rate=ds_hit)
    print(f"  prompt_tokens={pt:,} of which {cr:,} were cache hits")
    print(f"  aider ${a:.7f}   correct ${c:.7f}   overstated {a / c:.1f}x")
    print(f"  `prompt_tokens - {ds_hit}` leaves {pt - ds_hit:.8f}, i.e. the "
          f"subtraction does nothing.\n")

    ok = worst > 1.05
    print("RESULT:", "reproduced" if ok else "NOT reproduced -- claim fails")
    return 0 if ok else 1


GUARD_URL = ("https://raw.githubusercontent.com/Aider-AI/aider/main/"
             "aider/coders/base_coder.py")


def premise() -> int:
    """Does the arithmetic above actually run?

    The original filing said `compute_costs_from_tokens` produces the printed
    cost. It does not: the caller tries LiteLLM's own calculator first and falls
    back only when that comes back empty. That was retracted publicly in
    `correction-aider-5516.md`, whose closing line is the reason this function
    exists -- "the reproduction I attached checked my arithmetic and never
    checked my premise."

    So the premise is checked, from upstream, rather than asserted in prose. A
    verifier that replays arithmetic and stops is the same artifact that caused
    the retraction.
    """
    import urllib.request
    print("Premise: is that arithmetic on the path a normal call takes?\n")
    try:
        src = urllib.request.urlopen(GUARD_URL, timeout=30).read().decode()
    except Exception as e:                                     # noqa: BLE001
        print(f"  could not fetch base_coder.py ({e.__class__.__name__}).")
        print("  PREMISE UNCHECKED -- offline. The arithmetic above still holds;")
        print("  whether it executes is exactly what this could not confirm, so")
        print("  do not read the result above as the original claim.")
        return 0

    lines = src.splitlines()
    call = guard = fallback = None
    for i, line in enumerate(lines, 1):
        if "litellm.completion_cost(" in line and call is None:
            call = i
        elif call and guard is None and line.strip() == "if not cost:":
            guard = i
        elif guard and fallback is None and "compute_costs_from_tokens(" in line:
            fallback = i
            break

    if not (call and guard and fallback):
        print("  the guard is no longer where it was: completion_cost="
              f"{call}, `if not cost:`={guard}, fallback={fallback}")
        print("  PREMISE CHANGED -- re-read the caller before citing this.")
        return 1

    print(f"  base_coder.py:{call}  cost = litellm.completion_cost(...)")
    print(f"  base_coder.py:{guard}  if not cost:")
    print(f"  base_coder.py:{fallback}      cost = self.compute_costs_from_tokens(...)\n")
    print("  So the fallback runs only when completion_cost returns falsy or")
    print("  raises -- an uncosted model, or an exception. Measured on")
    print("  claude-haiku-4-5 through litellm 1.83.9, completion_cost returned")
    print("  $0.0068015 non-streaming and $0.0005892 streaming, both correct,")
    print("  so the fallback was not reached in either.\n")
    print("SCOPE: latent. The two defects above are real and unretracted, and")
    print("       they do not produce the cost aider prints in the normal case.")
    print("       The original headline claim that they do was withdrawn in")
    print("       correction-aider-5516.md.")
    return 0


def live() -> int:
    """Re-establish the premise: prompt_tokens does not move when the same
    prompt goes from uncached, to a cache write, to a cache read."""
    try:
        import litellm
    except ImportError:
        print("litellm not installed; skipping live check", file=sys.stderr)
        return 0
    litellm.suppress_debug_info = True
    big = "You are a precise assistant. " * 900     # over haiku's 4096 minimum

    def msgs(mark):
        blk = {"type": "text", "text": big}
        if mark:
            blk["cache_control"] = {"type": "ephemeral"}
        return [{"role": "system", "content": [blk]},
                {"role": "user", "content": "Reply with the single word: ok"}]

    seen = []
    # Numbered rather than named. A warm entry from an earlier run turns the
    # second call into a read, and a label asserting "write" would then be
    # describing something that did not happen.
    for tag, mark in (("call 1 unmarked", False), ("call 2 marked", True),
                      ("call 3 marked", True)):
        u = litellm.completion(model="anthropic/claude-haiku-4-5",
                               messages=msgs(mark), max_tokens=8).usage
        cw = getattr(u, "cache_creation_input_tokens", 0) or 0
        cr = getattr(u, "cache_read_input_tokens", 0) or 0
        print(f"  {tag:<16} prompt_tokens={u.prompt_tokens:<8} write={cw:<8} "
              f"read={cr:<8} remainder={u.prompt_tokens - cw - cr}")
        seen.append(u.prompt_tokens)
    same = len(set(seen)) == 1
    print(f"\n  prompt_tokens constant across all three: {same}")
    print("  =>", "it is the total, cache included" if same
          else "PREMISE FAILED: it moved, so the claim needs revisiting")
    return 0 if same else 1


if __name__ == "__main__":
    rc = arithmetic()
    print("\n" + "=" * 62 + "\n")
    rc |= premise()
    if "--live" in sys.argv:
        print("\n" + "=" * 62 + "\n")
        rc |= live()
    raise SystemExit(rc)
