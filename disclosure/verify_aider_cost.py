"""Reproduce the Aider cost double-count, arithmetic first, network optional.

The claim: `Coder.calculate_and_show_tokens_and_cost` charges every cached
token twice -- once at its cache rate, and again at the full input rate inside
`prompt_tokens` -- because it assumes `usage.prompt_tokens` is the uncached
remainder. It is not. LiteLLM reports it as the total, cache included.

    python3 disclosure/verify_aider_cost.py           # arithmetic only
    python3 disclosure/verify_aider_cost.py --live    # + one real API call

The arithmetic path takes no key, no network and no aider install, so a
maintainer can run it in five seconds. `--live` re-establishes the premise
against whatever LiteLLM is installed, and costs about a tenth of a cent.

Observed values below come from two independent runs recorded in
`tier-b/evidence/litellm-marker-survival.json` (litellm 1.83.9, 2026-07-31,
claude-haiku-4-5) and a re-run on 2026-08-03. Both show `prompt_tokens` holding
constant across an uncached call, a cache write and a cache read, which is the
whole premise.
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


# (label, prompt_tokens, cache_creation, cache_read) as reported by litellm.
OBSERVED = [
    ("uncached",                5415,    0,    0),
    ("cache write",             5415, 5402,    0),
    ("cache read",              5415,    0, 5402),
    # 2026-07-31 run, same shape at a different prefix size
    ("cache read (31 Jul run)", 15635,   0, 15624),
]

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
    if "--live" in sys.argv:
        print("\n" + "=" * 62 + "\n")
        rc |= live()
    raise SystemExit(rc)
