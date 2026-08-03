Correction: my headline claim here is wrong, and I would rather say so than let it sit.

I wrote that `compute_costs_from_tokens` produces the number `Cost: $X message, $Y session` prints. It does not, in the normal case. `base_coder.py:2036-2043` tries LiteLLM's own calculator first and only falls back to the token math when that returns nothing:

```python
try:
    cost = litellm.completion_cost(completion_response=completion)
except Exception:
    cost = 0

if not cost:
    cost = self.compute_costs_from_tokens(...)
```

I had read the fallback and not the caller. That is my error and it is the substance of the issue, not a detail.

I then measured whether the fallback is actually reached, on `claude-haiku-4-5` through litellm 1.83.9, with a cached prefix:

| | `completion_cost()` | usage | fallback reached |
|---|---|---|---|
| non-streaming | `$0.0068015` | write 5,402 | no |
| streaming | `$0.0005892` | read 5,402 | no |

It returns a usable cost both times, and both values are right — `$0.0068015` against a hand-computed `$0.0067615` plus output for the write, and `$0.0005892` against `$0.000549` plus output for the read. LiteLLM prices the cache classes correctly and aider uses that number. So on these completions the buggy function never runs.

**What that leaves.** The two defects in `compute_costs_from_tokens` are still real — I stand behind the arithmetic, and `verify_aider_cost.py` still reproduces it:

- The Anthropic branch adds `prompt_tokens` on top of the two cache classes, and `prompt_tokens` already contains them. 10.9x on a fully-cached call.
- The deepseek branch computes `(prompt_tokens - input_cost_per_token_cache_hit)`, subtracting a price from a token count. `2.8e-08` leaves the count unchanged, so the whole prompt is charged at full rate on top of the cache-hit charge.

But they are **latent**, reached only when `litellm.completion_cost` returns falsy or raises — an uncosted model, or an exception. Not the everyday path I claimed.

That is a much smaller issue than the one I filed, and I no longer think the framing justifies the length of the original post. Happy for you to close this, relabel it as a latent fallback bug, or take just the deepseek unit error, which is a one-line fix and wrong regardless of whether it runs.

Apologies for the noise. I should have traced the caller before writing, and the reproduction I attached checked my arithmetic without checking my premise — which is the more useful lesson for me.
