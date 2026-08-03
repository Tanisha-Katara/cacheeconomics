I got the main claim wrong.

`compute_costs_from_tokens` is not what produces the printed cost in the normal
case. `base_coder.py:2036-2043` tries LiteLLM's own calculator first, and only
falls back to the token math when that comes back empty:

```python
try:
    cost = litellm.completion_cost(completion_response=completion)
except Exception:
    cost = 0

if not cost:
    cost = self.compute_costs_from_tokens(...)
```

I read the fallback and never read its caller.

So I went and measured whether the fallback gets reached at all. claude-haiku-4-5
through litellm 1.83.9, cached prefix:

| | `completion_cost()` | usage | fallback |
|---|---|---|---|
| non-streaming | $0.0068015 | write 5,402 | not reached |
| streaming | $0.0005892 | read 5,402 | not reached |

Both numbers are also correct. Hand-computing the write gives $0.0067615 plus
output, and the read $0.000549 plus output. LiteLLM prices the cache classes
properly and aider uses its answer.

The two defects in `compute_costs_from_tokens` are still real, and the
arithmetic in `verify_aider_cost.py` still holds. The Anthropic branch adds
`prompt_tokens` on top of the two cache classes that `prompt_tokens` already
contains, which is 10.9x on a fully cached call. The deepseek branch subtracts a
price from a token count, so `(prompt_tokens - 2.8e-08)` changes nothing and the
whole prompt gets charged at full rate on top of the cache-hit charge.

But they only run when `completion_cost` returns nothing or throws, which means
an uncosted model or an exception. That is a much smaller bug than the one I
described, and it does not warrant a post the length of the one I wrote.

Close this, relabel it as a latent fallback bug, or take just the deepseek line,
which is wrong whether or not it executes. Whichever you prefer.

Sorry for the noise. The reproduction I attached checked my arithmetic and never
checked my premise.
