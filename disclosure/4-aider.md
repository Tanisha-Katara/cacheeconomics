# Aider — cached tokens are billed twice in the cost display

**Repo:** `Aider-AI/aider` · 47,906★ · Apache-2.0
**Verified at:** `5dc9490bb35f9729ef2c95d00a19ccd30c26339c` (2026-05-22)
**Status:** not yet filed
**Reproduction:** `python3 disclosure/verify_aider_cost.py` — no key, no network,
no aider install. `--live` adds one real API call, about a tenth of a cent.

---

## The claim

`Coder.calculate_and_show_tokens_and_cost` charges every cached token twice:
once at its cache rate, and again at the full input rate inside `prompt_tokens`.
Both branches of the `if` do it, by different mechanisms. The number this
produces is what `Cost: $X message, $Y session` shows
(`base_coder.py:2046-2047`, `2059-2060`).

The error is zero on an uncached request and grows with how well the cache is
working, so it is largest for exactly the users who tuned for caching.

## The premise, which is where this starts

`base_coder.py:2085-2087` states the assumption in a comment:

```
# Anthropic
# cache_creation_input_tokens + cache_read_input_tokens + prompt
#    == total tokens that were
```

That reading of `usage.prompt_tokens` is not what LiteLLM reports. Same prompt,
three consecutive calls, `litellm 1.83.9`, `claude-haiku-4-5`:

| call | `prompt_tokens` | `cache_creation` | `cache_read` | remainder |
|---|---|---|---|---|
| unmarked | 5,415 | 0 | 0 | 5,415 |
| marked, cold | 5,415 | 5,402 | 0 | 13 |
| marked, warm | 5,415 | 0 | 5,402 | 13 |

`prompt_tokens` does not move. It is the total input with the cache classes
included, not the uncached remainder. An independent run on 2026-07-31 at a
different prefix size shows the same thing (15,635 constant across all three);
both are in
[`tier-b/evidence/litellm-marker-survival.json`](https://github.com/Tanisha-Katara/cacheeconomics/blob/main/tier-b/evidence/litellm-marker-survival.json).

## Anthropic branch — `base_coder.py:2094-2097`

```python
# hard code the anthropic adjustments, no-ops for other models since cache_x_tokens==0
cost += cache_write_tokens * input_cost_per_token * 1.25
cost += cache_hit_tokens * input_cost_per_token * 0.10
cost += prompt_tokens * input_cost_per_token          # already contains both
```

This is the branch every Claude model takes: `input_cost_per_token_cache_hit`
is `None` for `claude-haiku-4-5`, `claude-sonnet-4-5` and `claude-opus-4-5` in
`litellm.model_cost`, so the guard on line 2089 is false.

Applying it to the rows above, at haiku's $1/M input:

| call | aider | correct | overstated |
|---|---|---|---|
| uncached | $0.0054150 | $0.0054150 | 1.0x |
| cache write | $0.0121675 | $0.0067655 | 1.8x |
| cache read | $0.0059552 | $0.0005532 | **10.8x** |

## deepseek branch — `base_coder.py:2089-2092`

```python
if input_cost_per_token_cache_hit:
    # must be deepseek
    cost += input_cost_per_token_cache_hit * cache_hit_tokens
    cost += (prompt_tokens - input_cost_per_token_cache_hit) * input_cost_per_token
```

The second line subtracts a **price** from a **token count**.
`input_cost_per_token_cache_hit` is `2.8e-08`, so `prompt_tokens - 2.8e-08`
leaves `prompt_tokens` unchanged to eight decimal places and the whole prompt is
charged at the full input rate on top of the cache-hit charge. It reads like
`prompt_tokens - cache_hit_tokens` was intended. On 10,000 prompt tokens of
which 9,000 were hits, that is 5.7x.

## Suggested fix

LiteLLM already publishes the exact per-token prices in the same `model_cost`
dict aider is already reading, so the hardcoded multipliers can go and the two
branches collapse into one:

```python
input_cost_per_token = self.main_model.info.get("input_cost_per_token") or 0
output_cost_per_token = self.main_model.info.get("output_cost_per_token") or 0
cache_write_cost = (self.main_model.info.get("cache_creation_input_token_cost")
                    or input_cost_per_token * 1.25)
cache_read_cost = (self.main_model.info.get("cache_read_input_token_cost")
                   or self.main_model.info.get("input_cost_per_token_cache_hit")
                   or input_cost_per_token * 0.10)

# prompt_tokens is the total input for both Anthropic and deepseek. The
# uncached part is what is left after removing both cache classes.
uncached_tokens = max(0, prompt_tokens - cache_write_tokens - cache_hit_tokens)

cost += cache_write_tokens * cache_write_cost
cost += cache_hit_tokens * cache_read_cost
cost += uncached_tokens * input_cost_per_token
cost += completion_tokens * output_cost_per_token
```

`max(0, ...)` because a provider that reports the classes inconsistently should
produce a slightly wrong number rather than a negative one.

Two things this gets for free. `cache_creation_input_token_cost_above_1hr` is in
the same dict, so a 1-hour write would price correctly if aider ever emits one
(today it never does — `chat_chunks.py:53` emits `{"type": "ephemeral"}` with no
`ttl`, which is the 5-minute lifetime). And `prompt_cache_min_tokens` is there
too, which matters because a `cache_control` marker on a prefix below the
model's minimum is ignored silently, with no error and
`cache_creation_input_tokens: 0`.

## Scope and caveats

- Measured on Anthropic through LiteLLM. The deepseek branch is read from
  source and arithmetic, not run against deepseek.
- This is about the **reported** cost. Nothing here says aider spends more than
  it should; the API bill is unaffected. What is affected is every decision a
  user makes from the number aider prints, and the benchmark cost figures.
- I have not checked whether `aider --cost` or the leaderboard cost columns are
  computed by this same path.
- `Aider-AI/aider` has an Individual CLA, so this is filed as an issue with the
  patch inline rather than as a PR. Happy to sign and open one if preferred.
