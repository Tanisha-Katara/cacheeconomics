`Coder.calculate_and_show_tokens_and_cost` charges every cached token twice: once at its cache rate, and again at the full input rate inside `prompt_tokens`. Both branches of the `if` do it, by different mechanisms. That number is what `Cost: $X message, $Y session` prints (`base_coder.py:2046-2047`, `2059-2060`).

The error is zero on an uncached request and grows with how well the cache is working, so it is largest for exactly the users who tuned for caching.

Verified at `5dc9490bb35f9729ef2c95d00a19ccd30c26339c`.

## The premise

`base_coder.py:2085-2087` states the assumption in a comment:

```
# Anthropic
# cache_creation_input_tokens + cache_read_input_tokens + prompt
#    == total tokens that were
```

That is not what LiteLLM reports. Same prompt, three consecutive calls, litellm 1.83.9, `claude-haiku-4-5`:

| call | `prompt_tokens` | `cache_creation` | `cache_read` | remainder |
|---|---|---|---|---|
| 1, unmarked | 17,111 | 0 | 0 | 17,111 |
| 2, marked | 17,111 | 0 | 17,102 | 9 |
| 3, marked | 17,111 | 0 | 17,102 | 9 |

`prompt_tokens` does not move. It is the total input with the cache classes included, not the uncached remainder.

Raw usage for every call is in [`prompt-tokens-semantics.json`](https://github.com/Tanisha-Katara/cacheeconomics/blob/main/tier-b/evidence/prompt-tokens-semantics.json). An independent run on 2026-07-31 at a different prefix size shows the same behaviour (15,635 constant across all three) in [`litellm-marker-survival.json`](https://github.com/Tanisha-Katara/cacheeconomics/blob/main/tier-b/evidence/litellm-marker-survival.json). Call tags in the first file are positional rather than claims about cache state — the LiteLLM leg ran after a direct-SDK leg had already warmed the same prefix, so its second call reports a read. The premise is only that `prompt_tokens` is identical in all three.

## Anthropic branch — `base_coder.py:2094-2097`

```python
# hard code the anthropic adjustments, no-ops for other models since cache_x_tokens==0
cost += cache_write_tokens * input_cost_per_token * 1.25
cost += cache_hit_tokens * input_cost_per_token * 0.10
cost += prompt_tokens * input_cost_per_token          # already contains both
```

Every Claude model takes this branch: `input_cost_per_token_cache_hit` is `None` for `claude-haiku-4-5`, `claude-sonnet-4-5` and `claude-opus-4-5` in `litellm.model_cost`, so the guard on line 2089 is false.

At haiku's $1/M input:

| call | aider | correct | overstated |
|---|---|---|---|
| 1, uncached | $0.0171110 | $0.0171110 | 1.0x |
| 2, cached | $0.0188212 | $0.0017192 | **10.9x** |
| 3, cached | $0.0188212 | $0.0017192 | **10.9x** |

On the 31 July run, which caught a genuine cold write, the same arithmetic gives 1.8x on the write and 10.9x on the read.

## deepseek branch — `base_coder.py:2089-2092`

```python
if input_cost_per_token_cache_hit:
    # must be deepseek
    cost += input_cost_per_token_cache_hit * cache_hit_tokens
    cost += (prompt_tokens - input_cost_per_token_cache_hit) * input_cost_per_token
```

The second line subtracts a **price** from a **token count**. `input_cost_per_token_cache_hit` is `2.8e-08`, so `prompt_tokens - 2.8e-08` leaves `prompt_tokens` unchanged to eight decimal places, and the whole prompt is charged at the full input rate on top of the cache-hit charge. It reads like `prompt_tokens - cache_hit_tokens` was intended. On 10,000 prompt tokens of which 9,000 were hits, that is 5.7x.

## Reproduction

```bash
git clone https://github.com/Tanisha-Katara/cacheeconomics
python3 cacheeconomics/disclosure/verify_aider_cost.py
```

No key, no network, no aider install — it transcribes the cost function and replays it against the committed usage artifacts. Takes about five seconds. `--live` adds three real API calls (uncached, write, read; one call cannot show `prompt_tokens` holding constant across them) for under a cent.

## Suggested fix

LiteLLM already publishes exact per-token prices in the same `model_cost` dict aider is already reading, so the hardcoded multipliers can go and the two branches collapse into one:

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

`max(0, ...)` so a provider reporting the classes inconsistently produces a slightly wrong number rather than a negative one.

Two things this gets for free. `cache_creation_input_token_cost_above_1hr` is in the same dict, so a 1-hour write would price correctly if aider ever emits one — today it never does, `chat_chunks.py:53` emits `{"type": "ephemeral"}` with no `ttl`, which is the 5-minute lifetime. And `prompt_cache_min_tokens` is there too, which matters because a `cache_control` marker on a prefix below the model's minimum is ignored silently, with no error and `cache_creation_input_tokens: 0`.

## Scope and caveats

- Measured on Anthropic through LiteLLM. The deepseek branch is read from source and arithmetic, not run against deepseek.
- This is about the **reported** cost. Nothing here says aider spends more than it should — the API bill is unaffected. What is affected is every decision a user makes from the number aider prints, and any cost figure derived from this path.
- I have not checked whether the leaderboard cost columns are computed the same way.

Happy to open a PR. I have not signed the Individual CLA yet, so I have put the patch here rather than assume; say the word and I will.
