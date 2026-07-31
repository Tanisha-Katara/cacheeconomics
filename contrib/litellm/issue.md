# `prompt_cache_min_tokens`: four `claude-fable-5` keys disagree with the direct entry, and 499 entries that claim caching support record no minimum at all

Two related problems in `model_prices_and_context_window.json`, found while building a cache-cost analyzer that reads this file as a source of truth. The first is a wrong value, the second is a coverage gap. Both surface through `get_prompt_cache_min_tokens()`.

## 1. `claude-fable-5` contradicts itself

| key | `prompt_cache_min_tokens` |
|---|---|
| `claude-fable-5` | **512** |
| `anthropic.claude-fable-5` | 1024 |
| `us.anthropic.claude-fable-5` | 1024 |
| `eu.anthropic.claude-fable-5` | 1024 |
| `global.anthropic.claude-fable-5` | 1024 |

Anthropic publishes 512 for Fable 5, which the direct entry already has. The four platform entries say 1024.

Effect: on those four, a 700-token prefix is treated as too short to cache. `is_prompt_caching_valid_prompt()` returns `False`, the caller skips caching, and a prefix that would have cached is re-sent at full rate on every request.

I checked every other Anthropic model in the file for the same kind of internal disagreement. This is the only one. Every other recorded minimum is correct, including the non-monotonic ones (512 on Opus 5, 1024 on Opus 4.8 and Sonnet 5, 2048 on Opus 4.7 and Haiku 3.5, 4096 on Opus 4.6, Opus 4.5 and Haiku 4.5).

## 2. 499 of 623 entries claiming caching support have no minimum

`get_prompt_cache_min_tokens()` reads the cost map and falls back to `DEFAULT_MINIMUM_PROMPT_CACHE_TOKEN_COUNT` (1024) when a model has no entry. That value feeds `is_prompt_caching_valid_prompt()` and `PromptCachingDeploymentCheck`.

For 29 of those entries the model is an Anthropic one whose minimum is already recorded elsewhere in this file, on another key. `claude-haiku-4-5` shows the shape: eighteen keys for that model set `supports_prompt_caching: true`, fourteen carry `4096` (direct, Vertex, Bedrock, `us.`, `eu.`, `apac.`, `au.`, `jp.`, `global.`), and four carry nothing:

| key | `prompt_cache_min_tokens` |
|---|---|
| `azure_ai/claude-haiku-4-5` | *absent, resolves to 1024* |
| `openrouter/anthropic/claude-haiku-4.5` | *absent, resolves to 1024* |
| `snowflake/claude-haiku-4-5` | *absent, resolves to 1024* |
| `vercel_ai_gateway/anthropic/claude-haiku-4.5` | *absent, resolves to 1024* |

## Why the direction of the error matters

Below the minimum, Anthropic processes the request without caching and returns no error. `cache_creation_input_tokens` comes back as 0 and nothing else signals it, so this fails silently.

Across the 29 entries:

- **12** have a real minimum above 1024, so `is_prompt_caching_valid_prompt()` returns `True` for prompts Anthropic will not cache. The caller marks the prompt, nothing is written, and there is no signal.
- **3** are `claude-fable-5` at 512, so caching is skipped for prompts that would have cached. Same direction as problem 1.
- **14** happen to match 1024 and are unaffected today, though they would drift if any of those models changed.

`PromptCachingDeploymentCheck` is less exposed: `_get_min_token_count_for_deployments` takes the lowest minimum in a group, and its docstring explains why a low threshold only costs a lookup. `is_prompt_caching_valid_prompt` is the path where a wrong value produces a wrong answer.

## Reproducing

```bash
curl -sLO https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json
python3 - <<'PY'
import json
d = json.load(open('model_prices_and_context_window.json'))
print({k: d[k].get('prompt_cache_min_tokens') for k in d if 'fable-5' in k})
claims = [k for k, v in d.items()
          if isinstance(v, dict) and v.get('supports_prompt_caching') is True]
print(f'{len([k for k in claims if d[k].get("prompt_cache_min_tokens") is None])}'
      f' of {len(claims)} claim caching support with no minimum')
PY
```

## Happy to send a PR

I have both changes and a test ready for `tests/test_litellm/`, following the contributing guide. Problem 1 is four values. Problem 2 is 29 keys, one field, no keys added or removed.

I deliberately left out 28 further entries with the same gap (`claude-3-opus`, `claude-3-haiku`, `claude-3-5-sonnet`, `claude-3-7-sonnet` and their aliases) because I do not hold a primary source for those minimums and would rather leave a value absent than guess one, which is the same argument this issue makes about the fallback.

Two questions before I open it:

1. For problem 2, would you prefer the values written out per key, or a resolution step that falls back to the base model when an alias has none? The second fixes the whole class including the 28 I left out, but it is a code change rather than a data change, and the design is yours.
2. Is there a convention for citing a source on a cost-map value? I could not find one, and these are numbers that go stale quietly. Problem 1 is what that looks like when it happens.

Filed by Tanisha Katara, KCG Consulting LLC.
