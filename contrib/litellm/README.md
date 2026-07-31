# LiteLLM contribution — `prompt_cache_min_tokens` coverage

Gate 6 of the plan. Pass criterion is maintainer engagement, not a merge.

## What is here

| File | What it is |
|---|---|
| `issue.md` | The issue to file at [BerriAI/litellm](https://github.com/BerriAI/litellm). Ends with two questions, because the second one changes whether this is a data PR or a code PR |
| `proposed-minimums.json` | 29 keys and their minimums. Apply to `model_prices_and_context_window.json` |
| `test_prompt_cache_min_tokens_aliases.py` | Goes to `tests/test_litellm/`. Their contributing guide makes at least one test a hard requirement, and mocked-only in that directory |

## The finding, in one paragraph

`get_prompt_cache_min_tokens()` reads `prompt_cache_min_tokens` from the cost
map and falls back to `DEFAULT_MINIMUM_PROMPT_CACHE_TOKEN_COUNT` (1024) when a
model has no entry. 499 of the 623 entries that set `supports_prompt_caching:
true` have no minimum. For 29 of those, the same model carries a minimum on a
different alias in the same file. Twelve have a real minimum above 1024, so
`is_prompt_caching_valid_prompt()` returns `True` for prompts Anthropic will not
cache — and Anthropic returns no error in that case, so it fails silently.

## Reproducing it

```bash
curl -sLO https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json
python3 - <<'PY'
import json
d = json.load(open('model_prices_and_context_window.json'))
claims = [k for k, v in d.items()
          if isinstance(v, dict) and v.get('supports_prompt_caching') is True]
missing = [k for k in claims if d[k].get('prompt_cache_min_tokens') is None]
print(f'{len(missing)} of {len(claims)} claim caching support with no minimum')
PY
```

## Scope discipline

Values come from our registry, which records Anthropic's published minimums with
a source URL, a quote and a check date per row. Only models we hold a primary
source for are proposed.

28 further Anthropic-family entries have the same gap and are **deliberately
left out**: `claude-3-opus`, `claude-3-haiku`, `claude-3-5-sonnet`,
`claude-3-7-sonnet` and their aliases. We have no primary source for those
minimums, and an absent value is better than a guessed one — which is the same
argument the issue makes about the fallback.

We also checked every minimum already in the file, model by model, for internal
disagreement. Exactly one model has it: `claude-fable-5` reads 512 on the direct
entry and 1024 on four Bedrock and regional entries. Anthropic publishes 512.
Every other recorded minimum is correct, including the non-monotonic ones.

That check was added because an earlier draft of the issue asserted the opposite
without having run it.

## Status

**Filed 2026-07-29 as [BerriAI/litellm#35011](https://github.com/BerriAI/litellm/issues/35011)**
by Tanisha-Katara. Awaiting maintainer response. Gate 6 passes on engagement,
not on a merge.

Pre-filing checks, all done:

- [x] Re-fetched the price file at filing time; byte-identical to the research
      copy, counts unchanged at 499/623
- [x] No existing issue mentions `prompt_cache_min_tokens` (0 results)
- [x] Ran the issue's own reproduction snippet verbatim; output matches the
      issue text

Two claims in the first draft were wrong and were caught before filing. It
asserted a key `anthropic.claude-haiku-4-5` that does not exist, and claimed
every recorded minimum was correct when `claude-fable-5` disagrees with itself
across four keys. The second became the lead finding.

## If a PR is invited

The issue ends on two questions, one of which decides the shape of the change:
per-key values, or alias-to-base resolution. Resolution fixes the whole class
including the 28 excluded here, but it is a code change and theirs to design.
Do not open a PR before that is answered.
