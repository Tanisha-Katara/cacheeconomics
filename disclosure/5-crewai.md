# CrewAI — `total_tokens` omits every cached token on the native Anthropic route

**Repo:** `crewAIInc/crewAI` · 56,569★ · MIT
**Verified at:** `main`, 2026-08-03 · line numbers checked live by
`disclosure/verify_claims.py`
**Reproduction:** `python3 disclosure/verify_crewai_tokens.py` — no key, no
network, no crewAI install
**Status:** filed 2026-08-03 as https://github.com/crewAIInc/crewAI/issues/6788

---

## The claim

On the native Anthropic provider, `UsageMetrics.total_tokens` counts neither
cache reads nor cache writes. Those are the tokens you are billed for on a
cached workload, and on a well-cached one they are almost all of them.

The LiteLLM route for the same model does count them, so the two routes report
different totals for identical traffic.

## Where

`llms/providers/anthropic/completion.py:1977`:

```python
result: dict[str, Any] = {
    "input_tokens": input_tokens,
    "output_tokens": output_tokens,
    "total_tokens": input_tokens + output_tokens,     # <--
    "cached_prompt_tokens": cache_read_tokens,
    "cache_creation_tokens": cache_creation_tokens,
}
```

Anthropic's `usage.input_tokens` is the **uncached remainder**, not the input
total. Measured against the API directly, one prompt, three consecutive calls,
`claude-haiku-4-5`, anthropic-sdk 0.120.2. Raw usage for every call is in
[`tier-b/evidence/prompt-tokens-semantics.json`](https://github.com/Tanisha-Katara/cacheeconomics/blob/main/tier-b/evidence/prompt-tokens-semantics.json):

| call | `input_tokens` | `cache_creation` | `cache_read` | `output` | sent |
|---|---|---|---|---|---|
| 1, unmarked | 17,111 | 0 | 0 | 4 | 17,115 |
| 2, marked | 9 | 17,102 | 0 | 4 | 17,115 |
| 3, marked | 9 | 0 | 17,102 | 8 | 17,119 |

So on the third call `total_tokens` is `9 + 8 = 17` for a request that moved
17,119 tokens — an understatement of about 1,000x, and 1,317x on the second.
`verify_crewai_tokens.py` recomputes both columns straight from that artifact.
The dict is then consumed by
`base_llm.py:954-964` via `UsageMetrics.from_provider_dict`, which recomputes
`total_tokens = prompt_tokens + completion_tokens` with `prompt_tokens` resolved
from the `input_tokens` alias (`usage_metrics.py:127-129, 149`), so the
undercount survives normalisation and lands in `self._token_usage`.

## Why the LiteLLM route disagrees

`utilities/token_counter_callback.py:54` uses `usage.prompt_tokens`, and LiteLLM
reports that as the **total** including both cache classes — 5,415 on all three
calls above, never moving. It then adds cached tokens to a separate counter
without touching the total (`base_token_process.py:25-26`), which is right and
does not double-count.

Same crew, same model, `is_litellm=True` versus the native provider
(`llm.py:478`), and `total_tokens` differs by three orders of magnitude on a
fully-cached call. The exact ratio is a property of the fixture, not of crewAI —
it is set by how much of the prompt is cacheable. What generalises is the sign
and the mechanism.

## How big in practice

Those ratios come from a near-100% cache hit on a deliberately large static
prefix, which is the ceiling rather than a typical run. For a realistic figure:
across 14,375 requests of my own agent traffic, cache reads and writes were 97%
of all input tokens. A `total_tokens` built this way would report about 3% of
what was sent.

CrewAI reports tokens rather than dollars, so nothing here is a wrong invoice.
It is a wrong denominator for anything computed from it — cost estimates
downstream, the evaluation metrics in
`experimental/evaluation/metrics/reasoning_metrics.py:68-69` that divide
`total_tokens` by call count, and any OTel span consuming the same field.

## A test currently pins the behaviour

`tests/llms/anthropic/test_anthropic.py:1655-1660`:

```python
mock_response.usage = MagicMock(
    input_tokens=100, output_tokens=50,
    cache_read_input_tokens=30, cache_creation_input_tokens=20,
)
usage = llm._extract_anthropic_token_usage(mock_response)
assert usage["total_tokens"] == 150
```

That request sent 150 input tokens (100 + 30 + 20) and produced 50 output, so
200 passed through it. The assertion encodes 150, which is the undercount, and
would need to change with the fix. Worth flagging rather than leaving to be
discovered in review — I have shipped this exact shape myself, where the test
that should have caught a bug had recorded it as the contract instead.

## Suggested fix

```python
input_tokens = getattr(usage, "input_tokens", 0)
output_tokens = getattr(usage, "output_tokens", 0)
cache_read_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0
cache_creation_tokens = getattr(usage, "cache_creation_input_tokens", 0) or 0

# Anthropic reports input_tokens as the uncached remainder; the cache classes
# are billed separately and are additional to it.
prompt_tokens = input_tokens + cache_read_tokens + cache_creation_tokens

result: dict[str, Any] = {
    "input_tokens": prompt_tokens,
    "output_tokens": output_tokens,
    "total_tokens": prompt_tokens + output_tokens,
    "cached_prompt_tokens": cache_read_tokens,
    "cache_creation_tokens": cache_creation_tokens,
}
```

This makes the native route agree with the LiteLLM route. `cached_prompt_tokens`
and `cache_creation_tokens` keep reporting the split, so nothing that reads them
today changes.

One thing to decide rather than something I would assert: whether `input_tokens`
in this dict should become the total, or stay the remainder with only
`total_tokens` corrected. The alias table at `usage_metrics.py:127-129` maps
`input_tokens` onto `prompt_tokens`, so leaving it as the remainder fixes
`total_tokens` and leaves `prompt_tokens` still undercounting. That is the
reason for the version above, but it is your call.

## Scope and caveats

- Measured against Anthropic directly (sdk 0.120.2) and through LiteLLM 1.83.9,
  recorded in the artifact linked above. `verify_crewai_tokens.py` recomputes
  the claim from it with no key and no network, so nothing here has to be taken
  on trust.
- The measurement is of the Anthropic API's reporting, not of crewAI running.
  I have not executed a crew end to end and watched `UsageMetrics`; the link
  from those usage fields to `total_tokens` is read from source and traced
  above. If you run one and see something different, I would want to know.
- Line numbers are against `main` on 2026-08-03 and drift; `disclosure/verify_claims.py` re-checks them.
- The Bedrock provider (`llms/providers/bedrock/completion.py:2071`) reads
  `usage.get("totalTokens", ...)`, which Bedrock populates itself, so it is
  probably unaffected. I have not verified that against a live Bedrock call and
  am not claiming it.
- I have not checked the Gemini or OpenAI native providers for the same shape.


---

## Correction posted 2026-08-03

The evidence citation in this draft was wrong and was corrected publicly at
https://github.com/crewAIInc/crewAI/issues/6788#issuecomment-5168933668.

`prompt-tokens-semantics.json` has `cache_creation_input_tokens: 0` on all
three LiteLLM calls, so it demonstrates reads rather than both cache classes.
The write case is in `litellm-marker-survival.json`, which the issue did not
link. The claim holds; the citation did not support it.
