One correction to my own citation, before anyone spends time on this.

I wrote that LiteLLM "reports that as the **total** including both cache classes — 17,111 on all three calls above." The three calls I linked have `cache_creation_input_tokens: 0` throughout. They demonstrate that `prompt_tokens` includes cache **reads**. They do not show a write, so they do not support "both classes" on their own.

The write case is in a separate run, [`litellm-marker-survival.json`](https://github.com/Tanisha-Katara/cacheeconomics/blob/main/tier-b/evidence/litellm-marker-survival.json) from 2026-07-31: `prompt_tokens` 15,635 with `cache_creation_input_tokens` 15,624 on the same call, and 15,635 again on the uncached control. So `prompt_tokens` is the total across both classes — it just takes both artifacts to show it, and I cited only one.

Nothing else changes. The native-provider undercount is established by the direct Anthropic SDK rows, which are in the artifact I did link, and by the source path through `from_provider_dict` into `_token_usage`.
