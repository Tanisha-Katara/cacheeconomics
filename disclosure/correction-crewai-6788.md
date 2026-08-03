Correcting my own citation before anyone spends time on it.

I wrote that LiteLLM reports `prompt_tokens` as the total "including both cache
classes — 17,111 on all three calls above". All three of those calls have
`cache_creation_input_tokens: 0`. They show `prompt_tokens` covering cache
reads. They say nothing about writes.

The write case is in a run I did not link,
[`litellm-marker-survival.json`](https://github.com/Tanisha-Katara/cacheeconomics/blob/main/tier-b/evidence/litellm-marker-survival.json)
from 31 July: `prompt_tokens` 15,635 alongside `cache_creation_input_tokens`
15,624 on the same call, and 15,635 again on the uncached control. So
`prompt_tokens` does cover both classes. It takes both files to show it and I
cited one.

Nothing else moves. The native-provider undercount rests on the direct Anthropic
SDK rows, which are in the artifact I did link, and on the source path through
`from_provider_dict` into `_token_usage`.
