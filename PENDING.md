# Known Limitations

This file is the public backlog for known-and-not-done work. It is intentionally
short: launch readers should be able to distinguish a current limitation from
old merge history. Detailed historical audit notes remain in git history.

## 1. Provider coverage is deliberately default-deny

cacheeconomics only publishes checks and prices for provider surfaces that have
dated registry evidence. Missing provider facts, contested rows, partner
surfaces without customer-specific rates, and self-hosted stacks without stable
cache-hit accounting are reported as abstentions rather than guessed.

This is product behavior, not a documentation gap. Expanding support means
adding sourced registry rows and tests for the surface.

## 2. Bedrock, Vertex, OpenAI, DeepSeek, and self-hosted stacks are not the same

The registry records surfaces separately because cache semantics and billing
can differ even when model names look similar. Anthropic-style cache markers,
OpenAI/DeepSeek implicit or explicit caching, partner billing, and self-hosted
gateway behavior should not be collapsed into one generic "LLM caching" model.

The next useful expansion is not "support every model." It is support for more
surface-specific trace fields, usage counters, rates, and refusal reasons.

## 3. Recommendations are scoped to the evidence that produced them

Some traces can contain several cache behaviors at once: one agent cadence may
make a one-hour TTL worthwhile while another part of the same trace is mostly
an efficiency or rebuild problem. Reports rank findings and show basis lines,
but they are not blanket instructions to apply a single TTL or marker policy to
all traffic.

When a finding looks contradictory, rerun with `--detail` and check which
requests, sessions, or agents produced it.

## 4. Token estimates are safety gates, not proof

Without exact tokenizer counts, structural analysis falls back to byte-share
estimates. The minimum-cache check uses the measured worst overestimate guard
from `segment.ESTIMATOR_WORST_OVERESTIMATE`, but that measurement is still a
bounded data point from this repository's evidence, not a universal theorem.

Use `tier-b/count_tokens.py` or your own trusted tokenizer path when exact
structural dollar figures matter. The installed package remains socket-free;
token counting is a separate, explicit egress step.

## 5. Live marker placement is conservative by design

`CachePlugin` can place markers automatically, but mutation is opt-in and it
stands down on cold data, existing markers, unsafe positions, uncertain
minimums, unsupported surfaces, or requests that would require moving prompt
content. That means it may refuse to optimize a request a human could tune by
hand after inspecting application context.

This is the intended launch posture. The first public users should see false
negatives before they see silent rewrites.

## 6. Public project operations are still lightweight

The repository now has minimal contribution, security, changelog, and issue
templates. It does not yet have a formal governance model, release automation,
or a compatibility promise beyond the package metadata and changelog.

Before a 1.0 release, add automated PyPI publishing, signed release notes, and a
clear API stability policy.
