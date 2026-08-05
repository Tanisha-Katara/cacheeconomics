# cacheeconomics

[![ci](https://github.com/Tanisha-Katara/cacheeconomics/actions/workflows/ci.yml/badge.svg)](https://github.com/Tanisha-Katara/cacheeconomics/actions/workflows/ci.yml)
[![pypi](https://img.shields.io/pypi/v/cacheeconomics)](https://pypi.org/project/cacheeconomics/)

[![Cache is the new cash](docs/assets/cache-is-new-cash.jpg)](https://commodiverus388593.substack.com/p/cache-economics-how-to-stop-paying)

Local prompt-cache economics for LLM agents.

cacheeconomics turns a vague "input tokens" bill into a concrete fix list:
which tokens were fresh input, which were cache reads, which were cache writes,
what they cost, and what caching behavior to fix first.

Read the launch essay:
[Cache Economics: How to Stop Paying](https://commodiverus388593.substack.com/p/cache-economics-how-to-stop-paying).

```bash
pip install cacheeconomics
cacheeconomics claude-code
```

If you use Claude Code, that is enough to inspect the transcripts already on
your machine. No instrumentation, no API key, no network call from the installed
package.

## Why It Exists

LLM dashboards often show one number called "input tokens." For prompt caching,
that hides the useful part.

Provider-side prompt caching can price input-like tokens very differently:

| token class | typical price shape |
|---|---:|
| fresh input | 1x |
| cache read | 0.1x |
| 5 minute cache write | 1.25x |
| 1 hour cache write | 2x |

That is a 20x spread between the cheapest and most expensive token bucket.
cacheeconomics separates those buckets and tells you which behavior is wasting
money: expiry, rebuilds, bad marker placement, model switches, tiny prefixes,
or TTLs that do not match request cadence.

## What You Get

- A local report over Claude Code transcripts, LiteLLM logs, request-body
  exports, or captured proxy traffic.
- Cache health metrics, including cache-read share and prefix efficiency.
- Ranked findings with a concrete "do this" recommendation.
- Policy bake-offs for marker placement and TTL choices.
- CI-friendly cache checks before a config ships.
- Optional live marker placement through `CachePlugin`, observe-first and
  opt-in for mutation.

## Who It Helps

- **Claude Code users and vibe coders** who want to know why usage spiked,
  whether compaction/model switches are rebuilding prefixes, and which session
  pattern to fix first.
- **Claude API agent builders** shipping coding agents, browser agents, research
  agents, or internal tools.
- **Platform and gateway teams** running LiteLLM, Bedrock, Vertex, custom
  gateways, or multi-tenant traffic.
- **FinOps and engineering leads** who need spend claims tied to evidence.
- **Framework maintainers and auditors** investigating cache-marker,
  token-accounting, and TTL bugs.

## Common Commands

| what you have | command |
|---|---|
| Claude Code history | `cacheeconomics claude-code` |
| LiteLLM proxy logs | `cacheeconomics analyze log.jsonl --from litellm` |
| gateway request bodies | `cacheeconomics analyze bodies.jsonl --from bodies` |
| policy bake-off | `cacheeconomics bakeoff trace.jsonl --by-agent` |
| CI cache check | `cacheeconomics checks --target-id anthropic/direct --prefix-tokens 900 --model claude-opus-5 --breakpoints 5` |
| registry summary | `cacheeconomics registry` |

`--from bodies` needs `CACHEECONOMICS_HMAC_KEY` or `--key-file`, because segment
ids are keyed hashes of prompt content. The tool refuses bare prompt digests.

For Bedrock, Vertex, or another partner-operated surface, pass the surface and
your effective rate from the bill:

```bash
cacheeconomics analyze trace.jsonl \
  --target-id amazon-bedrock/converse \
  --effective-rate 1.10
```

## What It Can Find

- A cron agent rebuilding a five-minute cache every seven minutes.
- A cache marker below the model's minimum prefix size.
- A static tool schema sitting in an expensive write bucket.
- A model switch splitting cache reuse.
- A one-hour TTL that only helps inside the five-minute-to-one-hour reuse band.
- Prompt sections that change too early and invalidate everything after them.

The schedule effect is the clearest example: in the browser-use case study,
moving from a two-minute cadence to a seven-minute cadence turned cache writes
from hundreds of tokens per request into thousands, because the five-minute
cache expired before reuse. The full write-up is in
[case-studies/schedule-decides-your-cache-bill.md](case-studies/schedule-decides-your-cache-bill.md).

## Live Placement

`CachePlugin` can sit inside a LiteLLM proxy and place cache markers on outgoing
requests.

It is conservative by default:

- observes before mutating
- never double-injects existing markers
- stands down on cold data
- stands down near model minimums
- stands down on unsupported or contested registry rows
- does not move prompt content

Install the optional dependency only when mounting the live plugin:

```bash
pip install "cacheeconomics[litellm]"
```

## Support Boundaries

cacheeconomics is strongest on Claude/Anthropic-style prompt caching, where the
provider exposes separate cache read/write counters and developers can control
markers or TTLs.

It does not claim every model or every cache is the same thing:

- Anthropic direct, Bedrock, and Vertex are separate billing/control surfaces.
- OpenAI and DeepSeek have different cache-control models.
- DeepSeek-style support means hit/miss economics, not Anthropic breakpoint
  placement.
- Open-weight or self-hosted support depends on the serving stack exposing
  stable cache semantics, usage counters, and rates.
- Unknown or contested registry facts become abstentions, not guesses.

This is not a replacement for observability platforms such as Langfuse,
Helicone, Phoenix, LiteLLM, Portkey, or GPTCache. It is narrower: a local tool
for provider-side prompt-cache economics and the concrete cache changes to try
first.

## Privacy

The installed package opens no sockets. It imports no network library, and CI
tests that claim.

Network-touching tools are separate and explicit:

- `tier-b/count_tokens.py` sends prompt prefixes to a tokenizer.
- `tier-b/capture_proxy.py` is a forwarding proxy for measurement.

Prompt text is optional. Hashes, structure, and token counts are enough for the
main findings. Segment identifiers are keyed HMAC-SHA-256, and multi-tenant
identity can be scoped with `--tenant`.

## Repository Map

```text
harness/cacheeconomics/  package source
web/                     browser demo bundle
tier-a/                  static findings on open-source agents
tier-b/                  live experiments and evidence rows
case-studies/            narrative write-ups
disclosure/              upstream disclosures and verifiers
contrib/                 upstream contribution material
```

Only `harness/` ships in the wheel. Evidence and experiments stay in the repo
for auditability.

## Development

```bash
git clone https://github.com/Tanisha-Katara/cacheeconomics.git
cd cacheeconomics
pip install .

python3 -m pytest -q
python3 web/build_bundle.py   # after changing harness/cacheeconomics
```

Python 3.9 or newer. The package has no runtime dependencies. CI runs on Python
3.9 and 3.13 with only `pytest` installed.

Before contributing, read [CONTRIBUTING.md](CONTRIBUTING.md). Security reports
and accidental egress concerns belong in the private flow described in
[SECURITY.md](SECURITY.md), not in public issues.

Known limitations are tracked in [PENDING.md](PENDING.md).

## License

Apache-2.0. Copyright 2026 KCG Consulting LLC.
