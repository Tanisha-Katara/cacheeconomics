# cacheeconomics

[![ci](https://github.com/Tanisha-Katara/cacheeconomics/actions/workflows/ci.yml/badge.svg)](https://github.com/Tanisha-Katara/cacheeconomics/actions/workflows/ci.yml)
[![pypi](https://img.shields.io/pypi/v/cacheeconomics)](https://pypi.org/project/cacheeconomics/)

You point it at logs you already have. It tells you what your prompt caching
actually cost, which part of the prompt is wasting the money, and how much of
that is recoverable.

It runs on your machine, opens no sockets, and will not print a dollar figure it
cannot tie to an invoice.

## Why this is not just adding up tokens

Your provider bills input tokens at four different rates: fresh input at 1x, a
cache read at 0.1x, a five-minute cache write at 1.25x, a one-hour write at 2x.
Your usage counters report the totals. They do not report which decisions put
tokens in the expensive buckets, and that is where the money is.

Three specific ways it leaks:

A one-hour cache is only cheaper inside a band. It costs 2x to write where the
short one costs 1.25x, so it wins only when your requests are more than five
minutes and less than an hour apart. Faster than that, the cheap entry was still
alive and you overpaid. Slower, both had expired and you overpaid. We measured
this on live API calls: $0.190 for the 5m arm against $0.114 for the 1h arm on
the same prefix, 40.2% cheaper, with all eight probes matching predictions
written down before the run. The per-call receipts are in `tier-b/evidence/`.

Below a model's minimum, a cache marker does nothing and nobody tells you. The
threshold is 512, 1024, 2048 or 4096 tokens depending on the model, and it does
not track model generation the way you would expect: `claude-opus-5` is 512,
`claude-opus-4-6` is 4096. Under it the provider quietly processes your request
uncached and reports `cache_creation_input_tokens: 0`. No error.

A write nobody reads is pure premium. You paid 1.25x or 2x for a prefix and then
changed something at the top of it. Usage counters cannot tell you that happened,
because "we rebuilt this instead of extending it" is a fact about prompt
structure, not about token totals.

## What you need to run it

One of these, in rough order of how little work they are for you:

Your Claude Code transcripts. Already on your disk, nothing to instrument. Run
`cacheeconomics claude-code`.

Your LiteLLM proxy logs. If you run LiteLLM you are already writing
`StandardLoggingPayload` records. That file is the input. This path needs a
recent enough LiteLLM to include `prompt_tokens_details`; without the cache
class split the tool will tell you it cannot read the rows rather than guess.

Request bodies your gateway already logs. Langfuse, Helicone and the LiteLLM
proxy all keep full bodies. This is the first path that can answer "what would a
different prompt layout have cost", because it can see prompt structure.

On this path the tool has to say how many tokens each part of your prompt is
worth, and it cannot count them offline: there is no local tokenizer for these
models. By default it divides your billed total between the segments in
proportion to their bytes, and measured against the provider's own tokenizer
that split is off by 19.2% at the median and 181% at worst, because dense JSON
tool schemas run about 2.74 bytes per token where English prose runs 5.22. Every
structural finding is costed from it. `tier-b/count_tokens.py` replaces the
estimate with exact counts and brings the same measurement to 0.2%; it is a
separate script because it talks to the provider and the analyzer does not.

An instrumented capture, using the recorder in this package. Needs a small code
change on your side and gives the strongest answers.

The file you have decides what the tool will claim. It works this out from the
contents, not from what you tell it, so an export cannot ask for more confidence
than it earns.

## Install and run

```bash
pip install cacheeconomics
```

Python 3.9 or newer and nothing else. `pip install "cacheeconomics[litellm]"`
only if you are mounting the live proxy plugin.

Or from a checkout, which is what you want if you also need the measurement
work and the tests:

```bash
git clone https://github.com/Tanisha-Katara/cacheeconomics.git
cd cacheeconomics && pip install .
```

```bash
# your own Claude Code usage, read locally
cacheeconomics claude-code --allow-unreconciled

# a trace, reconciled against the bill you actually received
cacheeconomics analyze trace.jsonl --invoice-usd 4820.16

# LiteLLM proxy logs, unchanged
cacheeconomics analyze standard_logging.jsonl --from litellm --invoice-usd 4820.16

# Bedrock or Vertex: the rate comes off your cloud bill, not from us
cacheeconomics analyze trace.jsonl --target-id amazon-bedrock/converse --effective-rate 1.10

# compare cache placement policies over the same requests
cacheeconomics bakeoff trace.jsonl --by-agent

# check a cache config before you ship it
cacheeconomics checks --prefix-tokens 900 --model claude-opus-5 --breakpoints 5

# what does this build actually know?
cacheeconomics registry
```

`checks` works as a CI gate. Exit 0 means everything passed, 2 means something
failed, and 3 means nothing failed but something could not be evaluated, usually
a model with no dated registry entry. That third code exists because "I did not
check" is not "this passed", and collapsing them gives you a green build where
the check that catches silently-ignored cache markers never ran. Exit 1 means the
tool itself broke.

## What it refuses to tell you

This part matters more than the analysis, because it is what makes the analysis
worth reading.

No invoice, no dollars. Every figure stays hidden until it reconciles against
money that actually left your account, within 5%. Without `--invoice-usd` the
report opens with FIGURES WITHHELD and prints `[figure withheld]` where each
number would be. `--allow-unreconciled` releases them for internal drafts, and it
is a flag you have to type rather than a default, so publishing an unreconciled
number is always somebody's decision.

Structure and usage have to agree. Spend comes from usage counters; structural
findings come from segment sizes. If those two descriptions of the same prompt
disagree by more than 5%, you still get the findings and you do not get their
dollar figures.

Coverage is measured in money, not rows. A structural claim needs to cover 90% of
your billed tokens before a figure attaches to it. Nine small structured requests
next to one huge unstructured one is 90% of rows and can be 8% of the bill, and a
recommendation costed from 8% of your spend does not describe your workload.

Bedrock and Vertex are not priced here. AWS and Google operate and invoice those,
so the rates are not ours to publish and we do not borrow Anthropic's. Pass
`--effective-rate` from your own cloud bill, which is the better number anyway:
partner rates vary by region and endpoint type, and nobody at your volume pays
list.

Every number says what kind of number it is. Measured means observed in usage
fields. Modeled means projected, always as a range, with the pessimistic end as
the headline. Verified means observed in production after a change shipped.
Nothing gets promoted between them quietly.

## Privacy

Everything runs locally. Nothing in the package opens a socket, the registry is
packaged data, and every input is a path on your disk.

Prompt text is optional. Hashes, structure and token counts are enough for every
finding here.

Segment identifiers are keyed HMAC-SHA-256 rather than bare digests, because a
bare digest of a short policy line is confirmable by anyone holding a guess.
There is deliberately no `--key` flag: `argv` shows up in `ps` and lands in shell
history, so the key comes from `CACHEECONOMICS_HMAC_KEY` or a `--key-file` whose
permissions get checked before it is read.

Keying and scoping are different things, and the difference matters on a shared
gateway. A key stops someone guessing your content. It does nothing about
equality, because equality is the whole job of an identifier: under one key,
identical text produces an identical id no matter who sent it. So identity is
scoped to the tenant as well. Pass `--tenant` on a multi-tenant export and two
tenants sending the same policy block stop sharing an id. That is also the
arithmetically correct answer, since caches are isolated per tenant and those two
requests could never have shared a cache entry.

## What is in here

```
harness/cacheeconomics/
  registry.py   dated capability and pricing registry, refuses to guess
  cost.py       the four token classes, date-effective pricing
  money.py      a dollar figure that cannot be printed until it is released
  trace.py      ingest, tiering, reconciliation
  analyzer.py   the finding rules (EFF-1, VOL-1, MIN-1, TTL-1, FAN-1, ...)
  report.py     text and self-contained HTML
  simulate.py   replay a trace against a modelled cache; the policy bake-off
  allocate.py   placement policies, including a model of LiteLLM auto-injection
  tiers.py      the DP tier allocator and an exact evaluator that agrees with it
  monitor.py    runtime diagnostics, bounded state
  plugin.py     a gateway plugin that decides inside the request path
  recorder.py   instrumented capture
  segment.py    the single traversal of a request body
  cli.py        the commands above
  data/         providers.json, pricing.json
```

```
web/            a browser demo; the analyzer runs in a worker via Pyodide
tier-a/         static findings on three open-source agents, cited to file and line
tier-b/         the live TTL experiment and its per-call evidence
disclosure/     findings as filed against the projects they affect
contrib/        upstream contributions (BerriAI/litellm#35011)
```

The measurement work is here rather than in the installed package, which ships
`harness/` only. `tier-a/FINDINGS.md` cites a file and line for every claim and
logs its own correction where the first screen was wrong. `tier-b/evidence/`
holds the per-call rows behind the 40.2% number, including one run that reports
`script_sha256: null` because it predates run-time hashing.

Eight provider surfaces are described. The recorded rates are Anthropic
first-party list prices, and the table names which surfaces they are valid for:
`anthropic/direct` and `anthropic/claude-platform-on-aws`. A test fails if a
model is ever added there without a dated rate.

The scope is default-deny, so a surface earns those rates by being named rather
than by being absent from an exclusion list. The alternative is how this went
wrong once already: before the scope existed, a Bedrock trace priced straight off
the Anthropic table and produced a confident total no AWS bill would match.

## Running it live

`plugin.py` can sit inside a LiteLLM proxy and place cache markers on outgoing
requests. It observes by default and only mutates when you ask.

The observing half is tested here. The mutating half is not: nothing in this
repository has watched a real LiteLLM proxy forward one of these requests, so we
do not know whether `cache_control` on a message content block survives LiteLLM's
translation to the Anthropic wire format. If it gets stripped you would get churn
and no cache writes while the plugin's own counters cheerfully reported
placements that never arrived. That is why `mutate` defaults to False and why
this paragraph exists.

## Development

```bash
python3 -m pytest -q          # 1011 tests, no dependencies
python3 web/build_bundle.py   # after changing anything under harness/cacheeconomics
```

CI runs the suite on Python 3.9 and 3.13 with only pytest installed, so an
accidental dependency fails the build rather than shipping.

## License

Apache-2.0. Copyright 2026 KCG Consulting LLC.
