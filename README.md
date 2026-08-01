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
different prompt layout have cost", because it can see prompt structure. It
needs the counting step below before those answers carry dollar figures.

An instrumented capture, using the recorder in this package. Needs a small code
change on your side and gives the strongest answers about structure. It needs
the counting step too: the recorder measures bytes like everything else.

Nothing at all, if your agent lets you change its base URL.
`tier-b/capture_proxy.py` forwards to the provider and writes a bodies export as
it goes, so you get the wire without exporting anything and without touching the
agent's code. That is how the browser-use measurement below was done, on a
project nobody instrumented for us.

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

## Measuring an agent you did not write

Most agents let you point them at a different base URL, because that is how
people use proxies and gateways. That is enough.

```bash
python3 tier-b/capture_proxy.py --out run.jsonl --port 8787 &
# then start the agent with its base URL set to http://127.0.0.1:8787
```

It forwards every request to the provider byte for byte and writes each one to
`run.jsonl` with the response beside it. Nothing is mutated, so what you measure
is the agent's own behaviour rather than the behaviour of an agent with
something in its path. The output is a bodies export, so it feeds
`--from bodies` directly.

It also stamps a session key derived from the stable prefix, because the wire
does not carry one and the analysis needs it. Without that every request lands
in one reuse chain, and a workload that interleaves call types reads as a single
conversation whose tools keep changing. That produced a confident and completely
wrong finding the first time this was run.

We used it on browser-use 0.13.7. Three real browser tasks, 33 requests, and the
result is in `tier-b/evidence/browser-use-interactive.json`: caching removes 54%
of its input spend, four requests place markers on prefixes below the model
minimum where the provider silently ignores them, and the one-hour TTL idea does
not apply at that cadence at all. The last of those is a finding against our own
hypothesis, which is why it is in the repository.

## Counting the tokens

Skip this and you still get every finding. You just will not get a dollar figure
on the ones about prompt structure.

Here is why. To say "34,000 tokens sit behind that volatile block" the tool has
to know how many tokens each part of your prompt is worth, and it cannot count
them on your machine: there is no local tokenizer for these models. So by
default it takes the input total your provider billed and divides it between the
segments in proportion to their bytes.

That split is worse than it sounds. Measured against the provider's own
tokenizer, it is off by 19.2% at the median and 181% at worst, because
bytes-per-token is not one number: dense JSON tool schemas run about 2.74 bytes
per token where English prose runs 5.22. A prompt mixing them, which is every
agent prompt, gets the prose over-allocated and the tools starved at the same
time. The per-segment rows are in `tier-b/evidence/inferred-token-split.json`.

This report will not publish spend that reconciles worse than 5% against your
invoice. Costing a recommendation from a 19% split while holding the invoice to
5% would be two standards, so structural findings arrive without figures until
the sizes are counted.

Counting them takes one command:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python3 tier-b/count_tokens.py bodies.jsonl -o bodies-counted.jsonl
cacheeconomics analyze bodies-counted.jsonl --from bodies --invoice-usd 4820.16
```

The same measurement against the same tokenizer then lands at 0.2%.

Three things worth knowing before you run it.

It is free and it is fast, because the endpoint costs nothing and the work is
cached on the prefix. Counting a prefix once covers every request that shares
it, which is every request, because a prefix that is not shared is not being
cached in the first place. On the demo trace 286 requests needed 345 calls. The
cache is written next to the output, so a second run costs nothing at all.

It sends prompt content to a tokenizer, and you choose which one. By default
that is Anthropic. If your workload already runs there it is the same content
over the same wire to the same company; if it runs on Bedrock or Vertex it is a
new egress path to a different vendor. Either way `--endpoint` points it at your
own gateway instead, so the egress can stay inside your perimeter.

You can also see exactly what it would send before agreeing to any of it:

```bash
python3 tier-b/count_tokens.py bodies.jsonl -o counted.jsonl --dry-run
```

That reports the call count and the host and sends nothing. It writes nothing
either, deliberately: an earlier version wrote an output file full of zeroes
that the analyzer then read as exact counts, which is the kind of
authoritative-looking wrong answer the rest of this tool exists to refuse.

And you can skip the whole step. Nothing breaks: you get every finding, and the
ones about prompt structure arrive without dollar figures.

It is a separate script rather than a flag, deliberately. The installed package
imports no network library at all and a test asserts it, because zero egress is
the thing a client is trusting when they hand over a trace, and they can check
it by grepping the wheel. A flag would have made that claim depend on reading
the flag's implementation.

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

Structural findings need counted sizes. A recommendation to move a block of
your prompt is costed from how many tokens that block is, and a byte-share
estimate of that is 19.2% off at the median. Those findings still report; their
dollar figures wait for `tier-b/count_tokens.py`.

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

Both halves have now been watched against a real LiteLLM. The worry was that
LiteLLM normalises Anthropic-shaped bodies through an OpenAI-shaped intermediate
and drops `cache_control`, which would give you churn and no cache writes while
the plugin's own counters cheerfully reported placements that never arrived.

It does not. On litellm 1.83.9, an unmarked control writes nothing, a marked
call writes 15,624 tokens, the same body a moment later reads all 15,624 back,
and a request routed through the handler with `mutate=True` writes too. The runs
are in `tier-b/evidence/litellm-marker-survival.json` and the script starts cold
each time, so you can repeat it.

`mutate` still defaults to False. Knowing the mechanism works is not the same as
deciding to rewrite somebody's live traffic, and that second decision is yours.

## Development

```bash
python3 -m pytest -q          # 1011 tests, no dependencies
python3 web/build_bundle.py   # after changing anything under harness/cacheeconomics
```

CI runs the suite on Python 3.9 and 3.13 with only pytest installed, so an
accidental dependency fails the build rather than shipping.

## License

Apache-2.0. Copyright 2026 KCG Consulting LLC.
