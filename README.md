# cacheeconomics

[![ci](https://github.com/Tanisha-Katara/cacheeconomics/actions/workflows/ci.yml/badge.svg)](https://github.com/Tanisha-Katara/cacheeconomics/actions/workflows/ci.yml)
[![pypi](https://img.shields.io/pypi/v/cacheeconomics)](https://pypi.org/project/cacheeconomics/)

Your provider bills input tokens at four different prices. Fresh input is 1x. A
cache read is 0.1x. A five-minute cache write is 1.25x. A one-hour write is 2x.
That is a 20x spread between the cheapest token you can send and the most
expensive one. Nearly every dashboard adds all four together and shows you a
single number labelled "input tokens".

cacheeconomics pulls that number apart. Point it at logs you already have and it
tells you what each class actually cost, which part of your prompt is sitting in
the expensive buckets, and how much of that you can get back. It runs on your
machine, opens no sockets, and will not print a dollar figure it cannot tie to an
invoice.

## Try it in thirty seconds

If you use Claude Code, the data is already on your disk.

```bash
pip install cacheeconomics
cacheeconomics claude-code
```

Nothing to instrument, no API key, nothing leaves the machine. This is what it
said about my own history, from 190 transcripts and 14,362 requests. Sections 1
and 5 are trimmed here; everything else is verbatim:

```
  cacheeconomics · what your prompt caching actually costs
  ────────────────────────────────────────────────────────────────────────────

  Your provider charges four different prices for the same token. Sending text
  fresh costs 1x. Reading it back out of the cache costs 0.1x. Putting it into
  the cache costs 1.25x, or 2x for the version that survives an hour. Your
  bill adds all four together and shows you one number called "input tokens".

  This pulls that number back apart, from logs you already have, and tells you
  which of the four you have been buying.

── 1 · what I read ───────────────────────────────────────────────────────────

  volume          14,362 requests over 31 days
  could be read   14,362 of 14,362 (100%)
  depth           token counts only. Enough for spend and cadence; the prompts
                  themselves are not here, so nothing can say which *part* of
                  a prompt costs you
  privacy         read on this machine. Nothing was sent anywhere

── 2 · how your caching is doing ─────────────────────────────────────────────

  Caching is working: most of what gets written to cache is read back before
  it expires.

  measure             value   in plain english
  ─────────────────── ──────  ────────────────────────────────────────────────
  input from cache    97%     of everything you sent, the share billed at the
                              cheap 0.1x read rate instead of full price
  prefix efficiency   97%     of every token you paid to put into cache, the
                              share something read back before it expired.
                              This is the one that says whether caching is
                              working

── 3 · what it is costing you — 3 findings, worst first ──────────────────────

  FIGURES WITHHELD — no invoice was supplied, so nothing here has been
  reconciled against money actually spent.

  The findings themselves still stand. This is a rule about publishing money,
  not a doubt about the analysis. Step 1 at the end turns the numbers on.

  #   severity  what it costs  what is happening
  ─── ───────── ────────────── ────────────────────────────────────────────
  1   MEDIUM    not costed     Sessions switching model mid-conversation
      do this   Keep a session on one model. Where a cheaper model is wanted
                for a sub-task, run it as a separate call rather than
                switching the main loop.
      basis     SPL-1 · measured · confidence high · quality risk medium

  2   MEDIUM    not costed     The cached prefix is being rebuilt, not
                               extended
      do this   Find what changes the prefix between turns before touching a
                TTL. Compaction in particular trades a smaller prompt for a
                full rewrite, and immediately after it every read you were
                getting at 0.1x is rebought at 1.25x.
      basis     REB-1 · measured · confidence high · quality risk low

  3   LOW       not costed     Caching is paying for itself
      do this   No action indicated on this measure. If the bill is still too
                high, the driver is input volume rather than cache
                configuration, and the questions are prompt size and turn
                count.
      basis     CAC-1 · measured · confidence high · quality risk low

  'not costed' — this rule names a mechanism and does not produce a dollar
  amount at this depth. Section 1 says what depth you gave it.

  Every row above is the short version. Re-run with --detail for the reasoning
  behind each one, the counts, and what was excluded from them.

── 4 · what to do next ───────────────────────────────────────────────────────

  1.  Get a dollar figure: re-run with --invoice-usd <amount> from the
      provider bill covering this window. Every number stays hidden until it
      reconciles to within 5% of money that actually left an account.

  2.  Or, for an internal look before the bill arrives: add
      --allow-unreconciled. It releases the figures and stamps the report
      DRAFT, which is not something to forward.

  3.  Reach the structural findings: this input carries usage counters but not
      prompt structure, so nothing here can say *which part* of the prompt
      costs you. Export request bodies from your gateway and re-run with
      --from bodies, or point the agent at tier-b/capture_proxy.py if you
      cannot export.

  4.  Act on SPL-1 first (sessions switching model mid-conversation). It is
      the highest-severity finding here, and the 'do this' line in its row is
      the change to make. Re-run with --detail for the reasoning behind it.

── 5 · the fine print — what this is based on, and what it could not see ─────

  ·   14,362 assistant turns from 43 sessions across 190 transcripts. Each is
      one billed request.
  ·   Transcripts record the conversation, not the wire request. The system
      prompt and tool definitions are absent, so prefix structure cannot be
      recovered and no counterfactual is derivable.
```

It always closes on numbered next steps chosen from what that particular run
could not do, so you never have to guess what to type after it. `--detail` puts
the reasoning back under each row.

Give it your invoice and the money column fills in:

```bash
cacheeconomics claude-code --invoice-usd 4820.16
```

```
  #   severity  what it costs  what is happening
  ─── ───────── ────────────── ────────────────────────────────────────────
  1   HIGH      ~$214/mo       'research-agent' request cadence sits inside
                               the one-hour window
      do this   Set a one-hour TTL on the static prefix for this agent.
                Outside the five-minute-to-one-hour band the five-minute
                default is cheaper, so this is not a blanket change.
      basis     TTL-1 · modeled · confidence medium · quality risk low
```

## What the clock does to your bill

This is the finding I did not expect, and it is the one worth stealing whether or
not you ever install this.

I ran [browser-use](https://github.com/browser-use/browser-use) 0.13.7 on one
task at five different schedules, captured every real request through a
forwarding proxy, and counted the tokens against the provider's own tokenizer.
Same code, same task, same prompts. The only thing I changed was how often it
ran.

| schedule | cache writes per request |
|---|---|
| back to back, ~7s apart | 39 |
| every 2 minutes | 536 |
| every 7 minutes | 3,597 |
| every 10 minutes | 4,109 |
| every 15 minutes | 3,417 |

A hundredfold jump between two minutes and seven, because a five-minute cache
entry survives five minutes. Faster than that and the agent reads its cache.
Slower and it rebuilds the whole thing, every single run, forever, paying a
premium for a cache it never once reads.

Turning those rewrites into reads is worth 13–17% of input spend on the three
slow schedules and nothing at all on the fast two. If your agent runs on a cron,
this is probably happening to you right now and no dashboard you own will show
it.

The write-up, including [the part of my own prediction the data
refuted](case-studies/schedule-decides-your-cache-bill.md), is in
`case-studies/`. Raw per-call rows are in `tier-b/evidence/`.

## Two more ways this leaks

**A cache marker below the minimum does nothing, and nobody tells you.** The
threshold is 512, 1024, 2048 or 4096 tokens depending on the model, and it does
not track model generation the way you would guess: `claude-opus-5` is 512,
`claude-opus-4-6` is 4096. Under it, the provider processes your request
uncached, writes nothing, returns `cache_creation_input_tokens: 0`, and raises no
error. I found four of these in the browser-use run.

**The one-hour cache only wins inside a band.** It costs 2x to write where the
short one costs 1.25x, so it pays only when requests are more than five minutes
and less than an hour apart. I measured this on live API calls: $0.190 for the 5m
arm against $0.114 for the 1h arm on the same prefix, 40.2% cheaper, with all
eight probes matching predictions written down before the run.

## What you can point it at

| what you have | how to run it | what you get |
|---|---|---|
| Claude Code transcripts | `cacheeconomics claude-code` | spend, ratios, rebuild and lifetime findings |
| LiteLLM proxy logs | `analyze log.jsonl --from litellm` | the same, across every model your proxy sees |
| request bodies from your gateway | `analyze bodies.jsonl --from bodies` | plus *which part* of the prompt costs you |
| an agent you did not write | `tier-b/capture_proxy.py` | the same as bodies, no code change |

The last row is how the browser-use measurement was done. Most agents let you
point them at a different base URL, because that is how people use gateways, and
that is all you need:

```bash
python3 tier-b/capture_proxy.py --out run.jsonl --port 8787 &
# start the agent with its base URL set to http://127.0.0.1:8787
```

It forwards every request byte for byte and writes each one down with its
response. Nothing is mutated, so you are measuring the agent rather than
measuring an agent with something in its path.

The file you supply decides what the tool will claim, and it works that out from
the contents rather than from what you tell it. An export cannot ask for more
confidence than it earns.

## Other things it does

```bash
# compare cache placement policies over the same requests
cacheeconomics bakeoff trace.jsonl --by-agent

# check a cache config before you ship it. works as a CI gate
cacheeconomics checks --prefix-tokens 900 --model claude-opus-5 --breakpoints 5

# what does this build actually know about pricing?
cacheeconomics registry

# Bedrock or Vertex: the rate comes off your cloud bill, not from me
cacheeconomics analyze trace.jsonl --target-id amazon-bedrock/converse --effective-rate 1.10
```

`checks` exits 0 for pass and 2 for fail, and 3 when nothing failed but something
could not be evaluated, usually a model with no dated registry entry. That third
code exists because "I did not check" is not "this passed", and collapsing them
gives you a green build where the check that catches silently-ignored cache
markers never ran.

## Counting the tokens

Skip this and you still get every finding. You just will not get dollar figures
on the ones about prompt structure. Here is why.

To say "34,000 tokens sit behind that volatile block", the tool has to know how
many tokens each part of your prompt is worth, and it cannot count them locally.
There is no local tokenizer for these models. So by default it takes the input
total your provider billed and splits it between segments in proportion to their
bytes.

That split is worse than it sounds: 19.2% off at the median, 181% at worst.
Bytes-per-token is not one number. Dense JSON tool schemas run about 2.74 bytes
per token where English prose runs 5.22. A prompt mixing them, which is every
agent prompt, over-allocates the prose and starves the tools at the same time.

Counting properly is one command, and it is the default:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python3 tier-b/run_diagnostic.py bodies.jsonl --invoice-usd 4820.16
```

That counts, then analyses. The same measurement lands at 0.2% instead of 19.2%.
It is free and fast, because the endpoint costs nothing and the work caches on the
prefix. A prefix that is not shared is not being cached in the first place, so
one count covers every request that shares it. On the demo trace, 286 requests
needed 345 calls, and a second run costs nothing at all.

It does send prompt content to a tokenizer, and you pick which one. `--endpoint`
points it at your own gateway. To see exactly what it would send before agreeing
to any of it:

```bash
python3 tier-b/count_tokens.py bodies.jsonl -o counted.jsonl --dry-run
```

That reports the call count and the host and sends nothing. It writes nothing
either, deliberately. An earlier version wrote an output file full of zeroes
that the analyzer then read as exact counts, which is exactly the kind of
confident wrong answer the rest of this exists to refuse.

`--estimate-only` skips it entirely.

It is a separate script rather than a flag because the installed package imports
no network library at all and a test asserts it. Zero egress is what a client is
trusting when they hand over a trace, and they can check it by grepping the
wheel. A flag would have made that claim depend on reading the flag's
implementation.

## What it refuses to tell you

**No invoice, no dollars.** Every figure stays hidden until it reconciles to
within 5% of money that actually left your account. `--allow-unreconciled`
releases them for internal drafts and stamps the report DRAFT.

**Structure and usage have to agree.** Spend comes from usage counters,
structural findings from segment sizes. If those two descriptions of the same
prompt disagree by more than 5%, you get the findings without their dollar
figures.

**Coverage is measured in money, not rows.** A structural claim needs to cover
90% of your billed tokens before a figure attaches. Nine small structured
requests next to one huge unstructured one is 90% of rows and can be 8% of the
bill.

**Bedrock and Vertex are not priced here.** AWS and Google operate and invoice
those, so those rates are not mine to publish and I do not borrow Anthropic's.
Pass `--effective-rate` from your own bill. That is the better number anyway:
partner rates vary by region and endpoint, and nobody at volume pays list.

**Every number says what kind of number it is.** Measured means observed in usage
fields. Modeled means projected, always as a range, with the pessimistic end as
the headline. Verified means observed in production after a change shipped.
Nothing gets promoted between them quietly.

## Privacy

Everything runs locally. Nothing in the package opens a socket, the registry is
packaged data, every input is a path on your disk.

Prompt text is optional. Hashes, structure and token counts are enough for every
finding here. Segment identifiers are keyed HMAC-SHA-256 rather than bare
digests, because a bare digest of a short policy line is confirmable by anyone
holding a guess. There is deliberately no `--key` flag: `argv` shows up in `ps`
and lands in shell history, so the key comes from `CACHEECONOMICS_HMAC_KEY` or a
`--key-file` whose permissions get checked before it is read.

Keying and scoping are different things, and the difference matters on a shared
gateway. A key stops someone guessing your content. It does nothing about
equality, which is the whole job of an identifier: under one key, identical text
produces an identical id no matter who sent it. So identity is scoped to the
tenant too. Pass `--tenant` on a multi-tenant export and two tenants sending the
same policy block stop sharing an id. That is also the arithmetically correct
answer, since caches are isolated per tenant and those requests could never have
shared an entry.

## What is in here

```
harness/cacheeconomics/
  registry.py   dated capability and pricing registry, refuses to guess
  cost.py       the four token classes, date-effective pricing
  money.py      a dollar figure that cannot be printed until it is released
  trace.py      ingest, tiering, reconciliation
  analyzer.py   the finding rules (EFF-1, VOL-1, MIN-1, TTL-1, REB-1, ...)
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

web/            a browser demo; the analyzer runs in a worker via Pyodide
tier-a/         static findings on three open-source agents, cited to file and line
tier-b/         the live experiments and their per-call evidence
case-studies/   the browser-use write-up
disclosure/     findings as filed against the projects they affect
contrib/        upstream contributions (BerriAI/litellm#35011)
```

Only `harness/` ships in the wheel. The measurement work stays here, where you
can check it: `tier-a/FINDINGS.md` cites a file and line for every claim and logs
its own correction where the first screen was wrong, and `tier-b/evidence/` holds
the per-call rows behind the 40.2% number, including one run that reports
`script_sha256: null` because it predates run-time hashing.

Eight provider surfaces are described. The recorded rates are Anthropic
first-party list prices and the table names which surfaces they are valid for.
The scope is default-deny, so a surface earns those rates by being named rather
than by being absent from an exclusion list. That is not theoretical: before the
scope existed, a Bedrock trace priced straight off the Anthropic table and
produced a confident total no AWS bill would ever match.

## Running it live

`plugin.py` can sit inside a LiteLLM proxy and place cache markers on outgoing
requests. It observes by default and only mutates when you ask.

The worry was that LiteLLM normalises Anthropic-shaped bodies through an
OpenAI-shaped intermediate and drops `cache_control`, which would give you churn
and no cache writes while the plugin's own counters cheerfully reported
placements that never arrived. It does not. On litellm 1.83.9, an unmarked
control writes nothing, a marked call writes 15,624 tokens, the same body a
moment later reads all 15,624 back, and a request routed through the handler with
`mutate=True` writes too. The runs are in
`tier-b/evidence/litellm-marker-survival.json` and the script starts cold each
time, so you can repeat it.

`mutate` still defaults to False. Knowing the mechanism works is not the same as
deciding to rewrite somebody's live traffic, and that second decision is yours.

## Development

```bash
git clone https://github.com/Tanisha-Katara/cacheeconomics.git
cd cacheeconomics && pip install .

python3 -m pytest -q          # 1040 tests, no dependencies
python3 web/build_bundle.py   # after changing anything under harness/cacheeconomics
```

Python 3.9 or newer and nothing else. CI runs the suite on 3.9 and 3.13 with only
pytest installed, so an accidental dependency fails the build rather than
shipping. `pip install "cacheeconomics[litellm]"` only if you are mounting the
live proxy plugin.

## License

Apache-2.0. Copyright 2026 KCG Consulting LLC.
</content>
