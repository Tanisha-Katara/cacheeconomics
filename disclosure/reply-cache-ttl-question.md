# Reply draft — reader question on Claude Code cache TTL, model mixing, and ratios

Everything below was measured on my own Claude Code history before writing:
14,375 requests, 43 sessions, 190 transcripts, 31 days. Commands to reproduce
are at the bottom. The one modelled number is labelled.

---

The TTL question has a surprising answer, and I only got it by measuring rather
than reasoning about it. I had exactly your instinct.

Claude Code writes 1h caches. A 1h write costs 2x where a 5m write costs 1.25x,
so switching should hand back 37% of write cost. I checked the gap between
consecutive requests inside a session across my own 14,375 requests:

| gap | share |
|---|---|
| under 5 min — the 5m entry is still alive | 98.32% |
| 5 min to 1 hour — only the 1h entry survives | 1.36% |
| over an hour — both are dead | 0.32% |

Median gap 5 seconds, p90 25 seconds. So the 1h cache is buying nothing 98% of
the time and the switch looks free.

It is not, though the margin is smaller than I first calculated. Those 195 gaps
sit on prefixes averaging 467,000 tokens. When an entry dies, the next request
writes what it would have read. Costed at opus-5 list, with the multipliers
taken from the registry rather than typed in:

```
cheaper writes on every 1h write            +$482
prefix rewritten at the 195 band gaps       -$524
                                            -----
net of switching to 5m                       -$41
```

So switching costs me about $41 over the window, against $1,285 spent writing
1h caches. That is 3% — close enough to a wash that I would not describe it as
a finding, and the useful part is that it is not the clear win the frequency
argument suggests.

Two things worth saying about that arithmetic. The rebuild line is a modelled
counterfactual; I have not run the 5m arm. And I got it wrong the first time I
did it by hand, charging the full 1.25x rebuild *and* the read as separate
costs, when the read is something you stop paying under 5m rather than
something you lose. That doubled the answer. The correct netting is 0.75x on
every 1h write against 1.15x on the prefix at each band gap, and I only caught
it when I put the rule in the tool and made it agree with a second
implementation.

The mechanism generalises even if the number does not: 1.4% of your turns can
carry as much cost as the other 98.6%, once the prefix is that large. Counting
gaps by frequency gives you the wrong sign. Your 5-hour control problem is
real, and it turns out to be the cheaper problem to have.

**Your ratio.** 1 : 10 : 100–1000 is healthy, and I can be precise about why.
Mine is 1 : 10 : 311. The number that settles it is reads / (reads + writes) —
of every token you paid to store, the share something read back before it
expired. Mine is 97%. A session that extends its prefix rather than rebuilding
it tends to run near 75%, so we are both well clear. Sending my same traffic
uncached would cost 6.3x what it does now.

At that point the cache is not your lever. Volume is, and the questions become
prompt size and turn count.

**Different models for sub-agents.** Caches are isolated per model, so each
model keeps its own pool. Sub-agents having their own pool is expected and is
not waste. The cost lands when the *main loop* switches model mid-conversation:
the accumulated prefix does not carry across, and the next request writes it
cold. In my history that is 8 switches and 731,000 tokens written cold, about
$7. Small for me, and it scales with prefix size — mine are large, so if yours
are too, it is worth counting rather than assuming.

**Does it change between releases?** One thing does, and it fails silently. The
minimum prefix a model will cache moves between models, and not in the
direction you would guess:

```
opus-4-6  4096      opus-4-8  1024      sonnet-5   1024
opus-4-7  2048      opus-5     512      haiku-4-5  4096
```

Below the minimum the provider processes the request uncached, writes nothing,
and returns no error at all. A prefix that cached fine on one model can stop
caching on another with nothing anywhere saying so. Note where haiku-4-5 sits —
it is the model people reach for on cheap sub-agents and it has the highest
threshold of the set.

**On verbosity.** I can only report my own logs, and this is not a controlled
comparison — different tasks, different weeks:

```
                 median   mean    p90
claude-opus-5       792   1,089  2,248
claude-opus-4-8     793   1,539  3,020
```

Identical at the median, and 4-8 is longer at the tail. That does not refute
what you have seen, since this is one person's task mix. It is the sort of
claim worth measuring on your own logs before you rewrite prompts around it,
because the counters are already there.

**On your framing.** Agreed on metrics, with one addition that has to come
first: almost every tool adds cache reads, cache writes and fresh input into a
single "input tokens" number. Those bill at 0.1x, 1.25x or 2x, and 1x. None of
this is visible until you pull them apart, which is the actual reason the
question is hard rather than anything about the models.

One honest gap on my side, now closed. My tool had a rule for "your cadence
sits in the 1h window and you are not using it" and nothing for the reverse,
which is your case and mine. Your question is what surfaced it. It ships as
TTL-2, it nets both sides rather than counting the premium, and on my history
it returns "the one-hour lifetime is earning its premium here" — which is the
answer, and one the tool previously gave by saying nothing at all.

Writing it also caught a second thing: on a trace carrying both lifetimes, the
new rule and the old one could recommend moving the TTL in opposite directions
at once. There is now a test that fails if they ever do.

---

Reproduce:

```bash
pip install cacheeconomics
cacheeconomics claude-code            # ratios, findings, next steps
cacheeconomics claude-code --detail   # the reasoning behind each one
```

The gap distribution and the netting above are what TTL-2 computes; the model
comparison was done directly off the same transcripts.
