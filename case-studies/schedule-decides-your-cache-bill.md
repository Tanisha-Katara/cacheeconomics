# If your agent runs on a timer, check your cache writes

We ran the same open-source agent on five different schedules and measured what
each one cost. Below a five-minute gap between runs, its cache worked fine.
Above five minutes, it rewrote that cache on every single run and read nothing
back, wasting about 15% of its input spend.

Same code. Same task. Same prompts. Only the clock changed.

## Why five minutes

A cache entry lives five minutes. Write it, and any request in the next five
minutes reads it at a tenth of the normal input price. Wait longer and the entry
is gone, so you write it again at 1.25 times the normal price.

An agent on a ten-minute timer therefore never reads anything. It writes, the
entry dies, it writes again. Forever. It is paying a premium for a cache it
never uses.

There is a one-hour lifetime that fixes this. It costs 2x to write instead of
1.25x, and survives an hour, so a ten-minute job writes once and reads five
times instead of writing six times.

Whether that trade is worth it depends entirely on how often you run. That is
not something you can read off your code.

## The measurement

[browser-use](https://github.com/browser-use/browser-use) 0.13.7, one task, five
schedules. We captured every real request and response through a forwarding
proxy, counted the tokens against Anthropic's own tokenizer, and analysed the
result. Nothing simulated.

Forget percentages for a second. Just count how many tokens each run wrote to
cache, per request:

| schedule | cache writes per request |
|---|---|
| back to back, 7s apart | 39 |
| every 2 minutes | 536 |
| every 7 minutes | 3,597 |
| every 10 minutes | 4,109 |
| every 15 minutes | 3,417 |

A hundredfold jump between two minutes and seven. Under five minutes the agent
finds its cache. Over five minutes it finds nothing and rebuilds, every run.

Turning those wasted rewrites into reads recovers 14%, 17% and 13% of input
spend for the three slow schedules. For the two fast ones it recovers nothing,
and switching them to a one-hour TTL would make them worse: you would pay 2x to
write a cache you were already reading fine.

It is 15% and not 90% because only part of each prompt is cacheable. This agent
reads web pages, and the page is different every time.

## What we got wrong

We wrote our prediction down before running it: nothing below five minutes,
rising steadily through the band, gone again past an hour.

The cliff held. The rise did not. Fourteen, seventeen, thirteen is flat, and the
wobble is noise on about twenty requests per schedule. Our first measurement was
the ten-minute one at 17%, and if we had stopped there we would have published a
peak that does not exist.

We also never ran anything slower than an hour, where both cache types are dead
and the saving should vanish. That part is arithmetic, not evidence, and we have
labelled it that way.

## browser-use is fine

Its caching works. On the interactive run it removed 54% of input spend versus
sending the same traffic uncached. We measured it, we changed nothing, and we are
not claiming we saved anyone money.

Two things did turn up. Four requests placed a cache marker on a prefix of about
810 tokens against a 4,096-token minimum, so the provider ignored them silently
and returned no error. And there is no code path in the project that emits a
one-hour TTL at all, which is also true of OpenHands and SWE-agent. We filed
against all three on 28 July 2026.

## Go and look at your own numbers

You do not need our tool for the first check. Your provider already returns
`cache_creation_input_tokens` and `cache_read_input_tokens` on every response.
Pull both for one scheduled job and compare them.

If creation is large and read is near zero, you are in the case above. If read
dominates, you are fine and should change nothing.

Most tooling adds those two together into one input-token number, which is
exactly why this stays invisible. That is the part worth being annoyed about.

---

One agent, one task, one model, about twenty requests per schedule. The
five-minute threshold is a property of the cache and applies to any workload on
Anthropic. The 15% is a property of this workload and yours will differ, which is
the point of measuring rather than quoting it.

Runs, raw counts and the analysis are in
[`tier-b/evidence/`](https://github.com/Tanisha-Katara/cacheeconomics/tree/main/tier-b/evidence).

Being precise about which number traces to which file, because a review found I
had not been. The 7-second and 10-minute rows are derivable from
`browser-use-interactive.json` and `browser-use-scheduled.json`, which carry
`cache_creation_input_tokens` and a request count. The 2-minute, 7-minute and
15-minute rows are not: `interval-sweep.json` records each schedule's
recoverable share and its measured dollars, and no per-schedule creation total,
so those three cells cannot be recomputed from what is committed. They came from
the same sweep and I have no reason to doubt them, but you cannot check them
here, and that is a gap in the evidence rather than a detail about it. The
recoverable percentages are auditable for every schedule.
