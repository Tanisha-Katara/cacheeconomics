# Reply draft — browser-use/browser-use#5321

Responding to @Mefisto04, who asked whether to wire up the 1h TTL or remove the
unreachable 1h metrics, and offered to implement either.

Answer first, then the measurement behind it. Since filing I ran browser-use at
five schedules, and the result partly walks back the obvious reading of my own
issue, so that is stated rather than implied.

---

Neither, I think. Make the lifetime configurable and leave the default at `5m`.

That keeps interactive users exactly where they are, gives scheduled deployments
a way to fix a real cost, and means `prompt_cache_creation_1h_tokens` in
`views.py` stops being unreachable without anyone deleting it. It looks like a
parameter on `ChatAnthropic` threaded into `_serialize_cache_control`.

Here is why not just switching to 1h, which is what my issue implied and which I
now think would be wrong.

I ran 0.13.7 on one task at five schedules, capturing real requests through a
forwarding proxy and reading the provider's own counters back. Cache writes per
request:

| schedule | cache writes per request |
|---|---|
| back to back, ~7s apart | 39 |
| every 2 minutes | 536 |
| every 7 minutes | 3,597 |
| every 10 minutes | 4,109 |
| every 15 minutes | 3,417 |

A 5-minute entry survives five minutes. Anything faster than that reads its
cache; anything slower rewrites it every run. The jump between 2 and 7 minutes is
the entry dying between runs.

For the three slow schedules a 1h TTL recovers 13-17% of input spend. For the two
fast ones it recovers nothing and costs more: a 1h write is 2x against the 5m
write's 1.25x, so you would pay extra to write a cache you were already reading
fine.

Interactive browser-use is the fast case. My run had a median gap of 7.3 seconds
and 0 of 26 gaps above five minutes. A 1h default would make the common case more
expensive, and I did not know that when I filed.

The caching here already works, worth saying. On the interactive run it removed
54% of input spend against sending the same traffic uncached.

One separate thing the same run surfaced. Four requests placed a cache marker on
a prefix of about 810 tokens against a 4,096-token minimum for that model. Below
the minimum the provider processes the request uncached, writes nothing and
returns no error, so those markers did nothing and nothing said so. Different
bug; happy to file it separately if you would rather keep this issue to one
thing.

Runs and raw counts:
https://github.com/Tanisha-Katara/cacheeconomics/tree/main/tier-b/evidence

Assignment is the maintainers' call rather than mine.
