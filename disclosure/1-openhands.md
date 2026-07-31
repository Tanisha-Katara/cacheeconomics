# OpenHands / software-agent-sdk

Repo: https://github.com/OpenHands/software-agent-sdk
Pinned at: `1f9f0b1aa0356e082d971e8a5cf82256d67fe576`

**Filed 2026-07-28 as OpenHands/software-agent-sdk#4292**
<https://github.com/OpenHands/software-agent-sdk/issues/4292>

---

Title: No way to reach the 1-hour cache TTL from `_apply_prompt_caching`

### What I found

`_apply_prompt_caching()` sets two breakpoints, and both get the default 5-minute lifetime:

- `openhands-sdk/openhands/sdk/llm/message.py:200` — `data["cache_control"] = {"type": "ephemeral"}`
- same at `message.py:214` (images) and `message.py:384` (message level)

Grepping the LLM layer (`message.py`, `llm.py`, `utils/model_features.py`) for `ttl`, `1h`, or `3600` returns nothing. `{"type": "ephemeral"}` with no `ttl` key is the 5-minute cache. The 1-hour cache needs `{"type": "ephemeral", "ttl": "1h"}`.

What made me write this up rather than shrug is the comment sitting right above it. At `llm.py:2614-2618` the dynamic system block is deliberately left unmarked:

```python
# Two-block structure: static (index 0) + dynamic (index 1)
# Mark only the static block; ensure dynamic is unmarked
sys_content[0].cache_prompt = True
sys_content[1].cache_prompt = False
```

The stated reason is "to enable cross-conversation cache sharing." That's the exact case a longer TTL exists for. A prefix shared across conversations is the one most likely to sit idle for more than five minutes between uses. The intent is clearly there; the five-minute lifetime just caps how far it reaches.

### What I measured

I ran a controlled experiment on `claude-sonnet-4-6` with a 16,442-token static prefix. Two arms on one schedule, `max_tokens=0` so only prefill gets billed.

| probe | 5m arm | 1h arm |
|---|---|---|
| t=0 | WRITE | WRITE |
| t=420s | MISS (full rewrite) | HIT |
| t=480s | HIT (stability control) | HIT |
| t=900s | MISS (full rewrite) | HIT |

Cost on the cached prefix: $0.190 for the 5m arm against $0.114 for the 1h arm, so 40.2% cheaper. All eight probes matched expectations I'd written down in advance, and worst clock drift was 2.4 seconds.

### Where this stops

I don't want to imply 1h is just better, because it isn't. Combining the measured 300s expiry with the write multipliers (5m write costs 1.25x input, 1h write 2x, read 0.1x):

| Inter-request gap | Cheaper option | Why |
|---|---|---|
| under 5 min | 5m | both stay warm, and the 5m write is cheaper |
| 5 min to 1 hr | 1h | the 5m arm rewrites on every request |
| over 1 hr | 5m | both go cold every time, cheaper write wins again |

So the 1-hour TTL only pays inside that band. Whether OpenHands deployments sit in it, I honestly have no idea. It depends on how often a shared system prefix gets reused across concurrent conversations, and that isn't something you can read out of the source. You'd know far better than I would.

My point is narrower than "you should switch." Right now there's no way to reach the 1-hour cache at all, so a deployment that is in the band can't act on it even if someone worked out that it should.

### Suggested change

Thread a TTL through `cache_prompt` so the static system block can opt into `{"type": "ephemeral", "ttl": "1h"}`, gated behind the `supports_prompt_cache` check that already exists in `model_features.py`. Whether it should default that way is your call. Making it reachable seems clearly right either way.

I'm happy to open a PR if that's useful. Tell me which configuration convention you'd want it to follow and I'll match it.

### One more thing

`model_features.py:141`, the comment saying "Do NOT add Gemini: explicit cache_control markers freeze its cache", is correct and not at all obvious. Gemini's cache is a named server-side resource rather than a span you annotate, and several other projects get that wrong. Worth keeping.

---

The harness and pinned sources for every line reference above are available if you want to reproduce any of it. Worth being clear that the measurement uses a synthetic prefix on a controlled schedule, not a replay of real OpenHands traffic. It establishes the mechanism, not what it's worth on your workload.

Filed by Tanisha Katara, CEO, KCG Consulting LLC.
