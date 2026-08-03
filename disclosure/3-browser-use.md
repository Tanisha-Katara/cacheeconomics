# browser-use

Repo: https://github.com/browser-use/browser-use
Pinned at: `f0aa3a8bb03779c71a5aa262d389e3bfe6b77cdc`

**Filed 2026-07-28 as browser-use/browser-use#5321**
<https://github.com/browser-use/browser-use/issues/5321>

---

Title: Usage model reports 1-hour cache tokens the serializer can never produce

### What I found

`AnthropicChatModel` reads both cache-creation buckets off the response and surfaces the 1-hour one:

```python
# browser_use/llm/anthropic/chat.py:195-196
getattr(cache_creation, 'ephemeral_5m_input_tokens', None),
getattr(cache_creation, 'ephemeral_1h_input_tokens', None),

# chat.py:216
prompt_cache_creation_1h_tokens=cache_creation_1h_tokens,
```

But the serializer only ever emits the 5-minute form:

```python
# browser_use/llm/anthropic/serializer.py:54-57
def _serialize_cache_control(use_cache: bool) -> CacheControlEphemeralParam | None:
    ...
    return CacheControlEphemeralParam(type='ephemeral')
```

`CacheControlEphemeralParam(type='ephemeral')` with no `ttl` is the 5-minute cache. An hour requires `{"type": "ephemeral", "ttl": "1h"}`, and no `ttl` appears anywhere in the Anthropic provider.

So `prompt_cache_creation_1h_tokens` is always zero, structurally. It isn't wrong exactly. It's a metric with no code path behind it, which is the sort of thing that misleads someone later who reads the usage model and reasonably assumes both TTLs are in play.

### What I measured

Controlled experiment on `claude-sonnet-4-6` with a 16,442-token static prefix, two arms sharing one schedule, `max_tokens=0` so only prefill gets billed.

| probe | 5m arm | 1h arm |
|---|---|---|
| t=0 | WRITE | WRITE |
| t=420s | MISS (full rewrite) | HIT |
| t=480s | HIT (stability control) | HIT |
| t=900s | MISS (full rewrite) | HIT |

$0.190 for the 5m arm against $0.114 for the 1h arm on the cached prefix, so 40.2% cheaper. All eight probes matched expectations written down before the run, worst clock drift 2.4 seconds.

### Where this stops

1h isn't universally better, and browser-use might be the least likely of the agents I looked at to benefit. From the measured 300s expiry and the write multipliers:

| Inter-request gap | Cheaper option |
|---|---|
| under 5 min | 5m, since both stay warm and a 1.25x write beats 2x |
| 5 min to 1 hr | 1h, since the 5m arm rewrites every request |
| over 1 hr | 5m, since both go cold every time |

Browser steps tend to be fast, so consecutive requests may well sit inside the five-minute window. If that's the case the current default is already right and the only thing worth doing here is dropping the unreachable metric.

One correction while I'm here, since I got this wrong myself at first and it seems like a common assumption. Images are cacheable. Anthropic's docs list "Images & Documents: Content blocks in the `messages.content` array, in user turns" among cacheable blocks. The real issue for a browser agent isn't that screenshots can't be cached, it's that they change every step, so a screenshot sitting before a breakpoint invalidates the stable content behind it. That's about placement, not cacheability.

### Suggested change

Threading a `ttl` through `_serialize_cache_control` alone is **not** sufficient. There are three places that construct `CacheControlEphemeralParam`, and two of them bypass that helper:

- `serializer.py:57` — inside `_serialize_cache_control`
- `serializer.py:118` — direct, on the string-content path
- `chat.py:362` — direct

So the options are either to consolidate all three behind a single TTL-aware helper, or to patch each site individually. Consolidating seems better: the next TTL-related change then has one place to touch, and it removes the chance of a partial fix that looks complete.

The alternative remains dropping the 1h fields from the usage model, since nothing can populate them today.

Both are defensible. Wiring it up is better if any deployment has gaps over five minutes, dropping the fields if you're confident none do. Either beats where it stands now, where the metric implies a capability that isn't wired up.

Happy to open a PR for whichever you prefer.

---

Harness and pinned sources available if you want to reproduce this. The measurement uses a synthetic text prefix on a controlled schedule, so it establishes the mechanism rather than what it's worth on browser-use's actual traffic.

Filed by Tanisha Katara, CEO, KCG Consulting LLC.


---

## Line-number drift since filing

Filed 2026-07-28 citing `chat.py:195-196`. As of 2026-08-03 upstream has those counters at lines 200-201. The claim is unchanged: the file still reads both `ephemeral_5m_input_tokens` and `ephemeral_1h_input_tokens`, and the serializer still never emits a 1h marker. But a maintainer opening line 195 today sees something else and stops reading, so the filed issue is worth a short correcting comment. `disclosure/verify_claims.py` now tracks 200/201.
