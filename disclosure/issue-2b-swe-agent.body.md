`_set_cache_control` emits the 5-minute cache at all three call sites:

- `sweagent/agent/history_processors.py:59` — `"cache_control": {"type": "ephemeral"}`
- same at lines 63 and 67

No `ttl` key anywhere. `{"type": "ephemeral"}` is the 5-minute cache; an hour needs `{"type": "ephemeral", "ttl": "1h"}`.

The clear-then-set pattern at lines 293-299 is a deliberate rolling breakpoint, so this is considered code, not an oversight. That's why I'm raising the TTL specifically and not the placement.

### What I measured

Controlled experiment on `claude-sonnet-4-6`, 16,442-token static prefix, two arms sharing one schedule, `max_tokens=0` so only prefill is billed.

| probe | 5m arm | 1h arm |
|---|---|---|
| t=0 | WRITE | WRITE |
| t=420s | MISS (full rewrite) | HIT |
| t=480s | HIT (stability control) | HIT |
| t=900s | MISS (full rewrite) | HIT |

$0.190 for the 5m arm against $0.114 for the 1h arm on the cached prefix, so 40.2% cheaper. Every probe matched an expectation written down before the run, worst drift 2.4 seconds.

### Where this stops

1h isn't universally better. From the measured 300s expiry plus the write multipliers:

| Inter-request gap | Cheaper option |
|---|---|
| under 5 min | 5m, since both stay warm and its write is cheaper |
| 5 min to 1 hr | 1h, since the 5m arm rewrites every request |
| over 1 hr | 5m, since both go cold anyway |

Whether SWE-agent trajectories land in that band comes down to step duration, and I couldn't establish it. `TrajectoryStep` does carry `execution_time`, but the public submissions in `SWE-bench/experiments` only contain `README.md`, `metadata.yaml` and `results`. No `.traj` files. So I have no measurement and I'm not going to pretend otherwise.

Two things make the band plausible, and you can check both far more cheaply than I can. `config.delay` at `models.py:673-677` deliberately spaces queries, so a delay configured above roughly five minutes would push every step past the TTL. Long-running actions between model calls would widen the gap on their own regardless of what `delay` is set to.

### Suggested change

Make the TTL configurable on the static prefix. If issue 1 lands first you'd be able to measure the effect directly rather than accepting an external claim about it, which seems like the better order to do these in.

Happy to open PRs for either or both.

---

Harness and pinned sources available if you want to reproduce any of this. The measurement is a synthetic prefix on a controlled schedule, so it establishes the mechanism and not what it's worth on real SWE-bench runs.

Filed by Tanisha Katara, CEO, KCG Consulting LLC.

Analysed at commit `3ea751c087f32b16e039a2233dd6eefecef325d5`
